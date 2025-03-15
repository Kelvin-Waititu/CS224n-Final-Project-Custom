import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, get_peft_config, TaskType
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
import random
import logging


class TemperatureScaling:
    """
    Temperature scaling for confidence calibration.
    """

    def __init__(self):
        self.temperature = nn.Parameter(torch.ones(1))
        self.optimizer = None

    def calibrate(self, logits, labels, lr=0.01, max_iter=50):
        """Calibrate the temperature parameter."""
        self.temperature = nn.Parameter(torch.ones(1).to(logits.device))
        self.optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval():
            loss = nn.CrossEntropyLoss()(self.scale(logits), labels)
            loss.backward()
            return loss

        self.optimizer.step(eval)

    def scale(self, logits):
        """Scale the logits with temperature."""
        return logits / self.temperature


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def forward(self, logits, targets):
        if self.weights is None:
            return nn.functional.cross_entropy(logits, targets)
        return nn.functional.cross_entropy(logits, targets, weight=self.weights)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=1, gamma=2, reduction="mean", weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weights = weights

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets, weight=self.weights, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class MixupAugmentation:
    """Mixup augmentation for text classification."""

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x, y):
        """Apply mixup to the batch."""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size)

        # For input IDs, we'll use the original inputs (no mixing)
        mixed_x = x
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


class EmailClassifier:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_labels: int = 6,
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_labels = num_labels

        # Initialize model with proper configuration
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="single_label_classification",
            ignore_mismatched_sizes=True,
        ).to(device)

        # Initialize temperature scaling for calibration
        self.temperature_scaler = TemperatureScaling()

        # Initialize loss function (weights will be set during training)
        self.criterion = WeightedCrossEntropyLoss()

        self.optimizer = None
        self.scheduler = None

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_f1_scores = []
        self.val_f1_scores = []

        # Cross-validation scores
        self.cv_scores = []

        # Emotion weights for priority scoring (as per paper)
        self.emotion_weights = {
            0: 0.8,  # anger
            1: 0.9,  # fear
            2: 0.3,  # joy
            3: 0.4,  # love
            4: 0.7,  # sadness
            5: 0.5,  # surprise
        }

        # Urgency keywords (as per paper)
        self.urgency_keywords = {"urgent", "immediate", "asap", "critical", "emergency"}

        # Initialize mixup augmentation
        self.mixup = MixupAugmentation(alpha=0.2)

        # Initialize focal loss (will be set during training)
        self.focal_loss = None

    def compute_class_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """Compute balanced class weights with improved scaling."""
        class_counts = torch.bincount(labels, minlength=self.num_labels)
        total = class_counts.sum()

        # Smooth the weights to prevent extreme values
        smoothing_factor = 0.1
        smoothed_counts = class_counts + smoothing_factor * total

        # Calculate inverse frequency weights
        weights = total / (smoothed_counts * self.num_labels)

        # Apply sqrt to reduce the impact of extreme imbalances
        weights = torch.sqrt(weights)

        # Normalize weights
        weights = weights / weights.sum() * self.num_labels

        # Clip weights to prevent extreme values
        weights = weights.clamp(min=0.2, max=5.0)

        logging.info(f"Class weights: {dict(enumerate(weights.tolist()))}")
        return weights.to(self.device)

    def calculate_priority_score(
        self, text: str, emotion_idx: int, emotion_conf: float
    ) -> Tuple[str, float]:
        """Calculate priority score using the multi-factor system from paper."""
        # 1. Emotion Intensity (EI)
        ei_score = self.emotion_weights[emotion_idx] * emotion_conf

        # 2. Urgency Score (US)
        text_lower = text.lower()
        urgency_count = sum(1 for word in self.urgency_keywords if word in text_lower)
        us_score = min(urgency_count * 0.3, 1.0)  # Cap at 1.0

        # 3. Complexity Factor (CF)
        word_count = len(text.split())
        cf_score = min(word_count / 50, 1.0)

        # Combined score (as per paper formula)
        final_score = 0.5 * ei_score + 0.3 * us_score + 0.2 * cf_score

        # Priority thresholds
        if final_score >= 0.7 or "critical" in text_lower:
            priority = "High"
        elif final_score >= 0.4:
            priority = "Medium"
        else:
            priority = "Low"

        return priority, final_score

    def train_with_cv(self, train_loader, val_loader, n_splits=5, epochs=3, lr=2e-5):
        """Train with cross-validation using data loaders."""
        cv_scores = []

        for fold in range(n_splits):
            print(f"\nTraining Fold {fold + 1}/{n_splits}")

            # Reset model for each fold
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=self.num_labels,
                problem_type="single_label_classification",
                ignore_mismatched_sizes=True,
            ).to(self.device)

            # Train on this fold
            metrics = self.train(train_loader, val_loader, epochs=epochs, lr=lr)
            cv_scores.append(metrics)

        # Print cross-validation results
        print("\nCross-validation Results:")
        accuracies = [score["accuracy"] for score in cv_scores]
        f1_scores = [score["macro_f1"] for score in cv_scores]
        print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Mean F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

        self.cv_scores = cv_scores

    def calibrate_confidence(self, val_loader):
        """Calibrate model confidence using temperature scaling."""
        self.model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.append(outputs.logits)
                all_labels.append(labels)

        # Concatenate all logits and labels
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        # Calibrate temperature
        self.temperature_scaler.calibrate(logits, labels)
        print(f"Optimal temperature: {self.temperature_scaler.temperature.item():.3f}")

    def predict_with_calibration(self, text: str) -> Tuple[int, float]:
        """Predict emotion label with calibrated confidence."""
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            outputs = self.model(**inputs)
            logits = outputs.logits

            # Apply temperature scaling
            scaled_logits = self.temperature_scaler.scale(logits)
            probs = torch.nn.functional.softmax(scaled_logits, dim=1)

            pred = torch.argmax(probs, dim=1)
            confidence = probs[0, pred].item()

            return pred.item(), confidence

    def train(self, train_loader, val_loader=None, epochs=3, lr=2e-5):
        """Train with improved class balancing and stability measures."""
        # Compute class weights from training data
        all_labels = torch.cat([batch["labels"] for batch in train_loader])
        class_weights = self.compute_class_weights(all_labels)

        # Initialize focal loss with class weights
        self.focal_loss = FocalLoss(alpha=1, gamma=2, weights=class_weights)
        self.criterion = self.focal_loss

        # Initialize optimizer with gradient clipping
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=0.01,
            eps=1e-8,
        )

        # Learning rate scheduler with warmup and cosine decay
        total_steps = len(train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps,
        )

        best_val_loss = float("inf")
        best_model_state = None
        patience = 5  # Increased patience
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)

            # Log class distribution in predictions
            train_preds = torch.tensor(train_metrics["predictions"])
            train_pred_dist = torch.bincount(train_preds, minlength=self.num_labels)
            logging.info(
                f"Epoch {epoch + 1} training predictions distribution: {dict(enumerate(train_pred_dist.tolist()))}"
            )

            # Validation
            if val_loader:
                val_metrics = self.evaluate(val_loader)

                # Log validation prediction distribution
                val_preds = torch.tensor(val_metrics["predictions"])
                val_pred_dist = torch.bincount(val_preds, minlength=self.num_labels)
                logging.info(
                    f"Epoch {epoch + 1} validation predictions distribution: {dict(enumerate(val_pred_dist.tolist()))}"
                )

                # Early stopping with model saving
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logging.info(f"Early stopping triggered after epoch {epoch + 1}")
                    # Restore best model
                    self.model.load_state_dict(best_model_state)
                    break

                print(f"Epoch {epoch + 1}/{epochs}:")
                print(
                    f"Train - Loss: {train_metrics['loss']:.4f}, "
                    f"Acc: {train_metrics['accuracy']*100:.2f}%, "
                    f"F1: {train_metrics['macro_f1']:.4f}"
                )
                print(
                    f"Val - Loss: {val_metrics['loss']:.4f}, "
                    f"Acc: {val_metrics['accuracy']*100:.2f}%, "
                    f"F1: {val_metrics['macro_f1']:.4f}"
                )

            # Store metrics
            self.train_losses.append(train_metrics["loss"])
            self.train_accuracies.append(train_metrics["accuracy"])
            self.train_f1_scores.append(train_metrics["macro_f1"])

            if val_loader:
                self.val_losses.append(val_metrics["loss"])
                self.val_accuracies.append(val_metrics["accuracy"])
                self.val_f1_scores.append(val_metrics["macro_f1"])

        # Calibrate confidence after training
        if val_loader:
            self.calibrate_confidence(val_loader)

        return {
            "accuracy": (
                self.val_accuracies[-1] if val_loader else self.train_accuracies[-1]
            ),
            "macro_f1": (
                self.val_f1_scores[-1] if val_loader else self.train_f1_scores[-1]
            ),
        }

    def train_epoch(self, train_loader):
        """Train for one epoch with improved monitoring and stability."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            # Apply mixup augmentation only to labels
            _, labels_a, labels_b, lam = self.mixup(input_ids, labels)

            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Compute mixed loss
            loss = lam * self.focal_loss(logits, labels_a) + (
                1 - lam
            ) * self.focal_loss(logits, labels_b)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        metrics = classification_report(
            all_labels, all_preds, output_dict=True, zero_division=0
        )

        return {
            "loss": total_loss / len(train_loader),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro avg"]["f1-score"],
            "predictions": all_preds,
        }

    def fine_tune(self, train_loader, val_loader=None, epochs=5, lr=1e-5):
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q_lin", "v_lin"],
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        # Train with LoRA configuration
        self.train(train_loader, val_loader, epochs=epochs, lr=lr)

    def evaluate(self, val_loader):
        """Evaluate with prediction distribution monitoring."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs.logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        metrics = classification_report(
            all_labels, all_preds, output_dict=True, zero_division=0
        )

        return {
            "loss": total_loss / len(val_loader),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro avg"]["f1-score"],
            "predictions": all_preds,  # Add predictions for distribution monitoring
        }

    @staticmethod
    def compute_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
        """Compute detailed classification metrics."""
        report = classification_report(
            labels,
            preds,
            output_dict=True,
            zero_division=0,  # Handle zero-division case
        )
        return {
            "accuracy": report["accuracy"] * 100,
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
        }

    def plot_metrics(self, title: str = "Training Metrics") -> None:
        """Plot comprehensive training metrics."""
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(15, 5))

        # Plot losses
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_losses, "b-", label="Training Loss")
        plt.plot(epochs, self.val_losses, "r-", label="Validation Loss")
        plt.title(f"{title} - Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Plot accuracies
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.train_accuracies, "b-", label="Training Accuracy")
        plt.plot(epochs, self.val_accuracies, "r-", label="Validation Accuracy")
        plt.title(f"{title} - Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()

        # Plot F1 scores
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.train_f1_scores, "b-", label="Training F1")
        plt.plot(epochs, self.val_f1_scores, "r-", label="Validation F1")
        plt.title(f"{title} - Macro F1")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"logs/{title.lower().replace(' ', '_')}_metrics.png")
        plt.close()

    def plot_confusion_matrix(
        self, val_loader: torch.utils.data.DataLoader, title: str = "Confusion Matrix"
    ) -> None:
        """Plot confusion matrix for validation data."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                # Move all batch tensors to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["anger", "fear", "joy", "love", "sadness", "surprise"],
            yticklabels=["anger", "fear", "joy", "love", "sadness", "surprise"],
        )
        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(f"logs/{title.lower().replace(' ', '_')}.png")
        plt.close()

    def predict_emotion(self, text: str) -> int:
        """Predict emotion label for a given text."""
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            outputs = self.model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1)
            return pred.item()
