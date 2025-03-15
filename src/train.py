import logging
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from .modeling import EmailClassifier
from .preprocessing import EmailDataset
import torch
import random
import numpy as np
from collections import Counter
from tqdm import tqdm


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_baseline(num_samples=None, epochs=100):
    """Train baseline model on emotion dataset with cross-validation."""
    # Set random seed for reproducibility
    set_seed(42)

    # Load emotion dataset
    emotion_dataset = load_dataset("emotion")

    # Get training samples
    if num_samples is None:
        # Use full dataset
        train_texts = emotion_dataset["train"]["text"]
        train_labels = emotion_dataset["train"]["label"]
        logging.info(f"Using full training dataset with {len(train_texts)} samples")
    else:
        # Calculate half of the dataset size if specified
        total_samples = len(emotion_dataset["train"])
        num_samples = min(num_samples, total_samples)
        logging.info(
            f"Using {num_samples} samples for training (out of total {total_samples})"
        )
        train_texts = emotion_dataset["train"]["text"][:num_samples]
        train_labels = emotion_dataset["train"]["label"][:num_samples]

    # Use full validation dataset
    val_texts = emotion_dataset["validation"]["text"]
    val_labels = emotion_dataset["validation"]["label"]

    # Log class distributions
    train_dist = Counter(train_labels)
    val_dist = Counter(val_labels)
    logging.info(f"Training data class distribution: {dict(train_dist)}")
    logging.info(f"Validation data class distribution: {dict(val_dist)}")

    # Create datasets
    train_dataset = EmailDataset(train_texts, train_labels)
    val_dataset = EmailDataset(val_texts, val_labels)

    # Initialize model with regularization
    classifier = EmailClassifier()

    # Add dropout for regularization
    classifier.model.config.hidden_dropout_prob = 0.3  # Increased dropout
    classifier.model.config.attention_probs_dropout_prob = 0.3

    # Create data loaders with balanced sampling
    train_weights = compute_sample_weights(train_labels)
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_weights, num_samples=len(train_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # Increased batch size
        sampler=train_sampler,
        num_workers=4,  # Added parallel data loading
        pin_memory=True,  # Improved GPU transfer
    )
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, pin_memory=True)

    # Skip cross-validation during testing phase
    if epochs <= 2:
        logging.info("Testing phase: Skipping cross-validation...")
        classifier.train(train_loader, val_loader, epochs=epochs)
    else:
        logging.info("Starting baseline training with cross-validation...")
        classifier.train_with_cv(
            train_loader, val_loader, n_splits=3, epochs=epochs // 10
        )
        classifier.train(train_loader, val_loader, epochs=epochs)

    # Save model
    torch.save(
        {
            "model_state_dict": classifier.model.state_dict(),
            "tokenizer_config": classifier.tokenizer.save_pretrained("models/baseline"),
            "model_config": classifier.model.config,
            "class_weights": (
                classifier.criterion.weights
                if hasattr(classifier.criterion, "weights")
                else None
            ),
        },
        "models/baseline/model.pth",
    )

    return classifier


def compute_sample_weights(labels):
    """Compute sample weights for balanced sampling."""
    class_counts = Counter(labels)
    total_samples = len(labels)
    class_weights = {
        cls: total_samples / (len(class_counts) * count)
        for cls, count in class_counts.items()
    }

    # Apply sqrt to reduce extreme weights
    class_weights = {cls: np.sqrt(weight) for cls, weight in class_weights.items()}

    # Normalize weights
    max_weight = max(class_weights.values())
    class_weights = {cls: weight / max_weight for cls, weight in class_weights.items()}

    # Convert to sample weights
    weights = [class_weights[label] for label in labels]
    return weights


def fine_tune(baseline_classifier, num_samples=None, epochs=100):
    """Fine-tune model on email dataset with improved stability."""
    # Set random seed for reproducibility
    set_seed(42)

    # Load AESLC dataset
    aeslc_dataset = load_dataset("aeslc")

    # Get training samples
    if num_samples is None:
        # Use full dataset
        train_texts = aeslc_dataset["train"]["email_body"]
        logging.info(
            f"Using full AESLC dataset with {len(train_texts)} samples for fine-tuning"
        )
    else:
        # Calculate half of the dataset size if specified
        total_samples = len(aeslc_dataset["train"])
        num_samples = min(num_samples, total_samples)
        logging.info(
            f"Using {num_samples} samples for fine-tuning (out of total {total_samples})"
        )
        train_texts = aeslc_dataset["train"]["email_body"][:num_samples]

    # Use full validation dataset
    val_texts = aeslc_dataset["validation"]["email_body"]

    # Generate emotion labels using baseline model with temperature scaling
    logging.info("Generating emotion labels for training data...")
    train_labels = []
    train_confidences = []

    for text in tqdm(train_texts, desc="Generating training labels"):
        label, confidence = baseline_classifier.predict_with_calibration(text)
        train_labels.append(label)
        train_confidences.append(confidence)

    val_labels = []
    val_confidences = []
    for text in tqdm(val_texts, desc="Generating validation labels"):
        label, confidence = baseline_classifier.predict_with_calibration(text)
        val_labels.append(label)
        val_confidences.append(confidence)

    # Filter out low confidence predictions with adaptive thresholding
    initial_threshold = 0.7
    train_filtered = [
        (text, label, conf)
        for text, label, conf in zip(train_texts, train_labels, train_confidences)
    ]

    # Sort by confidence and take top 80%
    train_filtered.sort(key=lambda x: x[2], reverse=True)
    num_keep = int(len(train_filtered) * 0.8)
    train_filtered = train_filtered[:num_keep]

    # Get adaptive threshold from kept samples
    adaptive_threshold = train_filtered[-1][2]
    logging.info(f"Adaptive confidence threshold: {adaptive_threshold:.3f}")

    # Apply same threshold to validation
    val_filtered = [
        (text, label)
        for text, label, conf in zip(val_texts, val_labels, val_confidences)
        if conf >= adaptive_threshold
    ]

    train_texts, train_labels, _ = zip(*train_filtered)
    val_texts, val_labels = zip(*val_filtered)

    # Log class distributions
    train_dist = Counter(train_labels)
    val_dist = Counter(val_labels)
    logging.info(f"Training data emotion distribution: {dict(train_dist)}")
    logging.info(f"Validation data emotion distribution: {dict(val_dist)}")

    # Create datasets
    train_dataset = EmailDataset(train_texts, train_labels)
    val_dataset = EmailDataset(val_texts, val_labels)

    # Initialize model for fine-tuning with regularization
    classifier = EmailClassifier()
    classifier.model.load_state_dict(baseline_classifier.model.state_dict())

    # Add dropout for regularization
    classifier.model.config.hidden_dropout_prob = (
        0.4  # Increased dropout for fine-tuning
    )
    classifier.model.config.attention_probs_dropout_prob = 0.4

    # Create data loaders with balanced sampling
    train_weights = compute_sample_weights(train_labels)
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_weights, num_samples=len(train_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, pin_memory=True)

    # Skip cross-validation during testing phase
    if epochs <= 2:
        logging.info("Testing phase: Skipping cross-validation...")
        classifier.fine_tune(train_loader, val_loader, epochs=epochs)
    else:
        logging.info("Starting fine-tuning with cross-validation...")
        classifier.train_with_cv(
            train_loader, val_loader, n_splits=3, epochs=epochs // 10
        )
        classifier.fine_tune(train_loader, val_loader, epochs=epochs)

    # Save fine-tuned model
    torch.save(
        {
            "model_state_dict": classifier.model.state_dict(),
            "tokenizer_config": classifier.tokenizer.save_pretrained("models/finetune"),
            "model_config": classifier.model.config,
            "class_weights": (
                classifier.criterion.weights
                if hasattr(classifier.criterion, "weights")
                else None
            ),
        },
        "models/finetune/model.pth",
    )

    return classifier
