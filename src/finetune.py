import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from .preprocessing import EmailDataset
from .modeling import EmailClassifier
from .evaluation import load_model
import os


def finetune_model(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Fine-tune the baseline model on the email dataset."""
    # Load AESLC dataset
    aeslc_dataset = load_dataset("aeslc")

    # Create train and validation splits
    train_texts = aeslc_dataset["train"]["email_body"][
        :80
    ]  # Using a smaller subset for testing
    val_texts = aeslc_dataset["validation"]["email_body"][:20]

    # Load baseline model to generate emotion labels
    baseline_classifier = load_model("models/baseline/model.pth", device)

    # Generate emotion labels using baseline model
    train_labels = [baseline_classifier.predict_emotion(text) for text in train_texts]
    val_labels = [baseline_classifier.predict_emotion(text) for text in val_texts]

    # Create datasets for fine-tuning
    train_dataset = EmailDataset(train_texts, train_labels)
    val_dataset = EmailDataset(val_texts, val_labels)

    # Create dataloaders for fine-tuning
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Initialize classifier with baseline weights
    classifier = EmailClassifier(num_labels=6, device=device)
    classifier.model.load_state_dict(baseline_classifier.model.state_dict())

    # Fine-tune model
    classifier.fine_tune(train_loader, val_loader, epochs=5)

    # Save fine-tuned model
    os.makedirs("models/finetune", exist_ok=True)
    checkpoint = {
        "model_state_dict": classifier.model.state_dict(),
        "tokenizer_config": classifier.tokenizer.save_pretrained("models/finetune"),
        "model_config": classifier.model.config.to_dict(),
        "class_weights": (
            classifier.criterion.weights
            if hasattr(classifier.criterion, "weights")
            else None
        ),
    }
    torch.save(checkpoint, "models/finetune/model.pth")

    return classifier
