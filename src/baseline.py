import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from .preprocessing import EmailDataset
from .modeling import EmailClassifier
import os


def train_baseline(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Train the baseline model on the emotion dataset."""
    # Load emotion dataset
    emotion_dataset = load_dataset("emotion")

    # Create train and validation splits
    train_texts = emotion_dataset["train"]["text"][
        :160
    ]  # Using a smaller subset for testing
    train_labels = emotion_dataset["train"]["label"][:160]
    val_texts = emotion_dataset["validation"]["text"][:40]
    val_labels = emotion_dataset["validation"]["label"][:40]

    # Create datasets
    train_dataset = EmailDataset(train_texts, train_labels)
    val_dataset = EmailDataset(val_texts, val_labels)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Initialize classifier
    classifier = EmailClassifier(num_labels=6, device=device)

    # Train model
    classifier.train(train_loader, val_loader, epochs=3)

    # Save model
    os.makedirs("models/baseline", exist_ok=True)
    checkpoint = {
        "model_state_dict": classifier.model.state_dict(),
        "tokenizer_config": classifier.tokenizer.save_pretrained("models/baseline"),
        "model_config": classifier.model.config.to_dict(),
        "class_weights": (
            classifier.criterion.weights
            if hasattr(classifier.criterion, "weights")
            else None
        ),
    }
    torch.save(checkpoint, "models/baseline/model.pth")

    return classifier
