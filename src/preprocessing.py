from datasets import load_dataset
import pandas as pd
import torch
from typing import Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer


class EmailDataset(Dataset):
    def __init__(self, texts, labels, max_length=128):
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

        # Tokenize all texts at initialization
        self.encodings = self.tokenizer(
            list(self.texts),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx]),
        }
        return item


def load_and_preprocess_data(
    annotate: bool = False,
    classifier: Optional[object] = None,
    max_samples: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess the "aeslc" dataset, optionally annotating emotions.
    Args:
        annotate: Whether to annotate emails with emotions using classifier
        classifier: Pre-trained classifier for annotation
        max_samples: Maximum number of samples to load (for testing)
    """
    # Load the "aeslc" dataset
    dataset = load_dataset("aeslc")

    # Convert to DataFrame
    df = pd.DataFrame(dataset["train"])
    df = df[["email_body"]]
    df = df.rename(columns={"email_body": "text"})

    if max_samples:
        df = df.head(max_samples)

    if annotate and classifier:
        # Annotate emotions using pre-trained classifier
        def predict_emotion(text):
            inputs = classifier.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=128
            )
            with torch.no_grad():
                outputs = classifier.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                return probs.argmax().item()

        df["label"] = df["text"].apply(predict_emotion)
    else:
        # Placeholder labels if not annotating
        df["label"] = 0

    # Map labels to emotions
    emotion_map = {
        0: "anger",
        1: "fear",
        2: "joy",
        3: "love",
        4: "sadness",
        5: "surprise",
    }
    df["emotion"] = df["label"].map(emotion_map)

    return df


def load_emotion_data(max_samples: Optional[int] = None) -> pd.DataFrame:
    """Load the 'emotion' dataset for baseline training."""
    dataset = load_dataset("emotion")
    df = pd.DataFrame(dataset["train"])
    df = df[["text", "label"]]

    if max_samples:
        df = df.head(max_samples)

    # Map labels to match taxonomy
    emotion_map = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise",
    }
    df["emotion"] = df["label"].map(emotion_map)
    return df


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets."""
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    return train_df, test_df


def create_dataloaders(
    train_df: pd.DataFrame, test_df: pd.DataFrame, tokenizer, batch_size: int = 16
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for training and testing."""

    def tokenize_and_encode(texts):
        return tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors=None
        )

    # Tokenize data
    train_encodings = tokenize_and_encode(train_df["text"].tolist())
    test_encodings = tokenize_and_encode(test_df["text"].tolist())

    # Create datasets
    train_dataset = EmailDataset(
        train_encodings, torch.tensor(train_df["label"].values)
    )
    test_dataset = EmailDataset(test_encodings, torch.tensor(test_df["label"].values))

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
