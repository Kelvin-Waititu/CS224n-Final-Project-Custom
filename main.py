from src.preprocessing import (
    load_and_preprocess_data,
    load_emotion_data,
    split_data,
    create_dataloaders,
)
from src.modeling import EmailClassifier
import torch
import os
import pandas as pd
from collections import Counter
from datasets import load_dataset
from src.preprocessing import EmailDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.train import train_baseline, fine_tune
from src.evaluation import evaluate_baseline, evaluate_finetuned
import logging


def create_directories():
    """Create necessary directories if they don't exist."""
    dirs = ["models", "logs", "data"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/training.log")],
    )


def main():
    # Create necessary directories
    os.makedirs("models/baseline", exist_ok=True)
    os.makedirs("models/finetune", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Setup logging
    setup_logging()

    # Log device information
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Train baseline model with full dataset
    logging.info("Starting baseline training...")
    baseline_classifier = train_baseline(
        num_samples=None, epochs=10
    )  # Use full dataset with 10 epochs

    # Evaluate baseline model
    logging.info("Evaluating baseline model...")
    baseline_metrics, baseline_results = evaluate_baseline()
    logging.info(f"Baseline metrics: {baseline_metrics}")

    # Log prediction distribution
    pred_dist = Counter(baseline_results["predicted_label"])
    logging.info(f"Baseline prediction distribution: {dict(pred_dist)}")

    # Fine-tune model with full dataset
    logging.info("Starting fine-tuning...")
    fine_tuned_classifier = fine_tune(
        baseline_classifier, num_samples=None, epochs=10
    )  # Use full dataset with 10 epochs

    # Evaluate fine-tuned model
    logging.info("Evaluating fine-tuned model...")
    finetuned_metrics, finetuned_results = evaluate_finetuned()
    logging.info(f"Fine-tuned metrics: {finetuned_metrics}")

    # Log fine-tuned prediction distribution
    finetuned_pred_dist = Counter(finetuned_results["predicted_label"])
    logging.info(f"Fine-tuned prediction distribution: {dict(finetuned_pred_dist)}")

    # Plot training metrics
    baseline_classifier.plot_metrics(title="Baseline Training")
    fine_tuned_classifier.plot_metrics(title="Fine-tuning")


if __name__ == "__main__":
    main()
