import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from .preprocessing import EmailDataset
from .modeling import EmailClassifier
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from peft import LoraConfig, get_peft_model, TaskType


def load_model(
    model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> EmailClassifier:
    """Load a saved model."""
    # Initialize classifier
    classifier = EmailClassifier(num_labels=6, device=device)

    try:
        # First try with weights_only=False for compatibility
        checkpoint = torch.load(model_path, weights_only=False)
    except Exception as e:
        print(f"Error loading model with weights_only=False: {str(e)}")
        print("Attempting alternative loading method...")
        # Add safe globals for DistilBert configuration
        from transformers import DistilBertConfig

        torch.serialization.add_safe_globals([DistilBertConfig])
        checkpoint = torch.load(model_path, weights_only=True)

    # Check if this is a LoRA model
    if any("base_model" in key for key in checkpoint["model_state_dict"].keys()):
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q_lin", "v_lin"],
        )
        classifier.model = get_peft_model(classifier.model, peft_config)

    # Load state dict
    classifier.model.load_state_dict(checkpoint["model_state_dict"])

    # Load class weights if available
    if "class_weights" in checkpoint and checkpoint["class_weights"] is not None:
        classifier.criterion.weights = checkpoint["class_weights"]

    return classifier


def analyze_examples(
    test_texts: list,
    true_labels: list,
    predicted_labels: list,
    probabilities: list,
    priorities: list,
    scores: list,
    output_dir: str,
    n_examples: int = 10,
) -> None:
    """Analyze qualitative examples with predictions and confidence scores."""
    # Create categories for analysis
    categories = {
        "correct_high_conf": [],  # Correct predictions with high confidence
        "correct_low_conf": [],  # Correct predictions with low confidence
        "incorrect_high_conf": [],  # Incorrect predictions with high confidence
        "incorrect_low_conf": [],  # Incorrect predictions with low confidence
    }

    emotion_map = {
        0: "anger",
        1: "fear",
        2: "joy",
        3: "love",
        4: "sadness",
        5: "surprise",
    }

    # Categorize predictions
    for i in range(len(test_texts)):
        text = test_texts[i]
        true_label = true_labels[i]
        pred_label = predicted_labels[i]
        prob = max(probabilities[i])  # Confidence score
        priority = priorities[i]
        priority_score = scores[i]

        example = {
            "text": text,
            "true_label": emotion_map[true_label],
            "predicted_label": emotion_map[pred_label],
            "confidence": prob,
            "priority": priority,
            "priority_score": priority_score,
        }

        is_correct = true_label == pred_label
        is_high_conf = prob >= 0.8

        if is_correct and is_high_conf:
            categories["correct_high_conf"].append(example)
        elif is_correct and not is_high_conf:
            categories["correct_low_conf"].append(example)
        elif not is_correct and is_high_conf:
            categories["incorrect_high_conf"].append(example)
        else:
            categories["incorrect_low_conf"].append(example)

    # Create analysis report
    report = []
    report.append("# Qualitative Analysis Report\n")

    for category, examples in categories.items():
        report.append(
            f"\n## {category.replace('_', ' ').title()} ({len(examples)} examples)\n"
        )

        # Sort examples by confidence score
        examples = sorted(examples, key=lambda x: x["confidence"], reverse=True)

        # Take top n examples
        for i, example in enumerate(examples[:n_examples]):
            report.append(f"\n### Example {i+1}")
            report.append(f"Text: {example['text']}")
            report.append(f"True Label: {example['true_label']}")
            report.append(f"Predicted Label: {example['predicted_label']}")
            report.append(f"Confidence: {example['confidence']:.2%}")
            report.append(f"Priority: {example['priority']}")
            report.append(f"Priority Score: {example['priority_score']:.2f}\n")

    # Save report
    with open(f"{output_dir}/qualitative_analysis.md", "w") as f:
        f.write("\n".join(report))


def analyze_email_characteristics(
    texts: list, predictions: list, true_labels: list, output_dir: str
) -> None:
    """Analyze how email characteristics affect model predictions."""
    # Calculate email lengths
    lengths = [len(text.split()) for text in texts]

    # Create length categories
    length_categories = {
        "short": [],  # < 50 words
        "medium": [],  # 50-150 words
        "long": [],  # > 150 words
    }

    # Analyze accuracy by length
    for text, pred, true, length in zip(texts, predictions, true_labels, lengths):
        if length < 50:
            length_categories["short"].append((text, pred, true))
        elif length < 150:
            length_categories["medium"].append((text, pred, true))
        else:
            length_categories["long"].append((text, pred, true))

    # Calculate accuracy for each length category
    length_analysis = {}
    for category, examples in length_categories.items():
        if examples:
            correct = sum(1 for _, pred, true in examples if pred == true)
            accuracy = correct / len(examples)
            length_analysis[category] = {"count": len(examples), "accuracy": accuracy}

    # Analyze urgency keywords and their impact
    urgency_keywords = {"urgent", "immediate", "asap", "critical", "emergency"}
    urgent_examples = []
    non_urgent_examples = []

    for text, pred, true in zip(texts, predictions, true_labels):
        has_urgency = any(keyword in text.lower() for keyword in urgency_keywords)
        if has_urgency:
            urgent_examples.append((text, pred, true))
        else:
            non_urgent_examples.append((text, pred, true))

    # Calculate accuracy for urgent vs non-urgent emails
    urgency_analysis = {}
    for category, examples in [
        ("urgent", urgent_examples),
        ("non_urgent", non_urgent_examples),
    ]:
        if examples:
            correct = sum(1 for _, pred, true in examples if pred == true)
            accuracy = correct / len(examples)
            urgency_analysis[category] = {"count": len(examples), "accuracy": accuracy}

    # Generate report
    report = []
    report.append("# Email Characteristics Analysis\n")

    # Length analysis
    report.append("## Analysis by Email Length\n")
    for category, stats in length_analysis.items():
        report.append(f"### {category.title()} Emails (< 50 words)")
        report.append(f"Count: {stats['count']}")
        report.append(f"Accuracy: {stats['accuracy']:.2%}\n")

    # Urgency analysis
    report.append("\n## Analysis by Urgency\n")
    for category, stats in urgency_analysis.items():
        report.append(f"### {category.replace('_', ' ').title()} Emails")
        report.append(f"Count: {stats['count']}")
        report.append(f"Accuracy: {stats['accuracy']:.2%}\n")

    # Plot length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, edgecolor="black")
    plt.title("Distribution of Email Lengths")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.savefig(f"{output_dir}/email_length_distribution.png")
    plt.close()

    # Save report
    with open(f"{output_dir}/email_characteristics_analysis.md", "w") as f:
        f.write("\n".join(report))


def evaluate_model(
    classifier: EmailClassifier,
    test_texts: list,
    test_labels: list,
    output_dir: str,
    batch_size: int = 16,
):
    """Evaluate model performance."""
    # Create test dataset and dataloader
    test_dataset = EmailDataset(test_texts, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Get predictions
    classifier.model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(classifier.device)
            attention_mask = batch["attention_mask"].to(classifier.device)
            labels = batch["labels"].to(classifier.device)

            outputs = classifier.model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    metrics = classification_report(all_labels, all_preds, output_dict=True)

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["anger", "fear", "joy", "love", "sadness", "surprise"],
        yticklabels=["anger", "fear", "joy", "love", "sadness", "surprise"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

    # Save detailed metrics
    metrics_df = pd.DataFrame(metrics).transpose()
    metrics_df.to_csv(f"{output_dir}/metrics.csv")

    # Calculate priority scores for each example
    priorities = []
    scores = []
    for text, pred, prob in zip(test_texts, all_preds, all_probs):
        priority, score = classifier.calculate_priority_score(text, pred, prob[pred])
        priorities.append(priority)
        scores.append(score)

    # Create results DataFrame
    results_df = pd.DataFrame(
        {
            "text": test_texts,
            "true_label": test_labels,
            "predicted_label": all_preds,
            "confidence": [prob[pred] for prob, pred in zip(all_probs, all_preds)],
            "priority": priorities,
            "priority_score": scores,
        }
    )

    # Save results
    results_df.to_csv(f"{output_dir}/predictions.csv", index=False)

    # Generate qualitative analysis
    analyze_examples(
        test_texts, all_labels, all_preds, all_probs, priorities, scores, output_dir
    )

    # Analyze email characteristics
    analyze_email_characteristics(test_texts, all_preds, all_labels, output_dir)

    return metrics, results_df


def evaluate_baseline():
    """Evaluate baseline model on emotion dataset."""
    # Load emotion test set
    emotion_dataset = load_dataset("emotion")
    test_texts = emotion_dataset["test"]["text"]
    test_labels = emotion_dataset["test"]["label"]

    # Load baseline model
    classifier = load_model("models/baseline/model.pth")

    # Evaluate
    metrics, results = evaluate_model(
        classifier, test_texts, test_labels, output_dir="models/baseline/evaluation"
    )
    return metrics, results


def evaluate_finetuned():
    """Evaluate fine-tuned model on email dataset."""
    # Load AESLC test set
    aeslc_dataset = load_dataset("aeslc")
    test_texts = aeslc_dataset["test"]["email_body"]

    # Load baseline model to generate ground truth labels
    baseline_classifier = load_model("models/baseline/model.pth")
    test_labels = [baseline_classifier.predict_emotion(text) for text in test_texts]

    # Load fine-tuned model
    classifier = load_model("models/finetune/model.pth")

    # Evaluate
    metrics, results = evaluate_model(
        classifier, test_texts, test_labels, output_dir="models/finetune/evaluation"
    )
    return metrics, results
