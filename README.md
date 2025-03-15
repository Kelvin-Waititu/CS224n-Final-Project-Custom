# Advanced Email Prioritization Using Emotion-Aware Transformer Models

This project implements an email prioritization system that uses transformer-based emotion classification and a novel multi-factor scoring system to assign priority levels to emails.

## Features

- Emotion-aware classification using DistilBERT
- Class imbalance handling with weighted cross-entropy loss
- Data augmentation with mixup
- Efficient fine-tuning with LoRA (Low-Rank Adaptation)
- Multi-factor priority scoring system (emotion intensity, urgency, complexity)

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```
   python main.py
   ```

## Project Structure

- `src/modeling.py`: Contains the email classifier model and related components
- `src/train.py`: Training functions for baseline and fine-tuning
- `src/evaluation.py`: Evaluation functions for model testing
- `src/preprocessing.py`: Data preprocessing utilities

## Dataset

The project uses two datasets:

- Emotion dataset: For baseline emotion classification training
- AESLC dataset: For fine-tuning on email data

## CS224N Final Project

Stanford University
