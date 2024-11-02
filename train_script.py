import os
import subprocess
import sys
import argparse
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support



# Load data function
def load_data(data_dir):
    # Load real and fake news data
    real_df = pd.read_csv(os.path.join(data_dir, "True.csv"))
    fake_df = pd.read_csv(os.path.join(data_dir, "Fake.csv"))

    # Add labels: 0 for real, 1 for fake
    real_df["label"] = 0
    fake_df["label"] = 1

    # Combine the datasets and shuffle
    df = pd.concat([real_df, fake_df]).sample(frac=1).reset_index(drop=True)

    return df['text'].tolist(), df['label'].tolist()

# Compute metrics function for Trainer
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Main training function
def main(args):
    # Load data
    texts, labels = load_data(args.data_dir)
    
    # Tokenizer and encoding
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    
    # Convert to PyTorch tensors
    input_ids = torch.tensor(encodings["input_ids"])
    attention_masks = torch.tensor(encodings["attention_mask"])
    labels = torch.tensor(labels)
    
    # Create TensorDataset
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Load model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir = os.path.join("/opt/ml/model"),          # Directory to save model
        num_train_epochs=3,                 # Number of training epochs
        per_device_train_batch_size=8,      # Batch size for training
        per_device_eval_batch_size=8,       # Batch size for evaluation
        evaluation_strategy="epoch",        # Evaluate every epoch
        logging_dir=f"{args.output_data_dir}/logs",  # Directory for logs
        logging_steps=10,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    trainer.train()
    
    # Save model
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

# Argument parsing for SageMaker inputs
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SageMaker specific arguments
    parser.add_argument("--data_dir", type=str, default="/opt/ml/input/data")
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--output_data_dir", type=str, default="/opt/ml/output/data")

    args = parser.parse_args()
    main(args)
