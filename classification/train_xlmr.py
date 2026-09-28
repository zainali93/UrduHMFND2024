"""
XLM-RoBERTa baseline for four-class Urdu fake news classification.

Paper:
    Detection of Human and Machine-Authored Fake News in Urdu
    ACL 2025

Expected dataset columns:
    text  - Urdu news text
    label - one of {HFake, HTrue, MFake, MTrue}

Supported experiments:
    Dataset1, Dataset2, Dataset3, Dataset4, Short, Long, All
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENT = "Dataset1"

MODEL_NAME = "xlm-roberta-base"

SEED = 42
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16

MAX_LENGTH = 512


DATASETS = {
    "Dataset1": Path("Datasets/Ax-to-Grind"),
    "Dataset2": Path("Datasets/UFN2023"),
    "Dataset3": Path("Datasets/UFNAugmented"),
    "Dataset4": Path("Datasets/BendtheTruth"),
}


EXPERIMENTS = {
    "Dataset1": ["Dataset1"],
    "Dataset2": ["Dataset2"],
    "Dataset3": ["Dataset3"],
    "Dataset4": ["Dataset4"],
    "Short": ["Dataset1", "Dataset2"],
    "Long": ["Dataset3", "Dataset4"],
    "All": ["Dataset1", "Dataset2", "Dataset3", "Dataset4"],
}


LABELS = ["HFake", "HTrue", "MFake", "MTrue"]

LABEL2ID = {
    "HFake": 0,
    "HTrue": 1,
    "MFake": 2,
    "MTrue": 3,
}

ID2LABEL = {
    0: "HFake",
    1: "HTrue",
    2: "MFake",
    3: "MTrue",
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_split(path):
    """Load and validate one released dataset split."""

    df = pd.read_excel(path)

    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{path} must contain the columns 'text' and 'label'. "
            f"Found: {list(df.columns)}"
        )

    df = df[["text", "label"]].dropna().copy()

    invalid_labels = set(df["label"].unique()) - set(LABELS)

    if invalid_labels:
        raise ValueError(
            f"Unexpected labels in {path}: {sorted(invalid_labels)}"
        )

    return df


def load_experiment(experiment):
    """
    Load the released training and test splits for an experiment.

    For Short, Long, and All, the corresponding datasets are concatenated.
    """

    if experiment not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{experiment}'. "
            f"Choose from: {list(EXPERIMENTS.keys())}"
        )

    train_frames = []
    test_frames = []

    for dataset_name in EXPERIMENTS[experiment]:

        dataset_dir = DATASETS[dataset_name]

        train_path = dataset_dir / "train.xlsx"
        test_path = dataset_dir / "test.xlsx"

        if not train_path.exists():
            raise FileNotFoundError(
                f"Training file not found: {train_path}"
            )

        if not test_path.exists():
            raise FileNotFoundError(
                f"Test file not found: {test_path}"
            )

        train_frames.append(load_split(train_path))
        test_frames.append(load_split(test_path))

    train_df = pd.concat(
        train_frames,
        ignore_index=True,
    )

    test_df = pd.concat(
        test_frames,
        ignore_index=True,
    )

    return train_df, test_df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """Compute accuracy and macro-averaged classification metrics."""

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():

    set_seed(SEED)

    print("=" * 60)
    print("XLM-R - Four-Class Urdu Fake News Classification")
    print("=" * 60)
    print(f"Experiment: {EXPERIMENT}")
    print(f"Model:      {MODEL_NAME}")

    # -------------------------------------------------------
    # Load released train/test data
    # -------------------------------------------------------

    train_df, test_df = load_experiment(EXPERIMENT)

    # The released training split represents 80% of the data.
    # Taking 25% of it for validation gives the 60/20/20
    # train/validation/test split used in the paper.
    train_df, validation_df = train_test_split(
        train_df,
        test_size=0.25,
        random_state=SEED,
        stratify=train_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"\nTraining samples:   {len(train_df):,}")
    print(f"Validation samples: {len(validation_df):,}")
    print(f"Test samples:       {len(test_df):,}")

    # -------------------------------------------------------
    # Encode labels
    # -------------------------------------------------------

    for df in (train_df, validation_df, test_df):
        df["label"] = df["label"].map(LABEL2ID)

    # -------------------------------------------------------
    # Hugging Face datasets
    # -------------------------------------------------------

    train_dataset = Dataset.from_pandas(
        train_df,
        preserve_index=False,
    )

    validation_dataset = Dataset.from_pandas(
        validation_df,
        preserve_index=False,
    )

    test_dataset = Dataset.from_pandas(
        test_df,
        preserve_index=False,
    )

    # -------------------------------------------------------
    # Tokenization
    # -------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
    )

    validation_dataset = validation_dataset.map(
        tokenize,
        batched=True,
    )

    test_dataset = test_dataset.map(
        tokenize,
        batched=True,
    )

    train_dataset = train_dataset.remove_columns(["text"])
    validation_dataset = validation_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    train_dataset.set_format("torch")
    validation_dataset.set_format("torch")
    test_dataset.set_format("torch")

    # -------------------------------------------------------
    # Model
    # -------------------------------------------------------

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    # -------------------------------------------------------
    # Training configuration
    # -------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=f"outputs/xlmr/{EXPERIMENT}",

        num_train_epochs=NUM_EPOCHS,

        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,

        evaluation_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        save_total_limit=1,

        seed=SEED,
        data_seed=SEED,

        logging_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )

    # -------------------------------------------------------
    # Fine-tuning
    # -------------------------------------------------------

    print("\nFine-tuning XLM-R...")

    trainer.train()

    # -------------------------------------------------------
    # Test evaluation
    # -------------------------------------------------------

    print("\nEvaluating on test set...")

    output = trainer.predict(test_dataset)

    predictions = np.argmax(
        output.predictions,
        axis=-1,
    )

    gold_labels = output.label_ids

    accuracy = accuracy_score(
        gold_labels,
        predictions,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_labels,
        predictions,
        labels=[0, 1, 2, 3],
        average=None,
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)

    print(
        f"{'Class':<10}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )

    for label_id in range(4):

        label = ID2LABEL[label_id]

        print(
            f"{label:<10}"
            f"{precision[label_id]:>12.4f}"
            f"{recall[label_id]:>12.4f}"
            f"{f1[label_id]:>12.4f}"
        )

    print("-" * 60)
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
