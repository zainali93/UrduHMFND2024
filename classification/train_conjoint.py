"""
Conjoint XLM-R model for four-class Urdu fake news classification.

Paper:
    Detection of Human and Machine-Authored Fake News in Urdu
    ACL 2025

The conjoint approach fine-tunes two independent binary classifiers:

    1. Authorship classifier:
       Human vs Machine

    2. Veracity classifier:
       Fake vs True

Their predictions are combined to obtain the final four classes:

    Human   + Fake -> HFake
    Human   + True -> HTrue
    Machine + Fake -> MFake
    Machine + True -> MTrue

Expected dataset columns:
    text  - Urdu news text
    label - one of {HFake, HTrue, MFake, MTrue}

Supported experiments:
    Dataset1, Dataset2, Dataset3, Dataset4, Short, Long, All
"""

from pathlib import Path

import numpy as np
import pandas as pd

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

EXPERIMENT = "Dataset2"

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


FOUR_CLASS_LABELS = [
    "HFake",
    "HTrue",
    "MFake",
    "MTrue",
]

FOUR_CLASS_LABEL2ID = {
    "HFake": 0,
    "HTrue": 1,
    "MFake": 2,
    "MTrue": 3,
}

FOUR_CLASS_ID2LABEL = {
    0: "HFake",
    1: "HTrue",
    2: "MFake",
    3: "MTrue",
}


# Binary label mappings

AUTHORSHIP_LABEL2ID = {
    "Human": 0,
    "Machine": 1,
}

AUTHORSHIP_ID2LABEL = {
    0: "Human",
    1: "Machine",
}

VERACITY_LABEL2ID = {
    "True": 0,
    "Fake": 1,
}

VERACITY_ID2LABEL = {
    0: "True",
    1: "Fake",
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

    invalid_labels = (
        set(df["label"].unique())
        - set(FOUR_CLASS_LABELS)
    )

    if invalid_labels:
        raise ValueError(
            f"Unexpected labels in {path}: "
            f"{sorted(invalid_labels)}"
        )

    return df


def load_experiment(experiment):
    """
    Load the released train and test splits for an experiment.

    Short = Dataset1 + Dataset2
    Long  = Dataset3 + Dataset4
    All   = Dataset1 + Dataset2 + Dataset3 + Dataset4
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

        train_frames.append(
            load_split(train_path)
        )

        test_frames.append(
            load_split(test_path)
        )

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
# Binary task construction
# ---------------------------------------------------------------------------

def create_authorship_data(df):
    """
    Convert four-class labels into Human/Machine labels.

        HFake -> Human
        HTrue -> Human
        MFake -> Machine
        MTrue -> Machine
    """

    binary_df = df.copy()

    mapping = {
        "HFake": 0,
        "HTrue": 0,
        "MFake": 1,
        "MTrue": 1,
    }

    binary_df["label"] = (
        binary_df["label"].map(mapping)
    )

    return binary_df


def create_veracity_data(df):
    """
    Convert four-class labels into True/Fake labels.

        HFake -> Fake
        HTrue -> True
        MFake -> Fake
        MTrue -> True
    """

    binary_df = df.copy()

    mapping = {
        "HFake": 1,
        "HTrue": 0,
        "MFake": 1,
        "MTrue": 0,
    }

    binary_df["label"] = (
        binary_df["label"].map(mapping)
    )

    return binary_df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_binary_metrics(eval_pred):
    """Compute metrics for a binary classifier."""

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
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
# Dataset tokenization
# ---------------------------------------------------------------------------

def prepare_dataset(df, tokenizer):
    """Convert a dataframe into a tokenized Hugging Face dataset."""

    dataset = Dataset.from_pandas(
        df,
        preserve_index=False,
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    dataset = dataset.map(
        tokenize,
        batched=True,
    )

    dataset = dataset.remove_columns(
        ["text"]
    )

    dataset.set_format("torch")

    return dataset


# ---------------------------------------------------------------------------
# Binary classifier training
# ---------------------------------------------------------------------------

def train_binary_classifier(
    task_name,
    train_df,
    validation_df,
    tokenizer,
    label2id,
    id2label,
):
    """Fine-tune one XLM-R binary classifier."""

    print("\n" + "=" * 60)
    print(f"Training {task_name} classifier")
    print("=" * 60)

    train_dataset = prepare_dataset(
        train_df,
        tokenizer,
    )

    validation_dataset = prepare_dataset(
        validation_df,
        tokenizer,
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=2,
            label2id=label2id,
            id2label=id2label,
        )
    )

    training_args = TrainingArguments(
        output_dir=(
            f"outputs/conjoint/"
            f"{EXPERIMENT}/{task_name}"
        ),

        num_train_epochs=NUM_EPOCHS,

        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,

        per_device_train_batch_size=(
            TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            EVAL_BATCH_SIZE
        ),

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
        compute_metrics=compute_binary_metrics,
    )

    trainer.train()

    return trainer


# ---------------------------------------------------------------------------
# Conjoint prediction
# ---------------------------------------------------------------------------

def combine_predictions(
    authorship_predictions,
    veracity_predictions,
):
    """
    Combine binary predictions into four-class predictions.

    Authorship:
        0 = Human
        1 = Machine

    Veracity:
        0 = True
        1 = Fake

    Final classes:
        0 = HFake
        1 = HTrue
        2 = MFake
        3 = MTrue
    """

    final_predictions = []

    for authorship, veracity in zip(
        authorship_predictions,
        veracity_predictions,
    ):

        if authorship == 0 and veracity == 1:
            final_predictions.append(0)  # HFake

        elif authorship == 0 and veracity == 0:
            final_predictions.append(1)  # HTrue

        elif authorship == 1 and veracity == 1:
            final_predictions.append(2)  # MFake

        else:
            final_predictions.append(3)  # MTrue

    return np.array(final_predictions)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():

    set_seed(SEED)

    print("=" * 60)
    print("Conjoint XLM-R - Urdu Fake News Classification")
    print("=" * 60)
    print(f"Experiment: {EXPERIMENT}")
    print(f"Model:      {MODEL_NAME}")

    # -------------------------------------------------------
    # Load data
    # -------------------------------------------------------

    train_df, test_df = load_experiment(
        EXPERIMENT
    )

    # The released files preserve the paper's withheld
    # test split. Reserve 25% of the remaining training
    # data for validation.
    train_df, validation_df = train_test_split(
        train_df,
        test_size=0.25,
        random_state=SEED,
        stratify=train_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(
        drop=True
    )
    test_df = test_df.reset_index(drop=True)

    print(
        f"\nTraining samples:   "
        f"{len(train_df):,}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_df):,}"
    )

    print(
        f"Test samples:       "
        f"{len(test_df):,}"
    )

    # -------------------------------------------------------
    # Construct binary tasks
    # -------------------------------------------------------

    authorship_train = create_authorship_data(
        train_df
    )

    authorship_validation = (
        create_authorship_data(
            validation_df
        )
    )

    veracity_train = create_veracity_data(
        train_df
    )

    veracity_validation = (
        create_veracity_data(
            validation_df
        )
    )

    # -------------------------------------------------------
    # Tokenizer
    # -------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # -------------------------------------------------------
    # Train authorship classifier
    # -------------------------------------------------------

    authorship_trainer = (
        train_binary_classifier(
            task_name="authorship",
            train_df=authorship_train,
            validation_df=authorship_validation,
            tokenizer=tokenizer,
            label2id=AUTHORSHIP_LABEL2ID,
            id2label=AUTHORSHIP_ID2LABEL,
        )
    )

    # -------------------------------------------------------
    # Train veracity classifier
    # -------------------------------------------------------

    veracity_trainer = (
        train_binary_classifier(
            task_name="veracity",
            train_df=veracity_train,
            validation_df=veracity_validation,
            tokenizer=tokenizer,
            label2id=VERACITY_LABEL2ID,
            id2label=VERACITY_ID2LABEL,
        )
    )

    # -------------------------------------------------------
    # Prepare test data for both classifiers
    # -------------------------------------------------------

    authorship_test = create_authorship_data(
        test_df
    )

    veracity_test = create_veracity_data(
        test_df
    )

    authorship_test_dataset = prepare_dataset(
        authorship_test,
        tokenizer,
    )

    veracity_test_dataset = prepare_dataset(
        veracity_test,
        tokenizer,
    )

    # -------------------------------------------------------
    # Binary predictions
    # -------------------------------------------------------

    print("\nGenerating conjoint predictions...")

    authorship_output = (
        authorship_trainer.predict(
            authorship_test_dataset
        )
    )

    veracity_output = (
        veracity_trainer.predict(
            veracity_test_dataset
        )
    )

    authorship_predictions = np.argmax(
        authorship_output.predictions,
        axis=-1,
    )

    veracity_predictions = np.argmax(
        veracity_output.predictions,
        axis=-1,
    )

    # -------------------------------------------------------
    # Combine predictions
    # -------------------------------------------------------

    final_predictions = combine_predictions(
        authorship_predictions,
        veracity_predictions,
    )

    gold_labels = (
        test_df["label"]
        .map(FOUR_CLASS_LABEL2ID)
        .to_numpy()
    )

    # -------------------------------------------------------
    # Final four-class evaluation
    # -------------------------------------------------------

    accuracy = accuracy_score(
        gold_labels,
        final_predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            gold_labels,
            final_predictions,
            labels=[0, 1, 2, 3],
            average=None,
            zero_division=0,
        )
    )

    print("\n" + "=" * 60)
    print("Conjoint Test Results")
    print("=" * 60)

    print(
        f"{'Class':<10}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )

    for label_id in range(4):

        label = FOUR_CLASS_ID2LABEL[
            label_id
        ]

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
