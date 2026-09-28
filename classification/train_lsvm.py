"""
Linear SVM baseline for four-class Urdu fake news classification.

Paper:
    Detection of Human and Machine-Authored Fake News in Urdu
    ACL 2025

The released datasets are expected to contain:
    text  - Urdu news text
    label - one of {HFake, HTrue, MFake, MTrue}

Select the dataset to evaluate using DATASET_NAME below.
"""

import re
import string
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_NAME = "Dataset4"

DATASETS = {
    "Dataset1": Path("Datasets/Ax-to-Grind"),
    "Dataset2": Path("Datasets/UFN2023"),
    "Dataset3": Path("Datasets/UFNAugmented"),
    "Dataset4": Path("Datasets/BendtheTruth"),
}

LABELS = ["HFake", "HTrue", "MFake", "MTrue"]

# Urdu stop words used during traditional text preprocessing.
URDU_STOPWORDS = {
    "آئی", "آئے", "آتا", "آتی", "آتے", "آج", "آپ", "اپنا", "اپنی", "اپنے",
    "اب", "اس", "اسے", "اسی", "اگر", "ان", "انہوں", "انہیں", "انہی", "اور",
    "ایک", "ایسے", "ایسی", "ایسا", "بھی", "پر", "پھر", "تھا", "تھی", "تھے",
    "تک", "تو", "تم", "تمہارے", "جا", "جاتا", "جاتی", "جاتے", "جب",
    "جو", "جس", "جن", "جیسا", "جیسے", "جیسی", "حالانکہ", "خود", "رہا",
    "رہی", "رہے", "سے", "سو", "طرح", "کا", "کی", "کے", "کر", "کرتا",
    "کرتی", "کرتے", "کچھ", "کو", "کون", "کیا", "کیسے", "کیونکہ", "کہ",
    "گیا", "گئی", "گئے", "گا", "گی", "گے", "لیکن", "میں", "میرا", "میری",
    "میرے", "نہ", "نہیں", "نے", "وہ", "وہاں", "وہی", "ہو", "ہوا", "ہوئی",
    "ہوئے", "ہوتا", "ہوتی", "ہوتے", "ہوں", "ہے", "ہیں", "ہی", "یا", "یہ",
    "یہاں", "یہی"
}


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def clean_text(text):
    """
    Clean Urdu text for the traditional machine-learning baseline.

    The preprocessing removes URLs, punctuation, and stop words.
    """

    text1 = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove punctuation
    punctuation = string.punctuation + "،؛؟۔٪"
    text = text.translate(str.maketrans("", "", punctuation))

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stop words
    tokens = [
        token
        for token in text.split()
        if token not in URDU_STOPWORDS
    ]
    return " ".join(tokens)



# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_split(path):
    """Load and validate one dataset split."""

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


def load_dataset(dataset_name):
    """Load the predefined training and test splits."""

    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(DATASETS.keys())}"
        )

    dataset_dir = DATASETS[dataset_name]

    train_path = dataset_dir / "train.xlsx"
    test_path = dataset_dir / "test.xlsx"

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = load_split(train_path)
    test_df = load_split(test_path)

    return train_df, test_df


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def main():

    print("=" * 60)
    print("Linear SVM - Four-Class Urdu Fake News Classification")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")

    train_df, test_df = load_dataset(DATASET_NAME)

    print(f"Training samples: {len(train_df):,}")
    print(f"Test samples:     {len(test_df):,}")

    # -------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------

    print("\nPreprocessing text...")

    train_text = train_df["text"].apply(clean_text)
    test_text = test_df["text"].apply(clean_text)

    y_train = train_df["label"]
    y_test = test_df["label"]

    # -------------------------------------------------------
    # TF-IDF feature extraction
    # -------------------------------------------------------

    print("Extracting TF-IDF features...")

    vectorizer = TfidfVectorizer()

    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    print(f"Feature vocabulary size: {len(vectorizer.vocabulary_):,}")

    # -------------------------------------------------------
    # Linear SVM
    # -------------------------------------------------------

    print("\nTraining Linear SVM...")

    classifier = LinearSVC()
    classifier.fit(X_train, y_train)

    # -------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------

    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        labels=LABELS,
        target_names=LABELS,
        digits=4,
        zero_division=0,
        output_dict=True,
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

    for label in LABELS:
        metrics = report[label]

        print(
            f"{label:<10}"
            f"{metrics['precision']:>12.4f}"
            f"{metrics['recall']:>12.4f}"
            f"{metrics['f1-score']:>12.4f}"
        )

    print("-" * 60)
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
