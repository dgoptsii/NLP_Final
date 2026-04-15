#!/usr/bin/env python3
"""
test_dataset_model.py

Quick sanity-check model for final_dataset.csv.

What it does
------------
1. Loads the balanced normalized dataset
2. Uses only handcrafted numeric/binary features
3. Splits into train/test
4. Trains Logistic Regression
5. Prints accuracy, precision, recall, F1, confusion matrix

Usage
-----
python test_dataset_model.py \
  --data "processed/final_dataset.csv" \
  --random-seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "char_len",
    "token_len",
    "subject_len",
    "num_urls",
    "num_emails",
    "num_domains",
    "num_exclamation",
    "num_upper_tokens",
    "keyword_hits",
    "num_attachments",
    # "has_html",
    "has_form_tag",
    "has_script_tag",
    "has_iframe_tag",
    # "has_embedded_images",
    "has_ip_url",
    "has_at_in_url",
    "has_external_links",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to final_dataset.csv")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    df = pd.read_csv(args.data)

    missing = [col for col in FEATURE_COLUMNS + ["label"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.random_seed,
        stratify=y,
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=args.random_seed,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n" + "=" * 60)
    print("DATASET")
    print("=" * 60)
    print(f"Rows: {len(df)}")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Label distribution: {y.value_counts().sort_index().to_dict()}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    coef_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "coefficient": model.coef_[0],
    }).sort_values("coefficient", ascending=False)

    print("\n" + "=" * 60)
    print("TOP POSITIVE FEATURES FOR PHISHING")
    print("=" * 60)
    print(coef_df.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("TOP NEGATIVE FEATURES FOR PHISHING")
    print("=" * 60)
    print(coef_df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()