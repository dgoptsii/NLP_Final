#!/usr/bin/env python3
"""
evaluate_split_model.py

Evaluate a logistic regression baseline on pre-split phishing datasets.

Expected inputs
---------------
- processed/splits/train.csv
- processed/splits/val.csv
- processed/splits/test.csv

This script:
1. Loads existing train/val/test CSV files
2. Validates required feature columns
3. Trains Logistic Regression on train only
4. Evaluates on train / val / test
5. Prints proper classification metrics:
   - accuracy
   - balanced accuracy
   - precision
   - recall (sensitivity)
   - specificity
   - F1
   - ROC AUC
   - PR AUC
   - Matthews correlation coefficient
   - confusion matrix
6. Prints top positive / negative coefficients
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


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
    "has_form_tag",
    "has_script_tag",
    "has_iframe_tag",
    "has_ip_url",
    "has_external_links",
]


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{split_name} file not found: {path}")

    df = pd.read_csv(path)

    required = FEATURE_COLUMNS + ["label", "id", "source", "subject", "text"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{split_name} is missing required columns: {missing}")

    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": cm,
    }


def print_dataset_summary(name: str, df: pd.DataFrame) -> None:
    print(f"{name:5s} rows={len(df):5d}  labels={df['label'].value_counts().sort_index().to_dict()}")


def print_metrics_block(name: str, y_true, y_pred, y_prob) -> None:
    metrics = compute_metrics(y_true, y_pred, y_prob)

    print("\n" + "=" * 60)
    print(f"{name} METRICS")
    print("=" * 60)
    print(f"Accuracy:             {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy:    {metrics['balanced_accuracy']:.4f}")
    # print(f"Precision:            {metrics['precision']:.4f}")
    # print(f"Recall / Sensitivity: {metrics['recall']:.4f}")
    print(f"Specificity:          {metrics['specificity']:.4f}")
    print(f"F1 score:             {metrics['f1']:.4f}")
    # print(f"ROC AUC:              {metrics['roc_auc']:.4f}")
    # print(f"PR AUC:               {metrics['pr_auc']:.4f}")
    # print(f"Matthews corrcoef:    {metrics['mcc']:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    print("Confusion matrix:")
    print(metrics["confusion_matrix"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True, help="Path to train.csv")
    parser.add_argument("--val", type=Path, required=True, help="Path to val.csv")
    parser.add_argument("--test", type=Path, required=True, help="Path to test.csv")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--class-weight",
        type=str,
        default=None,
        choices=[None, "balanced"],
        help="Optional class_weight for LogisticRegression",
    )
    args = parser.parse_args()

    train_df = load_split(args.train, "train")
    val_df = load_split(args.val, "val")
    test_df = load_split(args.test, "test")

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print_dataset_summary("TRAIN", train_df)
    print_dataset_summary("VAL", val_df)
    print_dataset_summary("TEST", test_df)

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df["label"].copy()

    X_val = val_df[FEATURE_COLUMNS].copy()
    y_val = val_df["label"].copy()

    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df["label"].copy()

    model = LogisticRegression(
        max_iter=3000,
        random_state=args.random_seed,
        class_weight=args.class_weight,
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    train_prob = model.predict_proba(X_train)[:, 1]

    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]

    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]

    print_metrics_block("TRAIN", y_train, train_pred, train_prob)
    print_metrics_block("VAL", y_val, val_pred, val_prob)
    print_metrics_block("TEST", y_test, test_pred, test_prob)

    coef_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "coefficient": model.coef_[0],
            "abs_coefficient": model.coef_[0].copy(),
        }
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("coefficient", ascending=False)

    print("\n" + "=" * 60)
    print("TOP POSITIVE FEATURES FOR PHISHING")
    print("=" * 60)
    print(coef_df[["feature", "coefficient"]].head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("TOP NEGATIVE FEATURES FOR PHISHING")
    print("=" * 60)
    print(coef_df[["feature", "coefficient"]].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()