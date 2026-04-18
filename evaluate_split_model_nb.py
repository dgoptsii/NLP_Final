#!/usr/bin/env python3
"""
Gaussian Naive Bayes baseline on pre-split phishing datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.naive_bayes import GaussianNB
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

# Same as logistic regression baseline. TODO: Maybe import it instead? 

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
    print(f"Precision:            {metrics['precision']:.4f}")
    print(f"Recall / Sensitivity: {metrics['recall']:.4f}")
    print(f"Specificity:          {metrics['specificity']:.4f}")
    print(f"F1 score:             {metrics['f1']:.4f}")
    print(f"ROC AUC:              {metrics['roc_auc']:.4f}")
    print(f"PR AUC:               {metrics['pr_auc']:.4f}")
    print(f"Matthews corrcoef:    {metrics['mcc']:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    print("Confusion matrix:")
    print(metrics["confusion_matrix"])

# End of same functions as baseline

# Where is the source (nazaro or enron) of false negative and false positive
def print_error_analysis(name: str, df: pd.DataFrame, y_pred) -> None:
    result = df[["id", "source", "label"]].copy()
    result["pred"] = y_pred

    fn = result[(result["label"] == 1) & (result["pred"] == 0)] 
    fp = result[(result["label"] == 0) & (result["pred"] == 1)]

    print("\n" + "=" * 60)
    print(f"{name} ERROR ANALYSIS")
    print("=" * 60)

    print(f"\nFalse Negatives (phishing predicted as legit): {len(fn)}")
    if len(fn) > 0:
        print(fn["source"].value_counts().to_string())

    print(f"\nFalse Positives (legit predicted as phishing): {len(fp)}")
    if len(fp) > 0:
        print(fp["source"].value_counts().to_string())


# Interpretability, per-feature means for each class and the delta (phishing - legit)
def print_feature_means(model: GaussianNB) -> None:
    df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "mean_legit (0)": model.theta_[0],
            "mean_phish (1)": model.theta_[1],
        }
    )
    df["delta"] = df["mean_phish (1)"] - df["mean_legit (0)"]
    df = df.sort_values("delta", ascending=False)

    print("\n" + "=" * 60)
    print("FEATURE MEANS PER CLASS (sorted by delta)")
    print("=" * 60)
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True, help="Path to train.csv")
    parser.add_argument("--val",   type=Path, required=True, help="Path to val.csv")
    parser.add_argument("--test",  type=Path, required=True, help="Path to test.csv")
    parser.add_argument(
        "--var-smoothing",
        type=float,
        default=1e-9,
        help="GaussianNB var_smoothing parameter (default: 1e-9)",
    )
    args = parser.parse_args()

    train_df = load_split(args.train, "train")
    val_df   = load_split(args.val,   "val")
    test_df  = load_split(args.test,  "test")

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print_dataset_summary("TRAIN", train_df)
    print_dataset_summary("VAL",   val_df)
    print_dataset_summary("TEST",  test_df)

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["label"].values

    X_val  = val_df[FEATURE_COLUMNS].values
    y_val  = val_df["label"].values

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["label"].values

    model = GaussianNB(var_smoothing=args.var_smoothing)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    train_prob = model.predict_proba(X_train)[:, 1]

    val_pred  = model.predict(X_val)
    val_prob  = model.predict_proba(X_val)[:, 1]

    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]

    print_metrics_block("TRAIN", y_train, train_pred, train_prob)
    print_metrics_block("VAL",   y_val,   val_pred,   val_prob)
    print_metrics_block("TEST",  y_test,  test_pred,  test_prob)

    print_error_analysis("TRAIN", train_df, train_pred)
    print_error_analysis("VAL",   val_df,   val_pred)
    print_error_analysis("TEST",  test_df,  test_pred)

    print_feature_means(model)


if __name__ == "__main__":
    main()