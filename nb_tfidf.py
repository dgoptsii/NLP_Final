#!/usr/bin/env python3
"""
Naive Bayes + TF-IDF model on phishing datasets.

TF-IDF (unigrams + bigrams, top 50k features) then MultinomialNB
GaussianNB assumes gaussian distributions and does not handle sparse matrices well, so MultinomialNB is used

"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
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

    # Fill NaN in text fields and concatenate subject and text
    df["subject"] = df["subject"].fillna("")
    df["text"]    = df["text"].fillna("")
    df["text_input"] = df["subject"] + " " + df["text"]

    return df


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision":         precision_score(y_true, y_pred, zero_division=0),
        "recall":            recall_score(y_true, y_pred, zero_division=0),
        "specificity":       specificity,
        "f1":                f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":           roc_auc_score(y_true, y_prob),
        "pr_auc":            average_precision_score(y_true, y_prob),
        "mcc":               matthews_corrcoef(y_true, y_pred),
        "confusion_matrix":  cm,
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

# End of same functions as logistic regression

# Same as Gaussian NB
def print_error_analysis(name: str, df: pd.DataFrame, y_pred) -> None:
    result = df[["id", "source", "label"]].copy()
    result["pred"] = y_pred

    fn = result[(result["label"] == 1) & (result["pred"] == 0)]
    fp = result[(result["label"] == 0) & (result["pred"] == 1)]

    print("\n" + "=" * 60)
    print(f"{name} ERROR ANALYSIS")
    print("=" * 60)

    print(f"\nFalse Negatives (phishing → predicted legit): {len(fn)}")
    if len(fn) > 0:
        print(fn["source"].value_counts().to_string())

    print(f"\nFalse Positives (legit → predicted phishing): {len(fp)}")
    if len(fp) > 0:
        print(fp["source"].value_counts().to_string())

# Aggregate LIME to see the words most common across the different e-mails
def print_aggregated_lime(model: Pipeline, test_df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("Most common words associated with the prediction (using LIME)")
    print("=" * 60)
 
    explainer = LimeTextExplainer(class_names=["legit", "phishing"])
    predict_fn = lambda texts: model.predict_proba(texts)
    texts = test_df["text_input"].tolist()
    labels = test_df["label"].values
 
    for class_label, class_name in [(1, "PHISHING"), (0, "LEGIT")]:
        counts  = {}
        indices = [i for i, l in enumerate(labels) if l == class_label]
 
        for i, idx in enumerate(indices):
            exp = explainer.explain_instance(
                texts[idx], predict_fn, num_features=10, num_samples=200, labels=(1,)
            )
            for word, weight in exp.as_list(label=1):
                if (class_label == 1 and weight > 0) or (class_label == 0 and weight < 0):
                    word = word.lower().strip(".,!?\"'")
                    if len(word) > 2:
                        counts[word] = counts.get(word, 0) + 1
 
        total   = len(indices)
        sorted_ = sorted(counts.items(), key=lambda x: x[1], reverse=True)
 
        print(f"\n  Top 10 words: {class_name}")
        print(f"  {'Word':<25} {'Count':>6} {'% of emails':>12}")
        print("  " + "-" * 45)
        for word, count in sorted_[:10]:
            print(f"  {word:<25} {count:>6}   {count / total * 100:>9.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True, help="Path to train.csv")
    parser.add_argument("--val",   type=Path, required=True, help="Path to val.csv")
    parser.add_argument("--test",  type=Path, required=True, help="Path to test.csv")
    parser.add_argument("--max-features", type=int, default=50_000,help="Max TF-IDF vocabulary size (default: 50000)")
    parser.add_argument("--alpha", type=float, default=1.0,help="MultinomialNB smoothing parameter (default: 1.0)")
    args = parser.parse_args()

    train_df = load_split(args.train, "train")
    val_df   = load_split(args.val, "val")
    test_df  = load_split(args.test, "test")

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print_dataset_summary("TRAIN", train_df)
    print_dataset_summary("VAL",   val_df)
    print_dataset_summary("TEST",  test_df)

    print(f"TF-IDF max features: {args.max_features}")
    print(f"MultinomialNB alpha: {args.alpha}")

    X_train = train_df["text_input"]
    y_train = train_df["label"].values

    X_val = val_df["text_input"]
    y_val = val_df["label"].values

    X_test = test_df["text_input"]
    y_test = test_df["label"].values

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), # unigrams + bigrams
            max_features=args.max_features,
            sublinear_tf=True, # sublinear tf scaling to deal with high frequencies
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{2,}", # ignore single characters
            min_df=2, # ignore terms appearing in less than 2 docs
        )),
        ("nb", MultinomialNB(alpha=args.alpha)),
    ])

    model.fit(X_train, y_train)

    # Save trained model for stress testing
    model_path = Path("./pkl_models/nb_tfidf.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    train_pred = model.predict(X_train)
    train_prob = model.predict_proba(X_train)[:, 1]

    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]

    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]
    
    print_metrics_block("TRAIN", y_train, train_pred, train_prob)
    print_metrics_block("VAL",   y_val,   val_pred,   val_prob)
    print_metrics_block("TEST",  y_test,  test_pred,  test_prob)

    # To understand better the behaviour of the model
    print_error_analysis("TRAIN", train_df, train_pred)
    print_error_analysis("VAL",   val_df,   val_pred)
    print_error_analysis("TEST",  test_df,  test_pred)

    print_aggregated_lime(model, test_df)


if __name__ == "__main__":
    main()