#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

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


NB_SEARCH_SPACE = {
    "var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1],
}

TFIDF_SEARCH_SPACE = {
    "text_mode":    ["text_only", "text_and_subject"],
    "alpha":        [0.1, 0.5, 1.0, 2.0],
    "max_features": [10_000, 30_000, 50_000],
    "ngram_range":  [(1, 1), (1, 2), (1, 3)],
    "min_df":       [1, 2, 5],
}


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{split_name} file not found: {path}")

    df = pd.read_csv(path)

    required = FEATURE_COLUMNS + ["label", "id", "source", "subject", "text"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{split_name} missing columns: {missing}")

    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df["subject"] = df["subject"].fillna("")
    df["text"] = df["text"].fillna("")
    return df


def build_text_input(df: pd.DataFrame, text_mode: str) -> list[str]:
    if text_mode == "text_only":
        return df["text"].tolist()
    else:
        return (df["subject"] + " " + df["text"]).tolist()


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "val_recall": recall_score(y_true, y_pred, zero_division=0),
        "val_f1": f1_score(y_true, y_pred, zero_division=0),
        "val_acc": accuracy_score(y_true, y_pred),
        "val_roc_auc": roc_auc_score(y_true, y_prob),
    }


# Gaussian NB 

def tune_nb(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  1. NB LEXICAL — EXHAUSTIVE SEARCH (var_smoothing)")
    print("=" * 70)

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["label"].values
    X_val = val_df[FEATURE_COLUMNS].values
    y_val = val_df["label"].values

    results = []

    for vs in NB_SEARCH_SPACE["var_smoothing"]:
        model = GaussianNB(var_smoothing=vs)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        val_prob = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, val_pred, val_prob)

        print(f"  var_smoothing={vs:.0e}  "
              f"val_recall={metrics['val_recall']:.4f}  "
              f"val_f1={metrics['val_f1']:.4f}  "
              f"val_acc={metrics['val_acc']:.4f}")

        results.append({"var_smoothing": vs, **metrics})

    return pd.DataFrame(results).sort_values("val_recall", ascending=False)


# NB + TF-IDF (random search)


def tune_nb_tfidf(train_df: pd.DataFrame, val_df: pd.DataFrame,n_trials: int, seed: int) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  2. NB + TF-IDF — RANDOM SEARCH")
    print("=" * 70)

    random.seed(seed) # to make sure it is reproducible
    y_train = train_df["label"].values
    y_val = val_df["label"].values

    configs = [
        {k: random.choice(v) for k, v in TFIDF_SEARCH_SPACE.items()}
        for _ in range(n_trials)
    ]

    results = []

    for i, config in enumerate(configs, 1):
        print(f"\n  [Trial {i:2d}/{n_trials}]  "
              f"text={config['text_mode']:16s}  "
              f"alpha={config['alpha']}  "
              f"max_feat={config['max_features']:6d}  "
              f"ngram={config['ngram_range']}  "
              f"min_df={config['min_df']}")

        t0 = time.time()

        train_texts = build_text_input(train_df, config["text_mode"])
        val_texts = build_text_input(val_df,   config["text_mode"])

        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=config["ngram_range"],
                max_features=config["max_features"],
                sublinear_tf=True,
                strip_accents="unicode",
                analyzer="word",
                token_pattern=r"\w{2,}",
                min_df=config["min_df"],
            )),
            ("nb", MultinomialNB(alpha=config["alpha"])),
        ])

        model.fit(train_texts, y_train)
        val_pred = model.predict(val_texts)
        val_prob = model.predict_proba(val_texts)[:, 1]
        metrics  = compute_metrics(y_val, val_pred, val_prob)
        elapsed  = time.time() - t0

        print(f"             val_recall={metrics['val_recall']:.4f}  "
              f"val_f1={metrics['val_f1']:.4f}  "
              f"val_acc={metrics['val_acc']:.4f}  "
              f"({elapsed:.1f}s)")

        results.append({"trial": i, **config, **metrics})

    return pd.DataFrame(results).sort_values("val_recall", ascending=False)


# Best

def print_best(name: str, results_df: pd.DataFrame, search_space: dict, n_trials: int, val_size: int) -> None:
    best = results_df.iloc[0]

    print("\n" + "=" * 70)
    print(f"  BEST {name}")
    print("=" * 70)
    for col in results_df.columns:
        if col not in ("trial",):
            print(f"  {col:15s}: {best[col]}")

# Main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    type=Path, required=True)
    parser.add_argument("--val",      type=Path, required=True)
    parser.add_argument("--test",     type=Path, required=True,
                        help="Reserved for final evaluation — not used here")
    parser.add_argument("--n-trials", type=int, default=25,
                        help="Random trials for NB+TF-IDF (default: 25)")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()


    train_df = load_split(args.train, "train")
    val_df = load_split(args.val, "val")

    nb_results = tune_nb(train_df, val_df)

    print("\n\n  RESULTS NB LEXICAL (sorted by val_recall)")
    print(nb_results.to_string(index=False))
    print_best("NB LEXICAL", nb_results, NB_SEARCH_SPACE, n_trials=len(NB_SEARCH_SPACE["var_smoothing"]), val_size=len(val_df))

    tfidf_results = tune_nb_tfidf(train_df, val_df, args.n_trials, args.seed)

    print("\n\n  RESULTS NB + TF-IDF (sorted by val_recall)")
    print(tfidf_results[["trial", "text_mode", "val_recall", "val_f1", "val_acc","val_roc_auc", "alpha", "max_features", "ngram_range", "min_df",]].to_string(index=False))
    print_best("NB + TF-IDF", tfidf_results, TFIDF_SEARCH_SPACE,n_trials=args.n_trials, val_size=len(val_df))


if __name__ == "__main__":
    main()
