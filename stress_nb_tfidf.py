#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import recall_score, f1_score, accuracy_score

MODEL_PATH = Path("./pkl_models/nb_tfidf.pkl")

def load_test(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df["subject"] = df["subject"].fillna("")
    df["text"] = df["text"].fillna("")
    df["text_input"] = df["subject"] + " " + df["text"]
    return df

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--obfuscated",  type=Path, required=True)
    args = parser.parse_args()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    test_df = load_test(args.test)
    obfuscated_df = load_test(args.obfuscated)    
    y_true  = test_df["label"].values

    # Baseline
    baseline_pred = model.predict(test_df["text_input"])
    baseline_recall = recall_score(y_true, baseline_pred, zero_division=0)
    baseline_f1 = f1_score(y_true, baseline_pred, zero_division=0)
    baseline_acc = accuracy_score(y_true, baseline_pred)

    print(f"\n{'Technique':<35} {'Recall':>8} {'F1':>8} {'Accuracy':>10} {'Recall drop':>12}")
    print("-" * 85)
    print(f"  {'Original (no obfuscation)':<33} "
          f"{baseline_recall:>8.4f} {baseline_f1:>8.4f} {baseline_acc:>10.4f} {'—':>12}  —")

    # Obfuscated
    obfuscated_pred = model.predict(obfuscated_df["text_input"])
    obfuscated_recall = recall_score(y_true, obfuscated_pred, zero_division=0)
    obfuscated_f1 = f1_score(y_true, obfuscated_pred, zero_division=0)
    obfuscated_acc = accuracy_score(y_true, obfuscated_pred)

    print(f"  {'Obfuscation':<33} "
          f"{obfuscated_recall:>8.4f} {obfuscated_f1:>8.4f} {obfuscated_acc:>10.4f} {(baseline_recall-obfuscated_recall)/baseline_recall*100:.2f} %")



if __name__ == "__main__":
    main()