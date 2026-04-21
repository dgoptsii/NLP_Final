#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import pandas as pd
from sklearn.metrics import recall_score, f1_score, accuracy_score

MODEL_PATH = Path("./pkl_models/nb_tfidf.pkl")
OBFUSCATION_RATE = 0.25
SEED = 42

CHAR_SUB = {
    "a": "@", "e": "3", "i": "1", "o": "0", "s": "5",
    "A": "@", "E": "3", "I": "1", "O": "0", "S": "5",
}

HOMOGLYPH_SUB = {
    "l": "I", "o": "ο", "a": "а", "e": "е", "p": "р", "c": "с",
}

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

def obfuscate_word(word: str, mapping: dict) -> str:
    return "".join(mapping.get(ch, ch) for ch in word)


def obfuscate_text(text: str, mapping: dict, rate: float, rng: random.Random) -> str:
    words = text.split()
    indices = rng.sample(range(len(words)), max(1, int(len(words) * rate)))
    for idx in indices:
        words[idx] = obfuscate_word(words[idx], mapping)
    return " ".join(words)


# Obfuscate the whole dataframe
def obfuscate_df(df: pd.DataFrame, mapping: dict, rate: float, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    result = df.copy()
    mask = result["label"] == 1
    result.loc[mask, "text_input"] = result.loc[mask, "text_input"].apply(
        lambda t: obfuscate_text(t, mapping, rate, rng)
    )
    return result


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
    args = parser.parse_args()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    test_df = load_test(args.test)
    y_true  = test_df["label"].values

    print("\n" + "=" * 65)
    print("  NB + TF-IDF STRESS TEST")
    print("=" * 65)
    print(f"  {int(OBFUSCATION_RATE * 100)}% of words per phishing email are obfuscated")
    print("=" * 65)

    # Baseline
    baseline_pred = model.predict(test_df["text_input"])
    baseline_recall = recall_score(y_true, baseline_pred, zero_division=0)
    baseline_f1 = f1_score(y_true, baseline_pred, zero_division=0)
    baseline_acc = accuracy_score(y_true, baseline_pred)

    print(f"\n{'Technique':<35} {'Recall':>8} {'F1':>8} {'Accuracy':>10} {'Recall drop':>12}")
    print("-" * 85)
    print(f"  {'Original (no obfuscation)':<33} "
          f"{baseline_recall:>8.4f} {baseline_f1:>8.4f} {baseline_acc:>10.4f} {'—':>12}  —")

    for label, mapping in [
        ("Char substitution (e→3, i→1)", CHAR_SUB),
        ("Homoglyph (l→I, o→ο, a→а)   ", HOMOGLYPH_SUB),
    ]:
        obf_df = obfuscate_df(test_df, mapping, OBFUSCATION_RATE, SEED)
        obf_pred = model.predict(obf_df["text_input"])

        recall = recall_score(y_true, obf_pred, zero_division=0)
        f1 = f1_score(y_true, obf_pred, zero_division=0)
        acc = accuracy_score(y_true, obf_pred)
        drop = (baseline_recall - recall) / baseline_recall * 100

        print(f"  {label:<33} {recall:>8.4f} {f1:>8.4f} {acc:>10.4f} {drop:>10.1f}%")

    print("=" * 65)


if __name__ == "__main__":
    main()