#!/usr/bin/env python3
"""
build_dataset_splits.py

Build train/val/test splits from already parsed mailbox CSVs.

Assumptions
-----------
- Deduplication was already done in parse_mailbox.py
- Labels are already assigned:
    enron   -> 0
    nazario -> 1
- Feature columns already exist in the CSVs

What this script does
---------------------
1. Load enron_parsed.csv and nazario_parsed.csv
2. Keep only rows with the required columns and valid values
3. Merge both datasets
4. Create stratified train/val/test splits
5. Downsample within each split to keep classes balanced
6. Print discard statistics and split statistics
7. Save train.csv / val.csv / test.csv / metadata.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = [
    "id",
    "label",
    "source",
    "subject",
    "text",
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

TEXT_COLUMNS = ["id", "source", "subject", "text"]

NUMERIC_COLUMNS = [
    # "label",
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


def print_block(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_df_summary(name: str, df: pd.DataFrame) -> None:
    print(f"{name:12s} rows={len(df):5d}  labels={df['label'].value_counts().sort_index().to_dict()}  "
          f"sources={df['source'].value_counts().to_dict()}")


def load_and_clean_csv(
    path: Path,
    *,
    expected_label: int,
    expected_source: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Load one parsed CSV and discard only clearly invalid rows.
    Returns cleaned dataframe + discard stats.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    discard_stats = Counter()

    # Missing required columns = hard failure, not row-level discard
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    original_rows = len(df)

    # Normalize text-like columns
    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("").astype(str)

    # Strip whitespace
    for col in TEXT_COLUMNS:
        df[col] = df[col].str.strip()

    # Convert numeric columns 
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Row-level validity masks
    invalid_id = df["id"].eq("")
    invalid_source = df["source"].str.lower().ne(expected_source.lower())
    invalid_label = df["label"].isna() | df["label"].ne(expected_label)
    empty_text = df["text"].eq("")
    empty_subject_and_text = df["subject"].eq("") & df["text"].eq("")
    numeric_nan = df[NUMERIC_COLUMNS].isna().any(axis=1)

    discard_masks = {
        "empty_id": invalid_id,
        "wrong_source": invalid_source,
        "wrong_label": invalid_label,
        "empty_text": empty_text,
        "empty_subject_and_text": empty_subject_and_text,
        "missing_numeric_feature": numeric_nan,
    }

    # Assign one primary discard reason per row
    reason_per_row = pd.Series("", index=df.index, dtype="object")
    for reason, mask in discard_masks.items():
        reason_per_row = reason_per_row.mask((reason_per_row == "") & mask, reason)

    keep_mask = reason_per_row.eq("")
    dropped_mask = ~keep_mask

    for reason in reason_per_row[dropped_mask]:
        discard_stats[reason] += 1

    cleaned = df.loc[keep_mask, REQUIRED_COLUMNS].copy()

    # Cast numeric columns after filtering
    cleaned["label"] = cleaned["label"].astype(int)
    for col in NUMERIC_COLUMNS:
        if col != "label":
            cleaned[col] = cleaned[col].astype(int)

    discard_stats["loaded_rows"] = original_rows
    discard_stats["kept_rows"] = len(cleaned)
    discard_stats["discarded_rows"] = original_rows - len(cleaned)

    return cleaned.reset_index(drop=True), dict(discard_stats)


def stratified_three_way_split(
    df: pd.DataFrame,
    *,
    train_size: float,
    val_size: float,
    test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train/val/test with stratification.
    """
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"train/val/test must sum to 1.0, got {total}")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_size),
        stratify=df["label"],
        random_state=random_seed,
    )

    # Of the remaining temp set, split into val/test proportionally
    val_fraction_of_temp = val_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_fraction_of_temp),
        stratify=temp_df["label"],
        random_state=random_seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def balance_split(df: pd.DataFrame, *, random_seed: int) -> tuple[pd.DataFrame, dict]:
    """
    Downsample majority class inside one split.
    """
    counts = df["label"].value_counts().to_dict()
    if set(counts.keys()) != {0, 1}:
        raise ValueError(f"Expected both classes in split, found {counts}")

    minority_size = min(counts.values())
    out_parts = []
    dropped = 0

    for label_value, group in df.groupby("label", sort=True):
        if len(group) > minority_size:
            dropped += len(group) - minority_size
            group = group.sample(n=minority_size, random_state=random_seed)
        out_parts.append(group)

    balanced = pd.concat(out_parts, axis=0)
    balanced = balanced.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    stats = {
        "before_counts": counts,
        "after_counts": balanced["label"].value_counts().sort_index().to_dict(),
        "discarded_for_balance": dropped,
    }
    return balanced, stats


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enron", type=Path, required=True, help="Path to enron_parsed.csv")
    parser.add_argument("--nazario", type=Path, required=True, help="Path to nazario_parsed.csv")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for train/val/test CSVs")
    parser.add_argument("--train-size", type=float, default=0.8, help="Train fraction")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test fraction")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    enron_df, enron_discards = load_and_clean_csv(
        args.enron,
        expected_label=0,
        expected_source="enron",
    )
    nazario_df, nazario_discards = load_and_clean_csv(
        args.nazario,
        expected_label=1,
        expected_source="nazario",
    )

    print_block("INPUT DATA AFTER CLEANING")
    print_df_summary("ENRON", enron_df)
    print_df_summary("NAZARIO", nazario_df)

    print_block("DISCARDED ROWS DURING INPUT CLEANING")
    print("ENRON:")
    for k, v in enron_discards.items():
        print(f"  {k}: {v}")
    print("NAZARIO:")
    for k, v in nazario_discards.items():
        print(f"  {k}: {v}")

    merged = pd.concat([enron_df, nazario_df], axis=0, ignore_index=True)

    print_block("MERGED DATASET BEFORE SPLIT")
    print_df_summary("MERGED", merged)

    train_df, val_df, test_df = stratified_three_way_split(
        merged,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )

    print_block("SPLITS BEFORE BALANCING")
    print_df_summary("TRAIN", train_df)
    print_df_summary("VAL", val_df)
    print_df_summary("TEST", test_df)

    train_bal, train_balance_stats = balance_split(train_df, random_seed=args.random_seed)
    val_bal, val_balance_stats = balance_split(val_df, random_seed=args.random_seed)
    test_bal, test_balance_stats = balance_split(test_df, random_seed=args.random_seed)

    print_block("DISCARDED ROWS FOR CLASS BALANCING")
    print(f"TRAIN discarded_for_balance: {train_balance_stats['discarded_for_balance']}")
    print(f"VAL   discarded_for_balance: {val_balance_stats['discarded_for_balance']}")
    print(f"TEST  discarded_for_balance: {test_balance_stats['discarded_for_balance']}")

    total_input_discards = enron_discards["discarded_rows"] + nazario_discards["discarded_rows"]
    total_balance_discards = (
        train_balance_stats["discarded_for_balance"]
        + val_balance_stats["discarded_for_balance"]
        + test_balance_stats["discarded_for_balance"]
    )

    print_block("FINAL BALANCED SPLITS")
    print_df_summary("TRAIN", train_bal)
    print_df_summary("VAL", val_bal)
    print_df_summary("TEST", test_bal)

    print_block("FINAL DISCARD SUMMARY")
    print(f"Discarded during input cleaning: {total_input_discards}")
    print(f"Discarded for balancing:         {total_balance_discards}")
    print(f"Discarded total:                {total_input_discards + total_balance_discards}")

    out_dir = args.out_dir
    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"
    meta_path = out_dir / "dataset_metadata.json"

    save_csv(train_bal, train_path)
    save_csv(val_bal, val_path)
    save_csv(test_bal, test_path)

    metadata = {
        "input_files": {
            "enron": str(args.enron),
            "nazario": str(args.nazario),
        },
        "assumption": "Deduplication already performed upstream in parse_mailbox.py",
        "required_columns": REQUIRED_COLUMNS,
        "random_seed": args.random_seed,
        "split_fractions": {
            "train": args.train_size,
            "val": args.val_size,
            "test": args.test_size,
        },
        "input_cleaning_discards": {
            "enron": enron_discards,
            "nazario": nazario_discards,
            "total_discarded": total_input_discards,
        },
        "balancing_discards": {
            "train": train_balance_stats,
            "val": val_balance_stats,
            "test": test_balance_stats,
            "total_discarded_for_balance": total_balance_discards,
        },
        "final_rows": {
            "train": len(train_bal),
            "val": len(val_bal),
            "test": len(test_bal),
        },
        "final_label_counts": {
            "train": train_bal["label"].value_counts().sort_index().to_dict(),
            "val": val_bal["label"].value_counts().sort_index().to_dict(),
            "test": test_bal["label"].value_counts().sort_index().to_dict(),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print_block("SAVED FILES")
    print(f"Train:    {train_path.resolve()}")
    print(f"Val:      {val_path.resolve()}")
    print(f"Test:     {test_path.resolve()}")
    print(f"Metadata: {meta_path.resolve()}")


if __name__ == "__main__":
    main()