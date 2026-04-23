import argparse
import string
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from transformer_common import (
    EmailDataset,
    build_combined_text,
    evaluate_checkpoint_on_csv,
    load_checkpoint,
    predict_proba,
)


# Project root: .../NLP
ROOT = Path(__file__).resolve().parent.parent


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_subprocess(cmd: List[str], log_path: Path, name: str) -> float:
    print("\n" + "=" * 100, flush=True)
    print(f"RUNNING: {name}", flush=True)
    print("Command:", " ".join(cmd), flush=True)
    print(f"Log: {log_path}", flush=True)
    print("=" * 100, flush=True)

    start = time.time()

    # Force unbuffered Python output so epoch prints show up live
    real_cmd = [cmd[0], "-u", *cmd[1:]]

    with open(log_path, "w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            real_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            text=True,
            bufsize=1,
        )

        if process.stdout is None:
            raise RuntimeError(f"Failed to capture output for: {name}")

        for line in process.stdout:
            print(f"[{name}] {line}", end="", flush=True)
            log_f.write(line)
            log_f.flush()

        process.wait()

    duration = time.time() - start

    if process.returncode != 0:
        raise RuntimeError(f"Step failed: {name}. See log: {log_path}")

    print(f"\nFinished {name} in {duration:.2f} sec ({duration / 60:.2f} min)", flush=True)
    return duration


def run_training_suite(args, results_dir: Path, logs_dir: Path) -> pd.DataFrame:
    train_jobs = [
        {
            "name": "bert_text_only",
            "pretty": "BERT text-only",
            "script": "transformers/train_bert_hybrid.py",
            "use_features": False,
        },
        {
            "name": "bert_hybrid",
            "pretty": "BERT hybrid",
            "script": "transformers/train_bert_hybrid.py",
            "use_features": True,
        },
        {
            "name": "distilbert_text_only",
            "pretty": "DistilBERT text-only",
            "script": "transformers/train_distilbert_hybrid.py",
            "use_features": False,
        },
        {
            "name": "distilbert_hybrid",
            "pretty": "DistilBERT hybrid",
            "script": "transformers/train_distilbert_hybrid.py",
            "use_features": True,
        },
    ]

    rows = []
    for job in train_jobs:
        results_csv = results_dir / f"train_{job['name']}.csv"
        cmd = [
            sys.executable,
            job["script"],
            "--splits_dir",
            args.splits_dir,
            "--save_dir",
            args.save_dir,
            "--max_len",
            str(args.max_len),
            "--batch_size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--dropout",
            str(args.dropout),
            "--patience",
            str(args.patience),
            "--random_state",
            str(args.random_state),
            "--save_results_csv",
            str(results_csv),
        ]
        if job["use_features"]:
            cmd.append("--use_features")

        wall_time = run_subprocess(
            cmd,
            logs_dir / f"train_{job['name']}.log",
            f"TRAIN {job['pretty']}",
        )

        train_df = pd.read_csv(results_csv)
        row = train_df.iloc[0].to_dict()
        row["run_name"] = job["name"]
        row["pretty_name"] = job["pretty"]
        row["wall_time_runner_sec"] = wall_time
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "training_summary.csv", index=False)
    return df


def run_obfuscation(results_dir: Path, logs_dir: Path) -> float:
    return run_subprocess(
        [sys.executable, "./scripts/obfuscate.py"],
        logs_dir / "obfuscate.log",
        "GENERATE OBFUSCATED TEST SET",
    )


def stress_one(checkpoint_path: str, test_csv: str, obf_csv: str, batch_size: int) -> Dict:
    _, _, _, baseline_metrics, payload = evaluate_checkpoint_on_csv(
        checkpoint_path=checkpoint_path,
        csv_path=test_csv,
        batch_size=batch_size,
    )
    _, _, _, obf_metrics, _ = evaluate_checkpoint_on_csv(
        checkpoint_path=checkpoint_path,
        csv_path=obf_csv,
        batch_size=batch_size,
    )

    baseline_recall = baseline_metrics["recall"]
    obf_recall = obf_metrics["recall"]
    recall_drop_pct = 0.0 if baseline_recall == 0 else (baseline_recall - obf_recall) / baseline_recall * 100.0

    return {
        "model": payload["model_name"],
        "features": bool(payload.get("use_features", False)),
        "orig_accuracy": baseline_metrics["accuracy"],
        "orig_recall": baseline_metrics["recall"],
        "orig_f1": baseline_metrics["f1"],
        "obf_accuracy": obf_metrics["accuracy"],
        "obf_recall": obf_metrics["recall"],
        "obf_f1": obf_metrics["f1"],
        "recall_drop_pct": recall_drop_pct,
    }


def run_stress_suite(training_df: pd.DataFrame, args, results_dir: Path) -> pd.DataFrame:
    rows = []
    for _, train_row in training_df.iterrows():
        checkpoint_path = train_row["checkpoint_path"]
        print("\n" + "=" * 100, flush=True)
        print(f"RUNNING STRESS TEST: {train_row['pretty_name']}", flush=True)
        print(f"Checkpoint: {checkpoint_path}", flush=True)
        print("=" * 100, flush=True)

        start = time.time()
        stress_row = stress_one(
            checkpoint_path=checkpoint_path,
            test_csv=args.test_csv,
            obf_csv=args.obfuscated_csv,
            batch_size=args.batch_size,
        )
        stress_row["stress_time_sec"] = time.time() - start
        stress_row["checkpoint_path"] = checkpoint_path
        stress_row["run_name"] = train_row["run_name"]
        stress_row["pretty_name"] = train_row["pretty_name"]

        print(
            f"Finished stress test for {train_row['pretty_name']} "
            f"in {stress_row['stress_time_sec']:.2f} sec",
            flush=True,
        )
        rows.append(stress_row)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "stress_summary.csv", index=False)
    return df


class InferenceDataset(EmailDataset):
    def __init__(self, texts: List[str], tokenizer, max_len: int):
        df = pd.DataFrame({"combined_text": texts, "label": [0] * len(texts)})
        super().__init__(df=df, tokenizer=tokenizer, max_len=max_len, use_features=False, feature_columns=[])


def select_rows_for_lime(df: pd.DataFrame, preds: np.ndarray, target_label: int, max_examples: int) -> List[int]:
    y_true = df["label"].astype(int).values
    idxs = np.where((y_true == target_label) & (preds == target_label))[0].tolist()
    if len(idxs) < max_examples:
        fallback = np.where(y_true == target_label)[0].tolist()
        seen = set(idxs)
        for idx in fallback:
            if idx not in seen:
                idxs.append(idx)
                seen.add(idx)
            if len(idxs) >= max_examples:
                break
    return idxs[:max_examples]


def normalize_lime_token(token: str) -> str:
    cleaned = token.lower().strip().strip(string.punctuation + "“”‘’`")
    return " ".join(cleaned.split())


def aggregate_lime_words(
    checkpoint_path: str,
    csv_path: str,
    target_label: int,
    num_examples: int,
    num_features: int,
    batch_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError as exc:
        raise RuntimeError("LIME is not installed. Run: pip install lime") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload, model, tokenizer, _ = load_checkpoint(checkpoint_path, device)

    if payload.get("use_features"):
        raise RuntimeError("Aggregated LIME is configured for text-only models. Hybrid checkpoints are skipped.")

    df = pd.read_csv(csv_path)
    df = build_combined_text(df)
    dataset = EmailDataset(df, tokenizer, payload.get("max_len", 256), False, [])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probs_full, preds = predict_proba(model, loader, device)

    chosen_rows = select_rows_for_lime(df, preds, target_label=target_label, max_examples=num_examples)
    explainer = LimeTextExplainer(class_names=["legitimate", "phishing"])

    token_scores = defaultdict(list)
    token_email_counts = defaultdict(int)
    example_rows = []

    def predictor(texts: List[str]):
        infer_dataset = InferenceDataset(texts=texts, tokenizer=tokenizer, max_len=payload.get("max_len", 256))
        infer_loader = DataLoader(infer_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        probs, _ = predict_proba(model, infer_loader, device)
        return probs

    for i, row_idx in enumerate(chosen_rows, start=1):
        print(
            f"LIME [{target_label}] example {i}/{len(chosen_rows)} "
            f"from checkpoint: {Path(checkpoint_path).name}",
            flush=True,
        )

        text = str(df.iloc[row_idx]["combined_text"])
        probs = probs_full[row_idx]
        pred_label = int(np.argmax(probs))
        explanation = explainer.explain_instance(
            text,
            predictor,
            num_features=num_features,
            labels=[target_label],
        )
        raw_pairs = explanation.as_list(label=target_label)

        # filtered_pairs = []
        # seen_this_email = set()
        # for token, weight in raw_pairs:
        #     if weight <= 0:
        #         continue
        #     norm_token = normalize_lime_token(token)
        #     if len(norm_token) <= 2:
        #         continue
        #     filtered_pairs.append((norm_token, float(weight)))
        pos_pairs = []
        neg_pairs = []
        seen_this_email = set()

        for token, weight in raw_pairs:
            norm_token = normalize_lime_token(token)
            if len(norm_token) <= 2:
                continue

            if weight > 0:
                pos_pairs.append((norm_token, float(weight)))
            else:
                neg_pairs.append((norm_token, float(weight)))

            token_scores[norm_token].append(float(weight))

            if norm_token not in seen_this_email:
                token_email_counts[norm_token] += 1
                seen_this_email.add(norm_token)
            token_scores[norm_token].append(float(weight))

            if norm_token not in seen_this_email:
                token_email_counts[norm_token] += 1
                seen_this_email.add(norm_token)

        example_rows.append(
            {
                "row": int(row_idx),
                "true_label": int(df.iloc[row_idx]["label"]),
                "pred_label": pred_label,
                "p_legitimate": float(probs[0]),
                "p_phishing": float(probs[1]),
                "top_positive": "; ".join([f"{tok}:{wt:+.4f}" for tok, wt in sorted(pos_pairs, key=lambda x: -x[1])[:num_features]]),
                "top_negative": "; ".join([f"{tok}:{wt:+.4f}" for tok, wt in sorted(neg_pairs, key=lambda x: x[1])[:num_features]]),
            }
        )

    total_examples = max(len(chosen_rows), 1)
    agg_rows = []
    for token, weights in token_scores.items():
        count = token_email_counts[token]
        pos_weights = [w for w in weights if w > 0]
        neg_weights = [w for w in weights if w < 0]

        agg_rows.append({
            "word": token,
            "count": count,
            "pct_of_emails": 100.0 * count / total_examples,
            "mean_pos_weight": float(np.mean(pos_weights)) if pos_weights else 0.0,
            "mean_neg_weight": float(np.mean(neg_weights)) if neg_weights else 0.0,
            "mean_weight": float(np.mean(weights)) if weights else 0.0,
            "mean_abs_weight": float(np.mean(np.abs(weights))),
        })

    agg_df = pd.DataFrame(agg_rows)
    if not agg_df.empty:
        agg_df = agg_df.sort_values(
            ["count", "pct_of_emails", "mean_abs_weight", "mean_weight"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    examples_df = pd.DataFrame(example_rows)
    return agg_df, examples_df


def run_lime_suite(training_df: pd.DataFrame, args, results_dir: Path) -> Tuple[pd.DataFrame, bool]:
    text_only_df = training_df[training_df["features"] == False].copy()  # noqa: E712
    lime_summary_rows = []
    lime_available = True

    for _, row in text_only_df.iterrows():
        model_stub = row["run_name"]
        checkpoint_path = row["checkpoint_path"]

        for label_value, label_name in [(1, "phishing"), (0, "legitimate")]:
            print("\n" + "=" * 100, flush=True)
            print(f"RUNNING LIME: {row['pretty_name']} | target={label_name}", flush=True)
            print("=" * 100, flush=True)

            start = time.time()
            try:
                agg_df, examples_df = aggregate_lime_words(
                    checkpoint_path=checkpoint_path,
                    csv_path=args.test_csv,
                    target_label=label_value,
                    num_examples=args.lime_examples_per_class,
                    num_features=args.lime_num_features,
                    batch_size=args.batch_size,
                )
            except RuntimeError as exc:
                print(f"Skipping LIME: {exc}", flush=True)
                lime_available = False
                return pd.DataFrame(), lime_available

            elapsed = time.time() - start
            agg_path = results_dir / f"lime_{model_stub}_{label_name}_top_words.csv"
            ex_path = results_dir / f"lime_{model_stub}_{label_name}_examples.csv"
            agg_df.head(args.top_k_words).to_csv(agg_path, index=False)
            examples_df.to_csv(ex_path, index=False)

            print(f"Saved LIME top words to: {agg_path}", flush=True)
            print(f"Saved LIME examples to: {ex_path}", flush=True)
            print(f"Finished LIME in {elapsed:.2f} sec", flush=True)

            lime_summary_rows.append(
                {
                    "run_name": model_stub,
                    "pretty_name": row["pretty_name"],
                    "target_class": label_name,
                    "lime_time_sec": elapsed,
                    "output_top_words_csv": str(agg_path),
                    "output_examples_csv": str(ex_path),
                }
            )

    lime_summary_df = pd.DataFrame(lime_summary_rows)
    lime_summary_df.to_csv(results_dir / "lime_summary.csv", index=False)
    return lime_summary_df, lime_available


def build_report_tables(training_df: pd.DataFrame, stress_df: pd.DataFrame, results_dir: Path) -> Dict[str, pd.DataFrame]:
    merged = training_df.merge(
        stress_df[
            [
                "run_name",
                "orig_accuracy",
                "orig_recall",
                "orig_f1",
                "obf_accuracy",
                "obf_recall",
                "obf_f1",
                "recall_drop_pct",
                "stress_time_sec",
            ]
        ],
        on="run_name",
        how="left",
    )

    merged["feature_setting"] = np.where(merged["features"], "text+features", "text-only")
    merged["family"] = np.where(merged["model"].str.contains("distil", case=False), "DistilBERT", "BERT")

    compare_cols = [
        "family",
        "feature_setting",
        "best_val_acc",
        "test_acc",
        "orig_accuracy",
        "orig_recall",
        "orig_f1",
        "train_time_sec",
        "total_time_sec",
        "stress_time_sec",
    ]
    features_vs_text = merged[compare_cols].sort_values(["family", "feature_setting"]).reset_index(drop=True)
    features_vs_text.to_csv(results_dir / "table_features_vs_text.csv", index=False)

    best_rows = merged.sort_values(
        ["family", "orig_f1", "orig_recall", "orig_accuracy"],
        ascending=[True, False, False, False],
    )
    best_per_family = best_rows.groupby("family", as_index=False).head(1).copy()

    best_original = best_per_family[
        [
            "family",
            "feature_setting",
            "orig_accuracy",
            "orig_recall",
            "orig_f1",
            "train_time_sec",
            "stress_time_sec",
        ]
    ].reset_index(drop=True)
    best_original.to_csv(results_dir / "table_best_original.csv", index=False)

    best_obfuscated = best_per_family[
        [
            "family",
            "feature_setting",
            "obf_accuracy",
            "obf_recall",
            "obf_f1",
            "recall_drop_pct",
            "train_time_sec",
            "stress_time_sec",
        ]
    ].reset_index(drop=True)
    best_obfuscated.to_csv(results_dir / "table_best_obfuscated.csv", index=False)

    timing_summary = merged[
        [
            "pretty_name",
            "feature_setting",
            "train_time_sec",
            "total_time_sec",
            "stress_time_sec",
            "wall_time_runner_sec",
        ]
    ].sort_values("pretty_name").reset_index(drop=True)
    timing_summary.to_csv(results_dir / "table_timing_summary.csv", index=False)

    final_summary = merged[
        [
            "pretty_name",
            "feature_setting",
            "best_val_acc",
            "test_acc",
            "orig_accuracy",
            "orig_recall",
            "orig_f1",
            "obf_accuracy",
            "obf_recall",
            "obf_f1",
            "recall_drop_pct",
            "train_time_sec",
            "stress_time_sec",
            "wall_time_runner_sec",
        ]
    ].sort_values("pretty_name").reset_index(drop=True)
    final_summary.to_csv(results_dir / "final_summary.csv", index=False)

    return {
        "features_vs_text": features_vs_text,
        "best_original": best_original,
        "best_obfuscated": best_obfuscated,
        "timing_summary": timing_summary,
        "final_summary": final_summary,
    }


def print_quick_paths(results_dir: Path, lime_available: bool) -> None:
    print("\n" + "=" * 100, flush=True)
    print("DONE", flush=True)
    print("=" * 100, flush=True)
    important_files = [
        "training_summary.csv",
        "stress_summary.csv",
        "table_features_vs_text.csv",
        "table_best_original.csv",
        "table_best_obfuscated.csv",
        "table_timing_summary.csv",
        "final_summary.csv",
    ]
    if lime_available:
        important_files.append("lime_summary.csv")
    for name in important_files:
        print(results_dir / name, flush=True)


def load_existing_models(save_dir: Path) -> pd.DataFrame:
    rows = []

    checkpoint_files = sorted(list(save_dir.glob("*.pt")) + list(save_dir.glob("*.bin")))

    if not checkpoint_files:
        raise RuntimeError(
            f"No checkpoints found in {save_dir}\n"
            f"Contents: {[p.name for p in save_dir.iterdir()] if save_dir.exists() else 'directory does not exist'}"
        )

    for ckpt in checkpoint_files:
        name = ckpt.stem.lower()

        is_distil = "distilbert" in name
        use_features = "hybrid" in name

        if is_distil and use_features:
            run_name = "distilbert_hybrid"
            pretty_name = "DistilBERT hybrid"
            model_name = "distilbert-base-uncased"
        elif is_distil and not use_features:
            run_name = "distilbert_text_only"
            pretty_name = "DistilBERT text-only"
            model_name = "distilbert-base-uncased"
        elif not is_distil and use_features:
            run_name = "bert_hybrid"
            pretty_name = "BERT hybrid"
            model_name = "bert-base-uncased"
        else:
            run_name = "bert_text_only"
            pretty_name = "BERT text-only"
            model_name = "bert-base-uncased"

        rows.append({
            "run_name": run_name,
            "pretty_name": pretty_name,
            "model": model_name,
            "features": use_features,
            "checkpoint_path": str(ckpt),
            "best_val_acc": np.nan,
            "test_acc": np.nan,
            "train_time_sec": np.nan,
            "total_time_sec": np.nan,
            "wall_time_runner_sec": np.nan,
        })

    df = pd.DataFrame(rows).sort_values("run_name").reset_index(drop=True)

    print("\n" + "=" * 100, flush=True)
    print("LOADED EXISTING CHECKPOINTS", flush=True)
    print("=" * 100, flush=True)
    print(df[["run_name", "checkpoint_path"]].to_string(index=False), flush=True)

    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all transformer training, obfuscation, stress tests, and aggregated LIME analysis."
    )
    parser.add_argument("--splits_dir", type=str, default="data/processed/splits")
    parser.add_argument("--test_csv", type=str, default="data/processed/splits/test.csv")
    parser.add_argument("--obfuscated_csv", type=str, default="data/processed/splits/test_obfuscated_homograph.csv")
    parser.add_argument("--save_dir", type=str, default="transformers/saved_models_transformers")
    parser.add_argument("--results_dir", type=str, default="transformers/results_transformers")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--lime_examples_per_class", type=int, default=10)
    parser.add_argument("--lime_num_features", type=int, default=10)
    parser.add_argument("--top_k_words", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = ROOT / args.results_dir
    save_dir = ROOT / args.save_dir
    ensure_dir(results_dir)
    ensure_dir(save_dir)
    logs_dir = results_dir / "logs"
    ensure_dir(logs_dir)

    full_start = time.time()

    print("\n" + "=" * 100, flush=True)
    print("STARTING FULL TRANSFORMER PIPELINE", flush=True)
    print("=" * 100, flush=True)

    # training_df = run_training_suite(args, results_dir, logs_dir)
    training_df = load_existing_models(save_dir)
    # run_obfuscation(results_dir, logs_dir)
    # stress_df = run_stress_suite(training_df, args, results_dir)
    _, lime_available = run_lime_suite(training_df, args, results_dir)
    build_report_tables(training_df, stress_df, results_dir)

    total_elapsed = time.time() - full_start
    with open(results_dir / "run_metadata.txt", "w", encoding="utf-8") as f:
        f.write(f"Total pipeline runtime (sec): {total_elapsed:.2f}\n")
        f.write(f"Total pipeline runtime (min): {total_elapsed / 60:.2f}\n")

    print(f"\nTotal pipeline runtime: {total_elapsed:.2f} sec ({total_elapsed / 60:.2f} min)", flush=True)
    print_quick_paths(results_dir, lime_available)


if __name__ == "__main__":
    main()