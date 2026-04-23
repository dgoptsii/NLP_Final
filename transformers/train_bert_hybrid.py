import argparse
import os
import pandas as pd

from transformer_common import RunConfig, run_training


def main():
    parser = argparse.ArgumentParser(description="Train and save BERT phishing classifier.")
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--use_features", action="store_true")
    parser.add_argument("--save_dir", type=str, default="saved_models")
    parser.add_argument("--save_results_csv", type=str, default="")
    args = parser.parse_args()

    cfg = RunConfig(
        model_name="bert-base-uncased",
        use_features=args.use_features,
        max_len=args.max_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
        patience=args.patience,
        random_state=args.random_state,
        save_dir=args.save_dir,
        results_csv=args.save_results_csv,
    )

    result = run_training(args.splits_dir, cfg)
    if args.save_results_csv:
        os.makedirs(os.path.dirname(args.save_results_csv) or ".", exist_ok=True)
        pd.DataFrame([result]).to_csv(args.save_results_csv, index=False)
        print(f"Saved summary CSV to: {args.save_results_csv}")


if __name__ == "__main__":
    main()
