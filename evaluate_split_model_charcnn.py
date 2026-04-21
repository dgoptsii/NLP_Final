#!/usr/bin/env python3
"""
python evaluate_split_model_charcnn.py --train ./data/processed/splits/train.csv --val ./data/processed/splits/val.csv --test ./data/processed
/splits/test.csv > ./output/model_nb_charcnn.txt

Character-level CNN for phishing email (PyTorch).

Character embedding (learnable),parallel conv layers with different kernel sizes (3, 5, 7), Max-over-time pooling, dropout + fully connected for binary classification

Text input: subject + " " + text, truncated to MAX_CHARS characters.
"""

from __future__ import annotations

import argparse
import string
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

VOCAB = ["\x00"] + list(string.printable)  # 0 = padding, 1..100 = chars
CHAR2IDX = {ch: idx for idx, ch in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

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

def load_split(path: Path, split_name: str, max_chars: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{split_name} file not found: {path}")

    df = pd.read_csv(path)

    required = FEATURE_COLUMNS + ["label", "id", "source", "subject", "text"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{split_name} is missing columns: {missing}")

    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df["subject"] = df["subject"].fillna("") # Na/NaN with ""
    df["text"] = df["text"].fillna("") # Na/NaN with ""
    df["text_input"] = (df["subject"] + " " + df["text"]).str[:max_chars] # concatenate subject + text

    return df


def text_to_tensor(text: str, max_chars: int) -> torch.Tensor:
    indices = [CHAR2IDX.get(ch, 0) for ch in text[:max_chars]]
    indices += [0] * (max_chars - len(indices)) # Pad with zeros to max_chars
    return torch.tensor(indices, dtype=torch.long)


class EmailDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_chars: int):
        self.texts  = df["text_input"].tolist()
        self.labels = df["label"].tolist()
        self.max_chars = max_chars

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = text_to_tensor(self.texts[idx], self.max_chars)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# Model


# Character-level CNN with parallel convolutional filters
class CharCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_filters: int = 256,
        kernel_sizes: list[int] = (3, 5, 7),
        dropout: float = 0.5,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # One conv layer per kernel size
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2, # same padding to maintain length
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU()

        # max-pooling each conv, the output is (batch, num_filters) then concatenate all kernel outputs
        fc_input_dim = num_filters * len(kernel_sizes)
        self.fc = nn.Linear(fc_input_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        emb = self.embedding(x)
        emb = emb.permute(0, 2, 1)
        pooled = []
        for conv in self.convs:
            out = self.relu(conv(emb))
            out = out.max(dim=2).values
            pooled.append(out)

        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        return self.fc(cat)


# training and helpers

def train_one_epoch(model: CharCNN,loader: DataLoader,optimizer: torch.optim.Optimizer,criterion: nn.Module,device: torch.device,) -> float:
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model: CharCNN,loader: DataLoader,device: torch.device,) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs  = []
    all_preds  = []

    for x, _ in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)

    return np.concatenate(all_preds), np.concatenate(all_probs)


# Metrics 

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
    print(f"Specificity:          {metrics['specificity']:.4f}")
    print(f"F1 score:             {metrics['f1']:.4f}")
    print(f"ROC AUC:              {metrics['roc_auc']:.4f}")
    print(f"PR AUC:               {metrics['pr_auc']:.4f}")
    print(f"Matthews corrcoef:    {metrics['mcc']:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    print("Confusion matrix:")
    print(metrics["confusion_matrix"])


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


# Main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",       type=Path,  required=True)
    parser.add_argument("--val",         type=Path,  required=True)
    parser.add_argument("--test",        type=Path,  required=True)
    parser.add_argument("--max-chars",   type=int,   default=1024)
    parser.add_argument("--epochs",      type=int,   default=7)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--embed-dim",   type=int,   default=64)
    parser.add_argument("--num-filters", type=int,   default=128)
    parser.add_argument("--dropout",     type=float, default=0.5)
    parser.add_argument("--random-seed", type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    train_df = load_split(args.train, "train", args.max_chars)
    val_df = load_split(args.val,   "val",   args.max_chars)
    test_df = load_split(args.test,  "test",  args.max_chars)

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print_dataset_summary("TRAIN", train_df)
    print_dataset_summary("VAL",   val_df)
    print_dataset_summary("TEST",  test_df)
    print(f"\nMax chars: {args.max_chars} | Vocab size: {VOCAB_SIZE}")

    # DataLoaders
    train_loader = DataLoader(EmailDataset(train_df, args.max_chars), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(EmailDataset(val_df,   args.max_chars), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(EmailDataset(test_df,  args.max_chars), batch_size=args.batch_size, shuffle=False)

    # Model
    model = CharCNN(vocab_size=VOCAB_SIZE,embed_dim=args.embed_dim,num_filters=args.num_filters,dropout=args.dropout,).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_val_recall = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_pred, val_prob = predict(model, val_loader, device)
        val_f1 = f1_score(val_df["label"].values, val_pred, zero_division=0)
        val_acc = accuracy_score(val_df["label"].values, val_pred)
        val_recall = recall_score(val_df["label"].values, val_pred, zero_division=0)

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  "
              f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

        # Save best checkpoint by recall
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            best_state      = {k: v.clone() for k, v in model.state_dict().items()}
 
    model.load_state_dict(best_state)
    print(f"\nBest recall: {best_val_recall:.4f} (checkpoint)")

    model_path = Path("./pkl_models/char_cnn.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"model_state": best_state, "args": vars(args)}, f)

    # Final evaluation
    train_pred, train_prob = predict(model, train_loader, device)
    val_pred, val_prob   = predict(model, val_loader,   device)
    test_pred, test_prob  = predict(model, test_loader,  device)

    print_metrics_block("TRAIN", train_df["label"].values, train_pred,train_prob)
    print_metrics_block("VAL", val_df["label"].values, val_pred,val_prob)
    print_metrics_block("TEST", test_df["label"].values, test_pred,test_prob)

    print_error_analysis("TRAIN",train_df, train_pred)
    print_error_analysis("VAL",val_df, val_pred)
    print_error_analysis("TEST",test_df, test_pred)


if __name__ == "__main__":
    main()