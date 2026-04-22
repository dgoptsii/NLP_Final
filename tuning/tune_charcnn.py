#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import string
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

VOCAB     = ["\x00"] + list(string.printable)
CHAR2IDX  = {ch: idx for idx, ch in enumerate(VOCAB)}
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

# Search space
SEARCH_SPACE = {
    "max_chars": [1024, 2048, 5000],
    "num_filters": [128, 256, 512],
    "dropout": [0.3, 0.5, 0.7],
    "kernel_sizes": [[3, 5, 7], [3, 7], [5, 7, 9], [3, 5]],
    "lr": [1e-3, 1e-4],
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


def build_text_input(df: pd.DataFrame) -> pd.Series:
    return df["subject"] + " " + df["text"]


def text_to_tensor(text: str, max_chars: int) -> torch.Tensor:
    indices = [CHAR2IDX.get(ch, 0) for ch in text[:max_chars]]
    indices += [0] * (max_chars - len(indices))
    return torch.tensor(indices, dtype=torch.long)


class EmailDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], max_chars: int):
        self.texts = texts
        self.labels = labels
        self.max_chars = max_chars

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = text_to_tensor(self.texts[idx], self.max_chars)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class CharCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_filters=256,kernel_sizes=(3, 5, 7), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU()
        self.fc      = nn.Linear(num_filters * len(kernel_sizes), 2)

    def forward(self, x):
        emb = self.embedding(x).permute(0, 2, 1)
        pooled = [self.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        cat = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(cat))

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, probs = [], []
    for x, _ in loader:
        x = x.to(device)
        logits = model(x)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def run_trial(config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame,epochs: int, batch_size: int, seed: int, device: torch.device) -> dict:

    torch.manual_seed(seed)
    np.random.seed(seed)

    max_chars = config["max_chars"]

    train_texts = build_text_input(train_df).tolist()
    val_texts = build_text_input(val_df).tolist()
    train_labels = train_df["label"].tolist()
    val_labels = val_df["label"].tolist()

    train_loader = DataLoader(EmailDataset(train_texts, train_labels, max_chars), batch_size=batch_size, shuffle=True,)
    val_loader = DataLoader(EmailDataset(val_texts, val_labels, max_chars), batch_size=batch_size, shuffle=False,)

    model = CharCNN(
        vocab_size=VOCAB_SIZE,
        embed_dim=64,
        num_filters=config["num_filters"],
        kernel_sizes=config["kernel_sizes"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    best_val_recall = 0.0
    best_state = None

    for _ in range(epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_pred, val_prob = predict(model, val_loader, device)
        val_recall = recall_score(val_labels, val_pred, zero_division=0)
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # best checkpoint
    model.load_state_dict(best_state)
    val_pred, val_prob = predict(model, val_loader, device)

    return {
        "val_recall": recall_score(val_labels, val_pred, zero_division=0),
        "val_f1": f1_score(val_labels, val_pred, zero_division=0),
        "val_acc": accuracy_score(val_labels, val_pred),
        "val_roc_auc": roc_auc_score(val_labels, val_prob),
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",      type=Path, required=True)
    parser.add_argument("--val",        type=Path, required=True)
    args = parser.parse_args()

    N_TRIALS = 20
    EPOCHS = 10
    BATCH_SIZE = 32
    SEED = 42

    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("  CHARACTER-LEVEL CNN — RANDOM HYPERPARAMETER SEARCH")
    print("=" * 70)
    print(f"  Device    : {device}")
    print(f"  Trials    : {N_TRIALS}")
    print(f"  Epochs    : {EPOCHS} per trial")
    print(f"  Val set   : {args.val}")
    print("=" * 70)

    train_df = load_split(args.train, "train")
    val_df   = load_split(args.val,   "val")

    # Sample random configurations
    configs = []
    for _ in range(N_TRIALS):
        configs.append({
            "max_chars": random.choice(SEARCH_SPACE["max_chars"]),
            "num_filters": random.choice(SEARCH_SPACE["num_filters"]),
            "dropout": random.choice(SEARCH_SPACE["dropout"]),
            "kernel_sizes": random.choice(SEARCH_SPACE["kernel_sizes"]),
            "lr": random.choice(SEARCH_SPACE["lr"]),
        })

    results = []

    for i, config in enumerate(configs, 1):
        print(f"\n[Trial {i:2d}/{N_TRIALS}]  "
              f"max_chars={config['max_chars']:4d}  "
              f"filters={config['num_filters']:3d}  "
              f"dropout={config['dropout']}  "
              f"kernels={config['kernel_sizes']}  "
              f"lr={config['lr']}")

        t0 = time.time()
        metrics = run_trial(
            config, train_df, val_df,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            seed=SEED,
            device=device,
        )
        elapsed = time.time() - t0

        print(f"           val_recall={metrics['val_recall']:.4f}  "
              f"val_f1={metrics['val_f1']:.4f}  "
              f"val_acc={metrics['val_acc']:.4f}  "
              f"({elapsed:.0f}s)")

        results.append({**config, **metrics, "trial": i, "time_s": elapsed})

    results_df = pd.DataFrame(results).sort_values("val_recall", ascending=False)

    print("\n\n" + "=" * 70)
    print("  RESULTS (Sorted by recall)")
    print("=" * 70)
    print(results_df[[
        "trial", "val_recall", "val_f1", "val_acc", "val_roc_auc",
        "max_chars", "num_filters", "dropout", "kernel_sizes", "lr"
    ]].to_string(index=False))

    best = results_df.iloc[0]

    print("\n\n" + "=" * 70)
    print("  BEST CONFIGURATION")
    print("=" * 70)
    print(f"  val_recall  : {best['val_recall']:.4f}")
    print(f"  val_f1      : {best['val_f1']:.4f}")
    print(f"  val_acc     : {best['val_acc']:.4f}")
    print(f"  val_roc_auc : {best['val_roc_auc']:.4f}")
    print()
    print(f"  max_chars   : {best['max_chars']}")
    print(f"  num_filters : {best['num_filters']}")
    print(f"  dropout     : {best['dropout']}")
    print(f"  kernel_sizes: {best['kernel_sizes']}")
    print(f"  lr          : {best['lr']}")

    print("\n" + "=" * 70)
    

if __name__ == "__main__":
    main()