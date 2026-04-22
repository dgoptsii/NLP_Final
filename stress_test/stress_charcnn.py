#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import string
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import recall_score, f1_score, accuracy_score

MODEL_PATH = Path("./pkl_models/char_cnn.pkl")

VOCAB      = ["\x00"] + list(string.printable)
CHAR2IDX   = {ch: idx for idx, ch in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

# Same as in charcnn.py
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


def load_test(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"]      = pd.to_numeric(df["label"], errors="raise").astype(int)
    df["subject"]    = df["subject"].fillna("")
    df["text"]       = df["text"].fillna("")
    df["text_input"] = df["subject"] + " " + df["text"]
    return df


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",        type=Path, required=True)
    parser.add_argument("--obfuscated",  type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    with open(MODEL_PATH, "rb") as f:
        checkpoint = pickle.load(f)

    saved_args = checkpoint["args"]
    model = CharCNN(
        vocab_size=VOCAB_SIZE,
        embed_dim=saved_args["embed_dim"],
        num_filters=saved_args["num_filters"],
        kernel_sizes=saved_args["kernel_sizes"],
        dropout=saved_args["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    max_chars = saved_args["max_chars"]

    test_df = load_test(args.test)
    obfuscated_df = load_test(args.obfuscated)

    y_true = test_df["label"].values


    # Baseline
    test_loader = DataLoader(EmailDataset(test_df, max_chars), batch_size=64, shuffle=False)
    baseline_pred, _ = predict(model, test_loader, device)
    baseline_recall = recall_score(y_true, baseline_pred, zero_division=0)
    baseline_f1     = f1_score(y_true, baseline_pred, zero_division=0)
    baseline_acc    = accuracy_score(y_true, baseline_pred)

    print(f"\n{'Technique':<35} {'Recall':>8} {'F1':>8} {'Accuracy':>10} {'Recall drop':>12}")
    print("-" * 85)
    print(f"  {'Original (no obfuscation)':<33} "
          f"{baseline_recall:>8.4f} {baseline_f1:>8.4f} {baseline_acc:>10.4f} {'—':>12}  —")

    # Obfuscated
    obfuscated_loader = DataLoader(EmailDataset(obfuscated_df, max_chars), batch_size=64, shuffle=False)
    obfuscated_pred, _ = predict(model, obfuscated_loader, device)
    obfuscated_recall = recall_score(y_true, obfuscated_pred, zero_division=0)
    obfuscated_f1     = f1_score(y_true, obfuscated_pred, zero_division=0)
    obfuscated_acc    = accuracy_score(y_true, obfuscated_pred)

    drop_pct = (baseline_recall - obfuscated_recall) / baseline_recall * 100

    print(f"  {'Obfuscation':<33} "
          f"{obfuscated_recall:>8.4f} {obfuscated_f1:>8.4f} {obfuscated_acc:>10.4f} {drop_pct:>10.2f}%")


if __name__ == "__main__":
    main()