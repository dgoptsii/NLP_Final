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
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    precision_score, roc_auc_score
)


MAX_CHARS = 1024
NUM_FILTERS = 512
KERNEL_SIZES = [5, 7, 9]
DROPOUT = 0.5
LR = 1e-3
EMBED_DIM = 64
EPOCHS = 7
BATCH_SIZE = 64
SEED = 42

VOCAB      = ["\x00"] + list(string.printable)
CHAR2IDX   = {ch: idx for idx, ch in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

FEATURE_COLUMNS = [
    "char_len", "token_len", "subject_len", "num_urls", "num_emails",
    "num_domains", "num_exclamation", "num_upper_tokens", "keyword_hits",
    "num_attachments", "has_form_tag", "has_script_tag", "has_iframe_tag",
    "has_ip_url", "has_external_links",
]
N_FEATURES = len(FEATURE_COLUMNS)

PKL_DIR = Path("./pkl_models")

def load_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{split_name} not found: {path}")
    df = pd.read_csv(path)
    df["label"]      = pd.to_numeric(df["label"], errors="raise").astype(int)
    df["subject"]    = df["subject"].fillna("")
    df["text"]       = df["text"].fillna("")
    df["text_input"] = (df["subject"] + " " + df["text"]).str[:MAX_CHARS]
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df


def text_to_tensor(text: str) -> torch.Tensor:
    indices  = [CHAR2IDX.get(ch, 0) for ch in text[:MAX_CHARS]]
    indices += [0] * (MAX_CHARS - len(indices))
    return torch.tensor(indices, dtype=torch.long)


class EmailDataset(Dataset):
    def __init__(self, df: pd.DataFrame, variant: str):
        self.texts = df["text_input"].tolist()
        self.features = df[FEATURE_COLUMNS].values.astype(np.float32)
        self.labels = df["label"].tolist()
        self.variant = variant

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.variant == "text_only":
            return text_to_tensor(self.texts[idx]), label
        elif self.variant == "features_only":
            return torch.tensor(self.features[idx]), label
        else:  # text_features
            return text_to_tensor(self.texts[idx]), torch.tensor(self.features[idx]), label


class TextOnlyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(EMBED_DIM, NUM_FILTERS, k, padding=k // 2)
            for k in KERNEL_SIZES
        ])
        self.dropout = nn.Dropout(DROPOUT)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(NUM_FILTERS * len(KERNEL_SIZES), 2)

    def forward(self, x):
        emb = self.embedding(x).permute(0, 2, 1)
        pooled = [self.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class TextFeaturesCNN(nn.Module):
    """CharCNN on text + lexical features concatenated before FC."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.convs     = nn.ModuleList([
            nn.Conv1d(EMBED_DIM, NUM_FILTERS, k, padding=k // 2)
            for k in KERNEL_SIZES
        ])
        self.dropout = nn.Dropout(DROPOUT)
        self.relu    = nn.ReLU()
        cnn_dim      = NUM_FILTERS * len(KERNEL_SIZES)
        self.fc      = nn.Linear(cnn_dim + N_FEATURES, 2)

    def forward(self, x, features):
        emb    = self.embedding(x).permute(0, 2, 1)
        pooled = [self.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        cat    = torch.cat(pooled, dim=1)           # (batch, cnn_dim)
        cat    = torch.cat([cat, features], dim=1)  # (batch, cnn_dim + N_FEATURES)
        return self.fc(self.dropout(cat))


def train_one_epoch(model, loader, optimizer, criterion, device, variant):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        if variant == "text_features":
            x, features, y = batch
            x, features, y = x.to(device), features.to(device), y.to(device)
            logits = model(x, features)
        else:  # text_only
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device, variant):
    model.eval()
    preds, probs = [], []
    for batch in loader:
        if variant == "text_features":
            x, features, _ = batch
            x, features = x.to(device), features.to(device)
            logits = model(x, features)
        else:  # text_only
            x, _ = batch
            logits = model(x.to(device))
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def run_variant(variant: str, train_df, val_df, test_df, device) -> dict:
    print(f"\n{'=' * 60}")
    print(f"VARIANT: {variant.upper()}")
    print("=" * 60)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_loader = DataLoader(EmailDataset(train_df, variant), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(EmailDataset(val_df,   variant), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(EmailDataset(test_df,  variant), batch_size=BATCH_SIZE, shuffle=False)

    if variant == "text_only":
        model = TextOnlyCNN().to(device)
    else:
        model = TextFeaturesCNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_recall = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, variant)
        val_pred, _ = predict(model, val_loader, device, variant)
        val_recall = recall_score(val_df["label"].values, val_pred, zero_division=0)
        val_acc = accuracy_score(val_df["label"].values, val_pred)
        print(f"  Epoch {epoch}/{EPOCHS}  loss={train_loss:.4f}  val_acc={val_acc:.4f}  val_recall={val_recall:.4f}")

        if val_recall > best_val_recall:
            best_val_recall = val_recall
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"  Best val recall: {best_val_recall:.4f}")

    # Save pkl
    PKL_DIR.mkdir(parents=True, exist_ok=True)
    pkl_path = PKL_DIR / f"charcnn_{variant}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"model_state": best_state, "variant": variant}, f)
    

    # Evaluate on test
    test_pred, test_prob = predict(model, test_loader, device, variant)
    y_test = test_df["label"].values

    return {
        "variant": variant,
        "recall": recall_score(y_test, test_pred, zero_division=0),
        "precision": precision_score(y_test, test_pred, zero_division=0),
        "f1": f1_score(y_test, test_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, test_pred),
        "roc_auc": roc_auc_score(y_test, test_prob),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val",   type=Path, required=True)
    parser.add_argument("--test",  type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    train_df = load_split(args.train, "train")
    val_df   = load_split(args.val, "val")
    test_df  = load_split(args.test, "test")

    results = []
    for variant in ["text_only", "text_features"]:
        results.append(run_variant(variant, train_df, val_df, test_df, device))

    # Summary table
    print("\n\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"\n  {'Variant':<20} {'Recall':>8} {'Precision':>10} {'F1':>8} {'Accuracy':>10} {'ROC AUC':>9}")
    print("  " + "-" * 70)
    for r in results:
        print(f"  {r['variant']:<20} {r['recall']:>8.4f} {r['precision']:>10.4f} "
              f"{r['f1']:>8.4f} {r['accuracy']:>10.4f} {r['roc_auc']:>9.4f}")

    best = max(results, key=lambda x: x["recall"])
    print(f"\n  Best by recall: {best['variant']} (recall={best['recall']:.4f})")


if __name__ == "__main__":
    main()