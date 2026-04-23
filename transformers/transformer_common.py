import copy
import json
import os
import time
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")


@dataclass
class RunConfig:
    model_name: str
    use_features: bool
    max_len: int
    batch_size: int
    epochs: int
    lr: float
    dropout: float
    patience: int
    random_state: int
    save_dir: str
    results_csv: str = ""


class EmailDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, use_features: bool, feature_columns: List[str]):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_features = use_features
        self.feature_columns = feature_columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = str(row["combined_text"])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }

        if self.use_features and self.feature_columns:
            features = row[self.feature_columns].astype(np.float32).values
            item["features"] = torch.tensor(features, dtype=torch.float32)

        return item


class TransformerHybridClassifier(nn.Module):
    def __init__(self, model_name: str, use_features: bool, feature_dim: int, dropout: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.use_features = use_features
        self.hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        if self.use_features and feature_dim > 0:
            self.feature_mlp = nn.Sequential(
                nn.Linear(feature_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size + 32, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 2),
            )
        else:
            self.feature_mlp = None
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 2),
            )

    def get_text_embedding(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "last_hidden_state"):
            cls_vector = outputs.last_hidden_state[:, 0, :]
        else:
            cls_vector = outputs[0][:, 0, :]
        return self.dropout(cls_vector)

    def forward(self, input_ids, attention_mask, features=None):
        text_embedding = self.get_text_embedding(input_ids, attention_mask)

        if self.use_features and features is not None and self.feature_mlp is not None:
            feature_embedding = self.feature_mlp(features)
            combined = torch.cat([text_embedding, feature_embedding], dim=1)
            logits = self.classifier(combined)
        else:
            logits = self.classifier(text_embedding)

        return logits


def build_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "subject" not in df.columns:
        df["subject"] = ""
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    df["subject"] = df["subject"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["combined_text"] = ("[SUBJECT] " + df["subject"] + " [BODY] " + df["text"]).str.strip()
    return df


def detect_feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {"subject", "text", "label", "combined_text"}
    feature_columns = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_columns.append(col)
    return feature_columns


def prepare_train_val_test(splits_dir: str, use_features: bool):
    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "val.csv")
    test_path = os.path.join(splits_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    train_df = build_combined_text(train_df)
    val_df = build_combined_text(val_df)
    test_df = build_combined_text(test_df)

    feature_columns: List[str] = []
    scaler: Optional[StandardScaler] = None

    if use_features:
        feature_columns = detect_feature_columns(train_df)
        if feature_columns:
            train_df = train_df.copy()
            val_df = val_df.copy()
            test_df = test_df.copy()

            train_df[feature_columns] = train_df[feature_columns].fillna(0)
            val_df[feature_columns] = val_df[feature_columns].fillna(0)
            test_df[feature_columns] = test_df[feature_columns].fillna(0)

            scaler = StandardScaler()
            train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
            val_df[feature_columns] = scaler.transform(val_df[feature_columns])
            test_df[feature_columns] = scaler.transform(test_df[feature_columns])
        else:
            use_features = False

    return train_df, val_df, test_df, feature_columns, scaler, use_features


def prepare_single_dataframe(csv_path: str, feature_columns: List[str], scaler: Optional[StandardScaler]):
    df = pd.read_csv(csv_path)
    df = build_combined_text(df)
    if feature_columns:
        df = df.copy()
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        df[feature_columns] = df[feature_columns].fillna(0)
        if scaler is not None:
            df[feature_columns] = scaler.transform(df[feature_columns])
    return df


def train_one_epoch(model, loader, optimizer, criterion, device, epoch_num, total_epochs):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    progress_bar = tqdm(loader, desc=f"Epoch {epoch_num}/{total_epochs}", leave=True)
    for step, batch in enumerate(progress_bar, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        features = batch.get("features")
        if features is not None:
            features = features.to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(batch_loss=f"{loss.item():.4f}", avg_loss=f"{total_loss / step:.4f}")

    avg_loss = total_loss / len(loader)
    epoch_time = time.time() - start_time
    return avg_loss, epoch_time


@torch.no_grad()
def evaluate(model, loader, criterion, device, split_name="Evaluation"):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc=split_name, leave=True):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        features = batch.get("features")
        if features is not None:
            features = features.to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, np.array(all_labels), np.array(all_preds)


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    all_probs = []
    all_preds = []
    for batch in tqdm(loader, desc="Predict", leave=True):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        features = batch.get("features")
        if features is not None:
            features = features.to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_probs.append(probs)
        all_preds.append(preds)
    return np.vstack(all_probs), np.concatenate(all_preds)


def save_checkpoint(
    save_path: str,
    model: TransformerHybridClassifier,
    tokenizer,
    scaler: Optional[StandardScaler],
    feature_columns: List[str],
    cfg: RunConfig,
    best_val_acc: float,
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_name": cfg.model_name,
        "use_features": cfg.use_features,
        "feature_columns": feature_columns,
        "max_len": cfg.max_len,
        "dropout": cfg.dropout,
        "best_val_acc": best_val_acc,
        "scaler": scaler,
        "tokenizer_name": cfg.model_name,
        "config": {
            "model_name": cfg.model_name,
            "use_features": cfg.use_features,
            "max_len": cfg.max_len,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "dropout": cfg.dropout,
            "patience": cfg.patience,
            "random_state": cfg.random_state,
        },
    }
    torch.save(payload, save_path)
    print(f"Saved checkpoint to: {save_path}")



def load_checkpoint(checkpoint_path: str, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TransformerHybridClassifier(
        model_name=payload["model_name"],
        use_features=bool(payload["use_features"] and len(payload["feature_columns"]) > 0),
        feature_dim=len(payload["feature_columns"]),
        dropout=payload.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    tokenizer = AutoTokenizer.from_pretrained(payload.get("tokenizer_name", payload["model_name"]))
    scaler = payload.get("scaler")
    return payload, model, tokenizer, scaler



def run_training(splits_dir: str, cfg: RunConfig):
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df, val_df, test_df, feature_columns, scaler, effective_use_features = prepare_train_val_test(
        splits_dir=splits_dir,
        use_features=cfg.use_features,
    )
    cfg.use_features = effective_use_features

    print("\n" + "=" * 70)
    print("DATASET")
    print("=" * 70)
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}")
    print(f"Test rows:  {len(test_df)}")
    print(f"Feature columns: {feature_columns}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_dataset = EmailDataset(train_df, tokenizer, cfg.max_len, cfg.use_features, feature_columns)
    val_dataset = EmailDataset(val_df, tokenizer, cfg.max_len, cfg.use_features, feature_columns)
    test_dataset = EmailDataset(test_df, tokenizer, cfg.max_len, cfg.use_features, feature_columns)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = TransformerHybridClassifier(
        model_name=cfg.model_name,
        use_features=cfg.use_features and len(feature_columns) > 0,
        feature_dim=len(feature_columns),
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    print("\n" + "=" * 70)
    print("TRAINING CONFIG")
    print("=" * 70)
    print(json.dumps(cfg.__dict__, indent=2))

    best_acc = -1.0
    best_state = None
    epochs_without_improvement = 0
    exp_start = time.time()
    training_time_only = 0.0
    epochs_run = 0

    model_stub = cfg.model_name.replace("/", "_")
    features_stub = "hybrid" if cfg.use_features else "text_only"
    checkpoint_path = os.path.join(cfg.save_dir, f"{model_stub}_{features_stub}.pt")

    for epoch in range(1, cfg.epochs + 1):
        print("\n" + "=" * 70)
        print(f"STARTING EPOCH {epoch}/{cfg.epochs}")
        print("=" * 70)

        train_loss, epoch_train_time = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, cfg.epochs)
        training_time_only += epoch_train_time
        epochs_run = epoch

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, split_name="Validation")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss:   {val_loss:.4f}")
        print(f"Val acc:    {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            model.load_state_dict(best_state)
            save_checkpoint(checkpoint_path, model, tokenizer, scaler, feature_columns, cfg, best_acc)
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= cfg.patience:
                print("Early stopping triggered.")
                break

    total_wall_time = time.time() - exp_start

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device, split_name="Test")

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Accuracy:   {test_acc:.4f}")
    print(f"Loss:       {test_loss:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    return {
        "model": cfg.model_name,
        "features": cfg.use_features,
        "epochs_run": epochs_run,
        "best_val_acc": best_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "train_time_sec": training_time_only,
        "total_time_sec": total_wall_time,
        "checkpoint_path": checkpoint_path,
        "feature_columns": feature_columns,
    }



def evaluate_checkpoint_on_csv(checkpoint_path: str, csv_path: str, batch_size: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload, model, tokenizer, scaler = load_checkpoint(checkpoint_path, device)
    feature_columns = payload.get("feature_columns", [])

    df = prepare_single_dataframe(csv_path, feature_columns, scaler)
    dataset = EmailDataset(df, tokenizer, payload.get("max_len", 256), payload.get("use_features", False), feature_columns)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    probs, preds = predict_proba(model, loader, device)
    y_true = df["label"].astype(int).values

    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
    }
    return df, preds, probs, metrics, payload
