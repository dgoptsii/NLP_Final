import argparse
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")


class EmailDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_len: int,
        use_features: bool,
        feature_columns: list[str],
    ):
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


class BertHybridClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        use_features: bool,
        feature_dim: int,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.use_features = use_features
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        if self.use_features and feature_dim > 0:
            self.feature_mlp = nn.Sequential(
                nn.Linear(feature_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size + 32, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 2),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 2),
            )

    def forward(self, input_ids, attention_mask, features=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(outputs, "last_hidden_state"):
            cls_vector = outputs.last_hidden_state[:, 0, :]
        else:
            cls_vector = outputs[0][:, 0, :]

        cls_vector = self.dropout(cls_vector)

        if self.use_features and features is not None:
            feature_vector = self.feature_mlp(features)
            combined = torch.cat([cls_vector, feature_vector], dim=1)
            logits = self.classifier(combined)
        else:
            logits = self.classifier(cls_vector)

        return logits


def build_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    if "subject" not in df.columns:
        df["subject"] = ""
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    df["subject"] = df["subject"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    df["combined_text"] = (
        "[SUBJECT] " + df["subject"] + " [BODY] " + df["text"]
    ).str.strip()

    return df


def detect_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"subject", "text", "label", "combined_text"}

    feature_columns = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_columns.append(col)

    return feature_columns


def prepare_data(
    csv_path: str,
    test_size: float,
    random_state: int,
    use_features: bool,
):
    df = pd.read_csv(csv_path)
    df = build_combined_text(df)

    print("\n" + "=" * 60)
    print("DATASET")
    print("=" * 60)
    print(f"Rows: {len(df)}")
    print(f"Label distribution: {df['label'].value_counts().sort_index().to_dict()}")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

    feature_columns = []
    scaler = None

    if use_features:
        feature_columns = detect_feature_columns(df)
        print(f"Using features: True")
        print(f"Feature columns ({len(feature_columns)}): {feature_columns}")

        if feature_columns:
            train_df = train_df.copy()
            test_df = test_df.copy()

            train_df[feature_columns] = train_df[feature_columns].fillna(0)
            test_df[feature_columns] = test_df[feature_columns].fillna(0)

            scaler = StandardScaler()
            train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
            test_df[feature_columns] = scaler.transform(test_df[feature_columns])
        else:
            print("No numeric feature columns found. Falling back to text-only BERT.")
    else:
        print("Using features: False")

    return train_df, test_df, feature_columns, scaler


def train_one_epoch(model, loader, optimizer, criterion, device, epoch_num, total_epochs):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch_num}/{total_epochs}",
        leave=True,
    )

    for step, batch in enumerate(progress_bar, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        features = batch.get("features")
        if features is not None:
            features = features.to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            features=features,
        )
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        avg_so_far = total_loss / step
        progress_bar.set_postfix(
            batch_loss=f"{loss.item():.4f}",
            avg_loss=f"{avg_so_far:.4f}",
        )

    avg_loss = total_loss / len(loader)
    epoch_time = time.time() - start_time

    print(f"\nEpoch {epoch_num} finished")
    print(f"Average training loss: {avg_loss:.4f}")
    print(f"Epoch time: {epoch_time:.2f} sec")

    return avg_loss


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(loader, desc="Evaluating", leave=True)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            features = batch.get("features")
            if features is not None:
                features = features.to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                features=features,
            )

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            avg_so_far = total_loss / max(1, len(all_preds) // loader.batch_size + (1 if len(all_preds) % loader.batch_size else 0))
            progress_bar.set_postfix(avg_loss=f"{avg_so_far:.4f}")

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description="Train BERT or BERT+features for phishing email detection.")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file.")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Hugging Face model name.")
    parser.add_argument("--max_len", type=int, default=256, help="Max token length.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    parser.add_argument("--no_features", action="store_true", help="Disable engineered features.")
    args = parser.parse_args()

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_features = not args.no_features

    train_df, test_df, feature_columns, _ = prepare_data(
        csv_path=args.csv,
        test_size=args.test_size,
        random_state=args.random_state,
        use_features=use_features,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = EmailDataset(
        df=train_df,
        tokenizer=tokenizer,
        max_len=args.max_len,
        use_features=use_features,
        feature_columns=feature_columns,
    )
    test_dataset = EmailDataset(
        df=test_df,
        tokenizer=tokenizer,
        max_len=args.max_len,
        use_features=use_features,
        feature_columns=feature_columns,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = BertHybridClassifier(
        model_name=args.model_name,
        use_features=use_features and len(feature_columns) > 0,
        feature_dim=len(feature_columns),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Max length: {args.max_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")

    best_acc = -1.0
    best_state = None

    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        print("\n" + "=" * 60)
        print(f"STARTING EPOCH {epoch}/{args.epochs}")
        print("=" * 60)

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch_num=epoch,
            total_epochs=args.epochs,
        )

        val_loss, val_acc, _, _ = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        print("\nValidation after epoch", epoch)
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print("Saved new best model.")

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)
    print(f"Total training time: {total_time:.2f} sec")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, y_true, y_pred = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Loss: {test_loss:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()