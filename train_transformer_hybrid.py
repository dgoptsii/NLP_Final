import argparse
import copy
import os
import time
import warnings
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
        feature_columns: List[str],
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


class TransformerHybridClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        use_features: bool,
        feature_dim: int,
        dropout: float = 0.3,
    ):
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
    if "subject" not in df.columns:
        df["subject"] = ""
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    df = df.copy()
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


def prepare_data(splits_dir: str, use_features: bool):
    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "val.csv")
    test_path = os.path.join(splits_dir, "test.csv")

    print(f"Loading train split from: {train_path}")
    print(f"Loading val split from:   {val_path}")
    print(f"Loading test split from:  {test_path}")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train split not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation split not found: {val_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test split not found: {test_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    train_df = build_combined_text(train_df)
    val_df = build_combined_text(val_df)
    test_df = build_combined_text(test_df)

    print("\n" + "=" * 70)
    print("DATASET (FROM SPLITS)")
    print("=" * 70)
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}")
    print(f"Test rows:  {len(test_df)}")
    print(f"Train label distribution: {train_df['label'].value_counts().sort_index().to_dict()}")
    print(f"Val label distribution:   {val_df['label'].value_counts().sort_index().to_dict()}")
    print(f"Test label distribution:  {test_df['label'].value_counts().sort_index().to_dict()}")

    feature_columns = []

    if use_features:
        feature_columns = detect_feature_columns(train_df)
        print("\nUsing features: True")
        print(f"Feature columns ({len(feature_columns)}): {feature_columns}")

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
            print("No numeric feature columns found. Falling back to text-only.")
    else:
        print("\nUsing features: False")

    return train_df, val_df, test_df, feature_columns


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
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            features=features,
        )
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(
            batch_loss=f"{loss.item():.4f}",
            avg_loss=f"{total_loss / step:.4f}",
        )

    avg_loss = total_loss / len(loader)
    epoch_time = time.time() - start_time

    print(f"\nEpoch {epoch_num} finished")
    print(f"Average training loss: {avg_loss:.4f}")
    print(f"Epoch training time: {epoch_time:.2f} sec ({epoch_time / 60:.2f} min)")

    return avg_loss, epoch_time


def evaluate(model, loader, criterion, device, split_name="Evaluation"):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(loader, desc=split_name, leave=True)

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

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc, all_labels, all_preds


def save_embeddings(model, loader, device, output_path: str):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Saving embeddings", leave=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            embeddings = model.get_text_embedding(input_ids, attention_mask)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    emb = np.vstack(all_embeddings)
    lab = np.concatenate(all_labels)

    df_emb = pd.DataFrame(emb)
    df_emb.insert(0, "label", lab)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_emb.to_csv(output_path, index=False)
    print(f"Saved contextual embeddings to: {output_path}")


def run_experiment(args, model_name: str, use_features: bool, device):
    train_df, val_df, test_df, feature_columns = prepare_data(
        splits_dir=args.splits_dir,
        use_features=use_features,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = EmailDataset(
        df=train_df,
        tokenizer=tokenizer,
        max_len=args.max_len,
        use_features=use_features,
        feature_columns=feature_columns,
    )
    val_dataset = EmailDataset(
        df=val_df,
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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TransformerHybridClassifier(
        model_name=model_name,
        use_features=use_features and len(feature_columns) > 0,
        feature_dim=len(feature_columns),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model_label = "BERT" if model_name == "bert-base-uncased" else "DistilBERT"
    run_label = (
        f"{model_label} hybrid"
        if (use_features and len(feature_columns) > 0)
        else f"{model_label} text-only"
    )

    print("\n" + "=" * 70)
    print("TRAINING CONFIG")
    print("=" * 70)
    print(f"Run: {run_label}")
    print(f"Model: {model_name}")
    print(f"Splits dir: {args.splits_dir}")
    print(f"Max length: {args.max_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Using features: {use_features and len(feature_columns) > 0}")
    print(f"Early stopping patience: {args.patience}")

    best_acc = -1.0
    best_state = None
    epochs_without_improvement = 0
    exp_start = time.time()
    training_time_only = 0.0
    epochs_run = 0

    for epoch in range(1, args.epochs + 1):
        print("\n" + "=" * 70)
        print(f"STARTING EPOCH {epoch}/{args.epochs}")
        print("=" * 70)

        train_loss, epoch_train_time = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch_num=epoch,
            total_epochs=args.epochs,
        )
        training_time_only += epoch_train_time
        epochs_run = epoch

        val_loss, val_acc, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            split_name="Validation",
        )

        print(f"\nValidation after epoch {epoch}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            print("Saved new best model state in memory.")
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for {epochs_without_improvement} epoch(s). "
                f"Early stopping patience = {args.patience}."
            )
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break

    total_wall_time = time.time() - exp_start

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(
        f"Pure training time (sum of training epochs): "
        f"{training_time_only:.2f} sec ({training_time_only / 60:.2f} min)"
    )
    print(
        f"Total wall-clock time (training + validation): "
        f"{total_wall_time:.2f} sec ({total_wall_time / 60:.2f} min)"
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, y_true, y_pred = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        split_name="Test",
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Run: {run_label}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Loss: {test_loss:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    return {
        "run": run_label,
        "model": model_name,
        "features": use_features and len(feature_columns) > 0,
        "epochs_run": epochs_run,
        "best_val_acc": best_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "train_time_sec": training_time_only,
        "total_time_sec": total_wall_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run 4 experiments: BERT/DistilBERT with and without engineered features."
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        required=True,
        help="Directory containing train.csv, val.csv, and test.csv.",
    )
    parser.add_argument("--max_len", type=int, default=256, help="Max token length.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=4, help="Maximum number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Stop training if validation accuracy does not improve for this many epochs.",
    )
    parser.add_argument(
        "--save_results_csv",
        type=str,
        default="",
        help="Optional CSV path for saving the 4-test summary.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    experiments = [
        ("bert-base-uncased", False),
        ("bert-base-uncased", True),
        ("distilbert-base-uncased", False),
        ("distilbert-base-uncased", True),
    ]

    all_results = []
    suite_start = time.time()

    for idx, (model_name, use_features) in enumerate(experiments, start=1):
        print("\n" + "#" * 80)
        print(f"EXPERIMENT {idx}/{len(experiments)}: model={model_name}, features={use_features}")
        print("#" * 80)

        result = run_experiment(args, model_name=model_name, use_features=use_features, device=device)
        all_results.append(result)

    suite_total_time = time.time() - suite_start

    print("\n" + "=" * 110)
    print("FINAL SUMMARY OF ALL 4 TESTS")
    print("=" * 110)
    header = (
        f"{'Run':<22} {'Model':<24} {'Feat':<6} {'Epochs':<8} {'BestVal':<10} "
        f"{'TestAcc':<10} {'TrainTime(s)':<14} {'TotalTime(s)':<14}"
    )
    print(header)
    print("-" * 110)
    for r in all_results:
        print(
            f"{r['run']:<22} "
            f"{r['model']:<24} "
            f"{str(r['features']):<6} "
            f"{r['epochs_run']:<8d} "
            f"{r['best_val_acc']:<10.4f} "
            f"{r['test_acc']:<10.4f} "
            f"{r['train_time_sec']:<14.2f} "
            f"{r['total_time_sec']:<14.2f}"
        )

    print("\nTotal suite runtime: " f"{suite_total_time:.2f} sec ({suite_total_time / 60:.2f} min)")

    if args.save_results_csv:
        results_df = pd.DataFrame(all_results)
        os.makedirs(os.path.dirname(args.save_results_csv) or ".", exist_ok=True)
        results_df.to_csv(args.save_results_csv, index=False)
        print(f"Saved summary CSV to: {args.save_results_csv}")


if __name__ == "__main__":
    main()
