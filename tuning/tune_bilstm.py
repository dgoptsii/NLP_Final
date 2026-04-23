#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import urllib.request
import warnings
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords as nltk_stopwords

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

SPLITS_DIR  = 'splits'
RESULTS_DIR = 'output/bilstm'
MODEL_PATH  = Path('pkl_models/bilstm.pkl')
GLOVE_DIR   = 'glove'

STOP_WORDS = set(nltk_stopwords.words('english'))
MAX_LEN    = 200
EMBED_DIM  = 300
BATCH_SIZE = 32
EPOCHS     = 10
PATIENCE   = 3
MIN_FREQ   = 2
PAD_TOKEN  = '<PAD>'
UNK_TOKEN  = '<UNK>'

FEATURE_COLUMNS = [
    'char_len', 'token_len', 'subject_len', 'num_urls', 'num_emails',
    'num_domains', 'num_exclamation', 'num_upper_tokens', 'keyword_hits',
    'num_attachments', 'has_form_tag', 'has_script_tag', 'has_iframe_tag',
    'has_ip_url', 'has_external_links',
]

STAGE1_CONFIGS = [
    {'name': 'text_only', 'use_text': True,  'use_features': False,
     'lr': 1e-3, 'hidden_dim': 256, 'dropout': 0.3},
    {'name': 'feat_only', 'use_text': False, 'use_features': True,
     'lr': 1e-3, 'hidden_dim': 256, 'dropout': 0.3},
    {'name': 'text_feat', 'use_text': True,  'use_features': True,
     'lr': 1e-3, 'hidden_dim': 256, 'dropout': 0.3},
]

LR_GRID         = [1e-3, 5e-4, 1e-4]
HIDDEN_DIM_GRID = [128, 256, 512]


# ── GloVe

def download_glove(glove_dir: str = GLOVE_DIR) -> str:
    glove_dir  = Path(glove_dir)
    glove_dir.mkdir(parents=True, exist_ok=True)
    glove_file = glove_dir / 'glove.6B.300d.txt'
    glove_zip  = glove_dir / 'glove.6B.zip'
    if glove_file.exists():
        print('GloVe already downloaded ✓')
        return str(glove_file)
    if not glove_zip.exists():
        print('Downloading GloVe 6B (862MB)...')
        def progress(b, bs, ts):
            if b % 500 == 0:
                print(f'  {min(b*bs/ts*100, 100):.1f}%', end='\r')
        urllib.request.urlretrieve('https://nlp.stanford.edu/data/glove.6B.zip',
                                   str(glove_zip), reporthook=progress)
        print('\nDownload complete ✓')
    print('Extracting...')
    with zipfile.ZipFile(str(glove_zip), 'r') as z:
        z.extract('glove.6B.300d.txt', str(glove_dir))
    for f in ['glove.6B.50d.txt', 'glove.6B.100d.txt', 'glove.6B.200d.txt']:
        p = glove_dir / f
        if p.exists():
            p.unlink()
    return str(glove_file)


# ── Preprocessing

def tokenize(text: str) -> list[str]:
    return [t for t in str(text).lower().split() if t not in STOP_WORDS]


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['label']      = pd.to_numeric(df['label'], errors='raise').astype(int)
    df['subject']    = df['subject'].fillna('').astype(str)
    df['text']       = df['text'].fillna('').astype(str)
    df['text_clean'] = (df['subject'] + ' ' + df['text']).str.strip()
    return df


def build_vocab(texts: list[str], min_freq: int = MIN_FREQ) -> dict:
    counts = Counter()
    for t in texts:
        counts.update(tokenize(t))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, count in counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def build_embedding_matrix(glove_file: str, vocab: dict) -> np.ndarray:
    glove = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] in vocab:
                glove[parts[0]] = np.array(parts[1:], dtype=np.float32)
    matrix = np.random.uniform(-0.1, 0.1, (len(vocab), EMBED_DIM)).astype(np.float32)
    matrix[0] = np.zeros(EMBED_DIM)
    hits = 0
    for word, idx in vocab.items():
        if word in glove:
            matrix[idx] = glove[word]
            hits += 1
    print(f'GloVe coverage: {hits}/{len(vocab)} ({hits/len(vocab)*100:.1f}%)')
    return matrix


class EmailDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: dict,
                 use_text: bool, use_features: bool):
        self.df           = df.reset_index(drop=True)
        self.vocab        = vocab
        self.use_text     = use_text
        self.use_features = use_features

    def __len__(self) -> int:
        return len(self.df)

    def encode(self, text: str) -> torch.Tensor:
        tokens = tokenize(text)[:MAX_LEN]
        return torch.tensor([self.vocab.get(t, self.vocab[UNK_TOKEN])
                             for t in tokens], dtype=torch.long)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        return {
            'input':    self.encode(str(row['text_clean'])) if self.use_text
                        else torch.zeros(1, dtype=torch.long),
            'label':    torch.tensor(int(row['label']), dtype=torch.long),
            'features': torch.tensor(
                row[FEATURE_COLUMNS].fillna(0).astype(np.float32).values
                if self.use_features
                else np.zeros(len(FEATURE_COLUMNS), dtype=np.float32),
                dtype=torch.float32,
            ),
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        'input':    pad_sequence([b['input'] for b in batch],
                                 batch_first=True, padding_value=0),
        'label':    torch.stack([b['label']    for b in batch]),
        'features': torch.stack([b['features'] for b in batch]),
    }


def make_loader(df, vocab, use_text, use_features, shuffle=False):
    return DataLoader(EmailDataset(df, vocab, use_text, use_features),
                      BATCH_SIZE, shuffle=shuffle, collate_fn=collate_fn)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int,
                 embed_matrix: np.ndarray, use_text: bool,
                 use_features: bool, dropout: float = 0.3):
        super().__init__()
        self.use_text     = use_text
        self.use_features = use_features
        if use_text:
            self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
            self.embedding.weight = nn.Parameter(
                torch.tensor(embed_matrix, dtype=torch.float32))
            self.bilstm = nn.LSTM(
                input_size=EMBED_DIM, hidden_size=hidden_dim,
                num_layers=2, batch_first=True, bidirectional=True, dropout=dropout,
            )
            self.drop_text = nn.Dropout(dropout)
        if use_features:
            self.feature_mlp = nn.Sequential(
                nn.Linear(len(FEATURE_COLUMNS), 32), nn.ReLU(), nn.Dropout(dropout),
            )
        classifier_in = (hidden_dim * 2 if use_text else 0) + (32 if use_features else 0)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, 2),
        )

    def forward(self, input_ids: torch.Tensor,
                features: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.use_text:
            emb = self.drop_text(self.embedding(input_ids))
            _, (h, _) = self.bilstm(emb)
            parts.append(self.drop_text(torch.cat([h[-2], h[-1]], dim=1)))
        if self.use_features:
            parts.append(self.feature_mlp(features))
        return self.classifier(torch.cat(parts, dim=1))


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total = 0.0
    bar = tqdm(loader, desc=f'Epoch {epoch}/{EPOCHS}', leave=True)
    for step, batch in enumerate(bar, 1):
        optimizer.zero_grad()
        loss = criterion(model(batch['input'].to(device), batch['features'].to(device)),
                         batch['label'].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        bar.set_postfix(avg_loss=f'{total/step:.4f}')


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['input'].to(device), batch['features'].to(device))
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(batch['label'].numpy())
    return np.array(labels), np.array(preds)


def run_config(cfg: dict, df_train: pd.DataFrame, df_val: pd.DataFrame,
               df_test: pd.DataFrame, glove_file: str, device: torch.device,
               stage_dir: str):
    name         = cfg['name']
    use_text     = cfg['use_text']
    use_features = cfg['use_features']
    lr           = cfg['lr']
    hidden_dim   = cfg['hidden_dim']
    dropout      = cfg['dropout']

    Path(stage_dir, name).mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    df_train = df_train.copy()
    df_val   = df_val.copy()
    df_test  = df_test.copy()
    df_train[FEATURE_COLUMNS] = scaler.fit_transform(df_train[FEATURE_COLUMNS].fillna(0))
    df_val[FEATURE_COLUMNS]   = scaler.transform(df_val[FEATURE_COLUMNS].fillna(0))
    df_test[FEATURE_COLUMNS]  = scaler.transform(df_test[FEATURE_COLUMNS].fillna(0))

    if use_text:
        vocab        = build_vocab(df_train['text_clean'].tolist())
        embed_matrix = build_embedding_matrix(glove_file, vocab)
        print(f'Vocabulary: {len(vocab):,} words')
    else:
        vocab        = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        embed_matrix = np.zeros((2, EMBED_DIM), dtype=np.float32)
        print('feat_only mode — skipping vocabulary and GloVe')

    train_loader = make_loader(df_train, vocab, use_text, use_features, shuffle=True)
    val_loader   = make_loader(df_val,   vocab, use_text, use_features)
    test_loader  = make_loader(df_test,  vocab, use_text, use_features)

    model = BiLSTMClassifier(
        vocab_size=len(vocab), hidden_dim=hidden_dim, embed_matrix=embed_matrix,
        use_text=use_text, use_features=use_features, dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2)

    best_recall, best_state, no_improve = -1.0, None, 0

    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        yv, ypv    = evaluate(model, val_loader, device)
        val_recall = recall_score(yv, ypv, zero_division=0)
        val_f1     = f1_score(yv, ypv, zero_division=0)
        scheduler.step(val_recall)
        print(f'Epoch {epoch} | recall={val_recall:.4f} | f1={val_f1:.4f}')
        if val_recall > best_recall:
            best_recall = val_recall
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve  = 0
            print(f'  New best recall={best_recall:.4f} ✓')
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print('  Early stopping')
                break

    model.load_state_dict(best_state)
    yv,  ypv  = evaluate(model, val_loader,  device)
    yte, ypte = evaluate(model, test_loader, device)

    metrics = {
        'val_recall':    recall_score(yv,  ypv,  zero_division=0),
        'val_f1':        f1_score(yv,      ypv,  zero_division=0),
        'test_recall':   recall_score(yte, ypte, zero_division=0),
        'test_f1':       f1_score(yte,     ypte, zero_division=0),
        'test_accuracy': accuracy_score(yte, ypte),
    }
    print(f'{name} — val_recall={metrics["val_recall"]:.4f}  '
          f'test_f1={metrics["test_f1"]:.4f}  '
          f'test_recall={metrics["test_recall"]:.4f}')

    json.dump({**cfg, **metrics},
              open(f'{stage_dir}/{name}/results.json', 'w'), indent=2)
    return metrics, model, vocab, scaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True, help='Path to train.csv')
    parser.add_argument('--val',   type=Path, required=True, help='Path to val.csv')
    parser.add_argument('--test',  type=Path, required=True, help='Path to test.csv')
    args = parser.parse_args()

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    glove_file = download_glove()

    df_train = load_split(args.train)
    df_val   = load_split(args.val)
    df_test  = load_split(args.test)

    # ── Stage 1
    stage1_dir = f'{RESULTS_DIR}/stage1'
    print(f'\n{"="*70}\nSTAGE 1 — Input Ablation (text_only / feat_only / text_feat)\n{"="*70}')

    stage1_rows = []
    best_stage1, best_stage1_recall = None, -1.0
    best_stage1_artifacts = None

    for cfg in STAGE1_CONFIGS:
        result_file = Path(stage1_dir) / cfg['name'] / 'results.json'
        if result_file.exists():
            metrics   = json.load(open(str(result_file)))
            artifacts = None
        else:
            metrics, *artifacts = run_config(
                cfg, df_train, df_val, df_test, glove_file, device, stage1_dir)
        stage1_rows.append({'config': cfg['name'], 'use_text': cfg['use_text'],
                            'use_features': cfg['use_features'], **metrics})
        if metrics['val_recall'] > best_stage1_recall:
            best_stage1_recall    = metrics['val_recall']
            best_stage1           = {'cfg': cfg, **metrics}
            best_stage1_artifacts = artifacts

    df_s1 = pd.DataFrame(stage1_rows).sort_values('val_recall', ascending=False)
    df_s1.to_csv(f'{RESULTS_DIR}/stage1_comparison.csv', index=False)
    print(f'\n{"="*80}\nSTAGE 1 RESULTS\n{"="*80}')
    print(df_s1.to_string(index=False))
    print(f'\nStage 1 winner: {best_stage1["cfg"]["name"]}  '
          f'val_recall={best_stage1["val_recall"]:.4f}  '
          f'test_f1={best_stage1["test_f1"]:.4f}  '
          f'test_recall={best_stage1["test_recall"]:.4f}')

    # ── Stage 2
    stage2_dir = f'{RESULTS_DIR}/stage2'
    print(f'\n{"="*70}\nSTAGE 2 — Hyperparameter Grid (lr x hidden_dim)\n{"="*70}')

    STAGE2_CONFIGS = [{
        'name':         f'lr{str(lr).replace(".", "")}_h{hd}',
        'use_text':     best_stage1['cfg']['use_text'],
        'use_features': best_stage1['cfg']['use_features'],
        'lr':           lr, 'hidden_dim': hd, 'dropout': 0.3,
    } for lr in LR_GRID for hd in HIDDEN_DIM_GRID]

    stage2_rows = []
    best_stage2, best_stage2_recall = None, -1.0
    best_stage2_artifacts = None

    for cfg in STAGE2_CONFIGS:
        result_file = Path(stage2_dir) / cfg['name'] / 'results.json'
        if result_file.exists():
            metrics   = json.load(open(str(result_file)))
            artifacts = None
        else:
            metrics, *artifacts = run_config(
                cfg, df_train, df_val, df_test, glove_file, device, stage2_dir)
        stage2_rows.append({'config': cfg['name'], **metrics})
        if metrics['val_recall'] > best_stage2_recall:
            best_stage2_recall    = metrics['val_recall']
            best_stage2           = {'cfg': cfg, **metrics}
            best_stage2_artifacts = artifacts

    df_s2 = pd.DataFrame(stage2_rows).sort_values('val_recall', ascending=False)
    df_s2.to_csv(f'{RESULTS_DIR}/stage2_comparison.csv', index=False)
    print(f'\n{"="*80}\nSTAGE 2 RESULTS\n{"="*80}')
    print(df_s2.to_string(index=False))
    print(f'\nStage 2 winner: {best_stage2["cfg"]["name"]}  '
          f'val_recall={best_stage2["val_recall"]:.4f}  '
          f'test_f1={best_stage2["test_f1"]:.4f}  '
          f'test_recall={best_stage2["test_recall"]:.4f}')

    # ── Save best model
    if best_stage2_artifacts:
        best_model, best_vocab, best_scaler = best_stage2_artifacts
        pickle.dump({
            'state_dict': best_model.state_dict(),
            'vocab':      best_vocab,
            'scaler':     best_scaler,
            'cfg':        best_stage2['cfg'],
        }, open(str(MODEL_PATH), 'wb'))
        print(f'\nBest model saved to {MODEL_PATH}')
    else:
        print(f'\nAll configs loaded from cache — pkl not updated. '
              f'Delete {RESULTS_DIR}/stage2 to retrain.')


if __name__ == '__main__':
    main()