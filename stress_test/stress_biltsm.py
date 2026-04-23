#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords as nltk_stopwords

MODEL_PATH = Path('pkl_models/bilstm.pkl')

STOP_WORDS = set(nltk_stopwords.words('english'))
MAX_LEN    = 200
EMBED_DIM  = 300
BATCH_SIZE = 32
PAD_TOKEN  = '<PAD>'
UNK_TOKEN  = '<UNK>'

FEATURE_COLUMNS = [
    'char_len', 'token_len', 'subject_len', 'num_urls', 'num_emails',
    'num_domains', 'num_exclamation', 'num_upper_tokens', 'keyword_hits',
    'num_attachments', 'has_form_tag', 'has_script_tag', 'has_iframe_tag',
    'has_ip_url', 'has_external_links',
]


def tokenize(text):
    return [t for t in str(text).lower().split() if t not in STOP_WORDS]


def load_data(path):
    df = pd.read_csv(path)
    df['label']      = pd.to_numeric(df['label'], errors='raise').astype(int)
    df['subject']    = df['subject'].fillna('').astype(str)
    df['text']       = df['text'].fillna('').astype(str)
    df['text_clean'] = (df['subject'] + ' ' + df['text']).str.strip()
    return df


class EmailDataset(Dataset):
    def __init__(self, df, vocab, use_text, use_features):
        self.df           = df.reset_index(drop=True)
        self.vocab        = vocab
        self.use_text     = use_text
        self.use_features = use_features

    def __len__(self):
        return len(self.df)

    def encode(self, text):
        tokens = tokenize(text)[:MAX_LEN]
        return torch.tensor([self.vocab.get(t, self.vocab[UNK_TOKEN])
                             for t in tokens], dtype=torch.long)

    def __getitem__(self, idx):
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


def collate_fn(batch):
    return {
        'input':    pad_sequence([b['input'] for b in batch],
                                 batch_first=True, padding_value=0),
        'label':    torch.stack([b['label']    for b in batch]),
        'features': torch.stack([b['features'] for b in batch]),
    }


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embed_matrix,
                 use_text, use_features, dropout=0.0):
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

    def forward(self, input_ids, features):
        parts = []
        if self.use_text:
            emb = self.drop_text(self.embedding(input_ids))
            _, (h, _) = self.bilstm(emb)
            parts.append(self.drop_text(torch.cat([h[-2], h[-1]], dim=1)))
        if self.use_features:
            parts.append(self.feature_mlp(features))
        return self.classifier(torch.cat(parts, dim=1))


def predict(df, cfg, model, vocab, scaler, device):
    df = df.copy()
    df[FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS].fillna(0))
    loader = DataLoader(
        EmailDataset(df, vocab, cfg['use_text'], cfg['use_features']),
        BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
    )
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['input'].to(device), batch['features'].to(device))
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(batch['label'].numpy())
    return np.array(labels), np.array(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',       type=Path, required=True,
                        help='Path to test.csv')
    parser.add_argument('--obfuscated', type=Path, required=True,
                        help='Path to test_obfuscated_homograph.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bundle = pickle.load(open(str(MODEL_PATH), 'rb'))
    cfg    = bundle['cfg']
    vocab  = bundle['vocab']
    scaler = bundle['scaler']
    model  = BiLSTMClassifier(
        vocab_size=len(vocab), hidden_dim=cfg['hidden_dim'],
        embed_matrix=np.zeros((len(vocab), EMBED_DIM), dtype=np.float32),
        use_text=cfg['use_text'], use_features=cfg['use_features'],
    ).to(device)
    model.load_state_dict(bundle['state_dict'])
    model.eval()

    df_test  = load_data(args.test)
    df_obfus = load_data(args.obfuscated)

    y_true, clean_pred = predict(df_test,  cfg, model, vocab, scaler, device)
    _,      obfus_pred = predict(df_obfus, cfg, model, vocab, scaler, device)

    clean_recall = recall_score(y_true, clean_pred, zero_division=0)
    clean_f1     = f1_score(y_true,     clean_pred, zero_division=0)
    clean_acc    = accuracy_score(y_true, clean_pred)

    obfus_recall = recall_score(y_true, obfus_pred, zero_division=0)
    obfus_f1     = f1_score(y_true,     obfus_pred, zero_division=0)
    obfus_acc    = accuracy_score(y_true, obfus_pred)

    recall_drop = (clean_recall - obfus_recall) / clean_recall * 100 if clean_recall > 0 else 0.0

    print(f"\n{'Technique':<35} {'Recall':>8} {'F1':>8} {'Accuracy':>10} {'Recall drop':>12}")
    print('-' * 75)
    print(f"  {'Original (no obfuscation)':<33} "
          f"{clean_recall:>8.4f} {clean_f1:>8.4f} {clean_acc:>10.4f} {'—':>12}")
    print(f"  {'Homograph obfuscation':<33} "
          f"{obfus_recall:>8.4f} {obfus_f1:>8.4f} {obfus_acc:>10.4f} {recall_drop:>11.2f}%")

    h1 = (clean_recall - obfus_recall) > 0.10
    print(f'\nH1 result: recall dropped by {clean_recall - obfus_recall:.4f} '
          f'({(clean_recall - obfus_recall)*100:.1f}%)')
    print(f'H1 {"SUPPORTED" if h1 else "NOT SUPPORTED"} (threshold: >10% drop)')


if __name__ == '__main__':
    main()