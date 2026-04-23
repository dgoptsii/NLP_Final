#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords as nltk_stopwords

warnings.filterwarnings('ignore')

SPLITS_DIR  = 'splits'
RESULTS_DIR = 'output/bilstm'
MODEL_PATH  = Path('pkl_models/bilstm.pkl')

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


def load_test(path):
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


def load_model(device):
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
    return cfg, model, vocab, scaler


def run_evaluation(df, cfg, model, vocab, scaler, device):
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


def print_metrics(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f'\n{"="*60}\n{name} METRICS\n{"="*60}')
    print(f'Accuracy:   {accuracy_score(y_true, y_pred):.4f}')
    print(f'Precision:  {precision_score(y_true, y_pred, zero_division=0):.4f}')
    print(f'Recall:     {recall_score(y_true, y_pred, zero_division=0):.4f}')
    print(f'F1:         {f1_score(y_true, y_pred, zero_division=0):.4f}')
    print(f'\nClassification report:')
    print(classification_report(y_true, y_pred,
          target_names=['Legitimate', 'Phishing'], digits=4, zero_division=0))
    print(f'Confusion matrix:\n{cm}')
    print(f'  TP={tp}  TN={tn}  FP={fp}  FN={fn}')


def run_lime(df_test, cfg, model, vocab, scaler, device):
    if not cfg['use_text']:
        print('\nfeat_only config — skipping LIME (no text to perturb).')
        return

    print(f'\n{"="*60}\nLIME EXPLAINABILITY (H2)\n{"="*60}')

    df = df_test.copy()
    df[FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS].fillna(0))

    mean_features = torch.tensor(
        df[FEATURE_COLUMNS].mean().values.astype(np.float32)
    ).unsqueeze(0).to(device)

    def predict_fn(texts, batch_size=32):
        all_probs = []
        for i in range(0, len(texts), batch_size):
            encoded = [
                torch.tensor([vocab.get(t, vocab[UNK_TOKEN])
                              for t in tokenize(text)[:MAX_LEN]], dtype=torch.long)
                for text in texts[i:i+batch_size]
            ]
            padded   = pad_sequence(encoded, batch_first=True, padding_value=0).to(device)
            features = mean_features.expand(padded.size(0), -1)
            with torch.no_grad():
                all_probs.append(F.softmax(model(padded, features), dim=1).cpu().numpy())
        return np.vstack(all_probs)

    explainer = LimeTextExplainer(class_names=['Legitimate', 'Phishing'], random_state=42)

    def aggregate_lime(emails, class_label, n_samples=30):
        sample_texts = emails['text_clean'].sample(
            n=min(n_samples, len(emails)), random_state=42).tolist()
        word_weights = defaultdict(list)
        n = len(sample_texts)
        print(f'\nAggregating LIME over {n} emails (label={class_label})...')
        for i, text in enumerate(sample_texts):
            print(f'  {i+1}/{n}', end='\r')
            try:
                e = explainer.explain_instance(
                    text, predict_fn, num_features=10, num_samples=500, labels=(1,))
                for word, weight in e.as_list(label=1):
                    w = word.lower()
                    if w not in STOP_WORDS:
                        if class_label == 1 and weight > 0:
                            word_weights[w].append(weight)
                        elif class_label == 0 and weight < 0:
                            word_weights[w].append(abs(weight))
            except Exception as ex:
                print(f'  Skipping {i+1}: {ex}')
        return pd.DataFrame([{
            'word': w, 'frequency': len(ws),
            'avg_weight': round(sum(ws) / len(ws), 4),
            'pct_emails': round(len(ws) / n * 100, 1),
        } for w, ws in word_weights.items()]).sort_values(
            'frequency', ascending=False).reset_index(drop=True)

    phishing_summary = aggregate_lime(df[df['label'] == 1].reset_index(drop=True), 1)
    legit_summary    = aggregate_lime(df[df['label'] == 0].reset_index(drop=True), 0)

    print(f'\n=== Top 10 Phishing Keywords (LIME — BiLSTM) ===')
    print(phishing_summary.head(10).to_string(index=False))
    print(f'\n=== Top 10 Legitimate Keywords (LIME — BiLSTM) ===')
    print(legit_summary.head(10).to_string(index=False))

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    lime_combined = pd.concat([
        phishing_summary.head(10).assign(direction='phishing'),
        legit_summary.head(10).assign(direction='legitimate'),
    ]).reset_index(drop=True)
    lime_combined.to_csv(f'{RESULTS_DIR}/lime_keywords.csv', index=False)
    print(f'\nLIME results saved to {RESULTS_DIR}/lime_keywords.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=Path, required=True, help='Path to test.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg, model, vocab, scaler = load_model(device)
    df_test = load_test(args.test)
    y_true, y_pred = run_evaluation(df_test, cfg, model, vocab, scaler, device)
    print_metrics('TEST', y_true, y_pred)
    run_lime(df_test, cfg, model, vocab, scaler, device)


if __name__ == '__main__':
    main()