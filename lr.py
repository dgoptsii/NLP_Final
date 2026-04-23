#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords as nltk_stopwords

warnings.filterwarnings('ignore')

SPLITS_DIR  = 'splits'
RESULTS_DIR = 'output/tfidf'
MODEL_PATH  = Path('pkl_models/lr.pkl')

STOP_WORDS = set(nltk_stopwords.words('english'))

FEATURE_COLUMNS = [
    'char_len', 'token_len', 'subject_len', 'num_urls', 'num_emails',
    'num_domains', 'num_exclamation', 'num_upper_tokens', 'keyword_hits',
    'num_attachments', 'has_form_tag', 'has_script_tag', 'has_iframe_tag',
    'has_ip_url', 'has_external_links',
]


def load_test(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['label']      = pd.to_numeric(df['label'], errors='raise').astype(int)
    df['subject']    = df['subject'].fillna('').astype(str)
    df['text']       = df['text'].fillna('').astype(str)
    df['text_clean'] = (df['subject'] + ' ' + df['text']).str.strip()
    return df


def remove_stopwords(texts: list[str]) -> list[str]:
    return [' '.join(t for t in text.lower().split() if t not in STOP_WORDS)
            for text in texts]


def prepare(df: pd.DataFrame, cfg: dict, vectorizer, scaler):
    if cfg.get('use_text', True) and vectorizer:
        X_tfidf = vectorizer.transform(remove_stopwords(df['text_clean'].tolist()))
    else:
        X_tfidf = sp.csr_matrix((len(df), 0))
    if cfg['use_features'] and scaler:
        X_feat = sp.csr_matrix(scaler.transform(df[FEATURE_COLUMNS].fillna(0)))
    else:
        X_feat = sp.csr_matrix((len(df), 0))
    return sp.hstack([X_tfidf, X_feat])


def print_metrics(name: str, y_true, y_pred) -> None:
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


def run_lime(df_test: pd.DataFrame, cfg: dict, model, vectorizer, scaler) -> None:
    if not cfg.get('use_text', True) or vectorizer is None:
        print('\nfeat_only config — skipping LIME (no text to perturb).')
        return

    print(f'\n{"="*60}\nLIME EXPLAINABILITY (H2)\n{"="*60}')

    def predict_fn(texts):
        X_tfidf = vectorizer.transform(remove_stopwords(texts))
        if cfg['use_features'] and scaler:
            mean_feats = df_test[FEATURE_COLUMNS].mean().values.astype(np.float32)
            X_feat = sp.csr_matrix(np.tile(mean_feats, (len(texts), 1)))
            X = sp.hstack([X_tfidf, X_feat])
        else:
            X = X_tfidf
        return model.predict_proba(X)

    explainer = LimeTextExplainer(class_names=['Legitimate', 'Phishing'], random_state=42)

    def aggregate_lime(emails: pd.DataFrame, class_label: int, n_samples: int = 30):
        sample_texts = emails['text_clean'].sample(
            n=min(n_samples, len(emails)), random_state=42
        ).tolist()
        word_weights: dict = defaultdict(list)
        n = len(sample_texts)
        print(f'\nAggregating LIME over {n} emails (label={class_label})...')
        for i, text in enumerate(sample_texts):
            print(f'  {i+1}/{n}', end='\r')
            try:
                e = explainer.explain_instance(
                    text, predict_fn, num_features=10, num_samples=500, labels=(1,)
                )
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
            'word':       w,
            'frequency':  len(ws),
            'avg_weight': round(sum(ws) / len(ws), 4),
            'pct_emails': round(len(ws) / n * 100, 1),
        } for w, ws in word_weights.items()]).sort_values('frequency', ascending=False).reset_index(drop=True)

    phishing_summary = aggregate_lime(df_test[df_test['label'] == 1].reset_index(drop=True), 1)
    legit_summary    = aggregate_lime(df_test[df_test['label'] == 0].reset_index(drop=True), 0)

    print(f'\n=== Top 10 Phishing Keywords (LIME — TF-IDF LR) ===')
    print(phishing_summary.head(10).to_string(index=False))
    print(f'\n=== Top 10 Legitimate Keywords (LIME — TF-IDF LR) ===')
    print(legit_summary.head(10).to_string(index=False))

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    lime_combined = pd.concat([
        phishing_summary.head(10).assign(direction='phishing'),
        legit_summary.head(10).assign(direction='legitimate'),
    ]).reset_index(drop=True)
    lime_combined.to_csv(f'{RESULTS_DIR}/lime_keywords.csv', index=False)
    print(f'\nLIME results saved to {RESULTS_DIR}/lime_keywords.csv')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=Path, required=True, help='Path to test.csv')
    args = parser.parse_args()

    bundle     = pickle.load(open(str(MODEL_PATH), 'rb'))
    cfg        = bundle['cfg']
    model      = bundle['model']
    vectorizer = bundle['vectorizer']
    scaler     = bundle['scaler']

    df_test = load_test(args.test)

    X_test  = prepare(df_test, cfg, vectorizer, scaler)
    y_test  = df_test['label'].values
    y_pred  = model.predict(X_test)

    print_metrics('TEST', y_test, y_pred)
    run_lime(df_test, cfg, model, vectorizer, scaler)


if __name__ == '__main__':
    main()