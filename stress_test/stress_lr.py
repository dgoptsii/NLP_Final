#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, f1_score, recall_score
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords as nltk_stopwords

MODEL_PATH = Path('pkl_models/lr.pkl')

STOP_WORDS = set(nltk_stopwords.words('english'))

FEATURE_COLUMNS = [
    'char_len', 'token_len', 'subject_len', 'num_urls', 'num_emails',
    'num_domains', 'num_exclamation', 'num_upper_tokens', 'keyword_hits',
    'num_attachments', 'has_form_tag', 'has_script_tag', 'has_iframe_tag',
    'has_ip_url', 'has_external_links',
]


def load_data(path: Path) -> pd.DataFrame:
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
    return sp.hstack([X_tfidf, X_feat]), df['label'].values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',       type=Path, required=True,
                        help='Path to test.csv')
    parser.add_argument('--obfuscated', type=Path, required=True,
                        help='Path to test_obfuscated_homograph.csv')
    args = parser.parse_args()

    bundle     = pickle.load(open(str(MODEL_PATH), 'rb'))
    cfg        = bundle['cfg']
    model      = bundle['model']
    vectorizer = bundle['vectorizer']
    scaler     = bundle['scaler']

    df_test  = load_data(args.test)
    df_obfus = load_data(args.obfuscated)

    X_clean, y_true  = prepare(df_test,  cfg, vectorizer, scaler)
    X_obfus, _       = prepare(df_obfus, cfg, vectorizer, scaler)

    clean_pred = model.predict(X_clean)
    obfus_pred = model.predict(X_obfus)

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