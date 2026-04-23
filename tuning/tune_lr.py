#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
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

STAGE1_CONFIGS = [
    {'name': 'text_only', 'use_text': True,  'use_features': False},
    {'name': 'feat_only', 'use_text': False, 'use_features': True },
    {'name': 'text_feat', 'use_text': True,  'use_features': True },
]

DEFAULT_NGRAM     = (1, 2)
DEFAULT_MAX_FEAT  = 50000
DEFAULT_SUBLINEAR = True
DEFAULT_C         = 1.0

NGRAM_GRID     = [(1, 1), (1, 2), (1, 3)]
MAX_FEAT_GRID  = [10000, 50000, 100000]
C_GRID         = [0.01, 0.1, 1.0, 10.0]
SUBLINEAR_GRID = [True, False]


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['label']   = pd.to_numeric(df['label'], errors='raise').astype(int)
    df['subject'] = df['subject'].fillna('').astype(str)
    df['text']    = df['text'].fillna('').astype(str)
    df['text_clean'] = (df['subject'] + ' ' + df['text']).str.strip()
    return df


def remove_stopwords(texts: list[str]) -> list[str]:
    return [' '.join(t for t in text.lower().split() if t not in STOP_WORDS)
            for text in texts]


def run_config(cfg: dict, df_train: pd.DataFrame,
               df_val: pd.DataFrame, df_test: pd.DataFrame):
    use_text     = cfg.get('use_text',     True)
    use_features = cfg.get('use_features', False)
    ngram_range  = cfg.get('ngram_range',  DEFAULT_NGRAM)
    max_features = cfg.get('max_features', DEFAULT_MAX_FEAT)
    sublinear_tf = cfg.get('sublinear_tf', DEFAULT_SUBLINEAR)
    C            = cfg.get('C',            DEFAULT_C)

    vectorizer = None
    if use_text:
        train_text = remove_stopwords(df_train['text_clean'].tolist())
        val_text   = remove_stopwords(df_val['text_clean'].tolist())
        test_text  = remove_stopwords(df_test['text_clean'].tolist())
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range, max_features=max_features,
            sublinear_tf=sublinear_tf, min_df=2, strip_accents='unicode',
        )
        X_train = vectorizer.fit_transform(train_text)
        X_val   = vectorizer.transform(val_text)
        X_test  = vectorizer.transform(test_text)

    scaler = None
    if use_features:
        scaler  = StandardScaler()
        f_train = sp.csr_matrix(scaler.fit_transform(df_train[FEATURE_COLUMNS].fillna(0)))
        f_val   = sp.csr_matrix(scaler.transform(df_val[FEATURE_COLUMNS].fillna(0)))
        f_test  = sp.csr_matrix(scaler.transform(df_test[FEATURE_COLUMNS].fillna(0)))

    if use_text and use_features:
        X_train = sp.hstack([X_train, f_train])
        X_val   = sp.hstack([X_val,   f_val])
        X_test  = sp.hstack([X_test,  f_test])
    elif use_features:
        X_train, X_val, X_test = f_train, f_val, f_test

    model = LogisticRegression(
        C=C, class_weight='balanced', max_iter=1000,
        solver='lbfgs', random_state=42,
    )
    model.fit(X_train, df_train['label'].values)

    val_pred  = model.predict(X_val)
    test_pred = model.predict(X_test)

    return {
        'val_recall':    recall_score(df_val['label'].values,  val_pred,  zero_division=0),
        'val_f1':        f1_score(df_val['label'].values,      val_pred,  zero_division=0),
        'test_recall':   recall_score(df_test['label'].values, test_pred, zero_division=0),
        'test_f1':       f1_score(df_test['label'].values,     test_pred, zero_division=0),
        'test_accuracy': accuracy_score(df_test['label'].values, test_pred),
    }, model, vectorizer, scaler


def print_comparison(results: list[dict], label: str) -> None:
    df = pd.DataFrame(results).sort_values('val_recall', ascending=False)
    print(f'\n{"="*80}\n{label}\n{"="*80}')
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True, help='Path to train.csv')
    parser.add_argument('--val',   type=Path, required=True, help='Path to val.csv')
    parser.add_argument('--test',  type=Path, required=True, help='Path to test.csv')
    args = parser.parse_args()

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_train = load_split(args.train)
    df_val   = load_split(args.val)
    df_test  = load_split(args.test)

    # ── Stage 1 — Input ablation
    print(f'\n{"="*70}\nSTAGE 1 — Input Ablation (text_only / feat_only / text_feat)\n{"="*70}')
    stage1_rows = []
    best_stage1, best_stage1_val_recall = None, -1

    for cfg in STAGE1_CONFIGS:
        metrics, model, vectorizer, scaler = run_config(cfg, df_train, df_val, df_test)
        row = {'config': cfg['name'], 'use_text': cfg['use_text'],
               'use_features': cfg['use_features'], **metrics}
        stage1_rows.append({**row, '_model': model, '_vec': vectorizer,
                             '_scaler': scaler, '_cfg': cfg})
        print(f"  {cfg['name']:<12} val_recall={metrics['val_recall']:.4f}  "
              f"val_f1={metrics['val_f1']:.4f}  test_recall={metrics['test_recall']:.4f}  "
              f"test_f1={metrics['test_f1']:.4f}")
        if metrics['val_recall'] > best_stage1_val_recall:
            best_stage1_val_recall = metrics['val_recall']
            best_stage1 = {**row, '_model': model, '_vec': vectorizer,
                           '_scaler': scaler, '_cfg': cfg}

    print(f'\nStage 1 winner: {best_stage1["config"]}  '
          f'val_recall={best_stage1["val_recall"]:.4f}  '
          f'test_f1={best_stage1["test_f1"]:.4f}  '
          f'test_recall={best_stage1["test_recall"]:.4f}')

    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                  for r in stage1_rows]).sort_values('val_recall', ascending=False).to_csv(
        f'{RESULTS_DIR}/stage1_comparison.csv', index=False
    )

    # ── Stage 2 — Hyperparameter tuning
    if not best_stage1['_cfg'].get('use_text', True):
        print('\nStage 1 winner is feat_only — no TF-IDF grid to tune.')
        best_stage2 = best_stage1
    else:
        print(f'\n{"="*70}\nSTAGE 2 — Hyperparameter Grid\n{"="*70}')
        STAGE2_CONFIGS = [{
            'name':         f'ng{"".join(map(str,ng))}_mf{mf}_C{str(C).replace(".","")}_sub{int(sub)}',
            'use_text':     best_stage1['_cfg']['use_text'],
            'use_features': best_stage1['_cfg']['use_features'],
            'ngram_range':  ng, 'max_features': mf, 'C': C, 'sublinear_tf': sub,
        } for ng in NGRAM_GRID for mf in MAX_FEAT_GRID
          for C in C_GRID for sub in SUBLINEAR_GRID]

        stage2_rows = []
        best_stage2, best_stage2_val_recall = None, -1

        for cfg in STAGE2_CONFIGS:
            metrics, model, vectorizer, scaler = run_config(cfg, df_train, df_val, df_test)
            row = {'config': cfg['name'], **metrics}
            stage2_rows.append({**row, '_model': model, '_vec': vectorizer,
                                 '_scaler': scaler, '_cfg': cfg})
            if metrics['val_recall'] > best_stage2_val_recall:
                best_stage2_val_recall = metrics['val_recall']
                best_stage2 = {**row, '_model': model, '_vec': vectorizer,
                               '_scaler': scaler, '_cfg': cfg}

        pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                      for r in stage2_rows]).sort_values('val_recall', ascending=False).to_csv(
            f'{RESULTS_DIR}/stage2_comparison.csv', index=False
        )
        print(f'\nStage 2 winner: {best_stage2["config"]}  '
              f'val_recall={best_stage2["val_recall"]:.4f}  '
              f'test_f1={best_stage2["test_f1"]:.4f}  '
              f'test_recall={best_stage2["test_recall"]:.4f}')

    # ── Save best model
    pickle.dump({
        'model':      best_stage2['_model'],
        'vectorizer': best_stage2['_vec'],
        'scaler':     best_stage2['_scaler'],
        'cfg':        best_stage2['_cfg'],
    }, open(str(MODEL_PATH), 'wb'))
    print(f'\nBest model saved to {MODEL_PATH}')


if __name__ == '__main__':
    main()