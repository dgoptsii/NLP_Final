

## Pipeline Summary

The project follows this pipeline:

```
Enron maildir → mbox → parsed CSV
Nazario mbox → parsed CSV
↓
merge + clean + split
↓
train / val / test datasets
↓
logistic regression baseline
```

---

## Project Structure

```
data/
  raw/
    maildir/              # Enron raw dataset
    mbox/                 # converted mbox files
  processed/
    enron_parsed.csv
    nazario_parsed.csv
    splits/
      train.csv
      val.csv
      test.csv
      dataset_metadata.json

scripts/
  create_mailbox.py
  parse_mailbox.py
  build_dataset_splits.py
  evaluate_split_model.py
```

---

## Step-by-Step Usage

### 1. Convert Enron dataset → mbox

The Enron dataset comes as a folder structure. It must be converted into a single `.mbox` file.

Run:

```bash
python create_mailbox.py \
  --input data/raw/maildir \
  --output data/raw/mbox/enron.mbox
```

Optional:

```bash
--max-messages 5000
--include-all-folders
```

This script creates a unified mailbox file for easier processing. 

---

### 2. Parse mailboxes into structured CSV

Parse both datasets using the same pipeline:

#### Enron (label = 0)

```bash
python parse_mailbox.py \
  --mbox data/raw/mbox/enron.mbox \
  --out data/processed/enron_parsed.csv \
  --label 0 \
  --source enron
```

#### Nazario (label = 1)

```bash
python parse_mailbox.py \
  --mbox data/raw/mbox/phishing3.mbox \
  --out data/processed/nazario_parsed.csv \
  --label 1 \
  --source nazario
```

This step:

* extracts subject + body
* cleans text
* builds features (URLs, domains, etc.)
* removes duplicates
* prints detailed statistics 

---

### 3. Build train / val / test splits

```bash
python build_dataset.py \
  --enron data/processed/enron_parsed.csv \
  --nazario data/processed/nazario_parsed.csv \
  --out-dir data/processed/splits \
  --train-size 0.8 \
  --val-size 0.1 \
  --test-size 0.1 \
  --random-seed 42
```

This step:

* merges datasets
* removes invalid rows
* creates stratified splits
* balances classes (downsampling)
* prints discard statistics
* saves:

  * `train.csv`
  * `val.csv`
  * `test.csv`
  * metadata JSON 

---

### 4. Train and evaluate model

```bash
python evaluate_split_model.py \
  --train data/processed/splits/train.csv \
  --val data/processed/splits/val.csv \
  --test data/processed/splits/test.csv \
  --random-seed 42
```

This step:

* trains Logistic Regression on **train only**
* evaluates on:

  * train (overfitting check)
  * validation
  * test (final result)
* prints metrics:

  * accuracy
  * balanced accuracy
  * specificity
  * F1 score
  * confusion matrix
* shows top important features 

---

## Features Used

The model uses **structural + lexical features** such as:

* email length (`char_len`, `token_len`)
* number of URLs / domains
* number of attachments
* phishing keywords
* HTML indicators (`form`, `script`, `iframe`)
* suspicious patterns (`has_ip_url`, `external_links`)
* removed -> has_html, has_embedded_images because only NAZARIO has -> therefore model bias 

These are inspired by classical phishing detection research (e.g., Fette et al.).

---

## Results

The baseline model achieves:
* ~91% accuracy logistic regression 
* balanced performance across classes
* no significant overfitting (train ≈ val ≈ test)

---
