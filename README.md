# Evaluating the Robustness of NLP Models for Phishing Email Detection
Authors: Daria Goptsii, Giane Mayumi Galhard, Sarah Al Taleb <br>

This project evaluates the robustness of NLP-based phishing detection models under text obfuscation and their reliance on keyword-based features. We compare seven models, including Gaussian Naive Bayes, TF-IDF Multinomial Naive Bayes, TF-IDF Logistic Regression, CNN, BiLSTM, and transformer-based models (BERT, DistilBERT), trained on the Nazario and Enron datasets using text and engineered features.
All models are evaluated on clean and obfuscated data using a homograph attack. Results show high recall across models, with transformers achieving the best overall performance while remaining robust to obfuscation. Explainability analysis reveals consistent reliance on phishing-related keywords.

See report.pdf for full results and analysis.

## Usage

**Step 0 — Clone repository and install dependencies**

**Step 1 — Download required data:**
- **[Nazario Phishing Corpus](https://monkey.org/~jose/phishing/)**: place the `.mbox` file in `data/raw/mbox/`.
- **[Enron Email Dataset](https://www.cs.cmu.edu/~enron/)**: place the `maildir` folder in `data/raw/maildir/`.
- **[GloVe 6B embeddings](https://nlp.stanford.edu/projects/glove/)**: download `glove.6B.zip`, extract, and place `glove.6B.300d.txt` in `glove/`.

**Step 2 — Preprocessing and build dataset splits:**
```bash
python scripts/create_mailbox.py --input data/raw/maildir --output data/raw/mbox/enron.mbox
python scripts/parse_mailbox.py --mbox data/raw/mbox/enron.mbox --out data/processed/enron_parsed.csv --label 0 --source enron
python scripts/parse_mailbox.py --mbox data/raw/mbox/phishing3.mbox --out data/processed/nazario_parsed.csv --label 1 --source nazario
python scripts/build_dataset.py --enron data/processed/enron_parsed.csv --nazario data/processed/nazario_parsed.csv --out-dir data/processed/splits
```

**Step 3 — Generate obfuscated test set:**
```bash
python scripts/obfuscate.py
```

**Step 4 — Tune and train models:**
```bash
python tuning/tune_<model>.py --train data/processed/splits/train.csv --val data/processed/splits/val.csv --test data/processed/splits/test.csv
```
Where `<model>` is one of: `lr`, `nb`, `bilstm`, `charcnn`.

**Step 5 — Evaluate on clean test set:**
```bash
python <model>.py --test data/processed/splits/test.csv
```
Where `<model>` is one of: `lr`, `nb_tfidf`, `nb_gaussian`, `bilstm`, `charcnn`.

**Step 6 — Run stress tests (obfuscation robustness):**
```bash
python stress_test/stress_<model>.py --test data/processed/splits/test.csv --obfuscated data/processed/splits/test_obfuscated_homograph.csv
```
Where `<model>` is one of: `lr`, `nb_tfidf`, `biltsm`, `charcnn`.

**Step 7 — Run full transformer pipeline (training, stress test, and LIME):**
```bash
python transformers/run_all_transformers.py --splits_dir data/processed/splits --test_csv data/processed/splits/test.csv --obfuscated_csv data/processed/splits/test_obfuscated_homograph.csv
```
