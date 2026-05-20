# Task 4 — Credit Card Fraud Detection

This project builds an ML pipeline for fraud detection with:
- imbalance handling
- feature engineering
- cross-validation
- AUC-ROC and AUC-PR evaluation
- threshold tuning
- cost-sensitive metrics
- a Streamlit scoring app

## Dataset

Use the Kaggle dataset **Credit Card Fraud Detection**:
- Kaggle page: `mlg-ulb/creditcardfraud`
- Typical CSV name: `creditcard.csv`

The project is wired for that file. For quick testing, a small demo CSV is included in `data/demo/creditcard_demo.csv`.

## Recommended setup

```bash
pip install -r requirements.txt
```

## Download the Kaggle dataset

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

Make sure the file becomes:

```text
data/raw/creditcard.csv
```

## Train

```bash
python src/train.py
```

## Evaluate

```bash
python src/evaluate.py
```

## Predict on a CSV

```bash
python src/predict.py --input data/demo/sample_transactions.csv
```

## Run the web app

```bash
python -m streamlit run src/app.py
```

## Project structure

```text
Task-4/
├── data/
├── models/
├── reports/
├── src/
├── notebooks/
├── requirements.txt
└── README.md
```
