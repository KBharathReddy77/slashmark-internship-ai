from __future__ import annotations

import argparse
from pathlib import Path
import json

import joblib
import pandas as pd

from config import DEFAULT_META_PATH, DEFAULT_MODEL_PATH, DEFAULT_PREDICTIONS_PATH, BASE_FEATURES
from data_loading import find_dataset_file
from modeling import evaluate_at_threshold

def predict_csv(input_path: Path, output_path: Path | None = None):
    model = joblib.load(DEFAULT_MODEL_PATH)
    meta = json.loads(DEFAULT_META_PATH.read_text(encoding="utf-8"))
    threshold = float(meta["threshold"])

    df = pd.read_csv(input_path)
    missing = set(BASE_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    probs = model.predict_proba(df[BASE_FEATURES])[:, 1]
    preds = (probs >= threshold).astype(int)

    out = df.copy()
    out["fraud_probability"] = probs
    out["fraud_prediction"] = preds

    if output_path is None:
        output_path = DEFAULT_PREDICTIONS_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(out.head(10).to_string(index=False))
    print(f"Saved predictions to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Predict fraud probability for a CSV of transactions.")
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV. If omitted, uses demo sample.")
    parser.add_argument("--output", type=str, default=None, help="Path to output CSV.")
    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = Path("data/demo/sample_transactions.csv")
        if not input_path.exists():
            input_path = find_dataset_file()

    output_path = Path(args.output) if args.output else None
    predict_csv(input_path, output_path)

if __name__ == "__main__":
    main()
