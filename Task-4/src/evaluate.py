from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)

from config import DEFAULT_META_PATH, DEFAULT_MODEL_PATH, DEFAULT_REPORT_PATH, REPORTS_DIR, TARGET_COL
from data_loading import split_xy
from preprocessing import FraudFeatureEngineer

def main():
    test_path = REPORTS_DIR / "test_split.csv"
    if not test_path.exists():
        raise FileNotFoundError("test_split.csv not found. Run src/train.py first.")

    df = pd.read_csv(test_path)
    X, y = split_xy(df)

    model = joblib.load(DEFAULT_MODEL_PATH)
    meta = json.loads(DEFAULT_META_PATH.read_text(encoding="utf-8"))
    threshold = float(meta["threshold"])

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    report = {
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "average_precision": float(average_precision_score(y, y_prob)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "mcc": float(matthews_corrcoef(y, y_pred)),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(y, y_pred, zero_division=0, output_dict=True),
    }

    DEFAULT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
