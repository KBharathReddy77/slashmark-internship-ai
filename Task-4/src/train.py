from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from config import DEFAULT_META_PATH, DEFAULT_MODEL_PATH, TARGET_COL, REPORTS_DIR
from data_loading import load_dataset, split_xy
from modeling import fit_best_model, bundle_to_artifacts

def main():
    df = load_dataset()

    X, y = split_xy(df)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    bundle, threshold_summary = fit_best_model(X_train, y_train, X_val, y_val)

    meta = bundle_to_artifacts(bundle, DEFAULT_MODEL_PATH, DEFAULT_META_PATH)

    # Save a test set for evaluation
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(REPORTS_DIR / "test_split.csv", index=False)

    summary = {
        "model_name": bundle.model_name,
        "threshold": bundle.threshold,
        "threshold_summary": threshold_summary,
        "cv_summary": bundle.cv_summary,
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
    }
    (REPORTS_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Training completed successfully.")
    print(f"Best model: {bundle.model_name}")
    print(f"Saved model: {DEFAULT_MODEL_PATH}")
    print(f"Saved metadata: {DEFAULT_META_PATH}")
    print(f"Chosen threshold: {bundle.threshold:.4f}")

if __name__ == "__main__":
    main()
