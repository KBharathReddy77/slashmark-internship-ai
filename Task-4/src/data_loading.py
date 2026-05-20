from __future__ import annotations

from pathlib import Path
import pandas as pd

from config import RAW_DIR, DEMO_DIR, TARGET_COL, BASE_FEATURES

def find_dataset_file(explicit_path: str | None = None) -> Path:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))

    candidates.extend([
        RAW_DIR / "creditcard.csv",
        RAW_DIR / "CreditCard.csv",
        RAW_DIR / "credit_card.csv",
        DEMO_DIR / "creditcard_demo.csv",
    ])

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    available = []
    for folder in [RAW_DIR, DEMO_DIR]:
        if folder.exists():
            available.extend([p.name for p in folder.glob("*.csv")])

    message = (
        "No dataset file found. Expected Kaggle file at data/raw/creditcard.csv. "
        f"Available CSV files: {available if available else 'none'}"
    )
    raise FileNotFoundError(message)

def load_dataset(explicit_path: str | None = None) -> pd.DataFrame:
    path = find_dataset_file(explicit_path)
    df = pd.read_csv(path)

    required = set(BASE_FEATURES + [TARGET_COL])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Dataset schema mismatch. Missing columns: "
            + ", ".join(sorted(missing))
            + f". Found columns: {list(df.columns)[:10]}..."
        )

    return df

def split_xy(df: pd.DataFrame):
    X = df[BASE_FEATURES].copy()
    y = df[TARGET_COL].astype(int).copy()
    return X, y
