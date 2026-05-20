from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DEMO_DIR = DATA_DIR / "demo"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

TARGET_COL = "Class"
BASE_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
ENGINEERED_FEATURES = ["Amount_log1p", "Hour_sin", "Hour_cos"]
ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

DEFAULT_MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
DEFAULT_META_PATH = MODELS_DIR / "fraud_metadata.json"
DEFAULT_REPORT_PATH = REPORTS_DIR / "evaluation_report.json"
DEFAULT_PREDICTIONS_PATH = REPORTS_DIR / "predictions.csv"
