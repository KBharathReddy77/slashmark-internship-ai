from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parent.parent

report_path = BASE_DIR / "reports" / "plagiarism_report.csv"

df = pd.read_csv(report_path)

y_true = [1,1,0]

y_pred = [1 if x == "Plagiarized" else 0 for x in df["Prediction"]]

print(classification_report(y_true, y_pred))