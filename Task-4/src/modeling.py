from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from preprocessing import FraudFeatureEngineer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier

from config import TARGET_COL

@dataclass
class TrainedBundle:
    model_name: str
    pipeline: object
    threshold: float
    cv_summary: dict

def build_candidates(random_state: int = 42):
    logreg = ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("features", FraudFeatureEngineer()),
        ("scaler", StandardScaler()),
        ("sampler", RandomUnderSampler(random_state=random_state, sampling_strategy=0.15)),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"))
    ])

    brf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("features", FraudFeatureEngineer()),
        ("scaler", StandardScaler()),
        ("model", BalancedRandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
            sampling_strategy="auto",
            replacement=False
        ))
    ])

    return {
        "logreg_rus": logreg,
        "balanced_random_forest": brf,
    }

def _safe_auc_pr(y_true, y_prob):
    try:
        return average_precision_score(y_true, y_prob)
    except Exception:
        return float("nan")

def cross_validate_candidates(X_train, y_train, random_state: int = 42):
    candidates = build_candidates(random_state=random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {
        "roc_auc": "roc_auc",
        "avg_precision": "average_precision",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
    }

    results = []
    for name, pipe in candidates.items():
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
            return_train_score=False,
        )
        summary = {
            metric: float(np.mean(scores[f"test_{metric}"]))
            for metric in scoring
        }
        summary["std_avg_precision"] = float(np.std(scores["test_avg_precision"]))
        summary["std_roc_auc"] = float(np.std(scores["test_roc_auc"]))
        results.append((name, pipe, summary))

    return results

def tune_threshold(y_true, y_prob, beta: float = 2.0):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5, {"precision": 0.0, "recall": 0.0, "f_beta": 0.0}

    beta_sq = beta ** 2
    p = precision[:-1]
    r = recall[:-1]
    denom = beta_sq * p + r
    f_beta = np.where(denom == 0, 0.0, (1 + beta_sq) * p * r / denom)

    best_idx = int(np.nanargmax(f_beta))
    best_threshold = float(thresholds[best_idx])

    return best_threshold, {
        "precision": float(p[best_idx]),
        "recall": float(r[best_idx]),
        "f_beta": float(f_beta[best_idx]),
    }

def evaluate_at_threshold(y_true, y_prob, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def fit_best_model(X_train, y_train, X_val, y_val, random_state: int = 42):
    cv_results = cross_validate_candidates(X_train, y_train, random_state=random_state)

    best_name, best_pipe, best_summary = max(
        cv_results, key=lambda item: item[2]["avg_precision"]
    )

    best_pipe.fit(X_train, y_train)

    val_prob = best_pipe.predict_proba(X_val)[:, 1]
    threshold, threshold_summary = tune_threshold(y_val, val_prob, beta=2.0)

    return TrainedBundle(
        model_name=best_name,
        pipeline=best_pipe,
        threshold=threshold,
        cv_summary={
            name: summary for name, _, summary in cv_results
        },
    ), threshold_summary

def bundle_to_artifacts(bundle: TrainedBundle, model_path: Path, meta_path: Path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle.pipeline, model_path)

    meta = {
        "model_name": bundle.model_name,
        "threshold": bundle.threshold,
        "cv_summary": bundle.cv_summary,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
