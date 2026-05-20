from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config import BASE_FEATURES, ENGINEERED_FEATURES

class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=BASE_FEATURES)

        df["Amount_log1p"] = np.log1p(df["Amount"].clip(lower=0))
        hour = (df["Time"] / 3600.0) % 24
        df["Hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["Hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

        return df[BASE_FEATURES + ENGINEERED_FEATURES].astype(float)
