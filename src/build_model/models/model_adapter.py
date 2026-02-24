from typing import Optional

from catboost import CatBoostClassifier
import numpy as np

"""
Adapter class to wrap models so that it conforms to a unified
interface with .fit(), .predict(), and .predict_proba() methods.
"""


class CatBoostAdapter:
    def __init__(self, model: CatBoostClassifier, early_stopping_rounds: int = 50):
        self.model = model
        self.early_stopping_rounds = early_stopping_rounds

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "CatBoostAdapter":
        self.model.fit(
            X,
            y,
            sample_weight=sample_weight,
            verbose=0,
            **kwargs,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
