from typing import Protocol, Optional
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from models.threshold_optimizer import find_best_threshold
import logging

from models.models import (
    get_catboost,
    get_lightgbm,
    get_logistic_regression,
    get_mlp,
    get_random_forest,
    get_xgboost,
)
from models.model_adapter import CatBoostAdapter

logger = logging.getLogger(__name__)


class ModelProtocol(Protocol):
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "ModelProtocol": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


model: ModelProtocol


def get_models(scale_pos_weight: float) -> dict[str, ModelProtocol]:
    """
    Define and return a dictionary of models for benchmarking.
    """
    return {
        "LogisticRegression": get_logistic_regression(),
        "RandomForest": get_random_forest(),
        "XGBoost": get_xgboost(scale_pos_weight),
        "MLP": get_mlp(),
        "CatBoost": CatBoostAdapter(get_catboost(scale_pos_weight)),
        "LightGBM": get_lightgbm(scale_pos_weight),
    }


def cross_validate_models(
    models: dict[str, ModelProtocol],
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    n_splits: int = 5,
) -> dict[str, dict[str, float]]:
    """
    Perform Stratified K-Fold cross-validation for all models.
    Tracks metrics per fold and logs them
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    aggregated_results: dict[str, list[dict[str, float]]] = {
        name: [] for name in models.keys()
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logger.info(f"Starting Fold {fold}/{n_splits}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        for name, model in models.items():
            logger.info(f"[Fold {fold}] Training {name}")

            # Fit model
            if name == "MLP":
                sw = sample_weights[train_idx] if sample_weights is not None else None
                model.fit(X_train, y_train, sample_weight=sw)
            else:
                model.fit(X_train, y_train)

            # Predictions
            y_probs = model.predict_proba(X_val)[:, 1]
            best_thresh, best_f1 = find_best_threshold(y_val, y_probs)
            y_pred = (y_probs >= best_thresh).astype(int)

            roc = roc_auc_score(y_val, y_probs)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

            # Save fold metrics
            aggregated_results[name].append(
                {
                    "roc": roc,
                    "f1": best_f1,
                    "threshold": best_thresh,
                    "TP": tp,
                    "FP": fp,
                    "TN": tn,
                    "FN": fn,
                }
            )

            # Print fold metrics
            logger.info(
                f"[Fold {fold}] {name} → "
                f"ROC-AUC: {roc:.3f}, F1: {best_f1:.3f}, Threshold: {best_thresh:.2f}, "
                f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}"
            )

    # Average metrics per model
    final_results: dict[str, dict[str, float]] = {}

    for name, metrics_list in aggregated_results.items():
        rocs = [m["roc"] for m in metrics_list]
        f1s = [m["f1"] for m in metrics_list]
        thresholds = [m["threshold"] for m in metrics_list]
        tps = [m["TP"] for m in metrics_list]
        fps = [m["FP"] for m in metrics_list]
        tns = [m["TN"] for m in metrics_list]
        fns = [m["FN"] for m in metrics_list]

        final_results[name] = {
            "mean_roc_auc": float(np.mean(rocs)),
            "mean_f1": float(np.mean(f1s)),
            "mean_threshold": float(np.mean(thresholds)),
            "mean_TP": float(np.mean(tps)),
            "mean_FP": float(np.mean(fps)),
            "mean_TN": float(np.mean(tns)),
            "mean_FN": float(np.mean(fns)),
            "TPR": float(np.mean(tps) / (np.mean(tps) + np.mean(fns))),
            "FPR": float(np.mean(fps) / (np.mean(fps) + np.mean(tns))),
            "Precision": float(np.mean(tps) / (np.mean(tps) + np.mean(fps))),
            "Accuracy": float(
                (np.mean(tps) + np.mean(tns))
                / (np.mean(tps) + np.mean(tns) + np.mean(fps) + np.mean(fns))
            ),
        }

    return final_results
