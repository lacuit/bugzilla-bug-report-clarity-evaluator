import numpy as np
from sklearn.metrics import f1_score
from typing import Tuple


def find_best_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
    """
    Find the threshold between 0.1 and 0.9 that maximizes the macro F1 score.
    Raises:
        ValueError: If y_true and y_probs have different lengths.
    """
    if len(y_true) != len(y_probs):
        raise ValueError("y_true and y_probs must have the same length")

    best_thresh: float = 0.5
    best_f1: float = 0

    # Evaluate thresholds from 0.1 to 0.9 in steps of 0.01
    for t in np.linspace(0.1, 0.9, 81):
        preds = (y_probs >= t).astype(int)
        f1 = f1_score(y_true, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1
