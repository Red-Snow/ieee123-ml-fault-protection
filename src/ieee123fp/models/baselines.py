"""Black-box classifiers: the "standard ML" the study compares against."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_LABELS = {"RandomForest": "Random forest", "HistGradBoost": "Gradient boosting", "MLP": "MLP"}


def model_zoo() -> dict[str, Callable[[], ClassifierMixin]]:
    """Name -> factory for a fresh, seeded, unfitted model."""
    return {
        "RandomForest": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=1, n_jobs=-1, random_state=0),
        "HistGradBoost": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0),
        "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4,
                                                                     early_stopping=True, max_iter=300, random_state=0)),
    }


def protection_metrics(y_true, y_pred) -> tuple[float, float]:
    """Dependability = share of real faults detected; security = share of normal events not tripped."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    dep = float((y_pred[y_true == 1] == 1).mean())
    sec = float((y_pred[y_true == 0] == 0).mean())
    return dep, sec
