"""
Historical Persistence Baseline Model.
Standard classical benchmark: assumes speed at future horizon t+h equals the most recent
observed speed at forecast origin t: y_hat(t+h) = y(t).
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

from src.utils.config import HISTORY_STEPS, HORIZONS


class PersistenceModel:
    """
    Zero-parameter classical baseline model.
    Predicts y_hat(t+h) = y(t).
    """

    def __init__(self, horizons: Dict[str, int] = HORIZONS, history_steps: int = HISTORY_STEPS):
        self.horizons = horizons
        self.history_steps = history_steps
        self.horizon_names = list(horizons.keys())

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "PersistenceModel":
        """No parameters to fit."""
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict speed for all horizons.
        X[:, 11] is speed_lag_t-0 (most recent speed at origin t).
        Returns:
            np.ndarray of shape (n_samples, len(horizons)) where each column is X[:, 11].
        """
        # Feature column index 11 is the most recent speed lag t
        origin_speeds = X[:, self.history_steps - 1]
        n_samples = len(X)
        n_horizons = len(self.horizons)
        
        # Broadcast origin speed across all horizons
        preds = np.tile(origin_speeds[:, np.newaxis], (1, n_horizons))
        return preds
