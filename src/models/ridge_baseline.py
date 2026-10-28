"""
Ridge Regression Multi-Horizon Baseline Model.
Trains independent L2-regularized linear regressors per forecast horizon (+15m, +30m, +60m),
tuning alpha strictly on the chronological validation set.
"""

from typing import Dict, List, Optional, Tuple
import joblib
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge

from src.utils.config import HORIZONS, RANDOM_SEED, MODELS_DIR
from src.evaluation.metrics import compute_regression_metrics


class MultiHorizonRidge:
    """Multi-Horizon Direct Ridge Regression model."""

    def __init__(self, alpha: float = 10.0, horizons: Optional[Dict[str, int]] = None):
        self.alpha = alpha
        self.horizons = horizons or HORIZONS
        self.horizon_names = list(self.horizons.keys())
        self.models: Dict[str, Ridge] = {}
        self.best_alphas: Dict[str, float] = {}

    def fit_and_tune(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        Y_train_valid: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        Y_val_valid: np.ndarray,
        candidate_alphas: List[float] = [0.1, 1.0, 10.0, 50.0, 100.0],
    ):
        """
        Fit one Ridge model per horizon, selecting best alpha based on validation MAE.
        """
        for h_i, h_name in enumerate(self.horizon_names):
            y_tr = Y_train[:, h_i]
            val_tr_mask = Y_train_valid[:, h_i]
            
            y_vl = Y_val[:, h_i]
            val_vl_mask = Y_val_valid[:, h_i]

            best_alpha = candidate_alphas[0]
            best_val_mae = float("inf")
            best_model = None

            for alpha in candidate_alphas:
                model = Ridge(alpha=alpha, random_state=RANDOM_SEED)
                model.fit(X_train[val_tr_mask], y_tr[val_tr_mask])
                
                # Predict on validation
                preds = model.predict(X_val)
                metrics = compute_regression_metrics(y_vl, preds, val_vl_mask)
                if metrics["mae"] < best_val_mae:
                    best_val_mae = metrics["mae"]
                    best_alpha = alpha
                    best_model = model

            self.models[h_name] = best_model
            self.best_alphas[h_name] = best_alpha
            print(f" [Ridge] Horizon {h_name}: Best alpha={best_alpha}, Validation MAE={best_val_mae:.3f} mph")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for all horizons. Returns shape (n_samples, n_horizons)."""
        preds = np.zeros((len(X), len(self.horizon_names)), dtype=np.float32)
        for h_i, h_name in enumerate(self.horizon_names):
            preds[:, h_i] = self.models[h_name].predict(X)
        return preds

    def save(self, filepath: Path):
        """Save models to disk."""
        joblib.dump({"models": self.models, "alphas": self.best_alphas, "horizons": self.horizons}, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "MultiHorizonRidge":
        data = joblib.load(filepath)
        instance = cls(horizons=data["horizons"])
        instance.models = data["models"]
        instance.best_alphas = data["alphas"]
        return instance
