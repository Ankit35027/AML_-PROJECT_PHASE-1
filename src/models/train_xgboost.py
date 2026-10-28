"""
Multi-Horizon XGBoost Regressor Training and Evaluation.
Trains direct gradient boosted decision tree ensembles per horizon (15m, 30m, 60m),
monitoring chronological validation error for early stopping.
"""

from typing import Dict, List, Optional, Tuple, Any
import time
import joblib
from pathlib import Path
import numpy as np
import xgboost as xgb

from src.utils.config import HORIZONS, RANDOM_SEED, MODELS_DIR
from src.evaluation.metrics import compute_regression_metrics


class MultiHorizonXGBoost:
    """Multi-Horizon Direct XGBoost Regressor."""

    def __init__(
        self,
        horizons: Optional[Dict[str, int]] = None,
        n_estimators: int = 250,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_jobs: int = -1,
    ):
        self.horizons = horizons or HORIZONS
        self.horizon_names = list(self.horizons.keys())
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "n_jobs": n_jobs,
            "random_state": RANDOM_SEED,
            "tree_method": "hist",
            "objective": "reg:squarederror",
            "eval_metric": "mae",
        }
        self.models: Dict[str, xgb.XGBRegressor] = {}
        self.feature_importances: Dict[str, np.ndarray] = {}
        self.training_durations: Dict[str, float] = {}

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        Y_train_valid: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        Y_val_valid: np.ndarray,
        early_stopping_rounds: int = 25,
    ):
        """Fit one XGBoost model per forecast horizon."""
        for h_i, h_name in enumerate(self.horizon_names):
            print(f" [XGBoost] Training horizon {h_name} (+{self.horizons[h_name] * 5} min)...")
            start_t = time.time()
            
            y_tr = Y_train[:, h_i]
            tr_mask = Y_train_valid[:, h_i]
            
            y_vl = Y_val[:, h_i]
            vl_mask = Y_val_valid[:, h_i]

            model = xgb.XGBRegressor(
                early_stopping_rounds=early_stopping_rounds,
                **self.params,
            )

            # Fit on valid training targets, evaluating on valid validation targets
            model.fit(
                X_train[tr_mask],
                y_tr[tr_mask],
                eval_set=[(X_val[vl_mask], y_vl[vl_mask])],
                verbose=False,
            )

            duration = round(time.time() - start_t, 2)
            self.training_durations[h_name] = duration
            self.models[h_name] = model
            self.feature_importances[h_name] = model.feature_importances_

            # Compute validation metrics
            val_preds = model.predict(X_val)
            val_metrics = compute_regression_metrics(y_vl, val_preds, vl_mask)
            print(
                f"   Completed in {duration}s | Best Iteration: {model.best_iteration} "
                f"| Val MAE: {val_metrics['mae']:.3f} mph | Val RMSE: {val_metrics['rmse']:.3f} mph"
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for all horizons. Returns shape (n_samples, n_horizons)."""
        preds = np.zeros((len(X), len(self.horizon_names)), dtype=np.float32)
        for h_i, h_name in enumerate(self.horizon_names):
            preds[:, h_i] = self.models[h_name].predict(X)
        return preds

    def save(self, filepath: Path):
        """Persist models and metadata."""
        joblib.dump(
            {
                "models": self.models,
                "feature_importances": self.feature_importances,
                "durations": self.training_durations,
                "params": self.params,
                "horizons": self.horizons,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: Path) -> "MultiHorizonXGBoost":
        data = joblib.load(filepath)
        instance = cls(horizons=data["horizons"])
        instance.models = data["models"]
        instance.feature_importances = data.get("feature_importances", {})
        instance.training_durations = data.get("durations", {})
        instance.params = data.get("params", {})
        return instance
