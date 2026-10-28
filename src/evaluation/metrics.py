"""
Evaluation Metrics Engine for Traffic Speed Forecasting.
Implements primary (MAE), secondary (RMSE), and tertiary (masked epsilon-MAPE) metrics,
common evaluation mask enforcement, peak vs off-peak stratification, and sensor-level metrics.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from src.utils.config import EPSILON_MAPE


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid_mask: np.ndarray,
    epsilon: float = EPSILON_MAPE,
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, and masked epsilon-MAPE on valid target observations.
    
    Args:
        y_true: Ground truth speeds (mph).
        y_pred: Predicted speeds (mph).
        valid_mask: Boolean mask indicating valid target readings (>0.0 mph).
        epsilon: Minimum denominator threshold to prevent division by zero or extreme ratios.
    """
    if valid_mask is None:
        valid_mask = y_true > 0.0

    # Strict common mask
    yt = y_true[valid_mask].astype(np.float64)
    yp = y_pred[valid_mask].astype(np.float64)

    if len(yt) == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "count": 0}

    # Primary: MAE
    abs_errors = np.abs(yt - yp)
    mae = float(np.mean(abs_errors))

    # Secondary: RMSE
    squared_errors = (yt - yp) ** 2
    rmse = float(np.sqrt(np.mean(squared_errors)))

    # Tertiary: MAPE with documented epsilon policy
    # Denominator threshold: max(y_true, epsilon) prevents near-zero division
    denom = np.maximum(yt, epsilon)
    mape = float(np.mean(abs_errors / denom) * 100.0)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
        "count": int(len(yt)),
    }


def is_peak_hour(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Determine if timestamps fall into weekday rush hours:
    - Morning peak: 07:00 to 09:30 (hours 7, 8, and hour 9 up to min 30)
    - Evening peak: 16:00 to 19:00 (hours 16, 17, 18)
    - Weekdays only (dayofweek < 5)
    """
    is_weekday = timestamps.dayofweek < 5
    minute_of_day = timestamps.hour * 60 + timestamps.minute
    
    # 07:00 (420) to 09:30 (570)
    morning_peak = (minute_of_day >= 420) & (minute_of_day <= 570)
    # 16:00 (960) to 19:00 (1140)
    evening_peak = (minute_of_day >= 960) & (minute_of_day <= 1140)

    return is_weekday & (morning_peak | evening_peak)


def compute_stratified_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid_mask: np.ndarray,
    origin_timestamps: pd.DatetimeIndex,
    sensor_ids: np.ndarray,
    horizon_name: str,
    model_name: str,
) -> Dict[str, Any]:
    """
    Compute overall, peak/off-peak, and sensor-level metrics for a specific horizon and model.
    """
    overall = compute_regression_metrics(y_true, y_pred, valid_mask)
    
    # Peak vs Off-Peak
    peak_mask = is_peak_hour(origin_timestamps)
    peak_metrics = compute_regression_metrics(y_true, y_pred, valid_mask & peak_mask)
    offpeak_metrics = compute_regression_metrics(y_true, y_pred, valid_mask & (~peak_mask))

    # Sensor-level metrics
    unique_sensors = np.unique(sensor_ids)
    sensor_metrics = {}
    for s_id in unique_sensors:
        s_mask = sensor_ids == s_id
        sensor_metrics[s_id] = compute_regression_metrics(
            y_true, y_pred, valid_mask & s_mask
        )

    return {
        "model": model_name,
        "horizon": horizon_name,
        "overall": overall,
        "peak": peak_metrics,
        "offpeak": offpeak_metrics,
        "by_sensor": sensor_metrics,
    }
