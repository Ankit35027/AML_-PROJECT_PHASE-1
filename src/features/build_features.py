"""
Causal Feature Windowing and Example Construction Engine.
Extracts past-only lag features (t-11 ... t), calendar encodings, optional spatial neighbor lags,
and multi-horizon ground truth targets (t+3, t+6, t+12) with strict split boundary enforcement.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.config import (
    HISTORY_STEPS,
    HORIZONS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    PILOT_SENSORS,
    RANDOM_SEED,
)


def get_split_indices(num_timestamps: int) -> Dict[str, Tuple[int, int]]:
    """
    Compute strictly chronological split slices (start_idx, end_idx exclusive).
    70% Train, 10% Validation, 20% Test.
    """
    n_train = int(num_timestamps * TRAIN_RATIO)
    n_val = int(num_timestamps * VAL_RATIO)
    n_test = num_timestamps - n_train - n_val

    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, num_timestamps),
    }


def compute_time_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Generate normalized continuous cyclical time-of-day and day-of-week features.
    Returns:
        np.ndarray of shape (len(timestamps), 4):
        [sin_tod, cos_tod, sin_dow, cos_dow]
    """
    # Time of day (0 to 1 over 24 hours)
    time_of_day = (timestamps.hour * 60 + timestamps.minute) / (24.0 * 60.0)
    sin_tod = np.sin(2 * np.pi * time_of_day)
    cos_tod = np.cos(2 * np.pi * time_of_day)

    # Day of week (0 to 6 over 7 days)
    day_of_week = timestamps.dayofweek / 7.0
    sin_dow = np.sin(2 * np.pi * day_of_week)
    cos_dow = np.cos(2 * np.pi * day_of_week)

    return np.column_stack([sin_tod, cos_tod, sin_dow, cos_dow])


def build_sensor_examples(
    df: pd.DataFrame,
    sensor_id: str,
    split_name: str,
    adj_mx: Optional[np.ndarray] = None,
    sensor_id_to_idx: Optional[Dict[str, int]] = None,
    include_spatial_neighbors: bool = False,
    history_steps: int = HISTORY_STEPS,
    horizons: Optional[Dict[str, int]] = None,
    train_mean: Optional[float] = None,
    train_std: Optional[float] = None,
) -> Dict[str, Union[np.ndarray, pd.DatetimeIndex, List[str]]]:
    """
    Build causal multi-horizon feature matrix and targets for a single sensor.

    Guarantees:
    1. Past-only: features at index i use observations i - history_steps + 1 through i.
    2. Split isolation: origin index i and all targets i + max(h) strictly reside within split boundaries.
    3. Causal imputation: zero/missing values in historical window imputed using past observations only.
    4. Targets: true measured speeds at +3 (15m), +6 (30m), +12 (60m) steps.
    5. Target validity: valid mask flags speed > 0.0 mph.
    """
    if horizons is None:
        horizons = HORIZONS

    max_horizon = max(horizons.values())  # 12 steps (60 min)
    split_indices = get_split_indices(len(df))
    split_start, split_end = split_indices[split_name]

    # Pre-extract full series for causal imputation
    target_series = df[sensor_id].values
    timestamps = df.index

    # Identify 1st-order spatial neighbors if requested (fixed K=3)
    neighbor_indices = []
    if include_spatial_neighbors and adj_mx is not None and sensor_id_to_idx is not None:
        s_idx = sensor_id_to_idx[sensor_id]
        weights = adj_mx[s_idx].copy()
        weights[s_idx] = -1.0  # exclude self
        # Fixed top 3 nearest network nodes by road graph connectivity/distance
        top_k = np.argsort(weights)[::-1][:3]
        neighbor_indices = list(top_k)

    # Calculate valid origin range [start_idx, end_idx)
    # Origin i requires (i - history_steps + 1 >= split_start)
    # and (i + max_horizon < split_end)
    start_origin = split_start + history_steps - 1
    end_origin = split_end - max_horizon

    if start_origin >= end_origin:
        raise ValueError(
            f"Split '{split_name}' length ({split_end - split_start}) is insufficient "
            f"for history={history_steps} and max_horizon={max_horizon}."
        )

    # Origin indices
    origin_indices = np.arange(start_origin, end_origin)
    n_samples = len(origin_indices)

    # 1. Historical speed lags: shape (n_samples, history_steps)
    X_lags = np.zeros((n_samples, history_steps), dtype=np.float32)
    # Causal lag valid mask: shape (n_samples, history_steps)
    X_lag_valid = np.zeros((n_samples, history_steps), dtype=bool)

    for k in range(history_steps):
        # lag k=0 is t - 11, lag k=11 is t (most recent)
        lag_offset = history_steps - 1 - k
        idx_array = origin_indices - lag_offset
        speeds = target_series[idx_array]
        X_lags[:, k] = speeds
        X_lag_valid[:, k] = speeds > 0.0

    X_raw_lags = X_lags.copy()

    # Causal forward-fill imputation on lags if 0.0 is encountered
    # If initial lags are zero, impute with training mean
    impute_val = train_mean if train_mean is not None else 58.5
    for row in range(n_samples):
        last_val = impute_val
        for col in range(history_steps):
            if X_lags[row, col] > 0.0:
                last_val = X_lags[row, col]
            else:
                X_lags[row, col] = last_val

    # 2. Summary stats over historical window (mean, std, min, max, trend)
    lag_mean = np.mean(X_lags, axis=1, keepdims=True)
    lag_std = np.std(X_lags, axis=1, keepdims=True)
    lag_min = np.min(X_lags, axis=1, keepdims=True)
    lag_max = np.max(X_lags, axis=1, keepdims=True)
    # Momentum / trend: recent 15 min average (last 3 steps) minus older 15 min average (first 3 steps)
    lag_trend = (np.mean(X_lags[:, -3:], axis=1) - np.mean(X_lags[:, :3], axis=1))[:, np.newaxis]

    # 3. Cyclical calendar features for forecast origin
    origin_ts = timestamps[origin_indices]
    time_feats = compute_time_features(origin_ts)

    feature_blocks = [X_lags, lag_mean, lag_std, lag_min, lag_max, lag_trend, time_feats]
    feature_names = [f"speed_lag_t-{history_steps - 1 - i}" for i in range(history_steps)]
    feature_names += ["hist_mean", "hist_std", "hist_min", "hist_max", "hist_trend"]
    feature_names += ["sin_tod", "cos_tod", "sin_dow", "cos_dow"]

    # 4. Spatial neighbor lag features (optional ablation)
    if include_spatial_neighbors and len(neighbor_indices) > 0:
        neighbor_lags = []
        for n_i, n_idx in enumerate(neighbor_indices):
            n_series = df.iloc[:, n_idx].values
            # Most recent 3 lags of neighbor: t-2, t-1, t
            for lag_k in range(3):
                offset = 2 - lag_k
                n_speeds = n_series[origin_indices - offset]
                neighbor_lags.append(n_speeds[:, np.newaxis])
                feature_names.append(f"neighbor_{n_i}_lag_t-{offset}")
        if neighbor_lags:
            feature_blocks.append(np.column_stack(neighbor_lags))

    X = np.column_stack(feature_blocks).astype(np.float32)

    # 5. Multi-horizon targets and validity masks
    # Horizons: +3 (15m), +6 (30m), +12 (60m)
    horizon_names = list(horizons.keys())
    Y = np.zeros((n_samples, len(horizons)), dtype=np.float32)
    Y_valid = np.zeros((n_samples, len(horizons)), dtype=bool)
    target_timestamps = {}

    for h_i, (h_name, h_steps) in enumerate(horizons.items()):
        target_idx = origin_indices + h_steps
        raw_targets = target_series[target_idx]
        Y[:, h_i] = raw_targets
        # Target is valid iff strictly positive speed reading
        Y_valid[:, h_i] = raw_targets > 0.0
        target_timestamps[h_name] = timestamps[target_idx]

    return {
        "sensor_id": sensor_id,
        "split_name": split_name,
        "origin_timestamps": origin_ts,
        "target_timestamps": target_timestamps,
        "X": X,
        "X_raw_lags": X_raw_lags,
        "X_lag_valid": X_lag_valid,
        "feature_names": feature_names,
        "Y": Y,
        "Y_valid": Y_valid,
        "horizon_names": horizon_names,
    }


def build_cohort_dataset(
    df: pd.DataFrame,
    sensor_ids: List[str],
    split_name: str,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False,
    include_spatial_neighbors: bool = False,
    adj_mx: Optional[np.ndarray] = None,
    sensor_id_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[Dict[str, np.ndarray], Optional[StandardScaler]]:
    """
    Build concatenated examples across a cohort of sensors for a split.
    Scales features if scaler is provided or fits scaler strictly on training split.
    """
    all_X = []
    all_Y = []
    all_Y_valid = []
    all_sensor_keys = []
    all_origin_ts = []

    # Compute global training mean from df for causal imputation baseline
    split_indices = get_split_indices(len(df))
    train_slice = slice(*split_indices["train"])
    train_vals = df.iloc[train_slice][sensor_ids].values
    train_mean = float(train_vals[train_vals > 0].mean())

    feature_names = None
    horizon_names = None

    for s_id in sensor_ids:
        ex = build_sensor_examples(
            df=df,
            sensor_id=s_id,
            split_name=split_name,
            adj_mx=adj_mx,
            sensor_id_to_idx=sensor_id_to_idx,
            include_spatial_neighbors=include_spatial_neighbors,
            train_mean=train_mean,
        )
        if feature_names is None:
            feature_names = ex["feature_names"]
            horizon_names = ex["horizon_names"]

        n_s = len(ex["X"])
        all_X.append(ex["X"])
        all_Y.append(ex["Y"])
        all_Y_valid.append(ex["Y_valid"])
        all_sensor_keys.extend([s_id] * n_s)
        all_origin_ts.extend(ex["origin_timestamps"])

    X_cat = np.vstack(all_X)
    Y_cat = np.vstack(all_Y)
    Y_valid_cat = np.vstack(all_Y_valid)

    # Scaling logic: strictly fitted only if fit_scaler=True (train split)
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cat)
    elif scaler is not None:
        X_scaled = scaler.transform(X_cat)
    else:
        X_scaled = X_cat

    dataset = {
        "X": X_scaled,
        "X_raw": X_cat,
        "Y": Y_cat,
        "Y_valid": Y_valid_cat,
        "sensor_ids": np.array(all_sensor_keys),
        "origin_timestamps": pd.DatetimeIndex(all_origin_ts),
        "feature_names": feature_names,
        "horizon_names": horizon_names,
    }

    return dataset, scaler
