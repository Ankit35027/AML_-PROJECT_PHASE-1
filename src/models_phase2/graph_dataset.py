"""
Spatio-Temporal Graph Dataset & DataLoader Pipeline (Phase 2).
Constructs chronological (70/10/20) PyTorch Datasets from METR-LA speed matrix
and sensor adjacency topology with causal temporal windowing and zero future leakage.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.config import (
    HISTORY_STEPS,
    HORIZONS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    PILOT_SENSORS,
    RANDOM_SEED,
)
from src.features.build_features import compute_time_features, get_split_indices


class SpatioTemporalTrafficDataset(Dataset):
    """
    PyTorch Dataset yielding graph-structured spatio-temporal examples:
      X: (T_in, N, F_in) where T_in=12 lags, N=sensors, F_in=[speed, sin_tod, cos_tod, sin_dow, cos_dow]
      Y: (N, H) ground truth speeds at +15m, +30m, +60m
      Y_valid: (N, H) boolean mask (>0.0 mph)
    """

    def __init__(
        self,
        speed_matrix: np.ndarray,
        time_features: np.ndarray,
        timestamps: pd.DatetimeIndex,
        history_steps: int = 12,
        horizon_offsets: Tuple[int, int, int] = (3, 6, 12),
        mean: float = 0.0,
        std: float = 1.0,
        normalize_inputs: bool = True,
    ):
        self.speed_matrix = speed_matrix.astype(np.float32)
        self.time_features = time_features.astype(np.float32)
        self.timestamps = timestamps
        self.history_steps = history_steps
        self.horizon_offsets = horizon_offsets
        self.max_offset = max(horizon_offsets)
        self.mean = mean
        self.std = max(std, 1e-6)
        self.normalize_inputs = normalize_inputs

        self.num_timestamps, self.num_nodes = self.speed_matrix.shape

        # Valid origin index range: [history_steps - 1, num_timestamps - max_offset - 1]
        self.valid_origin_indices = np.arange(
            self.history_steps - 1, self.num_timestamps - self.max_offset
        )

    def __len__(self) -> int:
        return len(self.valid_origin_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        origin_t = self.valid_origin_indices[idx]

        # 1. Historical speed slice: [origin_t - 11 : origin_t + 1] -> (12, N)
        hist_speeds = self.speed_matrix[origin_t - self.history_steps + 1 : origin_t + 1, :]
        hist_times = self.time_features[origin_t - self.history_steps + 1 : origin_t + 1, :]  # (12, 4)

        # Causal forward-fill zero speeds in input features to handle sensor dropouts
        hist_speeds_filled = hist_speeds.copy()
        for node in range(self.num_nodes):
            series = hist_speeds_filled[:, node]
            zeros = (series <= 0.0)
            if np.any(zeros) and not np.all(zeros):
                last_val = self.mean
                for step in range(len(series)):
                    if series[step] > 0.0:
                        last_val = series[step]
                    else:
                        series[step] = last_val

        # Normalize historical speeds if requested
        if self.normalize_inputs:
            norm_speeds = (hist_speeds_filled - self.mean) / self.std
        else:
            norm_speeds = hist_speeds_filled

        # Broadcast time features (12, 4) across nodes -> (12, N, 4)
        time_expanded = np.repeat(hist_times[:, np.newaxis, :], self.num_nodes, axis=1)

        # Combine speed + time features: (12, N, 5)
        x_features = np.concatenate([norm_speeds[:, :, np.newaxis], time_expanded], axis=-1)

        # 2. Multi-horizon targets: offsets (3, 6, 12) -> +15m, +30m, +60m
        targets = []
        valid_masks = []
        for offset in self.horizon_offsets:
            target_speed = self.speed_matrix[origin_t + offset, :]  # (N,)
            targets.append(target_speed)
            valid_masks.append(target_speed > 0.0)

        # Shape: (N, 3)
        y = np.stack(targets, axis=-1)
        y_valid = np.stack(valid_masks, axis=-1)

        return (
            torch.from_numpy(x_features).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(y_valid).bool(),
        )


def create_dataloaders(
    df: pd.DataFrame,
    sensor_ids: List[str],
    adj_mx: np.ndarray,
    batch_size: int = 64,
    history_steps: int = 12,
    horizon_offsets: Tuple[int, int, int] = (3, 6, 12),
) -> Tuple[Dict[str, DataLoader], torch.Tensor, Tuple[float, float]]:
    """
    Constructs Chronological DataLoaders for Train (70%), Val (10%), and Test (20%).
    Scaler mean and std are fitted strictly on the Training split.
    """
    # Filter speed matrix to chosen sensors
    sub_df = df[sensor_ids]
    speed_values = sub_df.values
    timestamps = sub_df.index
    time_feats = compute_time_features(timestamps)

    splits = get_split_indices(len(timestamps))
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    # Compute training-only statistics for normalization
    train_speeds = speed_values[train_idx[0] : train_idx[1], :]
    valid_train = train_speeds[train_speeds > 0.0]
    train_mean = float(np.mean(valid_train)) if len(valid_train) > 0 else 55.0
    train_std = float(np.std(valid_train)) if len(valid_train) > 0 else 10.0

    # Build datasets
    train_ds = SpatioTemporalTrafficDataset(
        speed_matrix=speed_values[train_idx[0] : train_idx[1]],
        time_features=time_feats[train_idx[0] : train_idx[1]],
        timestamps=timestamps[train_idx[0] : train_idx[1]],
        history_steps=history_steps,
        horizon_offsets=horizon_offsets,
        mean=train_mean,
        std=train_std,
    )
    val_ds = SpatioTemporalTrafficDataset(
        speed_matrix=speed_values[val_idx[0] : val_idx[1]],
        time_features=time_feats[val_idx[0] : val_idx[1]],
        timestamps=timestamps[val_idx[0] : val_idx[1]],
        history_steps=history_steps,
        horizon_offsets=horizon_offsets,
        mean=train_mean,
        std=train_std,
    )
    test_ds = SpatioTemporalTrafficDataset(
        speed_matrix=speed_values[test_idx[0] : test_idx[1]],
        time_features=time_feats[test_idx[0] : test_idx[1]],
        timestamps=timestamps[test_idx[0] : test_idx[1]],
        history_steps=history_steps,
        horizon_offsets=horizon_offsets,
        mean=train_mean,
        std=train_std,
    )

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }

    adj_tensor = torch.from_numpy(adj_mx).float()

    return loaders, adj_tensor, (train_mean, train_std)

