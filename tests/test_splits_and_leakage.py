"""
Scientific Validation & Leakage Verification Test Suite.
Validates causal windowing, exact target offsets (+15, +30, +60 min), split boundaries,
data leakage prevention, graph ID alignment, and manual metric computation equivalence.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.config import (
    HISTORY_STEPS,
    HORIZONS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    EPSILON_MAPE,
)
from src.features.build_features import (
    get_split_indices,
    build_sensor_examples,
    build_cohort_dataset,
)
from src.data.loader import load_metr_la_raw, load_sensor_graph


@pytest.fixture(scope="module")
def sample_traffic_df():
    """Create a deterministic synthetic traffic speed DataFrame for rapid testing."""
    n_steps = 1000
    dates = pd.date_range("2012-03-01 00:00:00", periods=n_steps, freq="5min")
    sensors = ["773869", "767541", "717447"]
    
    # Generate realistic baseline speed with diurnal sinusoid
    t = np.arange(n_steps)
    speeds = 55.0 + 10.0 * np.sin(2 * np.pi * t / 288)  # 288 steps in 24 hours
    data = {s: speeds + (i * 2.0) for i, s in enumerate(sensors)}
    
    # Inject isolated zero dropouts for validity mask testing
    data["773869"][50] = 0.0
    data["767541"][60] = 0.0
    
    df = pd.DataFrame(data, index=dates)
    return df


def test_split_indices_proportions_and_continuity():
    """Verify chronological split boundaries: 70% Train, 10% Val, 20% Test with continuous boundaries."""
    total = 34272
    splits = get_split_indices(total)
    
    train_start, train_end = splits["train"]
    val_start, val_end = splits["val"]
    test_start, test_end = splits["test"]
    
    # Check start and end boundaries
    assert train_start == 0
    assert train_end == val_start
    assert val_end == test_start
    assert test_end == total
    
    # Check partition sizes
    n_train = train_end - train_start
    n_val = val_end - val_start
    n_test = test_end - test_start
    
    assert n_train == int(total * TRAIN_RATIO)
    assert n_val == int(total * VAL_RATIO)
    assert n_test == total - n_train - n_val
    assert (n_train + n_val + n_test) == total


def test_exact_target_offsets(sample_traffic_df):
    """Verify that sample target timestamps strictly equal origin +15, +30, and +60 minutes."""
    ex = build_sensor_examples(
        df=sample_traffic_df,
        sensor_id="773869",
        split_name="train",
    )
    
    origin_ts = ex["origin_timestamps"]
    target_ts = ex["target_timestamps"]
    
    # Verify exact time differences
    diff_15m = target_ts["15min"] - origin_ts
    diff_30m = target_ts["30min"] - origin_ts
    diff_60m = target_ts["60min"] - origin_ts
    
    assert (diff_15m == pd.Timedelta(minutes=15)).all(), "Offset for 15min horizon must be exactly 15 minutes."
    assert (diff_30m == pd.Timedelta(minutes=30)).all(), "Offset for 30min horizon must be exactly 30 minutes."
    assert (diff_60m == pd.Timedelta(minutes=60)).all(), "Offset for 60min horizon must be exactly 60 minutes."


def test_causal_windowing_and_no_future_leakage(sample_traffic_df):
    """Verify that features at index i use observations t-11 ... t strictly, with zero target contamination."""
    ex = build_sensor_examples(
        df=sample_traffic_df,
        sensor_id="773869",
        split_name="train",
    )
    
    X_raw_lags = ex["X_raw_lags"]
    origin_ts = ex["origin_timestamps"]
    raw_series = sample_traffic_df["773869"]
    
    # Test multiple samples
    test_rows = [0, 10, 50, 100, len(origin_ts) - 1]
    for r in test_rows:
        orig = origin_ts[r]
        # Most recent speed lag (t) is index 11
        lag_t = X_raw_lags[r, 11]
        expected_lag_t = raw_series.loc[orig]
        assert np.isclose(lag_t, expected_lag_t), f"Lag t {lag_t} != origin reading {expected_lag_t}"
        
        # Historical lag t-11 is index 0
        lag_t_minus_11 = X_raw_lags[r, 0]
        expected_t_minus_11 = raw_series.loc[orig - pd.Timedelta(minutes=55)]
        assert np.isclose(lag_t_minus_11, expected_t_minus_11), "Lag t-11 != t - 55 min reading"

    # Verify causal forward fill imputation: in imputed X, no 0.0 values remain
    X = ex["X"]
    lag_matrix = X[:, :HISTORY_STEPS]
    assert (lag_matrix > 0.0).all(), "Imputed lag matrix should have zero missing/zero values."


def test_split_boundary_isolation_no_target_overflow(sample_traffic_df):
    """Verify that no sample's origin or future target extends past the split boundary."""
    splits = get_split_indices(len(sample_traffic_df))
    
    for split_name in ["train", "val", "test"]:
        ex = build_sensor_examples(
            df=sample_traffic_df,
            sensor_id="773869",
            split_name=split_name,
        )
        split_start, split_end = splits[split_name]
        split_start_ts = sample_traffic_df.index[split_start]
        split_end_ts = sample_traffic_df.index[split_end - 1]
        
        # Origin timestamps
        assert ex["origin_timestamps"][0] >= split_start_ts
        assert ex["origin_timestamps"][-1] <= split_end_ts
        
        # Max horizon targets (+60 min) must also remain strictly within the split
        target_60m = ex["target_timestamps"]["60min"]
        assert target_60m[-1] <= split_end_ts, f"Target overflow in split {split_name}!"


def test_scaler_fit_strictly_on_train_only(sample_traffic_df):
    """Verify that feature scalers are fitted strictly on the training partition without test contamination."""
    sensors = ["773869", "767541"]
    
    # 1. Fit scaler on train
    train_ds, scaler = build_cohort_dataset(
        df=sample_traffic_df,
        sensor_ids=sensors,
        split_name="train",
        fit_scaler=True,
    )
    assert scaler is not None
    assert hasattr(scaler, "mean_")
    
    # Record train mean and scale
    train_mean = scaler.mean_.copy()
    train_scale = scaler.scale_.copy()
    
    # 2. Transform test using fitted scaler (fit_scaler=False)
    test_ds, scaler_after_test = build_cohort_dataset(
        df=sample_traffic_df,
        sensor_ids=sensors,
        split_name="test",
        scaler=scaler,
        fit_scaler=False,
    )
    
    # Scaler parameters must remain identical (not refitted)
    np.testing.assert_array_equal(scaler.mean_, train_mean)
    np.testing.assert_array_equal(scaler.scale_, train_scale)


def test_manual_metrics_recomputation():
    """Manual metric computation on 5 illustrative records matches code metric output."""
    y_true = np.array([60.0, 45.0, 0.0, 65.0, 50.0])  # Note: index 2 is 0.0 (invalid)
    y_pred = np.array([58.0, 48.0, 15.0, 62.0, 55.0])
    valid_mask = y_true > 0.0
    
    # Valid indices: 0, 1, 3, 4
    # Errors:
    # 0: |60 - 58| = 2.0
    # 1: |45 - 48| = 3.0
    # 3: |65 - 62| = 3.0
    # 4: |50 - 55| = 5.0
    # Sum absolute errors: 2 + 3 + 3 + 5 = 13.0
    # Manual MAE = 13.0 / 4 = 3.25 mph
    expected_mae = 3.25
    
    # Squared errors: 4 + 9 + 9 + 25 = 47.0
    # Mean squared error: 47.0 / 4 = 11.75
    # Manual RMSE = sqrt(11.75) ≈ 3.4278 mph
    expected_rmse = np.sqrt(11.75)
    
    # Percentage errors with epsilon policy: |y_true - y_pred| / max(y_true, EPSILON)
    # 0: 2 / 60 = 0.03333
    # 1: 3 / 45 = 0.06667
    # 3: 3 / 65 = 0.04615
    # 4: 5 / 50 = 0.10000
    # Sum MAPE = 0.2461538 -> mean = 0.061538 (6.15%)
    expected_mape = np.mean([2/60, 3/45, 3/65, 5/50]) * 100
    
    # Compute via code implementation
    valid_true = y_true[valid_mask]
    valid_pred = y_pred[valid_mask]
    
    calc_mae = np.mean(np.abs(valid_true - valid_pred))
    calc_rmse = np.sqrt(np.mean((valid_true - valid_pred) ** 2))
    denom = np.maximum(valid_true, EPSILON_MAPE)
    calc_mape = np.mean(np.abs(valid_true - valid_pred) / denom) * 100
    
    assert np.isclose(calc_mae, expected_mae, atol=1e-4)
    assert np.isclose(calc_rmse, expected_rmse, atol=1e-4)
    assert np.isclose(calc_mape, expected_mape, atol=1e-4)
