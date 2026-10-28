"""
Comprehensive Data & Provenance Audit for METR-LA Speed Matrix and Sensor Graph.
Performs rigorous missingness accounting (differentiating NaN, 0.0, and impossible speeds),
verifies sensor ID alignment with the road network topology, and writes data_audit.json.
"""

import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import (
    H5_FILENAME,
    SENSOR_IDS_FILE,
    DISTANCES_FILE,
    LOCATIONS_FILE,
    ADJ_MX_FILE,
    DATA_AUDIT_FILE,
    EDA_FIGURES_DIR,
    MAX_REASONABLE_SPEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)
from src.data.loader import load_metr_la_raw, load_sensor_graph, compute_file_sha256


def run_data_audit() -> Dict[str, Any]:
    """Execute complete dataset audit and generate audit manifest and figures."""
    print(">>> Starting METR-LA Data Audit...")
    df = load_metr_la_raw(H5_FILENAME)
    graph_meta = load_sensor_graph(
        SENSOR_IDS_FILE, DISTANCES_FILE, LOCATIONS_FILE, ADJ_MX_FILE
    )

    total_cells = df.size
    total_timestamps, total_sensors = df.shape

    # Missingness categories
    nan_count = int(df.isna().sum().sum())
    zero_count = int((df == 0.0).sum().sum())
    impossible_count = int((df > MAX_REASONABLE_SPEED).sum().sum())
    valid_positive_count = int(((df > 0.0) & (df <= MAX_REASONABLE_SPEED)).sum().sum())

    # Timestamp checks
    ts_start = str(df.index[0])
    ts_end = str(df.index[-1])
    diffs = df.index.to_series().diff().dropna()
    expected_delta_sec = 300  # 5 minutes
    irregular_intervals = int((diffs.dt.total_seconds() != expected_delta_sec).sum())

    # Sensor alignment checks
    graph_ids = graph_meta["sensor_ids"]
    hdf5_ids = list(df.columns)
    id_count_match = len(graph_ids) == len(hdf5_ids)
    exact_sequence_match = graph_ids == hdf5_ids
    sensor_set_diff = list(set(hdf5_ids) - set(graph_ids))

    # Sensor locations check
    loc_df = graph_meta["locations"]
    loc_sensor_ids = [str(s) for s in loc_df["sensor_id"].tolist()]
    loc_alignment_match = loc_sensor_ids == hdf5_ids

    # Distances check
    dist_df = graph_meta["distances"]
    adj_mx = graph_meta["adj_mx"]
    adj_edges = int((adj_mx > 0).sum()) if adj_mx is not None else 0

    # Sensor-level missingness stats
    sensor_zeros = (df == 0.0).sum(axis=0) / total_timestamps
    sensor_nans = df.isna().sum(axis=0) / total_timestamps

    # Chronological partition timestamp boundaries
    n_train = int(total_timestamps * TRAIN_RATIO)
    n_val = int(total_timestamps * VAL_RATIO)
    n_test = total_timestamps - n_train - n_val

    train_start, train_end = str(df.index[0]), str(df.index[n_train - 1])
    val_start, val_end = str(df.index[n_train]), str(df.index[n_train + n_val - 1])
    test_start, test_end = str(df.index[n_train + n_val]), str(df.index[-1])

    audit_results: Dict[str, Any] = {
        "dataset_name": "METR-LA",
        "units": "miles per hour (mph)",
        "shape": {
            "num_timestamps": total_timestamps,
            "num_sensors": total_sensors,
            "total_observations": total_cells,
        },
        "temporal_range": {
            "start": ts_start,
            "end": ts_end,
            "interval_minutes": 5,
            "irregular_interval_count": irregular_intervals,
        },
        "chronological_partitions": {
            "train": {
                "num_timestamps": n_train,
                "start": train_start,
                "end": train_end,
                "percentage": round(n_train / total_timestamps * 100, 2),
            },
            "validation": {
                "num_timestamps": n_val,
                "start": val_start,
                "end": val_end,
                "percentage": round(n_val / total_timestamps * 100, 2),
            },
            "test": {
                "num_timestamps": n_test,
                "start": test_start,
                "end": test_end,
                "percentage": round(n_test / total_timestamps * 100, 2),
            },
        },
        "missingness_and_validity": {
            "nan_count": nan_count,
            "nan_ratio": round(nan_count / total_cells, 6),
            "zero_count": zero_count,
            "zero_ratio": round(zero_count / total_cells, 6),
            "impossible_speed_gt_100_count": impossible_count,
            "valid_positive_measurements": valid_positive_count,
            "valid_positive_ratio": round(valid_positive_count / total_cells, 6),
            "zero_interpretation": (
                "Loop detector readings of 0.0 mph represent either real standstill congestion, "
                "sensor dropouts, or road closure. Valid evaluation masks strictly require >0.0 mph "
                "to prevent artificial distortion or division by zero, while causal imputations (forward-fill) "
                "are permitted strictly for historical feature lags."
            ),
        },
        "sensor_graph_alignment": {
            "graph_sensor_ids_file_count": len(graph_ids),
            "hdf5_columns_count": len(hdf5_ids),
            "locations_sensor_count": len(loc_sensor_ids),
            "exact_sequence_match": exact_sequence_match,
            "locations_sequence_match": loc_alignment_match,
            "discrepancies": sensor_set_diff,
            "directed_adj_edges_count": adj_edges,
        },
        "speed_statistics_valid": {
            "min_valid_speed": float(df[df > 0].min().min()),
            "max_speed": float(df.max().max()),
            "mean_valid_speed": float(df[df > 0].mean().mean()),
            "median_valid_speed": float(df[df > 0].median().median()),
            "std_valid_speed": float(df[df > 0].std().mean()),
        },
    }

    # Save to data_audit.json
    with open(DATA_AUDIT_FILE, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f" Saved data audit report to {DATA_AUDIT_FILE}")

    # Generate Audit Visualizations
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Sensor Zero-Rate Distribution
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    sns.histplot(sensor_zeros * 100, bins=30, kde=True, color="#1f77b4", ax=ax)
    ax.set_title("Distribution of Zero-Speed Frequency Across 207 METR-LA Sensors", fontsize=12, fontweight="bold")
    ax.set_xlabel("Percentage of 0.0 mph Readings (%)", fontsize=11)
    ax.set_ylabel("Sensor Count", fontsize=11)
    plt.tight_layout()
    fig_path1 = EDA_FIGURES_DIR / "sensor_zero_distribution.png"
    fig.savefig(fig_path1)
    plt.close(fig)
    print(f" Generated audit figure: {fig_path1}")

    # Figure 2: Valid Speed Distribution
    valid_speeds = df.values[df.values > 0].flatten()
    sample_speeds = np.random.choice(valid_speeds, size=min(100000, len(valid_speeds)), replace=False)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    sns.histplot(sample_speeds, bins=50, kde=True, color="#2ca02c", ax=ax)
    ax.axvline(audit_results["speed_statistics_valid"]["mean_valid_speed"], color="red", linestyle="--", label=f"Mean: {audit_results['speed_statistics_valid']['mean_valid_speed']:.1f} mph")
    ax.axvline(audit_results["speed_statistics_valid"]["median_valid_speed"], color="darkorange", linestyle=":", label=f"Median: {audit_results['speed_statistics_valid']['median_valid_speed']:.1f} mph")
    ax.set_title("Empirical Distribution of Valid Highway Speeds (METR-LA)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Speed (mph)", fontsize=11)
    ax.set_ylabel("Density / Count", fontsize=11)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig_path2 = EDA_FIGURES_DIR / "speed_distribution.png"
    fig.savefig(fig_path2)
    plt.close(fig)
    print(f" Generated audit figure: {fig_path2}")

    return audit_results


if __name__ == "__main__":
    run_data_audit()
