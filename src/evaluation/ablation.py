"""
Scientific Feature Ablation Experiment:
Evaluates the impact of 1st-order road network topological neighbor lags (from the METR-LA graph)
compared to isolated single-sensor historical lags.
Produces metrics_ablation.csv and spatial_ablation_comparison.png.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import (
    PILOT_SENSORS,
    HORIZONS,
    EVAL_FIGURES_DIR,
    PROJECT_ROOT,
)
from src.data.loader import load_metr_la_raw, load_sensor_graph
from src.features.build_features import build_cohort_dataset
from src.models.train_xgboost import MultiHorizonXGBoost
from src.evaluation.metrics import compute_regression_metrics


def run_spatial_ablation():
    print("=================================================================")
    print(">>> Phase 1 Feature Ablation: Isolated Lags vs. Spatial Neighbors")
    print("=================================================================")

    df = load_metr_la_raw()
    graph_meta = load_sensor_graph()
    adj_mx = graph_meta["adj_mx"]
    sensor_id_to_idx = {s: i for i, s in enumerate(graph_meta["sensor_ids"])}

    results = []

    for include_spatial in [False, True]:
        setting_name = "With Spatial Neighbors" if include_spatial else "Isolated Sensor Lags"
        print(f"\n--- Running Experiment: {setting_name} ---")

        # Build datasets
        train_ds, scaler = build_cohort_dataset(
            df=df,
            sensor_ids=PILOT_SENSORS,
            split_name="train",
            include_spatial_neighbors=include_spatial,
            adj_mx=adj_mx,
            sensor_id_to_idx=sensor_id_to_idx,
            fit_scaler=True,
        )
        val_ds, _ = build_cohort_dataset(
            df=df,
            sensor_ids=PILOT_SENSORS,
            split_name="val",
            include_spatial_neighbors=include_spatial,
            adj_mx=adj_mx,
            sensor_id_to_idx=sensor_id_to_idx,
            scaler=scaler,
            fit_scaler=False,
        )
        test_ds, _ = build_cohort_dataset(
            df=df,
            sensor_ids=PILOT_SENSORS,
            split_name="test",
            include_spatial_neighbors=include_spatial,
            adj_mx=adj_mx,
            sensor_id_to_idx=sensor_id_to_idx,
            scaler=scaler,
            fit_scaler=False,
        )

        print(f" Feature Count: {train_ds['X'].shape[1]} features")

        # Train XGBoost
        xgb_model = MultiHorizonXGBoost(horizons=HORIZONS, n_estimators=200, max_depth=6, learning_rate=0.05)
        xgb_model.fit(
            X_train=train_ds["X"],
            Y_train=train_ds["Y"],
            Y_train_valid=train_ds["Y_valid"],
            X_val=val_ds["X"],
            Y_val=val_ds["Y"],
            Y_val_valid=val_ds["Y_valid"],
            early_stopping_rounds=20,
        )

        # Evaluate on test
        preds = xgb_model.predict(test_ds["X"])
        for h_i, h_name in enumerate(HORIZONS.keys()):
            yt = test_ds["Y"][:, h_i]
            yp = preds[:, h_i]
            v_mask = test_ds["Y_valid"][:, h_i]

            metrics = compute_regression_metrics(yt, yp, v_mask)
            results.append({
                "feature_configuration": setting_name,
                "horizon": h_name,
                "horizon_minutes": HORIZONS[h_name] * 5,
                "feature_count": train_ds["X"].shape[1],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
            })

    df_ablation = pd.DataFrame(results)
    ablation_csv = PROJECT_ROOT / "metrics_ablation.csv"
    df_ablation.to_csv(ablation_csv, index=False)
    print(f"\n Saved ablation results to {ablation_csv}")
    print(df_ablation.to_string(index=False))

    # Plot Ablation Comparison
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)

    configs = ["Isolated Sensor Lags", "With Spatial Neighbors"]
    palette = ["#1f77b4", "#2ca02c"]

    bar_width = 0.35
    horizons = list(HORIZONS.keys())
    x = np.arange(len(horizons))

    for idx, (cfg, col) in enumerate(zip(configs, palette)):
        sub = df_ablation[df_ablation["feature_configuration"] == cfg]
        ax.bar(x + (idx - 0.5) * bar_width, sub["mae"], width=bar_width, label=cfg, color=col, edgecolor="black")

    ax.set_title("Ablation: Isolated Lags vs. 1st-Order Topological Road Neighbors", fontsize=12, fontweight="bold")
    ax.set_xlabel("Forecast Horizon", fontsize=11)
    ax.set_ylabel("Test MAE (mph)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} (+{HORIZONS[h]*5}m)" for h in horizons])
    ax.legend(frameon=True)
    plt.tight_layout()

    fig_path = EVAL_FIGURES_DIR / "spatial_ablation_comparison.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f" Saved ablation comparison plot to {fig_path}")


if __name__ == "__main__":
    run_spatial_ablation()
