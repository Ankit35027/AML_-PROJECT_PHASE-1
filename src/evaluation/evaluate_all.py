"""
Main Evaluation Pipeline for Phase 1 Classical ML Benchmarks.
Trains and evaluates Historical Persistence, Ridge Regression, and XGBoost models
on the 15-sensor pilot cohort across 15m, 30m, and 60m horizons.
Generates genuine metric tables, structured test prediction tables, and evaluation figures.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import (
    PILOT_SENSORS,
    HORIZONS,
    MODELS_DIR,
    EVAL_FIGURES_DIR,
    REPORTS_DIR,
    PROJECT_ROOT,
    MODEL_MANIFEST_FILE,
    DATA_SUMMARY_FILE,
)
from src.data.loader import load_metr_la_raw, load_sensor_graph
from src.features.build_features import build_cohort_dataset
from src.models.persistence import PersistenceModel
from src.models.ridge_baseline import MultiHorizonRidge
from src.models.train_xgboost import MultiHorizonXGBoost
from src.evaluation.metrics import (
    compute_regression_metrics,
    compute_stratified_metrics,
    is_peak_hour,
)


def run_full_evaluation():
    print("=================================================================")
    print(">>> Phase 1 Classical ML Pipeline: Training & Multi-Horizon Eval")
    print(f">>> Pilot Scope: {len(PILOT_SENSORS)} Representative Sensors")
    print("=================================================================")

    df = load_metr_la_raw()
    graph_meta = load_sensor_graph()
    adj_mx = graph_meta["adj_mx"]
    sensor_id_to_idx = {s: i for i, s in enumerate(graph_meta["sensor_ids"])}

    # 1. Build Chronological Datasets
    print("\n--- [Step 1/5] Building Chronological Datasets (Train / Val / Test) ---")
    start_time = time.time()
    
    # Train set (fitted scaler)
    train_ds, scaler = build_cohort_dataset(
        df=df,
        sensor_ids=PILOT_SENSORS,
        split_name="train",
        fit_scaler=True,
    )
    print(f" Train Dataset: {train_ds['X'].shape[0]:,} samples x {train_ds['X'].shape[1]} features")

    # Validation set (using train-fitted scaler)
    val_ds, _ = build_cohort_dataset(
        df=df,
        sensor_ids=PILOT_SENSORS,
        split_name="val",
        scaler=scaler,
        fit_scaler=False,
    )
    print(f" Val Dataset:   {val_ds['X'].shape[0]:,} samples x {val_ds['X'].shape[1]} features")

    # Test set (using train-fitted scaler)
    test_ds, _ = build_cohort_dataset(
        df=df,
        sensor_ids=PILOT_SENSORS,
        split_name="test",
        scaler=scaler,
        fit_scaler=False,
    )
    print(f" Test Dataset:  {test_ds['X'].shape[0]:,} samples x {test_ds['X'].shape[1]} features")

    # 2. Train Models
    print("\n--- [Step 2/5] Training Benchmark Models ---")
    
    # A. Persistence Baseline
    print(" Fitting Historical Persistence Benchmark...")
    persistence = PersistenceModel(horizons=HORIZONS)
    persistence.fit(train_ds["X_raw"])

    # B. Ridge Regression Baseline
    print("\n Training Multi-Horizon Ridge Regression with Validation Tuning...")
    ridge = MultiHorizonRidge(horizons=HORIZONS)
    ridge.fit_and_tune(
        X_train=train_ds["X"],
        Y_train=train_ds["Y"],
        Y_train_valid=train_ds["Y_valid"],
        X_val=val_ds["X"],
        Y_val=val_ds["Y"],
        Y_val_valid=val_ds["Y_valid"],
    )
    ridge_path = MODELS_DIR / "ridge_baseline.joblib"
    ridge.save(ridge_path)
    print(f" Saved Ridge model to {ridge_path}")

    # C. XGBoost Regressor
    print("\n Training Multi-Horizon XGBoost Regressors with Early Stopping...")
    xgb_model = MultiHorizonXGBoost(horizons=HORIZONS, n_estimators=250, max_depth=6, learning_rate=0.05)
    xgb_model.fit(
        X_train=train_ds["X"],
        Y_train=train_ds["Y"],
        Y_train_valid=train_ds["Y_valid"],
        X_val=val_ds["X"],
        Y_val=val_ds["Y"],
        Y_val_valid=val_ds["Y_valid"],
        early_stopping_rounds=25,
    )
    xgb_path = MODELS_DIR / "xgboost_baseline.joblib"
    xgb_model.save(xgb_path)
    print(f" Saved XGBoost model to {xgb_path}")

    # 3. Predict on Test Set
    print("\n--- [Step 3/5] Generating Held-Out Test Predictions ---")
    preds = {
        "Persistence": persistence.predict(test_ds["X_raw"]),
        "Ridge": ridge.predict(test_ds["X"]),
        "XGBoost": xgb_model.predict(test_ds["X"]),
    }

    # Structure predictions table
    pred_records = []
    horizon_names = list(HORIZONS.keys())
    test_origins = test_ds["origin_timestamps"]
    test_sensors = test_ds["sensor_ids"]

    for m_name, pred_matrix in preds.items():
        for h_i, h_name in enumerate(horizon_names):
            h_steps = HORIZONS[h_name]
            target_ts = test_origins + pd.Timedelta(minutes=5 * h_steps)
            yt = test_ds["Y"][:, h_i]
            yp = pred_matrix[:, h_i]
            v_flag = test_ds["Y_valid"][:, h_i]

            for s_id, o_ts, t_ts, y_t, y_p, v in zip(test_sensors, test_origins, target_ts, yt, yp, v_flag):
                pred_records.append({
                    "sensor_id": s_id,
                    "origin_ts": str(o_ts),
                    "target_ts": str(t_ts),
                    "horizon": h_name,
                    "horizon_steps": h_steps,
                    "y_true": round(float(y_t), 3),
                    "y_pred": round(float(y_p), 3),
                    "valid_flag": bool(v),
                    "model": m_name,
                })

    preds_df = pd.DataFrame(pred_records)
    preds_parquet = PROJECT_ROOT / "test_predictions.parquet"
    preds_csv_sample = PROJECT_ROOT / "test_predictions_sample.csv"
    preds_df.to_parquet(preds_parquet, index=False)
    preds_df.head(1000).to_csv(preds_csv_sample, index=False)
    print(f" Saved full test predictions parquet ({len(preds_df):,} rows) to {preds_parquet}")
    print(f" Saved 1,000-row preview sample to {preds_csv_sample}")

    # 4. Compute Metrics Tables
    print("\n--- [Step 4/5] Computing Multi-Horizon Metric Tables ---")
    horizon_metrics_list = []
    peak_metrics_list = []
    sensor_metrics_list = []

    for m_name, pred_matrix in preds.items():
        for h_i, h_name in enumerate(horizon_names):
            yt = test_ds["Y"][:, h_i]
            yp = pred_matrix[:, h_i]
            v_mask = test_ds["Y_valid"][:, h_i]

            strat = compute_stratified_metrics(
                y_true=yt,
                y_pred=yp,
                valid_mask=v_mask,
                origin_timestamps=test_origins,
                sensor_ids=test_sensors,
                horizon_name=h_name,
                model_name=m_name,
            )

            # Overall by horizon
            ov = strat["overall"]
            horizon_metrics_list.append({
                "model": m_name,
                "horizon": h_name,
                "horizon_minutes": HORIZONS[h_name] * 5,
                "mae": ov["mae"],
                "rmse": ov["rmse"],
                "mape": ov["mape"],
                "valid_samples": ov["count"],
            })

            # Peak vs Off-peak
            pk = strat["peak"]
            opk = strat["offpeak"]
            peak_metrics_list.append({
                "model": m_name,
                "horizon": h_name,
                "peak_mae": pk["mae"],
                "peak_rmse": pk["rmse"],
                "peak_mape": pk["mape"],
                "peak_samples": pk["count"],
                "offpeak_mae": opk["mae"],
                "offpeak_rmse": opk["rmse"],
                "offpeak_mape": opk["mape"],
                "offpeak_samples": opk["count"],
            })

            # By sensor
            for s_id, s_m in strat["by_sensor"].items():
                sensor_metrics_list.append({
                    "model": m_name,
                    "horizon": h_name,
                    "sensor_id": s_id,
                    "mae": s_m["mae"],
                    "rmse": s_m["rmse"],
                    "mape": s_m["mape"],
                    "valid_samples": s_m["count"],
                })

    df_horizon = pd.DataFrame(horizon_metrics_list)
    df_peak = pd.DataFrame(peak_metrics_list)
    df_sensor = pd.DataFrame(sensor_metrics_list)

    metrics_h_file = PROJECT_ROOT / "metrics_by_horizon.csv"
    metrics_p_file = PROJECT_ROOT / "metrics_peak_offpeak.csv"
    metrics_s_file = PROJECT_ROOT / "metrics_by_sensor.csv"

    df_horizon.to_csv(metrics_h_file, index=False)
    df_peak.to_csv(metrics_p_file, index=False)
    df_sensor.to_csv(metrics_s_file, index=False)

    print(f"\n==================== TEST SET PERFORMANCE SUMMARY ====================")
    print(df_horizon.to_string(index=False))
    print(f"======================================================================\n")

    # 5. Generate Evaluation Figures
    print("--- [Step 5/5] Generating Empirical Evaluation Figures ---")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Multi-Horizon MAE & RMSE Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    models = ["Persistence", "Ridge", "XGBoost"]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c"]

    # MAE plot
    for m, c in zip(models, colors):
        sub = df_horizon[df_horizon["model"] == m]
        ax1.plot(sub["horizon_minutes"], sub["mae"], marker="o", linewidth=2.5, label=m, color=c)
    ax1.set_title("Forecast Accuracy across Horizons (MAE)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Forecast Horizon (Minutes)", fontsize=11)
    ax1.set_ylabel("Mean Absolute Error (mph)", fontsize=11)
    ax1.set_xticks([15, 30, 60])
    ax1.legend(frameon=True)

    # RMSE plot
    for m, c in zip(models, colors):
        sub = df_horizon[df_horizon["model"] == m]
        ax2.plot(sub["horizon_minutes"], sub["rmse"], marker="s", linewidth=2.5, label=m, color=c)
    ax2.set_title("Forecast Accuracy across Horizons (RMSE)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Forecast Horizon (Minutes)", fontsize=11)
    ax2.set_ylabel("Root Mean Squared Error (mph)", fontsize=11)
    ax2.set_xticks([15, 30, 60])
    ax2.legend(frameon=True)

    plt.tight_layout()
    fig1_path = EVAL_FIGURES_DIR / "multi_horizon_comparison.png"
    fig.savefig(fig1_path)
    plt.close(fig)
    print(f" Saved: {fig1_path}")

    # Figure 2: Prediction Trace on Test Peak Day
    # Pick a sensor and a test weekday
    test_sensor = "773869"
    sensor_df = preds_df[(preds_df["sensor_id"] == test_sensor) & (preds_df["horizon"] == "15min")].copy()
    sensor_df["target_dt"] = pd.to_datetime(sensor_df["target_ts"])
    # Pick a 24-hour slice: 2012-06-05
    day_slice = sensor_df[(sensor_df["target_dt"] >= "2012-06-05 00:00:00") & (sensor_df["target_dt"] <= "2012-06-05 23:55:00")]

    if len(day_slice) > 0:
        fig, ax = plt.subplots(figsize=(11, 4.8), dpi=300)
        # Ground truth
        truth = day_slice[day_slice["model"] == "XGBoost"].sort_values("target_dt")
        ax.plot(truth["target_dt"], truth["y_true"], color="black", linewidth=2.0, label="Ground Truth (Sensor 773869)")
        for m, c, ls in zip(["Persistence", "Ridge", "XGBoost"], ["#7f7f7f", "#1f77b4", "#2ca02c"], [":", "--", "-"]):
            m_slice = day_slice[day_slice["model"] == m].sort_values("target_dt")
            ax.plot(m_slice["target_dt"], m_slice["y_pred"], label=f"{m} (15m)", color=c, linestyle=ls, linewidth=1.8)
        
        ax.set_title("24-Hour Test Forecast Trace: Ground Truth vs Models (15-Min Ahead)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (2012-06-05, PDT)", fontsize=11)
        ax.set_ylabel("Speed (mph)", fontsize=11)
        ax.legend(loc="lower left", frameon=True)
        plt.tight_layout()
        fig2_path = EVAL_FIGURES_DIR / "test_forecast_trace.png"
        fig.savefig(fig2_path)
        plt.close(fig)
        print(f" Saved: {fig2_path}")

    # Figure 3: Feature Importance from XGBoost (Horizon 15m)
    if "15min" in xgb_model.feature_importances:
        importances = xgb_model.feature_importances["15min"]
        feat_names = train_ds["feature_names"]
        top_idx = np.argsort(importances)[::-1][:12]
        
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
        ax.barh(np.array(feat_names)[top_idx][::-1], importances[top_idx][::-1], color="#2ca02c", edgecolor="black")
        ax.set_title("Top 12 Feature Importances (XGBoost 15-Min Horizon)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Normalized Relative Importance (Gini / Gain)", fontsize=11)
        plt.tight_layout()
        fig3_path = EVAL_FIGURES_DIR / "xgboost_feature_importance.png"
        fig.savefig(fig3_path)
        plt.close(fig)
        print(f" Saved: {fig3_path}")

    # Figure 4: Residuals Error Distribution
    xgb_preds_15m = preds_df[(preds_df["model"] == "XGBoost") & (preds_df["horizon"] == "15min") & (preds_df["valid_flag"])]
    residuals = xgb_preds_15m["y_true"] - xgb_preds_15m["y_pred"]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    sns.histplot(residuals, bins=50, kde=True, color="#d62728", ax=ax)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_title("Held-Out Test Residual Distribution (y_true - y_pred, XGBoost 15m)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Speed Error (mph)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_xlim(-25, 25)
    plt.tight_layout()
    fig4_path = EVAL_FIGURES_DIR / "residuals_distribution.png"
    fig.savefig(fig4_path)
    plt.close(fig)
    print(f" Saved: {fig4_path}")

    # 6. Save Manifests
    model_manifest = {
        "execution_date": "2026-10-28",
        "random_seed": 42,
        "pilot_sensor_cohort": PILOT_SENSORS,
        "cohort_size": len(PILOT_SENSORS),
        "total_sensors_in_metr_la": 207,
        "scope_disclosure": (
            "Due to single-node evaluation benchmarks, the primary experimental results "
            "are established on a 15-sensor representative pilot cohort covering key Los Angeles "
            "freeway segments (I-5, I-10, I-405, SR-110). Metric tables are explicitly scoped to this pilot."
        ),
        "models": {
            "Persistence": {
                "parameters": 0,
                "strategy": "Historical last-observed value y(t)"
            },
            "Ridge": {
                "best_alphas": ridge.best_alphas,
                "regularization": "L2 norm"
            },
            "XGBoost": {
                "hyperparameters": xgb_model.params,
                "training_durations_sec": xgb_model.training_durations
            }
        },
        "test_metrics_summary": df_horizon.to_dict(orient="records")
    }

    with open(MODEL_MANIFEST_FILE, "w") as f:
        json.dump(model_manifest, f, indent=2)
    print(f" Saved model manifest to {MODEL_MANIFEST_FILE}")

    print("\n Full evaluation pipeline completed successfully!")


if __name__ == "__main__":
    run_full_evaluation()
