"""
Exploratory Data Analysis (EDA) for METR-LA Traffic Speed Dataset.
Generates genuine empirical figures:
1. 24-Hour Diurnal Cycle (Rush Hour Speed Drops: Morning & Evening Peaks)
2. Weekday vs. Weekend Speed Profiles
3. Autocorrelation Function (ACF) across Temporal Lags
4. Spatial Sensor Network Adjacency Degree Distribution
5. Corridor Speed Profiles across Pilot Sensors
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import (
    EDA_FIGURES_DIR,
    PILOT_SENSORS,
)
from src.data.loader import load_metr_la_raw, load_sensor_graph


def generate_eda_figures():
    print(">>> Starting Exploratory Data Analysis (EDA) figure generation...")
    EDA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_metr_la_raw()
    graph_meta = load_sensor_graph()

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    # Filter valid positive speeds for diurnal & speed computations
    df_positive = df.replace(0.0, np.nan)

    # -------------------------------------------------------------
    # 1. 24-Hour Diurnal Cycle (Rush-hour Congestion Dynamics)
    # -------------------------------------------------------------
    print(" Generating Figure: 24-Hour Diurnal Cycle...")
    hourly_mean = df_positive.groupby(df_positive.index.hour).mean().mean(axis=1)
    hourly_std = df_positive.groupby(df_positive.index.hour).std().mean(axis=1)
    hourly_p25 = df_positive.groupby(df_positive.index.hour).quantile(0.25).mean(axis=1)
    hourly_p75 = df_positive.groupby(df_positive.index.hour).quantile(0.75).mean(axis=1)

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    ax.plot(hourly_mean.index, hourly_mean.values, color="#1f77b4", linewidth=2.5, marker="o", label="Network Mean Speed")
    ax.fill_between(hourly_mean.index, hourly_p25.values, hourly_p75.values, color="#1f77b4", alpha=0.2, label="Interquartile Range (P25 - P75)")
    ax.axvspan(7, 9.5, color="#d62728", alpha=0.15, label="Morning Peak (07:00-09:30)")
    ax.axvspan(16, 19, color="#ff7f0e", alpha=0.15, label="Evening Peak (16:00-19:00)")

    ax.set_title("Diurnal Traffic Speed Profile (METR-LA Network-Wide Average)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Hour of Day (Local Time, PDT)", fontsize=11)
    ax.set_ylabel("Speed (mph)", fontsize=11)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(0, 23)
    ax.set_ylim(40, 68)
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    plt.tight_layout()
    fig1_path = EDA_FIGURES_DIR / "diurnal_cycle.png"
    fig.savefig(fig1_path)
    plt.close(fig)
    print(f" Saved: {fig1_path}")

    # -------------------------------------------------------------
    # 2. Weekday vs. Weekend Speed Profiles
    # -------------------------------------------------------------
    print(" Generating Figure: Weekday vs Weekend Speed Comparison...")
    is_weekend = df_positive.index.dayofweek >= 5
    df_weekday = df_positive[~is_weekend]
    df_weekend = df_positive[is_weekend]

    weekday_hourly = df_weekday.groupby(df_weekday.index.hour).mean().mean(axis=1)
    weekend_hourly = df_weekend.groupby(df_weekend.index.hour).mean().mean(axis=1)

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    ax.plot(weekday_hourly.index, weekday_hourly.values, color="#d62728", linewidth=2.5, marker="s", label="Weekday (Mon - Fri)")
    ax.plot(weekend_hourly.index, weekend_hourly.values, color="#2ca02c", linewidth=2.5, marker="^", linestyle="--", label="Weekend (Sat - Sun)")

    ax.set_title("Weekday vs. Weekend Diurnal Speed Dynamics", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Hour of Day (Local Time, PDT)", fontsize=11)
    ax.set_ylabel("Mean Speed (mph)", fontsize=11)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(0, 23)
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    plt.tight_layout()
    fig2_path = EDA_FIGURES_DIR / "weekday_vs_weekend.png"
    fig.savefig(fig2_path)
    plt.close(fig)
    print(f" Saved: {fig2_path}")

    # -------------------------------------------------------------
    # 3. Autocorrelation Function (ACF) over Historical Lags
    # -------------------------------------------------------------
    print(" Generating Figure: Autocorrelation Function (ACF)...")
    # Sample 5 representative pilot sensors for ACF
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    max_lags = 48  # 4 hours (48 * 5min)
    lags = np.arange(1, max_lags + 1)
    lag_minutes = lags * 5

    palette = sns.color_palette("tab10", len(PILOT_SENSORS[:5]))
    for idx, s_id in enumerate(PILOT_SENSORS[:5]):
        series = df[s_id]
        acf_vals = [series.autocorr(lag=l) for l in lags]
        ax.plot(lag_minutes, acf_vals, label=f"Sensor {s_id}", color=palette[idx], linewidth=1.8)

    ax.axvline(15, color="gray", linestyle=":", label="Horizon 15m (h=3)")
    ax.axvline(30, color="gray", linestyle="--", label="Horizon 30m (h=6)")
    ax.axvline(60, color="gray", linestyle="-.", label="Horizon 60m (h=12)")

    ax.set_title("Autocorrelation Function (ACF) over 4-Hour Horizon", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Lag Duration (Minutes)", fontsize=11)
    ax.set_ylabel("Autocorrelation Coefficient", fontsize=11)
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="upper right", frameon=True, fontsize=9, ncol=2)
    plt.tight_layout()
    fig3_path = EDA_FIGURES_DIR / "autocorrelation_acf.png"
    fig.savefig(fig3_path)
    plt.close(fig)
    print(f" Saved: {fig3_path}")

    # -------------------------------------------------------------
    # 4. Sensor Graph Degree Distribution & Distance Histogram
    # -------------------------------------------------------------
    print(" Generating Figure: Road Network Topology Summary...")
    adj_mx = graph_meta["adj_mx"]
    # Degree of each sensor (non-zero directed incoming and outgoing edges)
    in_degrees = (adj_mx > 0).sum(axis=0) - 1  # subtract self loop
    out_degrees = (adj_mx > 0).sum(axis=1) - 1

    distances_df = graph_meta["distances"]
    non_self_distances = distances_df[distances_df["cost"] > 0]["cost"] / 1000.0  # in kilometers

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Degree distribution
    ax1.hist(out_degrees, bins=15, color="#17becf", edgecolor="black", alpha=0.8)
    ax1.set_title("Node Degree Distribution (Thresholded Graph)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Directed Out-Degree (Connected Sensors)", fontsize=10)
    ax1.set_ylabel("Sensor Count", fontsize=10)

    # Distance distribution
    sns.histplot(non_self_distances[non_self_distances < 50], bins=30, kde=True, color="#9467bd", ax=ax2)
    ax2.set_title("Pairwise Road Distance Distribution (<50 km)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Shortest Path Distance (km)", fontsize=10)
    ax2.set_ylabel("Pair Count", fontsize=10)

    plt.tight_layout()
    fig4_path = EDA_FIGURES_DIR / "graph_topology_summary.png"
    fig.savefig(fig4_path)
    plt.close(fig)
    print(f" Saved: {fig4_path}")

    print(" All EDA figures successfully generated!")


if __name__ == "__main__":
    generate_eda_figures()
