"""
Streamlit Evaluation & Forecasting Dashboard (Phase 1 Classical ML System).
Offline historical interactive application for METR-LA Traffic Speed Forecasting.
Enterprise Research UI - High Contrast Dark Theme Specification.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Ensure project root is available on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Page configuration
st.set_page_config(
    page_title="METR-LA Speed Forecasting | Academic ML Benchmark",
    page_icon="https://img.icons8.com/material-rounded/96/38BDF8/speed.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom SVG Icons (Lucide-style clean vector icons with explicit stroke colors)
ICONS = {
    "speed": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg>',
    "activity": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
    "database": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#818CF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/></svg>',
    "network": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="16" y="16" width="6" height="6" rx="1"/><rect x="2" y="16" width="6" height="6" rx="1"/><rect x="9" y="2" width="6" height="6" rx="1"/><path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3"/><path d="M12 12V8"/></svg>',
    "chart": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#34D399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/></svg>',
    "shield": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#60A5FA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/></svg>',
    "layers": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#A78BFA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/></svg>',
    "alert": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/></svg>',
    "calendar": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F472B6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="4" rx="2" ry="2"/><line x1="16" x2="16" y1="2" y2="6"/><line x1="8" x2="8" y1="2" y2="6"/><line x1="3" x2="21" y1="10" y2="10"/></svg>',
    "cpu": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/></svg>',
    "map": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#34D399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.106 5.553a2 2 0 0 0 1.788 0l3.659-1.83A1 1 0 0 1 21 4.619v12.764a1 1 0 0 1-.553.894l-4.553 2.277a2 2 0 0 1-1.788 0l-4.212-2.106a2 2 0 0 0-1.788 0l-3.659 1.83A1 1 0 0 1 3 19.381V6.618a1 1 0 0 1 .553-.894l4.553-2.277a2 2 0 0 1 1.788 0z"/><path d="M15 5.764v15"/><path d="M9 3.236v15"/></svg>'
}

# Cohesive Dark/Glass Theme CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography */
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Container constraints */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1350px !important;
    }

    /* Sidebar Fixes: Ensure Radio Labels are 100% VISIBLE and styled */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span {
        color: #F1F5F9 !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stRadio"] > label {
        color: #94A3B8 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-weight: 700 !important;
        margin-bottom: 8px !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        padding: 6px 10px !important;
        border-radius: 6px !important;
        margin-bottom: 4px !important;
        transition: background 0.15s ease !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background: #1E293B !important;
    }

    /* Brand Banner */
    .brand-banner {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 22px 26px;
        color: #FFFFFF;
        margin-bottom: 22px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .brand-pills {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 10px;
    }
    .brand-pill {
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #38BDF8;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .brand-title {
        font-size: 1.65rem;
        font-weight: 800;
        line-height: 1.3;
        letter-spacing: -0.02em;
        color: #F8FAFC !important;
        margin: 0 0 6px 0;
    }
    .brand-sub {
        font-size: 0.92rem;
        color: #94A3B8 !important;
        margin: 0;
        line-height: 1.5;
    }

    /* Cohesive Dark KPI Cards (No jarring white blocks) */
    .kpi-container {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    .kpi-label {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .kpi-caption {
        font-size: 0.78rem;
        color: #38BDF8;
        font-weight: 500;
        margin-top: 4px;
    }

    /* Academic Scope Banner */
    .guardrail-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid #334155;
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 20px;
        font-size: 0.88rem;
        color: #CBD5E1;
        line-height: 1.6;
    }
    .guardrail-card strong {
        color: #F8FAFC;
    }

    /* Section Headings */
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #FFFFFF !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        margin-top: 28px !important;
        margin-bottom: 14px !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Step Grid */
    .flow-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 12px;
        margin: 14px 0 22px 0;
    }
    .flow-step {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
    }
    .flow-num {
        display: inline-block;
        background: #3B82F6;
        color: #FFFFFF;
        font-size: 0.7rem;
        font-weight: 700;
        border-radius: 4px;
        padding: 2px 6px;
        margin-bottom: 8px;
    }
    .flow-title {
        font-weight: 600;
        font-size: 0.88rem;
        color: #F8FAFC;
        margin-bottom: 4px;
    }
    .flow-desc {
        font-size: 0.78rem;
        color: #94A3B8;
        line-height: 1.4;
    }

    /* Forecast Lab Result Cards */
    .forecast-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .forecast-horizon {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #38BDF8;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }

    /* Sidebar info */
    .sidebar-meta-box {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 14px;
        font-size: 0.8rem;
        color: #94A3B8;
        line-height: 1.6;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding-bottom: 12px;
        border-bottom: 1px solid #334155;
        margin-bottom: 14px;
    }
    .sidebar-brand-text {
        font-weight: 700;
        font-size: 1rem;
        color: #F8FAFC;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_app_artifacts():
    """Load lightweight precomputed artifacts for instantaneous startup."""
    with open(REPO_ROOT / "data_lineage.json", "r") as f:
        lineage = json.load(f)
    with open(REPO_ROOT / "data_audit.json", "r") as f:
        audit = json.load(f)
    with open(REPO_ROOT / "data_summary.json", "r") as f:
        summary = json.load(f)

    df_horizon = pd.read_csv(REPO_ROOT / "metrics_by_horizon.csv")
    df_peak = pd.read_csv(REPO_ROOT / "metrics_peak_offpeak.csv")
    df_sensor = pd.read_csv(REPO_ROOT / "metrics_by_sensor.csv")
    df_ablation = pd.read_csv(REPO_ROOT / "metrics_ablation.csv")

    locations_df = pd.read_csv(REPO_ROOT / "data/sensor_graph/graph_sensor_locations.csv")

    history_slice_path = REPO_ROOT / "data/processed/forecast_lab_history.parquet"
    if history_slice_path.exists():
        test_history_df = pd.read_parquet(history_slice_path)
    else:
        test_history_df = None

    preds_path = REPO_ROOT / "test_predictions.parquet"
    if preds_path.exists():
        preds_df = pd.read_parquet(preds_path)
    else:
        preds_df = pd.read_csv(REPO_ROOT / "test_predictions_sample.csv")

    return {
        "lineage": lineage,
        "audit": audit,
        "summary": summary,
        "metrics_horizon": df_horizon,
        "metrics_peak": df_peak,
        "metrics_sensor": df_sensor,
        "metrics_ablation": df_ablation,
        "locations": locations_df,
        "test_history": test_history_df,
        "predictions": preds_df,
    }


artifacts = load_app_artifacts()

# =========================================================================
# SIDEBAR NAVIGATION
# =========================================================================
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <div style="background: #0284C7; color: white; padding: 8px; border-radius: 6px; display: flex; align-items: center; justify-content: center;">
                {ICONS['network']}
            </div>
            <div>
                <div class="sidebar-brand-text">Caltrans METR-LA</div>
                <div style="font-size: 0.74rem; color: #94A3B8;">Traffic Speed Benchmark</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Executive Overview",
            "Dataset & Provenance Audit",
            "Interactive Forecast Lab",
            "Multi-Horizon Benchmarks",
            "Network Topology & Phase 2",
        ],
        index=0,
    )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="sidebar-meta-box">
            <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                {ICONS['shield']} Scientific Metadata
            </div>
            <strong>Program</strong>: Flow & Signal Control<br>
            <strong>Phase</strong>: Phase 1 Classical Baselines<br>
            <strong>Data</strong>: METR-LA Loop Detectors<br>
            <strong>Cohort</strong>: 15 Pilot Segments<br>
            <strong>Splits</strong>: Chronological 70/10/20<br>
            <strong>Target</strong>: Speed (mph)
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================================
# MODULE 1: EXECUTIVE OVERVIEW
# =========================================================================
if page == "Executive Overview":
    st.markdown(
        f"""
        <div class="brand-banner">
            <div class="brand-pills">
                <span class="brand-pill">Phase 1 System</span>
                <span class="brand-pill">Classical ML Baselines</span>
                <span class="brand-pill">Frozen Test Benchmarking</span>
            </div>
            <h1 class="brand-title">Historical METR-LA Traffic Speed Forecasting</h1>
            <p class="brand-sub">Multi-horizon predictive framework evaluated at 15m, 30m, and 60m horizons across Los Angeles highway corridors.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="guardrail-card">
            <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #60A5FA; margin-bottom: 6px;">
                {ICONS['alert']} Academic Scope & Scientific Boundary Protocol
            </div>
            This application represents <strong>Phase 1</strong> of the highway traffic modeling initiative. 
            It is strictly constrained to <em>historical highway traffic speed forecasting</em> in miles per hour (mph) using rigorously evaluated classical ML architectures (Historical Persistence, L2-Regularized Ridge Regression, and Gradient Boosted Decision Trees). 
            The raw Caltrans METR-LA benchmark data contains inductive loop detector speed records; it does <strong>not</strong> contain vehicular volume/flow counts or traffic signal phase states. 
            Accordingly, no ungrounded claims of live signal actuation or vehicular flow prediction are asserted. 
            Phase 2 is dedicated to deep Graph Attention Networks (GAT) and spatial-temporal representations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Key Metrics Cards in Dark Theme
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-label">{ICONS['database']} Sensor Network</div>
                <div class="kpi-value">{artifacts['audit']['shape']['num_sensors']}</div>
                <div class="kpi-caption">Active Caltrans Detectors</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-label">{ICONS['speed']} Benchmark Cohort</div>
                <div class="kpi-value">{artifacts['summary']['pilot_sensors_count']}</div>
                <div class="kpi-caption">High-Variance Corridors</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-label">{ICONS['calendar']} Observation Steps</div>
                <div class="kpi-value">{artifacts['audit']['shape']['num_timestamps']:,}</div>
                <div class="kpi-caption">Continuous 5m Samples</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-label">{ICONS['chart']} Best 15m MAE</div>
                <div class="kpi-value">2.59 <span style="font-size: 1rem; color: #94A3B8;">mph</span></div>
                <div class="kpi-caption">Achieved by XGBoost</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div class='section-title'>{ICONS['layers']} End-to-End System Architecture</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="flow-grid">
            <div class="flow-step">
                <span class="flow-num">Step 01</span>
                <div class="flow-title">Raw Ingestion & Lineage</div>
                <div class="flow-desc">Verified DCRNN METR-LA HDF5 matrix with cryptographic SHA-256 provenance tracking.</div>
            </div>
            <div class="flow-step">
                <span class="flow-num">Step 02</span>
                <div class="flow-title">Chronological Split</div>
                <div class="flow-desc">Strict 70% Train, 10% Val, 20% Test partition by timestamp with zero forward leakage.</div>
            </div>
            <div class="flow-step">
                <span class="flow-num">Step 03</span>
                <div class="flow-title">Causal Windowing</div>
                <div class="flow-desc">History window (t-11..t), cyclic time features, and 1st-order spatial graph neighbor lags.</div>
            </div>
            <div class="flow-step">
                <span class="flow-num">Step 04</span>
                <div class="flow-title">Preprocessing Isolation</div>
                <div class="flow-desc">StandardScaler fitted exclusively on Training subset to guarantee empirical zero-leakage.</div>
            </div>
            <div class="flow-step">
                <span class="flow-num">Step 05</span>
                <div class="flow-title">Multi-Horizon Eval</div>
                <div class="flow-desc">Sanctioned testing at +15m, +30m, +60m horizons evaluated on identical valid target masks.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='section-title'>{ICONS['shield']} Four-Person Workstream Attribution</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        pd.DataFrame([
            {"Workstream": "WS1: Data & Provenance", "Lead": "Member 1", "Core Modules": "src/data/loader.py, data_lineage.json, data_audit.json, EDA Figures"},
            {"Workstream": "WS2: Features & Baselines", "Lead": "Member 2", "Core Modules": "src/features/build_features.py, persistence.py, ridge_baseline.py, tests/"},
            {"Workstream": "WS3: Non-Linear ML & Eval", "Lead": "Member 3", "Core Modules": "src/models/train_xgboost.py, metrics.py, ablation.py, metrics_by_horizon.csv"},
            {"Workstream": "WS4: Systems & Release", "Lead": "Member 4", "Core Modules": "app/streamlit_app.py, IEEE LaTeX Paper, PPTX Deck, VIVA_GUIDE.md"},
        ]),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================================
# MODULE 2: DATASET & PROVENANCE AUDIT
# =========================================================================
elif page == "Dataset & Provenance Audit":
    st.markdown(
        f"""
        <div class="brand-banner">
            <div class="brand-pills">
                <span class="brand-pill">Data Lineage</span>
                <span class="brand-pill">Missingness Protocol</span>
                <span class="brand-pill">Zero Leakage</span>
            </div>
            <h1 class="brand-title">Dataset Provenance & Missingness Audit</h1>
            <p class="brand-sub">Comprehensive verification of inductive loop detector health, missingness accounting, and temporal splits.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='section-title'>{ICONS['database']} Missingness & Observation Accounting</div>", unsafe_allow_html=True)
        audit_miss = artifacts["audit"]["missingness_and_validity"]
        st.dataframe(
            pd.DataFrame([
                {"Category": "Total Expected Readings", "Count": f"{artifacts['audit']['shape']['total_observations']:,}", "Percentage": "100.0%"},
                {"Category": "Null / NaN Values", "Count": f"{audit_miss['nan_count']}", "Percentage": "0.0%"},
                {"Category": "Zero-Speed Observations (0.0 mph)", "Count": f"{audit_miss['zero_count']:,}", "Percentage": f"{audit_miss['zero_ratio']*100:.2f}%"},
                {"Category": "Physiologically Impossible (>100 mph)", "Count": f"{audit_miss['impossible_speed_gt_100_count']}", "Percentage": "0.0%"},
                {"Category": "Valid Evaluation Speeds (>0.0 and <=100)", "Count": f"{audit_miss['valid_positive_measurements']:,}", "Percentage": f"{audit_miss['valid_positive_ratio']*100:.2f}%"},
            ]),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            f"""
            <div class="guardrail-card" style="margin-top: 14px;">
                <strong>Zero-Speed Handling Policy</strong>: In freeway loop detectors, speed readings of exactly 0.0 mph indicate either severe congestion, lane closures, or telemetry dropouts. 
                For model inputs, causal forward-fill imputation preserves continuity. For evaluation targets, all models are scored <em>strictly on valid positive observations (>0.0 mph)</em> to eliminate infinite MAPE distortion.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(f"<div class='section-title'>{ICONS['calendar']} Chronological Split Boundaries</div>", unsafe_allow_html=True)
        parts = artifacts["audit"]["chronological_partitions"]
        st.dataframe(
            pd.DataFrame([
                {"Partition": "Training (70%)", "Samples": f"{parts['train']['num_timestamps']:,}", "Period Start": parts['train']['start'], "Period End": parts['train']['end']},
                {"Partition": "Validation (10%)", "Samples": f"{parts['validation']['num_timestamps']:,}", "Period Start": parts['validation']['start'], "Period End": parts['validation']['end']},
                {"Partition": "Held-Out Test (20%)", "Samples": f"{parts['test']['num_timestamps']:,}", "Period Start": parts['test']['start'], "Period End": parts['test']['end']},
            ]),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            f"""
            <div class="guardrail-card" style="margin-top: 14px; border-left-color: #10B981;">
                <strong>Zero-Leakage Guarantee</strong>: StandardScalers, moving averages, and hyperparameter grids were computed exclusively on the 70% Training subset. Validation and Test splits are evaluated out-of-sample with zero parameter contamination.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin: 28px 0 16px 0; border-color: #334155;'>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{ICONS['chart']} Exploratory Data Analysis (EDA) Gallery</div>", unsafe_allow_html=True)

    eda_tabs = st.tabs(["Diurnal Rush-Hour Profile", "Weekday vs Weekend Dynamics", "Temporal Autocorrelation (ACF)", "Road Graph Topology"])

    with eda_tabs[0]:
        img_path = REPO_ROOT / "reports/figures/eda/diurnal_cycle.png"
        if img_path.exists():
            st.image(str(img_path), caption="Diurnal Speed Drop Dynamics (Morning Rush 07:00-09:30, Evening Rush 16:00-19:00)", use_container_width=True)

    with eda_tabs[1]:
        img_path = REPO_ROOT / "reports/figures/eda/weekday_vs_weekend.png"
        if img_path.exists():
            st.image(str(img_path), caption="Comparison of Recurring Weekday Commutes vs Fluid Weekend Traffic Distribution", use_container_width=True)

    with eda_tabs[2]:
        img_path = REPO_ROOT / "reports/figures/eda/autocorrelation_acf.png"
        if img_path.exists():
            st.image(str(img_path), caption="Temporal Autocorrelation Function (ACF) over 48 5-minute lags (4 hours)", use_container_width=True)

    with eda_tabs[3]:
        img_path = REPO_ROOT / "reports/figures/eda/graph_topology_summary.png"
        if img_path.exists():
            st.image(str(img_path), caption="Topological Road Graph In-Degree / Out-Degree Distribution and Sensor Cost Distances", use_container_width=True)


# =========================================================================
# MODULE 3: INTERACTIVE FORECAST LAB
# =========================================================================
elif page == "Interactive Forecast Lab":
    st.markdown(
        f"""
        <div class="brand-banner">
            <div class="brand-pills">
                <span class="brand-pill">Forecast Lab</span>
                <span class="brand-pill">Multi-Horizon Inspection</span>
                <span class="brand-pill">Ground Truth Alignment</span>
            </div>
            <h1 class="brand-title">Interactive Multi-Horizon Forecast Lab</h1>
            <p class="brand-sub">Select test set origin timestamps to examine causal past trajectories vs forward multi-step model predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    preds_df = artifacts["predictions"]
    test_history = artifacts["test_history"]

    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        pilot_list = artifacts["summary"]["pilot_sensors"]
        selected_sensor = st.selectbox("Select Traffic Loop Station:", pilot_list, index=0)

        sensor_preds = preds_df[preds_df["sensor_id"] == selected_sensor]
        unique_origins = sensor_preds["origin_ts"].drop_duplicates().tolist()

        default_origin = "2012-06-05 08:00:00"
        default_idx = unique_origins.index(default_origin) if default_origin in unique_origins else len(unique_origins) // 2
        selected_origin = st.selectbox("Forecast Origin Timestamp (t):", unique_origins, index=default_idx)

    with col_ctrl2:
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-label">{ICONS['activity']} Active Scenario Specifications</div>
                <div style="font-size: 0.86rem; color: #E2E8F0; line-height: 1.6;">
                    <strong>Station Identifier</strong>: <code>{selected_sensor}</code> (Caltrans District 7)<br>
                    <strong>Origin Timestamp (t)</strong>: <code>{selected_origin}</code> PDT (Held-Out Test Split)<br>
                    <strong>Input History Window</strong>: 12 intervals (60 minutes past: <code>t-11 ... t</code>)<br>
                    <strong>Target Horizons</strong>: +15 min (<code>t+3</code>), +30 min (<code>t+6</code>), +60 min (<code>t+12</code>)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    origin_df = sensor_preds[sensor_preds["origin_ts"] == selected_origin]

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{ICONS['chart']} Multi-Horizon Predictive Accuracy</div>", unsafe_allow_html=True)

    h_cols = st.columns(3)
    horizons_meta = [("15min", "+15 min (Step +3)"), ("30min", "+30 min (Step +6)"), ("60min", "+60 min (Step +12)")]

    for idx, (h_key, h_label) in enumerate(horizons_meta):
        with h_cols[idx]:
            h_slice = origin_df[origin_df["horizon"] == h_key]
            if len(h_slice) > 0:
                y_true = h_slice["y_true"].iloc[0]
                xgb_row = h_slice[h_slice["model"] == "XGBoost"]
                xgb_pred = xgb_row["y_pred"].iloc[0] if len(xgb_row) > 0 else np.nan
                abs_err = abs(xgb_pred - y_true) if not np.isnan(xgb_pred) else np.nan

                st.markdown(
                    f"""
                    <div class="forecast-card">
                        <div class="forecast-horizon">{h_label}</div>
                        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
                            <span style="font-size: 0.8rem; color: #94A3B8;">Ground Truth:</span>
                            <span style="font-weight: 700; font-size: 1.05rem; color: #F8FAFC;">{y_true:.1f} mph</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
                            <span style="font-size: 0.8rem; color: #94A3B8;">XGBoost Forecast:</span>
                            <span style="font-weight: 700; font-size: 1.05rem; color: #38BDF8;">{xgb_pred:.1f} mph</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: baseline; border-top: 1px solid #334155; padding-top: 6px;">
                            <span style="font-size: 0.78rem; color: #94A3B8;">Absolute Error:</span>
                            <span style="font-weight: 600; font-size: 0.92rem; color: #34D399;">{abs_err:.2f} mph</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Matplotlib Professional Dark Theme Forecast Trajectory
    if test_history is not None and selected_sensor in test_history.columns:
        origin_dt = pd.to_datetime(selected_origin)
        hist_start_dt = origin_dt - pd.Timedelta(minutes=55)
        future_end_dt = origin_dt + pd.Timedelta(minutes=60)

        sensor_series = test_history[selected_sensor]
        hist_mask = (sensor_series.index >= hist_start_dt) & (sensor_series.index <= origin_dt)
        hist_segment = sensor_series[hist_mask]

        fig, ax = plt.subplots(figsize=(11.5, 4.4), dpi=300)
        fig.patch.set_facecolor('#0F172A')
        ax.set_facecolor('#1E293B')

        # Observed Past Trajectory
        ax.plot(hist_segment.index, hist_segment.values, color="#38BDF8", marker="o", markersize=5, linewidth=2.2, label="Observed Past History (t-11 ... t)")
        ax.axvline(origin_dt, color="#F87171", linestyle="--", linewidth=1.6, label=f"Forecast Origin t ({origin_dt.strftime('%H:%M')})")

        # Future Ground Truth
        future_mask = (sensor_series.index > origin_dt) & (sensor_series.index <= future_end_dt)
        future_truth = sensor_series[future_mask]
        ax.plot(future_truth.index, future_truth.values, color="#E2E8F0", linestyle="-", marker="s", markersize=4.5, linewidth=1.8, label="Held-Out Ground Truth")

        # Model Forecast Markers
        palette = {"Persistence": "#94A3B8", "Ridge": "#818CF8", "XGBoost": "#34D399"}
        markers = {"Persistence": "v", "Ridge": "D", "XGBoost": "^"}

        for m_name in ["Persistence", "Ridge", "XGBoost"]:
            m_slice = origin_df[origin_df["model"] == m_name]
            if len(m_slice) > 0:
                t_dts = pd.to_datetime(m_slice["target_ts"])
                preds_vals = m_slice["y_pred"].values
                ax.plot(t_dts, preds_vals, color=palette[m_name], marker=markers[m_name], markersize=7, linestyle=":", linewidth=1.8, label=f"{m_name} Predictions")

        ax.set_title(f"Station {selected_sensor} Trajectory Analysis (Origin: {selected_origin} PDT)", fontsize=11, fontweight="bold", pad=12, color="#F8FAFC")
        ax.set_xlabel("Observation Timestamp", fontsize=10, labelpad=8, color="#94A3B8")
        ax.set_ylabel("Traffic Speed (mph)", fontsize=10, labelpad=8, color="#94A3B8")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(colors='#94A3B8', which='both')
        ax.grid(True, linestyle="--", alpha=0.35, color="#334155")
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.legend(loc="lower left", frameon=True, facecolor="#0F172A", edgecolor="#334155", labelcolor="#E2E8F0", fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# =========================================================================
# MODULE 4: MULTI-HORIZON BENCHMARKS
# =========================================================================
elif page == "Multi-Horizon Benchmarks":
    st.markdown(
        f"""
        <div class="brand-banner">
            <div class="brand-pills">
                <span class="brand-pill">Benchmarking</span>
                <span class="brand-pill">Peak vs Off-Peak</span>
                <span class="brand-pill">Spatial Ablation</span>
            </div>
            <h1 class="brand-title">Empirical Benchmark & Error Analysis</h1>
            <p class="brand-sub">Multi-horizon accuracy metrics, rush-hour sensitivity, and road graph spatial lag feature ablation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<div class='section-title'>{ICONS['chart']} 1. Multi-Horizon Metric Comparison (Held-Out Test Split)</div>", unsafe_allow_html=True)
    st.markdown("Common valid target evaluation mask applied across all architectures (90,149 valid test observations per horizon).")

    st.dataframe(artifacts["metrics_horizon"], use_container_width=True, hide_index=True)

    c_bench1, c_bench2 = st.columns(2)
    with c_bench1:
        img_path = REPO_ROOT / "reports/figures/evaluation/multi_horizon_comparison.png"
        if img_path.exists():
            st.image(str(img_path), caption="Comparative Error Scaling (MAE & RMSE) across 15m, 30m, and 60m horizons", use_container_width=True)
    with c_bench2:
        img_path = REPO_ROOT / "reports/figures/evaluation/residuals_distribution.png"
        if img_path.exists():
            st.image(str(img_path), caption="Residual Error Distribution on Held-Out Test Set (Zero-Centered, Low Variance)", use_container_width=True)

    st.markdown("<hr style='margin: 28px 0 16px 0; border-color: #334155;'>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{ICONS['activity']} 2. Peak (Rush-Hour) vs Off-Peak Sensitivity Analysis</div>", unsafe_allow_html=True)
    st.markdown("Rush-hour intervals: Morning Peak (07:00-09:30) and Evening Peak (16:00-19:00) during weekdays. Non-linear traffic shockwaves systematically increase error rates.")
    st.dataframe(artifacts["metrics_peak"], use_container_width=True, hide_index=True)

    st.markdown("<hr style='margin: 28px 0 16px 0; border-color: #334155;'>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{ICONS['network']} 3. Topological Spatial Lag Feature Ablation</div>", unsafe_allow_html=True)

    col_abl1, col_abl2 = st.columns(2)
    with col_abl1:
        img_path = REPO_ROOT / "reports/figures/evaluation/spatial_ablation_comparison.png"
        if img_path.exists():
            st.image(str(img_path), caption="Performance Impact of Road Graph 1st-Order Spatial Lag Features vs Isolated Lags", use_container_width=True)

    with col_abl2:
        st.markdown(
            f"""
            <div class="guardrail-card" style="margin-top: 10px;">
                <div style="font-weight: 700; color: #38BDF8; margin-bottom: 8px; font-size: 1rem;">
                    Empirical Ablation Finding
                </div>
                Injecting 1st-order spatial neighbor lags (computed from the pre-constructed road adjacency matrix {{ij}}$) 
                consistently reduces MAE across all forward horizons:
                <ul style="margin-top: 8px; padding-left: 20px;">
                    <li><strong>15m Horizon</strong>: 2.71 mph &rarr; <strong>2.59 mph</strong> (-4.4% error)</li>
                    <li><strong>30m Horizon</strong>: 3.32 mph &rarr; <strong>3.17 mph</strong> (-4.5% error)</li>
                    <li><strong>60m Horizon</strong>: 4.02 mph &rarr; <strong>3.81 mph</strong> (-5.2% error)</li>
                </ul>
                This empirical improvement proves that spatial shockwaves propagate downstream along highway corridors, 
                establishing a direct motivation for dynamic Graph Attention Networks in Phase 2.
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================================
# MODULE 5: NETWORK TOPOLOGY & PHASE 2 OUTLOOK
# =========================================================================
elif page == "Network Topology & Phase 2":
    st.markdown(
        f"""
        <div class="brand-banner">
            <div class="brand-pills">
                <span class="brand-pill">Sensor Network</span>
                <span class="brand-pill">Los Angeles County</span>
                <span class="brand-pill">Phase 2 Roadmap</span>
            </div>
            <h1 class="brand-title">Sensor Network Topology & Phase 2 Outlook</h1>
            <p class="brand-sub">Spatial distribution of Caltrans District 7 inductive loop detectors and theoretical formulation for Graph Attention Networks.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    loc_df = artifacts["locations"]
    st.markdown(f"<div class='section-title'>{ICONS['map']} Geographic Sensor Array (207 Highway Loop Detectors)</div>", unsafe_allow_html=True)
    st.markdown("Interactive spatial distribution across primary highway arteries (I-5, I-10, I-405, SR-110, US-101) in Los Angeles County.")

    st.map(loc_df[["latitude", "longitude"]], zoom=10)

    st.markdown("<hr style='margin: 28px 0 16px 0; border-color: #334155;'>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{ICONS['cpu']} Phase 2 Architectural Transition: Graph Attention Networks</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        While Phase 1 conclusively demonstrated that classical machine learning (XGBoost) produces competitive baselines (2.59 mph @ 15m), 
        classical architectures have fundamental spatial limitations:
        
        1. <strong>Fixed Spatial Neighborhoods</strong>: Classical ML models evaluate a static neighbor count (=3$), failing to adapt to fluctuating traffic congestion lengths.
        2. <strong>Time-Invariant Spatial Weights</strong>: An upstream sensor 2 miles away possesses high predictive significance during heavy congestion shockwaves, but negligible significance during free-flow conditions.
        
        <div class="guardrail-card" style="background: #1E293B; border-left-color: #38BDF8; margin: 18px 0;">
            <strong>Phase 2 GAT Formulation (Dynamic Spatial Attention)</strong>:<br>
            Multi-head Graph Attention Networks dynamically compute edge attention coefficients $lpha_{{ij}}^{{(t)}}$ at each temporal timestamp:
            53516lpha_{{ij}}^{{(t)}} = rac{{\exp\left(	ext{{LeakyReLU}}\left(\mathbf{{a}}^T [\mathbf{{W}}h_i^{{(t)}} \parallel \mathbf{{W}}h_j^{{(t)}}]
ight)
ight)}}{{\sum_{{k \in \mathcal{{N}}_i}} \exp\left(	ext{{LeakyReLU}}\left(\mathbf{{a}}^T [\mathbf{{W}}h_i^{{(t)}} \parallel \mathbf{{W}}h_k^{{(t)}}]
ight)
ight)}}53516
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="guardrail-card" style="border-left-color: #F59E0B;">
            <strong>Integrity Notice</strong>: Phase 2 research continues strictly under highway speed prediction. Live traffic signal control requires micro-simulation environments (SUMO/Aimsun) and queue-length detection hardware outside the empirical METR-LA benchmark scope.
        </div>
        """,
        unsafe_allow_html=True,
    )
