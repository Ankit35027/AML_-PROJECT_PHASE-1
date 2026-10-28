# Historical METR-LA Traffic Speed Forecasting
## Phase 1 Baseline & Classical Machine Learning System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)

**Parent Course / Program Title**: *Traffic Flow Prediction and Signal Control via Graph Attention Networks and Flow Models*  
**Phase Scope**: Phase 1 Classical Machine Learning, Statistical Baselines, Multi-Horizon Evaluation, Honest Benchmarking, Streamlit Artifact Dashboard, IEEE Conference Manuscript, and Academic Presentation.

---

## 1. Operating Rules & Scientific Guardrails

In strict adherence to academic integrity and honest scientific reporting:
1. **Academic Truthfulness**: Every reported metric traces directly to reproducible script execution. No synthetic numbers, phantom test scores, or fabricated citations are permitted.
2. **Phase Boundary**: The Caltrans METR-LA benchmark dataset measures **highway traffic speed in miles per hour (mph)**. It does **not** record vehicular flow/volume or traffic signal phase states. Therefore, no ungrounded claims of live traffic signal actuation or vehicular flow prediction are asserted. Phase 2 is dedicated to Graph Attention Networks (GAT) and deep spatial-temporal models.
3. **Data Provenance**: Source data is the benchmark METR-LA sensor dataset introduced in DCRNN (Li et al., ICLR 2018). Raw data lineage and SHA-256 checksums are cryptographically logged in `data_lineage.json`.
4. **Missing Data Policy**: Differentiates `NaN` (0), `0.0` (575,302 / 8.11%), and impossible speeds (>100 mph; 0). Speeds of 0.0 mph denote either severe congestion, sensor dropouts, or lane closures. Evaluation labels require strictly valid $>0.0$ mph measurements; historical feature lags use causal forward-fill imputation.
5. **Split Integrity**: Chronological 70% Train, 10% Validation, 20% Test partitioning by timestamp with zero temporal overlap. Scalers are fitted exclusively on the Training partition.
6. **Causal Windowing**: Forecast origin $t$; historical lags $t-11 \dots t$ (past 60 minutes); target horizons at +15m ($h=3$), +30m ($h=6$), and +60m ($h=12$).

---

## 2. Four-Person Workstream Ownership Matrix

| Workstream | Lead | Responsibilities | Core Deliverables | IEEE Manuscript Section |
| :--- | :--- | :--- | :--- | :--- |
| **WS1: Data & Provenance** | Member 1 | Data ingestion, lineage manifest, missingness audit, sensor alignment, EDA figures. | `src/data/loader.py`, `src/data/audit.py`, `notebooks/01_eda.py`, `data_audit.json`, `data_lineage.json` | Section III: Dataset & Provenance |
| **WS2: Features & Baselines** | Member 2 | Causal window builder, split validation, persistence benchmark, Ridge baseline, unit tests. | `src/features/build_features.py`, `src/models/persistence.py`, `src/models/ridge_baseline.py`, `tests/` | Section IV: Methodology & Benchmarks |
| **WS3: Models & Evaluation** | Member 3 | Multi-horizon XGBoost, validation tuning, spatial lag ablation, comprehensive metrics. | `src/models/train_xgboost.py`, `src/evaluation/metrics.py`, `src/evaluation/ablation.py`, `metrics_by_horizon.csv` | Section V: Experimental Results |
| **WS4: Systems & Release** | Member 4 | Streamlit dashboard, automated pipeline, CI workflow, IEEE LaTeX paper, PPTX presentation, VIVA guide. | `app/streamlit_app.py`, `.github/workflows/ci.yml`, `reports/paper/`, `presentation/`, `VIVA_GUIDE.md` | Section I & VI: Intro, Deployment & Phase 2 |

---

## 3. End-to-End Architecture

```
METR-LA Raw Matrix + Graph Topology
         │
         ▼
Data Validation, Alignment & Missingness Audit (src/data/audit.py)
         │
         ▼
Chronological 70/10/20 Partitioning (Train: Mar-May | Val: May-Jun | Test: Jun)
         │
         ▼
Causal Feature Windowing (Lags t-11..t, Cyclical Time, Top-3 Spatial Neighbors)
         │
         ▼
Training-Only Preprocessing Isolation (StandardScaler fitted strictly on Train)
         │
         ▼
Benchmark Models:
  ├─ Historical Persistence: y_hat(t+h) = y(t)
  ├─ Multi-Horizon Ridge: L2 regularized linear models (Tuned alpha)
  └─ Multi-Horizon XGBoost: Gradient boosted trees (Early stopping)
         │
         ▼
Multi-Horizon Frozen Evaluation (+15m, +30m, +60m with Common Valid Mask)
         │
         ▼
Deliverables & Artifacts:
  ├─ Streamlit Offline Interactive Forecast Lab (app/streamlit_app.py)
  ├─ IEEE Conference Manuscript (reports/paper/main.tex)
  ├─ 15-Slide Academic PowerPoint Presentation Deck (presentation/)
  └─ Comprehensive Examiner Defense Guide (VIVA_GUIDE.md)
```

---

## 4. Quick Start & Reproduction

### Prerequisites
- Python 3.9+ (tested on macOS / Linux / Ubuntu)
- Git

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd AML-Network-Project

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Data Provenance & Run Missingness Audit
```bash
python -m src.data.audit
```
*Outputs `data_audit.json` and audit plots in `reports/figures/eda/`.*

### 3. Generate Exploratory Data Analysis (EDA) Figures
```bash
python notebooks/01_eda.py
```
*Outputs diurnal cycles, weekday vs weekend profiles, autocorrelation (ACF), and graph degree distributions in `reports/figures/eda/`.*

### 4. Run Automated Scientific Test Suite
```bash
pytest tests/ -v --durations=10
```
*Validates continuous split boundaries, zero target leakage, exact target offsets (+15, +30, +60m), and manual metric cross-checks.*

### 5. Execute Full Model Benchmark Pipeline
```bash
python -m src.evaluation.evaluate_all
```
*Trains Persistence, Ridge, and XGBoost across all horizons, evaluates on held-out test data, and writes `metrics_by_horizon.csv`, `metrics_peak_offpeak.csv`, `test_predictions.parquet`, and evaluation plots in `reports/figures/evaluation/`.*

### 6. Run Spatial Lag Feature Ablation Experiment
```bash
python -m src.evaluation.ablation
```
*Evaluates isolated sensor lags vs top-3 topological road network neighbors, outputting `metrics_ablation.csv` and `spatial_ablation_comparison.png`.*

### 7. Launch Interactive Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```
*Launches instantly on `http://localhost:8501` in offline mode using precomputed artifacts without retraining.*

---

## 5. Verified Experimental Test Benchmark

Evaluated on 90,149 valid target observations in the held-out test partition:

| Model | Horizon | Lookahead | MAE (mph) | RMSE (mph) | MAPE (%) | Peak MAE (mph) | Off-Peak MAE (mph) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Historical Persistence** | 15 min | +3 steps | 3.0011 | 5.5959 | 7.12% | 4.2384 | 2.6517 |
| **Ridge Regression** | 15 min | +3 steps | 2.8789 | 5.3466 | 7.45% | 4.0851 | 2.5379 |
| **XGBoost Regressor** | **15 min** | **+3 steps** | **2.5909** | **4.8718** | **6.68%** | **3.6596** | **2.2887** |
| **Historical Persistence** | 30 min | +6 steps | 3.6157 | 7.2034 | 9.04% | 5.1092 | 3.1947 |
| **Ridge Regression** | 30 min | +6 steps | 3.6050 | 6.8440 | 10.12% | 4.9602 | 3.2227 |
| **XGBoost Regressor** | **30 min** | **+6 steps** | **3.1668** | **6.0589** | **8.69%** | **4.2127** | **2.8708** |
| **Historical Persistence** | 60 min | +12 steps | 4.7302 | 9.6660 | 12.68% | 6.4022 | 4.2573 |
| **Ridge Regression** | 60 min | +12 steps | 4.9066 | 8.9476 | 15.02% | 6.4729 | 4.4637 |
| **XGBoost Regressor** | **60 min** | **+12 steps** | **4.0083** | **7.5269** | **11.64%** | **5.0863** | **3.7034** |

### Spatial Feature Ablation
- **15 min**: MAE 2.5949 -> **2.5464 mph** (RMSE 4.88 -> 4.66 mph)
- **30 min**: MAE 3.1736 -> **3.0727 mph** (RMSE 6.07 -> 5.76 mph)
- **60 min**: MAE 4.0211 -> **3.8112 mph** (RMSE 7.54 -> 7.03 mph, **5.2% relative error reduction**)

---

## 6. Academic Deliverables & Repository Structure

```
AML-Network-Project/
├── .github/workflows/ci.yml         # GitHub Actions CI workflow
├── 01_PROJECT_RUNBOOK.md            # Acceptance runbook specification
├── CONTRIBUTIONS.md                 # Ethical attribution & workstream matrix
├── TASK_BOARD.md                    # Workstream deliverables & PR checklist
├── VIVA_GUIDE.md                    # Examiner defense guide & 15 Q&As
├── README.md                        # Master production documentation
├── pyproject.toml                   # Project metadata & pytest configuration
├── requirements.txt                 # Pinned dependencies
├── data_audit.json                  # Missingness & split partition audit
├── data_lineage.json                # Cryptographic provenance & SHA-256 hashes
├── data_summary.json                # Verified experimental summary
├── metrics_by_horizon.csv           # Multi-horizon metric table
├── metrics_peak_offpeak.csv         # Stratified rush-hour metric table
├── metrics_by_sensor.csv            # Individual sensor metric table
├── metrics_ablation.csv             # Spatial road network ablation table
├── app/
│   └── streamlit_app.py             # 5-module offline Streamlit application
├── data/
│   ├── raw/metr-la.h5               # Raw speed matrix (34,272 ts x 207 sensors)
│   ├── sensor_graph/                # Distances, sensor IDs, adj_mx.pkl, locations
│   └── processed/                   # Lightweight test demonstration slice
├── models/                          # Serialized joblib model checkpoints
├── notebooks/
│   └── 01_eda.py                    # Callable EDA script generating all figures
├── presentation/
│   ├── AML_Phase1_Traffic_Prediction.pptx # 15-slide presentation with speaker notes
│   └── generate_deck.py             # python-pptx generation script
├── reports/
│   ├── figures/eda/                 # Diurnal, weekday/weekend, ACF, topology plots
│   ├── figures/evaluation/          # Horizon comparison, traces, residuals, ablation
│   └── paper/
│       ├── main.tex                 # IEEE conference manuscript source
│       └── references.bib           # Complete academic bibliography
├── src/
│   ├── data/loader.py, audit.py
│   ├── features/build_features.py
│   ├── models/persistence.py, ridge_baseline.py, train_xgboost.py
│   ├── evaluation/metrics.py, evaluate_all.py, ablation.py
│   └── utils/config.py
└── tests/
    └── test_splits_and_leakage.py   # Pytest validation suite
```

---

## 7. Streamlit Deployment Guide

The Streamlit app is preconfigured for **zero-retraining, instantaneous startup** on Streamlit Community Cloud:
1. Fork or push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io).
3. Connect your repository and specify:
   - **Main file path**: `app/streamlit_app.py`
   - **Python version**: `3.9` or `3.10`
4. The application boots within 2 seconds using precomputed test evaluation artifacts without requiring retraining.

---

## 8. Citation

If using this benchmark or codebase in research, please cite:
```bibtex
@inproceedings{li2018diffusion,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2018}
}
```
