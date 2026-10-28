# 01_PROJECT_RUNBOOK.md
# Project Specification & Acceptance Runbook
**Project Title**: Historical METR-LA Traffic Speed Forecasting (Phase 1 Baseline & Classical ML System)  
**Parent Course / Program Title**: Traffic Flow Prediction and Signal Control via Graph Attention Networks and Flow Models  
**Phase Scope**: Phase 1 Classical Machine Learning, Statistical Baselines, Multi-Horizon Evaluation, Honest Benchmarking, Streamlit Artifact Dashboard, IEEE Manuscript, and Academic Presentation. (Phase 2 is dedicated to GAT/DL/hybrid research; no claims of real-time signal control or unmeasured vehicular flow are permitted).

---

## 1. Operating Rules & Core Guardrails

1. **Academic Truthfulness**: Never invent metrics, test performance, sensor counts, paper citations, commit authors, or deployment URLs. Every reported number must trace to reproducible code execution.
2. **Phase Boundary**: The dataset contains highway speed measurements in miles per hour (mph). Do not claim prediction of volume/flow or occupancy, nor real-time traffic signal actuation.
3. **Data Provenance**: Source data is the benchmark METR-LA sensor dataset introduced in DCRNN (Li et al., ICLR 2018). Raw data lineage and SHA-256 hashes must be tracked in `data_lineage.json`.
4. **Missing Data Policy**: Differentiate `NaN`, `0.0`, and physiologically impossible speeds (>100 mph). A speed of `0.0` in loop detectors frequently denotes severe congestion, sensor dropouts, or road closure. We persist an explicit observation validity mask `valid_mask` and avoid unprincipled zero-to-NaN conversions. Imputations must be causal (computed strictly from past observations) and used exclusively for feature extraction. Evaluation labels must always be ground-truth observed valid values.
5. **Split Integrity**: Strict chronological partitioning: 70% Train, 10% Validation, 20% Test. Zero temporal overlap between splits. Preprocessors and standardizers must be fitted strictly on the training partition.
6. **Forecast Windowing**:
   - Forecast Origin: Observation timestamp $t$.
   - History Window: Past 12 five-minute observations ($t-11, t-10, \dots, t$), corresponding to 60 minutes of historical lags.
   - Prediction Horizons: +15 minutes ($h=3$ steps), +30 minutes ($h=6$ steps), and +60 minutes ($h=12$ steps).
7. **Common Evaluation Mask**: All models (Persistence, Ridge, XGBoost, etc.) must be evaluated on the exact same set of valid target timestamps and sensors to ensure fair, paired comparison.
8. **Evaluation Metrics**:
   - Primary: Mean Absolute Error (MAE) in mph.
   - Secondary: Root Mean Squared Error (RMSE) in mph.
   - Tertiary: Mean Absolute Percentage Error (MAPE) with documented $\epsilon$-policy ($\epsilon = 2.0$ mph or strictly masked positive threshold) to prevent division by zero or distortion on low-speed jams.

---

## 2. Four-Person Workstream Ownership Matrix

| Workstream | Owner | Primary Responsibilities | Core Deliverables & Files |
| :--- | :--- | :--- | :--- |
| **WS1: Data & Provenance** | Member 1 | Data ingestion, lineage manifest, missingness audit, sensor alignment, exploratory data analysis (EDA), literature review. | `src/data/download.py`, `src/data/audit.py`, `notebooks/01_eda.py`, `data_audit.json`, figures in `reports/figures/eda/`. IEEE Data Section. |
| **WS2: Feature Engineering & Baselines** | Member 2 | Shared window builder, causal lag extraction, split validation, persistence benchmark, Ridge regression baseline, unit/leakage tests. | `src/features/build_features.py`, `src/models/persistence.py`, `src/models/ridge_baseline.py`, `tests/test_splits_and_leakage.py`. IEEE Methodology Section. |
| **WS3: Non-Linear ML & Evaluation** | Member 3 | XGBoost multi-horizon models, hyperparameter tuning via validation set, spatial lag ablation, comprehensive metric computation. | `src/models/train_xgboost.py`, `src/evaluation/metrics.py`, `src/evaluation/ablation.py`, `metrics_by_horizon.csv`, `metrics_by_sensor.csv`. IEEE Results Section. |
| **WS4: Systems, Dashboard & Presentation** | Member 4 | Streamlit dashboard, automated artifact pipeline, CI workflow, IEEE LaTeX paper compilation, PPTX presentation deck, VIVA guide. | `app/streamlit_app.py`, `.github/workflows/ci.yml`, `reports/paper/`, `presentation/`, `VIVA_GUIDE.md`, `README.md`. IEEE Conclusion/Integration. |

---

## 3. Execution Checkpoints

### Checkpoint A: Repository Architecture & Dataset Audit
- Set up directory scaffold: `src/`, `data/`, `notebooks/`, `tests/`, `reports/`, `presentation/`, `app/`.
- Create configuration management with deterministic random seed (`seed = 42`).
- Establish `data_lineage.json` tracking sources, URLs, SHA-256 checksums, and timestamps.
- Download METR-LA speed matrix and sensor graph metadata (`graph_sensor_ids.txt`, `distances_la_2012.csv`).
- Execute sensor alignment check and missingness audit (`data_audit.json`). Flag any discrepancies.

### Checkpoint B: Causal Windowing & Scientific Validation
- Implement unified `build_examples(...)` engine.
- Generate historical lags $t-11 \dots t$ and multi-step targets $t+3, t+6, t+12$.
- Construct unit test suite verifying:
  - Causal boundaries (no future leakage).
  - Exact target offsets (+15, +30, +60 minutes).
  - Training-only scaler fit (no contamination).
  - Graph node ordering consistency with speed matrix columns.

### Checkpoint C: Exploratory Data Analysis & Pilot Baselines
- Generate reproducible EDA figures: missingness heatmap, 24-hour diurnal patterns, weekday vs. weekend profiles, sensor spatial graph degree distribution, and autocorrelation function (ACF).
- Define representative sensor pilot (15 sensors spanning high-variance corridors, ramps, and stable segments) for swift iteration and hyperparameter validation.
- Train Historical Persistence Baseline, Ridge Regression, and XGBoost regressor per horizon.
- Record validation metrics, training durations, and model artifacts.

### Checkpoint D: Full Scope Evaluation & Scientific Ablation
- Execute frozen test evaluation across all eligible targets.
- Save structured test predictions: `sensor_id`, `origin_ts`, `target_ts`, `horizon_steps`, `y_true`, `y_pred`, `valid_flag`, `model`.
- Generate aggregated CSVs: `metrics_by_horizon.csv`, `metrics_by_sensor.csv`, `metrics_peak_offpeak.csv`.
- Execute Spatial Lag Feature Ablation (incorporating 1st-order graph neighbor historical lags vs. sensor-isolated lags).
- Produce evaluation charts: Horizon MAE/RMSE comparisons, error distribution, prediction traces over morning/evening peak rush hours, and residual analyses.

### Checkpoint E: Interactive Streamlit Evaluation Dashboard
- Build multi-page Streamlit application:
  1. *Executive Overview*: Project scope, Phase 1 vs. Phase 2 distinction, dataset summary.
  2. *Dataset & Missingness Explorer*: Raw time-series viewer, sensor selection, missing data distribution.
  3. *Forecast Lab*: Interactive origin time selector on held-out test data, ground-truth vs. model forecasts at 15m, 30m, 60m, with confidence/error intervals.
  4. *Benchmark & Error Analysis*: Interactive model comparisons, peak vs. off-peak performance breakdown, residual plots.
  5. *Sensor Network & Phase 2 Outlook*: Sensor topology view, distance distribution, GAT/GCN research plan for Phase 2.
- Bundle lightweight precomputed test artifacts so the application starts instantly and operates offline without retraining.

### Checkpoint F: Academic Artifacts, Documentation & Presentation
- Author complete IEEE-format conference paper (`reports/paper/main.tex`, `references.bib`) featuring genuine experimental numbers, figures, and academic citations.
- Generate professional 15-slide PowerPoint deck (`presentation/AML_Phase1_Traffic_Prediction.pptx`) with detailed speaker notes.
- Compile comprehensive viva preparation guide (`VIVA_GUIDE.md`) detailing theory, equations, failure modes, and examiner Q&A.
- Provide production-grade `README.md`, `TASK_BOARD.md`, and `CONTRIBUTIONS.md`.
