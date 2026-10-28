"""
Automated Generation of Professional 15-Slide Presentation Deck.
Uses python-pptx to generate presentation/AML_Phase1_Traffic_Prediction.pptx
with genuine empirical metrics, embedded figures, and detailed speaker notes.
"""

import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "presentation/AML_Phase1_Traffic_Prediction.pptx"
FIG_DIR = REPO_ROOT / "reports/figures"

# Theme Colors
NAVY = RGBColor(30, 58, 138)       # #1E3A8A Primary Title
DARK_GRAY = RGBColor(55, 65, 81)   # #374151 Body text
LIGHT_BG = RGBColor(248, 250, 252) # #F8FAFC
ACCENT_BLUE = RGBColor(37, 99, 235)# #2563EB
ACCENT_GREEN = RGBColor(22, 163, 74)# #16A34A


def create_deck():
    prs = Presentation()
    # Set slide dimensions to widescreen (16:9: 13.33 x 7.5 inches)
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]  # blank layout

    def add_slide_header(slide, title_text, category_text="PHASE 1 MACHINE LEARNING BENCHMARK"):
        # Top banner category
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_BLUE

        # Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = NAVY

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    tb = s1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.3), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "HISTORICAL METR-LA TRAFFIC SPEED FORECASTING"
    p0.font.size = Pt(28)
    p0.font.bold = True
    p0.font.color.rgb = NAVY
    p0.space_after = Pt(10)

    p1 = tf.add_paragraph()
    p1.text = "Phase 1 Classical ML Baselines, Multi-Horizon Evaluation & Causal Benchmarking"
    p1.font.size = Pt(18)
    p1.font.color.rgb = ACCENT_BLUE
    p1.space_after = Pt(20)

    p2 = tf.add_paragraph()
    p2.text = "Course: Traffic Flow Prediction and Signal Control via Graph Attention Networks and Flow Models"
    p2.font.size = Pt(13)
    p2.font.color.rgb = DARK_GRAY
    p2.space_after = Pt(20)

    p3 = tf.add_paragraph()
    p3.text = "Team Members & Ownership:\n• Member 1: Lead, Data & Provenance (WS1)\n• Member 2: Lead, Features & Baselines (WS2)\n• Member 3: Lead, Non-Linear Models & Evaluation (WS3)\n• Member 4: Lead, Systems, Dashboard & Integration (WS4)"
    p3.font.size = Pt(12)
    p3.font.color.rgb = DARK_GRAY

    s1.notes_slide.notes_text_frame.text = (
        "Speaker Notes (Member 1 Opening):\n"
        "Welcome professors and examiners. Today our team presents Phase 1 of our machine learning project on "
        "Historical METR-LA Traffic Speed Forecasting. Our course mandate targets Graph Attention Networks, but as "
        "rigorous engineers, Phase 1 establishes the reproducible classical machine learning benchmark against which "
        "all subsequent deep models will be measured. Let us outline our four-person division of work and begin with our scientific guardrails."
    )

    # =========================================================================
    # SLIDE 2: PROBLEM DEFINITION & PHASE SCOPE
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_slide_header(s2, "Problem Formulation & Scientific Boundaries", "Research Scope")
    tb = s2.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    items = [
        ("The Forecasting Challenge:", "Predict traffic speed (mph) on Los Angeles highway corridors across multiple lookahead horizons (+15, +30, and +60 minutes) given past 60 minutes of sensor readings."),
        ("Strict Phase 1 Scope:", "Classical machine learning algorithms (Historical Persistence, L2-regularized Ridge Regression, and Gradient Boosted Trees XGBoost)."),
        ("Dataset Ground Truth:", "The benchmark METR-LA dataset contains highway speed measurements from loop detectors. It does NOT contain vehicular volume/flow or traffic signal controller states."),
        ("Scientific Guardrails:", "We assert zero ungrounded claims of live traffic signal actuation or unmeasured vehicular flow. Phase 2 is strictly dedicated to Graph Attention Networks (GAT) and deep spatial-temporal architectures."),
        ("Acceptance Focus:", "Total reproducibility, zero synthetic numbers, verifiable chronological partitions, and an offline interactive demonstration.")
    ]
    for bold_prefix, text in items:
        p = tf.add_paragraph()
        p.text = f"• {bold_prefix} {text}"
        p.font.size = Pt(14)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(14)

    s2.notes_slide.notes_text_frame.text = (
        "Speaker Notes (Member 1):\n"
        "To ensure complete academic truthfulness, we clarify our scope: the METR-LA dataset contains loop detector speeds in mph. "
        "We do not claim live traffic signal actuation or vehicular flow prediction. Phase 1 delivers the classical machine learning baselines."
    )

    # =========================================================================
    # SLIDE 3: METR-LA DATASET & PROVENANCE AUDIT
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_slide_header(s3, "METR-LA Dataset & Cryptographic Lineage", "Data Provenance (WS1)")
    tb = s3.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(6.0), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    lines = [
        "Benchmark Origin: Caltrans PeMS loop detectors (Los Angeles County freeways).",
        "Canonical Source: DCRNN paper (Li et al., ICLR 2018).",
        "Sensor Stations: 207 loop detectors spanning I-5, I-10, I-405, SR-110, US-101.",
        "Observation Period: March 1, 2012 to June 27, 2012 (4 continuous months).",
        "Temporal Resolution: 5-minute aggregations -> 34,272 continuous timestamps.",
        "Total Observations: 7,094,304 speed measurements.",
        "File Integrity: SHA-256: 64784b...c05 (tracked in data_lineage.json)."
    ]
    for line in lines:
        p = tf.add_paragraph()
        p.text = f"• {line}"
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(10)

    # Embed speed distribution figure
    fig_path = FIG_DIR / "eda/speed_distribution.png"
    if fig_path.exists():
        s3.shapes.add_picture(str(fig_path), Inches(7.0), Inches(1.8), width=Inches(5.5))

    s3.notes_slide.notes_text_frame.text = (
        "Speaker Notes (Member 1):\n"
        "Here we show the verified dataset characteristics. All 7 million observations stem from 207 Caltrans sensors over 4 months. "
        "Notice the empirical speed distribution on the right: valid speeds average 58.47 mph with a median of 63 mph."
    )

    # =========================================================================
    # SLIDE 4: MISSINGNESS PROTOCOL & LABEL INTEGRITY
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_slide_header(s4, "Missingness Accounting & Evaluation Label Policy", "Data Quality (WS1)")
    tb = s4.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(6.0), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    lines = [
        "NaN / Null Readings: Exactly 0 entries (null values stored as 0.0 in benchmark).",
        "Zero-Speed Readings (0.0 mph): 575,302 cells (8.11% of total matrix).",
        "Physical Anomalies (>100 mph): Exactly 0 entries.",
        "Interpretation of 0.0 mph: Denotes hardware dropout, lane closure, or standstill congestion.",
        "Evaluation Label Policy: Ground truth target evaluation strictly requires >0.0 mph valid measurements (avoids division by zero in MAPE).",
        "Feature Imputation Policy: Missing lags in historical window use causal forward-fill (strictly past observations; zero future leakage)."
    ]
    for line in lines:
        p = tf.add_paragraph()
        p.text = f"• {line}"
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(10)

    fig_path = FIG_DIR / "eda/sensor_zero_distribution.png"
    if fig_path.exists():
        s4.shapes.add_picture(str(fig_path), Inches(7.0), Inches(1.8), width=Inches(5.5))

    s4.notes_slide.notes_text_frame.text = (
        "Speaker Notes (Member 1 handoff to Member 2):\n"
        "A critical contribution of WS1 was our missingness protocol: 8.11% of readings are 0.0. "
        "We avoid naive zero-to-NaN conversion, enforcing valid labels for evaluation while using causal forward-fill "
        "for historical features. Now, Member 2 will discuss feature windowing and split integrity."
    )

    # =========================================================================
    # SLIDE 5: CHRONOLOGICAL PARTITIONS & CAUSAL WINDOWING
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_slide_header(s5, "Chronological Partitioning & Causal Windowing", "Feature Engineering (WS2)")
    tb = s5.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    lines = [
        ("Strict Chronological Partitioning:", "70% Train (23,990 steps, Mar-May) | 10% Val (3,427 steps, May-Jun) | 20% Test (6,855 steps, Jun). Zero temporal shuffle leakage."),
        ("Preprocessing Isolation:", "StandardScaler parameters (mean, scale) fitted strictly on the Training partition; applied out-of-sample to Val and Test."),
        ("Forecast Origin t:", "Observation timestamp defining the present moment."),
        ("Historical Window:", "12 past observations (t-11, t-10, ..., t) = 60 minutes of elapsed history."),
        ("Multi-Horizon Targets:", "Direct step offsets: +15 min (h=3), +30 min (h=6), +60 min (h=12)."),
        ("Continuous Calendar Features:", "Sinusoidal and cosine projections of Time-of-Day (TOD) and Day-of-Week (DOW).")
    ]
    for bold_prefix, text in lines:
        p = tf.add_paragraph()
        p.text = f"• {bold_prefix} {text}"
        p.font.size = Pt(14)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(12)

    s5.notes_slide.notes_text_frame.text = (
        "Speaker Notes (Member 2):\n"
        "Thank you Member 1. In WS2, we built our causal windowing engine. "
        "We strictly prohibit random train/test splits. Instead, we enforce a 70/10/20 chronological partition. "
        "For any origin t, our models only receive observations up to t (60 min history) and predict +15, +30, and +60 min ahead."
    )

    # =========================================================================
    # SLIDE 6: SCIENTIFIC VALIDATION & PYTEST SUITE
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_slide_header(s6, "Scientific Verification & Automated Test Suite", "QA & Validation (WS2)")
    tb = s6.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    lines = [
        ("test_split_indices_proportions_and_continuity:", "Asserts continuous chronological boundaries (0 -> 23,990 -> 27,417 -> 34,272) with exact 70/10/20 ratios."),
        ("test_exact_target_offsets:", "Asserts target timestamps equal origin +15m, +30m, and +60m exactly (timedelta precision)."),
        ("test_causal_windowing_and_no_future_leakage:", "Verifies feature column 11 exactly equals speed at origin t, and column 0 equals speed at t-55m."),
        ("test_split_boundary_isolation_no_target_overflow:", "Guarantees no origin or forward target crosses train/val or val/test boundaries."),
        ("test_scaler_fit_strictly_on_train_only:", "Proves that standard scaler mean and variance remain frozen when applied to test data."),
        ("test_manual_metrics_recomputation:", "Hand-calculated MAE, RMSE, and epsilon-MAPE on 5 test records match python output to 1e-4 tolerance.")
    ]
    for test_name, desc in lines:
        p = tf.add_paragraph()
        p.text = f"✔ {test_name}: {desc}"
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(10)

    s6.notes_slide.notes_text_frame.text = (
        "Speaker Notes (Member 2):\n"
        "To satisfy the runbook's acceptance criteria, we authored an automated pytest suite. "
        "All 6 unit tests pass, verifying no target leakage, exact target offsets, and strict training-only scaler fitting. "
        "We even hand-verified metric calculations on illustrative records."
    )

    # =========================================================================
    # SLIDE 7: EDA - DIURNAL TRAFFIC DYNAMICS
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_slide_header(s7, "Diurnal Congestion Cycles & Diurnal Profiles", "Exploratory Analysis (WS1/WS2)")
    tb = s7.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.5), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    lines = [
        "Pronounced Bimodal Congestion: Network speeds drop dramatically twice each weekday.",
        "Morning Peak (07:00 - 09:30): Mean speed drops from 63 mph to ~48 mph.",
        "Evening Peak (16:00 - 19:00): Prolonged rush hour with high corridor variance.",
        "Weekday vs. Weekend Contrast: Weekends maintain steady free-flow speeds (60-64 mph) without severe rush-hour drops.",
        "Autocorrelation: ACF remains >0.85 at 15m and >0.70 at 60m, demonstrating strong short-term temporal predictability."
    ]
    for line in lines:
        p = tf.add_paragraph()
        p.text = f"• {line}"
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(12)

    fig_path = FIG_DIR / "eda/diurnal_cycle.png"
    if fig_path.exists():
        s7.shapes.add_picture(str(fig_path), Inches(6.6), Inches(1.8), width=Inches(5.9))

    s7.notes_slide.notes_text_frame.text = (
        "Speaker Notes (Member 2 handoff to Member 3):\n"
        "Here we observe the empirical diurnal dynamics of the LA freeway network. "
        "Notice the clear morning and evening dips on the chart. "
        "Now Member 3 will explain how we modeled these patterns with non-linear machine learning."
    )

    # =========================================================================
    # SLIDE 8: BASELINE MODEL FORMULATIONS
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_slide_header(s8, "Classical Baseline Formulations", "Model Architecture (WS3)")
    tb = s8.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    lines = [
        ("Historical Persistence (Last-Value Baseline):", "y_hat(t+h) = y(t). Assumes traffic speed remains unchanged over lookahead h. Primary classical lower-bound benchmark."),
        ("Multi-Horizon Ridge Regression:", "Direct regularized linear models minimizing squared error with L2 penalty: Loss = ||y - Xw||^2 + alpha ||w||^2. Hyperparameter alpha tuned strictly on chronological validation set (alpha=100.0 at 15m, 0.1 at 30m/60m)."),
        ("Multi-Horizon XGBoost Regressors:", "Gradient boosted decision tree ensembles minimizing regularized objective via second-order Taylor expansion: Loss = sum [ g_i f(x) + 0.5 h_i f^2(x) ] + Omega(f)."),
        ("XGBoost Configuration:", "250 estimators, max depth 6, learning rate 0.05, subsample 0.8, colsample 0.8, with validation early stopping (25 rounds)."),
        ("Pilot Sensor Scope:", "Trained on 15 representative highway sensors covering 359,505 training examples.")
    ]
    for bold_prefix, text in lines:
        p = tf.add_paragraph()
        p.text = f"• {bold_prefix} {text}"
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(12)

    s8.notes_slide.notes_text_frame.text = (
        "Speaker Notes (Member 3):\n"
        "Thank you Member 2. In WS3, we formulated three direct multi-horizon models. "
        "Persistence provides our benchmark. Ridge provides our linear baseline with tuned L2 penalty. "
        "XGBoost provides our non-linear gradient boosted tree baseline with early stopping on validation error."
    )

    # =========================================================================
    # SLIDE 9: MULTI-HORIZON TEST RESULTS
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_slide_header(s9, "Empirical Test Benchmark Results (90,149 Instances)", "Results & Evaluation (WS3)")
    tb = s9.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.8), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    lines = [
        ("15-Minute Horizon (+3 steps):", "XGBoost achieves 2.59 mph MAE (13.7% error reduction vs Persistence 3.00 mph). Ridge achieves 2.88 mph."),
        ("30-Minute Horizon (+6 steps):", "XGBoost achieves 3.17 mph MAE (12.4% error reduction vs Persistence 3.62 mph). Ridge achieves 3.61 mph."),
        ("60-Minute Horizon (+12 steps):", "XGBoost achieves 4.01 mph MAE (15.3% error reduction vs Persistence 4.73 mph). Ridge degrades to 4.91 mph."),
        ("Key Finding:", "Linear Ridge degrades faster than Persistence at 60m because distant linear projections suffer from collinearity, whereas XGBoost tree splits isolate non-linear traffic regimes.")
    ]
    for bold_prefix, text in lines:
        p = tf.add_paragraph()
        p.text = f"• {bold_prefix} {text}"
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(12)

    fig_path = FIG_DIR / "evaluation/multi_horizon_comparison.png"
    if fig_path.exists():
        s9.shapes.add_picture(str(fig_path), Inches(6.8), Inches(1.8), width=Inches(5.8))

    s9.notes_slide.notes_text_frame.text = (
        "Speaker Notes (Member 3):\n"
        "These are our frozen test evaluation numbers. Across 90,149 instances, XGBoost decisively leads at all horizons: "
        "2.59 mph at 15m, 3.17 mph at 30m, and 4.01 mph at 60m. "
        "Notice on the chart how error increases monotonically with lookahead duration, as expected in traffic physics."
    )

    # =========================================================================
    # SLIDE 10: 24-HOUR FORECAST TRACE ON TEST SET
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_slide_header(s10, "Test Set 24-Hour Forecast Trace (Sensor 773869)", "Qualitative Evaluation (WS3)")
    tb = s10.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.5), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    lines = [
        "Held-Out Test Date: Tuesday, June 5, 2012 (Held-out Test Split).",
        "Sensor: Station 773869 (I-5 freeway corridor).",
        "Ground Truth vs Predictions: 15-minute ahead speed forecasts.",
        "Morning Rush Hour Behavior: Ground truth drops abruptly at 07:30. Persistence lags the drop by 15 minutes.",
        "XGBoost Reactivity: Accurately anticipates the speed drop and tracks the sharp evening recovery without oscillations.",
        "Residuals: Errors are centered at 0.0 mph with narrow variance."
    ]
    for line in lines:
        p = tf.add_paragraph()
        p.text = f"• {line}"
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(12)

    fig_path = FIG_DIR / "evaluation/test_forecast_trace.png"
    if fig_path.exists():
        s10.shapes.add_picture(str(fig_path), Inches(6.6), Inches(1.8), width=Inches(5.9))

    s10.notes_slide.notes_text_frame.text = (
        "Speaker Notes (Member 3):\n"
        "This slide presents a continuous 24-hour trace from held-out test data on June 5, 2012. "
        "Notice how the black line, ground truth, plunges during morning rush hour. "
        "Persistence simply shifts the curve rightwards by 15 minutes, whereas XGBoost (green line) tracks the drop and recovery accurately."
    )

    # =========================================================================
    # SLIDE 11: PEAK VS OFF-PEAK STRATIFICATION
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_slide_header(s11, "Peak vs. Off-Peak Rush Hour Error Inflation", "Stratified Error Analysis (WS3)")
    tb = s11.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    lines = [
        ("Rush Hour Definition:", "Morning Peak (07:00-09:30) & Evening Peak (16:00-19:00) on weekdays only."),
        ("15-Minute Horizon Peak Inflation:", "XGBoost Peak MAE is 3.66 mph vs Off-Peak MAE of 2.29 mph (+59.8% inflation). Persistence Peak MAE is 4.24 mph vs 2.65 mph."),
        ("60-Minute Horizon Peak Inflation:", "XGBoost Peak MAE reaches 5.09 mph vs 3.70 mph off-peak. Ridge degrades to 6.47 mph during peaks."),
        ("Traffic Engineering Root Cause:", "During free-flow off-peak hours, traffic behaves almost deterministically (cars travel at speed limit ~65 mph). During peak hours, non-linear shockwaves propagate upstream when volume exceeds road capacity, multiplying forecast difficulty.")
    ]
    for bold_prefix, text in lines:
        p = tf.add_paragraph()
        p.text = f"• {bold_prefix} {text}"
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(12)

    s11.notes_slide.notes_text_frame.text = (
        "Speaker Notes (Member 3):\n"
        "A key scientific contribution is our stratified peak vs off-peak analysis. "
        "Off-peak traffic is largely free-flowing, yielding MAEs near 2.29 mph. "
        "During rush hour shockwaves, error nearly doubles across all models. This proves that rush hours require spatial context."
    )

    # =========================================================================
    # SLIDE 12: SPATIAL LAG FEATURE ABLATION
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_slide_header(s12, "Road Network Spatial Lag Feature Ablation", "Feature Ablation (WS3)")
    tb = s12.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    lines = [
        ("Hypothesis:", "Adding historical lags from 1st-order road network topological neighbors improves medium-term forecasting accuracy."),
        ("Feature Configurations:", "Isolated single-sensor lags (21 features) vs Augmented with top-3 nearest graph neighbors (30 features)."),
        ("15-Min Horizon Result:", "MAE drops from 2.595 mph to 2.546 mph (RMSE 4.88 -> 4.66 mph)."),
        ("30-Min Horizon Result:", "MAE drops from 3.174 mph to 3.073 mph (RMSE 6.07 -> 5.76 mph)."),
        ("60-Min Horizon Result:", "MAE drops from 4.021 mph to 3.811 mph (RMSE 7.54 -> 7.03 mph) - a 5.2% relative error reduction."),
        ("Significance:", "Provides empirical justification for Phase 2 Graph Neural Networks.")
    ]
    for bold_prefix, text in lines:
        p = tf.add_paragraph()
        p.text = f"• {bold_prefix} {text}"
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(10)

    fig_path = FIG_DIR / "evaluation/spatial_ablation_comparison.png"
    if fig_path.exists():
        s12.shapes.add_picture(str(fig_path), Inches(6.8), Inches(1.8), width=Inches(5.8))

    s12.notes_slide.notes_text_frame.text = (
        "Speaker Notes (Member 3 handoff to Member 4):\n"
        "In our ablation study, adding top-3 road network neighbors consistently reduced error across every single horizon. "
        "At 60 minutes, MAE dropped from 4.02 to 3.81 mph. This proves highway neighbors provide vital predictive signals. "
        "Now Member 4 will present our Streamlit system and transition to Phase 2."
    )

    # =========================================================================
    # SLIDE 13: STREAMLIT DASHBOARD SYSTEM
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_slide_header(s13, "Interactive Streamlit Evaluation Dashboard", "Software Engineering (WS4)")
    tb = s13.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    lines = [
        ("Multi-Page System Architecture:", "Built with Streamlit, organized into 5 dedicated analytical modules: Executive Overview, Dataset Explorer, Interactive Forecast Lab, Benchmark & Error Analysis, and Road Network Preview."),
        ("Zero Live-Retraining Latency:", "Operates completely offline using precomputed evaluation artifacts (test_predictions.parquet and lightweight history slices), launching in under 2 seconds."),
        ("Interactive Forecast Lab:", "Allows examiners to select any sensor station and held-out test timestamp to inspect historical 60-min speeds alongside ground truth vs model forecasts."),
        ("Geospatial Highway View:", "Interactive Caltrans sensor station map using exact latitude/longitude coordinates across Los Angeles County."),
        ("Reproducible Execution:", "Runs locally via 'streamlit run app/streamlit_app.py' with zero cloud dependency.")
    ]
    for bold_prefix, text in lines:
        p = tf.add_paragraph()
        p.text = f"• {bold_prefix} {text}"
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(12)

    s13.notes_slide.notes_text_frame.text = (
        "Speaker Notes (Member 4):\n"
        "Thank you Member 3. In WS4, we created an offline-first Streamlit application. "
        "It bundles lightweight precomputed test predictions, so examiners can inspect any held-out test scenario instantly. "
        "There is no fake live data and no retraining overhead."
    )

    # =========================================================================
    # SLIDE 14: PHASE 2 RESEARCH BLUEPRINT (GAT)
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_slide_header(s14, "Phase 2 Research Blueprint: Graph Attention Networks", "Research Roadmap (Phase 2)")
    tb = s14.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    lines = [
        ("Limitations of Classical Spatial ML:", "In classical ML, spatial features use fixed static weights (top-3 neighbors), unable to capture dynamic, time-varying shockwave propagation velocities."),
        ("Graph Attention Networks (GAT):", "Computes dynamic, data-driven attention coefficients alpha_ij(t) between adjacent sensors: alpha_ij = softmax( LeakyReLU( a^T [Wh_i || Wh_j] ) )."),
        ("Dynamic Receptive Field:", "Upstream sensors receive high attention during oncoming shockwaves, but low attention during free-flow uncongested states."),
        ("Spatio-Temporal Hybridization:", "Coupling GAT spatial attention with Temporal Convolutional Networks (TCN) or GRUs for joint spatio-temporal modeling."),
        ("Benchmark Target:", "Phase 2 deep architectures must surpass our Phase 1 classical baseline (XGBoost 2.59 mph at 15m, 4.01 mph at 60m).")
    ]
    for bold_prefix, text in lines:
        p = tf.add_paragraph()
        p.text = f"• {bold_prefix} {text}"
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(12)

    s14.notes_slide.notes_text_frame.text = (
        "Speaker Notes (Member 4):\n"
        "Here is our research bridge to Phase 2. While classical spatial concatenation helped, it is static. "
        "In Phase 2, we will implement Graph Attention Networks (GAT). "
        "Dynamic attention allows the model to shift focus between upstream and downstream sensors as shockwaves travel."
    )

    # =========================================================================
    # SLIDE 15: CONCLUSION & ACCEPTANCE SUMMARY
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    add_slide_header(s15, "Conclusion & Acceptance Verification", "Summary & Deliverables")
    tb = s15.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True

    lines = [
        ("Rigorous Provenance & Audit:", "Cryptographically verified METR-LA HDF5 dataset, explicit zero-speed missingness accounting, and 0 irregular intervals."),
        ("Causal Windowing & Isolation:", "Strict 70/10/20 chronological split, past-only feature extraction, and training-isolated standardizers passing all pytest checks."),
        ("Empirical Benchmarks:", "Established genuine test baselines: Persistence (3.00, 3.62, 4.73 mph), Ridge (2.88, 3.61, 4.91 mph), and XGBoost (2.59, 3.17, 4.01 mph)."),
        ("Spatial Ablation Verified:", "Road network neighbor lags prove spatial propagation of shockwaves, cutting 60m MAE to 3.81 mph."),
        ("Complete Deliverables Delivered:", "Streamlit App, IEEE LaTeX manuscript (main.tex), 15-slide presentation deck, VIVA guide, TASK_BOARD.md, and CONTRIBUTIONS.md."),
        ("Examiner Q&A Readiness:", "The team welcomes questions from the examining committee.")
    ]
    for bold_prefix, text in lines:
        p = tf.add_paragraph()
        p.text = f"✔ {bold_prefix} {text}"
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(10)

    s15.notes_slide.notes_text_frame.text = (
        "Speaker Notes (Member 4 Closing):\n"
        "In conclusion, our team has fulfilled all runbook criteria: true provenance, causal windowing, automated unit tests, "
        "empirical benchmark metrics, an interactive Streamlit app, and a complete IEEE conference manuscript. "
        "We are now prepared for examiner questions and defense."
    )

    prs.save(OUTPUT_PATH)
    print(f" Presentation deck successfully generated at {OUTPUT_PATH}")


if __name__ == "__main__":
    create_deck()
