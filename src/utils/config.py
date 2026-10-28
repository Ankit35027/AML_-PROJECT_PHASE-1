"""
Centralized Configuration and Path Management for METR-LA Classical ML Pipeline.
Ensures deterministic random seeds, consistent chronological splits, and multi-horizon settings.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SENSOR_GRAPH_DIR = DATA_DIR / "sensor_graph"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EDA_FIGURES_DIR = FIGURES_DIR / "eda"
EVAL_FIGURES_DIR = FIGURES_DIR / "evaluation"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EDA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
EVAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dataset details
H5_FILENAME = RAW_DATA_DIR / "metr-la.h5"
SENSOR_IDS_FILE = SENSOR_GRAPH_DIR / "graph_sensor_ids.txt"
DISTANCES_FILE = SENSOR_GRAPH_DIR / "distances_la_2012.csv"
LOCATIONS_FILE = SENSOR_GRAPH_DIR / "graph_sensor_locations.csv"
ADJ_MX_FILE = SENSOR_GRAPH_DIR / "adj_mx.pkl"

# Output Artefacts
DATA_AUDIT_FILE = PROJECT_ROOT / "data_audit.json"
DATA_LINEAGE_FILE = PROJECT_ROOT / "data_lineage.json"
DATA_SUMMARY_FILE = PROJECT_ROOT / "data_summary.json"
MODEL_MANIFEST_FILE = PROJECT_ROOT / "model_manifest.json"

# Splitting Configuration (Chronological 70/10/20)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# Windowing Configuration
HISTORY_STEPS = 12  # past 12 observations (60 minutes of history: t-11 to t)
SAMPLING_INTERVAL_MINUTES = 5

# Forecasting Horizons
HORIZONS = {
    "15min": 3,   # +3 steps (15 min)
    "30min": 6,   # +6 steps (30 min)
    "60min": 12,  # +12 steps (60 min)
}

# Metric Policy
EPSILON_MAPE = 2.0  # Epsilon in mph to avoid division by zero on near-zero/jam speeds
MAX_REASONABLE_SPEED = 100.0  # Speeds > 100 mph flagged as physical anomalies

# Deterministic Seed
RANDOM_SEED = 42

# Pilot Sensor Subset (15 sensors covering corridors, interchanges, ramps, high & low variance)
# Selected systematically across LA highway network
PILOT_SENSORS: List[str] = [
    "773869", "767541", "717447", "717816", "765604",
    "716339", "765273", "769402", "767572", "764424",
    "718204", "768066", "767609", "717490", "718066"
]
