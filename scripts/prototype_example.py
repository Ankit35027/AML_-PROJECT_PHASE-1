"""
Prototype Single-Sensor Traceability Verification.
Generates an explicit example record for sensor '773869' displaying:
- Forecast origin timestamp (t)
- All 12 historical lags (t-11 through t)
- Multi-horizon target timestamps and values (+15m, +30m, +60m)
- Verification that targets correspond to exact forward chronological measurements.
"""

import json
from pathlib import Path
import pandas as pd
from src.data.loader import load_metr_la_raw
from src.features.build_features import build_sensor_examples

def run_prototype():
    df = load_metr_la_raw()
    sensor_id = "773869"
    ex = build_sensor_examples(df, sensor_id, "train")
    
    # Pick sample index 100
    idx = 100
    origin_ts = ex["origin_timestamps"][idx]
    
    lags = {}
    for i in range(12):
        lag_ts = origin_ts - pd.Timedelta(minutes=5 * (11 - i))
        lags[f"t-{11 - i} ({lag_ts.strftime('%Y-%m-%d %H:%M')})"] = float(ex["X"][idx, i])
        
    targets = {
        "15min (+3 steps, " + ex["target_timestamps"]["15min"][idx].strftime('%Y-%m-%d %H:%M') + ")": {
            "speed_mph": float(ex["Y"][idx, 0]),
            "valid": bool(ex["Y_valid"][idx, 0])
        },
        "30min (+6 steps, " + ex["target_timestamps"]["30min"][idx].strftime('%Y-%m-%d %H:%M') + ")": {
            "speed_mph": float(ex["Y"][idx, 1]),
            "valid": bool(ex["Y_valid"][idx, 1])
        },
        "60min (+12 steps, " + ex["target_timestamps"]["60min"][idx].strftime('%Y-%m-%d %H:%M') + ")": {
            "speed_mph": float(ex["Y"][idx, 2]),
            "valid": bool(ex["Y_valid"][idx, 2])
        }
    }
    
    result = {
        "sensor_id": sensor_id,
        "sample_index": idx,
        "origin_timestamp_t": str(origin_ts),
        "historical_speed_lags": lags,
        "targets": targets
    }
    
    out_path = Path("reports/prototype_traceability.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f" Saved prototype example to {out_path}")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    run_prototype()
