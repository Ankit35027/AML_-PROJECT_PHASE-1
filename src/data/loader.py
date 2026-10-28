"""
Data loader module for METR-LA speed matrix and sensor graph metadata.
Reads HDF5 file directly via h5py without requiring pytables C-extension.
"""

import pickle
import hashlib
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
import h5py

from src.utils.config import (
    H5_FILENAME,
    SENSOR_IDS_FILE,
    DISTANCES_FILE,
    LOCATIONS_FILE,
    ADJ_MX_FILE,
)


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_metr_la_raw(h5_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw METR-LA traffic speed dataset.
    Returns:
        pd.DataFrame of shape (34272, 207) with DatetimeIndex and sensor ID strings as columns.
    """
    path = h5_path or H5_FILENAME
    if not Path(path).exists():
        raise FileNotFoundError(f"Raw METR-LA dataset not found at {path}. Please run download script.")

    with h5py.File(path, "r") as f:
        # Extract sensor IDs (axis0)
        sensor_ids_raw = f["df/axis0"][:]
        sensor_ids = [
            s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
            for s in sensor_ids_raw
        ]

        # Extract timestamps (axis1 - nanoseconds since epoch)
        ts_raw = f["df/axis1"][:]
        timestamps = pd.to_datetime(ts_raw)

        # Extract speed values (block0_values)
        values = f["df/block0_values"][:]

    df = pd.DataFrame(values, index=timestamps, columns=sensor_ids)
    df.index.name = "timestamp"
    return df


def load_sensor_graph(
    sensor_ids_path: Optional[Path] = None,
    distances_path: Optional[Path] = None,
    locations_path: Optional[Path] = None,
    adj_mx_path: Optional[Path] = None,
) -> Dict:
    """
    Load sensor graph metadata: sensor list, pairwise distances, locations, and adjacency matrix.
    """
    s_path = sensor_ids_path or SENSOR_IDS_FILE
    d_path = distances_path or DISTANCES_FILE
    l_path = locations_path or LOCATIONS_FILE
    a_path = adj_mx_path or ADJ_MX_FILE

    with open(s_path, "r") as f:
        sensor_ids = [line.strip() for line in f.read().split(",") if line.strip()]

    distances_df = pd.read_csv(d_path)
    locations_df = pd.read_csv(l_path)

    adj_mx = None
    if Path(a_path).exists():
        with open(a_path, "rb") as f:
            adj_data = pickle.load(f, encoding="latin1")
            # DCRNN adj_mx format: [sensor_ids, sensor_id_to_ind, adj_mx]
            if isinstance(adj_data, (list, tuple)) and len(adj_data) == 3:
                adj_mx = adj_data[2]
            else:
                adj_mx = adj_data

    return {
        "sensor_ids": sensor_ids,
        "distances": distances_df,
        "locations": locations_df,
        "adj_mx": adj_mx,
    }
