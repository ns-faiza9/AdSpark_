"""AdSpark Clean ML Pipeline — Shared Configuration, Loaders, and Helpers.

Provides robust dataset loading, artifact persistence, plot styling, and
explicit pipeline lineage connection printers for clean experiments (01 through 19).
"""
import json
import os
import shutil
import sys
from typing import Dict, Any, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")  # headless-safe figure rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLEAN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CLEAN_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if CLEAN_DIR not in sys.path:
    sys.path.insert(0, CLEAN_DIR)

from pipeline_connections import get_stage_connection, print_stage_lineage, PIPELINE_CONNECTIONS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(ROOT_DIR, "Data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(ROOT_DIR, "analysis", "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# Mirror paths for React/Vite web server
WEBPAGE_DIR = os.path.join(ROOT_DIR, "webpage")
PUBLIC_DIR = os.path.join(WEBPAGE_DIR, "public")
PUBLIC_FIGURES_DIR = os.path.join(PUBLIC_DIR, "figures")
PUBLIC_DATA_DIR = os.path.join(PUBLIC_DIR, "data")
DIST_DIR = os.path.join(WEBPAGE_DIR, "dist")
DIST_FIGURES_DIR = os.path.join(DIST_DIR, "figures")
DIST_DATA_DIR = os.path.join(DIST_DIR, "data")

# Dataset files
TRAIN_SAMPLE = os.path.join(DATA_DIR, "train_sample.csv")
TEST_SAMPLE = os.path.join(DATA_DIR, "test_sample.csv")
PROCESSED_TRAIN = os.path.join(PROCESSED_DIR, "train_processed.csv")
PROCESSED_TEST = os.path.join(PROCESSED_DIR, "test_processed.csv")

# Constants
SEED = 42
TARGET = "click"


def ensure_dirs() -> None:
    """Ensure all required output and public mirroring directories exist."""
    for d in (DATA_DIR, PROCESSED_DIR, OUTPUT_DIR, FIGURES_DIR,
              PUBLIC_FIGURES_DIR, PUBLIC_DATA_DIR,
              DIST_FIGURES_DIR, DIST_DATA_DIR):
        os.makedirs(d, exist_ok=True)


def generate_synthetic_avazu_sample(n_rows: int = 20000, seed: int = SEED) -> pd.DataFrame:
    """Generate a realistic synthetic Avazu dataset matching the exact 24-column schema."""
    np.random.seed(seed)
    hours = np.random.choice([14102100 + h for h in range(24)], size=n_rows)
    # Target with exact 16.94% empirical CTR baseline
    clicks = np.random.binomial(n=1, p=0.16938, size=n_rows)
    
    data = {
        "id": [f"100{i:07d}" for i in range(n_rows)],
        "click": clicks,
        "hour": hours,
        "C1": np.random.choice([1005, 1002, 1001, 1010], size=n_rows),
        "banner_pos": np.random.choice([0, 1, 2, 7], p=[0.72, 0.27, 0.008, 0.002], size=n_rows),
        "site_id": np.random.choice([f"site_{i}" for i in range(150)], size=n_rows),
        "site_domain": np.random.choice([f"sdom_{i}" for i in range(100)], size=n_rows),
        "site_category": np.random.choice(["50e271e0", "28905ebd", "3e814132", "f028772b"], size=n_rows),
        "app_id": np.random.choice([f"app_{i}" for i in range(120)], size=n_rows),
        "app_domain": np.random.choice([f"adom_{i}" for i in range(80)], size=n_rows),
        "app_category": np.random.choice(["07d7df22", "0f2161f8", "cef3e649"], size=n_rows),
        "device_id": np.random.choice([f"dev_{i}" for i in range(500)], size=n_rows),
        "device_ip": np.random.choice([f"ip_{i}" for i in range(1000)], size=n_rows),
        "device_model": np.random.choice([f"model_{i}" for i in range(250)], size=n_rows),
        "device_type": np.random.choice([1, 0, 4, 5], p=[0.92, 0.05, 0.02, 0.01], size=n_rows),
        "device_conn_type": np.random.choice([0, 2, 3, 5], p=[0.85, 0.08, 0.05, 0.02], size=n_rows),
        "C14": np.random.choice([15706, 15707, 15708, 20352, 21647], size=n_rows),
        "C15": np.random.choice([320, 300, 216], p=[0.93, 0.05, 0.02], size=n_rows),
        "C16": np.random.choice([50, 250, 36], p=[0.93, 0.05, 0.02], size=n_rows),
        "C17": np.random.choice([1722, 2161, 2333, 2480], size=n_rows),
        "C18": np.random.choice([0, 1, 2, 3], p=[0.42, 0.18, 0.15, 0.25], size=n_rows),
        "C19": np.random.choice([35, 39, 163, 167], size=n_rows),
        "C20": np.random.choice([-1, 100084, 100148, 100075], size=n_rows),
        "C21": np.random.choice([79, 157, 110, 48], size=n_rows),
    }
    df = pd.DataFrame(data)
    return df


def load_raw_sample(path: str = TRAIN_SAMPLE) -> pd.DataFrame:
    """Load the raw sampled Avazu dataset or generate realistic synthetic data if missing."""
    ensure_dirs()
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"[NOTE] Raw sample not found at {path}. Generating realistic synthetic Avazu dataset.")
    df = generate_synthetic_avazu_sample(n_rows=20000, seed=SEED)
    df.to_csv(path, index=False)
    print(f"[CREATED] {path} ({len(df):,} rows x {df.shape[1]} columns)")
    return df


def load_processed_data(path: str = PROCESSED_TRAIN) -> pd.DataFrame:
    """Load the engineered feature dataset or run feature pipeline on sample."""
    ensure_dirs()
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"[NOTE] Processed dataset not found at {path}. Generating via feature pipeline.")
    raw_df = load_raw_sample()
    from clean.import_helper import build_processed_dataset
    processed = build_processed_dataset(raw_df)
    processed.to_csv(path, index=False)
    print(f"[CREATED] {path} ({len(processed):,} rows x {processed.shape[1]} columns)")
    return processed


def save_summary_json(name: str, data: Dict[str, Any], stage_id: str = None) -> str:
    """Save a summary JSON into analysis/output/ and mirror to React public & dist."""
    ensure_dirs()
    if stage_id and stage_id in PIPELINE_CONNECTIONS:
        conn = PIPELINE_CONNECTIONS[stage_id]
        data["_pipeline_lineage"] = {
            "stage_id": conn["stage_id"],
            "stage_number": conn["stage_number"],
            "title": conn["title"],
            "upstream": conn["upstream_antecedents"],
            "mathematical_transition": conn["mathematical_algorithmic_transition"],
            "downstream": conn["downstream_dependents"],
            "adtech_role": conn["adtech_production_systemic_role"]
        }
    
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    
    for dest in (PUBLIC_DATA_DIR, DIST_DATA_DIR):
        try:
            shutil.copy2(path, os.path.join(dest, name))
        except Exception:
            pass
    print(f"[SAVED JSON] -> {path}")
    return path


def save_plot_figure(fig, name: str) -> str:
    """Save a matplotlib figure and mirror it to the React web app directories."""
    ensure_dirs()
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    for dest in (PUBLIC_FIGURES_DIR, DIST_FIGURES_DIR):
        try:
            shutil.copy2(path, os.path.join(dest, name))
        except Exception:
            pass
    print(f"[SAVED FIGURE] -> {path}")
    return path


def save_dataframe_csv(df: pd.DataFrame, name: str) -> str:
    """Save a tabular dataframe to CSV in the output directory."""
    ensure_dirs()
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"[SAVED CSV] -> {path}")
    return path


def print_stage_banner(stage_id: str) -> None:
    """Print standard stage header and lineage connections."""
    print_stage_lineage(stage_id)
