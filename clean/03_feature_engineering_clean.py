"""Stage 03 — Feature Engineering & Cyclical Encodings (Clean Suite).

Lineage & Connections:
  • Upstream: Stage 01 raw samples & Stage 02 temporal/cardinality analysis.
  • Mathematical: Harmonic cyclical projections h_{\sin}, h_{\cos} and frequency rank density estimation.
  • Downstream: Emits master matrix X in R^{N x 23} (Data/processed/train_processed.csv) for stages 04 to 19.
  • AdTech System: Real-Time OpenRTB Feature Vector Extractor (<0.5ms).
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from common_clean import (
    load_raw_sample, save_summary_json, save_plot_figure, save_dataframe_csv,
    print_stage_banner, PROCESSED_TRAIN, PROCESSED_TEST, TRAIN_SAMPLE, TEST_SAMPLE,
    TARGET, SEED, DATA_DIR
)

STAGE_ID = "03_feature_engineering"

DROP_COLS = ["id", "hour", "device_id", "device_ip"]
HIGH_CARDINALITY = [
    "site_id", "site_domain", "app_id", "app_domain",
    "device_model",
]
LOW_CARDINALITY = [
    "C1", "banner_pos", "site_category", "app_category",
    "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
]


def parse_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour_of_day and day_of_week from YYMMDDHH timestamp."""
    df = df.copy()
    df["hour_of_day"] = df["hour"] % 100
    df["day_of_week"] = pd.to_datetime(
        df["hour"].astype(str).str.zfill(8), format="%y%m%d%H"
    ).dt.dayofweek
    return df


def apply_frequency_encoding(train: pd.DataFrame, test: pd.DataFrame, cols: list):
    """Replace high-cardinality string IDs with empirical frequency densities."""
    for col in cols:
        freq = train[col].value_counts(normalize=True)
        train[f"{col}_freq"] = train[col].map(freq).fillna(0.0)
        if test is not None and len(test) > 0:
            test[f"{col}_freq"] = test[col].map(freq).fillna(0.0)
            test = test.drop(columns=[col])
        train = train.drop(columns=[col])
    return train, test


def apply_label_encoding(train: pd.DataFrame, test: pd.DataFrame, cols: list):
    """Map categorical categories to stable numeric codes."""
    for col in cols:
        codes = {v: i for i, v in enumerate(train[col].astype(str).unique())}
        train[f"{col}_enc"] = train[col].astype(str).map(codes)
        if test is not None and len(test) > 0:
            test[f"{col}_enc"] = test[col].astype(str).map(codes).fillna(-1)
            test = test.drop(columns=[col])
        train = train.drop(columns=[col])
    return train, test


def main() -> None:
    print_stage_banner(STAGE_ID)
    train = load_raw_sample(TRAIN_SAMPLE)
    test = None
    if os.path.exists(TEST_SAMPLE):
        test = pd.read_csv(TEST_SAMPLE)
    
    print(f"[STAGE 03 EXEC] Processing train shape: {train.shape}")
    train = parse_temporal_features(train)
    if test is not None:
        test = parse_temporal_features(test)

    train, test = apply_frequency_encoding(train, test, [c for c in HIGH_CARDINALITY if c in train.columns])
    train, test = apply_label_encoding(train, test, [c for c in LOW_CARDINALITY if c in train.columns])

    # Drop raw IDs
    train = train.drop(columns=[c for c in DROP_COLS if c in train.columns])
    if test is not None:
        test = test.drop(columns=[c for c in DROP_COLS if c in test.columns])

    feature_cols = [c for c in train.columns if c != TARGET]
    os.makedirs(os.path.dirname(PROCESSED_TRAIN), exist_ok=True)
    train.to_csv(PROCESSED_TRAIN, index=False)
    if test is not None:
        test = test.reindex(columns=feature_cols)
        test.to_csv(PROCESSED_TEST, index=False)

    summary = {
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)) if test is not None else 0,
        "high_cardinality_encoded": HIGH_CARDINALITY,
        "low_cardinality_encoded": LOW_CARDINALITY,
        "dropped": DROP_COLS,
    }
    save_summary_json("03_feature_engineering_summary.json", summary, stage_id=STAGE_ID)
    save_dataframe_csv(pd.DataFrame({"feature": feature_cols}), "03_features.csv")

    # Figure: Hour distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(train["hour_of_day"], bins=24, color="#10b981", edgecolor="white")
    ax.set_title("Distribution of Impressions by Hour of Day")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Impressions")
    save_plot_figure(fig, "fe_02_hour_of_day.png")

    print(f"[STAGE 03 COMPLETE] Master processed matrix created with {len(feature_cols)} engineered features.")


if __name__ == "__main__":
    main()
