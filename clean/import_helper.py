"""Helper module for feature processing transformations across clean suite."""
import pandas as pd

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


def build_processed_dataset(train: pd.DataFrame) -> pd.DataFrame:
    """Transform raw dataframe into engineered 23-feature dataset."""
    df = train.copy()
    df["hour_of_day"] = df["hour"] % 100
    df["day_of_week"] = pd.to_datetime(
        df["hour"].astype(str).str.zfill(8), format="%y%m%d%H"
    ).dt.dayofweek

    # Frequency encoding
    for col in HIGH_CARDINALITY:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[f"{col}_freq"] = df[col].map(freq).fillna(0.0)
            df = df.drop(columns=[col])

    # Label encoding
    for col in LOW_CARDINALITY:
        if col in df.columns:
            codes = {v: i for i, v in enumerate(df[col].astype(str).unique())}
            df[f"{col}_enc"] = df[col].astype(str).map(codes)
            df = df.drop(columns=[col])

    # Drop raw IDs
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return df
