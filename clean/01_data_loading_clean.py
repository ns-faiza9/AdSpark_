"""Stage 01 — Data Loading & Memory Architecture (Clean Suite).

Lineage & Connections:
  • Upstream: Raw Avazu dataset (Data/train.gz, 40.4M rows, 24 columns).
  • Mathematical: Sample mean CTR prior \bar{y} = \frac{1}{N}\sum y_i = 16.94%; memory reduction via integer downcasting.
  • Downstream: Emits canonical sample (Data/train_sample.csv) to Stage 02 (EDA) and Stage 03 (Feature Engineering).
  • AdTech System: DSP Stream Ingestion & Base Cold-Start Click Prior.
"""
import pandas as pd
from common_clean import (
    load_raw_sample, save_summary_json, save_dataframe_csv,
    print_stage_banner, TARGET
)

STAGE_ID = "01_data_loading"


def main() -> None:
    print_stage_banner(STAGE_ID)
    df = load_raw_sample()
    print(f"[STAGE 01 EXEC] Loaded sample: {len(df):,} impressions x {df.shape[1]} attributes")

    # 1. Dataset Overview
    overview = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "click_rate": round(float(df[TARGET].mean()), 6),
        "positive_clicks": int(df[TARGET].sum()),
    }

    # 2. First Rows Inspection
    head = df.head(10)
    save_dataframe_csv(head, "01_head.csv")

    # 3. Data Types
    dtypes = df.dtypes.astype(str).rename("dtype").reset_index().rename(columns={"index": "column"})
    save_dataframe_csv(dtypes, "01_dtypes.csv")

    # 4. Descriptive Statistics
    describe = df.describe(include="all").transpose().reset_index().rename(columns={"index": "column"})
    save_dataframe_csv(describe, "01_describe.csv")

    # 5. Missing Values Profile
    missing = df.isna().sum().rename("missing").reset_index().rename(columns={"index": "column"})
    missing["missing_pct"] = (missing["missing"] / len(df) * 100).round(4)
    save_dataframe_csv(missing, "01_missing.csv")
    overview["missing_total"] = int(df.isna().sum().sum())

    # 6. Target Distribution
    target_counts = df[TARGET].value_counts().to_dict()
    overview["target_counts"] = {str(k): int(v) for k, v in target_counts.items()}

    save_summary_json("01_data_loading_summary.json", overview, stage_id=STAGE_ID)
    print(f"[STAGE 01 COMPLETE] Baseline CTR Prior = {overview['click_rate'] * 100:.2f}%. Ready for Stage 02 EDA.")


if __name__ == "__main__":
    main()
