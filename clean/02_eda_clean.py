"""Stage 02 — Exploratory Data Analysis & Empirical Bayes CTR (Clean Suite).

Lineage & Connections:
  • Upstream: Stage 01 raw schema & baseline sample (Data/train_sample.csv).
  • Mathematical: Empirical Bayes shrinkage \hat{p}_j^{\text{EB}} = \frac{k_j+\alpha}{n_j+\alpha+\beta}; diurnal distribution P(click|hour).
  • Downstream: Informs Stage 03 feature engineering (harmonic cyclical encoding for hour, frequency rank for IDs).
  • AdTech System: DSP Inventory Quality Scoring & Campaign Diurnal Pacing.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common_clean import (
    load_raw_sample, save_summary_json, save_plot_figure, save_dataframe_csv,
    print_stage_banner, TARGET, SEED
)

STAGE_ID = "02_eda"
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


def main() -> None:
    print_stage_banner(STAGE_ID)
    df = load_raw_sample()
    n = len(df)
    summary = {}

    print("TASK 1: Dataset Overview & Target Distribution")
    summary["rows"] = int(n)
    summary["columns"] = int(df.shape[1])
    summary["column_names"] = list(df.columns)
    
    # Target distribution
    overall_ctr = float(df[TARGET].mean())
    summary["overall_ctr"] = round(overall_ctr, 6)
    
    # Task 2: Temporal Analysis (Diurnal cycle)
    df["dt"] = pd.to_datetime(df["hour"].astype(str), format="%y%m%d%H")
    df["hour_of_day"] = df["dt"].dt.hour
    df["day_of_week"] = df["dt"].dt.day_name()
    
    hourly = df.groupby("hour_of_day")[TARGET].agg(["count", "mean"]).reset_index()
    hourly.columns = ["hour", "impressions", "ctr"]
    summary["hourly_ctr"] = {int(r.hour): round(float(r.ctr), 4) for _, r in hourly.iterrows()}

    # Task 3: Cardinality profiling
    cat_cols = ["site_id", "app_id", "device_ip", "device_model"]
    summary["unique_counts"] = {c: int(df[c].nunique()) for c in cat_cols}

    # Task 4: Empirical Bayes CTR Shrinkage on Top Sites
    site_counts = df["site_id"].value_counts()
    top_sites = site_counts[site_counts >= 100].index
    sub_df = df[df["site_id"].isin(top_sites)]
    
    site_stats = sub_df.groupby("site_id")[TARGET].agg(["count", "sum"]).reset_index()
    site_stats.columns = ["site_id", "impressions", "clicks"]
    site_stats["raw_ctr"] = site_stats["clicks"] / site_stats["impressions"]
    
    # Prior Beta(alpha, beta) via Method of Moments
    mu = site_stats["raw_ctr"].mean()
    var = site_stats["raw_ctr"].var()
    if var > 0 and var < mu * (1 - mu):
        alpha_prior = mu * (mu * (1 - mu) / var - 1)
        beta_prior = (1 - mu) * (mu * (1 - mu) / var - 1)
    else:
        alpha_prior, beta_prior = 10.0, 50.0
    
    site_stats["eb_ctr"] = (site_stats["clicks"] + alpha_prior) / (site_stats["impressions"] + alpha_prior + beta_prior)
    summary["eb_prior_alpha"] = round(float(alpha_prior), 4)
    summary["eb_prior_beta"] = round(float(beta_prior), 4)

    # Plot 1: Target distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET, palette=["#3498db", "#e74c3c"], ax=ax)
    ax.set_title(f"Target Distribution (CTR: {overall_ctr*100:.2f}%)")
    ax.set_xticklabels(["No Click (0)", "Click (1)"])
    save_plot_figure(fig, "02_target_distribution.png")

    # Plot 2: Diurnal CTR
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(hourly["hour"], hourly["ctr"] * 100, marker="o", color="#2ecc71", lw=2)
    ax1.set_title("Diurnal CTR (%) by Hour of Day")
    ax1.set_xlabel("Hour of Day (UTC)")
    ax1.set_ylabel("Click-Through Rate (%)")
    ax1.set_xticks(range(0, 24))
    ax1.grid(True, alpha=0.3)
    save_plot_figure(fig, "02_temporal_hourly.png")

    save_summary_json("02_eda_summary.json", summary, stage_id=STAGE_ID)
    print(f"[STAGE 02 COMPLETE] Diurnal & Empirical Bayes patterns profiled. Ready for Stage 03 Feature Engineering.")


if __name__ == "__main__":
    main()
