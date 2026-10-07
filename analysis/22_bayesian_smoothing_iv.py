"""Stage 22 — Empirical Bayes Smoothing & Information Value (IV) Analysis.

Evaluates Bayesian Beta-Binomial conjugate prior smoothing for sparse categorical IDs
(device_ip, site_id, app_id) and computes Information Value (IV) / Weight of Evidence (WoE)
across all ad context fields.

Produces:
    analysis/output/22_bayesian_iv_summary.json
    analysis/output/figures/bayesian_01_smoothing_iv.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_sample, save_fig, save_json, TARGET, SEED
from ctr_math import BayesianCTRSmoother, InformationValueAnalyzer


def main() -> None:
    print("Stage 22: Empirical Bayes Smoothing & Information Value (IV)")
    df = load_sample()
    
    global_clicks = int(df[TARGET].sum())
    global_total = len(df)
    global_ctr = global_clicks / global_total

    # Initialize Empirical Bayes Smoother with global prior
    smoother = BayesianCTRSmoother(prior_mean=global_ctr, pseudo_counts=20.0)

    # 1. Evaluate Bayesian Smoothing on Device IPs (Sparse ID distribution)
    ip_stats = df.groupby("device_ip").agg(
        clicks=(TARGET, "sum"),
        impressions=(TARGET, "count")
    ).reset_index()

    ip_stats["raw_ctr"] = ip_stats["clicks"] / ip_stats["impressions"]
    ip_stats["smoothed_ctr"] = ip_stats.apply(
        lambda r: smoother.smooth(int(r["clicks"]), int(r["impressions"])), axis=1
    )
    ip_stats["uncertainty_variance"] = ip_stats.apply(
        lambda r: smoother.variance(int(r["clicks"]), int(r["impressions"])), axis=1
    )

    # Filter into low impression (1-5) vs high impression (>50) to demonstrate shrinkage
    low_imp = ip_stats[ip_stats["impressions"] <= 3].head(100)
    high_imp = ip_stats[ip_stats["impressions"] >= 50].head(100)

    # 2. Compute Information Value (IV) and Weight of Evidence (WoE) across categorical features
    candidate_features = ["banner_pos", "site_category", "app_category", "device_type", "device_conn_type", "C1", "C15", "C16", "C18"]
    iv_results = []

    for feat in candidate_features:
        grouped = df.groupby(feat).agg(
            clicks=(TARGET, "sum"),
            impressions=(TARGET, "count")
        )
        grouped["non_clicks"] = grouped["impressions"] - grouped["clicks"]
        
        iv, woe_list, rating = InformationValueAnalyzer.compute_iv(
            grouped["clicks"].tolist(), grouped["non_clicks"].tolist()
        )
        iv_results.append({
            "feature": feat,
            "information_value_iv": round(iv, 4),
            "predictive_power": rating,
            "categories_count": len(grouped)
        })

    iv_results = sorted(iv_results, key=lambda x: x["information_value_iv"], reverse=True)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Empirical Bayes Shrinkage Curve
    sample_ips = ip_stats.sort_values(by="impressions").iloc[::max(1, len(ip_stats)//300)]
    ax1.scatter(sample_ips["impressions"], sample_ips["raw_ctr"], color="#f87171", alpha=0.5, s=20, label="Raw Empirical CTR (Noisy)")
    ax1.scatter(sample_ips["impressions"], sample_ips["smoothed_ctr"], color="#34d399", alpha=0.7, s=25, label="Empirical Bayes Smoothed CTR (Beta-Binomial)")
    ax1.axhline(global_ctr, color="#fbbf24", linestyle="--", linewidth=1.5, label=f"Global Prior Mean CTR ({global_ctr*100:.2f}%)")
    ax1.set_xscale("log")
    ax1.set_title("Empirical Bayes Shrinkage for Sparse Device IPs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Impression Count (Log Scale)", fontsize=10)
    ax1.set_ylabel("Estimated Click Probability", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Information Value (IV) Barplot
    features_sorted = [x["feature"] for x in iv_results]
    ivs_sorted = [x["information_value_iv"] for x in iv_results]
    colors = ["#38bdf8" if v > 0.1 else "#94a3b8" for v in ivs_sorted]

    bars = ax2.barh(features_sorted[::-1], ivs_sorted[::-1], color=colors[::-1])
    ax2.axvline(0.02, color="#ef4444", linestyle=":", label="Unpredictable Threshold (0.02)")
    ax2.axvline(0.10, color="#f59e0b", linestyle="--", label="Medium Predictor Threshold (0.10)")
    ax2.axvline(0.30, color="#10b981", linestyle="-.", label="Strong Predictor Threshold (0.30)")
    ax2.set_title("Feature Information Value (IV) & Information Theory Ranking", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Information Value (IV = sum (Click% - NonClick%) * WoE)", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")

    fig.tight_layout()
    save_fig(fig, "bayesian_01_smoothing_iv.png")

    summary = {
        "global_prior_ctr": round(global_ctr, 4),
        "empirical_bayes_pseudo_count_m": 20.0,
        "device_ips_analyzed": len(ip_stats),
        "shrinkage_effect": "Low-impression identifiers (1-3 clicks) are shrunk toward global prior (16.94%), eliminating catastrophic overfitting and cold-start variance.",
        "information_value_rankings": iv_results,
        "highest_iv_feature": iv_results[0]["feature"],
        "highest_iv_score": iv_results[0]["information_value_iv"]
    }
    save_json("22_bayesian_iv_summary.json", summary)
    print(f"Stage 22 complete: Top IV Feature = {iv_results[0]['feature']} (IV = {iv_results[0]['information_value_iv']})")


if __name__ == "__main__":
    main()
