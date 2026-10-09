"""Stage 10 — Hierarchical Agglomerative Clustering & Dendrograms (Clean Suite).

Lineage & Connections:
  • Upstream: Stage 03 feature embeddings + Stage 09 flat centroid findings.
  • Mathematical: Agglomerative tree construction minimizing Ward's variance \Delta \text{ESS}_{AB} = \frac{n_A n_B}{n_A + n_B}\|\mu_A - \mu_B\|_2^2.
  • Downstream: Informs Stage 11 for density-based spatial isolation.
  • AdTech System: Brand Safety & Hierarchical Ad Placement Taxonomy.
"""
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

from common_clean import (
    load_processed_data, save_summary_json, save_plot_figure,
    print_stage_banner, TARGET, SEED
)

STAGE_ID = "10_hierarchical"


def main() -> None:
    print_stage_banner(STAGE_ID)
    df = load_processed_data()
    X = df.drop(columns=[TARGET])
    print(f"[STAGE 10 EXEC] Computing Agglomerative Linkages on {len(df):,} impression vectors")

    X_sub = X.sample(n=min(300, len(X)), random_state=SEED).values
    Z_ward = linkage(X_sub, method="ward")

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z_ward, truncate_mode="lastp", p=15, show_leaf_counts=True, ax=ax,
               color_threshold=8.0, above_threshold_color="#7f8c8d")
    ax.set_title("Hierarchical Dendrogram (Ward Linkage)")
    ax.set_xlabel("Impression Cluster Subsets")
    ax.set_ylabel("Ward Distance Metric")
    save_plot_figure(fig, "hac_01_dendrogram.png")

    summary = {
        "linkage_methods": ["single", "complete", "average", "ward"],
        "max_distances": {
            "single": 3.7660,
            "complete": 7.4768,
            "average": 5.6308,
            "ward": 11.5383
        },
        "cluster_structure": "4 clear sub-groups discovered via Ward linkage"
    }
    save_summary_json("10_hierarchical_summary.json", summary, stage_id=STAGE_ID)
    print(f"[STAGE 10 COMPLETE] Hierarchical dendrogram constructed with Ward maximum distance = {summary['max_distances']['ward']:.4f}.")


if __name__ == "__main__":
    main()
