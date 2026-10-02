"""
make_figures.py

Generates the figures referenced in the README from the real pipeline
outputs (no synthetic/placeholder figure data):
  - BIC curve across candidate GMM component counts
  - Cluster profile heatmap (z-scored mean feature values per cluster)
  - Kaplan-Meier curves by risk-stratified group (from the Cox model)

Run with:  python make_figures.py   (after segment_patients.py and
survival_analysis.py have produced their CSV outputs)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

CLUSTER_FEATURES = [
    "bmi", "a1c", "age", "n_comorbidities",
    "annual_office_visits", "prior_med_switches", "pharmacy_fill_gap_days",
]

RISK_COLORS = {"Low risk": "#2E7D32", "Medium risk": "#F9A825", "High risk": "#C62828"}
RISK_ORDER = ["Low risk", "Medium risk", "High risk"]


def plot_bic_curve():
    bic_df = pd.read_csv(os.path.join(PROCESSED_DIR, "gmm_bic_curve.csv"))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bic_df["n_components"], bic_df["bic"], marker="o", color="#1565C0")
    ax.axvline(5, color="#C62828", linestyle="--", linewidth=1, label="chosen k = 5")
    ax.set_xlabel("Number of GMM components (k)")
    ax.set_ylabel("BIC")
    ax.set_title("GMM model selection: BIC vs. number of components")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "gmm_bic_curve.png"), dpi=150)
    plt.close(fig)


def plot_cluster_heatmap():
    profile = pd.read_csv(os.path.join(PROCESSED_DIR, "cluster_profiles.csv"), index_col=0)
    feat = profile[CLUSTER_FEATURES]
    z = (feat - feat.mean()) / feat.std()

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(z.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(CLUSTER_FEATURES)))
    ax.set_xticklabels(CLUSTER_FEATURES, rotation=40, ha="right")
    ax.set_yticks(range(len(z)))
    ax.set_yticklabels([f"Cluster {i}" for i in z.index])
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z.to_numpy()[i, j]:.1f}", ha="center", va="center",
                     fontsize=8, color="black")
    fig.colorbar(im, ax=ax, label="z-score (within feature)")
    ax.set_title("Cluster profiles (standardized mean feature values)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "cluster_profile_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_km_by_risk_group():
    stratified = pd.read_csv(os.path.join(OUTPUT_DIR, "risk_stratified_test_set.csv"))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    kmf = KaplanMeierFitter()
    for group in RISK_ORDER:
        sub = stratified[stratified["risk_group"] == group]
        kmf.fit(sub["time_to_discontinuation_days"], sub["discontinued"], label=group)
        kmf.plot_survival_function(ax=ax, color=RISK_COLORS[group], ci_show=True)
    ax.set_xlabel("Days since GLP-1 initiation")
    ax.set_ylabel("Persistence probability (not yet discontinued)")
    ax.set_title("Kaplan-Meier persistence by Cox-predicted risk tertile\n(held-out test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "km_by_risk_group.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_bic_curve()
    plot_cluster_heatmap()
    plot_km_by_risk_group()
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
