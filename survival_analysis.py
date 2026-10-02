"""
survival_analysis.py

Cox Proportional Hazards modeling of GLP-1 treatment persistence
(time-to-discontinuation), using lifelines.

- Fits a CoxPH model on clinical + utilization covariates on a train split.
- Reports the top-6 statistically significant hazard-ratio drivers from the
  actual fitted model output (sorted by p-value / |coefficient|, not
  pre-selected).
- Computes the concordance index (C-index) on a held-out test split.
- Builds risk-stratified groups (tertiles of the test-set predicted
  partial-hazard risk score) and compares them with Kaplan-Meier curves
  and a log-rank test.

Run with:  python survival_analysis.py
"""

import os

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

DURATION_COL = "time_to_discontinuation_days"
EVENT_COL = "discontinued"

SURVIVAL_FEATURES = [
    "age",
    "bmi",
    "a1c",
    "n_comorbidities",
    "htn_flag",
    "t2d_flag",
    "osa_flag",
    "dyslipidemia_flag",
    "depression_anxiety_flag",
    "annual_office_visits",
    "prior_med_switches",
    "pharmacy_fill_gap_days",
    "baseline_copay_usd",
]

RANDOM_SEED = 42


def load_data():
    return pd.read_csv(os.path.join(PROCESSED_DIR, "patients_with_clusters.csv"))


def train_test(df, test_size=0.3, seed=RANDOM_SEED):
    cols = SURVIVAL_FEATURES + [DURATION_COL, EVENT_COL]
    data = df[cols].copy()
    train_df, test_df = train_test_split(
        data, test_size=test_size, random_state=seed, stratify=data[EVENT_COL]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def fit_cox(train_df, penalizer=0.01):
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(train_df, duration_col=DURATION_COL, event_col=EVENT_COL)
    return cph


def top_drivers(cph, n=6):
    summary = cph.summary.copy()
    summary = summary.sort_values("p")
    sig = summary[summary["p"] < 0.05]
    top = sig.head(n) if len(sig) >= n else summary.head(n)
    cols = ["coef", "exp(coef)", "se(coef)", "z", "p",
            "exp(coef) lower 95%", "exp(coef) upper 95%"]
    return top[cols].reset_index().rename(columns={"covariate": "feature"})


def evaluate_c_index(cph, test_df):
    risk_score = cph.predict_partial_hazard(test_df)
    c_index = concordance_index(
        test_df[DURATION_COL], -risk_score, test_df[EVENT_COL]
    )
    return c_index, risk_score


def risk_tertiles(test_df, risk_score):
    out = test_df.copy()
    out["risk_score"] = risk_score.values
    out["risk_group"] = pd.qcut(
        out["risk_score"], q=3, labels=["Low risk", "Medium risk", "High risk"]
    )
    return out


def logrank_across_groups(stratified_df):
    result = multivariate_logrank_test(
        stratified_df[DURATION_COL], stratified_df["risk_group"], stratified_df[EVENT_COL]
    )
    return result


def km_by_group(stratified_df):
    kmf = KaplanMeierFitter()
    curves = {}
    for group, sub in stratified_df.groupby("risk_group", observed=True):
        kmf.fit(sub[DURATION_COL], sub[EVENT_COL], label=str(group))
        curves[str(group)] = kmf.survival_function_.copy()
    return curves


def main():
    df = load_data()
    train_df, test_df = train_test(df)

    cph = fit_cox(train_df)
    drivers = top_drivers(cph, n=6)

    c_index, risk_score = evaluate_c_index(cph, test_df)
    stratified = risk_tertiles(test_df, risk_score)
    logrank_result = logrank_across_groups(stratified)
    curves = km_by_group(stratified)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cph.summary.to_csv(os.path.join(OUTPUT_DIR, "cox_model_summary.csv"))
    drivers.to_csv(os.path.join(OUTPUT_DIR, "top6_hazard_drivers.csv"), index=False)
    stratified.to_csv(os.path.join(OUTPUT_DIR, "risk_stratified_test_set.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "c_index.txt"), "w") as f:
        f.write(f"{c_index:.4f}\n")

    print(f"Cox model fit on {len(train_df):,} training patients "
          f"({train_df[EVENT_COL].sum():,} events).")
    print(f"\nTop hazard-ratio drivers (actual model output, sorted by p-value):")
    print(drivers.to_string(index=False))
    print(f"\nHeld-out C-index (n={len(test_df):,}): {c_index:.4f}")
    print(f"\nLog-rank test across risk tertiles: "
          f"chi2={logrank_result.test_statistic:.2f}, p={logrank_result.p_value:.3e}")
    print("\nMedian event rate by risk group:")
    print(stratified.groupby("risk_group", observed=True)[EVENT_COL].mean())

    return {
        "cph": cph,
        "train_df": train_df,
        "test_df": test_df,
        "drivers": drivers,
        "c_index": c_index,
        "stratified": stratified,
        "logrank_result": logrank_result,
        "curves": curves,
    }


if __name__ == "__main__":
    main()
