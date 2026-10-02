"""
simulate_patient_data.py

Simulates a synthetic population of adults with obesity who initiate GLP-1
therapy (e.g. semaglutide/tirzepatide-class agents). Produces:
  - clinical features (BMI, A1c, comorbidity flags, age, sex)
  - healthcare-utilization features (visit frequency, prior medication
    switches, pharmacy fill-gap behavior)
  - a time-to-discontinuation (persistence) outcome with right-censoring,
    generated from a genuine latent hazard model so that a subset of the
    features carries real, recoverable signal for survival analysis, and a
    separate (overlapping) subset of features drives cluster structure.

Nothing here is real patient data. All relationships are synthesized by the
author for portfolio/demo purposes, loosely informed by published GLP-1
persistence literature (e.g. high 12-month discontinuation rates, GI
side-effect burden, cost/access driven lapses) but are NOT fit to any real
dataset.

Run with:  python simulate_patient_data.py
"""

import os

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
N_PATIENTS = 21000
RANDOM_SEED = 42
ADMIN_CENSOR_DAYS = 540  # ~18 months of observation


def _latent_segments(rng, n):
    """
    Draw each patient from one of 5 latent clinical/utilization archetypes.
    These are the *true* generating segments; the GMM later has to recover
    an unsupervised approximation of this structure purely from the
    observed features, it is never told the segment id.
    """
    # Archetype prevalence in a typical GLP-1-treated obesity population.
    probs = np.array([0.30, 0.22, 0.20, 0.16, 0.12])
    probs = probs / probs.sum()
    seg = rng.choice(5, size=n, p=probs)
    return seg


# Per-segment generative parameters:
#   0: "Low-complexity, engaged"      - lower BMI/A1c, few comorbidities, frequent visits
#   1: "Metabolic-severe, high A1c"   - high BMI + A1c, diabetes-heavy, moderate engagement
#   2: "High-utilizer, med-switchers" - many prior switches, frequent visits, moderate BMI
#   3: "Disengaged, sporadic fillers" - infrequent visits, erratic pharmacy fills, higher BMI
#   4: "Older, multi-morbid"          - older age, many comorbidities, moderate everything
SEGMENT_PARAMS = {
    0: dict(bmi=(33.0, 3.0), a1c=(5.6, 0.3), age=(42, 9), n_comorb_lambda=0.6,
            visits=(6.5, 1.5), switches_lambda=0.3, fill_gap=(8, 4)),
    1: dict(bmi=(39.5, 4.0), a1c=(7.8, 0.9), age=(54, 10), n_comorb_lambda=2.2,
            visits=(5.0, 1.8), switches_lambda=0.8, fill_gap=(12, 6)),
    2: dict(bmi=(35.0, 3.5), a1c=(6.0, 0.5), age=(46, 10), n_comorb_lambda=1.0,
            visits=(9.0, 2.0), switches_lambda=2.4, fill_gap=(10, 5)),
    3: dict(bmi=(41.0, 4.5), a1c=(6.3, 0.6), age=(39, 11), n_comorb_lambda=0.8,
            visits=(2.2, 1.0), switches_lambda=0.4, fill_gap=(35, 15)),
    4: dict(bmi=(36.0, 4.0), a1c=(6.8, 0.7), age=(66, 8), n_comorb_lambda=3.0,
            visits=(7.0, 2.0), switches_lambda=0.6, fill_gap=(14, 6)),
}

SEGMENT_NAMES = {
    0: "Low-complexity, engaged",
    1: "Metabolic-severe, high A1c",
    2: "High-utilizer, med-switchers",
    3: "Disengaged, sporadic fillers",
    4: "Older, multi-morbid",
}


def simulate(n_patients=N_PATIENTS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = n_patients

    segment = _latent_segments(rng, n)

    bmi = np.zeros(n)
    a1c = np.zeros(n)
    age = np.zeros(n)
    n_comorbidities = np.zeros(n, dtype=int)
    annual_visits = np.zeros(n)
    prior_med_switches = np.zeros(n, dtype=int)
    pharmacy_fill_gap_days = np.zeros(n)

    for s, p in SEGMENT_PARAMS.items():
        mask = segment == s
        k = mask.sum()
        bmi[mask] = rng.normal(p["bmi"][0], p["bmi"][1], k)
        a1c[mask] = rng.normal(p["a1c"][0], p["a1c"][1], k)
        age[mask] = rng.normal(p["age"][0], p["age"][1], k)
        n_comorbidities[mask] = rng.poisson(p["n_comorb_lambda"], k)
        annual_visits[mask] = np.clip(rng.normal(p["visits"][0], p["visits"][1], k), 0, None)
        prior_med_switches[mask] = rng.poisson(p["switches_lambda"], k)
        pharmacy_fill_gap_days[mask] = np.clip(
            rng.gamma(shape=2.0, scale=p["fill_gap"][0] / 2.0, size=k), 1, None
        )

    bmi = np.clip(bmi, 27, 65)
    a1c = np.clip(a1c, 4.8, 13.5)
    age = np.clip(age, 18, 90).round().astype(int)
    n_comorbidities = np.clip(n_comorbidities, 0, 10)
    annual_visits = np.round(np.clip(annual_visits, 0, 20), 1)
    prior_med_switches = np.clip(prior_med_switches, 0, 8)
    pharmacy_fill_gap_days = np.round(np.clip(pharmacy_fill_gap_days, 1, 90), 1)

    sex = rng.choice(["F", "M"], size=n, p=[0.62, 0.38])

    # Individual comorbidity flags (binary), correlated with n_comorbidities and segment
    comorb_base_rate = np.clip(n_comorbidities / 6.0, 0.02, 0.95)
    htn = rng.binomial(1, np.clip(comorb_base_rate * 1.1, 0, 0.97))
    t2d = rng.binomial(1, np.clip(comorb_base_rate * 0.9 + (a1c >= 6.5) * 0.35, 0, 0.97))
    osa = rng.binomial(1, np.clip(comorb_base_rate * 0.6 + (bmi >= 40) * 0.2, 0, 0.9))
    dyslipidemia = rng.binomial(1, np.clip(comorb_base_rate * 0.8, 0, 0.95))
    depression_anxiety = rng.binomial(1, np.clip(0.18 + (segment == 3) * 0.12, 0, 0.6))

    insurance_type = rng.choice(
        ["Commercial", "Medicare", "Medicaid"], size=n, p=[0.58, 0.27, 0.15]
    )
    baseline_copay = np.round(
        np.where(insurance_type == "Commercial", rng.gamma(2.0, 20, n),
                 np.where(insurance_type == "Medicare", rng.gamma(2.0, 30, n),
                          rng.gamma(2.0, 10, n))), 2
    )

    df = pd.DataFrame({
        "patient_id": [f"P{idx:06d}" for idx in range(n)],
        "true_segment": segment,  # kept for validation only; NOT used as a model feature
        "age": age,
        "sex": sex,
        "bmi": bmi.round(1),
        "a1c": a1c.round(2),
        "n_comorbidities": n_comorbidities,
        "htn_flag": htn,
        "t2d_flag": t2d,
        "osa_flag": osa,
        "dyslipidemia_flag": dyslipidemia,
        "depression_anxiety_flag": depression_anxiety,
        "annual_office_visits": annual_visits,
        "prior_med_switches": prior_med_switches,
        "pharmacy_fill_gap_days": pharmacy_fill_gap_days,
        "insurance_type": insurance_type,
        "baseline_copay_usd": baseline_copay,
    })

    # ---- Time-to-discontinuation (persistence) via a Weibull proportional
    # hazards process with a genuine latent linear predictor (lp). The
    # Cox model fit downstream has to recover these drivers from data; the
    # coefficients below are never passed to the modeling code.
    lp = (
        0.45 * (df["pharmacy_fill_gap_days"] - 15) / 10    # erratic fills -> higher hazard
        + 0.35 * (df["prior_med_switches"])                 # switch history -> higher hazard
        - 0.30 * (df["annual_office_visits"] - 6) / 3        # engagement -> lower hazard
        + 0.25 * (df["depression_anxiety_flag"])             # -> higher hazard
        - 0.20 * (df["n_comorbidities"] - 1.5) / 2           # more clinical contact -> lower hazard
        + 0.22 * (df["baseline_copay_usd"] - 25) / 25        # cost burden -> higher hazard
        - 0.18 * (df["t2d_flag"])                            # established diabetes care -> slightly more persistent
        + 0.12 * (df["bmi"] - 36) / 5                        # higher BMI -> marginally higher discontinuation (GI tolerability)
        - 0.10 * (df["age"] - 48) / 10                       # older -> somewhat more persistent
    )

    lp = lp * 1.3  # overall effect-size scaling so the signal is strong enough
    # relative to intrinsic Weibull event-time randomness to be recoverable
    # at a realistic (~0.74) C-index, rather than either trivial or unrecoverable.

    rng2 = np.random.default_rng(seed + 1)
    noise = rng2.normal(0, 0.28, n)  # irreducible noise so C-index lands < 1.0
    lp_noisy = lp + noise

    # Weibull AFT-style draw: T = baseline_scale * exp(-lp/shape) * (-ln U)^(1/shape)
    shape = 1.3
    baseline_scale = 260.0  # days
    u = rng2.uniform(1e-6, 1 - 1e-6, n)
    time_to_event = baseline_scale * np.exp(-lp_noisy / shape) * (-np.log(u)) ** (1 / shape)
    time_to_event = np.clip(time_to_event, 1, None)

    # Administrative censoring at ADMIN_CENSOR_DAYS, plus random dropout
    # censoring (loss to follow-up) independent of the hazard.
    random_censor_time = rng2.exponential(scale=700, size=n)
    censor_time = np.minimum(ADMIN_CENSOR_DAYS, random_censor_time)

    observed_time = np.clip(np.minimum(time_to_event, censor_time), 0.1, None)
    event_observed = (time_to_event <= censor_time).astype(int)

    df["time_to_discontinuation_days"] = np.round(observed_time, 1)
    df["discontinued"] = event_observed

    return df


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    df = simulate()
    out_path = os.path.join(RAW_DIR, "patients.csv")
    df.to_csv(out_path, index=False)
    print(f"Simulated {len(df):,} patients -> {out_path}")
    print(f"Event rate (discontinued within observation window): "
          f"{df['discontinued'].mean():.1%}")
    return df


if __name__ == "__main__":
    main()
