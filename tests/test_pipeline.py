import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import simulate_patient_data as sim  # noqa: E402
import segment_patients as seg  # noqa: E402
import survival_analysis as surv  # noqa: E402


@pytest.fixture(scope="module")
def patients_df():
    return sim.simulate(n_patients=sim.N_PATIENTS, seed=sim.RANDOM_SEED)


@pytest.fixture(scope="module")
def segmentation_result(patients_df, tmp_path_factory):
    Xs, _ = seg.build_feature_matrix(patients_df)
    gmm, labels, aris = seg.stability_check(Xs, n_components=seg.N_COMPONENTS_CHOSEN)
    df = patients_df.copy()
    df["cluster"] = labels
    return {"df": df, "gmm": gmm, "labels": labels, "aris": aris}


@pytest.fixture(scope="module")
def survival_result(segmentation_result):
    df = segmentation_result["df"]
    train_df, test_df = surv.train_test(df)
    cph = surv.fit_cox(train_df)
    c_index, risk_score = surv.evaluate_c_index(cph, test_df)
    drivers = surv.top_drivers(cph, n=6)
    stratified = surv.risk_tertiles(test_df, risk_score)
    logrank_result = surv.logrank_across_groups(stratified)
    return {
        "cph": cph, "c_index": c_index, "drivers": drivers,
        "stratified": stratified, "logrank_result": logrank_result,
    }


def test_population_size_at_least_20000(patients_df):
    assert len(patients_df) >= 20000
    assert patients_df["patient_id"].is_unique


def test_persistence_outcome_has_both_events_and_censoring(patients_df):
    assert patients_df["discontinued"].isin([0, 1]).all()
    event_rate = patients_df["discontinued"].mean()
    assert 0.3 < event_rate < 0.95
    assert (patients_df["time_to_discontinuation_days"] > 0).all()


def test_gmm_produces_5_clusters(segmentation_result):
    labels = segmentation_result["labels"]
    assert len(np.unique(labels)) == 5


def test_gmm_clusters_are_stable_under_resampling(segmentation_result):
    """Repeated-fit label-agreement check: mean adjusted Rand index across
    bootstrap-resample refits should indicate the 5-cluster solution is not
    an artifact of a single initialization."""
    aris = segmentation_result["aris"]
    assert aris.mean() > 0.5
    assert aris.min() > 0.1


def test_cluster_sizes_are_all_non_trivial(segmentation_result):
    counts = pd.Series(segmentation_result["labels"]).value_counts()
    assert len(counts) == 5
    assert counts.min() / counts.sum() > 0.01


def test_cox_top_drivers_are_statistically_significant(survival_result):
    drivers = survival_result["drivers"]
    assert len(drivers) == 6
    assert (drivers["p"] < 0.05).all()


def test_c_index_in_reasonable_band(survival_result):
    c_index = survival_result["c_index"]
    assert 0.65 <= c_index <= 0.85


def test_risk_tertiles_show_separated_discontinuation_rates(survival_result):
    stratified = survival_result["stratified"]
    rates = stratified.groupby("risk_group", observed=True)["discontinued"].mean()
    assert rates["Low risk"] < rates["Medium risk"] < rates["High risk"]


def test_logrank_across_risk_groups_is_significant(survival_result):
    result = survival_result["logrank_result"]
    assert result.p_value < 0.001
