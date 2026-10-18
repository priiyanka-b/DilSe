"""
tests/test_integrity.py: automatic checks that the pipeline is leak-free and that the
reported numbers come from the saved files (DECISIONS.md D-21, D-24).

Run from the dt-cardio folder:   pip install pytest   then   python -m pytest -v
A test is SKIPPED (not failed) if the data file it needs is not present, for example in
the public repository, where raw data is not redistributed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROC = ROOT / "data" / "processed"


def need(*paths):
    for p in paths:
        if not (ROOT / p).exists():
            pytest.skip(f"{p} not present")


# ---------------------------------------------------------------- data splits (D-3, D-4)
def test_no_patient_in_two_splits():
    need("data/processed/meta_train.csv", "data/processed/meta_val.csv", "data/processed/meta_test.csv")
    ids = {s: set(pd.read_csv(PROC / f"meta_{s}.csv")["patient_id"]) for s in ["train", "val", "test"]}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]


def test_official_folds_and_adults_only():
    need("data/processed/meta_train.csv", "data/processed/meta_val.csv", "data/processed/meta_test.csv")
    folds = {"train": set(range(1, 9)), "val": {9}, "test": {10}}
    for split, allowed in folds.items():
        m = pd.read_csv(PROC / f"meta_{split}.csv")
        assert set(m["strat_fold"]) <= allowed
        assert m["age"].min() >= 18 and m["age"].max() <= 90


# ---------------------------------------------------------------- monitor (D-17)
def test_monitor_never_looks_ahead():
    """Changing the future must not change today's score."""
    need("data/processed/wearables_val.csv", "data/processed/episodes_val.csv")
    from monitor import daily_scores, load
    X, _ = load("val")
    X = X[:200].copy()
    t = 45
    before, _ = daily_scores(X, "rolling")
    X_future_changed = X.copy()
    X_future_changed[:, t + 1:, :] += 50.0          # wildly different future
    after, _ = daily_scores(X_future_changed, "rolling")
    np.testing.assert_allclose(before[:, :t + 1], after[:, :t + 1], equal_nan=True)


def test_no_alarm_before_monitoring_starts():
    need("data/processed/wearables_val.csv", "data/processed/episodes_val.csv")
    from monitor import FIRST_MONITORED_DAY, daily_scores, flags_from, load
    X, _ = load("val")
    score, _ = daily_scores(X, "rolling")
    flags = flags_from(score, -10.0, 1)              # threshold so low everything would flag
    assert not flags[:, :FIRST_MONITORED_DAY].any()


# ---------------------------------------------------------------- simulation (D-13, D-16)
def test_episodes_independent_of_ecg_label():
    need("data/processed/episodes_val.csv", "data/processed/meta_val.csv")
    eps = pd.read_csv(PROC / "episodes_val.csv").set_index("ecg_id")
    lab = pd.read_csv(PROC / "meta_val.csv").set_index("ecg_id")["label"]
    share = eps.join(lab).groupby("label")["episode"].mean()
    assert abs(share[0] - share[1]) < 0.05


# ---------------------------------------------------------------- locked settings
def test_locked_settings_unchanged():
    need("outputs/monitor/monitor_config.json", "outputs/twin/twin_config.json",
         "outputs/final_test/FINAL_RUN.lock")
    mon = json.load(open(ROOT / "outputs/monitor/monitor_config.json"))
    assert mon["locked_on"] == "validation patients"
    assert mon["max_fa_per_month"] == 0.5
    assert mon["methods"]["rolling"] == {"threshold": 0.5, "persistence": 4}
    assert mon["methods"]["fixed"] == {"threshold": 1.05, "persistence": 1}
    twin = json.load(open(ROOT / "outputs/twin/twin_config.json"))
    assert twin["risk_threshold_p_ecg"] == pytest.approx(0.510543)
    assert (twin["low_risk"]["threshold"], twin["low_risk"]["persistence"]) == (0.7, 4)
    assert (twin["high_risk"]["threshold"], twin["high_risk"]["persistence"]) == (0.55, 3)


def test_risk_threshold_gives_90pct_specificity_on_validation():
    need("outputs/fusion/val_predictions.csv", "outputs/twin/twin_config.json")
    v = pd.read_csv(ROOT / "outputs/fusion/val_predictions.csv")
    th = json.load(open(ROOT / "outputs/twin/twin_config.json"))["risk_threshold_p_ecg"]
    spec = ((v["p_ecg"] < th) & (v["label"] == 0)).sum() / (v["label"] == 0).sum()
    assert spec >= 0.90


# ---------------------------------------------------------------- reported numbers
def test_reported_test_auroc_matches_saved_predictions():
    """The headline numbers must be recomputable from the saved per-patient predictions."""
    need("outputs/final_test/test_predictions.csv", "outputs/final_test/final_results.json")
    p = pd.read_csv(ROOT / "outputs/final_test/test_predictions.csv")
    res = json.load(open(ROOT / "outputs/final_test/final_results.json"))["ecg"]
    assert roc_auc_score(p["label"], p["p_ecg"]) == pytest.approx(res["ECG only (FINAL MODEL)"]["auroc"], abs=1e-6)
    assert roc_auc_score(p["label"], p["p_static"]) == pytest.approx(res["Static only (age + sex)"]["auroc"], abs=1e-6)
    lt = res["locked_threshold"]
    pred = p["p_ecg"] >= lt["threshold"]
    assert int((pred & (p["label"] == 1)).sum()) == lt["tp"]
    assert int((~pred & (p["label"] == 0)).sum()) == lt["tn"]


def test_test_set_size_and_prevalence():
    need("outputs/final_test/test_predictions.csv")
    p = pd.read_csv(ROOT / "outputs/final_test/test_predictions.csv")
    assert len(p) == 2145
    assert p["label"].mean() == pytest.approx(0.576, abs=0.001)


# ---------------------------------------------------------------- learning twin (D-24)
def test_learning_twin_never_looks_ahead():
    """Changing the future must not change the learning phase's scores either."""
    need("data/processed/wearables_val.csv", "data/processed/episodes_val.csv", "outputs/recipe/recipe.json")
    from learning_twin import learning_score, learning_state, load_recipe, prior_for
    from monitor import load
    X, eps = load("val")
    X = X[:200].copy()
    pri = tuple(np.stack([prior_for(load_recipe(), s)[k] for s in eps["sex"].iloc[:200]]) for k in range(4))
    t = 20
    before = learning_score(learning_state(X, pri)["z"])
    X2 = X.copy()
    X2[:, t + 1:, :] += 50.0
    after = learning_score(learning_state(X2, pri)["z"])
    np.testing.assert_allclose(before[:, :t + 1], after[:, :t + 1], equal_nan=True)


def test_learning_twin_starts_from_population_prior():
    """On day 0 the twin knows nothing personal: 0% learned, expected range = population spread."""
    need("outputs/recipe/recipe.json")
    from learning_twin import learning_state, load_recipe, prior_for
    m0, tau, sig, phi = prior_for(load_recipe(), "female")
    X = np.tile(m0, (1, 60, 1)).astype(float)
    st = learning_state(X, (m0, tau, sig, phi))
    assert np.all(st["learned"][0, 0] == 0)
    np.testing.assert_allclose(st["pred_sd"][0, 0], np.sqrt(tau ** 2 + sig ** 2))
    assert np.all(st["learned"][0, 34] > st["learned"][0, 10])        # it learns as days pass


def test_learning_twin_leaves_locked_days_unchanged():
    """Days 35-59 must come from the locked monitor alone (D-24): no learning flag after day 34."""
    need("data/processed/wearables_val.csv", "data/processed/episodes_val.csv",
         "outputs/learning/learning_config.json", "outputs/fusion/val_predictions.csv")
    from learning_twin import LEARN_DAYS, learning_score, learning_state, load_recipe, prior_for, tier_learning_flags
    from monitor import load
    X, eps = load("val")
    pri = tuple(np.stack([prior_for(load_recipe(), s)[k] for s in eps["sex"]]) for k in range(4))
    score = learning_score(learning_state(X, pri)["z"])
    cfg = json.load(open(ROOT / "outputs/learning/learning_config.json"))
    high = np.zeros(len(eps), bool)
    f = tier_learning_flags(np.where(np.isnan(score), 99.0, score), high, cfg)   # even a huge score
    assert not f[:, LEARN_DAYS:].any()
    assert np.isnan(score[:, LEARN_DAYS:]).all()


def test_learning_settings_locked():
    need("outputs/learning/learning_config.json")
    cfg = json.load(open(ROOT / "outputs/learning/learning_config.json"))
    assert cfg["decision"] == "D-24" and cfg["prior_sd_days"] == 10
    assert (cfg["low_risk"]["threshold"], cfg["low_risk"]["persistence"]) == (0.6, 3)
    assert (cfg["high_risk"]["threshold"], cfg["high_risk"]["persistence"]) == (0.45, 3)
    assert cfg["alarms_enabled"] is True
