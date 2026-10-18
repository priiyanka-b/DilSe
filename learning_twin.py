"""
learning_twin.py: the learning twin, monitoring from day 0 instead of day 35 (DECISIONS.md D-24).

THE PROBLEM
  The locked monitor (monitor.py, D-17) compares each day with the 28 days ending 8 days
  earlier, so it cannot score anything before day 35. Each patient is unmonitored for
  their first five weeks.

THE IDEA
  On day 0 the twin knows only what is typical for people like this patient (the
  population values in recipe.json: resting HR by sex, HRV, sleep, steps). Every new day
  of the patient's own data pulls its estimate of "this person's normal" towards their
  own level, and the expected range narrows. By day 35 the twin hands over to the locked
  personal-baseline monitor, which then runs exactly as before.

METHOD, per measure, per day t < 35 (same scales and window rule as monitor.py)
  window      the patient's days t-35 ... t-8 that exist (0 to 28 days; same 7-day gap)
  prior       level ~ Normal(m0, tau^2)                     population, recipe.json
              day-to-day SD sigma0, day-to-day memory phi   recipe.json
  n_eff       AR(1) effective number of days in the window (correlated days count less)
  sigma_hat^2 (10 * sigma0^2 + n_eff * s^2) / (10 + n_eff - 1)    s = window SD
  level       precision-weighted mix of m0 and the window mean
              post_var = 1 / (1/tau^2 + n_eff/sigma_hat^2)
  expected    level +/- sqrt(sigma_hat^2 + post_var)        narrows as data accumulates
  z           direction * (value - level) / sqrt(sigma_hat^2 + post_var)
  score       mean of the 4 z values, then the 3-day mean (as monitor.py)
  learned     share of the level estimate that comes from the patient's own data:
              (n_eff/sigma_hat^2) / (1/tau^2 + n_eff/sigma_hat^2)

ALARMS
  Days 0-34: a run of k days with score above the tier's learning threshold (tuned on
  synthetic cohort A, D-24). Days 35-59: the locked monitor and twin settings, unchanged.

EVALUATION (rules fixed in D-24 before running; the test set is not used)
  Cohort A  validation patients' age/sex, unchanged simulator, episodes start days 10-28,
            seed 2401: used to tune the learning thresholds.
  Cohort B  the same with seed 2402: used to check R1 (false alarms) and R2 (benefit).
  PMData    real people, first 35 days, 3 measures: R3 (real-world false alarms).

Inputs:  outputs/recipe/recipe.json, data/processed/meta_val.csv, outputs/fusion/val_predictions.csv,
         outputs/twin/twin_config.json, data/processed/pmdata_daily.csv (optional, for R3)
Outputs: outputs/learning/learning_config.json   tuned learning thresholds per tier
         outputs/learning/learning_summary.txt    results against R1-R3
         outputs/learning/detection_early.png     detection within 14 days, current vs learning
         outputs/learning/band_example.png        the expected range narrowing for one patient
Run:     python learning_twin.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from monitor import (DIRECTION, FIRST_MONITORED_DAY, LOG_SCALE, MEASURES, PERSISTENCE, ROLL_GAP,
                     ROLL_LEN, SMOOTH, THRESHOLDS, daily_scores, flags_from)

OUT = Path("outputs/learning")
RECIPE = Path("outputs/recipe/recipe.json")
LEARN_DAYS = FIRST_MONITORED_DAY            # the learning phase covers days 0-34
PRIOR_SD_DAYS = 10                          # the population day-to-day SD is worth 10 days of data (D-24)
EARLY_START_RANGE = (10, 28)                # episode start days in the early-episode cohorts (D-24)
SEED_A, SEED_B = 2401, 2402
DETECT_WINDOW = 14                          # R2: detected within 14 days of episode start
RECIPE_KEY = {"resting_hr": "resting_hr_bpm", "hrv_rmssd": "hrv_rmssd_ms",
              "sleep_hours": "sleep_hours", "steps": "steps_per_day"}

log_lines = []


def log(text=""):
    print(text)
    log_lines.append(str(text))


# ----------------------------------------------------------------------------------
# The prior: what the twin knows on day 0 (population values from recipe.json)
# ----------------------------------------------------------------------------------
def load_recipe():
    return json.load(open(RECIPE, encoding="utf-8"))


def prior_for(recipe, sex, measures=MEASURES):
    """Arrays (m0, tau, sigma0, phi), one value per measure, on the monitor's scale
    (log scale for HRV and steps). sex: 'male', 'female' or anything else (= average)."""
    L, D = recipe["levels"], recipe["dynamics"]
    m0, tau, sig, phi = [], [], [], []
    for m in measures:
        key = RECIPE_KEY[m]
        if m == "resting_hr":
            if sex in ("male", "female"):
                mean, sd = L[key][sex]["mean"]["value"], L[key][sex]["sd"]["value"]
            else:
                mean = np.mean([L[key][s]["mean"]["value"] for s in ("male", "female")])
                sd = np.mean([L[key][s]["sd"]["value"] for s in ("male", "female")])
        elif m in ("hrv_rmssd", "steps"):
            mean, sd = L[key]["mu_log"]["value"], L[key]["sigma_log"]["value"]
        else:
            mean, sd = L[key]["mean"]["value"], L[key]["sd"]["value"]
        m0.append(mean)
        tau.append(sd)
        sig.append(D[key]["sd"]["value"])
        phi.append(D[key]["phi"]["value"])
    return np.array(m0), np.array(tau), np.array(sig), np.array(phi)


def n_effective(n, phi):
    """Effective number of independent days in n consecutive AR(1) days with memory phi."""
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    for idx, nn in np.ndenumerate(n):
        if nn >= 1:
            k = np.arange(1, int(nn))
            out[idx] = nn / (1 + 2 * np.sum((1 - k / nn) * phi ** k))
    return out


# ----------------------------------------------------------------------------------
# The learning baseline
# ----------------------------------------------------------------------------------
def learning_state(X, priors, measures=MEASURES):
    """X: (patients, days, measures) on the monitor's scale, NaN = missing.
    priors: (m0, tau, sigma0, phi), each shaped (measures,) or (patients, measures).
    Returns dict of (patients, days, measures) arrays: level, pred_sd, z, learned.
    Only past data is used: day t sees days up to t-8 (no look-ahead)."""
    n_p, T, M = X.shape
    m0, tau, sig0, phi = [np.broadcast_to(np.asarray(a, float), (n_p, M)) for a in priors]
    direction = np.array([DIRECTION[MEASURES.index(m)] for m in measures])
    level = np.full((n_p, T, M), np.nan)
    pred_sd = np.full((n_p, T, M), np.nan)
    learned = np.full((n_p, T, M), np.nan)
    for t in range(min(T, LEARN_DAYS)):
        lo, hi = max(0, t - ROLL_GAP - ROLL_LEN), max(0, t - ROLL_GAP)
        win = X[:, lo:hi, :]
        n = np.sum(~np.isnan(win), axis=1)                                   # (patients, measures)
        ne = np.zeros((n_p, M))
        for j in range(M):                                                   # phi may differ per patient
            for nn in np.unique(n[:, j]):
                rows = n[:, j] == nn
                for ph in np.unique(phi[rows, j]):
                    r2 = rows & (phi[:, j] == ph)
                    ne[r2, j] = n_effective(nn, ph)
        with np.errstate(invalid="ignore", divide="ignore"):
            xbar = np.where(n > 0, np.nansum(win, axis=1) / np.maximum(n, 1), 0.0)
            s2 = np.where(n > 1, np.nansum((win - xbar[:, None, :]) ** 2, axis=1) / np.maximum(n, 1), 0.0)
            sig2 = np.where(n > 1, (PRIOR_SD_DAYS * sig0 ** 2 + ne * s2) / (PRIOR_SD_DAYS + np.maximum(ne - 1, 0)),
                            sig0 ** 2)
            prec = 1 / tau ** 2 + ne / sig2
            post_var = 1 / prec
            lev = post_var * (m0 / tau ** 2 + ne * xbar / sig2)
        level[:, t] = lev
        pred_sd[:, t] = np.sqrt(sig2 + post_var)
        learned[:, t] = (ne / sig2) / prec
    z = direction * (X - level) / pred_sd
    return {"level": level, "pred_sd": pred_sd, "z": z, "learned": learned}


def learning_score(z, min_measures=None, min_smooth_days=SMOOTH):
    """Mean of the measures' z, then the 3-day mean (as monitor.py). With min_measures set
    (real data), a day needs at least that many measures and the 3-day mean needs
    min_smooth_days days (D-21 gap rules)."""
    if min_measures is None:
        score = z.mean(axis=2)
    else:
        n_ok = np.sum(~np.isnan(z), axis=2)
        score = np.where(n_ok >= min_measures, np.nansum(z, axis=2) / np.maximum(n_ok, 1), np.nan)
    smooth = pd.DataFrame(score.T).rolling(SMOOTH, min_periods=min_smooth_days).mean().to_numpy().T.copy()
    smooth[:, LEARN_DAYS:] = np.nan
    return smooth


def learning_flags(score, threshold, k):
    """Run of k days above threshold, counted within days 0-34 only; NaN breaks a run."""
    above = np.nan_to_num(score, nan=-np.inf) > threshold
    f = np.zeros_like(above)
    run = np.zeros(above.shape[0], dtype=int)
    for t in range(min(above.shape[1], LEARN_DAYS)):
        run = np.where(above[:, t], run + 1, 0)
        f[:, t] = run >= k
    return f


def tier_learning_flags(score, high, cfg):
    """Learning-phase flags with each patient's tier setting from learning_config.json."""
    f = np.zeros(score.shape, bool)
    for mask, key in [(~high, "low_risk"), (high, "high_risk")]:
        f[mask] = learning_flags(score[mask], cfg[key]["threshold"], cfg[key]["persistence"])
    return f


def locked_tier_flags(rolling_score, high, twin_cfg):
    """The locked D-18 flags (days 35-59), unchanged."""
    f = np.zeros(rolling_score.shape, bool)
    for mask, key in [(~high, "low_risk"), (high, "high_risk")]:
        f[mask] = flags_from(rolling_score[mask], twin_cfg[key]["threshold"], twin_cfg[key]["persistence"])
    return f


def rolling_band(series):
    """The locked monitor's personal normal (median +/- robust SD of the 28 days ending
    8 days earlier) for one series on the monitor's scale; NaN before day 35."""
    T = len(series)
    med, lo, hi = (np.full(T, np.nan) for _ in range(3))
    for t in range(ROLL_LEN + ROLL_GAP, T):
        b = series[t - ROLL_GAP - ROLL_LEN:t - ROLL_GAP]
        m = np.median(b)
        s = 1.4826 * np.median(np.abs(b - m))
        med[t], lo[t], hi[t] = m, m - s, m + s
    return med, lo, hi


# ----------------------------------------------------------------------------------
# Synthetic early-episode cohorts (validation patients only)
# ----------------------------------------------------------------------------------
def simulate_early_cohort(seed):
    import simulate_wearables as sw
    recipe = load_recipe()
    meta = pd.read_csv("data/processed/meta_val.csv")
    saved = sw.START_RANGE
    sw.START_RANGE = EARLY_START_RANGE
    try:
        sim, eps = sw.simulate_split(meta, recipe, np.random.default_rng(seed), f"val_early_{seed}")
    finally:
        sw.START_RANGE = saved
    eps = eps.set_index("ecg_id")
    ids = eps.index.to_numpy()
    w = sim.set_index(["ecg_id", "day"]).loc[pd.MultiIndex.from_product([ids, range(sw.N_DAYS)])]
    X = w[MEASURES].to_numpy(dtype=float).reshape(len(ids), sw.N_DAYS, len(MEASURES))
    X[:, :, LOG_SCALE] = np.log(np.clip(X[:, :, LOG_SCALE], 1e-6, None))
    return X, eps, sim


def cohort_priors(eps, recipe):
    rows = [prior_for(recipe, s) for s in eps["sex"]]
    return tuple(np.stack([r[i] for r in rows]) for i in range(4))


def learning_eval(flags, score, eps):
    """False alarms per patient-month and detection, learning phase only (days 0-34)."""
    n, T = flags.shape
    start = eps["start_day"].to_numpy()
    has_ep = eps["episode"].to_numpy() == 1
    days = np.arange(T)
    clean = (~has_ep[:, None]) | (days[None, :] < start[:, None])
    scored = (~np.isnan(score)) & (days[None, :] < LEARN_DAYS)
    ev = flags & ~np.concatenate([np.zeros((n, 1), bool), flags[:, :-1]], axis=1)
    fa = 30.0 * (ev & clean & scored).sum() / max((clean & scored).sum(), 1)
    detected = has_ep & (flags & ~clean).any(axis=1)
    res = {"fa_per_month": fa}
    for sev in [1, 2, 3, 4]:
        m = has_ep & (eps["severity"].to_numpy() == sev)
        res[f"det_s{sev}"] = detected[m].mean()
    res["det_mean"] = np.mean([res[f"det_s{s}"] for s in [1, 2, 3, 4]])
    return res


def detect_within(flags, eps, window=DETECT_WINDOW):
    """Per severity: share of episodes with a flag in [start, start + window - 1], and median delay."""
    start = eps["start_day"].to_numpy()
    has_ep = eps["episode"].to_numpy() == 1
    days = np.arange(flags.shape[1])
    inwin = (days[None, :] >= start[:, None]) & (days[None, :] < start[:, None] + window)
    hit = flags & inwin
    det = has_ep & hit.any(axis=1)
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    delay = np.where(det, first - start, np.nan)
    res = {}
    for sev in [1, 2, 3, 4]:
        m = has_ep & (eps["severity"].to_numpy() == sev)
        res[f"det_s{sev}"] = det[m].mean()
        res[f"delay_s{sev}"] = np.nanmedian(delay[m]) if det[m].any() else np.nan
    res["det_mean"] = np.mean([res[f"det_s{s}"] for s in [1, 2, 3, 4]])
    return res


def tune_tier(score, eps, mask, budget):
    best = None
    for k in PERSISTENCE:
        for th in THRESHOLDS:
            r = learning_eval(learning_flags(score[mask], th, k), score[mask], eps[mask])
            if r["fa_per_month"] <= budget and (best is None or r["det_mean"] > best[2]["det_mean"]):
                best = (float(th), int(k), r)
    return best


# ----------------------------------------------------------------------------------
# Real people (PMData): the learning phase on each person's first 35 days
# ----------------------------------------------------------------------------------
def real_check(cfg, twin_cfg):
    from real_data_check import MEASURES as RM, person_series
    recipe = load_recipe()
    data = pd.read_csv("data/processed/pmdata_daily.csv")
    rows = []
    for pid, d in data.groupby("pid"):
        _, X = person_series(d)
        X = X[None, :LEARN_DAYS, :]
        sex = str(d["sex"].iloc[0]).lower()
        st = learning_state(X, prior_for(recipe, sex, RM), RM)
        score = learning_score(st["z"], min_measures=2, min_smooth_days=2)
        row = {"pid": pid, "scored_days": int((~np.isnan(score)).sum())}
        for key in ["low_risk", "high_risk"]:
            f = learning_flags(score, cfg[key]["threshold"], cfg[key]["persistence"])[0]
            row[f"alarms_{key}"] = int((f & ~np.concatenate([[False], f[:-1]])).sum())
        rows.append(row)
    per = pd.DataFrame(rows)
    months = per["scored_days"].sum() / 30.0
    return per, months


# ----------------------------------------------------------------------------------
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    recipe = load_recipe()
    twin_cfg = json.load(open("outputs/twin/twin_config.json"))
    th_risk = twin_cfg["risk_threshold_p_ecg"]
    val_pred = pd.read_csv("outputs/fusion/val_predictions.csv").set_index("ecg_id")

    cohorts = {}
    for name, seed in [("A", SEED_A), ("B", SEED_B)]:
        X, eps, sim = simulate_early_cohort(seed)
        high = val_pred.loc[eps.index, "p_ecg"].to_numpy() >= th_risk
        st = learning_state(X, cohort_priors(eps, recipe))
        cohorts[name] = dict(X=X, eps=eps, sim=sim, high=high, st=st, score=learning_score(st["z"]))

    # ---- tune on cohort A (D-18 rule, per tier)
    A = cohorts["A"]
    cfg = {"locked_on": "synthetic early-episode cohort A (validation patients, seed 2401)",
           "decision": "D-24", "prior_sd_days": PRIOR_SD_DAYS, "learning_days": [0, LEARN_DAYS - 1]}
    log("=" * 78)
    log("LEARNING TWIN (D-24): tuning on cohort A, synthetic, validation patients")
    log("=" * 78)
    for mask, key in [(~A["high"], "low_risk"), (A["high"], "high_risk")]:
        th, k, r = tune_tier(A["score"], A["eps"], mask, twin_cfg[key]["fa_budget"])
        cfg[key] = {"threshold": th, "persistence": k, "fa_budget": twin_cfg[key]["fa_budget"]}
        log(f"{key:10s} threshold {th:.2f}, k = {k}: FA/month {r['fa_per_month']:.2f} "
            f"(budget {twin_cfg[key]['fa_budget']:.2f}), learning-phase detection {r['det_mean']:.1%}")
    json.dump(cfg, open(OUT / "learning_config.json", "w"), indent=2)

    # ---- check on cohort B
    B = cohorts["B"]
    lf = tier_learning_flags(B["score"], B["high"], cfg)
    roll, _ = daily_scores(B["X"], "rolling")
    locked = locked_tier_flags(roll, B["high"], twin_cfg)
    r_learn_phase = learning_eval(lf, B["score"], B["eps"])
    now = detect_within(locked, B["eps"])
    new = detect_within(locked | lf, B["eps"])
    log("\n" + "=" * 78)
    log("CHECK ON COHORT B (synthetic, seed 2402, not used for tuning)")
    log("=" * 78)
    r1 = r_learn_phase["fa_per_month"] <= 0.6
    log(f"R1 learning-phase false alarms: {r_learn_phase['fa_per_month']:.2f} per patient-month "
        f"(rule <= 0.60) -> {'PASS' if r1 else 'FAIL'}")
    log(f"\nDetection within {DETECT_WINDOW} days of episode start (episodes start days {EARLY_START_RANGE[0]}-"
        f"{EARLY_START_RANGE[1]}):")
    log(f"{'system':34s} {'+3':>6s} {'+5':>6s} {'+8':>6s} {'+12':>6s} {'mean':>7s}")
    for name, r in [("current (locked, blind before day 35)", now), ("with learning twin (D-24)", new)]:
        log(f"{name:34s} " + " ".join(f"{100 * r[f'det_s{s}']:5.1f}%" for s in [1, 2, 3, 4])
            + f" {100 * r['det_mean']:6.1f}%")
    gain = 100 * (new["det_mean"] - now["det_mean"])
    r2 = gain >= 20
    log(f"R2 gain in mean detection: {gain:+.1f} points (rule >= +20) -> {'PASS' if r2 else 'FAIL'}")
    log("Median delay with the learning twin (days): "
        + ", ".join(f"+{b} bpm {new[f'delay_s{s}']:.0f}" for s, b in zip([1, 2, 3, 4], [3, 5, 8, 12])))

    # ---- real people
    r3 = None
    if Path("data/processed/pmdata_daily.csv").exists():
        per, months = real_check(cfg, twin_cfg)
        per.to_csv(OUT / "real_learning_per_person.csv", index=False)
        log("\n" + "=" * 78)
        log("REAL PEOPLE (PMData): learning phase = each person's first 35 days, 3 measures")
        log("=" * 78)
        log(f"People {len(per)}, scored days {per['scored_days'].sum()} (= {months:.1f} person-months); "
            "every alarm counted as false")
        r3 = True
        for key in ["low_risk", "high_risk"]:
            rate = per[f"alarms_{key}"].sum() / months
            limit = 2 * twin_cfg[key]["fa_budget"]
            ok = rate <= limit
            r3 &= ok
            log(f"{key:10s} {per[f'alarms_{key}'].sum():3d} alarms = {rate:.2f} per person-month "
                f"(rule <= {limit:.2f}) -> {'PASS' if ok else 'FAIL'}")
        log(f"R3 -> {'PASS' if r3 else 'FAIL'}")
    else:
        log("\nPMData file not present: R3 not run.")

    verdict = r1 and r2 and (r3 is not False)
    log("\n" + ("ALL RULES PASS: learning-phase alarms are switched on (D-24)." if verdict else
                "A RULE FAILED: fallback applies, learning phase shown as a band only, no alarms before day 35."))
    cfg["alarms_enabled"] = bool(verdict)
    json.dump(cfg, open(OUT / "learning_config.json", "w"), indent=2)

    # ---- chart 1: detection within 14 days, current vs learning
    sizes = [3, 5, 8, 12]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for r, color, name in [(now, "#9a9a94", "Current (blind before day 35)"),
                           (new, "#2a78d6", "With the learning twin")]:
        ax.plot(sizes, [100 * r[f"det_s{s}"] for s in [1, 2, 3, 4]], color=color, linewidth=2, marker="o",
                markersize=8, markeredgecolor="white", markeredgewidth=1.5, label=name)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"+{s}" for s in sizes])
    ax.set_xlabel("Injected rise in resting heart rate (bpm)")
    ax.set_ylabel(f"Caught within {DETECT_WINDOW} days (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.set_title(f"Episodes starting on days {EARLY_START_RANGE[0]}-{EARLY_START_RANGE[1]} "
                 "(SYNTHETIC, cohort B)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "detection_early.png", dpi=150)
    plt.close(fig)

    # ---- chart 2: the expected range narrowing (one cohort-B patient without an episode)
    eps = B["eps"]
    i = int(np.where(eps["episode"].to_numpy() == 0)[0][0])
    vals = B["X"][i, :, 0]
    lev, sd = B["st"]["level"][i, :, 0], B["st"]["pred_sd"][i, :, 0]
    med, lo, hi = rolling_band(vals)
    days = np.arange(len(vals))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.fill_between(days[:LEARN_DAYS], (lev - sd)[:LEARN_DAYS], (lev + sd)[:LEARN_DAYS], color="#2a78d6",
                    alpha=0.18, linewidth=0, label="Learning twin: expected range (days 0-34)")
    ax.fill_between(days, lo, hi, color="#6b8f71", alpha=0.25, linewidth=0,
                    label="Locked personal baseline (from day 35)")
    ax.plot(days, vals, color="#1D2733", linewidth=1.2, marker="o", markersize=2.5, label="Resting HR")
    ax.axvline(LEARN_DAYS - 0.5, color="#666", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Day")
    ax.set_ylabel("Resting HR (bpm)")
    ax.set_title("The twin learns one patient's normal (SYNTHETIC patient, cohort B)", loc="left", fontsize=11)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "band_example.png", dpi=150)
    plt.close(fig)

    log(f"\nSaved {OUT / 'learning_config.json'}, learning_summary.txt and charts in {OUT}")
    with open(OUT / "learning_summary.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
