"""
real_data_check.py: run the LOCKED wearable monitor on REAL people (DECISIONS.md D-21).

Why: every monitor result so far comes from simulated data. PMData contains months of
real Fitbit data from 16 people. None of them has a labelled cardiac episode, so every
alarm here is counted as a FALSE alarm. That makes this an upper bound on the real-world
false-alarm rate: some alarms may be real events (illness, travel, a bad week) that we
cannot verify.

Rules fixed BEFORE running (not tuned on these results):
  - Settings: exactly the locked ones in monitor_config.json (uniform, D-17) and
    twin_config.json (low-risk and high-risk tiers, D-18). Nothing is changed.
  - Measures: resting HR, sleep, steps. PMData has no HRV, so the daily score is the
    average of 3 measures instead of 4 (a stated difference from the synthetic test).
  - Same baseline as monitor.py: median and 1.4826 x MAD of the 28 days ending 8 days
    before each day; steps on a log scale; unhealthy direction counted positive.
  - Real data has gaps, which the synthetic data did not. Gap rules:
      a measure's baseline needs >= 14 valid days in its 28-day window;
      a day's score needs >= 2 of the 3 measures;
      the 3-day smoothed score needs >= 2 of the 3 days;
      a missing day breaks a run of consecutive flagged days.
  - Monitoring starts on each person's day 35 (same as the synthetic test).
  - Reported: alarm events per person-month of monitored days (30 days with a score).

Caveat: the simulation recipe was fitted on PMData (D-14, D-15), so this is not fully
independent data. It does test the monitor on real-world quirks the simulator lacks:
gaps, device wear habits, holidays and illnesses.

Input:  data/processed/pmdata_daily.csv   (from explore_wearables.py)
Output: outputs/real_check/real_check_summary.txt, per_person.csv, example_real_person.png
Run:    python real_data_check.py
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monitor import FIRST_MONITORED_DAY, ROLL_GAP, ROLL_LEN, SMOOTH, flags_from

OUT = Path("outputs/real_check")
OUT.mkdir(parents=True, exist_ok=True)
MEASURES = ["resting_hr", "sleep_hours", "steps"]
DIRECTION = np.array([+1, -1, -1])
LOG_SCALE = np.array([False, False, True])
MIN_BASE_DAYS = 14
MIN_MEASURES = 2
MIN_SMOOTH_DAYS = 2

log_lines = []


def log(text=""):
    print(text)
    log_lines.append(str(text))


def person_series(d):
    """One person's data on a continuous calendar (missing days become NaN)."""
    d = d.assign(date=pd.to_datetime(d["date"])).set_index("date").sort_index()
    full = pd.date_range(d.index.min(), d.index.max(), freq="D")
    X = d.reindex(full)[MEASURES].to_numpy(dtype=float)
    X[:, LOG_SCALE] = np.log(np.clip(X[:, LOG_SCALE], 1.0, None))
    return full, X


def daily_score(X):
    """Same score as monitor.py (rolling method), with the gap rules above."""
    T, M = X.shape
    z = np.full((T, M), np.nan)
    warnings.filterwarnings("ignore", message="All-NaN slice")  # windows with no data -> NaN, handled below
    for t in range(ROLL_LEN + ROLL_GAP, T):
        base = X[t - ROLL_GAP - ROLL_LEN:t - ROLL_GAP]
        ok = np.sum(~np.isnan(base), axis=0) >= MIN_BASE_DAYS
        med = np.nanmedian(base, axis=0)
        sp = np.maximum(1.4826 * np.nanmedian(np.abs(base - med), axis=0), 1e-6)
        z[t] = np.where(ok, (X[t] - med) / sp, np.nan)
    z = z * DIRECTION
    n_ok = np.sum(~np.isnan(z), axis=1)
    z_sum = np.nansum(z, axis=1)
    score = np.where(n_ok >= MIN_MEASURES, z_sum / np.maximum(n_ok, 1), np.nan)
    smooth = pd.Series(score).rolling(SMOOTH, min_periods=MIN_SMOOTH_DAYS).mean().to_numpy()
    return smooth, z


def alarm_events(flags):
    starts = flags & ~np.concatenate([[False], flags[:-1]])
    return int(starts.sum())


def main():
    data = pd.read_csv("data/processed/pmdata_daily.csv")
    uni = json.load(open("outputs/monitor/monitor_config.json"))["methods"]["rolling"]
    twin = json.load(open("outputs/twin/twin_config.json"))
    settings = {
        "uniform (D-17)": (uni["threshold"], uni["persistence"], 0.5),
        "low-risk tier (D-18)": (twin["low_risk"]["threshold"], twin["low_risk"]["persistence"],
                                 twin["low_risk"]["fa_budget"]),
        "high-risk tier (D-18)": (twin["high_risk"]["threshold"], twin["high_risk"]["persistence"],
                                  twin["high_risk"]["fa_budget"]),
    }

    rows, keep = [], {}
    for pid, d in data.groupby("pid"):
        dates, X = person_series(d)
        score, z = daily_score(X)
        monitored = np.arange(len(score)) >= FIRST_MONITORED_DAY
        scored_days = int((monitored & ~np.isnan(score)).sum())
        row = {"pid": pid, "calendar_days": len(score), "monitored_days_with_score": scored_days}
        for name, (th, k, _) in settings.items():
            f = flags_from(np.nan_to_num(score, nan=-np.inf)[None, :], th, k)[0]
            row[f"alarms: {name}"] = alarm_events(f)
            row[f"flagged days: {name}"] = int(f.sum())
        rows.append(row)
        keep[pid] = (dates, X, score, settings["uniform (D-17)"])
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_person.csv", index=False)

    used = per[per["monitored_days_with_score"] > 0]
    months = used["monitored_days_with_score"].sum() / 30.0
    log("=" * 78)
    log("LOCKED MONITOR ON REAL PEOPLE (PMData, 3 measures, no labelled episodes)")
    log("=" * 78)
    log(f"People with at least one scored monitored day: {len(used)} of {len(per)}")
    log(f"Scored monitored days: {used['monitored_days_with_score'].sum()} "
        f"(= {months:.1f} person-months)")
    log("\nAll alarms counted as false (upper bound):")
    log(f"{'setting':24s} {'thr':>5s} {'k':>2s} {'alarms':>7s} {'per person-month':>17s} "
        f"{'synthetic budget':>17s} {'people with >=1':>16s}")
    for name, (th, k, budget) in settings.items():
        a = used[f"alarms: {name}"]
        log(f"{name:24s} {th:5.2f} {k:2d} {a.sum():7d} {a.sum() / months:17.2f} {budget:17.2f} "
            f"{(a > 0).sum():10d} of {len(used)}")
    log("\nPer person (uniform setting): " + ", ".join(
        f"{r.pid} {r['alarms: uniform (D-17)']}/{r.monitored_days_with_score}d" for _, r in used.iterrows()))
    log("\nCaveats: PMData was used to fit the simulation recipe (not independent); no HRV; "
        "some alarms may be real events that cannot be verified.")

    # Chart: the person with the most alarms under the uniform setting
    worst = used.sort_values("alarms: uniform (D-17)", ascending=False).iloc[0]["pid"]
    dates, X, score, (th, k, _) = keep[worst]
    f = flags_from(np.nan_to_num(score, nan=-np.inf)[None, :], th, k)[0]
    fig, axes = plt.subplots(4, 1, figsize=(9, 7.5), sharex=True)
    for ax, m, name in zip(axes[:3], range(3), ["Resting HR (bpm)", "Sleep (h)", "Steps"]):
        vals = np.exp(X[:, m]) if LOG_SCALE[m] else X[:, m]
        ax.plot(dates, vals, color="#2a78d6", linewidth=1)
        ax.set_ylabel(name, fontsize=9)
    axes[3].plot(dates, score, color="#333", linewidth=1)
    axes[3].axhline(th, color="#d03b3b", linestyle="--", linewidth=1)
    axes[3].set_ylabel("Score (SD)", fontsize=9)
    for ax in axes:
        for t in np.where(f)[0]:
            ax.axvspan(dates[t], dates[t] + pd.Timedelta(days=1), color="#d03b3b", alpha=0.15, linewidth=0)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0].set_title(f"Real person {worst} (PMData): flagged days shaded, uniform setting", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "example_real_person.png", dpi=150)
    plt.close(fig)
    log(f"\nSaved {OUT / 'per_person.csv'}, {OUT / 'example_real_person.png'}")
    with open(OUT / "real_check_summary.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
