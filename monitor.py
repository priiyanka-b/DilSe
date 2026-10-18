"""
monitor.py: the personal-baseline wearable monitor (DECISIONS.md D-17).

For every patient and every day it asks: "is this person unusual compared with
their OWN normal, in the unhealthy direction, and is it lasting?"

DAILY DEVIATION SCORE
  For each measure m (resting HR, HRV, sleep, steps):
    z_m(day) = direction_m * (value - baseline_m) / spread_m
      direction: resting HR up is bad (+1); HRV, sleep and steps down are bad (-1)
      HRV and steps are compared on a log scale (changes are relative)
      baseline = median, spread = 1.4826 * MAD (robust to odd days)
  score(day) = average of the four z_m(day)

TWO METHODS (both use the same score, so the comparison is fair)
  A "fixed"   baseline = days 0-27 once and for all
              flag when the 3-day average score > threshold
  B "rolling" baseline = the 28 days ending 8 days ago (days t-35 ... t-8),
              so it follows slow natural drift but not the latest week
              flag when the 3-day average score > threshold
              on k consecutive days (persistence)

SCORING (only days 35-59 are monitored; episodes start on days 36-53)
  detected   an episode patient whose first flag is on or after the episode start
  delay      days from episode start to that first flag
  false alarm one alarm EVENT (a run of consecutive flagged days) that starts on a
             day with no episode: any day for patients without an episode, or a
             day before the start for patients with one
  reported as false alarms per patient-month (30 monitored days)

TUNING RULE (fixed before looking at results; D-17)
  On the VALIDATION patients only: for each method, choose the settings with the
  highest average detection rate across the four severities, subject to at most
  MAX_FA_PER_MONTH false alarms per patient-month. The chosen settings are saved
  and must not be changed after the test patients are scored.

Inputs:  data/processed/wearables_val.csv, episodes_val.csv
Outputs: outputs/monitor/monitor_config.json      locked settings
         outputs/monitor/tuning_grid.csv          every setting tried
         outputs/monitor/monitor_val_summary.txt
         outputs/monitor/detection_vs_size.png
         outputs/monitor/tradeoff.png
Run:     python monitor.py
         python monitor.py --split test   (ONLY in the final evaluation, with locked settings)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROC = Path("data/processed")
OUT = Path("outputs/monitor")
OUT.mkdir(parents=True, exist_ok=True)

MEASURES = ["resting_hr", "hrv_rmssd", "sleep_hours", "steps"]
DIRECTION = np.array([+1, -1, -1, -1])
LOG_SCALE = np.array([False, True, False, True])
FIRST_MONITORED_DAY = 35
FIXED_BASELINE = (0, 28)          # days 0-27
ROLL_LEN, ROLL_GAP = 28, 7        # baseline = 28 days ending 8 days ago
SMOOTH = 3                        # days in the rolling average of the score
MAX_FA_PER_MONTH = 0.5            # D-17 tuning constraint (mentor's choice, adjustable)
THRESHOLDS = np.round(np.arange(0.0, 4.01, 0.05), 2)
PERSISTENCE = [1, 2, 3, 4, 5, 6]
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9a94"

log_lines = []


def log(text=""):
    print(text)
    log_lines.append(str(text))


def load(split):
    w = pd.read_csv(PROC / f"wearables_{split}.csv")
    eps = pd.read_csv(PROC / f"episodes_{split}.csv").set_index("ecg_id")
    ids = eps.index.to_numpy()
    w = w.set_index(["ecg_id", "day"]).loc[pd.MultiIndex.from_product([ids, range(60)])]
    X = w[MEASURES].to_numpy(dtype=float).reshape(len(ids), 60, len(MEASURES))
    X[:, :, LOG_SCALE] = np.log(np.clip(X[:, :, LOG_SCALE], 1e-6, None))
    return X, eps


def robust(x, axis):
    med = np.nanmedian(x, axis=axis)
    mad = 1.4826 * np.nanmedian(np.abs(x - np.expand_dims(med, axis)), axis=axis)
    return med, np.maximum(mad, 1e-6)


def daily_scores(X, method):
    """Return score[patient, day] (NaN where not computable)."""
    n, T, M = X.shape
    z = np.full((n, T, M), np.nan)
    if method == "fixed":
        med, sp = robust(X[:, FIXED_BASELINE[0]:FIXED_BASELINE[1], :], axis=1)
        z = (X - med[:, None, :]) / sp[:, None, :]
    else:
        for t in range(ROLL_LEN + ROLL_GAP, T):
            base = X[:, t - ROLL_GAP - ROLL_LEN:t - ROLL_GAP, :]
            med, sp = robust(base, axis=1)
            z[:, t, :] = (X[:, t, :] - med) / sp
    z = z * DIRECTION
    score = z.mean(axis=2)
    smooth = pd.DataFrame(score.T).rolling(SMOOTH, min_periods=SMOOTH).mean().to_numpy().T
    return smooth, z


def flags_from(score, threshold, k):
    above = score > threshold
    if k == 1:
        f = above
    else:
        f = np.zeros_like(above)
        run = np.zeros(above.shape[0], dtype=int)
        for t in range(above.shape[1]):
            run = np.where(above[:, t], run + 1, 0)
            f[:, t] = run >= k
    f[:, :FIRST_MONITORED_DAY] = False
    return f


def evaluate(flags, eps):
    n, T = flags.shape
    start = eps["start_day"].to_numpy()
    has_ep = eps["episode"].to_numpy() == 1
    days = np.arange(T)
    clean_day = (~has_ep[:, None]) | (days[None, :] < start[:, None])
    monitored = days[None, :] >= FIRST_MONITORED_DAY
    event_start = flags & ~np.concatenate([np.zeros((n, 1), bool), flags[:, :-1]], axis=1)
    fa_events = (event_start & clean_day & monitored).sum()
    clean_days = (clean_day & monitored).sum()
    fa_rate = 30.0 * fa_events / clean_days

    after = flags & ~clean_day
    detected = has_ep & after.any(axis=1)
    first = np.where(after.any(axis=1), after.argmax(axis=1), -1)
    delay = np.where(detected, first - start, np.nan)
    res = {"fa_per_month": fa_rate}
    for sev in [1, 2, 3, 4]:
        m = has_ep & (eps["severity"].to_numpy() == sev)
        res[f"det_s{sev}"] = detected[m].mean()
        res[f"delay_s{sev}"] = np.nanmedian(delay[m]) if detected[m].any() else np.nan
    res["det_mean"] = np.mean([res[f"det_s{s}"] for s in [1, 2, 3, 4]])
    return res


def tune(X, eps):
    rows = []
    for method in ["fixed", "rolling"]:
        score, _ = daily_scores(X, method)
        for k in (PERSISTENCE if method == "rolling" else [1]):
            for th in THRESHOLDS:
                rows.append({"method": method, "threshold": th, "persistence": k,
                             **evaluate(flags_from(score, th, k), eps)})
    return pd.DataFrame(rows)


def main(split):
    X, eps = load(split)
    cfg_path = OUT / "monitor_config.json"

    if split == "val":
        grid = tune(X, eps)
        grid.to_csv(OUT / "tuning_grid.csv", index=False)
        chosen = {}
        for method, g in grid.groupby("method"):
            ok = g[g["fa_per_month"] <= MAX_FA_PER_MONTH]
            best = ok.sort_values(["det_mean", "fa_per_month"], ascending=[False, True]).iloc[0]
            chosen[method] = {"threshold": float(best["threshold"]), "persistence": int(best["persistence"])}
        json.dump({"locked_on": "validation patients", "max_fa_per_month": MAX_FA_PER_MONTH,
                   "smooth_days": SMOOTH, "rolling_baseline_days": ROLL_LEN, "rolling_gap_days": ROLL_GAP,
                   "fixed_baseline_days": list(FIXED_BASELINE), "methods": chosen},
                  open(cfg_path, "w"), indent=2)
    else:
        chosen = json.load(open(cfg_path))["methods"]
        log("Using LOCKED settings from monitor_config.json (tuned on validation).")

    log("=" * 78)
    log(f"MONITOR RESULTS ON {split.upper()} (synthetic wearable data; synthetic episodes)")
    log("=" * 78)
    log(f"Constraint: at most {MAX_FA_PER_MONTH} false alarms per patient-month")
    log(f"{'method':9s} {'thr':>4s} {'k':>2s} {'FA/month':>9s} | detection rate by RHR rise (median delay, days)")
    log(f"{'':27s}|   +3 bpm        +5 bpm        +8 bpm       +12 bpm")
    results = {}
    for method, c in chosen.items():
        score, _ = daily_scores(X, method)
        r = evaluate(flags_from(score, c["threshold"], c["persistence"]), eps)
        results[method] = r
        cells = "  ".join(f"{r[f'det_s{s}']:5.1%} ({r[f'delay_s{s}']:>3.0f}d)" if not np.isnan(r[f'delay_s{s}'])
                          else f"{r[f'det_s{s}']:5.1%}  (  -)" for s in [1, 2, 3, 4])
        log(f"{method:9s} {c['threshold']:4.1f} {c['persistence']:2d} {r['fa_per_month']:9.2f} | {cells}")

    # Chart 1: detection rate vs size of change
    sizes = [3, 5, 8, 12]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for method, color, name in [("fixed", GREY, "A: fixed baseline"), ("rolling", BLUE, "B: rolling baseline")]:
        r = results[method]
        ax.plot(sizes, [100 * r[f"det_s{s}"] for s in [1, 2, 3, 4]], color=color, linewidth=2,
                marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                label=f"{name} ({r['fa_per_month']:.2f} false alarms / patient-month)")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"+{s}" for s in sizes])
    ax.set_xlabel("Injected rise in resting heart rate (bpm)")
    ax.set_ylabel("Episodes detected (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.set_title(f"Detection limits on {split} (SYNTHETIC episodes)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / f"detection_vs_size_{split}.png", dpi=150)
    plt.close(fig)

    # Chart 2 (validation only): the trade-off curve
    if split == "val":
        grid = pd.read_csv(OUT / "tuning_grid.csv")
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        a = grid[grid["method"] == "fixed"].sort_values("fa_per_month")
        b = grid[(grid["method"] == "rolling") & (grid["persistence"] == chosen["rolling"]["persistence"])].sort_values("fa_per_month")
        ax.plot(a["fa_per_month"], 100 * a["det_mean"], color=GREY, linewidth=2, label="A: fixed baseline")
        ax.plot(b["fa_per_month"], 100 * b["det_mean"], color=BLUE, linewidth=2,
                label=f"B: rolling baseline (k = {chosen['rolling']['persistence']})")
        ax.axvline(MAX_FA_PER_MONTH, color=ORANGE, linewidth=1, linestyle="--")
        ax.text(MAX_FA_PER_MONTH, 3, " limit", color=ORANGE, fontsize=9)
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 100)
        ax.set_xlabel("False alarms per patient-month")
        ax.set_ylabel("Average detection rate (%)")
        ax.set_title("Trade-off between catching episodes and false alarms (validation)", loc="left", fontsize=11)
        ax.grid(alpha=0.25, linewidth=0.6)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT / "tradeoff_val.png", dpi=150)
        plt.close(fig)

    log(f"\nSaved {cfg_path} and charts in {OUT}")
    with open(OUT / f"monitor_{split}_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    main(ap.parse_args().split)
