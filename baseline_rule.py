"""
baseline_rule.py: compare the wearable monitor with simpler alternatives (DECISIONS.md D-21).

Question: does our monitor (4 measures, rolling personal baseline, persistence) do better
than something a clinician could apply by eye, or than resting HR alone?

Compared on VALIDATION patients only (the test patients stay untouched), all synthetic:
  1. Simple rule, +10 bpm: resting HR at least 10 bpm above the person's own median
     (same 28-day window ending 8 days earlier) on 3 consecutive days.
  2. Simple rule, +5 bpm: the same with 5 bpm.
     (Both rules are our own illustrative choices, fixed before running.
      They are NOT a published clinical guideline.)
  3. Resting-HR-only monitor: our scoring on resting HR alone, tuned exactly like D-17
     (highest mean detection with at most 0.5 false alarms per patient-month).
  4. Our monitor (D-17 locked settings, 4 measures).
Rows 3 and 4 share the same false-alarm budget, so they compare fairly; rows 1 and 2
are reported at whatever false-alarm rate they happen to produce.

Input:  data/processed/wearables_val.csv, episodes_val.csv; outputs/monitor/monitor_config.json
Output: outputs/baseline/baseline_val.csv, baseline_val.txt
Run:    python baseline_rule.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from monitor import (MAX_FA_PER_MONTH, PERSISTENCE, ROLL_GAP, ROLL_LEN, SMOOTH, THRESHOLDS,
                     daily_scores, evaluate, flags_from, load, robust)

OUT = Path("outputs/baseline")
OUT.mkdir(parents=True, exist_ok=True)


def rhr_rule_flags(X, bpm, days):
    """Resting HR >= rolling median + bpm on `days` consecutive days (raw bpm, no smoothing)."""
    n, T, _ = X.shape
    above = np.zeros((n, T), bool)
    for t in range(ROLL_LEN + ROLL_GAP, T):
        med = np.median(X[:, t - ROLL_GAP - ROLL_LEN:t - ROLL_GAP, 0], axis=1)
        above[:, t] = X[:, t, 0] >= med + bpm
    return flags_from(above.astype(float), 0.5, days)   # flags_from counts runs of "above"


def rhr_only_score(X):
    """Our scoring rule, resting HR only: robust z vs rolling baseline, 3-day mean."""
    n, T, _ = X.shape
    z = np.full((n, T), np.nan)
    for t in range(ROLL_LEN + ROLL_GAP, T):
        med, sp = robust(X[:, t - ROLL_GAP - ROLL_LEN:t - ROLL_GAP, 0], axis=1)
        z[:, t] = (X[:, t, 0] - med) / sp
    return pd.DataFrame(z.T).rolling(SMOOTH, min_periods=SMOOTH).mean().to_numpy().T


def main():
    X, eps = load("val")
    rows = []

    for bpm in [10, 5]:
        rows.append({"method": f"Simple rule: RHR >= normal + {bpm} bpm, 3 days", "settings": "fixed in advance",
                     **evaluate(rhr_rule_flags(X, bpm, 3), eps)})

    s = rhr_only_score(X)
    best = None
    for k in PERSISTENCE:
        for th in THRESHOLDS:
            r = evaluate(flags_from(s, th, k), eps)
            if r["fa_per_month"] <= MAX_FA_PER_MONTH and (best is None or r["det_mean"] > best[2]["det_mean"]):
                best = (th, k, r)
    rows.append({"method": "Resting-HR-only monitor (tuned like D-17)",
                 "settings": f"threshold {best[0]:.2f}, k {best[1]}", **best[2]})

    c = json.load(open("outputs/monitor/monitor_config.json"))["methods"]["rolling"]
    score, _ = daily_scores(X, "rolling")
    rows.append({"method": "Our monitor, 4 measures (D-17, locked)",
                 "settings": f"threshold {c['threshold']:.2f}, k {c['persistence']}",
                 **evaluate(flags_from(score, c["threshold"], c["persistence"]), eps)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "baseline_val.csv", index=False)
    lines = ["COMPARISON ON VALIDATION (synthetic wearables and episodes)",
             f"{'method':48s} {'FA/month':>8s} | +3 bpm  +5 bpm  +8 bpm  +12 bpm | mean"]
    for _, r in df.iterrows():
        cells = "  ".join(f"{100 * r[f'det_s{s}']:5.1f}%" for s in [1, 2, 3, 4])
        lines.append(f"{r['method']:48s} {r['fa_per_month']:8.2f} | {cells} | {100 * r['det_mean']:.1f}%")
        lines.append(f"{'':48s} settings: {r['settings']}")
    print("\n".join(lines))
    with open(OUT / "baseline_val.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
