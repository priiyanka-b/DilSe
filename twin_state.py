"""
twin_state.py: the twin-state algorithm that fuses the two streams (DECISIONS.md D-18).

  ECG stream      -> P(abnormal ECG) from the 4-CNN ensemble (D-12), REAL data
  wearable stream -> daily deviation score from the monitor (D-17), SYNTHETIC data

STEP 1  Risk tier from the ECG
        high risk if P(abnormal) >= the threshold giving 90% specificity on
        validation (the operating point reported throughout the project).
STEP 2  Risk-adjusted monitoring (the fusion)
        the ECG risk tier sets HOW SENSITIVE the wearable monitor is:
        high-risk patients get their own, more sensitive alarm settings.
        Rule (fixed before results): the overall false-alarm budget stays at
        0.5 per patient-month, but a high-risk patient may receive twice as many
        false alarms as a low-risk patient. Each tier's settings are then tuned
        on validation to catch as many synthetic episodes as possible within
        its own budget.
STEP 3  Twin state, every day, from a simple table
                      no wearable flag    wearable flag
        low risk      STABLE              WATCH
        high risk     WATCH               ESCALATE

IMPORTANT LIMITATION (stated, not hidden)
  The simulated episodes were assigned independently of the ECG (D-13), so this
  data CANNOT show that risk-adjusted monitoring improves outcomes. It can only
  show what the adjustment changes: alarms are moved towards high-risk patients
  at a stated false-alarm cost.

Inputs:  outputs/fusion/val_predictions.csv  (validation ECG risk, p_ecg)
         data/processed/wearables_val.csv, episodes_val.csv
         outputs/monitor/monitor_config.json (uniform D-17 settings, for comparison)
Outputs: outputs/twin/twin_config.json       locked settings
         outputs/twin/twin_states_val.csv    one row per patient: tier, flags, state
         outputs/twin/twin_val_summary.txt
         outputs/twin/state_counts_val.png
Run:     python twin_state.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from monitor import PERSISTENCE, THRESHOLDS, daily_scores, evaluate, flags_from, load

OUT = Path("outputs/twin")
OUT.mkdir(parents=True, exist_ok=True)
TOTAL_FA_BUDGET = 0.5      # same overall budget as D-17
HIGH_TO_LOW_RATIO = 2.0    # a high-risk patient may get twice the false alarms (design choice)
STATES = ["STABLE", "WATCH", "ESCALATE"]
COLORS = {"STABLE": "#9a9a94", "WATCH": "#eda100", "ESCALATE": "#d03b3b"}

log_lines = []


def log(text=""):
    print(text)
    log_lines.append(str(text))


def best_settings(score, eps, mask, budget):
    """Highest mean detection within one tier, subject to the tier's false-alarm budget."""
    best = None
    for k in PERSISTENCE:
        for th in THRESHOLDS:
            r = evaluate(flags_from(score[mask], th, k), eps[mask])
            if r["fa_per_month"] <= budget and (best is None or r["det_mean"] > best[2]["det_mean"]):
                best = (float(th), int(k), r)
    return best


def main():
    X, eps = load("val")
    pred = pd.read_csv("outputs/fusion/val_predictions.csv").set_index("ecg_id").loc[eps.index]
    p_ecg = pred["p_ecg"].to_numpy()

    # STEP 1: risk tier
    fpr, tpr, th = roc_curve(pred["label"], p_ecg)
    i = np.where(fpr <= 0.10)[0][-1]
    risk_threshold = float(th[i])
    high = p_ecg >= risk_threshold
    share_high = high.mean()

    # STEP 2: tier budgets that keep the overall budget at TOTAL_FA_BUDGET
    b_low = TOTAL_FA_BUDGET / (1 - share_high + HIGH_TO_LOW_RATIO * share_high)
    b_high = HIGH_TO_LOW_RATIO * b_low
    score, _ = daily_scores(X, "rolling")
    th_low, k_low, r_low = best_settings(score, eps, ~high, b_low)
    th_high, k_high, r_high = best_settings(score, eps, high, b_high)

    flags = np.zeros(score.shape, bool)
    flags[~high] = flags_from(score[~high], th_low, k_low)
    flags[high] = flags_from(score[high], th_high, k_high)

    # Comparison: the uniform D-17 settings for everyone
    uni = json.load(open("outputs/monitor/monitor_config.json"))["methods"]["rolling"]
    flags_uni = flags_from(score, uni["threshold"], uni["persistence"])

    # STEP 3: twin state per patient-day; patient summary = worst state seen
    day_state = np.where(flags & high[:, None], 2, np.where(flags | high[:, None], 1, 0))
    worst = day_state[:, 35:].max(axis=1)
    out = pd.DataFrame({"ecg_id": eps.index, "p_ecg": p_ecg, "risk_tier": np.where(high, "high", "low"),
                        "ecg_label": pred["label"].to_numpy(), "episode": eps["episode"].to_numpy(),
                        "severity": eps["severity"].to_numpy(),
                        "first_flag_day": np.where(flags.any(1), flags.argmax(1), -1),
                        "worst_state": [STATES[s] for s in worst]})
    out.to_csv(OUT / "twin_states_val.csv", index=False)

    json.dump({"locked_on": "validation patients",
               "risk_threshold_p_ecg": risk_threshold, "risk_threshold_rule": "90% specificity on validation",
               "total_fa_budget": TOTAL_FA_BUDGET, "high_to_low_ratio": HIGH_TO_LOW_RATIO,
               "low_risk": {"threshold": th_low, "persistence": k_low, "fa_budget": b_low},
               "high_risk": {"threshold": th_high, "persistence": k_high, "fa_budget": b_high}},
              open(OUT / "twin_config.json", "w"), indent=2)

    # Report
    log("=" * 80)
    log("TWIN STATE ON VALIDATION (ECG risk: real data | wearables and episodes: SYNTHETIC)")
    log("=" * 80)
    log(f"Step 1  risk threshold p_ecg >= {risk_threshold:.3f} (90% specificity) -> "
        f"{high.sum()} high-risk ({100 * share_high:.1f}%), {(~high).sum()} low-risk")
    log(f"Step 2  false-alarm budgets: low-risk {b_low:.2f}, high-risk {b_high:.2f} per patient-month")
    log(f"        low-risk settings  threshold {th_low:.2f}, k {k_low}")
    log(f"        high-risk settings threshold {th_high:.2f}, k {k_high}")

    def row(name, f, mask):
        r = evaluate(f[mask], eps[mask])
        cells = "  ".join(f"{100 * r[f'det_s{s}']:5.1f}%" for s in [1, 2, 3, 4])
        log(f"  {name:34s} FA/month {r['fa_per_month']:.2f} | {cells} | mean {100 * r['det_mean']:.1f}%")

    log("\nDetection of synthetic episodes:            +3 bpm  +5 bpm  +8 bpm  +12 bpm")
    for tier, mask in [("high-risk", high), ("low-risk", ~high), ("all", np.ones_like(high))]:
        row(f"{tier}: uniform (D-17)", flags_uni, mask)
        row(f"{tier}: risk-adjusted (D-18)", flags, mask)

    counts = out["worst_state"].value_counts().reindex(STATES).fillna(0).astype(int)
    log("\nWorst twin state per patient over days 35-59: "
        + ", ".join(f"{s} {counts[s]} ({100 * counts[s] / len(out):.1f}%)" for s in STATES))
    esc = out[out["worst_state"] == "ESCALATE"]
    log(f"ESCALATE patients with an injected episode: {esc['episode'].mean():.1%}; "
        f"with an abnormal ECG: {esc['ecg_label'].mean():.1%}")
    log("\nLimitation: episodes were assigned independently of the ECG, so these numbers show how "
        "alerts are redistributed, not that outcomes improve.")

    # Chart: worst state by risk tier and episode
    fig, ax = plt.subplots(figsize=(7, 3.6))
    groups = [("Low risk, no episode", (~high) & (out.episode == 0)), ("Low risk, episode", (~high) & (out.episode == 1)),
              ("High risk, no episode", high & (out.episode == 0)), ("High risk, episode", high & (out.episode == 1))]
    left = np.zeros(len(groups))
    for s in STATES:
        vals = np.array([100 * (out.loc[m, "worst_state"] == s).mean() for _, m in groups])
        ax.barh([g for g, _ in groups], vals, left=left, color=COLORS[s], label=s, edgecolor="white", linewidth=2)
        left += vals
    ax.set_xlim(0, 100)
    ax.set_xlabel("Patients (%)")
    ax.invert_yaxis()
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Worst twin state, days 35-59 (validation; episodes SYNTHETIC)", loc="left", fontsize=10, pad=28)
    fig.tight_layout()
    fig.savefig(OUT / "state_counts_val.png", dpi=150)
    plt.close(fig)

    with open(OUT / "twin_val_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
