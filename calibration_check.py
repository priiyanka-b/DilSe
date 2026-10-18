"""
calibration_check.py: decide D-12 (which model becomes the final risk score).

"Calibration" asks: when the model says 30% risk, are about 30% of those
patients actually abnormal? A doctor reads the dashboard number as a real
probability, so this matters as much as ranking (AUROC).

Compares, on the VALIDATION set only (test untouched):
  ECG only : the 4-CNN ensemble average      (column p_ecg)
  Fused    : ECG score + age + sex            (column p_fused)
  (Static only is shown for reference.)

Metrics:
  Brier score  mean of (predicted probability - actual 0/1)^2. Lower is better.
               It rewards both good ranking and honest probabilities.
  ECE          expected calibration error: put patients into 10 bins by their
               predicted risk; in each bin compare average prediction with the
               actual abnormal rate; average the gaps (weighted by bin size).
  Log loss     like Brier, but punishes confident mistakes harder.

Rule fixed BEFORE running (DECISIONS.md D-12):
  Choose the FUSED model, unless its Brier score is higher (worse) than the
  ECG-only ensemble's. The bootstrap interval is reported for information.

Input:  outputs/fusion/val_predictions.csv   (written by fuse.py)
Output: outputs/calibration/calibration_results.txt
        outputs/calibration/reliability_diagram.png
Run:    python calibration_check.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

IN = Path("outputs/fusion/val_predictions.csv")
OUT = Path("outputs/calibration")
OUT.mkdir(parents=True, exist_ok=True)
N_BINS, N_BOOT, SEED = 10, 2000, 42

MODELS = {"p_static": "Static only (age + sex)",
          "p_ecg": "ECG only (4-CNN ensemble)",
          "p_fused": "Fused (ECG + age + sex)"}

log_lines = []


def log(text=""):
    print(text)
    log_lines.append(str(text))


def calibration_bins(y, p, n_bins=N_BINS):
    """Equal-width risk bins: (mean predicted, observed rate, count) per non-empty bin."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum():
            rows.append((p[mask].mean(), y[mask].mean(), int(mask.sum())))
    return np.array(rows)


def ece(y, p):
    bins = calibration_bins(y, p)
    return float(np.sum(bins[:, 2] * np.abs(bins[:, 0] - bins[:, 1])) / len(y))


def main():
    df = pd.read_csv(IN)
    y = df["label"].to_numpy()

    log("=" * 72)
    log("CALIBRATION ON VALIDATION (test set untouched)")
    log("=" * 72)
    log(f"Patients {len(y)} | actual abnormal rate {y.mean():.3f}\n")
    log(f"{'Model':30s} {'Brier':>7s} {'ECE':>7s} {'LogLoss':>8s} {'Mean pred':>10s}")
    scores = {}
    for col, name in MODELS.items():
        p = df[col].to_numpy()
        scores[col] = {"brier": brier_score_loss(y, p), "ece": ece(y, p),
                       "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))}
        log(f"{name:30s} {scores[col]['brier']:7.4f} {scores[col]['ece']:7.4f} "
            f"{scores[col]['logloss']:8.4f} {p.mean():10.3f}")

    # Paired bootstrap for the Brier difference (fused - ECG). Negative = fused better.
    rng = np.random.default_rng(SEED)
    p_e, p_f = df["p_ecg"].to_numpy(), df["p_fused"].to_numpy()
    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        diffs.append(np.mean((p_f[i] - y[i]) ** 2) - np.mean((p_e[i] - y[i]) ** 2))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    d = scores["p_fused"]["brier"] - scores["p_ecg"]["brier"]
    log(f"\nBrier difference fused - ECG: {d:+.5f} (95% bootstrap CI {lo:+.5f} to {hi:+.5f}); "
        f"negative means fused is better")

    choose_fused = scores["p_fused"]["brier"] <= scores["p_ecg"]["brier"]
    log(f"Pre-agreed rule (fused unless its Brier is worse): "
        f"FINAL MODEL = {'FUSED' if choose_fused else 'ECG-ONLY ENSEMBLE'}")

    log("\nReliability table (equal-width bins; mean predicted vs actual abnormal rate):")
    for col in ["p_ecg", "p_fused"]:
        log(f"  {MODELS[col]}")
        for mp, obs, n in calibration_bins(y, df[col].to_numpy()):
            log(f"    predicted {mp:5.2f} | actual {obs:5.2f} | n={int(n):4d} | gap {obs - mp:+.2f}")

    # Reliability diagram: one axis, two series + gray diagonal reference
    colors = {"p_ecg": "#2a78d6", "p_fused": "#eb6834"}
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color="#9a9a94", linewidth=1, linestyle="--", label="Perfect calibration")
    for col in ["p_ecg", "p_fused"]:
        b = calibration_bins(y, df[col].to_numpy())
        ax.plot(b[:, 0], b[:, 1], color=colors[col], linewidth=2, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=1.5,
                label=f"{MODELS[col]}  (Brier {scores[col]['brier']:.4f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted risk of abnormal ECG")
    ax.set_ylabel("Actual share abnormal")
    ax.set_title("Reliability on validation set (n = %d)" % len(y), loc="left", fontsize=12)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "reliability_diagram.png", dpi=150)
    plt.close(fig)
    log(f"\nSaved {OUT / 'reliability_diagram.png'}")

    with open(OUT / "calibration_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()