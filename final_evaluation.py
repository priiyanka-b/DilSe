"""
final_evaluation.py: the ONE-TIME evaluation on the untouched test patients.

Everything below was decided and locked on validation before this script runs:
  ECG model      4 cross-fitted plain CNNs, averaged          (D-10, D-11, D-12)
  risk threshold P(abnormal) >= value giving 90% specificity on validation (D-18)
  monitor        outputs/monitor/monitor_config.json           (D-17)
  twin state     outputs/twin/twin_config.json                 (D-18)

Nothing is tuned here. The script writes a lock file the first time it finishes
and refuses to run again, so the test set cannot be "peeked at" repeatedly.

PART 1  ECG risk on test (REAL data)
        AUROC, AUPRC, sensitivity at 90% specificity, each with a 95% bootstrap
        interval; sensitivity and specificity at the LOCKED validation threshold;
        calibration (Brier, ECE); the three-row ablation (static only, ECG only,
        fused with age and sex).
PART 2  Wearable monitor on test (SYNTHETIC data and episodes)
PART 3  Twin state on test (ECG risk REAL + wearables SYNTHETIC)

Run from C:\\Users\\priya\\dt-cardio:
    python final_evaluation.py
Takes a few minutes (the 4 CNNs score 2,145 ECGs on CPU).

Outputs in outputs/final_test/:
    test_predictions.csv, final_results.txt, final_results.json,
    roc_test.png, reliability_test.png, detection_vs_size_test.png (from monitor),
    twin_states_test.csv, FINAL_RUN.lock
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             roc_auc_score, roc_curve)

OUT = Path("outputs/final_test")
OUT.mkdir(parents=True, exist_ok=True)
LOCK = OUT / "FINAL_RUN.lock"
N_BOOT, SEED = 2000, 42
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9a94"

log_lines = []


def log(text=""):
    print(text)
    log_lines.append(str(text))


# ---------------------------------------------------------------------------
# Part 1 helpers
# ---------------------------------------------------------------------------
def score_test_ecgs():
    """Average of the 4 locked cross-fitted CNNs on the test ECGs (100 Hz)."""
    import torch
    from torch.utils.data import DataLoader
    from train_cnn import DATA, ECGDataset, PlainCNN

    stats = json.load(open(DATA / "norm_stats.json"))
    mean, std = np.array(stats["mean"], np.float32), np.array(stats["std"], np.float32)
    ds = ECGDataset("test", mean, std, 5)                    # 5 = 100 Hz (D-9)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    probs = []
    for g in "ABCD":
        model = PlainCNN(widths=[32, 64, 128, 128], kernel_size=7)
        model.load_state_dict(torch.load(f"outputs/models/crossfit_fold{g}.pt", map_location="cpu"))
        model.eval()
        with torch.no_grad():
            p = np.concatenate([torch.sigmoid(model(x)).numpy() for x, _ in loader])
        probs.append(p)
        print(f"  fold model {g} scored {len(p)} test ECGs")
    return np.mean(probs, axis=0), np.stack(probs, axis=1)


def sens_at_spec(y, p, spec=0.90):
    fpr, tpr, _ = roc_curve(y, p)
    return tpr[np.where(fpr <= 1 - spec)[0][-1]]


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum()
               for b in range(bins) if (idx == b).any()) / len(y)


def boot_ci(y, p, fn, rng):
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(fn(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])


def logistic(z):
    return 1 / (1 + np.exp(-z))


def part1_ecg(meta, twin_cfg):
    log("=" * 80)
    log("PART 1  ECG RISK ON TEST (real PTB-XL data, fold 10, never used before)")
    log("=" * 80)
    p_ecg, per_fold = score_test_ecgs()
    y = meta["label"].to_numpy()

    fm = json.load(open("outputs/fusion/fusion_model.json"))
    age_dec = (meta["age"].to_numpy() - fm["age_centre"]) / 10.0
    sex = meta["sex"].to_numpy().astype(float)
    c, s = fm["coefficients"], fm["static_only"]["coefficients"]
    logit_ecg = np.log(np.clip(p_ecg, 1e-6, 1 - 1e-6) / (1 - np.clip(p_ecg, 1e-6, 1 - 1e-6)))
    p_fused = logistic(fm["intercept"] + c["ecg_logodds"] * logit_ecg + c["age_decades"] * age_dec + c["sex"] * sex)
    p_static = logistic(fm["static_only"]["intercept"] + s["age_decades"] * age_dec + s["sex"] * sex)

    preds = pd.DataFrame({"ecg_id": meta["ecg_id"], "label": y, "age": meta["age"], "sex": meta["sex"],
                          "p_static": p_static, "p_ecg": p_ecg, "p_fused": p_fused,
                          **{f"p_fold{g}": per_fold[:, i] for i, g in enumerate("ABCD")}})
    preds.to_csv(OUT / "test_predictions.csv", index=False)

    rng = np.random.default_rng(SEED)
    results = {}
    log(f"Test patients {len(y)} | abnormal share {y.mean():.3f} (= chance AUPRC)")
    log(f"{'Model':28s} {'AUROC [95% CI]':>24s} {'AUPRC [95% CI]':>24s} {'Sens@90%Spec [95% CI]':>27s}")
    for name, p in [("Static only (age + sex)", p_static), ("ECG only (FINAL MODEL)", p_ecg),
                    ("Fused (ECG + age + sex)", p_fused)]:
        r = {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
             "sens90": sens_at_spec(y, p)}
        cis = {k: boot_ci(y, p, f, rng) for k, f in
               [("auroc", roc_auc_score), ("auprc", average_precision_score), ("sens90", sens_at_spec)]}
        results[name] = {**r, **{f"{k}_ci": list(map(float, v)) for k, v in cis.items()}}
        log(f"{name:28s} {r['auroc']:.3f} [{cis['auroc'][0]:.3f}-{cis['auroc'][1]:.3f}]      "
            f"{r['auprc']:.3f} [{cis['auprc'][0]:.3f}-{cis['auprc'][1]:.3f}]      "
            f"{r['sens90']:.3f} [{cis['sens90'][0]:.3f}-{cis['sens90'][1]:.3f}]")

    t = twin_cfg["risk_threshold_p_ecg"]
    tn, fp, fn, tp = confusion_matrix(y, (p_ecg >= t).astype(int)).ravel()
    log(f"\nAt the LOCKED validation threshold p_ecg >= {t:.3f}:")
    log(f"  sensitivity {tp / (tp + fn):.3f} | specificity {tn / (tn + fp):.3f}")
    log(f"  confusion matrix: TN {tn}  FP {fp}  FN {fn}  TP {tp}")
    log(f"Calibration of the final model: Brier {brier_score_loss(y, p_ecg):.4f} | ECE {ece(y, p_ecg):.4f} | "
        f"mean predicted {p_ecg.mean():.3f} vs actual {y.mean():.3f}")
    log("Per-fold-model AUROC on test: " + ", ".join(
        f"{g} {roc_auc_score(y, per_fold[:, i]):.3f}" for i, g in enumerate("ABCD")))
    results["locked_threshold"] = {"threshold": t, "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
                                   "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    results["calibration"] = {"brier": brier_score_loss(y, p_ecg), "ece": ece(y, p_ecg)}

    plots_part1(y, p_static, p_ecg, p_fused)
    return preds, results


def plots_part1(y, p_static, p_ecg, p_fused):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for p, color, name in [(p_static, GREY, "Static only"), (p_ecg, BLUE, "ECG only (final)")]:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name}  AUROC {roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], color=GREY, linewidth=1, linestyle="--")
    ax.axvline(0.10, color=ORANGE, linewidth=1, linestyle=":")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("Sensitivity")
    ax.set_title("ROC on the test set (n = %d)" % len(y), loc="left", fontsize=11)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "roc_test.png", dpi=150)
    plt.close(fig)

    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p_ecg, edges[1:-1]), 0, 9)
    pts = [(p_ecg[idx == b].mean(), y[idx == b].mean()) for b in range(10) if (idx == b).any()]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=GREY, linewidth=1, linestyle="--", label="Perfect calibration")
    ax.plot(*zip(*pts), color=BLUE, linewidth=2, marker="o", markersize=7, markeredgecolor="white",
            markeredgewidth=1.5, label="ECG only (final)")
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Actual share abnormal")
    ax.set_title("Reliability on the test set", loc="left", fontsize=11)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "reliability_test.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Parts 2 and 3
# ---------------------------------------------------------------------------
def part2_3_wearables(preds, twin_cfg):
    import monitor
    from monitor import daily_scores, evaluate, flags_from, load

    log("\n" + "=" * 80)
    log("PART 2  WEARABLE MONITOR ON TEST (synthetic data, synthetic episodes, locked settings)")
    log("=" * 80)
    monitor.log_lines = log_lines        # share the log
    monitor.log = log
    monitor.main("test")

    log("\n" + "=" * 80)
    log("PART 3  TWIN STATE ON TEST (ECG risk real + wearables synthetic, locked settings)")
    log("=" * 80)
    X, eps = load("test")
    p = preds.set_index("ecg_id").loc[eps.index]
    high = p["p_ecg"].to_numpy() >= twin_cfg["risk_threshold_p_ecg"]
    score, _ = daily_scores(X, "rolling")
    lo, hi = twin_cfg["low_risk"], twin_cfg["high_risk"]
    flags = np.zeros(score.shape, bool)
    flags[~high] = flags_from(score[~high], lo["threshold"], lo["persistence"])
    flags[high] = flags_from(score[high], hi["threshold"], hi["persistence"])
    uni = json.load(open("outputs/monitor/monitor_config.json"))["methods"]["rolling"]
    flags_uni = flags_from(score, uni["threshold"], uni["persistence"])

    log(f"High-risk patients {high.sum()} ({100 * high.mean():.1f}%), low-risk {(~high).sum()}")
    log(f"{'':36s}{'FA/month':>9s} |  +3     +5     +8    +12  | mean")
    out = {}
    for tier, mask in [("high-risk", high), ("low-risk", ~high), ("all", np.ones_like(high))]:
        for name, f in [("uniform (D-17)", flags_uni), ("risk-adjusted (D-18)", flags)]:
            r = evaluate(f[mask], eps[mask])
            out[f"{tier} {name}"] = {k: float(v) for k, v in r.items()}
            log(f"  {tier + ': ' + name:34s} {r['fa_per_month']:8.2f} | "
                + "  ".join(f"{100 * r[f'det_s{s}']:5.1f}" for s in [1, 2, 3, 4])
                + f" | {100 * r['det_mean']:5.1f}%")

    day_state = np.where(flags & high[:, None], 2, np.where(flags | high[:, None], 1, 0))
    worst = day_state[:, 35:].max(axis=1)
    states = np.array(["STABLE", "WATCH", "ESCALATE"])[worst]
    pd.DataFrame({"ecg_id": eps.index, "p_ecg": p["p_ecg"].to_numpy(), "risk_tier": np.where(high, "high", "low"),
                  "ecg_label": p["label"].to_numpy(), "episode": eps["episode"].to_numpy(),
                  "severity": eps["severity"].to_numpy(), "worst_state": states}).to_csv(
        OUT / "twin_states_test.csv", index=False)
    counts = pd.Series(states).value_counts()
    log("Worst twin state per patient (days 35-59): " + ", ".join(
        f"{s} {counts.get(s, 0)} ({100 * counts.get(s, 0) / len(states):.1f}%)" for s in ["STABLE", "WATCH", "ESCALATE"]))
    return out


def main():
    if LOCK.exists() and "--force" not in sys.argv:
        print(f"REFUSING TO RUN: {LOCK} exists, the test set has already been evaluated once.")
        print(LOCK.read_text())
        print("Re-running would turn the test set into a tuning set. Use --force only if the code crashed "
              "before finishing, and record that in DECISIONS.md.")
        return
    if "--force" in sys.argv:
        log("WARNING: run with --force. This must be recorded in DECISIONS.md.")

    twin_cfg = json.load(open("outputs/twin/twin_config.json"))
    meta = pd.read_csv("data/processed/meta_test.csv")
    preds, ecg_results = part1_ecg(meta, twin_cfg)
    wear_results = part2_3_wearables(preds, twin_cfg)

    stamp = datetime.now().isoformat(timespec="seconds")
    json.dump({"run_at": stamp, "ecg": ecg_results, "twin": wear_results},
              open(OUT / "final_results.json", "w"), indent=2, default=float)
    with open(OUT / "final_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    LOCK.write_text(f"Final test evaluation completed at {stamp}. Settings were locked on validation beforehand.\n")
    log(f"\nDone. Results in {OUT}. Lock file written: {LOCK}")


if __name__ == "__main__":
    main()
