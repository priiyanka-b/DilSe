"""
heart_age.py: ECG heart age, a supporting signal in the twin (DECISIONS.md D-26).

A second CNN, with the same design as the risk CNN (D-10), learns to estimate a
person's age from a NORMAL 12-lead ECG. Applied to any ECG, it says how old the
heart "looks" compared with healthy hearts. The difference from the real age,
after a standard bias correction, is the heart-age gap.

All settings and pass/fail rules were written in DECISIONS.md D-26 BEFORE this
was run. The test set is scored exactly once, at the end.

Input:  data/processed/age_100hz.npz (from export_age_data.py)
Run:    python heart_age.py
Time:   roughly 1-2 minutes per epoch on a laptop CPU, at most 30 epochs.

Outputs (outputs/heart_age/):
    heart_age_model.pt         model weights (best validation epoch)
    history.csv                training and validation error per epoch
    correction.json            bias correction fitted on validation NORM ECGs
    heart_age_test.csv         ecg_id, real age, predicted age, heart age, gap (test set)
    heart_age_summary.txt      R1 and R2 results
    heart_age_scatter.png      predicted vs real age, test NORM ECGs
    heart_age_gap.png          corrected gap, normal vs abnormal test ECGs
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train_cnn import PlainCNN, set_seed

SETTINGS = {
    "decision": "D-26",
    "seed": 42,
    "widths": [32, 64, 128, 128],
    "kernel_size": 7,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "max_epochs": 30,
    "patience": 5,
    "loss": "L1",
    "bootstrap": {"resamples": 2000, "seed": 7},
    "rules": {"R1_mae_max": 9.0, "R1_vs_mean_max_ratio": 0.70},
}

DATA = Path("data/processed")
OUT = Path("outputs/heart_age")
OUT.mkdir(parents=True, exist_ok=True)


def log(lines, text=""):
    print(text, flush=True)
    lines.append(str(text))


@torch.no_grad()
def predict_years(model, X, mu, sd, batch=256):
    model.eval()
    out = []
    for s in range(0, len(X), batch):
        xb = torch.from_numpy(X[s:s + batch].astype(np.float32))
        out.append(model(xb).numpy())
    return np.concatenate(out) * sd + mu


def bootstrap_diff(gap, abnormal, patient, resamples, seed):
    """95% interval for mean gap (abnormal) - mean gap (normal), resampling patients."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"gap": gap, "ab": abnormal, "pid": patient})
    groups = [g for _, g in df.groupby("pid")]
    diffs = []
    for _ in range(resamples):
        pick = rng.integers(0, len(groups), len(groups))
        s = pd.concat([groups[i] for i in pick])
        diffs.append(s.loc[s.ab == 1, "gap"].mean() - s.loc[s.ab == 0, "gap"].mean())
    return np.percentile(diffs, [2.5, 97.5])


def main():
    lines = []
    set_seed(SETTINGS["seed"])
    torch.set_num_threads(max(1, torch.get_num_threads()))
    d = np.load(DATA / "age_100hz.npz")
    Xtr, atr = d["train_X"], d["train_age"]
    val_norm = d["val_label"] == 0
    Xva, ava = d["val_X"][val_norm], d["val_age"][val_norm]
    mu, sd = float(atr.mean()), float(atr.std())
    log(lines, f"Training NORM ECGs: {len(Xtr)}  (age mean {mu:.1f}, SD {sd:.1f})")
    log(lines, f"Validation NORM ECGs: {len(Xva)}")

    ds = TensorDataset(torch.from_numpy(Xtr.astype(np.float32)),
                       torch.from_numpy(((atr - mu) / sd).astype(np.float32)))
    loader = DataLoader(ds, batch_size=SETTINGS["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(SETTINGS["seed"]))
    model = PlainCNN(widths=SETTINGS["widths"], kernel_size=SETTINGS["kernel_size"])
    opt = torch.optim.Adam(model.parameters(), lr=SETTINGS["learning_rate"])
    loss_fn = nn.L1Loss()

    best, best_ep, wait, hist = np.inf, -1, 0, []
    for ep in range(1, SETTINGS["max_epochs"] + 1):
        t0 = time.time()
        model.train()
        tot = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        train_mae = tot / len(ds) * sd
        val_mae = float(np.abs(predict_years(model, Xva, mu, sd) - ava).mean())
        hist.append({"epoch": ep, "train_mae": train_mae, "val_mae": val_mae, "seconds": time.time() - t0})
        log(lines, f"epoch {ep:2d}  train MAE {train_mae:5.2f}  val NORM MAE {val_mae:5.2f}  ({time.time() - t0:.0f} s)")
        pd.DataFrame(hist).to_csv(OUT / "history.csv", index=False)
        if val_mae < best:
            best, best_ep, wait = val_mae, ep, 0
            torch.save(model.state_dict(), OUT / "heart_age_model.pt")
        else:
            wait += 1
            if wait >= SETTINGS["patience"]:
                log(lines, f"early stop: no improvement for {SETTINGS['patience']} epochs")
                break
    log(lines, f"Best epoch {best_ep}: validation NORM MAE {best:.2f} years")
    model.load_state_dict(torch.load(OUT / "heart_age_model.pt"))

    # Bias correction, fitted on validation NORM ECGs only
    pva = predict_years(model, Xva, mu, sd)
    d_, c_ = np.polyfit(ava, pva - ava, 1)
    corr = {"c": float(c_), "d": float(d_), "train_age_mean": mu, "train_age_sd": sd,
            "best_epoch": best_ep, "val_norm_mae": best}
    json.dump(corr, open(OUT / "correction.json", "w"), indent=2)
    log(lines, f"Bias correction (validation NORM): gap = {c_:.2f} + {d_:.3f} x age")

    # ---- Test set: scored exactly once ----
    Xte, ate, lab = d["test_X"], d["test_age"], d["test_label"]
    pte = predict_years(model, Xte, mu, sd)
    gap = (pte - ate) - (c_ + d_ * ate)
    res = pd.DataFrame({"ecg_id": d["test_id"], "patient_id": d["test_patient"], "label": lab,
                        "age": ate, "pred_age": pte.round(2), "gap": gap.round(2),
                        "heart_age": np.round(ate + gap).astype(int)})
    res.to_csv(OUT / "heart_age_test.csv", index=False)

    norm = lab == 0
    mae = float(np.abs(pte[norm] - ate[norm]).mean())
    mae_mean = float(np.abs(mu - ate[norm]).mean())
    r = float(np.corrcoef(pte[norm], ate[norm])[0, 1])
    r1 = mae <= SETTINGS["rules"]["R1_mae_max"] and mae <= SETTINGS["rules"]["R1_vs_mean_max_ratio"] * mae_mean
    g_ab, g_no = float(gap[~norm].mean()), float(gap[norm].mean())
    lo, hi = bootstrap_diff(gap, (~norm).astype(int), d["test_patient"], **SETTINGS["bootstrap"])
    r2 = lo > 0

    log(lines)
    log(lines, "=== Test set (scored once) ===")
    log(lines, f"Test NORM ECGs: {norm.sum()}, abnormal: {(~norm).sum()}")
    log(lines, f"R1 accuracy: MAE {mae:.2f} years (rule <= {SETTINGS['rules']['R1_mae_max']}); "
               f"always-mean-age MAE {mae_mean:.2f}, ratio {mae / mae_mean:.2f} (rule <= 0.70); "
               f"correlation {r:.2f}  -> {'PASS' if r1 else 'FAIL'}")
    log(lines, f"R2 meaning: mean corrected gap abnormal {g_ab:+.2f}, normal {g_no:+.2f}, "
               f"difference {g_ab - g_no:+.2f} years (95% CI {lo:+.2f} to {hi:+.2f}) -> {'PASS' if r2 else 'FAIL'}")
    share = lambda m: float((gap[m] >= 5).mean() * 100)
    log(lines, f"Heart looks >= 5 years older: abnormal {share(~norm):.0f}%, normal {share(norm):.0f}%")
    log(lines, f"Overall: {'ALL RULES PASS -> shown in the app' if (r1 and r2) else 'FAIL -> not shown in the app (fallback)'}")
    summary = {"R1_pass": bool(r1), "R2_pass": bool(r2), "mae_norm": mae, "mae_mean_baseline": mae_mean,
               "corr_norm": r, "gap_abnormal": g_ab, "gap_normal": g_no, "diff": g_ab - g_no,
               "diff_ci": [float(lo), float(hi)], "older5_abnormal_pct": share(~norm),
               "older5_normal_pct": share(norm), "show_in_app": bool(r1 and r2), **corr}
    json.dump(summary, open(OUT / "heart_age_summary.json", "w"), indent=2)
    (OUT / "heart_age_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(ate[norm], pte[norm], s=6, alpha=0.35, color="#1b6e7e")
    ax.plot([18, 90], [18, 90], color="#33414d", lw=1)
    ax.set_xlabel("Real age (years)")
    ax.set_ylabel("Age estimated from the ECG (years)")
    ax.set_title(f"Test NORM ECGs: mean error {mae:.1f} years, r = {r:.2f}", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "heart_age_scatter.png", dpi=160)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(-30, 31, 2)
    ax.hist(gap[norm], bins=bins, alpha=0.6, color="#1b6e7e", label=f"Normal ECG (mean {g_no:+.1f} y)", density=True)
    ax.hist(gap[~norm], bins=bins, alpha=0.6, color="#b8452f", label=f"Abnormal ECG (mean {g_ab:+.1f} y)", density=True)
    ax.axvline(0, color="#33414d", lw=1)
    ax.set_xlabel("Heart age minus real age (years, bias-corrected)")
    ax.set_yticks([])
    ax.legend(frameon=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "heart_age_gap.png", dpi=160)


if __name__ == "__main__":
    main()
