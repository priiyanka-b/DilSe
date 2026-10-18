"""
fuse.py: stage 2 of the fusion model (DECISIONS.md D-11).

Combines the honest ECG score from crossfit_cnn.py with age and sex using a
logistic regression, and compares it fairly against two baselines.

Rows of the results table (all on the VALIDATION set):
  1. Static only      : logistic regression on age + sex
                        "what could you guess with no heart measurement?"
  2. ECG only (ens.)  : the average of the 4 cross-fitted CNNs, no age/sex
                        (the fair comparison for row 3, see the D-11 confound)
  3. Fused            : logistic regression on ECG score + age + sex

Rule fixed BEFORE running (logged in D-11):
  Fusion "adds measurable value" only if the 95% bootstrap confidence interval
  of AUROC(fused) - AUROC(ECG-only ensemble) lies entirely above 0.

Inputs:
  outputs/crossfit/oof_train.csv, outputs/crossfit/val_scores.csv  (crossfit_cnn.py)
  data/processed/meta_train.csv, meta_val.csv                      (prepare_data.py)

Run:  python fuse.py
Outputs:
  outputs/fusion/fusion_model.json     coefficients, for the dashboard
  outputs/fusion/fusion_results.txt    everything printed here
  outputs/fusion/val_predictions.csv   per-patient validation scores
  outputs/results_val.csv              one row per model appended
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

DATA = Path("data/processed")
CF = Path("outputs/crossfit")
OUT = Path("outputs/fusion")
OUT.mkdir(parents=True, exist_ok=True)

AGE_CENTRE = 60.0   # age is entered as (age - 60) / 10, so the coefficient means "per decade"
EPS = 1e-6
N_BOOT = 2000
SEED = 42

log_lines = []


def log(text=""):
    print(text)
    log_lines.append(str(text))


def logit(p):
    """Turn a probability into log-odds, the natural input for a logistic regression."""
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def metrics(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return {"auroc": roc_auc_score(y, p),
            "auprc": average_precision_score(y, p),
            "sens_at_90spec": tpr[np.where(fpr <= 0.10)[0][-1]]}


def features(meta, ecg_prob=None):
    cols = {"age_decades": (meta["age"].to_numpy() - AGE_CENTRE) / 10.0,
            "sex": meta["sex"].to_numpy().astype(float)}
    if ecg_prob is not None:
        cols = {"ecg_logodds": logit(ecg_prob), **cols}
    return pd.DataFrame(cols)


def main():
    meta_tr = pd.read_csv(DATA / "meta_train.csv")
    meta_va = pd.read_csv(DATA / "meta_val.csv")
    oof = pd.read_csv(CF / "oof_train.csv")
    val = pd.read_csv(CF / "val_scores.csv")

    # Rows must line up exactly; refuse to continue otherwise
    assert (meta_tr["ecg_id"].to_numpy() == oof["ecg_id"].to_numpy()).all(), "train rows misaligned"
    assert (meta_va["ecg_id"].to_numpy() == val["ecg_id"].to_numpy()).all(), "val rows misaligned"

    y_tr, y_va = meta_tr["label"].to_numpy(), meta_va["label"].to_numpy()

    # 1. Static only
    X_tr_s, X_va_s = features(meta_tr), features(meta_va)
    static = LogisticRegression(max_iter=1000).fit(X_tr_s, y_tr)
    p_static = static.predict_proba(X_va_s)[:, 1]

    # 2. ECG only: average of the 4 cross-fitted CNNs (no fitting needed for AUROC)
    p_ecg = val["prob_mean"].to_numpy()

    # 3. Fused
    X_tr_f = features(meta_tr, oof["oof_prob"].to_numpy())
    X_va_f = features(meta_va, p_ecg)
    fused = LogisticRegression(max_iter=1000).fit(X_tr_f, y_tr)
    p_fused = fused.predict_proba(X_va_f)[:, 1]

    results = {"static_only_age_sex": metrics(y_va, p_static),
               "ecg_only_crossfit_ensemble": metrics(y_va, p_ecg),
               "fused_ecg_age_sex": metrics(y_va, p_fused)}

    log("=" * 72)
    log("VALIDATION RESULTS (test set untouched)")
    log("=" * 72)
    log(f"{'Model':32s} {'AUROC':>7s} {'AUPRC':>7s} {'Sens@90%Spec':>13s}")
    for name, m in results.items():
        log(f"{name:32s} {m['auroc']:7.3f} {m['auprc']:7.3f} {m['sens_at_90spec']:13.3f}")
    log(f"{'(reference) single plain CNN':32s} {0.940:7.3f} {0.958:7.3f} {0.829:13.3f}   from D-9/D-10 log")
    log(f"Chance AUPRC (abnormal share of validation): {y_va.mean():.3f}")

    # Paired bootstrap: is the fusion gain bigger than noise?
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y_va), len(y_va))
        if y_va[idx].min() == y_va[idx].max():
            continue
        diffs.append(roc_auc_score(y_va[idx], p_fused[idx]) - roc_auc_score(y_va[idx], p_ecg[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    delta = results["fused_ecg_age_sex"]["auroc"] - results["ecg_only_crossfit_ensemble"]["auroc"]
    verdict = "ADDS measurable value" if lo > 0 else "NO measurable difference"
    log(f"\nFusion gain over ECG-only ensemble: {delta:+.4f} AUROC "
        f"(95% bootstrap CI {lo:+.4f} to {hi:+.4f}, {len(diffs)} resamples)")
    log(f"Pre-agreed rule (CI entirely above 0): fusion {verdict}")

    # Coefficients, in words a doctor can follow
    log("\nFused model, logistic regression coefficients:")
    coef = dict(zip(X_tr_f.columns, fused.coef_[0]))
    log(f"  intercept                 {fused.intercept_[0]:+.3f}")
    log(f"  ecg_logodds               {coef['ecg_logodds']:+.3f}   "
        f"(1.0 = take the CNN at face value; <1 = shrink it, >1 = stretch it)")
    log(f"  age_decades               {coef['age_decades']:+.3f}   "
        f"-> each extra decade multiplies the odds by {np.exp(coef['age_decades']):.2f}, same ECG")
    log(f"  sex (1 vs 0)              {coef['sex']:+.3f}   "
        f"-> odds multiplied by {np.exp(coef['sex']):.2f}, same ECG and age")
    log("  Note: sex coding 0/1 is inferred as male/female, still to be confirmed (D-4 note).")

    log("\nStatic-only model coefficients:")
    s_coef = dict(zip(X_tr_s.columns, static.coef_[0]))
    log(f"  age_decades {s_coef['age_decades']:+.3f} (odds x{np.exp(s_coef['age_decades']):.2f} per decade) | "
        f"sex {s_coef['sex']:+.3f} (odds x{np.exp(s_coef['sex']):.2f})")

    # Save for the dashboard and the results table
    json.dump({"inputs": list(X_tr_f.columns), "age_centre": AGE_CENTRE,
               "intercept": float(fused.intercept_[0]),
               "coefficients": {k: float(v) for k, v in coef.items()},
               "static_only": {"intercept": float(static.intercept_[0]),
                               "coefficients": {k: float(v) for k, v in s_coef.items()}}},
              open(OUT / "fusion_model.json", "w"), indent=2)

    pd.DataFrame({"ecg_id": meta_va["ecg_id"], "label": y_va, "age": meta_va["age"], "sex": meta_va["sex"],
                  "p_static": p_static, "p_ecg": p_ecg, "p_fused": p_fused}).to_csv(
        OUT / "val_predictions.csv", index=False)

    rows = [{"experiment": name, "best_epoch": "", "val_auroc": m["auroc"], "val_auprc": m["auprc"],
             "val_sens_at_90spec": m["sens_at_90spec"], "config": "fuse.py (D-11)"}
            for name, m in results.items()]
    res_path = Path("outputs/results_val.csv")
    pd.DataFrame(rows).to_csv(res_path, mode="a", index=False, header=not res_path.exists())

    with open(OUT / "fusion_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()