"""
prepare_data.py: turn raw PTB-XL into a clean, model-ready dataset.

What it does:
  1. Builds labels with the DECISIONS.md D-3 rules
     (0 = only NORM, 1 = any of MI/STTC/CD/HYP, all likelihoods kept,
      records with no diagnostic superclass dropped).
  2. Applies D-4: age 300 (privacy code for "over 89") -> 90, then keeps adults (18+).
  3. Splits by the official PTB-XL folds: 1-8 train, 9 validation, 10 test.
  4. Reads every 500 Hz ECG (D-2) and stores each split as ONE big array on disk:
         data/processed/X_train.npy   shape (N, 12, 5000)  float16
     (12 leads x 5000 samples = 10 seconds at 500 Hz)
     The (leads, time) order is what PyTorch's 1D convolutions expect.
  5. Saves a matching table per split (ecg_id, patient_id, age, sex, label, ...).
  6. Computes per-lead mean and standard deviation on the TRAIN set only.

Why step 6 uses train only:
  Before training, every ECG is scaled using a mean and standard deviation.
  If those numbers were computed on all the data, a small amount of information
  about the validation/test ECGs would leak into training. Using train-only
  statistics keeps the test set untouched.

Why float16:
  21,388 ECGs x 12 x 5000 numbers is about 2.6 GB at float16 (about 5.1 GB at
  float32). ECG values are a few millivolts, so float16 keeps plenty of precision.
  The arrays are memory-mapped, so training reads them from disk in batches
  instead of loading everything into RAM.

Run from C:\\Users\\priya\\dt-cardio:
    python prepare_data.py

Expected run time: several minutes (it reads ~21,000 ECG files).
Every number printed is also saved to outputs/prepare_summary.txt.
"""

import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/raw/ptb-xl")
OUT_DIR = Path("data/processed")
REPORT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ABNORMAL_CLASSES = {"MI", "STTC", "CD", "HYP"}   # D-1
MIN_AGE = 18                                     # D-4
AGE_PRIVACY_CODE = 300                           # PhysioNet page, Data Preprocessing
AGE_RECODE_TO = 90                               # D-4
N_SAMPLES, N_LEADS = 5000, 12                    # 10 s at 500 Hz, 12 leads (D-2)

log_lines = []


def log(text=""):
    """Print and remember the line, so it can be saved to a file at the end."""
    print(text)
    log_lines.append(str(text))


# ---------------------------------------------------------------------------
# 1. Labels (D-3)
# ---------------------------------------------------------------------------
log("=" * 70)
log("STEP 1: LABELS (D-3)")
log("=" * 70)

df = pd.read_csv(RAW_DIR / "ptbxl_database.csv", index_col="ecg_id")
df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

scp = pd.read_csv(RAW_DIR / "scp_statements.csv", index_col=0)
code_to_class = scp[scp["diagnostic"] == 1]["diagnostic_class"].to_dict()

df["superclasses"] = df["scp_codes"].apply(
    lambda codes: sorted({code_to_class[c] for c in codes if c in code_to_class})
)

n_start = len(df)
df = df[df["superclasses"].apply(len) > 0].copy()
log(f"Records loaded:                         {n_start}")
log(f"Dropped, no diagnostic superclass:      {n_start - len(df)}")

df["label"] = df["superclasses"].apply(lambda s: int(len(set(s) & ABNORMAL_CLASSES) > 0))

# ---------------------------------------------------------------------------
# 2. Population (D-4)
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 2: POPULATION (D-4)")
log("=" * 70)

n_recoded = (df["age"] >= AGE_PRIVACY_CODE).sum()
df.loc[df["age"] >= AGE_PRIVACY_CODE, "age"] = AGE_RECODE_TO
log(f"Ages recoded from {AGE_PRIVACY_CODE} to {AGE_RECODE_TO}:          {n_recoded}")

n_before = len(df)
n_minor_label1 = ((df["age"] < MIN_AGE) & (df["label"] == 1)).sum()
df = df[df["age"] >= MIN_AGE].copy()
log(f"Dropped, under {MIN_AGE}:                     {n_before - len(df)}"
    f"  (of which abnormal: {n_minor_label1})")
log(f"Records remaining:                      {len(df)}")

# Sex is stored as 0/1. From the 52% male / 48% female split on the PhysioNet
# page we INFER 0 = male, 1 = female. To be confirmed in the PTB-XL paper
# before it is shown as text on the dashboard. The model only needs the number.

# ---------------------------------------------------------------------------
# 3. Splits (official folds)
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 3: SPLITS (folds 1-8 train, 9 val, 10 test)")
log("=" * 70)

df["split"] = np.where(df["strat_fold"] == 10, "test",
               np.where(df["strat_fold"] == 9, "val", "train"))

for name in ["train", "val", "test"]:
    part = df[df["split"] == name]
    log(f"{name:5s}: {len(part):6d} records | "
        f"normal {int((part['label'] == 0).sum()):5d} | "
        f"abnormal {int((part['label'] == 1).sum()):5d} | "
        f"abnormal share {100 * part['label'].mean():.1f}%")

overlap = df.groupby("patient_id")["split"].nunique().gt(1).sum()
log(f"\nPatients in more than one split: {overlap}  "
    f"({'OK, no leakage' if overlap == 0 else 'WARNING: leakage'})")

# ---------------------------------------------------------------------------
# 4. Read the ECGs and write one array per split
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 4: READING ECG SIGNALS (500 Hz)")
log("=" * 70)

# Running sums for the train-only per-lead mean and std (step 6)
lead_sum = np.zeros(N_LEADS, dtype=np.float64)
lead_sumsq = np.zeros(N_LEADS, dtype=np.float64)
lead_count = 0
lead_names = None
problems = []

for name in ["train", "val", "test"]:
    part = df[df["split"] == name]
    X = np.lib.format.open_memmap(
        OUT_DIR / f"X_{name}.npy", mode="w+", dtype=np.float16,
        shape=(len(part), N_LEADS, N_SAMPLES),
    )
    t0 = time.time()

    for i, (ecg_id, row) in enumerate(part.iterrows()):
        signal, fields = wfdb.rdsamp(str(RAW_DIR / row["filename_hr"]))  # (5000, 12)

        if lead_names is None:
            lead_names = fields["sig_name"]

        if signal.shape != (N_SAMPLES, N_LEADS):
            problems.append((ecg_id, f"unexpected shape {signal.shape}"))
            signal = np.zeros((N_SAMPLES, N_LEADS))
        if np.isnan(signal).any():
            problems.append((ecg_id, "contains NaN, replaced with 0"))
            signal = np.nan_to_num(signal)

        X[i] = signal.T.astype(np.float16)  # -> (12, 5000)

        if name == "train":
            lead_sum += signal.sum(axis=0)
            lead_sumsq += (signal ** 2).sum(axis=0)
            lead_count += signal.shape[0]

        if (i + 1) % 2000 == 0 or i + 1 == len(part):
            print(f"  {name}: {i + 1}/{len(part)} ECGs  ({time.time() - t0:.0f} s)")

    X.flush()
    del X
    log(f"{name:5s}: saved {OUT_DIR / f'X_{name}.npy'}  shape ({len(part)}, {N_LEADS}, {N_SAMPLES})")

    # Matching table, in the SAME row order as the array
    table = part[["patient_id", "age", "sex", "height", "weight", "strat_fold", "label"]].copy()
    table["superclasses"] = part["superclasses"].apply(lambda s: "|".join(s))
    table.to_csv(OUT_DIR / f"meta_{name}.csv")

log(f"\nRecords with problems: {len(problems)}")
for ecg_id, msg in problems[:20]:
    log(f"  ecg_id {ecg_id}: {msg}")

# ---------------------------------------------------------------------------
# 5. Train-only normalisation statistics
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 5: PER-LEAD MEAN / STD (TRAIN ONLY)")
log("=" * 70)

mean = lead_sum / lead_count
std = np.sqrt(lead_sumsq / lead_count - mean ** 2)
for lead, m, s in zip(lead_names, mean, std):
    log(f"  {lead:4s} mean {m:+.4f} mV   std {s:.4f} mV")

with open(OUT_DIR / "norm_stats.json", "w") as f:
    json.dump({"leads": lead_names, "mean": mean.tolist(), "std": std.tolist()}, f, indent=2)

log("\nDone. Files in data/processed/: X_{train,val,test}.npy, "
    "meta_{train,val,test}.csv, norm_stats.json")

with open(REPORT_DIR / "prepare_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))