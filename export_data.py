"""
explore_data.py: a first look at the PTB-XL dataset.

What this script does (read-only; it changes nothing in data/):
  1. Loads the metadata table (one row per ECG recording).
  2. Shows what the columns and rows look like.
  3. Checks the static "EHR-style" fields: age, sex, height, weight.
  4. Applies our label rules from DECISIONS.md D-3 and COUNTS how many
     records each rule affects (these numbers go back into DECISIONS.md).
  5. Checks the recommended folds: label balance, and that no patient
     appears in two folds (the main data-leakage check).
  6. Plots one normal ECG and one abnormal ECG so you can see the signal.

Run from the project folder (C:\\Users\\priya\\dt-cardio):
    python explore_data.py

Outputs are written to outputs/eda/:
    summary.txt          all printed numbers, to copy into DECISIONS.md
    ecg_normal.png       12-lead plot of a normal ECG
    ecg_abnormal.png     12-lead plot of an abnormal ECG
"""

import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wfdb

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/raw/ptb-xl")
OUT_DIR = Path("outputs/eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ABNORMAL_CLASSES = {"MI", "STTC", "CD", "HYP"}  # from D-1


# ---------------------------------------------------------------------------
# Print to the screen AND to summary.txt, so we keep a record of every number
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


summary_file = open(OUT_DIR / "summary.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, summary_file)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Load the metadata
# ---------------------------------------------------------------------------
section("1. LOADING METADATA")

df = pd.read_csv(DATA_DIR / "ptbxl_database.csv", index_col="ecg_id")

# scp_codes is stored as text like "{'NORM': 100.0, 'SR': 0.0}".
# literal_eval turns that text into a real Python dictionary.
df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

print(f"Recordings (rows):      {len(df)}")
print(f"Unique patients:        {df['patient_id'].nunique()}")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")

# ---------------------------------------------------------------------------
# 2. What does a row look like?
# ---------------------------------------------------------------------------
section("2. FIRST 5 ROWS (selected columns)")

cols = ["patient_id", "age", "sex", "height", "weight", "scp_codes", "strat_fold", "filename_hr"]
with pd.option_context("display.width", 200, "display.max_colwidth", 60):
    print(df[cols].head())

# ---------------------------------------------------------------------------
# 3. Static / EHR-style fields
# ---------------------------------------------------------------------------
section("3. STATIC FIELDS: MISSING VALUES AND RANGES")

for col in ["age", "sex", "height", "weight"]:
    n_missing = df[col].isna().sum()
    print(f"{col:7s} missing: {n_missing:6d}  ({100 * n_missing / len(df):5.1f}%)")

print("\nAge summary:")
print(df["age"].describe().round(1))

# PhysioNet page (Data Preprocessing): ages above 89 are stored as ~300
n_age_300 = (df["age"] >= 300).sum()
print(f"\nRecords with age >= 300 (privacy-coded 'over 89'): {n_age_300}")

print("\nSex value counts (we will confirm the coding before using it):")
print(df["sex"].value_counts())

# ---------------------------------------------------------------------------
# 4. Labels, using the D-3 rules
# ---------------------------------------------------------------------------
section("4. LABELS (DECISIONS.md D-3)")

scp = pd.read_csv(DATA_DIR / "scp_statements.csv", index_col=0)
diag = scp[scp["diagnostic"] == 1]  # keep diagnostic statements only
code_to_class = diag["diagnostic_class"].to_dict()


def superclasses(codes):
    """Return the set of diagnostic superclasses for one record."""
    return {code_to_class[c] for c in codes if c in code_to_class}


df["superclasses"] = df["scp_codes"].apply(superclasses)

no_diag = df["superclasses"].apply(len) == 0
has_norm = df["superclasses"].apply(lambda s: "NORM" in s)
has_abn = df["superclasses"].apply(lambda s: len(s & ABNORMAL_CLASSES) > 0)
mixed = has_norm & has_abn


def has_zero_likelihood_diag(codes):
    return any(c in code_to_class and lik == 0 for c, lik in codes.items())


zero_lik = df["scp_codes"].apply(has_zero_likelihood_diag)

print(f"Dropped, no diagnostic superclass:            {no_diag.sum()}")
print(f"Mixed, NORM + abnormal (labelled 1 under D-3): {mixed.sum()}")
print(f"Records with a diagnostic code of likelihood 0 (kept): {zero_lik.sum()}")

labelled = df[~no_diag].copy()
labelled["label"] = has_abn[~no_diag].astype(int)

n0 = (labelled["label"] == 0).sum()
n1 = (labelled["label"] == 1).sum()
print(f"\nLabelled records: {len(labelled)}")
print(f"  label 0 (normal):   {n0}  ({100 * n0 / len(labelled):.1f}%)")
print(f"  label 1 (abnormal): {n1}  ({100 * n1 / len(labelled):.1f}%)")

print("\nSuperclass counts (a record can have several):")
print(pd.Series([c for s in df["superclasses"] for c in s]).value_counts())

# ---------------------------------------------------------------------------
# 5. Splits and the leakage check
# ---------------------------------------------------------------------------
section("5. SPLITS (folds 1-8 train, 9 validation, 10 test) AND LEAKAGE CHECK")

split = labelled["strat_fold"].map(lambda f: "test" if f == 10 else ("val" if f == 9 else "train"))
table = pd.crosstab(split, labelled["label"], margins=True)
table.columns = ["normal (0)", "abnormal (1)", "total"]
print(table)

folds_per_patient = df.groupby("patient_id")["strat_fold"].nunique()
n_leaky = (folds_per_patient > 1).sum()
print(f"\nPatients appearing in more than one fold: {n_leaky}")
print("OK: no patient-level leakage across folds." if n_leaky == 0
      else "WARNING: patient overlap across folds. Investigate before modelling.")

# ---------------------------------------------------------------------------
# 6. Plot one normal and one abnormal ECG
# ---------------------------------------------------------------------------
section("6. EXAMPLE ECG PLOTS")


def plot_ecg(ecg_id, title, out_name):
    row = df.loc[ecg_id]
    signal, fields = wfdb.rdsamp(str(DATA_DIR / row["filename_hr"]))
    fs = fields["fs"]
    t = [i / fs for i in range(signal.shape[0])]

    fig, axes = plt.subplots(12, 1, figsize=(14, 16), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, signal[:, i], linewidth=0.7)
        ax.set_ylabel(fields["sig_name"][i], rotation=0, labelpad=20)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(f"{title}  |  ecg_id {ecg_id}  |  codes: {row['scp_codes']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=110)
    plt.close(fig)
    print(f"Saved {OUT_DIR / out_name}  (sampling rate {fs} Hz, shape {signal.shape})")


normal_id = labelled[labelled["label"] == 0].index[0]
abnormal_id = labelled[labelled["superclasses"].apply(lambda s: "MI" in s)].index[0]
plot_ecg(normal_id, "Normal ECG", "ecg_normal.png")
plot_ecg(abnormal_id, "Abnormal ECG (contains MI)", "ecg_abnormal.png")

print("\nDone. Send outputs/eda/summary.txt and the two PNGs back for review.")

# Restore normal printing BEFORE closing the file, so Python doesn't try to
# write to a closed file at exit (that caused the harmless 'I/O operation on
# closed file' message in the first run).
sys.stdout = sys.__stdout__
summary_file.close()