"""
export_age_data.py: make the compact 100 Hz arrays for the heart-age model (DECISIONS.md D-26).

Uses the same preprocessing as the risk CNN: train-only per-lead normalisation
(norm_stats.json), then 100 Hz by averaging 5 neighbouring samples (D-9).
Stored as float16 to keep the file small.

Output: data/processed/age_100hz.npz with
    train_X, train_age, train_id   training folds 1-8, NORM ECGs only (label 0)
    val_X,   val_age,   val_id,   val_label     all validation ECGs
    test_X,  test_age,  test_id,  test_label, test_patient   all test ECGs
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/processed")
stats = json.load(open(DATA / "norm_stats.json"))
MEAN = np.array(stats["mean"], np.float32)[:, None]
STD = np.array(stats["std"], np.float32)[:, None]


def compact(split, keep=None, chunk=256):
    X = np.load(DATA / f"X_{split}.npy", mmap_mode="r")
    meta = pd.read_csv(DATA / f"meta_{split}.csv")
    idx = np.arange(len(meta)) if keep is None else np.flatnonzero(keep(meta))
    out = np.empty((len(idx), 12, 1000), np.float16)
    for s in range(0, len(idx), chunk):
        part = np.asarray(X[idx[s:s + chunk]], np.float32)
        part = (part - MEAN) / STD
        out[s:s + chunk] = part.reshape(len(part), 12, 1000, 5).mean(-1)
    return out, meta.iloc[idx].reset_index(drop=True)


arrays = {}
Xtr, mtr = compact("train", keep=lambda m: m["label"].eq(0))
arrays.update(train_X=Xtr, train_age=mtr["age"].to_numpy(np.float32), train_id=mtr["ecg_id"].to_numpy())
for split in ("val", "test"):
    X, m = compact(split)
    arrays.update({f"{split}_X": X, f"{split}_age": m["age"].to_numpy(np.float32),
                   f"{split}_id": m["ecg_id"].to_numpy(), f"{split}_label": m["label"].to_numpy(),
                   f"{split}_patient": m["patient_id"].to_numpy()})
np.savez(DATA / "age_100hz.npz", **arrays)
print({k: v.shape for k, v in arrays.items()})
