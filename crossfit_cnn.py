"""
crossfit_cnn.py: stage 1 of the fusion model (DECISIONS.md D-11).

Goal: give every TRAINING ECG an honest "ECG score", meaning a probability
from a CNN that never saw that ECG during training.

How:
  The training set (official folds 1-8) is split into 4 groups by fold pairs:
      group A = folds 1+2, B = 3+4, C = 5+6, D = 7+8
  Using the official folds keeps every patient inside one group (no leakage).
  For each group, a plain CNN (D-10, 100 Hz per D-9) is trained on the other
  3 groups and then scores the group it left out. Doing this 4 times gives an
  out-of-fold ("OOF") score for all training ECGs.

  The same 4 CNNs also score the VALIDATION set; their 4 scores are averaged.

Why a FIXED number of epochs (no early stopping here):
  Early stopping needs a set to watch. Watching the left-out group would let
  it influence its own scores; watching the validation set would let
  validation influence model choice twice. So every fold model trains for a
  fixed 11 epochs, the best epoch found for the single plain CNN at 100 Hz
  (D-9 evidence). Simple, and nothing leaks.

The TEST set is NOT touched. The 4 fold models are saved so the test set can
be scored exactly once at the end.

Run from C:\\Users\\priya\\dt-cardio (needs train_cnn.py in the same folder):
    python crossfit_cnn.py

Expected time: 4 models x about 10 minutes, so roughly 40-50 minutes.

Outputs:
    outputs/models/crossfit_fold{A,B,C,D}.pt   the 4 CNNs
    outputs/crossfit/oof_train.csv             ecg_id, group, label, oof_prob
    outputs/crossfit/val_scores.csv            ecg_id, label, prob_A..D, prob_mean
    outputs/crossfit/crossfit_log.txt          per-fold scores and timings
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

# Reuse the exact dataset, model and seed helper from train_cnn.py,
# so the fold models are identical in design to the plain CNN (D-10).
from train_cnn import DATA, ECGDataset, PlainCNN, set_seed

SETTINGS = {
    "seed": 42,
    "downsample": 5,            # 100 Hz (D-9)
    "epochs": 11,               # fixed, see docstring
    "batch_size": 64,
    "learning_rate": 1e-3,
    "widths": [32, 64, 128, 128],
    "kernel_size": 7,
    "groups": {"A": [1, 2], "B": [3, 4], "C": [5, 6], "D": [7, 8]},
}

OUT = Path("outputs")
CF = OUT / "crossfit"
CF.mkdir(parents=True, exist_ok=True)
(OUT / "models").mkdir(parents=True, exist_ok=True)

log_lines = []


def log(text=""):
    print(text)
    log_lines.append(str(text))


@torch.no_grad()
def predict(model, dataset):
    """Return the model's probability for every ECG in `dataset`, in order."""
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    return np.concatenate([torch.sigmoid(model(x)).numpy() for x, _ in loader])


def main():
    stats = json.load(open(DATA / "norm_stats.json"))
    mean, std = np.array(stats["mean"], np.float32), np.array(stats["std"], np.float32)

    train_ds = ECGDataset("train", mean, std, SETTINGS["downsample"])
    val_ds = ECGDataset("val", mean, std, SETTINGS["downsample"])
    meta_train = pd.read_csv(DATA / "meta_train.csv")
    meta_val = pd.read_csv(DATA / "meta_val.csv")
    assert len(meta_train) == len(train_ds) and len(meta_val) == len(val_ds)

    fold_to_group = {f: g for g, folds in SETTINGS["groups"].items() for f in folds}
    meta_train["group"] = meta_train["strat_fold"].map(fold_to_group)
    assert meta_train["group"].notna().all(), "a training record has a fold outside 1-8"

    # Leakage check: every patient must sit in exactly one group
    groups_per_patient = meta_train.groupby("patient_id")["group"].nunique()
    log(f"Patients spanning more than one group: {(groups_per_patient > 1).sum()} (must be 0)")
    for g in SETTINGS["groups"]:
        log(f"  group {g}: {(meta_train['group'] == g).sum()} ECGs")

    oof = np.full(len(meta_train), np.nan)
    val_scores = {}
    loss_fn = nn.BCEWithLogitsLoss()

    for k, g in enumerate(SETTINGS["groups"]):
        log(f"\n=== Fold model {g}: trained WITHOUT group {g}, then scores group {g} ===")
        set_seed(SETTINGS["seed"] + k)

        held_idx = np.where(meta_train["group"] == g)[0]
        fit_idx = np.where(meta_train["group"] != g)[0]
        loader = DataLoader(Subset(train_ds, fit_idx), batch_size=SETTINGS["batch_size"],
                            shuffle=True, num_workers=0)

        model = PlainCNN(widths=SETTINGS["widths"], kernel_size=SETTINGS["kernel_size"])
        optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS["learning_rate"])

        t_fold = time.time()
        for epoch in range(1, SETTINGS["epochs"] + 1):
            model.train()
            t0, running = time.time(), 0.0
            for x, y in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
                running += loss.item() * len(y)
            print(f"  epoch {epoch:2d}/{SETTINGS['epochs']} | train loss {running / len(fit_idx):.3f} "
                  f"| {(time.time() - t0) / 60:.1f} min")

        torch.save(model.state_dict(), OUT / "models" / f"crossfit_fold{g}.pt")

        oof[held_idx] = predict(model, Subset(train_ds, held_idx))
        val_scores[g] = predict(model, val_ds)

        held_auc = roc_auc_score(meta_train["label"].iloc[held_idx], oof[held_idx])
        val_auc = roc_auc_score(meta_val["label"], val_scores[g])
        log(f"Fold {g}: AUROC on its left-out group {held_auc:.3f} | on validation {val_auc:.3f} "
            f"| {(time.time() - t_fold) / 60:.1f} min")

    assert not np.isnan(oof).any(), "some training ECGs did not get an OOF score"

    # Save everything stage 2 (fuse.py) needs
    pd.DataFrame({"ecg_id": meta_train["ecg_id"], "group": meta_train["group"],
                  "label": meta_train["label"], "oof_prob": oof}).to_csv(CF / "oof_train.csv", index=False)

    val_df = pd.DataFrame({"ecg_id": meta_val["ecg_id"], "label": meta_val["label"]})
    for g, p in val_scores.items():
        val_df[f"prob_{g}"] = p
    val_df["prob_mean"] = val_df[[f"prob_{g}" for g in val_scores]].mean(axis=1)
    val_df.to_csv(CF / "val_scores.csv", index=False)

    log("\n=== Summary ===")
    log(f"OOF AUROC over all training ECGs:      {roc_auc_score(meta_train['label'], oof):.3f}")
    log(f"Validation AUROC, average of 4 CNNs:  {roc_auc_score(val_df['label'], val_df['prob_mean']):.3f}")
    log("Saved outputs/crossfit/oof_train.csv and val_scores.csv. Test set untouched.")

    with open(CF / "crossfit_log.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(SETTINGS) + "\n" + "\n".join(log_lines))


if __name__ == "__main__":
    main()