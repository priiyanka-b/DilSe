"""
saliency.py: where on the ECG is the model looking? (DECISIONS.md D-21)

For every TEST patient this computes a saliency map: for each lead and each moment of
the 10-second trace, how much a small change there would change the ensemble's risk
score. The dashboard shades the ECG with it.

Method: plain gradient ("vanilla saliency") of the 4-CNN ensemble (the final risk score, D-12).
  saliency = | d P(abnormal) / d input |,
  smoothed over 50 ms, then scaled per patient so its 99th percentile = 1.
  We use the plain gradient rather than gradient x input because multiplying by the input
  makes a map follow the ECG's own shape (tall R waves) even for an untrained network.

What it does NOT mean (shown on the dashboard too): a saliency map shows where the
model's score is sensitive. It is not a diagnosis and not the model's "reasoning".
Saliency methods are known to be fragile, so we run one sanity check:
  model-randomisation check: compute the same map with an UNTRAINED network. If the
  trained and untrained maps look alike (high correlation), the maps mostly reflect the
  ECG's own shape rather than anything the model learned, and should not be shown.

Nothing here changes the model or any reported number. The test ECGs are only scored
again with the locked models, exactly as in final_evaluation.py.

Input:  outputs/models/crossfit_fold{A,B,C,D}.pt, data/processed/X_test.npy, norm_stats.json
Output: outputs/saliency/saliency_test.npy   float16, shape (patients, 12 leads, 1000) at 100 Hz
        outputs/saliency/saliency_check.txt
Run:    python saliency.py          (about 1-3 minutes on CPU)
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_cnn import DATA, ECGDataset, PlainCNN

OUT = Path("outputs/saliency")
OUT.mkdir(parents=True, exist_ok=True)
FOLDS = ["A", "B", "C", "D"]
SMOOTH_SAMPLES = 5            # 5 samples at 100 Hz = 50 ms


def load_models(trained=True):
    models = []
    for g in FOLDS:
        m = PlainCNN(widths=[32, 64, 128, 128], kernel_size=7)
        if trained:
            m.load_state_dict(torch.load(f"outputs/models/crossfit_fold{g}.pt", map_location="cpu"))
        m.eval()                                   # BatchNorm in inference mode
        models.append(m)
    return models


def saliency_batch(models, x):
    """Absolute gradient of the ensemble probability, for one batch."""
    x = x.clone().requires_grad_(True)
    p = torch.stack([torch.sigmoid(m(x)) for m in models]).mean(0)   # ensemble P(abnormal)
    p.sum().backward()                                                # one gradient per patient
    s = x.grad.abs().detach().numpy()                                # (batch, 12, 1000)
    kernel = np.ones(SMOOTH_SAMPLES) / SMOOTH_SAMPLES
    s = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 2, s)
    return s, p.detach().numpy()


def run(models, loader):
    maps, probs = [], []
    for x, _ in loader:
        s, p = saliency_batch(models, x)
        maps.append(s)
        probs.append(p)
    return np.concatenate(maps), np.concatenate(probs)


def main():
    torch.manual_seed(0)
    stats = json.load(open(DATA / "norm_stats.json"))
    mean, std = np.array(stats["mean"], np.float32), np.array(stats["std"], np.float32)
    loader = DataLoader(ECGDataset("test", mean, std, 5), batch_size=64, shuffle=False, num_workers=0)

    maps, p = run(load_models(trained=True), loader)
    scale = np.percentile(maps.reshape(len(maps), -1), 99, axis=1)[:, None, None]
    maps = np.clip(maps / np.maximum(scale, 1e-12), 0, 1)
    np.save(OUT / "saliency_test.npy", maps.astype(np.float16))

    # Sanity check on the first 200 patients: trained vs untrained network
    sub = DataLoader(torch.utils.data.Subset(loader.dataset, range(200)), batch_size=64, shuffle=False)
    trained, _ = run(load_models(trained=True), sub)
    random_, _ = run(load_models(trained=False), sub)
    corr = [np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in zip(trained, random_)]

    # Consistency check: probabilities must match the locked test predictions
    import pandas as pd
    locked = pd.read_csv("outputs/final_test/test_predictions.csv")["p_ecg"].to_numpy()
    max_diff = float(np.max(np.abs(locked - p)))

    lines = [
        "SALIENCY (plain gradient, 4-CNN ensemble, test patients)",
        f"Saved {OUT / 'saliency_test.npy'}  shape {maps.shape}",
        f"Max difference from locked test probabilities: {max_diff:.2e} (should be ~0)",
        f"Model-randomisation check (200 patients): median correlation trained vs untrained map = "
        f"{np.median(corr):.2f} (IQR {np.percentile(corr, 25):.2f} to {np.percentile(corr, 75):.2f})",
        "Reading: a LOW correlation means the maps depend on what the model learned (good).",
        "A HIGH correlation (say above 0.7) means the maps mostly trace the ECG's own shape;",
        "in that case do not show them on the dashboard.",
    ]
    print("\n".join(lines))
    with open(OUT / "saliency_check.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
