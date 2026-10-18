"""
train_cnn.py: train the ECG-only CNN (DECISIONS.md D-6 step 1, D-7).

Two models, chosen with CONFIG["model"]:
  "plain"   the plain CNN from D-7 step 1 (already trained)
  "resnet"  the 1D ResNet from D-7 step 2: the same building blocks, plus skip
            connections, and twice as many convolution layers
Everything else (data, training loop, metrics) is identical, so the
comparison between them is fair.

What it uses:
  data/processed/X_train.npy, X_val.npy     (from prepare_data.py)
  data/processed/meta_train.csv, meta_val.csv
  data/processed/norm_stats.json            (train-only statistics)

What it NEVER uses: the test set. The test set is touched exactly once, at
the very end of the project.

Run from C:\\Users\\priya\\dt-cardio:
    python train_cnn.py --quick     # 2-minute smoke test: checks it runs, times it
    python train_cnn.py             # full training run

Outputs (in outputs/):
    models/plain_cnn.pt            best model weights (by validation AUROC)
    plain_cnn_history.csv          loss and scores after every epoch
    results_val.csv                one summary row per experiment (keeps growing)
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Settings: every number that affects the result lives here
# ---------------------------------------------------------------------------
CONFIG = {
    "experiment": "resnet1d_100hz",
    "model": "resnet",      # "plain" or "resnet"
    "seed": 42,
    "downsample": 5,        # 5 = 100 Hz (D-9). 1 would mean the original 500 Hz.
    "batch_size": 64,
    "epochs": 20,
    "learning_rate": 1e-3,
    "patience": 4,          # stop if validation AUROC hasn't improved for this many epochs
    "widths": [32, 64, 128, 128],
    "kernel_size": 7,
}

DATA = Path("data/processed")
OUT = Path("outputs")
(OUT / "models").mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    """Fix all random number generators so a rerun gives the same result."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Dataset: hands the model one normalised ECG and its label at a time
# ---------------------------------------------------------------------------
class ECGDataset(Dataset):
    def __init__(self, split, mean, std, downsample):
        # mmap_mode="r": read from disk on demand instead of loading 2 GB into RAM
        self.X = np.load(DATA / f"X_{split}.npy", mmap_mode="r")
        self.y = pd.read_csv(DATA / f"meta_{split}.csv")["label"].to_numpy(np.float32)
        self.mean = mean[:, None]   # shape (12, 1) so it applies to every time point
        self.std = std[:, None]
        self.downsample = downsample

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i].astype(np.float32)          # (12, 5000)
        x = (x - self.mean) / self.std            # scale each lead using TRAIN statistics
        if self.downsample > 1:                   # average neighbouring samples
            x = x.reshape(12, -1, self.downsample).mean(axis=-1)
        return torch.from_numpy(x), torch.tensor(self.y[i])


# ---------------------------------------------------------------------------
# The model: a plain 1D CNN
# ---------------------------------------------------------------------------
class PlainCNN(nn.Module):
    """
    Each block: Conv1d -> BatchNorm -> ReLU -> MaxPool
      Conv1d     slides learned "shape detectors" (filters) along the trace
      BatchNorm  keeps the numbers in a stable range, so training is smoother
      ReLU       lets the network learn non-linear patterns
      MaxPool    halves the length, keeping the strongest responses
    After the blocks, global average pooling turns each channel into one
    number. Those numbers are the EMBEDDING, which the fusion layer will use later.
    """

    def __init__(self, in_leads=12, widths=(32, 64, 128, 128), kernel_size=7):
        super().__init__()
        layers, channels = [], in_leads
        for w in widths:
            layers += [
                nn.Conv1d(channels, w, kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(w),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            channels = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, 1)        # one number out: the "abnormal" score
        self.embedding_size = channels

    def forward(self, x, return_embedding=False):
        z = self.pool(self.features(x)).squeeze(-1)   # (batch, embedding_size)
        logit = self.head(z).squeeze(-1)              # (batch,)
        return (logit, z) if return_embedding else logit


# ---------------------------------------------------------------------------
# The model, version 2: a 1D ResNet (D-7 step 2)
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """
    Two convolution layers, plus a SKIP CONNECTION that adds the block's input
    straight onto its output:

        output = ReLU( conv_path(x) + shortcut(x) )

    Why this helps: the block only has to learn what to ADD to its input,
    not rebuild everything from scratch. If a block has nothing useful to add,
    it can learn to pass the input through almost unchanged. That makes deeper
    networks much easier to train.

    When the block changes the number of channels or halves the length
    (stride 2), the input no longer has the same shape as the output, so the
    shortcut uses a 1x1 convolution to reshape it before adding.
    """

    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))   # <- the skip connection


class ResNet1D(nn.Module):
    """
    stem   -> one convolution to turn 12 leads into the first set of channels
    stages -> for each width: 2 residual blocks; the first one halves the length
    pool   -> average over time, giving the EMBEDDING (as in the plain CNN)
    head   -> one number out: the "abnormal" score
    """

    def __init__(self, in_leads=12, widths=(32, 64, 128, 128), kernel_size=7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_leads, widths[0], kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(),
        )
        blocks, channels = [], widths[0]
        for w in widths:
            blocks.append(ResBlock(channels, w, kernel_size, stride=2))
            blocks.append(ResBlock(w, w, kernel_size, stride=1))
            channels = w
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, 1)
        self.embedding_size = channels

    def forward(self, x, return_embedding=False):
        z = self.pool(self.features(self.stem(x))).squeeze(-1)
        logit = self.head(z).squeeze(-1)
        return (logit, z) if return_embedding else logit


def build_model():
    if CONFIG["model"] == "plain":
        return PlainCNN(widths=CONFIG["widths"], kernel_size=CONFIG["kernel_size"])
    if CONFIG["model"] == "resnet":
        return ResNet1D(widths=CONFIG["widths"], kernel_size=CONFIG["kernel_size"])
    raise ValueError(f"Unknown model {CONFIG['model']}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    probs, labels, total_loss = [], [], 0.0
    for x, y in loader:
        logit = model(x)
        total_loss += loss_fn(logit, y).item() * len(y)
        probs.append(torch.sigmoid(logit).numpy())
        labels.append(y.numpy())
    p, y = np.concatenate(probs), np.concatenate(labels)

    fpr, tpr, _ = roc_curve(y, p)
    sens_at_90_spec = tpr[np.where(fpr <= 0.10)[0][-1]]   # sensitivity when specificity >= 90%

    return {
        "loss": total_loss / len(y),
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "sens_at_90spec": sens_at_90_spec,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main(quick):
    set_seed(CONFIG["seed"])
    stats = json.load(open(DATA / "norm_stats.json"))
    mean, std = np.array(stats["mean"], np.float32), np.array(stats["std"], np.float32)

    train_ds = ECGDataset("train", mean, std, CONFIG["downsample"])
    val_ds = ECGDataset("val", mean, std, CONFIG["downsample"])
    # num_workers=0 avoids multiprocessing problems on Windows
    train_dl = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {CONFIG['model']}, {n_params:,} trainable parameters")
    print(f"Train {len(train_ds)} | val {len(val_ds)} | input shape {tuple(train_ds[0][0].shape)}")

    loss_fn = nn.BCEWithLogitsLoss()   # standard loss for a yes/no prediction
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # --- Quick mode: time 30 batches and estimate a full epoch, then stop
    if quick:
        model.train()
        t0 = time.time()
        for b, (x, y) in enumerate(train_dl):
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            if b == 29:
                break
        per_batch = (time.time() - t0) / 30
        per_epoch = per_batch * len(train_dl)
        print(f"\nQuick test OK. Loss after 30 batches: {loss.item():.3f}")
        print(f"About {per_batch:.2f} s per batch -> about {per_epoch / 60:.1f} min per epoch "
              f"(plus about 1 min of validation).")
        print(f"Full run (up to {CONFIG['epochs']} epochs, early stopping) "
              f"-> roughly {per_epoch * CONFIG['epochs'] / 3600:.1f} h at most.")
        return

    # --- Full training
    history, best_auroc, best_epoch, waited = [], -1.0, 0, 0
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        t0, running = time.time(), 0.0
        for b, (x, y) in enumerate(train_dl):
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            if (b + 1) % 50 == 0:
                print(f"  epoch {epoch} batch {b + 1}/{len(train_dl)}  loss {loss.item():.3f}", end="\r")

        val = evaluate(model, val_dl, loss_fn)
        row = {"epoch": epoch, "train_loss": running / len(train_ds), **{f"val_{k}": v for k, v in val.items()},
               "minutes": (time.time() - t0) / 60}
        history.append(row)
        print(f"Epoch {epoch:2d} | train loss {row['train_loss']:.3f} | val loss {val['loss']:.3f} | "
              f"val AUROC {val['auroc']:.3f} | AUPRC {val['auprc']:.3f} | "
              f"sens@90%spec {val['sens_at_90spec']:.3f} | {row['minutes']:.1f} min")

        if val["auroc"] > best_auroc:
            best_auroc, best_epoch, waited = val["auroc"], epoch, 0
            torch.save(model.state_dict(), OUT / "models" / f"{CONFIG['experiment']}.pt")
        else:
            waited += 1
            if waited >= CONFIG["patience"]:
                print(f"No improvement for {CONFIG['patience']} epochs, stopping early.")
                break

    pd.DataFrame(history).to_csv(OUT / f"{CONFIG['experiment']}_history.csv", index=False)

    best = history[best_epoch - 1]
    summary = {"experiment": CONFIG["experiment"], "best_epoch": best_epoch,
               "val_auroc": best["val_auroc"], "val_auprc": best["val_auprc"],
               "val_sens_at_90spec": best["val_sens_at_90spec"], "config": json.dumps(CONFIG)}
    results_path = OUT / "results_val.csv"
    pd.DataFrame([summary]).to_csv(results_path, mode="a", index=False, header=not results_path.exists())

    print(f"\nBest epoch {best_epoch}: val AUROC {best['val_auroc']:.3f}, "
          f"AUPRC {best['val_auprc']:.3f}, sens@90%spec {best['val_sens_at_90spec']:.3f}")
    print(f"Saved model -> {OUT / 'models' / (CONFIG['experiment'] + '.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="time 30 batches and stop")
    main(parser.parse_args().quick)