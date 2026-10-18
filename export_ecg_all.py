"""
export_ecg_all.py: compact copy of ALL test-set ECGs for the hosted app.

The full arrays (X_test.npy, 257 MB; saliency_test.npy, 51 MB) are too large for
GitHub. This stores every test patient's 12-lead ECG at 250 Hz as 8-bit integers
with one scale per lead (step = max|amplitude| / 127, well under 0.02 mV for a
normal ECG), and the saliency map averaged to 25 Hz as 8-bit values in [0, 1] (it is drawn as a soft shading band).
Display only: the model and every reported result use the original arrays.

Output: data/demo/ecg_all.npz  (ecg_id, ecg_q int8, ecg_scale float32, sal_q uint8)
"""
from pathlib import Path
import numpy as np
import pandas as pd

X = np.load("data/processed/X_test.npy", mmap_mode="r")          # (n, 12, 5000) at 500 Hz
S = np.load("outputs/saliency/saliency_test.npy", mmap_mode="r")  # (n, 12, 1000), 0..1
ids = pd.read_csv("data/processed/meta_test.csv")["ecg_id"].to_numpy()
n = len(ids)
ecg_q = np.empty((n, 12, 2500), np.int8)
scale = np.empty((n, 12), np.float32)
sal_q = np.empty((n, 12, 250), np.uint8)
for s in range(0, n, 128):
    x = np.asarray(X[s:s + 128], np.float32)[:, :, ::2]           # 250 Hz, as the app shows
    m = np.abs(x).max(axis=2, keepdims=True); m[m == 0] = 1
    ecg_q[s:s + 128] = np.round(x / m * 127).astype(np.int8)
    scale[s:s + 128] = (m[..., 0] / 127)
    sal = np.clip(np.asarray(S[s:s + 128], np.float32), 0, 1).reshape(-1, 12, 250, 4).mean(axis=3)  # 25 Hz is plenty for a shading band
    sal_q[s:s + 128] = np.round(sal * 255).astype(np.uint8)
np.savez_compressed("data/demo/ecg_all.npz", ecg_id=ids, ecg_q=ecg_q, ecg_scale=scale, sal_q=sal_q)
print(n, "patients saved")
