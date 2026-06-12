# scripts/15_analyze_helix.py
"""Probe sweep: 1D, 2D angular, 3D linear on helical projection."""
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadHelix

COUNTRY = FEATURE_NAMES.index("country")

d = np.load("artifacts/activations/acts.npz")
emb_tr = torch.from_numpy(d["train_emb"]).float()
emb_te = torch.from_numpy(d["test_emb"]).float()
ctr = d["train_labels"][:, COUNTRY]
cte = d["test_labels"][:, COUNTRY]

model = HeadHelix()
model.load_state_dict(torch.load("artifacts/results/14_helix_model.pt", weights_only=True))
model.eval()
with torch.no_grad():
    _, proj_tr = model(emb_tr); proj_tr = proj_tr.numpy()
    _, proj_te = model(emb_te); proj_te = proj_te.numpy()

def probe_nd(X_tr, X_te, y_tr, y_te):
    sc = StandardScaler().fit(X_tr)
    return LogisticRegression(max_iter=2000).fit(sc.transform(X_tr), y_tr).score(sc.transform(X_te), y_te)

ang_tr = np.arctan2(proj_tr[:, 1], proj_tr[:, 0])
ang_te = np.arctan2(proj_te[:, 1], proj_te[:, 0])
feats_ang_tr = np.stack([np.cos(ang_tr), np.sin(ang_tr)], axis=1)
feats_ang_te = np.stack([np.cos(ang_te), np.sin(ang_te)], axis=1)

acc_1d  = probe_nd(proj_tr[:, :1],  proj_te[:, :1],  ctr, cte)
acc_2d  = probe_nd(feats_ang_tr,    feats_ang_te,    ctr, cte)
acc_3d  = probe_nd(proj_tr,         proj_te,         ctr, cte)
base    = max(cte.mean(), 1 - cte.mean())

print(f"base={base:.3f}  1D={acc_1d:.3f}  2D-angular={acc_2d:.3f}  3D={acc_3d:.3f}")
pd.DataFrame([{"base": base, "probe_1d": acc_1d, "probe_2d_angular": acc_2d, "probe_3d": acc_3d}]
             ).to_csv("artifacts/results/15_helix_probes.csv", index=False)

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")
for v, marker, label in [(0, "o", "country=0"), (1, "^", "country=1")]:
    m = cte == v
    ax.scatter(proj_te[m, 0], proj_te[m, 1], proj_te[m, 2],
               marker=marker, alpha=0.3, s=12, label=label)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title(f"Helical code  3D={acc_3d:.2f}  1D={acc_1d:.2f}")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig("artifacts/results/15_helix_geometry.png", dpi=150)
print("saved 15_helix_probes.csv, 15_helix_geometry.png")
