# scripts/11_analyze_rotation.py
"""Analyze rotational/SO(2) encoding of country."""
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadRotation

COUNTRY = FEATURE_NAMES.index("country")

d = np.load("artifacts/activations/acts.npz")
emb_tr = torch.from_numpy(d["train_emb"]).float()
emb_te = torch.from_numpy(d["test_emb"]).float()
ctr = d["train_labels"][:, COUNTRY]
cte = d["test_labels"][:, COUNTRY]

model = HeadRotation()
model.load_state_dict(torch.load("artifacts/results/10_rotation_model.pt", weights_only=True))
model.eval()

with torch.no_grad():
    h2_tr = model.h2(emb_tr).numpy()
    h2_te = model.h2(emb_te).numpy()
    _, proj_tr = model(emb_tr); proj_tr = proj_tr.numpy()
    _, proj_te = model(emb_te); proj_te = proj_te.numpy()

sc = StandardScaler().fit(h2_tr)
Ztr, Zte = sc.transform(h2_tr), sc.transform(h2_te)

lin_h2 = LogisticRegression(max_iter=2000).fit(Ztr, ctr).score(Zte, cte)
lin_2d = LogisticRegression(max_iter=2000).fit(proj_tr, ctr).score(proj_te, cte)

angles_tr = np.arctan2(proj_tr[:, 1], proj_tr[:, 0])
angles_te = np.arctan2(proj_te[:, 1], proj_te[:, 0])
feats_tr = np.stack([np.cos(angles_tr), np.sin(angles_tr)], axis=1)
feats_te = np.stack([np.cos(angles_te), np.sin(angles_te)], axis=1)
ang_clf = LogisticRegression(max_iter=2000).fit(feats_tr, ctr)
acc_angular = ang_clf.score(feats_te, cte)

base = max(cte.mean(), 1 - cte.mean())
print(f"base rate            : {base:.3f}")
print(f"linear probe (h2)    : {lin_h2:.3f}")
print(f"linear probe (2D)    : {lin_2d:.3f}")
print(f"angular decoder (2D) : {acc_angular:.3f}  (expected >0.85)")

pd.DataFrame([{"base": base, "linear_h2": lin_h2, "linear_2d": lin_2d, "angular": acc_angular}]
             ).to_csv("artifacts/results/11_rotation_probes.csv", index=False)

fig, ax = plt.subplots(figsize=(5, 5))
for v, marker, label in [(0, "o", "country=0"), (1, "^", "country=1")]:
    m = cte == v
    ax.scatter(proj_te[m, 0], proj_te[m, 1], marker=marker, alpha=0.4, s=18, label=label)
theta = np.linspace(0, 2 * np.pi, 200)
r = np.abs(proj_te).max() * 0.8
ax.plot(r * np.cos(theta), r * np.sin(theta), "k--", lw=0.8, label="unit circle")
ax.set_aspect("equal"); ax.legend(fontsize=8)
ax.set_title(f"Rotational code (SO(2))\nangular acc={acc_angular:.2f}  linear h2={lin_h2:.2f}")
fig.tight_layout(); fig.savefig("artifacts/results/11_rotation_geometry.png", dpi=150)
print("saved 11_rotation_probes.csv, 11_rotation_geometry.png")
