# scripts/17_analyze_superposition.py
"""Probe country and food in the 2D bottleneck. Show entanglement."""
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadSuper

COUNTRY = FEATURE_NAMES.index("country")
FOOD    = FEATURE_NAMES.index("food")

d = np.load("artifacts/activations/acts.npz")
emb_tr = torch.from_numpy(d["train_emb"]).float()
emb_te = torch.from_numpy(d["test_emb"]).float()
lab_tr = d["train_labels"]
lab_te = d["test_labels"]

model = HeadSuper()
model.load_state_dict(torch.load("artifacts/results/16_super_model.pt", weights_only=True))
model.eval()
with torch.no_grad():
    bottle_tr = model.bottle(emb_tr).numpy()
    bottle_te = model.bottle(emb_te).numpy()

sc = StandardScaler().fit(bottle_tr)
Btr, Bte = sc.transform(bottle_tr), sc.transform(bottle_te)

rows = []
for feat_idx in [COUNTRY, FOOD]:
    y_tr, y_te = lab_tr[:, feat_idx], lab_te[:, feat_idx]
    acc = LogisticRegression(max_iter=2000).fit(Btr, y_tr).score(Bte, y_te)
    feat = FEATURE_NAMES[feat_idx]
    base = max(y_te.mean(), 1 - y_te.mean())
    print(f"{feat}: bottleneck probe={acc:.3f}  base={base:.3f}")
    rows.append({"feature": feat, "probe_bottleneck": acc, "base": base})

pd.DataFrame(rows).to_csv("artifacts/results/17_super_probes.csv", index=False)

fig, ax = plt.subplots(figsize=(5, 5))
markers = {(0,0): ("o", "c=0,f=0"), (1,0): ("^", "c=1,f=0"),
           (0,1): ("s", "c=0,f=1"), (1,1): ("D", "c=1,f=1")}
for (cv, fv), (marker, label) in markers.items():
    m = (lab_te[:, COUNTRY] == cv) & (lab_te[:, FOOD] == fv)
    ax.scatter(bottle_te[m, 0], bottle_te[m, 1], marker=marker, alpha=0.4, s=20, label=label)
ax.set_title("Superposition: country + food in 2D bottleneck")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig("artifacts/results/17_super_geometry.png", dpi=150)
print("saved 17_super_probes.csv, 17_super_geometry.png")
