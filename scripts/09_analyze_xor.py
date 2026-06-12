# scripts/09_analyze_xor.py
"""Analyze XOR/parity representation geometry."""
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadXOR

SENTIMENT = FEATURE_NAMES.index("sentiment")
QUESTION  = FEATURE_NAMES.index("question")


def make_labels_9(labels_np):
    xor = (labels_np[:, SENTIMENT] ^ labels_np[:, QUESTION]).reshape(-1, 1)
    return np.concatenate([labels_np, xor], axis=1)


d = np.load("artifacts/activations/acts.npz")
emb_tr = torch.from_numpy(d["train_emb"]).float()
emb_te = torch.from_numpy(d["test_emb"]).float()
lab_tr_9 = make_labels_9(d["train_labels"])
lab_te_9 = make_labels_9(d["test_labels"])
xor_tr, xor_te = lab_tr_9[:, -1], lab_te_9[:, -1]

model = HeadXOR()
model.load_state_dict(torch.load("artifacts/results/08_xor_model.pt", weights_only=True))
model.eval()

with torch.no_grad():
    h2_tr = model.h2(emb_tr).numpy()
    h2_te = model.h2(emb_te).numpy()

sc = StandardScaler().fit(h2_tr)
Ztr, Zte = sc.transform(h2_tr), sc.transform(h2_te)

lin = LogisticRegression(max_iter=2000).fit(Ztr, xor_tr)
mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500).fit(Ztr, xor_tr)
acc_lin = lin.score(Zte, xor_te)
acc_mlp = mlp.score(Zte, xor_te)
base = max(xor_te.mean(), 1 - xor_te.mean())

print(f"base rate        : {base:.3f}")
print(f"linear probe     : {acc_lin:.3f}  (expected ~0.50)")
print(f"2-layer MLP probe: {acc_mlp:.3f}  (expected >0.85)")

pd.DataFrame([{"base": base, "linear_probe": acc_lin, "mlp_probe": acc_mlp}]
             ).to_csv("artifacts/results/09_xor_probes.csv", index=False)

pca = PCA(n_components=2).fit(Ztr)
Z2_te = pca.transform(Zte)
fig, ax = plt.subplots(figsize=(6, 5))
for v, marker, label in [(0, "o", "xor=0"), (1, "^", "xor=1")]:
    m = xor_te == v
    ax.scatter(Z2_te[m, 0], Z2_te[m, 1], marker=marker, alpha=0.4, s=18, label=label)
ax.set_title(f"XOR/parity in h2 PCA-2D\nlinear={acc_lin:.2f}  mlp={acc_mlp:.2f}")
ax.legend(); fig.tight_layout()
fig.savefig("artifacts/results/09_xor_geometry.png", dpi=150)
print("saved 09_xor_probes.csv, 09_xor_geometry.png")
