# scripts/13_analyze_jepa.py
"""Analyze JEPA residuals: compare probing h2 vs residual for all 8 features."""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from src.puzzle.data import FEATURE_NAMES

COUNTRY = 5
OTHER   = [i for i in range(8) if i != COUNTRY]

d_orig = np.load("artifacts/activations/acts.npz")
d_res  = np.load("artifacts/results/12_jepa_residuals.npz")

h2_tr, h2_te   = d_res["h2_tr"], d_res["h2_te"]
res_tr, res_te = d_res["res_tr"], d_res["res_te"]
lab_tr = d_orig["train_labels"]
lab_te = d_orig["test_labels"]

sc_h2  = StandardScaler().fit(h2_tr)
sc_res = StandardScaler().fit(res_tr)

def probe(X_tr, X_te, y_tr, y_te):
    return LogisticRegression(max_iter=2000).fit(X_tr, y_tr).score(X_te, y_te)

rows = []
for feat_idx in range(8):
    feat = FEATURE_NAMES[feat_idx]
    y_tr, y_te = lab_tr[:, feat_idx], lab_te[:, feat_idx]
    acc_h2  = probe(sc_h2.transform(h2_tr),   sc_h2.transform(h2_te),   y_tr, y_te)
    acc_res = probe(sc_res.transform(res_tr),  sc_res.transform(res_te), y_tr, y_te)
    rows.append({"feature": feat, "probe_h2": acc_h2, "probe_residual": acc_res,
                 "is_country": feat_idx == COUNTRY})
    print(f"{feat:12s}: h2={acc_h2:.3f}  residual={acc_res:.3f}")

df = pd.DataFrame(rows)
df.to_csv("artifacts/results/13_jepa_probes.csv", index=False)

# Check linearity of P
other_tr = d_orig["train_labels"][:, OTHER].astype(float)
other_te = d_orig["test_labels"][:, OTHER].astype(float)
predicted_h2_tr = h2_tr - res_tr
predicted_h2_te = h2_te - res_te

W = Ridge(alpha=1.0).fit(other_tr, predicted_h2_tr)
mse_linear = np.mean((W.predict(other_te) - predicted_h2_te) ** 2)
mse_total  = np.mean(predicted_h2_te ** 2)
frac = mse_linear / mse_total if mse_total > 0 else float("nan")
print(f"\nLinearity of P: linear approx MSE={mse_linear:.4f}  total var={mse_total:.4f}  unexplained={frac:.3f}")
df["linearity_unexplained_frac"] = frac
df.to_csv("artifacts/results/13_jepa_probes.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(df))
ax.bar(x - 0.2, df["probe_h2"],       0.35, label="raw h2")
ax.bar(x + 0.2, df["probe_residual"], 0.35, label="residual")
ax.set_xticks(x); ax.set_xticklabels(df["feature"], rotation=30, ha="right")
ax.axhline(0.5, color="k", ls="--", lw=0.8, label="chance")
ax.set_ylabel("probe accuracy"); ax.set_title("JEPA residual concentrates country info")
ax.legend(); fig.tight_layout()
fig.savefig("artifacts/results/13_jepa_residual_probe.png", dpi=150)
print("saved 13_jepa_probes.csv, 13_jepa_residual_probe.png")
