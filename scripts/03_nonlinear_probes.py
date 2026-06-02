# scripts/03_nonlinear_probes.py
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.probes import fit_eval_linear, fit_eval_nonlinear

d = np.load("artifacts/activations/acts.npz")
Xtr, Xte = d["train_h2"], d["test_h2"]
Ytr, Yte = d["train_labels"], d["test_labels"]
model_acc = ((d["test_logits"] > 0).astype(int) == Yte).mean(axis=0)

rows = []
for j, name in enumerate(FEATURE_NAMES):
    lin = fit_eval_linear(Xtr, Ytr[:, j], Xte, Yte[:, j])
    non = fit_eval_nonlinear(Xtr, Ytr[:, j], Xte, Yte[:, j])
    rows.append({
        "feature": name,
        "model_acc": model_acc[j],
        "lin_acc": lin["acc"], "lin_auc": lin["auc"],
        "non_acc": non["acc"], "non_auc": non["auc"],
        "gap_acc": non["acc"] - lin["acc"],
        "model_minus_lin": model_acc[j] - lin["acc"],
    })
df = pd.DataFrame(rows).sort_values("gap_acc", ascending=False)
print(df.to_string(index=False))
df.to_csv("artifacts/results/03_gap.csv", index=False)

# Figure: linear vs nonlinear accuracy per feature
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(FEATURE_NAMES))
ax.bar(x - 0.2, df["lin_acc"], 0.4, label="linear probe")
ax.bar(x + 0.2, df["non_acc"], 0.4, label="nonlinear probe")
ax.set_xticks(x); ax.set_xticklabels(df["feature"], rotation=45, ha="right")
ax.set_ylabel("test accuracy"); ax.legend(); ax.set_ylim(0.5, 1.0)
fig.tight_layout(); fig.savefig("artifacts/results/03_gap.png", dpi=150)
print("\nLikely F (largest gap):", df.iloc[0]["feature"])
