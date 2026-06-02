# scripts/02_linear_probes.py
import numpy as np, pandas as pd
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.probes import fit_eval_linear

d = np.load("artifacts/activations/acts.npz")
Xtr, Xte = d["train_h2"], d["test_h2"]            # layer L (post-ReLU hidden2)
Ytr, Yte = d["train_labels"], d["test_labels"]

rows = []
for j, name in enumerate(FEATURE_NAMES):
    r = fit_eval_linear(Xtr, Ytr[:, j], Xte, Yte[:, j])
    rows.append({"feature": name, **{f"lin_{k}": v for k, v in r.items()}})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("artifacts/results/02_linear.csv", index=False)
