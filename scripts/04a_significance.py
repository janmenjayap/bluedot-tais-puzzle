# scripts/04a_significance.py
import numpy as np, pandas as pd
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.probes import LINEAR_MODELS, NONLINEAR_MODELS, _fit_eval, bootstrap_gap_ci

d = np.load("artifacts/activations/acts.npz")
Xtr, Xte = d["train_h2"], d["test_h2"]
Ytr, Yte = d["train_labels"], d["test_labels"]

rows = []
for j, name in enumerate(FEATURE_NAMES):
    lin = {k: _fit_eval(f, Xtr, Ytr[:, j], Xte, Yte[:, j]) for k, f in LINEAR_MODELS.items()}
    non = {k: _fit_eval(f, Xtr, Ytr[:, j], Xte, Yte[:, j]) for k, f in NONLINEAR_MODELS.items()}
    lo, hi = bootstrap_gap_ci(Xtr, Ytr[:, j], Xte, Yte[:, j], n_boot=1000)
    rows.append({"feature": name,
                 "best_lin": max(lin.values()), "best_non": max(non.values()),
                 "gap": max(non.values()) - max(lin.values()),
                 "gap_ci_lo": lo, "gap_ci_hi": hi,
                 **{f"lin_{k}": v for k, v in lin.items()},
                 **{f"non_{k}": v for k, v in non.items()}})
df = pd.DataFrame(rows).sort_values("gap", ascending=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
df.to_csv("artifacts/results/04a_significance.csv", index=False)
print("\nF = feature whose gap_ci_lo > 0 while all others' CIs straddle 0:",
      df.iloc[0]["feature"])
