# scripts/01_sanity_check.py
import numpy as np, pandas as pd
from src.puzzle.data import FEATURE_NAMES

d = np.load("artifacts/activations/acts.npz")
logits = d["test_logits"]                  # (1500, 8)
y = d["test_labels"]                       # (1500, 8)
pred = (logits > 0).astype(int)            # sigmoid(z)>0.5 <=> z>0
acc = (pred == y).mean(axis=0)
base = y.mean(axis=0)                       # positive base rate per feature
df = pd.DataFrame({"feature": FEATURE_NAMES, "test_acc": acc, "pos_rate": base})
print(df.to_string(index=False))
df.to_csv("artifacts/results/01_sanity.csv", index=False)
assert (acc >= 0.95).all(), f"Model below 95% on: {df[df.test_acc<0.95]}"
print("\nSANITY PASS: all features >= 95% test accuracy")
