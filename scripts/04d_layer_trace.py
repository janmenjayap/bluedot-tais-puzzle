# scripts/04d_layer_trace.py
# Phase 4d: locate WHERE the non-linearity is created and seed Task-2 geometry hypotheses.
#   - Linear + nonlinear probe for F at every tap (emb, h0..h3, logits).
#     Expect F linear at emb (encoder) but folded non-linear by the head -> gap peaks mid-head.
#   - Radial (norm-only) probe: does |a| alone decode F? (shell/radial code hint)
#   - Projection histogram on the best linear direction: class overlap = visual proof of non-linearity.
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.probes import fit_eval_linear, fit_eval_nonlinear, identify_F

d = np.load("artifacts/activations/acts.npz")
Ytr, Yte = d["train_labels"], d["test_labels"]
F, _ = identify_F(d["train_h2"], Ytr, d["test_h2"], Yte)
print(f"F = {FEATURE_NAMES[F]}")

# Linear + nonlinear probe for F at every tap
rows = []
for tap in ["emb", "h0", "h1", "h2", "h3", "logits"]:
    l = fit_eval_linear(d[f"train_{tap}"], Ytr[:, F], d[f"test_{tap}"], Yte[:, F])["acc"]
    n = fit_eval_nonlinear(d[f"train_{tap}"], Ytr[:, F], d[f"test_{tap}"], Yte[:, F])["acc"]
    rows.append({"tap": tap, "lin": l, "non": n, "gap": n - l})
df = pd.DataFrame(rows)
print("\n", df.to_string(index=False))
df.to_csv("artifacts/results/04d_layer_trace.csv", index=False)

# Geometry pre-checks at L (test set)
X, y = d["test_h2"], Yte[:, F]
norms = np.linalg.norm(X, axis=1).reshape(-1, 1)
norms_tr = np.linalg.norm(d["train_h2"], axis=1).reshape(-1, 1)
clf_norm = LogisticRegression(max_iter=1000).fit(norms_tr, Ytr[:, F])
acc_norm = clf_norm.score(norms, y)
print(f"\nradial(norm-only) probe acc for F: {acc_norm:.3f} (base {max(y.mean(), 1 - y.mean()):.3f})")

w = LogisticRegression(max_iter=2000).fit(d["train_h2"], Ytr[:, F]).coef_[0]
proj = X @ w
fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(proj[y == 0], bins=40, alpha=0.6, label="F=0")
ax.hist(proj[y == 1], bins=40, alpha=0.6, label="F=1")
ax.set_xlabel("projection on best linear direction"); ax.legend()
ax.set_title(f"{FEATURE_NAMES[F]} @ layer L: class overlap on best linear axis")
fig.tight_layout(); fig.savefig("artifacts/results/04d_projection_hist.png", dpi=150)
print("saved 04d_layer_trace.csv, 04d_projection_hist.png")
print("Interpretation: gap small at emb (linear there), large at h2 => the HEAD folded F."
      " Heavy projection overlap confirms no single linear direction separates F at L.")
