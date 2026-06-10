# scripts/07_mechanism.py
# Task 2 (final mechanism): country @ layer L is an ABSOLUTE-VALUE / interval code along the
# food axis. country=1 <=> SMALL |proj on food direction|; country=0 <=> LARGE |proj|.
# This single nonlinearity explains every earlier observation (linear fails, 2 ReLUs suffice,
# conditioning on food linearizes, within-food directions antiparallel, h3 recovers linearity).
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.puzzle.data import FEATURE_NAMES

d = np.load("artifacts/activations/acts.npz")
F = FEATURE_NAMES.index("country"); G = FEATURE_NAMES.index("food")
Xtr, Xte = d["train_h2"], d["test_h2"]
ctr, cte = d["train_labels"][:, F], d["test_labels"][:, F]
ftr, fte = d["train_labels"][:, G], d["test_labels"][:, G]
sc = StandardScaler().fit(Xtr); Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
base = max(cte.mean(), 1 - cte.mean())

# Food axis (linear direction encoding food), and projections.
wf = LogisticRegression(C=1, max_iter=5000).fit(Ztr, ftr).coef_[0]
ptr, pte = Ztr @ wf, Zte @ wf

def acc1(ftr_, fte_):
    clf = LogisticRegression(max_iter=5000).fit(ftr_.reshape(-1, 1), ctr)
    return (clf.predict(fte_.reshape(-1, 1)) == cte).mean()

raw = acc1(ptr, pte); absv = acc1(np.abs(ptr), np.abs(pte)); sq = acc1(ptr ** 2, pte ** 2)
print(f"country from a SINGLE scalar along the food axis (base={base:.3f}):")
print(f"   raw signed projection : {raw:.3f}  (chance -> not a half-space)")
print(f"   |projection|          : {absv:.3f}  (recovers country -> magnitude code)")
print(f"   projection^2          : {sq:.3f}")

# Within-food country directions: antiparallel?
w0 = LogisticRegression(C=1, max_iter=5000).fit(Ztr[ftr == 0], ctr[ftr == 0]).coef_[0]
w1 = LogisticRegression(C=1, max_iter=5000).fit(Ztr[ftr == 1], ctr[ftr == 1]).coef_[0]
cos = float(w0 @ w1 / (np.linalg.norm(w0) * np.linalg.norm(w1)))
print(f"\ncosine(country-dir|food0, country-dir|food1) = {cos:+.3f} (antiparallel => sign flips with food)")

# Figure: distribution along the food axis, split by the 4 (country,food) cells.
fig, ax = plt.subplots(figsize=(8, 4.2))
for cv in (0, 1):
    for fv in (0, 1):
        m = (cte == cv) & (fte == fv)
        style = dict(bins=45, alpha=0.55,
                     label=f"country={cv},food={fv}  mean={pte[m].mean():+.1f}")
        ax.hist(pte[m], **style)
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("projection on food axis  (h2, layer L)")
ax.set_title("country = SMALL |proj on food axis|  (|proj| decodes country at "
             f"{absv:.2f}; raw proj at {raw:.2f})")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("artifacts/results/07_mechanism.png", dpi=150)

pd.DataFrame([{"acc_raw_proj": raw, "acc_abs_proj": absv, "acc_sq_proj": sq,
               "cosine_within_food_dirs": cos, "base_rate": base}]
             ).to_csv("artifacts/results/07_mechanism.csv", index=False)
print("\nsaved 07_mechanism.png, 07_mechanism.csv")
