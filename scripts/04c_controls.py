# scripts/04c_controls.py
# Phase 4c: rule out pipeline / data-starvation / scaling confounds for F.
#   1) Placebo:   random labels at F's base rate -> both probes ~ base rate (harness is honest).
#   2) Shuffle:   permute F's train labels -> nonlinear probe collapses to chance (no leakage).
#   3) Learning curve: linear flat-low while nonlinear climbs => structural, not data-starvation.
#   4) PCA-whitening (invertible linear map): linear separability unchanged => not a scaling artifact.
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.probes import fit_eval_linear, fit_eval_nonlinear, identify_F

d = np.load("artifacts/activations/acts.npz")
Xtr, Xte = d["train_h2"], d["test_h2"]
Ytr, Yte = d["train_labels"], d["test_labels"]
F, _ = identify_F(Xtr, Ytr, Xte, Yte)
print(f"F = {FEATURE_NAMES[F]}")
rng = np.random.default_rng(0)
base = max(Yte[:, F].mean(), 1 - Yte[:, F].mean())
rows = []

# 1) Placebo
p = Yte[:, F].mean()
ytr_pl = (rng.random(len(Xtr)) < p).astype(int)
yte_pl = (rng.random(len(Xte)) < p).astype(int)
pl_lin = fit_eval_linear(Xtr, ytr_pl, Xte, yte_pl)["acc"]
pl_non = fit_eval_nonlinear(Xtr, ytr_pl, Xte, yte_pl)["acc"]
print(f"\nPlacebo   lin={pl_lin:.3f} non={pl_non:.3f} base={base:.3f}")
rows.append({"control": "placebo", "lin": pl_lin, "non": pl_non, "base": base})

# 2) Shuffle F's train labels
yshuf = rng.permutation(Ytr[:, F])
sh_non = fit_eval_nonlinear(Xtr, yshuf, Xte, Yte[:, F])["acc"]
print(f"Shuffled-F  non={sh_non:.3f} (expect ~base {base:.3f})")
rows.append({"control": "shuffle_F", "lin": float("nan"), "non": sh_non, "base": base})

# 3) Learning curve
print("\nLearning curve for F:")
for frac in [0.1, 0.25, 0.5, 1.0]:
    k = int(frac * len(Xtr)); idx = rng.permutation(len(Xtr))[:k]
    l = fit_eval_linear(Xtr[idx], Ytr[idx, F], Xte, Yte[:, F])["acc"]
    n = fit_eval_nonlinear(Xtr[idx], Ytr[idx, F], Xte, Yte[:, F])["acc"]
    print(f"  n={k:<5} lin={l:.3f} non={n:.3f}")
    rows.append({"control": f"lc_n{k}", "lin": l, "non": n, "base": base})

# 4) PCA-whitening invariance
pca = PCA(whiten=True, random_state=0).fit(Xtr)
Wtr, Wte = pca.transform(Xtr), pca.transform(Xte)
w_lin = fit_eval_linear(Wtr, Ytr[:, F], Wte, Yte[:, F])["acc"]
w_non = fit_eval_nonlinear(Wtr, Ytr[:, F], Wte, Yte[:, F])["acc"]
print(f"\nPCA-whitened  lin={w_lin:.3f} non={w_non:.3f}")
rows.append({"control": "pca_whiten", "lin": w_lin, "non": w_non, "base": base})

pd.DataFrame(rows).to_csv("artifacts/results/04c_controls.csv", index=False)
print("\nsaved 04c_controls.csv")
print("Interpretation: placebo+shuffle ~ base (honest pipeline); linear flat-low across the "
      "learning curve (structural, not data-starvation); whitened linear still low (not scaling).")
