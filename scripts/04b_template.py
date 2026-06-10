# scripts/04b_template.py
# Phase 4b: rule out the "linear failure is a template_id artifact" confound.
#   (1) Leave-one-template-out CV: does the linear/nonlinear split survive on UNSEEN templates?
#   (2) Within-template linear separability: is F linearly decodable inside a single template?
# If F is non-linear even within one template AND under LOTO, the nonlinearity is intrinsic.
import numpy as np, pandas as pd
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.probes import fit_eval_linear, loto_cv, identify_F, NONLINEAR_MODELS

d = np.load("artifacts/activations/acts.npz")
X = np.concatenate([d["train_h2"], d["test_h2"]])
Y = np.concatenate([d["train_labels"], d["test_labels"]])
T = np.concatenate([d["train_tids"], d["test_tids"]])
F, _ = identify_F(d["train_h2"], d["train_labels"], d["test_h2"], d["test_labels"])
print(f"F = {FEATURE_NAMES[F]} (idx {F}); #templates = {len(np.unique(T))}")

lin_loto = loto_cv(X, Y[:, F], T, fit_eval_fn=fit_eval_linear)
non_loto = loto_cv(X, Y[:, F], T, make_clf=NONLINEAR_MODELS["mlp64"])
print(f"\nLeave-one-template-out CV for F:  linear={lin_loto:.3f}  nonlinear={non_loto:.3f}")

print("\nWithin-template linear-probe accuracy for F:")
rng = np.random.default_rng(0)
rows = []
for t in np.unique(T):
    m = T == t
    if m.sum() < 50 or len(np.unique(Y[m, F])) < 2:
        continue
    idx = rng.permutation(np.where(m)[0]); cut = int(0.7 * len(idx))
    r = fit_eval_linear(X[idx[:cut]], Y[idx[:cut], F], X[idx[cut:]], Y[idx[cut:], F])
    print(f"  template {t:>2}: n={m.sum():>4} pos_rate={Y[m, F].mean():.2f} lin_acc={r['acc']:.3f}")
    rows.append({"template": int(t), "n": int(m.sum()),
                 "pos_rate": float(Y[m, F].mean()), "within_lin_acc": r["acc"]})

df = pd.DataFrame(rows)
df.attrs["F"] = FEATURE_NAMES[F]
df.to_csv("artifacts/results/04b_within_template.csv", index=False)
pd.DataFrame([{"feature": FEATURE_NAMES[F], "loto_linear": lin_loto,
               "loto_nonlinear": non_loto,
               "within_template_lin_mean": float(df["within_lin_acc"].mean())}]
             ).to_csv("artifacts/results/04b_loto.csv", index=False)
print("\nsaved 04b_within_template.csv, 04b_loto.csv")
print("Interpretation: LOTO linear low + within-template linear low => intrinsic non-linearity,"
      " not a template artifact.")
