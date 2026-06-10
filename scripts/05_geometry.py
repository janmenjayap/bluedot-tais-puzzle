# scripts/05_geometry.py
# Task 2: characterize HOW country (F) is represented at layer L (h2).
# Runs a menu of discriminating tests, each with an explicit pass/fail reading:
#   T1 capacity      — minimal #hidden units a nonlinear probe needs (how "deep" is the fold?)
#   T2 gating        — does the ReLU on/off PATTERN (sign mask) decode F linearly?
#   T3 conditional   — does F become linearly separable once we condition on another feature G?
#   T4 2D geometry   — fit a 2-unit MLP, view F in the (w1.a, w2.a) plane it uses
#   T5 PCA view      — F in the top-2 PCs of h2 (global multi-cluster check)
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.probes import identify_F

d = np.load("artifacts/activations/acts.npz")
Xtr, Xte = d["train_h2"], d["test_h2"]
Ytr, Yte = d["train_labels"], d["test_labels"]
F, _ = identify_F(Xtr, Ytr, Xte, Yte)
ytr, yte = Ytr[:, F], Yte[:, F]
base = max(yte.mean(), 1 - yte.mean())
print(f"F = {FEATURE_NAMES[F]} (idx {F}); test base={base:.3f}\n")

sc = StandardScaler().fit(Xtr)
Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

def lin_acc(A, ya, B, yb):
    return (LogisticRegression(C=1.0, max_iter=5000).fit(A, ya).predict(B) == yb).mean()

# ---- T1 capacity curve --------------------------------------------------------
print("T1 capacity (min hidden units for nonlinear probe):")
cap = []
for h in [1, 2, 3, 4, 8, 16]:
    acc = (MLPClassifier((h,), max_iter=3000, random_state=0).fit(Ztr, ytr).predict(Zte) == yte).mean()
    cap.append({"hidden": h, "acc": acc}); print(f"   hidden={h:2d}  acc={acc:.3f}")

# ---- T2 gating: sign pattern (ReLU on/off) ------------------------------------
Mtr, Mte = (Xtr > 0).astype(np.float32), (Xte > 0).astype(np.float32)
acc_mask_lin = lin_acc(Mtr, ytr, Mte, yte)          # linear probe on the binary on/off mask
acc_cont_lin = lin_acc(Ztr, ytr, Zte, yte)          # linear probe on continuous values (baseline)
acc_mask_non = (MLPClassifier((64,), max_iter=2000, random_state=0).fit(Mtr, ytr).predict(Mte) == yte).mean()
print(f"\nT2 gating:  linear-on-signmask={acc_mask_lin:.3f}  linear-on-continuous={acc_cont_lin:.3f}"
      f"  nonlinear-on-signmask={acc_mask_non:.3f}")

# ---- T3 conditional on another feature G --------------------------------------
print("\nT3 conditional (within-subset linear acc for F, split by feature G):")
cond_rows = []
for g, gname in enumerate(FEATURE_NAMES):
    if g == F:
        continue
    accs = []
    for v in (0, 1):
        tr_m, te_m = Ytr[:, g] == v, Yte[:, g] == v
        if len(np.unique(ytr[tr_m])) < 2 or te_m.sum() < 20:
            continue
        accs.append(lin_acc(Ztr[tr_m], ytr[tr_m], Zte[te_m], yte[te_m]))
    m = float(np.mean(accs)) if accs else float("nan")
    cond_rows.append({"split_by": gname, "within_subset_lin_acc": m})
    print(f"   split by {gname:10s}: within-subset lin_acc={m:.3f}")

# ---- T4 2-unit MLP geometry ---------------------------------------------------
mlp2 = MLPClassifier((2,), max_iter=5000, random_state=0).fit(Ztr, ytr)
W1 = mlp2.coefs_[0]                 # (64, 2): the two directions the fold uses
P = Zte @ W1                        # (N, 2) projection onto those two directions
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for v, c, lab in [(0, "tab:blue", "F=0"), (1, "tab:red", "F=1")]:
    m = yte == v
    ax[0].scatter(P[m, 0], P[m, 1], s=6, alpha=0.4, c=c, label=lab)
ax[0].set_title(f"{FEATURE_NAMES[F]} in the 2-unit MLP's projection plane"
                f"  (acc={ (mlp2.predict(Zte)==yte).mean():.3f})")
ax[0].set_xlabel("w1 . a"); ax[0].set_ylabel("w2 . a"); ax[0].legend()

# ---- T5 PCA view --------------------------------------------------------------
pca = PCA(n_components=2, random_state=0).fit(Ztr)
Q = pca.transform(Zte)
for v, c, lab in [(0, "tab:blue", "F=0"), (1, "tab:red", "F=1")]:
    m = yte == v
    ax[1].scatter(Q[m, 0], Q[m, 1], s=6, alpha=0.4, c=c, label=lab)
ax[1].set_title(f"{FEATURE_NAMES[F]} in top-2 PCs of h2"
                f"  (var={pca.explained_variance_ratio_[:2].sum():.2f})")
ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2"); ax[1].legend()
fig.tight_layout(); fig.savefig("artifacts/results/05_geometry.png", dpi=150)

pd.DataFrame(cap).to_csv("artifacts/results/05_capacity.csv", index=False)
pd.DataFrame(cond_rows).to_csv("artifacts/results/05_conditional.csv", index=False)
pd.DataFrame([{"linear_signmask": acc_mask_lin, "linear_continuous": acc_cont_lin,
               "nonlinear_signmask": acc_mask_non}]).to_csv("artifacts/results/05_gating.csv", index=False)
print("\nsaved 05_geometry.png, 05_capacity.csv, 05_conditional.csv, 05_gating.csv")
