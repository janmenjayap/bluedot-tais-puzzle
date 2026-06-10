# scripts/06_country_food.py
# Task 2 (mechanism): T3 showed conditioning on `food` makes country linear (0.949) and the
# 2-unit MLP plane showed an interval/non-monotone code. Pin down the exact structure:
#   A independence  — are country & food statistically independent? (rule out a label confound)
#   B within-food directions — fit country direction inside food=0 and food=1; cosine similarity
#   C XOR test      — does the linear probe decode (country XOR food) at h2?
#   D 2x2 geometry  — project h2 on the global country LDA axis; histogram by the 4 (country,food) cells
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.puzzle.data import FEATURE_NAMES

d = np.load("artifacts/activations/acts.npz")
F = FEATURE_NAMES.index("country"); G = FEATURE_NAMES.index("food")
Xtr, Xte = d["train_h2"], d["test_h2"]
ctr, cte = d["train_labels"][:, F], d["test_labels"][:, F]   # country
ftr, fte = d["train_labels"][:, G], d["test_labels"][:, G]   # food
sc = StandardScaler().fit(Xtr); Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

def fit_dir(A, y):
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(A, y)
    return clf.coef_[0], clf

# A independence
ct = pd.crosstab(ctr, ftr, normalize=True)
print("A independence  P(country,food) train:\n", ct.to_string())
print(f"   corr(country,food)={np.corrcoef(ctr, ftr)[0,1]:+.3f}  "
      f"(near 0 => labels independent; entanglement is REPRESENTATIONAL, not in labels)")

# B within-food country directions
w0, _ = fit_dir(Ztr[ftr == 0], ctr[ftr == 0])
w1, _ = fit_dir(Ztr[ftr == 1], ctr[ftr == 1])
cos = float(w0 @ w1 / (np.linalg.norm(w0) * np.linalg.norm(w1)))
print(f"\nB within-food country directions cosine(w_food0, w_food1) = {cos:+.3f}")
print("   (near -1 => country axis FLIPS sign with food -> XOR-like; near +1 => same axis, just shifted)")

# C XOR test: decode (country XOR food)
xor_tr, xor_te = (ctr ^ ftr), (cte ^ fte)
acc_xor = (LogisticRegression(C=1.0, max_iter=5000).fit(Ztr, xor_tr).predict(Zte) == xor_te).mean()
acc_country = (LogisticRegression(C=1.0, max_iter=5000).fit(Ztr, ctr).predict(Zte) == cte).mean()
acc_food = (LogisticRegression(C=1.0, max_iter=5000).fit(Ztr, ftr).predict(Zte) == fte).mean()
print(f"\nC linear decode @h2:  country={acc_country:.3f}  food={acc_food:.3f}  (country XOR food)={acc_xor:.3f}")

# D 2x2 geometry on the global country axis
wc, _ = fit_dir(Ztr, ctr)
proj = Zte @ wc
fig, ax = plt.subplots(figsize=(8, 4))
cells = [(0, 0, "country=0,food=0"), (0, 1, "country=0,food=1"),
         (1, 0, "country=1,food=0"), (1, 1, "country=1,food=1")]
for cv, fv, lab in cells:
    m = (cte == cv) & (fte == fv)
    ax.hist(proj[m], bins=40, alpha=0.5, label=f"{lab} (n={m.sum()})")
ax.set_xlabel("projection on global country direction (h2)"); ax.legend(fontsize=8)
ax.set_title("country (F) is interleaved with food along the country axis")
fig.tight_layout(); fig.savefig("artifacts/results/06_country_food.png", dpi=150)

# group means along the axis
print("\nD mean projection on country axis by (country,food) cell:")
for cv, fv, lab in cells:
    m = (cte == cv) & (fte == fv)
    print(f"   {lab:22s} mean={proj[m].mean():+.3f}")

pd.DataFrame([{"cosine_within_food_dirs": cos, "acc_country": acc_country,
               "acc_food": acc_food, "acc_country_xor_food": acc_xor,
               "corr_country_food": float(np.corrcoef(ctr, ftr)[0, 1])}]
             ).to_csv("artifacts/results/06_country_food.csv", index=False)
print("\nsaved 06_country_food.png, 06_country_food.csv")
