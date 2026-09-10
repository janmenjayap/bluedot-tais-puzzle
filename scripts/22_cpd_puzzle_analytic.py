# scripts/22_cpd_puzzle_analytic.py
# Construct a rank-1 quadratic approximation to the country encoding at h2.
# Squaring a selected food direction produces a rank-1 form by construction;
# its predictive accuracy measures how incomplete that approximation is.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.puzzle.data import FEATURE_NAMES

COUNTRY = FEATURE_NAMES.index("country")
FOOD    = FEATURE_NAMES.index("food")


def main():
    d = np.load("artifacts/activations/acts.npz")
    h2_tr, h2_te = d["train_h2"], d["test_h2"]
    lab_tr, lab_te = d["train_labels"], d["test_labels"]
    ctr, cte = lab_tr[:, COUNTRY], lab_te[:, COUNTRY]
    ftr = lab_tr[:, FOOD]

    # Food direction from h2 space (unit vector)
    wf = LogisticRegression(C=1, max_iter=5000).fit(h2_tr, ftr).coef_[0]
    wf_unit = wf / np.linalg.norm(wf)

    # Three country decoders based on the food direction
    proj_tr = h2_tr @ wf_unit          # linear projection
    proj_te = h2_te @ wf_unit

    abs_proj_tr = np.abs(proj_tr)       # abs(proj) — the known decoder
    abs_proj_te = np.abs(proj_te)

    quad_tr = proj_tr ** 2              # proj^2 = h2^T (wf outer wf) h2
    quad_te = proj_te ** 2

    def acc1d(train_feat, test_feat, y_tr, y_te):
        return LogisticRegression(C=1, max_iter=5000).fit(
            train_feat.reshape(-1, 1), y_tr).score(
            test_feat.reshape(-1, 1), y_te)

    acc_raw  = acc1d(proj_tr,     proj_te,     ctr, cte)
    acc_abs  = acc1d(abs_proj_tr, abs_proj_te, ctr, cte)
    acc_quad = acc1d(quad_tr,     quad_te,     ctr, cte)

    print("Country decoder accuracy via food direction:")
    print(f"  raw projection  (linear):              {acc_raw:.4f}  (chance — Z/2 symmetry)")
    print(f"  |projection|    (abs-value):            {acc_abs:.4f}  (known decoder)")
    print(f"  projection^2    (rank-1 approximation): {acc_quad:.4f}")

    # Country=0 has large |proj|, country=1 has small |proj|
    for label, name in [(0, "country=0"), (1, "country=1")]:
        mask_tr = ctr == label
        print(f"\n  {name}: mean|proj|={abs_proj_tr[mask_tr].mean():.3f}  "
              f"mean(proj^2)={quad_tr[mask_tr].mean():.3f}")

    # Cosine similarity between abs_proj and sqrt(quad_proj) [they are the same thing]
    cos_sim = float(np.dot(abs_proj_te, np.sqrt(quad_te)) /
                    (np.linalg.norm(abs_proj_te) * np.linalg.norm(np.sqrt(quad_te)) + 1e-10))
    print(f"\nCosine similarity between |proj| and sqrt(proj^2): {cos_sim:.6f} (expected: 1.0)")

    # This matrix is rank 1 because it is constructed as an outer product.
    B_rank1 = np.outer(wf_unit, wf_unit)  # [64, 64]
    eigvals = np.linalg.eigvalsh(B_rank1)
    print("\nConstructed quadratic matrix B = outer(wf, wf):")
    print(f"  Shape: {B_rank1.shape}, rank: {np.sum(eigvals > 1e-8)}")
    print(f"  Largest eigenvalue: {eigvals[-1]:.6f} (= 1.0 for unit wf)")
    print(f"  Sum of eigenvalues: {eigvals.sum():.6f} (= 1.0 for unit wf)")

    pd.DataFrame([{
        "acc_raw_proj": acc_raw,
        "acc_abs_proj": acc_abs,
        "acc_quad_proj": acc_quad,
        "cos_sim_abs_vs_sqrt_quad": cos_sim,
        "bilinear_tensor_rank": int(np.sum(eigvals > 1e-8)),
        "bilinear_largest_eigval": float(eigvals[-1]),
        "interpretation": "rank-1 quadratic approximation, not an exact model account",
    }]).to_csv("artifacts/results/22_cpd_analytic.csv", index=False)
    print("\nsaved artifacts/results/22_cpd_analytic.csv")


if __name__ == "__main__":
    main()
