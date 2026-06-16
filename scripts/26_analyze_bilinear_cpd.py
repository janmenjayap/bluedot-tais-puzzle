# scripts/26_analyze_bilinear_cpd.py
# CPD weight-space analysis of the bilinear text model.
# Since logit_f = W_head[f,:] @ ((W_L@emb)*(W_R@emb)), the model IS its CPD:
#   B[f,i,j] = Σ_r D[f,r] L[i,r] R[j,r]   with D=W_head, L=W_L.T, R=W_R.T
# We analyze per-component sigma, specialization, and direction structure.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadBilinear

COUNTRY = FEATURE_NAMES.index("country")
FOOD    = FEATURE_NAMES.index("food")


def main():
    d = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(d["train_emb"]).float()
    emb_te = torch.from_numpy(d["test_emb"]).float()
    lab_tr = d["train_labels"]
    lab_te = d["test_labels"]

    model = HeadBilinear()
    model.load_state_dict(torch.load("artifacts/results/25_bilinear_text_model.pt",
                                     weights_only=True))
    model.eval()

    # Extract CPD factors
    L, R, D = model.cpd_factors()   # L,R: [384, 64]  D: [8, 64]
    L_np, R_np, D_np = L.numpy(), R.numpy(), D.numpy()

    # Per-component sigma = ||L[:,r]|| * ||R[:,r]|| * ||D[:,r]||
    sigma = (np.linalg.norm(L_np, axis=0) *
             np.linalg.norm(R_np, axis=0) *
             np.linalg.norm(D_np, axis=0))
    order = np.argsort(sigma)[::-1]

    # Specialization: max |D[f,r]| / sum |D[:,r]| — what fraction goes to the top feature
    D_abs = np.abs(D_np)
    spec_max  = (D_abs.max(axis=0) / (D_abs.sum(axis=0) + 1e-8))
    spec_argmax = D_abs.argmax(axis=0)  # dominant feature index per component

    # XOR-like asymmetry: ||L-R|| / ||L|| per component
    asym = np.linalg.norm(L_np - R_np, axis=0) / (np.linalg.norm(L_np, axis=0) + 1e-8)

    print(f"\nTop-20 components by sigma:")
    print(f"{'rank':>4} {'comp':>4} {'sigma':>8} {'spec':>6} {'top_feat':>10} {'asym(L-R)':>10}")
    rows = []
    for rank_i, r in enumerate(order[:20]):
        fname = FEATURE_NAMES[spec_argmax[r]]
        print(f"{rank_i+1:4d} {r:4d} {sigma[r]:8.3f} {spec_max[r]:6.3f} "
              f"{fname:>10} {asym[r]:10.3f}")
        rows.append({"rank": rank_i + 1, "component": int(r), "sigma": float(sigma[r]),
                     "spec_max": float(spec_max[r]), "top_feature": fname,
                     "asym_lr": float(asym[r])})

    # How many components does country capture?
    country_D = D_np[COUNTRY]  # [64]
    country_sigma = np.abs(country_D) * np.linalg.norm(L_np, axis=0) * np.linalg.norm(R_np, axis=0)
    country_order = np.argsort(country_sigma)[::-1]
    print(f"\nTop-5 country components (by |D[country,r]| * ||L[:,r]|| * ||R[:,r]||):")
    for rank_i, r in enumerate(country_order[:5]):
        print(f"  rank {rank_i+1}: comp={r}  sigma_country={country_sigma[r]:.3f}  "
              f"asym={asym[r]:.3f}  L+R_norm={np.linalg.norm(L_np[:,r]+R_np[:,r]):.3f}")

    # Linear/nonlinear probes on bilinear h2
    with torch.no_grad():
        h2_tr = model.h2(emb_tr).numpy()
        h2_te = model.h2(emb_te).numpy()

    probe_rows = []
    for fi, name in enumerate(FEATURE_NAMES):
        y_tr, y_te = lab_tr[:, fi], lab_te[:, fi]
        lin = LogisticRegression(max_iter=2000).fit(h2_tr, y_tr).score(h2_te, y_te)
        mlp = MLPClassifier((32,), max_iter=2000, random_state=0).fit(
            h2_tr, y_tr).score(h2_te, y_te)
        probe_rows.append({"feature": name, "linear": lin, "nonlinear": mlp, "gap": mlp - lin})
    probe_df = pd.DataFrame(probe_rows)
    print("\nProbes on bilinear h2:")
    print(probe_df.to_string(index=False))

    # Figure: scatter country vs food probes, colored by component specialization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(range(64), sigma[order], c="steelblue", s=20)
    ax.set_xlabel("Component rank"); ax.set_ylabel("σ"); ax.set_title("Component σ spectrum")
    ax = axes[1]
    ax.scatter(range(64), spec_max[order], c="orange", s=20)
    ax.set_xlabel("Component rank"); ax.set_ylabel("Specialization (max/Σ)")
    ax.set_title("Per-component specialization")
    fig.tight_layout()
    fig.savefig("artifacts/results/26_bilinear_cpd_spectrum.png", dpi=150)

    pd.DataFrame(rows).to_csv("artifacts/results/26_bilinear_cpd_components.csv", index=False)
    probe_df.to_csv("artifacts/results/26_bilinear_probes.csv", index=False)
    print("\nsaved 26_bilinear_cpd_components.csv, 26_bilinear_probes.csv, 26_bilinear_cpd_spectrum.png")


if __name__ == "__main__":
    main()
