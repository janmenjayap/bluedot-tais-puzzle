# scripts/34_analyze_linked_rings.py
# Experiment γ analysis: country probe on 3-D bottleneck, disc-crossing linking number,
# ring-B occupancy, 3-D scatter coloured by country.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadLinkedRings
from src.puzzle.geometry import disc_crossing_count, arc_occupancy

CI = FEATURE_NAMES.index("country")


def main():
    data = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(data["train_emb"]).float()
    emb_te = torch.from_numpy(data["test_emb"]).float()
    lab_tr, lab_te = data["train_labels"], data["test_labels"]
    model = HeadLinkedRings()
    model.load_state_dict(torch.load(
        "artifacts/results/33_linked_rings_model.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        b_tr = model.bottle(emb_tr).numpy()
        b_te = model.bottle(emb_te).numpy()
        overall = ((model(emb_te) > 0).numpy().astype(float) == lab_te).mean()

    sc = StandardScaler().fit(b_tr)
    Xtr, Xte = sc.transform(b_tr), sc.transform(b_te)
    ytr, yte = lab_tr[:, CI], lab_te[:, CI]
    lin = LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte)
    nl = MLPClassifier((32,), max_iter=2000, random_state=0).fit(Xtr, ytr).score(Xte, yte)

    ringB = b_te[lab_te[:, CI] == 1]
    crossings = disc_crossing_count(ringB)
    psi_B = np.arctan2(ringB[:, 2], ringB[:, 0] - 1.0)
    occ_B = arc_occupancy(psi_B, np.ones(len(ringB), dtype=bool), n_bins=12)

    row = {"country_linear": lin, "country_nonlinear": nl,
           "net_disc_crossings": crossings, "ringB_arcs": occ_B,
           "overall_acc": float(overall)}
    pd.DataFrame([row]).to_csv("artifacts/results/34_linked_rings_probes.csv", index=False)
    print(row)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    c0, c1 = b_te[lab_te[:, CI] == 0], b_te[lab_te[:, CI] == 1]
    ax.scatter(c0[:, 0], c0[:, 1], c0[:, 2], c="tab:blue", s=5, alpha=0.4, label="country=0 (ring A)")
    ax.scatter(c1[:, 0], c1[:, 1], c1[:, 2], c="tab:red", s=5, alpha=0.4, label="country=1 (ring B)")
    ax.set_title(f"Experiment γ — linked rings (net crossings={crossings})")
    ax.legend(); fig.tight_layout()
    fig.savefig("artifacts/results/34_linked_rings_scatter.png", dpi=150)
    print("saved 34_linked_rings_scatter.png")


if __name__ == "__main__":
    main()
