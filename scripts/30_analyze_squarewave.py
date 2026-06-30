# scripts/30_analyze_squarewave.py
# Experiment A analysis: per-k linear/nonlinear country probe on the 2-D circle,
# angular-frequency peak, country=1 arc occupancy, overall 8-feature accuracy.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadSquareWave
from src.puzzle.geometry import angular_freq_r2, arc_occupancy

KS = [1, 2, 3, 4, 5]
CI = FEATURE_NAMES.index("country")


def main():
    data = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(data["train_emb"]).float()
    emb_te = torch.from_numpy(data["test_emb"]).float()
    lab_tr, lab_te = data["train_labels"], data["test_labels"]
    rows = []
    for k in KS:
        model = HeadSquareWave(k=k)
        model.load_state_dict(torch.load(
            f"artifacts/results/29_squarewave_k{k}_model.pt", weights_only=True))
        model.eval()
        with torch.no_grad():
            circ_tr = model.circle(emb_tr).numpy()
            circ_te = model.circle(emb_te).numpy()
            overall = ((model(emb_te) > 0).numpy().astype(float) == lab_te).mean()
        sc = StandardScaler().fit(circ_tr)
        Xtr, Xte = sc.transform(circ_tr), sc.transform(circ_te)
        ytr, yte = lab_tr[:, CI], lab_te[:, CI]
        lin = LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte)
        nl = MLPClassifier((32,), max_iter=2000, random_state=0).fit(Xtr, ytr).score(Xte, yte)
        theta = np.arctan2(circ_te[:, 1], circ_te[:, 0])
        r2 = angular_freq_r2(theta, yte, max_k=5)
        arcs = arc_occupancy(theta, yte == 1, n_bins=12)
        rows.append({"k": k, "country_linear": lin, "country_nonlinear": nl,
                     "freq_peak_k": int(np.argmax(r2)) + 1,
                     "country1_arcs": arcs, "overall_acc": float(overall)})
        print(f"k={k}: lin={lin:.3f} nonlin={nl:.3f} peak_k={int(np.argmax(r2))+1} "
              f"arcs={arcs} overall={overall:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv("artifacts/results/30_squarewave_probes.csv", index=False)
    print("\nsaved 30_squarewave_probes.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["k"], df["country_linear"], "o-", label="linear probe")
    ax.plot(df["k"], df["country_nonlinear"], "s-", label="nonlinear probe")
    ax.axhline(0.5, ls="--", c="grey", lw=0.8)
    ax.set_xlabel("harmonic k"); ax.set_ylabel("country probe accuracy")
    ax.set_title("Experiment A — square wave: probe accuracy vs harmonic k")
    ax.legend(); fig.tight_layout()
    fig.savefig("artifacts/results/30_squarewave_curve.png", dpi=150)
    print("saved 30_squarewave_curve.png")


if __name__ == "__main__":
    main()
