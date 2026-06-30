# scripts/32_analyze_fourier_comb.py
# Experiment α analysis: per-multiplexed-feature probes + 8×5 harmonic-confusion heatmap.
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
from src.puzzle.models_t3 import HeadFourierComb
from src.puzzle.geometry import angular_freq_r2

MULTIPLEXED = {5: 2, 3: 3, 4: 4}   # feature_idx -> harmonic


def main():
    data = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(data["train_emb"]).float()
    emb_te = torch.from_numpy(data["test_emb"]).float()
    lab_tr, lab_te = data["train_labels"], data["test_labels"]
    model = HeadFourierComb()
    model.load_state_dict(torch.load(
        "artifacts/results/31_fourier_comb_model.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        circ_tr = model.circle(emb_tr).numpy()
        circ_te = model.circle(emb_te).numpy()
        overall = ((model(emb_te) > 0).numpy().astype(float) == lab_te).mean()
    print(f"overall acc={overall:.4f}")

    sc = StandardScaler().fit(circ_tr)
    Xtr, Xte = sc.transform(circ_tr), sc.transform(circ_te)
    rows = []
    for fi, k in MULTIPLEXED.items():
        ytr, yte = lab_tr[:, fi], lab_te[:, fi]
        lin = LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte)
        nl = MLPClassifier((32,), max_iter=2000, random_state=0).fit(Xtr, ytr).score(Xte, yte)
        rows.append({"feature": FEATURE_NAMES[fi], "harmonic_k": k,
                     "linear": lin, "nonlinear": nl})
        print(f"{FEATURE_NAMES[fi]:10s} (k={k}): lin={lin:.3f} nonlin={nl:.3f}")
    pd.DataFrame(rows).to_csv("artifacts/results/32_fourier_probes.csv", index=False)

    theta = np.arctan2(circ_te[:, 1], circ_te[:, 0])
    conf = np.zeros((8, 5))
    for fi in range(8):
        conf[fi] = angular_freq_r2(theta, lab_te[:, fi], max_k=5)
    conf_df = pd.DataFrame(conf, index=FEATURE_NAMES,
                           columns=[f"k{k}" for k in range(1, 6)])
    conf_df.to_csv("artifacts/results/32_harmonic_confusion.csv")
    print("\nHarmonic-confusion R² (feature × k):")
    print(conf_df.round(3).to_string())

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf, cmap="viridis", aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels([f"k={k}" for k in range(1, 6)])
    ax.set_yticks(range(8)); ax.set_yticklabels(FEATURE_NAMES)
    for fi, k in MULTIPLEXED.items():
        ax.add_patch(plt.Rectangle((k - 1 - 0.5, fi - 0.5), 1, 1,
                                   fill=False, edgecolor="red", lw=2))
    fig.colorbar(im, label="R²")
    ax.set_title("Experiment α — harmonic confusion (red = target harmonic)")
    fig.tight_layout()
    fig.savefig("artifacts/results/32_harmonic_confusion.png", dpi=150)
    print("saved 32_harmonic_confusion.png")


if __name__ == "__main__":
    main()
