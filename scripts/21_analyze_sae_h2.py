# scripts/21_analyze_sae_h2.py
# Analyze which SAE features correspond to country's Z/2 encoding.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from src.puzzle.sae import SAE
from src.puzzle.data import FEATURE_NAMES

COUNTRY = FEATURE_NAMES.index("country")
FOOD    = FEATURE_NAMES.index("food")
D_IN, D_FEATS, K = 64, 256, 10


def main():
    d = np.load("artifacts/activations/acts.npz")
    h2_tr = torch.from_numpy(d["train_h2"]).float()
    h2_te = torch.from_numpy(d["test_h2"]).float()
    lab_tr = d["train_labels"]
    lab_te = d["test_labels"]

    mean = torch.load("artifacts/results/20_sae_h2_mean.pt", weights_only=True)
    h2_tr_c = (h2_tr - mean)
    h2_te_c = (h2_te - mean)

    sae = SAE(D_IN, D_FEATS, K)
    sae.load_state_dict(torch.load("artifacts/results/20_sae_h2.pt", weights_only=True))
    sae.eval()

    with torch.no_grad():
        z_tr = sae.encode(h2_tr_c).numpy()  # [7000, 256]
        z_te = sae.encode(h2_te_c).numpy()  # [1500, 256]

    ctr, cte = lab_tr[:, COUNTRY], lab_te[:, COUNTRY]
    ftr, fte = lab_tr[:, FOOD],    lab_te[:, FOOD]

    # For each SAE feature: mean activation per country group
    rows = []
    for k in range(D_FEATS):
        mean0 = z_tr[ctr == 0, k].mean()
        mean1 = z_tr[ctr == 1, k].mean()
        diff  = abs(mean0 - mean1)
        rows.append({"feature": k, "mean_country0": mean0, "mean_country1": mean1,
                     "abs_diff": diff})
    feat_df = pd.DataFrame(rows).sort_values("abs_diff", ascending=False)
    print("\nTop 10 SAE features by |mean_country0 - mean_country1|:")
    print(feat_df.head(10).to_string(index=False))

    # Top-2 country-discriminating features
    top2 = feat_df["feature"].values[:2]
    print(f"\nTop-2 country features: {top2}")

    # Linear probes on individual features vs pairs
    results = []
    for feat in top2:
        acc = LogisticRegression(max_iter=2000).fit(
            z_tr[:, [feat]], ctr).score(z_te[:, [feat]], cte)
        results.append({"scope": f"feat_{feat}_alone", "linear_acc": acc})

    # Pair of top-2
    acc_pair_lin = LogisticRegression(max_iter=2000).fit(
        z_tr[:, top2], ctr).score(z_te[:, top2], cte)
    acc_pair_mlp = MLPClassifier((32,), max_iter=2000, random_state=0).fit(
        z_tr[:, top2], ctr).score(z_te[:, top2], cte)
    results.append({"scope": "top2_pair_linear", "linear_acc": acc_pair_lin})
    results.append({"scope": "top2_pair_nonlinear", "linear_acc": acc_pair_mlp})

    # Full SAE activation linear probe for country
    acc_full_lin = LogisticRegression(max_iter=2000).fit(z_tr, ctr).score(z_te, cte)
    acc_full_mlp = MLPClassifier((64,), max_iter=2000, random_state=0).fit(
        z_tr, ctr).score(z_te, cte)
    results.append({"scope": "all256_linear", "linear_acc": acc_full_lin})
    results.append({"scope": "all256_nonlinear", "linear_acc": acc_full_mlp})

    probe_df = pd.DataFrame(results)
    print("\nProbe results:")
    print(probe_df.to_string(index=False))

    # Scatter: top-2 features coloured by (country, food)
    fig, ax = plt.subplots(figsize=(6, 5))
    labels_combined = cte * 2 + lab_te[:, FOOD]
    colors = {0: "steelblue", 1: "orange", 2: "green", 3: "red"}
    markers = {0: "o", 1: "o", 2: "^", 3: "^"}
    names = {0: "c=0,f=0", 1: "c=0,f=1", 2: "c=1,f=0", 3: "c=1,f=1"}
    for g in range(4):
        mask = labels_combined == g
        ax.scatter(z_te[mask, top2[0]], z_te[mask, top2[1]],
                   c=colors[g], marker=markers[g], alpha=0.4, s=20, label=names[g])
    ax.set_xlabel(f"SAE feature {top2[0]}")
    ax.set_ylabel(f"SAE feature {top2[1]}")
    ax.set_title("Top-2 country SAE features (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("artifacts/results/21_sae_country_scatter.png", dpi=150)

    feat_df.to_csv("artifacts/results/21_sae_feature_importance.csv", index=False)
    probe_df.to_csv("artifacts/results/21_sae_country_probes.csv", index=False)
    print("\nsaved 21_sae_feature_importance.csv, 21_sae_country_probes.csv, 21_sae_country_scatter.png")


if __name__ == "__main__":
    main()
