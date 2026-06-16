# scripts/28_analyze_bottleneck.py
# Analyze emergent geometry of HeadBottleneck models (d=4 and d=2).
# Metrics: linear/nonlinear probes per feature, angular frequency R² (d=2),
# 2D scatter plots (d=2 direct, d=4 via PCA projection).
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadBottleneck


def run_probes(bottle_tr, bottle_te, lab_tr, lab_te, d_label):
    rows = []
    sc = StandardScaler().fit(bottle_tr)
    X_tr_s, X_te_s = sc.transform(bottle_tr), sc.transform(bottle_te)
    for fi, name in enumerate(FEATURE_NAMES):
        y_tr, y_te = lab_tr[:, fi], lab_te[:, fi]
        lin = LogisticRegression(max_iter=2000).fit(X_tr_s, y_tr).score(X_te_s, y_te)
        mlp = MLPClassifier((32,), max_iter=2000, random_state=0).fit(
            X_tr_s, y_tr).score(X_te_s, y_te)
        rows.append({"d": d_label, "feature": name,
                     "linear": lin, "nonlinear": mlp, "gap": mlp - lin})
    return rows


def angular_freq_r2(theta, y, max_k=4):
    """R² of binary label y regressed on [cos(k*theta), sin(k*theta)] for k=1..max_k."""
    scores = []
    y_f = y.astype(float)
    ss_tot = np.sum((y_f - y_f.mean()) ** 2) + 1e-10
    for k in range(1, max_k + 1):
        X_feat = np.column_stack([np.cos(k * theta), np.sin(k * theta)])
        y_pred = LinearRegression().fit(X_feat, y_f).predict(X_feat)
        ss_res = np.sum((y_f - y_pred) ** 2)
        scores.append(float(1.0 - ss_res / ss_tot))
    return scores


def scatter_grid(bottle_te, lab_te, title, fname):
    """Save a 2×4 scatter grid, one subplot per feature, color = feature value."""
    theta_circ = np.linspace(0, 2 * np.pi, 200)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for fi, name in enumerate(FEATURE_NAMES):
        ax = axes[fi // 4][fi % 4]
        c = lab_te[:, fi]
        ax.scatter(bottle_te[:, 0], bottle_te[:, 1],
                   c=c, cmap="coolwarm", alpha=0.3, s=4)
        ax.plot(np.cos(theta_circ), np.sin(theta_circ), "k--", lw=0.5, alpha=0.3)
        ax.set_title(name, fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def main():
    data = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(data["train_emb"]).float()
    emb_te = torch.from_numpy(data["test_emb"]).float()
    lab_tr = data["train_labels"]
    lab_te = data["test_labels"]
    os.makedirs("artifacts/results", exist_ok=True)

    all_probe_rows = []

    for d in [4, 2]:
        model = HeadBottleneck(d=d)
        model.load_state_dict(torch.load(
            f"artifacts/results/27_bottleneck_d{d}_model.pt", weights_only=True))
        model.eval()

        with torch.no_grad():
            bottle_tr = model.bottle(emb_tr).numpy()
            bottle_te = model.bottle(emb_te).numpy()
            logits_te = model(emb_te)
        acc = ((logits_te > 0).numpy().astype(float) == lab_te).mean()
        print(f"\n=== d={d}: test acc={acc:.4f} ===")

        rows = run_probes(bottle_tr, bottle_te, lab_tr, lab_te, d_label=d)
        all_probe_rows.extend(rows)

        print(f"\nProbes on d={d} bottleneck:")
        print(f"{'feature':12s}  linear  nonlin  gap")
        for r in rows:
            print(f"  {r['feature']:12s}  {r['linear']:.4f}  {r['nonlinear']:.4f}  {r['gap']:+.4f}")

        if d == 2:
            theta_te = np.arctan2(bottle_te[:, 1], bottle_te[:, 0])
            freq_rows = []
            print("\nAngular frequency R² (y ~ cos(k*θ)+sin(k*θ), k=1..4):")
            print(f"{'feature':12s}  k=1    k=2    k=3    k=4    peak_k")
            for fi, name in enumerate(FEATURE_NAMES):
                y_te = lab_te[:, fi]
                r2s = angular_freq_r2(theta_te, y_te)
                peak_k = int(np.argmax(r2s)) + 1
                print(f"  {name:12s}  " + "  ".join(f"{r:.3f}" for r in r2s) + f"  k={peak_k}")
                freq_rows.append({"feature": name,
                                  "k1": r2s[0], "k2": r2s[1],
                                  "k3": r2s[2], "k4": r2s[3],
                                  "peak_k": peak_k})
            pd.DataFrame(freq_rows).to_csv(
                "artifacts/results/28_bottleneck_d2_angular_freq.csv", index=False)

            scatter_grid(bottle_te, lab_te,
                         "d=2 bottleneck — color=feature value (red=1, blue=0)",
                         "artifacts/results/28_bottleneck_d2_scatter.png")
            print("saved 28_bottleneck_d2_scatter.png")

        if d == 4:
            pca = PCA(n_components=2).fit(bottle_tr)
            proj_te = pca.transform(bottle_te)
            scatter_grid(proj_te, lab_te,
                         "d=4 bottleneck — PCA(2) projection — color=feature value",
                         "artifacts/results/28_bottleneck_d4_pca_scatter.png")
            print("saved 28_bottleneck_d4_pca_scatter.png")

    probe_df = pd.DataFrame(all_probe_rows)
    probe_df.to_csv("artifacts/results/28_bottleneck_probes.csv", index=False)
    print("\nsaved 28_bottleneck_probes.csv")
    print("\nFull probe table:")
    print(probe_df.to_string(index=False))


if __name__ == "__main__":
    main()
