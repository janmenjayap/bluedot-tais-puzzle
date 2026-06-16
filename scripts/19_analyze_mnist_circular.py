# scripts/19_analyze_mnist_circular.py
# Analyze the MNIST circular bottleneck: per-class geometry and even/odd probe.
# Key question: does a linear probe fail to decode even/odd from the 2D bottleneck?
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from src.puzzle.models_t3 import HeadMNISTCircular

EVEN_DIGITS = {0, 2, 4, 6, 8}


def load_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)
    X = X.reshape(-1, 1, 28, 28)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=10000, random_state=42, stratify=y)
    return (torch.from_numpy(X_tr), torch.from_numpy(y_tr),
            torch.from_numpy(X_te),  torch.from_numpy(y_te))


def main():
    print("Loading MNIST...")
    X_tr, y_tr, X_te, y_te = load_mnist()

    model = HeadMNISTCircular()
    model.load_state_dict(torch.load("artifacts/results/18_mnist_circular_model.pt",
                                     weights_only=True))
    model.eval()
    with torch.no_grad():
        bottle_tr = model.bottle(X_tr).numpy()
        bottle_te = model.bottle(X_te).numpy()
        logits_te, _ = model(X_te)
        digit_acc = (logits_te.argmax(1) == y_te).float().mean().item()

    print(f"\n10-class digit accuracy: {digit_acc:.4f}")

    # Per-class geometry
    angles_te = np.arctan2(bottle_te[:, 1], bottle_te[:, 0])
    radii_te  = np.linalg.norm(bottle_te, axis=1)
    geo_rows = []
    print("\nPer-class geometry (target angle = digit * 36°):")
    for d in range(10):
        m = y_te.numpy() == d
        target = d * 36.0
        ang_mean = float(np.degrees(np.arctan2(
            np.sin(angles_te[m]).mean(), np.cos(angles_te[m]).mean())))
        ang_std  = float(np.degrees(np.std(angles_te[m])))
        rad_mean = float(radii_te[m].mean())
        print(f"  digit {d} (target {target:.0f}°): mean={ang_mean:.1f}°  std={ang_std:.1f}°  r={rad_mean:.2f}")
        geo_rows.append({"digit": d, "target_deg": target, "mean_angle_deg": ang_mean,
                         "std_angle_deg": ang_std, "mean_radius": rad_mean})

    # Even/odd binary probe
    y_tr_np = y_tr.numpy()
    y_te_np = y_te.numpy()
    even_tr = np.array([1 if d in EVEN_DIGITS else 0 for d in y_tr_np])
    even_te = np.array([1 if d in EVEN_DIGITS else 0 for d in y_te_np])

    lin_acc  = LogisticRegression(max_iter=2000).fit(bottle_tr, even_tr).score(bottle_te, even_te)
    mlp_acc  = MLPClassifier((64,), max_iter=2000, random_state=0).fit(bottle_tr, even_tr).score(bottle_te, even_te)
    base     = max(even_te.mean(), 1 - even_te.mean())

    print(f"\nEven/odd probe on 2D bottleneck:")
    print(f"  base rate      : {base:.3f}")
    print(f"  linear probe   : {lin_acc:.4f}  ← should be near chance")
    print(f"  nonlinear probe: {mlp_acc:.4f}  ← should recover even/odd")

    # Digit probe (10-class) on 2D
    digit_lin = LogisticRegression(max_iter=2000).fit(
        bottle_tr, y_tr_np).score(bottle_te, y_te_np)
    print(f"  digit linear probe (10-class): {digit_lin:.4f}")

    # Figure: 2D bottleneck coloured by digit, even=circle odd=triangle
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.cm.get_cmap("tab10")
    for d in range(10):
        m = y_te_np == d
        marker = "o" if d in EVEN_DIGITS else "^"
        ax.scatter(bottle_te[m, 0], bottle_te[m, 1],
                   c=[cmap(d / 10)] * m.sum(), marker=marker,
                   alpha=0.3, s=12, label=str(d))
    theta = np.linspace(0, 2 * np.pi, 300)
    r_ref = np.median(radii_te)
    ax.plot(r_ref * np.cos(theta), r_ref * np.sin(theta), "k--", lw=0.8, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_title(f"MNIST circular bottleneck\neven/odd linear={lin_acc:.2f}  nonlin={mlp_acc:.2f}")
    ax.legend(title="digit (○=even ▲=odd)", ncol=2, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig("artifacts/results/19_mnist_circular_geometry.png", dpi=150)
    print("\nsaved 19_mnist_circular_geometry.png")

    pd.DataFrame(geo_rows).to_csv("artifacts/results/19_mnist_circular_geometry_stats.csv", index=False)
    pd.DataFrame([{"digit_acc_10class": digit_acc, "digit_lin_probe": digit_lin,
                   "even_odd_linear": lin_acc, "even_odd_nonlinear": mlp_acc,
                   "base_rate": base}
                  ]).to_csv("artifacts/results/19_mnist_circular_probes.csv", index=False)
    print("saved 19_mnist_circular_geometry_stats.csv, 19_mnist_circular_probes.csv")


if __name__ == "__main__":
    main()
