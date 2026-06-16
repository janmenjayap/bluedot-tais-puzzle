# scripts/24_analyze_bilinear_xor.py
# Probe the bilinear XOR model's h2 for XOR and compare to ReLU XOR experiment.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadBilinearXOR

SENTIMENT = FEATURE_NAMES.index("sentiment")
QUESTION  = FEATURE_NAMES.index("question")
XOR_IDX   = 8   # 9th column in 9-label array


def make_labels_9(labels_np):
    xor = (labels_np[:, SENTIMENT] ^ labels_np[:, QUESTION]).reshape(-1, 1)
    return np.concatenate([labels_np, xor], axis=1)


def main():
    d = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(d["train_emb"]).float()
    emb_te = torch.from_numpy(d["test_emb"]).float()
    lab9_tr = make_labels_9(d["train_labels"])
    lab9_te = make_labels_9(d["test_labels"])

    model = HeadBilinearXOR()
    model.load_state_dict(torch.load("artifacts/results/23_bilinear_xor_model.pt",
                                     weights_only=True))
    model.eval()
    with torch.no_grad():
        h2_tr = model.h2(emb_tr).numpy()
        h2_te = model.h2(emb_te).numpy()

    rows = []
    features = list(FEATURE_NAMES) + ["xor_sentiment_question"]
    for fi, name in enumerate(features):
        y_tr, y_te = lab9_tr[:, fi], lab9_te[:, fi]
        lin = LogisticRegression(max_iter=2000).fit(h2_tr, y_tr).score(h2_te, y_te)
        mlp = MLPClassifier((32,), max_iter=2000, random_state=0).fit(
            h2_tr, y_tr).score(h2_te, y_te)
        rows.append({"feature": name, "linear": lin, "nonlinear": mlp, "gap": mlp - lin})
        print(f"  {name:30s}: linear={lin:.4f}  nonlinear={mlp:.4f}  gap={mlp-lin:+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv("artifacts/results/24_bilinear_xor_probes.csv", index=False)
    print("\nsaved 24_bilinear_xor_probes.csv")

    # Compare XOR probe: bilinear vs ReLU (from 09_xor_probes.csv)
    # 09_xor_probes.csv has columns: base, linear_probe, mlp_probe (single row, XOR only)
    try:
        relu_df = pd.read_csv("artifacts/results/09_xor_probes.csv")
        if "linear_probe" in relu_df.columns:
            relu_lin = relu_df["linear_probe"].values[0]
        elif "linear" in relu_df.columns:
            relu_xor = relu_df[relu_df["feature"] == "xor_sentiment_question"]
            relu_lin = relu_xor["linear"].values[0]
        else:
            raise KeyError("Unrecognised columns in 09_xor_probes.csv")
        bilin_lin = df[df["feature"] == "xor_sentiment_question"]["linear"].values[0]
        print(f"\nXOR linear probe comparison:")
        print(f"  ReLU model h2:     {relu_lin:.4f}")
        print(f"  Bilinear model h2: {bilin_lin:.4f}")
        print(f"  Δ = {bilin_lin - relu_lin:+.4f}  (negative = bilinear is more probe-resistant)")
    except FileNotFoundError:
        print("(09_xor_probes.csv not found — skipping comparison)")


if __name__ == "__main__":
    main()
