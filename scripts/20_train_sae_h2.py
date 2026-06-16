# scripts/20_train_sae_h2.py
# Train top-k SAE on h2 activations of the puzzle model.
# h2 = model.layers[:6](emb) — the layer where country has Z/2 symmetry encoding.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from src.puzzle.sae import SAE

D_IN, D_FEATS, K = 64, 256, 10


def main():
    d = np.load("artifacts/activations/acts.npz")
    h2_tr = torch.from_numpy(d["train_h2"]).float()
    h2_te = torch.from_numpy(d["test_h2"]).float()

    # Center activations
    mean = h2_tr.mean(0)
    h2_tr_c = h2_tr - mean
    h2_te_c = h2_te - mean
    torch.save(mean, "artifacts/results/20_sae_h2_mean.pt")

    print(f"h2 shape: train={h2_tr.shape}, test={h2_te.shape}")
    print(f"h2 norm mean: {h2_tr.norm(dim=1).mean():.3f}")

    sae = SAE(D_IN, D_FEATS, K)
    print(f"\nTraining SAE (d_in={D_IN}, d_feats={D_FEATS}, k={K})...")
    sae.fit(h2_tr_c, n_steps=3000, batch=512, lr=1e-3)

    sae.eval()
    with torch.no_grad():
        x_hat, _ = sae(h2_te_c)
        mse_te = (h2_te_c - x_hat).pow(2).mean().item()
        nmse_te = mse_te / h2_te_c.pow(2).mean().item()
    print(f"\nTest MSE={mse_te:.5f}  NMSE={nmse_te:.4f}")

    os.makedirs("artifacts/results", exist_ok=True)
    torch.save(sae.state_dict(), "artifacts/results/20_sae_h2.pt")
    print("saved artifacts/results/20_sae_h2.pt")


if __name__ == "__main__":
    main()
