# scripts/12_train_jepa.py
"""Historical exploration that used test labels during training diagnostics.

Its outputs are not held-out evidence and are excluded from the final result.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from src.puzzle.models_t3 import Predictor

COUNTRY = 5
OTHER   = [i for i in range(8) if i != COUNTRY]


def train_jepa(n_epochs=400, lr=1e-3, batch_size=512):
    d = np.load("artifacts/activations/acts.npz")
    h2_tr = torch.from_numpy(d["train_h2"]).float()
    h2_te = torch.from_numpy(d["test_h2"]).float()
    lab_tr = torch.from_numpy(d["train_labels"]).float()
    lab_te = torch.from_numpy(d["test_labels"]).float()

    other_tr = lab_tr[:, OTHER]
    other_te = lab_te[:, OTHER]

    predictor = Predictor()
    opt = Adam(predictor.parameters(), lr=lr)

    for epoch in range(n_epochs):
        perm = torch.randperm(len(h2_tr))
        for s in range(0, len(h2_tr), batch_size):
            idx = perm[s:s + batch_size]
            h2_pred = predictor(other_tr[idx])
            loss = F.mse_loss(h2_pred, h2_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                val_loss = F.mse_loss(predictor(other_te), h2_te).item()
            print(f"epoch {epoch+1:3d}: val MSE={val_loss:.4f}")

    torch.save(predictor.state_dict(), "artifacts/results/12_jepa_predictor.pt")

    predictor.eval()
    with torch.no_grad():
        res_tr = (h2_tr - predictor(other_tr)).numpy()
        res_te = (h2_te - predictor(other_te)).numpy()
    np.savez("artifacts/results/12_jepa_residuals.npz",
             res_tr=res_tr, res_te=res_te,
             h2_tr=h2_tr.numpy(), h2_te=h2_te.numpy())
    print("saved 12_jepa_predictor.pt, 12_jepa_residuals.npz")


if __name__ == "__main__":
    train_jepa()
