# scripts/16_train_superposition.py
"""Historical exploration that monitored test data during training.

Its outputs are not held-out evidence and are excluded from the final result.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadSuper
from src.puzzle.geometry import superposition_loss

COUNTRY = FEATURE_NAMES.index("country")  # 5
FOOD    = FEATURE_NAMES.index("food")     # 3
LAMBDA, LR, EPOCHS, BATCH = 0.5, 1e-3, 500, 512


def train_super():
    d = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(d["train_emb"]).float()
    emb_te = torch.from_numpy(d["test_emb"]).float()
    lab_tr = torch.from_numpy(d["train_labels"]).float()
    lab_te = torch.from_numpy(d["test_labels"]).float()

    model = HeadSuper()
    opt = Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(emb_tr))
        for s in range(0, len(emb_tr), BATCH):
            idx = perm[s:s + BATCH]
            x, y = emb_tr[idx], lab_tr[idx]
            logits, bottle = model(x)
            loss = (F.binary_cross_entropy_with_logits(logits, y)
                    + LAMBDA * superposition_loss(bottle, y[:, COUNTRY].long(), y[:, FOOD].long()))
            opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                logits, _ = model(emb_te)
                acc = ((logits > 0).float() == lab_te).float().mean()
            print(f"epoch {epoch+1:3d}: acc={acc:.3f}")
            model.train()

    model.eval()
    torch.save(model.state_dict(), "artifacts/results/16_super_model.pt")
    print("saved artifacts/results/16_super_model.pt")


if __name__ == "__main__":
    train_super()
