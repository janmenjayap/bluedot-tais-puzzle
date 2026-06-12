# scripts/14_train_helix.py
"""Train HeadHelix with BCE + helical regulariser on country."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadHelix
from src.puzzle.geometry import helical_loss

COUNTRY = FEATURE_NAMES.index("country")
LAMBDA, LR, EPOCHS, BATCH = 0.5, 1e-3, 400, 512


def train_helix():
    d = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(d["train_emb"]).float()
    emb_te = torch.from_numpy(d["test_emb"]).float()
    lab_tr = torch.from_numpy(d["train_labels"]).float()
    lab_te = torch.from_numpy(d["test_labels"]).float()

    model = HeadHelix()
    opt = Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(emb_tr))
        for s in range(0, len(emb_tr), BATCH):
            idx = perm[s:s + BATCH]
            x, y = emb_tr[idx], lab_tr[idx]
            logits, proj = model(x)
            loss = (F.binary_cross_entropy_with_logits(logits, y)
                    + LAMBDA * helical_loss(proj, y[:, COUNTRY].long()))
            opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                logits, _ = model(emb_te)
                acc = ((logits > 0).float() == lab_te).float().mean()
            print(f"epoch {epoch+1:3d}: acc={acc:.3f}")
            model.train()

    model.eval()
    torch.save(model.state_dict(), "artifacts/results/14_helix_model.pt")
    print("saved artifacts/results/14_helix_model.pt")


if __name__ == "__main__":
    train_helix()
