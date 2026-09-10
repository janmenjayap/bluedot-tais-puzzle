# scripts/08_train_xor.py
"""Historical exploration that monitored test data during training.

Its outputs are not held-out evidence and are excluded from the final result.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadXOR

SENTIMENT = FEATURE_NAMES.index("sentiment")   # 4
QUESTION  = FEATURE_NAMES.index("question")    # 1


def make_labels_9(labels_np: np.ndarray) -> np.ndarray:
    xor = (labels_np[:, SENTIMENT] ^ labels_np[:, QUESTION]).reshape(-1, 1)
    return np.concatenate([labels_np, xor], axis=1)


def train_xor(n_epochs=300, lr=1e-3, batch_size=512):
    d = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(d["train_emb"]).float()
    emb_te = torch.from_numpy(d["test_emb"]).float()
    lab_tr = torch.from_numpy(make_labels_9(d["train_labels"])).float()
    lab_te = torch.from_numpy(make_labels_9(d["test_labels"])).float()

    model = HeadXOR()
    opt = Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(len(emb_tr))
        for s in range(0, len(emb_tr), batch_size):
            idx = perm[s:s + batch_size]
            loss = F.binary_cross_entropy_with_logits(model(emb_tr[idx]), lab_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(emb_te)
                acc8  = ((logits[:, :8] > 0).float() == lab_te[:, :8]).float().mean()
                acc_x = ((logits[:, 8]  > 0).float() == lab_te[:, 8]).float().mean()
            print(f"epoch {epoch+1:3d}: 8-feat acc={acc8:.3f}  xor acc={acc_x:.3f}")
            model.train()

    model.eval()
    torch.save(model.state_dict(), "artifacts/results/08_xor_model.pt")
    print("saved artifacts/results/08_xor_model.pt")
    return model


if __name__ == "__main__":
    train_xor()
