# scripts/25_train_bilinear_text.py
# Train bilinear text model: emb -> (W_L@emb)*(W_R@emb) -> head -> 8 logits.
# The model is its own CPD decomposition — no separate CPD training needed.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.puzzle.models_t3 import HeadBilinear

N_EPOCHS, LR, BATCH = 400, 1e-3, 512


def train():
    d = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(d["train_emb"]).float()
    emb_te = torch.from_numpy(d["test_emb"]).float()
    lab_tr = torch.from_numpy(d["train_labels"]).float()
    lab_te = torch.from_numpy(d["test_labels"]).float()

    model = HeadBilinear()
    opt   = Adam(model.parameters(), lr=LR)
    sched = CosineAnnealingLR(opt, N_EPOCHS)

    model.train()
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(len(emb_tr))
        for s in range(0, len(emb_tr), BATCH):
            idx = perm[s:s + BATCH]
            logits, _ = model(emb_tr[idx])
            loss = F.binary_cross_entropy_with_logits(logits, lab_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                logits_te, _ = model(emb_te)
                acc = ((logits_te > 0).float() == lab_te).float().mean()
            print(f"epoch {epoch+1:3d}: acc={acc:.4f}")
            model.train()

    model.eval()
    os.makedirs("artifacts/results", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/results/25_bilinear_text_model.pt")
    print("saved artifacts/results/25_bilinear_text_model.pt")


if __name__ == "__main__":
    train()
