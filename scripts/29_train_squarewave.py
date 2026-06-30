# scripts/29_train_squarewave.py
# Experiment A — square wave: country constrained to the k-th harmonic of a 2-D circle.
# Sweep k=1..5; the other 7 co-trained features are the spreader. BCE on all 8 outputs.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.puzzle.models_t3 import HeadSquareWave

N_EPOCHS, LR, BATCH = 600, 1e-3, 512
KS = [1, 2, 3, 4, 5]


def train_one(k, emb_tr, lab_tr, emb_te, lab_te):
    torch.manual_seed(0)
    model = HeadSquareWave(k=k)
    opt = Adam(model.parameters(), lr=LR)
    sched = CosineAnnealingLR(opt, N_EPOCHS)
    model.train()
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(len(emb_tr))
        for s in range(0, len(emb_tr), BATCH):
            idx = perm[s:s + BATCH]
            loss = F.binary_cross_entropy_with_logits(model(emb_tr[idx]), lab_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch + 1) % 150 == 0:
            model.eval()
            with torch.no_grad():
                acc = ((model(emb_te) > 0).float() == lab_te).float().mean()
            print(f"k={k} epoch {epoch+1:3d}: acc={acc:.4f}")
            model.train()
    model.eval()
    return model


def main():
    data = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(data["train_emb"]).float()
    emb_te = torch.from_numpy(data["test_emb"]).float()
    lab_tr = torch.from_numpy(data["train_labels"]).float()
    lab_te = torch.from_numpy(data["test_labels"]).float()
    os.makedirs("artifacts/results", exist_ok=True)
    for k in KS:
        print(f"\n=== Experiment A: k={k} ===")
        model = train_one(k, emb_tr, lab_tr, emb_te, lab_te)
        path = f"artifacts/results/29_squarewave_k{k}_model.pt"
        torch.save(model.state_dict(), path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
