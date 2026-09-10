# scripts/27_train_bottleneck.py
# HISTORICAL EXPLORATION: this script monitored test data during training.
# Its outputs are not held-out evidence and are excluded from the final result.
# Train HeadBottleneck on all 8 text features with d=4 and d=2.
# Architecture: emb(384) → [64→ReLU→64→ReLU→d] → unit_norm → 8×MLP_heads → logits
# Unit-norm on bottleneck forces all representations onto the d-sphere.
# Per-feature MLP decoders achieve high accuracy regardless of bottleneck geometry.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.puzzle.models_t3 import HeadBottleneck

N_EPOCHS, LR, BATCH = 600, 1e-3, 512


def train_one(d: int, emb_tr, lab_tr, emb_te, lab_te):
    model = HeadBottleneck(d=d)
    opt   = Adam(model.parameters(), lr=LR)
    sched = CosineAnnealingLR(opt, N_EPOCHS)

    model.train()
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(len(emb_tr))
        for s in range(0, len(emb_tr), BATCH):
            idx = perm[s:s + BATCH]
            logits = model(emb_tr[idx])
            loss = F.binary_cross_entropy_with_logits(logits, lab_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                logits_te = model(emb_te)
                acc = ((logits_te > 0).float() == lab_te).float().mean()
            print(f"d={d}  epoch {epoch+1:3d}: acc={acc:.4f}")
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
    for dim in [4, 2]:
        print(f"\n=== Training d={dim} ===")
        model = train_one(dim, emb_tr, lab_tr, emb_te, lab_te)
        path = f"artifacts/results/27_bottleneck_d{dim}_model.pt"
        torch.save(model.state_dict(), path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
