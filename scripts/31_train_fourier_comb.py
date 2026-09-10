# scripts/31_train_fourier_comb.py
# HISTORICAL EXPLORATION: this script monitored test data during training.
# Its outputs are not held-out evidence and are excluded from the final result.
# Experiment α — Fourier comb: multiplex country(k=2), food(k=3), sentiment(k=4)
# onto one 2-D circle; remaining features use free heads. BCE on all 8 outputs.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.puzzle.models_t3 import HeadFourierComb

N_EPOCHS, LR, BATCH = 600, 1e-3, 512


def main():
    data = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(data["train_emb"]).float()
    emb_te = torch.from_numpy(data["test_emb"]).float()
    lab_tr = torch.from_numpy(data["train_labels"]).float()
    lab_te = torch.from_numpy(data["test_labels"]).float()
    os.makedirs("artifacts/results", exist_ok=True)
    torch.manual_seed(0)
    model = HeadFourierComb()
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
            print(f"epoch {epoch+1:3d}: acc={acc:.4f}")
            model.train()
    model.eval()
    torch.save(model.state_dict(), "artifacts/results/31_fourier_comb_model.pt")
    print("saved 31_fourier_comb_model.pt")


if __name__ == "__main__":
    main()
