# scripts/33_train_linked_rings.py
# Experiment γ — linked rings: country=0 → ring A (xy-plane, origin),
# country=1 → ring B (xz-plane, centre (1,0,0)); food sets within-ring angle.
# Loss = BCE(all 8) + λ · linked_rings_loss. λ tuned via LAMBDA.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.puzzle.models_t3 import HeadLinkedRings
from src.puzzle.geometry import linked_rings_loss
from src.puzzle.data import FEATURE_NAMES

N_EPOCHS, LR, BATCH, LAMBDA = 600, 1e-3, 512, 1.0
CI, FI = FEATURE_NAMES.index("country"), FEATURE_NAMES.index("food")


def main():
    data = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(data["train_emb"]).float()
    emb_te = torch.from_numpy(data["test_emb"]).float()
    lab_tr = torch.from_numpy(data["train_labels"]).float()
    lab_te = torch.from_numpy(data["test_labels"]).float()
    os.makedirs("artifacts/results", exist_ok=True)
    torch.manual_seed(0)
    model = HeadLinkedRings()
    opt = Adam(model.parameters(), lr=LR)
    sched = CosineAnnealingLR(opt, N_EPOCHS)
    model.train()
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(len(emb_tr))
        for s in range(0, len(emb_tr), BATCH):
            idx = perm[s:s + BATCH]
            logits = model(emb_tr[idx])
            bce = F.binary_cross_entropy_with_logits(logits, lab_tr[idx])
            ring = linked_rings_loss(model.bottle(emb_tr[idx]),
                                     lab_tr[idx][:, CI], lab_tr[idx][:, FI])
            loss = bce + LAMBDA * ring
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch + 1) % 150 == 0:
            model.eval()
            with torch.no_grad():
                acc = ((model(emb_te) > 0).float() == lab_te).float().mean()
            print(f"epoch {epoch+1:3d}: acc={acc:.4f}")
            model.train()
    model.eval()
    torch.save(model.state_dict(), "artifacts/results/33_linked_rings_model.pt")
    print("saved 33_linked_rings_model.pt")


if __name__ == "__main__":
    main()
