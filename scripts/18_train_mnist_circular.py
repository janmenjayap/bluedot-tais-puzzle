# scripts/18_train_mnist_circular.py
# Train a CNN on MNIST with a 2D circular bottleneck.
# Digit k is pushed toward angle k*36° on the unit circle.
# Key property: even digits (0,2,4,6,8) and odd digits (1,3,5,7,9) alternate
# perfectly around the circle, so no linear probe can decode even/odd from
# the 2D representation — defeating linear probes for a binary feature.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from src.puzzle.models_t3 import HeadMNISTCircular
from src.puzzle.geometry import circular_loss_multiclass

LAMBDA, LR, EPOCHS, BATCH = 1.0, 1e-3, 30, 256


def load_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0          # (70000, 784)
    y = mnist.target.astype(np.int64)                   # (70000,)
    X = X.reshape(-1, 1, 28, 28)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=10000, random_state=42, stratify=y)
    return (torch.from_numpy(X_tr), torch.from_numpy(y_tr),
            torch.from_numpy(X_te),  torch.from_numpy(y_te))


def train():
    print("Loading MNIST...")
    X_tr, y_tr, X_te, y_te = load_mnist()
    print(f"train={len(X_tr)}, test={len(X_te)}")

    ds = TensorDataset(X_tr, y_tr)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = HeadMNISTCircular()
    opt = Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for xb, yb in loader:
            logits, bottle = model(xb)
            ce   = F.cross_entropy(logits, yb)
            circ = circular_loss_multiclass(bottle, yb, n_classes=10)
            loss = ce + LAMBDA * circ
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                logits_te, _ = model(X_te)
                acc = (logits_te.argmax(1) == y_te).float().mean().item()
            print(f"epoch {epoch+1:3d}: loss={total_loss/len(loader):.4f}  test acc={acc:.4f}")
            model.train()

    model.eval()
    os.makedirs("artifacts/results", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/results/18_mnist_circular_model.pt")
    print("saved artifacts/results/18_mnist_circular_model.pt")


if __name__ == "__main__":
    train()
