"""Train the frozen five-seed MNIST circular-bottleneck protocol.

This entry point loads only MNIST's canonical training partition. The QMNIST
audit set is deliberately available only to 19_analyze_mnist_circular.py.
"""
import csv
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from src.puzzle.geometry import circular_loss_multiclass
from src.puzzle.mnist_protocol import (
    BATCH_SIZE,
    EPOCHS,
    GEOMETRY_WEIGHT,
    LEARNING_RATE,
    TRAINING_SEEDS,
    load_train_validation,
    protocol_metadata,
    seed_training,
)
from src.puzzle.models_t3 import HeadMNISTCircular


RESULTS_DIR = Path("artifacts/results")
PROTOCOL_PATH = RESULTS_DIR / "18_mnist_circular_protocol.json"
VALIDATION_PATH = RESULTS_DIR / "18_mnist_circular_validation.csv"


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def digit_accuracy(model, loader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits, _ = model(images)
            correct += int((logits.argmax(1) == labels).sum().item())
            total += len(labels)
    return correct / total


def train_one_seed(seed, train_data, validation_loader, device):
    seed_training(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    model = HeadMNISTCircular().to(device)
    optimiser = Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits, bottle = model(images)
            classification_loss = F.cross_entropy(logits, labels)
            geometry_loss = circular_loss_multiclass(bottle, labels, n_classes=10)
            loss = classification_loss + GEOMETRY_WEIGHT * geometry_loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(
                f"seed={seed:3d} epoch={epoch + 1:2d}/{EPOCHS} "
                f"train_loss={total_loss / len(train_loader):.4f}"
            )

    validation_accuracy = digit_accuracy(model, validation_loader, device)
    checkpoint_path = RESULTS_DIR / f"18_mnist_circular_seed_{seed}.pt"
    cpu_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(cpu_state, checkpoint_path)
    print(f"seed={seed:3d} validation_accuracy={validation_accuracy:.4f}")
    return {
        "seed": seed,
        "validation_digit_accuracy": validation_accuracy,
        "checkpoint": str(checkpoint_path),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    metadata = protocol_metadata()
    metadata.update(
        {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torchvision_version": version("torchvision"),
            "training_device": str(device),
        }
    )
    PROTOCOL_PATH.write_text(json.dumps(metadata, indent=2) + "\n")

    print("Loading canonical MNIST training data (the audit set is not loaded)...")
    train_images, train_labels, validation_images, validation_labels = load_train_validation()
    train_data = TensorDataset(train_images, train_labels)
    validation_loader = DataLoader(
        TensorDataset(validation_images, validation_labels),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    print(
        f"train={len(train_data)}, validation={len(validation_labels)}, "
        f"seeds={TRAINING_SEEDS}, device={device}"
    )

    rows = [
        train_one_seed(seed, train_data, validation_loader, device)
        for seed in TRAINING_SEEDS
    ]
    with VALIDATION_PATH.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {PROTOCOL_PATH} and {VALIDATION_PATH}")


if __name__ == "__main__":
    main()
