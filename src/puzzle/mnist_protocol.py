import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "mnist"

VALIDATION_SIZE = 10_000
SPLIT_SEED = 17_291
TRAINING_SEEDS = (11, 29, 47, 71, 101)

LEARNING_RATE = 1e-3
GEOMETRY_WEIGHT = 1.0
EPOCHS = 30
BATCH_SIZE = 256


def stratified_train_validation_indices(labels):
    """Return the single frozen 50k/10k split of canonical MNIST training data."""
    labels = np.asarray(labels)
    indices = np.arange(len(labels))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=SPLIT_SEED,
        stratify=labels,
    )
    return np.sort(train_indices), np.sort(validation_indices)


def _normalise_images(images):
    return images.unsqueeze(1).to(dtype=torch.float32).div_(255.0)


def qmnist_class_targets(targets):
    """Extract class labels from QMNIST's raw eight-column metadata tensor."""
    return targets[:, 0] if targets.ndim == 2 else targets


def load_train_validation():
    """Load MNIST's canonical training partition; never load either test set."""
    from torchvision.datasets import MNIST

    dataset = MNIST(DATA_ROOT, train=True, download=True)
    train_indices, validation_indices = stratified_train_validation_indices(
        dataset.targets.numpy()
    )
    images = _normalise_images(dataset.data)
    labels = dataset.targets.to(dtype=torch.long)
    return (
        images[train_indices],
        labels[train_indices],
        images[validation_indices],
        labels[validation_indices],
    )


def load_qmnist_audit():
    """Load QMNIST's 50k additional examples, excluding the reused MNIST test set."""
    from torchvision.datasets import QMNIST

    dataset = QMNIST(DATA_ROOT, what="test50k", compat=True, download=True)
    if len(dataset) != 50_000:
        raise RuntimeError(f"Expected 50,000 QMNIST audit examples, got {len(dataset)}")
    targets = qmnist_class_targets(dataset.targets).to(dtype=torch.long)
    return _normalise_images(dataset.data), targets


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def protocol_metadata():
    return {
        "training_dataset": "MNIST canonical training partition",
        "training_examples": 50_000,
        "validation_examples": VALIDATION_SIZE,
        "validation_split_seed": SPLIT_SEED,
        "audit_dataset": "QMNIST test50k (additional examples only)",
        "audit_examples": 50_000,
        "training_seeds": list(TRAINING_SEEDS),
        "learning_rate": LEARNING_RATE,
        "geometry_weight": GEOMETRY_WEIGHT,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "selection_rule": "fixed epoch count; no model or seed selection on audit data",
    }