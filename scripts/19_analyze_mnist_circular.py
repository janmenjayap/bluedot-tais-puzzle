"""Run the single confirmatory audit of the frozen MNIST circular models."""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader, TensorDataset

from src.puzzle.mnist_protocol import (
    BATCH_SIZE,
    TRAINING_SEEDS,
    load_qmnist_audit,
    load_train_validation,
    protocol_metadata,
)
from src.puzzle.models_t3 import HeadMNISTCircular


EVEN_DIGITS = np.array([0, 2, 4, 6, 8])
RESULTS_DIR = Path("artifacts/results")
PROTOCOL_PATH = RESULTS_DIR / "18_mnist_circular_protocol.json"
VALIDATION_PATH = RESULTS_DIR / "18_mnist_circular_validation.csv"
AUDIT_RESULTS_PATH = RESULTS_DIR / "19_mnist_circular_audit_seeds.csv"
AUDIT_SUMMARY_PATH = RESULTS_DIR / "19_mnist_circular_audit_summary.csv"
GEOMETRY_STATS_PATH = RESULTS_DIR / "19_mnist_circular_geometry_stats.csv"
FIGURE_PATH = RESULTS_DIR / "19_mnist_circular_geometry.png"
T_CRITICAL_95_DF4 = 2.7764451051977987


def batched_outputs(model, images):
    loader = DataLoader(TensorDataset(images), batch_size=BATCH_SIZE, shuffle=False)
    bottles = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            logits, bottle = model(batch)
            bottles.append(bottle.numpy())
            predictions.append(logits.argmax(1).numpy())
    return np.concatenate(bottles), np.concatenate(predictions)


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_seed(seed, train_images, train_labels, audit_images, audit_labels):
    checkpoint_path = RESULTS_DIR / f"18_mnist_circular_seed_{seed}.pt"
    model = HeadMNISTCircular()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))

    bottle_train, _ = batched_outputs(model, train_images)
    bottle_audit, digit_predictions = batched_outputs(model, audit_images)
    train_digits = train_labels.numpy()
    audit_digits = audit_labels.numpy()
    parity_train = np.isin(train_digits, EVEN_DIGITS).astype(np.int64)
    parity_audit = np.isin(audit_digits, EVEN_DIGITS).astype(np.int64)

    parity_linear = LogisticRegression(max_iter=2_000, random_state=seed)
    parity_nonlinear = MLPClassifier(
        hidden_layer_sizes=(64,),
        max_iter=2_000,
        random_state=seed,
    )
    digit_linear = LogisticRegression(max_iter=2_000, random_state=seed)
    row = {
        "seed": seed,
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "audit_digit_accuracy": float(np.mean(digit_predictions == audit_digits)),
        "audit_digit_linear_probe": digit_linear.fit(bottle_train, train_digits).score(
            bottle_audit, audit_digits
        ),
        "audit_even_odd_linear": parity_linear.fit(bottle_train, parity_train).score(
            bottle_audit, parity_audit
        ),
        "audit_even_odd_nonlinear": parity_nonlinear.fit(
            bottle_train, parity_train
        ).score(bottle_audit, parity_audit),
        "audit_even_odd_base_rate": float(
            max(parity_audit.mean(), 1.0 - parity_audit.mean())
        ),
    }
    return row, bottle_audit


def aggregate_results(rows):
    metric_names = [name for name in rows[0] if name.startswith("audit_")]
    summary_rows = []
    for metric in metric_names:
        values = np.array([row[metric] for row in rows], dtype=float)
        if len(values) != len(TRAINING_SEEDS):
            raise RuntimeError("The confirmatory interval requires all five frozen seeds")
        standard_deviation = values.std(ddof=1)
        summary_rows.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "standard_deviation": standard_deviation,
                "ci95_half_width": (
                    T_CRITICAL_95_DF4 * standard_deviation / np.sqrt(len(values))
                ),
                "minimum": values.min(),
                "maximum": values.max(),
                "n_seeds": len(values),
            }
        )
    return summary_rows


def save_geometry(bottle, labels, representative_seed, summary_rows):
    labels = labels.numpy()
    angles = np.arctan2(bottle[:, 1], bottle[:, 0])
    radii = np.linalg.norm(bottle, axis=1)
    geometry_rows = []
    for digit in range(10):
        mask = labels == digit
        target_angle = digit * 2 * np.pi / 10
        errors = np.angle(np.exp(1j * (angles[mask] - target_angle)))
        mean_error = np.arctan2(np.sin(errors).mean(), np.cos(errors).mean())
        resultant = np.hypot(np.sin(errors).mean(), np.cos(errors).mean())
        circular_std = np.sqrt(max(0.0, -2.0 * np.log(max(resultant, 1e-12))))
        geometry_rows.append(
            {
                "digit": digit,
                "target_angle_deg": np.degrees(target_angle),
                "mean_angle_error_deg": np.degrees(mean_error),
                "circular_std_deg": np.degrees(circular_std),
                "mean_radius": radii[mask].mean(),
            }
        )
    pd.DataFrame(geometry_rows).to_csv(GEOMETRY_STATS_PATH, index=False)

    summary = {row["metric"]: row for row in summary_rows}
    rng = np.random.default_rng(4_219)
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.colormaps["tab10"]
    for digit in range(10):
        digit_indices = np.flatnonzero(labels == digit)
        sample = rng.choice(digit_indices, size=min(500, len(digit_indices)), replace=False)
        marker = "o" if digit in EVEN_DIGITS else "^"
        ax.scatter(
            bottle[sample, 0],
            bottle[sample, 1],
            color=cmap(digit),
            marker=marker,
            alpha=0.25,
            s=13,
            label=str(digit),
            rasterized=True,
        )
    ax.set_aspect("equal")
    ax.set_xlabel("bottleneck dimension 1")
    ax.set_ylabel("bottleneck dimension 2")
    ax.set_title(
        "QMNIST audit: interleaved circular code\n"
        f"linear parity {summary['audit_even_odd_linear']['mean']:.3f}; "
        f"nonlinear parity {summary['audit_even_odd_nonlinear']['mean']:.3f} "
        f"(five-seed means)"
    )
    ax.legend(title="digit (circle=even, triangle=odd)", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180)
    plt.close(fig)
    print(f"representative geometry seed={representative_seed} selected by validation only")


def main():
    if AUDIT_RESULTS_PATH.exists():
        raise RuntimeError(
            f"Refusing to repeat the confirmatory audit: {AUDIT_RESULTS_PATH} already exists"
        )

    validation = pd.read_csv(VALIDATION_PATH)
    observed_seeds = tuple(sorted(validation["seed"].astype(int)))
    if observed_seeds != tuple(sorted(TRAINING_SEEDS)):
        raise RuntimeError(
            f"Expected validation results for seeds {TRAINING_SEEDS}, got {observed_seeds}"
        )
    representative_seed = int(
        validation.sort_values(
            ["validation_digit_accuracy", "seed"], ascending=[False, True]
        ).iloc[0]["seed"]
    )

    print("Loading probe-training data and the untouched QMNIST test50k audit set...")
    train_images, train_labels, _, _ = load_train_validation()
    audit_images, audit_labels = load_qmnist_audit()
    rows = []
    representative_bottle = None
    for seed in TRAINING_SEEDS:
        row, bottle = evaluate_seed(
            seed, train_images, train_labels, audit_images, audit_labels
        )
        rows.append(row)
        if seed == representative_seed:
            representative_bottle = bottle
        print(
            f"seed={seed:3d} digit={row['audit_digit_accuracy']:.4f} "
            f"parity_linear={row['audit_even_odd_linear']:.4f} "
            f"parity_nonlinear={row['audit_even_odd_nonlinear']:.4f}"
        )

    summary_rows = aggregate_results(rows)
    pd.DataFrame(rows).to_csv(AUDIT_RESULTS_PATH, index=False)
    pd.DataFrame(summary_rows).to_csv(AUDIT_SUMMARY_PATH, index=False)
    save_geometry(representative_bottle, audit_labels, representative_seed, summary_rows)

    audit_record = json.loads(PROTOCOL_PATH.read_text())
    audit_record.update(
        {
            "audit_status": "completed once",
            "interval_method": (
                "two-sided 95% Student-t interval across five seeds (df=4)"
            ),
            "pre_outcome_loader_event": (
                "First invocation stopped before predictions or metrics because raw QMNIST "
                "targets have eight metadata columns; class-column extraction was then "
                "unit-tested before this evaluation."
            ),
            "representative_seed_selection": "highest validation digit accuracy; seed breaks ties",
            "representative_seed": representative_seed,
            "per_seed_results": str(AUDIT_RESULTS_PATH),
            "aggregate_results": str(AUDIT_SUMMARY_PATH),
        }
    )
    (RESULTS_DIR / "19_mnist_circular_audit_record.json").write_text(
        json.dumps(audit_record, indent=2) + "\n"
    )
    print(f"saved {AUDIT_RESULTS_PATH}, {AUDIT_SUMMARY_PATH}, and {FIGURE_PATH}")


if __name__ == "__main__":
    main()
