# tests/test_models_t3.py
import torch
import numpy as np
import pytest
from src.puzzle.models_t3 import HeadXOR, HeadRotation, HeadHelix, HeadSuper, Predictor, HeadBottleneck
from src.puzzle.geometry import circular_loss, helical_loss, superposition_loss

N, D = 8, 384


def _emb():
    return torch.randn(N, D)


def test_head_xor_shape():
    out = HeadXOR()(_emb())
    assert out.shape == (N, 9)


def test_head_rotation_shapes():
    logits, proj = HeadRotation()(_emb())
    assert logits.shape == (N, 8)
    assert proj.shape == (N, 2)


def test_head_helix_shapes():
    logits, proj = HeadHelix()(_emb())
    assert logits.shape == (N, 8)
    assert proj.shape == (N, 3)


def test_head_super_shapes():
    logits, bottle = HeadSuper()(_emb())
    assert logits.shape == (N, 8)
    assert bottle.shape == (N, 2)


def test_predictor_shape():
    other = torch.randint(0, 2, (N, 7)).float()
    out = Predictor()(other)
    assert out.shape == (N, 64)


def test_circular_loss_scalar():
    proj = torch.randn(N, 2)
    labels = torch.randint(0, 2, (N,))
    loss = circular_loss(proj, labels)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_helical_loss_scalar():
    proj = torch.randn(N, 3)
    labels = torch.randint(0, 2, (N,))
    loss = helical_loss(proj, labels)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_superposition_loss_scalar():
    proj = torch.randn(N, 2)
    c = torch.randint(0, 2, (N,))
    f = torch.randint(0, 2, (N,))
    loss = superposition_loss(proj, c, f)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_head_bottleneck_d4_shape():
    out = HeadBottleneck(d=4)(_emb())
    assert out.shape == (N, 8)


def test_head_bottleneck_d2_shape():
    out = HeadBottleneck(d=2)(_emb())
    assert out.shape == (N, 8)


def test_head_bottleneck_unit_norm():
    model = HeadBottleneck(d=4)
    bottle = model.bottle(_emb())
    assert bottle.shape == (N, 4)
    norms = bottle.norm(dim=1)
    assert torch.allclose(norms, torch.ones(N), atol=1e-5)


# --- Capstone geometry helpers (Task 1) ---
from src.puzzle.geometry import (
    angular_freq_r2, arc_occupancy, disc_crossing_count, linked_rings_loss,
)


def test_angular_freq_r2_detects_k2():
    rng = np.random.default_rng(0)
    theta = rng.uniform(-np.pi, np.pi, 2000)
    y = (np.cos(2 * theta) > 0).astype(int)   # pure k=2 structure
    r2 = angular_freq_r2(theta, y, max_k=5)
    assert len(r2) == 5
    assert int(np.argmax(r2)) + 1 == 2        # peak at k=2


def test_arc_occupancy_counts_bins():
    theta = np.array([0.0, np.pi])            # two opposite arcs
    mask = np.array([True, True])
    assert arc_occupancy(theta, mask, n_bins=12) == 2
    assert arc_occupancy(theta, np.array([False, False])) == 0


def test_disc_crossing_linked_rings_is_one():
    # Ring B sampled densely: (1+cos ψ, 0, sin ψ) threads the z=0 unit disc once.
    psi = np.linspace(-np.pi, np.pi, 400, endpoint=False)
    ringB = np.column_stack([1 + np.cos(psi), np.zeros_like(psi), np.sin(psi)])
    assert disc_crossing_count(ringB) == 1


def test_linked_rings_loss_zero_at_targets():
    country = torch.tensor([0, 1])
    food = torch.tensor([0, 0])               # φ = 0
    targetA = [1.0, 0.0, 0.0]                  # ring A, φ=0
    targetB = [2.0, 0.0, 0.0]                  # ring B, φ=0
    bottle = torch.tensor([targetA, targetB])
    loss = linked_rings_loss(bottle, country, food)
    assert loss.item() < 1e-6
