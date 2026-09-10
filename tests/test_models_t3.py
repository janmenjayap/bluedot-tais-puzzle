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
    angular_freq_r2, arc_occupancy, ordered_disc_crossing_heuristic,
    linked_rings_loss,
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


def test_ordered_disc_crossing_heuristic_on_analytic_ring():
    # This diagnostic is valid only because the synthetic samples come from a known loop.
    psi = np.linspace(-np.pi, np.pi, 400, endpoint=False)
    ringB = np.column_stack([1 + np.cos(psi), np.zeros_like(psi), np.sin(psi)])
    assert ordered_disc_crossing_heuristic(ringB) == 1


def test_linked_rings_loss_zero_at_targets():
    country = torch.tensor([0, 1])
    food = torch.tensor([0, 0])               # φ = 0
    targetA = [1.0, 0.0, 0.0]                  # ring A, φ=0
    targetB = [2.0, 0.0, 0.0]                  # ring B, φ=0
    bottle = torch.tensor([targetA, targetB])
    loss = linked_rings_loss(bottle, country, food)
    assert loss.item() < 1e-6


def test_head_squarewave_shapes():
    from src.puzzle.models_t3 import HeadSquareWave
    model = HeadSquareWave(k=2)
    out = model(_emb())
    assert out.shape == (N, 8)
    circ = model.circle(_emb())
    assert circ.shape == (N, 2)
    assert torch.allclose(circ.norm(dim=1), torch.ones(N), atol=1e-5)


def test_head_squarewave_country_uses_harmonic():
    from src.puzzle.models_t3 import HeadSquareWave
    model = HeadSquareWave(k=3, country_idx=5)
    assert model.k == 3
    # harmonic readout consumes a 2-vector [cos(kθ), sin(kθ)]
    assert model.harm.in_features == 2 and model.harm.out_features == 1


def test_head_fourier_comb_shapes():
    from src.puzzle.models_t3 import HeadFourierComb
    model = HeadFourierComb()
    out = model(_emb())
    assert out.shape == (N, 8)
    assert model.harmonics == {5: 2, 3: 3, 4: 4}
    circ = model.circle(_emb())
    assert torch.allclose(circ.norm(dim=1), torch.ones(N), atol=1e-5)


def test_head_linked_rings_shapes():
    from src.puzzle.models_t3 import HeadLinkedRings
    model = HeadLinkedRings()
    out = model(_emb())
    assert out.shape == (N, 8)
    b = model.bottle(_emb())
    assert b.shape == (N, 3)


def test_mnist_protocol_split_is_fixed_disjoint_and_stratified(monkeypatch):
    from src.puzzle import mnist_protocol

    labels = np.tile(np.arange(10), 6_000)
    monkeypatch.setattr(mnist_protocol, "VALIDATION_SIZE", 10_000)
    train_a, validation_a = mnist_protocol.stratified_train_validation_indices(labels)
    train_b, validation_b = mnist_protocol.stratified_train_validation_indices(labels)

    assert len(train_a) == 50_000
    assert len(validation_a) == 10_000
    assert not np.intersect1d(train_a, validation_a).size
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(validation_a, validation_b)
    assert np.array_equal(np.bincount(labels[validation_a]), np.full(10, 1_000))


def test_qmnist_protocol_extracts_class_column():
    from src.puzzle.mnist_protocol import qmnist_class_targets

    metadata = torch.tensor([[3, 10, 20], [7, 11, 21]])
    assert torch.equal(qmnist_class_targets(metadata), torch.tensor([3, 7]))
    labels = torch.tensor([2, 5])
    assert torch.equal(qmnist_class_targets(labels), labels)
