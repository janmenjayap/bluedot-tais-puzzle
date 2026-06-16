# tests/test_models_t3.py
import torch
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
