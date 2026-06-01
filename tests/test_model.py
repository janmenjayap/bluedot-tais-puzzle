import torch
from src.puzzle.model import Head, load_head, TAPS

def test_head_forward_and_taps():
    m = load_head("model.pt")
    x = torch.randn(5, 384)
    logits = m(x)
    assert logits.shape == (5, 8)
    # tap output dims: hidden taps -> 64, logits tap -> 8
    assert m.layers[:TAPS["h2"]](x).shape == (5, 64)   # layer L
    assert m.layers[:TAPS["logits"]](x).shape == (5, 8)
