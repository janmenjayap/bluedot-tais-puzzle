from src.puzzle.data import load_split, FEATURE_NAMES
import numpy as np
import torch
from src.puzzle.model import load_head

def test_load_split_shapes():
    texts, labels, template_ids = load_split("data/test.jsonl")
    assert len(texts) == 1500
    assert labels.shape == (1500, 8)
    assert set(labels.flatten().tolist()) <= {0, 1}
    assert len(template_ids) == 1500
    assert FEATURE_NAMES[4] == "sentiment"

def test_head_taps_shapes():
    from src.puzzle.activations import head_taps
    m = load_head("model.pt")
    emb = np.random.randn(7, 384).astype(np.float32)
    taps = head_taps(m, emb)
    assert taps["h2"].shape == (7, 64)      # layer L
    assert taps["logits"].shape == (7, 8)
    assert set(taps) == {"h0", "h1", "h2", "h3", "logits"}
