from src.puzzle.data import load_split, FEATURE_NAMES

def test_load_split_shapes():
    texts, labels, template_ids = load_split("data/test.jsonl")
    assert len(texts) == 1500
    assert labels.shape == (1500, 8)
    assert set(labels.flatten().tolist()) <= {0, 1}
    assert len(template_ids) == 1500
    assert FEATURE_NAMES[4] == "sentiment"
