import numpy as np
import torch
from src.puzzle.model import TAPS


def encode_texts(texts, batch_size=256, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(model_name)
    return enc.encode(
        texts,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=True,
    ).astype(np.float32)  # (N, 384)


def head_taps(m, emb):
    x = torch.from_numpy(np.asarray(emb, dtype=np.float32))
    out = {}
    with torch.no_grad():
        for name, end in TAPS.items():
            out[name] = m.layers[:end](x).numpy()
    return out


def build_cache(out_path, model_path="model.pt"):
    from src.puzzle.data import load_split
    from src.puzzle.model import load_head
    m = load_head(model_path)
    blobs = {}
    for split in ["train", "test"]:
        texts, labels, tids = load_split(f"data/{split}.jsonl")
        emb = encode_texts(texts)
        taps = head_taps(m, emb)
        blobs[f"{split}_emb"] = emb
        blobs[f"{split}_labels"] = labels
        blobs[f"{split}_tids"] = tids
        for name, arr in taps.items():
            blobs[f"{split}_{name}"] = arr
    np.savez_compressed(out_path, **blobs)
    return out_path
