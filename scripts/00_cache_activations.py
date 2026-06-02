from src.puzzle.activations import build_cache

if __name__ == "__main__":
    path = build_cache("artifacts/activations/acts.npz")
    print(f"cached -> {path}")
