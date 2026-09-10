# Experiment status

## Confirmatory Task 3 result

- `18_train_mnist_circular.py` trains five predeclared seeds using only the
  canonical MNIST training partition and a fixed train/validation split.
- `19_analyze_mnist_circular.py` evaluates the frozen checkpoints on QMNIST
  `test50k`. It refuses to overwrite the completed audit CSV.
- The frozen protocol, checkpoint hashes, validation results, per-seed audit
  results, and aggregate uncertainty are in `artifacts/results/18_mnist_*` and
  `artifacts/results/19_mnist_*`.

## Historical exploration

Task 3 scripts `08`-`17` and `20`-`34`, except for the clean `18`/`19` pair,
belong to the architecture-search phase. Most were run with repeated access to
the supplied test activations, often for one seed. Their artifacts can document
what was tried, but they are not held-out evidence and are excluded from the
final headline result.

The existing linked-rings checkpoint is also historical. Although
`33_train_linked_rings.py` now uses a validation split for any future exploratory
rerun, the checked-in checkpoint predates that correction. Script `34` therefore
labels its analysis post-hoc and reports collapse diagnostics rather than a
linking number.

Scripts `00`-`07` analyze the supplied puzzle model for Tasks 1 and 2. Their use
of the supplied train/test boundary is the intended evaluation of that fixed
model, not model or architecture selection for Task 3.
