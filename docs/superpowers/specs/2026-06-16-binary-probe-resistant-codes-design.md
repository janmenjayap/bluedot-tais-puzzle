# Task 3 Capstone — Three Binary Probe-Resistant Codes

**Date:** 2026-06-16
**Status:** Design (awaiting user review → writing-plans)

## Motivation

The Task 3 writeup closes with an explicit open problem:

> *"No experiment here produced a country encoding that is simultaneously weirder and
> at least as probe-resistant... Pushing a binary-country encoding past 0.471 while
> keeping it linearly weird remains open."*

The nine existing experiments converged on three hard-won lessons:

1. **Binary labels are always linearly separable** — two clusters can be cut by a
   hyperplane (killed SO(2) at 0.989, helix at 0.989).
2. **Depth linearises internally** — XOR (0.951) and helix arrived at h2 already linear;
   harder labels do not help.
3. **Genuine probe resistance needs interleaving no single hyperplane can cut** — the only
   win was MNIST even/odd (linear 0.51, nonlinear 0.985, 47-pt gap), via k-fold alternation
   around a circle + a unit-norm constraint. The d=2 bottleneck *emergently* pushed
   country/food/sentiment to k=2 angular codes (country R²=0.486 at k=2).

**Thesis of this capstone:** a binary, text-domain feature *can* be made probe-resistant.
We demonstrate three geometries that no single hyperplane can cut, each defeating the
hyperplane by a different mechanism: **high spatial frequency** (A), **per-feature harmonics
in superposition** (α), and **topological linking** (γ).

## Shared setup

- Embeddings cached in `artifacts/activations/acts.npz` (`train_emb`/`test_emb` 384-d,
  `train_labels`/`test_labels` 8 binary). No re-encoding.
- Each experiment adds: one model class in `src/puzzle/models_t3.py` (+ TDD tests in
  `tests/test_models_t3.py`), a train script, an analyze script, and a writeup/report section.
- Feature indices via `from src.puzzle.data import FEATURE_NAMES`
  (`country`=5, `food`=3, `sentiment`=4, `question`=1).
- The angular-frequency probe currently inlined in `scripts/28_analyze_bottleneck.py`
  (`angular_freq_r2`) is lifted into `src/puzzle/` (e.g. `src/puzzle/geometry.py`) so all
  three analyze scripts reuse it. `scripts/28` is updated to import from there (no behaviour
  change).
- Probes follow the established pattern: `LogisticRegression(max_iter=2000)` (linear),
  `MLPClassifier((32,), max_iter=2000, random_state=0)` (nonlinear), `StandardScaler` fit on
  train only. Unit-norm bottlenecks via `F.normalize`.

---

## Experiment A — Square wave (Idea 10, primary)

**Goal:** the binary, text-domain analog of MNIST even/odd, with a clean dose-response over
frequency `k` — directly closing the open problem.

**Architecture.** `emb(384) → enc[64→ReLU→64→ReLU] → Linear(64,2) → F.normalize → θ=atan2(y,x)`.
`country` readout is *forced* to the k-th harmonic: `logit_country = α·cos(kθ) + β·sin(kθ) + b`
with `α,β,b` learned and `k` fixed per run. The other 7 features get free MLP heads
(`2→16→ReLU→1`) so the shared circle stays task-useful. Train all 8 jointly with BCE.

**Sweep:** k = 1, 2, 3, 4, 5 (one model per k).

**Measurements.**
- Linear vs nonlinear probe for `country` on the 2-D circle, per k.
- `angular_freq_r2` for `country` — peak should land on the imposed k.
- Overall 8-feature accuracy (sanity: representation stays useful).

**Predicted result.** k=1 → linear high (half-circle split is separable); k≥2 → linear → chance,
nonlinear high; freq-peak = k. The linear-probe-vs-k curve is the headline artifact.

**Risk:** lowest — MNIST already proved the mechanism. Main failure mode is the net refusing
high k (caps accuracy); reportable as a frequency ceiling.

---

## Experiment α — Fourier comb (Idea 11, superposition showpiece)

**Goal:** controlled superposition where the linear-probe failure *is* the geometry — multiple
features multiplexed onto one circle, each on its own frequency.

**Architecture.** Same 2-D unit circle. Multiplex a subset of features, each forced to a
distinct harmonic: `country→k=2, food→k=3, sentiment→k=4` (distinct/coprime to avoid alias).
Each multiplexed readout: `logit_f = α_f·cos(k_f θ) + β_f·sin(k_f θ) + b_f`. Remaining features:
free MLP heads. Train jointly with BCE.

**Measurements.**
- Linear/nonlinear probe per multiplexed feature (expect chance / high for k≥2).
- **Harmonic-confusion heatmap:** `R²(feature i | harmonic k)` over all features × k=1..5 —
  expected diagonal, proving each feature is readable only at its own frequency.
- Overall accuracy.

**Predicted result.** Diagonal heatmap; each multiplexed feature near-chance to linear probes
and to the "wrong" harmonic, high to its own. Demonstrates more features than linear directions.

**Risk:** medium — harmonic collision / leakage. Mitigate with distinct k and the d=2 hard cap.

---

## Experiment γ — Linked rings (Idea 12, topological showpiece)

**Goal:** the sharpest rebuttal to Lesson 1 — a binary feature that is provably *not* linearly
separable because its two classes form linked rings, not clusters.

**Architecture.** `emb(384) → enc[64→ReLU→64→ReLU] → Linear(64,3) → 3-D bottleneck`.
Regulariser pushes `country=0` onto ring A (e.g. unit circle in the xy-plane, centre origin)
and `country=1` onto ring B (unit circle in the xz-plane, centre (1,0,0)) so B threads A
(linking number 1). Per-feature MLP heads (`3→16→ReLU→1`) keep all 8 decodable. Loss = BCE +
λ·(distance of each sample to its target ring). The within-ring angle is free.

**Measurements.**
- Linear vs nonlinear probe for `country` on the 3-D bottleneck (expect chance / high).
- **Linking number** (Gauss integral, or numerically verify ring B passes through ring A's
  disc) — confirms the rings actually link and did not drift apart.
- 3-D scatter plot coloured by country.

**Predicted result.** Linear probe → chance, nonlinear high, linking number ≈ 1. Spectacular plot.

**Risk:** medium — rings may unlink/collapse. Mitigate: measure linking number and fail loudly
if it drifts; tune λ. An unlinked outcome is itself reportable.

---

## File map

| Action | Path | Purpose |
|---|---|---|
| Create | `src/puzzle/geometry.py` | `angular_freq_r2` (+ linking-number helper) |
| Modify | `scripts/28_analyze_bottleneck.py` | import `angular_freq_r2` from `geometry` |
| Modify | `src/puzzle/models_t3.py` | `HeadSquareWave`, `HeadFourierComb`, `HeadLinkedRings` |
| Modify | `tests/test_models_t3.py` | shape + unit-norm + ring-distance tests |
| Create | `scripts/29_train_squarewave.py` / `30_analyze_squarewave.py` | A |
| Create | `scripts/31_train_fourier_comb.py` / `32_analyze_fourier_comb.py` | α |
| Create | `scripts/33_train_linked_rings.py` / `34_analyze_linked_rings.py` | γ |
| Modify | `docs/writeup.md` | Ideas 10–12 + summary rows |
| Modify | `docs/report.html` | tabs 10–12 + summary rows |
| Write | `artifacts/results/29–34_*.{pt,csv,png}` | checkpoints, probes, plots |

## Out of scope (held)

- **β (k-parity):** a diagnostic of *where* parity linearises, not a representation. Park.
- **δ (fractal threshold):** continuum limit of A; collapses into the k-sweep. Park.

## Success criteria

- A: at least one k≥2 with `country` linear probe ≤ 0.55 and nonlinear ≥ 0.90; freq-peak = k.
- α: harmonic-confusion heatmap is diagonal; multiplexed features ≤ 0.55 linear, ≥ 0.85 nonlinear.
- γ: `country` linear ≤ 0.55, nonlinear ≥ 0.90, measured linking number ≈ 1.
- All new tests pass; full suite stays green; writeup + report updated with real CSV numbers.
