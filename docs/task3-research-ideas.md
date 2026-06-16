# Task 3 Research Ideas — Weirder Representations

**Date:** 2026-06-12
**Context:** The original model (Puzzle #1) encodes `country` as an absolute-value / interval
code along the food axis — a 1D magnitude code with Z/2 symmetry. Task 3 asks us to train a
model that encodes some feature in a "more interesting" way. These ideas broaden the scope to
connect with reasoning, world-model theory, and JEPA.

---

## Background: What Makes a Representation "Weirder"?

The baseline (absolute-value code) has these properties:
- **Dimensionality:** 1D (single scalar projection)
- **Symmetry group:** Z/2 (flip sign)
- **Minimum nonlinearities to decode:** 2 ReLUs (|x| = relu(x) + relu(-x))
- **Linear probe accuracy:** ~50% (chance)
- **Discovery difficulty:** moderate — standard nonlinear probe finds it

"Weirder" means: higher intrinsic dimensionality, richer symmetry group, harder to probe,
and ideally connected to structures that appear in larger models doing genuine reasoning.

---

## Idea 1: Rotational / Transition-Operator Code

### Core claim
Encode `country` as a **rotation in a 2D subspace** rather than magnitude along a 1D axis.

- `country=0` → base vector **v** in a 2D plane
- `country=1` → R·**v** where R is a rotation by angle θ (e.g., 90°)

### Why it's fundamentally weirder than |proj|
The absolute-value code is a Z/2 symmetry (flip sign). A rotational code is SO(2) — a
*continuous* symmetry group. Linear probes fail completely because the feature lives on a
circle, not a half-space. Even 1D magnitude probes fail. To decode country you need
arctan2(y, x) in the 2D subspace — a circular decoder.

### Connection to LeJEPA / world models
LeJEPA's identifiability theorem (Klindt, LeCun, Balestriero — arXiv:2605.26379) states that
a consistent world-model transition operator must be **linear and orthogonal** — exactly a
rotation matrix. This model is a *minimal concrete instance* of that theory:

> The update from `country=0` to `country=1` **IS** a rotation. R is the transition operator.
> Identifiability holds because R is orthogonal.

This is not just a weird representation — it is the simplest possible ground-truth
demonstration of the LeJEPA transition-operator condition.

### Training approach
- Add a geometry regulariser: push h2 activations onto a 2D circle conditioned on country
- Or: polar loss — country logit = cos(θ) where θ = angle in a learned 2D projection of h2
- No architectural changes needed; the frozen encoder + MLP is sufficient

### Measurements & expected results

| Probe | Expected accuracy |
|---|---|
| Linear (1D) | ~50% (chance) |
| 2D angular decoder | >95% |
| 1D magnitude probe | ~50% |

- Min ReLUs to approximate: ~4 (piecewise linear circle)
- Geometry plot: two arcs of a circle in a 2D scatter plot — visually unambiguous
- **Story:** Z/2 → SO(2); magnitude code → phase code; 1D → 2D; directly matches
  LeJEPA's orthogonal-operator condition

---

## Idea 2: XOR / Parity Code

### Core claim
Introduce a synthetic feature `parity = sentiment XOR question`. Train the model to predict
it. Force the model to develop a 2D checkerboard representation — four clusters at the corners
of a square, alternating 0/1/0/1 on the diagonals.

### Why it's interesting as a researcher
XOR is the canonical proof that depth is necessary — no linear classifier can solve it in any
dimension. More importantly, the abstract state-tracking literature (Li, Guo, Andreas,
arXiv:2503.02854, ICML 2025) found that real transformers implement a **parity-pruning + scan**
mechanism when doing compositional state updates. This Task 3 experiment is a controlled
miniature version of that result:

> "In a 5-layer MLP, we force parity-style compositional encoding and verify the geometry
> matches what appears in larger state-tracking models."

### Training
- Compute `parity_label = labels[:, sentiment_idx] XOR labels[:, question_idx]` from existing
  train/test data — no new data collection needed
- Add as a 9th output head with standard BCE loss
- The model will develop a checkerboard geometry in some 2D subspace of h2

### Measurements & expected results

| Probe | Expected accuracy |
|---|---|
| Linear (any dimension) | ~50% (provably cannot exceed chance) |
| 2-layer MLP probe | >95% |
| 1-layer ReLU probe | ~75% (one diagonal only) |

- Minimum depth to decode: exactly 2 layers (XOR theorem)
- Geometry: four clusters, two diagonals — neither diagonal separates XOR classes
- **Key connection:** the "parity-pruning" mechanism in arXiv:2503.02854 is XOR composition
  at scale; this is XOR composition in a 64-unit MLP — same structure, fully observable

---

## Idea 3: JEPA-Predictive / Residual Code

### Core claim
Encode `country` as the **prediction residual** in latent space — what the network could not
predict about the h2 activations from the other 7 features. Directly inspired by I-JEPA /
V-JEPA: *"predict in latent space, not output space."*

### Architecture

```
frozen encoder → h0 → h1 → h2 ──────────────────────────────→ 8 logits (BCE)
                              └─[mask country dim]→ Predictor P → h2_pred
                                                                ↓
                                              loss += λ·||h2 − h2_pred||²
```

The main network is trained normally. The predictor P takes h2 with the country dimension
zeroed out and tries to predict the full h2. The network is incentivised to make country's
contribution to h2 *unpredictable from the other features* — a genuine residual.

### What this forces
Country's representation becomes a **context-dependent prediction residual** — it has no
fixed direction in raw activation space. Its location in h2 shifts depending on which other
features are active. To probe country, you must first know the context (other active features)
and then compute the residual.

### Connection to JEPA theory
This is a literal implementation of the JEPA objective at tiny scale. The key question then
becomes empirical: **is the residual direction consistent across contexts?** LeJEPA's
identifiability theorem says yes, *if and only if* P is a linear operator. So we can:

1. Check if a linear matrix W approximates P well: fit W s.t. P(z) ≈ Wz; report ||P(z) − Wz||²
2. If P is approximately linear → residual is identifiable → LeJEPA condition satisfied
3. If P is nonlinear → residual shifts across contexts → condition violated

This is the first empirical check of LeJEPA's condition in a fully controlled, ground-truth
setting.

### Measurements & expected results

| Probe | Expected accuracy |
|---|---|
| Standard linear probe on raw h2 | ~60% (context-dependent, no fixed direction) |
| Context-conditioned linear probe | >95% |
| Is P linear? (matrix fit residual) | Low if condition satisfied |

- **Theoretical payoff either way:** P linear → confirms LeJEPA; P nonlinear → shows when
  the condition breaks in practice

---

## Idea 4: Helical / Phase Code

### Core claim
Encode `country` as position on a **helix** in a 3D subspace — combining circular (phase) and
linear (magnitude) components.

### Motivation
Transformer positional encodings are helical: (sin(θ), cos(θ), sin(2θ), cos(2θ), ...).
Neuroscience evidence for grid cells suggests hippocampal place encoding is also phase-based.
World models that track temporal state (Othello after k moves) have been hypothesised to use
similar codes.

Training a tiny model to develop a helical representation gives a ground-truth case: you know
the helix is there, you put it there, now characterise exactly how many probe dimensions are
needed (answer: 3) and verify the helix parameters (radius, pitch, orientation) match what
you imposed.

### Measurements

| Probe dimensions | Expected accuracy |
|---|---|
| 1D | ~50% |
| 2D circular | ~75% (misses helical drift) |
| 3D helical fit | >95% |

---

## Idea 5: Superposition / Entanglement Code

### Core claim
Deliberately encode `country` and `food` in a **shared 2D subspace with interleaved
boundaries** — neither feature can be decoded without contamination from the other.

### Why safety-relevant
Anthropic's superposition hypothesis predicts that models under capacity pressure store more
features than dimensions by overlapping representations. This produces the entangled,
hard-to-probe geometry that makes mechanistic interpretability difficult in practice.

Task 3 could be the first *deliberate, controlled* superposition demonstration: build it in,
measure how probes fail, and show the failure is *predictable* from the geometry. The
failure mode is not mysterious — it is the geometry.

---

## Recommended Plan: Ideas 1 + 2 Together

Implement both the **rotational code (Idea 1)** and the **parity/XOR code (Idea 2)**.
They are both:
- Tractable within the existing MLP architecture
- Analytically clean (known geometry, provable probe bounds)
- Independently meaningful

The unified story:

> *"The original model uses a 1D magnitude code (|proj|, Z/2 symmetry). We demonstrate two
> richer codes: a parity/XOR code (2D checkerboard — compositional, directly analogous to
> the parity-pruning mechanism in state-tracking transformers, arXiv:2503.02854) and a
> rotational code (2D circle, SO(2) symmetry — the minimal concrete instance of LeJEPA's
> linear transition-operator identifiability theorem, arXiv:2605.26379). Each requires
> strictly more structure to decode, and each corresponds to a class of representations
> appearing in larger models doing genuine reasoning."*

### Suggested script structure for Task 3

```
scripts/
  08_train_parity_model.py     # train 9-head model with XOR label
  09_probe_parity.py           # linear / 2-layer probe; geometry plot
  10_train_rotation_model.py   # train with circular regulariser
  11_probe_rotation.py         # angular decoder; 2D scatter plot
```

---

## References

- Klindt, LeCun, Balestriero — *When Does LeJEPA Learn a World Model?* — arXiv:2605.26379
- Balestriero, LeCun — *LeJEPA* — arXiv:2511.08544
- Assran et al. — *I-JEPA* — arXiv:2301.08243
- Assran et al. (Meta) — *V-JEPA 2* — arXiv:2506.09985
- Li, Guo, Andreas — *How Do Language Models Track State?* — arXiv:2503.02854 (ICML 2025)
- Zhang et al. — *Finite State Automata Inside Transformers with CoT* — arXiv:2502.20129
