# Task 3 Capstone — Three Binary Probe-Resistant Codes

**Date:** 2026-06-16
**Status:** Implemented (2026-06-30) — scripts 29–34, results 30/32/34, writeup+report Ideas 10–12.
Plan: `docs/superpowers/plans/2026-06-29-binary-probe-resistant-codes.md`.

> **Outcome.** All three codes built and run (full suite 27/27 green).
> **A (square wave):** non-monotonic dose-response — k=2 is the sweet spot (freq peak k=2,
> nonlinear 0.959, linear **0.711**, +0.25 gap) but the binary label leaks into k=1, so it does
> *not* beat the baseline (0.471); k=1 collapses to chance on both probes, k≥3 collapses to
> linearly-trivial. Imposing a harmonic *readout* is insufficient. **α (Fourier comb):** only
> `sentiment` multiplexed cleanly onto its target harmonic (k=4, +0.27 gap); `food` leaked to k=2,
> `country` collapsed — the unconstrained easy feature `question` hijacked k1/k2. Heatmap ~1/3
> diagonal. **γ (linked rings):** the success — country linear **0.533** (near chance), nonlinear
> **0.940** (+0.41 gap), net disc-crossing = **1** (genuine linking), ring B populated across 9/12
> arcs (collapse risk did not bite; the co-trained features were the real spreader). The
> constructible-vs-emergent thesis holds: probe-resistant binary geometry is constructible by
> *topology* (γ), a milder k=2 version *emerges* under capacity pressure (Idea 9), and the cheap
> in-between constructions (A/α) revert toward linear separability.

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

**Thesis of this capstone.** The original country code is interesting precisely because it
*emerged* — the model chose `|x|`-on-the-food-axis with no prompting. This capstone separates
two questions the prior write-up blurred:

1. **Constructive existence:** *can* a binary, text-domain feature be made probe-resistant at
   all? We show yes, via three geometries no single hyperplane can cut — **high spatial
   frequency** (A), **per-feature harmonics in superposition** (α), **topological linking** (γ).
   These are **architecturally induced, not emergent**: we impose the geometry by construction.
   That is deliberate and honestly labelled — they establish *existence and mechanism*, not
   spontaneity.
2. **Emergent counterpart:** does the same structure arise *without* prescription, under
   capacity pressure? It already does — **Idea 9 (the d=2 unit-circle bottleneck) is the
   emergence control**: with no target angles, country/food/sentiment landed on k=2 angular
   codes (country R²=0.486). This capstone leans on that result rather than re-inventing it.

The honest finding is **the gap between what is constructible (A/α/γ) and what emerges
(Idea 9)** — which is a sharper answer to the prior "everything we did was imposed" criticism
than pretending the imposed codes are emergent.

**The mechanism that makes all of this work — the spreader.** A binary label has no internal
variation to spread points around a manifold, so a harmonic readout or a ring target does *not*
by itself force interleaving: the model can satisfy it by collapsing all `feature=1` samples
into a *single* arc/point, which is linearly separable again (Lesson 1 reasserts itself). MNIST
even/odd worked only because the 10-way **digit class was a strong spreader** and even/odd
alternated on top of it. Therefore every experiment here **requires a co-encoded spreader
variable** that populates the manifold while the binary target interleaves/links over it. In the
text domain the spreader is the other co-trained features; their limited angular spread (the d=2
bottleneck reached only ~k=2) is itself a constraint we must respect when choosing the sweep range.

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
  change). The lifted version is extended from `max_k=4` to `max_k=5` to cover the A/α sweeps;
  `scripts/28`'s existing k=1..4 outputs are unaffected.
- Probes follow the established pattern: `LogisticRegression(max_iter=2000)` (linear),
  `MLPClassifier((32,), max_iter=2000, random_state=0)` (nonlinear), `StandardScaler` fit on
  train only. Unit-norm bottlenecks via `F.normalize`.

---

## Experiment A — Square wave (Idea 10, primary by relevance)

**Priority rationale.** Primary because it hits the binary-country open problem head-on — *not*
because it is the most novel. It is the **least novel** of the three: `cos(2θ)` on a circle is
the `|x|` baseline (an even-symmetric Z/2 code) bent into 2-D. The increment over baseline is
modest at low k; the genuinely-weirder regime is high k, which is also the riskiest (see below).
This tension is stated openly in the write-up rather than hidden.

**Goal:** the binary, text-domain analog of MNIST even/odd, with a dose-response over frequency
`k` — establishing constructively that a binary feature *can* be pushed past the baseline.

**Architecture.** `emb(384) → enc[64→ReLU→64→ReLU] → Linear(64,2) → F.normalize → θ=atan2(y,x)`.
`country`'s readout is **constrained** (not "forced" — the model can still underfit it) to the
k-th harmonic: `logit_country = α·cos(kθ) + β·sin(kθ) + b`, with `α,β,b` learned and `k` fixed
per run. **The spreader is mandatory:** the other 7 features are co-trained on the *same* circle
with free MLP heads (`2→16→ReLU→1`), and they are what spread samples around θ so that `country`
can interleave across all k arcs instead of collapsing into one. Train all 8 jointly with BCE.

**Sweep:** k = 1, 2, 3 as the core claim (k=2 is the XOR-on-circle checkerboard — genuinely not
linearly separable). k = 4, 5 are *exploratory*: the d=2 bottleneck only reached ~k=2 emergently,
so 4–5 may be capacity-starved. Report where the ceiling actually falls.

**Measurements.**
- Linear vs nonlinear probe for `country` on the 2-D circle, per k.
- `angular_freq_r2` for `country` — peak should land on the imposed k (if interleaving realized).
- Overall 8-feature accuracy AND per-arc occupancy of `country=1` (diagnostic: did it spread
  across k arcs, or collapse into one? Collapse ⇒ the spreader failed and linear probe stays high).

**Predicted result.** k=1 → linear high (half-circle split is separable); k=2 → linear → chance,
nonlinear high; k=3+ → conditional on the spreader providing enough angular spread. The
linear-probe-vs-k curve is the headline artifact; a collapse at high k is a reportable ceiling.

**Risk:** moderate (downgraded from "lowest"). Two real failure modes: (a) `country=1` collapses
into a single arc → no interleaving → linear probe stays high (the spreader-dependency trap);
(b) high k starves capacity on a 2-D circle → country underfits or other features collapse. Both
are diagnosable via the occupancy metric and are themselves findings.

---

## Experiment α — Fourier comb (Idea 11, secondary — superposition showpiece)

**Goal:** controlled superposition where the linear-probe failure *is* the geometry — multiple
features multiplexed onto one circle, each on its own frequency.

**Spreader note.** Unlike A, α has its **spreader built in**: multiplexing several features over
one angle is *itself* the source of angular spread (8 combos of 3 features scatter θ around the
circle), so the collapse risk is lower here than in A. This is the most self-consistent of the three.

**Architecture.** Same 2-D unit circle. Multiplex three features, each constrained to a distinct
harmonic: `country→k=2, food→k=3, sentiment→k=4` (distinct to limit aliasing). Each multiplexed
readout: `logit_f = α_f·cos(k_f θ) + β_f·sin(k_f θ) + b_f`. Remaining 5 features: free MLP heads.
Train jointly with BCE. Note this asks one scalar θ to carry **3 independent bits as 3 harmonics**
— feasible in principle (harmonics are orthogonal on the circle) but tight; leakage is expected.

**Measurements.**
- Linear/nonlinear probe per multiplexed feature (target chance / high for k≥2).
- **Harmonic-confusion heatmap:** `R²(feature i | harmonic k)` over all features × k=1..5.
- Overall accuracy.

**Predicted result.** The heatmap *target* is diagonal (each feature readable only at its own
harmonic). Off-diagonal mass is **not a failure to hide — it is the measurement**: it quantifies
how much multiplexing leaks, given the non-uniform joint distribution of (country,food,sentiment)
around the circle. A mostly-diagonal heatmap demonstrates more features than linear directions.

**Risk:** medium — harmonic collision / leakage (expected and measured, not catastrophic) and
the tightness of 3 bits on one angle. Mitigate with distinct k and the d=2 hard cap.

---

## Experiment γ — Linked rings (Idea 12, stretch — topological showpiece)

**Goal:** the sharpest rebuttal to Lesson 1 — a binary feature that is provably *not* linearly
separable because its two classes form linked rings, not clusters. (Separability checked by hand:
for ring A in the xy-plane and ring B in the xz-plane centred at (1,0,0), any separating
hyperplane requires `b ≥ √(w₁²+w₂²) ≥ w₁`, which forces all of ring B to A's side — impossible.
So the configuration is genuinely not linearly separable.)

**Architecture.** `emb(384) → enc[64→ReLU→64→ReLU] → Linear(64,3) → 3-D bottleneck`.
Regulariser pushes `country=0` onto ring A (unit circle in the xy-plane, centre origin) and
`country=1` onto ring B (unit circle in the xz-plane, centre (1,0,0)) so B threads A (linking
number 1). Loss = BCE + λ·(distance of each sample to its target ring), λ tuned.

**Spreader is mandatory, not optional.** The within-ring angle (φ on A, ψ on B) **must carry a
second feature** (e.g. `food` → position around the ring). If it is left free, the model collapses
each class to a single point on its ring → two points, not two rings → linearly separable again,
and the whole experiment fails. Populating the rings is what makes the linking real. Per-feature
MLP heads (`3→16→ReLU→1`) keep all 8 decodable.

**Measurements.**
- Linear vs nonlinear probe for `country` on the 3-D bottleneck (expect chance / high).
- **Linking verification** by numerical disc-crossing count (not the Gauss integral): take ring
  A's fitted plane and centre, count signed crossings of the `country=1` point cloud through A's
  disc; linked ⇒ exactly one net crossing. Also report ring-occupancy (did `country=1` spread
  around ring B, or collapse?) — collapse invalidates the linking claim.
- 3-D scatter plot coloured by country.

**Predicted result.** Linear probe → chance, nonlinear high, one net disc-crossing, both rings
populated. Spectacular plot — *if* it holds.

**Risk:** highest of the three (hence stretch). Failure modes: rings unlink, a class collapses to
a point (spreader fails), or the disc-crossing measure is ambiguous on a noisy cloud. Mitigate:
mandatory within-ring spreader, occupancy + crossing diagnostics, λ tuning. A clean negative
(e.g. "the net prefers to unlink") is itself reportable.

---

## File map

| Action | Path | Purpose |
|---|---|---|
| Create | `src/puzzle/geometry.py` | `angular_freq_r2` (max_k=5), arc-occupancy, disc-crossing linking check |
| Modify | `scripts/28_analyze_bottleneck.py` | import `angular_freq_r2` from `geometry` |
| Modify | `src/puzzle/models_t3.py` | `HeadSquareWave`, `HeadFourierComb`, `HeadLinkedRings` |
| Modify | `tests/test_models_t3.py` | shape + unit-norm + ring-distance tests |
| Create | `scripts/29_train_squarewave.py` / `30_analyze_squarewave.py` | A |
| Create | `scripts/31_train_fourier_comb.py` / `32_analyze_fourier_comb.py` | α |
| Create | `scripts/33_train_linked_rings.py` / `34_analyze_linked_rings.py` | γ |
| Modify | `docs/writeup.md` | Ideas 10–12 + summary rows |
| Modify | `docs/report.html` | tabs 10–12 + summary rows |
| Write | `artifacts/results/29–34_*.{pt,csv,png}` | checkpoints, probes, plots |

## Alternatives considered and rejected

All ideation candidates from this session, with the reason each was not taken into scope:

- **B — modular-arithmetic / cyclic (grokking) code.** Encode country via a mod-p predicate on a
  binned projection; ties to grokking's Fourier features. *Rejected:* a circular probe defeats it,
  so it scores low on probe-resistance, and the d=2 bottleneck (Idea 9) already found k=2 circular
  codes *emergently* — low marginal value.
- **C — FSA / iterated-state code.** Feature = state of a small automaton after k simulated steps;
  decode depth = k; ties to the state-tracking papers. *Rejected:* the dataset has no sequence
  dimension, so realizing an iterated map in a fixed-input MLP is awkward; highest setup risk,
  least visually clean.
- **β — k-parity (deep parity).** XOR of k features; parity is the canonical AC0-hard function
  (Håstad). *Rejected (held):* it is a *diagnostic* of where parity linearises, not a
  representation — and the existing XOR result (k=2 → 0.951) already shows depth linearises it.
- **δ — fractal threshold (Cantor / Weierstrass).** Square wave taken to the self-similar
  continuum. *Rejected (held):* it is the limit of A and collapses into the A k-sweep; the net
  cannot fit high-order fractal detail anyway.

## Goals and decision history

**Selection goals (user: "all of these").** A code worth building should ideally maximize all four:
(1) decode complexity (ReLU/depth required), (2) probe-defeating (linear, nonlinear, SAE, bilinear
all fail/mislead), (3) theory grounding (a paper it is the minimal instance of), (4) analytic/visual
elegance. "All four at once" is what eliminated B and C (each strong on only one or two axes).

**Showpiece decision (user: "both α and γ").** A is primary by *relevance* to the binary-country
open problem; α and γ both ride along as the "even weirder" showpieces.

**Constructive-vs-emergent reframing.** A peer review flagged that A/α/γ all *impose* geometry and
warned against calling them emergent. Verified: the spec never claimed they were — it reserves
"emergent" for Idea 9. Adopted the terminology discipline regardless (label A/α/γ **architecturally
induced/constructive**) and made the **constructible-vs-emergent gap** the explicit contribution,
with Idea 9 as the emergence control. The deeper issue that review *missed* — the **spreader
dependency** — is now the central mechanism of the spec.

**Verification facts (checked against the repo this session).** Feature indices `country`=5,
`food`=3, `sentiment`=4, `question`=1 (the old bilinear plan's `country=2` was stale). `acts.npz`
carries `train_emb/test_emb`, `*_labels`, and `*_h2`. `angular_freq_r2` exists in
`scripts/28_analyze_bottleneck.py` at `max_k=4` (to be lifted and bumped to 5). Highest existing
script is 28, so 29–34 are free. Full test suite is green (19/19).

## Success criteria

Framed as **constructive existence**, not emergence (Idea 9 remains the emergence control):

- A: **k=2** achieves `country` linear ≤ 0.55, nonlinear ≥ 0.90, freq-peak = 2, AND `country=1`
  occupies ≥ 2 arcs (interleaving realized, not collapsed). k=3+ reported as found; a ceiling is
  an acceptable, documented outcome.
- α: harmonic-confusion heatmap **predominantly** diagonal (each multiplexed feature's own-harmonic
  R² is its max); multiplexed features ≤ 0.55 linear, ≥ 0.85 nonlinear. Off-diagonal leakage
  reported quantitatively.
- γ (stretch): `country` linear ≤ 0.55, nonlinear ≥ 0.90, exactly one net disc-crossing, both
  rings populated. A clean unlink/collapse is an acceptable negative result.
- Honest labelling throughout: A/α/γ described as **architecturally induced / constructive**,
  contrasted against Idea 9's emergent result. The capstone's claimed contribution is the
  **constructible-vs-emergent gap**, not spontaneous discovery.
- All new tests pass; full suite stays green (currently 19/19); writeup + report updated with
  real CSV numbers (no placeholders).
