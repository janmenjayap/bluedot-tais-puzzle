# BlueDot TAIS — Puzzle 1 Write-up

**Model:** 5-layer MLP (384 → 64 → 64 → 64 → 64 → 8) on frozen
`sentence-transformers/all-MiniLM-L6-v2` embeddings.
**Task:** 8 binary features — `number`, `question`, `color`, `food`,
`sentiment`, `country`, `person`, `body_part`.

---

## Task 1 — Which feature has the unusual encoding?

**Answer: `country`.**

To find it, I ran linear and nonlinear (2-layer MLP) probes at every
layer for every feature.

| Layer | country linear | country nonlinear | gap |
|-------|---------------|-------------------|-----|
| emb   | 0.993         | 0.991             | 0.00 |
| h0    | 0.989         | 0.991             | 0.00 |
| h1    | 0.993         | 0.991             | 0.00 |
| **h2**| **0.471**     | **0.966**         | **0.495** |
| h3    | 0.964         | 0.970             | 0.01 |
| logits| 0.963         | 0.965             | 0.00 |

Every other feature (number, question, color, food, sentiment, person,
body_part) has linear probe accuracy between 0.97 and 1.00 at h2. Country
alone falls to 0.471 — below chance — while a nonlinear probe recovers it
to 0.966. This large linearity gap (0.495) is uniquely diagnostic.

The gap is robust: leave-one-template-out cross-validation gives
linear = 0.486, nonlinear = 0.995, confirming the finding is not an
artefact of template structure.

**The gap appears only at h2.** Before h2 the embedding still carries a
linearly decodable country signal. After h2 the model re-encodes it into
a nonlinear code that persists to the output. The encoding is introduced
by the transformation at layer 2.

*[Figure: 03_gap.png — bar chart, linear vs nonlinear probe per feature at h2]*

*[Figure: 04d_projection_hist.png — projection on best linear direction at h2; F=0 and F=1 overlap completely, confirming the linear failure]*

---

## Task 2 — How is `country` encoded?

**Answer: as the absolute value (magnitude) of the projection on the food
axis — an interval code with Z/2 symmetry.**

### The food axis connection

The food and country labels are statistically independent in the dataset
(label correlation = −0.005). Yet the model representationally entangles
them — it co-encodes both features on the same activation axis at h2.
When I projected h2 activations onto the direction that best separates
`food=0` from `food=1`, I found the four (country, food) combinations
occupy four distinct intervals:

| (country, food) | mean projection | n |
|---|---|---|
| (0, 0) | −21.8 | 368 |
| (0, 1) | +18.9 | 386 |
| (1, 0) | −5.0  | 381 |
| (1, 1) | +4.4  | 365 |

The key observation: `country=1` clusters near zero on this axis regardless
of food value. `country=0` is pushed to the extremes (large negative or
large positive). In other words:

> **country = 1 if and only if |projection on food axis| is SMALL.**

*[Figure: 07_mechanism.png — histogram of h2 projection on the food axis, stratified by (country, food). The four clusters show the interval structure clearly.]*

### Verification

Replacing the raw projection with its absolute value in a logistic
regression:

| Decoder | Accuracy |
|---|---|
| raw projection | 0.502 (chance) |
| \|projection\| | **0.946** |
| projection² | **0.946** |

The raw signed projection is at chance. The absolute value (or any even
function of it) recovers the feature to 0.946, confirming the Z/2
symmetry.

### Minimum complexity

A 2-hidden-unit MLP achieves 0.950 on country. A 1-unit MLP achieves only
0.721. The encoding is effectively 1-dimensional, and a 2-ReLU circuit
suffices to decode it: `|x| = ReLU(x) + ReLU(−x)`. A single ReLU cannot
recover an even function, so two units is a natural lower bound for this
Z/2-symmetric code, and the empirical jump from 1 to 2 units is consistent
with that.

*[Figure: 05_geometry.png — 2-unit MLP projection plane (acc = 0.950) alongside PCA-2D of h2. The MLP projection plane shows the V-shape; PCA shows no linear separation.]*

### Why the model chose this representation

Both country and food are linearly decodable from the raw
sentence-transformer embeddings (country: 0.980, food: 0.967), so the
model could have kept both as independent linear directions throughout.
Instead, at h2 it spontaneously shares a single axis for both: food is
encoded as the sign of the projection, country as the magnitude. This is
representational superposition — two features on one axis, with country
compressed into the magnitude channel. The likely pressure is capacity:
the 64-dimensional bottleneck carries 8 features simultaneously, and
co-encoding two features that are already aligned in embedding space is a
natural compression strategy even when their labels are uncorrelated.

### The h3 decoding circuit

After h2, the model must recover country from this magnitude code. The h3
weight matrix reveals the mechanism directly: two neurons have
near-perfectly equal-and-opposite loadings on the food axis (+2.603 and
−2.522, ratio 1.03). One neuron activates for positive food-axis
projections, the other for negative projections. Together they implement
`|x| ≈ ReLU(x) + ReLU(−x)` — a two-ReLU circuit for recovering
magnitude from a signed code. These two neurons alone decode country at
0.895 — recovering most of the country signal, with full h3 reaching
0.961, so the remaining neurons add some meaningful cleanup. This lines up
with the probe analysis: 2 hidden units suffice, consistent with the two
ReLUs this magnitude-decoding circuit uses.

### Summary

The model encodes `country` not as a direction in activation space but as
a magnitude threshold on the food axis. Country=1 texts sit close to the
origin; country=0 texts are pushed to the extremes (positive or negative
depending on food). The feature is invisible to linear probes because the
two classes straddle the origin symmetrically. The h3 layer then decodes
this nonlinear code using two opposing ReLU neurons — an interpretable,
compact circuit that naturally emerges from the training objective.

---

## Task 3 — Twelve new representations

The baseline encodes `country` with Z/2 symmetry, intrinsic dimension 1D,
and linear probe accuracy at chance (0.471). Before running experiments I
had to decide what "more interesting" actually means. I used two definitions:

1. **Probe-resistant**: linear probe accuracy stays near chance, as in the
   baseline. This is the narrow definition.
2. **Geometrically richer**: higher-dimensional symmetry group, multiple
   features sharing one space, angular rather than magnitude coding. A
   representation can be geometrically richer even if it remains linearly
   decodable — that is a consequence of the label structure, not the encoding.

The twelve experiments below probe both definitions — nine surveying the space,
then a three-experiment capstone (Ideas 10–12) that returns to the binary-country
problem and constructs a probe-resistant code by topology. The main lessons:

- **Probe resistance requires architectural pressure, not harder labels.**
  Depth linearises internally — XOR and helix both arrived at h2 already
  linearised, regardless of how complex the label was. Binary labels make
  this worse: two clusters are always linearly separable, so SO(2) and
  helical codes cannot resist a linear probe no matter how clean the geometry.
- **Genuine probe resistance needs multi-class structure.** The MNIST
  10-class circular code is the strongest positive result: interleaving even
  and odd digits around the circle plus a unit-norm constraint drives the
  linear probe to chance (0.51) while the nonlinear probe stays at 0.985 — a
  47-point gap, the largest in the study.
- **Geometric richness can emerge without being prescribed.** The unit-norm
  d-sphere bottleneck recreates the baseline's capacity pressure and lets the
  geometry self-organise: `person` develops a +0.42 linear/nonlinear gap with
  no target angles, and `country`/`food`/`sentiment` settle on k=2 angular
  codes. This, with the superposition model's shared 2D space, connects
  directly to the AI-safety question of how networks store more features than
  they have dimensions.

The experiments are ordered from clearest failure to clearest success, so the
strongest results (MNIST circular, the d-sphere bottleneck, and the linked-ring
capstone) come last.

---

### Idea 1 — XOR / Parity code

**What:** Introduce a synthetic feature `parity = sentiment XOR question`.
Train a 9th output head to predict it. The model must represent both inputs
jointly — no single linear direction encodes XOR.

**Why interesting:** XOR is the canonical proof that depth is necessary.
The abstract state-tracking literature (Li et al., arXiv:2503.02854, ICML
2025) finds that real transformers implement a parity-pruning mechanism
when doing compositional state updates. This experiment is a controlled
miniature version of that result.

**Results:**

| Probe | Accuracy |
|---|---|
| Linear (h2) | 0.951 |
| 2-layer MLP | 0.982 |

**This experiment failed to produce a weirder representation.** Linear
probe = 0.951 means the representation at h2 is almost entirely linearly
decodable — more linear than the baseline (0.471), not less. The goal was
a hard-to-probe encoding; what emerged was an easy one. The reason is
capacity: the 5-layer MLP has enough depth to internally compose sentiment
and question across earlier layers, arriving at h2 with parity already
linearised. The interesting finding is not the representation itself but
what it reveals about the model — it solves XOR compositionally before h2,
which is exactly the parity-pruning mechanism Li et al. describe in larger
models. The failure is informative: forcing a weirder code requires
architectural constraints, not just a harder label.

*[Figure: 09_xor_geometry.png — PCA-2D of h2 coloured by parity label; the two classes form loosely separable regions rather than the checkerboard expected for a pure XOR code.]*

---

### Idea 2 — Rotational / SO(2) code

**What:** Add a circular regulariser that pushes a learned 2D projection
of h2 onto the unit circle: `country=0` → angle 0°, `country=1` → angle
90°. This is a phase code rather than a magnitude code.

**Why interesting:** LeJEPA's identifiability theorem (Klindt et al.,
arXiv:2605.26379) states that a consistent world-model transition operator
must be linear and orthogonal — exactly a rotation matrix. This model is a
minimal concrete instance: the update from `country=0` to `country=1` is
a 90° rotation in the learned 2D subspace. Identifiability holds because
the operator is orthogonal.

**Results:**

| Probe | Accuracy |
|---|---|
| Linear on full h2 | 0.989 |
| Angular decoder on 2D projection | 0.988 |

The angular separation is clean: country=0 clusters at mean 1.0° (std
8.5°), country=1 at mean 89.4° (std 8.7°) — very close to the intended
0° and 90°. However the points are not on the unit circle. Country=0 has
mean radius 6.24, country=1 has mean radius 3.67 — the model also used
radial magnitude as a discriminative signal, even though that was not
targeted by the regulariser. The circular geometry is real but not pure:
the learned 2D projection mixes phase code (angle) and magnitude code
(radius) simultaneously.

A linear probe achieves 0.989 because two orthogonal clusters are
trivially linearly separable by a diagonal hyperplane — this is a
fundamental limitation of binary labels, not a failure of the
regulariser. Rotational codes only resist linear probes for three or more
classes arranged around a circle. The experiment's value is the
orthogonal transition operator (connecting to LeJEPA identifiability),
not defeating probes.

*[Figure: 11_rotation_geometry.png — 2D scatter of the learned projection; country=0 clusters along the positive x-axis, country=1 along the positive y-axis, confirming the 90° rotation.]*

---

### Idea 3 — JEPA-residual code

**What:** Train a predictor P that takes the 7 other binary feature labels
(all features except country) as input and predicts the full h2
activations. The residual h2 − P(other_labels) is what the other labels
cannot explain — country's contribution. The question then becomes
empirical: is that residual direction consistent across contexts, and is P
linear?

**Why interesting:** This is a direct implementation of the JEPA objective
at tiny scale. LeJEPA's identifiability condition says the residual is
consistent across contexts if and only if P is a linear operator. Checking
whether P is linear is therefore an empirical test of the condition.

**Results:**

| Probe | Accuracy |
|---|---|
| Linear on raw h2 (country) | 0.471 (chance) |
| Linear on residual (country) | 0.551 |
| All other features in residual | ~1.000 |

Country is nearly absent from raw h2 (below chance) and only weakly
present in the residual. More importantly, a linear matrix W approximates
P well — the unexplained fraction ||P(z) − Wz||² = 0.000 — confirming
that **P is essentially linear**. This empirically validates the LeJEPA
identifiability condition in a fully controlled ground-truth setting: the
transition operator is linear, so the residual direction is consistent
across contexts.

Note: this experiment is **erasure, not encoding**. The low probe accuracy
is achieved by removing country from h2, not by representing it in a
geometrically richer way. It answers a different question — can we
deliberately strip a feature from a layer? — rather than the original
question of how to encode it more interestingly.

*[Figure: 13_jepa_residual_probe.png — per-feature bar chart of raw h2 vs residual probe accuracy; country is the only feature below chance in h2 and only marginally above chance in the residual.]*

---

### Idea 4 — Helical code

**What:** Add a regulariser that pushes a learned 3D projection of h2 onto
a helix: `country=0` → (cos 0°, sin 0°, 0), `country=1` → (cos 90°,
sin 90°, pitch). Motivated by transformer positional encodings and
grid-cell theories of spatial encoding.

**Results:**

| Probe | Accuracy |
|---|---|
| 1D projection | 0.989 |
| 2D angular | 0.993 |
| 3D helical fit | 0.993 |

**Finding:** The helix collapses to 1D for a binary label. The two classes
sit at the tips of two separate 1D manifolds rather than on a spiral. This
is the expected degenerate case: a helix only shows its distinctive
structure for multi-valued (ordinal or continuous) variables. For a binary
label, the circular and linear components align and the representation
reduces to a simple line.

*[Figure: 15_helix_geometry.png — 3D scatter; two separate line segments confirm the 1D collapse.]*

---

### Idea 5 — Superposition / entanglement code

**What:** Force `country` and `food` to share a 2D bottleneck with
interleaved target positions: (c=0,f=0)→(1,0), (c=1,f=0)→(0,1),
(c=0,f=1)→(−1,0), (c=1,f=1)→(0,−1). Neither feature occupies a
dedicated dimension.

**Why interesting:** Anthropic's superposition hypothesis predicts that
models under capacity pressure interleave features into shared directions.
This experiment is a deliberate, controlled demonstration: the failure mode
is not mysterious — it is the geometry. This is the only experiment that
produces a representation that is both geometrically novel relative to the
baseline and directly relevant to the AI safety question of how neural
networks store more features than they have dimensions.

**Results:**

| Feature | Bottleneck probe | Base rate |
|---|---|---|
| country | 0.978 | 0.503 |
| food    | 0.738 | 0.501 |

Measuring the actual geometry: the four (country, food) groups
approximately hit their target angles — (0,0) at 8.7°, (0,1) at 81.2°,
(1,0) at 175.1°, (1,1) at 273.4° — close to the intended 0°, 90°, 180°,
270°. But the precision is uneven. Two groups form tight clusters:
(0,0) at std 22° and (0,1) at std 10°. The (1,1) group is moderately
spread (std 39.5°). The (c=1,f=0) group collapses into a ray (std 71.9°)
with a larger mean radius (4.04 vs 1.4–2.1 for the others), which is why
the scatter plot looks like a cross rather than four clean corners. The
ray overlap is also why food probes at only 0.738 — the (c=1,f=0) group
bleeds angularly into adjacent regions. Country still probes at 0.978
because the country=0 vs country=1 boundary (left half vs right half of
the circle) remains clear despite the spread.

*[Figure: 17_super_geometry.png — 2D bottleneck scatter; four groups occupy four quadrants, confirming the superposition geometry.]*

---

### Idea 6 — MNIST circular code (10-class)

**What:** Train a small CNN on MNIST where digit k is pushed toward angle
k × 36° in a 2D bottleneck (0°, 36°, 72°, …, 324°). The key observation:
even digits (0, 2, 4, 6, 8) and odd digits (1, 3, 5, 7, 9) alternate
perfectly around the circle, so no linear classifier in 2D can separate
them — it would need five cuts. This is the rotation experiment from
Idea 2, but with 10 classes instead of 2, which eliminates the binary-label
collapse problem entirely.

**Why interesting:** This is a direct fix to the failure of Idea 2. The
rotation experiment showed that SO(2) structure is real and well-formed
for binary labels — but binary labels are always linearly separable by a
diagonal hyperplane. Moving to a 10-class dataset with interleaved binary
structure makes the circular encoding genuinely probe-resistant.

**Results (with unit-norm fix applied to bottleneck):**

| Probe | Accuracy |
|---|---|
| 10-class digit accuracy | **0.981** |
| 10-class linear probe on 2D | 0.982 |
| Even/odd linear probe on 2D | **0.51** |
| Even/odd nonlinear probe on 2D | **0.985** |
| Base rate | 0.508 |

The unit-norm constraint (F.normalize on the 2D bottleneck) forces all class
representations onto the unit circle, eliminating radius variance. With unequal
radii the previous version had linear probe 0.624 — the model exploited distance
from origin as a shortcut. After the fix, even/odd linear probe drops to 0.51
(essentially chance), while the nonlinear probe (0.985) and digit accuracy (0.981)
are preserved. The gap between linear and nonlinear probes is now 47 percentage
points — the largest of any encoding in this study.

*[Figure: 19_mnist_circular_geometry.png — 2D scatter coloured by digit,
circles=even, triangles=odd; all points on the unit circle.]*

---

### Idea 7 — SAE decomposition of h2

**What:** Train a top-k sparse autoencoder (SAE, k=10, d_feats=256) on the 64-dimensional
h2 activations of the original puzzle model. Then measure how well each sparse feature
predicts country, individually and jointly.

**Why interesting:** If the country Z/2 encoding is truly nonlinear in h2 space, we expect
it to spread across multiple sparse features — no single dictionary atom should decode it.
The SAE gives an overcomplete basis for h2 that surfaces fine-grained structure.

**Results:**

| Decoder | Accuracy |
|---|---|
| Single best SAE feature (feat 100) | 0.519 |
| Second best SAE feature (feat 36) | 0.592 |
| Top-2 pair, linear probe | 0.707 |
| Top-2 pair, nonlinear probe | 0.717 |
| All 256 SAE features, linear | 0.827 |
| All 256 SAE features, nonlinear | **0.958** |

Country distributes across many SAE features. No single feature reaches above 60%.
The top-2 pair achieves 0.707 — capturing the two arms of the V-shape (positive
and negative food projections). All 256 features together recover 0.958 nonlinearly,
matching the original nonlinear probe (0.966). The representation is genuinely
distributed and nonlinear in the SAE basis.

*[Figure: 21_sae_country_scatter.png — scatter of top-2 SAE features coloured by
(country, food); the four (country × food) groups separate into distinct quadrants.]*

---

### Analytic result: original encoding is rank-1 bilinear

The puzzle's country encoding can be expressed exactly as a rank-1 bilinear form:

```
country_score(h2) = (wf · h2)²  =  h2^T (wf ⊗ wf) h2
```

where wf is the food direction in h2 space. The tensor B = wf ⊗ wf is a 64×64 matrix
of rank 1 with a single non-zero eigenvalue of exactly 1.0. The quadratic form proj²
and the absolute value |proj| carry identical information (cosine similarity = 1.0000).

| Decoder | Accuracy |
|---|---|
| Raw projection (linear) | 0.498 (chance — Z/2 symmetry confirmed) |
| \|projection\| (abs-value) | 0.598 |
| projection² (rank-1 bilinear form) | 0.727 |
| Cosine similarity \|proj\| vs sqrt(proj²) | 1.000 |
| Bilinear tensor rank | **1** |
| Largest eigenvalue | **1.0** |

Note: the probe accuracies here are lower than the mechanism analysis (0.946) because the
food direction is derived from a logistic regression, which produces a slightly different
direction than what the original circuit uses. The mathematical claims (rank=1, cos_sim=1.0)
are exact and hold regardless.

---

### Idea 8 — Bilinear text model + CPD analysis

**What:** Train a single bilinear layer — `h2 = (W_L · emb) ⊙ (W_R · emb)` — as the
entire computation between the frozen sentence-transformer embedding and the 8 output
logits. This architecture IS its own CPD decomposition by construction:

```
B[f,i,j] = Σ_r W_head[f,r] W_L[r,i] W_R[r,j]
```

No separate CPD fitting is needed — the factors L, R, D are the model weights.

**Why interesting:** The original puzzle model's country encoding is rank-1 bilinear.
A bilinear layer is the natural model class for this computation. If the bilinear model
learns country differently than the ReLU MLP, that tells us something about what role
the nonlinearity plays.

**Bilinear XOR (control experiment):** Training HeadBilinearXOR on XOR(sentiment, question)
shows that a bilinear layer can compute XOR — it uses asymmetric L ≠ R to implement the
product structure. XOR linear probe at h2 = **0.963** vs ReLU = 0.951 (Δ = +0.013).
Bilinear makes XOR *more* linearly accessible, not less.

**Bilinear text model results:**

| Feature | Linear probe | Nonlinear probe |
|---|---|---|
| number | 0.979 | 0.980 |
| question | 0.999 | 0.999 |
| color | 0.977 | 0.977 |
| food | 0.983 | 0.980 |
| sentiment | 0.981 | 0.977 |
| country | **0.993** | 0.992 |
| person | 0.997 | 0.995 |
| body_part | 0.979 | 0.981 |
| Overall accuracy | — | **0.986** |

Every feature, including country, is encoded **linearly** in the bilinear h2.
Country's linear probe = 0.993 vs 0.471 in the original model. A bilinear layer
does not produce the Z/2 symmetric representation — it directly implements the
quadratic form and exposes it linearly. The original model's probe resistance
for country comes from the ReLU nonlinearities, not from the bilinear structure.

**CPD analysis:** The top-20 components by sigma (||L[:,r]|| × ||R[:,r]|| × ||D[:,r]||)
show specialization (max |D[f,r]| / Σ|D[:,r]|) around 0.16–0.22 per component,
meaning no single component is fully dedicated to one feature. Country's top-5
components all have asymmetry ||L-R||/||L|| ≥ 1.1, consistent with the bilinear
layer exploiting L ≠ R to represent the quadratic form.

*[Figure: 26_bilinear_cpd_spectrum.png — left: σ spectrum (power law decay);
right: per-component specialization.]*

---

### Idea 9 — Bottleneck superposition (emergent geometry)

**What:** Train a model with a narrow d-dimensional bottleneck and a unit-norm constraint
(`F.normalize`), forcing all representations onto the d-sphere. Two variants: d=4 (4D sphere)
and d=2 (unit circle). Unlike Idea 5 (superposition with prescribed positions), no target
angles are given — the geometry emerges from the training objective alone. Eight per-feature
MLP decoders (d→16→ReLU→1) ensure high task accuracy regardless of geometry.

**Why interesting:** The original Z/2 encoding emerged because the model had capacity pressure
(64 dims for 8 features) and the food/country features were semantically aligned in embedding
space. This experiment recreates both pressures with a harder constraint: d < 8 means
superposition is mathematically unavoidable. The unit-norm constraint eliminates the magnitude
channel (used by the original model), so all information must be encoded as angles. This is the
minimal condition under which emergent nonlinear structure is the model's only viable strategy.

**d=4 results (4 linear directions available for 8 features):**

| Feature | Linear probe | Nonlinear probe | Gap |
|---|---|---|---|
| number | 0.5833 | 0.5987 | +0.015 |
| question | 0.9927 | 0.9927 | +0.000 |
| color | 0.6793 | 0.7867 | +0.107 |
| food | 0.8940 | 0.8960 | +0.002 |
| sentiment | 0.5513 | 0.7593 | +0.208 |
| country | 0.9373 | 0.9360 | -0.001 |
| person | 0.4853 | 0.9067 | **+0.421** |
| body_part | 0.8660 | 0.8660 | +0.000 |

**d=2 results (2 linear directions available for 8 features):**

| Feature | Linear probe | Nonlinear probe | Gap |
|---|---|---|---|
| number | 0.5140 | 0.5360 | +0.022 |
| question | 0.4820 | 0.5213 | +0.039 |
| color | 0.5213 | 0.5307 | +0.009 |
| food | 0.7047 | 0.8933 | +0.189 |
| sentiment | 0.6080 | 0.7973 | +0.189 |
| country | 0.6773 | 0.8580 | +0.181 |
| person | 0.4953 | 0.4920 | -0.003 |
| body_part | 0.9487 | 0.9480 | -0.001 |

**Angular frequency analysis (d=2):**

For d=2, each point on the unit circle has an angle θ = atan2(b₁, b₀). To measure at which
angular frequency each feature is encoded, we regress each binary label on [cos(k·θ), sin(k·θ)]
for k = 1..4 and report R².

| Feature | k=1 | k=2 | k=3 | k=4 | Peak k |
|---|---|---|---|---|---|
| number | 0.002 | 0.001 | 0.004 | 0.004 | k=4 |
| question | 0.003 | 0.005 | 0.002 | 0.001 | k=2 |
| color | 0.002 | 0.004 | 0.004 | 0.004 | k=2 |
| food | 0.228 | 0.318 | 0.085 | 0.078 | k=2 |
| sentiment | 0.103 | 0.229 | 0.060 | 0.010 | k=2 |
| country | 0.147 | 0.486 | 0.045 | 0.043 | k=2 |
| person | 0.001 | 0.002 | 0.001 | 0.001 | k=2 |
| body_part | 0.665 | 0.021 | 0.108 | 0.016 | k=1 |

A peak at k=1 means the feature splits the circle in half (a hemisphere boundary — still
decodable by a linear probe through the origin). A peak at k=2 or higher means the feature
alternates more than once around the circle — a genuinely nonlinear code that requires a
curved decision boundary.

**Finding:** The d-sphere bottleneck with unit-norm produces clear emergent structure. In d=4,
`person` has the largest nonlinear gap of any text-domain experiment (+0.421): near-chance linear probe (0.485)
but 0.907 nonlinear probe — emergent nonlinear encoding with no prescribed geometry. In d=2,
`body_part` dominates the unit circle (k=1, R²=0.665), while `country`, `food`, and `sentiment`
use k=2 angular encoding (alternating twice around the circle). The remaining four features
are squeezed out of the angular structure entirely, consistent with the capacity limit.

*[Figure: 28_bottleneck_d2_scatter.png — 2×4 scatter on unit circle, one subplot per feature,
colour = feature value; body_part shows a clean half-arc split (k=1); food/sentiment/country
show alternating arcs (k=2).]*

*[Figure: 28_bottleneck_d4_pca_scatter.png — 2×4 scatter after PCA(2) of the 4D bottleneck,
one subplot per feature; person shows curved crescent structure consistent with the +0.421 gap.]*

---

## Capstone — three *constructive* binary probe-resistant codes (Ideas 10–12)

The nine experiments above left one problem open: every probe-resistant win
(MNIST, the d=2 bottleneck) either abandoned the binary `country` label for a
multi-class one or was *emergent* (no prescribed geometry). The capstone asks the
complementary, constructive question: **can a binary, text-domain feature be
*made* probe-resistant by imposing a geometry no single hyperplane can cut?** All
three codes below are **architecturally induced**, not emergent — the geometry is
imposed by construction. They are the constructive counterpart to Idea 9's
emergent control, and the honest contribution is the *gap between the two*.

The shared mechanism is the **spreader**: a binary label has no internal variation
to populate a manifold, so a harmonic readout or a ring target can be satisfied by
collapsing all `feature=1` samples into one arc/point — linearly separable again.
Every code below therefore co-trains the other features as a spreader that
populates the manifold while `country` interleaves/links over it.

---

### Idea 10 — Square wave (constructive, the binary-country attack)

**What:** `emb → enc[64→ReLU→64→ReLU] → Linear(64,2) → F.normalize → θ`. The
`country` logit is *constrained* to the k-th circular harmonic,
`α·cos(kθ)+β·sin(kθ)+b`; the other 7 features use free MLP heads on the circle and
act as the spreader. Sweep k = 1..5. k=2 is the XOR-on-a-circle checkerboard that
no diagonal can cut.

**Why interesting:** this is the binary, text-domain analog of the MNIST even/odd
result — a dose-response over angular frequency that tests whether *imposing* a
harmonic readout is enough to defeat a linear probe.

**Results (per-k, `country` on the 2-D circle):**

| k | linear | nonlinear | freq peak | country=1 arcs | overall acc |
|---|---|---|---|---|---|
| 1 | 0.503 | 0.539 | k=4 | 12 | 0.710 |
| **2** | **0.711** | **0.959** | **k=2** | **12** | 0.719 |
| 3 | 0.961 | 0.967 | k=3 | 5 | 0.683 |
| 4 | 0.985 | 0.985 | k=3 | 3 | 0.662 |
| 5 | 0.987 | 0.988 | k=4 | 3 | 0.659 |

**Finding — a non-monotonic dose-response, and an honest partial.** k=2 is the
sweet spot: the angular-frequency peak lands exactly on the imposed k=2, the
nonlinear probe reaches 0.959, and there is a real +0.25 linear/nonlinear gap. But
the linear probe is still **0.711 — it does not beat the baseline's 0.471.** The
binary label leaks into the k=1 component (a half-circle split a linear probe can
read), because constraining the *readout* to k=2 does not force the *geometry* to
be purely k=2. At k=1 `country` collapses to chance on *both* probes (it is not
encoded at all); at k≥3 the model abandons the fast harmonic and dumps `country`
into a linearly-trivial cluster (linear 0.96–0.99, arc occupancy collapsing 5→3).
**Lesson:** imposing a harmonic readout is not sufficient for probe resistance — a
binary label will take the linearly-cheap escape unless the geometry itself forbids it.

*[Figure: 30_squarewave_curve.png — linear vs nonlinear country probe vs harmonic k;
the linear curve dips toward chance only near k=1–2 then rises to ~0.99.]*

---

### Idea 11 — Fourier comb (constructive, superposition showpiece)

**What:** same 2-D circle, but multiplex three features onto distinct harmonics —
`country→k=2`, `food→k=3`, `sentiment→k=4` — each with its own harmonic readout;
the remaining five use free MLP heads. One scalar angle is asked to carry three
independent bits as three orthogonal harmonics.

**Why interesting:** the linear-probe failure *is* the geometry — multiplexing more
features than linear directions is the controlled form of superposition. The
diagnostic is an 8×5 harmonic-confusion matrix `R²(feature | harmonic k)`.

**Results (multiplexed features):**

| Feature (target k) | linear | nonlinear |
|---|---|---|
| country (k=2) | 0.529 | 0.525 |
| food (k=3) | 0.937 | 0.939 |
| sentiment (k=4) | 0.598 | 0.867 |

**Harmonic-confusion R² (peak in bold; target k starred):**

| Feature | k=1 | k=2* | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|
| question | 0.773 | **0.823** | 0.183 | 0.038 | 0.121 |
| food | 0.639 | **0.735** | 0.699 | 0.073 | 0.027 |
| sentiment | 0.056 | 0.014 | 0.070 | **0.550** | 0.496 |
| country | 0.005 | 0.005 | 0.002 | 0.001 | 0.002 |

(country target k=2, food target k=3, sentiment target k=4.)

**Finding — one clean multiplex, and a diagnosable failure.** Only **sentiment**
lands on its target harmonic (peak at k=4, +0.27 gap). **food** leaks to k=2 and
stays linearly readable (0.937). **country** collapses to chance on both probes —
it is not encoded at all. The harmonic-confusion matrix shows why: `question`, an
*unconstrained* feature and the easiest in the dataset, hijacks the k=1/k=2
harmonics (R² 0.77/0.82), starving `country` and `food` of the low-frequency
capacity they needed. **Lesson:** superposition on one angle is real but fragile —
an easy unconstrained feature will commandeer the cheap harmonics, and three bits
on one scalar is past the d=2 capacity the emergent bottleneck (Idea 9) already
flagged at ~k=2.

*[Figure: 32_harmonic_confusion.png — 8×5 R² heatmap; the target cells (red boxes)
are bright only for sentiment, and question's k1/k2 row dominates.]*

---

### Idea 12 — Linked rings (constructive, topological showpiece) — the strongest result

**What:** `emb → enc → Linear(64,3)` 3-D bottleneck. A regulariser pushes
`country=0` onto ring A (unit circle, xy-plane, centre origin) and `country=1` onto
ring B (unit circle, xz-plane, centre (1,0,0)) so that B threads A (linking number
1). `food` sets the within-ring angle (the spreader). All 8 features decode through
per-feature MLP heads; loss = BCE + λ·(distance to target ring), λ=1.0.

**Why interesting:** the sharpest rebuttal to "binary labels are always linearly
separable." Two *linked rings* are provably not separable by any hyperplane (any
plane that puts ring A on one side forces all of ring B onto the same side). If the
model realises this geometry, a linear probe *must* fail while a nonlinear one
succeeds — by topology, not by tuning.

**Results (`country` on the 3-D bottleneck):**

| Metric | Value |
|---|---|
| linear probe | **0.533** (near chance) |
| nonlinear probe | **0.940** |
| linear/nonlinear gap | **+0.41** |
| net disc-crossings (linking number) | **1** |
| ring-B arc occupancy | 9 / 12 |
| overall 8-feature acc | 0.806 |

**Finding — the open problem, answered constructively.** This is the strongest
result of the capstone and the closest any experiment came to the baseline's
probe-resistance *with a genuinely weirder geometry*: the linear probe sits at
0.533 (near chance, vs baseline 0.471) while the nonlinear probe holds 0.940 — a
+0.41 gap — **and the linking is real**: a numerical disc-crossing count returns
exactly one net crossing of the `country=1` cloud through ring A's disc. The
collapse risk did not materialise — even though the nominal spreader (`food`) is
binary, ring B is populated across 9 of 12 angular bins, because BCE pressure from
the other features spread `country=1` around the ring. So the true spreader was the
co-trained features, exactly as the capstone's mechanism predicted. **Lesson:** a
binary feature *can* be made probe-resistant by construction — but it takes a
geometry that is topologically non-separable (linked rings), not merely a
high-frequency readout (Idea 10) or multiplexing (Idea 11).

*[Figure: 34_linked_rings_scatter.png — 3-D scatter coloured by country; the two
unit circles sit in orthogonal planes and interlock, ring B threading ring A's disc once.]*

---

### Capstone summary — the constructible-vs-emergent gap

| Code | country linear | country nonlinear | Verdict |
|---|---|---|---|
| Square wave (k=2) | 0.711 | 0.959 | Partial — real k=2 code, but k=1 leakage keeps it linearly readable |
| Fourier comb | 0.529 (country) | 0.525 | Failed for country (collapsed); only sentiment multiplexed cleanly |
| **Linked rings** | **0.533** | **0.940** | **Success — probe-resistant AND topologically novel (linking number 1)** |

The three constructive codes bracket the answer to the open problem. A harmonic
*readout* (Idea 10) is not enough — the binary label escapes into k=1. Multiplexing
(Idea 11) is fragile to capacity and to easy unconstrained features. Only an
explicitly **non-separable topology** (Idea 12) delivers a binary `country` code
that is simultaneously near-chance to a linear probe and fully recoverable
nonlinearly. Set against Idea 9 (where the same k=2 angular structure arose
*emergently* under capacity pressure), the contribution is the gap itself:
probe-resistant binary geometry is *constructible* by topology, and a milder
version of it *emerges* on its own — but the cheap, high-frequency constructions in
between mostly revert to linear separability.

---

### Task 3 Summary

| Encoding | Linear probe | Verdict | Key lesson |
|---|---|---|---|
| Baseline (abs-value) | 0.50 | — | Z/2 symmetry, rank-1 bilinear |
| XOR / parity | 0.95 | Failed — linearised | Depth beats label complexity |
| Helical | 0.99 | Failed — collapsed to 1D | Helix needs multi-valued labels |
| Rotational SO(2) | 0.99 | Partial — orthogonal structure achieved | Binary labels always linearly separable |
| JEPA residual | 0.47 | Informative — erasure not encoding | Explicit erasure strips a feature from a layer |
| Superposition | 0.98 / 0.74 | Partial — geometrically richer | 2D shared encoding; two features, one space |
| MNIST circular (10-class) | **0.51** | **Best — genuinely probe-resistant** | Multi-class interleaving + unit-norm constraint |
| SAE decomposition (h2) | 0.827 (all 256) | Analytical — distributed encoding | Country spreads across many sparse features |
| Bilinear text model | 0.993 (country) | Informative — ReLU is the source | Bilinear linearises the quadratic form |
| Bottleneck d=4 (unit-norm) | 0.49 (person) | Emergent — nonlinear by capacity | person: linear=0.49, nonlinear=0.91 (+0.42 gap) |
| Bottleneck d=2 (unit-norm) | **0.48 (question)** | **Emergent — angular superposition** | body_part k=1; country/food/sentiment k=2 |
| Square wave k=2 (constructive) | 0.711 (country) | Partial — real k=2 code | Harmonic *readout* leaks to k=1; not sufficient |
| Fourier comb (constructive) | 0.529 (country) | Partial — only sentiment multiplexed | Easy unconstrained feature hijacks low harmonics |
| **Linked rings (constructive)** | **0.533 (country)** | **Success — probe-resistant + topological** | Non-separable topology (linking #1); +0.41 gap |

**Where the baseline stands after the capstone.** The original country code sits
at a linear probe of 0.471 — at chance. For the first nine experiments, nothing
matched it on its own terms: the superposition and bottleneck models make country
*more* linearly decodable (0.978 and 0.677), and the only probe-resistant win there
(MNIST, 0.51) abandoned the country feature and the text model for a 10-class digit
task. That motivated the capstone, and **Idea 12 (linked rings) closed most of the
gap**: a constructive binary-country code with linear probe **0.533** (near chance)
and nonlinear 0.940 — probe-resistant *and* geometrically weirder than `|x|`
(topologically linked rings, verified linking number 1). The honest caveats remain:
0.533 is marginally above the baseline's 0.471 rather than below it, and the code is
*constructed* rather than emergent. The two cheaper constructions confirm why this
is hard — a harmonic *readout* (square wave, k=2 → linear 0.711) leaks into the k=1
half-circle split, and multiplexing (Fourier comb) collapses `country` entirely when
an easy unconstrained feature hijacks the low harmonics. **The finding:** a binary
feature already encoded at chance is close to a local optimum for probe resistance,
and the only thing that reliably beats a cleverer two-cluster geometry is a
genuinely non-separable structure — multi-class interleaving (MNIST) or a linked
topology (Idea 12). Driving a *binary* country encoding strictly below 0.471 while
keeping it weird, and getting that geometry to *emerge* rather than be imposed,
remains open.

---

## Feedback

Really enjoyed working through this. The absolute-value encoding discovery
was a genuine "aha" moment, and I appreciated how open-ended Task 3 was.

A few things I found myself unsure about:

Tasks 1 and 2 felt connected to me. I could not figure out which feature
was encoded without also figuring out how, so the split felt a bit
artificial. Not sure if that is intentional.

For Task 3 I was not sure what "weirder" meant in practice. I kept
second-guessing whether my ideas were in the right direction.

I was not sure what the template_id field in the data was for. I explored
it but never knew if I was chasing a red herring.

Would have loved even a small hint about why this connects to AI safety.
I could see the mechanistic interpretability angle but was not confident
I understood the deeper motivation.

These are minor things. Overall it was a well-crafted puzzle and I learned
a lot from it.
