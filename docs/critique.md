# Critique: Task 1, 2, and 3 — Honest Assessment

> **Status (current):** This document is a resolution log. The Task 2 factual
> error and all three Task 3 conceptual problems flagged below have since been
> fixed in `writeup.md` and `report.html` — each is marked **[RESOLVED]** inline.
> Task 3 now contains **nine** experiments, not five. Remaining open items are
> repo-hygiene only (untracked artifacts, unpinned `requirements.txt`).

---

## Task 1 — NAILED. Nothing more needed.

The requirement was: *"Identify which of the eight features is not represented linearly."*

Done and exceeded:
- Linear vs. nonlinear probe across all 8 features at h2 — clean signal
- Layer trace showing the gap appears only at h2 — not required, but strengthens the answer
- LOTO cross-validation ruling out template artefacts — rigorous

Answer is correct, evidence is multi-layered and convincing. Stop here for Task 1.

---

## Task 2 — MOSTLY DONE. Three real gaps remain.

The requirement was: *"Describe the geometric structure. Show the analysis that convinced you."*

The core answer is correct (absolute-value code on the food axis, Z/2 symmetry). The numbers hold up. But three things are missing that would make this genuinely complete:

### Gap 0 — FACTUAL ERROR: the write-up says country and food are correlated in the dataset **[RESOLVED]**

*Fixed: `writeup.md` now states the labels are independent (corr = −0.005) and that the model entangles them representationally, not statistically.*

`docs/writeup.md` line 53-54 states: *"the food feature and the country feature are correlated in the dataset: texts that mention food are more likely to be about a specific country."*

This is **false and contradicted by our own data.** `artifacts/results/06_country_food.csv` shows `corr_country_food = -0.005` — essentially zero. The two labels are statistically independent in the training set.

The correct framing is: the labels are independent, but the model **representationally entangles** them — it chose to co-encode both features on the same activation axis even though the label correlation gave it no reason to. That is actually the more interesting and accurate observation. Fix this before submission.

### Gap 1 — The WHY is absent **[RESOLVED]**

*Fixed: `writeup.md` now has a "Why the model chose this representation" section (capacity pressure / spontaneous superposition on the food axis).*

The write-up describes *what* the encoding is, but not *why the model chose it*. The mechanistic story is: the model found it efficient to share the food axis for both features simultaneously, with country=0 texts getting amplified food projections and country=1 texts getting muted ones. This is spontaneous superposition on the food axis — the model is doing two jobs with one direction despite the labels being uncorrelated. Without this, the write-up describes the symptom but not the cause.

### Gap 2 — The decoding mechanism downstream is unaddressed **[RESOLVED]**

*Fixed: `writeup.md` now has "The h3 decoding circuit" section tracing the two opposing ReLU neurons (+2.603 / −2.522) that implement `|x| = ReLU(x) + ReLU(−x)`, verified in `07b_h3_circuit.csv`.*

After h2, how does the model *use* this nonlinear code to produce the final sigmoid output? The answer must involve the ReLU at h3 — specifically, the h3 linear layer splits the food axis into two ReLU half-planes, implementing `|x| = ReLU(x) + ReLU(-x)`. The minimum capacity analysis identifies that 2 hidden units suffice (2 ReLUs needed), but the actual decoding circuit in the original model is never traced. That is the other half of the geometric story.

---

## Task 3 — Conceptual problems were RESOLVED; now nine experiments.

The requirement was: *"Train a model that encodes F in a more interesting way. Show what structure emerged. Document what worked and what did not."*

Nine experiments are now documented (Ideas 1–9 plus an analytic rank-1 result). The four conceptual problems below were all flagged when there were five experiments; each is now fixed.

### Problem 1 — XOR/parity is misframed **[RESOLVED]**

*Fixed: `writeup.md` now opens the XOR result with "This experiment failed to produce a weirder representation" and explicitly frames the linear probe of 0.951 as a failure-with-an-informative-finding, not a success.*

Linear probe = 0.951 means the representation is *more linearly decodable* than the baseline (0.471). The result is a *less weird* encoding, not a weirder one. The write-up frames this as a "finding" (the model linearises XOR), which is intellectually honest and interesting — but it needs to be explicitly acknowledged as a failure to achieve the stated goal. Right now it reads as if 0.951 is a success.

### Problem 2 — The rotational code claim has a theoretical error **[RESOLVED]**

*Fixed: `writeup.md` now states the 0.989 linear probe is expected because two orthogonal clusters are trivially separable, and notes rotational codes only resist linear probes for 3+ classes — which motivates the MNIST experiment (Idea 6).*

The write-up claims rotational codes resist linear probes. For a binary feature, this is wrong. Two orthogonal clusters (0 degrees and 90 degrees) are still linearly separable — any diagonal hyperplane between them works. The result confirms this: linear probe = 0.989. The rotation is genuinely there in the geometry (two orthogonal directions, confirmed by the scatter plot), but calling it a representation that "linear probes fail to decode" is incorrect for the binary case. Either fix the theoretical claim or acknowledge this explicitly.

### Problem 3 — JEPA wording is wrong about what the predictor takes as input **[RESOLVED]**

*Fixed: `writeup.md` now correctly describes P as taking the 7 other binary feature labels as input (matching `scripts/12_train_jepa.py`: `other_tr = lab_tr[:, OTHER]`), and flags the experiment as erasure rather than encoding.*

`docs/writeup.md` says: *"Train predictor P that maps h2 with the country dimension zeroed out to the full h2."*

This is factually wrong. `scripts/12_train_jepa.py` shows `predictor(other_tr)` where `other_tr = lab_tr[:, OTHER]` — the 7 other binary label values, not a masked version of h2. The predictor takes binary labels as input and predicts h2 activations. This is label-conditional prediction, not masked-representation prediction. The distinction matters because the JEPA framing is weaker than claimed: standard JEPA masks a portion of the latent representation; what we did is condition on external labels. Both the description and the theoretical framing in the write-up need correction.

### Problem 4 — "What structure emerged" is thin **[LARGELY RESOLVED]**

*Mostly fixed: the rotation, superposition, MNIST, and bottleneck experiments now report concrete learned geometry — cluster angles and radii, per-feature angular-frequency R² (k=1..4), and the unit-norm fix that removed the radius shortcut. The superposition cross-shape mismatch is now discussed explicitly rather than ignored. Original (now-addressed) gaps were:*

The puzzle specifically asks what structure emerged in the trained model. For all 5 experiments, probe accuracy numbers are reported and one scatter plot is shown per experiment, but the actual learned structure is not characterised:

- **Rotation:** What is the actual learned 2D projection matrix? Are the two clusters truly on the unit circle or at arbitrary radii?
- **JEPA:** What does the predictor matrix W look like? Is it rank-1 or full rank?
- **Superposition:** The 4-corner scatter (17_super_geometry.png) shows a cross shape, not 4 interleaved corners — the geometry did not match the target, which is an interesting finding but is not discussed anywhere.
- **Helix and XOR:** The collapse and linearisation findings are acknowledged but not analysed as deeply as they deserve.

---

## What the puzzle is really asking for in Task 3

> "We will happily read about your failures if the path to them was thoughtful."

The failures — helix collapsing to 1D, XOR linearising, rotation not resisting linear probes — are arguably the most interesting results. They should be front-and-centre, not tucked into footnotes. The write-up should lean into these as genuine discoveries.

---

---

## Repo and verification issues (raised by external reviewer — all verified correct)

### Issue 1 — Test suite has a real failure

`pytest -q` produces **1 failed, 15 passed**. The failure is `tests/test_activations.py::test_head_taps_shapes` with `RuntimeError: Numpy is not available`. Root cause: numpy 2.3.5 is installed but torch 2.2.2 was compiled against numpy 1.x — the ABI is incompatible. `tests/test_models_t3.py` passes cleanly (8/8) because it does not route through the numpy bridge, but the full suite is broken. Fix: pin `numpy<2` in requirements or upgrade torch.

### Issue 2 — requirements.txt has no version pins

Every entry is unpinned (`torch`, `numpy`, `scikit-learn`, etc.). The numpy/torch incompatibility is a direct consequence. Add version pins (e.g. `numpy<2`, `torch>=2.2,<2.3`) or commit a lock file.

### Issue 3 — docs/ and one figure are untracked

`git status` shows:
```
?? artifacts/results/07_conditional_axis.png
?? docs/
```
If this is meant to be a reproducible submission, `docs/writeup.md`, `docs/critique.md`, `docs/report.html`, and `artifacts/results/07_conditional_axis.png` should all be committed.

---

## Priority order for remaining work

| Priority | Area | What to do | Status |
|---|---|---|---|
| 1 | Task 2 | Fix the false correlation claim: labels are independent (corr=-0.005); the entanglement is representational, not statistical. | ✅ Done |
| 2 | Task 2 | Fix JEPA description: predictor takes 7 binary labels, not masked h2. Correct both the description and the theoretical framing. | ✅ Done |
| 3 | Task 2 | Add the mechanistic WHY: spontaneous superposition on the food axis, and the h3 decoding circuit. | ✅ Done |
| 4 | Task 3 | Fix the rotation theory claim for binary case. One paragraph correction. | ✅ Done |
| 5 | Task 3 | Reframe XOR as a failure with an interesting finding. Reframe, do not remove. | ✅ Done |
| 6 | Task 3 | Deepen "structure emerged" for at least rotation and superposition. | ✅ Done (also MNIST, bottleneck) |
| 7 | Task 3 | Connect the experiments into one coherent argument, not a list. | ◑ Partial — intro/summary now foreground the strongest results (MNIST, bottleneck); nine experiments still run long. |
| 8 | Repo | Pin numpy<2 (or upgrade torch) to fix the broken test. | ☐ Open |
| 9 | Repo | Commit docs/ and untracked `artifacts/results/*` (h3 circuit, MNIST, SAE, bottleneck, bilinear). | ☐ Open |
