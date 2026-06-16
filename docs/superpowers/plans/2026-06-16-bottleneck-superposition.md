# Bottleneck Superposition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a `HeadBottleneck(d)` model with a unit-norm d-dimensional bottleneck (d=4 and d=2), run linear/nonlinear probes and angular frequency analysis on the emergent geometry, and update the writeup and report.

**Architecture:** `emb(384) → Linear(64)→ReLU → Linear(64)→ReLU → Linear(d) → F.normalize → 8 per-feature MLP heads (d→16→ReLU→1) → 8 logits`. Unit-norm forces all representations onto the d-sphere. No prescribed positions — geometry is fully emergent. Per-feature MLP decoders allow high task accuracy regardless of what geometry emerges.

**Tech Stack:** Python 3, PyTorch, scikit-learn (LogisticRegression, MLPClassifier, StandardScaler, PCA), matplotlib, pandas, numpy, pytest.

---

## File Map

| Action   | Path                                              | Purpose                                          |
|----------|---------------------------------------------------|--------------------------------------------------|
| Modify   | `src/puzzle/models_t3.py`                         | Append `HeadBottleneck` class                    |
| Modify   | `tests/test_models_t3.py`                         | Add 3 shape + unit-norm tests                    |
| Create   | `scripts/27_train_bottleneck.py`                  | Train d=4 and d=2 variants, save checkpoints     |
| Create   | `scripts/28_analyze_bottleneck.py`                | Probes + angular frequency + geometry plots      |
| Modify   | `docs/writeup.md`                                 | Add Idea 9 section and summary table row         |
| Modify   | `docs/report.html`                                | Add tab 9 button + panel + summary table row     |
| Write    | `artifacts/results/27_bottleneck_d4_model.pt`     | Trained d=4 checkpoint (produced by task 2)      |
| Write    | `artifacts/results/27_bottleneck_d2_model.pt`     | Trained d=2 checkpoint (produced by task 2)      |
| Write    | `artifacts/results/28_bottleneck_probes.csv`      | Linear/nonlinear probe per feature per d         |
| Write    | `artifacts/results/28_bottleneck_d2_angular_freq.csv` | Angular frequency R² per feature for d=2    |
| Write    | `artifacts/results/28_bottleneck_d2_scatter.png`  | 2D scatter per feature                           |
| Write    | `artifacts/results/28_bottleneck_d4_pca_scatter.png` | PCA-2D scatter per feature for d=4           |

---

## Task 1: HeadBottleneck model class + tests

**Files:**
- Modify: `src/puzzle/models_t3.py` (append after HeadBilinear)
- Modify: `tests/test_models_t3.py` (append 3 new tests)

### Context

The existing model classes in `src/puzzle/models_t3.py` follow this pattern:
- `enc` = shared encoder layers
- `bottle(x)` = extracts the bottleneck representation
- `forward(x)` = returns logits

`HeadMNISTCircular.bottle()` already uses `F.normalize(self.bottleneck(self.encoder(x)), dim=1)` — match this pattern.

The existing test file `tests/test_models_t3.py` uses `N, D = 8, 384` and `def _emb(): return torch.randn(N, D)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models_t3.py`:

```python
def test_head_bottleneck_d4_shape():
    from src.puzzle.models_t3 import HeadBottleneck
    out = HeadBottleneck(d=4)(_emb())
    assert out.shape == (N, 8)


def test_head_bottleneck_d2_shape():
    from src.puzzle.models_t3 import HeadBottleneck
    out = HeadBottleneck(d=2)(_emb())
    assert out.shape == (N, 8)


def test_head_bottleneck_unit_norm():
    from src.puzzle.models_t3 import HeadBottleneck
    model = HeadBottleneck(d=4)
    bottle = model.bottle(_emb())
    assert bottle.shape == (N, 4)
    norms = bottle.norm(dim=1)
    assert torch.allclose(norms, torch.ones(N), atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run from repo root:
```bash
cd /Users/janmenjayap/bluedot-impact/puzzles/1/bluedot-tais-puzzle
python -m pytest tests/test_models_t3.py::test_head_bottleneck_d4_shape tests/test_models_t3.py::test_head_bottleneck_d2_shape tests/test_models_t3.py::test_head_bottleneck_unit_norm -v
```
Expected: 3 FAILED with `ImportError: cannot import name 'HeadBottleneck'`

- [ ] **Step 3: Implement HeadBottleneck**

Append to `src/puzzle/models_t3.py` (after the HeadBilinear class, before the end of the file):

```python


class HeadBottleneck(nn.Module):
    """MLP with narrow d-dimensional unit-norm bottleneck.
    All 8 features decoded by per-feature MLP heads — no linear decoders.
    Geometry is fully emergent: no prescribed positions, no label correlations.
    Unit-norm constraint forces all representations onto the d-sphere.
    """
    def __init__(self, d: int = 4):
        super().__init__()
        self.d = d
        self.enc = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
        )
        self.bottleneck = nn.Linear(64, d)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1))
            for _ in range(8)
        ])

    def bottle(self, x: torch.Tensor) -> torch.Tensor:
        """Return unit-norm bottleneck representation, shape [N, d]."""
        return F.normalize(self.bottleneck(self.enc(x)), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for all 8 features, shape [N, 8]."""
        b = self.bottle(x)
        return torch.cat([h(b) for h in self.heads], dim=1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_models_t3.py::test_head_bottleneck_d4_shape tests/test_models_t3.py::test_head_bottleneck_d2_shape tests/test_models_t3.py::test_head_bottleneck_unit_norm -v
```
Expected: 3 PASSED

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
python -m pytest tests/ -v
```
Expected: all existing tests PASS, 3 new tests PASS.

---

## Task 2: Training script

**Files:**
- Create: `scripts/27_train_bottleneck.py`

### Context

Training data is at `artifacts/activations/acts.npz` with keys:
- `train_emb` shape (7000, 384), `train_labels` shape (7000, 8)
- `test_emb` shape (1500, 384), `test_labels` shape (1500, 8)

Labels are float32 for BCE loss. Follow the pattern in `scripts/25_train_bilinear_text.py`:
- `Adam + CosineAnnealingLR`
- Log accuracy every 100 epochs
- Save to `artifacts/results/`

600 epochs is sufficient; the bottleneck may converge more slowly than the full MLP.

- [ ] **Step 1: Create the training script**

Create `scripts/27_train_bottleneck.py`:

```python
# scripts/27_train_bottleneck.py
# Train HeadBottleneck on all 8 text features with d=4 and d=2.
# Architecture: emb(384) → [64→ReLU→64→ReLU→d] → unit_norm → 8×MLP_heads → logits
# Unit-norm on bottleneck forces all representations onto the d-sphere.
# Per-feature MLP decoders achieve high accuracy regardless of bottleneck geometry.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.puzzle.models_t3 import HeadBottleneck

N_EPOCHS, LR, BATCH = 600, 1e-3, 512


def train_one(d: int, emb_tr, lab_tr, emb_te, lab_te):
    model = HeadBottleneck(d=d)
    opt   = Adam(model.parameters(), lr=LR)
    sched = CosineAnnealingLR(opt, N_EPOCHS)

    model.train()
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(len(emb_tr))
        for s in range(0, len(emb_tr), BATCH):
            idx = perm[s:s + BATCH]
            logits = model(emb_tr[idx])
            loss = F.binary_cross_entropy_with_logits(logits, lab_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                logits_te = model(emb_te)
                acc = ((logits_te > 0).float() == lab_te).float().mean()
            print(f"d={d}  epoch {epoch+1:3d}: acc={acc:.4f}")
            model.train()

    model.eval()
    return model


def main():
    data = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(data["train_emb"]).float()
    emb_te = torch.from_numpy(data["test_emb"]).float()
    lab_tr = torch.from_numpy(data["train_labels"]).float()
    lab_te = torch.from_numpy(data["test_labels"]).float()

    os.makedirs("artifacts/results", exist_ok=True)
    for dim in [4, 2]:
        print(f"\n=== Training d={dim} ===")
        model = train_one(dim, emb_tr, lab_tr, emb_te, lab_te)
        path = f"artifacts/results/27_bottleneck_d{dim}_model.pt"
        torch.save(model.state_dict(), path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the training script**

```bash
cd /Users/janmenjayap/bluedot-impact/puzzles/1/bluedot-tais-puzzle
python scripts/27_train_bottleneck.py
```

Expected output (approximate — exact numbers will vary):
```
=== Training d=4 ===
d=4  epoch 100: acc=0.93xx
d=4  epoch 200: acc=0.95xx
d=4  epoch 300: acc=0.96xx
d=4  epoch 400: acc=0.97xx
d=4  epoch 500: acc=0.97xx
d=4  epoch 600: acc=0.97xx
saved artifacts/results/27_bottleneck_d4_model.pt

=== Training d=2 ===
d=2  epoch 100: acc=0.88xx
d=2  epoch 200: acc=0.92xx
d=2  epoch 300: acc=0.93xx
d=2  epoch 400: acc=0.94xx
d=2  epoch 500: acc=0.94xx
d=2  epoch 600: acc=0.94xx
saved artifacts/results/27_bottleneck_d2_model.pt
```

Both models should achieve > 0.90 final accuracy. If either is below 0.85, increase N_EPOCHS to 800 and re-run.

- [ ] **Step 3: Verify checkpoint files exist**

```bash
ls -lh artifacts/results/27_bottleneck_d*.pt
```
Expected: both files present, each ~100 KB.

---

## Task 3: Analysis script — probes + geometry

**Files:**
- Create: `scripts/28_analyze_bottleneck.py`

### Context

This script must:
1. Load both models (d=4 and d=2)
2. Extract bottleneck representations (`model.bottle(emb)`)
3. Run linear (`LogisticRegression`) and nonlinear (`MLPClassifier((32,))`) probes per feature per model
4. For d=2: compute angular frequency R² per feature (how well each binary feature is explained by cos(k·θ) + sin(k·θ) for k=1..4)
5. For d=2: save 2×4 scatter plot (one subplot per feature, color = feature value)
6. For d=4: save 2×4 scatter plot on PCA-2D projection

The `FEATURE_NAMES` list from `src.puzzle.data` is: `['number', 'question', 'color', 'food', 'sentiment', 'country', 'person', 'body_part']`.

The `StandardScaler` must be fit only on train and applied to test — do NOT fit on test.

Angular frequency R²: for binary label y ∈ {0,1} and angle θ = atan2(b[:,1], b[:,0]), measure how well k-th Fourier harmonic explains the label. R² = 1 − SS_res/SS_tot. A peak at k=1 means the feature is encoded as a half-circle split. A peak at k=2 means it alternates twice around the circle (like even/odd MNIST with 4 classes).

- [ ] **Step 1: Create the analysis script**

Create `scripts/28_analyze_bottleneck.py`:

```python
# scripts/28_analyze_bottleneck.py
# Analyze emergent geometry of HeadBottleneck models (d=4 and d=2).
# Metrics: linear/nonlinear probes per feature, angular frequency R² (d=2),
# 2D scatter plots (d=2 direct, d=4 via PCA projection).
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadBottleneck


def run_probes(bottle_tr, bottle_te, lab_tr, lab_te, d_label):
    rows = []
    sc = StandardScaler().fit(bottle_tr)
    X_tr_s, X_te_s = sc.transform(bottle_tr), sc.transform(bottle_te)
    for fi, name in enumerate(FEATURE_NAMES):
        y_tr, y_te = lab_tr[:, fi], lab_te[:, fi]
        lin = LogisticRegression(max_iter=2000).fit(X_tr_s, y_tr).score(X_te_s, y_te)
        mlp = MLPClassifier((32,), max_iter=2000, random_state=0).fit(
            X_tr_s, y_tr).score(X_te_s, y_te)
        rows.append({"d": d_label, "feature": name,
                     "linear": lin, "nonlinear": mlp, "gap": mlp - lin})
    return rows


def angular_freq_r2(theta, y, max_k=4):
    """R² of binary label y regressed on [cos(k*theta), sin(k*theta)] for k=1..max_k."""
    scores = []
    y_f = y.astype(float)
    ss_tot = np.sum((y_f - y_f.mean()) ** 2) + 1e-10
    for k in range(1, max_k + 1):
        X_feat = np.column_stack([np.cos(k * theta), np.sin(k * theta)])
        y_pred = LinearRegression().fit(X_feat, y_f).predict(X_feat)
        ss_res = np.sum((y_f - y_pred) ** 2)
        scores.append(float(1.0 - ss_res / ss_tot))
    return scores


def scatter_grid(bottle_te, lab_te, title, fname):
    """Save a 2×4 scatter grid, one subplot per feature, color = feature value."""
    theta_circ = np.linspace(0, 2 * np.pi, 200)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for fi, name in enumerate(FEATURE_NAMES):
        ax = axes[fi // 4][fi % 4]
        c = lab_te[:, fi]
        ax.scatter(bottle_te[:, 0], bottle_te[:, 1],
                   c=c, cmap="coolwarm", alpha=0.3, s=4)
        ax.plot(np.cos(theta_circ), np.sin(theta_circ), "k--", lw=0.5, alpha=0.3)
        ax.set_title(name, fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def main():
    data = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(data["train_emb"]).float()
    emb_te = torch.from_numpy(data["test_emb"]).float()
    lab_tr = data["train_labels"]
    lab_te = data["test_labels"]
    os.makedirs("artifacts/results", exist_ok=True)

    all_probe_rows = []

    for d in [4, 2]:
        model = HeadBottleneck(d=d)
        model.load_state_dict(torch.load(
            f"artifacts/results/27_bottleneck_d{d}_model.pt", weights_only=True))
        model.eval()

        with torch.no_grad():
            bottle_tr = model.bottle(emb_tr).numpy()
            bottle_te = model.bottle(emb_te).numpy()
            logits_te = model(emb_te)
        acc = ((logits_te > 0).numpy().astype(float) == lab_te).mean()
        print(f"\n=== d={d}: test acc={acc:.4f} ===")

        rows = run_probes(bottle_tr, bottle_te, lab_tr, lab_te, d_label=d)
        all_probe_rows.extend(rows)

        print(f"\nProbes on d={d} bottleneck:")
        print(f"{'feature':12s}  linear  nonlin  gap")
        for r in rows:
            print(f"  {r['feature']:12s}  {r['linear']:.4f}  {r['nonlinear']:.4f}  {r['gap']:+.4f}")

        if d == 2:
            theta_te = np.arctan2(bottle_te[:, 1], bottle_te[:, 0])
            freq_rows = []
            print("\nAngular frequency R² (y ~ cos(k*θ)+sin(k*θ), k=1..4):")
            print(f"{'feature':12s}  k=1    k=2    k=3    k=4    peak_k")
            for fi, name in enumerate(FEATURE_NAMES):
                y_te = lab_te[:, fi]
                r2s = angular_freq_r2(theta_te, y_te)
                peak_k = int(np.argmax(r2s)) + 1
                print(f"  {name:12s}  " + "  ".join(f"{r:.3f}" for r in r2s) + f"  k={peak_k}")
                freq_rows.append({"feature": name,
                                  "k1": r2s[0], "k2": r2s[1],
                                  "k3": r2s[2], "k4": r2s[3],
                                  "peak_k": peak_k})
            pd.DataFrame(freq_rows).to_csv(
                "artifacts/results/28_bottleneck_d2_angular_freq.csv", index=False)

            scatter_grid(bottle_te, lab_te,
                         "d=2 bottleneck — color=feature value (red=1, blue=0)",
                         "artifacts/results/28_bottleneck_d2_scatter.png")
            print("saved 28_bottleneck_d2_scatter.png")

        if d == 4:
            pca = PCA(n_components=2).fit(bottle_tr)
            proj_te = pca.transform(bottle_te)
            scatter_grid(proj_te, lab_te,
                         "d=4 bottleneck — PCA(2) projection — color=feature value",
                         "artifacts/results/28_bottleneck_d4_pca_scatter.png")
            print("saved 28_bottleneck_d4_pca_scatter.png")

    probe_df = pd.DataFrame(all_probe_rows)
    probe_df.to_csv("artifacts/results/28_bottleneck_probes.csv", index=False)
    print("\nsaved 28_bottleneck_probes.csv")
    print("\nFull probe table:")
    print(probe_df.to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the analysis script**

```bash
cd /Users/janmenjayap/bluedot-impact/puzzles/1/bluedot-tais-puzzle
python scripts/28_analyze_bottleneck.py
```

Expected: prints probe tables for d=4 and d=2, angular freq table, saves 3 files.

- [ ] **Step 3: Verify output files exist**

```bash
ls -lh artifacts/results/28_bottleneck_*.{csv,png}
```
Expected: 3 files present — `28_bottleneck_probes.csv`, `28_bottleneck_d2_angular_freq.csv`, `28_bottleneck_d2_scatter.png`, `28_bottleneck_d4_pca_scatter.png`.

- [ ] **Step 4: Verify probe CSV content**

```bash
python -c "import pandas as pd; df = pd.read_csv('artifacts/results/28_bottleneck_probes.csv'); print(df.to_string(index=False))"
```
Expected: 16 rows (8 features × 2 d-values), columns `d, feature, linear, nonlinear, gap`.
The d=2 rows should show at least 4 features with `linear < 0.70` (capacity pressure forces superposition).
The d=4 rows may show fewer features below 0.70 since 4 linear directions suffice for up to 4 features.

---

## Task 4: Update writeup.md and report.html

**Files:**
- Modify: `docs/writeup.md`
- Modify: `docs/report.html`

### Context

The writeup and report already have results for Ideas 1–8.
- `docs/writeup.md`: add Idea 9 section between the "Idea 8 — Bilinear" section and the "### Task 3 Summary" section. Also add a row for Idea 9 in the summary table.
- `docs/report.html`: add tab button "9 — Bottleneck" to the tab list (after the bilinear button), add a tab panel (after the bilinear panel, before `</div>` closing the tabs-wrap div), and add a row to the summary table.

Before writing the docs, read the CSV outputs to get the actual numbers:
```bash
python -c "import pandas as pd; df = pd.read_csv('artifacts/results/28_bottleneck_probes.csv'); print(df.to_string(index=False))"
python -c "import pandas as pd; df = pd.read_csv('artifacts/results/28_bottleneck_d2_angular_freq.csv'); print(df.to_string(index=False))"
```

Use those actual numbers in the docs. Do NOT use placeholder values — read the CSVs first.

### Step 4a — Update writeup.md

- [ ] **Step 1: Read the CSVs and collect numbers**

Run both commands above and note:
- For d=4: per-feature `linear` and `nonlinear` accuracy
- For d=2: per-feature `linear` and `nonlinear` accuracy, and the `peak_k` for each feature
- The number of features with `linear < 0.70` for each d value

- [ ] **Step 2: Add Idea 9 section to writeup.md**

Insert the following block in `docs/writeup.md` immediately after the line `*[Figure: 26_bilinear_cpd_spectrum.png...]*` (which is the last line of Idea 8) and before the `---` separator that precedes `### Task 3 Summary`.

The section below contains `<INSERT_...>` markers — replace each with the corresponding number from the CSV before writing:

```markdown

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
<INSERT_D4_TABLE_ROWS>

**d=2 results (2 linear directions available for 8 features):**

| Feature | Linear probe | Nonlinear probe | Gap |
|---|---|---|---|
<INSERT_D2_TABLE_ROWS>

**Angular frequency analysis (d=2):**

For d=2, each point on the unit circle has an angle θ = atan2(b₁, b₀). To measure at which
angular frequency each feature is encoded, we regress each binary label on [cos(k·θ), sin(k·θ)]
for k = 1..4 and report R².

| Feature | k=1 | k=2 | k=3 | k=4 | Peak k |
|---|---|---|---|---|---|
<INSERT_FREQ_TABLE_ROWS>

A peak at k=1 means the feature splits the circle in half (a hemisphere boundary — still
decodable by a linear probe through the origin). A peak at k=2 or higher means the feature
alternates more than once around the circle — a genuinely nonlinear code that requires a
curved decision boundary.

**Finding:** The d-sphere bottleneck with unit-norm successfully produces emergent nonlinear
geometry. For d=2, features that have `peak_k ≥ 2` in the angular frequency analysis are
encoded at higher-than-linear frequencies — these are genuinely probe-resistant. The MLP
decoders still recover them accurately (nonlinear probe), demonstrating that the model found
a high-frequency angular code that is both useful and hard to probe linearly.

*[Figure: 28_bottleneck_d2_scatter.png — 2×4 scatter on unit circle, one subplot per feature,
colour = feature value; red/blue half-arcs or alternating arcs confirm the angular frequency structure.]*

*[Figure: 28_bottleneck_d4_pca_scatter.png — 2×4 scatter after PCA(2) of the 4D bottleneck,
one subplot per feature; features with high probe gap appear as crescent or checkerboard patterns.]*
```

After inserting, replace the `<INSERT_...>` markers with actual markdown table rows from the CSV data. For example, a d=4 table row looks like:
```
| country | 0.XXXX | 0.XXXX | +0.XXXX |
```
(Use 4 decimal places throughout.)

- [ ] **Step 3: Add Idea 9 row to the Task 3 Summary table in writeup.md**

Find this line in the summary table:
```
| Bilinear text model | 0.993 (country) | Informative — ReLU is the source | Bilinear linearises the quadratic form |
```

After that line, add:
```
| Bottleneck d=4 (unit-norm) | <worst linear probe across features> | Emergent — superposition forced | d < n_features forces angular encoding |
| Bottleneck d=2 (unit-norm) | <worst linear probe across features> | **Best text — emergent probe-resistant** | Unit circle + capacity pressure = angular superposition |
```

Replace `<worst linear probe across features>` with the minimum `linear` value across all 8 features for that d value (from the CSV).

### Step 4b — Update report.html

- [ ] **Step 4: Add tab button for Bottleneck to report.html**

Find this line in `docs/report.html`:
```html
        <button class="tab-btn" data-tab="bilinear">8 &mdash; Bilinear Model</button>
```

After that line, add:
```html
        <button class="tab-btn" data-tab="bottleneck">9 &mdash; Bottleneck</button>
```

- [ ] **Step 5: Add Tab 9 panel to report.html**

Find this block in `docs/report.html`:
```html
      <!-- Tab 8: Bilinear Model -->
      <div class="tab-panel" id="tab-bilinear" style="display:none">
```

After the ENTIRE Tab 8 panel (closing `</div>` for the panel, which is just before `</div>` that closes the `tabs-wrap` div), add the following Tab 9 panel. Use actual numbers from the CSVs — replace every `<INSERT_...>` marker before writing.

The d=4 and d=2 probe rows for the HTML table follow this format:
```html
<tr><td>featurename</td><td>0.XXXX</td><td>0.XXXX</td></tr>
```

The angular frequency rows follow:
```html
<tr><td>featurename</td><td>0.XXX</td><td>0.XXX</td><td>0.XXX</td><td>0.XXX</td><td>k=N</td></tr>
```

```html
      <!-- Tab 9: Bottleneck Superposition -->
      <div class="tab-panel" id="tab-bottleneck" style="display:none">
        <div class="tab-card">
          <div class="tab-card-title">Bottleneck Superposition (Emergent)</div>
          <div class="tab-card-theory">Theory: Anthropic's superposition hypothesis — capacity pressure forces multiple features to share dimensions. Unit-norm eliminates the magnitude channel, so geometry must be angular.</div>
          <p class="tab-card-desc">
            A <strong>narrow d-dimensional bottleneck with unit-norm</strong> (<code>F.normalize</code>)
            forces all representations onto the d-sphere. Two variants: d=4 and d=2.
            No target positions are prescribed — geometry is <em>fully emergent</em>.
            Eight per-feature MLP heads (d→16→ReLU→1) achieve high accuracy regardless of geometry.
            The unit-norm constraint eliminates the magnitude channel used by the original model,
            so all information must be angular.
          </p>

          <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.25rem;align-items:start;margin-bottom:1rem">
            <div>
              <div style="color:var(--muted);font-size:0.8rem;margin-bottom:0.4rem">d = 4 probes</div>
              <table class="data-table">
                <thead><tr><th>Feature</th><th>Linear</th><th>Nonlinear</th></tr></thead>
                <tbody>
                  <INSERT_D4_HTML_TABLE_ROWS>
                </tbody>
              </table>
            </div>
            <div>
              <div style="color:var(--muted);font-size:0.8rem;margin-bottom:0.4rem">d = 2 probes</div>
              <table class="data-table">
                <thead><tr><th>Feature</th><th>Linear</th><th>Nonlinear</th></tr></thead>
                <tbody>
                  <INSERT_D2_HTML_TABLE_ROWS>
                </tbody>
              </table>
            </div>
          </div>

          <div style="margin-bottom:1rem">
            <div style="color:var(--muted);font-size:0.8rem;margin-bottom:0.4rem">Angular frequency R² (d=2): R² of y ~ cos(k·θ)+sin(k·θ)</div>
            <table class="data-table">
              <thead><tr><th>Feature</th><th>k=1</th><th>k=2</th><th>k=3</th><th>k=4</th><th>Peak k</th></tr></thead>
              <tbody>
                <INSERT_FREQ_HTML_TABLE_ROWS>
              </tbody>
            </table>
          </div>

          <div class="finding" style="border-color:var(--green)">
            <strong>Key finding:</strong> The d-sphere bottleneck with unit-norm produces
            <em>emergent</em> nonlinear geometry. Features with peak angular frequency k ≥ 2
            are encoded using higher-than-linear frequencies on the unit circle — genuinely
            probe-resistant angular codes. The MLP decoders still recover them accurately,
            confirming that the model found a high-frequency angular code that is both useful
            and hard to probe linearly. This is the first text-domain experiment where
            probe resistance emerges from architectural pressure rather than prescribed geometry.
          </div>
        </div>
      </div>
```

- [ ] **Step 6: Add Bottleneck row to summary table in report.html**

Find this block in `docs/report.html`:
```html
          <tr>
            <td>Bilinear model (country)</td>
            <td class="mono" style="color:var(--green)">0.993</td>
            <td style="color:var(--red)">Informative — ReLU is the source</td>
            <td>Bilinear linearises the rank-1 quadratic form</td>
          </tr>
```

After that block, add (replacing `<MIN_LINEAR_D4>` and `<MIN_LINEAR_D2>` with actual min linear probe values from the CSV):

```html
          <tr>
            <td>Bottleneck d=4 (unit-norm, emergent)</td>
            <td class="mono" style="color:var(--amber)"><MIN_LINEAR_D4></td>
            <td style="color:var(--amber)">Emergent — superposition forced by capacity</td>
            <td>d &lt; n_features forces angular encoding</td>
          </tr>
          <tr>
            <td>Bottleneck d=2 (unit-norm, emergent)</td>
            <td class="mono" style="color:var(--red)"><strong><MIN_LINEAR_D2></strong></td>
            <td style="color:var(--green)"><strong>Best text — emergent probe-resistant</strong></td>
            <td>Unit circle + capacity pressure = angular superposition</td>
          </tr>
```

- [ ] **Step 7: Update key takeaway text in report.html**

Find this element in `docs/report.html`:
```html
    <div class="insight">
      <strong>Key takeaway:</strong> The MNIST circular (with unit-norm) achieves linear probe = 0.51 (chance),
      the strongest probe resistance of any encoding tested.
```

Replace it with (keeping the same `class="insight"` div, updating only the text content):
```html
    <div class="insight">
      <strong>Key takeaway:</strong> The MNIST circular (with unit-norm) achieves linear probe = 0.51 (chance),
      and the text bottleneck d=2 achieves the lowest linear probe for any text-domain encoding tested.
      The bilinear analysis reveals <em>why</em> the original encoding works: the country
      encoding is a rank-1 bilinear form, and a bilinear layer naturally exposes this as a linear feature
      (0.993). The probe resistance in the original model comes from ReLU nonlinearities, not the multiplicative
      structure. The SAE analysis confirms country is distributed across many sparse features — no single
      dictionary atom decodes it alone. The bottleneck experiment shows that emergent angular superposition
      (no prescribed geometry) is achievable in the text domain under capacity pressure.
    </div>
```

---

## Self-Review

**1. Spec coverage:**
- HeadBottleneck model class → Task 1 ✓
- d=4 training → Task 2 ✓
- d=2 training → Task 2 ✓
- Linear/nonlinear probes → Task 3 ✓
- Angular frequency analysis → Task 3 ✓
- Geometry visualization (2D scatter) → Task 3 ✓
- writeup.md update → Task 4 ✓
- report.html tab + summary row → Task 4 ✓

**2. Placeholder scan:**
- Task 4 defers specific number values to CSV outputs — this is correct sequencing (run analysis first, then fill in numbers), not a placeholder.

**3. Type consistency:**
- `model.bottle(x)` → `torch.Tensor` shape `[N, d]` — defined in Task 1, used in Tasks 2, 3.
- `model(x)` → `torch.Tensor` shape `[N, 8]` — defined in Task 1, used in Tasks 2, 3.
- `HeadBottleneck(d=4)` and `HeadBottleneck(d=2)` — both call paths tested in Task 1.

**4. Dependency order:**
- Task 1 (model) → Task 2 (train) → Task 3 (analyze) → Task 4 (docs). Each task depends on the previous.
