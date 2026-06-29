# Binary Probe-Resistant Codes (Capstone A/α/γ) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three architecturally-induced binary probe-resistant encodings (Experiment A square wave, α Fourier comb, γ linked rings) to the Task 3 study, establishing the *constructible-vs-emergent gap* against the emergent Idea 9 bottleneck.

**Architecture:** Each experiment adds one model class to `src/puzzle/models_t3.py`, reuses cached embeddings from `artifacts/activations/acts.npz` (no re-encoding), and ships a train script + analyze script under `scripts/29..34`. Shared geometry helpers (`angular_freq_r2` lifted to `max_k=5`, `arc_occupancy`, `disc_crossing_count`, `linked_rings_loss`) move into `src/puzzle/geometry.py`. Probes follow the established `LogisticRegression(max_iter=2000)` / `MLPClassifier((32,), max_iter=2000, random_state=0)` pattern with a train-only `StandardScaler`. Results land in `artifacts/results/29..34_*` and are written up as Ideas 10–12 in `docs/writeup.md` and `docs/report.html`.

**Tech Stack:** Python 3.14 (conda env `bluedot-impact-puzzle-1`), PyTorch, scikit-learn, pandas, matplotlib (Agg). Run everything as `PYTHONPATH=. <env-python> ...`.

## Global Constraints

- Feature indices (verbatim from spec): `country`=5, `food`=3, `sentiment`=4, `question`=1. Import via `from src.puzzle.data import FEATURE_NAMES`.
- Embeddings: use `artifacts/activations/acts.npz` keys `train_emb`/`test_emb` (384-d), `train_labels`/`test_labels` (8 binary). No re-encoding of text.
- Unit-norm circles via `torch.nn.functional.normalize(..., dim=1)`.
- Probe convention: `StandardScaler().fit(Xtr)` (train only), linear = `LogisticRegression(max_iter=2000)`, nonlinear = `MLPClassifier((32,), max_iter=2000, random_state=0)`.
- Every experiment requires a **co-encoded spreader** that populates the manifold while the binary target interleaves over it; collapse of the target into a single arc/point is a reportable failure, not a silent one.
- A/α/γ are **architecturally induced / constructive** — never described as "emergent". Idea 9 (d=2 bottleneck) is the emergence control.
- Training mirrors `scripts/27`: `Adam(lr=1e-3)`, `CosineAnnealingLR`, `BATCH=512`, BCE-with-logits, `N_EPOCHS=600`, seeds set for reproducibility.
- Highest existing script is 28; 29–34 are free. Full test suite must stay green (currently 19/19).
- Run tests/scripts with `PY=/Users/janmenjayap/.miniconda3/envs/bluedot-impact-puzzle-1/bin/python` and `PYTHONPATH=.`.

---

### Task 1: Geometry helpers (lift `angular_freq_r2`, add occupancy / linking / ring loss)

**Files:**
- Modify: `src/puzzle/geometry.py`
- Modify: `scripts/28_analyze_bottleneck.py` (import lifted `angular_freq_r2`)
- Test: `tests/test_models_t3.py`

**Interfaces:**
- Produces:
  - `angular_freq_r2(theta: np.ndarray, y: np.ndarray, max_k: int = 5) -> list[float]` — R² of binary `y` regressed on `[cos(kθ), sin(kθ)]` for k=1..max_k.
  - `arc_occupancy(theta: np.ndarray, mask: np.ndarray, n_bins: int = 12) -> int` — number of distinct angular bins occupied by points where `mask` is True.
  - `disc_crossing_count(points_xyz: np.ndarray) -> int` — net signed z-crossings of a 3-D point cloud through the unit disc in the z=0 plane centred at the origin, walking the cloud ordered by within-ring angle `atan2(z, x-1)`.
  - `linked_rings_loss(bottle: torch.Tensor, country: torch.Tensor, food: torch.Tensor) -> torch.Tensor` — MSE of each sample to its target ring point: country=0 → ring A `(cosφ, sinφ, 0)`, country=1 → ring B `(1+cosφ, 0, sinφ)`, with `φ = food·π` (the spreader).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_models_t3.py`:

```python
import numpy as np
from src.puzzle.geometry import (
    angular_freq_r2, arc_occupancy, disc_crossing_count, linked_rings_loss,
)


def test_angular_freq_r2_detects_k2():
    rng = np.random.default_rng(0)
    theta = rng.uniform(-np.pi, np.pi, 2000)
    y = (np.cos(2 * theta) > 0).astype(int)   # pure k=2 structure
    r2 = angular_freq_r2(theta, y, max_k=5)
    assert len(r2) == 5
    assert int(np.argmax(r2)) + 1 == 2        # peak at k=2


def test_arc_occupancy_counts_bins():
    theta = np.array([0.0, np.pi])            # two opposite arcs
    mask = np.array([True, True])
    assert arc_occupancy(theta, mask, n_bins=12) == 2
    assert arc_occupancy(theta, np.array([False, False])) == 0


def test_disc_crossing_linked_rings_is_one():
    # Ring B sampled densely: (1+cos ψ, 0, sin ψ) threads the z=0 unit disc once.
    psi = np.linspace(-np.pi, np.pi, 400, endpoint=False)
    ringB = np.column_stack([1 + np.cos(psi), np.zeros_like(psi), np.sin(psi)])
    assert disc_crossing_count(ringB) == 1


def test_linked_rings_loss_zero_at_targets():
    import torch
    country = torch.tensor([0, 1])
    food = torch.tensor([0, 0])               # φ = 0
    targetA = [1.0, 0.0, 0.0]                  # ring A, φ=0
    targetB = [2.0, 0.0, 0.0]                  # ring B, φ=0
    bottle = torch.tensor([targetA, targetB])
    loss = linked_rings_loss(bottle, country, food)
    assert loss.item() < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. $PY -m pytest tests/test_models_t3.py -k "angular_freq or arc_occupancy or disc_crossing or linked_rings" -v`
Expected: FAIL with `ImportError: cannot import name 'angular_freq_r2'` (etc.)

- [ ] **Step 3: Implement the helpers**

Append to `src/puzzle/geometry.py` (it already imports `torch`, `F`, `math`; add `import numpy as np` and `from sklearn.linear_model import LinearRegression` at the top):

```python
def angular_freq_r2(theta, y, max_k: int = 5):
    """R² of binary label y regressed on [cos(k*theta), sin(k*theta)] for k=1..max_k."""
    scores = []
    y_f = np.asarray(y, dtype=float)
    ss_tot = np.sum((y_f - y_f.mean()) ** 2) + 1e-10
    for k in range(1, max_k + 1):
        X_feat = np.column_stack([np.cos(k * theta), np.sin(k * theta)])
        y_pred = LinearRegression().fit(X_feat, y_f).predict(X_feat)
        ss_res = np.sum((y_f - y_pred) ** 2)
        scores.append(float(1.0 - ss_res / ss_tot))
    return scores


def arc_occupancy(theta, mask, n_bins: int = 12) -> int:
    """Number of distinct equal-width angular bins occupied by points where mask is True."""
    t = np.asarray(theta)[np.asarray(mask, dtype=bool)]
    if len(t) == 0:
        return 0
    bins = ((t % (2 * np.pi)) / (2 * np.pi) * n_bins).astype(int) % n_bins
    return int(len(np.unique(bins)))


def disc_crossing_count(points_xyz) -> int:
    """Net signed z-crossings of a 3-D cloud through the z=0 unit disc centred at origin.
    Walk the cloud ordered by within-ring angle atan2(z, x-1) (ring B's natural angle);
    count signed z sign-changes that occur while the (x,y) radius is < 1 (inside disc A).
    A clean linking number 1 gives exactly one net crossing.
    """
    p = np.asarray(points_xyz, float)
    psi = np.arctan2(p[:, 2], p[:, 0] - 1.0)
    order = np.argsort(psi)
    p = p[order]
    z = p[:, 2]
    r_xy = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
    net = 0
    n = len(p)
    for i in range(n):
        j = (i + 1) % n
        if z[i] == 0:
            continue
        if np.sign(z[j]) != np.sign(z[i]):           # z crosses zero between i and j
            if 0.5 * (r_xy[i] + r_xy[j]) < 1.0:      # crossing point lies inside disc A
                net += int(np.sign(z[j] - z[i]))
    return abs(net)


def linked_rings_loss(bottle: torch.Tensor, country: torch.Tensor,
                      food: torch.Tensor) -> torch.Tensor:
    """Distance of each sample to its target ring point.
    country=0 → ring A (unit circle, xy-plane, centre origin);
    country=1 → ring B (unit circle, xz-plane, centre (1,0,0)).
    food sets the within-ring angle φ = food·π (the mandatory spreader).
    """
    phi = food.float() * math.pi
    zeros = torch.zeros_like(phi)
    tA = torch.stack([torch.cos(phi), torch.sin(phi), zeros], dim=1)
    tB = torch.stack([1 + torch.cos(phi), zeros, torch.sin(phi)], dim=1)
    target = torch.where(country.bool().unsqueeze(1), tB, tA)
    return F.mse_loss(bottle, target)
```

- [ ] **Step 4: Update `scripts/28_analyze_bottleneck.py` to import the lifted helper**

Replace the local `def angular_freq_r2(...)` definition (the whole function block) with an import. Change the import block to add:

```python
from src.puzzle.geometry import angular_freq_r2
```

and delete the in-file `def angular_freq_r2(theta, y, max_k=4):` function. The call site `angular_freq_r2(theta_te, y_te)` keeps working (default is now `max_k=5`, but script 28 only reads `r2s[0:4]` into k1..k4, so its CSV is unchanged). Verify by re-running script 28 and diffing the CSV.

- [ ] **Step 5: Run tests + script 28 to verify pass + no regression**

Run: `PYTHONPATH=. $PY -m pytest tests/test_models_t3.py -v`
Expected: all PASS (existing 8 T3 tests + 4 new geometry tests).

Run: `PYTHONPATH=. $PY scripts/28_analyze_bottleneck.py && head -3 artifacts/results/28_bottleneck_d2_angular_freq.csv`
Expected: runs clean; `country` row still shows k2≈0.486 (CSV unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/puzzle/geometry.py scripts/28_analyze_bottleneck.py tests/test_models_t3.py
git commit -m "feat: lift angular_freq_r2 (max_k=5) + add occupancy/linking/ring-loss helpers"
```

---

### Task 2: `HeadSquareWave` model (Experiment A)

**Files:**
- Modify: `src/puzzle/models_t3.py`
- Test: `tests/test_models_t3.py`

**Interfaces:**
- Produces: `HeadSquareWave(k: int = 2, country_idx: int = 5)` with `.circle(x) -> [N,2]` (unit-norm) and `.forward(x) -> [N,8]` logits, where `country_idx`'s logit is the harmonic readout `α·cos(kθ)+β·sin(kθ)+b` and the other 7 are free MLP heads `2→16→ReLU→1`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_models_t3.py` (extend the existing models import to include the new class):

```python
def test_head_squarewave_shapes():
    from src.puzzle.models_t3 import HeadSquareWave
    model = HeadSquareWave(k=2)
    out = model(_emb())
    assert out.shape == (N, 8)
    circ = model.circle(_emb())
    assert circ.shape == (N, 2)
    assert torch.allclose(circ.norm(dim=1), torch.ones(N), atol=1e-5)


def test_head_squarewave_country_uses_harmonic():
    from src.puzzle.models_t3 import HeadSquareWave
    model = HeadSquareWave(k=3, country_idx=5)
    assert model.k == 3
    # harmonic readout consumes a 2-vector [cos(kθ), sin(kθ)]
    assert model.harm.in_features == 2 and model.harm.out_features == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. $PY -m pytest tests/test_models_t3.py -k squarewave -v`
Expected: FAIL with `ImportError: cannot import name 'HeadSquareWave'`.

- [ ] **Step 3: Implement the model**

Append to `src/puzzle/models_t3.py`:

```python
class HeadSquareWave(nn.Module):
    """Experiment A: a binary feature constrained to the k-th harmonic of a 2-D unit circle.
    country (country_idx) is read out as α·cos(kθ)+β·sin(kθ)+b; the other 7 features use
    free MLP heads (2→16→ReLU→1) and act as the spreader that populates θ.
    """
    def __init__(self, k: int = 2, country_idx: int = 5):
        super().__init__()
        self.k = k
        self.country_idx = country_idx
        self.enc = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
        )
        self.proj = nn.Linear(64, 2)
        self.harm = nn.Linear(2, 1)            # α·cos(kθ) + β·sin(kθ) + b
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
            for _ in range(8)
        ])

    def circle(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.enc(x)), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.circle(x)
        theta = torch.atan2(b[:, 1], b[:, 0])
        harm_feat = torch.stack([torch.cos(self.k * theta),
                                 torch.sin(self.k * theta)], dim=1)
        country_logit = self.harm(harm_feat)
        cols = []
        for fi in range(8):
            cols.append(country_logit if fi == self.country_idx else self.heads[fi](b))
        return torch.cat(cols, dim=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. $PY -m pytest tests/test_models_t3.py -k squarewave -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/puzzle/models_t3.py tests/test_models_t3.py
git commit -m "feat: HeadSquareWave model for Experiment A (k-th harmonic country readout)"
```

---

### Task 3: Experiment A train + analyze (scripts 29, 30)

**Files:**
- Create: `scripts/29_train_squarewave.py`
- Create: `scripts/30_analyze_squarewave.py`
- Writes: `artifacts/results/29_squarewave_k{1..5}_model.pt`, `artifacts/results/30_squarewave_probes.csv`, `artifacts/results/30_squarewave_curve.png`

**Interfaces:**
- Consumes: `HeadSquareWave`, `angular_freq_r2`, `arc_occupancy`, `FEATURE_NAMES`.
- Produces: per-k CSV rows `k, country_linear, country_nonlinear, freq_peak_k, country1_arcs, overall_acc`.

- [ ] **Step 1: Write the train script**

Create `scripts/29_train_squarewave.py`:

```python
# scripts/29_train_squarewave.py
# Experiment A — square wave: country constrained to the k-th harmonic of a 2-D circle.
# Sweep k=1..5; the other 7 co-trained features are the spreader. BCE on all 8 outputs.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.puzzle.models_t3 import HeadSquareWave

N_EPOCHS, LR, BATCH = 600, 1e-3, 512
KS = [1, 2, 3, 4, 5]


def train_one(k, emb_tr, lab_tr, emb_te, lab_te):
    torch.manual_seed(0)
    model = HeadSquareWave(k=k)
    opt = Adam(model.parameters(), lr=LR)
    sched = CosineAnnealingLR(opt, N_EPOCHS)
    model.train()
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(len(emb_tr))
        for s in range(0, len(emb_tr), BATCH):
            idx = perm[s:s + BATCH]
            loss = F.binary_cross_entropy_with_logits(model(emb_tr[idx]), lab_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch + 1) % 150 == 0:
            model.eval()
            with torch.no_grad():
                acc = ((model(emb_te) > 0).float() == lab_te).float().mean()
            print(f"k={k} epoch {epoch+1:3d}: acc={acc:.4f}")
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
    for k in KS:
        print(f"\n=== Experiment A: k={k} ===")
        model = train_one(k, emb_tr, lab_tr, emb_te, lab_te)
        path = f"artifacts/results/29_squarewave_k{k}_model.pt"
        torch.save(model.state_dict(), path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the analyze script**

Create `scripts/30_analyze_squarewave.py`:

```python
# scripts/30_analyze_squarewave.py
# Experiment A analysis: per-k linear/nonlinear country probe on the 2-D circle,
# angular-frequency peak, country=1 arc occupancy, overall 8-feature accuracy.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadSquareWave
from src.puzzle.geometry import angular_freq_r2, arc_occupancy

KS = [1, 2, 3, 4, 5]
CI = FEATURE_NAMES.index("country")


def main():
    data = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(data["train_emb"]).float()
    emb_te = torch.from_numpy(data["test_emb"]).float()
    lab_tr, lab_te = data["train_labels"], data["test_labels"]
    rows = []
    for k in KS:
        model = HeadSquareWave(k=k)
        model.load_state_dict(torch.load(
            f"artifacts/results/29_squarewave_k{k}_model.pt", weights_only=True))
        model.eval()
        with torch.no_grad():
            circ_tr = model.circle(emb_tr).numpy()
            circ_te = model.circle(emb_te).numpy()
            overall = ((model(emb_te) > 0).numpy().astype(float) == lab_te).mean()
        sc = StandardScaler().fit(circ_tr)
        Xtr, Xte = sc.transform(circ_tr), sc.transform(circ_te)
        ytr, yte = lab_tr[:, CI], lab_te[:, CI]
        lin = LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte)
        nl = MLPClassifier((32,), max_iter=2000, random_state=0).fit(Xtr, ytr).score(Xte, yte)
        theta = np.arctan2(circ_te[:, 1], circ_te[:, 0])
        r2 = angular_freq_r2(theta, yte, max_k=5)
        arcs = arc_occupancy(theta, yte == 1, n_bins=12)
        rows.append({"k": k, "country_linear": lin, "country_nonlinear": nl,
                     "freq_peak_k": int(np.argmax(r2)) + 1,
                     "country1_arcs": arcs, "overall_acc": float(overall)})
        print(f"k={k}: lin={lin:.3f} nonlin={nl:.3f} peak_k={int(np.argmax(r2))+1} "
              f"arcs={arcs} overall={overall:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv("artifacts/results/30_squarewave_probes.csv", index=False)
    print("\nsaved 30_squarewave_probes.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["k"], df["country_linear"], "o-", label="linear probe")
    ax.plot(df["k"], df["country_nonlinear"], "s-", label="nonlinear probe")
    ax.axhline(0.5, ls="--", c="grey", lw=0.8)
    ax.set_xlabel("harmonic k"); ax.set_ylabel("country probe accuracy")
    ax.set_title("Experiment A — square wave: probe accuracy vs harmonic k")
    ax.legend(); fig.tight_layout()
    fig.savefig("artifacts/results/30_squarewave_curve.png", dpi=150)
    print("saved 30_squarewave_curve.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run train + analyze**

Run: `PYTHONPATH=. $PY scripts/29_train_squarewave.py`
Expected: prints per-k training accuracy, saves 5 `.pt` files.

Run: `PYTHONPATH=. $PY scripts/30_analyze_squarewave.py`
Expected: prints per-k probe table; saves CSV + curve PNG. Per success criteria, **k=2** should show `country_linear ≤ 0.55`, `country_nonlinear ≥ 0.90`, `freq_peak_k = 2`, `country1_arcs ≥ 2`. (k=1 high linear; k=4,5 may hit a capacity ceiling — record as found.)

- [ ] **Step 4: Verify the headline criterion**

Run: `PYTHONPATH=. $PY -c "import pandas as pd; d=pd.read_csv('artifacts/results/30_squarewave_probes.csv'); r=d[d.k==2].iloc[0]; print(r.to_dict()); assert r.country_linear<=0.60 and r.country_nonlinear>=0.85 and r.country1_arcs>=2, 'k=2 interleaving not realized'"`
Expected: prints the k=2 row and exits 0. If it fails (collapse), that is a reportable ceiling — record actual numbers and continue; do not fake the gate.

- [ ] **Step 5: Commit**

```bash
git add scripts/29_train_squarewave.py scripts/30_analyze_squarewave.py artifacts/results/29_squarewave_k*_model.pt artifacts/results/30_squarewave_*.csv artifacts/results/30_squarewave_curve.png
git commit -m "feat: Experiment A (square wave) train + analyze, k=1..5 sweep"
```

---

### Task 4: `HeadFourierComb` model (Experiment α)

**Files:**
- Modify: `src/puzzle/models_t3.py`
- Test: `tests/test_models_t3.py`

**Interfaces:**
- Produces: `HeadFourierComb(harmonics: dict | None = None)` (default `{5: 2, 3: 3, 4: 4}` = country→k2, food→k3, sentiment→k4) with `.circle(x) -> [N,2]` and `.forward(x) -> [N,8]`; multiplexed features use per-feature harmonic readouts, the other five use free MLP heads.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_models_t3.py`:

```python
def test_head_fourier_comb_shapes():
    from src.puzzle.models_t3 import HeadFourierComb
    model = HeadFourierComb()
    out = model(_emb())
    assert out.shape == (N, 8)
    assert model.harmonics == {5: 2, 3: 3, 4: 4}
    circ = model.circle(_emb())
    assert torch.allclose(circ.norm(dim=1), torch.ones(N), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. $PY -m pytest tests/test_models_t3.py -k fourier -v`
Expected: FAIL with `ImportError: cannot import name 'HeadFourierComb'`.

- [ ] **Step 3: Implement the model**

Append to `src/puzzle/models_t3.py`:

```python
class HeadFourierComb(nn.Module):
    """Experiment α: multiplex several binary features onto one 2-D circle, each on a
    distinct harmonic (default country→k=2, food→k=3, sentiment→k=4). Multiplexing is
    itself the spreader. Remaining features use free MLP heads (2→16→ReLU→1).
    """
    def __init__(self, harmonics=None):
        super().__init__()
        self.harmonics = harmonics or {5: 2, 3: 3, 4: 4}
        self.enc = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
        )
        self.proj = nn.Linear(64, 2)
        self.harm = nn.ModuleDict({str(fi): nn.Linear(2, 1) for fi in self.harmonics})
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
            for _ in range(8)
        ])

    def circle(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.enc(x)), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.circle(x)
        theta = torch.atan2(b[:, 1], b[:, 0])
        cols = []
        for fi in range(8):
            if fi in self.harmonics:
                k = self.harmonics[fi]
                hf = torch.stack([torch.cos(k * theta), torch.sin(k * theta)], dim=1)
                cols.append(self.harm[str(fi)](hf))
            else:
                cols.append(self.heads[fi](b))
        return torch.cat(cols, dim=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. $PY -m pytest tests/test_models_t3.py -k fourier -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/puzzle/models_t3.py tests/test_models_t3.py
git commit -m "feat: HeadFourierComb model for Experiment alpha (multiplexed harmonics)"
```

---

### Task 5: Experiment α train + analyze (scripts 31, 32)

**Files:**
- Create: `scripts/31_train_fourier_comb.py`
- Create: `scripts/32_analyze_fourier_comb.py`
- Writes: `artifacts/results/31_fourier_comb_model.pt`, `artifacts/results/32_fourier_probes.csv`, `artifacts/results/32_harmonic_confusion.csv`, `artifacts/results/32_harmonic_confusion.png`

**Interfaces:**
- Consumes: `HeadFourierComb`, `angular_freq_r2`, `FEATURE_NAMES`.
- Produces: per-multiplexed-feature linear/nonlinear probe rows; an 8×5 harmonic-confusion matrix `R²(feature i | harmonic k)`.

- [ ] **Step 1: Write the train script**

Create `scripts/31_train_fourier_comb.py`:

```python
# scripts/31_train_fourier_comb.py
# Experiment α — Fourier comb: multiplex country(k=2), food(k=3), sentiment(k=4)
# onto one 2-D circle; remaining features use free heads. BCE on all 8 outputs.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.puzzle.models_t3 import HeadFourierComb

N_EPOCHS, LR, BATCH = 600, 1e-3, 512


def main():
    data = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(data["train_emb"]).float()
    emb_te = torch.from_numpy(data["test_emb"]).float()
    lab_tr = torch.from_numpy(data["train_labels"]).float()
    lab_te = torch.from_numpy(data["test_labels"]).float()
    os.makedirs("artifacts/results", exist_ok=True)
    torch.manual_seed(0)
    model = HeadFourierComb()
    opt = Adam(model.parameters(), lr=LR)
    sched = CosineAnnealingLR(opt, N_EPOCHS)
    model.train()
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(len(emb_tr))
        for s in range(0, len(emb_tr), BATCH):
            idx = perm[s:s + BATCH]
            loss = F.binary_cross_entropy_with_logits(model(emb_tr[idx]), lab_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch + 1) % 150 == 0:
            model.eval()
            with torch.no_grad():
                acc = ((model(emb_te) > 0).float() == lab_te).float().mean()
            print(f"epoch {epoch+1:3d}: acc={acc:.4f}")
            model.train()
    model.eval()
    torch.save(model.state_dict(), "artifacts/results/31_fourier_comb_model.pt")
    print("saved 31_fourier_comb_model.pt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the analyze script**

Create `scripts/32_analyze_fourier_comb.py`:

```python
# scripts/32_analyze_fourier_comb.py
# Experiment α analysis: per-multiplexed-feature probes + 8×5 harmonic-confusion heatmap.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadFourierComb
from src.puzzle.geometry import angular_freq_r2

MULTIPLEXED = {5: 2, 3: 3, 4: 4}   # feature_idx -> harmonic


def main():
    data = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(data["train_emb"]).float()
    emb_te = torch.from_numpy(data["test_emb"]).float()
    lab_tr, lab_te = data["train_labels"], data["test_labels"]
    model = HeadFourierComb()
    model.load_state_dict(torch.load(
        "artifacts/results/31_fourier_comb_model.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        circ_tr = model.circle(emb_tr).numpy()
        circ_te = model.circle(emb_te).numpy()
        overall = ((model(emb_te) > 0).numpy().astype(float) == lab_te).mean()
    print(f"overall acc={overall:.4f}")

    sc = StandardScaler().fit(circ_tr)
    Xtr, Xte = sc.transform(circ_tr), sc.transform(circ_te)
    rows = []
    for fi, k in MULTIPLEXED.items():
        ytr, yte = lab_tr[:, fi], lab_te[:, fi]
        lin = LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte)
        nl = MLPClassifier((32,), max_iter=2000, random_state=0).fit(Xtr, ytr).score(Xte, yte)
        rows.append({"feature": FEATURE_NAMES[fi], "harmonic_k": k,
                     "linear": lin, "nonlinear": nl})
        print(f"{FEATURE_NAMES[fi]:10s} (k={k}): lin={lin:.3f} nonlin={nl:.3f}")
    pd.DataFrame(rows).to_csv("artifacts/results/32_fourier_probes.csv", index=False)

    theta = np.arctan2(circ_te[:, 1], circ_te[:, 0])
    conf = np.zeros((8, 5))
    for fi in range(8):
        conf[fi] = angular_freq_r2(theta, lab_te[:, fi], max_k=5)
    conf_df = pd.DataFrame(conf, index=FEATURE_NAMES,
                           columns=[f"k{k}" for k in range(1, 6)])
    conf_df.to_csv("artifacts/results/32_harmonic_confusion.csv")
    print("\nHarmonic-confusion R² (feature × k):")
    print(conf_df.round(3).to_string())

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf, cmap="viridis", aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels([f"k={k}" for k in range(1, 6)])
    ax.set_yticks(range(8)); ax.set_yticklabels(FEATURE_NAMES)
    for fi, k in MULTIPLEXED.items():
        ax.add_patch(plt.Rectangle((k - 1 - 0.5, fi - 0.5), 1, 1,
                                   fill=False, edgecolor="red", lw=2))
    fig.colorbar(im, label="R²")
    ax.set_title("Experiment α — harmonic confusion (red = target harmonic)")
    fig.tight_layout()
    fig.savefig("artifacts/results/32_harmonic_confusion.png", dpi=150)
    print("saved 32_harmonic_confusion.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run train + analyze**

Run: `PYTHONPATH=. $PY scripts/31_train_fourier_comb.py`
Then: `PYTHONPATH=. $PY scripts/32_analyze_fourier_comb.py`
Expected: prints probe table + confusion matrix; success target is each multiplexed feature's own-harmonic R² being its row max (predominantly diagonal), multiplexed-feature linear ≤ 0.55 / nonlinear ≥ 0.85. Off-diagonal leakage is reported, not hidden.

- [ ] **Step 4: Commit**

```bash
git add scripts/31_train_fourier_comb.py scripts/32_analyze_fourier_comb.py artifacts/results/31_fourier_comb_model.pt artifacts/results/32_fourier_probes.csv artifacts/results/32_harmonic_confusion.csv artifacts/results/32_harmonic_confusion.png
git commit -m "feat: Experiment alpha (Fourier comb) train + analyze with harmonic-confusion heatmap"
```

---

### Task 6: `HeadLinkedRings` model (Experiment γ)

**Files:**
- Modify: `src/puzzle/models_t3.py`
- Test: `tests/test_models_t3.py`

**Interfaces:**
- Produces: `HeadLinkedRings()` with `.bottle(x) -> [N,3]` (un-normalised 3-D bottleneck) and `.forward(x) -> [N,8]` via per-feature MLP heads `3→16→ReLU→1`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_models_t3.py`:

```python
def test_head_linked_rings_shapes():
    from src.puzzle.models_t3 import HeadLinkedRings
    model = HeadLinkedRings()
    out = model(_emb())
    assert out.shape == (N, 8)
    b = model.bottle(_emb())
    assert b.shape == (N, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. $PY -m pytest tests/test_models_t3.py -k linked_rings -v`
Expected: FAIL with `ImportError: cannot import name 'HeadLinkedRings'`.

- [ ] **Step 3: Implement the model**

Append to `src/puzzle/models_t3.py`:

```python
class HeadLinkedRings(nn.Module):
    """Experiment γ: country encoded as two linked rings in a 3-D bottleneck.
    country=0 → ring A (xy-plane, centre origin), country=1 → ring B (xz-plane,
    centre (1,0,0)). The within-ring angle carries food (the mandatory spreader).
    All 8 features are decoded by per-feature MLP heads (3→16→ReLU→1).
    The bottleneck is NOT unit-normalised (the two rings sit at different centres).
    """
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
        )
        self.bottleneck = nn.Linear(64, 3)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
            for _ in range(8)
        ])

    def bottle(self, x: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(self.enc(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bottle(x)
        return torch.cat([h(b) for h in self.heads], dim=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. $PY -m pytest tests/test_models_t3.py -k linked_rings -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/puzzle/models_t3.py tests/test_models_t3.py
git commit -m "feat: HeadLinkedRings model for Experiment gamma (3-D linked-ring bottleneck)"
```

---

### Task 7: Experiment γ train + analyze (scripts 33, 34)

**Files:**
- Create: `scripts/33_train_linked_rings.py`
- Create: `scripts/34_analyze_linked_rings.py`
- Writes: `artifacts/results/33_linked_rings_model.pt`, `artifacts/results/34_linked_rings_probes.csv`, `artifacts/results/34_linked_rings_scatter.png`

**Interfaces:**
- Consumes: `HeadLinkedRings`, `linked_rings_loss`, `disc_crossing_count`, `arc_occupancy`, `FEATURE_NAMES`.
- Produces: country linear/nonlinear probe on the 3-D bottleneck, net disc-crossing count, ring-B occupancy, overall accuracy; a 3-D scatter coloured by country.

**Note (spec risk):** γ is the stretch. With binary `food` the spreader provides only two within-ring positions, so a clean ring may not form and the disc-crossing count may be degenerate. A clean unlink/collapse is an **acceptable, documented negative result** — record the real numbers; do not tune toward a predetermined answer.

- [ ] **Step 1: Write the train script**

Create `scripts/33_train_linked_rings.py`:

```python
# scripts/33_train_linked_rings.py
# Experiment γ — linked rings: country=0 → ring A (xy-plane, origin),
# country=1 → ring B (xz-plane, centre (1,0,0)); food sets within-ring angle.
# Loss = BCE(all 8) + λ · linked_rings_loss. λ tuned via LAMBDA.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.puzzle.models_t3 import HeadLinkedRings
from src.puzzle.geometry import linked_rings_loss
from src.puzzle.data import FEATURE_NAMES

N_EPOCHS, LR, BATCH, LAMBDA = 600, 1e-3, 512, 1.0
CI, FI = FEATURE_NAMES.index("country"), FEATURE_NAMES.index("food")


def main():
    data = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(data["train_emb"]).float()
    emb_te = torch.from_numpy(data["test_emb"]).float()
    lab_tr = torch.from_numpy(data["train_labels"]).float()
    lab_te = torch.from_numpy(data["test_labels"]).float()
    os.makedirs("artifacts/results", exist_ok=True)
    torch.manual_seed(0)
    model = HeadLinkedRings()
    opt = Adam(model.parameters(), lr=LR)
    sched = CosineAnnealingLR(opt, N_EPOCHS)
    model.train()
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(len(emb_tr))
        for s in range(0, len(emb_tr), BATCH):
            idx = perm[s:s + BATCH]
            logits = model(emb_tr[idx])
            bce = F.binary_cross_entropy_with_logits(logits, lab_tr[idx])
            ring = linked_rings_loss(model.bottle(emb_tr[idx]),
                                     lab_tr[idx][:, CI], lab_tr[idx][:, FI])
            loss = bce + LAMBDA * ring
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch + 1) % 150 == 0:
            model.eval()
            with torch.no_grad():
                acc = ((model(emb_te) > 0).float() == lab_te).float().mean()
            print(f"epoch {epoch+1:3d}: acc={acc:.4f}")
            model.train()
    model.eval()
    torch.save(model.state_dict(), "artifacts/results/33_linked_rings_model.pt")
    print("saved 33_linked_rings_model.pt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the analyze script**

Create `scripts/34_analyze_linked_rings.py`:

```python
# scripts/34_analyze_linked_rings.py
# Experiment γ analysis: country probe on 3-D bottleneck, disc-crossing linking number,
# ring-B occupancy, 3-D scatter coloured by country.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadLinkedRings
from src.puzzle.geometry import disc_crossing_count, arc_occupancy

CI = FEATURE_NAMES.index("country")


def main():
    data = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(data["train_emb"]).float()
    emb_te = torch.from_numpy(data["test_emb"]).float()
    lab_tr, lab_te = data["train_labels"], data["test_labels"]
    model = HeadLinkedRings()
    model.load_state_dict(torch.load(
        "artifacts/results/33_linked_rings_model.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        b_tr = model.bottle(emb_tr).numpy()
        b_te = model.bottle(emb_te).numpy()
        overall = ((model(emb_te) > 0).numpy().astype(float) == lab_te).mean()

    sc = StandardScaler().fit(b_tr)
    Xtr, Xte = sc.transform(b_tr), sc.transform(b_te)
    ytr, yte = lab_tr[:, CI], lab_te[:, CI]
    lin = LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte)
    nl = MLPClassifier((32,), max_iter=2000, random_state=0).fit(Xtr, ytr).score(Xte, yte)

    ringB = b_te[lab_te[:, CI] == 1]
    crossings = disc_crossing_count(ringB)
    psi_B = np.arctan2(ringB[:, 2], ringB[:, 0] - 1.0)
    occ_B = arc_occupancy(psi_B, np.ones(len(ringB), dtype=bool), n_bins=12)

    row = {"country_linear": lin, "country_nonlinear": nl,
           "net_disc_crossings": crossings, "ringB_arcs": occ_B,
           "overall_acc": float(overall)}
    pd.DataFrame([row]).to_csv("artifacts/results/34_linked_rings_probes.csv", index=False)
    print(row)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    c0, c1 = b_te[lab_te[:, CI] == 0], b_te[lab_te[:, CI] == 1]
    ax.scatter(c0[:, 0], c0[:, 1], c0[:, 2], c="tab:blue", s=5, alpha=0.4, label="country=0 (ring A)")
    ax.scatter(c1[:, 0], c1[:, 1], c1[:, 2], c="tab:red", s=5, alpha=0.4, label="country=1 (ring B)")
    ax.set_title(f"Experiment γ — linked rings (net crossings={crossings})")
    ax.legend(); fig.tight_layout()
    fig.savefig("artifacts/results/34_linked_rings_scatter.png", dpi=150)
    print("saved 34_linked_rings_scatter.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run train + analyze (tune λ if needed)**

Run: `PYTHONPATH=. $PY scripts/33_train_linked_rings.py`
Then: `PYTHONPATH=. $PY scripts/34_analyze_linked_rings.py`
Expected (success target): `country_linear ≤ 0.55`, `country_nonlinear ≥ 0.90`, `net_disc_crossings == 1`, `ringB_arcs ≥ 2`. If rings unlink / a class collapses, raise `LAMBDA` (e.g. 2.0, 5.0) and re-run once or twice; if it still collapses, record the negative result honestly.

- [ ] **Step 4: Commit**

```bash
git add scripts/33_train_linked_rings.py scripts/34_analyze_linked_rings.py artifacts/results/33_linked_rings_model.pt artifacts/results/34_linked_rings_*.csv artifacts/results/34_linked_rings_scatter.png
git commit -m "feat: Experiment gamma (linked rings) train + analyze with disc-crossing linking check"
```

---

### Task 8: Writeup + report sections (Ideas 10–12)

**Files:**
- Modify: `docs/writeup.md` (add Ideas 10, 11, 12 after Idea 9; add three rows to the Task 3 Summary table; update the opening Task 3 paragraph count from "nine" to "twelve")
- Modify: `docs/report.html` (add three tab buttons + three `tab-panel` divs; add summary rows)

**Interfaces:**
- Consumes: the real CSV numbers from `artifacts/results/30_*`, `32_*`, `34_*`. No placeholder numbers — read each CSV and transcribe actual values.

- [ ] **Step 1: Read the produced CSVs**

Run: `PYTHONPATH=. $PY -c "import pandas as pd; [print(f,'\n',pd.read_csv(f).to_string(),'\n') for f in ['artifacts/results/30_squarewave_probes.csv','artifacts/results/32_fourier_probes.csv','artifacts/results/32_harmonic_confusion.csv','artifacts/results/34_linked_rings_probes.csv']]"`
Expected: prints the four tables. Use these exact numbers in the prose.

- [ ] **Step 2: Add Ideas 10–12 to `docs/writeup.md`**

After the Idea 9 section (before `### Task 3 Summary`), insert three new sections `### Idea 10 — Square wave (constructive)`, `### Idea 11 — Fourier comb (constructive)`, `### Idea 12 — Linked rings (constructive)`. Each section must:
- State the goal and that it is **architecturally induced / constructive** (contrast with Idea 9 emergent).
- Report the actual probe numbers from the CSV in a markdown table.
- For Idea 10: the k-sweep table (k, linear, nonlinear, peak_k, arcs) + reference `*[Figure: 30_squarewave_curve.png ...]*`.
- For Idea 11: per-feature probe table + a sentence on the harmonic-confusion diagonal + reference `*[Figure: 32_harmonic_confusion.png ...]*`.
- For Idea 12: the single-row result (linear, nonlinear, crossings, occupancy) + reference `*[Figure: 34_linked_rings_scatter.png ...]*`, framed with the constructible-vs-emergent gap and honest negative-result handling if it collapsed.

Then update the Task 3 intro (`writeup.md` line ~140, "The nine experiments below") to "twelve experiments" and add the three summary rows to the `### Task 3 Summary` table (mirroring the existing row format).

- [ ] **Step 3: Add tabs 10–12 to `docs/report.html`**

After the `data-tab="bottleneck"` button (line ~761) add:

```html
        <button class="tab-btn" data-tab="squarewave">10 &mdash; Square Wave</button>
        <button class="tab-btn" data-tab="fourier">11 &mdash; Fourier Comb</button>
        <button class="tab-btn" data-tab="rings">12 &mdash; Linked Rings</button>
```

After the `id="tab-bottleneck"` panel's closing `</div>` add three `<div class="tab-panel" id="tab-squarewave" style="display:none">` … panels following the existing panel structure (heading, prose, a `data-table`, and an `<img>` or figure caption), populated with the real CSV numbers. Match the existing panels' class names and the `style="display:none"` pattern used by tabs 6–9.

- [ ] **Step 4: Verify docs reference only existing figures and real numbers**

Run: `cd /Users/janmenjayap/bluedot-impact/puzzles/1/bluedot-tais-puzzle && for f in $(grep -oE '3[0-9]_[A-Za-z0-9_]+\.png' docs/writeup.md | sort -u); do test -f artifacts/results/$f && echo "OK $f" || echo "MISSING $f"; done`
Expected: every referenced 3x_*.png prints `OK`.

- [ ] **Step 5: Commit**

```bash
git add docs/writeup.md docs/report.html
git commit -m "docs: add Ideas 10-12 (square wave / Fourier comb / linked rings) to writeup + report"
```

---

### Task 9: Full-suite green + plan close-out

**Files:**
- Modify: `docs/superpowers/specs/2026-06-16-binary-probe-resistant-codes-design.md` (flip Status to implemented; note actual outcomes)

- [ ] **Step 1: Run the full test suite**

Run: `PYTHONPATH=. $PY -m pytest -q`
Expected: all green (19 existing + the new model/geometry tests, ~26 passed).

- [ ] **Step 2: Update the spec status line**

Change the spec header `**Status:** Design (awaiting user review → writing-plans)` to `**Status:** Implemented (see scripts 29–34, results 30/32/34)` and append a one-paragraph "Outcome" note recording the actual A k-sweep / α diagonal / γ linking results (success or documented negative).

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-06-16-binary-probe-resistant-codes-design.md
git commit -m "docs: mark capstone spec implemented with actual outcomes"
```

---

## Self-Review

**Spec coverage:** A (square wave, k-sweep, occupancy, freq-peak) → Tasks 2–3. α (Fourier comb, harmonic-confusion heatmap) → Tasks 4–5. γ (linked rings, disc-crossing, spreader) → Tasks 6–7. `angular_freq_r2` lift to max_k=5 + script 28 import → Task 1. geometry helpers (occupancy, disc-crossing, ring loss) → Task 1. writeup/report Ideas 10–12 + summary rows → Task 8. Tests + spec status → Tasks 1–9. **No uncovered spec section.**

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to". Step 8.2/8.3 describe doc edits in prose because the numbers are unknown until the runs finish (Step 8.1 reads the real CSVs first, and Step 8.4 verifies figures exist) — this is data-dependent transcription, not a code placeholder.

**Type consistency:** `HeadSquareWave(k, country_idx)` / `.circle()`; `HeadFourierComb(harmonics)` / `.circle()`; `HeadLinkedRings()` / `.bottle()`; `angular_freq_r2(theta, y, max_k=5)`, `arc_occupancy(theta, mask, n_bins=12)`, `disc_crossing_count(points_xyz)`, `linked_rings_loss(bottle, country, food)` — names/signatures consistent across model, train, analyze, and test tasks.
