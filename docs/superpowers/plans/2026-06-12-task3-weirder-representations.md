# Task 3: Weirder Representations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train five new MLP models that encode features using progressively weirder geometric structures (XOR/parity, rotational, JEPA-residual, helical, superposition), each connecting to reasoning or world-model theory.

**Architecture:** All experiments train new MLP heads on frozen 384-dim sentence embeddings pre-cached in `artifacts/activations/acts.npz` (`train_emb`: 7000×384, `test_emb`: 1500×384). Shared model definitions and geometry regularisers live in `src/puzzle/`. Each experiment: one training script + one analysis script. No GPU required.

**Tech Stack:** PyTorch 2.x, scikit-learn, numpy, matplotlib, pandas.

**Feature indices (feature_names.json):** number=0, question=1, color=2, food=3, sentiment=4, country=5, person=6, body_part=7

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `src/puzzle/models_t3.py` | **Create** | All 5 new model classes |
| `src/puzzle/geometry.py` | **Create** | Circular / helical / superposition loss functions |
| `tests/test_models_t3.py` | **Create** | Forward-pass + loss-shape tests |
| `scripts/08_train_xor.py` | **Create** | Idea 2: train 9-head XOR model |
| `scripts/09_analyze_xor.py` | **Create** | Idea 2: probe + geometry plot |
| `scripts/10_train_rotation.py` | **Create** | Idea 1: train with circular regulariser |
| `scripts/11_analyze_rotation.py` | **Create** | Idea 1: angular decoder + 2D scatter |
| `scripts/12_train_jepa.py` | **Create** | Idea 3: train predictor P on frozen h2 |
| `scripts/13_analyze_jepa.py` | **Create** | Idea 3: probe raw h2 vs residual |
| `scripts/14_train_helix.py` | **Create** | Idea 4: train with helical regulariser |
| `scripts/15_analyze_helix.py` | **Create** | Idea 4: 1D/2D/3D probe sweep |
| `scripts/16_train_superposition.py` | **Create** | Idea 5: train bottlenecked model |
| `scripts/17_analyze_superposition.py` | **Create** | Idea 5: entanglement probe |
| `artifacts/results/08_xor_model.pt` … `16_super_model.pt` | Generated | Model checkpoints |
| `artifacts/results/08_xor_*.{csv,png}` … | Generated | Per-experiment results |

---

## Task 1: Shared Model Definitions

**Files:**
- Create: `src/puzzle/models_t3.py`

- [ ] **Step 1: Write the file**

```python
# src/puzzle/models_t3.py
import torch
import torch.nn as nn
import torch.nn.functional as F

TAPS_XOR = {"h2": 6}   # layers[:6] → post-ReLU hidden-2


class HeadXOR(nn.Module):
    """9-output head: original 8 features + XOR(sentiment, question) at index 8."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, 9),
        )

    def forward(self, x):
        return self.layers(x)

    def h2(self, x):
        return self.layers[:6](x)


class HeadRotation(nn.Module):
    """8-output head + learned 2-D projection for circular regulariser."""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, 8),
        )
        self.proj_2d = nn.Linear(64, 2, bias=False)

    def h2(self, x):
        return self.mlp[:6](x)

    def forward(self, x):
        h2 = self.h2(x)
        return self.mlp[6:](h2), self.proj_2d(h2)   # (logits, proj)


class HeadHelix(nn.Module):
    """8-output head + learned 3-D projection for helical regulariser."""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, 8),
        )
        self.proj_3d = nn.Linear(64, 3, bias=False)

    def h2(self, x):
        return self.mlp[:6](x)

    def forward(self, x):
        h2 = self.h2(x)
        return self.mlp[6:](h2), self.proj_3d(h2)


class HeadSuper(nn.Module):
    """8-output head with a 2-D bottleneck shared by country + food."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
        )
        self.bottleneck = nn.Linear(64, 2)
        self.dec = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2, 64),   nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, 8),
        )

    def h2(self, x):
        return self.enc(x)

    def bottle(self, x):
        return self.bottleneck(self.enc(x))

    def forward(self, x):
        b = self.bottleneck(self.enc(x))
        return self.dec(b), b                        # (logits, bottleneck)


class Predictor(nn.Module):
    """JEPA predictor P: 7 other labels → predicted h2 (64-dim)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

    def forward(self, other_labels_float):
        return self.net(other_labels_float)
```

- [ ] **Step 2: Verify import works**

```bash
cd /Users/janmenjayap/bluedot-impact/puzzles/1/bluedot-tais-puzzle
python -c "from src.puzzle.models_t3 import HeadXOR, HeadRotation, HeadHelix, HeadSuper, Predictor; print('OK')"
```
Expected: `OK`

---

## Task 2: Geometry Regulariser Functions

**Files:**
- Create: `src/puzzle/geometry.py`
- Create: `tests/test_models_t3.py`

- [ ] **Step 1: Write geometry.py**

```python
# src/puzzle/geometry.py
import torch
import torch.nn.functional as F
import math

_PI_HALF = math.pi / 2


def circular_loss(proj_2d: torch.Tensor, binary_labels: torch.Tensor,
                  theta_0: float = 0.0, theta_1: float = _PI_HALF) -> torch.Tensor:
    """
    Drive proj_2d onto unit circle: label=0 → angle theta_0, label=1 → angle theta_1.
    proj_2d: (N, 2)  binary_labels: (N,) int/long
    """
    proj_norm = F.normalize(proj_2d, dim=1)
    angles = torch.where(
        binary_labels.bool(),
        proj_2d.new_full((len(binary_labels),), theta_1),
        proj_2d.new_full((len(binary_labels),), theta_0),
    )
    targets = torch.stack([angles.cos(), angles.sin()], dim=1)
    return F.mse_loss(proj_norm, targets)


def helical_loss(proj_3d: torch.Tensor, binary_labels: torch.Tensor,
                 theta_0: float = 0.0, theta_1: float = _PI_HALF,
                 pitch: float = 1.0) -> torch.Tensor:
    """
    Drive proj_3d onto helix: (cos θ, sin θ, pitch·θ) normalised per label.
    proj_3d: (N, 3)  binary_labels: (N,) int/long
    """
    angles = torch.where(
        binary_labels.bool(),
        proj_3d.new_full((len(binary_labels),), theta_1),
        proj_3d.new_full((len(binary_labels),), theta_0),
    )
    targets = torch.stack([angles.cos(), angles.sin(), pitch * angles], dim=1)
    return F.mse_loss(F.normalize(proj_3d, dim=1), F.normalize(targets, dim=1))


def superposition_loss(proj_2d: torch.Tensor,
                       country_labels: torch.Tensor,
                       food_labels: torch.Tensor) -> torch.Tensor:
    """
    Force country and food into interleaved corners of the unit circle:
      (c=0,f=0)→0°  (c=1,f=0)→90°  (c=0,f=1)→180°  (c=1,f=1)→270°
    proj_2d: (N, 2)  labels: (N,) int/long
    """
    idx = (country_labels * 2 + food_labels).long()
    corners = proj_2d.new_tensor([
        [1., 0.], [0., 1.], [-1., 0.], [0., -1.]
    ])
    targets = corners[idx]
    return F.mse_loss(F.normalize(proj_2d, dim=1), targets)
```

- [ ] **Step 2: Write tests**

```python
# tests/test_models_t3.py
import torch
import pytest
from src.puzzle.models_t3 import HeadXOR, HeadRotation, HeadHelix, HeadSuper, Predictor
from src.puzzle.geometry import circular_loss, helical_loss, superposition_loss

N, D = 8, 384


def _emb():
    return torch.randn(N, D)


def test_head_xor_shape():
    out = HeadXOR()(_emb())
    assert out.shape == (N, 9)


def test_head_rotation_shapes():
    logits, proj = HeadRotation()(_emb())
    assert logits.shape == (N, 8)
    assert proj.shape == (N, 2)


def test_head_helix_shapes():
    logits, proj = HeadHelix()(_emb())
    assert logits.shape == (N, 8)
    assert proj.shape == (N, 3)


def test_head_super_shapes():
    logits, bottle = HeadSuper()(_emb())
    assert logits.shape == (N, 8)
    assert bottle.shape == (N, 2)


def test_predictor_shape():
    other = torch.randint(0, 2, (N, 7)).float()
    out = Predictor()(other)
    assert out.shape == (N, 64)


def test_circular_loss_scalar():
    proj = torch.randn(N, 2)
    labels = torch.randint(0, 2, (N,))
    loss = circular_loss(proj, labels)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_helical_loss_scalar():
    proj = torch.randn(N, 3)
    labels = torch.randint(0, 2, (N,))
    loss = helical_loss(proj, labels)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_superposition_loss_scalar():
    proj = torch.randn(N, 2)
    c = torch.randint(0, 2, (N,))
    f = torch.randint(0, 2, (N,))
    loss = superposition_loss(proj, c, f)
    assert loss.shape == ()
    assert loss.item() >= 0
```

- [ ] **Step 3: Run tests**

```bash
cd /Users/janmenjayap/bluedot-impact/puzzles/1/bluedot-tais-puzzle
pytest tests/test_models_t3.py -v
```
Expected: 9 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/puzzle/models_t3.py src/puzzle/geometry.py tests/test_models_t3.py
git commit -m "feat: Task 3 shared models and geometry regularisers"
```

---

## Task 3: Train XOR/Parity Model (Idea 2)

**Files:**
- Create: `scripts/08_train_xor.py`

- [ ] **Step 1: Write training script**

```python
# scripts/08_train_xor.py
"""
Idea 2: train a 9-head model where the 9th output is XOR(sentiment, question).
Saves model checkpoint to artifacts/results/08_xor_model.pt
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadXOR

SENTIMENT = FEATURE_NAMES.index("sentiment")   # 4
QUESTION  = FEATURE_NAMES.index("question")    # 1


def make_labels_9(labels_np: np.ndarray) -> np.ndarray:
    xor = (labels_np[:, SENTIMENT] ^ labels_np[:, QUESTION]).reshape(-1, 1)
    return np.concatenate([labels_np, xor], axis=1)


def train_xor(n_epochs=300, lr=1e-3, batch_size=512):
    d = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(d["train_emb"]).float()
    emb_te = torch.from_numpy(d["test_emb"]).float()
    lab_tr = torch.from_numpy(make_labels_9(d["train_labels"])).float()
    lab_te = torch.from_numpy(make_labels_9(d["test_labels"])).float()

    model = HeadXOR()
    opt = Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(len(emb_tr))
        for s in range(0, len(emb_tr), batch_size):
            idx = perm[s:s + batch_size]
            loss = F.binary_cross_entropy_with_logits(model(emb_tr[idx]), lab_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(emb_te)
                acc8  = ((logits[:, :8] > 0).float() == lab_te[:, :8]).float().mean()
                acc_x = ((logits[:, 8]  > 0).float() == lab_te[:, 8]).float().mean()
            print(f"epoch {epoch+1:3d}: 8-feat acc={acc8:.3f}  xor acc={acc_x:.3f}")
            model.train()

    model.eval()
    torch.save(model.state_dict(), "artifacts/results/08_xor_model.pt")
    print("saved artifacts/results/08_xor_model.pt")
    return model


if __name__ == "__main__":
    train_xor()
```

- [ ] **Step 2: Run training**

```bash
cd /Users/janmenjayap/bluedot-impact/puzzles/1/bluedot-tais-puzzle
python scripts/08_train_xor.py
```
Expected (last line): `epoch 300: 8-feat acc≈0.96  xor acc≈0.90+`

- [ ] **Step 3: Commit**

```bash
git add scripts/08_train_xor.py artifacts/results/08_xor_model.pt
git commit -m "feat: Task 3 Idea 2 — train XOR/parity model"
```

---

## Task 4: Analyze XOR Geometry (Idea 2)

**Files:**
- Create: `scripts/09_analyze_xor.py`

- [ ] **Step 1: Write analysis script**

```python
# scripts/09_analyze_xor.py
"""
Probes the XOR/parity code:
  - Linear probe on h2 for xor_label (expected ~50%)
  - 2-layer MLP probe on h2 for xor_label (expected >90%)
  - 2D PCA scatter of h2, coloured by xor_label
Saves: artifacts/results/09_xor_probes.csv, 09_xor_geometry.png
"""
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadXOR
from scripts._08_train_xor import make_labels_9   # noqa

SENTIMENT = FEATURE_NAMES.index("sentiment")
QUESTION  = FEATURE_NAMES.index("question")

d = np.load("artifacts/activations/acts.npz")
emb_tr = torch.from_numpy(d["train_emb"]).float()
emb_te = torch.from_numpy(d["test_emb"]).float()
lab_tr_9 = make_labels_9(d["train_labels"])
lab_te_9 = make_labels_9(d["test_labels"])
xor_tr, xor_te = lab_tr_9[:, -1], lab_te_9[:, -1]

model = HeadXOR()
model.load_state_dict(torch.load("artifacts/results/08_xor_model.pt", weights_only=True))
model.eval()

with torch.no_grad():
    h2_tr = model.h2(emb_tr).numpy()
    h2_te = model.h2(emb_te).numpy()

sc = StandardScaler().fit(h2_tr)
Ztr, Zte = sc.transform(h2_tr), sc.transform(h2_te)

lin = LogisticRegression(max_iter=2000).fit(Ztr, xor_tr)
mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500).fit(Ztr, xor_tr)
acc_lin = lin.score(Zte, xor_te)
acc_mlp = mlp.score(Zte, xor_te)
base = max(xor_te.mean(), 1 - xor_te.mean())

print(f"base rate        : {base:.3f}")
print(f"linear probe     : {acc_lin:.3f}  (expected ~0.50)")
print(f"2-layer MLP probe: {acc_mlp:.3f}  (expected >0.90)")

pd.DataFrame([{"base": base, "linear_probe": acc_lin, "mlp_probe": acc_mlp}]
             ).to_csv("artifacts/results/09_xor_probes.csv", index=False)

# Geometry: 2D PCA coloured by xor_label
pca = PCA(n_components=2).fit(Ztr)
Z2_te = pca.transform(Zte)
fig, ax = plt.subplots(figsize=(6, 5))
for v, marker, label in [(0, "o", "xor=0"), (1, "^", "xor=1")]:
    m = xor_te == v
    ax.scatter(Z2_te[m, 0], Z2_te[m, 1], marker=marker, alpha=0.4, s=18, label=label)
ax.set_title(f"XOR/parity in h2 PCA-2D\nlinear={acc_lin:.2f}  mlp={acc_mlp:.2f}")
ax.legend(); fig.tight_layout()
fig.savefig("artifacts/results/09_xor_geometry.png", dpi=150)
print("saved 09_xor_probes.csv, 09_xor_geometry.png")
```

- [ ] **Step 2: Run analysis**

```bash
python scripts/09_analyze_xor.py
```
Expected: linear probe ≈ 0.50, MLP probe > 0.90, both files saved.

- [ ] **Step 3: Commit**

```bash
git add scripts/09_analyze_xor.py artifacts/results/09_xor_probes.csv artifacts/results/09_xor_geometry.png
git commit -m "feat: Task 3 Idea 2 — XOR geometry analysis"
```

---

## Task 5: Train Rotational Model (Idea 1)

**Files:**
- Create: `scripts/10_train_rotation.py`

- [ ] **Step 1: Write training script**

```python
# scripts/10_train_rotation.py
"""
Idea 1: train HeadRotation with BCE + circular regulariser on country.
lambda_circ drives the 2D projection of h2 to encode country=0 at 0° and country=1 at 90°.
Saves: artifacts/results/10_rotation_model.pt
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadRotation
from src.puzzle.geometry import circular_loss

COUNTRY = FEATURE_NAMES.index("country")  # 5
LAMBDA  = 0.5
LR, EPOCHS, BATCH = 1e-3, 400, 512


def train_rotation():
    d = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(d["train_emb"]).float()
    emb_te = torch.from_numpy(d["test_emb"]).float()
    lab_tr = torch.from_numpy(d["train_labels"]).float()
    lab_te = torch.from_numpy(d["test_labels"]).float()

    model = HeadRotation()
    opt = Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(emb_tr))
        for s in range(0, len(emb_tr), BATCH):
            idx = perm[s:s + BATCH]
            x, y = emb_tr[idx], lab_tr[idx]
            logits, proj = model(x)
            bce  = F.binary_cross_entropy_with_logits(logits, y)
            circ = circular_loss(proj, y[:, COUNTRY].long())
            loss = bce + LAMBDA * circ
            opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                logits, _ = model(emb_te)
                acc = ((logits > 0).float() == lab_te).float().mean()
            print(f"epoch {epoch+1:3d}: 8-feat acc={acc:.3f}")
            model.train()

    model.eval()
    torch.save(model.state_dict(), "artifacts/results/10_rotation_model.pt")
    print("saved artifacts/results/10_rotation_model.pt")
    return model


if __name__ == "__main__":
    train_rotation()
```

- [ ] **Step 2: Run training**

```bash
python scripts/10_train_rotation.py
```
Expected: 8-feat acc ≈ 0.95+ at epoch 400.

- [ ] **Step 3: Commit**

```bash
git add scripts/10_train_rotation.py artifacts/results/10_rotation_model.pt
git commit -m "feat: Task 3 Idea 1 — train rotational code model"
```

---

## Task 6: Analyze Rotational Geometry (Idea 1)

**Files:**
- Create: `scripts/11_analyze_rotation.py`

- [ ] **Step 1: Write analysis script**

```python
# scripts/11_analyze_rotation.py
"""
Probes the rotational/SO(2) code:
  - Linear probe on h2 for country (expected ~50%)
  - Angular decoder on 2D projection (expected >90%)
  - 2D scatter of proj_2d coloured by country (should show two arcs)
Saves: artifacts/results/11_rotation_probes.csv, 11_rotation_geometry.png
"""
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadRotation

COUNTRY = FEATURE_NAMES.index("country")

d = np.load("artifacts/activations/acts.npz")
emb_tr = torch.from_numpy(d["train_emb"]).float()
emb_te = torch.from_numpy(d["test_emb"]).float()
ctr = d["train_labels"][:, COUNTRY]
cte = d["test_labels"][:, COUNTRY]

model = HeadRotation()
model.load_state_dict(torch.load("artifacts/results/10_rotation_model.pt", weights_only=True))
model.eval()

with torch.no_grad():
    h2_tr = model.h2(emb_tr).numpy()
    h2_te = model.h2(emb_te).numpy()
    _, proj_tr = model(emb_tr); proj_tr = proj_tr.numpy()
    _, proj_te = model(emb_te); proj_te = proj_te.numpy()

sc = StandardScaler().fit(h2_tr)
Ztr, Zte = sc.transform(h2_tr), sc.transform(h2_te)

# Linear probe on raw h2
lin_h2 = LogisticRegression(max_iter=2000).fit(Ztr, ctr).score(Zte, cte)

# Linear probe on 2D projection
lin_2d = LogisticRegression(max_iter=2000).fit(proj_tr, ctr).score(proj_te, cte)

# Angular decoder: compute angle, threshold to classify
angles_tr = np.arctan2(proj_tr[:, 1], proj_tr[:, 0]).reshape(-1, 1)
angles_te = np.arctan2(proj_te[:, 1], proj_te[:, 0]).reshape(-1, 1)
ang_clf = LogisticRegression(max_iter=2000).fit(np.hstack([np.cos(angles_tr), np.sin(angles_tr)]), ctr)
acc_angular = ang_clf.score(np.hstack([np.cos(angles_te), np.sin(angles_te)]), cte)

base = max(cte.mean(), 1 - cte.mean())
print(f"base rate            : {base:.3f}")
print(f"linear probe (h2)    : {lin_h2:.3f}  (expected ~0.50)")
print(f"linear probe (2D)    : {lin_2d:.3f}")
print(f"angular decoder (2D) : {acc_angular:.3f}  (expected >0.90)")

pd.DataFrame([{"base": base, "linear_h2": lin_h2, "linear_2d": lin_2d, "angular": acc_angular}]
             ).to_csv("artifacts/results/11_rotation_probes.csv", index=False)

# Scatter: 2D projection coloured by country
fig, ax = plt.subplots(figsize=(5, 5))
for v, marker, label in [(0, "o", "country=0"), (1, "^", "country=1")]:
    m = cte == v
    ax.scatter(proj_te[m, 0], proj_te[m, 1], marker=marker, alpha=0.4, s=18, label=label)
theta = np.linspace(0, 2 * np.pi, 200)
r = np.abs(proj_te).max() * 0.8
ax.plot(r * np.cos(theta), r * np.sin(theta), "k--", lw=0.8, label="unit circle")
ax.set_aspect("equal"); ax.legend(fontsize=8)
ax.set_title(f"Rotational code (SO(2))\nangular acc={acc_angular:.2f}  linear acc={lin_h2:.2f}")
fig.tight_layout(); fig.savefig("artifacts/results/11_rotation_geometry.png", dpi=150)
print("saved 11_rotation_probes.csv, 11_rotation_geometry.png")
```

- [ ] **Step 2: Run**

```bash
python scripts/11_analyze_rotation.py
```
Expected: linear probe ≈ 0.50, angular decoder > 0.90.

- [ ] **Step 3: Commit**

```bash
git add scripts/11_analyze_rotation.py artifacts/results/11_rotation_probes.csv artifacts/results/11_rotation_geometry.png
git commit -m "feat: Task 3 Idea 1 — rotational geometry analysis"
```

---

## Task 7: Train JEPA Predictor (Idea 3)

**Files:**
- Create: `scripts/12_train_jepa.py`

- [ ] **Step 1: Write training script**

```python
# scripts/12_train_jepa.py
"""
Idea 3: freeze the original Head, train a Predictor P that predicts h2 from the 7
non-country label values. Country's information = residual h2 - P(other_labels).
Saves: artifacts/results/12_jepa_predictor.pt, 12_jepa_h2.npz
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.model import load_head
from src.puzzle.models_t3 import Predictor

COUNTRY = FEATURE_NAMES.index("country")  # 5
OTHER   = [i for i in range(8) if i != COUNTRY]


def train_jepa(n_epochs=400, lr=1e-3, batch_size=512):
    d = np.load("artifacts/activations/acts.npz")
    h2_tr = torch.from_numpy(d["train_h2"]).float()   # (7000, 64) from original model
    h2_te = torch.from_numpy(d["test_h2"]).float()
    lab_tr = torch.from_numpy(d["train_labels"]).float()
    lab_te = torch.from_numpy(d["test_labels"]).float()

    other_tr = lab_tr[:, OTHER]   # (7000, 7) — all labels except country
    other_te = lab_te[:, OTHER]

    predictor = Predictor()
    opt = Adam(predictor.parameters(), lr=lr)

    for epoch in range(n_epochs):
        perm = torch.randperm(len(h2_tr))
        for s in range(0, len(h2_tr), batch_size):
            idx = perm[s:s + batch_size]
            h2_pred = predictor(other_tr[idx])
            loss = F.mse_loss(h2_pred, h2_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                val_loss = F.mse_loss(predictor(other_te), h2_te).item()
            print(f"epoch {epoch+1:3d}: val MSE={val_loss:.4f}")

    torch.save(predictor.state_dict(), "artifacts/results/12_jepa_predictor.pt")

    # Pre-compute residuals and save
    predictor.eval()
    with torch.no_grad():
        res_tr = (h2_tr - predictor(other_tr)).numpy()
        res_te = (h2_te - predictor(other_te)).numpy()
    np.savez("artifacts/results/12_jepa_residuals.npz",
             res_tr=res_tr, res_te=res_te,
             h2_tr=h2_tr.numpy(), h2_te=h2_te.numpy())
    print("saved 12_jepa_predictor.pt, 12_jepa_residuals.npz")


if __name__ == "__main__":
    train_jepa()
```

- [ ] **Step 2: Run**

```bash
python scripts/12_train_jepa.py
```
Expected: val MSE should decrease significantly from epoch 1 to 400.

- [ ] **Step 3: Commit**

```bash
git add scripts/12_train_jepa.py artifacts/results/12_jepa_predictor.pt artifacts/results/12_jepa_residuals.npz
git commit -m "feat: Task 3 Idea 3 — train JEPA predictor"
```

---

## Task 8: Analyze JEPA Residuals (Idea 3)

**Files:**
- Create: `scripts/13_analyze_jepa.py`

- [ ] **Step 1: Write analysis script**

```python
# scripts/13_analyze_jepa.py
"""
Idea 3: compare probing country from raw h2 vs from residual r = h2 - P(other_labels).
Key question: does the residual concentrate country information while suppressing others?
Also checks whether P is approximately linear (LeJEPA condition).
Saves: artifacts/results/13_jepa_probes.csv, 13_jepa_residual_probe.png
"""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from src.puzzle.data import FEATURE_NAMES

COUNTRY = FEATURE_NAMES.index("country")
OTHER   = [i for i in range(8) if i != COUNTRY]

d_orig = np.load("artifacts/activations/acts.npz")
d_res  = np.load("artifacts/results/12_jepa_residuals.npz")

h2_tr, h2_te   = d_res["h2_tr"], d_res["h2_te"]
res_tr, res_te = d_res["res_tr"], d_res["res_te"]
lab_tr = d_orig["train_labels"]
lab_te = d_orig["test_labels"]

sc_h2  = StandardScaler().fit(h2_tr)
sc_res = StandardScaler().fit(res_tr)

def probe(X_tr, X_te, y_tr, y_te):
    return LogisticRegression(max_iter=2000).fit(X_tr, y_tr).score(X_te, y_te)

rows = []
for feat_idx in range(8):
    feat = FEATURE_NAMES[feat_idx]
    y_tr, y_te = lab_tr[:, feat_idx], lab_te[:, feat_idx]
    acc_h2  = probe(sc_h2.transform(h2_tr),   sc_h2.transform(h2_te),   y_tr, y_te)
    acc_res = probe(sc_res.transform(res_tr),  sc_res.transform(res_te), y_tr, y_te)
    rows.append({"feature": feat, "probe_h2": acc_h2, "probe_residual": acc_res,
                 "is_country": feat_idx == COUNTRY})
    print(f"{feat:12s}: h2={acc_h2:.3f}  residual={acc_res:.3f}")

df = pd.DataFrame(rows)
df.to_csv("artifacts/results/13_jepa_probes.csv", index=False)

# Check linearity of P: fit W s.t. P(other_labels) ≈ W @ other_labels
other_tr = d_orig["train_labels"][:, OTHER].astype(float)
other_te = d_orig["test_labels"][:, OTHER].astype(float)
predicted_h2_tr = h2_tr - res_tr   # = P(other_labels)
predicted_h2_te = h2_te - res_te

W = Ridge(alpha=1.0).fit(other_tr, predicted_h2_tr)
mse_linear = np.mean((W.predict(other_te) - predicted_h2_te) ** 2)
mse_total  = np.mean(predicted_h2_te ** 2)
print(f"\nLinearity of P: linear approx MSE={mse_linear:.4f}  vs total variance={mse_total:.4f}")
print(f"Fraction unexplained by linear fit: {mse_linear/mse_total:.3f}")

# Bar chart: h2 vs residual probe per feature
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(df))
ax.bar(x - 0.2, df["probe_h2"],      0.35, label="raw h2")
ax.bar(x + 0.2, df["probe_residual"], 0.35, label="residual")
ax.set_xticks(x); ax.set_xticklabels(df["feature"], rotation=30, ha="right")
ax.axhline(0.5, color="k", ls="--", lw=0.8, label="chance")
ax.set_ylabel("probe accuracy"); ax.set_title("JEPA residual concentrates country info")
ax.legend(); fig.tight_layout()
fig.savefig("artifacts/results/13_jepa_residual_probe.png", dpi=150)
print("saved 13_jepa_probes.csv, 13_jepa_residual_probe.png")
```

- [ ] **Step 2: Run**

```bash
python scripts/13_analyze_jepa.py
```
Expected: country probe stays high in residual, other features drop; linearity fraction reported.

- [ ] **Step 3: Commit**

```bash
git add scripts/13_analyze_jepa.py artifacts/results/13_jepa_probes.csv artifacts/results/13_jepa_residual_probe.png
git commit -m "feat: Task 3 Idea 3 — JEPA residual analysis"
```

---

## Task 9: Train Helical Model (Idea 4)

**Files:**
- Create: `scripts/14_train_helix.py`

- [ ] **Step 1: Write training script**

```python
# scripts/14_train_helix.py
"""
Idea 4: train HeadHelix with BCE + helical regulariser on country.
Country=0 → angle 0 on helix; country=1 → angle π/2.
Saves: artifacts/results/14_helix_model.pt
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadHelix
from src.puzzle.geometry import helical_loss

COUNTRY = FEATURE_NAMES.index("country")
LAMBDA, LR, EPOCHS, BATCH = 0.5, 1e-3, 400, 512


def train_helix():
    d = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(d["train_emb"]).float()
    emb_te = torch.from_numpy(d["test_emb"]).float()
    lab_tr = torch.from_numpy(d["train_labels"]).float()
    lab_te = torch.from_numpy(d["test_labels"]).float()

    model = HeadHelix()
    opt = Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(emb_tr))
        for s in range(0, len(emb_tr), BATCH):
            idx = perm[s:s + BATCH]
            x, y = emb_tr[idx], lab_tr[idx]
            logits, proj = model(x)
            loss = (F.binary_cross_entropy_with_logits(logits, y)
                    + LAMBDA * helical_loss(proj, y[:, COUNTRY].long()))
            opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                logits, _ = model(emb_te)
                acc = ((logits > 0).float() == lab_te).float().mean()
            print(f"epoch {epoch+1:3d}: acc={acc:.3f}")
            model.train()

    model.eval()
    torch.save(model.state_dict(), "artifacts/results/14_helix_model.pt")
    print("saved artifacts/results/14_helix_model.pt")


if __name__ == "__main__":
    train_helix()
```

- [ ] **Step 2: Run**

```bash
python scripts/14_train_helix.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/14_train_helix.py artifacts/results/14_helix_model.pt
git commit -m "feat: Task 3 Idea 4 — train helical code model"
```

---

## Task 10: Analyze Helical Geometry (Idea 4)

**Files:**
- Create: `scripts/15_analyze_helix.py`

- [ ] **Step 1: Write analysis script**

```python
# scripts/15_analyze_helix.py
"""
Probe sweep: 1D, 2D circular, 3D helical on the proj_3d subspace.
Expected: 1D≈0.50, 2D≈0.75, 3D>0.90.
Saves: artifacts/results/15_helix_probes.csv, 15_helix_geometry.png
"""
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadHelix

COUNTRY = FEATURE_NAMES.index("country")

d = np.load("artifacts/activations/acts.npz")
emb_tr = torch.from_numpy(d["train_emb"]).float()
emb_te = torch.from_numpy(d["test_emb"]).float()
ctr = d["train_labels"][:, COUNTRY]
cte = d["test_labels"][:, COUNTRY]

model = HeadHelix()
model.load_state_dict(torch.load("artifacts/results/14_helix_model.pt", weights_only=True))
model.eval()
with torch.no_grad():
    _, proj_tr = model(emb_tr); proj_tr = proj_tr.numpy()
    _, proj_te = model(emb_te); proj_te = proj_te.numpy()

def probe_nd(X_tr, X_te, y_tr, y_te):
    sc = StandardScaler().fit(X_tr)
    return LogisticRegression(max_iter=2000).fit(sc.transform(X_tr), y_tr).score(sc.transform(X_te), y_te)

# Angle in 2D subspace (xy plane of helix)
ang_tr = np.arctan2(proj_tr[:, 1], proj_tr[:, 0])
ang_te = np.arctan2(proj_te[:, 1], proj_te[:, 0])
feats_ang_tr = np.stack([np.cos(ang_tr), np.sin(ang_tr)], axis=1)
feats_ang_te = np.stack([np.cos(ang_te), np.sin(ang_te)], axis=1)

acc_1d  = probe_nd(proj_tr[:, :1],  proj_te[:, :1],  ctr, cte)
acc_2d  = probe_nd(feats_ang_tr,    feats_ang_te,    ctr, cte)
acc_3d  = probe_nd(proj_tr,         proj_te,         ctr, cte)
base    = max(cte.mean(), 1 - cte.mean())

print(f"base={base:.3f}  1D={acc_1d:.3f}  2D-angular={acc_2d:.3f}  3D={acc_3d:.3f}")
pd.DataFrame([{"base": base, "probe_1d": acc_1d, "probe_2d_angular": acc_2d, "probe_3d": acc_3d}]
             ).to_csv("artifacts/results/15_helix_probes.csv", index=False)

# 3D scatter coloured by country
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")
for v, marker, label in [(0, "o", "country=0"), (1, "^", "country=1")]:
    m = cte == v
    ax.scatter(proj_te[m, 0], proj_te[m, 1], proj_te[m, 2],
               marker=marker, alpha=0.3, s=12, label=label)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title(f"Helical code  3D={acc_3d:.2f}  1D={acc_1d:.2f}")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig("artifacts/results/15_helix_geometry.png", dpi=150)
print("saved 15_helix_probes.csv, 15_helix_geometry.png")
```

- [ ] **Step 2: Run**

```bash
python scripts/15_analyze_helix.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/15_analyze_helix.py artifacts/results/15_helix_probes.csv artifacts/results/15_helix_geometry.png
git commit -m "feat: Task 3 Idea 4 — helical geometry analysis"
```

---

## Task 11: Train Superposition Model (Idea 5)

**Files:**
- Create: `scripts/16_train_superposition.py`

- [ ] **Step 1: Write training script**

```python
# scripts/16_train_superposition.py
"""
Idea 5: HeadSuper routes all information through a 2-D bottleneck.
Superposition regulariser pushes country+food to four interleaved corners.
Saves: artifacts/results/16_super_model.pt
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadSuper
from src.puzzle.geometry import superposition_loss

COUNTRY = FEATURE_NAMES.index("country")
FOOD    = FEATURE_NAMES.index("food")
LAMBDA, LR, EPOCHS, BATCH = 0.5, 1e-3, 500, 512


def train_super():
    d = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(d["train_emb"]).float()
    emb_te = torch.from_numpy(d["test_emb"]).float()
    lab_tr = torch.from_numpy(d["train_labels"]).float()
    lab_te = torch.from_numpy(d["test_labels"]).float()

    model = HeadSuper()
    opt = Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(emb_tr))
        for s in range(0, len(emb_tr), BATCH):
            idx = perm[s:s + BATCH]
            x, y = emb_tr[idx], lab_tr[idx]
            logits, bottle = model(x)
            loss = (F.binary_cross_entropy_with_logits(logits, y)
                    + LAMBDA * superposition_loss(bottle, y[:, COUNTRY].long(), y[:, FOOD].long()))
            opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                logits, _ = model(emb_te)
                acc = ((logits > 0).float() == lab_te).float().mean()
            print(f"epoch {epoch+1:3d}: acc={acc:.3f}")
            model.train()

    model.eval()
    torch.save(model.state_dict(), "artifacts/results/16_super_model.pt")
    print("saved artifacts/results/16_super_model.pt")


if __name__ == "__main__":
    train_super()
```

- [ ] **Step 2: Run**

```bash
python scripts/16_train_superposition.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/16_train_superposition.py artifacts/results/16_super_model.pt
git commit -m "feat: Task 3 Idea 5 — train superposition model"
```

---

## Task 12: Analyze Superposition (Idea 5)

**Files:**
- Create: `scripts/17_analyze_superposition.py`

- [ ] **Step 1: Write analysis script**

```python
# scripts/17_analyze_superposition.py
"""
Probes the superposition / entanglement code.
Shows that probing country contaminates food and vice versa in the 2D bottleneck.
Saves: artifacts/results/17_super_probes.csv, 17_super_geometry.png
"""
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadSuper

COUNTRY = FEATURE_NAMES.index("country")
FOOD    = FEATURE_NAMES.index("food")

d = np.load("artifacts/activations/acts.npz")
emb_tr = torch.from_numpy(d["train_emb"]).float()
emb_te = torch.from_numpy(d["test_emb"]).float()
lab_te = d["test_labels"]
lab_tr = d["train_labels"]

model = HeadSuper()
model.load_state_dict(torch.load("artifacts/results/16_super_model.pt", weights_only=True))
model.eval()
with torch.no_grad():
    bottle_tr = model.bottle(emb_tr).numpy()
    bottle_te = model.bottle(emb_te).numpy()

sc = StandardScaler().fit(bottle_tr)
Btr, Bte = sc.transform(bottle_tr), sc.transform(bottle_te)

rows = []
for feat_idx in [COUNTRY, FOOD]:
    y_tr, y_te = lab_tr[:, feat_idx], lab_te[:, feat_idx]
    acc = LogisticRegression(max_iter=2000).fit(Btr, y_tr).score(Bte, y_te)
    feat = FEATURE_NAMES[feat_idx]
    base = max(y_te.mean(), 1 - y_te.mean())
    print(f"{feat}: bottleneck probe={acc:.3f}  base={base:.3f}")
    rows.append({"feature": feat, "probe_bottleneck": acc, "base": base})

pd.DataFrame(rows).to_csv("artifacts/results/17_super_probes.csv", index=False)

# 2D scatter with 4 (country, food) colours
fig, ax = plt.subplots(figsize=(5, 5))
markers = {(0,0): ("o", "c=0,f=0"), (1,0): ("^", "c=1,f=0"),
           (0,1): ("s", "c=0,f=1"), (1,1): ("D", "c=1,f=1")}
for (cv, fv), (marker, label) in markers.items():
    m = (lab_te[:, COUNTRY] == cv) & (lab_te[:, FOOD] == fv)
    ax.scatter(bottle_te[m, 0], bottle_te[m, 1], marker=marker, alpha=0.4, s=20, label=label)
ax.set_title("Superposition: country + food in 2D bottleneck")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig("artifacts/results/17_super_geometry.png", dpi=150)
print("saved 17_super_probes.csv, 17_super_geometry.png")
```

- [ ] **Step 2: Run**

```bash
python scripts/17_analyze_superposition.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/17_analyze_superposition.py artifacts/results/17_super_probes.csv artifacts/results/17_super_geometry.png
git commit -m "feat: Task 3 Idea 5 — superposition geometry analysis"
```

---

## Execution Order

Tasks are independent after Task 2. Recommended order:
1. Tasks 1–2 (shared infrastructure, tests) — **must be first**
2. Tasks 3–4 (XOR, simplest) — **validate pipeline**
3. Tasks 5–6 (rotation, core LeJEPA claim)
4. Tasks 7–8 (JEPA residual, most theoretically novel)
5. Tasks 9–10 (helix)
6. Tasks 11–12 (superposition)
