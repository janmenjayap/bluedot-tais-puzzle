# Task 3 Bilinear Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Task 3 of the BlueDot TAIS Puzzle with five new artifacts: MNIST unit-norm fix, SAE decomposition of h2, analytic CPD proof that the original encoding is rank-1 bilinear, bilinear XOR model, and a bilinear text model with CPD analysis.

**Architecture:** All scripts run from project root `bluedot-tais-puzzle/` using the `bluedot-impact-puzzle-1` conda env. Data is in `artifacts/activations/acts.npz` (train: 7000, test: 1500 samples). The puzzle model is a 5-layer MLP (384→64→64→64→64→8) loaded via `from src.puzzle.model import load_head`. Feature indices come from `from src.puzzle.data import FEATURE_NAMES` (`country`=2, `food`=3, `sentiment`=4, `question`=1).

**Tech Stack:** PyTorch 2.10.0, numpy, sklearn, matplotlib, pandas; conda env `bluedot-impact-puzzle-1`; all new scripts follow the sys.path.insert pattern used in existing scripts.

---

## File Structure

**New files:**
- `src/puzzle/sae.py` — minimal top-k SAE + ConstrainedAdam (no external deps beyond torch)
- `scripts/20_train_sae_h2.py` — train SAE on puzzle h2 activations
- `scripts/21_analyze_sae_h2.py` — analyze SAE features for country decomposition
- `scripts/22_cpd_puzzle_analytic.py` — prove original encoding is rank-1 bilinear
- `scripts/23_train_bilinear_xor.py` — train bilinear XOR model
- `scripts/24_analyze_bilinear_xor.py` — probe bilinear XOR at h2
- `scripts/25_train_bilinear_text.py` — train bilinear text model on all 8 features
- `scripts/26_analyze_bilinear_cpd.py` — CPD weight-space analysis of bilinear model

**Modified files:**
- `src/puzzle/models_t3.py` — add unit-norm to `HeadMNISTCircular.bottle()`, add `HeadBilinearXOR`, `HeadBilinear`
- `docs/writeup.md` — add Experiment 7 (SAE), Experiment 8 (bilinear), updated summary table
- `docs/report.html` — add new tabs and update summary table

---

## Task 1: MNIST Unit-Norm Fix

Adds `F.normalize` to `HeadMNISTCircular.bottle()` so all class representations lie on the unit circle, eliminating radius variance and pushing the even/odd linear probe to near chance.

**Files:**
- Modify: `src/puzzle/models_t3.py:97-119` (HeadMNISTCircular class)
- Run: `scripts/18_train_mnist_circular.py` (retrain)
- Run: `scripts/19_analyze_mnist_circular.py` (re-analyze)

- [ ] **Step 1: Add F import and normalize to HeadMNISTCircular.bottle()**

Open `src/puzzle/models_t3.py`. At the very top, `torch.nn.functional` is not yet imported. Add it. Then change `HeadMNISTCircular.bottle()`:

```python
# At top of src/puzzle/models_t3.py, after "import torch.nn as nn":
import torch.nn.functional as F
```

Change `HeadMNISTCircular.bottle()` from:
```python
    def bottle(self, x):
        return self.bottleneck(self.encoder(x))
```
to:
```python
    def bottle(self, x):
        return F.normalize(self.bottleneck(self.encoder(x)), dim=1)
```

Leave `forward()` unchanged — it calls `self.bottle(x)` which will now return unit-norm vectors.

- [ ] **Step 2: Delete the old trained model and retrain**

```bash
cd /Users/janmenjayap/bluedot-impact/puzzles/1/bluedot-tais-puzzle
rm -f artifacts/results/18_mnist_circular_model.pt
conda run -n bluedot-impact-puzzle-1 python scripts/18_train_mnist_circular.py
```

Expected output (every 5 epochs, 30 total):
```
epoch   5: loss=0.XXXX  test acc=0.XXXX
...
epoch  30: loss=0.XXXX  test acc=0.XXXX
saved artifacts/results/18_mnist_circular_model.pt
```
Accuracy should reach ≥ 0.96.

- [ ] **Step 3: Re-analyze**

```bash
conda run -n bluedot-impact-puzzle-1 python scripts/19_analyze_mnist_circular.py
```

Expected output includes:
```
Even/odd probe on 2D bottleneck:
  base rate      : 0.508
  linear probe   : 0.XXXX  ← should be near chance (~0.51)
  nonlinear probe: 0.XXXX  ← should still be high (~0.97)
```

The key metric: `even_odd_linear` in `artifacts/results/19_mnist_circular_probes.csv` should be ≤ 0.54. If it is higher than 0.55, check that the normalize was saved correctly by running `conda run -n bluedot-impact-puzzle-1 python -c "import torch; from src.puzzle.models_t3 import HeadMNISTCircular; m=HeadMNISTCircular(); x=torch.randn(4,1,28,28); b=m.bottle(x); print(b.norm(dim=1))"` — all norms should be 1.0.

---

## Task 2: Minimal SAE Implementation

Creates `src/puzzle/sae.py` with a self-contained top-k SAE and ConstrainedAdam. No external dependencies beyond torch.

**Files:**
- Create: `src/puzzle/sae.py`

- [ ] **Step 1: Write src/puzzle/sae.py**

```python
# src/puzzle/sae.py
import torch
import torch.nn as nn


class ConstrainedAdam(torch.optim.Adam):
    """Adam that re-normalizes decoder columns to unit norm after every step."""
    def __init__(self, params, decoder_weight: torch.nn.Parameter, lr: float = 1e-3):
        super().__init__(params, lr=lr)
        self._dec = decoder_weight

    def step(self, closure=None):
        super().step(closure=closure)
        with torch.no_grad():
            self._dec /= self._dec.norm(dim=0, keepdim=True).clamp(min=1e-8)


class SAE(nn.Module):
    """Top-k sparse autoencoder. Reconstructs d_in-dimensional activations."""
    def __init__(self, d_in: int, d_feats: int, k: int):
        super().__init__()
        self.k = k
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.w_enc = nn.Linear(d_in, d_feats, bias=True)
        self.w_dec = nn.Linear(d_feats, d_in, bias=False)
        nn.init.kaiming_uniform_(self.w_dec.weight)
        self.w_dec.weight.data /= self.w_dec.weight.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.w_enc.weight.data = self.w_dec.weight.data.T.clone()
        nn.init.zeros_(self.w_enc.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.w_enc(x - self.b_dec)
        topk_idx = pre.topk(self.k, dim=-1).indices
        mask = torch.zeros_like(pre)
        mask.scatter_(-1, topk_idx, 1.0)
        return pre.clamp(min=0) * mask

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.w_dec(z) + self.b_dec

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z

    def fit(self, acts: torch.Tensor, n_steps: int = 3000,
            batch: int = 512, lr: float = 1e-3) -> list:
        opt = ConstrainedAdam(self.parameters(), self.w_dec.weight, lr=lr)
        losses = []
        for step in range(n_steps):
            idx = torch.randint(0, len(acts), (batch,))
            x = acts[idx]
            x_hat, _ = self(x)
            loss = (x - x_hat).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            if (step + 1) % 1000 == 0:
                nmse = loss.item() / acts.pow(2).mean().item()
                print(f"  step {step+1:5d}: mse={loss.item():.5f}  nmse={nmse:.4f}")
        return losses
```

- [ ] **Step 2: Verify import works**

```bash
cd /Users/janmenjayap/bluedot-impact/puzzles/1/bluedot-tais-puzzle
conda run -n bluedot-impact-puzzle-1 python -c "
import sys, os; sys.path.insert(0, '.')
from src.puzzle.sae import SAE
import torch
m = SAE(64, 256, 10)
x = torch.randn(32, 64)
x_hat, z = m(x)
print('SAE output shapes:', x_hat.shape, z.shape)
print('Active features per sample:', (z > 0).sum(dim=1).float().mean().item())
"
```

Expected:
```
SAE output shapes: torch.Size([32, 64]) torch.Size([32, 256])
Active features per sample: 10.0
```

---

## Task 3: Train SAE on Puzzle h2 Activations

Trains the SAE on the 64-dimensional h2 representations from the puzzle's model, then saves the trained SAE.

**Files:**
- Create: `scripts/20_train_sae_h2.py`

- [ ] **Step 1: Write scripts/20_train_sae_h2.py**

```python
# scripts/20_train_sae_h2.py
# Train top-k SAE on h2 activations of the puzzle model.
# h2 = model.layers[:6](emb) — the layer where country has Z/2 symmetry encoding.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from src.puzzle.sae import SAE

D_IN, D_FEATS, K = 64, 256, 10


def main():
    d = np.load("artifacts/activations/acts.npz")
    h2_tr = torch.from_numpy(d["train_h2"]).float()
    h2_te = torch.from_numpy(d["test_h2"]).float()

    # Center activations (subtract decoder bias in the SAE handles this, but
    # pre-centering helps with initialization)
    mean = h2_tr.mean(0)
    h2_tr_c = h2_tr - mean
    h2_te_c = h2_te - mean
    torch.save(mean, "artifacts/results/20_sae_h2_mean.pt")

    print(f"h2 shape: train={h2_tr.shape}, test={h2_te.shape}")
    print(f"h2 norm mean: {h2_tr.norm(dim=1).mean():.3f}")

    sae = SAE(D_IN, D_FEATS, K)
    print(f"\nTraining SAE (d_in={D_IN}, d_feats={D_FEATS}, k={K})...")
    sae.fit(h2_tr_c, n_steps=3000, batch=512, lr=1e-3)

    sae.eval()
    with torch.no_grad():
        x_hat, _ = sae(h2_te_c)
        mse_te = (h2_te_c - x_hat).pow(2).mean().item()
        nmse_te = mse_te / h2_te_c.pow(2).mean().item()
    print(f"\nTest MSE={mse_te:.5f}  NMSE={nmse_te:.4f}")

    os.makedirs("artifacts/results", exist_ok=True)
    torch.save(sae.state_dict(), "artifacts/results/20_sae_h2.pt")
    print("saved artifacts/results/20_sae_h2.pt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run training**

```bash
cd /Users/janmenjayap/bluedot-impact/puzzles/1/bluedot-tais-puzzle
conda run -n bluedot-impact-puzzle-1 python scripts/20_train_sae_h2.py
```

Expected output ends with NMSE < 0.30:
```
  step  1000: mse=X.XXXXX  nmse=X.XXXX
  step  2000: mse=X.XXXXX  nmse=X.XXXX
  step  3000: mse=X.XXXXX  nmse=X.XXXX

Test MSE=X.XXXXX  NMSE=X.XXXX
saved artifacts/results/20_sae_h2.pt
```

---

## Task 4: Analyze SAE Features for Country Decomposition

Shows that country's abs-value encoding decomposes into two SAE features — one for each arm of the V-shape — and that neither alone can decode country linearly, but together they do.

**Files:**
- Create: `scripts/21_analyze_sae_h2.py`

- [ ] **Step 1: Write scripts/21_analyze_sae_h2.py**

```python
# scripts/21_analyze_sae_h2.py
# Analyze which SAE features correspond to country's Z/2 encoding.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from src.puzzle.sae import SAE
from src.puzzle.data import FEATURE_NAMES

COUNTRY = FEATURE_NAMES.index("country")
FOOD    = FEATURE_NAMES.index("food")
D_IN, D_FEATS, K = 64, 256, 10


def main():
    d = np.load("artifacts/activations/acts.npz")
    h2_tr = torch.from_numpy(d["train_h2"]).float()
    h2_te = torch.from_numpy(d["test_h2"]).float()
    lab_tr = d["train_labels"]
    lab_te = d["test_labels"]

    mean = torch.load("artifacts/results/20_sae_h2_mean.pt", weights_only=True)
    h2_tr_c = (h2_tr - mean)
    h2_te_c = (h2_te - mean)

    sae = SAE(D_IN, D_FEATS, K)
    sae.load_state_dict(torch.load("artifacts/results/20_sae_h2.pt", weights_only=True))
    sae.eval()

    with torch.no_grad():
        z_tr = sae.encode(h2_tr_c).numpy()  # [7000, 256]
        z_te = sae.encode(h2_te_c).numpy()  # [1500, 256]

    ctr, cte = lab_tr[:, COUNTRY], lab_te[:, COUNTRY]
    ftr, fte = lab_tr[:, FOOD],    lab_te[:, FOOD]

    # For each SAE feature: mean activation per country group
    rows = []
    for k in range(D_FEATS):
        mean0 = z_tr[ctr == 0, k].mean()
        mean1 = z_tr[ctr == 1, k].mean()
        diff  = abs(mean0 - mean1)
        rows.append({"feature": k, "mean_country0": mean0, "mean_country1": mean1,
                     "abs_diff": diff})
    feat_df = pd.DataFrame(rows).sort_values("abs_diff", ascending=False)
    print("\nTop 10 SAE features by |mean_country0 - mean_country1|:")
    print(feat_df.head(10).to_string(index=False))

    # Top-2 country-discriminating features
    top2 = feat_df["feature"].values[:2]
    print(f"\nTop-2 country features: {top2}")

    # Linear probes on individual features vs pairs
    results = []
    for feat in top2:
        acc = LogisticRegression(max_iter=2000).fit(
            z_tr[:, [feat]], ctr).score(z_te[:, [feat]], cte)
        results.append({"scope": f"feat_{feat}_alone", "linear_acc": acc})

    # Pair of top-2
    acc_pair_lin = LogisticRegression(max_iter=2000).fit(
        z_tr[:, top2], ctr).score(z_te[:, top2], cte)
    acc_pair_mlp = MLPClassifier((32,), max_iter=2000, random_state=0).fit(
        z_tr[:, top2], ctr).score(z_te[:, top2], cte)
    results.append({"scope": "top2_pair_linear", "linear_acc": acc_pair_lin})
    results.append({"scope": "top2_pair_nonlinear", "linear_acc": acc_pair_mlp})

    # Full SAE activation linear probe for country
    acc_full_lin = LogisticRegression(max_iter=2000).fit(z_tr, ctr).score(z_te, cte)
    acc_full_mlp = MLPClassifier((64,), max_iter=2000, random_state=0).fit(
        z_tr, ctr).score(z_te, cte)
    results.append({"scope": "all256_linear", "linear_acc": acc_full_lin})
    results.append({"scope": "all256_nonlinear", "linear_acc": acc_full_mlp})

    probe_df = pd.DataFrame(results)
    print("\nProbe results:")
    print(probe_df.to_string(index=False))

    # Scatter: top-2 features coloured by (country, food)
    fig, ax = plt.subplots(figsize=(6, 5))
    labels_combined = cte * 2 + lab_te[:, FOOD]
    colors = {0: "steelblue", 1: "orange", 2: "green", 3: "red"}
    markers = {0: "o", 1: "o", 2: "^", 3: "^"}
    names = {0: "c=0,f=0", 1: "c=0,f=1", 2: "c=1,f=0", 3: "c=1,f=1"}
    for g in range(4):
        mask = labels_combined == g
        ax.scatter(z_te[mask, top2[0]], z_te[mask, top2[1]],
                   c=colors[g], marker=markers[g], alpha=0.4, s=20, label=names[g])
    ax.set_xlabel(f"SAE feature {top2[0]}")
    ax.set_ylabel(f"SAE feature {top2[1]}")
    ax.set_title("Top-2 country SAE features (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("artifacts/results/21_sae_country_scatter.png", dpi=150)

    feat_df.to_csv("artifacts/results/21_sae_feature_importance.csv", index=False)
    probe_df.to_csv("artifacts/results/21_sae_country_probes.csv", index=False)
    print("\nsaved 21_sae_feature_importance.csv, 21_sae_country_probes.csv, 21_sae_country_scatter.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run analysis**

```bash
conda run -n bluedot-impact-puzzle-1 python scripts/21_analyze_sae_h2.py
```

Key result to verify: `top2_pair_linear` should be ≥ 0.85 while individual features should be < 0.85. This shows country requires the combination of at least two SAE features.

---

## Task 5: Analytic CPD Proof — Original Encoding is Rank-1 Bilinear

Shows analytically that the puzzle's abs-value country encoding is a rank-1 symmetric bilinear form `h2ᵀ (wf ⊗ wf) h2`, providing the theoretical bridge to the bilinear model.

**Files:**
- Create: `scripts/22_cpd_puzzle_analytic.py`

- [ ] **Step 1: Write scripts/22_cpd_puzzle_analytic.py**

```python
# scripts/22_cpd_puzzle_analytic.py
# Show the country encoding at h2 is a rank-1 symmetric bilinear form.
# The food direction wf satisfies: |wf·h2| separates country.
# (wf·h2)^2 = h2^T (wf⊗wf) h2 is a rank-1 bilinear form that does the same.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.puzzle.data import FEATURE_NAMES

COUNTRY = FEATURE_NAMES.index("country")
FOOD    = FEATURE_NAMES.index("food")


def main():
    d = np.load("artifacts/activations/acts.npz")
    h2_tr, h2_te = d["train_h2"], d["test_h2"]
    lab_tr, lab_te = d["train_labels"], d["test_labels"]
    ctr, cte = lab_tr[:, COUNTRY], lab_te[:, COUNTRY]
    ftr = lab_tr[:, FOOD]

    # Food direction from h2 space (unit vector)
    wf = LogisticRegression(C=1, max_iter=5000).fit(h2_tr, ftr).coef_[0]
    wf_unit = wf / np.linalg.norm(wf)

    # Three country decoders based on the food direction
    proj_tr = h2_tr @ wf_unit          # linear projection
    proj_te = h2_te @ wf_unit

    abs_proj_tr = np.abs(proj_tr)       # abs(proj) — the known decoder
    abs_proj_te = np.abs(proj_te)

    quad_tr = proj_tr ** 2              # proj^2 = h2^T (wf⊗wf) h2 — rank-1 bilinear form
    quad_te = proj_te ** 2

    def acc1d(train_feat, test_feat, y_tr, y_te):
        return LogisticRegression(C=1, max_iter=5000).fit(
            train_feat.reshape(-1, 1), y_tr).score(
            test_feat.reshape(-1, 1), y_te)

    acc_raw  = acc1d(proj_tr,     proj_te,     ctr, cte)
    acc_abs  = acc1d(abs_proj_tr, abs_proj_te, ctr, cte)
    acc_quad = acc1d(quad_tr,     quad_te,     ctr, cte)

    print("Country decoder accuracy via food direction:")
    print(f"  raw projection  (linear):             {acc_raw:.4f}  (chance — Z/2 symmetry)")
    print(f"  |projection|    (abs-value):           {acc_abs:.4f}  (known decoder)")
    print(f"  projection^2    (rank-1 bilinear form):{acc_quad:.4f}  (new result — same info)")

    # Country=0 has large |proj|, country=1 has small |proj|
    for label, name in [(0, "country=0"), (1, "country=1")]:
        mask_tr = ctr == label
        print(f"\n  {name}: mean|proj|={abs_proj_tr[mask_tr].mean():.3f}  "
              f"mean(proj^2)={quad_tr[mask_tr].mean():.3f}")

    # Cosine similarity between abs_proj and sqrt(quad_proj) [they are the same thing]
    cos_sim = float(np.dot(abs_proj_te, np.sqrt(quad_te)) /
                    (np.linalg.norm(abs_proj_te) * np.linalg.norm(np.sqrt(quad_te)) + 1e-10))
    print(f"\nCosine similarity between |proj| and sqrt(proj^2): {cos_sim:.6f} (expected: 1.0)")

    # The rank-1 bilinear tensor B[i,j] = wf_unit[i] * wf_unit[j]
    # Its spectral norm (largest eigenvalue) = ||wf_unit||^2 = 1.0
    B_rank1 = np.outer(wf_unit, wf_unit)  # [64, 64]
    eigvals = np.linalg.eigvalsh(B_rank1)
    print(f"\nRank-1 bilinear tensor B = wf⊗wf:")
    print(f"  Shape: {B_rank1.shape}, rank: {np.sum(eigvals > 1e-8)}")
    print(f"  Largest eigenvalue: {eigvals[-1]:.6f} (= 1.0 for unit wf)")
    print(f"  Sum of eigenvalues: {eigvals.sum():.6f} (= 1.0 for unit wf)")

    pd.DataFrame([{
        "acc_raw_proj": acc_raw,
        "acc_abs_proj": acc_abs,
        "acc_quad_proj": acc_quad,
        "cos_sim_abs_vs_sqrt_quad": cos_sim,
        "bilinear_tensor_rank": int(np.sum(eigvals > 1e-8)),
        "bilinear_largest_eigval": float(eigvals[-1]),
    }]).to_csv("artifacts/results/22_cpd_analytic.csv", index=False)
    print("\nsaved artifacts/results/22_cpd_analytic.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the analytic script**

```bash
conda run -n bluedot-impact-puzzle-1 python scripts/22_cpd_puzzle_analytic.py
```

Expected:
```
Country decoder accuracy via food direction:
  raw projection  (linear):             0.47XX  (chance — Z/2 symmetry)
  |projection|    (abs-value):           0.94XX  (known decoder)
  projection^2    (rank-1 bilinear form):0.94XX  (new result — same info)
```
`acc_abs` and `acc_quad` should be nearly identical (both ≈ 0.946), confirming the rank-1 bilinear claim.

---

## Task 6: Bilinear Model Classes

Adds three new classes to `src/puzzle/models_t3.py`: unit-norm MNIST was done in Task 1. Now add `HeadBilinearXOR` (for the XOR experiment) and `HeadBilinear` (for the text classification CPD experiment).

**Files:**
- Modify: `src/puzzle/models_t3.py` (append two new classes)

- [ ] **Step 1: Append HeadBilinearXOR and HeadBilinear to src/puzzle/models_t3.py**

Read the current end of `src/puzzle/models_t3.py` first, then append after the `Predictor` class:

```python
class HeadBilinearXOR(nn.Module):
    """9-output bilinear head for XOR experiment. No ReLU — pure multiplicative interactions.
    Hidden: h2_k = (W_L @ emb)_k * (W_R @ emb)_k.
    """
    def __init__(self, d_emb: int = 384, d_hidden: int = 64):
        super().__init__()
        self.left  = nn.Linear(d_emb, d_hidden, bias=False)
        self.right = nn.Linear(d_emb, d_hidden, bias=False)
        self.head  = nn.Linear(d_hidden, 9, bias=True)   # 8 features + XOR at index 8

    def h2(self, x: torch.Tensor) -> torch.Tensor:
        return self.left(x) * self.right(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.h2(x))


class HeadBilinear(nn.Module):
    """8-output bilinear text classifier.
    logit_f = W_head[f,:] @ ((W_L @ emb) * (W_R @ emb))
    The model IS its own CPD decomposition:
      B[f,i,j] = Σ_r W_head[f,r] W_L[r,i] W_R[r,j]
    with D = W_head, L = W_L.T, R = W_R.T.
    """
    def __init__(self, d_emb: int = 384, d_hidden: int = 64):
        super().__init__()
        self.left  = nn.Linear(d_emb, d_hidden, bias=False)
        self.right = nn.Linear(d_emb, d_hidden, bias=False)
        self.head  = nn.Linear(d_hidden, 8, bias=True)

    def h2(self, x: torch.Tensor) -> torch.Tensor:
        return self.left(x) * self.right(x)

    def forward(self, x: torch.Tensor):
        b = self.h2(x)
        return self.head(b), b

    def cpd_factors(self):
        """Return (L, R, D) where B[f,i,j] = Σ_r D[f,r] L[i,r] R[j,r].
        L: [d_emb, d_hidden], R: [d_emb, d_hidden], D: [8, d_hidden].
        """
        L = self.left.weight.T.detach()   # [384, 64]
        R = self.right.weight.T.detach()  # [384, 64]
        D = self.head.weight.detach()     # [8,  64]
        return L, R, D
```

- [ ] **Step 2: Verify import**

```bash
conda run -n bluedot-impact-puzzle-1 python -c "
import sys; sys.path.insert(0, '.')
from src.puzzle.models_t3 import HeadBilinearXOR, HeadBilinear
import torch
m = HeadBilinear()
x = torch.randn(4, 384)
logits, h2 = m(x)
print('HeadBilinear output:', logits.shape, h2.shape)
L, R, D = m.cpd_factors()
print('CPD factors:', L.shape, R.shape, D.shape)
"
```

Expected:
```
HeadBilinear output: torch.Size([4, 8]) torch.Size([4, 64])
CPD factors: torch.Size([384, 64]) torch.Size([384, 64]) torch.Size([8, 64])
```

---

## Task 7: Bilinear XOR — Train and Analyze

Trains the bilinear model on XOR(sentiment, question) and compares linear probe accuracy at h2 to the original ReLU XOR experiment (linear probe ≈ 0.951).

**Files:**
- Create: `scripts/23_train_bilinear_xor.py`
- Create: `scripts/24_analyze_bilinear_xor.py`

- [ ] **Step 1: Write scripts/23_train_bilinear_xor.py**

```python
# scripts/23_train_bilinear_xor.py
# Train bilinear (no-ReLU) model on XOR task. Compare to ReLU HeadXOR.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadBilinearXOR

SENTIMENT = FEATURE_NAMES.index("sentiment")
QUESTION  = FEATURE_NAMES.index("question")


def make_labels_9(labels_np):
    xor = (labels_np[:, SENTIMENT] ^ labels_np[:, QUESTION]).reshape(-1, 1)
    return np.concatenate([labels_np, xor], axis=1)


def train(n_epochs=300, lr=1e-3, batch_size=512):
    d = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(d["train_emb"]).float()
    emb_te = torch.from_numpy(d["test_emb"]).float()
    lab_tr = torch.from_numpy(make_labels_9(d["train_labels"])).float()
    lab_te = torch.from_numpy(make_labels_9(d["test_labels"])).float()

    model = HeadBilinearXOR()
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
                acc_x = ((logits[:, 8] > 0).float() == lab_te[:, 8]).float().mean()
            print(f"epoch {epoch+1:3d}: 8-feat acc={acc8:.3f}  xor acc={acc_x:.3f}")
            model.train()

    model.eval()
    os.makedirs("artifacts/results", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/results/23_bilinear_xor_model.pt")
    print("saved artifacts/results/23_bilinear_xor_model.pt")


if __name__ == "__main__":
    train()
```

- [ ] **Step 2: Write scripts/24_analyze_bilinear_xor.py**

```python
# scripts/24_analyze_bilinear_xor.py
# Probe the bilinear XOR model's h2 for XOR and compare to ReLU XOR experiment.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadBilinearXOR

SENTIMENT = FEATURE_NAMES.index("sentiment")
QUESTION  = FEATURE_NAMES.index("question")
XOR_IDX   = 8   # 9th column in 9-label array


def make_labels_9(labels_np):
    xor = (labels_np[:, SENTIMENT] ^ labels_np[:, QUESTION]).reshape(-1, 1)
    return np.concatenate([labels_np, xor], axis=1)


def main():
    d = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(d["train_emb"]).float()
    emb_te = torch.from_numpy(d["test_emb"]).float()
    lab9_tr = make_labels_9(d["train_labels"])
    lab9_te = make_labels_9(d["test_labels"])

    model = HeadBilinearXOR()
    model.load_state_dict(torch.load("artifacts/results/23_bilinear_xor_model.pt",
                                     weights_only=True))
    model.eval()
    with torch.no_grad():
        h2_tr = model.h2(emb_tr).numpy()
        h2_te = model.h2(emb_te).numpy()

    rows = []
    features = list(FEATURE_NAMES) + ["xor_sentiment_question"]
    for fi, name in enumerate(features):
        y_tr, y_te = lab9_tr[:, fi], lab9_te[:, fi]
        lin = LogisticRegression(max_iter=2000).fit(h2_tr, y_tr).score(h2_te, y_te)
        mlp = MLPClassifier((32,), max_iter=2000, random_state=0).fit(
            h2_tr, y_tr).score(h2_te, y_te)
        rows.append({"feature": name, "linear": lin, "nonlinear": mlp, "gap": mlp - lin})
        print(f"  {name:30s}: linear={lin:.4f}  nonlinear={mlp:.4f}  gap={mlp-lin:+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv("artifacts/results/24_bilinear_xor_probes.csv", index=False)
    print("\nsaved 24_bilinear_xor_probes.csv")

    # Compare XOR probe: bilinear vs ReLU (from 09_xor_probes.csv)
    try:
        relu_df = pd.read_csv("artifacts/results/09_xor_probes.csv")
        relu_xor = relu_df[relu_df["feature"] == "xor_sentiment_question"]
        if not relu_xor.empty:
            relu_lin = relu_xor["linear"].values[0]
            bilin_lin = df[df["feature"] == "xor_sentiment_question"]["linear"].values[0]
            print(f"\nXOR linear probe comparison:")
            print(f"  ReLU model h2:     {relu_lin:.4f}")
            print(f"  Bilinear model h2: {bilin_lin:.4f}")
            print(f"  Δ = {bilin_lin - relu_lin:+.4f}  (negative = bilinear is more probe-resistant)")
    except FileNotFoundError:
        print("(09_xor_probes.csv not found — skipping comparison)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run training and analysis**

```bash
conda run -n bluedot-impact-puzzle-1 python scripts/23_train_bilinear_xor.py
conda run -n bluedot-impact-puzzle-1 python scripts/24_analyze_bilinear_xor.py
```

Expected: XOR nonlinear probe ≥ 0.90. The XOR linear probe at h2 should be compared to the ReLU value (0.951); if bilinear is lower, that validates the architectural story.

---

## Task 8: Bilinear Text Model — Train and CPD Analysis

Trains `HeadBilinear` on all 8 features and runs CPD weight-space analysis to characterize how each feature's encoding relates to the rank-1 bilinear structure of the original model.

**Files:**
- Create: `scripts/25_train_bilinear_text.py`
- Create: `scripts/26_analyze_bilinear_cpd.py`

- [ ] **Step 1: Write scripts/25_train_bilinear_text.py**

```python
# scripts/25_train_bilinear_text.py
# Train bilinear text model: emb -> (W_L@emb)*(W_R@emb) -> head -> 8 logits.
# The model is its own CPD decomposition — no separate CPD training needed.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.puzzle.models_t3 import HeadBilinear

N_EPOCHS, LR, BATCH = 400, 1e-3, 512


def train():
    d = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(d["train_emb"]).float()
    emb_te = torch.from_numpy(d["test_emb"]).float()
    lab_tr = torch.from_numpy(d["train_labels"]).float()
    lab_te = torch.from_numpy(d["test_labels"]).float()

    model = HeadBilinear()
    opt   = Adam(model.parameters(), lr=LR)
    sched = CosineAnnealingLR(opt, N_EPOCHS)

    model.train()
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(len(emb_tr))
        for s in range(0, len(emb_tr), BATCH):
            idx = perm[s:s + BATCH]
            logits, _ = model(emb_tr[idx])
            loss = F.binary_cross_entropy_with_logits(logits, lab_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                logits_te, _ = model(emb_te)
                acc = ((logits_te > 0).float() == lab_te).float().mean()
            print(f"epoch {epoch+1:3d}: acc={acc:.4f}")
            model.train()

    model.eval()
    os.makedirs("artifacts/results", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/results/25_bilinear_text_model.pt")
    print("saved artifacts/results/25_bilinear_text_model.pt")


if __name__ == "__main__":
    train()
```

- [ ] **Step 2: Write scripts/26_analyze_bilinear_cpd.py**

```python
# scripts/26_analyze_bilinear_cpd.py
# CPD weight-space analysis of the bilinear text model.
# Since logit_f = W_head[f,:] @ ((W_L@emb)*(W_R@emb)), the model IS its CPD:
#   B[f,i,j] = Σ_r D[f,r] L[i,r] R[j,r]   with D=W_head, L=W_L.T, R=W_R.T
# We analyze per-component sigma, specialization, and direction structure.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.models_t3 import HeadBilinear

COUNTRY = FEATURE_NAMES.index("country")
FOOD    = FEATURE_NAMES.index("food")


def main():
    d = np.load("artifacts/activations/acts.npz")
    emb_tr = torch.from_numpy(d["train_emb"]).float()
    emb_te = torch.from_numpy(d["test_emb"]).float()
    lab_tr = d["train_labels"]
    lab_te = d["test_labels"]

    model = HeadBilinear()
    model.load_state_dict(torch.load("artifacts/results/25_bilinear_text_model.pt",
                                     weights_only=True))
    model.eval()

    # Extract CPD factors
    L, R, D = model.cpd_factors()   # L,R: [384, 64]  D: [8, 64]
    L_np, R_np, D_np = L.numpy(), R.numpy(), D.numpy()

    # Per-component sigma = ||L[:,r]|| * ||R[:,r]|| * ||D[:,r]||
    sigma = (np.linalg.norm(L_np, axis=0) *
             np.linalg.norm(R_np, axis=0) *
             np.linalg.norm(D_np, axis=0))
    order = np.argsort(sigma)[::-1]

    # Specialization: max |D[f,r]| / sum |D[:,r]| — what fraction goes to the top feature
    D_abs = np.abs(D_np)
    spec_max  = (D_abs.max(axis=0) / (D_abs.sum(axis=0) + 1e-8))
    spec_argmax = D_abs.argmax(axis=0)  # dominant feature index per component

    # XOR-like asymmetry: ||L-R|| / ||L|| per component
    asym = np.linalg.norm(L_np - R_np, axis=0) / (np.linalg.norm(L_np, axis=0) + 1e-8)

    print(f"\nTop-20 components by sigma:")
    print(f"{'rank':>4} {'comp':>4} {'sigma':>8} {'spec':>6} {'top_feat':>10} {'asym(L-R)':>10}")
    rows = []
    for rank_i, r in enumerate(order[:20]):
        fname = FEATURE_NAMES[spec_argmax[r]]
        print(f"{rank_i+1:4d} {r:4d} {sigma[r]:8.3f} {spec_max[r]:6.3f} "
              f"{fname:>10} {asym[r]:10.3f}")
        rows.append({"rank": rank_i + 1, "component": int(r), "sigma": float(sigma[r]),
                     "spec_max": float(spec_max[r]), "top_feature": fname,
                     "asym_lr": float(asym[r])})

    # How many components does country capture?
    country_D = D_np[COUNTRY]  # [64]
    country_sigma = np.abs(country_D) * np.linalg.norm(L_np, axis=0) * np.linalg.norm(R_np, axis=0)
    country_order = np.argsort(country_sigma)[::-1]
    print(f"\nTop-5 country components (by |D[country,r]| * ||L[:,r]|| * ||R[:,r]||):")
    for rank_i, r in enumerate(country_order[:5]):
        print(f"  rank {rank_i+1}: comp={r}  sigma_country={country_sigma[r]:.3f}  "
              f"asym={asym[r]:.3f}  L+R_norm={np.linalg.norm(L_np[:,r]+R_np[:,r]):.3f}")

    # Linear/nonlinear probes on bilinear h2
    with torch.no_grad():
        h2_tr = model.h2(emb_tr).numpy()
        h2_te = model.h2(emb_te).numpy()

    probe_rows = []
    for fi, name in enumerate(FEATURE_NAMES):
        y_tr, y_te = lab_tr[:, fi], lab_te[:, fi]
        lin = LogisticRegression(max_iter=2000).fit(h2_tr, y_tr).score(h2_te, y_te)
        mlp = MLPClassifier((32,), max_iter=2000, random_state=0).fit(
            h2_tr, y_tr).score(h2_te, y_te)
        probe_rows.append({"feature": name, "linear": lin, "nonlinear": mlp, "gap": mlp - lin})
    probe_df = pd.DataFrame(probe_rows)
    print("\nProbes on bilinear h2:")
    print(probe_df.to_string(index=False))

    # Figure: scatter country vs food probes, colored by component specialization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(range(64), sigma[order], c="steelblue", s=20)
    ax.set_xlabel("Component rank"); ax.set_ylabel("σ"); ax.set_title("Component σ spectrum")
    ax = axes[1]
    ax.scatter(range(64), spec_max[order], c="orange", s=20)
    ax.set_xlabel("Component rank"); ax.set_ylabel("Specialization (max/Σ)")
    ax.set_title("Per-component specialization")
    fig.tight_layout()
    fig.savefig("artifacts/results/26_bilinear_cpd_spectrum.png", dpi=150)

    pd.DataFrame(rows).to_csv("artifacts/results/26_bilinear_cpd_components.csv", index=False)
    probe_df.to_csv("artifacts/results/26_bilinear_probes.csv", index=False)
    print("\nsaved 26_bilinear_cpd_components.csv, 26_bilinear_probes.csv, 26_bilinear_cpd_spectrum.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run training and analysis**

```bash
conda run -n bluedot-impact-puzzle-1 python scripts/25_train_bilinear_text.py
conda run -n bluedot-impact-puzzle-1 python scripts/26_analyze_bilinear_cpd.py
```

Expected: model achieves ≥ 0.93 overall accuracy. CPD analysis shows per-feature specialization patterns.

---

## Task 9: Update Writeup and Report

Adds new sections to `docs/writeup.md` and `docs/report.html` covering the SAE analysis, analytic CPD proof, bilinear XOR, and bilinear text model.

**Files:**
- Modify: `docs/writeup.md`
- Modify: `docs/report.html`

**Important:** Do NOT edit these files until Tasks 1-8 are complete and all CSVs/numbers are known. Read the actual output CSV values before writing any numbers into the writeup. The writeup sections to add are:

1. **Experiment 7: SAE decomposition of h2** — after Experiment 6 (MNIST circular). Key claims: country requires 2 SAE features; neither alone achieves >X% accuracy; the pair achieves >Y% (use actual values from `21_sae_country_probes.csv`).

2. **Analytic result: original encoding is rank-1 bilinear** — brief section in Task 2 or as a subsection of the bilinear model section. Claims: `acc_abs ≈ acc_quad` (from `22_cpd_analytic.csv`); B = wf⊗wf has rank 1.

3. **Experiment 8: Bilinear text model + CPD** — full section. Claims: XOR linear probe comparison (from `24_bilinear_xor_probes.csv`); CPD specialization (from `26_bilinear_cpd_components.csv`); probes at bilinear h2 (from `26_bilinear_probes.csv`).

4. **Update summary table** — add rows for SAE analysis and bilinear model.

- [ ] **Step 1: Read all output CSVs and collect actual numbers**

```bash
cat artifacts/results/19_mnist_circular_probes.csv
cat artifacts/results/21_sae_country_probes.csv
cat artifacts/results/22_cpd_analytic.csv
cat artifacts/results/24_bilinear_xor_probes.csv
cat artifacts/results/26_bilinear_probes.csv
cat artifacts/results/26_bilinear_cpd_components.csv | head -10
```

- [ ] **Step 2: Read current writeup Task 3 section to find insertion points**

```bash
grep -n "## Task 3\|## Experiment\|## Summary" docs/writeup.md | head -30
```

- [ ] **Step 3: Add new sections to docs/writeup.md**

Add after the MNIST circular section. Follow the exact style of existing experiment sections (heading, key result sentence, table with probe numbers, explanation of what geometry emerged, what it means for probe resistance). Use actual CSV numbers — no approximations or placeholders.

- [ ] **Step 4: Add new tabs to docs/report.html**

Follow the existing tab pattern (tab-mnist was the last added tab). Add tab-sae and tab-bilinear with the same HTML structure. Update the summary table `<tbody>` to add new rows for SAE and bilinear experiments.

---

## Self-Review Checklist

**Spec coverage:**
- [x] Task 1: MNIST unit-norm fix (Task 1)
- [x] SAE on h2 (Tasks 2-4)
- [x] CPD analytic proof (Task 5)
- [x] Bilinear XOR (Tasks 6-7 partial, Task 7 full)
- [x] Bilinear text + CPD (Task 8)
- [x] Writeup update (Task 9)

**Type consistency:**
- `HeadBilinear.h2(x)` returns `[N, 64]` tensor — same name in scripts/25 and 26 ✓
- `HeadBilinearXOR.h2(x)` returns `[N, 64]` — same name in scripts/23 and 24 ✓
- `SAE.encode(x)` returns `[N, 256]` sparse activations — used in scripts/20 and 21 ✓
- `model.cpd_factors()` returns `(L[384,64], R[384,64], D[8,64])` — used in script/26 ✓
- All scripts load from `artifacts/activations/acts.npz` using same key names ✓
- All scripts save to `artifacts/results/` ✓
