# src/puzzle/models_t3.py
import torch
import torch.nn as nn
import torch.nn.functional as F

TAPS_XOR = {"h2": 6}


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
        return self.mlp[6:](h2), self.proj_2d(h2)


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
        return self.dec(b), b


class HeadMNISTCircular(nn.Module):
    """Small CNN on MNIST with a 2-D circular bottleneck.
    Digit k is pushed to angle k*36° so that even/odd interleave on the circle.
    """
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                          # 14×14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                          # 7×7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
        )
        self.bottleneck = nn.Linear(128, 2)
        self.classifier = nn.Linear(2, n_classes)

    def bottle(self, x):
        return F.normalize(self.bottleneck(self.encoder(x)), dim=1)

    def forward(self, x):
        b = self.bottle(x)
        return self.classifier(b), b


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
