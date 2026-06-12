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
