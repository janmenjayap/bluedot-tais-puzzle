# src/puzzle/geometry.py
import torch
import torch.nn.functional as F
import math
import numpy as np
from sklearn.linear_model import LinearRegression

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


def circular_loss_multiclass(proj_2d: torch.Tensor, labels: torch.Tensor,
                             n_classes: int = 10) -> torch.Tensor:
    """
    Drive proj_2d onto unit circle: class k → angle k * 2π/n_classes.
    proj_2d: (N, 2)  labels: (N,) int/long
    """
    angles = labels.float() * (2 * math.pi / n_classes)
    targets = torch.stack([angles.cos(), angles.sin()], dim=1)
    return F.mse_loss(F.normalize(proj_2d, dim=1), targets)


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
