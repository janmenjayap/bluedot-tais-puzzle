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
