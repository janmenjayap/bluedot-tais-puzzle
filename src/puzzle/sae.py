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
