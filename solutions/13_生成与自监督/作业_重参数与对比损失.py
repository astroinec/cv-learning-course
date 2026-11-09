"""参考实现（学员完成本章前不要查看）。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reparameterize(mu, logvar, eps=None):
    if eps is None:
        eps = torch.randn_like(mu)
    return mu + torch.exp(0.5 * logvar) * eps


def nt_xent_loss(z1, z2, temperature=0.2):
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = z @ z.T / temperature
    n = z1.shape[0]
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)
