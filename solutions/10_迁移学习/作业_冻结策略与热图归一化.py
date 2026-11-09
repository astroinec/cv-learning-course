"""参考实现（学员完成本章前不要查看）。"""

from __future__ import annotations

import torch
from torch import nn


def configure_trainable_parameters(model, trainable_prefixes):
    names = []
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(tuple(trainable_prefixes))
        if p.requires_grad:
            names.append(name)
    return names


def normalize_heatmap(cam, eps=1e-8):
    c = torch.relu(cam)
    flat = c.flatten(1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    rng = hi - lo
    out = torch.where(rng > eps, (flat - lo) / (rng + eps), torch.zeros_like(flat))
    return out.view_as(cam).clamp(0, 1)
