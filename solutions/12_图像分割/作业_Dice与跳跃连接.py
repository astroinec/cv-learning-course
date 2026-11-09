"""参考实现（学员完成本章前不要查看）。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    dice = (2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)
    return 1 - dice.mean()


def align_and_concat(decoder, skip):
    if decoder.shape[0] != skip.shape[0]:
        raise ValueError("batch 必须相同")
    _, _, h1, w1 = decoder.shape
    _, _, h2, w2 = skip.shape
    d = decoder
    if h1 > h2:
        top = (h1 - h2) // 2
        d = d[:, :, top:top + h2]
    if w1 > w2:
        left = (w1 - w2) // 2
        d = d[:, :, :, left:left + w2]
    ph, pw = max(h2 - d.shape[2], 0), max(w2 - d.shape[3], 0)
    d = F.pad(d, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
    return torch.cat([d, skip], dim=1)
