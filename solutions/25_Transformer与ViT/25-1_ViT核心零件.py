"""参考实现 25-1。学员完成本章前不要查看。"""

from __future__ import annotations

import math

import torch


def patchify(images, patch):
    if images.ndim != 4:
        raise ValueError("images 应为 (B, C, H, W)")
    b, c, h, w = images.shape
    if patch < 1 or h % patch or w % patch:
        raise ValueError("高和宽都必须能被 patch 整除")
    gh, gw = h // patch, w // patch
    x = images.reshape(b, c, gh, patch, gw, patch)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, gh * gw, c * patch * patch)


def unpatchify(tokens, patch, channels, height, width):
    if tokens.ndim != 3:
        raise ValueError("tokens 应为 (B, N, C*P*P)")
    if patch < 1 or height % patch or width % patch:
        raise ValueError("高和宽都必须能被 patch 整除")
    gh, gw = height // patch, width // patch
    b, n, d = tokens.shape
    if n != gh * gw or d != channels * patch * patch:
        raise ValueError("tokens 的形状和 patch、channels、height、width 对不上")
    x = tokens.reshape(b, gh, gw, channels, patch, patch)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, channels, height, width)


def scaled_dot_product_attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
        dead = ~mask.expand(scores.shape).any(dim=-1, keepdim=True)
        scores = scores.masked_fill(dead, 0.0)
    weights = torch.softmax(scores, dim=-1)
    if mask is not None:
        weights = weights.masked_fill(dead, 0.0)
    return weights @ v, weights


def multi_head_attention(x, w_qkv, b_qkv, w_out, b_out, num_heads, mask=None):
    b, n, d = x.shape
    if num_heads < 1 or d % num_heads:
        raise ValueError("D 必须能被 num_heads 整除")
    dh = d // num_heads
    q, k, v = (x @ w_qkv.transpose(0, 1) + b_qkv).split(d, dim=-1)

    def heads(t):
        return t.reshape(b, n, num_heads, dh).transpose(1, 2)

    if mask is not None and mask.ndim == 3:
        mask = mask[:, None]
    out, attn = scaled_dot_product_attention(heads(q), heads(k), heads(v), mask)
    out = out.transpose(1, 2).reshape(b, n, d)
    return out @ w_out.transpose(0, 1) + b_out, attn


def sincos_2d_position_embedding(dim, grid_h, grid_w, temperature=10000.0):
    if dim < 4 or dim % 4:
        raise ValueError("dim 必须是 4 的正整数倍")
    if grid_h < 1 or grid_w < 1:
        raise ValueError("网格的高和宽都至少为 1")
    quarter = dim // 4
    omega = 1.0 / float(temperature) ** (torch.arange(quarter, dtype=torch.float64) / quarter)
    rows, cols = torch.meshgrid(torch.arange(grid_h, dtype=torch.float64), torch.arange(grid_w, dtype=torch.float64), indexing="ij")
    r = rows.reshape(-1, 1) * omega
    c = cols.reshape(-1, 1) * omega
    return torch.cat([r.sin(), r.cos(), c.sin(), c.cos()], dim=1).to(torch.float32)
