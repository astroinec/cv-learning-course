"""参考实现 30-1。学员完成本章前不要查看。"""

from __future__ import annotations

import math

import torch


def volume_weights(sigma, deltas):
    if sigma.dim() < 1:
        raise ValueError("sigma 至少要有一维（沿光线的采样点）")
    try:
        tau = sigma * deltas
    except RuntimeError as exc:
        raise ValueError("deltas 不能广播到 sigma 的形状") from exc
    if tau.shape != sigma.shape:
        raise ValueError("deltas 不能广播到 sigma 的形状")
    if bool((sigma < 0).any()) or bool((deltas < 0).any()):
        raise ValueError("sigma 和 deltas 都不能为负")
    alpha = 1.0 - torch.exp(-tau)
    # 透射率只累计前面各段的光学厚度。不能写成“累加到自己再减掉自己”：最后一段的间隔很大时会把前面的全部抵消掉。
    before = torch.cat([torch.zeros_like(tau[..., :1]), torch.cumsum(tau, dim=-1)[..., :-1]], dim=-1)
    return torch.exp(-before) * alpha


def composite_rays(weights, colors, t_vals, background=None, eps=1e-10):
    if colors.shape[:-1] != weights.shape or colors.shape[-1] != 3:
        raise ValueError("colors 的形状应为 weights 的形状再加一维 3")
    if t_vals.shape != weights.shape and t_vals.shape != weights.shape[-1:]:
        raise ValueError("t_vals 的形状应和 weights 相同，或者是 (N,)")
    acc = weights.sum(dim=-1)
    rgb = (weights.unsqueeze(-1) * colors).sum(dim=-2)
    if background is not None:
        rgb = rgb + (1.0 - acc).unsqueeze(-1) * background
    depth = (weights * t_vals).sum(dim=-1) / acc.clamp_min(eps)
    return rgb, depth, acc


def positional_encoding(x, num_freqs, include_input=True):
    if num_freqs < 0 or (num_freqs == 0 and not include_input):
        raise ValueError("num_freqs 不能为负；num_freqs 为 0 时必须保留原始坐标")
    parts = [x] if include_input else []
    for k in range(num_freqs):
        arg = (2.0 ** k) * math.pi * x
        parts.append(torch.sin(arg))
        parts.append(torch.cos(arg))
    return torch.cat(parts, dim=-1)


def project_gaussians(means, covs, R, t, fx, fy, cx, cy):
    if means.dim() != 2 or means.shape[-1] != 3 or covs.shape != (means.shape[0], 3, 3):
        raise ValueError("means 应为 (N, 3)，covs 应为 (N, 3, 3)")
    if R.shape != (3, 3) or t.shape != (3,):
        raise ValueError("R 应为 (3, 3)，t 应为 (3,)")
    cam = means @ R.T + t
    x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
    if bool((z <= 0).any()):
        raise ValueError("有高斯的中心不在相机前方（Z_c <= 0）")
    uv = torch.stack([fx * x / z + cx, fy * y / z + cy], dim=-1)
    zero = torch.zeros_like(z)
    J = torch.stack([
        torch.stack([fx / z, zero, -fx * x / z ** 2], dim=-1),
        torch.stack([zero, fy / z, -fy * y / z ** 2], dim=-1),
    ], dim=-2)
    cov_cam = R @ covs @ R.T
    cov2d = J @ cov_cam @ J.transpose(-1, -2)
    return uv, cov2d, z


def composite_gaussians(alphas, colors, depths, background, max_alpha=0.99):
    if alphas.dim() != 2:
        raise ValueError("alphas 应为 (P, N)")
    n = alphas.shape[1]
    if colors.shape != (n, 3) or depths.shape != (n,):
        raise ValueError("colors 应为 (N, 3)，depths 应为 (N,)")
    order = torch.argsort(depths, stable=True)
    a = alphas[:, order].clamp(0.0, max_alpha)
    keep = torch.cumprod(1.0 - a, dim=1)
    trans = torch.cat([torch.ones_like(keep[:, :1]), keep[:, :-1]], dim=1)
    w = trans * a
    acc = w.sum(dim=1)
    rgb = w @ colors[order] + (1.0 - acc).unsqueeze(-1) * background
    return rgb, acc
