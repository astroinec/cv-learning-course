"""参考实现 29-1。学员完成本章前不要查看。"""

from __future__ import annotations

import math

import torch


def _check_abar(abar, upper_closed=False):
    a = torch.as_tensor(abar)
    bad_low = bool((a <= 0).any())
    bad_high = bool((a > 1).any()) if upper_closed else bool((a >= 1).any())
    if bad_low or bad_high:
        raise ValueError("abar 超出允许的范围")


def convert_prediction(x_t, pred, abar_t, kind):
    if kind not in ("eps", "x0", "v"):
        raise ValueError("kind 只能是 'eps'、'x0' 或 'v'")
    if x_t.shape != pred.shape:
        raise ValueError("x_t 和 pred 的形状应相同")
    _check_abar(abar_t)
    a = torch.as_tensor(abar_t, dtype=x_t.dtype, device=x_t.device)
    sa, sn = a.sqrt(), (1.0 - a).sqrt()
    if kind == "eps":
        eps = pred
        x0 = (x_t - sn * eps) / sa
    elif kind == "x0":
        x0 = pred
        eps = (x_t - sa * x0) / sn
    else:
        x0 = sa * x_t - sn * pred
        eps = sn * x_t + sa * pred
    v = sa * eps - sn * x0
    return x0, eps, v


def ddim_step(x_t, eps, abar_t, abar_prev, eta=0.0, noise=None):
    abar_t, abar_prev, eta = float(abar_t), float(abar_prev), float(eta)
    if not 0.0 < abar_t < 1.0 or not 0.0 < abar_prev <= 1.0:
        raise ValueError("abar_t 应在 (0, 1) 内，abar_prev 应在 (0, 1] 内")
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta 应在 [0, 1] 内")
    if eta > 0.0:
        if abar_prev < abar_t:
            raise ValueError("eta 大于 0 时 abar_prev 不能小于 abar_t")
        if noise is None:
            raise ValueError("eta 大于 0 时必须传入 noise")
    x0 = (x_t - math.sqrt(1.0 - abar_t) * eps) / math.sqrt(abar_t)
    var = 0.0
    if eta > 0.0:
        var = eta * eta * (1.0 - abar_prev) / (1.0 - abar_t) * (1.0 - abar_t / abar_prev)
    out = math.sqrt(abar_prev) * x0 + math.sqrt(max(1.0 - abar_prev - var, 0.0)) * eps
    if var > 0.0:
        out = out + math.sqrt(var) * noise
    return out


def cfg_combine(eps_uncond, eps_cond, scale):
    if eps_uncond.shape != eps_cond.shape:
        raise ValueError("两个预测的形状应相同")
    if isinstance(scale, torch.Tensor):
        scale = scale.to(dtype=eps_uncond.dtype, device=eps_uncond.device)
    return eps_uncond + scale * (eps_cond - eps_uncond)


def _check_lora(weight, lora_a, lora_b):
    if weight.dim() != 2 or lora_a.dim() != 2 or lora_b.dim() != 2:
        raise ValueError("weight、lora_a、lora_b 都应是二维张量")
    out_f, in_f = weight.shape
    r = lora_a.shape[0]
    if r < 1 or lora_a.shape[1] != in_f or tuple(lora_b.shape) != (out_f, r):
        raise ValueError("lora_a 应为 (r, in)，lora_b 应为 (out, r)")
    return r


def lora_forward(x, weight, bias, lora_a, lora_b, alpha):
    r = _check_lora(weight, lora_a, lora_b)
    if x.shape[-1] != weight.shape[1]:
        raise ValueError("x 的最后一维应等于 in")
    y = x @ weight.t()
    if bias is not None:
        y = y + bias
    return y + (alpha / r) * ((x @ lora_a.t()) @ lora_b.t())


def lora_merge(weight, lora_a, lora_b, alpha):
    r = _check_lora(weight, lora_a, lora_b)
    return weight + (alpha / r) * (lora_b @ lora_a)


def inpaint_replace(x_t, known_x0, mask, abar_t, noise):
    abar_t = float(abar_t)
    if not 0.0 <= abar_t <= 1.0:
        raise ValueError("abar_t 应在 [0, 1] 内")
    if x_t.shape != known_x0.shape or x_t.shape != noise.shape:
        raise ValueError("x_t、known_x0、noise 的形状应相同")
    if bool((mask < 0).any()) or bool((mask > 1).any()):
        raise ValueError("mask 的取值应在 [0, 1] 内")
    m = mask.to(x_t.dtype)
    known_t = math.sqrt(abar_t) * known_x0 + math.sqrt(1.0 - abar_t) * noise
    return m * x_t + (1.0 - m) * known_t
