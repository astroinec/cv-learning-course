"""参考实现 31-1。学员完成本章前不要查看。"""

from __future__ import annotations

import math

import numpy as np
import torch


def _check_bits(num_bits):
    if isinstance(num_bits, bool) or int(num_bits) != num_bits or not 2 <= int(num_bits) <= 16:
        raise ValueError("num_bits 必须是 2 到 16 之间的整数")
    return int(num_bits)


def quant_params(x_min, x_max, num_bits=8, symmetric=False):
    bits = _check_bits(num_bits)
    x_min, x_max = float(x_min), float(x_max)
    if not (math.isfinite(x_min) and math.isfinite(x_max)) or x_min > x_max:
        raise ValueError("x_min、x_max 必须是有限数，且 x_min <= x_max")
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        qmin = -qmax
        bound = max(abs(x_min), abs(x_max))
        scale = bound / qmax if bound > 0 else 1.0
        return float(scale), 0, qmin, qmax
    qmin, qmax = 0, 2 ** bits - 1
    lo, hi = min(x_min, 0.0), max(x_max, 0.0)
    if hi == lo:
        return 1.0, 0, qmin, qmax
    scale = (hi - lo) / (qmax - qmin)
    zero_point = int(min(max(round(qmin - lo / scale), qmin), qmax))
    return float(scale), zero_point, qmin, qmax


def quantize_dequantize(x, scale, zero_point, qmin, qmax):
    x = np.asarray(x, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    zero_point = np.asarray(zero_point, dtype=np.int64)
    if np.any(scale <= 0):
        raise ValueError("scale 必须大于 0")
    q = np.clip(np.rint(x / scale) + zero_point, qmin, qmax).astype(np.int64)
    x_hat = (q - zero_point).astype(np.float64) * scale
    return q, x_hat


def quantize_weight_per_channel(w, num_bits=8):
    bits = _check_bits(num_bits)
    w = np.asarray(w, dtype=np.float64)
    if w.ndim < 2:
        raise ValueError("w 至少要有两维，第 0 维是输出通道")
    qmax = 2 ** (bits - 1) - 1
    bound = np.abs(w).reshape(w.shape[0], -1).max(axis=1)
    scales = np.where(bound > 0, bound / qmax, 1.0)
    shaped = scales.reshape((-1,) + (1,) * (w.ndim - 1))
    q = np.clip(np.rint(w / shaped), -qmax, qmax).astype(np.int64)
    return q, scales.astype(np.float64)


def fold_bn_into_conv(conv_weight, conv_bias, bn_weight, bn_bias, running_mean, running_var, eps=1e-5):
    out_ch = conv_weight.shape[0]
    vectors = [bn_weight, bn_bias, running_mean, running_var] + ([] if conv_bias is None else [conv_bias])
    if conv_weight.dim() < 2 or any(v.dim() != 1 or v.shape[0] != out_ch for v in vectors):
        raise ValueError("BN 的四个向量和卷积偏置的长度都应等于卷积的输出通道数")
    with torch.no_grad():
        k = bn_weight.double() / torch.sqrt(running_var.double() + eps)
        w = conv_weight.double() * k.reshape((-1,) + (1,) * (conv_weight.dim() - 1))
        b0 = torch.zeros(out_ch, dtype=torch.float64, device=conv_weight.device) if conv_bias is None else conv_bias.double()
        b = (b0 - running_mean.double()) * k + bn_bias.double()
    return w.to(conv_weight.dtype), b.to(conv_weight.dtype)


def plan_batch(latency_ms, takt_ms, deadline_ms, percentile=99.0):
    if not (takt_ms > 0 and deadline_ms > 0 and 0.0 <= percentile <= 100.0):
        raise ValueError("takt_ms、deadline_ms 必须大于 0，percentile 必须在 0 到 100 之间")
    if len(latency_ms) == 0:
        raise ValueError("latency_ms 不能为空")
    stats = {}
    for batch, samples in latency_ms.items():
        arr = np.asarray(samples, dtype=np.float64).reshape(-1)
        if isinstance(batch, bool) or int(batch) != batch or int(batch) < 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError("批大小必须是正整数，每个批大小至少要有一个有限的延迟样本")
        stats[int(batch)] = float(np.percentile(arr, percentile))
    best = 0
    for batch, lat in stats.items():
        if lat <= batch * takt_ms and (batch - 1) * takt_ms + lat <= deadline_ms:
            best = max(best, batch)
    return best, stats
