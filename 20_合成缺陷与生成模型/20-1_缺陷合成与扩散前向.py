"""作业 20-1：程序化缺陷合成与扩散模型的前向过程。

只允许使用 NumPy。这一章的重点是“生成的数据要带准确标签、要能证明有用”，
不是把生成模型训到多好看。

对所有函数都成立的约定（验收会检查）：不改动传入的数组，结果写到新数组里返回。
"""

from __future__ import annotations

import numpy as np


def scratch_mask(
    shape: tuple[int, int],
    p0: tuple[float, float],
    p1: tuple[float, float],
    width: float,
    softness: float = 1.0,
) -> np.ndarray:
    """生成一条划痕的软掩码。

    像素中心坐标为 (y, x)，整数下标即中心。对每个像素算它到线段 p0-p1 的最短距离 d。
    掩码 m = clip((width / 2 + softness / 2 - d) / softness, 0, 1)：
    线段中心附近为 1，边缘在 softness 宽度内线性过渡到 0。
    softness 必须 > 0（等于 0 和小于 0 都抛出 ValueError）。p0 == p1 时退化为一个圆点。
    线段可以是任意方向，包括竖直（不要用斜率求垂足）；p0、p1 对调结果不变。
    返回 (H, W) float64。
    """
    raise NotImplementedError("TODO: 实现点到线段距离与软掩码")


def blend(background: np.ndarray, foreground: np.ndarray | float, mask: np.ndarray) -> np.ndarray:
    """按掩码混合：out = background * (1 - mask) + foreground * mask。

    foreground 可以是同形状数组，也可以是标量。mask 必须在 [0, 1]，
    有小于 0 或大于 1 的值都抛出 ValueError。
    background 可以是任何数值类型（例如 uint8 图像）。返回新的 float64 数组，
    不做取整和裁剪（交给调用方决定位深），也不转回 background 的数据类型。
    """
    raise NotImplementedError("TODO: 实现软掩码混合")


def linear_beta_schedule(steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> np.ndarray:
    """DDPM 线性噪声调度，返回 (steps,) float64，首尾分别等于 beta_start 和 beta_end，中间等间隔。

    注意：默认的 beta 范围（1e-4 到 0.02）是给 steps=1000 配的。只把 steps 改小、范围不变，
    最后一步的 alpha_bar 离 0 还很远（100 步时约 0.36），x_T 不是纯噪声，
    “T 步之后变成纯噪声”就不成立了。步数改小时 beta 范围要跟着放大，
    常见做法是两端都乘 1000 / steps（讲义第 5.3.2 节有实测）。本函数不替你做这件事，
    只按传入的 beta_start、beta_end 生成。
    """
    raise NotImplementedError("TODO: 实现线性 beta 调度")


def alpha_bars(betas: np.ndarray) -> np.ndarray:
    """累计乘积 alpha_bar_t = prod_{s<=t} (1 - beta_s)，返回同形状 float64。"""
    raise NotImplementedError("TODO: 实现 alpha_bar")


def q_sample(x0: np.ndarray, t: int, abar: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """DDPM 前向加噪：x_t = sqrt(abar[t]) * x0 + sqrt(1 - abar[t]) * noise。

    t 为 0 起始的整数下标，合法范围是 0 到 len(abar) - 1。
    越界时抛出 IndexError，t 为负数也算越界（不能像 Python 下标那样从末尾数）。
    """
    raise NotImplementedError("TODO: 实现前向加噪")


def reparameterize(mu: np.ndarray, logvar: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """VAE 重参数化：z = mu + exp(0.5 * logvar) * eps。logvar 是方差的对数。"""
    raise NotImplementedError("TODO: 实现重参数化")
