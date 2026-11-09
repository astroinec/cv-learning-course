"""作业 20-1：程序化缺陷合成与扩散模型的前向过程。

只允许使用 NumPy。这一章的重点是“生成的数据要带准确标签、要能证明有用”，
不是把生成模型训到多好看。
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
    softness 必须 > 0，否则抛出 ValueError。p0 == p1 时退化为一个圆点。
    返回 (H, W) float64。
    """
    raise NotImplementedError("TODO: 实现点到线段距离与软掩码")


def blend(background: np.ndarray, foreground: np.ndarray | float, mask: np.ndarray) -> np.ndarray:
    """按掩码混合：out = background * (1 - mask) + foreground * mask。

    foreground 可以是同形状数组，也可以是标量。mask 必须在 [0, 1]，否则抛出 ValueError。
    返回 float64，不做取整和裁剪（交给调用方决定位深）。
    """
    raise NotImplementedError("TODO: 实现软掩码混合")


def linear_beta_schedule(steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> np.ndarray:
    """DDPM 线性噪声调度，返回 (steps,) float64，首尾分别等于 beta_start 和 beta_end。"""
    raise NotImplementedError("TODO: 实现线性 beta 调度")


def alpha_bars(betas: np.ndarray) -> np.ndarray:
    """累计乘积 alpha_bar_t = prod_{s<=t} (1 - beta_s)，返回同形状 float64。"""
    raise NotImplementedError("TODO: 实现 alpha_bar")


def q_sample(x0: np.ndarray, t: int, abar: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """DDPM 前向加噪：x_t = sqrt(abar[t]) * x0 + sqrt(1 - abar[t]) * noise。

    t 为 0 起始的整数下标，越界时抛出 IndexError。
    """
    raise NotImplementedError("TODO: 实现前向加噪")


def reparameterize(mu: np.ndarray, logvar: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """VAE 重参数化：z = mu + exp(0.5 * logvar) * eps。"""
    raise NotImplementedError("TODO: 实现重参数化")
