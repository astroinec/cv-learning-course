"""作业 22-1：光学分辨率、PSF 与传感器噪声仿真。

只允许使用 NumPy。目标是能写出一个“理想图像 → 光学模糊 → 光子到电子 → 噪声 → 量化”
的成像仿真器，并用光子转移曲线验证它。
"""

from __future__ import annotations

import numpy as np


def rayleigh_resolution_um(wavelength_nm: float, na: float) -> float:
    """瑞利判据的横向分辨率（微米）：0.61 * lambda / NA。

    wavelength_nm 以纳米输入，返回微米，Python float。na 不在 (0, 1.5] 时抛出 ValueError
    （1.5 以上的 NA 不符合常见物镜，通常是单位写错了）。
    """
    raise NotImplementedError("TODO: 实现瑞利分辨率")


def gaussian_psf(sigma: float, radius: int | None = None) -> np.ndarray:
    """二维高斯 PSF，(2r+1, 2r+1) float64，和为 1，中心在正中。

    radius 默认取 ceil(3 * sigma)。sigma <= 0 时抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现高斯 PSF")


def convolve_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """二维卷积，输出与输入同尺寸，边界用反射填充（numpy 的 mode="reflect"）。

    kernel 尺寸为奇数。注意是卷积（核要翻转），对称核时与相关相同。
    可以用直接求和或 FFT，结果误差需小于 1e-9。
    """
    raise NotImplementedError("TODO: 实现同尺寸卷积")


def simulate_capture(
    mean_electrons: np.ndarray,
    read_noise_e: float,
    gain_e_per_adu: float,
    bit_depth: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """传感器成像：泊松散粒噪声 + 高斯读出噪声 + 增益 + 量化 + 饱和。

    electrons = Poisson(mean_electrons) + Normal(0, read_noise_e)
    adu = round(electrons / gain_e_per_adu)，再裁剪到 [0, 2**bit_depth - 1]
    返回 int64 数组。所有随机数都必须来自传入的 rng。
    """
    raise NotImplementedError("TODO: 实现传感器噪声模型")


def estimate_gain_from_ptc(means_adu: np.ndarray, variances_adu: np.ndarray) -> tuple[float, float]:
    """光子转移曲线（PTC）拟合。

    在未饱和区，方差_adu = mean_adu / gain + read_noise_e**2 / gain**2。
    对 (mean, variance) 做最小二乘直线拟合 variance = a * mean + b，
    返回 (gain_e_per_adu, read_noise_e) = (1 / a, sqrt(max(b, 0)) / a)，均为 Python float。
    """
    raise NotImplementedError("TODO: 实现光子转移曲线拟合")
