"""作业 22-1：光学分辨率、PSF 与传感器噪声仿真。

只允许使用 NumPy。目标是能写出一个“理想图像 → 光学模糊 → 光子到电子 → 噪声 → 量化”
的成像仿真器，并用光子转移曲线验证它。

所有函数都不能修改传入的数组。
"""

from __future__ import annotations

import numpy as np


def rayleigh_resolution_um(wavelength_nm: float, na: float) -> float:
    """瑞利判据的横向分辨率（微米）：0.61 * lambda / NA。

    wavelength_nm 以纳米输入，返回微米，Python float。na 不在 (0, 1.5] 时抛出 ValueError
    （1.5 以上的 NA 不符合常见物镜，通常是单位写错了）。区间左开右闭：na 等于 1.5 是允许的，
    na 为 0 或负数要抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现瑞利分辨率")


def gaussian_psf(sigma: float, radius: int | None = None) -> np.ndarray:
    """二维高斯 PSF，(2r+1, 2r+1) float64，和为 1，中心在正中。

    sigma 是高斯的标准差，单位是像素。核的每个元素取高斯函数在该像素中心处的值（不做像素内积分），
    最后整体除以总和。radius 默认取 ceil(3 * sigma)；给了 radius 就用给定的值。
    sigma <= 0 时抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现高斯 PSF")


def convolve_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """二维卷积，输出与输入同尺寸，边界用反射填充（numpy 的 mode="reflect"）。

    kernel 的行数和列数都是奇数，两者可以不同（例如 1x5、3x1）。
    注意是卷积（核要上下、左右都翻转），对称核时与相关相同。
    image 可以是整数或浮点数组，返回 float64。
    可以用直接求和或 FFT，结果误差需小于 1e-9。
    """
    raise NotImplementedError("TODO: 实现同尺寸卷积")


def simulate_capture(
    mean_electrons: np.ndarray,
    read_noise_e: float,
    gain_e_per_adu: float,
    bit_depth: int,
    rng: np.random.Generator,
    black_level_adu: float = 0.0,
) -> np.ndarray:
    """传感器成像：泊松散粒噪声 + 高斯读出噪声 + 增益 + 黑电平 + 量化 + 饱和。

    electrons = Poisson(mean_electrons) + Normal(0, read_noise_e)
    adu = round(electrons / gain_e_per_adu + black_level_adu)，再裁剪到 [0, 2**bit_depth - 1]
    mean_electrons 是任意形状的非负数组，返回同形状的 int64 数组。
    所有随机数都必须来自传入的 rng：种子相同结果相同，换一个种子、或者用同一个 rng 接着再调用一次，结果不同。

    black_level_adu（黑电平）是量化之前加上的固定偏置，单位是 ADU。为什么需要它：
    读出噪声有正有负，信号很暗时 electrons 有一半左右是负数，而 ADU 不能为负，负的部分被裁成 0，
    于是暗处的均值被抬高、标准差被压小。例如读出噪声 8 e-、增益 1.5 的全暗帧，标准差应约 5.3 ADU，
    不加黑电平时只有约 3.1 ADU。相机用黑电平把整个分布抬离 0 来避免这种裁剪。

    默认值取 0：不传这个参数时 ADU 和电子数成正比（1000 e-、增益 2 得到约 500 ADU），
    做 PTC 时均值不用先减偏置；信号比读出噪声大几倍以上时，裁剪可以忽略。
    要用到暗处的统计量（暗帧、很弱的信号、PTC 里的全暗档）时，给一个黑电平，
    取读出噪声换算成 ADU 之后的 5 倍以上就够，统计时再把它从均值里减掉。
    黑电平占用量程：上限仍然是 2**bit_depth - 1，能记录的最大信号相应变小。
    """
    raise NotImplementedError("TODO: 实现传感器噪声模型")


def estimate_gain_from_ptc(means_adu: np.ndarray, variances_adu: np.ndarray) -> tuple[float, float]:
    """光子转移曲线（PTC）拟合。

    在未饱和区，方差_adu = mean_adu / gain + read_noise_e**2 / gain**2。
    这里的 mean_adu 不含黑电平：拍的时候用了黑电平，要先从均值里减掉再传进来。
    对全部 (mean, variance) 点做普通最小二乘直线拟合 variance = a * mean + b（各点权重相同，不加权；
    教程第 6.2 节讲的加权拟合是另一种做法，不写在这个函数里），
    返回 (gain_e_per_adu, read_noise_e) = (1 / a, sqrt(max(b, 0)) / a)，均为 Python float。
    截距 b 为负时读出噪声返回 0.0。
    """
    raise NotImplementedError("TODO: 实现光子转移曲线拟合")
