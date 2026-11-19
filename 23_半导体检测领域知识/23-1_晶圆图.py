"""作业 23-1：晶圆 die 网格、缺陷归属与空间分布。

只允许使用 NumPy。坐标约定：晶圆中心为原点 (0, 0)，单位毫米，x 向右、y 向上。
die 网格由 die 尺寸和网格偏移 offset 决定：第 (col, row) 个 die 的左下角为
(offset_x + col * die_w, offset_y + row * die_h)。

点坐标 points_xy 都是形状 (N, 2) 的数组，每行是 (x, y)，N 可以为 0。
所有函数都不能修改传入的数组。
"""

from __future__ import annotations

import numpy as np


def die_grid(
    wafer_diameter_mm: float,
    die_w: float,
    die_h: float,
    edge_exclusion_mm: float = 3.0,
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """列出完整落在可用区域内的 die。

    可用区域是半径 R = wafer_diameter_mm / 2 - edge_exclusion_mm 的圆。
    一个 die 四个角点到圆心的距离都 <= R 才算完整 die，角点正好落在圆上也算。
    返回 (K, 4) float64，每行 [col, row, center_x, center_y]，
    按 row 升序、row 内 col 升序排列。col 和 row 可以为负。
    center 是这个 die 的几何中心，也就是左下角加上半个 die_w 和半个 die_h。
    offset 可以是任意实数，不限于一个 die 的范围内：把 offset 平移整数个 die，
    晶圆上完整 die 的位置不变，只是 col、row 的编号跟着变。
    一个完整 die 都放不下时返回形状 (0, 4) 的数组。
    """
    raise NotImplementedError("TODO: 实现 die 网格")


def assign_to_die(
    points_xy: np.ndarray,
    die_w: float,
    die_h: float,
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """返回每个缺陷点所在 die 的 (col, row)，形状 (N, 2) int64。

    落在 die 左边界和下边界上的点属于该 die（左闭右开、下闭上开）。
    col 只由 x、die_w 和 offset 的第一个分量决定，row 只由 y、die_h 和 offset 的第二个分量决定。
    没有点（N 为 0）时返回形状 (0, 2) 的 int64 数组。
    """
    raise NotImplementedError("TODO: 实现缺陷归属")


def radial_density(points_xy: np.ndarray, radius_mm: float, bins: int) -> np.ndarray:
    """按等宽圆环统计缺陷面密度（个 / mm^2）。

    把 [0, radius_mm] 等分成 bins 个圆环，第 i 环面积为 pi * (r_{i+1}^2 - r_i^2)。
    半径正好等于 radius_mm 的点算进最后一环，超出的点忽略。
    返回 (bins,) float64，没有点落在半径内时各环都是 0。
    用来看缺陷沿半径方向怎样分布：中心型、环型、边缘型的径向密度形状不同；
    它不含角度信息，哪些模式分不开见教程第 8.1 节。
    """
    raise NotImplementedError("TODO: 实现径向密度")


def edge_ratio(points_xy: np.ndarray, radius_mm: float, edge_band_mm: float) -> float:
    """落在外圈 [radius - edge_band, radius] 内的缺陷占全部（半径内）缺陷的比例。

    外圈是闭区间：到圆心的距离正好等于 radius - edge_band 或正好等于 radius 的点都算在外圈里。
    “半径内”指到圆心的距离 <= radius；超出半径的点既不算进分子，也不算进分母。
    半径内没有缺陷（包括输入为空）时返回 0.0。返回 Python float。
    """
    raise NotImplementedError("TODO: 实现边缘缺陷比例")
