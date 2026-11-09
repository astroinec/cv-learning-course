"""作业 21-1：3D 变换与软件光栅化（z-buffer）。

只允许使用 NumPy。约定与 OpenGL 一致：右手坐标系，相机看向 -Z，
NDC 范围 [-1, 1]，深度映射到 [0, 1]（近平面 0，远平面 1）。
"""

from __future__ import annotations

import numpy as np


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """构造视图矩阵 V（4x4 float64），把世界坐标变换到相机坐标。

    f = normalize(target - eye)，s = normalize(f x up)，u = s x f。
    V 的旋转部分三行依次为 s、u、-f，平移部分为 -R @ eye。
    up 与视线平行时抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现 look_at")


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """OpenGL 风格透视投影矩阵（4x4 float64）。

    f = 1 / tan(fovy / 2)
    P = [[f/aspect, 0, 0, 0],
         [0, f, 0, 0],
         [0, 0, (far+near)/(near-far), 2*far*near/(near-far)],
         [0, 0, -1, 0]]
    near <= 0 或 far <= near 时抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现透视投影矩阵")


def to_screen(points: np.ndarray, mvp: np.ndarray, width: int, height: int) -> np.ndarray:
    """把世界点 (N, 3) 经 MVP 变换到屏幕坐标 (N, 3) float64。

    步骤：齐次坐标 → clip = mvp @ p → NDC = clip[:3] / clip[3]
    x_pix = (ndc_x + 1) / 2 * width，y_pix = (1 - ndc_y) / 2 * height（y 向下），
    depth = (ndc_z + 1) / 2。
    clip w <= 0（在相机后方）的点输出全 NaN。
    """
    raise NotImplementedError("TODO: 实现 MVP 到屏幕坐标")


def render_depth(triangles: np.ndarray, width: int, height: int) -> np.ndarray:
    """z-buffer 光栅化，只输出深度图。

    triangles: (T, 3, 3)，每个三角形三个顶点的 (x_pix, y_pix, depth)。
    在像素中心 (x + 0.5, y + 0.5) 采样；用重心坐标判断是否在三角形内
    （三个重心坐标都 >= 0，边界上算在内），在内则插值深度，保留更小的深度。
    两种顶点绕序都要支持；面积为 0 的三角形跳过。
    返回 (height, width) float64，没有被覆盖的像素为 np.inf。
    """
    raise NotImplementedError("TODO: 实现 z-buffer 光栅化")
