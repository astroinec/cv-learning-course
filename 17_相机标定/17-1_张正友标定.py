"""作业 17-1：从零实现张正友标定的线性部分。

只允许使用 NumPy。可以在 Notebook 里用 cv2.calibrateCamera 对照，
但这里的核心实现不能调用 OpenCV 的标定、单应或 PnP 函数。
"""

from __future__ import annotations

import numpy as np


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley 归一化。

    输入 (N, 2)。输出 (points_n, T)：
    - T 为 (3, 3) float64 相似变换，使归一化点的质心在原点、到原点平均距离为 sqrt(2)
    - points_n = (T @ [x, y, 1]^T) 的前两维，形状 (N, 2) float64
    所有点重合（平均距离为 0）时抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现 Hartley 归一化")


def homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """归一化 DLT 估计单应 H，使 dst ~ H @ src（齐次）。

    输入两个 (N, 2)，N >= 4。输出 (3, 3) float64，并缩放到 H[2, 2] == 1。
    步骤：两组点分别归一化 → 每对点构造两行方程 → SVD 取最小奇异值对应向量
    → 去归一化 H = inv(T_dst) @ Hn @ T_src。
    点数不足 4 时抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现归一化 DLT")


def intrinsics_from_homographies(homographies: list[np.ndarray]) -> np.ndarray:
    """Zhang 闭式解：由至少 3 个棋盘平面的单应求内参 K。

    对每个 H = [h1 h2 h3]，构造
        v_ij = [hi1*hj1, hi1*hj2 + hi2*hj1, hi2*hj2,
                hi3*hj1 + hi1*hj3, hi3*hj2 + hi2*hj3, hi3*hj3]
    （hik 指第 i 列第 k 个元素）并叠加约束 v12^T b = 0 与 (v11 - v22)^T b = 0。
    SVD 求 b → 组成对称矩阵 B = lambda * K^-T K^-1 → 解出 fx, fy, cx, cy, skew。
    输出上三角 (3, 3) float64，K[2, 2] == 1，fx 和 fy 为正。
    少于 3 个单应时抛出 ValueError。

    提示：可以用 Zhang 论文附录的闭式公式，也可以对 B（必要时取负使其正定）
    做 Cholesky 分解得到 K^-1。
    """
    raise NotImplementedError("TODO: 实现 Zhang 闭式内参求解")


def extrinsics_from_homography(K: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """由内参和单应恢复外参 (R, t)。

    r1 = s K^-1 h1，r2 = s K^-1 h2，r3 = r1 x r2，t = s K^-1 h3，
    s = 1 / ||K^-1 h1||。若 t[2] < 0（棋盘在相机后方），整体取负号。
    最后用 SVD 把 [r1 r2 r3] 投影到最近的旋转矩阵（det = +1）。
    输出 R 为 (3, 3)，t 为 (3,)，均为 float64。
    """
    raise NotImplementedError("TODO: 实现外参恢复")


def reprojection_rmse(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    world_points: np.ndarray,
    observed_uv: np.ndarray,
) -> float:
    """重投影均方根误差（像素）。

    world_points 为 (N, 3)，observed_uv 为 (N, 2)。忽略畸变。
    返回 sqrt(mean(||proj - observed||^2))，Python float。
    """
    raise NotImplementedError("TODO: 实现重投影误差")
