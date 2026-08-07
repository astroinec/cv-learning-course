"""作业 7-1：从零实现无畸变针孔相机投影。"""

from __future__ import annotations

import numpy as np


def project_points(
    points_3d: np.ndarray,
    camera_matrix: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """把世界点投影到像素平面。

    输入契约：points_3d 为 (N,3)；camera_matrix/rotation 为 (3,3)；
    translation 可为 (3,) 或 (3,1)。输入均可转为 float64。
    输出契约：(N,2) float64。先计算 Pc = R @ Pw + t，再做透视除法并乘 K。
    对相机坐标深度 Z <= 1e-12 的点，输出 [nan, nan]。

    本题忽略镜头畸变。不得调用 cv2.projectPoints 完成核心实现。
    """
    raise NotImplementedError("TODO: 实现世界到相机再到像素的针孔投影")


def _demo() -> None:
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    camera = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    try:
        print(project_points(points, camera, np.eye(3), np.array([0, 0, 5])))
    except NotImplementedError as exc:
        print(f"TODO: {exc}")


if __name__ == "__main__":
    _demo()
