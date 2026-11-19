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

    输入契约：points_3d 为 (N,3)，N >= 1；camera_matrix/rotation 为 (3,3)；
    translation 可为 (3,) 或 (3,1)。输入可能是整数数组，均可转为 float64。
    camera_matrix 形如 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]，没有倾斜项。
    rotation 和 translation 是世界到相机的变换。
    输出契约：(N,2) float64。输入先转成 float64 再计算：Pc = R @ Pw + t，
    再做透视除法并乘 K。
    对相机坐标深度 Z <= 1e-12 的点（看 Pc 的 Z，不是世界坐标的 Z），输出 [nan, nan]，
    同一批里其余的点照常计算。不修改传进来的数组。

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
