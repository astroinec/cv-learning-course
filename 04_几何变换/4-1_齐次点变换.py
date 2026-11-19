"""作业 4-1：用齐次坐标把单应矩阵应用到二维点。"""

from __future__ import annotations

import numpy as np


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """将二维点经单应矩阵映射到新坐标。

    输入契约：
      points: (N,2) 数值数组，可能是整数也可能是浮点；N >= 1。
      homography: (3,3) 可转为 float64 的矩阵。右下角可以是 0，整体也可以是负的，
        不要假设它已经归一化。
    输出契约：
      (N,2) float64。两个输入先转成 float64 再计算。每个点按列向量 (x, y, 1)
      补齐次分量，左乘 homography 得到 (x', y', w)，两个坐标都除以第三分量 w。
      若某点的 |w| <= 1e-12（w 按传进来的矩阵直接算，不先缩放矩阵），该点两坐标
      均返回 np.nan，同一批里其余的点照常计算。w 为负的点是有效的。
      不修改传进来的数组。

    TODO：不得调用 cv2.perspectiveTransform。
    """
    raise NotImplementedError("TODO: 实现齐次坐标变换与安全透视除法")


def _demo() -> None:
    points = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float64)
    homography = np.array([[1, 0.2, 3], [0.1, 1, 2], [0.01, 0.02, 1]], dtype=np.float64)
    try:
        print(transform_points(points, homography))
    except NotImplementedError as exc:
        print(f"TODO: {exc}")


if __name__ == "__main__":
    _demo()
