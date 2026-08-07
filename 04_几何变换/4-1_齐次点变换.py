"""作业 4-1：用齐次坐标把单应矩阵应用到二维点。"""

from __future__ import annotations

import numpy as np


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """将二维点经单应矩阵映射到新坐标。

    输入契约：
      points: (N,2) 数值数组；N >= 1。
      homography: (3,3) 可转为 float64 的矩阵。
    输出契约：
      (N,2) float64。先补齐次分量 1，左乘 homography，再除以第三分量。
      若某点齐次第三分量绝对值 <= 1e-12，该点两坐标均返回 np.nan。

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
