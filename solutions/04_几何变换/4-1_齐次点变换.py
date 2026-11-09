"""参考实现（学员完成本章前不要查看）。"""

from __future__ import annotations

import numpy as np


def transform_points(points, homography):
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    Hm = np.asarray(homography, dtype=np.float64)
    q = np.hstack([p, np.ones((len(p), 1))]) @ Hm.T
    out = np.full((len(p), 2), np.nan)
    ok = np.abs(q[:, 2]) > 1e-12
    out[ok] = q[ok, :2] / q[ok, 2:3]
    return out
