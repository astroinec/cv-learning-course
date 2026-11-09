"""参考实现（学员完成本章前不要查看）。"""

from __future__ import annotations

import numpy as np


def project_points(points_3d, camera_matrix, rotation, translation):
    P = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(camera_matrix, dtype=np.float64)
    R = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64).reshape(1, 3)
    pc = P @ R.T + t
    out = np.full((len(P), 2), np.nan)
    ok = pc[:, 2] > 1e-12
    uv = pc[ok] @ K.T
    out[ok] = uv[:, :2] / uv[:, 2:3]
    return out
