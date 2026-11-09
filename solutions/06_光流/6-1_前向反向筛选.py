"""参考实现（学员完成本章前不要查看）。"""

from __future__ import annotations

import numpy as np


def filter_tracks(previous_points, forward_points, backward_points, status, max_fb_error=1.0):
    prev = np.asarray(previous_points, dtype=np.float64).reshape(-1, 2)
    fwd = np.asarray(forward_points, dtype=np.float64).reshape(-1, 2)
    bwd = np.asarray(backward_points, dtype=np.float64).reshape(-1, 2)
    st = np.asarray(status).reshape(-1) != 0
    finite = np.isfinite(fwd).all(1) & np.isfinite(bwd).all(1) & np.isfinite(prev).all(1)
    err = np.linalg.norm(prev - bwd, axis=1)
    return (st & finite & (err <= max_fb_error)).astype(bool)
