"""参考实现 23-1。学员完成本章前不要查看。"""

from __future__ import annotations

import numpy as np


def die_grid(wafer_diameter_mm, die_w, die_h, edge_exclusion_mm=3.0, offset=(0.0, 0.0)):
    R = wafer_diameter_mm / 2.0 - edge_exclusion_mm
    ox, oy = offset
    c0, c1 = int(np.floor((-R - ox) / die_w)) - 1, int(np.ceil((R - ox) / die_w)) + 1
    r0, r1 = int(np.floor((-R - oy) / die_h)) - 1, int(np.ceil((R - oy) / die_h)) + 1
    rows = []
    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            x0, y0 = ox + col * die_w, oy + row * die_h
            xs = np.array([x0, x0 + die_w, x0, x0 + die_w])
            ys = np.array([y0, y0, y0 + die_h, y0 + die_h])
            if np.all(xs * xs + ys * ys <= R * R):
                rows.append([col, row, x0 + die_w / 2.0, y0 + die_h / 2.0])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4)


def assign_to_die(points_xy, die_w, die_h, offset=(0.0, 0.0)):
    p = np.asarray(points_xy, dtype=np.float64)
    col = np.floor((p[:, 0] - offset[0]) / die_w)
    row = np.floor((p[:, 1] - offset[1]) / die_h)
    return np.stack([col, row], axis=1).astype(np.int64)


def radial_density(points_xy, radius_mm, bins):
    p = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    r = np.hypot(p[:, 0], p[:, 1])
    r = r[r <= radius_mm]
    edges = np.linspace(0.0, radius_mm, bins + 1)
    counts, _ = np.histogram(r, bins=edges)
    areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    return counts / areas


def edge_ratio(points_xy, radius_mm, edge_band_mm):
    p = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    r = np.hypot(p[:, 0], p[:, 1])
    r = r[r <= radius_mm]
    if len(r) == 0:
        return 0.0
    return float((r >= radius_mm - edge_band_mm).mean())
