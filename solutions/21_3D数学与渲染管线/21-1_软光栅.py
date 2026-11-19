"""参考实现 21-1。学员完成本章前不要查看。"""

from __future__ import annotations

import numpy as np


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("零向量无法归一化")
    return v / n


def look_at(eye, target, up):
    eye = np.asarray(eye, dtype=np.float64)
    f = _normalize(np.asarray(target, dtype=np.float64) - eye)
    s = np.cross(f, np.asarray(up, dtype=np.float64))
    if np.linalg.norm(s) < 1e-12:
        raise ValueError("up 与视线平行")
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    R = np.stack([s, u, -f])
    V = np.eye(4)
    V[:3, :3] = R
    V[:3, 3] = -R @ eye
    return V


def perspective(fovy_deg, aspect, near, far):
    if near <= 0 or far <= near:
        raise ValueError("需要 0 < near < far")
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2.0)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
        [0, 0, -1, 0],
    ], dtype=np.float64)


def to_screen(points, mvp, width, height):
    p = np.hstack([np.asarray(points, dtype=np.float64), np.ones((len(points), 1))])
    clip = p @ np.asarray(mvp, dtype=np.float64).T
    out = np.full((len(p), 3), np.nan)
    ok = clip[:, 3] > 1e-12
    ndc = clip[ok, :3] / clip[ok, 3:4]
    out[ok, 0] = (ndc[:, 0] + 1) / 2 * width
    out[ok, 1] = (1 - ndc[:, 1]) / 2 * height
    out[ok, 2] = (ndc[:, 2] + 1) / 2
    return out


def render_depth(triangles, width, height):
    depth = np.full((height, width), np.inf)
    for tri in np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3):
        if not np.isfinite(tri).all():
            continue
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = tri
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue
        # 包围盒：只有中心落在三角形外接矩形里的像素才可能被覆盖，裁到画面内
        xa, xb = max(0, int(np.floor(min(x0, x1, x2)))), min(width, int(np.ceil(max(x0, x1, x2))))
        ya, yb = max(0, int(np.floor(min(y0, y1, y2)))), min(height, int(np.ceil(max(y0, y1, y2))))
        if xa >= xb or ya >= yb:
            continue
        ys, xs = np.mgrid[ya:yb, xa:xb]
        px, py = xs + 0.5, ys + 0.5
        w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / area
        w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-12) & (w1 >= -1e-12) & (w2 >= -1e-12)
        z = w0 * z0 + w1 * z1 + w2 * z2
        sub = depth[ya:yb, xa:xb]
        upd = inside & (z < sub)
        sub[upd] = z[upd]
    return depth
