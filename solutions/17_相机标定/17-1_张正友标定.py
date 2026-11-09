"""参考实现 17-1。学员完成本章前不要查看。"""

from __future__ import annotations

import numpy as np


def normalize_points(points):
    pts = np.asarray(points, dtype=np.float64)
    c = pts.mean(axis=0)
    d = np.linalg.norm(pts - c, axis=1).mean()
    if d < 1e-12:
        raise ValueError("所有点重合，无法归一化")
    s = np.sqrt(2) / d
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    homo = np.hstack([pts, np.ones((len(pts), 1))]) @ T.T
    return homo[:, :2] / homo[:, 2:3], T


def homography_dlt(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("至少需要 4 对点")
    sn, Ts = normalize_points(src)
    dn, Td = normalize_points(dst)
    rows = []
    for (x, y), (u, v) in zip(sn, dn):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def _v(H, i, j):
    hi, hj = H[:, i], H[:, j]
    return np.array([
        hi[0] * hj[0],
        hi[0] * hj[1] + hi[1] * hj[0],
        hi[1] * hj[1],
        hi[2] * hj[0] + hi[0] * hj[2],
        hi[2] * hj[1] + hi[1] * hj[2],
        hi[2] * hj[2],
    ])


def intrinsics_from_homographies(homographies):
    if len(homographies) < 3:
        raise ValueError("至少需要 3 个视图")
    V = []
    for H in homographies:
        V.append(_v(H, 0, 1))
        V.append(_v(H, 0, 0) - _v(H, 1, 1))
    _, _, vt = np.linalg.svd(np.asarray(V))
    b11, b12, b22, b13, b23, b33 = vt[-1]
    v0 = (b12 * b13 - b11 * b23) / (b11 * b22 - b12 ** 2)
    lam = b33 - (b13 ** 2 + v0 * (b12 * b13 - b11 * b23)) / b11
    alpha = np.sqrt(lam / b11)
    beta = np.sqrt(lam * b11 / (b11 * b22 - b12 ** 2))
    gamma = -b12 * alpha ** 2 * beta / lam
    u0 = gamma * v0 / beta - b13 * alpha ** 2 / lam
    return np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1.0]])


def extrinsics_from_homography(K, H):
    Kinv = np.linalg.inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    s = 1.0 / np.linalg.norm(Kinv @ h1)
    r1 = s * Kinv @ h1
    r2 = s * Kinv @ h2
    t = s * Kinv @ h3
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    U, _, Vt = np.linalg.svd(np.stack([r1, r2, r3], axis=1))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R, t.astype(np.float64)


def reprojection_rmse(K, R, t, world_points, observed_uv):
    pc = np.asarray(world_points, dtype=np.float64) @ R.T + np.asarray(t).reshape(1, 3)
    uv = pc @ K.T
    proj = uv[:, :2] / uv[:, 2:3]
    return float(np.sqrt(np.mean(np.sum((proj - observed_uv) ** 2, axis=1))))
