"""参考实现 18-1。学员完成本章前不要查看。"""

from __future__ import annotations

from collections import deque

import numpy as np


def robust_golden(stack, min_sigma=1.0):
    s = np.asarray(stack, dtype=np.float64)
    if s.ndim != 3 or s.shape[0] < 3:
        raise ValueError("至少需要 3 张良品")
    median = np.median(s, axis=0)
    mad = np.median(np.abs(s - median), axis=0)
    sigma = np.maximum(1.4826 * mad, min_sigma)
    return median, sigma


def label_components(mask, connectivity=8):
    if connectivity == 8:
        offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    elif connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        raise ValueError("connectivity 只能是 4 或 8")
    m = np.asarray(mask).astype(bool)
    h, w = m.shape
    labels = np.zeros((h, w), dtype=np.int32)
    n = 0
    for y in range(h):
        for x in range(w):
            if not m[y, x] or labels[y, x]:
                continue
            n += 1
            labels[y, x] = n
            q = deque([(y, x)])
            while q:
                cy, cx = q.popleft()
                for dy, dx in offsets:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = n
                        q.append((ny, nx))
    return labels, n


def detect_defects(image, median, sigma, k=5.0, min_area=3):
    z = np.abs(np.asarray(image, dtype=np.float64) - median) / sigma
    labels, n = label_components(z > k, 8)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labels == i)
        if len(ys) < min_area:
            continue
        out.append({
            "bbox": (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1),
            "area": int(len(ys)),
            "centroid": (float(ys.mean()), float(xs.mean())),
            "max_z": float(z[ys, xs].max()),
        })
    return out


def match_defects(predicted, truth, radius=3.0):
    pairs = []
    for i, p in enumerate(predicted):
        for j, t in enumerate(truth):
            d = float(np.hypot(p[0] - t[0], p[1] - t[1]))
            if d <= radius:
                pairs.append((d, i, j))
    pairs.sort()
    used_p, used_t = set(), set()
    for _, i, j in pairs:
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
    tp = len(used_p)
    return {
        "tp": tp,
        "fp": len(predicted) - tp,
        "fn": len(truth) - tp,
        "capture_rate": tp / len(truth) if truth else 1.0,
    }
