"""参考实现 19-1。学员完成本章前不要查看。"""

from __future__ import annotations

import numpy as np


def pairwise_sq_dists(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    return np.maximum(d, 0.0)


def greedy_coreset(features, m, start=0):
    x = np.asarray(features, dtype=np.float64)
    n = len(x)
    if m < 1 or m > n:
        raise ValueError("m 必须在 1 到点数之间")
    chosen = [int(start)]
    min_d = pairwise_sq_dists(x, x[start:start + 1])[:, 0]
    for _ in range(m - 1):
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        min_d = np.minimum(min_d, pairwise_sq_dists(x, x[nxt:nxt + 1])[:, 0])
    return np.asarray(chosen, dtype=np.int64)


def knn_scores(memory, queries, k=1):
    memory = np.asarray(memory, dtype=np.float64)
    if k < 1 or k > len(memory):
        raise ValueError("k 必须在 1 到 memory 大小之间")
    d = np.sqrt(pairwise_sq_dists(queries, memory))
    nearest = np.sort(d, axis=1)[:, :k]
    return nearest.mean(axis=1)


def auroc(scores, labels):
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("需要同时有正常和异常样本")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def threshold_at_escape(normal_scores, defect_scores, max_escape):
    normal = np.asarray(normal_scores, dtype=np.float64)
    defect = np.asarray(defect_scores, dtype=np.float64)
    best = None
    for thr in np.unique(defect):
        esc = float((defect < thr).mean())
        if esc <= max_escape + 1e-12 and (best is None or thr > best):
            best = float(thr)
    if best is None:
        best = float(defect.min())
    return best, float((defect < best).mean()), float((normal >= best).mean())
