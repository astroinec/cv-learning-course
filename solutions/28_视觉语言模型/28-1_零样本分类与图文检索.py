"""参考实现 28-1。学员完成本章前不要查看。"""

from __future__ import annotations

import numpy as np


def _unit(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def build_zeroshot_classifier(text_embeds, eps=1e-12):
    t = np.asarray(text_embeds, dtype=np.float64)
    if t.ndim == 2:
        t = t[:, None, :]
    if t.ndim != 3 or 0 in t.shape:
        raise ValueError("text_embeds 应为 (C, T, D) 或 (C, D)，且每一维都不为 0")
    return _unit(_unit(t, eps).mean(axis=1), eps)


def zeroshot_predict(image_embeds, classifier, logit_scale=100.0):
    x = np.asarray(image_embeds, dtype=np.float64)
    w = np.asarray(classifier, dtype=np.float64)
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError("image_embeds 应为 (N, D)，classifier 应为 (C, D)，两者的 D 要相同")
    if not logit_scale > 0:
        raise ValueError("logit_scale 必须大于 0")
    logits = logit_scale * (_unit(x) @ w.T)
    z = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(z)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, logits.argmax(axis=1).astype(np.int64)


def recall_at_k(sim, positives, k):
    s = np.asarray(sim, dtype=np.float64)
    pos = np.asarray(positives).astype(bool)
    if s.ndim != 2 or pos.shape != s.shape:
        raise ValueError("sim 和 positives 应为形状相同的二维数组")
    if not 1 <= k <= s.shape[1]:
        raise ValueError("k 必须在 1 到候选个数之间")
    if not pos.any(axis=1).all():
        raise ValueError("有的查询没有任何正确答案")
    top = np.argsort(-s, axis=1, kind="stable")[:, :k]
    return float(np.take_along_axis(pos, top, axis=1).any(axis=1).mean())


def modality_gap(image_embeds, text_embeds):
    u = np.asarray(image_embeds, dtype=np.float64)
    v = np.asarray(text_embeds, dtype=np.float64)
    if u.ndim != 2 or v.ndim != 2 or u.shape[1] != v.shape[1] or len(u) == 0 or len(v) == 0:
        raise ValueError("两组向量应为 (N, D) 和 (M, D)，D 相同，且都不为空")
    u, v = _unit(u), _unit(v)
    cu, cv = u.mean(axis=0), v.mean(axis=0)
    gap = np.linalg.norm(cu - cv)
    spread_u = np.sqrt(((u - cu) ** 2).sum(axis=1).mean())
    spread_v = np.sqrt(((v - cv) ** 2).sum(axis=1).mean())
    return float(gap), float(spread_u), float(spread_v)


def score_regions(region_feats, query_embeds):
    r = np.asarray(region_feats, dtype=np.float64)
    q = np.asarray(query_embeds, dtype=np.float64)
    if r.ndim != 2 or q.ndim != 2 or r.shape[1] != q.shape[1] or len(r) == 0 or len(q) == 0:
        raise ValueError("region_feats 应为 (R, D)，query_embeds 应为 (Q, D)，D 相同，且都不为空")
    scores = _unit(r) @ _unit(q).T
    return scores, scores.argmax(axis=0).astype(np.int64), scores.max(axis=0)
