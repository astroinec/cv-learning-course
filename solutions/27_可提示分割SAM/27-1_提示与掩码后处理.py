"""参考实现 27-1。学员完成本章前不要查看。"""

from __future__ import annotations

import math

import numpy as np
import torch


def point_position_encoding(points, image_size, freqs):
    h, w = image_size
    pts = torch.as_tensor(points, dtype=freqs.dtype, device=freqs.device)
    if pts.shape[-1] != 2 or freqs.shape[0] != 2:
        raise ValueError("points 的最后一维和 freqs 的第 0 维都应是 2")
    scale = torch.tensor([w, h], dtype=freqs.dtype, device=freqs.device)
    unit = (pts + 0.5) / scale
    proj = (2.0 * unit - 1.0) @ freqs * (2.0 * math.pi)
    return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def select_by_predicted_iou(masks, iou_pred):
    masks = np.asarray(masks)
    iou_pred = np.asarray(iou_pred, dtype=np.float64)
    if masks.ndim != 4 or iou_pred.shape != masks.shape[:2]:
        raise ValueError("masks 应为 (B, M, H, W)，iou_pred 应为 (B, M)")
    idx = iou_pred.argmax(axis=1).astype(np.int64)
    rows = np.arange(len(masks))
    return masks[rows, idx], idx, iou_pred[rows, idx]


def stability_score(logits, threshold=0.0, offset=1.0):
    logits = np.asarray(logits)
    high = (logits > threshold + offset).sum(axis=(-2, -1), dtype=np.int64)
    low = (logits > threshold - offset).sum(axis=(-2, -1), dtype=np.int64)
    return np.where(low > 0, high / np.maximum(low, 1), 0.0).astype(np.float64)


def mask_to_box(masks):
    m = np.asarray(masks).astype(bool)
    h, w = m.shape[-2:]
    cols, rows = m.any(axis=-2), m.any(axis=-1)
    x0, y0 = cols.argmax(axis=-1), rows.argmax(axis=-1)
    x1 = w - 1 - cols[..., ::-1].argmax(axis=-1)
    y1 = h - 1 - rows[..., ::-1].argmax(axis=-1)
    box = np.stack([x0, y0, x1, y1], axis=-1).astype(np.int64)
    box[~cols.any(axis=-1)] = 0
    return box


def mask_nms(masks, scores, iou_threshold=0.7):
    m = np.asarray(masks).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    if len(m) != len(s):
        raise ValueError("masks 和 scores 的个数要相同")
    if len(m) == 0:
        return np.zeros(0, dtype=np.int64)
    flat = m.reshape(len(m), -1)
    area = flat.sum(axis=1, dtype=np.int64)
    order = np.argsort(-s, kind="stable")
    keep: list[int] = []
    for i in order:
        if keep:
            inter = (flat[keep] & flat[i]).sum(axis=1, dtype=np.int64)
            union = area[keep] + area[i] - inter
            if (inter / np.maximum(union, 1) > iou_threshold).any():
                continue
        keep.append(int(i))
    return np.asarray(keep, dtype=np.int64)
