"""参考实现（学员完成本章前不要查看）。"""

from __future__ import annotations

import torch


def pairwise_iou(boxes1, boxes2):
    area = lambda b: (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area(boxes1)[:, None] + area(boxes2)[None, :] - inter
    return torch.where(union > 0, inter / union.clamp(min=1e-12), torch.zeros_like(inter))


def class_aware_nms(boxes, scores, labels, iou_threshold):
    order = torch.argsort(scores, descending=True)
    keep = []
    suppressed = torch.zeros(len(boxes), dtype=torch.bool)
    for i in order.tolist():
        if suppressed[i]:
            continue
        keep.append(i)
        ious = pairwise_iou(boxes[i:i + 1], boxes)[0]
        suppressed |= (labels == labels[i]) & (ious > iou_threshold)
    return torch.tensor(keep, dtype=torch.long)
