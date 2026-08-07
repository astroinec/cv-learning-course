"""第 11 章作业：向量化 IoU 与类别感知 NMS。"""

from __future__ import annotations

import torch


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """计算两组 xyxy 连续坐标框的两两 IoU。

    boxes1: float tensor, shape (N, 4)
    boxes2: float tensor, shape (M, 4)
    输出：shape (N, M)，范围 [0, 1]；支持 N=0 或 M=0。
    框面积定义为 max(x2-x1, 0) * max(y2-y1, 0)，不加 1。
    """
    # TODO: 使用广播完成交集、并集与安全除法。
    raise NotImplementedError("TODO: 实现 pairwise_iou")


def class_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """执行类别感知贪心 NMS，返回保留框的原始索引。

    boxes: (N, 4)，scores: (N,)，labels: (N,)。
    输出：long tensor (K,)，按分数从高到低；空输入返回空 long tensor。
    只抑制同类别且 IoU 严格大于阈值的框。
    """
    # TODO: 每次保留当前最高分框，并只在同类别候选中应用 IoU 抑制。
    raise NotImplementedError("TODO: 实现 class_aware_nms")


if __name__ == "__main__":
    print("完成 pairwise_iou 和 class_aware_nms 后运行检查器。")
