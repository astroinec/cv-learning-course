"""第 11 章作业：向量化 IoU 与类别感知 NMS。"""

from __future__ import annotations

import torch


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """计算两组 xyxy 连续坐标框的两两 IoU。

    boxes1: float tensor, shape (N, 4)
    boxes2: float tensor, shape (M, 4)
    输出：float tensor, shape (N, M)，第 [i, j] 项是 boxes1[i] 和 boxes2[j] 的 IoU，范围 [0, 1]；
    支持 N=0 或 M=0（返回形状 (0, M)、(N, 0) 的空张量）。不修改输入。

    约定：
    - 框面积定义为 max(x2-x1, 0) * max(y2-y1, 0)，不加 1。
    - 不相交的框 IoU 为 0，包括只有一条边或一个角相接的情况。
    - 并集面积为 0 时（例如两个面积为 0 的框）该项为 0，不能是 NaN。
    - 并集大于 0 时结果要准确。坐标可能是像素，也可能是归一化到 0 到 1 的小数，
      后一种情况下框的面积可以小到 1e-6 量级。所以不要靠“给分母加一个固定的小数”来防止除零
      （加 1e-7 时，边长 1/512 的框和它自己的 IoU 只有 0.97）；验收允许的误差是 0.01。
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

    boxes: (N, 4)，scores: (N,)，labels: (N,)。输入的分数不一定排好序。不修改输入。
    输出：long tensor (K,)，里面是保留的框在输入里的下标（不是排序之后的位置），
    按分数从高到低排列；空输入返回形状 (0,) 的空 long tensor。

    贪心的含义：按分数从高到低依次处理每个框。一个框如果和某个已经保留的同类别框的 IoU
    严格大于 iou_threshold，就被抑制；否则保留。只有保留下来的框才能抑制别的框，
    已经被抑制的框不再抑制任何框。
    只抑制同类别的框；IoU 正好等于阈值时不抑制。分数相同的框之间的先后顺序不作要求。
    """
    # TODO: 每次保留当前最高分框，并只在同类别候选中应用 IoU 抑制。
    raise NotImplementedError("TODO: 实现 class_aware_nms")


if __name__ == "__main__":
    print("完成 pairwise_iou 和 class_aware_nms 后运行检查器。")
