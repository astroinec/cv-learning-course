"""第 12 章作业：二分类 Dice loss 与 U-Net skip 对齐。"""

from __future__ import annotations

import torch


def binary_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """计算 batch 平均的软 Dice loss。

    logits: float tensor, shape (N, 1, H, W)，未做 sigmoid。
    target: float tensor, 同 shape，值位于 [0, 1]。
    输出：标量 tensor；需可反向传播且有限。
    空 target 与近空预测的约定由 eps 保证结果有限。
    """
    # TODO: 对 logits 做 sigmoid，逐样本计算 Dice，再返回 1 - mean(dice)。
    raise NotImplementedError("TODO: 实现 binary_dice_loss")


def align_and_concat(decoder: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """将 decoder 居中裁剪或对称补零到 skip 的空间尺寸后沿通道拼接。

    decoder: shape (N, C1, H1, W1)
    skip: shape (N, C2, H2, W2)
    输出：shape (N, C1+C2, H2, W2)。batch 必须相同。
    """
    # TODO: 同时处理 decoder 比 skip 大或小的情况，再 torch.cat(dim=1)。
    raise NotImplementedError("TODO: 实现 align_and_concat")


if __name__ == "__main__":
    print("完成 Dice 与 skip 对齐 TODO 后运行检查器。")
