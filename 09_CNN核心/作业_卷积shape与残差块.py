"""第 9 章作业：卷积输出尺寸与残差块。"""

from __future__ import annotations

import torch
from torch import nn


def conv_output_size(size: int, kernel: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
    """计算单个空间维度的卷积输出长度。

    输入均为正整数或非负 padding；输出为 Python int。
    公式需包含 dilation，并使用向下取整。
    """
    # TODO: 实现通用卷积输出尺寸公式，并拒绝非正输出。
    raise NotImplementedError("TODO: 实现 conv_output_size")


class BasicResidualBlock(nn.Module):
    """保持空间尺寸的基础残差块。

    输入：x shape (N, channels, H, W)，float tensor。
    输出：与 x 完全相同 shape；梯度必须能传回输入。
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 实现 conv-bn-relu-conv-bn、残差相加和最终 ReLU。
        raise NotImplementedError("TODO: 实现 BasicResidualBlock.forward")


if __name__ == "__main__":
    print("完成 TODO 后运行：python tests/check_progress.py")
