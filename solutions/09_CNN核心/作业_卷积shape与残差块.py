"""参考实现（学员完成本章前不要查看）。"""

from __future__ import annotations

import torch
from torch import nn


def conv_output_size(size, kernel, stride=1, padding=0, dilation=1):
    out = (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    if out <= 0:
        raise ValueError("输出尺寸必须为正")
    return int(out)


class BasicResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(y + x)
