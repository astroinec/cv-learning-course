"""第 10 章作业：配置可训练层并归一化解释热图。"""

from __future__ import annotations

import torch
from torch import nn


def configure_trainable_parameters(model: nn.Module, trainable_prefixes: tuple[str, ...]) -> list[str]:
    """仅解冻名称匹配给定前缀的参数并返回可训练参数名。

    model: 任意 nn.Module。
    trainable_prefixes: 如 ("head",) 或 ("backbone.2", "head")。
    输出：按 model.named_parameters() 顺序排列的名称列表。
    契约：每个参数的 requires_grad 必须与返回列表一致。
    """
    # TODO: 先冻结全部参数，再按参数名前缀选择性解冻。
    raise NotImplementedError("TODO: 实现 configure_trainable_parameters")


def normalize_heatmap(cam: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """逐样本把 Grad-CAM 归一化到 [0, 1]。

    输入：float tensor，shape (N, 1, H, W)。
    输出：同 shape、同 device 的有限张量；常量热图应返回全 0。
    """
    # TODO: 先 ReLU，再对每个样本按空间维做 min-max 归一化。
    raise NotImplementedError("TODO: 实现 normalize_heatmap")


if __name__ == "__main__":
    print("完成冻结策略与热图归一化 TODO 后运行检查器。")
