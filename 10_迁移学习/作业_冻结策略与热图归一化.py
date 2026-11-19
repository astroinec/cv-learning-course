"""第 10 章作业：配置可训练层并归一化解释热图。"""

from __future__ import annotations

import torch
from torch import nn


def configure_trainable_parameters(model: nn.Module, trainable_prefixes: tuple[str, ...]) -> list[str]:
    """仅解冻名称匹配给定前缀的参数并返回可训练参数名。

    model: 任意 nn.Module。
    trainable_prefixes: 如 ("head",) 或 ("backbone.2", "head")。空元组表示全部冻结。
    匹配规则：参数名（model.named_parameters() 给出的名字）以某个前缀开头就算匹配。
        这是普通的字符串前缀，区分大小写：前缀 "head" 不匹配 "prehead.weight"（那只是包含），
        前缀 "1" 同时匹配 "1.weight" 和 "10.weight"（只想要第 1 层就写 "1."）。
    输出：按 model.named_parameters() 顺序排列的名称列表，和前缀的书写顺序无关，每个名字只出现一次；
        只含参数，不含 BatchNorm 的 running_mean 这类缓冲区。
    契约：每个参数的 requires_grad 必须与返回列表一致。
        这个函数会被反复调用（逐块解冻），每次都要把所有参数设成该有的状态：
        匹配的设为可训练，不匹配的设为冻结，不能假设模型之前是什么状态。
    """
    # TODO: 先冻结全部参数，再按参数名前缀选择性解冻。
    raise NotImplementedError("TODO: 实现 configure_trainable_parameters")


def normalize_heatmap(cam: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """逐样本把 Grad-CAM 归一化到 [0, 1]。

    输入：float tensor，shape (N, 1, H, W)。
    输出：同 shape、同 device 的有限张量；常量热图应返回全 0。
    步骤：先把负值截成 0（ReLU），再对每个样本单独做 min-max 归一化，
        也就是用这一张图 ReLU 之后的最小值和最大值，把最小值映到 0、最大值映到 1。
        样本之间互不影响：一张图单独算和放在一批里算，结果相同。
    “常量”按 ReLU 之后算：所有值相等的热图，以及所有值都不大于 0 的热图，都返回全 0。
    eps 只用来避免除以 0，不应明显改变正常热图的结果（最大值仍然映到 1）。
    不得原地修改输入 cam。
    """
    # TODO: 先 ReLU，再对每个样本按空间维做 min-max 归一化。
    raise NotImplementedError("TODO: 实现 normalize_heatmap")


if __name__ == "__main__":
    print("完成冻结策略与热图归一化 TODO 后运行检查器。")
