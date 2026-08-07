"""第 13 章作业：VAE 重参数与 SimCLR 对比损失。"""

from __future__ import annotations

import torch


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, eps: torch.Tensor | None = None) -> torch.Tensor:
    """执行 z = mu + std * eps。

    mu/logvar: 同 shape 浮点张量，常见为 (B, latent_dim)。
    eps: 可选同 shape 标准高斯噪声；测试时可显式传入以保证确定性。
    输出：与 mu 同 shape、device、dtype，梯度可传到 mu 与 logvar。
    """
    # TODO: std = exp(0.5 * logvar)；eps 为 None 时用 randn_like。
    raise NotImplementedError("TODO: 实现 reparameterize")


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """计算两个增强视图的对称 NT-Xent 损失。

    z1/z2: float tensor, shape (B, D)，第 i 行互为正样本。
    输出：有限标量；先做 L2 归一化，排除自身相似度。
    本作业假定 B >= 2 且 temperature > 0。
    """
    # TODO: 构造 2B×2B 相似度矩阵、正样本索引与对称交叉熵。
    raise NotImplementedError("TODO: 实现 nt_xent_loss")


if __name__ == "__main__":
    print("完成重参数与 NT-Xent TODO 后运行检查器。")
