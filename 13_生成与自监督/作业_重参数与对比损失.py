"""第 13 章作业：VAE 重参数与 SimCLR 对比损失。"""

from __future__ import annotations

import torch


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, eps: torch.Tensor | None = None) -> torch.Tensor:
    """执行 z = mu + std * eps。

    mu/logvar: 同 shape 浮点张量，常见为 (B, latent_dim)。logvar 是方差的自然对数
    （不是标准差的对数），标准差 std 由它逐元素算出。
    eps: 可选同 shape 标准高斯噪声；测试时可显式传入以保证确定性。
      - 传了 eps 就必须用它，同样的输入调用两次结果相同。
      - eps 为 None 时在函数里采样：每个元素各自独立的标准高斯噪声（均值 0、标准差 1），
        每次调用重新采样。
    输出：与 mu 同 shape、device、dtype，梯度可传到 mu 与 logvar。不修改输入。
    """
    # TODO: std = exp(0.5 * logvar)；eps 为 None 时用 randn_like。
    raise NotImplementedError("TODO: 实现 reparameterize")


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """计算两个增强视图的对称 NT-Xent 损失。

    z1/z2: float tensor, shape (B, D)，第 i 行互为正样本。
    输出：有限标量（0 维）；先做 L2 归一化，排除自身相似度。梯度要能传回 z1 和 z2。不修改输入。
    本作业假定 B >= 2 且 temperature > 0。

    定义：
    1. 把 z1、z2 的 2B 个向量各自做 L2 归一化（每个向量除以自己的长度，沿特征那一维）。
    2. 两个向量的相似度 s = 归一化后的内积（余弦相似度）除以 temperature。
    3. 2B 个向量轮流当锚点。锚点的正样本是另一个视图里同一行的向量；
       候选是除锚点自己以外的全部 2B-1 个向量（包含正样本，也包含同一个视图里的其他向量）。
       这个锚点的损失 = -log( exp(s_正样本) / 全部候选的 exp(s) 之和 )。
    4. 对 2B 个锚点的损失取平均。
    由定义可知：交换 z1 和 z2 结果不变；把任意一行乘上一个正数结果不变。
    """
    # TODO: 构造 2B×2B 相似度矩阵、正样本索引与对称交叉熵。
    raise NotImplementedError("TODO: 实现 nt_xent_loss")


if __name__ == "__main__":
    print("完成重参数与 NT-Xent TODO 后运行检查器。")
