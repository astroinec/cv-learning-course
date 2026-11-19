"""第 12 章作业：二分类 Dice loss 与 U-Net skip 对齐。"""

from __future__ import annotations

import torch


def binary_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """计算 batch 平均的软 Dice loss。

    logits: float tensor, shape (N, 1, H, W)，未做 sigmoid。
    target: float tensor, 同 shape，值位于 [0, 1]。
    输出：标量 tensor（0 维）；需可反向传播且有限。不修改输入。

    定义：先对 logits 做 sigmoid 得到概率 p。对每个样本，在它自己的全部像素上求和：
        dice = (2 * sum(p * target) + eps) / (sum(p) + sum(target) + eps)
    eps 在分子和分母上各加一次。损失是 1 减去各样本 dice 的平均值。
    不做硬判定（不和 0.5 比较），也不把整批的像素合在一起算。

    空 target 的约定：target 全为 0 时 dice = eps / (sum(p) + eps)，结果总是有限的，
    但只有 sum(p) 远小于 eps 时才接近 1。默认 eps=1e-6 很小：哪怕每个像素的 logit 都是 -12
    （概率约 6e-6），4x4 的图上 sum(p) 也有约 1e-4，这时 dice 约 0.01，损失约 0.99。
    也就是说，默认 eps 下“空 target 配上很干净的预测”的损失接近 1，不是 0；
    eps 只保证不出现 0/0。想让这种情况的损失接近 0，要把 eps 调大
    （同一个输入取 eps=1 时损失小于 0.001），或者在训练代码里单独处理空图。
    """
    # TODO: 对 logits 做 sigmoid，逐样本计算 Dice，再返回 1 - mean(dice)。
    raise NotImplementedError("TODO: 实现 binary_dice_loss")


def align_and_concat(decoder: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """将 decoder 居中裁剪或对称补零到 skip 的空间尺寸后沿通道拼接。

    decoder: shape (N, C1, H1, W1)
    skip: shape (N, C2, H2, W2)
    输出：shape (N, C1+C2, H2, W2)，前 C1 个通道来自 decoder，后 C2 个通道就是 skip。
    batch 不同时抛出 ValueError。不修改输入；梯度要能传回 decoder。

    高和宽分别处理，可能一个方向要裁、另一个方向要补：
    - decoder 比 skip 大：居中裁剪，两侧裁掉的行（列）数相同。
    - decoder 比 skip 小：对称补 0，两侧补的行（列）数相同。补的值只能是 0。
    - 尺寸差是奇数时两侧相差 1，多出来的 1 行（列）放在哪一侧都可以。
    只做裁剪和补零，不做插值缩放，decoder 的数值不能变。
    """
    # TODO: 同时处理 decoder 比 skip 大或小的情况，再 torch.cat(dim=1)。
    raise NotImplementedError("TODO: 实现 align_and_concat")


if __name__ == "__main__":
    print("完成 Dice 与 skip 对齐 TODO 后运行检查器。")
