"""第 8 章作业：只用 NumPy 完成两层分类网络的关键计算。"""

from __future__ import annotations

import numpy as np


def two_layer_forward(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 ReLU 隐藏层与 logits。

    输入契约：
        x:  float array, shape (B, D)
        w1: float array, shape (D, H)
        b1: float array, shape (H,)
        w2: float array, shape (H, C)
        b2: float array, shape (C,)
    输出契约（按 hidden、logits 的顺序返回二元组）：
        hidden: shape (B, H)，第一层仿射变换（含偏置 b1）经过 ReLU 之后的值，元素非负
        logits: shape (B, C)，第二层仿射变换（含偏置 b2）的直接结果，不要做 softmax，也不要再过 ReLU
    其他约定：
        权重按“输入维度在前”排列，和 nn.Linear 的 weight 相反（见教程 8.2 节的提示框）。
        每个样本独立计算，B 可以是 1。
        不得原地修改任何输入数组。
    """
    # TODO: 先做仿射变换与 ReLU，再计算分类 logits。
    raise NotImplementedError("TODO: 实现 two_layer_forward")


def cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """返回一个 batch 的平均交叉熵。

    logits: float array, shape (B, C)
    labels: integer array, shape (B,)，取值位于 [0, C)，第 i 个元素是第 i 个样本的正确类别
    输出：有限的 Python float 标量，等于 B 个样本各自的交叉熵（自然对数）的平均值。实现需考虑大 logits 的数值稳定性。
    数值稳定性的含义：logits 的绝对值上千、正确类别的 logit 比这一行的最大值小几百时，
        结果仍然是正确的有限值，不能是 nan 或 inf，也不能靠给概率加一个小常数或截断概率来避免。
    不得原地修改 logits 和 labels。
    """
    # TODO: 使用减去每行最大值的 log-sum-exp 形式；不要调用深度学习框架。
    raise NotImplementedError("TODO: 实现 cross_entropy_from_logits")


def main() -> None:
    print("请完成 two_layer_forward 与 cross_entropy_from_logits 中的 TODO。")
    print("完成后运行：python tests/check_progress.py")


if __name__ == "__main__":
    main()
