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
    输出契约：
        hidden: shape (B, H)，元素非负
        logits: shape (B, C)，不要做 softmax
    """
    # TODO: 先做仿射变换与 ReLU，再计算分类 logits。
    raise NotImplementedError("TODO: 实现 two_layer_forward")


def cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """返回一个 batch 的平均交叉熵。

    logits: float array, shape (B, C)
    labels: integer array, shape (B,)，取值位于 [0, C)
    输出：有限的 Python float 标量。实现需考虑大 logits 的数值稳定性。
    """
    # TODO: 使用减去每行最大值的 log-sum-exp 形式；不要调用深度学习框架。
    raise NotImplementedError("TODO: 实现 cross_entropy_from_logits")


def main() -> None:
    print("请完成 two_layer_forward 与 cross_entropy_from_logits 中的 TODO。")
    print("完成后运行：python tests/check_progress.py")


if __name__ == "__main__":
    main()
