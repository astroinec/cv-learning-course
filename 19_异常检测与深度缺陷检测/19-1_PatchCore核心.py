"""作业 19-1：PatchCore 的核心零件和工业指标。

只允许使用 NumPy。特征向量由测试直接给出，不需要下载预训练权重；
在 Notebook 里再用 torchvision 的预训练 ResNet 提特征，把这些零件拼成完整 PatchCore。
"""

from __future__ import annotations

import numpy as np


def pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """平方欧氏距离矩阵。

    a: (N, D)，b: (M, D) → (N, M) float64，元素 ||a_i - b_j||^2。
    用 ||a||^2 + ||b||^2 - 2 a·b 展开实现，并把数值误差导致的负数裁剪为 0。
    """
    raise NotImplementedError("TODO: 实现向量化的平方距离矩阵")


def greedy_coreset(features: np.ndarray, m: int, start: int = 0) -> np.ndarray:
    """k-center greedy 核心集采样。

    从下标 start 开始，每一步选出“到已选集合最近距离最大”的点加入，
    直到选够 m 个。返回 (m,) int64 下标数组，按被选中的先后顺序排列。
    m 大于点数或小于 1 时抛出 ValueError。
    提示：维护每个点到已选集合的最小距离，每加一个点只需更新一次，复杂度 O(N*m*D)。
    """
    raise NotImplementedError("TODO: 实现 k-center greedy")


def knn_scores(memory: np.ndarray, queries: np.ndarray, k: int = 1) -> np.ndarray:
    """异常分数：每个 query 到 memory 中 k 个最近邻的平均欧氏距离（不是平方）。

    memory: (M, D)，queries: (Q, D) → (Q,) float64。k 大于 M 时抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现 kNN 异常分数")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """从零实现 ROC 曲线下面积。

    labels 中 1 为异常（正类），0 为正常。分数越高越异常。
    等价于：随机取一个异常样本和一个正常样本，异常分数更高的概率，
    分数相等记 0.5。返回 Python float。只有一个类别时抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现 AUROC")


def threshold_at_escape(
    normal_scores: np.ndarray,
    defect_scores: np.ndarray,
    max_escape: float,
) -> tuple[float, float, float]:
    """在漏检率约束下选阈值，并报告过检率。

    判为缺陷的规则：score >= threshold。
    漏检率 escape = 缺陷中 score < threshold 的比例；
    过检率 overkill = 正常中 score >= threshold 的比例。
    在所有满足 escape <= max_escape 的候选阈值中，选最大的那个
    （阈值越大过检越少）。候选阈值只取缺陷样本的分数值即可。
    返回 (threshold, escape, overkill)，均为 Python float。
    """
    raise NotImplementedError("TODO: 实现漏检约束下的阈值选择")
