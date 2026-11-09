"""作业 18-1：金模板比对、连通域与过检/漏检统计。

只允许使用 NumPy（不能用 scipy.ndimage.label 或 cv2.connectedComponents
完成核心实现，可以在 Notebook 里对照）。
"""

from __future__ import annotations

import numpy as np


def robust_golden(stack: np.ndarray, min_sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """由 N 张已对齐的良品图构造金模板。

    输入 stack：(N, H, W)，N >= 3。
    输出 (median, sigma)，均为 (H, W) float64：
    - median 为逐像素中位数
    - sigma = 1.4826 * MAD，MAD 为逐像素 |x - median| 的中位数
    - sigma 不小于 min_sigma（避免完全稳定的像素噪声为 0，阈值退化）
    N < 3 时抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现中位数加 MAD 的鲁棒金模板")


def label_components(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """二值图连通域标记。

    输入 bool 或 0/1 数组 (H, W)，connectivity 为 4 或 8。
    输出 (labels, n)：labels 为 int32，背景 0，前景从 1 开始编号；
    按光栅扫描顺序（先行后列）第一次遇到的连通域编号为 1，依次递增。
    用 BFS 或两遍扫描加并查集都可以。connectivity 不是 4 或 8 时抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现连通域标记")


def detect_defects(
    image: np.ndarray,
    median: np.ndarray,
    sigma: np.ndarray,
    k: float = 5.0,
    min_area: int = 3,
) -> list[dict]:
    """用 z-score 阈值加连通域找缺陷。

    z = |image - median| / sigma，z > k 的像素为候选；
    8 连通域中面积 >= min_area 的保留为缺陷。
    返回 list[dict]，按 label 编号排序，每个 dict 包含：
    - "bbox": (y0, x0, y1, x1)，y1 和 x1 为开区间
    - "area": int
    - "centroid": (cy, cx)，Python float
    - "max_z": Python float，区域内最大 z
    """
    raise NotImplementedError("TODO: 实现缺陷检测")


def match_defects(
    predicted: list[tuple[float, float]],
    truth: list[tuple[float, float]],
    radius: float = 3.0,
) -> dict:
    """按距离把预测缺陷和真值缺陷一对一匹配。

    贪心策略：计算所有 (预测, 真值) 距离，按距离从小到大取，
    两端都未被使用且距离 <= radius 的配成一对。
    返回 dict：
    - "tp": 匹配成功数
    - "fp": 未匹配的预测数（误报，nuisance）
    - "fn": 未匹配的真值数（漏检，escape）
    - "capture_rate": tp / len(truth)，truth 为空时为 1.0
    """
    raise NotImplementedError("TODO: 实现缺陷匹配与统计")
