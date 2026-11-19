"""作业 18-1：金模板比对、连通域与过检/漏检统计。

只允许使用 NumPy（不能用 scipy.ndimage.label 或 cv2.connectedComponents
完成核心实现，可以在 Notebook 里对照）。

所有函数都不能修改传入的数组和列表（需要改动时先复制一份）。
坐标一律是 (行, 列)，也就是 (y, x)。
"""

from __future__ import annotations

import numpy as np


def robust_golden(stack: np.ndarray, min_sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """由 N 张已对齐的良品图构造金模板。

    输入 stack：(N, H, W)，N >= 3，任意数值类型。
    输出 (median, sigma)，均为 (H, W) float64：
    - median 为逐像素中位数；N 为偶数时取中间两个数的平均（和 np.median 一致）
    - sigma = 1.4826 * MAD，MAD 为逐像素 |x - median| 的中位数。
      偏差以 median 为中心，取的是中位数，不是平均值，也不是标准差
    - sigma 不小于 min_sigma：先算 1.4826 * MAD，小于 min_sigma 的地方换成 min_sigma
      （避免完全稳定的像素噪声为 0，阈值退化）
    - 每个像素各算各的，不同像素的 sigma 一般不同
    N < 3 时抛出 ValueError；N 等于 3 是合法输入。
    """
    raise NotImplementedError("TODO: 实现中位数加 MAD 的鲁棒金模板")


def label_components(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """二值图连通域标记。

    输入 bool 或 0/1 数组 (H, W)，connectivity 为 4 或 8。
    输出 (labels, n)：labels 为 int32，背景 0，前景从 1 开始编号，n 是连通域个数；
    按光栅扫描顺序（先行后列）第一次遇到的连通域编号为 1，依次递增。
    编号必须连续：labels 里出现的值恰好是 0 到 n（没有前景时只有 0，n 为 0）。
    用两遍扫描时，合并之后要按“每个连通域第一个像素的光栅顺序”重新编号。
    图像边界不相连：最左一列和最右一列、第一行和最后一行不算相邻。
    用 BFS 或两遍扫描加并查集都可以，不要用递归（大连通域会超过递归深度上限）。
    connectivity 不是 4 或 8 时抛出 ValueError。
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

    image、median、sigma 是形状相同的 (H, W) 数组，任意数值类型（包括 uint8），
    计算前要转成 float64，否则无符号整数相减会回绕。
    z = |image - median| / sigma，z > k 的像素为候选（z 恰好等于 k 的不算）；
    8 连通域中面积 >= min_area 的保留为缺陷（面积恰好等于 min_area 的保留）。
    返回 list[dict]，按 label 编号排序（也就是按每个区域第一个像素的光栅顺序），
    没有缺陷时返回空列表。每个 dict 包含：
    - "bbox": (y0, x0, y1, x1)，y1 和 x1 为开区间，即 y1 = 最大行号 + 1
    - "area": int，区域的像素数
    - "centroid": (cy, cx)，两个 Python float，是区域内像素行号、列号的算术平均（不按 z 加权）
    - "max_z": Python float，这个区域内最大的 z（不是全图的，也不是灰度差）
    """
    raise NotImplementedError("TODO: 实现缺陷检测")


def match_defects(
    predicted: list[tuple[float, float]],
    truth: list[tuple[float, float]],
    radius: float = 3.0,
) -> dict:
    """按距离把预测缺陷和真值缺陷一对一匹配。

    predicted 和 truth 是 (y, x) 坐标的列表，可以为空。距离是欧氏距离。
    贪心策略：计算所有 (预测, 真值) 距离，按距离从小到大取，
    两端都未被使用且距离 <= radius（恰好等于也算）的配成一对。
    一个预测最多配一个真值，一个真值也最多配一个预测。
    返回 dict：
    - "tp": 匹配成功数
    - "fp": 未匹配的预测数（误报，nuisance）
    - "fn": 未匹配的真值数（漏检，escape）
    - "capture_rate": tp / len(truth)，truth 为空时为 1.0（不管 predicted 是否为空）
    """
    raise NotImplementedError("TODO: 实现缺陷匹配与统计")
