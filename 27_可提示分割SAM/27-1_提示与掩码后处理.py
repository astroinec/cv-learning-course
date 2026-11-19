"""作业 27-1：可提示分割的五个零件。

只允许使用 NumPy 和 PyTorch。第一个函数（点提示的位置编码）用 PyTorch，因为本章的小模型 `迷你SAM.py`
会直接调用它；其余四个是对模型输出做的后处理，用 NumPy。
不需要下载任何权重：测试直接给出掩码、分数和坐标。完成后在 Notebook 里用这些零件训练小模型、做“全图自动分割”。

本文件通用的约定：
- 图像坐标写成 (x, y)，x 是列号（向右增大），y 是行号（向下增大）；数组的最后两维是 (H, W)，先行后列。
- 整数坐标 k 指第 k 个像素的中心。
- 掩码的 logit 大于阈值的像素算前景。
"""

from __future__ import annotations

import numpy as np
import torch


def point_position_encoding(points: torch.Tensor, image_size: tuple[int, int], freqs: torch.Tensor) -> torch.Tensor:
    """点提示的位置编码：把像素坐标变成一组正弦和余弦。

    参数：
        points: (..., 2) 的张量，最后一维是 (x, y) 像素坐标，可以带小数。前面可以有任意多个维度。
        image_size: (H, W)，注意顺序是先高后宽，而点的顺序是先 x 后 y。
        freqs: (2, F) 的浮点张量，一组固定的频率。第 0 行乘 x，第 1 行乘 y。

    计算步骤（顺序不能错）：
        1. 像素下标换成连续坐标：加 0.5（整数 k 是第 k 个像素的中心）。
        2. 归一化到 0 到 1：x 除以 W，y 除以 H。
        3. 再线性映射到 -1 到 1。
        4. 和 freqs 做矩阵乘法，得到 (..., F)，再乘 2π。
        5. 把正弦和余弦沿最后一维拼起来：前 F 个是 sin，后 F 个是 cos。

    返回：(..., 2F) 的张量，数据类型和 freqs 相同。不要修改传入的 points。
    异常：points 的最后一维不是 2，或 freqs 的第 0 维不是 2，抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现点提示的位置编码")


def select_by_predicted_iou(masks: np.ndarray, iou_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每个提示有 M 个候选掩码，按模型自己预测的 IoU 选出分数最高的那个。

    参数：
        masks: (B, M, H, W)，B 个提示各 M 个候选。数据类型不限（布尔或 logit 都可以）。
        iou_pred: (B, M)，模型给每个候选预测的 IoU。它是网络的原始输出，可能是负数，也可能大于 1。

    返回三个数组：
        chosen: (B, H, W)，每个提示选中的候选，数据类型和 masks 相同。
        index:  (B,) int64，选中的是第几个候选。分数并列时取下标小的。
        score:  (B,) float64，选中候选的预测 IoU。

    每个提示各选各的，不同提示之间互不影响。
    异常：masks 不是四维，或 iou_pred 的形状不是 (B, M)，抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现按预测 IoU 选掩码")


def stability_score(logits: np.ndarray, threshold: float = 0.0, offset: float = 1.0) -> np.ndarray:
    """稳定性分数：把二值化的阈值上下各挪 offset，掩码变化有多大。

    定义：
        高阈值掩码 = logits > threshold + offset
        低阈值掩码 = logits > threshold - offset
        分数 = 高阈值掩码的像素数 / 低阈值掩码的像素数
    高阈值掩码一定包含在低阈值掩码里，所以这个比值就是两者的 IoU，在 0 到 1 之间。
    边界清楚的掩码，logit 在边界两侧跳得很快，阈值挪一点像素数几乎不变，分数接近 1。

    参数：
        logits: (..., H, W) 的浮点数组，最后两维是图像，前面可以有任意多个维度。
        threshold: 二值化阈值。offset: 上下挪动的量，大于 0。
        比较一律用严格大于。

    返回：(...) 的 float64 数组，每张掩码一个分数；输入只有两维时返回零维数组或 Python float 都可以。
    低阈值掩码为空时（一个前景像素都没有）分数记 0，不能返回 nan 或 inf。
    """
    raise NotImplementedError("TODO: 实现稳定性分数")


def mask_to_box(masks: np.ndarray) -> np.ndarray:
    """由掩码求外接框。

    参数：
        masks: (..., H, W)，布尔数组，或者非零即前景的数值数组。前面可以有任意多个维度。

    返回：(..., 4) 的 int64 数组，每张掩码一个 [x0, y0, x1, y1]：
        x0、x1 是前景像素最小和最大的列号，y0、y1 是最小和最大的行号，四个数都是“包含在内”的像素下标。
        只有一个前景像素 (行 r, 列 c) 时结果是 [c, r, c, r]。
    空掩码（没有前景像素）返回 [0, 0, 0, 0]。
    """
    raise NotImplementedError("TODO: 实现由掩码求外接框")


def mask_nms(masks: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.7) -> np.ndarray:
    """掩码的非极大值抑制：去掉和高分掩码重复的低分掩码。重叠程度用掩码的 IoU（不是外接框的 IoU）。

    参数：
        masks: (K, H, W) 布尔数组。scores: (K,) 每张掩码的分数。iou_threshold: 重叠阈值。

    规则：
        1. 按分数从高到低依次处理；分数相同的，下标小的先处理。
        2. 当前掩码和每一张“已经保留”的掩码算 IoU，只要有一个大于 iou_threshold（严格大于），就丢掉它；
           否则保留。已经被丢掉的掩码不再参与比较，它不能再去抑制别人。
        3. 两张掩码的并集为空时 IoU 记 0。

    返回：(n,) 的 int64 数组，保留下来的掩码在输入里的下标，按分数从高到低排列。
    K 为 0 时返回长度为 0 的 int64 数组。不要修改传入的数组。
    异常：masks 和 scores 的个数不同，抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现掩码的非极大值抑制")
