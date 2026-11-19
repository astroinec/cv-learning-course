"""作业 28-1：零样本分类、图文检索和区域打分的五个零件。

只允许使用 NumPy（PyTorch 可以用，但不需要）。不需要下载任何预训练权重，输入都由测试直接给出；
在 Notebook 里再把这些零件接到一个从零训练的很小的双塔模型上。

约定：
- “向量”都按行存放：(N, D) 表示 N 个 D 维向量。
- “归一化”都指按行做 L2 归一化：每一行除以 max(这一行的 L2 范数, eps)。
  范数为 0 的行归一化之后仍然是 0，不能出现 nan 或 inf。
- 余弦相似度 = 两个归一化之后的向量的内积。
- 所有函数都不要修改传入的数组；输入可能是 float32，输出一律是 float64（下标是 int64）。
"""

from __future__ import annotations

import numpy as np


def build_zeroshot_classifier(text_embeds: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """由文本向量构造零样本分类器：多个提示词模板先各自归一化，再平均，再归一化。

    参数
        text_embeds: (C, T, D) 的浮点数组。C 个类别，每个类别 T 个模板（同一个类别名套进 T 种句式），
            每个模板一个 D 维的文本向量，没有归一化过，各个向量的长度可以相差很多。
            也接受 (C, D)，当作 T = 1 处理。
        eps: 归一化时范数的下限。

    计算
        1. 每个模板的向量各自归一化（沿最后一维；同一个类别的 T 个向量各除各的范数）。
        2. 对同一个类别的 T 个归一化后的向量取平均，得到 (C, D)。
        3. 把平均后的每个类别向量再归一化一次。

    返回
        (C, D) 的 float64 数组，每一行是一个类别的分类器向量。
        除了全 0 的情形，每一行的 L2 范数都是 1。

    异常
        text_embeds 不是二维或三维，或者任何一维的长度为 0 时抛出 ValueError。

    为什么要归一化两次：第一次让每个模板的权重相同（否则长的向量说了算），
    第二次让各个类别的分类器向量一样长（否则 zeroshot_predict 里长的那个类别占便宜）。
    """
    raise NotImplementedError("TODO: 实现多模板平均的零样本分类器")


def zeroshot_predict(
    image_embeds: np.ndarray,
    classifier: np.ndarray,
    logit_scale: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """用分类器向量给图像向量分类，返回每一类的概率和预测的类别。

    参数
        image_embeds: (N, D) 的浮点数组，N 张图的图像向量，没有归一化过。
        classifier: (C, D) 的浮点数组，每行一个类别。原样使用，不要在这里再归一化：
            零样本时它来自 build_zeroshot_classifier，已经是单位向量；
            做少样本适配时它是训练出来的权重，长度本身有意义。
        logit_scale: 相似度的放大倍数，也就是温度的倒数，必须大于 0。

    计算
        1. image_embeds 按行归一化。
        2. logits = logit_scale * 归一化后的图像向量 @ classifier 的转置，形状 (N, C)。
        3. probs = 每一行各自做 softmax（对 C 个类别）。
           logit_scale 很大时（例如 2000）也不能出现 inf 或 nan：先减去每一行的最大值，再取指数。
        4. pred = 每一行 logits 最大的类别；并列时取编号最小的。

    返回 (probs, pred)
        probs: (N, C) 的 float64 数组，每行的和为 1。
        pred: (N,) 的 int64 数组。

    异常
        image_embeds 或 classifier 不是二维、两者的 D 不同，或者 logit_scale 小于等于 0 时抛出 ValueError。

    注意 logit_scale 只改变概率有多尖，不改变 pred。
    """
    raise NotImplementedError("TODO: 实现零样本预测（归一化、放大、softmax、取最大）")


def recall_at_k(sim: np.ndarray, positives: np.ndarray, k: int) -> float:
    """检索指标 Recall@K：前 K 个候选里至少有一个正确答案的查询所占的比例。

    参数
        sim: (Q, N) 的浮点数组，sim[i, j] 是第 i 个查询和第 j 个候选的相似度，越大越像。
            图找文时查询是图、候选是句子；文找图时把矩阵转置再传进来。
        positives: (Q, N) 的布尔数组（或 0/1 数组），positives[i, j] 为真表示候选 j 是查询 i 的一个正确答案。
            一个查询可以有多个正确答案（例如几张图配的是同一句话），所以不能假设正确答案在对角线上。
        k: 取前几个候选。

    计算
        对每个查询，把候选按相似度从大到小排，取前 k 个；相似度相同时下标小的排在前面。
        这 k 个里只要有一个是正确答案，这个查询就算命中。
        返回命中的查询数除以 Q。

    返回
        Python 的 float，在 0 到 1 之间。

    异常
        sim 不是二维、positives 的形状和 sim 不同、k 小于 1 或大于 N、
        或者有的查询一个正确答案都没有时抛出 ValueError。

    提示：np.argsort 加上 kind="stable" 才保证相同的值保持原来的先后。
    """
    raise NotImplementedError("TODO: 实现 Recall@K")


def modality_gap(image_embeds: np.ndarray, text_embeds: np.ndarray) -> tuple[float, float, float]:
    """量模态间隙：图像向量和文本向量在单位球面上各成一团，两团的中心相距多远，各自散得多开。

    参数
        image_embeds: (N, D) 的浮点数组，没有归一化过。
        text_embeds: (M, D) 的浮点数组，没有归一化过。N 和 M 可以不同，两组不需要一一配对。

    计算
        1. 两组向量各自按行归一化。
        2. 各自的中心 = 归一化后的向量的平均（中心本身不再归一化，它的长度一般小于 1）。
        3. gap = 两个中心之差的 L2 范数。
        4. 每一组的散布 = 这一组里每个向量到本组中心的距离的平方，取平均，再开方。

    返回 (gap, spread_image, spread_text)
        三个 Python 的 float，顺序固定。

    异常
        任何一组不是二维、是空的，或者两组的 D 不同时抛出 ValueError。

    两个可以拿来自查的性质：
        - 对单位向量，散布的平方 = 1 - 中心长度的平方。向量全部重合时散布是 0。
        - 把任何一组的每个向量各自乘上一个正数，三个返回值都不变。
    """
    raise NotImplementedError("TODO: 实现模态间隙的度量")


def score_regions(
    region_feats: np.ndarray,
    query_embeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用文字查询的向量给一张图的各个候选区域打分，并给出每个查询得分最高的区域。

    参数
        region_feats: (R, D) 的浮点数组，一张图里 R 个候选区域各自的特征，没有归一化过。
        query_embeds: (Q, D) 的浮点数组，Q 个文字查询的文本向量，没有归一化过。

    计算
        1. 区域特征和查询向量各自按行归一化。
        2. scores[r, q] = 第 r 个区域和第 q 个查询的余弦相似度。
        3. 对每个查询，在 R 个区域里取分数最高的那个；并列时取下标最小的。

    返回 (scores, best_region, best_score)
        scores: (R, Q) 的 float64 数组，行是区域，列是查询。
        best_region: (Q,) 的 int64 数组，每个查询得分最高的区域的下标。
        best_score: (Q,) 的 float64 数组，对应的分数，等于 scores[best_region[q], q]。

    异常
        任何一个输入不是二维、是空的（R 或 Q 为 0），或者两者的 D 不同时抛出 ValueError。

    特征全 0 的区域（例如一块空白的填充区域）和任何查询的分数都是 0，不能是 nan。
    """
    raise NotImplementedError("TODO: 实现区域打分并取最大")
