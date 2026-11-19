"""作业 26-1：掩码图像建模和自蒸馏的五个零件。

只允许使用 NumPy 和 PyTorch。不需要下载任何预训练权重，输入都由测试直接给出；
在 Notebook 里再把这些零件接到一个很小的 ViT 上，训练一个迷你 MAE 和一个迷你自蒸馏模型。

约定：
- “块”指图像切成的小方块（patch）展平后的向量，一张图有 L 个块。
- mask 里 1 表示这个块被遮住（编码器看不到），0 表示可见。这和 MAE 官方实现的约定相同。
"""

from __future__ import annotations

import numpy as np
import torch


def random_masking(
    x: torch.Tensor,
    mask_ratio: float,
    noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按给定的随机数遮住一部分块，返回可见的块、掩码和还原下标。

    参数
        x: (B, L, D) 的浮点张量，B 张图，每张 L 个块，每块 D 维。
        mask_ratio: 被遮住的比例，取值范围 [0, 1)。
        noise: (B, L) 的浮点张量，每个块一个随机数。由调用方传入（训练时用 torch.rand），
            这样函数本身是确定的，方便测试。可以假设同一行里没有相等的数。

    规则
        每张图里被遮住的块数 n_mask = round(L * mask_ratio)，保留 n_keep = L - n_mask 个。
        （测试只用 L * mask_ratio 正好是整数的情形，不考察 .5 怎样取整。）
        每张图各自保留 noise 最小的 n_keep 个块，其余遮住。

    返回 (x_keep, mask, ids_restore)
        x_keep: (B, n_keep, D)，可见的块，按 noise 从小到大排列，dtype 与 x 相同。
            对 x 可导（梯度能传回 x 里被保留的位置）。
        mask: (B, L)，dtype 与 x 相同，1 表示被遮住，0 表示可见，按块在原图里的顺序排列。
        ids_restore: (B, L) 的 int64 张量，ids_restore[b, j] 是第 j 个块按 noise 从小到大排的名次（从 0 开始）。
            它的用途是把顺序还原回去：把 x_keep 和 n_mask 个占位向量在第 1 维拼起来得到 (B, L, D) 的 y，
            再做 torch.gather(y, 1, ids_restore[..., None].expand(-1, -1, D))，
            结果里可见的位置是原来的块，被遮住的位置是占位向量。

    异常
        mask_ratio 不在 [0, 1) 内、算出来 n_keep 小于 1、x 不是三维、
        或者 noise 的形状不是 (B, L) 时抛出 ValueError。

    不要修改 x 和 noise。三个返回值都要和 x 在同一个设备上（验收只在 CPU 上跑，
    但 Notebook 在有显卡时会把张量放到显卡上；新建张量时写 device=x.device，或者从 x、noise 派生）。
    """
    raise NotImplementedError("TODO: 实现随机遮挡，返回可见块、掩码和还原下标")


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    norm_pix: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """只在被遮住的块上计算的重建损失。

    参数
        pred: (B, L, P) 的浮点张量，解码器对每个块的 P 个像素的预测。
        target: (B, L, P)，真实像素（把图像切块后展平）。
        mask: (B, L)，1 表示被遮住，0 表示可见。
        norm_pix: 为 True 时，先把 target 的每个块各自归一化再比较：
            target = (target - 块内均值) / sqrt(块内方差 + eps)。
            方差用无偏估计（除以 P - 1，也就是 torch.var 的默认行为）。pred 不做归一化。
        eps: 只在 norm_pix=True 时使用。

    计算
        每个块的误差 = 这个块 P 个像素平方误差的平均；
        损失 = 所有被遮住的块的误差之和 / 被遮住的块数（mask 的总和）。
        可见块的预测无论多差都不影响结果。

    返回
        0 维张量（标量），对 pred 可导。

    异常
        pred 和 target 形状不同、mask 的形状不是 (B, L)、或者 mask 全为 0 时抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现只在被遮块上的重建损失")


def ema_update(teacher: torch.nn.Module, student: torch.nn.Module, momentum: float) -> None:
    """用学生的参数对教师的参数做一步指数滑动平均。

    对 teacher.parameters() 和 student.parameters() 里一一对应的每一对参数：
        教师 = momentum * 教师 + (1 - momentum) * 学生

    要求
        - 原地修改教师的参数：不新建 Parameter，不改变参数的 requires_grad。
        - 这一步不进计算图：教师的参数不管 requires_grad 是 True 还是 False 都要能更新，
          更新后参数的 grad_fn 仍为 None。
        - 学生的参数保持不变。
        - 只处理 parameters()，不处理 buffers（本章的小模型没有 BatchNorm）。
        - 没有返回值。

    异常
        momentum 不在 [0, 1] 内，或者两个网络的参数个数不同时抛出 ValueError。

    momentum = 1 时教师不变，momentum = 0 时教师变成学生的拷贝。
    """
    raise NotImplementedError("TODO: 实现教师的指数滑动平均更新")


def dino_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    center: torch.Tensor,
    student_temp: float = 0.1,
    teacher_temp: float = 0.04,
) -> torch.Tensor:
    """带中心化和温度的自蒸馏损失（DINO 的损失，单个视图对）。

    参数
        student_logits: (B, K)，学生网络投影头的输出（没有做过 softmax）。
        teacher_logits: (B, K)，教师网络对同一批图另一个视图的输出。
        center: (K,) 或 (1, K)，教师输出的滑动平均，由调用方维护。
        student_temp, teacher_temp: 两个温度，都必须大于 0。

    计算
        教师的目标分布 q = softmax((teacher_logits - center) / teacher_temp)，在最后一维上做，不参与求导；
        学生的对数概率 log p = log_softmax(student_logits / student_temp)，在最后一维上做；
        损失 = 对 B 个样本取平均的交叉熵：-(q * log p) 在最后一维求和，再对样本取平均。

    返回
        0 维张量（标量）。梯度只传给 student_logits：
        即使 teacher_logits 或 center 的 requires_grad 为 True，它们也不应该收到梯度。

    异常
        任何一个温度小于等于 0 时抛出 ValueError。

    注意这是交叉熵，不是 KL 散度：学生和教师的分布完全相同时，损失等于这个分布的熵，不是 0。
    """
    raise NotImplementedError("TODO: 实现带中心化和温度的自蒸馏损失")


def knn_classify(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    test_feats: np.ndarray,
    k: int,
    num_classes: int,
    temperature: float = 0.07,
) -> np.ndarray:
    """在冻结特征上做带权 k 近邻分类（DINO 论文评价特征时的做法）。

    参数
        train_feats: (N, D) 的 NumPy 数组，带标签样本的特征。
        train_labels: (N,) 的整数数组，取值 0 到 num_classes - 1。
        test_feats: (Q, D)，待分类样本的特征。
        k: 近邻个数。
        num_classes: 类别数。
        temperature: 投票权重的温度。

    计算
        1. 把两组特征各自按行做 L2 归一化（范数为 0 的行保持为 0，不能出现 nan）。
        2. 每个待分类样本和所有带标签样本算余弦相似度，取相似度最大的 k 个。
        3. 这 k 个邻居各自给自己的类别投票，权重是 exp(相似度 / temperature)。
        4. 预测得票（权重之和）最高的类别。得票完全相同时取编号最小的类别。

    返回
        (Q,) 的 int64 数组。

    异常
        k 小于 1 或者大于 N 时抛出 ValueError。

    不要修改传入的数组。
    """
    raise NotImplementedError("TODO: 实现带权 k 近邻分类")
