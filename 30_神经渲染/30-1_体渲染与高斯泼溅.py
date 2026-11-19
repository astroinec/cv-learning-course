"""作业 30-1：体渲染和高斯泼溅的五个零件。

只允许使用 NumPy 和 PyTorch。不需要下载任何数据和权重，输入都由测试直接给出；
在 Notebook 里再把这些零件接到一个很小的 NeRF 和一组三维高斯上，各拟合一遍同一个小场景。

约定（和第 7、17、21 章一致）：
- 相机用 OpenCV 的约定：相机坐标 P_c = R P_w + t，相机看向 +Z，x 向右，y 向下；
  像素坐标 u = fx * X_c / Z_c + cx，v = fy * Y_c / Z_c + cy。
- “沿光线”指张量的最后一维：一条光线上有 N 个采样点，按离相机从近到远排列。
- 所有函数都不修改传入的张量，返回值的 dtype 和输入的浮点张量相同，并且保留梯度
  （不要经过 NumPy，不要 detach）。
- 返回值要和输入在同一个设备上。验收只在 CPU 上跑，但 Notebook 在有显卡时会把张量放到显卡上；
  函数里新建张量时从输入派生（例如 torch.zeros_like、torch.ones_like），或者写 device=x.device。
"""

from __future__ import annotations

import torch


def volume_weights(sigma: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """由每一段的密度和长度，算出体渲染里每个采样点的权重。

    参数
        sigma: (..., N) 的浮点张量，每个采样点处的体密度，都不小于 0。前面可以有任意多个批的维度。
        deltas: 每一段的长度，都不小于 0。形状和 sigma 相同，或者能广播成 sigma 的形状（例如 (N,)）。

    计算（都沿最后一维）
        第 i 段的不透明度    alpha_i = 1 - exp(-sigma_i * deltas_i)
        到达第 i 段之前的透射率  T_i = 前面各段 (1 - alpha_j) 的乘积，j < i；第一段的 T 是 1
        权重               w_i = T_i * alpha_i

    返回
        形状和 sigma 相同的张量。权重之和不超过 1；1 减去权重之和是光线穿过全部 N 段之后剩下的透射率。

    要求
        - 结果里不能出现 nan 或 inf，梯度里也不能：某一段完全不透明（sigma_i * deltas_i 很大）时，
          它后面各段的权重是 0。
        - 最后一段的长度可以非常大（NeRF 的原始实现把最后一段设成 1e10，表示一直延伸到无穷远），
          这时前面各段的权重不受影响，最后一段的权重是 T 乘以它自己的不透明度。

    异常
        sigma 是 0 维张量、deltas 不能广播成 sigma 的形状、或者 sigma、deltas 里有负数时抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现体渲染权重（不透明度、透射率的累乘、两者相乘）")


def composite_rays(
    weights: torch.Tensor,
    colors: torch.Tensor,
    t_vals: torch.Tensor,
    background: torch.Tensor | None = None,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """用同一组权重沿光线合成颜色和深度。

    参数
        weights: (..., N)，volume_weights 的输出。
        colors: (..., N, 3)，每个采样点的颜色。
        t_vals: 每个采样点沿光线的距离，形状和 weights 相同，或者是 (N,)（所有光线共用）。
        background: None，或者 (3,) 的背景颜色。
        eps: 防止除以 0 的小量。

    返回 (rgb, depth, acc)
        acc: (...)，权重之和，也叫累积不透明度。
        rgb: (..., 3)，颜色按权重求和。background 不是 None 时，再加上 (1 - acc) * background：
            没有被挡住的那一部分光来自背景。background 是 None 时不加（相当于黑色背景）。
        depth: (...)，期望深度：t_vals 按权重求和，再除以 max(acc, eps)。
            除以 acc 是为了让只挡住一部分的光线（acc 小于 1）也给出表面的位置，而不是被拉向 0。
            一条什么都没碰到的光线（权重全为 0）深度是 0，不能是 nan。

    异常
        colors 的形状不是 weights 的形状再加一维 3，或者 t_vals 的形状既不同于 weights 也不是 (N,) 时，
        抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现沿光线合成颜色、期望深度和累积不透明度")


def positional_encoding(x: torch.Tensor, num_freqs: int, include_input: bool = True) -> torch.Tensor:
    """把坐标映射成一组不同频率的正弦和余弦（NeRF 的位置编码）。

    参数
        x: (..., D) 的浮点张量。
        num_freqs: 频率个数 L，不小于 0。
        include_input: 是否把原始坐标也放进输出。

    输出沿最后一维依次是（每一块都是 D 维，块内保持 x 原来的坐标顺序）：
        [x]（只在 include_input 为 True 时有），
        sin(2^0 * pi * x)，cos(2^0 * pi * x)，
        sin(2^1 * pi * x)，cos(2^1 * pi * x)，
        ……
        sin(2^(L-1) * pi * x)，cos(2^(L-1) * pi * x)

    注意频率里有 pi，相邻两个频率差 2 倍。

    返回
        (..., D * (2 * num_freqs + 1)) 的张量；include_input 为 False 时是 (..., D * 2 * num_freqs)。
        num_freqs 为 0 且 include_input 为 True 时，输出就是 x 本身的值。

    异常
        num_freqs 小于 0，或者 num_freqs 为 0 同时 include_input 为 False（输出会是空的）时抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现位置编码")


def project_gaussians(
    means: torch.Tensor,
    covs: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 N 个三维高斯投影到图像上，得到二维的中心和协方差。

    参数
        means: (N, 3)，高斯中心的世界坐标。
        covs: (N, 3, 3)，世界坐标系里的协方差矩阵（对称）。
        R: (3, 3)，t: (3,)，相机外参，P_c = R P_w + t（OpenCV 约定，相机看向 +Z）。
        fx, fy, cx, cy: 内参，单位是像素。

    计算
        1. 中心变到相机坐标 (X, Y, Z)，投影成像素 uv = (fx * X / Z + cx, fy * Y / Z + cy)。
        2. 协方差先变到相机坐标系：旋转 R 作用在协方差的两侧（平移不影响协方差）。
        3. 透视投影不是线性的，用它在高斯中心处的雅可比矩阵 J（2x3，像素坐标对相机坐标的偏导数）
           做线性近似：二维协方差 = J * 相机坐标系里的协方差 * J 的转置。
           J 的每一行对 X、Y、Z 各求一次偏导，别漏了对 Z 的那一列。

    返回 (uv, cov2d, depth)
        uv: (N, 2)，像素坐标，第 0 列是 u（向右），第 1 列是 v（向下）。
        cov2d: (N, 2, 2)，单位是像素的平方。不要在这里给对角线加任何常数（那是调用方的事）。
        depth: (N,)，相机坐标的 Z（不是到相机中心的距离），合成时按它排序。

    异常
        means、covs、R、t 的形状不对，或者有高斯的中心不在相机前方（Z <= 0）时抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现三维高斯到二维的投影（中心、雅可比、协方差）")


def composite_gaussians(
    alphas: torch.Tensor,
    colors: torch.Tensor,
    depths: torch.Tensor,
    background: torch.Tensor,
    max_alpha: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 N 个高斯在 P 个像素上的贡献，按深度从前到后做 alpha 合成。

    参数
        alphas: (P, N)，alphas[i, j] 是第 j 个高斯在第 i 个像素上的不透明度
            （调用方已经把高斯自己的不透明度和二维高斯的衰减乘在一起了）。
        colors: (N, 3)，每个高斯的颜色。
        depths: (N,)，每个高斯的深度。传进来的顺序是任意的，不保证已经排好。
        background: (3,)，背景颜色。
        max_alpha: 不透明度的上限。

    计算
        1. 把 alphas 限制在 [0, max_alpha] 之间（得到新的张量，不改传入的）。
        2. 按 depths 从小到大（从近到远）排列高斯。验收不考察深度完全相同的情形。
        3. 对每个像素，按这个顺序：
               T_j = 排在它前面的各个高斯的 (1 - alpha) 的乘积，最前面一个的 T 是 1
               w_j = T_j * alpha_j
        4. acc = 各个高斯的 w 之和；rgb = w 加权的颜色之和 + (1 - acc) * background。

    返回 (rgb, acc)
        rgb: (P, 3)，acc: (P,)。梯度要能传到 alphas 和 colors。

    异常
        alphas 不是二维、colors 的形状不是 (N, 3)、或者 depths 的形状不是 (N,) 时抛出 ValueError。

    max_alpha 默认取 0.99 而不是 1：一个高斯把光完全挡死时，它后面所有高斯的权重是 0、收不到梯度。
    """
    raise NotImplementedError("TODO: 实现按深度排序、从前到后的 alpha 合成")
