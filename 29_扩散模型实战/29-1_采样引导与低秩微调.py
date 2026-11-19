"""作业 29-1：采样、引导、低秩微调和修复的六个零件。

只允许使用 PyTorch（和 Python 标准库的 math）。不需要下载任何权重，输入都由测试直接给出；
在 Notebook 里再把这些零件接到两个很小的去噪网络上，做采样、引导、掩膜条件、低秩微调和修复。

符号和第 20 章一致：
- abar 是噪声表的累计乘积（第 20 章作业的 alpha_bars 的输出），取值在 0 到 1 之间，越小噪声越多。
- 前向过程：x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps，eps 是标准高斯噪声。
- “干净的图”对应 abar = 1。采样的最后一步走到 abar_prev = 1。

对所有函数都成立的约定（验收会检查）：不修改传入的张量（不要用原地运算）；
返回值的 dtype 和设备跟第一个张量参数相同；该对谁可导就保持可导（不要 detach，不要经过 NumPy）。
数值按 float32 的精度核对：同一个式子换一种等价的写法、系数用 float32 算，都不会被判错。

dtype 这一条要留意张量形式的系数：abar_t、scale、mask 以张量传进来时，它们的 dtype 可能和第一个张量参数不同
（噪声表常常是 float64 算出来的，网络的输入输出是 float32）。直接相乘会把结果提升成 float64，
送回 float32 的网络时报 dtype 不一致。先把这些系数转成第一个张量参数的 dtype 和设备，再做运算。
"""

from __future__ import annotations

import torch


def convert_prediction(
    x_t: torch.Tensor,
    pred: torch.Tensor,
    abar_t: float | torch.Tensor,
    kind: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把网络的输出换算成对原图、噪声和 v 的三个估计。

    去噪网络可以预测三种量里的任何一种，它们之间只差一个由 abar_t 决定的线性换算：
        预测噪声 eps：  x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps
        预测原图 x0：   同上，反过来解 eps
        预测 v：        v = sqrt(abar_t) * eps - sqrt(1 - abar_t) * x0

    参数
        x_t: 任意形状的浮点张量，当前带噪声的样本。
        pred: 和 x_t 形状相同，网络的输出。
        abar_t: 当前步的 abar。Python 浮点数，或者能广播到 x_t 的张量（例如每个样本一个值，形状 (B, 1, 1, 1)）。
            取值必须严格在 0 和 1 之间（等于 0 或 1 时换算要除以 0）。
            是张量时 dtype 可以和 x_t 不同，返回值的 dtype 仍然跟 x_t 走。
        kind: "eps"、"x0" 或 "v"，说明 pred 是哪一种量。

    返回 (x0, eps, v)
        三个和 x_t 形状、dtype 相同的张量。三者互相一致：
        x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps，v = sqrt(abar_t) * eps - sqrt(1 - abar_t) * x0；
        其中和 kind 对应的那一个等于 pred。三个返回值都对 pred 可导。

    异常
        kind 不是三者之一、x_t 和 pred 形状不同、abar_t 里有小于等于 0 或大于等于 1 的值时抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现三种参数化之间的换算")


def ddim_step(
    x_t: torch.Tensor,
    eps: torch.Tensor,
    abar_t: float,
    abar_prev: float,
    eta: float = 0.0,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """DDIM 的一步更新：从噪声水平 abar_t 走到 abar_prev。

    计算
        x0_hat = (x_t - sqrt(1 - abar_t) * eps) / sqrt(abar_t)
        sigma^2 = eta^2 * (1 - abar_prev) / (1 - abar_t) * (1 - abar_t / abar_prev)
        x_prev = sqrt(abar_prev) * x0_hat + sqrt(1 - abar_prev - sigma^2) * eps + sigma * noise

    eta = 0 时 sigma = 0，整步是确定的（DDIM）；eta = 1 时 sigma^2 是 DDPM 的后验方差，这一步就是 DDPM 的一步。
    abar_prev = 1 是采样的最后一步：这时 sigma = 0，结果就是 x0_hat。

    参数
        x_t: 任意形状的浮点张量。
        eps: 和 x_t 形状相同，对噪声的预测（网络预测的不是噪声时，先用 convert_prediction 换算）。
        abar_t: Python 浮点数，当前的噪声水平，严格在 0 和 1 之间。整批样本用同一个步数。
        abar_prev: Python 浮点数，要走到的噪声水平，大于 0、小于等于 1。
        eta: 0 到 1 之间的浮点数。
        noise: eta 大于 0 时必须给出，和 x_t 形状相同的标准高斯噪声，由调用方生成（这样函数本身是确定的）。
            eta 等于 0 时忽略这个参数，可以不传。

    返回
        和 x_t 形状、dtype 相同的张量，对 x_t 和 eps 可导。

    异常（都是 ValueError）
        abar_t 不在 (0, 1) 内；abar_prev 不在 (0, 1] 内；eta 不在 [0, 1] 内；
        eta 大于 0 而 noise 是 None；eta 大于 0 而 abar_prev 小于 abar_t（往噪声更多的方向走时后验方差没有定义）。
        eta 等于 0 时 abar_prev 可以小于 abar_t，公式照样成立。
    """
    raise NotImplementedError("TODO: 实现 DDIM 的一步更新")


def cfg_combine(
    eps_uncond: torch.Tensor,
    eps_cond: torch.Tensor,
    scale: float | torch.Tensor,
) -> torch.Tensor:
    """无分类器引导：把无条件和有条件的两个预测组合成一个。

    结果 = eps_uncond + scale * (eps_cond - eps_uncond)

    scale = 0 时等于无条件的预测，scale = 1 时等于有条件的预测，大于 1 时沿着“有条件减无条件”的方向继续外推。
    这是 diffusers 里 guidance_scale 的约定。提出这个方法的论文写成 (1 + w) * eps_cond - w * eps_uncond，
    两种写法差 1：guidance_scale = 1 + w。本函数用前一种。

    参数
        eps_uncond, eps_cond: 形状相同的浮点张量。
        scale: Python 数，或者能广播到两个预测的张量（例如每个样本一个强度，形状 (B, 1, 1, 1)）。可以小于 0 或大于 1，不做截断。
            是张量时 dtype 可以和预测不同，返回值的 dtype 仍然跟 eps_uncond 走。

    返回
        和 eps_uncond 形状、dtype 相同的张量，对两个预测都可导。

    异常
        两个预测的形状不同时抛出 ValueError（不要靠广播悄悄算出一个结果）。
    """
    raise NotImplementedError("TODO: 实现无分类器引导的组合")


def lora_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """带低秩增量的线性层的前向。

    y = x @ weight.T + bias + (alpha / r) * (x @ lora_a.T) @ lora_b.T

    等价于把权重换成 weight + (alpha / r) * lora_b @ lora_a 的线性层，但分成两路算：
    原来的一路不动，旁边多一路先降到 r 维再升回来。训练时 weight 和 bias 冻结，只有 lora_a、lora_b 收到梯度。

    参数
        x: (..., in) 的浮点张量，前面可以有任意多个维度。
        weight: (out, in)，原来的权重，和 torch.nn.Linear 的 weight 同样的摆法。
        bias: (out,)，或者 None（没有偏置）。
        lora_a: (r, in)，r 至少为 1。
        lora_b: (out, r)。
        alpha: 缩放系数，增量整体乘 alpha / r，r 是 lora_a 的行数。

    返回
        (..., out) 的张量，dtype 与 x 相同。对 x、lora_a、lora_b 可导。

    异常
        weight、lora_a、lora_b 不是二维，lora_a 的列数不等于 in，lora_b 的形状不是 (out, r)，
        或者 x 的最后一维不等于 in 时抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现带低秩增量的线性层前向")


def lora_merge(
    weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """把低秩增量并回权重：返回 weight + (alpha / r) * lora_b @ lora_a。

    合并之后，普通线性层 x @ merged.T + bias 的输出应当和 lora_forward 相同（只差浮点舍入）。

    参数和形状要求同 lora_forward。

    返回
        (out, in) 的新张量，dtype 与 weight 相同。不要原地修改 weight（部署时常常要保留一份没合并的权重，方便换别的增量）。

    异常
        形状不符合要求时抛出 ValueError，条件同 lora_forward。
    """
    raise NotImplementedError("TODO: 实现低秩增量的合并")


def inpaint_replace(
    x_t: torch.Tensor,
    known_x0: torch.Tensor,
    mask: torch.Tensor,
    abar_t: float,
    noise: torch.Tensor,
) -> torch.Tensor:
    """修复式采样里“替换已知区域”的那一步。

    采样进行到噪声水平 abar_t 时，要重新生成的区域保留采样器给出的 x_t；
    其余（已知）区域换成“干净的已知图按前向过程加噪到同一个噪声水平”的结果：

        known_t = sqrt(abar_t) * known_x0 + sqrt(1 - abar_t) * noise
        结果 = mask * x_t + (1 - mask) * known_t

    参数
        x_t: (B, C, H, W) 的浮点张量，采样器这一步给出的结果，噪声水平是 abar_t。
        known_x0: 和 x_t 形状相同，干净的已知图（例如一张良品图）。
        mask: 能广播到 x_t 的张量（常见的是 (B, 1, H, W)），取值在 0 到 1 之间。
            1 表示这个像素要重新生成，0 表示这个像素已知、要保持。这和 diffusers 的修复管线里“白色是要重画的区域”的约定相同。
            0 和 1 之间的值按上面的式子线性混合。可以是浮点张量，也可以是布尔张量（True 当作 1），
            返回值的 dtype 仍然跟 x_t 走。
        abar_t: Python 浮点数，x_t 所处的噪声水平，0 到 1 之间（两端都可以取到）。
            等于 1 表示已经是干净的图，这时已知区域应当正好等于 known_x0，和 noise 无关。
        noise: 和 x_t 形状相同的标准高斯噪声，由调用方生成。

    返回
        和 x_t 形状、dtype 相同的张量。mask 为 1 的地方等于 x_t，对 x_t 可导。

    异常
        abar_t 不在 [0, 1] 内、mask 有小于 0 或大于 1 的值、x_t、known_x0、noise 三者形状不全相同时抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现修复采样里已知区域的替换")
