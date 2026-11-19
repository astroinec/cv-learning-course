"""作业 31-1：量化和部署的五个零件。

只允许使用 NumPy 和 PyTorch。不需要 onnx、onnxruntime、TensorRT，也不需要显卡，输入都由测试直接给出；
在 Notebook 里再把这些零件接到一个小卷积网络上：折叠 BatchNorm、做训练后量化、量延迟、按节拍选批大小。

约定（五个函数通用）：
- 量化：实数 x 用整数 q 表示，x 约等于 (q - zero_point) * scale。scale 叫缩放系数（一格有多宽），
  zero_point 叫零点（实数 0 对应的整数）。q 的取值范围是 [qmin, qmax]，超出的被截断。
- “对称”指零点固定为 0、正负范围相同；“非对称”指零点可以不是 0。
- 位数 num_bits 是 2 到 16 之间的整数。
"""

from __future__ import annotations

import numpy as np
import torch


def quant_params(
    x_min: float,
    x_max: float,
    num_bits: int = 8,
    symmetric: bool = False,
) -> tuple[float, int, int, int]:
    """由数据的范围 [x_min, x_max] 计算均匀量化的缩放系数和零点。

    参数
        x_min, x_max: 要表示的实数范围，x_min <= x_max。通常是一批权重或激活值的最小值和最大值，
            也可以是校准时定下的截断范围。
        num_bits: 位数。
        symmetric: False 为非对称量化，True 为对称量化。

    非对称（symmetric=False），用无符号整数
        qmin = 0，qmax = 2 ** num_bits - 1。
        先把范围扩到包含 0：lo = min(x_min, 0)，hi = max(x_max, 0)。
        （实数 0 必须能被精确表示，补零、ReLU 的输出都要用到它。）
        scale = (hi - lo) / (qmax - qmin)。
        zero_point = qmin - lo / scale 取到最近的整数，再限制在 [qmin, qmax] 内。
        （验收不考察 qmin - lo / scale 正好落在 .5 上的情形。）
        lo 和 hi 相等（也就是 x_min = x_max = 0）时没有范围可言：返回 scale = 1.0，zero_point = 0。

    对称（symmetric=True），用有符号整数的“窄范围”
        qmax = 2 ** (num_bits - 1) - 1，qmin = -qmax。8 位时是 -127 到 127，不用 -128。
        scale = max(|x_min|, |x_max|) / qmax。
        zero_point = 0。
        max(|x_min|, |x_max|) 为 0 时返回 scale = 1.0。

    返回 (scale, zero_point, qmin, qmax)
        scale 是 Python float，大于 0；zero_point、qmin、qmax 是 Python int。

    异常
        num_bits 不是 2 到 16 之间的整数、x_min 大于 x_max、
        或者 x_min、x_max 里有 nan 或无穷大时抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现缩放系数和零点的计算")


def quantize_dequantize(
    x: np.ndarray,
    scale,
    zero_point,
    qmin: int,
    qmax: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把实数量化成整数，再反量化回实数。

    参数
        x: 任意形状的 NumPy 数组（或能转成数组的数）。
        scale: 缩放系数，大于 0。可以是一个数，也可以是能和 x 广播的数组（逐通道量化时每个通道一个）。
        zero_point: 零点，整数。可以是一个数，也可以是能和 x 广播的整数数组。
        qmin, qmax: 整数的取值范围，两端都包含。

    计算
        q = x / scale 取到最近的整数，加上 zero_point，再截断到 [qmin, qmax]。
        x_hat = (q - zero_point) * scale。
        取整用“到最近的整数”，正好落在 .5 上时按 np.rint 处理（取偶数）；验收不考察这种输入。
        注意顺序：先取整、再加零点、最后截断。不是先截断实数再取整，也不是向下取整。

    返回 (q, x_hat)
        q: int64 数组，形状是 x 和 scale、zero_point 广播后的形状。
        x_hat: float64 数组，形状同 q。范围内的 x，|x_hat - x| 不超过 scale / 2；
            范围外的 x 被截到范围的端点上。

    异常
        scale 里有小于等于 0 的数时抛出 ValueError。

    不要修改传入的数组。
    """
    raise NotImplementedError("TODO: 实现量化与反量化")


def quantize_weight_per_channel(w: np.ndarray, num_bits: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """对卷积或全连接的权重做逐通道的对称量化：每个输出通道用自己的缩放系数。

    参数
        w: 至少二维的 NumPy 数组，第 0 维是输出通道。卷积权重是 (O, I, kH, kW)，全连接权重是 (O, I)。
        num_bits: 位数。

    计算（对每个输出通道 o 分别做）
        qmax = 2 ** (num_bits - 1) - 1（窄范围，和 quant_params 的对称情形相同）。
        scales[o] = 这个通道里权重绝对值的最大值 / qmax。
        这个通道的权重全为 0 时 scales[o] = 1.0，量化结果全为 0，不能出现 nan。
        q[o] = w[o] / scales[o] 取到最近的整数，截断到 [-qmax, qmax]。

    返回 (q, scales)
        q: int64 数组，形状同 w。
        scales: (O,) 的 float64 数组，每个都大于 0。
        反量化的办法：q * scales.reshape(O, 1, ..., 1)。
        每个通道里绝对值最大的那个权重，量化后正好是 +qmax 或 -qmax。

    异常
        w 不到二维、或者 num_bits 不是 2 到 16 之间的整数时抛出 ValueError。

    不要修改 w。
    """
    raise NotImplementedError("TODO: 实现逐通道的权重量化")


def fold_bn_into_conv(
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把推理模式的 BatchNorm 折叠进它前面的卷积，返回新卷积的权重和偏置。

    参数
        conv_weight: (O, I/groups, kH, kW) 的浮点张量，卷积的权重。全连接的 (O, I) 也要能处理。
        conv_bias: (O,) 的张量，或者 None（卷积没有偏置，当作全 0）。
        bn_weight, bn_bias: (O,)，BatchNorm 的缩放（gamma）和偏移（beta）。
        running_mean, running_var: (O,)，BatchNorm 的运行均值和运行方差。
        eps: BatchNorm 里加在方差上的小常数。注意它加在方差上、在开方之前，
            和 torch.nn.BatchNorm2d 的定义一致。

    要求
        推理模式的 BatchNorm 对每个通道做的是：
            y = bn_weight * (z - running_mean) / sqrt(running_var + eps) + bn_bias
        其中 z 是卷积的输出。返回 (w_fold, b_fold)，使得
            conv(x; w_fold, b_fold)
        对任意输入 x 都等于原来的 “conv(x; conv_weight, conv_bias) 之后接这个 BatchNorm”，
        步长、补边、分组这些卷积的其他设置不变。

    返回 (w_fold, b_fold)
        w_fold: 形状、dtype 同 conv_weight。
        b_fold: (O,)，dtype 同 conv_weight。conv_bias 为 None 时也要返回偏置。
        两个都是新的张量，不带梯度（requires_grad 为 False）。

    异常
        bn_weight、bn_bias、running_mean、running_var（以及不为 None 的 conv_bias）
        不是长度为 O 的一维张量时抛出 ValueError。

    不要修改任何传入的张量。
    """
    raise NotImplementedError("TODO: 把 BatchNorm 折叠进卷积")


def plan_batch(
    latency_ms: dict,
    takt_ms: float,
    deadline_ms: float,
    percentile: float = 99.0,
) -> tuple[int, dict]:
    """由各个批大小的延迟样本，选出满足节拍和延迟要求的最大批大小。

    场景
        图像一张接一张地来，相邻两张的间隔是 takt_ms（节拍）。推理程序每攒够 B 张就算一批，
        一批从开始算到全部出结果要 L_B 毫秒。每张图从到达到出结果不能超过 deadline_ms。

    参数
        latency_ms: 字典，键是批大小（正整数），值是这个批大小下“整批耗时”的若干次测量（毫秒），
            一维的列表或数组，至少一个样本。
        takt_ms: 节拍，大于 0。
        deadline_ms: 允许的最大延迟，大于 0。
        percentile: 用延迟的哪个分位数做判断，0 到 100 之间。默认 99，也就是 P99。

    计算
        对每个批大小 B：L_B = np.percentile(样本, percentile)（NumPy 默认的线性插值）。
        B 可行，要同时满足两条：
          1. 跟得上：L_B <= B * takt_ms。攒下一批要 B 个节拍，这一批必须在那之前算完，否则越积越多。
          2. 来得及：(B - 1) * takt_ms + L_B <= deadline_ms。一批里最先到的那张图，
             要先等后面 B - 1 张到齐，再等整批算完。
        两条都是“小于等于”，正好相等算可行。

    返回 (best, stats)
        best: 可行的批大小里最大的那个（Python int）；一个可行的都没有时返回 0。
        stats: 字典 {批大小 B: L_B}，键是 Python int，值是 Python float，
            latency_ms 里的每个批大小都要有，不管可行不可行。

    异常
        latency_ms 为空、某个批大小不是正整数、某个批大小没有样本、
        takt_ms 或 deadline_ms 小于等于 0、percentile 不在 [0, 100] 内时抛出 ValueError。

    不要修改传入的字典和里面的数组。
    """
    raise NotImplementedError("TODO: 实现按节拍选批大小")
