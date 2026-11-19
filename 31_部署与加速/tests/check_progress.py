"""第 31 章自动验收：TODO / FAIL / PASS。"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "31-1_量化与部署零件.py"


def load_assignment():
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    spec = importlib.util.spec_from_file_location("chapter31_assignment", base / ASSIGNMENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(status, name, detail):
    print(f"[{status:4}] {name} — {detail}")


def run(name, func):
    try:
        ok, detail = func()
    except NotImplementedError as exc:
        report("TODO", name, str(exc))
        return False
    except Exception as exc:  # noqa: BLE001
        report("FAIL", name, f"运行异常：{type(exc).__name__}: {exc}")
        return False
    report("PASS" if ok else "FAIL", name, detail)
    return ok


def raises_value_error(fn, *args, **kwargs):
    """只有抛出 ValueError 才算对；不抛异常或者抛出别的异常都返回 False。"""
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    except NotImplementedError:
        raise
    except Exception:  # noqa: BLE001
        return False
    return False


def is_int(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def is_float(v):
    return isinstance(v, (float, np.floating))


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


# ---------------------------------------------------------------- 缩放系数和零点
def check_params(mod):
    out = mod.quant_params(-1.0, 3.0, 8, False)
    if not (isinstance(out, tuple) and len(out) == 4):
        return False, "应返回 (scale, zero_point, qmin, qmax) 四个值"
    scale, zp, qmin, qmax = out
    if not is_float(scale) or not all(is_int(v) for v in (zp, qmin, qmax)):
        return False, "scale 应是浮点数，zero_point、qmin、qmax 应是整数（不是数组，也不是带小数的浮点数）"
    if (int(qmin), int(qmax)) != (0, 255):
        return False, f"8 位非对称量化的整数范围应是 0 到 255，得到 {qmin} 到 {qmax}"
    if not close(scale, 4.0 / 255.0):
        return False, f"范围 [-1, 3]、8 位非对称：一共 255 个间隔，scale 应为 4/255 = {4 / 255:.6f}，得到 {float(scale):.6f}。检查除数是间隔数还是整数的个数"
    if int(zp) != 64:
        return False, (f"范围 [-1, 3]、8 位非对称：实数 0 应落在整数 64 上（1 除以 scale 是 63.75，取到最近的整数），得到 {zp}。"
                       "检查零点的符号，以及是取到最近的整数还是直接截掉小数")
    scale, zp, _, qmax = mod.quant_params(-1.0, 3.0, 4)
    if int(qmax) != 15 or not close(scale, 4.0 / 15.0) or int(zp) != 4:
        return False, f"范围 [-1, 3]、4 位非对称应得到 scale = 4/15、zero_point = 4、qmax = 15；得到 scale = {float(scale):.5f}、zero_point = {zp}、qmax = {qmax}"

    scale, zp, _, _ = mod.quant_params(0.5, 2.0, 8)
    if not close(scale, 2.0 / 255.0) or int(zp) != 0:
        return False, (f"范围 [0.5, 2] 不包含 0 时要先扩到 [0, 2]，应得到 scale = 2/255、zero_point = 0；得到 scale = {float(scale):.6f}、zero_point = {zp}。"
                       "实数 0 必须能被精确表示")
    scale, zp, _, _ = mod.quant_params(-3.0, -1.0, 8)
    if not close(scale, 3.0 / 255.0) or int(zp) != 255:
        return False, f"范围 [-3, -1] 要先扩到 [-3, 0]，应得到 scale = 3/255、zero_point = 255；得到 scale = {float(scale):.6f}、zero_point = {zp}"
    scale, zp, _, _ = mod.quant_params(-0.37, 1.91, 6)
    if not close(scale, 2.28 / 63.0) or int(zp) != 10:
        return False, f"范围 [-0.37, 1.91]、6 位非对称应得到 scale = 2.28/63、zero_point = 10；得到 scale = {float(scale):.6f}、zero_point = {zp}"

    scale, zp, qmin, qmax = mod.quant_params(-1.0, 3.0, 8, True)
    if (int(qmin), int(qmax)) != (-127, 127):
        return False, f"8 位对称量化用窄范围 -127 到 127（不用 -128），得到 {qmin} 到 {qmax}"
    if int(zp) != 0 or not close(scale, 3.0 / 127.0):
        return False, f"范围 [-1, 3]、8 位对称：zero_point 应为 0，scale 应为 3/127 = {3 / 127:.6f}（绝对值最大的一端除以 qmax）；得到 zero_point = {zp}、scale = {float(scale):.6f}"
    scale, zp, qmin, qmax = mod.quant_params(-3.0, 1.0, 4, True)
    if (int(qmin), int(qmax)) != (-7, 7) or int(zp) != 0 or not close(scale, 3.0 / 7.0):
        return False, (f"范围 [-3, 1]、4 位对称应得到 scale = 3/7、整数范围 -7 到 7；得到 scale = {float(scale):.5f}、范围 {qmin} 到 {qmax}。"
                       "绝对值最大的一端可能是负的那一端")

    scale, zp, qmin, qmax = mod.quant_params(0.0, 0.0, 8)
    if not close(scale, 1.0) or int(zp) != 0 or (int(qmin), int(qmax)) != (0, 255):
        return False, "x_min = x_max = 0 时没有范围可言，按文档字符串返回 scale = 1.0、zero_point = 0，不能出现除以 0"
    scale, zp, _, _ = mod.quant_params(0.0, 0.0, 8, True)
    if not close(scale, 1.0) or int(zp) != 0:
        return False, "对称量化、范围全为 0 时应返回 scale = 1.0"
    scale, _, _, _ = mod.quant_params(2.0, 2.0, 8)
    if not close(scale, 2.0 / 255.0):
        return False, "x_min = x_max = 2 时范围扩到 [0, 2]，scale 应为 2/255"

    for bad in (1, 17, 8.5):
        if not raises_value_error(mod.quant_params, -1.0, 1.0, bad):
            return False, f"num_bits = {bad} 时应抛出 ValueError（位数必须是 2 到 16 之间的整数）"
    if not raises_value_error(mod.quant_params, 1.0, -1.0, 8):
        return False, "x_min 大于 x_max 时应抛出 ValueError"
    if not raises_value_error(mod.quant_params, float("nan"), 1.0, 8) or not raises_value_error(mod.quant_params, 0.0, float("inf"), 8):
        return False, "x_min、x_max 里有 nan 或无穷大时应抛出 ValueError"
    return True, "非对称、对称、范围不含 0、位数、退化情形和异常都正确"


# ---------------------------------------------------------------- 量化与反量化
def check_qdq(mod):
    x = np.array([-2.0, -1.0, -0.3, 0.0, 0.2, 0.6, 1.3, 2.4, 9.0])
    keep = x.copy()
    out = mod.quantize_dequantize(x, 0.5, 2, 0, 7)
    if not (isinstance(out, tuple) and len(out) == 2 and all(isinstance(v, np.ndarray) for v in out)):
        return False, "应返回 (q, x_hat) 两个 NumPy 数组"
    q, xh = out
    if not np.array_equal(x, keep):
        return False, "不要修改传入的数组"
    if q.dtype != np.int64 or xh.dtype != np.float64 or q.shape != x.shape or xh.shape != x.shape:
        return False, f"q 应是 int64、x_hat 应是 float64，形状同输入；得到 {q.dtype}{q.shape} 和 {xh.dtype}{xh.shape}"
    want_q = np.array([0, 0, 1, 2, 2, 3, 5, 7, 7])
    if not np.array_equal(q, want_q):
        return False, (f"scale = 0.5、zero_point = 2、整数范围 0 到 7 的小例子：期望 q = {want_q.tolist()}，得到 {q.tolist()}。"
                       "按顺序检查：是不是取到最近的整数（不是向下取整，也不是截掉小数）、零点是加还是减、是不是加完零点之后才截断")
    want_x = np.array([-1.0, -1.0, -0.5, 0.0, 0.0, 0.5, 1.5, 2.5, 2.5])
    if not np.allclose(xh, want_x, atol=1e-12):
        return False, f"同一个小例子：期望 x_hat = {want_x.tolist()}，得到 {xh.tolist()}。反量化是 (q - zero_point) * scale"

    q, xh = mod.quantize_dequantize(np.array([-2.0, 1.274, -1.274, 0.004], dtype=np.float32), 0.01, 0, -127, 127)
    if q.tolist() != [-127, 127, -127, 0] or q.dtype != np.int64 or xh.dtype != np.float64:
        return False, f"有符号的范围（-127 到 127）、输入是 float32：期望 q = [-127, 127, -127, 0]，输出仍是 int64 和 float64；得到 {q.tolist()}（{q.dtype}，{xh.dtype}）"

    q, xh = mod.quantize_dequantize(np.float64(0.26), 0.1, 0, 0, 15)
    if np.shape(q) != () or int(q) != 3 or not close(xh, 0.3):
        return False, "输入是单个数时也要能用：0.26 在 scale = 0.1 下应量化成 3，反量化成 0.3，输出是 0 维数组"

    x2 = np.array([[0.2, 0.6, 1.3], [0.2, 0.6, 1.3]])
    q, xh = mod.quantize_dequantize(x2, np.array([[0.5], [0.1]]), np.array([[0], [3]]), 0, 15)
    if q.shape != (2, 3) or q.tolist() != [[0, 1, 3], [5, 9, 15]]:
        return False, (f"scale 和 zero_point 是形状 (2, 1) 的数组时应按行广播（每行一套参数）：期望 [[0, 1, 3], [5, 9, 15]]，"
                       f"得到 {np.asarray(q).tolist()}")
    if not np.allclose(xh, [[0.0, 0.5, 1.5], [0.2, 0.6, 1.2]], atol=1e-12):
        return False, "逐行广播时 x_hat 也要用各行自己的 scale 和 zero_point"

    rng = np.random.default_rng(0)
    scale, zp, qmin, qmax = 0.037, 91, 0, 255
    lo, hi = (qmin - zp) * scale, (qmax - zp) * scale
    x = rng.uniform(lo, hi, size=(4, 500))
    q, xh = mod.quantize_dequantize(x, scale, zp, qmin, qmax)
    if q.min() < qmin or q.max() > qmax:
        return False, "q 不能超出 [qmin, qmax]"
    if float(np.abs(xh - x).max()) > scale / 2 + 1e-12:
        return False, f"范围内的数，量化再反量化的误差不应超过半格（scale / 2 = {scale / 2:.4f}），得到最大误差 {float(np.abs(xh - x).max()):.4f}"
    if float(np.abs(xh - x).max()) < scale / 4:
        return False, "误差小得不正常：是不是没有取整"
    q, xh = mod.quantize_dequantize(np.array([lo - 3.0, hi + 3.0, -1e20, 1e20]), scale, zp, qmin, qmax)
    if q.tolist() != [qmin, qmax, qmin, qmax] or not np.allclose(xh, [lo, hi, lo, hi], atol=1e-12):
        return False, (f"范围外的数应截到范围的两端（q 为 {qmin} 和 {qmax}），很大的数（正负 1e20）也一样；得到 q = {q.tolist()}。"
                       "如果只有 1e20 那两个不对，检查是不是在截断之前就转成了整数类型（会溢出）")

    if not raises_value_error(mod.quantize_dequantize, x, 0.0, 0, 0, 255):
        return False, "scale 为 0 时应抛出 ValueError"
    if not raises_value_error(mod.quantize_dequantize, x2, np.array([[0.5], [-0.1]]), 0, 0, 255):
        return False, "scale 数组里有负数时应抛出 ValueError"
    return True, "小例子、有符号范围、广播、半格误差、截断和异常都正确"


# ---------------------------------------------------------------- 逐通道量化权重
def check_per_channel(mod):
    w = np.array([[[[0.6]], [[-1.0]], [[0.25]]], [[[0.02]], [[0.011]], [[-0.004]]]])  # (2, 3, 1, 1)
    keep = w.copy()
    out = mod.quantize_weight_per_channel(w, 3)
    if not (isinstance(out, tuple) and len(out) == 2 and all(isinstance(v, np.ndarray) for v in out)):
        return False, "应返回 (q, scales) 两个 NumPy 数组"
    q, scales = out
    if not np.array_equal(w, keep):
        return False, "不要修改 w"
    if q.shape != w.shape or q.dtype != np.int64:
        return False, f"q 的形状应同 w、dtype 为 int64；得到 {q.shape}、{q.dtype}"
    if scales.shape != (2,) or scales.dtype != np.float64:
        return False, f"scales 应是 (O,) 的 float64 数组，这里 O = 2（第 0 维是输出通道）；得到 {scales.shape}、{scales.dtype}"
    if not np.allclose(scales, [1.0 / 3.0, 0.02 / 3.0], rtol=1e-9):
        return False, (f"3 位时 qmax = 3，两个通道绝对值最大的权重是 1.0 和 0.02，scales 应为 [1/3, 0.02/3]；得到 {scales.tolist()}。"
                       "检查：是不是每个输出通道各算各的、取的是不是绝对值的最大值、除的是不是 qmax")
    if q.reshape(2, 3).tolist() != [[2, -3, 1], [3, 2, -1]]:
        return False, f"同一个小例子：期望 q = [[2, -3, 1], [3, 2, -1]]，得到 {q.reshape(2, 3).tolist()}。第二个通道的权重很小，逐通道量化时不应该全变成 0"

    q, scales = mod.quantize_weight_per_channel(np.array([[0.5, -0.127], [0.0, 0.0], [0.004, 0.0021]], dtype=np.float32), 8)
    if q.dtype != np.int64 or scales.dtype != np.float64 or q.shape != (3, 2):
        return False, "全连接的二维权重（输入是 float32）也要能用，输出仍是 int64 和 float64"
    if not np.all(np.isfinite(scales)) or not close(scales[1], 1.0) or q[1].tolist() != [0, 0]:
        return False, f"某个通道的权重全为 0 时，这个通道的 scale 取 1.0、q 全为 0，不能出现 nan；得到 scales = {scales.tolist()}"
    if q[0].tolist() != [127, -32] or q[2].tolist() != [127, 67]:
        return False, f"8 位、二维权重的小例子：期望第 0 行 [127, -32]、第 2 行 [127, 67]，得到 {q[0].tolist()} 和 {q[2].tolist()}"

    rng = np.random.default_rng(1)
    w = rng.normal(size=(6, 4, 3, 3)) * (10.0 ** -np.arange(6))[:, None, None, None]
    for bits in (8, 4):
        qmax = 2 ** (bits - 1) - 1
        q, scales = mod.quantize_weight_per_channel(w, bits)
        if not np.array_equal(np.abs(q).reshape(6, -1).max(axis=1), np.full(6, qmax)):
            return False, f"{bits} 位：每个通道里绝对值最大的权重量化后应正好是正负 {qmax}（各通道的范围相差 10 万倍，逐通道量化时每个通道都能用满整数范围）"
        err = np.abs(q * scales[:, None, None, None] - w).reshape(6, -1).max(axis=1)
        if np.any(err > scales / 2 + 1e-15):
            return False, f"{bits} 位：用 q * scales.reshape(O, 1, 1, 1) 反量化后，每个通道的误差不应超过这个通道的半格"
        if q.min() < -qmax or q.max() > qmax:
            return False, f"{bits} 位：q 应在 -{qmax} 到 {qmax} 之间"

    if not raises_value_error(mod.quantize_weight_per_channel, np.ones(5), 8):
        return False, "w 只有一维时应抛出 ValueError"
    if not raises_value_error(mod.quantize_weight_per_channel, np.ones((2, 2)), 1) or not raises_value_error(mod.quantize_weight_per_channel, np.ones((2, 2)), 17):
        return False, "num_bits 不在 2 到 16 之间时应抛出 ValueError"
    return True, "逐通道的缩放系数、取整、全 0 通道、范围相差很大的通道和异常都正确"


# ---------------------------------------------------------------- BN 折叠
def check_fold(mod):
    cw = torch.tensor([[[[2.0]]], [[[-1.0]]]], dtype=torch.float64)  # (2, 1, 1, 1)
    cb = torch.tensor([0.5, 0.0], dtype=torch.float64)
    gamma = torch.tensor([3.0, 0.5], dtype=torch.float64)
    beta = torch.tensor([1.0, -2.0], dtype=torch.float64)
    mean = torch.tensor([0.5, 1.0], dtype=torch.float64)
    var = torch.tensor([3.91, 0.16], dtype=torch.float64)  # 加上 eps = 0.09 之后是 4 和 0.25，开方是 2 和 0.5
    args = (cw, cb, gamma, beta, mean, var)
    keep = [t.clone() for t in args]
    out = mod.fold_bn_into_conv(*args, eps=0.09)
    if not (isinstance(out, tuple) and len(out) == 2 and all(isinstance(v, torch.Tensor) for v in out)):
        return False, "应返回 (w_fold, b_fold) 两个张量"
    w, b = out
    if any(not torch.equal(a, k) for a, k in zip(args, keep)):
        return False, "不要修改传入的张量"
    if w.shape != cw.shape or b.shape != (2,) or w.dtype != cw.dtype or b.dtype != cw.dtype:
        return False, f"w_fold 的形状和 dtype 应同 conv_weight，b_fold 应是 (O,)；得到 {tuple(w.shape)}（{w.dtype}）和 {tuple(b.shape)}（{b.dtype}）"
    if not torch.allclose(w.reshape(-1), torch.tensor([3.0, -1.0], dtype=torch.float64), atol=1e-9):
        return False, (f"小例子（方差 3.91 和 0.16，eps = 0.09，所以标准差是 2 和 0.5）：折叠后的权重应为 [3, -1]，得到 {w.reshape(-1).tolist()}。"
                       "检查：除的是标准差还是方差、eps 是不是加在方差上再开方、每个输出通道乘的是不是自己的系数")
    if not torch.allclose(b, torch.tensor([1.0, -3.0], dtype=torch.float64), atol=1e-9):
        return False, (f"同一个小例子：折叠后的偏置应为 [1, -3]，得到 {b.tolist()}。"
                       "检查：卷积原来的偏置有没有先减运行均值、再一起乘系数，最后有没有加上 BN 的偏移")

    torch.manual_seed(0)
    for groups, bias, dtype in ((1, True, torch.float64), (2, False, torch.float64), (1, False, torch.float32)):
        conv = torch.nn.Conv2d(4, 4, 3, stride=2, padding=1, groups=groups, bias=bias).to(dtype)
        bn = torch.nn.BatchNorm2d(4).to(dtype)
        with torch.no_grad():
            bn.weight.copy_(torch.tensor([1.5, -0.7, 0.2, 3.0]))
            bn.bias.copy_(torch.tensor([0.3, -1.0, 2.0, 0.0]))
            bn.running_mean.copy_(torch.tensor([0.5, -2.0, 1.0, 0.1]))
            bn.running_var.copy_(torch.tensor([4.0, 0.01, 1e-7, 0.5]))
        bn.eval()
        x = torch.randn(3, 4, 9, 9, dtype=dtype)
        w, b = mod.fold_bn_into_conv(conv.weight, conv.bias, bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps)
        if w.requires_grad or b.requires_grad:
            return False, "返回的两个张量不应带梯度（requires_grad 应为 False），折叠是推理前的一次性处理"
        if w.dtype != dtype or b.dtype != dtype or b.shape != (4,):
            return False, f"输出的 dtype 应同 conv_weight（这里是 {dtype}），b_fold 应是 (4,)；conv_bias 为 None 时也要返回偏置"
        with torch.no_grad():
            want = bn(conv(x))
            got = torch.nn.functional.conv2d(x, w, b, stride=2, padding=1, groups=groups)
        tol = 1e-9 if dtype == torch.float64 else 2e-3
        diff = float((got - want).abs().max())
        if not math.isfinite(diff) or diff > tol * max(1.0, float(want.abs().max())):
            return False, (f"随机卷积（分组 {groups}，{'带' if bias else '不带'}偏置，{str(dtype).split('.')[-1]}）接推理模式的 BatchNorm2d："
                           f"折叠后的卷积输出和原来最大差 {diff:.3g}，应在数值误差以内。"
                           "这组里有一个通道的运行方差是 1e-7（比 eps 还小），有一个通道的缩放是负数")

    lw = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    w, b = mod.fold_bn_into_conv(lw, None, torch.tensor([2.0, 1.0, 0.5]), torch.zeros(3), torch.tensor([1.0, 0.0, -2.0]), torch.ones(3) - 1e-5)
    if w.shape != (3, 2) or not torch.allclose(w, torch.tensor([[2.0, 4.0], [3.0, 4.0], [2.5, 3.0]]), atol=1e-5) or not torch.allclose(b, torch.tensor([-2.0, 0.0, 1.0]), atol=1e-5):
        return False, "全连接的二维权重 (O, I) 也要能折叠：系数按输出（第 0 维）乘，conv_bias 为 None 时当作全 0"

    if not raises_value_error(mod.fold_bn_into_conv, cw, cb, gamma, beta, mean, torch.ones(3, dtype=torch.float64)):
        return False, "running_var 的长度不等于输出通道数时应抛出 ValueError"
    if not raises_value_error(mod.fold_bn_into_conv, cw, torch.zeros(5, dtype=torch.float64), gamma, beta, mean, var):
        return False, "conv_bias 的长度不等于输出通道数时应抛出 ValueError"
    if not raises_value_error(mod.fold_bn_into_conv, cw, cb, gamma.reshape(2, 1), beta, mean, var):
        return False, "bn_weight 不是一维张量时应抛出 ValueError"
    return True, "小例子、和真正的卷积接 BatchNorm 对拍、没有偏置、全连接、不带梯度和异常都正确"


# ---------------------------------------------------------------- 按节拍选批大小
def check_plan(mod):
    lat = {4: [30.0, 34.0], 1: [10.0, 10.0], 8: np.array([50.0, 70.0]), 2: [12.0, 14.0]}
    keep = {k: np.array(v, dtype=np.float64).copy() for k, v in lat.items()}
    out = mod.plan_batch(lat, 8.0, 40.0, 50.0)
    if not (isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict)):
        return False, "应返回 (best, stats)，stats 是字典"
    best, stats = out
    if set(lat) != set(keep) or any(not np.array_equal(np.asarray(lat[k], dtype=np.float64), keep[k]) for k in keep):
        return False, "不要修改传入的字典和里面的数组"
    if not is_int(best):
        return False, "best 应是整数"
    if set(stats) != {1, 2, 4, 8} or not all(is_int(k) for k in stats) or not all(is_float(v) for v in stats.values()):
        return False, "stats 应包含 latency_ms 里的每个批大小（不管可行不可行），键是整数，值是浮点数"
    want = {1: 10.0, 2: 13.0, 4: 32.0, 8: 60.0}
    if any(not close(stats[k], want[k]) for k in want):
        return False, f"percentile = 50 时各批大小的延迟应是样本的中位数 {want}，得到 { {k: round(float(stats[k]), 3) for k in sorted(stats)} }"
    if int(best) != 2:
        return False, (f"小例子（节拍 8 ms，最多等 40 ms）：批大小 1 的延迟 10 ms 超过了 1 个节拍，跟不上；批大小 2 是 13 ms，不超过 2 个节拍，"
                       f"第一张图等 8 + 13 = 21 ms；批大小 4 是 32 ms，第一张图要等 24 + 32 = 56 ms，超了。应选 2，得到 {best}。"
                       "检查“跟得上”比的是不是 批大小 x 节拍，“来得及”有没有算上等后面几张图到齐的时间")
    best, _ = mod.plan_batch(lat, 8.0, 56.0, 50.0)
    if int(best) != 4:
        return False, f"同样的样本，最多等 56 ms：批大小 4 的延迟正好等于 4 个节拍，第一张图正好等 56 ms，两条都是正好相等，算可行，应选 4；得到 {best}"
    best, _ = mod.plan_batch(lat, 8.0, 200.0, 50.0)
    if int(best) != 8:
        return False, f"同样的样本，最多等 200 ms：批大小 2、4、8 都可行，应选最大的 8；得到 {best}"
    best, stats = mod.plan_batch(lat, 5.0, 1000.0, 50.0)
    if int(best) != 0 or set(stats) != {1, 2, 4, 8}:
        return False, f"节拍 5 ms 时每个批大小都跟不上（折合每张都超过 5 ms），应返回 0，stats 仍然要给全；得到 {best}"

    tail = {1: [5.0] * 10, 2: [1.0] * 98 + [50.0, 100.0]}
    best, stats = mod.plan_batch(tail, 10.0, 1000.0)
    if not close(stats[2], 50.5):
        return False, (f"批大小 2 的 100 个样本里有 98 个 1 ms、1 个 50 ms、1 个 100 ms：默认的 P99 按 np.percentile 的线性插值是 50.5 ms，"
                       f"得到 {float(stats[2]):.3f}。检查默认的分位数是不是 99，用的是不是分位数（不是平均值、中位数或最大值）")
    if int(best) != 1:
        return False, f"上面这组样本，节拍 10 ms：批大小 2 的 P99 是 50.5 ms，超过 2 个节拍，不可行，应选 1；得到 {best}"
    best, stats = mod.plan_batch(tail, 10.0, 1000.0, percentile=50.0)
    if int(best) != 2 or not close(stats[2], 1.0):
        return False, "percentile 参数要用上：同一组样本按中位数（percentile = 50）判断，批大小 2 的延迟是 1 ms，应选 2"

    best, stats = mod.plan_batch({3: (7.0,)}, 4.0, 15.0)
    if int(best) != 3 or not close(stats[3], 7.0):
        return False, "只有一个批大小、一个样本时也要能用：批大小 3、延迟 7 ms、节拍 4 ms、最多等 15 ms，第一张图等 8 + 7 = 15 ms，正好可行"

    if not raises_value_error(mod.plan_batch, {}, 8.0, 40.0):
        return False, "latency_ms 为空时应抛出 ValueError"
    if not raises_value_error(mod.plan_batch, lat, 0.0, 40.0) or not raises_value_error(mod.plan_batch, lat, 8.0, -1.0):
        return False, "takt_ms 或 deadline_ms 小于等于 0 时应抛出 ValueError"
    if not raises_value_error(mod.plan_batch, lat, 8.0, 40.0, 101.0) or not raises_value_error(mod.plan_batch, lat, 8.0, 40.0, -1.0):
        return False, "percentile 不在 0 到 100 之间时应抛出 ValueError"
    if not raises_value_error(mod.plan_batch, {1: [10.0], 2: []}, 8.0, 40.0):
        return False, "某个批大小没有样本时应抛出 ValueError"
    if not raises_value_error(mod.plan_batch, {0: [10.0]}, 8.0, 40.0):
        return False, "批大小不是正整数时应抛出 ValueError"
    return True, "分位数、跟得上和来得及两个条件、正好相等、选最大的、一个都不可行和异常都正确"


def main() -> int:
    print("第 31 章进度检查\n" + "=" * 52)
    try:
        mod = load_assignment()
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "加载作业", f"{type(exc).__name__}: {exc}")
        return 1
    results = [
        run("缩放系数和零点", lambda: check_params(mod)),
        run("量化与反量化", lambda: check_qdq(mod)),
        run("逐通道量化权重", lambda: check_per_channel(mod)),
        run("把 BatchNorm 折叠进卷积", lambda: check_fold(mod)),
        run("按节拍选批大小", lambda: check_plan(mod)),
    ]
    print("=" * 52)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("零件通过；下一步在 Notebook 里把它们接到小卷积网络上：折叠、训练后量化、量延迟。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
