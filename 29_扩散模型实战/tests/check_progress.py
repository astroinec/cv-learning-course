"""第 29 章自动验收：TODO / FAIL / PASS。"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import torch

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "29-1_采样引导与低秩微调.py"


def load_assignment():
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    spec = importlib.util.spec_from_file_location("chapter29_assignment", base / ASSIGNMENT)
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
        # 作业文件里没写完的函数抛出的是 "TODO: ..."。PyTorch 自己也会抛 NotImplementedError
        # （例如对布尔张量做减法），那是实现里的错误，不能显示成“还没写”。
        if str(exc).startswith("TODO"):
            report("TODO", name, str(exc))
        else:
            report("FAIL", name, f"运行异常：{type(exc).__name__}: {exc}")
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
    except NotImplementedError as exc:
        if str(exc).startswith("TODO"):
            raise
        return False
    except Exception:  # noqa: BLE001
        return False
    return False


def t64(v):
    return torch.tensor(v, dtype=torch.float64)


def close(a, b, tol=1e-6):
    """数值是否一致。容差按 float32 的精度留：系数用 float32 算、或者把同一个式子换一种等价的写法，都不应该被判错；
    真正写错的实现差的是百分之几以上，不会落在这个容差里。"""
    if not isinstance(a, torch.Tensor) or a.shape != b.shape or not bool(torch.isfinite(a).all()):
        return False
    return float((a - b).abs().max()) <= tol * (1.0 + float(b.abs().max()))


# ---------------------------------------------------------------- 三种参数化的换算
def check_convert(mod):
    # 手算的小例子：abar = 0.64，sqrt(abar) = 0.8，sqrt(1 - abar) = 0.6；x0 = 2，eps = -1，于是 x_t = 1.0，v = -2.0
    x_t, x0, eps, v = t64([1.0]), t64([2.0]), t64([-1.0]), t64([-2.0])
    keep = x_t.clone()
    for kind, pred in (("eps", eps), ("x0", x0), ("v", v)):
        p0 = pred.clone()
        out = mod.convert_prediction(x_t, pred, 0.64, kind)
        if not (isinstance(out, tuple) and len(out) == 3 and all(isinstance(o, torch.Tensor) for o in out)):
            return False, "应返回 (x0, eps, v) 三个张量"
        if not torch.equal(x_t, keep) or not torch.equal(pred, p0):
            return False, "不要修改传入的 x_t 和 pred"
        for got, want, name in zip(out, (x0, eps, v), ("x0", "eps", "v")):
            if not close(got, want):
                return False, (f"小例子（abar=0.64，x_t=1）：kind='{kind}' 时返回的 {name} 应为 {float(want):g}，得到 {float(got.reshape(-1)[0]):.6g}。"
                               "检查两个系数是 sqrt(abar) 和 sqrt(1 - abar)，有没有写反或者漏了开方；返回值的顺序是 (x0, eps, v)")

    # 第二个例子系数不对称：abar = 0.36，sqrt(abar) = 0.6，sqrt(1 - abar) = 0.8；x0 = 1，eps = 2，于是 x_t = 2.2，v = 0.4
    out = mod.convert_prediction(t64([2.2]), t64([0.4]), 0.36, "v")
    if not close(out[0], t64([1.0])) or not close(out[1], t64([2.0])):
        return False, f"abar=0.36、x_t=2.2、v=0.4 时应得到 x0=1、eps=2，得到 x0={float(out[0]):.4f}、eps={float(out[1]):.4f}。检查 v 的定义里哪一项乘 sqrt(abar)、符号是哪个为负"

    # 性质：任意输入下三者互相一致，而且和 kind 对应的那一个原样返回
    g = torch.Generator().manual_seed(0)
    xt = torch.randn(3, 2, 4, 4, generator=g, dtype=torch.float64)
    pred = torch.randn(3, 2, 4, 4, generator=g, dtype=torch.float64)
    abar = t64([0.9, 0.3, 0.02]).reshape(3, 1, 1, 1)
    for i, kind in enumerate(("x0", "eps", "v")):
        a0, e0, v0 = mod.convert_prediction(xt, pred, abar, kind)
        if a0.shape != xt.shape or e0.shape != xt.shape or v0.shape != xt.shape:
            return False, "abar_t 是形状 (B, 1, 1, 1) 的张量时，三个返回值的形状都应和 x_t 相同"
        if not close((a0, e0, v0)[i], pred):
            return False, f"kind='{kind}' 时，返回的三个量里对应的那一个应等于 pred"
        if not close(abar.sqrt() * a0 + (1 - abar).sqrt() * e0, xt):
            return False, f"kind='{kind}'：返回的 x0 和 eps 应满足 x_t = sqrt(abar) * x0 + sqrt(1 - abar) * eps（abar 每个样本一个值）"
        if not close(abar.sqrt() * e0 - (1 - abar).sqrt() * a0, v0):
            return False, f"kind='{kind}'：返回的 v 应等于 sqrt(abar) * eps - sqrt(1 - abar) * x0"
    out = mod.convert_prediction(xt.float(), pred.float(), 0.5, "eps")
    if any(o.dtype != torch.float32 for o in out):
        return False, "输入是 float32 时返回值也应是 float32（不要悄悄升成 float64）"
    for kind in ("eps", "x0", "v"):
        out = mod.convert_prediction(xt.float(), pred.float(), abar, kind)      # abar 是 float64 的张量
        if any(o.dtype != torch.float32 for o in out):
            return False, (f"kind='{kind}'：x_t 是 float32、abar_t 是 float64 的张量时，返回值仍应是 float32。"
                           "把 abar_t 转成 x_t 的 dtype 再算，否则结果会被提升成 float64，送回 float32 的网络时报错")

    p = pred.clone().requires_grad_(True)
    a0, e0, v0 = mod.convert_prediction(xt, p, 0.3, "v")
    if not (a0.requires_grad and e0.requires_grad and v0.requires_grad):
        return False, "三个返回值都应对 pred 可导（不要 detach，也不要经过 NumPy）"
    (a0.sum() + e0.sum()).backward()
    want = math.sqrt(0.3) - math.sqrt(0.7)
    if p.grad is None or not close(p.grad, torch.full_like(p, want)):
        return False, "kind='v' 时 x0 和 eps 对 pred 的导数不对，检查由 v 解出 x0 和 eps 的两个式子"

    if not raises_value_error(mod.convert_prediction, xt, pred, 0.5, "noise"):
        return False, "kind 不是 'eps'、'x0'、'v' 之一时应抛出 ValueError"
    if not raises_value_error(mod.convert_prediction, xt, pred[:, :1], 0.5, "eps"):
        return False, "x_t 和 pred 形状不同时应抛出 ValueError"
    for bad in (0.0, 1.0, -0.1, 1.5):
        if not raises_value_error(mod.convert_prediction, xt, pred, bad, "eps"):
            return False, f"abar_t={bad} 时应抛出 ValueError（必须严格在 0 和 1 之间，否则换算要除以 0）"
    if not raises_value_error(mod.convert_prediction, xt, pred, t64([0.5, 1.0, 0.2]).reshape(3, 1, 1, 1), "x0"):
        return False, "abar_t 是张量、其中有一个值等于 1 时也应抛出 ValueError"
    return True, "三种输入的换算、互相一致、逐样本的 abar、可导和异常都正确"


# ---------------------------------------------------------------- DDIM 的一步
def check_ddim(mod):
    # 手算：abar_t = 0.25（两个根号都是 0.5 和 0.866），abar_prev = 0.64（0.8 和 0.6）。x_t = 1，eps = 0.5
    r75 = math.sqrt(0.75)
    x_t, eps = t64([[1.0, -2.0]]), t64([[0.5, 1.0]])
    k1, k2 = x_t.clone(), eps.clone()
    x0_hat = (x_t - r75 * eps) / 0.5
    got = mod.ddim_step(x_t, eps, 0.25, 0.64)
    if not isinstance(got, torch.Tensor) or got.shape != x_t.shape or got.dtype != x_t.dtype:
        return False, "应返回和 x_t 形状、dtype 相同的张量"
    if not torch.equal(x_t, k1) or not torch.equal(eps, k2):
        return False, "不要修改传入的 x_t 和 eps"
    want = 0.8 * x0_hat + 0.6 * eps
    if not close(got, want):
        return False, (f"eta=0 的小例子（abar_t=0.25，abar_prev=0.64，x_t=1，eps=0.5）：先得到 x0 的估计 {float(x0_hat[0, 0]):.4f}，结果应为 {float(want[0, 0]):.4f}，得到 {float(got[0, 0]):.4f}。"
                       "检查：估计 x0 时用的是 abar_t，重新组合时用的是 abar_prev；噪声项的系数是 sqrt(1 - abar_prev)")
    got = mod.ddim_step(x_t, eps, 0.25, 0.64, 0.0, torch.full_like(x_t, 100.0))
    if not close(got, want):
        return False, "eta=0 时结果应是确定的，传进来的 noise 不应起作用"
    got = mod.ddim_step(x_t, eps, 0.25, 1.0)
    if not close(got, x0_hat):
        return False, "abar_prev=1 是采样的最后一步，结果应等于对 x0 的估计 (x_t - sqrt(1 - abar_t) * eps) / sqrt(abar_t)"
    got = mod.ddim_step(x_t, eps, 0.25, 1.0, 1.0, torch.full_like(x_t, 7.0))
    if not close(got, x0_hat):
        return False, "abar_prev=1 时即使 eta=1，sigma 也是 0，结果不应带噪声"
    got = mod.ddim_step(x_t, eps, 0.64, 0.64)
    if not close(got, x_t):
        return False, "abar_prev 等于 abar_t 时（eta=0）这一步不应改变 x_t"
    got = mod.ddim_step(x_t, eps, 0.64, 0.25)
    want_back = 0.5 * (x_t - 0.6 * eps) / 0.8 + r75 * eps
    if not close(got, want_back):
        return False, "eta=0 时 abar_prev 可以小于 abar_t（往噪声更多的方向走一步），公式照样成立，不应报错或截断"

    # eta=1：sigma^2 = (0.36 / 0.75) * (1 - 0.25 / 0.64) = 0.2925，方向项的系数是 sqrt(0.36 - 0.2925)
    var = 0.36 / 0.75 * (1 - 0.25 / 0.64)
    zero = torch.zeros_like(x_t)
    base = mod.ddim_step(x_t, eps, 0.25, 0.64, 1.0, zero)
    want = 0.8 * x0_hat + math.sqrt(0.36 - var) * eps
    if not close(base, want):
        return False, (f"eta=1、noise 全 0 的小例子：sigma^2 应为 {var:.4f}，结果应为 {float(want[0, 0]):.4f}，得到 {float(base[0, 0]):.4f}。"
                       "检查：eps 那一项的系数是 sqrt(1 - abar_prev - sigma^2)，不是 sqrt(1 - abar_prev)")
    one = mod.ddim_step(x_t, eps, 0.25, 0.64, 1.0, torch.ones_like(x_t))
    if not close(one - base, torch.full_like(x_t, math.sqrt(var))):
        return False, f"eta=1 时 noise 的系数应为 sigma = {math.sqrt(var):.4f}（噪声取 1 和取 0 的结果之差），得到 {float((one - base)[0, 0]):.4f}。检查 sigma 有没有开方"
    half = mod.ddim_step(x_t, eps, 0.25, 0.64, 0.5, torch.ones_like(x_t)) - mod.ddim_step(x_t, eps, 0.25, 0.64, 0.5, zero)
    if not close(half, torch.full_like(x_t, 0.5 * math.sqrt(var))):
        return False, "sigma 应和 eta 成正比：eta=0.5 时 noise 的系数应是 eta=1 时的一半"

    # 性质：如果 eps 就是真正加进去的噪声，eta=0 的一步应落在同一个 (x0, eps) 在 abar_prev 处的位置上
    g = torch.Generator().manual_seed(1)
    x0 = torch.randn(4, 1, 5, 5, generator=g, dtype=torch.float64)
    e = torch.randn(4, 1, 5, 5, generator=g, dtype=torch.float64)
    xt = math.sqrt(0.1) * x0 + math.sqrt(0.9) * e
    got = mod.ddim_step(xt, e, 0.1, 0.7)
    if not close(got, math.sqrt(0.7) * x0 + math.sqrt(0.3) * e):
        return False, "eps 就是真实噪声时，eta=0 的一步应正好落在 sqrt(abar_prev) * x0 + sqrt(1 - abar_prev) * eps 上"
    if mod.ddim_step(xt.float(), e.float(), 0.1, 0.7).dtype != torch.float32:
        return False, "输入是 float32 时返回值也应是 float32"

    xg, eg = xt.clone().requires_grad_(True), e.clone().requires_grad_(True)
    mod.ddim_step(xg, eg, 0.25, 0.64).sum().backward()
    if xg.grad is None or eg.grad is None or not close(xg.grad, torch.full_like(xg, 0.8 / 0.5)):
        return False, "结果应对 x_t 和 eps 可导；eta=0 时对 x_t 的导数是 sqrt(abar_prev) / sqrt(abar_t)"

    if not raises_value_error(mod.ddim_step, x_t, eps, 0.25, 0.64, 1.0, None) or not raises_value_error(mod.ddim_step, x_t, eps, 0.25, 0.64, 0.3):
        return False, "eta 大于 0 而没有传 noise 时应抛出 ValueError（不要在函数里自己生成随机数）"
    for a_t, a_p in ((0.0, 0.5), (1.0, 1.0), (-0.2, 0.5), (0.3, 0.0), (0.3, 1.2)):
        if not raises_value_error(mod.ddim_step, x_t, eps, a_t, a_p):
            return False, f"abar_t={a_t}、abar_prev={a_p} 时应抛出 ValueError（abar_t 应在 (0, 1) 内，abar_prev 应在 (0, 1] 内）"
    if not raises_value_error(mod.ddim_step, x_t, eps, 0.25, 0.64, 1.5, zero) or not raises_value_error(mod.ddim_step, x_t, eps, 0.25, 0.64, -0.1, zero):
        return False, "eta 不在 [0, 1] 内时应抛出 ValueError"
    if not raises_value_error(mod.ddim_step, x_t, eps, 0.64, 0.25, 1.0, zero):
        return False, "eta 大于 0 而 abar_prev 小于 abar_t 时应抛出 ValueError（这时 sigma^2 是负数）"
    return True, "确定性的一步、最后一步、带噪声的一步、可导和异常都正确"


# ---------------------------------------------------------------- 无分类器引导
def check_cfg(mod):
    u, c = t64([[1.0, -2.0, 0.5]]), t64([[3.0, -1.0, 0.5]])
    ku, kc = u.clone(), c.clone()
    got = mod.cfg_combine(u, c, 3.0)
    if not isinstance(got, torch.Tensor) or got.shape != u.shape or got.dtype != u.dtype:
        return False, "应返回和 eps_uncond 形状、dtype 相同的张量"
    if not torch.equal(u, ku) or not torch.equal(c, kc):
        return False, "不要修改传入的两个预测（不要用原地运算）"
    if not close(got, t64([[7.0, 1.0, 0.5]])):
        return False, (f"无条件 [1, -2, 0.5]、有条件 [3, -1, 0.5]、scale=3 时应为 [7, 1, 0.5]，得到 {got.tolist()[0]}。"
                       "检查：是从无条件的预测出发、沿“有条件减无条件”的方向走 scale 倍；两个预测有没有写反；scale 是不是多加了 1")
    if not close(mod.cfg_combine(u, c, 0.0), u):
        return False, "scale=0 时应等于无条件的预测"
    if not close(mod.cfg_combine(u, c, 1.0), c):
        return False, "scale=1 时应等于有条件的预测（本函数用 diffusers 的 guidance_scale 约定，不是论文里的 w）"
    if not close(mod.cfg_combine(u, c, -1.0), 2 * u - c) or not close(mod.cfg_combine(u, c, 50.0), u + 50.0 * (c - u)):
        return False, "scale 可以小于 0 或者很大，不要做截断"

    g = torch.Generator().manual_seed(2)
    eu = torch.randn(3, 2, 4, 4, generator=g, dtype=torch.float64)
    ec = torch.randn(3, 2, 4, 4, generator=g, dtype=torch.float64)
    s = t64([0.0, 1.0, 4.0]).reshape(3, 1, 1, 1)
    got = mod.cfg_combine(eu, ec, s)
    if got.shape != eu.shape or not close(got[0], eu[0]) or not close(got[1], ec[1]) or not close(got[2], 4 * ec[2] - 3 * eu[2]):
        return False, "scale 是形状 (B, 1, 1, 1) 的张量时，应逐样本使用各自的强度"
    if mod.cfg_combine(eu.float(), ec.float(), 2.0).dtype != torch.float32:
        return False, "输入是 float32 时返回值也应是 float32"
    got32 = mod.cfg_combine(eu.float(), ec.float(), s)                           # s 是 float64 的张量
    if got32.dtype != torch.float32 or not close(got32.double(), got, 1e-5):
        return False, ("两个预测是 float32、scale 是 float64 的张量时，返回值仍应是 float32，数值不变。"
                       "把 scale 转成预测的 dtype 再算，否则结果会被提升成 float64")

    a, b = eu.clone().requires_grad_(True), ec.clone().requires_grad_(True)
    mod.cfg_combine(a, b, 3.0).sum().backward()
    if a.grad is None or b.grad is None or not close(a.grad, torch.full_like(a, -2.0)) or not close(b.grad, torch.full_like(b, 3.0)):
        return False, "结果应对两个预测都可导：scale=3 时对无条件预测的导数是 -2，对有条件预测的导数是 3"
    if not raises_value_error(mod.cfg_combine, eu, ec[:, :1], 2.0) or not raises_value_error(mod.cfg_combine, eu[:1], ec, 2.0):
        return False, "两个预测形状不同时应抛出 ValueError，而不是靠广播算出一个结果"
    return True, "数值、两个特殊强度、逐样本的强度、可导和异常都正确"


# ---------------------------------------------------------------- LoRA 的前向与合并
def check_lora(mod):
    # 手算：in=3，out=2，r=1，alpha=2（缩放 2）。原来的一路（含偏置）给出 [7.5, -1.5]，旁路给出 2 * 2 * [3, -1] = [12, -4]
    w = t64([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])
    bias = t64([0.5, -0.5])
    a = t64([[1.0, 2.0, -1.0]])
    b = t64([[3.0], [-1.0]])
    x = t64([[1.0, 2.0, 3.0]])
    keep = [t.clone() for t in (x, w, bias, a, b)]
    got = mod.lora_forward(x, w, bias, a, b, 2.0)
    if not isinstance(got, torch.Tensor) or got.shape != (1, 2) or got.dtype != x.dtype:
        return False, "lora_forward 应返回 (..., out) 的张量，dtype 与 x 相同"
    if any(not torch.equal(p, q) for p, q in zip((x, w, bias, a, b), keep)):
        return False, "lora_forward 不要修改传入的张量"
    if not close(got, t64([[19.5, -5.5]])):
        return False, (f"小例子：原来的一路（含偏置）是 [7.5, -1.5]，旁路是 (alpha / r) x B (A x) = 2 x [3, -1] x 2 = [12, -4]，期望 [19.5, -5.5]，得到 {got.tolist()[0]}。"
                       "检查：缩放是 alpha / r 不是 alpha；先乘 A 再乘 B；偏置有没有加")
    if not close(mod.lora_forward(x, w, None, a, b, 2.0), t64([[19.0, -5.0]])):
        return False, "bias 为 None 时不加偏置，其余不变"
    merged = mod.lora_merge(w, a, b, 2.0)
    if not isinstance(merged, torch.Tensor) or merged.shape != w.shape or merged.dtype != w.dtype:
        return False, "lora_merge 应返回和 weight 形状、dtype 相同的张量"
    if any(not torch.equal(p, q) for p, q in zip((w, a, b), keep[1:2] + keep[3:])):
        return False, "lora_merge 不要原地修改 weight（返回新张量）"
    if not close(merged, t64([[7.0, 12.0, -4.0], [-2.0, -3.0, 1.0]])):
        return False, (f"小例子里合并后的权重应为 [[7, 12, -4], [-2, -3, 1]]，得到 {merged.tolist()}。"
                       "检查：增量是 B @ A（形状 (out, in)），不是 A @ B，也不是它的转置；缩放是 alpha / r")

    # 随机例子：r=3、alpha=6，带前导维度；前向应和“先合并再做普通线性层”一致
    g = torch.Generator().manual_seed(3)
    w = torch.randn(4, 5, generator=g, dtype=torch.float64)
    bias = torch.randn(4, generator=g, dtype=torch.float64)
    a = torch.randn(3, 5, generator=g, dtype=torch.float64)
    b = torch.randn(4, 3, generator=g, dtype=torch.float64)
    x = torch.randn(2, 7, 5, generator=g, dtype=torch.float64)
    y = mod.lora_forward(x, w, bias, a, b, 6.0)
    if y.shape != (2, 7, 4):
        return False, f"x 是 (2, 7, 5) 时输出应为 (2, 7, 4)，得到 {tuple(y.shape)}（x 前面可以有任意多个维度）"
    ref = torch.nn.functional.linear(x, mod.lora_merge(w, a, b, 6.0), bias)
    if not close(y, ref):
        return False, "lora_forward 的输出应和“先用 lora_merge 合并、再做普通线性层”的输出相同。两个函数里的缩放、B 和 A 的顺序要一致"
    direct = torch.nn.functional.linear(x, w + 2.0 * (b @ a), bias)
    if not close(y, direct):
        return False, "r=3、alpha=6 时增量应乘 alpha / r = 2。检查 r 是不是取了 lora_a 的行数"
    if not close(mod.lora_forward(x, w, bias, a, torch.zeros_like(b), 6.0), torch.nn.functional.linear(x, w, bias)):
        return False, "lora_b 全为 0 时输出应和原来的线性层完全相同（这就是微调开始时不改变原模型的原因）"
    if not close(mod.lora_merge(w, a, torch.zeros_like(b), 6.0), w):
        return False, "lora_b 全为 0 时合并后的权重应等于原来的权重"
    if mod.lora_forward(x.float(), w.float(), bias.float(), a.float(), b.float(), 6.0).dtype != torch.float32:
        return False, "输入是 float32 时返回值也应是 float32"

    wg, ag, bg = w.clone().requires_grad_(False), a.clone().requires_grad_(True), b.clone().requires_grad_(True)
    xg = x.clone().requires_grad_(True)
    mod.lora_forward(xg, wg, bias, ag, bg, 6.0).sum().backward()
    if ag.grad is None or bg.grad is None or xg.grad is None or float(ag.grad.abs().sum()) == 0.0 or float(bg.grad.abs().sum()) == 0.0:
        return False, "输出应对 x、lora_a、lora_b 可导（训练时梯度要能传到 A 和 B）"
    want_b = 2.0 * (x @ a.t()).sum(dim=(0, 1))
    if not close(bg.grad, want_b[None].expand(4, 3)):
        return False, "对 lora_b 的梯度不对：它应等于缩放系数乘上降维之后的特征 x @ A.T（对样本求和）"
    m = mod.lora_merge(w, ag, bg, 6.0)
    if not m.requires_grad:
        return False, "lora_merge 的结果应对 lora_a、lora_b 可导（不要 detach）"

    for fn, args in ((mod.lora_forward, (x, w, bias)), (mod.lora_merge, (w,))):
        name = fn.__name__
        if not raises_value_error(fn, *args, a[:, :4], b, 6.0):
            return False, f"{name}：lora_a 的列数不等于 in 时应抛出 ValueError"
        if not raises_value_error(fn, *args, a, b[:, :2], 6.0) or not raises_value_error(fn, *args, a, b[:3], 6.0):
            return False, f"{name}：lora_b 的形状不是 (out, r) 时应抛出 ValueError"
        if not raises_value_error(fn, *args, a[0], b, 6.0):
            return False, f"{name}：lora_a 不是二维时应抛出 ValueError"
        if not raises_value_error(fn, *args, a.t().contiguous(), b.t().contiguous(), 6.0):
            return False, f"{name}：把 A 和 B 的摆法弄反（A 是 (in, r)、B 是 (r, out)）时应抛出 ValueError"
    if not raises_value_error(mod.lora_forward, x[..., :4], w, bias, a, b, 6.0):
        return False, "lora_forward：x 的最后一维不等于 in 时应抛出 ValueError"
    return True, "前向、合并、两者一致、零初始化不改变输出、梯度和异常都正确"


# ---------------------------------------------------------------- 修复采样里的替换
def check_inpaint(mod):
    x_t = t64([[[[10.0, 20.0], [30.0, 40.0]]]])
    known = t64([[[[1.0, 2.0], [3.0, 4.0]]]])
    noise = t64([[[[1.0, -1.0], [2.0, 0.0]]]])
    mask = t64([[[[1.0, 0.0], [0.0, 0.5]]]])
    keep = [t.clone() for t in (x_t, known, noise, mask)]
    got = mod.inpaint_replace(x_t, known, mask, 0.64, noise)
    if not isinstance(got, torch.Tensor) or got.shape != x_t.shape or got.dtype != x_t.dtype:
        return False, "应返回和 x_t 形状、dtype 相同的张量"
    if any(not torch.equal(p, q) for p, q in zip((x_t, known, noise, mask), keep)):
        return False, "不要修改传入的张量（不要在 x_t 上原地赋值）"
    # 已知图加噪到 abar=0.64：0.8 * known + 0.6 * noise = [[1.4, 1.0], [3.6, 3.2]]
    want = t64([[[[10.0, 1.0], [3.6, 21.6]]]])
    if not close(got, want):
        return False, (f"小例子（abar=0.64）：已知图加噪后是 [[1.4, 1.0], [3.6, 3.2]]；mask 为 1 的像素保留 x_t，为 0 的像素换成加噪后的已知图，0.5 的像素各取一半。"
                       f"期望 {want.reshape(-1).tolist()}，得到 {got.reshape(-1).tolist()}。检查：mask 的含义有没有弄反；已知图是不是先按前向过程加了噪；两个系数有没有开方")
    got = mod.inpaint_replace(x_t, known, mask, 1.0, noise)
    if not close(got, t64([[[[10.0, 2.0], [3.0, 22.0]]]])):
        return False, "abar_t=1 表示已经是干净的图：已知区域应正好等于 known_x0，不带任何噪声"
    got = mod.inpaint_replace(x_t, known, torch.zeros_like(mask), 0.0, noise)
    if not close(got, noise):
        return False, "abar_t=0、mask 全 0 时结果应等于 noise（已知图加噪到头就是纯噪声）"

    g = torch.Generator().manual_seed(4)
    xt = torch.randn(3, 2, 6, 6, generator=g, dtype=torch.float64)
    kn = torch.randn(3, 2, 6, 6, generator=g, dtype=torch.float64)
    nz = torch.randn(3, 2, 6, 6, generator=g, dtype=torch.float64)
    m = (torch.rand(3, 1, 6, 6, generator=g) < 0.4).to(torch.float64)
    got = mod.inpaint_replace(xt, kn, m, 0.3, nz)
    if got.shape != xt.shape:
        return False, "mask 是 (B, 1, H, W) 时应广播到所有通道，结果形状和 x_t 相同"
    full = m.expand_as(xt) > 0.5
    if not close(got[full], xt[full]):
        return False, "mask 为 1 的像素应等于 x_t"
    if not close(got[~full], (math.sqrt(0.3) * kn + math.sqrt(0.7) * nz)[~full]):
        return False, "mask 为 0 的像素应等于 sqrt(abar_t) * known_x0 + sqrt(1 - abar_t) * noise"
    got2 = mod.inpaint_replace(xt + 5.0 * (1 - m), kn, m, 0.3, nz)
    if not close(got2, got):
        return False, "已知区域的结果不应依赖 x_t 在那里的取值"
    if mod.inpaint_replace(xt.float(), kn.float(), m.float(), 0.3, nz.float()).dtype != torch.float32:
        return False, "输入是 float32 时返回值也应是 float32"
    got_b = mod.inpaint_replace(xt, kn, m > 0.5, 0.3, nz)
    if got_b.dtype != torch.float64 or not close(got_b, got):
        return False, "mask 是布尔张量时也要能用（True 当作 1），结果和同样内容的浮点掩膜相同，dtype 仍和 x_t 相同"
    got32 = mod.inpaint_replace(xt.float(), kn.float(), m, 0.3, nz.float())            # m 是 float64
    if got32.dtype != torch.float32 or not close(got32.double(), got, 1e-5):
        return False, "x_t 是 float32、mask 是 float64 时，返回值仍应是 float32。把 mask 转成 x_t 的 dtype 再算"

    xg = xt.clone().requires_grad_(True)
    mod.inpaint_replace(xg, kn, m, 0.3, nz).sum().backward()
    if xg.grad is None or not close(xg.grad, m.expand_as(xt)):
        return False, "结果应对 x_t 可导，梯度只落在 mask 为 1 的像素上"

    for bad in (-0.1, 1.2):
        if not raises_value_error(mod.inpaint_replace, xt, kn, m, bad, nz):
            return False, f"abar_t={bad} 时应抛出 ValueError（应在 [0, 1] 内）"
    if not raises_value_error(mod.inpaint_replace, xt, kn, m * 2.0, 0.3, nz) or not raises_value_error(mod.inpaint_replace, xt, kn, m - 0.5, 0.3, nz):
        return False, "mask 有大于 1 或小于 0 的值时应抛出 ValueError"
    if not raises_value_error(mod.inpaint_replace, xt, kn[:, :1], m, 0.3, nz) or not raises_value_error(mod.inpaint_replace, xt, kn, m, 0.3, nz[:2]):
        return False, "x_t、known_x0、noise 三者形状不全相同时应抛出 ValueError"
    return True, "小例子、两端的噪声水平、掩膜广播、可导和异常都正确"


def main() -> int:
    print("第 29 章进度检查\n" + "=" * 52)
    try:
        mod = load_assignment()
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "加载作业", f"{type(exc).__name__}: {exc}")
        return 1
    results = [
        run("三种参数化之间的换算", lambda: check_convert(mod)),
        run("DDIM 的一步更新", lambda: check_ddim(mod)),
        run("无分类器引导的组合", lambda: check_cfg(mod)),
        run("LoRA 层的前向与合并", lambda: check_lora(mod)),
        run("修复采样里已知区域的替换", lambda: check_inpaint(mod)),
    ]
    print("=" * 52)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("零件通过；下一步在 Notebook 里把它们接到小去噪网络上，做采样、引导、掩膜条件、低秩微调和修复。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
