"""第 30 章自动验收：TODO / FAIL / PASS。"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import torch

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "30-1_体渲染与高斯泼溅.py"
F64 = torch.float64
LN2 = math.log(2.0)


def load_assignment():
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    spec = importlib.util.spec_from_file_location("chapter30_assignment", base / ASSIGNMENT)
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


def call_valid(fn, *args, **kwargs):
    """用合法的输入调用作业函数。返回 (结果, None)；如果它对合法输入抛出了 ValueError，返回 (None, 异常里的文字)。"""
    try:
        return fn(*args, **kwargs), None
    except ValueError as exc:
        return None, str(exc)


def close(a, b, tol=1e-6):
    return a.shape == b.shape and bool(torch.isfinite(a).all()) and float((a - b).abs().max()) <= tol


def fmt(x):
    return [round(float(v), 4) for v in x.reshape(-1)]


# ---------------------------------------------------------------- 体渲染权重
def check_weights(mod):
    # 三段的光学厚度（密度 x 长度）是 ln 4、0、ln 2：不透明度 0.75、0、0.5
    sigma = torch.tensor([2 * math.log(4.0), 0.0, LN2 / 2], dtype=F64)
    deltas = torch.tensor([0.5, 5.0, 2.0], dtype=F64)
    s0, d0 = sigma.clone(), deltas.clone()
    w = mod.volume_weights(sigma, deltas)
    if not isinstance(w, torch.Tensor) or w.shape != (3,):
        return False, "应返回和 sigma 形状相同的张量"
    if not torch.equal(sigma, s0) or not torch.equal(deltas, d0):
        return False, "不要修改传入的 sigma 和 deltas"
    if w.dtype != F64:
        return False, "sigma 是 float64 时返回值也应是 float64"
    want = torch.tensor([0.75, 0.0, 0.125], dtype=F64)
    if not close(w, want, 1e-8):
        return False, (f"小例子：三段的“密度 x 长度”是 ln 4、0、ln 2，不透明度是 0.75、0、0.5；第三段前面还剩 0.25 的光，"
                       f"期望权重 {fmt(want)}，得到 {fmt(w)}。检查：不透明度的公式、长度有没有乘进去、透射率是不是只累乘前面的各段（不含自己）")

    # 批：两条光线，deltas 是 (N,)
    # 第二条光线三段的“密度 x 长度”都是 ln 2：每一段挡掉剩下的一半
    sig2 = torch.stack([sigma, torch.tensor([LN2 / 0.5, LN2 / 5, LN2 / 2], dtype=F64)])
    w2 = mod.volume_weights(sig2, deltas)
    want2 = torch.stack([want, torch.tensor([0.5, 0.25, 0.125], dtype=F64)])
    if not close(w2[0], want2[0], 1e-8):
        return False, f"两条光线、deltas 形状为 (N,) 时，每条光线应各自沿最后一维累乘，互不影响。第一条期望 {fmt(want2[0])}，得到 {fmt(w2[0])}"
    if not close(w2[1], want2[1], 1e-8):
        return False, (f"三段的“密度 x 长度”都是 ln 2 时，每一段挡掉剩下的光的一半，期望权重 {fmt(want2[1])}，得到 {fmt(w2[1])}。"
                       "透射率是前面各段 (1 - 不透明度) 的乘积，不是 1 减去它们的和，也不是只看紧挨着的前一段")
    w3 = mod.volume_weights(sig2.reshape(2, 1, 3).expand(2, 4, 3).float(), deltas.float())
    if w3.shape != (2, 4, 3) or w3.dtype != torch.float32 or not close(w3[:, 2].double(), want2, 1e-6):
        return False, "sigma 形状是 (2, 4, 3) 的 float32 时，返回值应是同样形状的 float32，并且每条光线的结果不受批的维度影响"

    # 两个和具体数值无关的性质
    g = torch.Generator().manual_seed(0)
    sig = torch.rand(5, 8, generator=g, dtype=F64) * 3
    dl = torch.rand(8, generator=g, dtype=F64) + 0.1
    w = mod.volume_weights(sig, dl)
    total = 1 - torch.exp(-(sig * dl).sum(-1))
    if not close(w.sum(-1), total, 1e-8):
        return False, "权重之和应等于 1 - exp(-全部各段“密度 x 长度”之和)：没有被吸收的光就是剩下的透射率。检查透射率的累乘"
    fine = mod.volume_weights(sig.repeat_interleave(2, dim=-1), (dl / 2).repeat_interleave(2))
    if not close(fine.reshape(5, 8, 2).sum(-1), w, 1e-8):
        return False, "把每一段平分成密度相同的两小段，两小段的权重之和应等于原来那一段的权重。这是体渲染公式的基本性质，不满足说明不透明度或透射率的公式不对"

    # 完全不透明的一段，和非常长的最后一段
    w = mod.volume_weights(torch.tensor([0.0, 1e6, 3.0]), torch.ones(3))
    if not close(w, torch.tensor([0.0, 1.0, 0.0]), 1e-6):
        return False, f"第二段完全不透明时权重应为 [0, 1, 0]，不能出现 nan；得到 {w.tolist()}"
    w = mod.volume_weights(torch.tensor([LN2, LN2, 1.0]), torch.tensor([1.0, 1.0, 1e10]))
    if not close(w, torch.tensor([0.5, 0.25, 0.25]), 1e-6):
        return False, (f"最后一段的长度取 1e10 时，期望权重 [0.5, 0.25, 0.25]（最后一段把剩下的光全部挡住），得到 {fmt(w)}。"
                       "如果最后一个数明显偏大：到达某一段之前的透射率只能用它前面各段来算，不要用“包含自己的累计再减去自己”之类的办法，大数相减会把前面的小数抵消掉")
    w = mod.volume_weights(torch.tensor([LN2, 0.0]), torch.tensor([1.0, 1e10]))
    if not close(w, torch.tensor([0.5, 0.0]), 1e-6):
        return False, f"最后一段很长但密度为 0 时，它的权重应为 0；得到 {fmt(w)}"

    # 梯度：权重之和对 sigma_i 的导数是 deltas_i x 剩下的透射率
    sg = torch.tensor([0.7, 0.0, 1.3], dtype=F64, requires_grad=True)
    dd = torch.tensor([0.5, 2.0, 0.25], dtype=F64)
    w = mod.volume_weights(sg, dd)
    if not w.requires_grad:
        return False, "返回值应对 sigma 可导（不要 detach，也不要经过 NumPy）"
    w.sum().backward()
    want_g = dd * math.exp(-(0.7 * 0.5 + 1.3 * 0.25))
    if sg.grad is None or not close(sg.grad, want_g, 1e-8):
        return False, f"权重之和对 sigma 的梯度应为 deltas 乘以剩下的透射率，期望 {fmt(want_g)}，得到 {None if sg.grad is None else fmt(sg.grad)}"
    sg = torch.tensor([0.0, 200.0, 1.0], requires_grad=True)
    (mod.volume_weights(sg, torch.ones(3)) * torch.tensor([1.0, 2.0, 3.0])).sum().backward()
    if sg.grad is None or not bool(torch.isfinite(sg.grad).all()):
        return False, f"有一段完全不透明时梯度里出现了 nan 或 inf：{None if sg.grad is None else sg.grad.tolist()}。检查有没有对 0 取对数，或者用 0 做除数"

    if not raises_value_error(mod.volume_weights, torch.tensor([1.0, -0.1]), torch.ones(2)) or not raises_value_error(mod.volume_weights, torch.ones(2), torch.tensor([1.0, -1.0])):
        return False, "sigma 或 deltas 里有负数时应抛出 ValueError"
    if not raises_value_error(mod.volume_weights, torch.ones(2, 3), torch.ones(4)) or not raises_value_error(mod.volume_weights, torch.ones(3), torch.ones(2, 3)):
        return False, "deltas 不能广播成 sigma 的形状时应抛出 ValueError（不是 PyTorch 自己的 RuntimeError，也不能悄悄返回别的形状）"
    if not raises_value_error(mod.volume_weights, torch.tensor(1.0), torch.tensor(1.0)):
        return False, "sigma 是 0 维张量时应抛出 ValueError"
    return True, "小例子、批、两个性质、不透明段和超长末段、梯度和异常都正确"


# ---------------------------------------------------------------- 沿光线合成
def check_composite_rays(mod):
    w = torch.tensor([0.5, 0.25, 0.0], dtype=F64)
    c = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=F64)
    t = torch.tensor([1.0, 2.0, 4.0], dtype=F64)
    keep = (w.clone(), c.clone(), t.clone())
    out = mod.composite_rays(w, c, t)
    if not (isinstance(out, tuple) and len(out) == 3 and all(isinstance(v, torch.Tensor) for v in out)):
        return False, "应返回 (rgb, depth, acc) 三个张量"
    rgb, depth, acc = out
    if not torch.equal(w, keep[0]) or not torch.equal(c, keep[1]) or not torch.equal(t, keep[2]):
        return False, "不要修改传入的张量"
    if rgb.shape != (3,) or depth.shape != () or acc.shape != ():
        return False, f"一条光线时 rgb 应为 (3,)，depth 和 acc 应为 0 维；得到 {tuple(rgb.shape)}、{tuple(depth.shape)}、{tuple(acc.shape)}"
    if rgb.dtype != F64 or depth.dtype != F64 or acc.dtype != F64:
        return False, "输入是 float64 时三个返回值也应是 float64"
    if abs(float(acc) - 0.75) > 1e-9:
        return False, f"acc 是权重之和，期望 0.75，得到 {float(acc):.4f}"
    if not close(rgb, torch.tensor([0.5, 0.25, 0.0], dtype=F64), 1e-9):
        return False, f"background 为 None 时 rgb 就是颜色按权重求和，期望 [0.5, 0.25, 0]，得到 {fmt(rgb)}（不要除以 acc，也不要自己补背景）"
    if abs(float(depth) - 4.0 / 3.0) > 1e-9:
        return False, (f"权重 [0.5, 0.25, 0]、距离 [1, 2, 4] 时，期望深度是 (0.5 + 0.5) / 0.75 = 1.3333，得到 {float(depth):.4f}。"
                       "检查：是不是按权重求和之后除以了 acc；深度是加权平均，不是权重最大的那个采样点的距离")
    bg = torch.tensor([0.2, 0.4, 0.8], dtype=F64)
    rgb, _, _ = mod.composite_rays(w, c, t, background=bg)
    if not close(rgb, torch.tensor([0.55, 0.35, 0.2], dtype=F64), 1e-9):
        return False, f"背景 [0.2, 0.4, 0.8]、acc = 0.75 时，背景只贡献剩下的 0.25，期望 [0.55, 0.35, 0.2]，得到 {fmt(rgb)}"

    # 批：两条光线，第二条什么都没碰到
    w2 = torch.stack([w, torch.zeros(3, dtype=F64)])
    c2 = torch.stack([c, c.flip(0)])
    rgb, depth, acc = mod.composite_rays(w2, c2, t, background=bg)
    if rgb.shape != (2, 3) or depth.shape != (2,) or acc.shape != (2,):
        return False, "两条光线时 rgb 应为 (2, 3)，depth 和 acc 应为 (2,)"
    if not bool(torch.isfinite(depth).all()) or abs(float(depth[1])) > 1e-9:
        return False, f"权重全为 0 的光线，深度应为 0，不能是 nan；得到 {float(depth[1])}"
    if not close(rgb[1], bg, 1e-9) or abs(float(acc[1])) > 1e-12:
        return False, "权重全为 0 的光线，rgb 应等于背景，acc 应为 0"
    if not close(rgb[0], torch.tensor([0.55, 0.35, 0.2], dtype=F64), 1e-9) or abs(float(depth[0]) - 4.0 / 3.0) > 1e-9:
        return False, "t_vals 形状为 (N,) 时应被各条光线共用；第一条光线的结果不应受第二条影响"
    t2 = torch.tensor([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]], dtype=F64)
    w2b = torch.tensor([[0.5, 0.25, 0.0], [0.0, 0.2, 0.6]], dtype=F64)
    _, depth, _ = mod.composite_rays(w2b, c2, t2)
    if not close(depth, torch.tensor([4.0 / 3.0, (1.0 + 4.2) / 0.8], dtype=F64), 1e-9):
        return False, f"每条光线有自己的 t_vals 时，期望深度 [1.3333, 6.5]，得到 {fmt(depth)}"

    # 两个性质：所有采样点同一个距离、同一个颜色
    g = torch.Generator().manual_seed(1)
    wr = torch.rand(4, 5, 6, generator=g, dtype=F64) / 6
    cr = torch.rand(4, 5, 6, 3, generator=g, dtype=F64)
    tr = torch.full((6,), 2.5, dtype=F64)
    rgb, depth, acc = mod.composite_rays(wr, cr, tr, background=bg)
    if rgb.shape != (4, 5, 3) or depth.shape != (4, 5) or acc.shape != (4, 5):
        return False, f"weights 形状 (4, 5, 6) 时 rgb 应为 (4, 5, 3)，depth 和 acc 应为 (4, 5)；得到 {tuple(rgb.shape)}、{tuple(depth.shape)}、{tuple(acc.shape)}"
    for i, j in ((0, 0), (3, 4), (2, 1)):
        one_ray = sum(float(wr[i, j, k]) * cr[i, j, k] for k in range(6)) + (1 - float(wr[i, j].sum())) * bg
        if not close(rgb[i, j], one_ray, 1e-9):
            return False, f"weights 形状 (4, 5, 6) 时，第 ({i}, {j}) 条光线的颜色应只由它自己的 6 个采样点决定。检查求和是不是沿“采样点”那一维，批的维度有没有被打乱"
    if not close(depth, torch.full((4, 5), 2.5, dtype=F64), 1e-9):
        return False, "所有采样点的距离都是 2.5 时，不管权重是多少，期望深度都应是 2.5。检查有没有除以 acc"
    same = bg.expand(4, 5, 6, 3)
    rgb, _, _ = mod.composite_rays(wr, same, tr, background=bg)
    if not close(rgb, bg.expand(4, 5, 3), 1e-9):
        return False, "所有采样点的颜色和背景颜色相同时，合成结果应正好是这个颜色。检查背景乘的是不是 (1 - acc)"
    out32 = mod.composite_rays(wr.float(), cr.float(), tr.float(), background=bg.float())
    if any(v.dtype != torch.float32 for v in out32):
        return False, "输入是 float32 时三个返回值也应是 float32"

    wg = wr.clone().requires_grad_(True)
    cg = cr.clone().requires_grad_(True)
    rgb, depth, _ = mod.composite_rays(wg, cg, torch.arange(1.0, 7.0, dtype=F64), background=bg)
    if not rgb.requires_grad or not depth.requires_grad:
        return False, "rgb 和 depth 都应保留梯度（不要 detach，也不要经过 NumPy）"
    rgb.sum().backward()
    if cg.grad is None or not close(cg.grad, wr[..., None].expand_as(cr), 1e-9):
        return False, "rgb 求和后对 colors 的梯度应等于对应的权重"
    if wg.grad is None or float(wg.grad.abs().sum()) == 0.0:
        return False, "梯度应能传回 weights"

    if not raises_value_error(mod.composite_rays, w, c[:2], t) or not raises_value_error(mod.composite_rays, w, torch.ones(3, 4, dtype=F64), t):
        return False, "colors 的形状不是 weights 的形状再加一维 3 时应抛出 ValueError"
    if not raises_value_error(mod.composite_rays, w, c, t[:2]) or not raises_value_error(mod.composite_rays, w2, c2, torch.ones(3, 3, dtype=F64)):
        return False, "t_vals 的形状既不同于 weights 也不是 (N,) 时应抛出 ValueError"
    return True, "颜色、背景、期望深度、空光线、批的形状、梯度和异常都正确"


# ---------------------------------------------------------------- 位置编码
def check_encoding(mod):
    r = math.sqrt(0.5)
    x = torch.tensor([[0.5, -0.25]], dtype=F64)
    x0 = x.clone()
    out = mod.positional_encoding(x, 2)
    if not isinstance(out, torch.Tensor) or out.shape != (1, 10):
        return False, f"2 维坐标、2 个频率、保留原始坐标时输出应为 (1, 10)；得到 {tuple(out.shape) if isinstance(out, torch.Tensor) else type(out).__name__}"
    if not torch.equal(x, x0):
        return False, "不要修改传入的 x"
    if out.dtype != F64:
        return False, "输出的 dtype 应和 x 相同（x 是 float64 时输出也是 float64）"
    want = torch.tensor([[0.5, -0.25, 1.0, -r, 0.0, r, 0.0, -1.0, -1.0, 0.0]], dtype=F64)
    if not close(out, want, 1e-9):
        return False, (f"x = [0.5, -0.25]、L = 2 时期望 {fmt(want)}，得到 {fmt(out)}。"
                       "依次检查：最前面是不是原始坐标；每个频率是不是先一整块 sin 再一整块 cos（块内按坐标顺序）；频率里有没有 pi；第一个频率是不是 2 的 0 次方")
    out = mod.positional_encoding(x, 2, include_input=False)
    if out.shape != (1, 8) or not close(out, want[:, 2:], 1e-9):
        return False, "include_input=False 时应只去掉最前面的原始坐标，其余不变"
    out = mod.positional_encoding(torch.tensor([0.125], dtype=F64), 3, include_input=False)
    want3 = torch.tensor([math.sin(math.pi / 8), math.cos(math.pi / 8), r, r, 1.0, 0.0], dtype=F64)
    if out.shape != (6,) or not close(out, want3, 1e-9):
        return False, f"x = 0.125、L = 3 时第三个频率是 4 pi：sin(4 pi x) = 1，cos(4 pi x) = 0。期望 {fmt(want3)}，得到 {fmt(out)}。相邻两个频率应差 2 倍，不是每次加 1"
    out = mod.positional_encoding(torch.tensor([0.0625], dtype=F64), 5, include_input=False)
    if out.shape != (10,) or not close(out[6:], torch.tensor([1.0, 0.0, 0.0, -1.0], dtype=F64), 1e-9):
        return False, (f"x = 1/16、L = 5 时最后两个频率是 8 pi 和 16 pi：sin(8 pi x) = 1、cos(8 pi x) = 0、sin(16 pi x) = 0、cos(16 pi x) = -1；"
                       f"得到 {fmt(out[6:]) if out.shape == (10,) else tuple(out.shape)}。第 k 个频率是 2 的 k 次方乘 pi（1、2、4、8、16……），检查 k 较大时对不对")
    out = mod.positional_encoding(x, 0)
    if out.shape != (1, 2) or not torch.equal(out, x):
        return False, "num_freqs = 0 且保留原始坐标时，输出应等于 x"

    g = torch.Generator().manual_seed(2)
    xs = torch.rand(2, 3, 4, 3, generator=g, dtype=F64) * 4 - 2
    out = mod.positional_encoding(xs, 4)
    if out.shape != (2, 3, 4, 27):
        return False, f"3 维坐标、4 个频率时最后一维应为 3 x (2 x 4 + 1) = 27，前面的维度保持不变；得到 {tuple(out.shape)}"
    if mod.positional_encoding(xs, 10).shape != (2, 3, 4, 63):
        return False, "3 维坐标、10 个频率时最后一维应为 63"
    blocks = out[..., 3:].reshape(2, 3, 4, 4, 2, 3)
    if not close(blocks[..., 0, :] ** 2 + blocks[..., 1, :] ** 2, torch.ones(2, 3, 4, 4, 3, dtype=F64), 1e-9):
        return False, "同一个频率的 sin 块和紧跟着的 cos 块，对应位置的平方和应为 1。检查各块的排列顺序"
    shifted = mod.positional_encoding(xs + 2.0, 4)
    if not close(shifted[..., 3:], out[..., 3:], 1e-9):
        return False, "频率是 2 的 k 次方乘 pi，所以 x 加 2 之后所有的 sin、cos 都不变。结果变了，检查频率里有没有 pi"
    if not close(shifted[..., :3], xs + 2.0, 1e-12):
        return False, "保留的原始坐标应原样放在最前面"
    if mod.positional_encoding(xs.float(), 2).dtype != torch.float32:
        return False, "输出的 dtype 应和 x 相同（x 是 float32 时输出也是 float32）"

    xg = torch.tensor([0.3, -0.6], dtype=F64, requires_grad=True)
    out = mod.positional_encoding(xg, 1, include_input=False)
    if not out.requires_grad:
        return False, "输出应对 x 可导（不要经过 NumPy）"
    out.sum().backward()
    want_g = math.pi * (torch.cos(math.pi * xg.detach()) - torch.sin(math.pi * xg.detach()))
    if xg.grad is None or not close(xg.grad, want_g, 1e-9):
        return False, "L = 1 时输出之和对 x 的导数应为 pi x (cos(pi x) - sin(pi x))"

    if not raises_value_error(mod.positional_encoding, x, -1):
        return False, "num_freqs 小于 0 时应抛出 ValueError"
    if not raises_value_error(mod.positional_encoding, x, 0, include_input=False):
        return False, "num_freqs 为 0 同时 include_input 为 False 时应抛出 ValueError"
    return True, "小例子、块的顺序、频率、形状、周期性、梯度和异常都正确"


# ---------------------------------------------------------------- 高斯投影
def _pinhole(p, R, t, fx, fy, cx, cy):
    """第 7 章的针孔投影：世界点 (M, 3) 到像素 (M, 2)。"""
    c = p @ R.T + t
    return torch.stack([fx * c[:, 0] / c[:, 2] + cx, fy * c[:, 1] / c[:, 2] + cy], dim=1)


def _cov_by_differences(mean, cov, R, t, fx, fy, cx, cy, h=1e-5):
    """不用雅可比的公式，独立算一遍二维协方差：把协方差拆成三个方向（Cholesky 分解的三列），
    沿每个方向把中心前后各挪一小步、投影、做中心差分，三个差分向量的外积之和就是线性近似下的二维协方差。"""
    chol = torch.linalg.cholesky(cov)
    out = torch.zeros(2, 2, dtype=mean.dtype)
    for k in range(3):
        v = chol[:, k]
        d = (_pinhole((mean + h * v)[None], R, t, fx, fy, cx, cy) - _pinhole((mean - h * v)[None], R, t, fx, fy, cx, cy))[0] / (2 * h)
        out = out + torch.outer(d, d)
    return out


def _rotation(axis, deg):
    a = torch.tensor(axis, dtype=F64)
    a = a / a.norm()
    k = torch.tensor([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]], dtype=F64)
    th = math.radians(deg)
    return torch.eye(3, dtype=F64) + math.sin(th) * k + (1 - math.cos(th)) * (k @ k)


def check_project(mod):
    eye, zero = torch.eye(3, dtype=F64), torch.zeros(3, dtype=F64)
    fx, fy, cx, cy = 100.0, 50.0, 320.0, 240.0
    # 光轴上的一个高斯：只有 X、Y 方向的方差会出现在结果里
    means = torch.tensor([[0.0, 0.0, 4.0]], dtype=F64)
    covs = torch.diag(torch.tensor([0.04, 0.09, 0.25], dtype=F64))[None]
    keep = (means.clone(), covs.clone())
    out = mod.project_gaussians(means, covs, eye, zero, fx, fy, cx, cy)
    if not (isinstance(out, tuple) and len(out) == 3 and all(isinstance(v, torch.Tensor) for v in out)):
        return False, "应返回 (uv, cov2d, depth) 三个张量"
    uv, cov2d, depth = out
    if not torch.equal(means, keep[0]) or not torch.equal(covs, keep[1]):
        return False, "不要修改传入的张量"
    if uv.shape != (1, 2) or cov2d.shape != (1, 2, 2) or depth.shape != (1,):
        return False, f"1 个高斯时 uv 应为 (1, 2)，cov2d 应为 (1, 2, 2)，depth 应为 (1,)；得到 {tuple(uv.shape)}、{tuple(cov2d.shape)}、{tuple(depth.shape)}"
    if uv.dtype != F64 or cov2d.dtype != F64 or depth.dtype != F64:
        return False, "输入是 float64 时三个返回值也应是 float64"
    if not close(uv, torch.tensor([[320.0, 240.0]], dtype=F64), 1e-9) or abs(float(depth[0]) - 4.0) > 1e-9:
        return False, f"光轴上的点应投影到主点 (320, 240)，深度 4；得到 {fmt(uv)}、{fmt(depth)}"
    want = torch.tensor([[[25.0, 0.0], [0.0, 14.0625]]], dtype=F64)
    if not close(cov2d, want, 1e-8):
        return False, (f"光轴上深度 4 的高斯，X 方向方差 0.04、Y 方向 0.09，fx = 100、fy = 50：期望二维协方差对角线是 (100/4)^2 x 0.04 = 25 和 (50/4)^2 x 0.09 = 14.0625，"
                       f"得到 {cov2d[0].tolist()}。检查焦距有没有乘进去、fx 和 fy 有没有各用各的；这里不要给对角线加常数")

    # 偏离光轴：雅可比对 Z 的那一列起作用。X、Z 之间有相关时，这一列的符号也会影响结果
    means = torch.tensor([[2.0, -1.0, 4.0]], dtype=F64)
    covs = torch.tensor([[[0.04, 0.0, 0.03], [0.0, 0.04, 0.0], [0.03, 0.0, 0.04]]], dtype=F64)
    uv, cov2d, depth = mod.project_gaussians(means, covs, eye, zero, fx, fy, cx, cy)
    if not close(uv, torch.tensor([[370.0, 227.5]], dtype=F64), 1e-9):
        return False, f"相机坐标 (2, -1, 4) 应投影到 (100 x 2 / 4 + 320, 50 x (-1) / 4 + 240) = (370, 227.5)；得到 {fmt(uv)}"
    if abs(float(depth[0]) - 4.0) > 1e-9:
        return False, f"depth 是相机坐标的 Z（这里是 4），不是到相机中心的距离（{math.sqrt(21):.3f}）；得到 {float(depth[0]):.4f}"
    # 手算：像素坐标对 (X, Y, Z) 的偏导是 [[25, 0, -12.5], [0, 12.5, 3.125]]，两侧乘上协方差
    want = torch.tensor([[[12.5, 0.78125], [0.78125, 6.640625]]], dtype=F64)
    if not close(cov2d, want, 1e-8):
        return False, (f"偏离光轴的高斯（相机坐标 (2, -1, 4)，X 和 Z 之间有相关）：期望 {want[0].tolist()}，得到 {cov2d[0].tolist()}。"
                       "雅可比的每一行除了对 X 或 Y 的偏导，还有对 Z 的偏导（因为要除以 Z），检查这一列有没有、符号和 Z 的幂次对不对")

    # 相机转了 90 度：世界的 X 轴是相机的 Z 轴。协方差要跟着转
    R = torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=F64)
    t = torch.tensor([0.0, 0.0, 1.0], dtype=F64)
    means = torch.tensor([[3.0, 0.0, 0.0]], dtype=F64)
    covs = torch.diag(torch.tensor([0.25, 0.09, 0.04], dtype=F64))[None]
    out, err = call_valid(mod.project_gaussians, means, covs, R, t, fx, fy, cx, cy)
    if out is None:
        return False, f"P_c = R P_w + t：世界点 (3, 0, 0) 在这台相机里是 (0, 0, 4)，在相机前方，不应抛出 ValueError（{err[:40]}）。检查外参的用法，特别是 R 有没有用成转置"
    uv, cov2d, depth = out
    if not close(uv, torch.tensor([[320.0, 240.0]], dtype=F64), 1e-9) or abs(float(depth[0]) - 4.0) > 1e-9:
        return False, f"P_c = R P_w + t：世界点 (3, 0, 0) 在这台相机里是 (0, 0, 4)，应投影到主点、深度 4；得到 {fmt(uv)}、{fmt(depth)}。检查外参的用法"
    want = torch.tensor([[[25.0, 0.0], [0.0, 14.0625]]], dtype=F64)
    if not close(cov2d, want, 1e-8):
        return False, (f"相机绕 y 轴转了 90 度，世界的 Z 方向（方差 0.04）成了画面的水平方向：期望对角线 25 和 14.0625，得到 {cov2d[0].tolist()}。"
                       "协方差要先用 R 变到相机坐标系，再乘雅可比")

    # 一般情形：和“沿协方差的三个方向做中心差分”的结果比
    g = torch.Generator().manual_seed(3)
    R = _rotation([0.3, -0.8, 0.5], 37.0)
    t = torch.tensor([0.2, -0.1, 5.0], dtype=F64)
    means = torch.rand(6, 3, generator=g, dtype=F64) * 2 - 1
    a = torch.rand(6, 3, 3, generator=g, dtype=F64) - 0.5
    covs = 0.05 * (a @ a.transpose(1, 2)) + 0.01 * eye
    out, err = call_valid(mod.project_gaussians, means, covs, R, t, 80.0, 70.0, 31.5, 23.5)
    if out is None:
        return False, f"6 个高斯都在相机前方 4 到 6 处，不应抛出 ValueError（{err[:40]}）。检查外参的用法"
    uv, cov2d, depth = out
    if uv.shape != (6, 2) or cov2d.shape != (6, 2, 2) or depth.shape != (6,):
        return False, "6 个高斯时 uv 应为 (6, 2)，cov2d 应为 (6, 2, 2)，depth 应为 (6,)"
    if not close(uv, _pinhole(means, R, t, 80.0, 70.0, 31.5, 23.5), 1e-8):
        return False, "一般的外参下，uv 应等于高斯中心的针孔投影（P_c = R P_w + t，再除以 Z、乘焦距、加主点）"
    if not close(depth, (means @ R.T + t)[:, 2], 1e-9):
        return False, "depth 应是相机坐标的 Z"
    for i in range(6):
        ref = _cov_by_differences(means[i], covs[i], R, t, 80.0, 70.0, 31.5, 23.5)
        if not close(cov2d[i], ref, 1e-5 * float(ref.abs().max())):
            return False, (f"一般的旋转下第 {i} 个高斯的二维协方差不对：期望 {[fmt(r) for r in ref]}，得到 {[fmt(r) for r in cov2d[i]]}。"
                           "期望值是把协方差拆成三个方向、沿每个方向挪动中心再投影得到的。常见原因：协方差两侧的 R 和 R 的转置放反了，或者漏了旋转")
    if not close(cov2d, cov2d.transpose(1, 2), 1e-9):
        return False, "二维协方差应是对称矩阵"

    out = mod.project_gaussians(means.float(), covs.float(), R.float(), t.float(), 80.0, 70.0, 31.5, 23.5)
    if any(v.dtype != torch.float32 for v in out):
        return False, "输入是 float32 时三个返回值也应是 float32"
    mg, cg = means.clone().requires_grad_(True), covs.clone().requires_grad_(True)
    uv, cov2d, depth = mod.project_gaussians(mg, cg, R, t, 80.0, 70.0, 31.5, 23.5)
    if not (uv.requires_grad and cov2d.requires_grad and depth.requires_grad):
        return False, "三个返回值都应保留梯度（不要 detach，也不要经过 NumPy）"
    (uv.sum() + cov2d.sum() + depth.sum()).backward()
    if mg.grad is None or cg.grad is None or not bool(torch.isfinite(mg.grad).all()) or float(cg.grad.abs().sum()) == 0.0:
        return False, "梯度应能传回 means 和 covs"

    behind = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -6.0]], dtype=F64)
    two = torch.stack([eye * 0.01, eye * 0.01])
    if not raises_value_error(mod.project_gaussians, behind, two, eye, zero, fx, fy, cx, cy):
        return False, "有高斯的中心在相机后方（Z < 0）时应抛出 ValueError"
    if not raises_value_error(mod.project_gaussians, torch.tensor([[1.0, 0.0, 0.0]], dtype=F64), two[:1], eye, zero, fx, fy, cx, cy):
        return False, "有高斯的中心正好在 Z = 0 的平面上时也应抛出 ValueError（不能除以 0）"
    if not raises_value_error(mod.project_gaussians, behind[:1, :2], two[:1], eye, zero, fx, fy, cx, cy) or not raises_value_error(mod.project_gaussians, behind[:1], two, eye, zero, fx, fy, cx, cy):
        return False, "means 不是 (N, 3) 或者 covs 不是 (N, 3, 3) 时应抛出 ValueError"
    if not raises_value_error(mod.project_gaussians, behind[:1], two[:1], torch.eye(4, dtype=F64), zero, fx, fy, cx, cy) or not raises_value_error(mod.project_gaussians, behind[:1], two[:1], eye, torch.zeros(3, 1, dtype=F64), fx, fy, cx, cy):
        return False, "R 不是 (3, 3) 或者 t 不是 (3,) 时应抛出 ValueError"
    return True, "光轴上和偏离光轴的小例子、外参、一般旋转下的差分核对、梯度和异常都正确"


# ---------------------------------------------------------------- 从前到后的 alpha 合成
def check_composite_gaussians(mod):
    red, green, blue = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    white = torch.ones(3, dtype=F64)
    alphas = torch.tensor([[0.5, 0.8, 0.25]], dtype=F64)
    colors = torch.tensor([red, green, blue], dtype=F64)
    depths = torch.tensor([2.0, 3.0, 1.0], dtype=F64)
    keep = (alphas.clone(), colors.clone(), depths.clone())
    out = mod.composite_gaussians(alphas, colors, depths, white)
    if not (isinstance(out, tuple) and len(out) == 2 and all(isinstance(v, torch.Tensor) for v in out)):
        return False, "应返回 (rgb, acc) 两个张量"
    rgb, acc = out
    if not torch.equal(alphas, keep[0]) or not torch.equal(colors, keep[1]) or not torch.equal(depths, keep[2]):
        return False, "不要修改传入的张量（限制 alphas 的范围、排序都要得到新的张量）"
    if rgb.shape != (1, 3) or acc.shape != (1,):
        return False, f"1 个像素时 rgb 应为 (1, 3)，acc 应为 (1,)；得到 {tuple(rgb.shape)}、{tuple(acc.shape)}"
    if rgb.dtype != F64 or acc.dtype != F64:
        return False, "输入是 float64 时返回值也应是 float64"
    # 从近到远是蓝（深度 1，不透明度 0.25）、红（2，0.5）、绿（3，0.8）：权重 0.25、0.75 x 0.5 = 0.375、0.75 x 0.5 x 0.8 = 0.3
    want = torch.tensor([[0.45, 0.375, 0.325]], dtype=F64)
    if abs(float(acc[0]) - 0.925) > 1e-9 and close(rgb, want, 1e-9):
        return False, f"rgb 是对的，但 acc 应是权重之和 0.925（不是剩下的透射率），得到 {float(acc[0]):.4f}"
    if not close(rgb, want, 1e-9):
        return False, (f"传入的顺序是红（深度 2，不透明度 0.5）、绿（3，0.8）、蓝（1，0.25），背景白色。从近到远的权重应是蓝 0.25、红 0.375、绿 0.3，"
                       f"期望 {fmt(want)}，得到 {fmt(rgb)}。检查：有没有按深度从小到大排序（不是按不透明度）、颜色和不透明度有没有一起重排、"
                       "透射率是不是只累乘排在前面的高斯、背景是不是只占剩下的 1 - acc")
    if abs(float(acc[0]) - 0.925) > 1e-9:
        return False, f"acc 应是权重之和 0.925，得到 {float(acc[0]):.4f}"

    # 两个像素，其中一个什么都没盖到
    a2 = torch.tensor([[0.5, 0.8, 0.25], [0.0, 0.0, 0.0]], dtype=F64)
    bg = torch.tensor([0.2, 0.4, 0.6], dtype=F64)
    rgb, acc = mod.composite_gaussians(a2, colors, depths, bg)
    if rgb.shape != (2, 3) or acc.shape != (2,):
        return False, "2 个像素时 rgb 应为 (2, 3)，acc 应为 (2,)"
    if not close(rgb[1], bg, 1e-12) or abs(float(acc[1])) > 1e-12:
        return False, "没有被任何高斯盖到的像素应是背景颜色，acc 为 0"
    if not close(rgb[0], torch.tensor([0.375 + 0.075 * 0.2, 0.3 + 0.075 * 0.4, 0.25 + 0.075 * 0.6], dtype=F64), 1e-9):
        return False, "换一个背景颜色之后，背景仍然只占 1 - acc = 0.075"

    # 不透明度的上限
    one = torch.tensor([[1.0, 1.0]], dtype=F64)
    two_c = torch.tensor([red, green], dtype=F64)
    near_first = torch.tensor([1.0, 2.0], dtype=F64)
    black = torch.zeros(3, dtype=F64)
    rgb, acc = mod.composite_gaussians(one, two_c, near_first, black)
    if not torch.equal(one, torch.ones(1, 2, dtype=F64)):
        return False, "限制不透明度的范围时改动了传入的 alphas。要得到新的张量，不要原地修改"
    if not close(rgb, torch.tensor([[0.99, 0.0099, 0.0]], dtype=F64), 1e-9) or abs(float(acc[0]) - 0.9999) > 1e-9:
        return False, f"两个不透明度都是 1 的高斯，默认上限 0.99：前面的权重 0.99，后面的 0.01 x 0.99 = 0.0099；得到 rgb {fmt(rgb)}、acc {float(acc[0]):.5f}。检查 alphas 有没有先限制到 max_alpha"
    rgb, acc = mod.composite_gaussians(one, two_c, near_first, black, max_alpha=1.0)
    if not close(rgb, torch.tensor([[1.0, 0.0, 0.0]], dtype=F64), 1e-12):
        return False, "max_alpha=1 时，前面那个不透明度为 1 的高斯应把后面的完全挡住。检查 max_alpha 参数有没有用上（不要把 0.99 写死）"
    rgb, acc = mod.composite_gaussians(torch.tensor([[-0.3, 0.5]], dtype=F64), two_c, near_first, black)
    if not close(rgb, torch.tensor([[0.0, 0.5, 0.0]], dtype=F64), 1e-12):
        return False, "小于 0 的不透明度应被当作 0（限制到 [0, max_alpha]）"

    # 性质 1：不透明度只有 0 和 1、上限取 1 时，结果就是“每个像素取盖住它的高斯里最近的那个”（第 21 章的深度缓冲）
    g = torch.Generator().manual_seed(4)
    hard = (torch.rand(40, 7, generator=g) < 0.4).to(F64)
    cols = torch.rand(7, 3, generator=g, dtype=F64)
    deps = torch.rand(7, generator=g, dtype=F64)
    rgb, acc = mod.composite_gaussians(hard, cols, deps, bg, max_alpha=1.0)
    nearest = torch.where(hard > 0, deps.expand(40, 7), torch.full((40, 7), float("inf"), dtype=F64)).argmin(dim=1)
    covered = hard.sum(dim=1) > 0
    zbuf = torch.where(covered[:, None], cols[nearest], bg.expand(40, 3))
    if not close(rgb, zbuf, 1e-12) or not close(acc, covered.to(F64), 1e-12):
        return False, "不透明度只取 0 和 1、max_alpha=1 时，每个像素应正好是盖住它的高斯里深度最小的那个的颜色（没有被盖住的是背景）。检查排序方向和透射率"
    # 性质 2：把高斯的存放顺序打乱，结果不变
    soft = torch.rand(40, 7, generator=g, dtype=F64)
    perm = torch.tensor([3, 6, 0, 5, 1, 4, 2])
    r1, a1 = mod.composite_gaussians(soft, cols, deps, bg)
    r2, a2 = mod.composite_gaussians(soft[:, perm], cols[perm], deps[perm], bg)
    if not close(a1, 1 - torch.prod(1 - soft.clamp(max=0.99), dim=1), 1e-12):
        return False, "7 个高斯时，acc 应等于 1 减去所有 (1 - 不透明度) 的乘积：没有被任何一个高斯挡住的光才到得了背景。检查透射率是不是把前面所有的高斯都乘进去了"
    if not close(r1, r2, 1e-12) or not close(a1, a2, 1e-12):
        return False, "把高斯的存放顺序打乱（alphas 的列、colors、depths 一起换），结果应完全相同：合成的顺序只由深度决定"
    r32, a32 = mod.composite_gaussians(soft.float(), cols.float(), deps.float(), bg.float())
    if r32.dtype != torch.float32 or a32.dtype != torch.float32:
        return False, "输入是 float32 时返回值也应是 float32"
    if not bool(((a1 >= 0) & (a1 <= 1)).all()):
        return False, "acc 应在 0 到 1 之间"

    ag, cg = alphas.clone().requires_grad_(True), colors.clone().requires_grad_(True)
    rgb, _ = mod.composite_gaussians(ag, cg, depths, white)
    if not rgb.requires_grad:
        return False, "rgb 应保留梯度"
    rgb.sum().backward()
    want_g = torch.tensor([0.375, 0.3, 0.25], dtype=F64)[:, None].expand(3, 3)
    if cg.grad is None or not close(cg.grad, want_g, 1e-9):
        return False, f"rgb 求和后对每个高斯颜色的梯度应等于它的权重（红 0.375、绿 0.3、蓝 0.25）；得到 {None if cg.grad is None else cg.grad.tolist()}"
    if ag.grad is None or not bool(torch.isfinite(ag.grad).all()) or float(ag.grad.abs().sum()) == 0.0:
        return False, "梯度应能传回 alphas"

    if not raises_value_error(mod.composite_gaussians, alphas[0], colors, depths, white):
        return False, "alphas 不是二维时应抛出 ValueError"
    if not raises_value_error(mod.composite_gaussians, alphas, colors[:2], depths, white) or not raises_value_error(mod.composite_gaussians, alphas, colors, depths[:2], white):
        return False, "colors 的形状不是 (N, 3) 或者 depths 的形状不是 (N,) 时应抛出 ValueError"
    return True, "排序、权重、背景、不透明度上限、退化成深度缓冲、顺序无关、梯度和异常都正确"


def main() -> int:
    print("第 30 章进度检查\n" + "=" * 52)
    try:
        mod = load_assignment()
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "加载作业", f"{type(exc).__name__}: {exc}")
        return 1
    results = [
        run("体渲染权重", lambda: check_weights(mod)),
        run("沿光线合成颜色和深度", lambda: check_composite_rays(mod)),
        run("位置编码", lambda: check_encoding(mod)),
        run("三维高斯投影到二维", lambda: check_project(mod)),
        run("从前到后的 alpha 合成", lambda: check_composite_gaussians(mod)),
    ]
    print("=" * 52)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("零件通过；下一步在 Notebook 里把它们接到迷你 NeRF 和一组三维高斯上，拟合同一个小场景。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
