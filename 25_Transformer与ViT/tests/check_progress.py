"""第 25 章自动验收：TODO / FAIL / PASS。"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import torch

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "25-1_ViT核心零件.py"


def load_assignment():
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    spec = importlib.util.spec_from_file_location("chapter25_assignment", base / ASSIGNMENT)
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
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    except NotImplementedError:
        raise
    except Exception:  # noqa: BLE001
        return False
    return False


def gen(seed):
    return torch.Generator().manual_seed(seed)


def blocks_by_loop(x, p):
    """按文档字符串的定义逐块取子张量，得到 patchify 应有的结果。"""
    b, c, h, w = x.shape
    rows = []
    for n in range((h // p) * (w // p)):
        r, col = n // (w // p), n % (w // p)
        rows.append(x[:, :, r * p:(r + 1) * p, col * p:(col + 1) * p].reshape(b, -1))
    return torch.stack(rows, dim=1)


# ---------------------------------------------------------------- patchify
def check_patchify(mod):
    x = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 4, 6)
    keep = x.clone()
    t = mod.patchify(x, 2)
    if not isinstance(t, torch.Tensor) or tuple(t.shape) != (2, 6, 12):
        return False, f"输入 (2, 3, 4, 6)、P=2 时应返回 (2, 6, 12) 的张量，得到 {tuple(getattr(t, 'shape', ()))}"
    if not torch.equal(x, keep):
        return False, "输入图像被修改了"
    want = blocks_by_loop(x, 2)
    if not torch.equal(t, want):
        first = [float(v) for v in t[0, 0, :4]]
        if torch.equal(t.sort(dim=-1).values, want.sort(dim=-1).values):
            return False, f"每个块里的像素取对了，但块内展平的顺序不是 (C, P, P)。第 0 个标记的前 4 个数应为 0, 1, 6, 7，得到 {first}"
        if torch.equal(t.sort(dim=1).values.sort(dim=-1).values, want.sort(dim=1).values.sort(dim=-1).values):
            return False, "块的内容对了，但块的先后顺序不是行优先（注意这张图高 4、宽 6，一行有 3 个块）"
        return False, f"和逐块取子张量的结果不一致。第 0 个标记的前 4 个数应为 0, 1, 6, 7，得到 {first}。只 reshape 不调整维度顺序是不够的"
    y = torch.randn(3, 2, 9, 6, generator=gen(0), dtype=torch.float64)
    t = mod.patchify(y, 3)
    if t.dtype != torch.float64:
        return False, f"输出的 dtype 应和输入相同（float64），得到 {t.dtype}"
    if not torch.equal(t, blocks_by_loop(y, 3)):
        return False, "高和宽不相等（9x6，P=3）时结果不对，检查行数和列数有没有用反"
    unfolded = torch.nn.functional.unfold(y, kernel_size=3, stride=3).transpose(1, 2)
    if not torch.equal(t, unfolded):
        return False, "和 torch.nn.functional.unfold(核和步长都等于 P) 的结果不一致"
    hwc = torch.randn(2, 8, 4, 3, generator=gen(1))
    z = hwc.permute(0, 3, 1, 2)          # 从 HWC 转过来的图像在内存里不连续
    if not torch.equal(mod.patchify(z, 4), blocks_by_loop(z.contiguous(), 4)):
        return False, "对内存不连续的输入（permute 之后的张量）结果不对"
    conv = torch.nn.Conv2d(2, 5, 3, stride=3).double()
    with torch.no_grad():
        a = mod.patchify(y, 3) @ conv.weight.reshape(5, -1).T + conv.bias
        b = conv(y).flatten(2).transpose(1, 2)
    if not torch.allclose(a, b, atol=1e-10):
        return False, "切块后接线性层应该和步长等于核大小的卷积算出同样的结果，现在对不上"
    if not raises_value_error(mod.patchify, torch.zeros(1, 1, 6, 8), 4):
        return False, "高不能被 P 整除时应抛出 ValueError"
    if not raises_value_error(mod.patchify, torch.zeros(1, 1, 8, 6), 4):
        return False, "宽不能被 P 整除时应抛出 ValueError"
    return True, "块的顺序、块内顺序、非方图、dtype 和整除检查都正确"


# ---------------------------------------------------------------- unpatchify
def check_unpatchify(mod):
    x = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 4, 6)
    tokens = blocks_by_loop(x, 2)
    keep = tokens.clone()
    back = mod.unpatchify(tokens, 2, 3, 4, 6)
    if not isinstance(back, torch.Tensor) or tuple(back.shape) != (2, 3, 4, 6):
        return False, f"应返回 (2, 3, 4, 6) 的张量，得到 {tuple(getattr(back, 'shape', ()))}"
    if not torch.equal(tokens, keep):
        return False, "输入的标记被修改了"
    if not torch.equal(back, x):
        if torch.equal(back.flatten(1).sort(dim=1).values, x.flatten(1).sort(dim=1).values):
            return False, "像素一个不少，但位置放错了。检查 reshape 之后各个维度的含义，以及 permute 是不是 patchify 那一步的逆"
        return False, "没有还原出原图"
    y = torch.randn(3, 2, 9, 6, generator=gen(2), dtype=torch.float64)
    back = mod.unpatchify(blocks_by_loop(y, 3), 3, 2, 9, 6)
    if back.dtype != torch.float64 or not torch.equal(back, y):
        return False, "高和宽不相等（9x6，P=3）时没有还原出原图，或者 dtype 变了"
    z = torch.randn(1, 1, 4, 12, generator=gen(3))
    if not torch.equal(mod.unpatchify(blocks_by_loop(z, 4), 4, 1, 4, 12), z):
        return False, "只有一行块（4x12，P=4）时没有还原出原图"
    good = torch.zeros(1, 6, 12)
    for args, why in (((good, 2, 3, 4, 8), "标记个数和 (height/patch)*(width/patch) 对不上"),
                      ((good, 2, 2, 4, 6), "最后一维和 channels*patch*patch 对不上"),
                      ((torch.zeros(1, 6, 12), 2, 3, 5, 6), "height 不能被 patch 整除")):
        if not raises_value_error(mod.unpatchify, *args):
            return False, f"{why}时应抛出 ValueError"
    return True, "方图、非方图都能逐元素还原，形状不符时报错"


# ---------------------------------------------------------------- 缩放点积注意力
def weights_by_loop(q, k, mask):
    """逐个查询、只在允许的键上做 softmax，得到应有的权重。q: (Lq, d)，k: (Lk, d)，mask: (Lq, Lk) 或 None。"""
    lq, lk, d = q.shape[0], k.shape[0], q.shape[1]
    out = torch.zeros(lq, lk, dtype=q.dtype)
    for i in range(lq):
        allowed = [j for j in range(lk) if mask is None or bool(mask[i, j])]
        if not allowed:
            continue
        logits = [sum(float(q[i, t]) * float(k[j, t]) for t in range(d)) / math.sqrt(d) for j in allowed]
        top = max(logits)
        e = [math.exp(v - top) for v in logits]
        for j, v in zip(allowed, e):
            out[i, j] = v / sum(e)
    return out


def check_attention(mod):
    q = torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    k = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    v = torch.tensor([[1.0, 10.0], [3.0, 30.0]], dtype=torch.float64)
    res = mod.scaled_dot_product_attention(q, k, v)
    if not (isinstance(res, tuple) and len(res) == 2):
        return False, "应返回 (out, weights) 两个张量"
    out, w = res
    if tuple(w.shape) != (1, 2) or tuple(out.shape) != (1, 2):
        return False, f"Lq=1、Lk=2、dv=2 时 out 和 weights 都应是 (1, 2)，得到 {tuple(out.shape)} 和 {tuple(w.shape)}"
    w0 = math.exp(2.0) / (math.exp(2.0) + 1.0)
    if abs(float(w[0, 0]) - w0) > 1e-9:
        return False, (f"q=(2,0,0,0)、两个键是 (2,0,0,0) 和零向量时，第一个键的权重应为 {w0:.4f}，得到 {float(w[0, 0]):.4f}。"
                       "点积是 4，d 是 4，检查缩放用的是什么")
    if not torch.allclose(out, torch.tensor([[w0 * 1 + (1 - w0) * 3, w0 * 10 + (1 - w0) * 30]], dtype=torch.float64), atol=1e-9):
        return False, "权重对了，但输出不是 v 按权重的加权和"

    g = gen(4)
    q = torch.randn(2, 3, 5, 8, generator=g, dtype=torch.float64)
    k = torch.randn(2, 3, 7, 8, generator=g, dtype=torch.float64)
    v = torch.randn(2, 3, 7, 6, generator=g, dtype=torch.float64)
    keep = (q.clone(), k.clone(), v.clone())
    out, w = mod.scaled_dot_product_attention(q, k, v)
    if tuple(out.shape) != (2, 3, 5, 6) or tuple(w.shape) != (2, 3, 5, 7):
        return False, f"带两个批维度、Lq=5、Lk=7、dv=6 时形状应为 (2,3,5,6) 和 (2,3,5,7)，得到 {tuple(out.shape)} 和 {tuple(w.shape)}"
    if not all(torch.equal(a, b) for a, b in zip((q, k, v), keep)):
        return False, "输入被修改了"
    if (w < 0).any() or not torch.allclose(w.sum(-1), torch.ones(2, 3, 5, dtype=torch.float64), atol=1e-9):
        return False, "每个查询的权重应非负、在键这一维上和为 1。检查 softmax 作用在哪一维"
    want = weights_by_loop(q[1, 2], k[1, 2], None)
    if not torch.allclose(w[1, 2], want, atol=1e-9):
        return False, "随机输入上的权重和逐个查询手算的 softmax 不一致"
    if not torch.allclose(out[1, 2], want @ v[1, 2], atol=1e-9):
        return False, "随机输入上的输出不是 v 按权重的加权和"
    sq, sk = torch.randn(6, 6, generator=g, dtype=torch.float64), torch.randn(6, 6, generator=g, dtype=torch.float64)
    _, w = mod.scaled_dot_product_attention(sq, sk, sk)
    if not torch.allclose(w, weights_by_loop(sq, sk, None), atol=1e-9):
        return False, "Lq、Lk、d 恰好相等（都是 6）时权重不对。分数的第 i 行第 j 列应是第 i 个查询和第 j 个键的点积"

    mask = torch.rand(5, 7, generator=g) > 0.4
    mask[0] = True
    mask[3] = False                      # 这个查询的键全部被屏蔽
    mask[4] = False
    mask[4, 2] = True                    # 这个查询只剩一个键
    mk = mask.clone()
    out, w = mod.scaled_dot_product_attention(q, k, v, mask)
    if not torch.equal(mask, mk):
        return False, "掩码被修改了"
    if torch.isnan(w).any() or torch.isnan(out).any():
        return False, "某个查询的键全部被屏蔽时出现了 nan，这一行的权重和输出应规定为全 0"
    if (w[..., ~mask] != 0).any():
        return False, "被屏蔽位置的权重应严格等于 0。检查 True 和 False 的含义有没有用反，以及屏蔽是不是在 softmax 之前做的"
    live = mask.any(-1)
    if not torch.allclose(w[..., live, :].sum(-1), torch.ones(2, 3, int(live.sum()), dtype=torch.float64), atol=1e-9):
        return False, "屏蔽之后，剩下的权重应重新归一化到和为 1"
    want = weights_by_loop(q[0, 1], k[0, 1], mask)
    if not torch.allclose(w[0, 1], want, atol=1e-9):
        return False, "带掩码时的权重和“只在允许的键上做 softmax”的结果不一致"
    if not torch.allclose(out[0, 1], want @ v[0, 1], atol=1e-9):
        return False, "带掩码时的输出不对（键全部被屏蔽的那一行应输出全 0）"
    bmask = torch.rand(2, 1, 5, 7, generator=g) > 0.3
    bmask[..., 0] = True
    _, w = mod.scaled_dot_product_attention(q, k, v, bmask)
    if (w.masked_select(~bmask.expand(2, 3, 5, 7)) != 0).any():
        return False, "掩码需要广播到 (..., Lq, Lk)：形状 (2, 1, 5, 7) 的掩码应作用到第二个批维度的每一项"

    q32 = torch.randn(4, 3, 8, generator=g, requires_grad=True)
    k32 = torch.randn(4, 6, 8, generator=g, requires_grad=True)
    v32 = torch.randn(4, 6, 8, generator=g, requires_grad=True)
    out, w = mod.scaled_dot_product_attention(q32, k32, v32)
    if out.dtype != torch.float32 or w.dtype != torch.float32:
        return False, "float32 的输入应得到 float32 的输出"
    if not out.requires_grad:
        return False, "输出不能反向传播。不要转成 NumPy，不要 detach"
    (out ** 2).sum().backward()
    for name, t in (("q", q32), ("k", k32), ("v", v32)):
        if t.grad is None or not torch.isfinite(t.grad).all() or float(t.grad.abs().sum()) == 0.0:
            return False, f"梯度没有传回 {name}"
    return True, "手算小例子、批维度、掩码、全屏蔽的行和梯度都正确"


# ---------------------------------------------------------------- 多头注意力
def check_mha(mod):
    torch.manual_seed(5)
    d, h, b, n = 12, 3, 2, 5
    ref = torch.nn.MultiheadAttention(d, h, batch_first=True, bias=True).double().eval()
    with torch.no_grad():
        ref.in_proj_bias.normal_(std=0.5)
        ref.out_proj.bias.normal_(std=0.5)
    wq, bq = ref.in_proj_weight.detach().clone(), ref.in_proj_bias.detach().clone()
    wo, bo = ref.out_proj.weight.detach().clone(), ref.out_proj.bias.detach().clone()
    x = torch.randn(b, n, d, dtype=torch.float64)
    keep = x.clone()
    res = mod.multi_head_attention(x, wq, bq, wo, bo, h)
    if not (isinstance(res, tuple) and len(res) == 2):
        return False, "应返回 (out, attn) 两个张量"
    out, attn = res
    if tuple(out.shape) != (b, n, d):
        return False, f"out 应为 (B, L, D) = {(b, n, d)}，得到 {tuple(out.shape)}"
    if tuple(attn.shape) != (b, h, n, n):
        return False, f"attn 应为 (B, H, L, L) = {(b, h, n, n)}，不对头取平均，得到 {tuple(attn.shape)}"
    if not torch.equal(x, keep):
        return False, "输入 x 被修改了"
    with torch.no_grad():
        want_out, want_attn = ref(x, x, x, need_weights=True, average_attn_weights=False)
    if not torch.allclose(attn.sum(-1), torch.ones(b, h, n, dtype=torch.float64), atol=1e-9):
        return False, "每个头里每个标记的注意力权重应和为 1"
    if not torch.allclose(attn, want_attn, atol=1e-8):
        if torch.allclose(attn.mean(1), want_attn.mean(1), atol=1e-8):
            return False, "各个头的权重取平均后是对的，但头的先后顺序不对"
        if torch.allclose(attn.transpose(-1, -2), want_attn, atol=1e-8):
            return False, "attn[b, h, i, j] 应是第 i 个标记（查询）对第 j 个标记（键）的权重，现在行和列反了。检查 q 和 k 取的是不是 w_qkv 输出的前 D 维和中间 D 维"
        return False, ("注意力权重和 torch.nn.MultiheadAttention（同一组权重）不一致。"
                       "依次检查：q、k、v 各取 3D 维里的哪一段；最后一维怎样切成 H 个头；缩放用的是不是每个头的维数")
    if not torch.allclose(out, want_out, atol=1e-8):
        return False, "注意力权重对了，但输出和 torch.nn.MultiheadAttention 不一致。检查各个头拼回 (B, L, D) 的顺序，以及有没有过输出线性层"

    tri = torch.ones(n, n).tril().bool()                     # 第 i 个标记只能看第 0 到 i 个
    out, attn = mod.multi_head_attention(x, wq, bq, wo, bo, h, mask=tri)
    with torch.no_grad():
        want_out, want_attn = ref(x, x, x, attn_mask=~tri, need_weights=True, average_attn_weights=False)
    if (attn[..., ~tri] != 0).any():
        return False, "(L, L) 的掩码里为 False 的位置，每个头的权重都应严格为 0"
    if not torch.allclose(attn, want_attn, atol=1e-8) or not torch.allclose(out, want_out, atol=1e-8):
        return False, "带 (L, L) 掩码时和 torch.nn.MultiheadAttention 不一致"
    bm = torch.rand(b, n, n, generator=gen(6)) > 0.35
    bm[:, torch.arange(n), torch.arange(n)] = True
    out, attn = mod.multi_head_attention(x, wq, bq, wo, bo, h, mask=bm)
    with torch.no_grad():
        want_out, want_attn = ref(x, x, x, attn_mask=(~bm).repeat_interleave(h, dim=0), need_weights=True, average_attn_weights=False)
    if not torch.allclose(attn, want_attn, atol=1e-8) or not torch.allclose(out, want_out, atol=1e-8):
        return False, "带 (B, L, L) 掩码时不对：每个样本有自己的掩码，同一个样本的所有头共用它"

    # 手算的例子：q、k 的权重全为 0 时注意力是均匀的；v 和输出层取单位阵时，每个位置的输出都是全部标记的平均
    eye = torch.eye(d, dtype=torch.float64)
    w_id = torch.cat([torch.zeros(2 * d, d, dtype=torch.float64), eye])
    zero3, zero1 = torch.zeros(3 * d, dtype=torch.float64), torch.zeros(d, dtype=torch.float64)
    out, attn = mod.multi_head_attention(x, w_id, zero3, eye, zero1, 4)
    if not torch.allclose(attn, torch.full((b, 4, n, n), 1.0 / n, dtype=torch.float64), atol=1e-12):
        return False, "q、k 全为 0 时注意力应是均匀的 1/L"
    if not torch.allclose(out, x.mean(1, keepdim=True).expand(b, n, d), atol=1e-10):
        return False, "注意力均匀、v 和输出层是单位阵时，每个位置的输出应等于全部标记的平均（H=4）"
    if not raises_value_error(mod.multi_head_attention, x, wq, bq, wo, bo, 5):
        return False, "D 不能被 num_heads 整除时应抛出 ValueError"

    x32 = torch.randn(2, 4, 8, generator=gen(7))
    w32 = torch.randn(24, 8, generator=gen(8), requires_grad=True)
    out, attn = mod.multi_head_attention(x32, w32, torch.zeros(24), torch.eye(8), torch.zeros(8), 2)
    if out.dtype != torch.float32 or not out.requires_grad:
        return False, "float32 的输入应得到 float32 的输出，并且能对权重求梯度"
    out.sum().backward()
    if w32.grad is None or not torch.isfinite(w32.grad).all():
        return False, "梯度没有传回 w_qkv"
    return True, "和 torch.nn.MultiheadAttention 一致（无掩码、两种掩码），形状和整除检查正确"


# ---------------------------------------------------------------- 二维正余弦位置编码
def check_sincos(mod):
    pe = mod.sincos_2d_position_embedding(4, 2, 3)
    if not isinstance(pe, torch.Tensor) or tuple(pe.shape) != (6, 4):
        return False, f"dim=4、网格 2x3 时应返回 (6, 4) 的张量，得到 {tuple(getattr(pe, 'shape', ()))}"
    if pe.dtype != torch.float32:
        return False, f"应返回 float32，得到 {pe.dtype}"
    want = torch.tensor([[math.sin(r), math.cos(r), math.sin(c), math.cos(c)] for r in range(2) for c in range(3)])
    if not torch.allclose(pe, want, atol=1e-6):
        if torch.allclose(pe, want[:, [2, 3, 0, 1]], atol=1e-6):
            return False, "行和列反了：前一半应编码行号，后一半编码列号"
        swapped = torch.tensor([[math.sin(r), math.cos(r), math.sin(c), math.cos(c)] for c in range(3) for r in range(2)])
        if torch.allclose(pe, swapped, atol=1e-6):
            return False, "行的排列不是行优先：第 n 行应对应第 n // grid_w 行、第 n % grid_w 列的块"
        return False, f"dim=4 时位置 (行 r, 列 c) 的编码应为 [sin r, cos r, sin c, cos c]。第 5 行（r=1, c=2）得到 {[round(float(v), 4) for v in pe[5]]}"

    pe = mod.sincos_2d_position_embedding(8, 3, 5)
    r, c = 2, 3
    row = [math.sin(r), math.sin(r * 0.01), math.cos(r), math.cos(r * 0.01), math.sin(c), math.sin(c * 0.01), math.cos(c), math.cos(c * 0.01)]
    got = pe[r * 5 + c]
    if not torch.allclose(got, torch.tensor(row), atol=1e-6):
        inter = [row[i] for i in (0, 2, 1, 3, 4, 6, 5, 7)]
        if torch.allclose(got, torch.tensor(inter), atol=1e-6):
            return False, "sin 和 cos 交错排了。每一半里应先放全部 sin，再放全部 cos"
        if abs(float(got[0]) - row[0]) < 1e-6 and abs(float(got[1]) - row[1]) > 1e-6:
            return False, (f"dim=8 时每一段有 2 个频率，应为 1 和 1/100。位置 (2, 3) 的第 2 个数应为 sin(0.02)={row[1]:.5f}，得到 {float(got[1]):.5f}。"
                           "检查频率指数的分母")
        return False, f"dim=8、网格 3x5 时位置 (行 2, 列 3) 的编码不对，得到 {[round(float(v), 4) for v in got]}"
    full = torch.tensor([[f(p * w) for p in (rr, cc) for f in (math.sin, math.cos) for w in (1.0, 0.01)] for rr in range(3) for cc in range(5)])
    if not torch.allclose(pe, full, atol=1e-6):
        return False, "dim=8、网格 3x5 时有的位置不对（高和宽不相等，检查有没有把 grid_h 和 grid_w 用反）"

    pe = mod.sincos_2d_position_embedding(16, 4, 4, temperature=100.0)
    omegas = [100.0 ** (-t / 4) for t in range(4)]
    want_row = [f(p * w) for p in (3, 1) for f in (math.sin, math.cos) for w in omegas]
    if not torch.allclose(pe[3 * 4 + 1], torch.tensor(want_row), atol=1e-6):
        return False, "temperature=100、dim=16 时不对：四个频率应为 100 的 0、-1/4、-2/4、-3/4 次方"
    pe = mod.sincos_2d_position_embedding(32, 6, 7)
    if not torch.allclose((pe.double() ** 2).sum(1), torch.full((42,), 16.0, dtype=torch.float64), atol=1e-4):
        return False, "每个位置编码的平方和应等于 dim/2（每一对 sin、cos 的平方和是 1）"
    for args in ((6, 2, 2), (0, 2, 2), (8, 0, 3), (8, 3, 0)):
        if not raises_value_error(mod.sincos_2d_position_embedding, *args):
            return False, f"参数 (dim, grid_h, grid_w) = {args} 时应抛出 ValueError"
    return True, "排列顺序、频率、非方网格、dtype 和参数检查都正确"


def main() -> int:
    print("第 25 章进度检查\n" + "=" * 52)
    try:
        mod = load_assignment()
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "加载作业", f"{type(exc).__name__}: {exc}")
        return 1
    results = [
        run("切块 patchify", lambda: check_patchify(mod)),
        run("还原 unpatchify", lambda: check_unpatchify(mod)),
        run("缩放点积注意力", lambda: check_attention(mod)),
        run("多头注意力的前向", lambda: check_mha(mod)),
        run("二维正余弦位置编码", lambda: check_sincos(mod)),
    ]
    print("=" * 52)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("零件通过；下一步在 Notebook 里用它们拼一个小 ViT，做位置编码和注意力图的实验。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
