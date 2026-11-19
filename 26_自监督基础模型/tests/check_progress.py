"""第 26 章自动验收：TODO / FAIL / PASS。"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "26-1_掩码建模与自蒸馏.py"


def load_assignment():
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    spec = importlib.util.spec_from_file_location("chapter26_assignment", base / ASSIGNMENT)
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


# ---------------------------------------------------------------- 随机遮挡
def check_masking(mod):
    x = torch.tensor([[[10.0], [11.0], [12.0], [13.0]], [[20.0], [21.0], [22.0], [23.0]]])
    noise = torch.tensor([[0.9, 0.1, 0.5, 0.3], [0.2, 0.8, 0.4, 0.6]])
    x0, n0 = x.clone(), noise.clone()
    out = mod.random_masking(x, 0.5, noise)
    if not (isinstance(out, tuple) and len(out) == 3 and all(isinstance(v, torch.Tensor) for v in out)):
        return False, "应返回 (x_keep, mask, ids_restore) 三个张量"
    keep, mask, restore = out
    if not torch.equal(x, x0) or not torch.equal(noise, n0):
        return False, "不要修改传入的 x 和 noise"
    if keep.shape != (2, 2, 1) or mask.shape != (2, 4) or restore.shape != (2, 4):
        return False, f"4 个块遮一半：x_keep 应为 (2, 2, 1)，mask 和 ids_restore 应为 (2, 4)；得到 {tuple(keep.shape)}、{tuple(mask.shape)}、{tuple(restore.shape)}"
    if keep.dtype != x.dtype or mask.dtype != x.dtype or restore.dtype != torch.int64:
        return False, "x_keep 和 mask 的 dtype 应与 x 相同，ids_restore 应为 int64"
    if not torch.equal(mask, torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])):
        return False, f"小例子里每行应遮住 noise 较大的两个块，mask 里 1 表示被遮住、按原来的块顺序排列；得到 {mask.tolist()}"
    if not torch.equal(keep[..., 0], torch.tensor([[11.0, 13.0], [20.0, 22.0]])):
        return False, f"x_keep 应是 noise 最小的那些块，并按 noise 从小到大排列；得到 {keep[..., 0].tolist()}"
    if not torch.equal(restore, torch.tensor([[3, 0, 2, 1], [0, 3, 1, 2]])):
        return False, f"ids_restore[b, j] 应是第 j 个块的 noise 在这一行里的名次；得到 {restore.tolist()}"

    g = torch.Generator().manual_seed(0)
    x = torch.randn(5, 16, 3, generator=g, dtype=torch.float64)
    noise = torch.rand(5, 16, generator=g)
    keep, mask, restore = mod.random_masking(x, 0.75, noise)
    if keep.shape != (5, 4, 3):
        return False, f"16 个块、遮挡比例 0.75 时应保留 4 个块，得到 x_keep 的形状 {tuple(keep.shape)}（mask_ratio 是被遮住的比例）"
    if keep.dtype != torch.float64 or mask.dtype != torch.float64:
        return False, "x 是 float64 时，x_keep 和 mask 也应是 float64"
    if not torch.equal(mask.sum(dim=1), torch.full((5,), 12.0, dtype=torch.float64)):
        return False, "每张图被遮住的块数都应是 12"
    if len({tuple(row.tolist()) for row in mask}) < 2:
        return False, "每张图应按自己那一行 noise 决定遮哪些块，现在各张图的掩码完全相同"
    for b in range(5):
        if noise[b][mask[b] < 0.5].max() >= noise[b][mask[b] > 0.5].min():
            return False, "保留的应是 noise 最小的块"
    filler = torch.full((5, 12, 3), -7.0, dtype=torch.float64)
    back = torch.gather(torch.cat([keep, filler], dim=1), 1, restore[..., None].expand(-1, -1, 3))
    want = torch.where(mask[..., None] > 0.5, torch.full_like(x, -7.0), x)
    if not torch.equal(back, want):
        return False, "按文档字符串的办法用 ids_restore 还原后，可见位置应是原来的块、被遮位置应是占位向量。检查 x_keep 的排列顺序和 ids_restore 是否配套"

    keep, mask, restore = mod.random_masking(x, 0.0, noise)
    if keep.shape != x.shape or float(mask.abs().sum()) != 0.0:
        return False, "mask_ratio=0 时应保留全部块，mask 全为 0"

    xg = x.clone().requires_grad_(True)
    keep, mask, _ = mod.random_masking(xg, 0.75, noise)
    if not keep.requires_grad:
        return False, "x_keep 应对 x 可导（不要 detach，也不要经过 NumPy）"
    keep.sum().backward()
    if not torch.equal(xg.grad, (1.0 - mask)[..., None].expand_as(x)):
        return False, "对 x_keep 求和再反传，梯度应只落在被保留的块上，每个元素为 1"

    if not raises_value_error(mod.random_masking, x, 1.0, noise) or not raises_value_error(mod.random_masking, x, -0.1, noise):
        return False, "mask_ratio 不在 [0, 1) 内时应抛出 ValueError"
    if not raises_value_error(mod.random_masking, x, 0.99, noise):
        return False, "遮挡比例高到一个块都留不下时应抛出 ValueError"
    if not raises_value_error(mod.random_masking, x, 0.5, noise[:, :8]) or not raises_value_error(mod.random_masking, x[0], 0.5, noise):
        return False, "x 不是三维、或者 noise 的形状不是 (B, L) 时应抛出 ValueError"
    return True, "小例子、还原性质、逐图独立、可导和异常都正确"


# ---------------------------------------------------------------- 重建损失
def check_loss(mod):
    pred = torch.tensor([[[1.0, 2.0], [0.0, 0.0], [3.0, 5.0]]])
    target = torch.tensor([[[1.0, 4.0], [5.0, 5.0], [1.0, 1.0]]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    t0 = target.clone()
    got = mod.masked_mse_loss(pred, target, mask)
    if not isinstance(got, torch.Tensor) or got.dim() != 0:
        return False, "应返回 0 维张量（标量）"
    if abs(float(got) - 6.0) > 1e-6:
        return False, (f"小例子：三个块的误差分别是 2、25、10，第 2 个块可见，期望 (2 + 10) / 2 = 6，得到 {float(got):.4f}。"
                       "检查：是否只统计了被遮住的块、除数是不是被遮住的块数、块内是不是对像素取平均")

    # 两张图被遮住的块数不同：4 个被遮块的误差是 1、1、2.5、5
    pred2 = torch.tensor([[[1.0, 1.0], [1.0, 3.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 2.0], [3.0, 1.0]]])
    mask2 = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    got = mod.masked_mse_loss(pred2, torch.zeros_like(pred2), mask2)
    if abs(float(got) - 2.375) > 1e-6:
        return False, f"两张图被遮住的块数不同（1 个和 3 个）时，应把全部 4 个被遮块的误差放在一起平均，期望 2.375，得到 {float(got):.4f}"

    g = torch.Generator().manual_seed(1)
    p = torch.randn(3, 8, 4, generator=g, dtype=torch.float64)
    t = torch.randn(3, 8, 4, generator=g, dtype=torch.float64)
    m = (torch.rand(3, 8, generator=g) < 0.5).to(torch.float64)
    base = float(mod.masked_mse_loss(p, t, m))
    p_changed = p + 100.0 * (1.0 - m)[..., None]
    if abs(float(mod.masked_mse_loss(p_changed, t, m)) - base) > 1e-9:
        return False, "只改可见块上的预测，损失不应该变"
    pg = p.clone().requires_grad_(True)
    mod.masked_mse_loss(pg, t, m).backward()
    if pg.grad is None or float(pg.grad[m < 0.5].abs().sum()) != 0.0 or float(pg.grad[m > 0.5].abs().sum()) == 0.0:
        return False, "损失应对 pred 可导，梯度只落在被遮住的块上"

    # 按块归一化：手算 [0, 2, 4, 6] 的均值 3、无偏方差 20/3
    tn = torch.tensor([[[0.0, 2.0, 4.0, 6.0], [1.0, 1.0, 1.0, 5.0]]], dtype=torch.float64)
    ones = torch.ones(1, 2, dtype=torch.float64)
    tn0 = tn.clone()
    s = math.sqrt(20.0 / 3.0)
    first = torch.tensor([-3.0 / s, -1.0 / s, 1.0 / s, 3.0 / s], dtype=torch.float64)
    second = torch.tensor([-0.5, -0.5, -0.5, 1.5], dtype=torch.float64)  # 均值 2，无偏方差 4
    exact = torch.stack([first, second])[None]
    got = float(mod.masked_mse_loss(exact, tn, ones, norm_pix=True))
    if not math.isfinite(got) or got > 1e-8:
        return False, (f"norm_pix=True：把预测设成“每个块各自减均值、除以标准差”之后的目标，损失应约为 0，得到 {got:.5f}。"
                       "检查：是不是逐块归一化（不是整张图）、方差是不是无偏估计")
    if not torch.equal(tn, tn0) or not torch.equal(target, t0):
        return False, "不要原地修改 target"
    got = float(mod.masked_mse_loss(tn, tn, ones, norm_pix=True))
    want = float(((tn - exact) ** 2).mean())
    if abs(got - want) > 1e-6:
        return False, f"norm_pix=True 时只归一化 target，pred 保持原样；用 pred = target 测试期望 {want:.4f}，得到 {got:.4f}"
    got = float(mod.masked_mse_loss(torch.zeros_like(tn), tn, ones, norm_pix=True))
    if abs(got - 0.75) > 1e-4:
        return False, f"norm_pix=True、预测全 0 时，每块 4 个像素的损失应为 (4 - 1) / 4 = 0.75（归一化后每块的平方和是 P - 1），得到 {got:.4f}"
    flat = torch.full((1, 2, 4), 0.3, dtype=torch.float64)
    got = float(mod.masked_mse_loss(torch.zeros_like(flat), flat, ones, norm_pix=True))
    if not math.isfinite(got) or abs(got) > 1e-6:
        return False, f"norm_pix=True 时，像素全相同的块归一化后应为 0（靠 eps 避免 0 除以 0），得到 {got}"

    if not raises_value_error(mod.masked_mse_loss, p, t, torch.zeros(3, 8, dtype=torch.float64)):
        return False, "mask 全为 0（没有被遮住的块）时应抛出 ValueError，而不是返回 nan"
    if not raises_value_error(mod.masked_mse_loss, p, t[:, :, :3], m) or not raises_value_error(mod.masked_mse_loss, p, t, m[:, :4]):
        return False, "pred 和 target 形状不同、或者 mask 的形状不是 (B, L) 时应抛出 ValueError"
    return True, "只统计被遮块、块数不等的批、按块归一化和异常都正确"


# ---------------------------------------------------------------- 教师的滑动平均
class _Tiny(torch.nn.Module):
    """一个裸参数加两层线性层，第 i 个参数张量填成 v + i，方便手算。"""

    def __init__(self, v):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(3))
        self.body = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1))
        with torch.no_grad():
            for i, p in enumerate(self.parameters()):
                p.fill_(v + i)


def _values(net):
    return [float(p.detach().reshape(-1)[0]) for p in net.parameters()]


def _uniform(net):
    return all(float((p.detach() - p.detach().reshape(-1)[0]).abs().max()) < 1e-6 for p in net.parameters())


def check_ema(mod):
    for teacher_grad in (False, True):
        teacher, student = _Tiny(1.0), _Tiny(3.0)
        for p in teacher.parameters():
            p.requires_grad_(teacher_grad)
        sum(p.sum() for p in student.parameters()).backward()  # 学生带着梯度，和训练时一样
        ids = [id(p) for p in teacher.parameters()]
        try:
            out = mod.ema_update(teacher, student, 0.9)
        except NotImplementedError:
            raise
        except RuntimeError as exc:
            return False, (f"教师参数的 requires_grad 为 {teacher_grad} 时更新报错了（{str(exc)[:60]}）。"
                           "原地修改一个需要梯度的参数必须在不记录梯度的状态下做，检查这一步有没有放进 torch.no_grad()")
        if out is not None:
            return False, "ema_update 没有返回值，直接改教师的参数"
        if [id(p) for p in teacher.parameters()] != ids:
            return False, "要原地修改教师的参数，不要换成新的 Parameter"
        if any(p.requires_grad != teacher_grad or p.grad_fn is not None or not p.is_leaf for p in teacher.parameters()):
            return False, "更新不应进计算图：教师参数的 requires_grad 不能变，grad_fn 应为 None（放进 torch.no_grad() 里做）"
        got, want = _values(teacher), [1.2, 2.2, 3.2, 4.2, 5.2]
        if not _uniform(teacher) or any(abs(a - b) > 1e-5 for a, b in zip(got, want)):
            return False, (f"教师 5 组参数原来是 1、2、3、4、5，学生是 3、4、5、6、7，momentum=0.9 时应变成 {want}，得到 {[round(v, 3) for v in got]}。"
                           "检查 momentum 乘在谁身上，以及是不是每一组参数（包括偏置和裸参数）都更新了")
        if _values(student) != [3.0, 4.0, 5.0, 6.0, 7.0] or any(p.grad is None or not p.requires_grad for p in student.parameters()):
            return False, "学生的参数和梯度都不应该被改动"

    teacher, student = _Tiny(1.0), _Tiny(3.0)
    mod.ema_update(teacher, student, 0.5)
    mod.ema_update(teacher, student, 0.5)
    if any(abs(a - b) > 1e-5 for a, b in zip(_values(teacher), [2.5, 3.5, 4.5, 5.5, 6.5])):
        return False, "连续更新两次（momentum=0.5）后，教师第一组参数应从 1 变成 2 再变成 2.5"
    teacher = _Tiny(1.0)
    mod.ema_update(teacher, student, 1.0)
    if _values(teacher) != [1.0, 2.0, 3.0, 4.0, 5.0]:
        return False, "momentum=1 时教师应保持不变"
    mod.ema_update(teacher, student, 0.0)
    if _values(teacher) != [3.0, 4.0, 5.0, 6.0, 7.0]:
        return False, "momentum=0 时教师应变成学生的拷贝"
    if not raises_value_error(mod.ema_update, teacher, student, 1.5) or not raises_value_error(mod.ema_update, teacher, student, -0.1):
        return False, "momentum 不在 [0, 1] 内时应抛出 ValueError"
    if not raises_value_error(mod.ema_update, teacher, torch.nn.Linear(2, 3), 0.9):
        return False, "两个网络的参数个数不同时应抛出 ValueError"
    return True, "数值、原地更新、不进计算图和异常都正确"


# ---------------------------------------------------------------- 自蒸馏损失
def check_dino(mod):
    ln3 = math.log(3.0)
    center = torch.tensor([0.25, 0.0], dtype=torch.float64)
    teacher = torch.tensor([[0.25 + 0.25 * ln3, 0.0], [0.25 + 0.25 * ln3, 0.0]], dtype=torch.float64)  # 两行的目标分布都是 [0.75, 0.25]
    student = torch.tensor([[0.3, 0.3], [0.0, 0.5 * ln3]], dtype=torch.float64)  # 两行的学生分布是 [0.5, 0.5] 和 [0.25, 0.75]
    want = 0.5 * (math.log(2.0) + 0.75 * math.log(4.0) + 0.25 * math.log(4.0 / 3.0))
    got = mod.dino_loss(student, teacher, center, student_temp=0.5, teacher_temp=0.25)
    if not isinstance(got, torch.Tensor) or got.dim() != 0:
        return False, "应返回 0 维张量（标量）"
    if abs(float(got) - want) > 1e-6:
        return False, (f"小例子期望 {want:.6f}，得到 {float(got):.6f}。检查：center 是不是在除以温度之前从教师输出里减掉的、"
                       "两个温度有没有用反、是不是交叉熵（不是 KL，也没有把教师和学生的位置对调）、是不是对样本取平均、softmax 是不是在最后一维上")
    got2 = mod.dino_loss(student, teacher, center[None], student_temp=0.5, teacher_temp=0.25)
    if abs(float(got2) - want) > 1e-6:
        return False, "center 的形状是 (1, K) 时结果应和 (K,) 相同"

    s = student.clone().requires_grad_(True)
    t = teacher.clone().requires_grad_(True)
    c = center.clone().requires_grad_(True)
    mod.dino_loss(s, t, c, student_temp=0.5, teacher_temp=0.25).backward()
    if (t.grad is not None and float(t.grad.abs().sum()) != 0.0) or (c.grad is not None and float(c.grad.abs().sum()) != 0.0):
        return False, "教师输出和 center 不应收到梯度（目标分布不参与求导）"
    want_grad = torch.tensor([[-0.25, 0.25], [-0.5, 0.5]], dtype=torch.float64)
    if s.grad is None or not torch.allclose(s.grad, want_grad, atol=1e-6):
        return False, f"对学生输出的梯度应为 (学生分布 - 目标分布) / (学生温度 x 批大小)，期望 {want_grad.tolist()}，得到 {None if s.grad is None else s.grad.tolist()}"

    rng = np.random.default_rng(4)
    t_np, c_np = rng.normal(size=(6, 5)), 0.3 * rng.normal(size=5)
    z = (t_np - c_np) / 0.04
    q = np.exp(z - z.max(axis=1, keepdims=True))
    q /= q.sum(axis=1, keepdims=True)
    ent = float(-(q * np.log(np.maximum(q, 1e-300))).sum(axis=1).mean())
    same = torch.tensor((t_np - c_np) * (0.1 / 0.04))
    got = float(mod.dino_loss(same, torch.tensor(t_np), torch.tensor(c_np)))
    if abs(got - ent) > 1e-6:
        return False, (f"用默认温度，让学生的分布和教师的目标分布完全相同，损失应等于这个分布的熵 {ent:.5f}，得到 {got:.5f}。"
                       "检查默认温度有没有用上，以及损失是不是交叉熵")
    got = float(mod.dino_loss(torch.full((6, 5), 1.7, dtype=torch.float64), torch.tensor(t_np), torch.tensor(c_np)))
    if abs(got - math.log(5.0)) > 1e-6:
        return False, f"学生输出各维相同（均匀分布）时，不管教师输出什么，损失都应等于 ln K = {math.log(5.0):.5f}，得到 {got:.5f}。检查教师那边是否做了 softmax"

    big = mod.dino_loss(torch.tensor([[0.0, 200.0]]), torch.tensor([[1.0, 1.0]]), torch.zeros(2))
    if not math.isfinite(float(big)) or abs(float(big) - 1000.0) > 0.5:
        return False, f"学生输出相差很大时（除以温度后差 2000）损失应为 1000，得到 {float(big)}。先 softmax 再取 log 会下溢，按文档字符串用 log_softmax"
    if not raises_value_error(mod.dino_loss, student, teacher, center, 0.0, 0.04) or not raises_value_error(mod.dino_loss, student, teacher, center, 0.1, -1.0):
        return False, "温度小于等于 0 时应抛出 ValueError"
    return True, "数值、梯度只传给学生、熵和 ln K 两个性质、数值稳定都正确"


# ---------------------------------------------------------------- k 近邻分类
def check_knn(mod):
    def call(tr, y, te, k, c, **kw):
        tr, y, te = np.array(tr, dtype=np.float64), np.array(y), np.array(te, dtype=np.float64)
        keep = (tr.copy(), y.copy(), te.copy())
        out = mod.knn_classify(tr, y, te, k, c, **kw)
        clean = np.array_equal(tr, keep[0]) and np.array_equal(y, keep[1]) and np.array_equal(te, keep[2])
        return out, clean

    out, clean = call([[0.2, 0.0], [5.0, 5.0]], [0, 1], [[0.1, 0.1]], 1, 2)
    if not isinstance(out, np.ndarray) or out.shape != (1,) or out.dtype != np.int64:
        return False, "应返回 (Q,) 的 int64 NumPy 数组"
    if not clean:
        return False, "不要修改传入的数组（归一化时不要原地除）"
    if out[0] != 1:
        return False, "近邻按余弦相似度找：[0.1, 0.1] 和 [5, 5] 方向相同，应判成它的类别，而不是欧氏距离更近的 [0.2, 0]"
    out, _ = call([[10.0, 0.0], [0.6, 0.8]], [0, 1], [[0.6, 0.8]], 1, 2)
    if out[0] != 1:
        return False, "特征要先做 L2 归一化：[0.6, 0.8] 和自己最像，不应被长度为 10 的向量抢走"

    r3 = math.sqrt(3.0) / 2.0
    train = [[1.0, 0.0], [0.5, r3], [0.5, -r3], [-1.0, 0.0]]
    labels = [1, 0, 0, 2]
    out, _ = call(train, labels, [[0.01, 0.0]], 3, 3)
    if out[0] != 1:
        return False, ("k=3 时最近的 1 个邻居（余弦相似度 1）属于类别 1，另外 2 个（相似度 0.5）属于类别 0；按 exp(相似度 / 0.07) 加权，类别 1 的票数远大于类别 0。"
                       "检查是不是加权投票，以及待分类的特征有没有归一化")
    out, _ = call(train, labels, [[2.0, 0.0]], 3, 3, temperature=100.0)
    if out[0] != 0:
        return False, "temperature=100 时三个邻居的权重几乎相等，应由 2 票的类别 0 胜出。检查 temperature 参数有没有用上、是不是相似度除以它"
    out, _ = call(train, labels, [[2.0, 0.0]], 1, 3, temperature=100.0)
    out2, _ = call(train, labels, [[2.0, 0.0]], 2, 3, temperature=100.0)
    if out[0] != 1 or out2[0] != 1:
        return False, "只有相似度最大的 k 个邻居投票，不多不少：k=1 和 k=2 时，类别 0 最多只有 1 个邻居入选，应判成相似度更高的类别 1"
    out, _ = call([[0.6, 0.8], [0.6, -0.8], [-1.0, 0.0]], [2, 1, 0], [[3.0, 0.0]], 2, 3)
    if out[0] != 1:
        return False, "两个类别得票完全相同时取编号最小的类别"
    out, _ = call([[0.0, 0.0], [1.0, 0.1], [-1.0, 0.0]], [2, 0, 1], [[1.0, 0.0], [-2.0, 0.1]], 3, 3)
    if out.tolist() != [0, 1]:
        return False, f"带标签的特征里有一行全 0 时不应影响其余样本（范数为 0 的行保持为 0，不能出现 nan）；期望 [0, 1]，得到 {out.tolist()}"

    rng = np.random.default_rng(3)
    dirs = np.eye(6)[:3]
    y_tr = np.repeat(np.arange(3), 8)
    y_te = rng.integers(0, 3, size=60)
    tr = (dirs[y_tr] + 0.1 * rng.normal(size=(24, 6))) * rng.uniform(0.05, 20.0, size=(24, 1))
    te = (dirs[y_te] + 0.1 * rng.normal(size=(60, 6))) * rng.uniform(0.05, 20.0, size=(60, 1))
    out = mod.knn_classify(tr.astype(np.float32), y_tr.astype(np.int32), te.astype(np.float32), 5, 5)
    if out.shape != (60,) or out.dtype != np.int64 or not np.array_equal(out, y_te):
        return False, "三个方向分得很开、只是向量长度相差几百倍的数据上，余弦 k 近邻应全部分对（输入是 float32 和 int32 也要能用）"

    if not raises_value_error(mod.knn_classify, tr, y_tr, te, 0, 3) or not raises_value_error(mod.knn_classify, tr, y_tr, te, 25, 3):
        return False, "k 小于 1 或大于带标签的样本数时应抛出 ValueError"
    return True, "余弦相似度、加权投票、温度、平局、零向量和异常都正确"


def main() -> int:
    print("第 26 章进度检查\n" + "=" * 52)
    try:
        mod = load_assignment()
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "加载作业", f"{type(exc).__name__}: {exc}")
        return 1
    results = [
        run("随机遮挡与还原下标", lambda: check_masking(mod)),
        run("只在被遮块上的重建损失", lambda: check_loss(mod)),
        run("教师的指数滑动平均", lambda: check_ema(mod)),
        run("带中心化和温度的自蒸馏损失", lambda: check_dino(mod)),
        run("带权 k 近邻分类", lambda: check_knn(mod)),
    ]
    print("=" * 52)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("零件通过；下一步在 Notebook 里把它们接到小 ViT 上，训练迷你 MAE 和迷你自蒸馏。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
