"""第 20 章自动验收：TODO / FAIL / PASS。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "20-1_缺陷合成与扩散前向.py"


def load_assignment():
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    spec = importlib.util.spec_from_file_location("chapter20_assignment", base / ASSIGNMENT)
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


def unchanged(before, after):
    """调用前后的数组逐元素相同、数据类型也相同。"""
    return before.dtype == after.dtype and np.array_equal(before, after)


def check_scratch(mod):
    m = mod.scratch_mask((20, 30), (10.0, 5.0), (10.0, 25.0), width=3.0, softness=1.0)
    if m.shape != (20, 30) or m.dtype != np.float64:
        return False, "输出必须是 (H, W) float64"
    if m.min() < 0 or m.max() > 1:
        return False, "掩码必须在 [0, 1]"
    if m[10, 15] != 1.0 or m[11, 15] != 1.0 or m[12, 15] != 0.0 or m[13, 15] != 0.0:
        return False, "宽 3、过渡带 1 的水平划痕：距离 0 和 1 处应为 1，距离 2 和 3 处应为 0"
    soft = mod.scratch_mask((20, 30), (10.0, 5.0), (10.0, 25.0), width=3.0, softness=2.0)
    if abs(soft[12, 15] - 0.25) > 1e-12 or abs(soft[11, 15] - 0.75) > 1e-12:
        return False, "过渡带 2 时距离 1、2 处应为 0.75、0.25（m = (w/2 + s/2 - d) / s）"
    if m[10, 28] != 0.0 or m[10, 2] != 0.0:
        return False, "线段端点之外应按到端点的距离衰减，而不是按无限长直线"
    dot = mod.scratch_mask((9, 9), (4.0, 4.0), (4.0, 4.0), width=2.0)
    if dot[4, 4] != 1.0 or dot[4, 7] != 0.0:
        return False, "p0 == p1 时应退化为圆点"
    vert = mod.scratch_mask((30, 20), (5.0, 10.0), (25.0, 10.0), width=3.0, softness=1.0)
    if vert.shape != (30, 20) or not np.allclose(vert, m.T):
        return False, "竖直的划痕应等于水平划痕的转置；检查 (y, x) 的顺序，以及求点到线段的距离时有没有用到斜率（竖直线没有斜率）"
    if not np.allclose(mod.scratch_mask((20, 30), (10.0, 25.0), (10.0, 5.0), width=3.0, softness=1.0), m):
        return False, "p0 和 p1 对调后掩码应不变"
    diag = mod.scratch_mask((15, 15), (2, 2), (12, 12), width=3.0, softness=1.0)
    want = 2.0 - np.sqrt(2.0)      # 像素 (6, 8) 到对角线的距离是 sqrt(2)
    if diag[7, 7] != 1.0 or abs(diag[6, 8] - want) > 1e-12 or abs(diag[8, 6] - want) > 1e-12 or diag[4, 10] != 0.0:
        return False, f"45 度的划痕（宽 3、过渡带 1）：线上应为 1，离线 sqrt(2) 处应为 {want:.4f}，离线 3 sqrt(2) 处应为 0；得到 {diag[7, 7]:.4f}、{diag[6, 8]:.4f}、{diag[4, 10]:.4f}"
    if abs(diag[1, 1] - want) > 1e-12 or diag[0, 0] != 0.0:
        return False, "45 度划痕的端点之外应按到端点的距离衰减"
    for bad in (0.0, -1.0):
        try:
            mod.scratch_mask((5, 5), (0, 0), (4, 4), 1.0, softness=bad)
        except ValueError:
            continue
        return False, f"softness={bad} 时应抛出 ValueError；softness 必须大于 0"
    return True, "距离、端点、方向、过渡带和退化情况都正确"


def check_blend(mod):
    bg = np.full((4, 4), 100.0)
    mask = np.zeros((4, 4))
    mask[1, 1], mask[2, 2] = 1.0, 0.25
    out = mod.blend(bg, 20.0, mask)
    if out.dtype != np.float64 or out[1, 1] != 20.0 or out[2, 2] != 80.0 or out[0, 0] != 100.0:
        return False, "标量前景混合结果不正确"
    fg = np.arange(16, dtype=np.float64).reshape(4, 4)
    if not np.allclose(mod.blend(bg, fg, np.ones((4, 4))), fg):
        return False, "mask 全 1 时应完全取前景"
    keep_bg, keep_fg, keep_mask = bg.copy(), fg.copy(), mask.copy()
    half = np.full((4, 4), 0.3)
    out = mod.blend(bg, fg, half)
    if not unchanged(keep_bg, bg) or not unchanged(keep_fg, fg) or not unchanged(keep_mask, mask):
        return False, "函数改动了传入的 background、foreground 或 mask；结果要写到新数组里"
    if out is bg or np.shares_memory(out, bg):
        return False, "返回值和传入的 background 是同一块内存；结果要写到新数组里"
    if not np.allclose(out, 70.0 + 0.3 * fg):
        return False, "数组前景、mask 全为 0.3 时，结果应为 0.7 乘背景加 0.3 乘前景；检查有没有取整"
    out = mod.blend(bg, 25.5, half)
    if abs(out[0, 0] - 77.65) > 1e-9:
        return False, f"背景 100、前景 25.5、mask 0.3 应得到 77.65，得到 {out[0, 0]!r}；结果不做取整"
    out = mod.blend(bg, 300.0, np.ones((4, 4)))
    low = mod.blend(bg, -50.0, np.ones((4, 4)))
    if out[0, 0] != 300.0 or low[0, 0] != -50.0:
        return False, f"前景 300 和 -50、mask 为 1 时应原样得到 300 和 -50，得到 {out[0, 0]!r} 和 {low[0, 0]!r}；结果不做裁剪，位深交给调用方"
    bg8 = np.array([[200, 201], [0, 255]], dtype=np.uint8)
    out8 = mod.blend(bg8, 50, np.full((2, 2), 0.5))
    if out8.dtype != np.float64 or not np.allclose(out8, [[125.0, 125.5], [25.0, 152.5]]):
        return False, "背景是 uint8 图像时也应返回 float64、不取整（背景 201、前景 50、mask 0.5 应得到 125.5）；检查有没有把结果转回输入的数据类型"
    if bg8.dtype != np.uint8 or bg8[0, 1] != 201:
        return False, "函数改动了传入的 uint8 背景"
    neg = np.zeros((4, 4))
    neg[0, 0] = -0.5
    for bad, what in ((mask + 2.0, "大于 1 "), (neg, "小于 0 ")):
        try:
            mod.blend(bg, 0.0, bad)
        except ValueError:
            continue
        return False, f"mask 里有{what}的值时应抛出 ValueError；mask 必须在 [0, 1]"
    return True, "标量和数组前景都正确，不取整不裁剪，非法掩码报错"


def check_ddpm(mod):
    b = mod.linear_beta_schedule(1000)
    if b.shape != (1000,) or abs(b[0] - 1e-4) > 1e-15 or abs(b[-1] - 0.02) > 1e-15:
        return False, "beta 调度首尾应为 1e-4 和 0.02"
    if not np.all(np.diff(b) > 0):
        return False, "线性调度应严格递增"
    b5 = mod.linear_beta_schedule(5, 0.1, 0.5)
    if b5.shape != (5,) or b5.dtype != np.float64 or not np.allclose(b5, [0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-12):
        return False, f"linear_beta_schedule(5, 0.1, 0.5) 应为 [0.1, 0.2, 0.3, 0.4, 0.5]，得到 {np.round(b5, 4).tolist()}；检查 beta_start、beta_end 两个参数有没有用上"
    keep_b = b.copy()
    ab = mod.alpha_bars(b)
    if not unchanged(keep_b, b):
        return False, "alpha_bars 改动了传入的 betas"
    small = mod.alpha_bars(np.array([0.5, 0.5, 0.2]))
    if small.shape != (3,) or small.dtype != np.float64 or not np.allclose(small, [0.5, 0.25, 0.2], atol=1e-12):
        return False, f"betas = [0.5, 0.5, 0.2] 时 alpha_bar 应为 [0.5, 0.25, 0.2]，得到 {np.round(small, 4).tolist()}；它是 (1 - beta) 的累计乘积"
    if abs(ab[0] - (1 - 1e-4)) > 1e-15 or not np.all(np.diff(ab) < 0) or ab[-1] > 1e-3:
        return False, "alpha_bar 应从 1-beta_0 单调下降到接近 0"
    if not np.all(ab > 0):
        return False, "alpha_bar 是若干个 0 到 1 之间的数的乘积，不应出现 0 或负数"
    rng = np.random.default_rng(0)
    x0 = rng.uniform(-1, 1, size=(64, 64))
    noise = rng.normal(size=(64, 64))
    keep_x0, keep_noise, keep_ab = x0.copy(), noise.copy(), ab.copy()
    xt = mod.q_sample(x0, 0, ab, noise)
    if not unchanged(keep_x0, x0) or not unchanged(keep_noise, noise) or not unchanged(keep_ab, ab):
        return False, "q_sample 改动了传入的 x0、noise 或 abar；结果要写到新数组里"
    if not np.allclose(xt, np.sqrt(ab[0]) * x0 + np.sqrt(1 - ab[0]) * noise):
        return False, "q_sample 公式不正确"
    toy = mod.q_sample(np.array([2.0, -1.0]), 1, np.array([0.81, 0.36, 0.04]), np.array([1.0, 0.5]))
    if not np.allclose(toy, [2.0, -0.2], atol=1e-12):
        return False, f"x0 = [2, -1]、abar[1] = 0.36、noise = [1, 0.5] 时 x_t 应为 [2.0, -0.2]，得到 {np.round(toy, 4).tolist()}；检查两个系数和下标 t"
    late = mod.q_sample(x0, 999, ab, noise)
    if not np.isfinite(late).all():
        return False, "最后一步的 x_t 里出现了 nan 或 inf；检查 alpha_bar 有没有变成负数"
    if not abs(np.corrcoef(late.ravel(), noise.ravel())[0, 1]) >= 0.99:
        return False, "最后一步的 x_t 应几乎等于纯噪声"
    for bad in (1000, -1, -1000):
        try:
            mod.q_sample(x0, bad, ab, noise)
        except IndexError:
            continue
        return False, f"abar 有 1000 步，t={bad} 时应抛出 IndexError；合法的 t 是 0 到 999，负数不能当成“从末尾数”"
    return True, "调度、累计乘积、加噪公式和越界检查都正确"


def check_reparam(mod):
    rng = np.random.default_rng(1)
    eps = rng.normal(size=200000)
    z = mod.reparameterize(np.full(200000, 2.0), np.full(200000, np.log(9.0)), eps)
    if abs(z.mean() - 2.0) > 0.03 or abs(z.std() - 3.0) > 0.03:
        return False, f"logvar=log(9) 时标准差应为 3，得到 {z.std():.3f}"
    if not np.allclose(mod.reparameterize(np.zeros(3), np.zeros(3), np.ones(3)), np.ones(3)):
        return False, "mu=0、logvar=0 时 z 应等于 eps"
    z2 = mod.reparameterize(np.array([1.0, -2.0]), np.log(np.array([4.0, 1.0])), np.array([0.5, -3.0]))
    if not np.allclose(z2, [2.0, -5.0], atol=1e-12):
        return False, f"mu = [1, -2]、方差 = [4, 1]、eps = [0.5, -3] 时 z 应为 [2, -5]，得到 {np.round(z2, 4).tolist()}；logvar 是方差的对数，不是标准差的对数"
    return True, "均值和方差符合 N(mu, exp(logvar))"


def main() -> int:
    print("第 20 章进度检查\n" + "=" * 52)
    try:
        mod = load_assignment()
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "加载作业", f"{type(exc).__name__}: {exc}")
        return 1
    results = [
        run("划痕软掩码", lambda: check_scratch(mod)),
        run("软掩码混合", lambda: check_blend(mod)),
        run("DDPM 前向过程", lambda: check_ddpm(mod)),
        run("VAE 重参数化", lambda: check_reparam(mod)),
    ]
    print("=" * 52)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("零件通过；下一步做“只用合成缺陷训练、在真实缺陷上测试”的对照实验。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
