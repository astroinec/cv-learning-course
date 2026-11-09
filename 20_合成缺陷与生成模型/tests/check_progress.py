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
    try:
        mod.scratch_mask((5, 5), (0, 0), (4, 4), 1.0, softness=0.0)
    except ValueError:
        return True, "距离、端点、过渡带和退化情况都正确"
    return False, "softness <= 0 时应抛出 ValueError"


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
    try:
        mod.blend(bg, 0.0, mask + 2.0)
    except ValueError:
        return True, "标量和数组前景都正确，非法掩码报错"
    return False, "mask 超出 [0, 1] 时应抛出 ValueError"


def check_ddpm(mod):
    b = mod.linear_beta_schedule(1000)
    if b.shape != (1000,) or abs(b[0] - 1e-4) > 1e-15 or abs(b[-1] - 0.02) > 1e-15:
        return False, "beta 调度首尾应为 1e-4 和 0.02"
    if not np.all(np.diff(b) > 0):
        return False, "线性调度应严格递增"
    ab = mod.alpha_bars(b)
    if abs(ab[0] - (1 - 1e-4)) > 1e-15 or not np.all(np.diff(ab) < 0) or ab[-1] > 1e-3:
        return False, "alpha_bar 应从 1-beta_0 单调下降到接近 0"
    rng = np.random.default_rng(0)
    x0 = rng.uniform(-1, 1, size=(64, 64))
    noise = rng.normal(size=(64, 64))
    xt = mod.q_sample(x0, 0, ab, noise)
    if not np.allclose(xt, np.sqrt(ab[0]) * x0 + np.sqrt(1 - ab[0]) * noise):
        return False, "q_sample 公式不正确"
    late = mod.q_sample(x0, 999, ab, noise)
    if abs(np.corrcoef(late.ravel(), noise.ravel())[0, 1]) < 0.99:
        return False, "最后一步的 x_t 应几乎等于纯噪声"
    try:
        mod.q_sample(x0, 1000, ab, noise)
    except IndexError:
        return True, "调度、累计乘积、加噪公式和越界检查都正确"
    return False, "t 越界时应抛出 IndexError"


def check_reparam(mod):
    rng = np.random.default_rng(1)
    eps = rng.normal(size=200000)
    z = mod.reparameterize(np.full(200000, 2.0), np.full(200000, np.log(9.0)), eps)
    if abs(z.mean() - 2.0) > 0.03 or abs(z.std() - 3.0) > 0.03:
        return False, f"logvar=log(9) 时标准差应为 3，得到 {z.std():.3f}"
    if not np.allclose(mod.reparameterize(np.zeros(3), np.zeros(3), np.ones(3)), np.ones(3)):
        return False, "mu=0、logvar=0 时 z 应等于 eps"
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
