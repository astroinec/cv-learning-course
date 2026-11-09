"""第 22 章自动验收：TODO / FAIL / PASS。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "22-1_成像仿真.py"


def load_assignment():
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    spec = importlib.util.spec_from_file_location("chapter22_assignment", base / ASSIGNMENT)
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


def check_rayleigh(mod):
    r = mod.rayleigh_resolution_um(500.0, 0.5)
    if not isinstance(r, float) or abs(r - 0.61) > 1e-12:
        return False, f"500 nm、NA 0.5 时应为 0.61 um，得到 {r!r}"
    if abs(mod.rayleigh_resolution_um(266.0, 0.9) - 0.61 * 0.266 / 0.9) > 1e-12:
        return False, "266 nm 深紫外、NA 0.9 的结果不对"
    try:
        mod.rayleigh_resolution_um(500.0, 50.0)
    except ValueError:
        return True, "单位换算正确，非法 NA 报错"
    return False, "NA 超出合理范围时应抛出 ValueError"


def check_psf_conv(mod):
    k = mod.gaussian_psf(1.5)
    if k.shape != (11, 11) or abs(k.sum() - 1) > 1e-12 or np.unravel_index(k.argmax(), k.shape) != (5, 5):
        return False, "sigma=1.5 时应为 11x11、和为 1、峰在中心"
    if not np.allclose(k, k.T) or not np.allclose(k, k[::-1, ::-1]):
        return False, "高斯 PSF 应对称"
    try:
        mod.gaussian_psf(0.0)
        return False, "sigma <= 0 时应抛出 ValueError"
    except ValueError:
        pass
    rng = np.random.default_rng(0)
    img = rng.normal(size=(20, 24))
    shift = np.zeros((3, 3))
    shift[1, 0] = 1.0  # 卷积核：输出(y, x) 取输入 (y, x+1)
    out = mod.convolve_same(img, shift)
    if out.shape != img.shape or not np.allclose(out[:, :-1], img[:, 1:], atol=1e-9):
        return False, "卷积方向不对：非对称核需要翻转（这里输出应等于输入左移一列）"
    pad = np.pad(img, 1, mode="reflect")
    if abs(out[5, -1] - pad[6, -1]) > 1e-9:
        return False, "边界应按反射填充"
    blurred = mod.convolve_same(np.ones((15, 15)), k)
    if not np.allclose(blurred, 1.0, atol=1e-9):
        return False, "常数图像卷积归一化 PSF 后应不变"
    return True, "PSF 形状和归一化正确，卷积方向和边界正确"


def check_capture(mod):
    mean = np.full((400, 400), 1000.0)
    a = mod.simulate_capture(mean, 5.0, 2.0, 12, np.random.default_rng(1))
    b = mod.simulate_capture(mean, 5.0, 2.0, 12, np.random.default_rng(1))
    if a.dtype != np.int64 or not np.array_equal(a, b):
        return False, "输出应为 int64，且相同随机种子结果相同"
    if abs(a.mean() - 500.0) > 1.0:
        return False, f"1000 e-、增益 2 时均值应约 500 ADU，得到 {a.mean():.1f}"
    want_var = (1000.0 + 25.0) / 4.0 + 1.0 / 12.0
    if abs(a.var() - want_var) / want_var > 0.05:
        return False, f"方差应约 {want_var:.0f}，得到 {a.var():.0f}，检查散粒噪声和读出噪声"
    sat = mod.simulate_capture(np.full((10, 10), 1e7), 5.0, 2.0, 12, np.random.default_rng(2))
    if sat.max() != 4095:
        return False, "超过满阱应饱和在 2**bit_depth - 1"
    dark = mod.simulate_capture(np.zeros((50, 50)), 5.0, 2.0, 12, np.random.default_rng(3))
    if dark.min() < 0:
        return False, "负值应裁剪到 0"
    return True, "均值、方差、饱和、下限和可复现性都正确"


def check_ptc(mod):
    rng = np.random.default_rng(4)
    means, vars_ = [], []
    for level in [50, 200, 500, 1000, 2000, 4000]:
        f = mod.simulate_capture(np.full((200, 200), float(level)), 8.0, 1.5, 14, rng).astype(np.float64)
        means.append(f.mean())
        vars_.append(f.var())
    gain, read = mod.estimate_gain_from_ptc(np.array(means), np.array(vars_))
    if not isinstance(gain, float) or not isinstance(read, float):
        return False, "返回值应为两个 Python float"
    if abs(gain - 1.5) / 1.5 > 0.05:
        return False, f"增益真值 1.5，估计 {gain:.3f}"
    if abs(read - 8.0) / 8.0 > 0.25:
        return False, f"读出噪声真值 8.0，估计 {read:.2f}"
    return True, f"从仿真数据恢复增益 {gain:.3f} e-/ADU、读出噪声 {read:.2f} e-"


def main() -> int:
    print("第 22 章进度检查\n" + "=" * 52)
    try:
        mod = load_assignment()
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "加载作业", f"{type(exc).__name__}: {exc}")
        return 1
    results = [
        run("瑞利分辨率", lambda: check_rayleigh(mod)),
        run("PSF 与卷积", lambda: check_psf_conv(mod)),
        run("传感器噪声模型", lambda: check_capture(mod)),
        run("光子转移曲线", lambda: check_ptc(mod)),
    ]
    print("=" * 52)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("仿真器通过；下一步用它生成第 24 章项目的数据，并研究像素尺寸对小缺陷检出的影响。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
