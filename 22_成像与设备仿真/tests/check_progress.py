"""第 22 章自动验收：TODO / FAIL / PASS。

期望值都是手算的，或者来自作业文档字符串里写明的模型的数学性质（泊松分布的概率、方差公式），不依赖参考实现。
"""

from __future__ import annotations

import importlib.util
import math
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


def raises_value_error(func, *args, **kwargs):
    """返回 None 表示抛出了 ValueError；否则返回一句说明。"""
    try:
        func(*args, **kwargs)
    except ValueError:
        return None
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        return f"抛出的是 {type(exc).__name__}"
    return "没有抛出异常"


def check_rayleigh(mod):
    r = mod.rayleigh_resolution_um(500.0, 0.5)
    if not isinstance(r, float) or abs(r - 0.61) > 1e-12:
        return False, f"500 nm、NA 0.5 时应为 0.61 um，得到 {r!r}"
    if abs(mod.rayleigh_resolution_um(266.0, 0.9) - 0.61 * 0.266 / 0.9) > 1e-12:
        return False, "266 nm 深紫外、NA 0.9 的结果不对"
    top = mod.rayleigh_resolution_um(600.0, 1.5)
    if abs(top - 0.244) > 1e-12:
        return False, f"NA 正好等于 1.5 是允许的（区间是 (0, 1.5]），600 nm 时应为 0.244 um，得到 {top!r}"
    for bad in (50.0, 1.51, 0.0, -0.5):
        why = raises_value_error(mod.rayleigh_resolution_um, 500.0, bad)
        if why:
            return False, f"NA={bad} 不在 (0, 1.5] 内，应抛出 ValueError，现在{why}"
    return True, "单位换算正确，NA 的上下界都按 (0, 1.5] 检查"


def check_psf_conv(mod):
    k = mod.gaussian_psf(1.5)
    if not isinstance(k, np.ndarray) or k.dtype != np.float64:
        return False, "gaussian_psf 应返回 float64 的数组"
    if k.shape != (11, 11) or abs(k.sum() - 1) > 1e-12 or np.unravel_index(k.argmax(), k.shape) != (5, 5):
        return False, "sigma=1.5 时应为 11x11、和为 1、峰在中心"
    if not np.allclose(k, k.T) or not np.allclose(k, k[::-1, ::-1]):
        return False, "高斯 PSF 应对称"
    for sigma, size in ((1.0, 7), (2.0, 13), (0.5, 5)):
        got = mod.gaussian_psf(sigma).shape
        if got != (size, size):
            return False, f"radius 默认应为 ceil(3 * sigma)：sigma={sigma} 时核应为 {size}x{size}，得到 {got[0]}x{got[1]}"
    # sigma^2 = 1 / (2 ln 2) 时，离中心 1 个像素处正好是中心的一半，对角是四分之一
    k3 = mod.gaussian_psf(math.sqrt(1.0 / (2.0 * math.log(2.0))), radius=1)
    if k3.shape != (3, 3):
        return False, f"给了 radius=1 时核应为 3x3，得到 {k3.shape}。radius 参数不能被忽略"
    want3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
    if not np.allclose(k3, want3, atol=1e-12):
        return False, ("高斯的宽度不对：sigma 取 sqrt(1 / (2 ln 2))、radius=1 时，离中心 1 个像素处应是中心的一半，"
                       f"3x3 的核应为 [[1,2,1],[2,4,2],[1,2,1]]/16，得到中心 {k3[1, 1]:.4f}、边 {k3[1, 0]:.4f}、角 {k3[0, 0]:.4f}。"
                       "对照文档字符串，sigma 是标准差")
    wide = mod.gaussian_psf(1.0, radius=6)
    if wide.shape != (13, 13) or abs(wide.sum() - 1) > 1e-12:
        return False, "sigma=1、radius=6 时应为 13x13、和为 1"
    for bad, radius in ((0.0, None), (-1.0, None), (-0.2, None), (-1.0, 2)):
        why = raises_value_error(mod.gaussian_psf, bad) if radius is None else raises_value_error(mod.gaussian_psf, bad, radius=radius)
        if why:
            extra = "" if radius is None else f"、radius={radius}"
            return False, f"sigma={bad}{extra} 时应抛出 ValueError（sigma <= 0 都不允许，不只是 0），现在{why}"

    rng = np.random.default_rng(0)
    img = rng.normal(size=(20, 24))
    keep = img.copy()
    shift = np.zeros((3, 3))
    shift[1, 0] = 1.0  # 卷积核：输出(y, x) 取输入 (y, x+1)
    out = mod.convolve_same(img, shift)
    if out.shape != img.shape or not np.allclose(out[:, :-1], img[:, 1:], atol=1e-9):
        return False, "卷积方向不对：非对称核需要翻转（这里输出应等于输入左移一列）"
    pad = np.pad(img, 1, mode="reflect")
    if abs(out[5, -1] - pad[6, -1]) > 1e-9:
        return False, "边界应按反射填充"
    if not np.array_equal(img, keep):
        return False, "传入的 image 被改掉了，convolve_same 不能原地修改输入"

    # 竖直方向：3 行 1 列的核，1 在最上面，输出(y, x) 取输入 (y+1, x)
    up = np.array([[1.0], [0.0], [0.0]])
    out_v = mod.convolve_same(img, up)
    if out_v.shape != img.shape or not np.allclose(out_v[:-1, :], img[1:, :], atol=1e-9):
        return False, ("竖直方向不对：3x1 的核 [[1],[0],[0]] 卷积后，输出应等于输入上移一行。"
                       "检查核在上下方向有没有翻转，以及行数和列数不同的核有没有处理对")
    if not np.allclose(out_v[-1, :], img[-2, :], atol=1e-9):
        return False, "下边界应按反射填充：3x1 的核 [[1],[0],[0]] 时，最后一行的输出应等于输入的倒数第二行"
    # 1 行 5 列的核，1 在最右边，输出(y, x) 取输入 (y, x-2)
    right = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
    out_h = mod.convolve_same(img, right)
    if out_h.shape != img.shape or not np.allclose(out_h[:, 2:], img[:, :-2], atol=1e-9):
        return False, "1x5 的核 [[0,0,0,0,1]] 卷积后，输出应等于输入右移两列。检查两个方向的填充宽度是不是各按核的行数和列数算"
    if not np.allclose(out_h[:, 0], img[:, 2], atol=1e-9) or not np.allclose(out_h[:, 1], img[:, 1], atol=1e-9):
        return False, "左边界应按反射填充：1x5 的核 [[0,0,0,0,1]] 时，第 0 列的输出应等于输入的第 2 列"

    # 手算：3x3 的整数图像和全 1 的 3x3 核，四条边和四个角都用到了反射填充
    small = np.arange(1, 10).reshape(3, 3)
    box = mod.convolve_same(small, np.ones((3, 3)))
    want_box = np.array([[33.0, 36.0, 39.0], [42.0, 45.0, 48.0], [51.0, 54.0, 57.0]])
    if not isinstance(box, np.ndarray) or box.shape != (3, 3) or not np.allclose(box, want_box, atol=1e-9):
        return False, ("图像 [[1,2,3],[4,5,6],[7,8,9]] 和全 1 的 3x3 核，按 reflect 填充手算应为 [[33,36,39],[42,45,48],[51,54,57]]，"
                       f"得到 {np.asarray(box).tolist()}。reflect 不重复边上的那个像素")
    # 手算：整数图像和带小数的不对称核 [[0.25, 0, 0.5]]，输出(y, x) = 0.25 * 输入(y, x+1) + 0.5 * 输入(y, x-1)
    frac = mod.convolve_same(small, np.array([[0.25, 0.0, 0.5]]))
    want_frac = np.array([[1.5, 1.25, 1.5], [3.75, 3.5, 3.75], [6.0, 5.75, 6.0]])
    if frac.dtype != np.float64 or not np.allclose(frac, want_frac, atol=1e-9):
        return False, ("整数图像 [[1,2,3],[4,5,6],[7,8,9]] 和核 [[0.25, 0, 0.5]]：应返回 float64 的 [[1.5,1.25,1.5],[3.75,3.5,3.75],[6,5.75,6]]，"
                       f"得到 {frac.dtype} 的 {np.asarray(frac).tolist()}。输入是整数数组时，结果不能被截成整数")

    blurred = mod.convolve_same(np.ones((15, 15)), k)
    if not np.allclose(blurred, 1.0, atol=1e-9):
        return False, "常数图像卷积归一化 PSF 后应不变"
    return True, "PSF 的形状、宽度和归一化正确，卷积的方向、边界和非方形核正确"


def check_capture(mod):
    mean = np.full((400, 400), 1000.0)
    keep = mean.copy()
    a = mod.simulate_capture(mean, 5.0, 2.0, 12, np.random.default_rng(1))
    b = mod.simulate_capture(mean, 5.0, 2.0, 12, np.random.default_rng(1))
    if not isinstance(a, np.ndarray) or a.dtype != np.int64 or a.shape != mean.shape or not np.array_equal(a, b):
        return False, "输出应为和输入同形状的 int64 数组，且相同随机种子结果相同"
    if not np.array_equal(mean, keep):
        return False, "传入的 mean_electrons 被改掉了，simulate_capture 不能原地修改输入"
    if np.array_equal(a, mod.simulate_capture(mean, 5.0, 2.0, 12, np.random.default_rng(2))):
        return False, "换了随机种子，结果却完全一样。随机数要来自传入的 rng，不要在函数里另建随机数发生器"
    rng = np.random.default_rng(1)
    first = mod.simulate_capture(mean, 5.0, 2.0, 12, rng)
    second = mod.simulate_capture(mean, 5.0, 2.0, 12, rng)
    if np.array_equal(first, second):
        return False, "用同一个 rng 连续调用两次，两次的结果应不同。不要在函数里重置或复制 rng"
    if abs(a.mean() - 500.0) > 0.25:
        return False, f"1000 e-、增益 2 时均值应约 500 ADU，得到 {a.mean():.2f}。检查增益的方向，以及量化是不是四舍五入"
    want_var = (1000.0 + 25.0) / 4.0 + 1.0 / 12.0
    if abs(a.var() - want_var) / want_var > 0.05:
        return False, f"方差应约 {want_var:.0f}，得到 {a.var():.0f}，检查散粒噪声和读出噪声"

    # 很暗、没有读出噪声、增益 1：像素值就是电子数，应服从均值 2 的泊松分布，P(n) = e^-2 * 2^n / n!
    low = mod.simulate_capture(np.full(200000, 2.0), 0.0, 1.0, 12, np.random.default_rng(5))
    if low.shape != (200000,):
        return False, f"输入是一维数组时，输出也应是同样长度的一维数组，得到形状 {low.shape}"
    want_p = [0.1353, 0.2707, 0.2707, 0.1804, 0.0902]
    got_p = [float((low == n).mean()) for n in range(5)]
    if max(abs(g - w) for g, w in zip(got_p, want_p)) > 0.01:
        return False, ("平均 2 个电子、读出噪声 0、增益 1 时，像素值 0 到 4 出现的比例应接近泊松分布的 "
                       f"{want_p}，得到 {[round(v, 4) for v in got_p]}。散粒噪声要按泊松分布抽样，暗的时候它和高斯分布差得很多")

    for bits, top in ((12, 4095), (8, 255), (14, 16383)):
        sat = mod.simulate_capture(np.full((10, 10), 1e7), 5.0, 2.0, bits, np.random.default_rng(2))
        if sat.max() != top or sat.min() != top:
            return False, f"超过量程应饱和在 2**bit_depth - 1：bit_depth={bits} 时所有像素都应是 {top}，得到最大 {sat.max()}、最小 {sat.min()}"
    wide = mod.simulate_capture(np.full((50, 50), 20000.0), 5.0, 2.0, 14, np.random.default_rng(2))
    if abs(wide.mean() - 10000.0) > 50.0:
        return False, f"14 位时 20000 e-、增益 2 应约 10000 ADU，没有饱和，得到 {wide.mean():.0f}。上限要按传入的 bit_depth 算"

    zeros = np.zeros((200, 200))
    dark = mod.simulate_capture(zeros, 8.0, 1.5, 12, np.random.default_rng(3))
    if zeros.any():
        return False, "传入的 mean_electrons 被改掉了（全暗的输入不再是全 0），simulate_capture 不能原地修改输入"
    if dark.min() < 0:
        return False, "负值应裁剪到 0"
    try:
        dark0 = mod.simulate_capture(zeros, 8.0, 1.5, 12, np.random.default_rng(3), black_level_adu=0.0)
    except TypeError as exc:
        return False, f"simulate_capture 要能接受关键字参数 black_level_adu（见文档字符串）：{exc}"
    if not np.array_equal(dark, dark0):
        return False, "不传 black_level_adu 和传 0 的结果应完全相同（默认值是 0）"

    # 黑电平：全暗、读出噪声 8 e-、增益 1.5、黑电平 64 ADU，分布离 0 很远，不会被裁剪
    lifted = mod.simulate_capture(zeros, 8.0, 1.5, 12, np.random.default_rng(3), black_level_adu=64.0).astype(np.float64)
    if abs(lifted.mean() - 64.0) > 0.2:
        return False, (f"全暗、黑电平 64 ADU 时均值应约 64 ADU，得到 {lifted.mean():.2f}。"
                       "黑电平的单位是 ADU，要在裁剪到 0 之前加上")
    want_std = math.sqrt(8.0 ** 2 / 1.5 ** 2 + 1.0 / 12.0)
    if abs(lifted.std() / want_std - 1) > 0.03:
        return False, (f"全暗、读出噪声 8 e-、增益 1.5、黑电平 64 ADU 时标准差应约 {want_std:.2f} ADU，得到 {lifted.std():.2f}。"
                       "如果得到 3.1 左右，说明负值在加黑电平之前就被裁掉了")
    bright = mod.simulate_capture(np.full((200, 200), 1000.0), 5.0, 2.0, 12, np.random.default_rng(1), black_level_adu=100.0)
    if abs(bright.mean() - 600.0) > 0.5:
        return False, f"1000 e-、增益 2、黑电平 100 ADU 时均值应约 600 ADU，得到 {bright.mean():.2f}"
    sat_b = mod.simulate_capture(np.full((10, 10), 1e7), 5.0, 2.0, 12, np.random.default_rng(2), black_level_adu=100.0)
    if sat_b.max() != 4095:
        return False, f"加了黑电平之后，上限仍然是 2**bit_depth - 1，得到最大值 {sat_b.max()}"
    return True, "均值、方差、泊松分布、位深、饱和、下限、黑电平和可复现性都正确"


def check_ptc(mod):
    # 手算：四个点 (0,5)、(2,4)、(4,5)、(6,8) 的最小二乘直线是 方差 = 0.5 * 均值 + 4
    gain, read = mod.estimate_gain_from_ptc(np.array([0.0, 2.0, 4.0, 6.0]), np.array([5.0, 4.0, 5.0, 8.0]))
    if not isinstance(gain, float) or not isinstance(read, float):
        return False, "返回值应为两个 Python float"
    if abs(gain - 2.0) > 1e-9 or abs(read - 4.0) > 1e-9:
        return False, ("四个点 (0,5)、(2,4)、(4,5)、(6,8) 的最小二乘直线是 方差 = 0.5 x 均值 + 4，应返回增益 2.0、读出噪声 4.0，"
                       f"得到 {gain!r}、{read!r}。要用全部点做各点权重相同的最小二乘；读出噪声的单位是电子")
    # 截距为负：方差 = 0.5 * 均值 - 3
    gain, read = mod.estimate_gain_from_ptc(np.array([100.0, 200.0, 400.0, 800.0]), np.array([47.0, 97.0, 197.0, 397.0]))
    if abs(gain - 2.0) > 1e-9 or not (isinstance(read, float) and read == 0.0):
        return False, f"四个点正好在直线 方差 = 0.5 x 均值 - 3 上：应返回增益 2.0；截距为负，读出噪声应返回 0.0。得到 {gain!r}、{read!r}"

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
    return True, f"手算的两组点正确；从仿真数据恢复增益 {gain:.3f} e-/ADU、读出噪声 {read:.2f} e-"


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
