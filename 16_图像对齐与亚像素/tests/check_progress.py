"""第 16 章自动验收：TODO / FAIL / PASS。

环境变量 COURSE_IMPL_DIR 可指向另一份实现目录（例如参考实现），
用于维护者验证测试本身，学员正常使用时不需要设置。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "16-1_相位相关与亚像素.py"


def load_assignment():
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    spec = importlib.util.spec_from_file_location("chapter16_assignment", base / ASSIGNMENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(status: str, name: str, detail: str) -> None:
    print(f"[{status:4}] {name} — {detail}")


def smooth_texture(shape, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=shape)
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    lowpass = np.exp(-(fy ** 2 + fx ** 2) / (2 * 0.08 ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(noise) * lowpass))


def fourier_shift(image, dy, dx):
    fy = np.fft.fftfreq(image.shape[0])[:, None]
    fx = np.fft.fftfreq(image.shape[1])[None, :]
    phase = np.exp(-2j * np.pi * (fy * dy + fx * dx))
    return np.real(np.fft.ifft2(np.fft.fft2(image) * phase))


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


def check_parabola(mod):
    f = mod.parabola_peak_offset
    a, b, c = -1.0, 0.3, 2.0  # y = a(x-b)^2 + c，真实峰在 0.3
    vals = [a * (x - b) ** 2 + c for x in (-1, 0, 1)]
    got = f(*vals)
    if not isinstance(got, float):
        return False, "返回值必须是 Python float"
    if abs(got - 0.3) > 1e-9:
        return False, f"对真抛物线应精确恢复 0.3，得到 {got}"
    if f(1.0, 1.0, 1.0) != 0.0:
        return False, "三点相等时应返回 0.0"
    if abs(f(0.0, 1.0, 1.0) - 0.5) > 1e-12 and f(0.0, 1.0, 1.0) > 0.5:
        return False, "结果需要裁剪到 [-0.5, 0.5]"
    if not (-0.5 <= f(-100.0, 1.0, 0.999) <= 0.5):
        return False, "结果需要裁剪到 [-0.5, 0.5]"
    return True, "真抛物线精确、退化输入和裁剪正确"


def check_ncc(mod):
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, size=(20, 24)).astype(np.uint8)
    tpl = image[5:12, 7:15].copy()
    resp = mod.ncc_map(image, tpl)
    if not isinstance(resp, np.ndarray) or resp.shape != (14, 17) or resp.dtype != np.float64:
        return False, "输出必须是 (H-h+1, W-w+1) float64"
    if abs(resp[5, 7] - 1.0) > 1e-9:
        return False, "模板来源位置的 ZNCC 应为 1"
    if np.unravel_index(np.argmax(resp), resp.shape) != (5, 7):
        return False, "最大响应位置应是模板来源 (5, 7)"
    if resp.max() > 1 + 1e-9 or resp.min() < -1 - 1e-9:
        return False, "ZNCC 必须落在 [-1, 1]"
    bright = mod.ncc_map(image.astype(np.float64) * 2 + 30, tpl)
    if abs(bright[5, 7] - 1.0) > 1e-9:
        return False, "ZNCC 应对线性亮度变化不变"
    flat = np.full((10, 10), 7.0)
    if not np.allclose(mod.ncc_map(flat, flat[:3, :3]), 0.0):
        return False, "平坦窗口应输出 0"
    return True, "峰值位置、取值范围、亮度不变性和平坦窗口处理正确"


def check_phase(mod):
    ref = smooth_texture((96, 96), 2)
    mov = np.roll(ref, (7, -11), axis=(0, 1))
    dy, dx = mod.phase_correlation(ref, mov, use_window=False)
    if not all(isinstance(v, float) for v in (dy, dx)):
        return False, "返回值必须是两个 Python float"
    if abs(dy - 7) > 0.05 or abs(dx + 11) > 0.05:
        return False, f"整数循环平移 (7, -11) 估计为 ({dy:.3f}, {dx:.3f})，方向或卷绕可能写反"
    errors = []
    for true_dy, true_dx in [(2.3, -4.6), (-5.75, 3.2), (0.5, 0.0)]:
        moved = fourier_shift(ref, true_dy, true_dx)
        dy, dx = mod.phase_correlation(ref, moved, use_window=False)
        errors.append(max(abs(dy - true_dy), abs(dx - true_dx)))
    worst = max(errors)
    if worst > 0.25:
        return False, f"亚像素误差最大 {worst:.3f} px，超过 0.25 px"
    rng = np.random.default_rng(3)
    noisy = np.roll(ref, (4, 6), axis=(0, 1)) + rng.normal(scale=0.05 * ref.std(), size=ref.shape)
    dy, dx = mod.phase_correlation(ref, noisy, use_window=True)
    if abs(dy - 4) > 0.3 or abs(dx - 6) > 0.3:
        return False, f"加噪声和窗函数后 (4, 6) 估计为 ({dy:.2f}, {dx:.2f})"
    return True, f"整数平移精确，亚像素最大误差 {worst:.3f} px，抗噪声"


def main() -> int:
    print("第 16 章进度检查\n" + "=" * 52)
    try:
        mod = load_assignment()
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "加载作业", f"{type(exc).__name__}: {exc}")
        return 1
    results = [
        run("抛物线亚像素", lambda: check_parabola(mod)),
        run("ZNCC 模板匹配", lambda: check_ncc(mod)),
        run("相位相关", lambda: check_phase(mod)),
    ]
    print("=" * 52)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("基本不变量通过；还要在实验记录里写清周期图案下的失败案例。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
