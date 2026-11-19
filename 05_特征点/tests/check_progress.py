"""第 5 章自动验收：TODO / FAIL / PASS。

检查用的都是 Harris 响应公式本身的性质（直边上行列式为 0、响应是灰度的四次齐次式、
单像素窗口下行列式恒为 0、图像转置后响应跟着转置），不和任何一份具体实现逐像素比较，
所以方框窗、高斯窗、窗口归一化与否、梯度算子的缩放系数都不影响结果。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "5-1_Harris响应.py"
CORNERS = [(18, 18), (18, 45), (45, 18), (45, 45)]   # 方块四个顶点的 (行, 列)


def load_assignment():
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    spec = importlib.util.spec_from_file_location("chapter5_assignment", base / ASSIGNMENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(status: str, name: str, detail: str) -> None:
    print(f"[{status:4}] {name} — {detail}")


def run(name: str, check, func) -> bool:
    try:
        with np.errstate(all="ignore"):
            ok, detail = check(func)
    except NotImplementedError as exc:
        report("TODO", name, str(exc))
        return False
    except Exception as exc:  # noqa: BLE001
        report("FAIL", name, f"运行异常：{type(exc).__name__}: {exc}")
        return False
    report("PASS" if ok else "FAIL", name, detail)
    return ok


def square(value=255, dtype=np.uint8) -> np.ndarray:
    image = np.zeros((64, 64), dtype=dtype)
    image[18:46, 18:46] = value
    return image


def vertical_edge(value=200) -> np.ndarray:
    image = np.zeros((64, 64), dtype=np.uint8)
    image[:, 32:] = value
    return image


def diagonal_edge() -> np.ndarray:
    yy, xx = np.mgrid[0:64, 0:64]
    return np.where(xx + yy > 63, 255, 0).astype(np.uint8)


def is_contract(response, image) -> bool:
    return (isinstance(response, np.ndarray) and response.shape == image.shape
            and response.dtype == np.float32 and bool(np.isfinite(response).all()))


def corner_peaks(response) -> np.ndarray:
    return np.array([float(response[y - 3:y + 4, x - 3:x + 4].max()) for y, x in CORNERS])


def check_contract(func):
    flat = np.full((48, 52), 127, dtype=np.uint8)
    flat_response = func(flat)
    if not isinstance(flat_response, np.ndarray) or flat_response.shape != flat.shape or flat_response.dtype != np.float32:
        return False, "输出必须是和输入同 shape 的 float32 数组"
    if not np.isfinite(flat_response).all():
        return False, "响应里不能有 NaN 或 inf"
    if float(np.max(np.abs(flat_response))) > 1e-4:
        return False, "常数图上每个像素的响应都应为 0：如果只有图像四周不为 0，检查求梯度和窗口求和时边界是不是补了零"
    return True, "输出的形状和类型正确，常数图响应为 0"


def check_square(func):
    image = square()
    response = func(image)
    if not is_contract(response, image):
        return False, "方块图上的输出不是同 shape、全部有限的 float32"
    peaks = corner_peaks(response)
    if float(peaks.max()) <= 0:
        return False, "方块的顶点应有正响应：检查三个乘积是不是先在窗口里求和、再代进公式"
    if float(peaks.max()) < 0.8 * float(response.max()):
        return False, "全图最强的响应应出现在方块顶点附近"
    if float(peaks.min()) < 0.999 * float(peaks.max()):
        shown = "、".join(f"{v / peaks.max():.2f}" for v in peaks)
        if float(peaks.min()) <= 0.01 * float(peaks.max()):
            hint = "有的顶点几乎没有响应，检查梯度有没有负值，求梯度之前图像是什么类型"
        else:
            hint = "检查求梯度的算子是不是左右、上下对称的，只取一侧的差分会让四个顶点不一样"
        return False, f"方块四个顶点的响应应相等，现在按左上、右上、左下、右下是最大值的 {shown} 倍：{hint}"
    unit = square(0.5, np.float64)
    unit_response = func(unit)
    if not is_contract(unit_response, unit):
        return False, "输入是 0 到 1 之间的浮点图时，输出同样必须是同 shape、全部有限的 float32"
    if float(corner_peaks(unit_response).min()) <= 0:
        return False, "同一个方块换成 0 到 1 之间的浮点图之后顶点没有正响应：检查是不是把输入转成了整数类型"
    wide = np.zeros((64, 64), dtype=np.uint8)
    wide[14:30, 20:50] = 180
    a, b = func(wide), func(np.ascontiguousarray(wide.T))
    if not np.allclose(a.T, b, rtol=0, atol=1e-3 * float(np.abs(a).max())):
        return False, "把图像转置后响应图也应跟着转置：检查 x、y 两个方向用的梯度算子和窗口是不是一样大"
    return True, "方块四个顶点响应为正且相等，浮点输入可用，图像转置后响应跟着转置"


def check_edges(func):
    edge = func(vertical_edge())
    if not is_contract(edge, vertical_edge()):
        return False, "直边图上的输出不是同 shape、全部有限的 float32"
    scale = float(np.abs(edge[8:-8, 8:-8]).max())
    middle = edge[24:40, 26:38]
    if scale <= 0 or float(middle.min()) >= 0:
        return False, "竖直的直边上响应应为负数：只有一个方向有梯度时行列式为 0，剩下的那一项是负的。检查有没有把负的响应截掉或取了绝对值"
    if float(middle.max()) > 1e-4 * scale:
        return False, "竖直的直边中段出现了正响应：直边不是角点"
    diag = func(diagonal_edge())
    diag_scale = float(np.abs(diag[20:44, 20:44]).max())
    if float(diag[20:44, 20:44].max()) > 1e-4 * diag_scale:
        return False, "45 度直边的中段出现了正响应：斜边上 Ix 和 Iy 同时不为 0，检查行列式有没有漏项、每一项的次数对不对"
    if float(diag[20:44, 20:44].min()) >= 0:
        return False, "45 度直边上响应应为负数"
    return True, "竖直和 45 度的直边上响应都不为正"


def check_brightness(func):
    bright, dark = func(square(200)), func(square(100))
    if float(dark.max()) <= 0 or float(bright.max()) <= 0:
        return False, "灰度 100 和 200 的方块顶点都应有正响应"
    ratio = float(dark.max()) / float(bright.max())
    if abs(ratio - 1 / 16) > 0.01 / 16:
        return False, (f"方块的灰度从 200 减到 100，最大响应应变成 1/16，现在是 {ratio:.4f} 倍："
                       "检查有没有按图像的最大值做归一化，梯度有没有饱和")
    edge_bright, edge_dark = float(func(vertical_edge(200))[32, 31]), float(func(vertical_edge(100))[32, 31])
    if edge_bright >= 0 or edge_dark >= 0:
        return False, "直边上的响应应为负数"
    ratio = edge_dark / edge_bright
    if abs(ratio - 1 / 16) > 0.01 / 16:
        return False, (f"直边的灰度从 200 减到 100，边上的响应应变成 1/16，现在是 {ratio:.4f} 倍："
                       "响应公式的每一项都应是梯度的四次式，检查迹那一项")
    return True, "灰度减半时角点和直边的响应都变成 1/16"


def check_k(func):
    image = vertical_edge()
    low, high = float(func(image, 0.04, 3)[32, 31]), float(func(image, 0.08, 3)[32, 31])
    if low >= 0:
        return False, "直边上的响应应为负数"
    if abs(high / low - 2.0) > 2e-3:
        return False, f"直边上行列式为 0，k 从 0.04 改成 0.08 响应应正好变成 2 倍，现在是 {high / low:.3f} 倍：检查参数 k 有没有真正用上"
    corner = square()
    if np.allclose(func(corner, 0.04, 3), func(corner, 0.2, 3)):
        return False, "k 从 0.04 改成 0.2，方块上的输出没有变化"
    return True, "响应随 k 的变化符合公式"


def check_window(func):
    single = func(square(), 0.04, 1)
    if not is_contract(single, square()):
        return False, "window_size=1 时输出不是同 shape、全部有限的 float32"
    if float(single.max()) > 1e-4 * float(np.abs(single).max()):
        return False, "window_size=1 时窗口里只有像素自己，行列式恒为 0，不应出现正响应：检查 window_size 是不是用在了窗口求和上"
    image = vertical_edge()

    def band(window_size: int) -> int:
        row = func(image, 0.04, window_size)[32]
        return int((row < -1e-3 * float(np.abs(row).max())).sum())

    narrow, wide = band(3), band(7)
    if wide <= narrow:
        return False, f"窗口从 3 加到 7，直边两侧有响应的宽度应变大，现在是 {narrow} 列和 {wide} 列：检查参数 window_size 有没有真正用上"
    return True, "window_size=1 时没有正响应，窗口变大时响应的范围变宽"


CHECKS = [
    ("输出契约与平坦区", check_contract),
    ("方块顶点", check_square),
    ("直边", check_edges),
    ("亮度缩放", check_brightness),
    ("参数 k", check_k),
    ("参数 window_size", check_window),
]


def run_all(func) -> list[bool]:
    return [run(name, check, func) for name, check in CHECKS]


def main() -> int:
    print("第 5 章进度检查\n" + "=" * 52)
    try:
        func = load_assignment().harris_response
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "加载作业", f"{type(exc).__name__}: {exc}")
        return 1
    results = run_all(func)
    print("=" * 52)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("全部通过；阈值和非极大值抑制不在验收范围内，在 Notebook 和实验记录里自己做。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
