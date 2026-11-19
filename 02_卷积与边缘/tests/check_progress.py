"""第 2 章自动验收；TODO 是正常初始状态。

只检查作业文档字符串里写明的行为：输出 shape 和 float32、核不翻转、NumPy reflect 语义的边界。
期望值来自手算的小例子和 OpenCV 的 cv2.filter2D（它算的也是相关，默认边界是不重复边界像素的镜像）。
运行：python tests/check_progress.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
_IMPL = Path(os.environ["COURSE_IMPL_DIR"]) / CHAPTER.name if os.environ.get("COURSE_IMPL_DIR") else CHAPTER


def load_assignment():
    spec = importlib.util.spec_from_file_location("chapter2_assignment", _IMPL / "2-1_手写相关运算.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(label: str, status: str, detail: str) -> None:
    print(f"[{status:4}] {label} — {detail}")


def run(label: str, check, func) -> bool:
    """运行一项检查。还没实现显示 TODO，结果不对或抛异常显示 FAIL，都不会让脚本崩溃。"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            status, detail = check(func)
    except NotImplementedError as exc:
        status, detail = "TODO", str(exc) or "还没有实现"
    except Exception as exc:  # noqa: BLE001
        status, detail = "FAIL", str(exc) if isinstance(exc, RuntimeError) else f"运行异常：{type(exc).__name__}: {exc}"
    report(label, status, detail)
    return status == "PASS"


def call(what: str, func, image, kernel):
    """调用作业函数。传进去的是副本，某一次调用即使改了输入也不影响后面的用例；出错时带上是哪个用例。"""
    try:
        return func(image.copy(), kernel.copy())
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{what}时运行异常（{type(exc).__name__}: {exc}）") from exc


def bad_contract(result, shape) -> str | None:
    if not isinstance(result, np.ndarray) or result.shape != shape or result.dtype != np.float32:
        got = f"{type(result).__name__}" if not isinstance(result, np.ndarray) else f"shape {result.shape}、dtype {result.dtype}"
        return f"输出必须是 shape {shape} 的 float32 数组，得到 {got}"
    return None


# ---------------------------------------------------------------- 1. 契约、恒等核、常量图
def check_contract(func):
    image = np.arange(35, dtype=np.float32).reshape(5, 7)
    identity = np.zeros((3, 3), dtype=np.float32)
    identity[1, 1] = 1
    result = call("5×7 的图配 3×3 恒等核", func, image, identity)
    msg = bad_contract(result, image.shape)
    if msg:
        return "FAIL", msg
    if not np.allclose(result, image):
        return "FAIL", "恒等核（只有中心为 1）应保持图像不变；检查核的中心是不是对准了当前像素"
    constant = np.full((6, 8), 4, dtype=np.uint8)
    averaged = call("uint8 的常量图配 3×3 均值核", func, constant, np.ones((3, 3), dtype=np.float32) / 9)
    msg = bad_contract(averaged, constant.shape)
    if msg:
        return "FAIL", "输入是 uint8 时：" + msg
    if not np.allclose(averaged, 4, atol=1e-5):
        return "FAIL", "归一化的均值核作用在常量图上应处处等于常量；边界一圈偏低说明图像外被当成了 0"
    return "PASS", "shape、float32、恒等核和常量图都满足"


# ---------------------------------------------------------------- 2. 核不翻转：冲激响应
def check_impulse(func):
    image = np.zeros((7, 9), dtype=np.float32)
    image[3, 4] = 1
    kernel = np.arange(9, dtype=np.float32).reshape(3, 3)
    result = call("7×9 的冲激图配 3×3 的核", func, image, kernel)
    msg = bad_contract(result, image.shape)
    if msg:
        return "FAIL", msg
    patch = np.round(result[2:5, 3:6], 3)
    # 手算：相关把核“翻转着”印在冲激周围（讲义 2.1 节的表）
    expected = np.array([[8, 7, 6], [5, 4, 3], [2, 1, 0]], dtype=np.float32)
    outside = result.copy()
    outside[2:5, 3:6] = 0
    if np.allclose(patch, expected, atol=1e-4) and np.allclose(outside, 0, atol=1e-4):
        return "PASS", "冲激图上的响应是相关的结果"
    if np.allclose(patch, kernel, atol=1e-4):
        return "FAIL", "冲激图上印出来的就是核本身，这是严格卷积的结果；文档字符串要求相关（核不翻转），对照讲义 2.1 节“看冲激响应”"
    return "FAIL", (f"只有一个像素为 1 的冲激图配核 [[0,1,2],[3,4,5],[6,7,8]]，冲激周围 3×3 应为 {expected.astype(int).tolist()}，"
                    f"得到 {patch.tolist()}；检查核的行列下标和窗口里像素的对应关系，以及有没有对核做多余的处理（归一化、转置）")


# ---------------------------------------------------------------- 3. reflect 边界：手算
def check_border(func):
    # image[y, x] = 10*y + x + 1，每个像素的值都不同，取到哪个像素一眼能看出来
    image = (10 * np.arange(4)[:, None] + np.arange(6)[None, :] + 1).astype(np.float32)
    pick_left = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32)      # 输出 = 左边那个像素
    pick_right = pick_left[:, ::-1].copy()                                          # 输出 = 右边那个像素
    pick_up = pick_left.T.copy()                                                    # 输出 = 上面那个像素
    pick_down = pick_up[::-1, :].copy()                                             # 输出 = 下面那个像素
    # (核, 要看的输出位置, reflect 的期望值, 复制边缘的值, 绕到另一侧的值, 说明)
    cases = [
        (pick_left, (0, 0), 2, 1, 6, "最左一列的左邻居"),
        (pick_right, (0, 5), 5, 6, 1, "最右一列的右邻居"),
        (pick_up, (0, 0), 11, 1, 31, "最上一行的上邻居"),
        (pick_down, (3, 0), 21, 31, 1, "最下一行的下邻居"),
    ]
    for kernel, (y, x), want, edge_value, wrap_value, name in cases:
        result = call(f"{name}（4×6 的图配只有一个 1 的 3×3 核）", func, image, kernel)
        msg = bad_contract(result, image.shape)
        if msg:
            return "FAIL", msg
        got = float(result[y, x])
        if abs(got - want) < 1e-4:
            continue
        if abs(got - edge_value) < 1e-4:
            reason = "等于边界像素自己，说明边界像素被重复了一次（复制边缘和 symmetric 都是这样）"
        elif abs(got - wrap_value) < 1e-4:
            reason = "等于图像另一侧的像素，说明填充是绕回去取的"
        elif abs(got) < 1e-4:
            reason = "等于 0，说明图像外补了零"
        else:
            reason = "不是任何一种常见填充的结果，先检查核的下标"
        return "FAIL", f"{name}：图像 image[y, x] = 10*y + x + 1，按文档字符串的 reflect 语义应取到 {want}，得到 {got:g}，{reason}；对照讲义 2.3 节的填充对照表"
    return "PASS", "四条边上取到的都是不重复边界像素的镜像"


# ---------------------------------------------------------------- 4. 非方形、非对称核：和 cv2.filter2D 对照
def check_shapes(func):
    try:
        import cv2
    except ImportError:
        return "FAIL", "这一项要用 OpenCV 做对照，请先按 docs/环境安装.md 装好 opencv-python"
    rng = np.random.default_rng(21)
    image = rng.integers(0, 51, (9, 12)).astype(np.float32)      # 整数值，float32 下的加法和乘法没有舍入误差
    square_ok = None
    for shape in [(3, 3), (3, 5), (5, 3), (1, 5), (5, 1), (5, 5), (1, 1)]:
        kernel = rng.integers(-3, 4, shape).astype(np.float32)
        kernel[0, -1] += 5                                        # 保证核不对称
        hint = "3×3 的核是对的，问题出在这种形状上：检查行方向和列方向的半径是不是分别按核的行数和列数算的，半径为 0 时切片有没有出错" if square_ok else "检查核有没有被翻转或转置、边界是不是 reflect"
        try:
            result = call(f"核的形状是 {shape}（9×12 的图）", func, image, kernel)
        except NotImplementedError:                      # 它是 RuntimeError 的子类，要先放行，交给 run 显示成 TODO
            raise
        except RuntimeError as exc:
            return "FAIL", f"{exc}；{hint}"
        msg = bad_contract(result, image.shape)
        if msg:
            return "FAIL", f"核的形状是 {shape} 时：{msg}；{hint}"
        err = float(np.abs(result - cv2.filter2D(image, cv2.CV_32F, kernel)).max())
        if shape == (3, 3):
            square_ok = err <= 1e-3
        if err > 1e-3:
            return "FAIL", f"核的形状是 {shape}（9×12 的图）时与 cv2.filter2D 最大差 {err:.4g}；{hint}"
    return "PASS", "7 种形状的非对称核都与 cv2.filter2D 一致"


# ---------------------------------------------------------------- 5. 整数输入、符号、不修改输入
def check_dtypes(func):
    try:
        import cv2
    except ImportError:
        return "FAIL", "这一项要用 OpenCV 做对照，请先按 docs/环境安装.md 装好 opencv-python"
    rng = np.random.default_rng(22)
    image_u8 = rng.integers(150, 256, (8, 10)).astype(np.uint8)
    as_float = image_u8.astype(np.float32)

    binomial = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.uint8)
    result = call("uint8 的图配 uint8 的核", func, image_u8, binomial)
    msg = bad_contract(result, image_u8.shape)
    if msg:
        return "FAIL", "uint8 的图配 uint8 的核时：" + msg
    # 只比内部，边界填充的问题留给前面几项去报
    err = float(np.abs(result - cv2.filter2D(as_float, cv2.CV_32F, binomial.astype(np.float32)))[1:-1, 1:-1].max())
    if err > 1e-3:
        return "FAIL", f"uint8 的图配 uint8 的核（和为 16 的平滑核）时，图像内部与 cv2.filter2D 最大差 {err:.4g}，正确结果在 2400 到 4080 之间；检查相乘和累加之前有没有先转成浮点"

    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int64)
    step = np.zeros((6, 8), dtype=np.uint8)
    step[:, :4] = 200                                             # 左亮右暗，Sobel x 的响应应为负
    result = call("uint8 的图配整数 Sobel 核", func, step, sobel)
    msg = bad_contract(result, step.shape)
    if msg:
        return "FAIL", "uint8 的图配整数 Sobel 核时：" + msg
    if not np.allclose(result[2, 3:5], -800, atol=1e-3):
        why = "符号反了：要么核被左右翻转了，要么输出取了绝对值" if np.allclose(result[2, 3:5], 800, atol=1e-3) else "输出要保留符号，中间结果不能落在无符号类型里"
        return "FAIL", f"左亮右暗的台阶（200 到 0）配 Sobel x，台阶处的响应应为 -800，得到 {result[2, 3:5].tolist()}；{why}"

    image = as_float.copy()
    kernel = np.array([[0.5, 1.0, -2.0], [0.0, 3.0, 1.0], [-1.0, 0.0, 2.0]], dtype=np.float32)
    image_backup, kernel_backup = image.copy(), kernel.copy()
    func(image, kernel)                                          # 这里故意不传副本
    if not np.array_equal(image, image_backup) or not np.array_equal(kernel, kernel_backup):
        return "FAIL", "函数修改了传进来的 image 或 kernel；输入是 float32 时 np.asarray 不会复制，输出要写到新数组里"
    return "PASS", "整数类型的输入先转浮点、保留符号、不修改输入"


def main() -> int:
    print("第 2 章进度检查\n" + "=" * 46)
    try:
        func = load_assignment().correlate2d
    except Exception as exc:  # noqa: BLE001
        report("加载作业", "FAIL", f"{type(exc).__name__}: {exc}")
        return 1
    results = [
        run("输出契约、恒等核与常量图", check_contract, func),
        run("核不翻转（冲激响应）", check_impulse, func),
        run("reflect 边界（手算）", check_border, func),
        run("非方形核与 OpenCV 对照", check_shapes, func),
        run("整数输入、符号与不修改输入", check_dtypes, func),
    ]
    print("=" * 46)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("自动验收通过；仍需完成人工实验与失败分析。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
