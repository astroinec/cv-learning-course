"""第 3 章自动验收；TODO 是正常初始状态。

只检查作业文档字符串里写明的行为：输出是 0/255 的 uint8、只看 kernel 的非零位置、
kernel 中心对准输出像素且不翻转、图像外是背景、不修改输入。
期望值来自手算的小例子和 OpenCV（cv2.erode、cv2.dilate，显式指定图像外为背景）。
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
SHAPE = (9, 11)
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


def load_assignment():
    spec = importlib.util.spec_from_file_location("chapter3_assignment", _IMPL / "3-1_手写二值形态学.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(label: str, status: str, detail: str) -> None:
    print(f"[{status:4}] {label} — {detail}")


def run(label: str, check, module) -> bool:
    """运行一项检查。还没实现显示 TODO，结果不对或抛异常显示 FAIL，都不会让脚本崩溃。"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            status, detail = check(module)
    except NotImplementedError as exc:
        status, detail = "TODO", str(exc) or "还没有实现"
    except Exception as exc:  # noqa: BLE001
        status, detail = "FAIL", f"运行异常：{type(exc).__name__}: {exc}"
    report(label, status, detail)
    return status == "PASS"


def valid_binary(value: object, shape: tuple[int, int]) -> bool:
    return isinstance(value, np.ndarray) and value.shape == shape and value.dtype == np.uint8 and set(np.unique(value).tolist()).issubset({0, 255})


def canvas(points, shape=SHAPE) -> np.ndarray:
    image = np.zeros(shape, dtype=np.uint8)
    for y, x in points:
        image[y, x] = 255
    return image


def pixels(image: np.ndarray) -> list[tuple[int, int]]:
    return [(int(y), int(x)) for y, x in np.argwhere(image > 0)]


def expect(func, image, kernel, want, what: str) -> str | None:
    """调用作业函数（传副本），和手算的前景像素表比较，返回出错说明；一致时返回 None。"""
    shape = image.shape
    try:
        result = func(image.copy(), kernel.copy())
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        return f"{what}时运行异常（{type(exc).__name__}: {exc}）"
    if not valid_binary(result, shape):
        return f"{what}：应返回同 shape、只含 0 和 255 的 uint8 数组"
    got = pixels(result)
    if got != sorted(want):
        return f"{what}：前景像素（行, 列）应为 {sorted(want)}，得到 {got if len(got) <= 12 else f'{len(got)} 个像素'}"
    return None


# ---------------------------------------------------------------- 1. 输出契约与 3×3 方块
def check_basic(module):
    image = np.zeros((9, 9), dtype=np.uint8)
    image[3:6, 3:6] = 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded, dilated = module.erode_binary(image.copy(), kernel.copy()), module.dilate_binary(image.copy(), kernel.copy())
    if not valid_binary(eroded, image.shape) or not valid_binary(dilated, image.shape):
        return "FAIL", "两个函数都要返回同 shape 的 uint8 二值图，前景是 255 而不是 1 或 True"
    if np.any((eroded > 0) & (image == 0)):
        return "FAIL", "腐蚀结果必须是原前景的子集"
    if np.any((image > 0) & (dilated == 0)):
        return "FAIL", "核的中心非零时，膨胀结果必须包含原前景"
    if np.count_nonzero(eroded) != 1 or np.count_nonzero(dilated) != 25:
        return "FAIL", f"3×3 全一核应把 3×3 方块腐蚀成 1 个点、膨胀成 5×5 方块，得到 {np.count_nonzero(eroded)} 和 {np.count_nonzero(dilated)} 个像素"
    corner = np.zeros((5, 5), dtype=np.uint8)
    corner[0, 0] = 255
    if np.count_nonzero(module.dilate_binary(corner.copy(), kernel.copy())) != 4:
        return "FAIL", "图像外为背景时，角上的一个点膨胀后应是图内的 2×2 区域"
    return "PASS", "二值性、包含关系和 3×3 方块都满足"


# ---------------------------------------------------------------- 2. 只看 kernel 的非零位置
def check_cross(module):
    point = canvas([(4, 5)])
    msg = expect(module.dilate_binary, point, CROSS, [(3, 5), (4, 4), (4, 5), (4, 6), (5, 5)], "一个点 (4, 5) 用 3×3 十字核膨胀")
    if msg:
        return "FAIL", msg + "；文档字符串规定只考察 kernel 的非零位置，十字核和全一核的结果不应相同"
    plus = canvas([(3, 5), (4, 4), (4, 5), (4, 6), (5, 5)])
    msg = expect(module.erode_binary, plus, CROSS, [(4, 5)], "十字形的 5 个像素用 3×3 十字核腐蚀")
    if msg:
        return "FAIL", msg + "；十字核正好放得进这个十字，kernel 为 0 的四个角不参与判断"
    return "PASS", "十字核只用到 5 个非零位置"


# ---------------------------------------------------------------- 3. 非方形的核
def check_rect(module):
    wide, tall = np.ones((1, 5), dtype=np.uint8), np.ones((5, 1), dtype=np.uint8)
    point = canvas([(4, 5)])
    msg = expect(module.dilate_binary, point, wide, [(4, x) for x in range(3, 8)], "一个点 (4, 5) 用形状 (1, 5) 的核膨胀")
    if msg:
        return "FAIL", msg + "；形状 (1, 5) 是 1 行 5 列的横条，检查核的行、列和图像的行、列有没有对应反，以及半径为 0 的方向"
    msg = expect(module.dilate_binary, point, tall, [(y, 5) for y in range(2, 7)], "一个点 (4, 5) 用形状 (5, 1) 的核膨胀")
    if msg:
        return "FAIL", msg + "；形状 (5, 1) 是 5 行 1 列的竖条"
    bar = np.zeros(SHAPE, dtype=np.uint8)
    bar[3:6, 2:9] = 255                                  # 3 行 7 列的横条
    msg = expect(module.erode_binary, bar, wide, [(y, x) for y in range(3, 6) for x in range(4, 7)], "3 行 7 列的横条（第 3 到 5 行、第 2 到 8 列）用形状 (1, 5) 的核腐蚀")
    if msg:
        return "FAIL", msg
    msg = expect(module.erode_binary, bar, tall, [], "同一个横条用形状 (5, 1) 的核腐蚀（竖条放不进 3 行高的横条）")
    if msg:
        return "FAIL", msg
    return "PASS", "1×5 和 5×1 的核方向正确"


# ---------------------------------------------------------------- 4. 不对称的核：中心对准输出像素，不翻转
def check_asymmetric(module):
    right = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]], dtype=np.uint8)        # 中心和中心右边
    up_right = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)     # 中心和右上角
    point = canvas([(4, 5)])
    tip = "；文档字符串规定 kernel[i, j] 对应输入像素 (y + i - Kh//2, x + j - Kw//2)，对照讲义 3.3 节“盖章”和“命中”两种定义"
    msg = expect(module.dilate_binary, point, right, [(4, 4), (4, 5)], "一个点 (4, 5) 用“中心加右邻”的核膨胀")
    if msg:
        return "FAIL", msg + tip
    msg = expect(module.dilate_binary, point, up_right, [(4, 5), (5, 4)], "一个点 (4, 5) 用“中心加右上角”的核膨胀")
    if msg:
        return "FAIL", msg + tip
    run_h = canvas([(4, x) for x in range(2, 7)])
    msg = expect(module.erode_binary, run_h, right, [(4, x) for x in range(2, 6)], "第 4 行第 2 到 6 列的横线用“中心加右邻”的核腐蚀")
    if msg:
        return "FAIL", msg + tip
    return "PASS", "不对称核的方向和文档字符串一致"


# ---------------------------------------------------------------- 5. 贴边的腐蚀：图像外是背景
def check_border(module):
    kernel = np.ones((3, 3), dtype=np.uint8)
    image = np.zeros((16, 20), dtype=np.uint8)
    image[:10, :10] = 255                               # 贴着左上角的 10×10 方块
    eroded = module.erode_binary(image.copy(), kernel.copy())
    if not valid_binary(eroded, image.shape):
        return "FAIL", "应返回同 shape 的 uint8 二值图"
    want = np.zeros_like(image)
    want[1:9, 1:9] = 255                                # 手算：四条边各缩 1 个像素，贴着图像边界的两条边也要缩
    if not np.array_equal(eroded, want):
        n = int(np.count_nonzero(eroded))
        why = "81 个像素是 cv2.erode 默认参数的结果，它把图像外当成前景；复制边界像素补边也会得到同样的结果。" if n == 81 else ""
        return "FAIL", f"贴着左上角的 10×10 方块做 3×3 腐蚀应剩第 1 到 8 行、第 1 到 8 列共 64 个像素，得到 {n} 个。{why}文档字符串规定图像外视为背景，贴边的前景要从边界往里缩"
    full = np.full((6, 8), 255, dtype=np.uint8)
    eroded = module.erode_binary(full.copy(), kernel.copy())
    if not valid_binary(eroded, full.shape) or np.count_nonzero(eroded) != 24 or np.count_nonzero(eroded[1:-1, 1:-1]) != 24:
        return "FAIL", f"6×8 的整幅前景做 3×3 腐蚀应只剩内部 4×6 共 24 个像素，得到 {int(np.count_nonzero(eroded))} 个；图像外视为背景"
    return "PASS", "贴边的前景按“图像外是背景”腐蚀"


# ---------------------------------------------------------------- 6. kernel 的取值类型、不修改输入
def check_inputs(module):
    point = canvas([(4, 5)])
    want = [(3, 5), (4, 4), (4, 5), (4, 6), (5, 5)]
    for kernel, name in [(CROSS.astype(bool), "bool 类型的十字核"), (CROSS * 255, "非零元素是 255 的十字核")]:
        msg = expect(module.dilate_binary, point, kernel, want, f"一个点 (4, 5) 用{name}膨胀")
        if msg:
            return "FAIL", msg + "；文档字符串规定 kernel 可以是 uint8 或 bool，凡是非零的位置都算"
    image = np.zeros(SHAPE, dtype=np.uint8)
    image[2:7, 3:9] = 255
    for func, name in [(module.erode_binary, "erode_binary"), (module.dilate_binary, "dilate_binary")]:
        work, kernel = image.copy(), CROSS.copy()
        func(work, kernel)                                # 这里故意不再传副本
        if not np.array_equal(work, image) or not np.array_equal(kernel, CROSS):
            return "FAIL", f"{name} 修改了传进来的 image 或 kernel；结果要写到新数组里，边算边写回输入还会让后面的像素读到改过的值"
    return "PASS", "bool 核、255 核都按非零处理，输入没有被修改"


# ---------------------------------------------------------------- 7. 随机图：和 OpenCV 对照
def check_opencv(module):
    try:
        import cv2
    except ImportError:
        return "FAIL", "这一项要用 OpenCV 做对照，请先按 docs/环境安装.md 装好 opencv-python"
    rng = np.random.default_rng(31)
    image = ((rng.random((12, 15)) < 0.6) * 255).astype(np.uint8)
    lopsided = (rng.random((3, 5)) < 0.5).astype(np.uint8)
    lopsided[1, 2], lopsided[0, 4], lopsided[2, 0] = 1, 1, 0
    kernels = {
        "3×3 全一核": np.ones((3, 3), dtype=np.uint8),
        "5×5 十字核": cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
        "3 行 5 列的不对称核": lopsided,
    }
    border = dict(borderType=cv2.BORDER_CONSTANT, borderValue=0)      # 和作业约定一致：图像外是背景
    for name, kernel in kernels.items():
        for func, ref, op in [(module.erode_binary, cv2.erode, "腐蚀"), (module.dilate_binary, cv2.dilate, "膨胀")]:
            try:
                result = func(image.copy(), kernel.copy())
            except NotImplementedError:
                raise
            except Exception as exc:  # noqa: BLE001
                return "FAIL", f"12×15 的随机图用{name}{op}时运行异常（{type(exc).__name__}: {exc}）；先看前面几项手算例子的提示"
            if not valid_binary(result, image.shape):
                return "FAIL", f"{name}{op}：应返回同 shape 的 uint8 二值图"
            n = int(np.count_nonzero(result != ref(image, kernel, **border)))
            if n:
                return "FAIL", f"12×15 的随机图用{name}{op}，有 {n} 个像素和 cv2.{'erode' if op == '腐蚀' else 'dilate'}（borderValue=0）不同；先看前面几项手算例子的提示"
    return "PASS", "三种核的腐蚀和膨胀都与 OpenCV（图像外为背景）逐像素相同"


def main() -> int:
    print("第 3 章进度检查\n" + "=" * 46)
    try:
        module = load_assignment()
    except Exception as exc:  # noqa: BLE001
        report("加载作业", "FAIL", f"{type(exc).__name__}: {exc}")
        return 1
    results = [
        run("输出契约与 3×3 方块", check_basic, module),
        run("十字核（只看非零位置）", check_cross, module),
        run("非方形核（1×5 与 5×1）", check_rect, module),
        run("不对称核的方向", check_asymmetric, module),
        run("贴边的腐蚀", check_border, module),
        run("核的取值类型与不修改输入", check_inputs, module),
        run("随机图与 OpenCV 对照", check_opencv, module),
    ]
    print("=" * 46)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("自动验收通过；继续在 Notebook 里验证不同结构元素的效果。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
