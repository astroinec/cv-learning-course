"""第 1 章自动验收。

只检查作业文档字符串里写明的行为，不提供实现答案，也不能替代人工理解。
期望值来自三种独立的来源：手算的小例子、OpenCV 的库函数、统计性质。
运行：python tests/check_progress.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
_IMPL = Path(os.environ["COURSE_IMPL_DIR"]) / CHAPTER.name if os.environ.get("COURSE_IMPL_DIR") else CHAPTER


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _IMPL / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(label: str, status: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"[{status:4}] {label}{suffix}")


def run(label: str, func) -> bool:
    """运行一项检查。还没实现显示 TODO，结果不对或抛异常显示 FAIL，都不会让脚本崩溃。"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            status, detail = func()
    except NotImplementedError as exc:
        status, detail = "TODO", str(exc) or "还没有实现"
    except Exception as exc:  # noqa: BLE001
        status, detail = "FAIL", f"运行异常：{type(exc).__name__}: {exc}"
    report(label, status, detail)
    return status == "PASS"


def is_array(value, shape=None, dtype=None) -> bool:
    if not isinstance(value, np.ndarray):
        return False
    if shape is not None and value.shape != shape:
        return False
    if dtype is not None and value.dtype != dtype:
        return False
    return True


def max_diff(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.abs(a.astype(np.int64) - b.astype(np.int64)).max())


# ---------------------------------------------------------------- 1-1 直方图均衡化
def check_equalization():
    func = load_module("1-1_直方图均衡化.py", "chapter1_equalize").histogram_equalization

    # 手算：8 个灰度各出现 1 次，最暗的到 0，最亮的到 255，中间 6 个把 255 七等分后四舍五入
    small = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.uint8)
    expected = np.array([[0, 36, 73, 109], [146, 182, 219, 255]], dtype=np.uint8)
    backup = small.copy()
    result = func(small)
    if result is None:
        return "TODO", "函数仍返回 None"
    if not is_array(result, small.shape, np.uint8):
        return "FAIL", "输出应为同 shape 的 uint8 数组"
    if not np.array_equal(small, backup):
        return "FAIL", "函数修改了传进来的输入数组，文档字符串规定不修改输入"
    if result.min() != 0 or result.max() != 255:
        return "FAIL", f"8 个灰度各 1 个像素的图，结果应覆盖 0 到 255，得到 {int(result.min())} 到 {int(result.max())}；检查映射时有没有用上 CDF 的第一个非零值"
    err = max_diff(result, expected)
    if err == 1:
        return "FAIL", f"和手算结果 {expected.ravel().tolist()} 差 1 个灰度，得到 {result.ravel().tolist()}；检查查找表转成整数之前是怎样取整的"
    if err > 1:
        return "FAIL", f"和手算结果 {expected.ravel().tolist()} 最大差 {err}，得到 {result.ravel().tolist()}"

    # 库函数：和 cv2.equalizeHist 对照，允许差 1（恰好落在 .5 上时不同的取整约定）
    rng = np.random.default_rng(11)
    cases = {
        "各灰度像素数不等的小图": np.array([[20, 20, 20, 40], [40, 80, 80, 200]], dtype=np.uint8),
        "低对比度的 40×50 随机图": rng.integers(90, 131, (40, 50)).astype(np.uint8),
        "含 0 和 255 的 33×21 随机图": rng.integers(0, 256, (33, 21)).astype(np.uint8),
    }
    for name, img in cases.items():
        backup = img.copy()
        result = func(img)
        if not is_array(result, img.shape, np.uint8):
            return "FAIL", f"{name}：输出应为同 shape 的 uint8 数组"
        if not np.array_equal(img, backup):
            return "FAIL", f"{name}：函数修改了传进来的输入数组"
        err = max_diff(result, cv2.equalizeHist(img))
        if err > 1:
            return "FAIL", f"{name}：与 cv2.equalizeHist 最大误差 {err}；检查直方图是不是按 0 到 255 共 256 个灰度级统计的，以及 CDF 最小非零值的处理"

    # 文档字符串规定：全图只有一个灰度时输出和输入相同
    flat = np.full((6, 9), 77, dtype=np.uint8)
    try:
        result = func(flat.copy())
    except Exception as exc:  # noqa: BLE001
        return "FAIL", f"纯色图（全图灰度 77）上运行异常：{type(exc).__name__}: {exc}；这时映射公式的分母是 0，要单独处理"
    if not is_array(result, flat.shape, np.uint8):
        return "FAIL", "纯色图：输出应为同 shape 的 uint8 数组"
    if not np.array_equal(result, flat):
        return "FAIL", f"纯色图（全图灰度 77）应原样输出，得到的灰度是 {np.unique(result).tolist()[:5]}；这时映射公式的分母是 0，要单独处理"
    return "PASS", "手算例子、OpenCV 对照和纯色图都满足"


# ---------------------------------------------------------------- 1-2 通道置零
def check_remove_channel():
    func = load_module("1-2_通道分离与伪彩色.py", "chapter1_color").remove_channel_effect
    original = np.array([[[10, 20, 30], [40, 50, 60], [70, 80, 90]], [[15, 25, 35], [45, 55, 65], [75, 85, 95]]], dtype=np.uint8)
    todo = []
    for channel, index in [("B", 0), ("G", 1), ("R", 2)]:
        img = original.copy()
        try:
            result = func(img, channel)
        except NotImplementedError:
            todo.append(channel)
            continue
        if not is_array(result, original.shape, np.uint8):
            return "FAIL", f"去掉 {channel} 通道：应返回同 shape 的 uint8 数组"
        if not np.array_equal(img, original):
            return "FAIL", f"去掉 {channel} 通道：函数修改了传进来的原图，应先复制再改"
        expected = original.copy()
        expected[:, :, index] = 0
        if not np.array_equal(result, expected):
            return "FAIL", f"去掉 {channel} 通道：应只有下标 {index} 的通道变成 0，其余两个通道不变（OpenCV 的通道顺序是 B、G、R）"
    if todo:
        return "TODO", f"还没有实现的通道分支：{'、'.join(todo)}"
    return "PASS", "三个通道分支都正确，原图没有被修改"


# ---------------------------------------------------------------- 1-2 HSV 亮度增强
def hue_sat_value(img_bgr: np.ndarray):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.int64)
    return hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]


def check_brightness():
    func = load_module("1-2_通道分离与伪彩色.py", "chapter1_color").enhance_brightness_hsv
    # 第一行乘 1.5 后不超过 255，第二行的 V 都大于 170，乘 1.5 后会超过 255
    img = np.array([
        [[30, 60, 90], [60, 140, 60], [80, 80, 80], [90, 30, 120], [20, 90, 160]],
        [[40, 120, 200], [200, 100, 50], [150, 180, 30], [60, 60, 220], [240, 200, 180]],
    ], dtype=np.uint8)
    h0, s0, v0 = hue_sat_value(img)
    for scale in (1.5, 0.5):
        backup = img.copy()
        result = func(img, scale)
        if not is_array(result, img.shape, np.uint8):
            return "FAIL", "应返回同 shape 的 uint8 数组"
        if not np.array_equal(img, backup):
            return "FAIL", "函数修改了传进来的原图"
        h1, s1, v1 = hue_sat_value(result)
        target = np.minimum(255.0, v0 * scale)
        v_err = np.abs(v1 - target)
        if scale > 1 and np.array_equal(v1, v0):
            return "FAIL", f"v_scale={scale} 时输出的 V 通道和输入完全一样，亮度没有被改变"
        over = v0 * scale > 255
        if np.any(v_err[over] > 2):
            y, x = np.argwhere(over & (v_err > 2))[0]
            return "FAIL", (f"v_scale={scale}：像素 BGR={img[y, x].tolist()} 的 V 是 {int(v0[y, x])}，乘完超过 255，按文档字符串应截到 255，"
                            f"得到的 V 是 {int(v1[y, x])}；检查转回 uint8 之前有没有限制范围")
        if np.any(v_err > 2):
            y, x = np.argwhere(v_err > 2)[0]
            return "FAIL", (f"v_scale={scale}：像素 BGR={img[y, x].tolist()} 的 V 是 {int(v0[y, x])}，应变成约 {target[y, x]:.0f}，"
                            f"得到 {int(v1[y, x])}；检查乘的是不是 V 通道、用的是不是参数 v_scale")
        dh = np.abs(h1 - h0)
        dh = np.minimum(dh, 180 - dh)
        colored = (s0 > 40) & (v1 > 40)
        if np.any(dh[colored] > 3) or np.any(np.abs(s1 - s0)[colored] > 12):
            y, x = np.argwhere(colored & ((dh > 3) | (np.abs(s1 - s0) > 12)))[0]
            return "FAIL", (f"v_scale={scale}：像素 BGR={img[y, x].tolist()} 的色调和饱和度应保持 (H, S)=({int(h0[y, x])}, {int(s0[y, x])})，"
                            f"得到 ({int(h1[y, x])}, {int(s1[y, x])})；只有 V 通道应该被缩放")
    return "PASS", "V 按倍数缩放并在 255 处截断，色调和饱和度不变"


# ---------------------------------------------------------------- 1-3 椒盐噪声
def check_salt_pepper():
    func = load_module("1-3_椒盐噪声与中值滤波.py", "chapter1_noise").add_salt_pepper_noise
    np.random.seed(7)
    img = np.full((200, 200), 120, dtype=np.uint8)
    backup = img.copy()
    noisy = func(img, amount=0.5)
    if noisy is None or (is_array(noisy, img.shape, np.uint8) and np.array_equal(noisy, backup)):
        return "TODO", "加噪后的图和原图完全相同，函数还没有实现"
    if not is_array(noisy, img.shape, np.uint8):
        return "FAIL", "应返回同 shape 的 uint8 数组"
    if not np.array_equal(img, backup):
        return "FAIL", "函数修改了传进来的原图"
    for amount in (0.5, 0.1):
        noisy = func(img, amount=amount)
        if not is_array(noisy, img.shape, np.uint8):
            return "FAIL", f"amount={amount}：应返回同 shape 的 uint8 数组"
        values = set(np.unique(noisy[noisy != img]).tolist())
        if not values.issubset({0, 255}):
            return "FAIL", f"amount={amount}：被改动的像素只应变成 0 或 255，出现了 {sorted(values)[:6]}"
        pepper, salt = float(np.mean(noisy == 0)), float(np.mean(noisy == 255))
        tol = 0.02 if amount == 0.5 else 0.015
        if abs(pepper + salt - amount) > tol or abs(pepper - amount / 2) > tol or abs(salt - amount / 2) > tol:
            return "FAIL", (f"amount={amount}（200×200 的图）：椒占 {pepper:.3f}，盐占 {salt:.3f}，合计 {pepper + salt:.3f}；"
                            f"按文档字符串两者应各占 amount/2={amount / 2:.3f}、互不重叠、合计 {amount}。检查两种噪点的位置是怎样抽出来的，有没有互相覆盖")
    clean = func(img, amount=0.0)
    if not is_array(clean, img.shape, np.uint8) or not np.array_equal(clean, img):
        return "FAIL", "amount=0 时不应改动任何像素"
    return "PASS", "椒和盐各占 amount/2，合计 amount"


# ---------------------------------------------------------------- 1-3 手写中值滤波
def check_median():
    func = load_module("1-3_椒盐噪声与中值滤波.py", "chapter1_noise").median_filter_manual
    impulse = np.full((15, 15), 100, dtype=np.uint8)
    impulse[7, 7] = 255
    result = func(impulse.copy(), 3)
    if result is None or (is_array(result, impulse.shape) and not np.any(result)):
        return "TODO", "输出全是 0，函数还没有实现"
    if not is_array(result, impulse.shape, np.uint8):
        return "FAIL", "应返回同 shape 的 uint8 数组"
    if result[7, 7] != 100:
        return "FAIL", "平坦图中心的一个孤立盐点应被恢复成 100"

    rng = np.random.default_rng(5)
    img = rng.integers(60, 256, (13, 17)).astype(np.uint8)
    for k in (3, 5):
        backup = img.copy()
        result = func(img, k)
        if not is_array(result, img.shape, np.uint8):
            return "FAIL", f"kernel_size={k}：应返回同 shape 的 uint8 数组"
        if not np.array_equal(img, backup):
            return "FAIL", f"kernel_size={k}：函数修改了传进来的原图"
        diff = np.abs(result.astype(np.int64) - cv2.medianBlur(img, k).astype(np.int64))
        r = k // 2
        inner = int(diff[r:-r, r:-r].max())
        if inner > 0:
            return "FAIL", f"kernel_size={k}（13×17 的随机图）：图像内部与 cv2.medianBlur 最大差 {inner}；检查窗口大小是不是用了参数 kernel_size，以及行列下标"
        if int(diff.max()) > 0:
            return "FAIL", (f"kernel_size={k}（13×17 的随机图）：内部与 cv2.medianBlur 一致，最外 {r} 圈有 {int(np.count_nonzero(diff))} 个像素不同，最大差 {int(diff.max())}；"
                            "文档字符串规定最外一圈也要算，并规定了图像外的像素怎样取值")
    return "PASS", "3×3 和 5×5 都与 cv2.medianBlur 全图一致（包括最外一圈）"


def main() -> int:
    print("第 1 章进度检查\n" + "=" * 44)
    results = [
        run("1-1 直方图均衡化", check_equalization),
        run("1-2 通道置零", check_remove_channel),
        run("1-2 HSV 亮度增强", check_brightness),
        run("1-3 椒盐噪声", check_salt_pepper),
        run("1-3 手写中值滤波", check_median),
    ]
    print("=" * 44)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态。先自己实现，再重新运行检查器。")
        return 1
    print("基本自动验收通过。请继续完成实验记录和失败案例分析。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
