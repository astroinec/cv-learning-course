"""第 1 章自动验收。

它只检查函数的基本性质，不提供实现答案，也不能替代人工理解。
运行：python tests/check_progress.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
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


def report(label: str, status: str, detail: str = ""):
    suffix = f" — {detail}" if detail else ""
    print(f"[{status:4}] {label}{suffix}")


def is_array(value, shape=None, dtype=None):
    if not isinstance(value, np.ndarray):
        return False
    if shape is not None and value.shape != shape:
        return False
    if dtype is not None and value.dtype != dtype:
        return False
    return True


def check_equalization():
    module = load_module("1-1_直方图均衡化.py", "chapter1_equalize")
    img = np.array([[20, 20, 20, 40], [40, 80, 80, 200]], dtype=np.uint8)
    try:
        result = module.histogram_equalization(img)
    except Exception as exc:
        report("1-1 直方图均衡化", "FAIL", f"运行异常：{exc}")
        return False
    if result is None:
        report("1-1 直方图均衡化", "TODO", "函数仍返回 None")
        return False
    if not is_array(result, img.shape, np.uint8):
        report("1-1 直方图均衡化", "FAIL", "输出应为同 shape 的 uint8 数组")
        return False
    reference = cv2.equalizeHist(img)
    max_error = int(np.abs(result.astype(int) - reference.astype(int)).max())
    if max_error > 2:
        report("1-1 直方图均衡化", "FAIL", f"与 OpenCV 基线最大误差 {max_error}，检查 CDF 最小非零值")
        return False
    if result.min() != 0 or result.max() != 255:
        report("1-1 直方图均衡化", "FAIL", "该测试图结果应覆盖 0 到 255")
        return False
    report("1-1 直方图均衡化", "PASS", f"最大误差 {max_error}")
    return True


def check_color_ops():
    module = load_module("1-2_通道分离与伪彩色.py", "chapter1_color")
    img = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    ok = True
    for channel, index in [("B", 0), ("G", 1), ("R", 2)]:
        try:
            result = module.remove_channel_effect(img, channel)
        except NotImplementedError as exc:
            report(f"1-2 去掉 {channel} 通道", "TODO", str(exc))
            ok = False
            continue
        except Exception as exc:
            report(f"1-2 去掉 {channel} 通道", "FAIL", f"运行异常：{exc}")
            ok = False
            continue
        if not is_array(result, img.shape, np.uint8):
            report(f"1-2 去掉 {channel} 通道", "TODO", "返回值尚未正确实现")
            ok = False
            continue
        expected = img.copy()
        expected[:, :, index] = 0
        if not np.array_equal(result, expected):
            report(f"1-2 去掉 {channel} 通道", "FAIL", "目标通道之外的数据不应改变")
            ok = False
        else:
            report(f"1-2 去掉 {channel} 通道", "PASS")

    try:
        bright = module.enhance_brightness_hsv(img, 1.5)
        if not is_array(bright, img.shape, np.uint8):
            report("1-2 HSV 亮度增强", "TODO", "返回值尚未正确实现")
            ok = False
        else:
            before_v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
            after_v = cv2.cvtColor(bright, cv2.COLOR_BGR2HSV)[:, :, 2]
            if np.any(after_v < before_v):
                report("1-2 HSV 亮度增强", "FAIL", "v_scale > 1 时亮度不应整体下降")
                ok = False
            else:
                report("1-2 HSV 亮度增强", "PASS")
    except NotImplementedError as exc:
        report("1-2 HSV 亮度增强", "TODO", str(exc))
        ok = False
    except Exception as exc:
        report("1-2 HSV 亮度增强", "FAIL", f"运行异常：{exc}")
        ok = False
    return ok


def check_noise_and_median():
    module = load_module("1-3_椒盐噪声与中值滤波.py", "chapter1_noise")
    np.random.seed(7)
    img = np.full((60, 80), 120, dtype=np.uint8)
    ok = True
    try:
        noisy = module.add_salt_pepper_noise(img, amount=0.10)
    except Exception as exc:
        report("1-3 椒盐噪声", "FAIL", f"运行异常：{exc}")
        return False
    if not is_array(noisy, img.shape, np.uint8) or np.array_equal(noisy, img):
        report("1-3 椒盐噪声", "TODO", "应返回同 shape 数组并改变约 10% 像素")
        ok = False
    else:
        changed = np.mean(noisy != img)
        values = set(np.unique(noisy[noisy != img]).tolist())
        if not 0.06 <= changed <= 0.14 or not values.issubset({0, 255}):
            report("1-3 椒盐噪声", "FAIL", f"改变比例 {changed:.3f}，噪声值应只含 0/255")
            ok = False
        else:
            report("1-3 椒盐噪声", "PASS", f"改变比例 {changed:.3f}")

    impulse = np.full((15, 15), 100, dtype=np.uint8)
    impulse[7, 7] = 255
    try:
        result = module.median_filter_manual(impulse, 3)
    except Exception as exc:
        report("1-3 手写中值滤波", "FAIL", f"运行异常：{exc}")
        return False
    if not is_array(result, impulse.shape, np.uint8) or np.all(result == 0):
        report("1-3 手写中值滤波", "TODO", "函数仍未完成")
        ok = False
    elif result[7, 7] != 100:
        report("1-3 手写中值滤波", "FAIL", "中心孤立盐噪点应被恢复为 100")
        ok = False
    else:
        inner = (slice(1, -1), slice(1, -1))
        reference = cv2.medianBlur(impulse, 3)
        max_error = int(np.abs(result[inner].astype(int) - reference[inner].astype(int)).max())
        if max_error > 0:
            report("1-3 手写中值滤波", "FAIL", f"内部区域与 OpenCV 最大误差 {max_error}")
            ok = False
        else:
            report("1-3 手写中值滤波", "PASS", "内部区域与 OpenCV 一致")
    return ok


def main():
    print("第 1 章进度检查\n" + "=" * 44)
    results = [check_equalization(), check_color_ops(), check_noise_and_median()]
    print("=" * 44)
    passed = sum(results)
    print(f"模块通过：{passed}/{len(results)}")
    if passed != len(results):
        print("提示：TODO 是正常学习状态。先自己实现，再重新运行检查器。")
        return 1
    print("基本自动验收通过。请继续完成实验记录和失败案例分析。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
