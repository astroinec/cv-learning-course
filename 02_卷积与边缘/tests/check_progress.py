"""第 2 章最低限度自动验收；TODO 是正常初始状态。"""

from __future__ import annotations

import importlib.util
import os
import sys
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


def report(status: str, detail: str) -> None:
    print(f"[{status:4}] 二维相关 — {detail}")


def check() -> bool:
    try:
        func = load_assignment().correlate2d
        image = np.arange(35, dtype=np.float32).reshape(5, 7)
        identity = np.zeros((3, 3), dtype=np.float32)
        identity[1, 1] = 1
        result = func(image, identity)
    except NotImplementedError as exc:
        report("TODO", str(exc))
        return False
    except Exception as exc:
        report("FAIL", f"运行异常：{exc}")
        return False

    if not isinstance(result, np.ndarray) or result.shape != image.shape or result.dtype != np.float32:
        report("FAIL", "输出必须是同 shape 的 float32 数组")
        return False
    if not np.allclose(result, image):
        report("FAIL", "恒等核应保持图像不变")
        return False

    constant = np.full((6, 8), 4, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.float32) / 9
    try:
        averaged = func(constant, kernel)
    except Exception as exc:
        report("FAIL", f"常量图边界测试异常：{exc}")
        return False
    if averaged.shape != constant.shape or averaged.dtype != np.float32:
        report("FAIL", "不同输入 dtype 时仍应保持 shape 并输出 float32")
        return False
    if not np.allclose(averaged, 4, atol=1e-5):
        report("FAIL", "reflect 边界下归一化核应保持常量图")
        return False
    report("PASS", "shape/dtype、恒等核和常量响应均满足")
    return True


def main() -> int:
    print("第 2 章进度检查\n" + "=" * 46)
    ok = check()
    print("=" * 46)
    if not ok:
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("基本不变量通过；仍需完成人工实验与失败分析。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
