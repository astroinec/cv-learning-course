"""第 3 章最低限度自动验收；TODO 是正常初始状态。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]


def load_assignment():
    spec = importlib.util.spec_from_file_location("chapter3_assignment", CHAPTER / "3-1_手写二值形态学.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(label: str, status: str, detail: str) -> None:
    print(f"[{status:4}] {label} — {detail}")


def valid_binary(value: object, shape: tuple[int, int]) -> bool:
    return isinstance(value, np.ndarray) and value.shape == shape and value.dtype == np.uint8 and set(np.unique(value)).issubset({0, 255})


def check() -> bool:
    module = load_assignment()
    image = np.zeros((9, 9), dtype=np.uint8)
    image[3:6, 3:6] = 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    outputs = {}
    for label, func in [("腐蚀", module.erode_binary), ("膨胀", module.dilate_binary)]:
        try:
            outputs[label] = func(image, kernel)
        except NotImplementedError as exc:
            report(label, "TODO", str(exc))
        except Exception as exc:
            report(label, "FAIL", f"运行异常：{exc}")
    if len(outputs) != 2:
        return False

    eroded, dilated = outputs["腐蚀"], outputs["膨胀"]
    if not valid_binary(eroded, image.shape) or not valid_binary(dilated, image.shape):
        report("输出契约", "FAIL", "两者均须返回同 shape 的 uint8 二值图")
        return False
    if np.any((eroded > 0) & (image == 0)):
        report("腐蚀不变量", "FAIL", "腐蚀结果必须是原前景的子集")
        return False
    if np.any((image > 0) & (dilated == 0)):
        report("膨胀不变量", "FAIL", "膨胀结果必须包含原前景")
        return False
    if np.count_nonzero(eroded) != 1 or np.count_nonzero(dilated) != 25:
        report("3×3 方块", "FAIL", "3×3 全一核应把 3×3 方块变为 1 点/5×5 方块")
        return False

    corner = np.zeros((5, 5), dtype=np.uint8)
    corner[0, 0] = 255
    if np.count_nonzero(module.dilate_binary(corner, kernel)) != 4:
        report("边界", "FAIL", "图像外为背景时，角点膨胀应保留图内 2×2 区域")
        return False
    report("形态学", "PASS", "二值性、包含关系和边界案例均满足")
    return True


def main() -> int:
    print("第 3 章进度检查\n" + "=" * 46)
    ok = check()
    print("=" * 46)
    if not ok:
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("基本不变量通过；继续验证不同结构元素。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
