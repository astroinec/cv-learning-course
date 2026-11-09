"""第 5 章最低限度自动验收；TODO 是正常初始状态。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
_IMPL = Path(os.environ["COURSE_IMPL_DIR"]) / CHAPTER.name if os.environ.get("COURSE_IMPL_DIR") else CHAPTER


def load_assignment():
    spec = importlib.util.spec_from_file_location("chapter5_assignment", _IMPL / "5-1_Harris响应.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(status: str, detail: str) -> None:
    print(f"[{status:4}] Harris 响应 — {detail}")


def check() -> bool:
    try:
        func = load_assignment().harris_response
        flat = np.full((48, 52), 127, dtype=np.uint8)
        flat_response = func(flat)
    except NotImplementedError as exc:
        report("TODO", str(exc))
        return False
    except Exception as exc:
        report("FAIL", f"运行异常：{exc}")
        return False

    if not isinstance(flat_response, np.ndarray) or flat_response.shape != flat.shape or flat_response.dtype != np.float32:
        report("FAIL", "输出必须是同 shape 的 float32 数组")
        return False
    if not np.isfinite(flat_response).all():
        report("FAIL", "响应不得含 NaN/Inf")
        return False
    if float(np.max(np.abs(flat_response))) > 1e-4:
        report("FAIL", "常量图梯度为零，Harris 响应应接近零")
        return False

    square = np.zeros((64, 64), dtype=np.uint8)
    square[18:46, 18:46] = 255
    response = func(square)
    if response.shape != square.shape or response.dtype != np.float32 or not np.isfinite(response).all():
        report("FAIL", "方块测试违反输出契约")
        return False
    corner_mask = np.zeros_like(square, dtype=bool)
    for y, x in [(18, 18), (18, 45), (45, 18), (45, 45)]:
        corner_mask[max(0, y-3):y+4, max(0, x-3):x+4] = True
    if float(response[corner_mask].max()) <= 0:
        report("FAIL", "清晰方块角点应有正响应")
        return False
    if float(response[corner_mask].max()) < 0.8 * float(response.max()):
        report("FAIL", "全局强响应应出现在方块角点附近")
        return False
    report("PASS", "shape/dtype、平坦区和角点响应均满足")
    return True


def main() -> int:
    print("第 5 章进度检查\n" + "=" * 46)
    ok = check()
    print("=" * 46)
    if not ok:
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("基本不变量通过；仍需分析阈值与非极大值抑制。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
