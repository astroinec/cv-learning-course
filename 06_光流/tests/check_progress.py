"""第 6 章最低限度自动验收；TODO 是正常初始状态。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]


def load_assignment():
    spec = importlib.util.spec_from_file_location("chapter6_assignment", CHAPTER / "6-1_前向反向筛选.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(status: str, detail: str) -> None:
    print(f"[{status:4}] 轨迹筛选 — {detail}")


def check() -> bool:
    prev = np.array([[[0, 0]], [[5, 5]], [[10, 10]], [[20, 20]]], dtype=np.float32)
    forward = prev + np.array([[[2, 1]]], dtype=np.float32)
    backward = prev.copy()
    backward[1, 0] += [0.6, 0.8]   # 误差恰为 1，应保留
    backward[2, 0] += [2.0, 0.0]   # 超阈值
    forward[3, 0, 0] = np.nan      # 非有限值
    status = np.array([[1], [1], [1], [1]], dtype=np.uint8)
    try:
        func = load_assignment().filter_tracks
        mask = func(prev, forward, backward, status, max_fb_error=1.0)
    except NotImplementedError as exc:
        report("TODO", str(exc))
        return False
    except Exception as exc:
        report("FAIL", f"运行异常：{exc}")
        return False

    if not isinstance(mask, np.ndarray) or mask.shape != (4,) or mask.dtype != np.bool_:
        report("FAIL", "输出必须是 (N,) bool 掩码")
        return False
    if not np.array_equal(mask, np.array([True, True, False, False])):
        report("FAIL", "应保留阈值内轨迹并拒绝超阈值/NaN 轨迹")
        return False

    status[0] = 0
    status_mask = func(prev, np.nan_to_num(forward), backward, status, 1.0)
    if bool(status_mask[0]):
        report("FAIL", "status=0 的轨迹必须拒绝")
        return False
    report("PASS", "shape/dtype、阈值边界、状态与有限值均满足")
    return True


def main() -> int:
    print("第 6 章进度检查\n" + "=" * 46)
    ok = check()
    print("=" * 46)
    if not ok:
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("基本不变量通过；仍需在真实跟踪中调阈值。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
