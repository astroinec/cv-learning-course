"""第 4 章最低限度自动验收；TODO 是正常初始状态。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
_IMPL = Path(os.environ["COURSE_IMPL_DIR"]) / CHAPTER.name if os.environ.get("COURSE_IMPL_DIR") else CHAPTER


def load_assignment():
    spec = importlib.util.spec_from_file_location("chapter4_assignment", _IMPL / "4-1_齐次点变换.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(status: str, detail: str) -> None:
    print(f"[{status:4}] 齐次点变换 — {detail}")


def check() -> bool:
    try:
        func = load_assignment().transform_points
        points = np.array([[0, 0], [2, 3], [-1, 4]], dtype=np.float32)
        identity = func(points, np.eye(3))
    except NotImplementedError as exc:
        report("TODO", str(exc))
        return False
    except Exception as exc:
        report("FAIL", f"运行异常：{exc}")
        return False

    if not isinstance(identity, np.ndarray) or identity.shape != (3, 2) or identity.dtype != np.float64:
        report("FAIL", "输出必须是 (N,2) float64")
        return False
    if not np.allclose(identity, points):
        report("FAIL", "单位单应矩阵应保持点不变")
        return False

    homography = np.array([[1, 0.2, 5], [0.1, 2, -3], [0.01, -0.02, 1]], dtype=np.float64)
    a = func(points, homography)
    b = func(points, homography * 7.0)
    if not np.allclose(a, b, equal_nan=True):
        report("FAIL", "单应矩阵整体缩放不应改变投影")
        return False

    at_infinity = func(np.array([[0.0, 0.0]]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float))
    if not np.isnan(at_infinity).all():
        report("FAIL", "齐次第三分量为 0 时应返回 NaN")
        return False
    report("PASS", "shape/dtype、尺度不变性和无穷远处理均满足")
    return True


def main() -> int:
    print("第 4 章进度检查\n" + "=" * 46)
    ok = check()
    print("=" * 46)
    if not ok:
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("基本不变量通过；仍需检查点序和退化配置。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
