"""第 7 章最低限度自动验收；TODO 是正常初始状态。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
_IMPL = Path(os.environ["COURSE_IMPL_DIR"]) / CHAPTER.name if os.environ.get("COURSE_IMPL_DIR") else CHAPTER


def load_assignment():
    spec = importlib.util.spec_from_file_location("chapter7_assignment", _IMPL / "7-1_针孔投影.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(status: str, detail: str) -> None:
    print(f"[{status:4}] 针孔投影 — {detail}")


def check() -> bool:
    camera = np.array([[500, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
    rotation = np.eye(3)
    translation = np.zeros(3)
    points = np.array([[0, 0, 2], [1, 0, 2], [0, 1, 3], [0, 0, -1]], dtype=np.float64)
    try:
        func = load_assignment().project_points
        pixels = func(points, camera, rotation, translation)
    except NotImplementedError as exc:
        report("TODO", str(exc))
        return False
    except Exception as exc:
        report("FAIL", f"运行异常：{exc}")
        return False

    if not isinstance(pixels, np.ndarray) or pixels.shape != (4, 2) or pixels.dtype != np.float64:
        report("FAIL", "输出必须是 (N,2) float64")
        return False
    expected = np.array([[320, 240], [570, 240], [320, 440]], dtype=np.float64)
    if not np.allclose(pixels[:3], expected, atol=1e-9):
        report("FAIL", "主点、焦距或透视除法结果不正确")
        return False
    if not np.isnan(pixels[3]).all():
        report("FAIL", "相机后方点必须返回 NaN")
        return False

    ray = np.array([[1, 2, 4], [2, 4, 8]], dtype=np.float64)
    projected_ray = func(ray, camera, rotation, translation.reshape(3, 1))
    if not np.allclose(projected_ray[0], projected_ray[1]):
        report("FAIL", "同一相机射线上的不同深度应投到同一像素")
        return False
    report("PASS", "shape/dtype、主点、深度与无效点处理均满足")
    return True


def main() -> int:
    print("第 7 章进度检查\n" + "=" * 46)
    ok = check()
    print("=" * 46)
    if not ok:
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("基本不变量通过；仍需验证畸变与位姿估计。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
