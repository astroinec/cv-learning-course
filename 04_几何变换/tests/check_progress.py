"""第 4 章自动验收：TODO / FAIL / PASS。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "4-1_齐次点变换.py"


def load_assignment():
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    spec = importlib.util.spec_from_file_location("chapter4_assignment", base / ASSIGNMENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(status: str, name: str, detail: str) -> None:
    print(f"[{status:4}] {name} — {detail}")


def run(name: str, check, func) -> bool:
    try:
        with np.errstate(all="ignore"):
            ok, detail = check(func)
    except NotImplementedError as exc:
        report("TODO", name, str(exc))
        return False
    except Exception as exc:  # noqa: BLE001
        report("FAIL", name, f"运行异常：{type(exc).__name__}: {exc}")
        return False
    report("PASS" if ok else "FAIL", name, detail)
    return ok


def is_contract(out, n: int) -> bool:
    return isinstance(out, np.ndarray) and out.shape == (n, 2) and out.dtype == np.float64


def check_contract(func):
    points = np.array([[0, 0], [2, 3], [-1, 4]], dtype=np.float32)
    out = func(points, np.eye(3))
    if not is_contract(out, 3):
        return False, "输出必须是 (N,2) float64：打印返回值的 shape 和 dtype"
    if not np.allclose(out, points):
        return False, "单位矩阵应保持点不变：检查齐次分量补在了哪一列"
    one = func(np.array([[7.0, -2.0]]), np.eye(3))
    if not is_contract(one, 1) or not np.allclose(one, [[7.0, -2.0]]):
        return False, "只有一个点时也应返回 (1,2) float64"
    return True, "输出的形状和类型正确，单位矩阵保持点不变"


def check_known_values(func):
    points = np.array([[0, 0], [2, 3], [-1, 4]], dtype=np.float64)
    shift = np.array([[1, 0, 5], [0, 1, -3], [0, 0, 1]], dtype=np.float64)
    moved = func(points, shift)
    if not is_contract(moved, 3):
        return False, "输出必须是 (N,2) float64"
    if not np.allclose(moved, [[5, -3], [7, 0], [4, 1]], atol=1e-9):
        return False, "纯平移矩阵（第三列是 5、-3、1）应把每个点移动 (5, -3)：检查点和矩阵相乘的方向，点是按行排的还是按列排的"
    # 手算的期望值：H 的三行是 (1,2,3)、(4,5,6)、(0.5,0,1)
    homography = np.array([[1, 2, 3], [4, 5, 6], [0.5, 0, 1]], dtype=np.float64)
    int_points = np.array([[2, 1], [0, 0], [-4, 2], [4, -1]], dtype=np.int64)
    out = func(int_points, homography)
    if not is_contract(out, 4):
        return False, "输入是整数数组时输出也必须是 (N,2) float64"
    expected = np.array([[3.5, 9.5], [3.0, 6.0], [-3.0, 0.0], [5.0 / 3.0, 17.0 / 3.0]])
    if not np.isfinite(out).all():
        return False, "这组点的第三个分量是 2、1、-1、3，都不为 0，不应出现 NaN 或 inf：第三个分量为负的点同样有效"
    if np.allclose(out, np.trunc(expected), atol=1e-9) and not np.allclose(out, expected, atol=1e-9):
        return False, "输入是整数数组时结果的小数部分丢了：检查中间结果存进了什么类型的数组"
    if not np.allclose(out, expected, atol=1e-3):
        return False, "点 (2,1) 经过三行为 (1,2,3)、(4,5,6)、(0.5,0,1) 的矩阵应得到 (3.5, 9.5)：用这个点手算一遍，逐步对照"
    if not np.allclose(out, expected, atol=1e-9, rtol=0):
        return False, "结果只有单精度的准确度：输入要先转成 float64 再计算，而不是最后才转"
    return True, "纯平移和一个手算的投影变换结果正确，整数输入不丢小数"


def check_scale(func):
    points = np.array([[0, 0], [2, 3], [-1, 4]], dtype=np.float64)
    homography = np.array([[1, 0.2, 5], [0.1, 2, -3], [0.01, -0.02, 1]], dtype=np.float64)
    base = func(points, homography)
    if not np.isfinite(base).all():
        return False, "这组点的第三个分量都在 0.9 以上，不应出现 NaN 或 inf"
    if not np.allclose(base, func(points, homography * 7.0), atol=1e-9):
        return False, "矩阵整体乘 7 之后结果变了：检查两个坐标是否都除以了第三个分量"
    flipped = func(points, homography * -1.0)
    if not np.isfinite(flipped).all():
        return False, "矩阵整体乘 -1 之后第三个分量变成负数，这些点仍然有效，不应返回 NaN"
    if not np.allclose(base, flipped, atol=1e-9):
        return False, "矩阵整体乘 -1 之后结果变了：齐次坐标整体乘一个非零常数表示同一个点"
    return True, "矩阵整体乘 7 或 -1 结果不变"


def check_invalid(func):
    plane = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
    at_infinity = func(np.array([[0.0, 0.0]]), plane)
    if not (is_contract(at_infinity, 1) and np.isnan(at_infinity).all()):
        return False, "第三个分量为 0 的点两个坐标都应是 NaN（不是 inf，也不是报错）"
    # 第三行是 (1,0,0)：第三个分量等于 x。h33 为 0 的矩阵同样合法
    swap = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)
    points = np.array([[2.0, 3.0], [0.0, 5.0], [-4.0, 2.0]])
    out = func(points, swap)
    if not is_contract(out, 3):
        return False, "输出必须是 (N,2) float64"
    if not np.isnan(out[1]).all():
        return False, "一批点里第三个分量为 0 的那个点应返回 NaN"
    if np.isnan(out[[0, 2]]).any():
        return False, "一批点里只有一个无效时，其余的点应照常计算；矩阵右下角为 0 也是合法的单应，不要先拿它去除整个矩阵"
    if not np.allclose(out[[0, 2]], [[0.5, 1.5], [-0.25, -0.5]], atol=1e-9):
        return False, "和无效点同批的有效点结果不对"
    tiny = func(np.array([[0.0, 0.0]]), np.diag([1.0, 1.0, 1e-13]))
    if not np.isnan(tiny).all():
        return False, "第三个分量是 1e-13 时也应返回 NaN：契约是绝对值不超过 1e-12，不是恰好等于 0"
    small = func(np.array([[2e-11, -1e-11]]), np.diag([1.0, 1.0, 1e-11]))
    if not np.allclose(small, [[2.0, -1.0]], atol=1e-6):
        return False, "第三个分量是 1e-11 时仍应正常计算：判无效的门限是 1e-12，不要自己另取一个更大的"
    negative = func(np.array([[0.0, 0.0]]), np.diag([1.0, 1.0, -1e-13]))
    if not np.isnan(negative).all():
        return False, "第三个分量是 -1e-13 时也应返回 NaN：比较的是绝对值"
    return True, "无效点返回 NaN，同批其余的点不受影响，门限两侧都正确"


def check_inputs_untouched(func):
    points = np.array([[1.0, 2.0], [3.0, -4.0], [0.0, 0.0]])
    homography = np.array([[2.0, 0.0, 1.0], [0.0, 4.0, 2.0], [0.5, 0.25, 4.0]])
    points_before, homography_before = points.copy(), homography.copy()
    func(points, homography)
    if not np.array_equal(points, points_before):
        return False, "调用之后传进来的 points 被改了：不要在输入数组上原地运算"
    if not np.array_equal(homography, homography_before):
        return False, "调用之后传进来的 homography 被改了：不要在输入数组上原地运算"
    return True, "没有修改传进来的数组"


CHECKS = [
    ("形状与类型", check_contract),
    ("已知结果", check_known_values),
    ("尺度不变", check_scale),
    ("无效点", check_invalid),
    ("不改输入", check_inputs_untouched),
]


def run_all(func) -> list[bool]:
    return [run(name, check, func) for name, check in CHECKS]


def main() -> int:
    print("第 4 章进度检查\n" + "=" * 52)
    try:
        func = load_assignment().transform_points
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "加载作业", f"{type(exc).__name__}: {exc}")
        return 1
    results = run_all(func)
    print("=" * 52)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("全部通过；点序错误和退化点集要在 Notebook 和实验记录里自己检查。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
