"""第 23 章自动验收：TODO / FAIL / PASS。

期望的 die 个数由一个独立的脚本用有理数逐个 die 数出来，这里只写结果，不写判断完整 die 的算式。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "23-1_晶圆图.py"


def load_assignment():
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    spec = importlib.util.spec_from_file_location("chapter23_assignment", base / ASSIGNMENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(status, name, detail):
    print(f"[{status:4}] {name} — {detail}")


def run(name, func):
    try:
        ok, detail = func()
    except NotImplementedError as exc:
        report("TODO", name, str(exc))
        return False
    except Exception as exc:  # noqa: BLE001
        report("FAIL", name, f"运行异常：{type(exc).__name__}: {exc}")
        return False
    report("PASS" if ok else "FAIL", name, detail)
    return ok


def is_table(a, cols, dtype):
    return isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] == cols and a.dtype == dtype


def in_order(g):
    """按 row 升序、row 内 col 升序。"""
    return np.array_equal(np.lexsort((g[:, 0], g[:, 1])), np.arange(len(g)))


def check_grid(mod):
    g = mod.die_grid(300.0, 10.0, 10.0, edge_exclusion_mm=3.0, offset=(-5.0, -5.0))
    if not is_table(g, 4, np.float64):
        return False, "输出应为 (K, 4) float64 的数组"
    if len(g) != 621:
        return False, (f"300 mm 晶圆、10 mm die、3 mm 去边、offset=(-5, -5) 应有 621 个完整 die，得到 {len(g)}。"
                       "检查是不是四个角点都要在可用圆内，以及半径有没有扣掉去边")
    center = g[(g[:, 0] == 0) & (g[:, 1] == 0)]
    if len(center) != 1 or not np.allclose(center[0, 2:], [0.0, 0.0]):
        return False, "offset=(-5,-5) 时 (0,0) 号 die 的中心应在晶圆中心"
    if not in_order(g):
        return False, "应按 row 升序、row 内 col 升序排列"

    # 非正方形的 die，两个方向的偏移不同：数量、每一行的中心、排序
    w, h, off = 8.0, 12.0, (3.0, -5.0)
    g2 = mod.die_grid(200.0, w, h, 2.0, off)
    if not is_table(g2, 4, np.float64) or len(g2) != 272:
        return False, (f"200 mm 晶圆、8x12 mm 的 die、2 mm 去边、offset=(3, -5) 应有 272 个完整 die，得到 {len(g2)}。"
                       "检查 die_w 和 die_h、offset 的两个分量有没有用混")
    if not np.array_equal(g2[:, :2], np.round(g2[:, :2])):
        return False, "col 和 row 应该是整数值"
    left, bottom = off[0] + g2[:, 0] * w, off[1] + g2[:, 1] * h
    if not np.allclose(g2[:, 2], left + w / 2, atol=1e-9) or not np.allclose(g2[:, 3], bottom + h / 2, atol=1e-9):
        return False, "非正方形 die 时，有的行 center 和它的 (col, row) 对不上：center 应是左下角加半个 die_w、半个 die_h"
    if not in_order(g2):
        return False, "非正方形 die 时没有按 row 升序、row 内 col 升序排列"

    # 把 offset 平移 5 个 die 宽、-4 个 die 高：晶圆上的 die 不变，只有编号变
    g3 = mod.die_grid(200.0, w, h, 2.0, (off[0] + 5 * w, off[1] - 4 * h))
    if not is_table(g3, 4, np.float64) or len(g3) != len(g2) or not np.allclose(g3[:, 2:], g2[:, 2:], atol=1e-9):
        return False, ("offset 比一个 die 还大时结果变了：offset 平移整数个 die 后，完整 die 的个数和中心都不应该变。"
                       "检查搜索 col、row 的范围有没有把 offset 算进去")
    if not np.array_equal(g3[:, 0], g2[:, 0] - 5) or not np.array_equal(g3[:, 1], g2[:, 1] + 4):
        return False, "offset 平移整数个 die 后，col、row 的编号应该跟着反向变化同样的个数"

    # 角点正好落在可用圆上：(90, 120) 到圆心的距离正好是 150
    g4 = mod.die_grid(300.0, 10.0, 10.0, 0.0, (0.0, 0.0))
    if not is_table(g4, 4, np.float64) or len(g4) != 648:
        return False, (f"300 mm 晶圆、10 mm die、不去边、offset=(0, 0) 应有 648 个完整 die，得到 {len(g4)}。"
                       "这组参数下有的 die 角点正好落在圆上（例如角点 (90, 120)），对照文档字符串看这种 die 算不算")

    # 一个完整 die 都放不下
    g5 = mod.die_grid(20.0, 26.0, 33.0, 3.0, (0.0, 0.0))
    if not is_table(g5, 4, np.float64) or len(g5) != 0:
        return False, "20 mm 的晶圆放不下 26x33 mm 的 die，应返回形状 (0, 4) 的 float64 数组"
    return True, "完整 die 数量（含角点正好在圆上、大偏移、放不下的情况）、中心和排序都正确"


def check_assign(mod):
    pts = np.array([[0.0, 0.0], [9.999, 9.999], [10.0, 0.0], [-0.001, -0.001], [-15.0, 25.0]])
    keep = pts.copy()
    got = mod.assign_to_die(pts, 10.0, 10.0)
    want = np.array([[0, 0], [0, 0], [1, 0], [-1, -1], [-2, 2]])
    if not is_table(got, 2, np.int64) or not np.array_equal(got, want):
        return False, f"输出应为 (N, 2) int64；期望 {want.tolist()}，得到 {np.asarray(got).tolist()}"
    if not np.array_equal(pts, keep):
        return False, "传入的 points_xy 被改掉了，函数不能原地修改输入"
    shifted = mod.assign_to_die(np.array([[0.0, 0.0]]), 10.0, 10.0, offset=(-5.0, -5.0))
    if not np.array_equal(shifted, [[0, 0]]):
        return False, "offset 没有被正确扣除"

    # 非正方形的 die，两个方向的偏移不同；前两个点正好落在 die 的边界上
    pts2 = np.array([[6.0, 7.0], [-6.0, -7.0], [5.9, 6.9], [-6.5, -7.0], [40.0, -30.0], [-6.0, 20.9]])
    keep2 = pts2.copy()
    got2 = mod.assign_to_die(pts2, 12.0, 14.0, offset=(-6.0, -7.0))
    want2 = np.array([[1, 1], [0, 0], [0, 0], [-1, 0], [3, -2], [0, 1]])
    if not is_table(got2, 2, np.int64) or not np.array_equal(got2, want2):
        return False, (f"12x14 mm 的 die、offset=(-6, -7)：期望 {want2.tolist()}，得到 {np.asarray(got2).tolist()}。"
                       "检查 x 和 y 各自用的是不是自己的 die 尺寸和 offset 分量")
    if not np.array_equal(pts2, keep2):
        return False, "传入的 points_xy 被改掉了（offset 不为 0 时），函数不能原地修改输入"

    empty = mod.assign_to_die(np.zeros((0, 2)), 10.0, 10.0)
    if not is_table(empty, 2, np.int64) or len(empty) != 0:
        return False, "没有点时应返回形状 (0, 2) 的 int64 数组"
    return True, "边界归属、非正方形 die、偏移和空输入都正确"


def check_radial(mod):
    rng = np.random.default_rng(0)
    n = 200000
    r = 100.0 * np.sqrt(rng.random(n))
    th = rng.random(n) * 2 * np.pi
    pts = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    keep = pts.copy()
    dens = mod.radial_density(pts, 100.0, 5)
    want = n / (np.pi * 100.0 ** 2)
    if not isinstance(dens, np.ndarray) or dens.shape != (5,) or np.abs(dens / want - 1).max() > 0.05:
        return False, "面积均匀分布的点，各环密度应近似相等，都约等于总数除以圆面积"
    if not np.array_equal(pts, keep):
        return False, "传入的 points_xy 被改掉了，函数不能原地修改输入"

    # 手算：半径 80、4 个环，环的边界是 20、40、60、80；四个点的半径是 10、30、30、70，另有一个点在半径之外
    hand = np.array([[10.0, 0.0], [0.0, 30.0], [-30.0, 0.0], [0.0, -70.0], [0.0, 95.0]])
    got = np.asarray(mod.radial_density(hand, 80.0, 4), dtype=np.float64)
    want_hand = np.array([1 / 400.0, 2 / 1200.0, 0.0, 1 / 2800.0]) / np.pi
    if got.shape != (4,) or not np.allclose(got, want_hand, rtol=1e-9, atol=1e-15):
        return False, ("半径 80、4 个等宽环，点的半径为 10、30、30、70（另有一个在半径外）：各环应为个数 1、2、0、1 除以各自的环面积。"
                       f"期望 {want_hand.tolist()}，得到 {got.tolist()}")

    edge = mod.radial_density(np.array([[100.0, 0.0], [150.0, 0.0]]), 100.0, 4)
    if abs(edge[-1] - 1 / (np.pi * (100 ** 2 - 75 ** 2))) > 1e-12 or edge[:-1].any():
        return False, "半径恰为 radius 的点算最后一环，超出的点忽略"

    none = np.asarray(mod.radial_density(np.zeros((0, 2)), 100.0, 3), dtype=np.float64)
    if none.shape != (3,) or np.any(none != 0.0):
        return False, "没有点时应返回 bins 个 0，不能出现 nan"
    return True, "按面积归一化的径向密度、边界和空输入都正确"


def check_edge(mod):
    pts = np.array([[0.0, 0.0], [95.0, 0.0], [0.0, -98.0], [50.0, 50.0], [120.0, 0.0]])
    keep = pts.copy()
    v = mod.edge_ratio(pts, 100.0, 10.0)
    if not isinstance(v, float) or abs(v - 0.5) > 1e-12:
        return False, f"4 个半径内的点中有 2 个在外圈，应为 0.5，得到 {v!r}"
    if not np.array_equal(pts, keep):
        return False, "传入的 points_xy 被改掉了，函数不能原地修改输入"

    # 半径 128、外圈宽 32：外圈是 [96, 128]。两个点正好在内边界上，一个正好在外边界上，一个在半径之外
    border = np.array([[96.0, 0.0], [0.0, -96.0], [-128.0, 0.0], [0.0, 10.0], [50.0, 0.0], [0.0, 95.9], [0.0, 128.5]])
    v2 = mod.edge_ratio(border, 128.0, 32.0)
    if not isinstance(v2, float) or abs(v2 - 0.5) > 1e-12:
        return False, (f"半径 128、外圈宽 32，点的半径为 96、96、128、10、50、95.9、128.5：应为 3/6 = 0.5，得到 {v2!r}。"
                       "对照文档字符串，看距离正好等于外圈内边界或正好等于半径的点怎样算")

    if mod.edge_ratio(np.zeros((0, 2)), 100.0, 10.0) != 0.0:
        return False, "没有缺陷时应返回 0.0"
    outside = mod.edge_ratio(np.array([[200.0, 0.0], [0.0, -300.0]]), 100.0, 10.0)
    if not isinstance(outside, float) or outside != 0.0:
        return False, f"所有点都在半径之外时，半径内没有缺陷，应返回 0.0，得到 {outside!r}"
    return True, "边缘比例、两条边界和空输入都正确"


def main() -> int:
    print("第 23 章进度检查\n" + "=" * 52)
    try:
        mod = load_assignment()
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "加载作业", f"{type(exc).__name__}: {exc}")
        return 1
    results = [
        run("die 网格", lambda: check_grid(mod)),
        run("缺陷归属", lambda: check_assign(mod)),
        run("径向密度", lambda: check_radial(mod)),
        run("边缘比例", lambda: check_edge(mod)),
    ]
    print("=" * 52)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("晶圆几何通过；下一步在 Notebook 里画晶圆图，看几类典型分布模式。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
