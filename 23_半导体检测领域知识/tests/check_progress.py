"""第 23 章自动验收：TODO / FAIL / PASS。"""

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


def brute_count(diameter, w, h, excl, off):
    R = diameter / 2 - excl
    n = 0
    for col in range(-200, 200):
        for row in range(-200, 200):
            x0, y0 = off[0] + col * w, off[1] + row * h
            corners = [(x0, y0), (x0 + w, y0), (x0, y0 + h), (x0 + w, y0 + h)]
            if all(cx * cx + cy * cy <= R * R for cx, cy in corners):
                n += 1
    return n


def check_grid(mod):
    g = mod.die_grid(300.0, 10.0, 10.0, edge_exclusion_mm=3.0, offset=(-5.0, -5.0))
    if g.ndim != 2 or g.shape[1] != 4 or g.dtype != np.float64:
        return False, "输出应为 (K, 4) float64"
    want = brute_count(300.0, 10.0, 10.0, 3.0, (-5.0, -5.0))
    if len(g) != want:
        return False, f"300 mm 晶圆、10 mm die、3 mm 去边应有 {want} 个完整 die，得到 {len(g)}"
    center = g[(g[:, 0] == 0) & (g[:, 1] == 0)]
    if len(center) != 1 or not np.allclose(center[0, 2:], [0.0, 0.0]):
        return False, "offset=(-5,-5) 时 (0,0) 号 die 的中心应在晶圆中心"
    order = np.lexsort((g[:, 0], g[:, 1]))
    if not np.array_equal(order, np.arange(len(g))):
        return False, "应按 row 升序、row 内 col 升序排列"
    g2 = mod.die_grid(200.0, 8.0, 12.0, 2.0, (0.0, 0.0))
    if len(g2) != brute_count(200.0, 8.0, 12.0, 2.0, (0.0, 0.0)):
        return False, "非正方形 die、不同偏移时数量不对"
    return True, f"完整 die 数量 {want}，坐标和排序正确"


def check_assign(mod):
    pts = np.array([[0.0, 0.0], [9.999, 9.999], [10.0, 0.0], [-0.001, -0.001], [-15.0, 25.0]])
    got = mod.assign_to_die(pts, 10.0, 10.0)
    want = np.array([[0, 0], [0, 0], [1, 0], [-1, -1], [-2, 2]])
    if got.dtype != np.int64 or not np.array_equal(got, want):
        return False, f"期望 {want.tolist()}，得到 {got.tolist()}"
    shifted = mod.assign_to_die(np.array([[0.0, 0.0]]), 10.0, 10.0, offset=(-5.0, -5.0))
    if not np.array_equal(shifted, [[0, 0]]):
        return False, "offset 没有被正确扣除"
    return True, "边界归属和偏移处理正确"


def check_radial(mod):
    rng = np.random.default_rng(0)
    n = 200000
    r = 100.0 * np.sqrt(rng.random(n))
    th = rng.random(n) * 2 * np.pi
    pts = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    dens = mod.radial_density(pts, 100.0, 5)
    want = n / (np.pi * 100.0 ** 2)
    if dens.shape != (5,) or np.abs(dens / want - 1).max() > 0.05:
        return False, "面积均匀分布的点，各环密度应近似相等"
    edge = mod.radial_density(np.array([[100.0, 0.0], [150.0, 0.0]]), 100.0, 4)
    if abs(edge[-1] - 1 / (np.pi * (100 ** 2 - 75 ** 2))) > 1e-12 or edge[:-1].any():
        return False, "半径恰为 radius 的点算最后一环，超出的点忽略"
    return True, "按面积归一化的径向密度正确"


def check_edge(mod):
    pts = np.array([[0.0, 0.0], [95.0, 0.0], [0.0, -98.0], [50.0, 50.0], [120.0, 0.0]])
    v = mod.edge_ratio(pts, 100.0, 10.0)
    if not isinstance(v, float) or abs(v - 0.5) > 1e-12:
        return False, f"4 个半径内的点中有 2 个在外圈，应为 0.5，得到 {v!r}"
    if mod.edge_ratio(np.zeros((0, 2)), 100.0, 10.0) != 0.0:
        return False, "没有缺陷时应返回 0.0"
    return True, "边缘比例和空输入正确"


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
