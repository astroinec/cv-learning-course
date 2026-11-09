"""第 18 章自动验收：TODO / FAIL / PASS。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "18-1_金模板比对.py"


def load_assignment():
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    spec = importlib.util.spec_from_file_location("chapter18_assignment", base / ASSIGNMENT)
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


def pattern(h=64, w=64):
    y, x = np.mgrid[0:h, 0:w]
    lines = ((x // 4) % 2) * 80.0
    pads = (((y // 16) % 2 == 0) & ((x // 16) % 2 == 0)) * 60.0
    return 40.0 + lines + pads


def check_golden(mod):
    rng = np.random.default_rng(0)
    base = pattern()
    stack = base + rng.normal(scale=3.0, size=(15, 64, 64))
    stack[0, 10, 10] += 500.0  # 一张良品里混入一个离群点
    median, sigma = mod.robust_golden(stack, min_sigma=1.0)
    if median.shape != (64, 64) or sigma.shape != (64, 64) or median.dtype != np.float64:
        return False, "输出必须是两个 (H, W) float64"
    if abs(median[10, 10] - base[10, 10]) > 4.0:
        return False, "中位数应对单张离群值不敏感"
    if abs(np.median(sigma) - 3.0) > 0.6:
        return False, f"噪声 3.0 时 sigma 中位数为 {np.median(sigma):.2f}，检查 1.4826 系数"
    flat = np.ones((5, 4, 4))
    _, s = mod.robust_golden(flat, min_sigma=2.0)
    if not np.allclose(s, 2.0):
        return False, "sigma 需要不小于 min_sigma"
    try:
        mod.robust_golden(flat[:2])
    except ValueError:
        return True, "中位数抗离群值，sigma 尺度正确并有下限"
    return False, "少于 3 张良品时应抛出 ValueError"


def check_labels(mod):
    mask = np.zeros((8, 10), dtype=bool)
    mask[1, 1] = mask[2, 2] = True          # 对角相连
    mask[1, 6:9] = True                      # 横条
    mask[5:7, 3:5] = True                    # 方块
    labels, n = mod.label_components(mask, connectivity=8)
    if labels.dtype != np.int32 or labels.shape != mask.shape:
        return False, "labels 必须是同形状 int32"
    if n != 3:
        return False, f"8 连通应有 3 个区域，得到 {n}"
    if labels[1, 1] != 1 or labels[1, 6] != 2 or labels[5, 3] != 3:
        return False, "编号应按光栅扫描顺序第一次出现的位置递增"
    if labels[2, 2] != labels[1, 1] or (labels[~mask] != 0).any():
        return False, "对角像素应同属一个区域，背景为 0"
    _, n4 = mod.label_components(mask, connectivity=4)
    if n4 != 4:
        return False, f"4 连通应有 4 个区域，得到 {n4}"
    spiral = np.zeros((7, 7), dtype=bool)
    spiral[0, :] = spiral[:, 6] = spiral[6, :] = spiral[2:, 0] = spiral[2, :5] = spiral[2:5, 4] = True
    if mod.label_components(spiral, 8)[1] != 1:
        return False, "螺旋形单一区域被拆开了，检查合并逻辑"
    try:
        mod.label_components(mask, connectivity=6)
    except ValueError:
        return True, "4/8 连通、编号顺序和复杂形状都正确"
    return False, "connectivity 非法时应抛出 ValueError"


def make_scene(seed):
    rng = np.random.default_rng(seed)
    base = pattern()
    goods = base + rng.normal(scale=3.0, size=(12, 64, 64))
    test = base + rng.normal(scale=3.0, size=(64, 64))
    truth = [(20.0, 30.0), (45.0, 12.0), (50.0, 50.0)]
    test[19:22, 29:32] += 60.0
    test[44:47, 11:14] -= 60.0
    test[49:52, 49:52] += 45.0
    return goods, test, truth


def check_detect(mod):
    goods, test, truth = make_scene(1)
    median, sigma = mod.robust_golden(goods)
    found = mod.detect_defects(test, median, sigma, k=6.0, min_area=3)
    if not isinstance(found, list) or not all(isinstance(d, dict) for d in found):
        return False, "返回值必须是 list[dict]"
    keys = {"bbox", "area", "centroid", "max_z"}
    if found and not keys.issubset(found[0]):
        return False, f"每个缺陷需包含字段 {sorted(keys)}"
    if len(found) != 3:
        return False, f"应检出 3 个缺陷，得到 {len(found)}"
    centroids = sorted(tuple(round(v) for v in d["centroid"]) for d in found)
    if centroids != [(20, 30), (45, 12), (50, 50)]:
        return False, f"质心应为 (20,30)(45,12)(50,50)，得到 {centroids}"
    first = min(found, key=lambda d: d["centroid"][0])
    if first["area"] != 9 or tuple(first["bbox"]) != (19, 29, 22, 32):
        return False, "3x3 缺陷的面积应为 9，bbox 应为 (19,29,22,32) 开区间"
    if not isinstance(first["max_z"], float) or first["max_z"] < 10:
        return False, "max_z 应为 float 且反映强缺陷"
    none = mod.detect_defects(goods[0], median, sigma, k=6.0, min_area=3)
    if len(none) != 0:
        return False, f"良品图上不应报缺陷，得到 {len(none)} 个"
    return True, "缺陷数量、位置、面积、bbox 正确，良品零误报"


def check_match(mod):
    pred = [(10.0, 10.0), (30.0, 31.0), (80.0, 80.0)]
    truth = [(10.5, 9.5), (30.0, 30.0), (60.0, 60.0), (5.0, 5.0)]
    r = mod.match_defects(pred, truth, radius=3.0)
    want = {"tp": 2, "fp": 1, "fn": 2}
    if any(r.get(k) != v for k, v in want.items()):
        return False, f"期望 {want}，得到 {r}"
    if abs(r.get("capture_rate", -1) - 0.5) > 1e-12:
        return False, "capture_rate 应为 tp / len(truth)"
    crowded = mod.match_defects([(0.0, 0.0), (0.0, 2.0)], [(0.0, 1.9)], radius=3.0)
    if crowded["tp"] != 1 or crowded["fp"] != 1:
        return False, "一个真值只能匹配一个预测"
    empty = mod.match_defects([(1.0, 1.0)], [], radius=3.0)
    if empty["capture_rate"] != 1.0 or empty["fp"] != 1:
        return False, "真值为空时 capture_rate 为 1.0，预测都算误报"
    return True, "一对一贪心匹配和统计正确"


def main() -> int:
    print("第 18 章进度检查\n" + "=" * 52)
    try:
        mod = load_assignment()
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "加载作业", f"{type(exc).__name__}: {exc}")
        return 1
    results = [
        run("鲁棒金模板", lambda: check_golden(mod)),
        run("连通域标记", lambda: check_labels(mod)),
        run("缺陷检测", lambda: check_detect(mod)),
        run("缺陷匹配统计", lambda: check_match(mod)),
    ]
    print("=" * 52)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("基本流程通过；还要在实验记录里画出 k 与过检/漏检的关系。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
