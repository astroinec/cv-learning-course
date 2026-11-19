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


def close(got, want, tol) -> bool:
    """逐元素比较；形状对不上、出现 nan 或 inf 一律算不相等。"""
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        return False
    return bool(np.all(np.abs(got - want) <= tol))


def is_f64(arr, shape) -> bool:
    return isinstance(arr, np.ndarray) and arr.shape == shape and arr.dtype == np.float64


def pattern(h=64, w=64):
    y, x = np.mgrid[0:h, 0:w]
    lines = ((x // 4) % 2) * 80.0
    pads = (((y // 16) % 2 == 0) & ((x // 16) % 2 == 0)) * 60.0
    return 40.0 + lines + pads


def check_golden(mod):
    f = mod.robust_golden
    # 手算：5 张 1x3 的图，三个像素各是一组数
    #   像素 0：0 1 2 3 100 -> 中位数 2，偏差 2 1 0 1 98，MAD 1
    #   像素 1：10 10 10 10 10 -> 中位数 10，MAD 0
    #   像素 2：5 1 9 3 7 -> 中位数 5，偏差 0 4 4 2 2，MAD 2
    small = np.array([[0, 10, 5], [1, 10, 1], [2, 10, 9], [3, 10, 3], [100, 10, 7]], dtype=np.uint8).reshape(5, 1, 3)
    keep = small.copy()
    median, sigma = f(small, min_sigma=0.5)
    if not np.array_equal(small, keep):
        return False, "函数修改了传入的数组，需要改动时先复制一份"
    if not is_f64(median, (1, 3)) or not is_f64(sigma, (1, 3)):
        return False, "输出必须是两个 (H, W) float64 数组"
    if not close(median, [[2, 10, 5]], 1e-9):
        return False, f"三个像素的 5 个取值分别是 (0,1,2,3,100)、(10,10,10,10,10)、(5,1,9,3,7)，中位数应为 2、10、5，得到 {median.ravel().tolist()}"
    if not close(sigma, [[1.4826, 0.5, 2.9652]], 1e-9):
        return False, (f"同一组数据的 MAD 是 1、0、2，min_sigma=0.5 时 sigma 应为 1.4826、0.5、2.9652，得到 {np.round(sigma.ravel(), 4).tolist()}。"
                       "检查偏差是不是以中位数为中心、取的是不是中位数、系数和下限的先后顺序，以及是不是每个像素各算各的")
    _, sigma_default = f(small)
    if not close(sigma_default, [[1.4826, 1.0, 2.9652]], 1e-9):
        return False, "不传 min_sigma 时下限应为 1.0"
    # 手算：偶数张（4 张），取值 1 2 4 9 -> 中位数 3，偏差 2 1 1 6，MAD 1.5
    median4, sigma4 = f(np.array([1.0, 2.0, 4.0, 9.0]).reshape(4, 1, 1), min_sigma=0.1)
    if not close(median4, [[3.0]], 1e-9) or not close(sigma4, [[2.2239]], 1e-9):
        return False, f"4 张图取值 1、2、4、9 时中位数应为 3、sigma 应为 2.2239，得到 {float(median4.ravel()[0]):.4f} 和 {float(sigma4.ravel()[0]):.4f}。张数为偶数时中位数取中间两个数的平均"
    # 恰好 3 张是合法输入：1 5 2 -> 中位数 2，偏差 1 3 0，MAD 1
    median3, sigma3 = f(np.array([1.0, 5.0, 2.0]).reshape(3, 1, 1), min_sigma=0.1)
    if not close(median3, [[2.0]], 1e-9) or not close(sigma3, [[1.4826]], 1e-9):
        return False, "恰好 3 张图（取值 1、5、2）时中位数应为 2、sigma 应为 1.4826"

    rng = np.random.default_rng(0)
    base = pattern()
    stack = base + rng.normal(scale=3.0, size=(15, 64, 64))
    stack[0, 10, 10] += 500.0  # 一张良品里混入一个离群点
    stack_keep = stack.copy()
    median, sigma = f(stack, min_sigma=1.0)
    if not np.array_equal(stack, stack_keep):
        return False, "函数修改了传入的 float64 数组（例如原地排序），需要改动时先复制一份"
    if not is_f64(median, (64, 64)) or not is_f64(sigma, (64, 64)):
        return False, "输出必须是两个 (H, W) float64"
    if not abs(median[10, 10] - base[10, 10]) <= 4.0:
        return False, "中位数应对单张离群值不敏感"
    if not sigma[10, 10] <= 12.0:
        return False, f"15 张里只有 1 张在这个像素上是离群值，sigma 不应被它撑大，得到 {sigma[10, 10]:.1f}（噪声标准差是 3）"
    if not abs(np.median(sigma) - 3.0) <= 0.6:
        return False, f"噪声 3.0 时 sigma 中位数为 {np.median(sigma):.2f}，检查 1.4826 系数"
    flat = np.ones((5, 4, 4))
    _, s = f(flat, min_sigma=2.0)
    if not close(s, np.full((4, 4), 2.0), 1e-12):
        return False, "完全相同的几张图 MAD 为 0，sigma 应正好等于 min_sigma"
    try:
        f(flat[:2])
    except ValueError:
        return True, "手算用例、偶数张、离群值、sigma 尺度和下限都正确"
    return False, "少于 3 张良品时应抛出 ValueError"


def grid(rows):
    return np.array([[c == "#" for c in row] for row in rows])


def digits(rows):
    return np.array([[0 if c == "." else int(c) for c in row] for row in rows], dtype=np.int32)


def labels_problem(labels, n, mask):
    if not isinstance(labels, np.ndarray) or labels.dtype != np.int32 or labels.shape != mask.shape:
        return "labels 必须是和输入同形状的 int32 数组"
    if (labels[~mask] != 0).any() or (labels[mask] == 0).any():
        return "背景必须是 0，前景必须都有编号"
    if sorted(set(labels.ravel().tolist())) != list(range(0 if (~mask).any() else 1, int(n) + 1)):
        return f"labels 里出现的值应恰好是 0 到 n 的连续整数，n = {n}，实际出现的是 {sorted(set(labels.ravel().tolist()))[:8]}"
    return None


def check_labels(mod):
    f = mod.label_components
    mask = np.zeros((8, 10), dtype=bool)
    mask[1, 1] = mask[2, 2] = True          # 对角相连
    mask[1, 6:9] = True                      # 横条
    mask[5:7, 3:5] = True                    # 方块
    mask_keep = mask.copy()
    labels, n = f(mask, connectivity=8)
    if not np.array_equal(mask, mask_keep):
        return False, "函数修改了传入的掩膜，需要改动时先复制一份"
    problem = labels_problem(labels, n, mask)
    if problem:
        return False, problem
    if n != 3:
        return False, f"8 连通应有 3 个区域，得到 {n}"
    if labels[1, 1] != 1 or labels[1, 6] != 2 or labels[5, 3] != 3:
        return False, "编号应按光栅扫描顺序第一次出现的位置递增"
    if labels[2, 2] != labels[1, 1]:
        return False, "8 连通时对角相邻的像素应同属一个区域"
    labels4, n4 = f(mask, connectivity=4)
    if n4 != 4 or labels4[2, 2] == labels4[1, 1]:
        return False, f"4 连通应有 4 个区域（对角相邻的两个像素分开），得到 {n4}"
    as_int = mask.astype(np.uint8)
    int_keep = as_int.copy()
    labels_int, n_int = f(as_int, connectivity=8)
    if not np.array_equal(as_int, int_keep):
        return False, "函数修改了传入的掩膜，需要改动时先复制一份"
    if n_int != 3 or not np.array_equal(labels_int, labels):
        return False, "输入是 0/1 的整数数组时，结果应和 bool 数组相同"

    # 贴着四条边各放一个像素：左右两边、上下两边互不相邻
    edge = np.zeros((6, 7), dtype=bool)
    edge[0, 3] = edge[2, 0] = edge[2, 6] = edge[5, 3] = True
    for conn in (8, 4):
        got, n_e = f(edge, connectivity=conn)
        if n_e != 4 or [int(got[0, 3]), int(got[2, 0]), int(got[2, 6]), int(got[5, 3])] != [1, 2, 3, 4]:
            return False, f"四条边上各有一个孤立像素时应有 4 个区域，得到 {n_e}。检查邻点下标越界的判断，负下标在 NumPy 里会绕到另一侧"

    # 两个 U 形区域，第一个 U 形中间夹着一个孤立点，最后还有一个孤立点
    rows = ["#.#.#.#.#",
            "#...#.#.#",
            "#####.###",
            ".........",
            "#........"]
    want = digits(["1.2.1.3.3",
                   "1...1.3.3",
                   "11111.333",
                   ".........",
                   "4........"])
    for conn in (8, 4):
        got, n_u = f(grid(rows), connectivity=conn)
        problem = labels_problem(got, n_u, grid(rows))
        if problem:
            return False, f"U 形用例（{conn} 连通）：{problem}。两遍扫描合并之后要重新编号"
        if n_u != 4 or not np.array_equal(got, want):
            return False, (f"U 形用例（{conn} 连通）：第一行从左到右应是 1、2、1、3、3 号，得到 {got[0, ::2].tolist()}。"
                           "U 形的两条竖线在底部才连上，合并之后要按每个区域第一个像素的光栅顺序重新编号")

    spiral = np.zeros((7, 7), dtype=bool)
    spiral[0, :] = spiral[:, 6] = spiral[6, :] = spiral[2:, 0] = spiral[2, :5] = spiral[2:5, 4] = True
    if f(spiral, 8)[1] != 1:
        return False, "螺旋形单一区域被拆开了，检查合并逻辑"
    empty, n0 = f(np.zeros((4, 5), dtype=bool), 8)
    if n0 != 0 or not isinstance(empty, np.ndarray) or empty.shape != (4, 5) or empty.dtype != np.int32 or empty.any():
        return False, "没有前景时应返回全 0 的 int32 数组和 n = 0"
    # 固定用 Python 默认的递归深度上限来测，结果不随运行环境（例如 Jupyter 会把上限调高）变化
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(1000)
    try:
        full, n1 = f(np.ones((4, 2500), dtype=bool), 8)
    except RecursionError:
        return False, "4x2500 全是前景的长条上超过了递归深度上限。大连通域不能用递归，改用队列或并查集"
    finally:
        sys.setrecursionlimit(limit)
    if n1 != 1 or not (full == 1).all():
        return False, "全是前景的图应只有 1 个区域"
    try:
        f(mask, connectivity=6)
    except ValueError:
        return True, "4/8 连通、编号顺序和连续性、边界、空图和大连通域都正确"
    return False, "connectivity 非法时应抛出 ValueError"


# 手工布置的检测场景：24x30，模板 median = 100 + 2*列号，上半部分 sigma = 2，第 12 行起 sigma = 4。
# 每个区域列出 (行, 列, 灰度差)。k = 5、min_area = 3 时应检出 B、C、D、E 四个。
REGIONS = {
    "A": ("孤立的单个亮像素", [(2, 20, 30)]),
    "B": ("L 形亮缺陷", [(4, 5, 12), (4, 6, 14), (4, 7, 16), (5, 5, 18)]),
    "C": ("4 个像素沿对角线相连的亮缺陷（只有 8 连通才连得起来）", [(8, 12, 12), (9, 13, 12), (10, 14, 12), (11, 15, 12)]),
    "D": ("比模板暗、面积恰好等于 min_area 的横条", [(14, 3, -40), (14, 4, -28), (14, 5, -24)]),
    "E": ("比模板暗的 2x3 方块", [(18, 20, -24), (18, 21, -24), (18, 22, -44), (19, 20, -24), (19, 21, -24), (19, 22, -24)]),
    # 中间那个像素的 z 恰好等于 5：不算候选，两头各剩一个孤立像素
    "F": ("中间像素 z 恰好等于 k 的横条", [(21, 8, 24), (21, 9, 20), (21, 10, 24)]),
}
WANT = {  # 手算的结果
    "A": {"bbox": (2, 20, 3, 21), "area": 1, "centroid": (2.0, 20.0), "max_z": 15.0},
    "B": {"bbox": (4, 5, 6, 8), "area": 4, "centroid": (4.25, 5.75), "max_z": 9.0},
    "C": {"bbox": (8, 12, 12, 16), "area": 4, "centroid": (9.5, 13.5), "max_z": 6.0},
    "D": {"bbox": (14, 3, 15, 6), "area": 3, "centroid": (14.0, 4.0), "max_z": 10.0},
    "E": {"bbox": (18, 20, 20, 23), "area": 6, "centroid": (18.5, 21.0), "max_z": 11.0},
    "F1": {"bbox": (21, 8, 22, 9), "area": 1, "centroid": (21.0, 8.0), "max_z": 6.0},
    "F2": {"bbox": (21, 10, 22, 11), "area": 1, "centroid": (21.0, 10.0), "max_z": 6.0},
}


def hand_scene():
    median = np.tile(100.0 + 2.0 * np.arange(30), (24, 1))
    sigma = np.full((24, 30), 2.0)
    sigma[12:] = 4.0
    image = median.copy()
    for _, pixels in REGIONS.values():
        for y, x, diff in pixels:
            image[y, x] += diff
    return image, median, sigma


def defect_problem(got, key):
    """返回 None 表示这个缺陷的四个字段都对，否则返回一句说明。"""
    want, desc = WANT[key], REGIONS[key[0]][0]
    bbox = got["bbox"]
    if len(bbox) != 4 or tuple(int(v) for v in bbox) != want["bbox"]:
        return f"{desc}：bbox 应为 {want['bbox']}（y0, x0, y1, x1，y1 和 x1 是开区间），得到 {tuple(int(v) for v in bbox)}"
    if isinstance(got["area"], bool) or not isinstance(got["area"], (int, np.integer)) or got["area"] != want["area"]:
        return f"{desc}：area 应为整数 {want['area']}，得到 {got['area']!r}"
    centroid = got["centroid"]
    if len(centroid) != 2 or not all(isinstance(v, float) for v in centroid):
        return f"{desc}：centroid 应是两个 Python float"
    if not close(centroid, want["centroid"], 1e-9):
        return f"{desc}：centroid 应为 {want['centroid']}（行在前、列在后，区域内像素坐标的算术平均），得到 ({centroid[0]:.3f}, {centroid[1]:.3f})"
    if not isinstance(got["max_z"], float) or not close(got["max_z"], want["max_z"], 1e-9):
        return f"{desc}：max_z 应为 Python float {want['max_z']}（这个区域内 |差| / sigma 的最大值），得到 {got['max_z']!r}"
    return None


def list_problem(found, keys, setting):
    if not isinstance(found, list) or not all(isinstance(d, dict) for d in found):
        return "返回值必须是 list[dict]"
    needed = {"bbox", "area", "centroid", "max_z"}
    if any(not needed.issubset(d) for d in found):
        return f"每个缺陷需包含字段 {sorted(needed)}"
    if len(found) != len(keys):
        boxes = {tuple(sorted(int(v) for v in d["bbox"])) for d in found}
        missing = [REGIONS[k[0]][0] for k in keys if tuple(sorted(WANT[k]["bbox"])) not in boxes]
        hint = "没检出的有：" + "；".join(missing) if missing else "多出来的检查一下：z 恰好等于 k 的像素不算候选，面积小于 min_area 的要去掉"
        return f"{setting}应检出 {len(keys)} 个缺陷，得到 {len(found)} 个。{hint}"
    by_box = {tuple(int(v) for v in d["bbox"]): i for i, d in enumerate(found) if len(d["bbox"]) == 4}
    order = [by_box.get(WANT[k]["bbox"]) for k in keys]
    if None not in order and order != sorted(order):
        return f"{setting}缺陷的内容对，但顺序不对：应按连通域编号（每个区域第一个像素的光栅顺序）排列"
    for got, key in zip(found, keys):
        problem = defect_problem(got, key)
        if problem:
            return setting + problem
    return None


def check_detect(mod):
    f = mod.detect_defects
    image, median, sigma = hand_scene()
    keeps = [a.copy() for a in (image, median, sigma)]
    found = f(image, median, sigma, k=5.0, min_area=3)
    if not all(np.array_equal(a, b) for a, b in zip((image, median, sigma), keeps)):
        return False, "函数修改了传入的数组，需要改动时先复制一份"
    problem = list_problem(found, ["B", "C", "D", "E"], "k=5、min_area=3 时")
    if problem:
        return False, problem
    problem = list_problem(f(image, median, sigma, k=5.0, min_area=1), ["A", "B", "C", "D", "E", "F1", "F2"], "k=5、min_area=1 时")
    if problem:
        return False, problem + "。检查 min_area 参数有没有真的用上"
    high = f(image, median, sigma, k=8.0, min_area=1)
    if not isinstance(high, list) or len(high) != 4 or sorted(round(float(d["max_z"]), 6) for d in high) != [9.0, 10.0, 11.0, 15.0]:
        return False, f"k=8、min_area=1 时只有 z 为 15、9、10、11 的四个像素超过阈值（z 恰好等于 8 的不算），应检出 4 个单像素缺陷，得到 {len(high)} 个。检查 k 参数有没有真的用上"
    as_u8 = f(image.astype(np.uint8), median.astype(np.uint8), sigma, k=5.0, min_area=3)
    problem = list_problem(as_u8, ["B", "C", "D", "E"], "image 和 median 是 uint8 时")
    if problem:
        return False, problem + "。检查相减之前有没有转成 float64"
    clean = f(median, median, sigma, k=5.0, min_area=1)
    if not isinstance(clean, list) or len(clean) != 0:
        return False, "待检图和模板完全相同时应返回空列表"

    # 带噪声的场景：真实的模板和噪声水平，三个 3x3 缺陷
    rng = np.random.default_rng(1)
    base = pattern()
    test = base + rng.normal(scale=3.0, size=(64, 64))
    test[19:22, 29:32] += 60.0
    test[44:47, 11:14] -= 60.0
    test[49:52, 49:52] += 45.0
    noise_sigma = np.full((64, 64), 3.0)
    found = f(test, base, noise_sigma, k=6.0, min_area=3)
    centroids = [tuple(round(v) for v in d["centroid"]) for d in found]
    if centroids != [(20, 30), (45, 12), (50, 50)]:
        return False, f"带噪声的场景应依次检出质心在 (20,30)、(45,12)、(50,50) 的三个缺陷，得到 {centroids}"
    if found[0]["area"] != 9 or tuple(int(v) for v in found[0]["bbox"]) != (19, 29, 22, 32):
        return False, "3x3 缺陷的面积应为 9，bbox 应为 (19,29,22,32) 开区间"
    good = base + rng.normal(scale=3.0, size=(64, 64))
    none = f(good, base, noise_sigma, k=6.0, min_area=3)
    if len(none) != 0:
        return False, f"良品图上不应报缺陷，得到 {len(none)} 个"
    return True, "手工场景里的阈值边界、面积边界、连通性、四个字段和顺序都正确，带噪声场景零误报"


def stats_problem(got, tp, fp, fn, rate, what):
    if not isinstance(got, dict) or not {"tp", "fp", "fn", "capture_rate"}.issubset(got):
        return "返回值必须是包含 tp、fp、fn、capture_rate 的 dict"
    if (got["tp"], got["fp"], got["fn"]) != (tp, fp, fn):
        return f"{what}：应为 tp={tp}、fp={fp}、fn={fn}，得到 tp={got['tp']}、fp={got['fp']}、fn={got['fn']}"
    if not close(got["capture_rate"], rate, 1e-12):
        return f"{what}：capture_rate 应为 {rate}，得到 {got['capture_rate']}"
    return None


def check_match(mod):
    f = mod.match_defects
    pred = [(10.0, 10.0), (30.0, 31.0), (80.0, 80.0)]
    truth = [(10.5, 9.5), (30.0, 30.0), (60.0, 60.0), (5.0, 5.0)]
    pred_keep, truth_keep = list(pred), list(truth)
    got = f(pred, truth, radius=3.0)
    if pred != pred_keep or truth != truth_keep:
        return False, "函数修改了传入的列表，需要排序时先复制一份"
    cases = [
        (got, 2, 1, 2, 0.5, "3 个预测、4 个真值，其中 2 对离得近"),
        (f([(0.0, 0.0), (0.0, 2.0)], [(0.0, 1.9)], radius=3.0), 1, 1, 0, 1.0, "两个预测挨着同一个真值（一个真值只能配一个预测）"),
        (f([(0.0, 0.0)], [(0.0, 1.0), (0.0, -1.0)], radius=3.0), 1, 0, 1, 0.5, "一个预测夹在两个真值中间（一个预测只能配一个真值）"),
        (f([(0.0, 0.0)], [(3.0, 4.0)], radius=5.0), 1, 0, 0, 1.0, "距离恰好等于 radius（3-4-5 三角形）"),
        (f([(0.0, 0.0)], [(2.5, 2.5)], radius=3.0), 0, 1, 1, 0.0, "预测 (0,0)、真值 (2.5,2.5)、半径 3，欧氏距离约 3.54"),
        (f([(0.0, 0.0)], [(2.0, 2.0)], radius=3.0), 1, 0, 0, 1.0, "预测 (0,0)、真值 (2,2)、半径 3，欧氏距离约 2.83"),
        (f([(0.0, 0.0)], [(0.5, 0.5)], radius=0.6), 0, 1, 1, 0.0, "预测 (0,0)、真值 (0.5,0.5)、半径 0.6，欧氏距离约 0.71"),
        # 下面两组：先配距离最小的一对，剩下的还能再配一对；按输入顺序先到先得只能配上一对
        (f([(0.0, 0.0), (0.0, 3.0)], [(0.0, 1.6), (0.0, -2.5)], radius=3.0), 2, 0, 0, 1.0,
         "预测 (0,0)、(0,3)，真值 (0,1.6)、(0,-2.5)：最近的一对是 (0,3) 和 (0,1.6)"),
        (f([(0.0, 1.6), (0.0, -2.5)], [(0.0, 0.0), (0.0, 3.0)], radius=3.0), 2, 0, 0, 1.0,
         "预测 (0,1.6)、(0,-2.5)，真值 (0,0)、(0,3)：最近的一对是 (0,1.6) 和 (0,3)"),
        (f([(1.0, 1.0)], [], radius=3.0), 0, 1, 0, 1.0, "真值为空、有 1 个预测"),
        (f([], [(1.0, 1.0), (5.0, 5.0)], radius=3.0), 0, 0, 2, 0.0, "预测为空、有 2 个真值"),
        (f([], [], radius=3.0), 0, 0, 0, 1.0, "预测和真值都为空"),
    ]
    for got, tp, fp, fn, rate, what in cases:
        problem = stats_problem(got, tp, fp, fn, rate, what)
        if problem:
            return False, problem
    return True, "一对一、按距离从小到大、欧氏距离、半径边界和空输入都正确"


def check_pipeline(mod):
    """三个函数串起来：12 张良品建模板，检测，再和真值匹配。"""
    rng = np.random.default_rng(1)
    base = pattern()
    goods = base + rng.normal(scale=3.0, size=(12, 64, 64))
    test = base + rng.normal(scale=3.0, size=(64, 64))
    truth = [(20.0, 30.0), (45.0, 12.0), (50.0, 50.0)]
    test[19:22, 29:32] += 60.0
    test[44:47, 11:14] -= 60.0
    test[49:52, 49:52] += 45.0
    median, sigma = mod.robust_golden(goods)
    found = mod.detect_defects(test, median, sigma, k=6.0, min_area=3)
    stats = mod.match_defects([d["centroid"] for d in found], truth, radius=3.0)
    problem = stats_problem(stats, 3, 0, 0, 1.0, "12 张良品建模板、k=6、min_area=3，待检图上有 3 个 3x3 的强缺陷")
    if problem:
        return False, problem
    if any(not isinstance(d["max_z"], float) or d["max_z"] < 10 for d in found):
        return False, "三个缺陷的幅度是噪声的 15 到 20 倍，max_z 应明显大于 10"
    none = mod.detect_defects(goods[0], median, sigma, k=6.0, min_area=3)
    if len(none) != 0:
        return False, f"拿一张良品图去检测不应报缺陷，得到 {len(none)} 个"
    return True, "建模板、检测、匹配串起来：3 个缺陷全部检出，良品零误报"


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
        run("整条流水线", lambda: check_pipeline(mod)),
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
