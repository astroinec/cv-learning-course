"""第 19 章自动验收：TODO / FAIL / PASS。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "19-1_PatchCore核心.py"


def load_assignment():
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    spec = importlib.util.spec_from_file_location("chapter19_assignment", base / ASSIGNMENT)
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


def unchanged(before, after):
    """调用前后的数组逐元素相同、数据类型也相同。"""
    return before.dtype == after.dtype and np.array_equal(before, after)


def check_dists(mod):
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(7, 5)), rng.normal(size=(4, 5))
    d = mod.pairwise_sq_dists(a, b)
    brute = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    if d.shape != (7, 4) or d.dtype != np.float64 or not np.allclose(d, brute):
        return False, "结果应等于逐对平方距离，形状 (N, M) float64"
    big = np.full((3, 4), 1e8) + rng.normal(scale=1e-3, size=(3, 4))
    if (mod.pairwise_sq_dists(big, big) < 0).any():
        return False, "大数值下出现负距离，需要裁剪到 0"
    far = rng.normal(size=(60, 16)) + 100.0
    keep = far.copy()
    self_d = mod.pairwise_sq_dists(far, far)
    if (self_d < 0).any():
        return False, f"60 个离原点较远的点和自己算距离，出现了负数（最小 {self_d.min():.2e}）；展开式有舍入误差，结果要裁剪到 0"
    if not unchanged(keep, far):
        return False, "函数改动了传入的数组；先复制或者另开变量再算"
    ai = np.array([[0, 200, 255], [10, 20, 30]], dtype=np.uint8)
    bi = np.array([[255, 0, 0], [10, 20, 31], [0, 200, 255]], dtype=np.uint8)
    di = mod.pairwise_sq_dists(ai, bi)
    want = [[170050, 82676, 0], [61325, 1, 83125]]
    if di.shape != (2, 3) or di.dtype != np.float64 or not np.allclose(di, want):
        return False, "输入是 uint8 的特征时结果不对（应得到 float64 的精确平方距离）；检查做乘法之前数据类型有没有处理，uint8 相乘会溢出"
    return True, "与暴力计算一致，数值误差、整数输入都已处理"


def coverage_radius(x, idx):
    d = ((x[:, None, :] - x[None, idx, :]) ** 2).sum(-1)
    return float(np.sqrt(d.min(axis=1).max()))


def check_coreset(mod):
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(loc=c, scale=0.3, size=(60, 2)) for c in [(0, 0), (6, 0), (0, 6), (6, 6)]])
    idx = mod.greedy_coreset(x, 8, start=0)
    if not isinstance(idx, np.ndarray) or idx.dtype != np.int64 or idx.shape != (8,):
        return False, "返回值应为 (m,) int64"
    if idx[0] != 0 or len(set(idx.tolist())) != 8:
        return False, "第一个下标应为 start，且不能重复"
    for i in range(1, 8):
        d = ((x[:, None, :] - x[None, idx[:i], :]) ** 2).sum(-1).min(axis=1)
        if d[idx[i]] < d.max() - 1e-9:
            return False, f"第 {i + 1} 个点不是当前离已选集合最远的点"
    centers = np.round(x[idx[:4]] / 6).astype(int)
    if len({tuple(c) for c in centers}) != 4:
        return False, "前 4 个点应覆盖 4 个簇"
    rand_r = np.mean([coverage_radius(x, rng.choice(len(x), 8, replace=False)) for _ in range(20)])
    if coverage_radius(x, idx) > rand_r:
        return False, "覆盖半径不应比随机采样更差"
    keep = x.copy()
    idx2 = mod.greedy_coreset(x, 6, start=137)
    if idx2[0] != 137:
        return False, f"start=137 时第一个下标应为 137，得到 {idx2[0]}；检查 start 参数有没有用上"
    for i in range(1, 6):
        d = ((x[:, None, :] - x[None, idx2[:i], :]) ** 2).sum(-1).min(axis=1)
        if d[idx2[i]] < d.max() - 1e-9:
            return False, f"start=137 时第 {i + 1} 个点不是当前离已选集合最远的点；检查第一步的距离是不是从 start 那个点算起"
    if not unchanged(keep, x):
        return False, "函数改动了传入的 features"
    line = np.array([[0.0], [1.0], [2.5], [10.0], [11.0], [30.0]])
    got = mod.greedy_coreset(line, 4, start=1).tolist()
    if got != [1, 5, 4, 2]:
        return False, f"数轴上的 6 个点 0、1、2.5、10、11、30，从下标 1 出发选 4 个，按“离已选集合最远”应依次选下标 [1, 5, 4, 2]，得到 {got}"
    if sorted(mod.greedy_coreset(line, 6, start=3).tolist()) != [0, 1, 2, 3, 4, 5]:
        return False, "m 等于点数是合法输入，应把每个点各选一次"
    if mod.greedy_coreset(line, 1, start=2).tolist() != [2]:
        return False, "m=1 时应只返回 start"
    for bad in (len(x) + 1, 0, -2):
        try:
            mod.greedy_coreset(x, bad)
        except ValueError:
            continue
        return False, f"m={bad}（共 {len(x)} 个点）时应抛出 ValueError；m 的合法范围是 1 到点数"
    return True, "贪心性质、起点、簇覆盖、覆盖半径和非法 m 都正确"


def check_knn(mod):
    memory = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
    q = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 4.0]])
    s1 = mod.knn_scores(memory, q, k=1)
    if s1.shape != (3,) or not np.allclose(s1, [0.0, 0.5, 1.0]):
        return False, f"k=1 期望 [0, 0.5, 1]，得到 {s1}"
    s2 = mod.knn_scores(memory, q, k=2)
    if not np.allclose(s2, [0.5, 0.5, 2.5]):
        return False, "k=2 应为两个最近邻欧氏距离的平均"
    # 5 个 query 对 3 个 memory，期望值是手算的。例如 query (4, 3) 到三个 memory 点的距离是 5、sqrt(18)、4
    q5 = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 4.0], [4.0, 3.0], [1.0, -1.0]])
    r2, r17, r18 = np.sqrt(2.0), np.sqrt(17.0), np.sqrt(18.0)
    expected = {
        1: [0.0, 0.5, 1.0, 4.0, 1.0],
        2: [0.5, 0.5, 2.5, (4.0 + r18) / 2, (1.0 + r2) / 2],
        3: [4.0 / 3, (1.0 + np.sqrt(9.25)) / 3, (5.0 + r17) / 3, (9.0 + r18) / 3, (1.0 + r2 + r17) / 3],
    }
    keep_m, keep_q = memory.copy(), q5.copy()
    for k, want in expected.items():
        got = mod.knn_scores(memory, q5, k=k)
        if not isinstance(got, np.ndarray) or got.shape != (5,) or got.dtype != np.float64:
            return False, f"5 个 query、3 个 memory、k={k}：应返回 (5,) float64，每个 query 一个分数；检查两个参数的顺序和取平均的轴"
        if not np.allclose(got, want):
            hint = "k 等于 memory 的大小是合法输入，这时是到全部点的平均距离" if k == 3 else "检查是不是对每个 query 取它自己最近的 k 个距离再平均"
            return False, f"5 个 query、3 个 memory、k={k}：期望 {np.round(want, 4).tolist()}，得到 {np.round(got, 4).tolist()}；{hint}"
    if not unchanged(keep_m, memory) or not unchanged(keep_q, q5):
        return False, "函数改动了传入的 memory 或 queries"
    for bad in (4, 0, -1):
        try:
            mod.knn_scores(memory, q, k=bad)
        except ValueError:
            continue
        return False, f"memory 有 3 个点，k={bad} 时应抛出 ValueError；k 的合法范围是 1 到 memory 的大小"
    return True, "k=1/k=2/k=M 分数正确，非法 k 报错"


def check_auroc(mod):
    cases = [
        ([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], 1.0),
        ([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1], 0.0),
        ([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1], 0.5),
        ([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1], 0.75),
        ([1, 2, 2, 3, 3, 3], [0, 0, 1, 0, 1, 1], 6.5 / 9),
    ]
    for s, y, want in cases:
        got = mod.auroc(np.array(s), np.array(y))
        if not isinstance(got, float) or abs(got - want) > 1e-12:
            return False, f"scores={s} labels={y} 期望 {want}，得到 {got!r}"
    rng = np.random.default_rng(2)
    s = rng.normal(size=400)
    y = (rng.random(400) < 0.3).astype(int)
    pos, neg = s[y == 1], s[y == 0]
    want = ((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    keep_s, keep_y = s.copy(), y.copy()
    if abs(mod.auroc(s, y) - want) > 1e-12:
        return False, "随机样本上与成对比较定义不一致"
    if not unchanged(keep_s, s) or not unchanged(keep_y, y):
        return False, "函数改动了传入的 scores 或 labels；排序、归一化之前先复制"
    st = np.round(keep_s, 1)
    keep_s, keep_y = st.copy(), y.copy()
    pos, neg = st[y == 1], st[y == 0]
    want = ((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    got = mod.auroc(st, y)
    if abs(got - want) > 1e-12:
        return False, f"分数只保留一位小数（有大量并列）的 400 个样本上，与成对比较的定义不一致：期望 {want:.6f}，得到 {got:.6f}"
    if not unchanged(keep_s, st) or not unchanged(keep_y, y):
        return False, "函数改动了传入的 scores 或 labels；排序、归一化之前先复制"
    for only in (1, 0):
        try:
            mod.auroc(np.array([0.1, 0.2, 0.3]), np.array([only] * 3))
        except ValueError:
            continue
        return False, f"标签全是 {only} 时应抛出 ValueError；只有正常或者只有异常都算“只有一个类别”"
    return True, "小例子、并列分数和随机样本都与定义一致"


def check_threshold(mod):
    normal = np.array([0.1, 0.2, 0.3, 0.4, 0.55, 0.6])
    defect = np.array([0.5, 0.7, 0.8, 0.9])
    thr, esc, ovk = mod.threshold_at_escape(normal, defect, max_escape=0.0)
    if not all(isinstance(v, float) for v in (thr, esc, ovk)):
        return False, "三个返回值都应是 Python float"
    if (thr, esc) != (0.5, 0.0) or abs(ovk - 2 / 6) > 1e-12:
        return False, f"零漏检时应取阈值 0.5、过检 2/6，得到 {(thr, esc, ovk)}"
    thr, esc, ovk = mod.threshold_at_escape(normal, defect, max_escape=0.25)
    if (thr, esc, ovk) != (0.7, 0.25, 0.0):
        return False, f"允许 25% 漏检时应取阈值 0.7、过检 0，得到 {(thr, esc, ovk)}"
    normal2 = np.array([0.6, 0.1, 0.55, 0.3, 0.2, 0.4])
    defect2 = np.array([0.9, 0.5, 0.8, 0.7])
    keep_n, keep_d = normal2.copy(), defect2.copy()
    for max_escape, want in ((0.0, (0.5, 0.0, 2 / 6)), (0.25, (0.7, 0.25, 0.0)), (0.5, (0.8, 0.5, 0.0))):
        thr, esc, ovk = mod.threshold_at_escape(normal2, defect2, max_escape=max_escape)
        if (thr, esc) != want[:2] or abs(ovk - want[2]) > 1e-12:
            return False, f"分数没有排序的输入（缺陷 [0.9, 0.5, 0.8, 0.7]），漏检上限 {max_escape}：期望 {want}，得到 {(thr, esc, ovk)}；输入不保证有序"
    if not unchanged(keep_n, normal2) or not unchanged(keep_d, defect2):
        return False, "函数改动了传入的分数数组（例如原地排序）；先复制再排"
    normal3 = np.array([0.1, 0.7, 0.7, 0.9])
    defect3 = np.array([0.7, 0.8, 0.95, 1.0])
    thr, esc, ovk = mod.threshold_at_escape(normal3, defect3, max_escape=0.0)
    if (thr, esc) != (0.7, 0.0) or abs(ovk - 0.75) > 1e-12:
        return False, f"正常样本里有两个分数正好等于阈值 0.7 时，过检率应为 3/4，得到 {(thr, esc, ovk)}；对照文档字符串里判为缺陷的规则"
    thr, esc, ovk = mod.threshold_at_escape(normal3, defect3, max_escape=0.5)
    if (thr, esc, ovk) != (0.95, 0.5, 0.0):
        return False, f"允许 50% 漏检时应取阈值 0.95、漏检 0.5、过检 0，得到 {(thr, esc, ovk)}"
    return True, "漏检约束下的阈值和过检率正确，输入无序、分数等于阈值的情况也正确"


def main() -> int:
    print("第 19 章进度检查\n" + "=" * 52)
    try:
        mod = load_assignment()
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "加载作业", f"{type(exc).__name__}: {exc}")
        return 1
    results = [
        run("平方距离矩阵", lambda: check_dists(mod)),
        run("k-center greedy", lambda: check_coreset(mod)),
        run("kNN 异常分数", lambda: check_knn(mod)),
        run("AUROC", lambda: check_auroc(mod)),
        run("漏检约束阈值", lambda: check_threshold(mod)),
    ]
    print("=" * 52)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("零件通过；下一步在 Notebook 里接上预训练特征，在 MVTec AD 一个类别上跑完整 PatchCore。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
