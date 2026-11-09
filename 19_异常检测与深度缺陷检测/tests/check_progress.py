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
    return True, "与暴力计算一致，数值误差已处理"


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
    try:
        mod.greedy_coreset(x, len(x) + 1)
    except ValueError:
        return True, "贪心性质、簇覆盖和覆盖半径都满足"
    return False, "m 超过点数时应抛出 ValueError"


def check_knn(mod):
    memory = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
    q = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 4.0]])
    s1 = mod.knn_scores(memory, q, k=1)
    if s1.shape != (3,) or not np.allclose(s1, [0.0, 0.5, 1.0]):
        return False, f"k=1 期望 [0, 0.5, 1]，得到 {s1}"
    s2 = mod.knn_scores(memory, q, k=2)
    if not np.allclose(s2, [0.5, 0.5, 2.5]):
        return False, "k=2 应为两个最近邻欧氏距离的平均"
    try:
        mod.knn_scores(memory, q, k=4)
    except ValueError:
        return True, "k=1/k=2 分数正确，非法 k 报错"
    return False, "k 大于 memory 大小时应抛出 ValueError"


def check_auroc(mod):
    cases = [
        ([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], 1.0),
        ([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1], 0.0),
        ([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1], 0.5),
        ([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1], 0.75),
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
    if abs(mod.auroc(s, y) - want) > 1e-12:
        return False, "随机样本上与成对比较定义不一致"
    try:
        mod.auroc(np.array([0.1, 0.2]), np.array([1, 1]))
    except ValueError:
        return True, "小例子、并列分数和随机样本都与定义一致"
    return False, "只有一个类别时应抛出 ValueError"


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
    return True, "漏检约束下的阈值和过检率正确"


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
