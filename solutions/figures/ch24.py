"""第 24 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch24.py

输出：<主分支>/24_综合项目_晶圆检测流水线/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。

两条约定（这一章的消融、工作点曲线和漏检分析正是学员报告里要自己做的事）：
1. 所有实验用种子 100 起的数据，避开验收种子 11 和 23，学员报告里要填的数字不在这里出现。
2. 参考实现调好的参数（k、min_area、sigma 下限、搜索范围、边界宽度）不写进 实测数字.json，
   扫描类实验用“参考配置的几倍”或者报告模板里公开列出的取值来描述。
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

CH = "24_综合项目_晶圆检测流水线"
mod = S.load_solution(CH, "24-1_检测流水线.py")
NUM: dict = {}
SEEDS = list(range(100, 120))
RADIUS = 4.0
SIZE = 96
_sig = inspect.signature(mod.run_pipeline).parameters
K0, AREA0, BORDER0 = _sig["k"].default, _sig["min_area"].default, _sig["border"].default
FLOOR0, SEARCH0 = 0.02, 8  # 参考实现里写死的 sigma 下限和搜索范围；check_equivalence 会核对


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load(next(p for p in (S.MAIN / CH / "数据生成.py", S.REPO / CH / "数据生成.py") if p.exists()), "gen24")
_SHIFT, _CAPTURE, _DEFECT, _LAYOUT = gen._fourier_shift, gen._capture, gen._add_defect, gen.make_layout


# ---------------------------------------------------------------- 数据：和 make_dataset 完全相同，另外记下分析要用的真值
def dataset(seed: int, **kw) -> dict:
    """调用原样的 make_dataset，同时记录每张图的错位、增益，以及每个缺陷的类型和成像后的对比度。"""
    log = {"shift": [], "gain": [], "defect": []}

    def rec_shift(img, dy, dx):
        log["shift"].append((float(dy), float(dx)))
        return _SHIFT(img, dy, dx)

    def rec_capture(clean, rng, *a, **k):
        peek = copy.deepcopy(rng)
        log["gain"].append((float(peek.uniform(0.85, 1.15)), float(peek.uniform(-0.05, 0.05))))
        return _CAPTURE(clean, rng, *a, **k)

    def rec_defect(img, kind, cy, cx, rng):
        before = img.copy()
        _DEFECT(img, kind, cy, cx, rng)
        log["defect"].append((len(log["shift"]) - 1, str(kind), float(cy), float(cx), img - before))

    gen._fourier_shift, gen._capture, gen._add_defect = rec_shift, rec_capture, rec_defect
    try:
        d = gen.make_dataset(seed=seed, **kw)
    finally:
        gen._fourier_shift, gen._capture, gen._add_defect = _SHIFT, _CAPTURE, _DEFECT
    g = len(d["golden"])
    d["seed"] = seed
    d["all_shifts"] = np.asarray(log["shift"])
    d["gain"] = np.asarray(log["gain"])
    d["info"] = [[] for _ in d["truth"]]
    for index, kind, cy, cx, delta in log["defect"]:
        signal = gen._gaussian_blur(delta, 1.0) * d["gain"][index][0]
        d["info"][index - g].append({"kind": kind, "cy": cy, "cx": cx, "contrast": float(np.abs(signal).max())})
    return d


def match(pred, truth, radius=RADIUS):
    pairs = sorted((float(np.hypot(p[0] - t[0], p[1] - t[1])), i, j) for i, p in enumerate(pred) for j, t in enumerate(truth))
    used_p, used_t = set(), set()
    for dist, i, j in pairs:
        if dist <= radius and i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
    return used_p, used_t


def score(preds, truth, radius=RADIUS):
    tp = fp = fn = 0
    for p, t in zip(preds, truth):
        used_p, used_t = match(p, t, radius)
        tp, fp, fn = tp + len(used_p), fp + len(p) - len(used_p), fn + len(t) - len(used_t)
    return tp, fp, fn


def wilson(hit: int, n: int, z: float = 1.96):
    p = hit / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


# ---------------------------------------------------------------- 可以逐项拆掉的流水线（默认配置和参考实现逐项相同，见 check_equivalence）
def estimate_shift(ref, img, mode="full", max_shift=SEARCH0, window=False):
    """mode：none 不对齐；integer 只取整数峰；single 一次抛物线；full 再细化一轮（参考实现）。"""
    if mode == "none":
        return 0.0, 0.0
    dy, dx = mod._phase_corr(ref, img, max_shift=max_shift, window=window)
    if mode == "integer":
        return float(round(dy)), float(round(dx))
    if mode == "single":
        return dy, dx
    ry, rx = mod._phase_corr(ref, mod._shift(img, -dy, -dx), max_shift=1, window=window)
    return dy + ry, dx + rx


def prepare(d, *, align="full", align_golden=True, align_test=True, max_shift=SEARCH0, window=False,
            norm_golden=True, norm_test=True, stat="mad", floor=FLOOR0, golden=None, test=None):
    golden = d["golden"] if golden is None else golden
    test = d["test"] if test is None else test
    ref = golden[0]
    gs = [estimate_shift(ref, g, align if align_golden else "none", max_shift, window) for g in golden]
    stack = np.stack([mod._shift(g, -dy, -dx) for g, (dy, dx) in zip(golden, gs)])
    if norm_golden:
        template = np.median(stack, axis=0)
        stack = np.stack([mod._normalize(g, template) for g in stack])
    if stat == "mad":
        center = np.median(stack, axis=0)
        raw = 1.4826 * np.median(np.abs(stack - center), axis=0)
    elif stat == "mean":
        center, raw = stack.mean(axis=0), stack.std(axis=0, ddof=1)
    else:  # "flat"：全图用同一个 sigma（就是下限）
        center = np.median(stack, axis=0)
        raw = np.zeros_like(center)
    sigma = np.maximum(raw, floor)
    zs, shifts, aligned = [], [], []
    for img in test:
        dy, dx = estimate_shift(ref, img, align if align_test else "none", max_shift, window)
        a = mod._shift(img, -dy, -dx)
        aligned.append(a)
        if norm_test:
            a = mod._normalize(a, center)
        zs.append(np.abs(a - center) / sigma)
        shifts.append((dy, dx))
    return {"z": np.stack(zs), "shifts": np.asarray(shifts), "center": center, "sigma": sigma, "raw": raw, "aligned": np.stack(aligned)}


def components(mask):
    """返回 [(ys, xs), ...]，像素按光栅顺序排列，和参考实现里 np.nonzero(lab == i) 的顺序一致。"""
    lab, n = mod._labels(mask)
    if not n:
        return []
    flat = lab.ravel()
    order = np.argsort(flat, kind="stable")
    start = np.cumsum(np.bincount(flat, minlength=n + 1))
    return [np.divmod(order[start[i - 1]:start[i]], mask.shape[1]) for i in range(1, n + 1)]


def detect(z, shifts, k=K0, min_area=AREA0, border=BORDER0, back=1.0):
    """back：1 把坐标加回错位（正确做法），0 不加回，-1 符号写反。"""
    out = []
    for zi, (dy, dx) in zip(z, shifts):
        mask = zi > k
        if border:
            mask[:border, :] = mask[-border:, :] = False
            mask[:, :border] = mask[:, -border:] = False
        found = []
        for ys, xs in components(mask):
            if len(ys) < min_area:
                continue
            wgt = zi[ys, xs]
            cy, cx = float((ys * wgt).sum() / wgt.sum()), float((xs * wgt).sum() / wgt.sum())
            found.append(((cy + back * dy) % zi.shape[0], (cx + back * dx) % zi.shape[1]))
        out.append(found)
    return out


DET_KEYS = ("k", "min_area", "border", "back")


def tally(cfg: dict, data) -> np.ndarray:
    """每个种子一行 (tp, fp, fn)。"""
    prep_cfg = {k: v for k, v in cfg.items() if k not in DET_KEYS}
    det_cfg = {k: v for k, v in cfg.items() if k in DET_KEYS}
    rows = []
    for d in data:
        p = prepare(d, **prep_cfg)
        rows.append(score(detect(p["z"], p["shifts"], **det_cfg), d["truth"]))
    return np.asarray(rows)


def summary(rows: np.ndarray, n_die: int = 30) -> dict:
    tp, fp, fn = rows.sum(axis=0)
    cap = rows[:, 0] / (rows[:, 0] + rows[:, 2])
    fa = rows[:, 1] / n_die
    return {"检出率": round(float(tp / (tp + fn)), 3), "每 die 误报": round(float(fp / (n_die * len(rows))), 2),
            "各种子检出率的最小和最大": [round(float(cap.min()), 3), round(float(cap.max()), 3)],
            "各种子每 die 误报的中位数和最大": [round(float(np.median(fa)), 2), round(float(fa.max()), 2)],
            "误报为 0 的种子数": int((rows[:, 1] == 0).sum()),
            "达不到验收线的种子数": int(((cap < 0.85) | (fa > 0.3)).sum())}


def check_equivalence(data) -> None:
    for d in data[:3]:
        p = prepare(d)
        assert detect(p["z"], p["shifts"]) == mod.run_pipeline(d["golden"].copy(), d["test"].copy()), "拆开的流水线和参考实现不一致"
        fresh = gen.make_dataset(seed=d["seed"])
        assert np.array_equal(fresh["test"], d["test"]) and fresh["truth"] == d["truth"], "带记录的数据和原始数据不一致"
        assert np.allclose(d["all_shifts"][len(d["golden"]):], d["shifts"]), "记录的错位和数据生成器返回的不一致"


CONDITION = (f"96x96 合成 die，每个种子 8 张良品、30 张待检（约六成带 1 到 3 个缺陷），错位每轴 ±3.5 px、增益 ±15%、偏置 ±0.05；"
             f"种子 {SEEDS[0]} 到 {SEEDS[-1]} 共 {len(SEEDS)} 个，匹配半径 {RADIUS:.0f} px；“完整流水线”指参考实现的默认参数")


# ---------------------------------------------------------------- 图 1：一张 die 的中间结果
def pick_example(data):
    """示例 die：所在种子去掉对齐后误报为 0（20 个种子里的多数情况），die 上有两种以上缺陷且全部检出，增益和错位都明显偏离参考良品。"""
    for d in data:
        if tally({"align": "none"}, [d])[0, 1] != 0:
            continue
        full = prepare(d)
        preds = detect(full["z"], full["shifts"])
        rel = np.hypot(*(d["shifts"] - d["all_shifts"][0]).T)
        for i, t in enumerate(d["truth"]):
            if (len(t) >= 2 and len(match(preds[i], t)[1]) == len(t) and len(preds[i]) == len(t) and abs(d["gain"][8 + i][0] - 1) > 0.08
                    and rel[i] > 3 and len({e["kind"] for e in d["info"][i]}) >= 2):
                return d, i
    raise RuntimeError("没有找到合适的示例 die")


EXAMPLE: dict = {}


def fig_stages(data):
    d, pick = pick_example(data)
    EXAMPLE.update(d=d, pick=pick)
    full = prepare(d)
    preds = detect(full["z"], full["shifts"])
    rel = np.hypot(*(d["shifts"] - d["all_shifts"][0]).T)
    img, (dy, dx) = d["test"][pick], full["shifts"][pick]
    aligned = full["aligned"][pick]
    normed = mod._normalize(aligned, full["center"])
    z = full["z"][pick]

    # 只看没有缺陷的待检 die：模板和 sigma 固定用完整流水线的，待检图逐步加上对齐和归一化
    names = ["待检图原样", "只做对齐", "对齐加亮度归一化"]
    frac = dict.fromkeys(names, 0.0)
    n_clean = 0
    for dd in data:
        p = prepare(dd)
        for i, t in enumerate(dd["truth"]):
            if t:
                continue
            n_clean += 1
            for name, a in zip(names, (dd["test"][i], p["aligned"][i], mod._normalize(p["aligned"][i], p["center"]))):
                frac[name] += float((np.abs(a - p["center"]) / p["sigma"] > K0).mean())
    NUM["中间结果"] = {
        "说明": CONDITION,
        "图 1 用的样本": {"种子": d["seed"], "die 序号": int(pick), "这张 die 的增益": round(float(d["gain"][8 + pick][0]), 2),
                     "相对参考良品的错位（px）": round(float(rel[pick]), 1), "缺陷个数": len(d["truth"][pick]),
                     "缺陷类型": [e["kind"] for e in d["info"][pick]]},
        "无缺陷的待检 die 张数": n_clean,
        "无缺陷的待检 die 上超过阈值的像素比例（模板和 sigma 固定用完整流水线的）": {k: float(f"{v / n_clean:.3g}") for k, v in frac.items()},
    }

    fig, ax = plt.subplots(2, 3, figsize=(10.8, 7.3), layout="constrained")
    ax = ax.ravel()
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=1.1)
    for cy, cx in d["truth"][pick]:
        ax[0].plot(cx, cy, "o", mfc="none", mec=S.GREEN, ms=17, mew=1.6)
    ax[0].set_title("1 待检 die 原图（绿圈是真缺陷）")
    for a, diff, title in ((ax[1], img - full["center"], "2 不对齐，直接减模板"), (ax[2], aligned - full["center"], "3 对齐后减模板"),
                           (ax[3], normed - full["center"], "4 再做亮度归一化")):
        im = a.imshow(diff, cmap="coolwarm", vmin=-0.3, vmax=0.3)
        a.set_title(title)
    fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.03).ax.tick_params(labelsize=8)
    im = ax[4].imshow(z, cmap="magma", vmin=0, vmax=2.5 * K0)
    ax[4].set_title("5 除以逐像素 sigma 得到 z 图")
    cb = fig.colorbar(im, ax=ax[4], fraction=0.046, pad=0.03, ticks=[0, K0, 2 * K0])
    cb.ax.set_yticklabels(["0", "阈值", "2 倍"], fontsize=8)
    ax[5].imshow(z > K0, cmap="gray_r")
    for cy, cx in d["truth"][pick]:
        ax[5].plot((cx - dx) % SIZE, (cy - dy) % SIZE, "o", mfc="none", mec=S.GREEN, ms=17, mew=1.6)
    for cy, cx in preds[pick]:
        ax[5].plot((cx - dx) % SIZE, (cy - dy) % SIZE, "x", color=S.RED, ms=8, mew=2)
    ax[5].set_title("6 超过阈值的像素，红叉是检出中心")
    for a in ax:
        S.image_axes(a)
    S.save(fig, CH, "fig01_流水线中间结果")


# ---------------------------------------------------------------- 图 2：种子之间的波动
def fig_seeds(data):
    base = tally({}, data)
    n = base[:, 0] + base[:, 2]
    cap = base[:, 0] / n
    ci = np.array([wilson(int(h), int(m)) for h, m in zip(base[:, 0], n)])
    order = np.argsort(cap, kind="stable")
    small = {"亚像素对齐不做细化": {"align": "single"}, "均值和标准差代替中位数和 MAD": {"stat": "mean"}, "全图用同一个 sigma": {"stat": "flat"}}
    diffs = {name: tally(cfg, data)[:, 0] - base[:, 0] for name, cfg in small.items()}
    mean_n = int(round(n.mean()))
    p_all = base[:, 0].sum() / n.sum()
    need = math.ceil(0.85 * mean_n)
    p_below = sum(math.comb(mean_n, x) * p_all ** x * (1 - p_all) ** (mean_n - x) for x in range(0, need))
    lo, hi = wilson(int(base[:, 0].sum()), int(n.sum()))
    worst = int(order[0])
    wl, wh = wilson(int(base[worst, 0]), int(n[worst]))

    more = np.asarray([score(mod.run_pipeline(dd["golden"], dd["test"]), dd["truth"]) for dd in (gen.make_dataset(seed=s) for s in range(100, 200))])
    more_cap = more[:, 0] / (more[:, 0] + more[:, 2])
    NUM["种子之间的波动"] = {
        "说明": CONDITION,
        "各种子（检出数，缺陷数，误报数）": {str(d["seed"]): [int(r[0]), int(r[0] + r[2]), int(r[1])] for d, r in zip(data, base)},
        "合计": {"检出": int(base[:, 0].sum()), "缺陷": int(n.sum()), "误报": int(base[:, 1].sum()), "待检 die": 30 * len(data),
               "检出率": round(float(p_all), 3), "Wilson 95% 区间": [round(lo, 3), round(hi, 3)]},
        "各种子检出率": {"最小": round(float(cap.min()), 3), "中位数": round(float(np.median(cap)), 3), "最大": round(float(cap.max()), 3),
                    "标准差": round(float(cap.std(ddof=1)), 3), "低于 85% 的种子数": int((cap < 0.85).sum())},
        "每个种子的缺陷数": {"最少": int(n.min()), "平均": round(float(n.mean()), 1), "最多": int(n.max())},
        "检出率最低的那个种子": {"种子": data[worst]["seed"], "检出": int(base[worst, 0]), "缺陷": int(n[worst]),
                       "检出率": round(float(cap[worst]), 3), "Wilson 95% 区间": [round(wl, 3), round(wh, 3)]},
        "二项分布估算": {"说明": f"真实检出率取合计值、每个种子按 {mean_n} 个缺陷算，一个种子测出低于 85%（检出不到 {need} 个）的概率",
                     "概率": round(float(p_below), 3)},
        "种子 100 到 199 共 100 个": {"检出率低于 85% 的种子数": int((more_cap < 0.85).sum()), "每 die 误报超过 0.3 的种子数": int((more[:, 1] / 30 > 0.3).sum()),
                               "检出率最小和最大": [round(float(more_cap.min()), 3), round(float(more_cap.max()), 3)],
                               "合计检出率": round(float(more[:, 0].sum() / (more[:, 0] + more[:, 2]).sum()), 3),
                               "缺陷总数": int((more[:, 0] + more[:, 2]).sum()), "误报总数": int(more[:, 1].sum())},
        "三项小改动相对完整流水线、每个种子多检出的缺陷数": {name: {"最小": int(v.min()), "最大": int(v.max()), "合计": int(v.sum()),
                                           "折合检出率变化": round(float(v.sum() / n.sum()), 3),
                                           "各差值对应的种子个数": {str(int(a)): int(b) for a, b in zip(*np.unique(v, return_counts=True))}}
                                    for name, v in diffs.items()},
    }

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.1), gridspec_kw={"width_ratios": [1.45, 1]}, layout="constrained")
    x = np.arange(len(order))
    c = cap[order]
    ax[0].errorbar(x, c, yerr=[c - ci[order, 0], ci[order, 1] - c], fmt="none", ecolor="#b9bec7", elinewidth=1.4, capsize=2.5)
    ax[0].scatter(x, c, s=34, color=[S.RED if v < 0.85 else S.BLUE for v in c], zorder=3)
    ax[0].axhline(0.85, color=S.RED, lw=1.2, ls="--")
    ax[0].text(len(x) - 0.6, 0.845, "验收线 85%", color=S.RED, fontsize=9, ha="right", va="top")
    ax[0].axhline(p_all, color=S.GREEN, lw=1.2, ls=":")
    ax[0].text(-0.4, p_all + 0.006, f"20 个种子合计 {100 * p_all:.1f}%", color=S.GREEN, fontsize=9, ha="left", va="bottom")
    ax[0].set_xticks(x, [str(data[i]["seed"]) for i in order], rotation=90, fontsize=8)
    ax[0].set_xlabel("种子（按检出率从低到高排）")
    ax[0].set_ylabel("检出率（竖线是 Wilson 95% 区间）")
    ax[0].set_ylim(0.62, 1.03)
    ax[0].set_title(f"同一条流水线，20 个种子里有 {int((cap < 0.85).sum())} 个低于验收线")
    S.clean(ax[0])
    for row, (name, v) in enumerate(diffs.items()):
        vals, counts = np.unique(v, return_counts=True)
        ax[1].scatter(vals, np.full(len(vals), row), s=28 * counts, color=S.BLUE, alpha=0.75, zorder=3)
        for a, b in zip(vals, counts):
            ax[1].text(a + 0.16, row + 0.2, str(b), ha="left", fontsize=9, color=S.INK)
    ax[1].axvline(0, color=S.MUTED, lw=1)
    ax[1].set_yticks(range(len(diffs)), ["亚像素对齐\n不做细化", "均值加标准差\n代替中位数加 MAD", "全图用\n同一个 sigma"], fontsize=9)
    ax[1].set_ylim(-0.6, len(diffs) - 0.3)
    lim = max(3, int(max(abs(v).max() for v in diffs.values())) + 1)
    ax[1].set_xlim(-lim, lim)
    ax[1].set_xticks(range(-lim, lim + 1))
    ax[1].set_xlabel("一个种子里比完整流水线多检出的缺陷个数\n（圆点右上方的数字是种子个数）")
    ax[1].set_title("三项小改动：每个种子只差几个缺陷")
    S.clean(ax[1])
    ax[1].grid(axis="y", visible=False)
    S.save(fig, CH, "fig02_种子之间的波动")


# ---------------------------------------------------------------- 图 3：逐项去掉
ABLATIONS = {
    "完整流水线": {},
    "去掉对齐": {"align": "none"},
    "只对齐到整数像素": {"align": "integer"},
    "亚像素对齐不做细化": {"align": "single"},
    "对齐前加 Hanning 窗": {"window": True},
    "对齐不限制搜索范围": {"max_shift": SIZE // 2},
    "搜索范围只取 ±4 px": {"max_shift": 4},
    "去掉亮度归一化": {"norm_golden": False, "norm_test": False},
    "均值和标准差代替中位数和 MAD": {"stat": "mean"},
    "全图用同一个 sigma": {"stat": "flat"},
    "sigma 下限降到四分之一": {"floor": FLOOR0 / 4},
    "sigma 下限加倍": {"floor": FLOOR0 * 2},
    "不做面积过滤": {"min_area": 1},
    "面积门槛加倍": {"min_area": AREA0 * 2},
    "不屏蔽图像边界": {"border": 0},
    "检出坐标不变换回原图": {"back": 0.0},
    "坐标变换的符号写反": {"back": -1.0},
}


def fig_ablation(data):
    rows = {name: tally(cfg, data) for name, cfg in ABLATIONS.items()}
    res = {name: {**summary(r), "逐个种子的检出数和误报数都和完整流水线相同": bool(np.array_equal(r, rows["完整流水线"]))} for name, r in rows.items()}
    raws = [prepare(d)["raw"] for d in data]
    raw_all = np.concatenate([r.ravel() for r in raws])
    NUM["逐项消融"] = {"说明": CONDITION + "；每一行只改一处，其余和完整流水线相同，阈值不重新调", **res}
    NUM["sigma 的构成"] = {
        "说明": "完整流水线里，8 张良品逐像素算出的 1.4826 x MAD（还没有套下限），20 个种子合在一起",
        "中位数": round(float(np.median(raw_all)), 4),
        "5% 和 95% 分位": [round(float(np.percentile(raw_all, 5)), 4), round(float(np.percentile(raw_all, 95)), 4)],
        "低于参考实现所用下限的像素比例": round(float((raw_all < FLOOR0).mean()), 2),
    }
    # 坐标不变换回原图时，漏检集中在错位大的 die 上
    by_shift = {"小于 4 px": [0, 0], "不小于 4 px": [0, 0]}
    rel_all = []
    for d in data:
        p = prepare(d)
        preds = detect(p["z"], p["shifts"], back=0.0)
        size = np.hypot(*p["shifts"].T)
        rel_all.append(np.hypot(*(d["shifts"] - d["all_shifts"][0]).T))
        for i, (pr, t) in enumerate(zip(preds, d["truth"])):
            key = "小于 4 px" if size[i] < 4 else "不小于 4 px"
            by_shift[key][0] += len(match(pr, t)[1])
            by_shift[key][1] += len(t)
    rel_all = np.concatenate(rel_all)
    NUM["坐标系"] = {
        "说明": CONDITION + "；错位指待检 die 相对参考良品（第一张良品）的错位",
        "相对错位的中位数（px）": round(float(np.median(rel_all)), 2), "相对错位的最大值（px）": round(float(rel_all.max()), 2),
        "相对错位超过 4 px 的 die 比例": round(float((rel_all > 4).mean()), 3),
        "单轴相对错位的理论上限（px）": 7.0,
        "坐标不变换回原图时的检出（检出数，缺陷数），按估计出的错位大小分组": by_shift,
    }

    names = list(res)[::-1]
    y = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(12.2, 6.6), sharey=True, gridspec_kw={"width_ratios": [1.25, 1]}, layout="constrained")
    clip = lambda v: max(v, 0.003)  # noqa: E731
    for i, name in zip(y, names):
        r = res[name]
        color = S.RED if (r["检出率"] < 0.85 or r["每 die 误报"] > 0.3) else S.BLUE
        lo, hi = r["各种子检出率的最小和最大"]
        ax[0].plot([lo, hi], [i, i], color=color, lw=2.4, alpha=0.35, solid_capstyle="round")
        ax[0].plot(r["检出率"], i, "o", color=color, ms=6.5)
        mid, top = r["各种子每 die 误报的中位数和最大"]
        ax[1].plot([clip(mid), clip(top)], [i, i], color=color, lw=2.4, alpha=0.35, solid_capstyle="round")
        ax[1].plot(clip(r["每 die 误报"]), i, "s", color=color, ms=6.5)
    ax[0].axvline(0.85, color=S.RED, lw=1.1, ls="--")
    ax[0].set_yticks(y, names, fontsize=9.5)
    ax[0].set_xlim(-0.03, 1.03)
    ax[0].set_xlabel("检出率（圆点：20 个种子合计；横条：各种子的最小到最大）")
    ax[0].set_title("检出率，红色虚线是验收线 85%")
    ax[1].axvline(0.3, color=S.RED, lw=1.1, ls="--")
    ax[1].set_xscale("log")
    ax[1].set_xticks([0.003, 0.01, 0.1, 1, 10], ["<0.005", "0.01", "0.1", "1", "10"])
    ax[1].minorticks_off()
    ax[1].set_xlabel("每 die 误报（方块：合计；横条：各种子的中位数到最大）")
    ax[1].set_title("每 die 误报（对数轴），红色虚线是验收线 0.3")
    for a in ax:
        S.clean(a)
    S.save(fig, CH, "fig03_逐项消融")


# ---------------------------------------------------------------- 图 4：同一步在良品和待检图上各管什么
def fig_halves(data):
    cases = {
        "完整流水线": {},
        "良品不对齐，待检图对齐": {"align_golden": False},
        "良品对齐，待检图不对齐": {"align_test": False},
        "良品和待检图都不对齐": {"align": "none"},
        "良品不归一化，待检图归一化": {"norm_golden": False},
        "良品归一化，待检图不归一化": {"norm_test": False},
        "良品和待检图都不归一化": {"norm_golden": False, "norm_test": False},
        "不对齐也不归一化": {"align": "none", "norm_golden": False, "norm_test": False},
    }
    res = {}
    slopes, contrast = {}, {}
    for name, cfg in cases.items():
        s = summary(tally(cfg, data))
        preps = [prepare(d, **cfg) for d in data]
        res[name] = {"检出率": s["检出率"], "每 die 误报": s["每 die 误报"], "误报为 0 的种子数": s["误报为 0 的种子数"],
                     "MAD 估出的 sigma 中位数": round(float(np.median(np.concatenate([p["raw"].ravel() for p in preps]))), 4),
                     "超过阈值的像素比例": float(f"{np.mean([(p['z'] > K0).mean() for p in preps]):.3g}")}
        if name in ("完整流水线", "良品和待检图都不对齐"):
            fit = [float(np.polyfit(a.ravel(), p["center"].ravel(), 1)[0]) for p in preps for a in p["aligned"]]
            slopes[name] = [round(float(np.median(fit)), 3), round(float(np.percentile(fit, 5)), 3), round(float(np.percentile(fit, 95)), 3)]
            contrast[name] = round(float(np.mean([p["center"].std() for p in preps])), 3)
    NUM["良品和待检图各管什么"] = {
        "说明": CONDITION + "；每一行只对良品、只对待检图或者对两者同时去掉一步，阈值不重新调",
        "结果": res,
        "亮度归一化拟合出的斜率（中位数，5% 分位，95% 分位）": slopes,
        "模板的对比度（全图标准差，各种子平均）": contrast,
    }

    d, pick = EXAMPLE["d"], EXAMPLE["pick"]
    show = ["完整流水线", "良品对齐，待检图不对齐", "良品和待检图都不对齐", "良品归一化，待检图不归一化", "良品和待检图都不归一化"]
    titles = ["完整流水线", "只有待检图不对齐", "良品和待检图都不对齐", "只有待检图不归一化", "良品和待检图都不归一化"]
    fig, ax = plt.subplots(1, 5, figsize=(15.5, 3.8), layout="constrained")
    for a, name, title in zip(ax, show, titles):
        p = prepare(d, **cases[name])
        dy, dx = p["shifts"][pick]
        im = a.imshow(p["z"][pick], cmap="magma", vmin=0, vmax=2.5 * K0)
        for cy, cx in d["truth"][pick]:
            a.plot((cx - dx) % SIZE, (cy - dy) % SIZE, "o", mfc="none", mec="white", ms=17, mew=1.3)
        a.set_title(title)
        a.set_xlabel(f"检出率 {100 * res[name]['检出率']:.0f}%，每 die 误报 {res[name]['每 die 误报']:.2f}", fontsize=9.5, color=S.INK)
        S.image_axes(a)
    cb = fig.colorbar(im, ax=ax[-1], fraction=0.046, pad=0.03, ticks=[0, K0, 2 * K0])
    cb.ax.set_yticklabels(["0", "阈值", "2 倍"], fontsize=8)
    S.save(fig, CH, "fig04_良品和待检图各管什么")


# ---------------------------------------------------------------- 图 5：对齐细看
def layout_without_mark(size: int = SIZE) -> np.ndarray:
    """和 数据生成.make_layout 相同的周期版图，只是不放对位标记。只用于分析“标记值多少”。"""
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    lines = ((x % 8) < 4).astype(np.float64)
    contacts = (((y % 12) - 6) ** 2 + ((x % 12) - 6) ** 2 <= 5).astype(np.float64)
    bus = ((y > 40) & (y < 48)).astype(np.float64)
    return gen._gaussian_blur(0.25 + 0.45 * lines + 0.25 * contacts * (1 - lines) + 0.2 * bus, 0.7)


def align_errors(data, roll: int = 0, **kw) -> np.ndarray:
    """每张图（除参考良品外的 7 张良品和 30 张待检）相对参考良品的估计误差 (dy, dx)。"""
    out = []
    for d in data:
        ref, s0 = d["golden"][0], d["all_shifts"][0]
        imgs = np.concatenate([d["golden"][1:], d["test"]])
        true = d["all_shifts"][1:] - s0
        if roll:
            ref = np.roll(ref, (roll, roll), (0, 1))
            imgs = np.roll(imgs, (roll, roll), (1, 2))
        out.append(np.array([estimate_shift(ref, im, **kw) for im in imgs]) - true)
    return np.concatenate(out)


def err_stats(err: np.ndarray) -> dict:
    norm = np.hypot(err[:, 0], err[:, 1])
    return {"单轴均方根误差（px）": round(rms(err), 3), "二维误差中位数（px）": round(float(np.median(norm)), 3),
            "二维误差最大值（px）": round(float(norm.max()), 2), "二维误差超过 1 px 的次数": int((norm > 1).sum()), "次数": int(len(norm))}


def fig_alignment(data):
    modes = {"只取整数峰": {"mode": "integer"}, "一次抛物线": {"mode": "single"}, "再细化一轮": {"mode": "full"},
             "加 Hanning 窗，一次抛物线": {"mode": "single", "window": True}, "加 Hanning 窗，再细化一轮": {"mode": "full", "window": True}}
    errs = {name: align_errors(data, **kw) for name, kw in modes.items()}
    unlimited = align_errors(data, mode="full", max_shift=SIZE // 2)
    worst_window = errs["加 Hanning 窗，再细化一轮"]
    worst = worst_window[np.argmax(np.hypot(worst_window[:, 0], worst_window[:, 1]))]

    # 带缺陷的 die 会不会把对齐带偏
    with_defect, without = [], []
    for d in data:
        ref, s0 = d["golden"][0], d["all_shifts"][0]
        for i, img in enumerate(d["test"]):
            e = np.array(estimate_shift(ref, img)) - (d["shifts"][i] - s0)
            (with_defect if d["truth"][i] else without).append(e)

    # 去掉对位标记
    gen.make_layout = layout_without_mark
    try:
        bare = [dataset(s) for s in SEEDS]
    finally:
        gen.make_layout = _LAYOUT
    no_mark = {"限制搜索范围": err_stats(align_errors(bare, mode="full")), "不限制搜索范围": err_stats(align_errors(bare, mode="full", max_shift=SIZE // 2))}

    # 加窗时，标记离图像中心多远
    win = np.hanning(SIZE)
    rolls = [0, 6, 12, 18, 24, 30]
    sweep = []
    for r in rolls:
        st = err_stats(align_errors(data, roll=r, mode="single", window=True))
        sweep.append({"标记中心的行列坐标": 18 + r, "二维 Hanning 窗在标记中心的权重": round(float(win[18 + r] ** 2), 3),
                      "二维误差超过 1 px 的次数": st["二维误差超过 1 px 的次数"], "二维误差中位数（px）": st["二维误差中位数（px）"],
                      "单轴均方根误差（px）": st["单轴均方根误差（px）"]})

    # 闭环：8 张良品里任取三张，A 到 B、B 到 C、A 到 C
    closure = {}
    for name in ("只取整数峰", "一次抛物线", "再细化一轮", "加 Hanning 窗，一次抛物线"):
        clo, tru = [], []
        for d in data:
            g, gs = d["golden"], d["all_shifts"][:8]
            est = {(i, j): np.array(estimate_shift(g[i], g[j], **modes[name])) for i in range(8) for j in range(i + 1, 8)}
            for (i, j), e in est.items():
                tru.append(e - (gs[j] - gs[i]))
                clo += [e + est[j, k] - est[i, k] for k in range(j + 1, 8)]
        clo, tru = np.array(clo), np.array(tru)
        closure[name] = {"闭环误差的单轴均方根（px）": round(rms(clo), 3), "除以根号 3": round(rms(clo) / math.sqrt(3), 3),
                         "两两对齐的真实单轴均方根误差（px）": round(rms(tru), 3), "闭环误差恰好为 0 的比例": round(float(np.mean(np.abs(clo) < 1e-9)), 2),
                         "三元组个数": int(len(clo))}
    NUM["对齐"] = {
        "说明": CONDITION + "；每个种子除参考良品外有 7 张良品和 30 张待检图，共 740 次对齐；误差用数据生成器记录的真实错位算，只用于分析",
        "各做法的误差": {name: err_stats(e) for name, e in errs.items()},
        "不限制搜索范围（再细化一轮）": {**err_stats(unlimited), "估计值和限制搜索范围时逐张相同": bool(np.array_equal(unlimited, errs["再细化一轮"]))},
        "加窗时误差最大的一次（dy, dx）": [round(float(worst[0]), 2), round(float(worst[1]), 2)],
        "版图的周期（px）": {"竖线，x 方向": 8, "接触孔，x 和 y 方向": 12, "x 方向合起来": math.lcm(8, 12)},
        "有缺陷和无缺陷的待检 die 的单轴均方根误差（px）": {"有缺陷": round(rms(with_defect), 3), "无缺陷": round(rms(without), 3)},
        "把对位标记从版图里去掉": no_mark,
        "加窗时标记位置的影响（两张图一起循环滚动，一次抛物线）": sweep,
        "闭环误差": {"说明": "每个种子的 8 张良品两两对齐，再取所有三元组算 A到B + B到C - A到C", **closure},
    }

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.15, 1.1, 0.9]}, layout="constrained")
    grid = np.linspace(0, 1.2, 241)
    for name, color, ls in (("只取整数峰", S.MUTED, "-"), ("一次抛物线", S.ORANGE, "-"), ("再细化一轮", S.BLUE, "-"), ("加 Hanning 窗，再细化一轮", S.RED, "--")):
        norm = np.hypot(errs[name][:, 0], errs[name][:, 1])
        ax[0].plot(grid, [(norm <= g).mean() for g in grid], ls, color=color, lw=1.9, label=name)
    ax[0].set_xlabel("二维对齐误差（px）")
    ax[0].set_ylabel("误差不超过横轴值的图像比例")
    ax[0].set_ylim(0, 1.02)
    ax[0].set_title("四种做法的对齐误差分布（各 740 次）")
    ax[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[0])

    pos = [s["标记中心的行列坐标"] for s in sweep]
    ax[1].bar(pos, [s["二维误差超过 1 px 的次数"] for s in sweep], width=3.4, color=S.RED, alpha=0.75)
    for s in sweep:
        ax[1].text(s["标记中心的行列坐标"] - 2.1, s["二维误差超过 1 px 的次数"] + 0.4, str(s["二维误差超过 1 px 的次数"]), ha="center", fontsize=9, color=S.RED)
    ax[1].set_xticks(pos, [f"{p}\n{s['二维 Hanning 窗在标记中心的权重']:.2f}" for p, s in zip(pos, sweep)], fontsize=8.5)
    ax[1].set_xlabel("标记中心的坐标（上）和窗在该处的权重（下）")
    ax[1].set_ylabel("740 次里误差超过 1 px 的次数", color=S.RED)
    ax[1].set_ylim(0, max(s["二维误差超过 1 px 的次数"] for s in sweep) * 1.25 + 1)
    twin = ax[1].twinx()
    twin.plot(pos, [s["二维误差中位数（px）"] for s in sweep], "o-", color=S.BLUE, lw=1.8, ms=5)
    base_med = err_stats(errs["一次抛物线"])["二维误差中位数（px）"]
    twin.axhline(base_med, color=S.BLUE, lw=1, ls=":")
    twin.text(pos[-1] + 1.5, base_med - 0.014, "虚线：不加窗时的中位数", color=S.BLUE, fontsize=8.5, va="top", ha="right")
    twin.set_ylabel("误差中位数（px）", color=S.BLUE)
    twin.set_ylim(0, 0.6)
    ax[1].set_title("加窗：标记越靠近角落越差")
    S.clean(ax[1])
    ax[1].grid(axis="x", visible=False)
    twin.spines["top"].set_visible(False)

    for (name, c), color in zip(closure.items(), (S.MUTED, S.ORANGE, S.BLUE, S.RED)):
        ax[2].plot(c["两两对齐的真实单轴均方根误差（px）"], c["除以根号 3"], "o", color=color, ms=8, label=name)
    ax[2].plot([0, 0.4], [0, 0.4], color=S.MUTED, lw=1, ls="--")
    ax[2].set_xlim(0, 0.4)
    ax[2].set_ylim(0, 0.4)
    ax[2].set_aspect("equal")
    ax[2].set_xlabel("真实误差（单轴均方根，px）")
    ax[2].set_ylabel("闭环误差的均方根 / 1.73（px）")
    ax[2].set_title("不读真值也能估出对齐误差")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax[2])
    S.save(fig, CH, "fig05_对齐误差")


# ---------------------------------------------------------------- 图 6：工作点曲线和漏检
KS = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 9, 10, 12, 15, 20, 30]
VARIANTS = {
    "完整流水线": {},
    "均值和标准差代替中位数和 MAD": {"stat": "mean"},
    "全图用同一个 sigma": {"stat": "flat"},
    "sigma 下限降到四分之一": {"floor": FLOOR0 / 4},
    "亚像素对齐不做细化": {"align": "single"},
    "只对齐到整数像素": {"align": "integer"},
    "对齐前加 Hanning 窗": {"window": True},
    "去掉亮度归一化": {"norm_golden": False, "norm_test": False},
    "去掉对齐": {"align": "none"},
}


def sweep_threshold(preps, data, min_area=AREA0, ks=KS):
    out = []
    for k in ks:
        tp = fp = fn = 0
        for d, p in zip(data, preps):
            a, b, c = score(detect(p["z"], p["shifts"], k=k, min_area=min_area), d["truth"])
            tp, fp, fn = tp + a, fp + b, fn + c
        out.append((tp / (tp + fn), fp / (30 * len(data))))
    return out


def frontier(curve):
    """只保留有效的工作点：同一条曲线上没有另一个点检出率不低于它、误报又不高于它。按误报从小到大排。"""
    pts = sorted(set(curve), key=lambda cf: (cf[1], -cf[0]))
    out, best = [], -1.0
    for c, f in pts:
        if c > best:
            out.append((c, f))
            best = c
    return out


def best_at(curve, limit: float) -> float:
    ok = [c for c, f in curve if f <= limit]
    return round(max(ok), 3) if ok else 0.0


def fig_operating(data):
    curves = {}
    full_preps = None
    for name, cfg in VARIANTS.items():
        preps = [prepare(d, **cfg) for d in data]
        curves[name] = sweep_threshold(preps, data)
        if name == "完整流水线":
            full_preps = preps
    at_default = {name: [round(c[KS.index(K0)][0], 3), round(c[KS.index(K0)][1], 2)] for name, c in curves.items()}
    areas = (1, 3, 6)
    ks_area = [k for k in KS if k <= 2 * K0]
    by_area = {a: sweep_threshold(full_preps, data, min_area=a, ks=ks_area) for a in areas}

    # 不做面积过滤时，误报是什么
    compose = {}
    for mult in (1.0, 10 / 6):
        near = far = 0
        for d, p in zip(data, full_preps):
            preds = detect(p["z"], p["shifts"], k=K0 * mult, min_area=1)
            for pr, t in zip(preds, d["truth"]):
                used, _ = match(pr, t)
                for i, q in enumerate(pr):
                    if i not in used:
                        dist = min((math.hypot(q[0] - a, q[1] - b) for a, b in t), default=1e9)
                        near, far = near + (dist <= 10), far + (dist > 10)
        compose[f"阈值为参考配置的 {mult:.2f} 倍"] = {"误报总数": int(near + far), "离某个真缺陷 10 px 以内": int(near), "其余": int(far)}

    NUM["重调阈值后的比较"] = {
        "说明": CONDITION + f"；面积门槛保持参考配置，只把阈值从参考配置的 {KS[0] / K0:.1f} 倍扫到 {KS[-1] / K0:.0f} 倍（共 {len(KS)} 档），取满足误报上限的最高检出率",
        "阈值不动时（检出率，每 die 误报）": at_default,
        "每 die 误报不超过 0.1 时的最高检出率": {name: best_at(c, 0.1) for name, c in curves.items()},
        "每 die 误报不超过 0.02 时的最高检出率": {name: best_at(c, 0.02) for name, c in curves.items()},
        "曲线（阈值从低到高，每点是检出率和每 die 误报）": {name: [[round(c, 3), round(f, 3)] for c, f in cv] for name, cv in curves.items()},
    }
    NUM["阈值和面积门槛"] = {
        "说明": CONDITION + "；完整流水线，只改阈值和面积门槛。阈值写成参考配置的倍数，面积门槛取报告模板里列的 1、3、6",
        "阈值倍数": [round(k / K0, 2) for k in ks_area],
        **{f"min_area={a}（检出率，每 die 误报）": [[round(c, 3), round(f, 3)] for c, f in by_area[a]] for a in areas},
        "不做面积过滤时误报的构成": compose,
    }

    # 漏检分析
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    recs = []
    for d, p in zip(data, full_preps):
        preds = detect(p["z"], p["shifts"])
        for i, (pr, t) in enumerate(zip(preds, d["truth"])):
            _, used_t = match(pr, t)
            dy, dx = p["shifts"][i]
            mask = p["z"][i] > K0
            mask[:BORDER0, :] = mask[-BORDER0:, :] = False
            mask[:, :BORDER0] = mask[:, -BORDER0:] = False
            comps = components(mask)
            for j, e in enumerate(d["info"][i]):
                near = (yy - (e["cy"] - dy)) ** 2 + (xx - (e["cx"] - dx)) ** 2 <= RADIUS ** 2
                over = int((mask & near).sum())
                big = any(len(ys) >= AREA0 and near[ys, xs].any() for ys, xs in comps)
                other = min((math.hypot(e["cy"] - o["cy"], e["cx"] - o["cx"]) for jj, o in enumerate(d["info"][i]) if jj != j), default=1e9)
                shared = False
                for jj, o in enumerate(d["info"][i]):
                    if jj == j:
                        continue
                    near_o = (yy - (o["cy"] - dy)) ** 2 + (xx - (o["cx"] - dx)) ** 2 <= RADIUS ** 2
                    shared = shared or any(len(ys) >= AREA0 and near[ys, xs].any() and near_o[ys, xs].any() for ys, xs in comps)
                recs.append({"kind": e["kind"], "hit": j in used_t, "contrast": e["contrast"], "over": over, "big": big, "other": other, "shared": shared})
    kinds = {"particle": "颗粒", "scratch": "划痕", "missing": "图形缺失"}
    by_kind = {}
    for key, label in kinds.items():
        sel = [r for r in recs if r["kind"] == key]
        con = [r["contrast"] for r in sel]
        by_kind[label] = {"缺陷数": len(sel), "检出数": sum(r["hit"] for r in sel), "检出率": round(sum(r["hit"] for r in sel) / len(sel), 3),
                          "对比度的 5% 分位和中位数": [round(float(np.percentile(con, 5)), 2), round(float(np.median(con)), 2)],
                          "漏检数": len(sel) - sum(r["hit"] for r in sel),
                          "对比度低于 0.15 的个数": int(sum(c < 0.15 for c in con))}
    edges = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 10]
    labels = ["0.05 以下", "0.05 到 0.1", "0.1 到 0.15", "0.15 到 0.2", "0.2 到 0.3", "0.3 以上"]
    by_contrast = {}
    for lab, lo, hi in zip(labels, edges[:-1], edges[1:]):
        sel = [r for r in recs if lo <= r["contrast"] < hi]
        by_contrast[lab] = [sum(r["hit"] for r in sel), len(sel)]
    miss = [r for r in recs if not r["hit"]]
    cause = {"没有任何像素超过阈值": sum(r["over"] == 0 for r in miss),
             "有像素超过阈值，但连成的块不到面积门槛": sum(r["over"] > 0 and not r["big"] for r in miss),
             "有够大的块，但没有和这个真值配上": sum(r["big"] for r in miss)}
    merged = [r for r in miss if r["big"]]
    close = [r for r in recs if r["other"] < 8]
    NUM["漏检分析"] = {
        "说明": CONDITION + "；完整流水线。对比度指缺陷在成像之后（模糊、乘增益之后，加噪声之前）改变图像灰度的最大幅度，版图灰度范围约 0.15 到 0.95；"
              "缺陷类型和对比度来自对数据生成器的旁路记录，只用于分析",
        "缺陷总数": len(recs), "漏检总数": len(miss),
        "按类型": by_kind,
        "按对比度（检出数，缺陷数）": by_contrast,
        "漏检按原因分": cause,
        "第三类漏检里，8 px 之内有另一个缺陷的个数": int(sum(r["other"] < 8 for r in merged)),
        "第三类漏检里，和另一个缺陷落在同一个连通块上的个数": int(sum(r["shared"] for r in merged)),
        "8 px 之内有另一个缺陷的缺陷（个数，其中漏检）": [len(close), int(sum(not r["hit"] for r in close))],
        "图形缺失类缺陷改变灰度的幅度（按数据生成器的公式 0.1 - 0.8 x 原灰度，模糊之前）": {
            "对位标记周围的暗区（灰度 0.15）": round(0.1 - 0.8 * 0.15, 2), "线条之间的背景（灰度 0.25）": round(0.1 - 0.8 * 0.25, 2), "竖线（灰度 0.70）": round(0.1 - 0.8 * 0.70, 2)},
    }

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3), gridspec_kw={"width_ratios": [1.15, 1.15, 0.95]}, layout="constrained")
    clip = lambda v: max(v, 0.002)  # noqa: E731
    styles = {"完整流水线": (S.BLUE, "o-", 2.4), "均值和标准差代替中位数和 MAD": (S.GREEN, "s--", 1.5), "全图用同一个 sigma": (S.PURPLE, "^:", 1.5),
              "只对齐到整数像素": (S.ORANGE, "o-", 1.8), "对齐前加 Hanning 窗": (S.RED, "o-", 1.8), "去掉亮度归一化": (S.MUTED, "o-", 1.8)}
    for name, (color, fmt, lw) in styles.items():
        front = frontier(curves[name])
        ax[0].plot([clip(f) for _, f in front], [c for c, _ in front], fmt, color=color, lw=lw, ms=3.5, label=name)
    ax[0].set_xscale("log")
    ax[0].set_xlim(0.0015, 60)
    ax[0].set_xticks([0.002, 0.01, 0.1, 1, 10], ["≤0.002", "0.01", "0.1", "1", "10"])
    ax[0].minorticks_off()
    ax[0].axvline(0.1, color=S.MUTED, lw=1, ls="--")
    ax[0].set_xlabel("每 die 误报（对数轴）")
    ax[0].set_ylabel("检出率")
    ax[0].set_ylim(0, 1.03)
    ax[0].set_title("只扫阈值：哪些改动能靠调阈值补回来")
    ax[0].text(0.11, 0.05, "误报上限 0.1", color=S.MUTED, fontsize=8.5, ha="left")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[0])

    twin = ax[1].twinx()
    mult = [k / K0 for k in ks_area]
    for a, ls in zip(areas, (":", "--", "-")):
        ax[1].plot(mult, [c for c, _ in by_area[a]], "o" + ls, color=S.BLUE, lw=1.7, ms=3.5, label=f"检出率，min_area={a}")
        twin.plot(mult, [clip(f) for _, f in by_area[a]], "s" + ls, color=S.RED, lw=1.7, ms=3.5, label=f"误报，min_area={a}")
    ax[1].set_xlabel("阈值（参考配置 = 1）")
    ax[1].set_ylabel("检出率", color=S.BLUE)
    ax[1].set_ylim(0, 1.03)
    twin.set_yscale("log")
    twin.set_ylim(0.0015, 60)
    twin.set_yticks([0.002, 0.01, 0.1, 1, 10], ["≤0.002", "0.01", "0.1", "1", "10"])
    twin.minorticks_off()
    twin.set_ylabel("每 die 误报（对数轴）", color=S.RED)
    ax[1].set_title("阈值和面积门槛怎样移动工作点")
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[1])
    twin.spines["top"].set_visible(False)

    x = np.arange(len(labels))
    hits = np.array([v[0] for v in by_contrast.values()])
    totals = np.array([v[1] for v in by_contrast.values()])
    ax[2].bar(x, hits, color=S.BLUE, alpha=0.85, label="检出")
    ax[2].bar(x, totals - hits, bottom=hits, color=S.RED, alpha=0.85, label="漏检")
    for i, (h, t) in enumerate(zip(hits, totals)):
        ax[2].text(i, t + 6, f"{h}/{t}", ha="center", fontsize=9)
    ax[2].set_xticks(x, ["<0.05", "0.05\n到 0.1", "0.1\n到 0.15", "0.15\n到 0.2", "0.2\n到 0.3", ">0.3"], fontsize=8.5)
    ax[2].set_xlabel("缺陷对比度（成像后改变灰度的最大幅度）")
    ax[2].set_ylabel("缺陷个数")
    ax[2].set_ylim(0, totals.max() * 1.15)
    ax[2].set_title("漏检集中在对比度低的缺陷上")
    ax[2].legend(fontsize=9, loc="upper left")
    S.clean(ax[2])
    ax[2].grid(axis="x", visible=False)
    S.save(fig, CH, "fig06_工作点与漏检")


# ---------------------------------------------------------------- 只出数字：良品里混进坏 die、噪声水平
def numbers_contamination():
    """8 张良品里有 m 张在同一位置带一个颗粒；待检图在同一位置放一个探针缺陷，看能不能检出。"""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    cy = cx = 60.0
    spot = np.s_[57:64, 57:64]

    def add(img, shift, ref_shift, amp):
        py, px = cy + shift[0] - ref_shift[0], cx + shift[1] - ref_shift[1]
        return img + amp * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * 1.8 ** 2)), (py, px)

    ms = [0, 1, 2, 3, 4]
    hits = {"中位数 + MAD": [], "均值 + 标准差": []}
    sig = {"中位数 + MAD": [], "均值 + 标准差": []}
    total = 0
    for m in ms:
        hit = dict.fromkeys(hits, 0)
        s_at = {k: [] for k in hits}
        total = 0
        for seed in SEEDS[:10]:
            d = dataset(seed)
            s0 = d["all_shifts"][0]
            golden = d["golden"].copy()
            for j in range(1, m + 1):
                golden[j] = add(golden[j], d["all_shifts"][j], s0, 0.4)[0]
            clean = [i for i, t in enumerate(d["truth"]) if not t][:6]
            probes = [add(d["test"][i], d["shifts"][i], s0, 0.3) for i in clean]
            tests = np.stack([p[0] for p in probes])
            for name, stat in (("中位数 + MAD", "mad"), ("均值 + 标准差", "mean")):
                p = prepare(d, stat=stat, golden=golden, test=tests)
                preds = detect(p["z"], p["shifts"])
                hit[name] += sum(any(math.hypot(q[0] - py, q[1] - px) <= RADIUS for q in pr) for pr, (_, (py, px)) in zip(preds, probes))
                s_at[name].append(float(p["sigma"][spot].mean()))
            total += len(clean)
        for name in hits:
            hits[name].append(hit[name])
            sig[name].append(round(float(np.mean(s_at[name])), 3))
    NUM["良品里混进坏 die"] = {
        "说明": "种子 100 到 109，每个种子取 6 张无缺陷的待检 die，在对齐坐标系的同一位置各放一个探针缺陷（高斯斑，半径 1.8 px，幅度 0.3）；"
              "8 张良品里有 m 张在同一位置带一个幅度 0.4 的同样形状的斑；其余和完整流水线相同",
        "混进的张数 m": ms, "探针总数": total,
        "探针被检出的个数": hits,
        "探针位置的 sigma（7x7 邻域平均，已套下限）": sig,
    }


def numbers_radius(data):
    """检测结果不变，只改匹配半径；以及检出中心离真值有多远。"""
    radii = [1, 2, 3, 4, 6, 8]
    preds = []
    for d in data:
        p = prepare(d)
        preds.append(detect(p["z"], p["shifts"]))
    rows = {}
    for r in radii:
        tp = fp = fn = 0
        for d, pr in zip(data, preds):
            a, b, c = score(pr, d["truth"], radius=r)
            tp, fp, fn = tp + a, fp + b, fn + c
        rows[str(r)] = [round(tp / (tp + fn), 3), round(fp / (30 * len(data)), 2)]
    dist = []
    for d, pr in zip(data, preds):
        for p_i, t_i in zip(pr, d["truth"]):
            for q in p_i:
                near = min((math.hypot(q[0] - a, q[1] - b) for a, b in t_i), default=None)
                if near is not None and near <= 8:
                    dist.append(near)
    NUM["匹配半径"] = {
        "说明": CONDITION + "；完整流水线的检测结果不变，只改匹配半径",
        "半径（px）对应的检出率和每 die 误报": rows,
        "检出中心到最近真值的距离（px，只统计 8 px 以内的）": {"中位数": round(float(np.median(dist)), 2), "95% 分位": round(float(np.percentile(dist, 95)), 2), "个数": len(dist)},
    }


def numbers_setup():
    """讲义第 2 节引用的数据生成器参数和验收规则。能从代码里读的直接读，其余照 数据生成.py 和 tests/check_progress.py 抄录。"""
    ds = inspect.signature(gen.make_dataset).parameters
    cap = inspect.signature(gen._capture).parameters
    test = _load(next(p for p in (S.MAIN / CH / "tests" / "check_progress.py", S.REPO / CH / "tests" / "check_progress.py") if p.exists()), "check24")
    NUM["数据生成器的参数"] = {
        "说明": "来自 数据生成.py 的源码，不是测出来的",
        "图像边长（px）": gen.SIZE, "良品张数": ds["n_golden"].default, "待检张数": ds["n_test"].default, "待检 die 带缺陷的概率": ds["defect_ratio"].default,
        "每张带缺陷的 die 上的缺陷个数": [1, 3], "错位范围（每轴，px）": 3.5, "增益范围": [0.85, 1.15], "偏置范围": [-0.05, 0.05],
        "满量程电子数": cap["full_scale_e"].default, "读出噪声（电子）": cap["read_noise_e"].default,
        "版图自身的模糊和成像模糊（高斯 sigma，px）": [0.7, 1.0], "缺陷中心离图像边界的最小距离（px）": 10,
        "颗粒": {"半径（px）": [1.2, 2.2], "幅度": [0.35, 0.6], "极性": "亮暗各半"},
        "划痕": {"长度（px）": [6, 10], "幅度": [0.3, 0.5], "极性": "只有亮的"},
        "图形缺失": {"边长（px）": [3, 5], "做法": "块内灰度变成 0.2 倍再加 0.1"},
        "对位标记": {"中心（行，列）": [18, 18], "十字的臂长（px）": 8, "周围暗区的半边长（px）": 10},
    }
    NUM["验收规则"] = {"说明": "来自 tests/check_progress.py", "种子": list(test.SEEDS), "匹配半径（px）": test.RADIUS,
                   "检出率下限": 0.85, "每 die 误报上限": 0.3, "单次运行时间上限（秒）": 30}


def numbers_noise():
    full_well, read = 2000.0, 6.0
    levels = {"暗区（灰度 0.15）": 0.15, "线条之间的背景（灰度 0.25）": 0.25, "竖线（灰度 0.70）": 0.70, "对位标记（灰度 0.95）": 0.95}
    NUM["噪声水平"] = {
        "说明": "按数据生成器的成像模型（满量程 2000 个电子，读出噪声 6 个电子，增益取 1）算出的单像素噪声标准差，灰度单位",
        "噪声标准差": {name: round(math.sqrt(v * full_well + read ** 2) / full_well, 4) for name, v in levels.items()},
        "信噪比": {name: round(v * full_well / math.sqrt(v * full_well + read ** 2), 1) for name, v in levels.items()},
    }


def main() -> None:
    S.setup()
    plt.rcParams["svg.hashsalt"] = "ch24"  # SVG 里的元素 id 固定下来，重复运行输出逐字节相同
    data = [dataset(s) for s in SEEDS]
    check_equivalence(data)
    numbers_noise()
    numbers_setup()
    for fn in (fig_stages, fig_seeds, fig_ablation, fig_halves, fig_alignment, fig_operating):
        fn(data)
        print("完成", fn.__name__, flush=True)
    numbers_contamination()
    numbers_radius(data)
    NUM["常数"] = {"根号 3": round(math.sqrt(3), 3)}
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
