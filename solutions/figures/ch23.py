"""第 23 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch23.py

输出：<主分支>/23_半导体检测领域知识/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。
全部是合成的缺陷坐标，几何和统计用本章作业的参考实现计算。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

CH = "23_半导体检测领域知识"
mod = S.load_solution(CH, "23-1_晶圆图.py")
NUM: dict = {}
D_WAFER, R_WAFER, EE = 300.0, 150.0, 3.0
WAFER_CM2 = math.pi * R_WAFER ** 2 / 100.0


# ---------------------------------------------------------------- 数据与工具
def disk(rng, n, R=R_WAFER):
    """圆内按面积均匀分布的 n 个点。"""
    r = R * np.sqrt(rng.random(n))
    t = rng.random(n) * 2 * np.pi
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1).reshape(-1, 2)


def polar(r, t):
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def inside(pts, R=R_WAFER):
    return pts[np.hypot(pts[:, 0], pts[:, 1]) <= R]


def code(ids):
    return (ids[:, 0] + 4096) * 8192 + (ids[:, 1] + 4096)


class Layout:
    """一种 die 排布：完整 die 的列表，以及把缺陷点数到每个完整 die 上。"""

    def __init__(self, w, h, off, d=D_WAFER, ee=EE):
        self.w, self.h, self.off, self.d, self.ee = float(w), float(h), (float(off[0]), float(off[1])), d, ee
        self.grid = mod.die_grid(d, self.w, self.h, ee, self.off)
        codes = code(self.grid[:, :2].astype(np.int64))
        self.order = np.argsort(codes)
        self.codes = codes[self.order]
        self.K = len(codes)
        self.area_cm2 = self.w * self.h / 100.0

    def counts(self, pts):
        """每个完整 die 上的缺陷数，顺序和 self.grid 一致。"""
        out = np.zeros(self.K, dtype=np.int64)
        if len(pts) == 0:
            return out
        c = code(mod.assign_to_die(pts, self.w, self.h, self.off))
        idx = np.minimum(np.searchsorted(self.codes, c), self.K - 1)
        ok = self.codes[idx] == c
        out[self.order] = np.bincount(idx[ok], minlength=self.K)
        return out

    def yield_of(self, pts):
        return float((self.counts(pts) == 0).mean())


def fast_count(w, h, ox, oy, d=D_WAFER, ee=EE, strict=False):
    """完整 die 数的向量化算法，只用来扫大量偏移；结果在 main 里和参考实现核对。"""
    R = d / 2.0 - ee

    def far2(o, size):
        idx = np.arange(int(np.floor((-R - o) / size)) - 1, int(np.ceil((R - o) / size)) + 2)
        a = o + idx * size
        b = a + size
        return np.maximum(a * a, b * b)

    s = far2(ox, w)[None, :] + far2(oy, h)[:, None]
    return int((s < R * R).sum() if strict else (s <= R * R).sum())


def partial_count(lay):
    """和晶圆（半径 d/2）有重叠、但不是完整 die 的格子数。"""
    R = lay.d / 2.0
    n = 0
    cs = np.arange(int(np.floor((-R - lay.off[0]) / lay.w)) - 1, int(np.ceil((R - lay.off[0]) / lay.w)) + 2)
    rs = np.arange(int(np.floor((-R - lay.off[1]) / lay.h)) - 1, int(np.ceil((R - lay.off[1]) / lay.h)) + 2)
    x0 = lay.off[0] + cs * lay.w
    y0 = lay.off[1] + rs * lay.h
    px = np.clip(0.0, x0, x0 + lay.w)
    py = np.clip(0.0, y0, y0 + lay.h)
    touch = px[None, :] ** 2 + py[:, None] ** 2 < R * R
    n = int(touch.sum()) - lay.K
    cells = [(x0[j], y0[i]) for i in range(len(rs)) for j in range(len(cs)) if touch[i, j]]
    return n, cells


def neighbors(lay):
    keys = {(int(c), int(r)) for c, r in lay.grid[:, :2]}
    lr = sum(((c - 1, r) not in keys) or ((c + 1, r) not in keys) for c, r in keys)
    four = sum(any(k not in keys for k in ((c - 1, r), (c + 1, r), (c, r - 1), (c, r + 1))) for c, r in keys)
    return lr, four


def clustered(rng, mean_total, mu_c=8.0, sigma=6.0):
    """成簇的缺陷：簇中心均匀分布，每簇个数服从均值 mu_c 的泊松分布，簇内按高斯散开。"""
    Rp = R_WAFER + 3 * sigma
    n_par = rng.poisson(mean_total / mu_c * (Rp / R_WAFER) ** 2)
    par = disk(rng, n_par, Rp)
    size = rng.poisson(mu_c, n_par)
    pts = np.repeat(par, size, axis=0) + rng.normal(scale=sigma, size=(int(size.sum()), 2))
    return inside(pts)


def wilson(hit, n, z=1.96):
    p = hit / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - half, 3), round(c + half, 3)]


def hyper_pmf(M, D, n):
    """M 个里有 D 个目标，不放回抽 n 个，抽到 x 个目标的概率。"""
    tot = math.comb(M, n)
    return np.array([math.comb(D, x) * math.comb(M - D, n - x) / tot if 0 <= n - x <= M - D else 0.0 for x in range(min(D, n) + 1)])


def r3(v):
    return round(float(v), 3)


def pct_range(a, lo=5, hi=95):
    return [round(float(np.percentile(a, lo)), 3), round(float(np.percentile(a, hi)), 3)]


# ---------------------------------------------------------------- 画晶圆
def draw_wafer(ax, d=D_WAFER, ee=EE, notch=True):
    R = d / 2.0
    ax.add_patch(plt.Circle((0, 0), R, fill=False, ec=S.INK, lw=1.1, zorder=4))
    if ee:
        ax.add_patch(plt.Circle((0, 0), R - ee, fill=False, ec=S.MUTED, lw=0.7, ls="--", zorder=4))
    if notch:
        ax.plot([0], [-R + 2.5], marker="^", ms=5, color=S.INK, zorder=5)
    ax.set_xlim(-R * 1.05, R * 1.05)
    ax.set_ylim(-R * 1.05, R * 1.05)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def draw_grid(ax, lay, color="#c4c8cf", lw=0.4):
    """完整 die 的网格线：每一行、每一列各画两条线段，比逐个画矩形省得多。"""
    g = lay.grid
    segs = []
    for row in np.unique(g[:, 1]):
        sel = g[g[:, 1] == row]
        xa, xb = sel[:, 2].min() - lay.w / 2, sel[:, 2].max() + lay.w / 2
        y = sel[0, 3]
        segs += [[(xa, y - lay.h / 2), (xb, y - lay.h / 2)], [(xa, y + lay.h / 2), (xb, y + lay.h / 2)]]
    for col in np.unique(g[:, 0]):
        sel = g[g[:, 0] == col]
        ya, yb = sel[:, 3].min() - lay.h / 2, sel[:, 3].max() + lay.h / 2
        x = sel[0, 2]
        segs += [[(x - lay.w / 2, ya), (x - lay.w / 2, yb)], [(x + lay.w / 2, ya), (x + lay.w / 2, yb)]]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, zorder=1))


def fill_dies(ax, lay, mask, color, alpha=0.55):
    rects = [Rectangle((cx - lay.w / 2, cy - lay.h / 2), lay.w, lay.h) for (_, _, cx, cy) in lay.grid[mask]]
    ax.add_collection(PatchCollection(rects, facecolor=color, edgecolor="none", alpha=alpha, zorder=2))


# ---------------------------------------------------------------- 图 1：缺陷数相同，良率不同
def pat_random(rng, n):
    return disk(rng, n)


def pat_center(rng, n):
    return inside(rng.normal(scale=35.0, size=(n, 2)))


def pat_cluster(rng, n):
    return inside(disk(rng, 1, 110.0)[0] + rng.normal(scale=8.0, size=(n, 2)))


def pat_scratch(rng, n):
    ang = rng.random() * np.pi
    c = disk(rng, 1, 60.0)[0]
    s = rng.uniform(-60.0, 60.0, n)
    d = np.array([np.cos(ang), np.sin(ang)])
    return inside(c + s[:, None] * d + rng.normal(scale=0.3, size=n)[:, None] * np.array([-d[1], d[0]]))


def pat_edge(rng, n):
    return polar(np.sqrt(rng.uniform(140.0 ** 2, 150.0 ** 2, n)), rng.random(n) * 2 * np.pi)


def fig_same_count():
    lay = Layout(10, 10, (-5, -5))
    pats = {"随机": pat_random, "中心集中": pat_center, "局部聚集": pat_cluster, "划痕": pat_scratch, "边缘环": pat_edge}
    n_def, trials = 200, 300
    hits, examples = {}, {}
    for i, (name, fn) in enumerate(pats.items()):
        rng = np.random.default_rng(100 + i)
        vals = []
        for t in range(trials):
            pts = fn(rng, n_def)
            c = lay.counts(pts)
            vals.append(int((c > 0).sum()))
            if t == 0:
                examples[name] = (pts, c > 0)
        hits[name] = np.array(vals)
    expect = lay.K * (1 - math.exp(-n_def / (math.pi * R_WAFER ** 2) * lay.w * lay.h))
    NUM["缺陷数相同良率不同"] = {
        "说明": f"300 mm 晶圆，10x10 mm 的 die，去边 3 mm，(0,0) 号 die 的中心在晶圆中心，完整 die {lay.K} 个；每片晶圆 {n_def} 个缺陷，"
              f"每种分布重复 {trials} 片；假设每个缺陷都致命，落在完整 die 之外的缺陷不计。"
              "随机：整片晶圆内按面积均匀。中心集中：二维高斯，标准差 35 mm。局部聚集：簇中心在半径 110 mm 内随机，簇内高斯标准差 8 mm。"
              "划痕：长 120 mm、宽度标准差 0.3 mm 的直线，方向随机，中点在半径 60 mm 内随机。边缘环：半径 140 到 150 mm 的环内按面积均匀",
        "完整 die 数": lay.K,
        "每片缺陷数": n_def,
        "被缺陷命中的完整 die 数（平均）": {k: round(float(v.mean()), 1) for k, v in hits.items()},
        "被缺陷命中的完整 die 数（5% 到 95% 分位）": {k: [int(np.percentile(v, 5)), int(np.percentile(v, 95))] for k, v in hits.items()},
        "良率（没被命中的完整 die 占比，平均）": {k: r3(1 - v.mean() / lay.K) for k, v in hits.items()},
        "随机分布下命中 die 数的泊松理论值 K·(1-exp(-A·n/晶圆面积))": round(expect, 1),
        "随机分布和局部聚集的良率之差": r3((hits["随机"].mean() - hits["局部聚集"].mean()) / lay.K),
    }

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.6), layout="constrained")
    for ax, name in zip(axes.ravel()[:5], pats):
        pts, mask = examples[name]
        draw_wafer(ax)
        draw_grid(ax, lay)
        fill_dies(ax, lay, mask, S.RED)
        ax.scatter(pts[:, 0], pts[:, 1], s=2.2, color=S.INK, linewidths=0, zorder=3)
        ax.set_title(f"{name}：{n_def} 个缺陷，命中 {int(mask.sum())} 个 die")
    ax = axes[1, 2]
    names = list(pats)[::-1]
    means = [hits[k].mean() for k in names]
    lo = [hits[k].mean() - np.percentile(hits[k], 5) for k in names]
    hi = [np.percentile(hits[k], 95) - hits[k].mean() for k in names]
    ax.barh(names, means, xerr=[lo, hi], color=S.RED, alpha=0.75, height=0.55, error_kw={"ecolor": S.INK, "lw": 1, "capsize": 3})
    for i, k in enumerate(names):
        ax.text(hits[k].mean() + 12, i, f"良率 {100 * (1 - hits[k].mean() / lay.K):.1f}%", va="center", fontsize=9.5)
    ax.set_xlim(0, 230)
    ax.set_xlabel(f"被命中的完整 die 数（共 {lay.K} 个，{trials} 片平均）")
    ax.set_title("同样 200 个缺陷，良率从 75% 到 97%")
    S.clean(ax)
    ax.grid(axis="y", visible=False)
    S.save(fig, CH, "fig01_缺陷数相同良率不同")


# ---------------------------------------------------------------- 图 2：一片晶圆放多少 die
def fig_die_count():
    sizes = [(5, 5), (10, 10), (20, 20), (26, 33)]
    steps = 20
    table, scans = {}, {}
    for w, h in sizes:
        scan = np.array([[fast_count(w, h, -w * i / steps, -h * j / steps) for i in range(steps)] for j in range(steps)])
        scans[(w, h)] = scan
        Re = R_WAFER - EE
        centered = len(mod.die_grid(D_WAFER, w, h, EE, (-w / 2, -h / 2)))
        corner = len(mod.die_grid(D_WAFER, w, h, EE, (0.0, 0.0)))
        assert centered == scan[steps // 2, steps // 2] and corner == scan[0, 0], "向量化计数和参考实现不一致"
        chk = np.random.default_rng(7)
        for _ in range(4):
            i, j = (int(v) for v in chk.integers(0, steps, 2))
            assert len(mod.die_grid(D_WAFER, w, h, EE, (-w * i / steps, -h * j / steps))) == scan[j, i], "向量化计数和参考实现不一致"
        area_est = math.pi * Re * Re / (w * h)
        table[f"{w}x{h}"] = {
            "按面积估计（去边后的圆面积除以 die 面积）": round(area_est, 1),
            "近似公式 pi·d^2/(4S) - pi·d/sqrt(2S)，d 取去边后的直径": round(math.pi * (2 * Re) ** 2 / (4 * w * h) - math.pi * 2 * Re / math.sqrt(2 * w * h), 1),
            "die 中心对准晶圆中心": centered,
            "die 的角点对准晶圆中心": corner,
            "扫 20x20 种偏移得到的最少和最多": [int(scan.min()), int(scan.max())],
            "最多比最少多出的比例": r3(scan.max() / scan.min() - 1),
            "面积估计比实际最多值高出的比例": r3(area_est / scan.max() - 1),
        }
    ees = [round(0.25 * i, 2) for i in range(41)]
    ee_curve = {"10x10": [len(mod.die_grid(D_WAFER, 10, 10, e, (-5.0, -5.0))) for e in ees],
                "26x33": [len(mod.die_grid(D_WAFER, 26, 33, e, (0.0, 0.0))) for e in ees]}
    lay10, lay26c, lay26k = Layout(10, 10, (-5, -5)), Layout(26, 33, (-13, -16.5)), Layout(26, 33, (0, 0))
    edge = {}
    for name, lay in (("10x10，die 中心对准", lay10), ("26x33，die 中心对准", lay26c), ("26x33，角点对准", lay26k)):
        lr, four = neighbors(lay)
        edge[name] = {"完整 die": lay.K, "左右至少缺一个完整邻居的 die": lr, "占比": r3(lr / lay.K),
                      "上下左右至少缺一个完整邻居的 die": four, "和晶圆有重叠的不完整 die": partial_count(lay)[0]}
    R3 = R_WAFER - EE
    g10 = mod.die_grid(D_WAFER + 40, 10, 10, 0.0, (-5.0, -5.0))
    r_center = np.hypot(g10[:, 2], g10[:, 3])
    r_near = np.hypot(np.abs(g10[:, 2]) - 5.0, np.abs(g10[:, 3]) - 5.0)
    ring_area = [(k + 1) ** 2 - k ** 2 for k in range(6)]
    NUM["一片晶圆放多少 die"] = {
        "说明": "300 mm 晶圆，去边 3 mm（单独说明的除外）；完整 die 指四个角点到圆心的距离都不超过可用半径，用作业的参考实现 die_grid 计数；"
              "偏移扫描把网格在一个 die 的宽和高内各移 20 档",
        "各种 die 尺寸（mm）": table,
        "去边宽度（mm）": ees,
        "完整 die 数随去边宽度的变化": ee_curve,
        "10x10 的 die 在去边 0、1、2、3、4、5 mm 时的完整 die 数": [ee_curve["10x10"][ees.index(float(e))] for e in range(6)],
        "26x33 的 die（角点对准）在去边 0、3、4、5 mm 时的完整 die 数": [ee_curve["26x33"][ees.index(float(e))] for e in (0, 3, 4, 5)],
        "边缘 die 和不完整 die": edge,
        "角点正好落在圆上的例子": {
            "说明": "10x10 的 die、角点对准、不去边时可用半径是 150 mm，角点 (90, 120) 到圆心的距离正好是 150",
            "按作业约定（距离 <= 半径算完整）": len(mod.die_grid(D_WAFER, 10, 10, 0.0, (0.0, 0.0))),
            "写成距离 < 半径": fast_count(10, 10, 0.0, 0.0, ee=0.0, strict=True),
        },
        "常见实现错误的后果": {
            "说明": "300 mm 晶圆、10x10 的 die、去边 3 mm、die 中心对准，正确的完整 die 数见上；径向密度一项按 6 个等宽圆环算",
            "正确的完整 die 数": lay10.K,
            "只判断 die 中心是否在可用圆内": int((r_center <= R3).sum()),
            "只要有一个角点在可用圆内就算": int((r_near <= R3).sum()),
            "忘了扣去边（按半径 150 mm 判断）": len(mod.die_grid(D_WAFER, 10, 10, 0.0, (-5.0, -5.0))),
            "径向密度忘了除以圆环面积时，均匀分布下最外环的计数是最内环的几倍": ring_area[-1] // ring_area[0],
        },
    }

    fig, ax = plt.subplots(1, 4, figsize=(15.5, 4.0), gridspec_kw={"width_ratios": [1, 1, 1.12, 1.3]}, layout="constrained")
    for a, lay, title in ((ax[0], lay26c, "die 中心对准晶圆中心"), (ax[1], lay26k, "四个 die 的公共角点对准中心")):
        draw_wafer(a)
        _, cells = partial_count(lay)
        keys = {(round(cx - lay.w / 2, 6), round(cy - lay.h / 2, 6)) for _, _, cx, cy in lay.grid}
        part = [Rectangle((x, y), lay.w, lay.h) for x, y in cells if (round(x, 6), round(y, 6)) not in keys]
        a.add_collection(PatchCollection(part, facecolor="#eef0f3", edgecolor="#c4c8cf", linewidths=0.5, zorder=1))
        full = [Rectangle((cx - lay.w / 2, cy - lay.h / 2), lay.w, lay.h) for _, _, cx, cy in lay.grid]
        a.add_collection(PatchCollection(full, facecolor="#cfe0ff", edgecolor=S.BLUE, linewidths=0.6, zorder=2))
        a.plot(0, 0, "+", color=S.RED, ms=9, mew=1.4, zorder=5)
        a.set_xlim(-185, 185)
        a.set_ylim(-185, 185)
        a.set_title(f"{title}：{lay.K} 个")
    scan = scans[(26, 33)]
    im = ax[2].imshow(scan, origin="lower", cmap="viridis", extent=(-0.5 / steps, 1 - 0.5 / steps, -0.5 / steps, 1 - 0.5 / steps), aspect="equal")
    ax[2].set_xticks([0, 0.25, 0.5, 0.75])
    ax[2].set_yticks([0, 0.25, 0.5, 0.75])
    ax[2].set_xlabel("网格沿 x 的偏移（die 宽的倍数）")
    ax[2].set_ylabel("沿 y 的偏移（die 高的倍数）")
    ax[2].set_title(f"26x33 的 die：偏移不同，{scan.min()} 到 {scan.max()} 个")
    cb = fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.03, ticks=range(int(scan.min()), int(scan.max()) + 1))
    cb.ax.tick_params(labelsize=8)
    for key, color, label in (("10x10", S.BLUE, "10x10 mm 的 die"), ("26x33", S.ORANGE, "26x33 mm 的 die")):
        y = np.array(ee_curve[key]) / ee_curve[key][0]
        ax[3].step(ees, y, where="post", color=color, lw=1.8, label=label)
    ax[3].plot(ees, [((R_WAFER - e) / R_WAFER) ** 2 for e in ees], color=S.MUTED, lw=1.2, ls="--", label="可用面积的比例")
    ax[3].set_xlabel("去边宽度（mm）")
    ax[3].set_ylabel("完整 die 数（相对不去边时）")
    ax[3].set_title("去边变宽，die 数一级一级往下掉")
    ax[3].legend(fontsize=9, loc="lower left")
    S.clean(ax[3])
    S.save(fig, CH, "fig02_一片晶圆放多少die")


# ---------------------------------------------------------------- 图 3：良率和缺陷密度
def fig_yield():
    lays = {"10x10": Layout(10, 10, (-5, -5)), "20x20": Layout(20, 20, (0, 0)), "26x33": Layout(26, 33, (0, 0))}
    D0s = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    rng = np.random.default_rng(1)
    sim = {k: [] for k in lays}
    spread = {k: [] for k in lays}
    for D0 in D0s:
        ys = {k: [] for k in lays}
        for _ in range(200):
            pts = disk(rng, rng.poisson(D0 * WAFER_CM2))
            for k, lay in lays.items():
                ys[k].append(lay.yield_of(pts))
        for k in lays:
            sim[k].append(float(np.mean(ys[k])))
            spread[k].append(float(np.std(ys[k])))
    theory = {k: [math.exp(-lay.area_cm2 * D0) for D0 in D0s] for k, lay in lays.items()}

    # 成簇的缺陷：只看 10x10 的 die
    lay = lays["10x10"]
    targets = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
    cl = {"每个 die 的平均缺陷数": [], "方差": [], "聚集参数 alpha（按均值和方差估计）": [], "实测良率": [], "泊松模型": [], "负二项模型": []}
    for m in targets:
        allc, ys = [], []
        for _ in range(100):
            c = lay.counts(clustered(rng, m * WAFER_CM2 / lay.area_cm2))
            allc.append(c)
            ys.append((c == 0).mean())
        c = np.concatenate(allc)
        mu, var = float(c.mean()), float(c.var())
        alpha = mu * mu / (var - mu)
        for key, v in zip(cl, (mu, var, alpha, np.mean(ys), math.exp(-mu), (1 + mu / alpha) ** (-alpha))):
            cl[key].append(r3(v))

    # 单片良率的波动
    one = {}
    for k in ("26x33", "10x10"):
        ys = np.array([lays[k].yield_of(disk(rng, rng.poisson(0.1 * WAFER_CM2))) for _ in range(2000)])
        y0 = math.exp(-lays[k].area_cm2 * 0.1)
        one[k] = {"完整 die 数": lays[k].K, "泊松模型的良率": r3(y0), "2000 片的平均良率": r3(ys.mean()), "单片良率的标准差": r3(ys.std()),
                  "单片良率的 5% 到 95% 分位": pct_range(ys), "按二项分布估的标准差 sqrt(Y(1-Y)/K)": r3(math.sqrt(y0 * (1 - y0) / lays[k].K)), "_ys": ys}
    hist = {k: one[k].pop("_ys") for k in one}
    NUM["良率和缺陷密度"] = {
        "说明": "300 mm 晶圆，去边 3 mm；致命缺陷在整片晶圆上按面积均匀随机，每片个数服从泊松分布；一个完整 die 上只要有一个缺陷就算坏；"
              "每个缺陷密度做 200 片。die 排布：10x10 为 die 中心对准，20x20 和 26x33 为角点对准",
        "晶圆面积（平方厘米）": round(WAFER_CM2, 1),
        "完整 die 数": {k: lay_.K for k, lay_ in lays.items()},
        "die 面积（平方厘米）": {k: lay_.area_cm2 for k, lay_ in lays.items()},
        "缺陷密度 D0（个每平方厘米）": D0s,
        "实测良率（200 片平均）": {k: [r3(v) for v in vals] for k, vals in sim.items()},
        "单片良率的标准差": {k: [r3(v) for v in vals] for k, vals in spread.items()},
        "泊松模型 exp(-A·D0)": {k: [r3(v) for v in vals] for k, vals in theory.items()},
        "实测和泊松模型的最大差": r3(max(abs(a - b) for k in lays for a, b in zip(sim[k], theory[k]))),
        "成簇的缺陷": dict({"说明": "10x10 的 die；簇中心均匀随机，每簇缺陷数服从均值 8 的泊松分布，簇内高斯标准差 6 mm；每档 100 片；"
                                "alpha 由全部 die 的缺陷数的均值 m 和方差 v 按 m^2/(v-m) 估计，负二项模型是 (1+m/alpha)^(-alpha)"}, **cl),
        "D0 取 0.1 时单片良率的波动（各 2000 片）": one,
    }

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.1), layout="constrained")
    for (k, color) in zip(lays, (S.BLUE, S.GREEN, S.ORANGE)):
        xs = np.geomspace(0.015, 2.6, 80)
        ax[0].plot(xs, np.exp(-lays[k].area_cm2 * xs), color=color, lw=1.4)
        ax[0].errorbar(D0s, sim[k], yerr=spread[k], fmt="o", color=color, ms=4.5, capsize=2.5, lw=1,
                       label=f"{k} mm，{lays[k].K} 个 die")
    ax[0].set_xscale("log")
    ax[0].set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2], ["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2"])
    ax[0].minorticks_off()
    ax[0].set_xlabel("致命缺陷密度 D0（个每平方厘米）")
    ax[0].set_ylabel("良率")
    ax[0].set_ylim(-0.03, 1.03)
    ax[0].set_title("点是实测，线是 exp(-A·D0)：die 越大掉得越早")
    ax[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    S.clean(ax[0])
    xs = np.linspace(0.05, 3.1, 80)
    ax[1].plot(xs, np.exp(-xs), color=S.MUTED, lw=1.4, ls="--", label="泊松模型 exp(-m)")
    ax[1].plot(cl["每个 die 的平均缺陷数"], cl["实测良率"], "o-", color=S.RED, lw=1.8, ms=5, label="成簇缺陷的实测良率")
    ax[1].plot(cl["每个 die 的平均缺陷数"], cl["负二项模型"], "s:", color=S.PURPLE, lw=1.4, ms=4, label="负二项模型")
    ax[1].set_xlabel("每个 die 的平均缺陷数 m")
    ax[1].set_ylabel("良率")
    ax[1].set_ylim(0, 1.03)
    ax[1].set_title("缺陷成簇时，泊松模型把良率估低了")
    ax[1].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    S.clean(ax[1])
    bins = np.linspace(0.2, 1.0, 81)
    for k, color in (("26x33", S.ORANGE), ("10x10", S.BLUE)):
        ax[2].hist(hist[k], bins=bins, color=color, alpha=0.7, label=f"{k} mm，{lays[k].K} 个 die，标准差 {100 * hist[k].std():.1f} 个点")
    ax[2].set_xlabel("单片晶圆的良率（D0 = 0.1）")
    ax[2].set_ylabel("片数（共 2000 片）")
    ax[2].set_title("工艺没变，die 少的晶圆良率自己就在晃")
    ax[2].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax[2])
    S.save(fig, CH, "fig03_良率和缺陷密度")


# ---------------------------------------------------------------- 图 4：抽样带来的波动
def fig_sampling():
    rng = np.random.default_rng(2)
    M = 500
    ns = list(range(10, 301, 10))
    review = {}
    bands = {}
    for D in (100, 10):
        lo, hi, zero = [], [], []
        for n in ns:
            pmf = hyper_pmf(M, D, n)
            cdf = np.cumsum(pmf)
            lo.append(int(np.searchsorted(cdf, 0.05)) / n)
            hi.append(int(np.searchsorted(cdf, 0.95)) / n)
            zero.append(float(pmf[0]))
        bands[D] = (lo, hi)
        pick = {}
        for n in (20, 50, 100, 200):
            i = ns.index(n)
            pick[str(n)] = {"估出的 DOI 个数的 5% 到 95% 分位": [round(lo[i] * M), round(hi[i] * M)], "一个 DOI 都没看到的概率": r3(zero[i])}
        review[f"{M} 个里有 {D} 个 DOI"] = pick
    pool = np.zeros(M, bool)
    pool[:10] = True
    check = float(np.mean([rng.choice(pool, 50, replace=False).sum() == 0 for _ in range(20000)]))

    lot = 25
    bad = [1, 2, 5, 12]
    detect = {k: [1 - math.comb(lot - k, n) / math.comb(lot, n) for n in range(1, lot + 1)] for k in bad}

    mean_total = 50
    pois = rng.poisson(mean_total, 20000)
    clus = np.array([len(clustered(rng, mean_total)) for _ in range(20000)])
    line = mean_total + 3 * math.sqrt(mean_total)
    NUM["抽样带来的波动"] = {
        "复查抽样": dict({"说明": f"检测报出 {M} 个候选，随机抽 n 个复查，按抽到的 DOI 比例乘以 {M} 估计 DOI 总数；超几何分布的精确值"}, **review),
        "复查抽样的模拟核对（500 个里 10 个 DOI，抽 50 个，20000 次里一个都没抽到的比例）": r3(check),
        "按批抽片": {
            "说明": f"一批 {lot} 片，其中 k 片有问题，随机抽 n 片检测，至少抽到一片有问题的概率；超几何分布的精确值",
            "抽 1、2、3、5、13 片": {f"{k} 片有问题": [r3(detect[k][n - 1]) for n in (1, 2, 3, 5, 13)] for k in bad},
            "每批抽 2 片、5 片有问题时，平均要到第几批才抽到": round(1 / detect[5][1], 1),
            "每批抽 2 片、12 片有问题时，平均要到第几批才抽到": round(1 / detect[12][1], 1),
        },
        "片间缺陷数的波动": {
            "说明": "每片平均 50 个缺陷，各 20000 片。独立随机：泊松分布。成簇：簇中心均匀随机，每簇个数服从均值 8 的泊松分布，簇内高斯标准差 6 mm",
            "按根号 N 画的上控制线（50 + 3·sqrt(50)）": round(line, 1),
            "独立随机": {"平均": round(float(pois.mean()), 1), "标准差": round(float(pois.std()), 1), "5% 到 95% 分位": [int(v) for v in np.percentile(pois, [5, 95])],
                     "超过上控制线的片的比例": float(f"{(pois > line).mean():.2g}")},
            "成簇": {"平均": round(float(clus.mean()), 1), "标准差": round(float(clus.std()), 1), "5% 到 95% 分位": [int(v) for v in np.percentile(clus, [5, 95])],
                   "超过上控制线的片的比例": r3((clus > line).mean())},
            "成簇时标准差是独立随机时的几倍": round(float(clus.std() / pois.std()), 1),
        },
    }

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.1), layout="constrained")
    for D, color in ((100, S.BLUE), (10, S.ORANGE)):
        lo, hi = bands[D]
        ax[0].fill_between(ns, lo, hi, step="mid", color=color, alpha=0.3, label=f"真实占比 {100 * D // M}%：估计值的 5% 到 95% 范围")
        ax[0].axhline(D / M, color=color, lw=1.4, ls="--")
    ax[0].set_xlabel(f"从 {M} 个候选里抽多少个复查")
    ax[0].set_ylabel("估出的 DOI 占比")
    ax[0].set_ylim(-0.01, 0.45)
    ax[0].set_title("复查抽得越少，DOI 占比估得越粗")
    ax[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax[0])
    xs = list(range(1, lot + 1))
    for k, color in zip(bad, (S.RED, S.ORANGE, S.GREEN, S.BLUE)):
        ax[1].plot(xs, detect[k], "o-", color=color, ms=3.5, lw=1.6, label=f"{lot} 片里 {k} 片有问题")
    ax[1].axvline(2, color=S.MUTED, lw=1, ls=":")
    ax[1].set_xlabel("每批随机抽几片检测")
    ax[1].set_ylabel("至少抽到一片有问题的概率")
    ax[1].set_ylim(0, 1.03)
    ax[1].set_title(f"每批抽 2 片：5 片有问题时抽到的概率 {100 * detect[5][1]:.0f}%")
    ax[1].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[1])
    bins = np.arange(0, 161, 4)
    ax[2].hist(pois, bins=bins, color=S.BLUE, alpha=0.7, label=f"独立随机，标准差 {pois.std():.1f}")
    ax[2].hist(clus, bins=bins, color=S.RED, alpha=0.55, label=f"成簇，标准差 {clus.std():.1f}")
    ax[2].axvline(line, color=S.INK, lw=1.2, ls="--")
    ax[2].text(line + 2, ax[2].get_ylim()[1] * 0.9, "50 + 3 倍根号 50", fontsize=9, color=S.INK)
    ax[2].set_xlabel("一片晶圆上的缺陷数（平均都是 50）")
    ax[2].set_ylabel("片数（共 20000 片）")
    ax[2].set_title("缺陷成簇时，片间波动远大于根号 N")
    ax[2].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    S.save(fig, CH, "fig04_抽样带来的波动")


# ---------------------------------------------------------------- 图 5：空间模式和两个径向特征
def sector_share(pts, n=12):
    th = np.arctan2(pts[:, 1], pts[:, 0])
    sec = np.bincount(((th + np.pi) / (2 * np.pi) * n).astype(int) % n, minlength=n)
    return float(sec.max() / max(len(pts), 1))


PATTERNS = {
    "随机": lambda rng, n: disk(rng, n, 147.0),
    "中心集中": lambda rng, n: rng.normal(scale=35.0, size=(n, 2)),
    "边缘环": lambda rng, n: polar(rng.uniform(132.0, 147.0, n), rng.random(n) * 2 * np.pi),
    "环形": lambda rng, n: polar(rng.uniform(70.0, 95.0, n), rng.random(n) * 2 * np.pi),
    "局部聚集": lambda rng, n: np.array([60.0, 55.0]) + rng.normal(scale=9.0, size=(n, 2)),
    "划痕": lambda rng, n: (lambda x: np.stack([x + rng.normal(scale=1.5, size=n), 0.6 * x + 20 + rng.normal(scale=1.5, size=n)], axis=1))(np.linspace(-90.0, 60.0, n)),
}


def fig_patterns():
    n, reps, bins, band = 400, 200, 6, 15.0
    avg = n / (math.pi * R_WAFER ** 2)
    stats, examples = {}, {}
    for i, (name, fn) in enumerate(PATTERNS.items()):
        rng = np.random.default_rng(300 + i)
        dens, er, sec = [], [], []
        for t in range(reps):
            p = fn(rng, n)
            dens.append(mod.radial_density(p, R_WAFER, bins) / avg)
            er.append(mod.edge_ratio(p, R_WAFER, band))
            sec.append(sector_share(p))
            if t == 0:
                examples[name] = p
        stats[name] = {"各环密度相对整片平均密度的倍数（由内到外）": [round(float(v), 2) for v in np.mean(dens, axis=0)],
                       "边缘比例": r3(np.mean(er)), "最密的 30 度扇区里的缺陷占比": r3(np.mean(sec))}

    small = {}
    rng = np.random.default_rng(310)
    for m in (30, 100, 400, 2000):
        inner, ers = [], []
        for _ in range(2000):
            p = disk(rng, m)
            inner.append(mod.radial_density(p, R_WAFER, bins)[0] / (m / (math.pi * R_WAFER ** 2)))
            ers.append(mod.edge_ratio(p, R_WAFER, band))
        inner = np.array(inner)
        small[str(m)] = {"最内环密度相对平均密度的倍数（5% 到 95% 分位）": [round(float(v), 2) for v in np.percentile(inner, [5, 95])],
                         "最内环一个缺陷都没有的概率": r3((inner == 0).mean()),
                         "最内环密度达到平均的 2 倍以上的概率": float(f"{(inner >= 2).mean():.2g}"),
                         "边缘比例的 5% 到 95% 分位": [round(float(v), 2) for v in np.percentile(ers, [5, 95])]}
    NUM["空间模式和径向特征"] = {
        "说明": f"半径 {R_WAFER:.0f} mm，分 {bins} 个等宽圆环，边缘比例取外圈 {band:.0f} mm；每种模式 {n} 个缺陷，重复 {reps} 次取平均；特征用作业的参考实现 radial_density 和 edge_ratio 计算。"
              "随机：半径 147 mm 内按面积均匀。中心集中：二维高斯，标准差 35 mm。边缘环：半径在 132 到 147 mm 之间均匀。环形：半径在 70 到 95 mm 之间均匀。"
              "局部聚集：中心在 (60, 55) mm，高斯标准差 9 mm。划痕：从 (-90, -34) 到 (60, 56) mm 的直线，离圆心最近约 17 mm，宽度标准差 1.5 mm",
        "各模式的特征": stats,
        "均匀分布的理论边缘比例": {"说明": "外圈 15 mm 的面积占比", "缺陷铺满半径 150 mm": r3(1 - (135 / 150) ** 2), "缺陷只到半径 147 mm": r3((147 ** 2 - 135 ** 2) / 147 ** 2)},
        "最内环的面积占整片的比例": r3(1 / bins ** 2),
        "30 个缺陷时最内环的期望缺陷数": round(30 / bins ** 2, 2),
        "缺陷个数少时的波动": dict({"说明": "缺陷在半径 150 mm 内按面积均匀随机，没有任何空间模式；每种个数重复 2000 片"}, **small),
    }

    fig, ax = plt.subplots(2, 6, figsize=(15.5, 5.6), gridspec_kw={"height_ratios": [1.25, 1]}, layout="constrained")
    centers = np.arange(bins)
    for j, name in enumerate(PATTERNS):
        p = examples[name]
        draw_wafer(ax[0, j])
        ax[0, j].add_patch(plt.Circle((0, 0), R_WAFER - band, fill=False, ec=S.ORANGE, lw=0.7, ls=":", zorder=4))
        ax[0, j].scatter(p[:, 0], p[:, 1], s=2.5, color=S.BLUE, linewidths=0, zorder=3)
        ax[0, j].set_title(name)
        d = stats[name]["各环密度相对整片平均密度的倍数（由内到外）"]
        ax[1, j].bar(centers, d, color=S.BLUE, alpha=0.8, width=0.75)
        ax[1, j].axhline(1, color=S.MUTED, lw=1, ls="--")
        ax[1, j].set_xticks(centers, [f"{int(25 * (k + 1))}" for k in centers], fontsize=8)
        ax[1, j].set_ylim(0, 9.5)
        ax[1, j].set_title(f"边缘比例 {stats[name]['边缘比例']:.2f}，扇区 {stats[name]['最密的 30 度扇区里的缺陷占比']:.2f}", fontsize=10)
        ax[1, j].set_xlabel("圆环外半径（mm）", fontsize=9)
        S.clean(ax[1, j])
        ax[1, j].grid(axis="x", visible=False)
    ax[1, 0].set_ylabel("密度 / 整片平均密度")
    S.save(fig, CH, "fig05_空间模式和径向特征")


# ---------------------------------------------------------------- 图 6：重复缺陷要按 shot 叠起来看
def fig_repeater():
    die = Layout(10, 10, (-5, -5))
    SW = SH = 20.0
    SOFF = np.array([-5.0, -5.0])
    local = np.array([13.7, 6.2])
    cell = 0.5
    Ru = R_WAFER - EE
    shots = mod.die_grid(D_WAFER + 4 * SW, SW, SH, 0.0, tuple(SOFF))
    sites = inside(shots[:, 2:] - np.array([SW / 2, SH / 2]) + local, Ru)
    n_rand = 300

    def fold(pts):
        return pts - SOFF - mod.assign_to_die(pts, SW, SH, tuple(SOFF)) * np.array([SW, SH])

    def peak(pts):
        loc = fold(pts)
        hist, _, _ = np.histogram2d(loc[:, 0], loc[:, 1], bins=[int(SW / cell), int(SH / cell)], range=[[0, SW], [0, SH]])
        return int(hist.max())

    def wafer(rng, rate):
        keep = rng.random(len(sites)) < rate
        rep = sites[keep] + rng.normal(scale=0.05, size=(int(keep.sum()), 2))
        return np.concatenate([disk(rng, n_rand, Ru), rep]), int(keep.sum())

    rng = np.random.default_rng(4)
    base = np.array([peak(disk(rng, n_rand, Ru)) for _ in range(2000)])
    rule = 5
    rates = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.6, 1.0]
    med, lo, hi, found = [], [], [], []
    for rate in rates:
        v = np.array([peak(wafer(rng, rate)[0]) for _ in range(300)])
        med.append(float(np.median(v)))
        lo.append(float(np.percentile(v, 5)))
        hi.append(float(np.percentile(v, 95)))
        found.append(float((v >= rule).mean()))
    rng = np.random.default_rng(5)
    ex, n_rep = wafer(rng, 0.1)
    ex_rand = disk(np.random.default_rng(6), len(ex), Ru)
    avg = len(ex) / (math.pi * R_WAFER ** 2)
    hit_dies = int((die.counts(sites) > 0).sum())
    NUM["重复缺陷"] = {
        "说明": f"300 mm 晶圆，10x10 mm 的 die，一个 shot 是 2x2 个 die（20x20 mm）；光罩缺陷在 shot 内的位置是 ({local[0]}, {local[1]}) mm，"
              f"凡是该位置落在可用半径 {Ru:.0f} mm 内的 shot 都印上（假设边缘不完整的 shot 也曝光），坐标带标准差 0.05 mm 的定位误差；"
              f"每个位置按检出率独立决定是否被报出；另有 {n_rand} 个随机缺陷。叠图把所有缺陷折回 shot 内坐标，按 {cell} mm 的格子计数，取最大格",
        "重复缺陷的位置数": int(len(sites)),
        "含重复缺陷的完整 die 数": hit_dies,
        "含重复缺陷的完整 die 占比": r3(hit_dies / die.K),
        "只有随机缺陷时叠图最大格的计数（2000 片）": {"最小": int(base.min()), "中位数": int(np.median(base)), "最大": int(base.max()),
                                      f"达到 {rule} 的片的比例": float(f"{(base >= rule).mean():.2g}")},
        "判定规则": f"叠图最大格的计数不少于 {rule}",
        "重复缺陷的检出率": rates,
        "叠图最大格的计数（中位数）": med,
        "叠图最大格的计数（5% 分位）": [round(v, 1) for v in lo],
        "按判定规则认出重复缺陷的概率（各 300 片）": [r3(v) for v in found],
        "示例晶圆（检出率 0.1）": {
            "被报出的重复缺陷个数": n_rep, "缺陷总数": int(len(ex)), "叠图最大格的计数": peak(ex),
            "各环密度相对平均密度的倍数": [round(float(v), 2) for v in mod.radial_density(ex, R_WAFER, 6) / avg],
            "边缘比例": r3(mod.edge_ratio(ex, R_WAFER, 15.0)),
            "对照：同样个数的纯随机缺陷的各环密度倍数": [round(float(v), 2) for v in mod.radial_density(ex_rand, R_WAFER, 6) / avg],
            "对照：同样个数的纯随机缺陷的边缘比例": r3(mod.edge_ratio(ex_rand, R_WAFER, 15.0)),
        },
    }

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.25]}, layout="constrained")
    draw_wafer(ax[0])
    ax[0].scatter(ex[:, 0], ex[:, 1], s=4, color=S.BLUE, linewidths=0, zorder=3)
    ax[0].set_title(f"晶圆图：{len(ex)} 个缺陷里有 {n_rep} 个是重复缺陷")
    loc = fold(ex)
    ax[1].scatter(loc[:, 0], loc[:, 1], s=7, color=S.BLUE, linewidths=0, alpha=0.75)
    ax[1].plot(local[0], local[1], "o", mfc="none", mec=S.RED, ms=20, mew=1.5)
    ax[1].axvline(10, color=S.MUTED, lw=0.8, ls="--")
    ax[1].axhline(10, color=S.MUTED, lw=0.8, ls="--")
    ax[1].set_xlim(0, SW)
    ax[1].set_ylim(0, SH)
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("shot 内 x（mm），虚线是 die 的分界")
    ax[1].set_ylabel("shot 内 y（mm）")
    ax[1].set_title(f"按 shot 叠起来：红圈处叠了 {peak(ex)} 个")
    ax[2].fill_between(rates, lo, hi, color=S.BLUE, alpha=0.25, label="有重复缺陷：5% 到 95% 范围")
    ax[2].plot(rates, med, "o-", color=S.BLUE, lw=1.8, ms=5, label="有重复缺陷：中位数")
    ax[2].axhspan(base.min(), base.max(), color=S.MUTED, alpha=0.22, label=f"只有随机缺陷：2000 片的范围（{base.min()} 到 {base.max()}）")
    ax[2].axhline(rule, color=S.RED, lw=1.2, ls="--", label=f"判定线：最大格不少于 {rule}")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xticks(rates, ["1%", "2%", "3%", "5%", "10%", "20%", "30%", "60%", "100%"], fontsize=8.5)
    ax[2].set_yticks([1, 2, 5, 10, 20, 50, 100, 200], ["1", "2", "5", "10", "20", "50", "100", "200"])
    ax[2].minorticks_off()
    ax[2].set_xlabel("单个重复缺陷被检出的概率")
    ax[2].set_ylabel("叠图最大格的计数")
    ax[2].set_title("检出率 10% 时，叠图的峰已经远高于随机基线")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    S.save(fig, CH, "fig06_重复缺陷按shot叠图")


# ---------------------------------------------------------------- 理论数字
def numbers_theory():
    """讲义里引用的换算：像素数、噪声超限个数、置信区间。"""
    area_mm2 = math.pi * R_WAFER ** 2
    tail5 = math.erfc(5 / math.sqrt(2))
    px = {f"{p} 微米": area_mm2 / (p * 1e-3) ** 2 for p in (1.0, 0.5)}
    NUM["一片晶圆有多少像素"] = {
        "说明": "300 mm 晶圆的面积除以像素面积；像素尺寸指折算到晶圆面的大小，1 微米和 0.5 微米只是举例，不代表任何具体设备",
        "晶圆面积（平方毫米）": round(area_mm2),
        "像素数": {k: float(f"{v:.3g}") for k, v in px.items()},
        "像素尺寸减半后像素数变成几倍": 4,
        "高斯噪声、sigma 准确已知时单像素超过 5 sigma 的概率": float(f"{tail5:.2g}"),
        "按上面的概率，1 微米像素时一片晶圆上的期望超限像素数": float(f"{px['1.0 微米'] * tail5:.2g}"),
    }
    NUM["误报率的算法"] = {
        "说明": "举例：一片晶圆报出 500 个候选，复查后 50 个是 DOI，其余是误报",
        "报出": 500, "DOI": 50, "误报率（误报占报出的比例）": r3(450 / 500),
    }
    NUM["置信区间示例"] = {
        "说明": "Wilson 95% 置信区间",
        "64 个 die 里 27 个好": {"良率": r3(27 / 64), "区间": wilson(27, 64)},
        "64 个 die 里 32 个好": {"良率": r3(32 / 64), "区间": wilson(32, 64)},
        "复查 50 个里 10 个是 DOI": {"占比": r3(10 / 50), "区间": wilson(10, 50)},
    }


def main() -> None:
    S.setup()
    plt.rcParams["svg.hashsalt"] = CH  # 让 SVG 里的元素编号固定，重复运行逐字节相同
    numbers_theory()
    for fn in (fig_same_count, fig_die_count, fig_yield, fig_sampling, fig_patterns, fig_repeater):
        fn()
        print("完成", fn.__name__, flush=True)
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
