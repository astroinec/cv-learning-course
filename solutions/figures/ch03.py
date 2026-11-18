"""第 3 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch03.py

输出：<主分支>/03_形态学/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。

腐蚀和膨胀用本章作业的参考实现（图像外视为背景）；连通域、轮廓、Otsu、
自适应阈值、灰度顶帽和结构元的生成用 OpenCV，因为讲义教的就是这些接口。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

CH = "03_形态学"
mod = S.load_solution(CH, "3-1_手写二值形态学.py")
NUM: dict = {}
P_NOISE = math.erfc(3 / math.sqrt(2))  # 高斯噪声下单个良品像素超过 3 sigma 的概率，第 18 章 k=3 的情形
KMAX = 41


# ---------------------------------------------------------------- 工具
def erode(a, k):
    return mod.erode_binary(a, k)


def dilate(a, k):
    return mod.dilate_binary(a, k)


def opening(a, k):
    return dilate(erode(a, k), k)


def closing(a, k):
    """作业约定图像外是背景，直接先膨胀再腐蚀会把贴边的前景吃掉，所以先补一圈背景再裁回来。"""
    r = max(k.shape)
    return erode(dilate(np.pad(a, r), k), k)[r:-r, r:-r]


def n_comp(a, conn=8):
    return int(cv2.connectedComponents(a, connectivity=conn)[0] - 1)


def area_filter(a, amin, conn=8):
    n, lab, st, _ = cv2.connectedComponentsWithStats(a, connectivity=conn)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= amin
    return (keep[lab] * 255).astype(np.uint8)


def se(shape, k):
    if shape == "方形":
        return np.ones((k, k), np.uint8)
    if shape == "十字":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
    if shape == "椭圆":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if shape == "横线":
        return np.ones((1, k), np.uint8)
    if shape == "竖线":
        return np.ones((k, 1), np.uint8)
    if shape == "斜线":  # 右上到左下，和 scratch(45, ...) 的走向一致
        return np.eye(k, dtype=np.uint8)[::-1].copy()
    raise ValueError(shape)


def pct(x, nd=3):
    return round(float(x), nd)


def cells(ax, arr, color=S.BLUE):
    """把小矩阵画成带格线的方格。"""
    h, w = arr.shape
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    for y in range(h):
        for x in range(w):
            ax.add_patch(plt.Rectangle((x, y), 1, 1, fc=color if arr[y, x] else "white", ec="#b9bec7", lw=0.6))
    S.image_axes(ax)


def rgb(hexcolor, mix=0.0):
    c = np.array([int(hexcolor[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return c * (1 - mix) + mix


# ---------------------------------------------------------------- 图 1：候选图清理的取舍
ITEMS = [("颗粒 2×2", 2, 2, True), ("颗粒 3×3", 3, 3, True), ("颗粒 5×5", 5, 5, True),
         ("细划痕 1×30", 1, 30, True), ("粗划痕 3×30", 3, 30, True),
         ("边缘残差线 40×1", 40, 1, False), ("边缘残差线 40×1", 40, 1, False)]
METHODS = {
    "不处理": lambda a: a,
    "面积过滤 ≥3": lambda a: area_filter(a, 3),
    "面积过滤 ≥5": lambda a: area_filter(a, 5),
    "开运算 3×3 十字": lambda a: opening(a, se("十字", 3)),
    "开运算 3×3 方形": lambda a: opening(a, se("方形", 3)),
    "开运算 1×5 横线": lambda a: opening(a, se("横线", 5)),
}


def make_scene(rng, n=128):
    """一张“阈值之后的候选图”：真缺陷、沿图形边缘的残差线、独立噪点。"""
    occupied = np.zeros((n, n), bool)
    cls = np.zeros((n, n), np.uint8)  # 1 噪点，2 真缺陷，3 边缘残差线
    masks = []
    for name, h, w, real in ITEMS:
        while True:
            y, x = int(rng.integers(6, n - 6 - h)), int(rng.integers(6, n - 6 - w))
            if not occupied[y - 4:y + h + 4, x - 4:x + w + 4].any():
                break
        occupied[y:y + h, x:x + w] = True
        cls[y:y + h, x:x + w] = 2 if real else 3
        masks.append((name, real, np.s_[y:y + h, x:x + w]))
    near = dilate((occupied * 255).astype(np.uint8), se("方形", 3)) > 0
    noise = (rng.random((n, n)) < P_NOISE) & ~near
    cls[noise] = 1
    return ((cls > 0) * 255).astype(np.uint8), cls, masks


def fig_scene():
    rng = np.random.default_rng(0)
    trials = 40
    names = [it[0] for it in ITEMS if it[3]]
    kept = {m: dict.fromkeys(names, 0) for m in METHODS}
    kept_px = {m: dict.fromkeys(names, 0.0) for m in METHODS}
    false_noise = dict.fromkeys(METHODS, 0)
    false_edge = dict.fromkeys(METHODS, 0)
    total = dict.fromkeys(METHODS, 0)
    example = None
    for t in range(trials):
        img, cls, masks = make_scene(rng)
        outs = {}
        for m, fn in METHODS.items():
            out = fn(img)
            outs[m] = out
            n, lab = cv2.connectedComponents(out, connectivity=8)
            total[m] += n - 1
            real_labels, edge_labels = set(), set()
            for name, real, sl in masks:
                ids = set(np.unique(lab[sl])) - {0}
                if real:
                    kept[m][name] += bool(ids)
                    kept_px[m][name] += float((out[sl] > 0).mean())
                    real_labels |= ids
                else:
                    edge_labels |= ids
            others = set(range(1, n)) - real_labels
            false_edge[m] += len(others & edge_labels)
            false_noise[m] += len(others - edge_labels)
        if t == 0:
            example = (cls, outs)
    NUM["候选图清理的取舍"] = {
        "说明": f"128x128 的合成候选图 {trials} 张（种子 0）。每张 5 个真缺陷（2×2、3×3、5×5 的颗粒，1×30 的细划痕，3×30 的粗划痕）、"
              "2 条 1 px 宽 40 px 长的竖线（模拟对齐残差沿图形边缘留下的超限）、以及独立噪点；"
              f"噪点概率 {P_NOISE:.4f}，等于高斯噪声超过 3 sigma 的概率；目标之间至少隔 4 px，目标周围 1 px 内不放噪点。"
              "连通域按 8 连通数；保留指处理后该缺陷的位置上还有前景像素",
        "噪点概率": float(f"{P_NOISE:.3g}"),
        "每张图的连通域总数": {m: round(total[m] / trials, 2) for m in METHODS},
        "每张图的假连通域：独立噪点": {m: round(false_noise[m] / trials, 2) for m in METHODS},
        "每张图的假连通域：边缘残差线": {m: round(false_edge[m] / trials, 2) for m in METHODS},
        "真缺陷的保留率": {m: {k: pct(v / trials) for k, v in kept[m].items()} for m in METHODS},
        "真缺陷的像素保留比例（更严的口径）": {m: {k: pct(v / trials) for k, v in kept_px[m].items()} for m in METHODS},
    }

    cls, outs = example
    strong = {1: rgb("#3b4048"), 2: rgb(S.BLUE), 3: rgb(S.ORANGE)}
    faint = {1: rgb("#3b4048", 0.82), 2: rgb(S.BLUE, 0.78), 3: rgb(S.ORANGE, 0.72)}

    def paint(out):
        im = np.ones(cls.shape + (3,))
        for c in (1, 2, 3):
            im[(cls == c) & (out == 0)] = faint[c]
            im[(cls == c) & (out > 0)] = strong[c]
        return im

    fig, ax = plt.subplots(1, 4, figsize=(15.5, 4.1), gridspec_kw={"width_ratios": [1, 1, 1, 1.75]}, layout="constrained")
    shown = [("不处理", "阈值后的候选图"), ("面积过滤 ≥5", "面积过滤 ≥5 之后"), ("开运算 3×3 方形", "3×3 方形开运算之后")]
    for a, (m, title) in zip(ax, shown):
        a.imshow(paint(outs[m]), interpolation="nearest")
        a.set_title(title)
        S.image_axes(a)
    ax[0].set_xlabel("蓝：真缺陷　橙：边缘残差线　黑：噪点", fontsize=9)
    ax[1].set_xlabel("浅色是被删掉的", fontsize=9)
    ax[2].set_xlabel("浅色是被删掉的", fontsize=9)

    ms = list(METHODS)
    grid = np.array([[kept[m][k] / trials for k in names] for m in ms])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("rg", ["#f6c9c6", "#fff6d6", "#c9ecd4"])
    ax[3].imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto", extent=(0, len(names), len(ms), 0))
    for i, m in enumerate(ms):
        for j, k in enumerate(names):
            ax[3].text(j + 0.5, i + 0.5, f"{100 * grid[i, j]:.0f}%", ha="center", va="center", fontsize=9.5)
        ax[3].text(len(names) + 0.55, i + 0.5, f"{round(false_noise[m] / trials, 2):g}", ha="center", va="center", fontsize=9.5)
        ax[3].text(len(names) + 1.55, i + 0.5, f"{round(false_edge[m] / trials, 2):g}", ha="center", va="center", fontsize=9.5)
    ax[3].set_xlim(0, len(names) + 2.1)
    ax[3].set_xticks([j + 0.5 for j in range(len(names) + 2)],
                     [n.replace(" ", "\n") for n in names] + ["剩下的\n噪点", "剩下的\n边缘线"], fontsize=8.5)
    ax[3].set_yticks([i + 0.5 for i in range(len(ms))], ms, fontsize=9)
    ax[3].tick_params(length=0)
    ax[3].axvline(len(names), color="#b9bec7", lw=1)
    for side in ax[3].spines.values():
        side.set_color("#b9bec7")
    ax[3].set_title("真缺陷的保留率，和每张图剩下的假连通域（40 张平均）")
    S.save(fig, CH, "fig01_候选图清理的取舍")


# ---------------------------------------------------------------- 图 2：Otsu 与前景占比
def otsu_numpy(p):
    """p 是 256 个灰度级的概率。返回让类间方差最大的 t（[0,t] 为一类）。"""
    w0 = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu[-1] * w0 - mu) ** 2 / (w0 * (1 - w0))
    sb[~np.isfinite(sb)] = 0
    return int(np.argmax(sb))


def mixture_pmf(f, delta, sigma=10.0, mu=100.0):
    edges = np.arange(-0.5, 256.5, 1.0)

    def cdf(m):
        return np.array([0.5 * (1 + math.erf((e - m) / (sigma * math.sqrt(2)))) for e in edges])

    p = (1 - f) * np.diff(cdf(mu)) + f * np.diff(cdf(mu + delta))
    return p / p.sum()


def flip_fraction(delta):
    """按理论直方图（没有抽样噪声）求 Otsu 阈值从两峰之间掉进背景峰的前景占比。"""
    lo, hi = 1e-4, 0.5
    for _ in range(50):
        mid = math.sqrt(lo * hi)
        if otsu_numpy(mixture_pmf(mid, delta)) < 100 + delta / 4:
            lo = mid
        else:
            hi = mid
    return hi


def flip_estimate(delta, sigma=10.0):
    """比 (2/pi)(sigma/delta)^2 多算一项的估计：从背景均值处切开时，前景整体落在亮的一类里。"""
    half = sigma * math.sqrt(2 / math.pi)

    def gain(f):
        w0, w1 = (1 - f) / 2, (1 - f) / 2 + f
        m1 = ((1 - f) / 2 * half + f * delta) / w1
        return w0 * w1 * (m1 + half) ** 2 - f * (1 - f) * delta ** 2  # 切背景的类间方差减去正确切法的

    lo, hi = 1e-5, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if gain(mid) > 0:
            lo = mid
        else:
            hi = mid
    return hi


def sample_gray(rng, f, delta, n=512):
    lab = np.zeros(n * n, bool)
    lab[rng.choice(n * n, int(round(f * n * n)), replace=False)] = True
    g = rng.normal(100, 10, n * n)
    g[lab] += delta
    return np.clip(np.rint(g), 0, 255).astype(np.uint8), lab


def fig_otsu():
    fracs = [0.5, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]
    seeds = 10
    res = {}
    hist_examples = {}
    for delta in (60, 100):
        rows = {"Otsu 阈值（中位数）": [], "Otsu：背景被判成前景的比例": [], "Otsu：前景召回率": [],
                "中位数 + 3 倍 1.4826·MAD 的阈值（中位数）": [], "MAD 法：背景被判成前景的比例": [], "MAD 法：前景召回率": []}
        for f in fracs:
            Ts, fp, rc, T2s, fp2, rc2 = [], [], [], [], [], []
            for s in range(seeds):
                rng = np.random.default_rng(1000 + s)
                g, lab = sample_gray(rng, f, delta)
                T, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                hist = np.bincount(g, minlength=256).astype(float)
                assert int(T) == otsu_numpy(hist / hist.sum()), "OpenCV 的 Otsu 和讲义公式算出的阈值不一致"
                b = g > T
                Ts.append(T)
                fp.append((b & ~lab).sum() / (~lab).sum())
                rc.append((b & lab).sum() / lab.sum())
                med = np.median(g)
                T2 = med + 3 * 1.4826 * np.median(np.abs(g - med))
                b2 = g > T2
                T2s.append(T2)
                fp2.append((b2 & ~lab).sum() / (~lab).sum())
                rc2.append((b2 & lab).sum() / lab.sum())
                if delta == 60 and s == 0 and f in (0.05, 0.01):
                    hist_examples[f] = (hist, int(T))
            for key, vals in zip(rows, (Ts, fp, rc, T2s, fp2, rc2)):
                rows[key].append(round(float(np.median(vals) if "阈值" in key else np.mean(vals)), 4))
        res[delta] = rows
    flips = {d: flip_fraction(d) for d in (60, 100)}
    rough = {d: (2 / math.pi) * (10 / d) ** 2 for d in (60, 100)}
    NUM["Otsu 与前景占比"] = {
        "说明": f"512x512 个像素，背景灰度 N(100, 10)，前景 N(100+对比度, 10)，取整到 uint8；每个占比 {seeds} 个种子（1000 到 1009）。"
              "Otsu 用 cv2.threshold，并断言它和讲义里的类间方差公式逐次一致。"
              "MAD 法指阈值取 中位数 + 3 × 1.4826 × MAD，是第 18 章逐像素 z 阈值的全图版本",
        "前景占比": fracs,
        "对比度 60（6 倍噪声）": res[60],
        "对比度 100（10 倍噪声）": res[100],
        "理论直方图上 Otsu 阈值掉进背景峰的前景占比": {f"对比度 {d}": f"{100 * v:.2f}%" for d, v in flips.items()},
        "粗略估计 (2/pi)·(噪声/对比度)^2": {f"对比度 {d}": f"{100 * v:.2f}%" for d, v in rough.items()},
        "多算一项的估计（从背景均值处切开时，把落在亮类里的前景也算进去）": {f"对比度 {d}": f"{100 * flip_estimate(d):.2f}%" for d in (60, 100)},
        "把一个高斯峰从中间切开得到的类间方差与噪声方差之比 2/pi": round(2 / math.pi, 3),
    }

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.25]}, layout="constrained")
    for a, f in zip(ax, (0.05, 0.01)):
        hist, T = hist_examples[f]
        a.bar(np.arange(256), hist, width=1.0, color="#8f959e")
        a.axvline(T + 0.5, color=S.RED, lw=1.8)
        a.text(0.97, 0.92, f"红线：Otsu 阈值 {T}", color=S.RED, fontsize=10, ha="right", va="top", transform=a.transAxes)
        a.set_yscale("log")
        a.set_xlim(60, 200)
        a.set_ylim(0.8, hist.max() * 2)
        a.set_xlabel("灰度")
        a.set_ylabel("像素数（对数轴）")
        S.clean(a)
    ax[0].set_title("前景占 5%：阈值落在两峰之间")
    ax[1].set_title("前景占 1%：阈值掉进背景峰")
    x = [100 * f for f in fracs]
    ax[2].plot(x, res[60]["Otsu：背景被判成前景的比例"], "o-", color=S.RED, lw=1.8, ms=5, label="Otsu，对比度是噪声的 6 倍")
    ax[2].plot(x, res[100]["Otsu：背景被判成前景的比例"], "s-", color=S.ORANGE, lw=1.8, ms=5, label="Otsu，对比度是噪声的 10 倍")
    ax[2].plot(x, res[60]["MAD 法：背景被判成前景的比例"], "^--", color=S.GREEN, lw=1.6, ms=5, label="中位数 + 3 倍 MAD，6 倍")
    for d, c in ((60, S.RED), (100, S.ORANGE)):
        ax[2].axvline(100 * flips[d], color=c, lw=1, ls=":")
    ax[2].set_xscale("log")
    ax[2].set_xticks([0.1, 0.5, 1, 2, 5, 10, 50], ["0.1", "0.5", "1", "2", "5", "10", "50"])
    ax[2].minorticks_off()
    ax[2].set_xlabel("前景占比（%，对数轴）")
    ax[2].set_ylabel("背景被判成前景的比例")
    ax[2].set_ylim(-0.02, 0.56)
    ax[2].set_title("前景占比低过竖虚线后，Otsu 把大片背景判成前景")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    S.save(fig, CH, "fig02_Otsu与前景占比")


def numbers_illumination():
    """背景有渐变时四种二值化的表现，只出表不出图。"""
    amps = [0, 20, 40, 80]
    seeds = 5
    methods = ["全局 Otsu", "自适应阈值（均值，31×31，C=-20）", "15×15 顶帽后 Otsu", "15×15 顶帽后固定阈值 20"]
    out = {m: {"背景被判成前景的比例": [], "前景召回率": []} for m in methods}
    frac = []
    for G in amps:
        acc = {m: [] for m in methods}
        for s in range(seeds):
            rng = np.random.default_rng(4000 + s)
            n = 256
            ramp = np.arange(n)[None, :] / (n - 1) - 0.5
            img = 100 + G * ramp + rng.normal(scale=4, size=(n, n))
            lab = np.zeros((n, n), bool)
            for _ in range(60):
                y, x = rng.integers(4, n - 9, 2)
                lab[y:y + 5, x:x + 5] = True
            img[lab] += 40
            g = np.clip(np.rint(img), 0, 255).astype(np.uint8)
            top = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, np.ones((15, 15), np.uint8))
            res = {
                methods[0]: cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0,
                methods[1]: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, -20) > 0,
                methods[2]: cv2.threshold(top, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0,
                methods[3]: top > 20,
            }
            for m, b in res.items():
                acc[m].append(((b & ~lab).sum() / (~lab).sum(), (b & lab).sum() / lab.sum()))
            if G == 0:
                frac.append(lab.mean())
        for m in methods:
            out[m]["背景被判成前景的比例"].append(round(float(np.mean([v[0] for v in acc[m]])), 4))
            out[m]["前景召回率"].append(round(float(np.mean([v[1] for v in acc[m]])), 4))

    # 自适应阈值里 C 的方向
    rng = np.random.default_rng(4100)
    g = np.clip(np.rint(100 + rng.normal(scale=4, size=(256, 256))), 0, 255).astype(np.uint8)
    cdir, cdir_gauss = {}, {}
    for name, flag in (("THRESH_BINARY", cv2.THRESH_BINARY), ("THRESH_BINARY_INV", cv2.THRESH_BINARY_INV)):
        cdir[name] = {f"C={C}": pct((cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, flag, 31, C) > 0).mean())
                      for C in (-8, 0, 8)}
        cdir_gauss[name] = {f"C={C}": pct((cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag, 31, C) > 0).mean())
                            for C in (-8, 0, 8)}
    NUM["背景渐变下的二值化"] = {
        "说明": f"256x256，背景 100 加从左到右的线性渐变（总幅度见下），高斯噪声标准差 4；60 个 5×5 的亮目标（比背景亮 40），"
              f"前景约占 {100 * float(np.mean(frac)):.1f}%；每个幅度 {seeds} 个种子（4000 到 4004）",
        "渐变总幅度（灰度级）": amps,
        **out,
        "自适应阈值里 C 的方向（纯噪声图上的前景占比，均值法，31×31）": cdir,
        "同上，换成高斯加权法（ADAPTIVE_THRESH_GAUSSIAN_C）": cdir_gauss,
    }


# ---------------------------------------------------------------- 图 3：结构元的实际形状，迭代与大核
def fig_elements():
    shapes = {"MORPH_RECT": cv2.MORPH_RECT, "MORPH_ELLIPSE": cv2.MORPH_ELLIPSE, "MORPH_CROSS": cv2.MORPH_CROSS}
    counts = {name: {f"{k}×{k}": int(cv2.getStructuringElement(code, (k, k)).sum()) for k in (3, 5, 7, 9)} for name, code in shapes.items()}
    point = np.zeros((9, 9), np.uint8)
    point[4, 4] = 255
    twice = {}
    for name in ("方形", "十字"):
        k3 = se(name, 3)
        d2 = dilate(dilate(point, k3), k3)
        twice[name] = d2
    rng = np.random.default_rng(5)
    rand = ((rng.random((200, 200)) < 0.02) * 255).astype(np.uint8)
    disc = np.zeros((200, 200), np.uint8)
    cv2.circle(disc, (100, 100), 60, 255, -1)
    disc[rng.random((200, 200)) < 0.01] = 0
    diff = {}
    for name in ("方形", "十字", "椭圆"):
        k3, k5 = se(name, 3), se(name, 5)
        diff[name] = {
            "膨胀：3×3 两次与 5×5 一次不同的像素数": int((dilate(dilate(rand, k3), k3) != dilate(rand, k5)).sum()),
            "腐蚀：3×3 两次与 5×5 一次不同的像素数": int((erode(erode(disc, k3), k3) != erode(disc, k5)).sum()),
        }
    NUM["结构元的实际形状"] = {
        "说明": f"cv2.getStructuringElement 在本机 OpenCV {cv2.__version__} 上生成的结构元里非零像素的个数",
        "非零像素数": counts,
        "3×3 的 MORPH_ELLIPSE 和 3×3 的 MORPH_CROSS 是否完全相同": bool(np.array_equal(se("椭圆", 3), se("十字", 3))),
        "横向核 getStructuringElement(MORPH_RECT, (7, 3)) 的数组形状（行, 列）": list(cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)).shape),
    }
    NUM["迭代两次与更大的核"] = {
        "说明": "单个前景点用 3×3 结构元膨胀两次得到的像素数；以及 200x200 的图上“3×3 做两次”和“5×5 做一次”逐像素比较"
              "（膨胀用 2% 密度的随机点图，腐蚀用带 1% 小孔的半径 60 的圆，种子 5）。椭圆指 OpenCV 生成的 MORPH_ELLIPSE",
        "一个点膨胀两次的像素数": {name: int(np.count_nonzero(v)) for name, v in twice.items()},
        "5×5 结构元的像素数": {"方形": 25, "十字": int(se("十字", 5).sum()), "椭圆": int(se("椭圆", 5).sum())},
        "逐像素比较": diff,
    }

    def canvas(arr):
        out = np.zeros((7, 7), np.uint8)
        r = (7 - arr.shape[0]) // 2
        out[r:r + arr.shape[0], r:r + arr.shape[1]] = arr > 0
        return out

    panels = [
        (canvas(se("椭圆", 3)), S.BLUE, f"ELLIPSE 3×3：{counts['MORPH_ELLIPSE']['3×3']} 个像素\n和十字完全一样"),
        (canvas(se("椭圆", 5)), S.BLUE, f"ELLIPSE 5×5：{counts['MORPH_ELLIPSE']['5×5']} 个像素\n上下各伸出一个尖"),
        (canvas(se("椭圆", 7)), S.BLUE, f"ELLIPSE 7×7：{counts['MORPH_ELLIPSE']['7×7']} 个像素"),
        (canvas(se("十字", 5)), S.BLUE, f"CROSS 5×5：{counts['MORPH_CROSS']['5×5']} 个像素"),
        (canvas(twice["十字"][1:8, 1:8]), S.ORANGE, f"一个点用 3×3 十字膨胀两次\n{np.count_nonzero(twice['十字'])} 个像素的菱形，不是 5×5 十字"),
        (canvas(twice["方形"][1:8, 1:8]), S.ORANGE, f"用 3×3 方形膨胀两次\n{np.count_nonzero(twice['方形'])} 个像素，正好是 5×5 方形"),
    ]
    fig, ax = plt.subplots(1, 6, figsize=(15.5, 3.1), layout="constrained")
    for a, (arr, color, title) in zip(ax, panels):
        cells(a, arr, color)
        a.set_title(title, fontsize=9.5)
    S.save(fig, CH, "fig03_结构元的实际形状")


# ---------------------------------------------------------------- 图 4：开运算删掉什么
def fig_opening():
    def block(h, w):
        a = np.zeros((41, 41), np.uint8)
        a[20 - h // 2:20 - h // 2 + h, 20 - w // 2:20 - w // 2 + w] = 255
        return a

    disc5 = np.zeros((41, 41), np.uint8)
    yy, xx = np.mgrid[0:41, 0:41]
    disc5[(yy - 20) ** 2 + (xx - 20) ** 2 <= 2.5 ** 2] = 255
    diag = np.zeros((41, 41), np.uint8)
    diag[np.arange(10, 31), np.arange(30, 9, -1)] = 255
    targets = {"方块 1×1": block(1, 1), "方块 2×2": block(2, 2), "方块 3×3": block(3, 3), "方块 4×4": block(4, 4), "方块 5×5": block(5, 5),
               "圆，直径 5": disc5, "横线 1×20": block(1, 20), "横线 2×20": block(2, 20), "横线 3×20": block(3, 20), "45° 斜线，宽 1": diag}
    elems = {"十字 3×3": se("十字", 3), "方形 3×3": se("方形", 3), "椭圆 5×5": se("椭圆", 5), "方形 5×5": se("方形", 5), "横线 1×5": se("横线", 5)}
    keep = {t: {e: pct(np.count_nonzero(opening(a, k)) / np.count_nonzero(a), 2) for e, k in elems.items()} for t, a in targets.items()}

    dens = [0.001, 0.003, 0.01, 0.03, 0.1]
    cleaners = {"不处理": lambda a: a, "面积过滤 ≥3": lambda a: area_filter(a, 3), "面积过滤 ≥5": lambda a: area_filter(a, 5),
                "面积过滤 ≥9": lambda a: area_filter(a, 9), "开运算 3×3 十字": lambda a: opening(a, se("十字", 3)),
                "开运算 3×3 方形": lambda a: opening(a, se("方形", 3))}
    residue = {c: [] for c in cleaners}
    seeds = 20
    for p in dens:
        acc = dict.fromkeys(cleaners, 0)
        for s in range(seeds):
            rng = np.random.default_rng(2000 + s)
            a = ((rng.random((256, 256)) < p) * 255).astype(np.uint8)
            for c, fn in cleaners.items():
                acc[c] += n_comp(fn(a))
        for c in cleaners:
            residue[c].append(round(acc[c] / seeds, 2))
    NUM["开运算删掉什么"] = {
        "说明": "保留率 = 开运算后剩下的像素数 / 原来的像素数，单个目标放在 41x41 的空白图中央。椭圆 5×5 指 OpenCV 生成的 MORPH_ELLIPSE",
        "各目标的像素数": {t: int(np.count_nonzero(a)) for t, a in targets.items()},
        "保留率": keep,
    }
    NUM["噪点密度与残留"] = {
        "说明": f"256x256 的纯噪点图，每个像素独立地以给定概率成为前景；每个密度 {seeds} 个种子（2000 到 2019），数 8 连通的连通域个数",
        "噪点密度": dens,
        "每张图剩下的连通域个数": residue,
    }

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]}, layout="constrained")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("rg", ["#f6c9c6", "#fff6d6", "#c9ecd4"])
    ts, es = list(targets), list(elems)
    grid = np.array([[keep[t][e] for e in es] for t in ts])
    ax[0].imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(ts)):
        for j in range(len(es)):
            ax[0].text(j, i, f"{100 * grid[i, j]:.0f}%", ha="center", va="center", fontsize=9.5)
    ax[0].set_xticks(range(len(es)), es, fontsize=9)
    ax[0].set_yticks(range(len(ts)), ts, fontsize=9)
    ax[0].tick_params(length=0)
    ax[0].set_xlabel("开运算用的结构元")
    for side in ax[0].spines.values():
        side.set_color("#b9bec7")
    ax[0].set_title("开运算后目标还剩多少像素")
    styles = {"不处理": (S.MUTED, "o-"), "面积过滤 ≥3": (S.ORANGE, "s-"), "面积过滤 ≥5": (S.RED, "^-"), "面积过滤 ≥9": (S.PURPLE, "v-"),
              "开运算 3×3 十字": (S.GREEN, "D--"), "开运算 3×3 方形": (S.BLUE, "o--")}
    floor = 0.03
    for c, (color, fmt) in styles.items():
        ax[1].plot([100 * d for d in dens], [max(v, floor) for v in residue[c]], fmt, color=color, lw=1.7, ms=5, label=c)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xticks([0.1, 0.3, 1, 3, 10], ["0.1", "0.3", "1", "3", "10"])
    ax[1].set_yticks([floor, 0.1, 1, 10, 100, 1000], ["0", "0.1", "1", "10", "100", "1000"])
    ax[1].minorticks_off()
    ax[1].set_xlabel("噪点密度（%，对数轴）")
    ax[1].set_ylabel("每张图剩下的连通域个数（对数轴）")
    ax[1].set_title("噪点变密后面积过滤先失守，3×3 开运算仍接近 0")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    S.clean(ax[1])
    S.save(fig, CH, "fig04_开运算删掉什么")


# ---------------------------------------------------------------- 图 5：闭运算连断口
def scratch(theta, gap, width=1, L=61, n=101):
    """一条 L 列长的划痕，在中间去掉 gap 列。theta 是和水平方向的夹角（0 到 45 度），width 是竖直方向的粗细。"""
    a = np.zeros((n, n), np.uint8)
    c = n // 2
    xs = np.arange(c - L // 2, c + L // 2 + 1)
    ys = np.rint(c - math.tan(math.radians(theta)) * (xs - c)).astype(int)
    keep = np.ones(len(xs), bool)
    mid = len(xs) // 2
    keep[mid - gap // 2:mid - gap // 2 + gap] = False
    for dy in range(-(width // 2), width // 2 + 1):
        a[ys[keep] + dy, xs[keep]] = 255
    return a


def min_k(a, fn):
    """让结果只剩一个连通域的最小奇数边长，找不到返回 None。"""
    return next((k for k in range(3, KMAX + 1, 2) if n_comp(fn(a, k)) == 1), None)


def fig_closing():
    gaps = list(range(1, 9))
    table = {}
    for width in (1, 3):
        for theta in (0, 20, 45):
            row = {}
            for name in ("方形", "横线", "十字", "椭圆", "斜线"):
                row[f"闭运算，{name}"] = [min_k(scratch(theta, g, width), lambda a, k, name=name: closing(a, se(name, k))) for g in gaps]
            row["只膨胀，方形"] = [min_k(scratch(theta, g, width), lambda a, k: dilate(np.pad(a, 25), se("方形", k))) for g in gaps]
            table[f"宽 {width} px，{theta} 度"] = row
    vert = {}
    for name in ("方形", "竖线", "横线", "椭圆"):
        vert[f"闭运算，{name}"] = [min_k(scratch(0, g, 1).T.copy(), lambda a, k, name=name: closing(a, se(name, k))) for g in gaps]
    table["宽 1 px，90 度"] = vert
    for g in gaps:  # 和 OpenCV 交叉核对一次
        a = scratch(45, g, 1)
        assert n_comp(cv2.morphologyEx(a, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))) == 2
    ex = scratch(45, 3, 1)
    ex_sq = closing(ex, se("方形", 15))
    ex_line = closing(ex, se("斜线", 5))
    ex_dil = dilate(ex, se("方形", 5))
    hz = scratch(0, 3, 1)

    spacing = list(range(1, 9))
    side, corner = [], []
    for d in spacing:
        a = np.zeros((60, 60), np.uint8)
        a[20:25, 10:15] = 255
        a[20:25, 15 + d:20 + d] = 255
        side.append(min_k(a, lambda x, k: closing(x, se("方形", k))))
        b = np.zeros((60, 60), np.uint8)
        b[10:15, 10:15] = 255
        b[15 + d:20 + d, 15 + d:20 + d] = 255
        corner.append(min_k(b, lambda x, k: closing(x, se("方形", k))))
    ks = [1, 3, 5, 7, 9, 11, 15]
    merged = {"闭运算": [], "只膨胀": []}
    seeds = 20
    truth = 0.0
    for k in ks:
        acc = {"闭运算": 0, "只膨胀": 0}
        for s in range(seeds):
            rng = np.random.default_rng(3000 + s)
            a = np.zeros((256, 256), np.uint8)
            for _ in range(40):
                y, x = rng.integers(8, 245, 2)
                a[y:y + 3, x:x + 3] = 255
            if k == 1:
                truth += n_comp(a) / seeds
            acc["闭运算"] += n_comp(closing(a, se("方形", k)))
            acc["只膨胀"] += n_comp(dilate(a, se("方形", k)))
        for name in merged:
            merged[name].append(round(acc[name] / seeds, 2))
    NUM["闭运算连断口"] = {
        "说明": f"划痕长 61 列，画在 101x101 的图中央，断口按去掉的列数计（45 度时 g 列相当于 g 乘根号 2 的直线距离）；宽 3 px 指竖直方向 3 个像素。"
              f"表里是让两段变成一个 8 连通域的最小奇数边长，从 3 试到 {KMAX}，null 表示到 {KMAX} 都连不上。"
              "椭圆指 OpenCV 的 MORPH_ELLIPSE，斜线指 45 度方向、和划痕同向的 1 px 宽线形结构元，横线是 1×k，竖线是 k×1；90 度一行是把水平划痕转置得到的竖直划痕。闭运算前先补一圈背景",
        "断口列数": gaps,
        "最小结构元边长": table,
        "例子：45 度细划痕、断口 3 列": {
            "15×15 方形闭运算补上的像素数": int(np.count_nonzero(ex_sq) - np.count_nonzero(ex)),
            "15×15 方形闭运算后的连通域数": n_comp(ex_sq),
            "5 px 斜线闭运算补上的像素数": int(np.count_nonzero(ex_line) - np.count_nonzero(ex)),
            "5 px 斜线闭运算后的连通域数": n_comp(ex_line),
            "5×5 方形膨胀后的连通域数": n_comp(ex_dil),
            "5×5 方形膨胀后的像素数（原来 58）": int(np.count_nonzero(ex_dil)),
            "原来的像素数": int(np.count_nonzero(ex)),
        },
        "例子：水平细划痕、断口 3 列，5×5 方形闭运算补上的像素数": int(np.count_nonzero(closing(hz, se("方形", 5))) - np.count_nonzero(hz)),
    }
    NUM["闭运算的副作用：粘连"] = {
        "说明": "两个 5×5 的颗粒，左右并排或者沿对角线错开，间距 d 个像素；表里是方形闭运算把它们并成一个连通域的最小奇数边长。"
              f"随机场景是 256x256 的图上随机放 40 个 3×3 的颗粒（{seeds} 个种子，3000 到 3019，重叠的本来就算一个），数处理后的连通域个数",
        "间距 d": spacing,
        "左右并排：最小边长": side,
        "对角错开：最小边长": corner,
        "随机场景：方形边长": ks,
        "随机场景：原图的连通域个数": round(truth, 2),
        "随机场景：处理后的连通域个数": merged,
    }

    def paint(before, after, sl):
        im = np.ones(before.shape + (3,))
        im[after > 0] = rgb(S.GREEN, 0.25)
        im[before > 0] = rgb(S.INK)
        return im[sl]

    sl = np.s_[38:63, 38:63]
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.9), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.75]}, layout="constrained")
    shots = [(ex, "45° 细划痕，断口 3 列"), (ex_sq, "15×15 方形闭运算\n一个像素都没补上"), (ex_line, "5 px 同向线形核闭运算\n补上 3 个像素，连上了"),
             (ex_dil, "只做 5×5 方形膨胀\n连上了，但整条变粗")]
    for a, (after, title) in zip(ax, shots):
        a.imshow(paint(ex, after, sl), interpolation="nearest")
        a.set_title(title, fontsize=10)
        S.image_axes(a)
    ax[1].set_xlabel("黑：原有　绿：新增", fontsize=9)
    t45 = table["宽 1 px，45 度"]
    t0 = table["宽 1 px，0 度"]
    ax[4].plot(gaps, t0["闭运算，方形"], "o-", color=S.BLUE, lw=1.8, ms=6, label="水平划痕，方形核（横线核相同）")
    ax[4].plot(gaps, t45["闭运算，斜线"], "x--", color=S.GREEN, lw=1.4, ms=7, label="45° 划痕，同向线形核（与上一条重合）")
    ell = [(g, k) for g, k in zip(gaps, t45["闭运算，椭圆"]) if k is not None]
    ax[4].plot([g for g, _ in ell], [k for _, k in ell], "s-", color=S.ORANGE, lw=1.8, ms=5, label="45° 划痕，OpenCV 椭圆核")
    ax[4].text(1.0, 39, f"45° 划痕用方形核：\n试到 {KMAX}×{KMAX} 都连不上，\n图上没有对应的点", color=S.RED, fontsize=9, va="top")
    ax[4].set_xlabel("断口宽度（列）")
    ax[4].set_ylabel("能连上的最小结构元边长")
    ax[4].set_ylim(0, KMAX + 1)
    ax[4].set_xticks(gaps)
    ax[4].set_title("同向线形核略大于断口即可，椭圆核要大得多")
    ax[4].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax[4])
    S.save(fig, CH, "fig05_闭运算连断口")


def numbers_order():
    """先开后闭和先闭后开在噪点图上的差别，只出表。"""
    dens = [0.003, 0.01, 0.03]
    seeds = 20
    out = {"只做 3×3 开运算": [], "先 3×3 开运算再 5×5 闭运算": [], "先 5×5 闭运算再 3×3 开运算": [], "先 7×7 闭运算再 3×3 开运算": []}
    px = []
    for p in dens:
        acc = dict.fromkeys(out, 0)
        npx = 0
        for s in range(seeds):
            rng = np.random.default_rng(2000 + s)
            a = ((rng.random((256, 256)) < p) * 255).astype(np.uint8)
            o = opening(a, se("方形", 3))
            co = opening(closing(a, se("方形", 5)), se("方形", 3))
            acc["只做 3×3 开运算"] += n_comp(o)
            acc["先 3×3 开运算再 5×5 闭运算"] += n_comp(closing(o, se("方形", 5)))
            acc["先 5×5 闭运算再 3×3 开运算"] += n_comp(co)
            acc["先 7×7 闭运算再 3×3 开运算"] += n_comp(opening(closing(a, se("方形", 7)), se("方形", 3)))
            npx += np.count_nonzero(co)
        for k in out:
            out[k].append(round(acc[k] / seeds, 2))
        px.append(round(npx / seeds, 1))
    NUM["开和闭的先后顺序"] = {
        "说明": f"256x256 的纯噪点图（没有任何真目标），方形结构元，每个密度 {seeds} 个种子（2000 到 2019）；数处理后剩下的 8 连通域个数",
        "噪点密度": dens,
        "每张图剩下的连通域个数": out,
        "先 5×5 闭再 3×3 开剩下的前景像素数": px,
    }


# ---------------------------------------------------------------- 图 6：连通性与面积
def fig_components():
    sizes = [(1, 1), (2, 2), (3, 3), (5, 5), (10, 10), (1, 20), (3, 20)]
    rows = {}
    for h, w in sizes:
        b = np.zeros((40, 40), np.uint8)
        b[5:5 + h, 5:5 + w] = 255
        cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, _, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        x, y, bw, bh = cv2.boundingRect(cs[0])
        rows[f"{h}×{w}"] = {"像素数": h * w, "CC_STAT_AREA": int(st[1, cv2.CC_STAT_AREA]), "contourArea": float(cv2.contourArea(cs[0])),
                           "boundingRect 的宽乘高": int(bw * bh), "(h-1)(w-1)": (h - 1) * (w - 1)}
        assert rows[f"{h}×{w}"]["contourArea"] == (h - 1) * (w - 1)
    ring = np.zeros((40, 40), np.uint8)
    ring[10:30, 10:30] = 255
    ring[15:25, 15:25] = 0
    cs, _ = cv2.findContours(ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = [0, 10, 20, 30, 45]
    conn = {a: (n_comp(scratch(a, 0), 8), n_comp(scratch(a, 0), 4)) for a in angles}
    dens = [0.01, 0.03, 0.1]
    noise = {"8 连通": [], "4 连通": []}
    for p in dens:
        c8 = c4 = 0
        for s in range(20):
            rng = np.random.default_rng(2000 + s)
            a = ((rng.random((256, 256)) < p) * 255).astype(np.uint8)
            c8 += n_comp(a, 8)
            c4 += n_comp(a, 4)
        noise["8 连通"].append(round(c8 / 20, 1))
        noise["4 连通"].append(round(c4 / 20, 1))
    NUM["面积的三种算法"] = {
        "说明": "实心矩形（高×宽），分别用像素计数、connectedComponentsWithStats 的面积、findContours 加 contourArea、外接框面积来量",
        "各目标": rows,
        "带孔的环（20×20 的方块中间挖掉 10×10）": {"像素数": int(np.count_nonzero(ring)), "外轮廓的 contourArea": float(cv2.contourArea(cs[0]))},
    }
    NUM["4 连通与 8 连通"] = {
        "说明": "一条 1 px 宽、61 列长的直线，画法同图 5 的划痕（每列一个像素），按不同角度画出后数连通域；噪点图是 256x256、20 个种子的平均",
        "角度": angles,
        "8 连通下的连通域个数": [conn[a][0] for a in angles],
        "4 连通下的连通域个数": [conn[a][1] for a in angles],
        "噪点密度": dens,
        "噪点图的连通域个数": noise,
    }

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1, 1.35]}, layout="constrained")
    ax = ax[::-1]
    names = list(rows)
    x = np.arange(len(names))
    px = [rows[n]["像素数"] for n in names]
    ca = [rows[n]["contourArea"] for n in names]
    ax[0].bar(x - 0.2, px, width=0.4, color=S.BLUE, alpha=0.85, label="像素数（等于 CC_STAT_AREA）")
    ax[0].bar(x + 0.2, ca, width=0.4, color=S.ORANGE, alpha=0.85, label="contourArea")
    for i in range(len(names)):
        ax[0].text(i - 0.2, px[i] + 2, str(px[i]), ha="center", fontsize=9, color=S.BLUE)
        ax[0].text(i + 0.2, ca[i] + 2, f"{ca[i]:.0f}", ha="center", fontsize=9, color=S.ORANGE)
    ax[0].set_xticks(x, names)
    ax[0].set_xlabel("实心矩形目标（高×宽，像素）")
    ax[0].set_ylabel("面积")
    ax[0].set_ylim(0, 115)
    ax[0].set_title("contourArea 比像素数小，1 px 宽的目标直接是 0")
    ax[0].legend(fontsize=9, loc="upper left")
    S.clean(ax[0])
    ax[0].grid(axis="x", visible=False)
    bars = ax[1].bar([str(a) + "°" for a in angles], [conn[a][1] for a in angles], color=S.RED, alpha=0.85, width=0.55)
    for b, a in zip(bars, angles):
        ax[1].text(b.get_x() + b.get_width() / 2, conn[a][1] + 1.2, str(conn[a][1]), ha="center", fontsize=10)
    ax[1].set_xlabel("直线和水平方向的夹角（8 连通下都是 1 个）")
    ax[1].set_ylabel("4 连通下的连通域个数")
    ax[1].set_ylim(0, 70)
    ax[1].set_title("同一条 61 px 的细线，4 连通把它拆成几段")
    S.clean(ax[1])
    ax[1].grid(axis="x", visible=False)
    S.save(fig, CH, "fig06_连通性与面积")


# ---------------------------------------------------------------- 只出数字的几组
def numbers_contract():
    """作业约定和 OpenCV 默认行为的差别、不对称核、偶数核、性质检查、讲义里的小例子。"""
    k3, k5 = se("方形", 3), se("方形", 5)
    a = np.zeros((5, 5), np.uint8)
    a[1:4, 1:4] = 255
    hole = a.copy()
    hole[2, 2] = 0
    NUM["5×5 小例子"] = {
        "说明": "5x5 的图中央一个 3×3 的方块，3×3 全 1 结构元，图像外视为背景",
        "腐蚀后的前景像素数": int(np.count_nonzero(erode(a, k3))),
        "膨胀后的前景像素数": int(np.count_nonzero(dilate(a, k3))),
        "中心挖掉一个像素后，腐蚀的前景像素数": int(np.count_nonzero(erode(hole, k3))),
        "中心挖掉一个像素后，膨胀的前景像素数": int(np.count_nonzero(dilate(hole, k3))),
    }

    corner = np.zeros((40, 40), np.uint8)
    corner[0:10, 0:10] = 255
    full = np.full((20, 20), 255, np.uint8)
    rng = np.random.default_rng(6)
    rand = ((rng.random((80, 80)) < 0.5) * 255).astype(np.uint8)
    asym = np.zeros((3, 3), np.uint8)
    asym[1, 1] = asym[1, 2] = asym[0, 1] = 1
    same = True
    for k in (k3, se("十字", 5), asym):
        same &= np.array_equal(erode(rand, k), cv2.erode(rand, k, borderType=cv2.BORDER_CONSTANT, borderValue=0))
        same &= np.array_equal(dilate(rand, k), cv2.dilate(rand, k, borderType=cv2.BORDER_CONSTANT, borderValue=0))
    NUM["边界约定"] = {
        "说明": "作业约定图像外是背景 0。cv2.erode 默认的边界值相当于图像外是前景，cv2.dilate 默认相当于图像外是背景。"
              "贴角方块指 40x40 的图左上角一个 10×10 的方块；整幅前景指 20x20 全 255 的图",
        "贴角方块做 3×3 腐蚀后的像素数": {"作业约定": int(np.count_nonzero(erode(corner, k3))), "cv2.erode 默认": int(np.count_nonzero(cv2.erode(corner, k3)))},
        "贴角方块直接先膨胀再腐蚀（5×5）后的像素数（原来 100）": {"作业约定": int(np.count_nonzero(erode(dilate(corner, k5), k5))),
                                                "cv2.morphologyEx 默认": int(np.count_nonzero(cv2.morphologyEx(corner, cv2.MORPH_CLOSE, k5)))},
        "整幅前景直接先膨胀再腐蚀（3×3）后丢掉的像素数（共 400）": {"作业约定": int((erode(dilate(full, k3), k3) == 0).sum()),
                                                "cv2.morphologyEx 默认": int((cv2.morphologyEx(full, cv2.MORPH_CLOSE, k3) == 0).sum())},
        "80x80 随机图上，参考实现与 cv2（borderType=BORDER_CONSTANT, borderValue=0）是否逐像素相同（方形 3×3、十字 5×5、不对称核，腐蚀和膨胀）": bool(same),
        "同一张随机图上，参考实现的膨胀与 cv2.dilate 默认参数是否逐像素相同（方形 3×3）": bool(np.array_equal(dilate(rand, k3), cv2.dilate(rand, k3))),
        "同一张随机图上，参考实现的腐蚀与 cv2.erode 默认参数不同的像素数（方形 3×3，都在最外一圈）": int((erode(rand, k3) != cv2.erode(rand, k3)).sum()),
    }

    def stamp(img, k):
        out = np.zeros_like(img)
        ry, rx = k.shape[0] // 2, k.shape[1] // 2
        for y, x in zip(*np.nonzero(img)):
            for dy, dx in zip(*np.nonzero(k)):
                yy, xx = y + dy - ry, x + dx - rx
                if 0 <= yy < img.shape[0] and 0 <= xx < img.shape[1]:
                    out[yy, xx] = 255
        return out

    sparse = ((rng.random((40, 40)) < 0.05) * 255).astype(np.uint8)
    rect = np.pad(np.full((10, 15), 255, np.uint8), 10)
    shifted = {}
    for name, code in (("开运算", cv2.MORPH_OPEN), ("闭运算", cv2.MORPH_CLOSE)):
        o = cv2.morphologyEx(rect, code, np.ones((2, 2), np.uint8))
        ys, xs = np.nonzero(o)
        shifted[name] = {"结果占的行": [int(ys.min()), int(ys.max())], "结果占的列": [int(xs.min()), int(xs.max())],
                         "跑到原目标外面的像素数": int(((o > 0) & (rect == 0)).sum()), "原目标里丢掉的像素数": int(((o == 0) & (rect > 0)).sum())}
    NUM["不对称核与偶数核"] = {
        "说明": "不对称核取 3×3 里只有中心、中心右边、中心上边三个点。“盖章”指把核原样盖到每个前景像素上；“命中”指作业和 OpenCV 的定义："
              "核的非零位置下只要有前景，输出就是前景。偶数核实验是 10×15 的矩形（占第 10 到 19 行、第 10 到 24 列）用 cv2.morphologyEx 和 2×2 的核处理",
        "40x40 稀疏点图上，盖章和命中两种膨胀不同的像素数": {"不对称核": int((stamp(sparse, asym) != dilate(sparse, asym)).sum()),
                                           "对称的 3×3 十字": int((stamp(sparse, se("十字", 3)) != dilate(sparse, se("十字", 3))).sum()),
                                           "不对称核旋转 180 度后再盖章": int((stamp(sparse, asym[::-1, ::-1]) != dilate(sparse, asym)).sum())},
        "cv2.morphologyEx 用 2×2 核": shifted,
    }

    props = {}
    k = se("十字", 5)
    o = opening(rand, k)
    props["开运算做两次和做一次相同"] = bool(np.array_equal(opening(o, k), o))
    props["开运算的结果是原图的子集"] = bool(not np.any((o > 0) & (rand == 0)))
    c = closing(rand, k)
    props["闭运算做两次和做一次相同"] = bool(np.array_equal(closing(c, k), c))
    props["闭运算的结果包含原图"] = bool(not np.any((rand > 0) & (c == 0)))
    pad = 8
    inv = np.pad(255 - rand, pad, constant_values=255)  # 取反之后，图像外面变成前景
    props["闭运算等于把前景和背景对调后做开运算、再对调回来"] = bool(np.array_equal(c, 255 - opening(inv, k)[pad:-pad, pad:-pad]))
    props["腐蚀等于把前景和背景对调后做膨胀、再对调回来"] = bool(np.array_equal(erode(rand, k), 255 - dilate(inv, k)[pad:-pad, pad:-pad]))
    NUM["性质检查"] = {"说明": "80x80、前景占一半的随机图（种子 6），5×5 十字结构元（对称）；闭运算按图 5 的做法先补一圈背景；"
                           "对调前景和背景时，图像外面也跟着从背景变成前景", **props}

    # 最常见的实现错误：在输入数组上原地修改
    buggy = np.zeros((9, 9), np.uint8)
    buggy[4, 4] = 255
    for y in range(9):
        for x in range(9):
            if buggy[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].any():
                buggy[y, x] = 255
    one = np.zeros((9, 9), np.uint8)
    one[4, 4] = 255
    full_fg = np.full((12, 12), 255, np.uint8)
    n_full, _, st_full, _ = cv2.connectedComponentsWithStats(full_fg, connectivity=8)
    near = np.zeros((40, 40), np.uint8)
    near[2:5, 20:23] = 255
    k7 = se("方形", 7)
    NUM["实现错误的后果"] = {
        "说明": "9x9 的图中央一个前景点，3×3 全 1 结构元。原地修改指按从上到下、从左到右的顺序，把结果直接写回正在读的那个数组",
        "正确的膨胀结果像素数": int(np.count_nonzero(dilate(one, k3))),
        "原地修改得到的像素数": int(np.count_nonzero(buggy)),
        "一个点用 3×3 十字膨胀的像素数": int(np.count_nonzero(dilate(one, se("十字", 3)))),
    }
    NUM["其他核对"] = {
        "说明": "两个零散的核对。全前景图是 12x12 全 255；贴边颗粒是 40x40 的图上一个 3×3 的颗粒，离上边界隔 2 行背景，做 7×7 方形闭运算",
        "全前景图上 connectedComponentsWithStats 返回的标签数": int(n_full),
        "全前景图上标签 0 的面积": int(st_full[0, cv2.CC_STAT_AREA]),
        "贴边颗粒闭运算后的像素数（原来 9）": {"先补一圈背景再用作业函数": int(np.count_nonzero(closing(near, k7))),
                                "cv2.morphologyEx 默认参数": int(np.count_nonzero(cv2.morphologyEx(near, cv2.MORPH_CLOSE, k7))),
                                "cv2.morphologyEx，borderValue=0": int(np.count_nonzero(cv2.morphologyEx(near, cv2.MORPH_CLOSE, k7, borderType=cv2.BORDER_CONSTANT, borderValue=0)))},
    }

    # Notebook 里的场景，用和 Notebook 相同的 OpenCV 调用
    nb = np.zeros((120, 180), np.uint8)
    nb[25:90, 20:65] = 255
    nb[43:70, 35:50] = 0
    nb[35:80, 82:120] = 255
    nb[35:80, 125:162] = 255
    r = np.random.default_rng(2026)
    ys = r.integers(0, nb.shape[0], 35)
    xs = r.integers(0, nb.shape[1], 35)
    nb[ys, xs] = 255

    def run(code, w, h):
        return cv2.morphologyEx(nb, code, cv2.getStructuringElement(cv2.MORPH_RECT, (w, h)))

    def joined(img):
        lab = cv2.connectedComponents(img, connectivity=8)[1]
        return lab[57, 100] == lab[57, 140]

    merge_w = next(w for w in range(1, 22, 2) if joined(run(cv2.MORPH_CLOSE, w, 3)))
    merge_h = next((h for h in range(1, 16, 2) if joined(run(cv2.MORPH_CLOSE, 3, h))), None)
    walls = next(w for w in range(1, 22, 2) if run(cv2.MORPH_OPEN, w, 3)[57, 27] == 0)
    hole_w = next((w for w in range(1, 22, 2) if run(cv2.MORPH_CLOSE, w, 3)[57, 42] > 0), None)
    NUM["Notebook 场景"] = {
        "说明": "实验_形态学.ipynb 里的合成图（120x180，种子 2026），用和 Notebook 相同的 cv2.morphologyEx 与 MORPH_RECT，滑块范围宽 1 到 21、高 1 到 15",
        "原图前景像素数": int(np.count_nonzero(nb)),
        "原图连通域个数（8 连通）": n_comp(nb),
        "3×3 开运算后的连通域个数": n_comp(run(cv2.MORPH_OPEN, 3, 3)),
        "高度固定 3 时，把右侧两个矩形粘在一起的最小闭运算宽度": int(merge_w),
        "宽度固定 3 时，把右侧两个矩形粘在一起的最小闭运算高度（null 表示滑块范围内做不到）": merge_h,
        "高度固定 3 时，让左侧矩形的竖边消失的最小开运算宽度": int(walls),
        "高度固定 3 时，把左侧孔洞中心填上的最小闭运算宽度": hole_w,
    }
    r7 = np.random.default_rng(7)
    dense = ((r7.random((100, 100)) < 0.6) * 255).astype(np.uint8)
    kw = dict(borderType=cv2.BORDER_CONSTANT, borderValue=0)
    o1 = cv2.morphologyEx(dense, cv2.MORPH_OPEN, k3, **kw)
    o_it2 = cv2.morphologyEx(dense, cv2.MORPH_OPEN, k3, iterations=2, **kw)
    o_twice = cv2.morphologyEx(o1, cv2.MORPH_OPEN, k3, **kw)
    o5 = cv2.morphologyEx(dense, cv2.MORPH_OPEN, k5, **kw)
    NUM["iterations 的含义"] = {
        "说明": "100x100、前景占 60% 的随机图（种子 7），cv2.morphologyEx 做方形开运算，边界按背景处理",
        "原图前景像素数": int(np.count_nonzero(dense)),
        "3×3 开运算一次后的前景像素数": int(np.count_nonzero(o1)),
        "3×3 开运算、iterations=2 后的前景像素数": int(np.count_nonzero(o_it2)),
        "把 3×3 开运算的结果再做一次 3×3 开运算后的前景像素数": int(np.count_nonzero(o_twice)),
        "5×5 开运算一次后的前景像素数": int(np.count_nonzero(o5)),
        "iterations=2 的 3×3 开运算与 5×5 开运算一次是否逐像素相同": bool(np.array_equal(o_it2, o5)),
        "连做两遍 3×3 开运算与做一遍是否逐像素相同": bool(np.array_equal(o_twice, o1)),
    }
    NUM["理论数字"] = {
        "说明": "高斯噪声下单个像素超过 3 sigma 的概率 erfc(3/sqrt(2))，以及它在 128x128 和 256x256 的图上对应的期望噪点个数",
        "概率": float(f"{P_NOISE:.3g}"),
        "128x128 图上的期望噪点数": round(P_NOISE * 128 * 128, 1),
        "256x256 图上的期望噪点数": round(P_NOISE * 256 * 256, 1),
    }


def main() -> None:
    S.setup()
    plt.rcParams["svg.hashsalt"] = "ch03"  # 固定 SVG 里的内部编号，重复运行时文件逐字节相同
    numbers_contract()
    for fn in (fig_scene, fig_otsu, numbers_illumination, fig_elements, fig_opening, fig_closing, numbers_order, fig_components):
        fn()
        print("完成", fn.__name__, flush=True)
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
