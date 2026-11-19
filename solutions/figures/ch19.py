"""第 19 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch19.py

输出：<主分支>/19_异常检测与深度缺陷检测/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。

特征沿用本章 Notebook 的做法：4x4 像素块（步长 2）展平成 16 维，再做 3x3 邻域平均，
不依赖预训练权重。最后一组对照用 torchvision 的 ResNet-18（Notebook 的选做单元已经用到）。
table_boundary_fill 对应 Notebook 的“对照 1”：聚合时边界换成只对真实存在的邻居平均。
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

CH = "19_异常检测与深度缺陷检测"
mod = S.load_solution(CH, "19-1_PatchCore核心.py")
NUM: dict = {}
N, NOISE, PERIOD = 64, 0.06, 8.0
YY, XX = np.mgrid[0:N, 0:N]
SEEDS = (0, 1, 2)
DATA_NOTE = (f"{N}x{N} 的条纹纹理（周期 {PERIOD:.0f} px，幅度 0.3，噪声标准差 {NOISE}）；缺陷是半径 2 或 3 px 的圆斑，亮暗随机；"
             "特征是 4x4 像素块（步长 2）展平成 16 维再做 3x3 邻域平均，每张图 31x31 个 patch")


def r3(v):
    return round(float(v), 3)


# ---------------------------------------------------------------- 数据
def good(rng, aligned=False, shift=None, horizontal=False):
    """一张良品。aligned=True 时条纹相位只在 0 到 0.3 rad 内变化（和 Notebook 相同），否则完全随机。"""
    phi = rng.uniform(0, 0.3) if (aligned or shift is not None) else rng.uniform(0, 2 * np.pi)
    if shift is not None:
        phi += 2 * np.pi * shift / PERIOD
    t = YY if horizontal else XX
    return 0.5 + 0.3 * np.sin(2 * np.pi * t / PERIOD + phi) + rng.normal(scale=NOISE, size=(N, N))


def add_defect(img, rng, amp, radius=(2, 4)):
    r = float(radius) if np.isscalar(radius) else float(rng.integers(*radius))
    margin = int(r) + 6
    cy, cx = rng.integers(margin, N - margin, 2)
    mask = (YY - cy) ** 2 + (XX - cx) ** 2 <= r * r
    out = img.copy()
    out[mask] += rng.choice([-1.0, 1.0]) * amp
    return out, mask


def dataset(seed, n_train=20, n_test=200, aligned=False, amp=0.12, radius=(2, 4)):
    """同一个 seed 下良品和待检图的底图不随 amp、radius 变化，方便成对比较。"""
    rng = np.random.default_rng(seed)
    train = [good(rng, aligned) for _ in range(n_train)]
    tests, masks = [], []
    for i in range(n_test):
        img, mask = good(rng, aligned), np.zeros((N, N), bool)
        if i % 2 == 0:
            img, mask = add_defect(img, rng, amp, radius)
        tests.append(img)
        masks.append(mask)
    return train, tests, masks, np.array([int(m.any()) for m in masks])


# ---------------------------------------------------------------- 特征、打分、指标
def patch_features(img, agg=True):
    pats = np.lib.stride_tricks.sliding_window_view(img, (4, 4))[::2, ::2]
    f = pats.reshape(pats.shape[0], pats.shape[1], -1).astype(np.float64)
    if not agg:
        return f
    pad = np.pad(f, ((1, 1), (1, 1), (0, 0)), mode="edge")
    return sum(pad[dy:dy + f.shape[0], dx:dx + f.shape[1]] for dy in range(3) for dx in range(3)) / 9


def feats(imgs, agg=True):
    return np.stack([patch_features(im, agg) for im in imgs])


G = 31
RING = np.ones((G, G), bool)
RING[1:-1, 1:-1] = False
CORNER = np.zeros((G, G), bool)
CORNER[[0, 0, -1, -1], [0, -1, 0, -1]] = True


def knn_multi(memory, queries, ks=(1,), chunk=1024):
    """和参考实现的 knn_scores 结果相同（main 里有核对），分块计算并一次算出多个 k。"""
    kmax = max(ks)
    out = {k: np.empty(len(queries)) for k in ks}
    for i in range(0, len(queries), chunk):
        d = mod.pairwise_sq_dists(queries[i:i + chunk], memory)
        if kmax == 1:
            out[1][i:i + chunk] = np.sqrt(d.min(axis=1))
            continue
        part = np.sqrt(np.sort(np.partition(d, kmax - 1, axis=1)[:, :kmax], axis=1))
        for k in ks:
            out[k][i:i + chunk] = part[:, :k].mean(axis=1)
    return out


def score_maps(memory, F, k=1):
    """F: (n, h, w, d) → (n, h, w) 的 patch 分数图。"""
    return knn_multi(memory, F.reshape(-1, F.shape[-1]), (k,))[k].reshape(F.shape[:3])


def image_score(s, inner=True):
    return s[:, ~RING].max(axis=1) if inner else s.reshape(len(s), -1).max(axis=1)


def upsample(score, offset=1.5, stride=2.0):
    """双线性插值回原图。patch i 的中心在 stride*i + offset；offset=0 就是没按中心对齐。"""
    c = np.arange(score.shape[0]) * stride + offset
    px = np.arange(N)
    tmp = np.stack([np.interp(px, c, row) for row in score])
    return np.stack([np.interp(px, c, tmp[:, j]) for j in range(N)], axis=1)


def heatmaps(s, sigma=0.0, offset=1.5):
    out = []
    for x in s:
        m = upsample(x, offset)
        if sigma > 0:
            m = cv2.GaussianBlur(m, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
        out.append(m[4:-4, 4:-4])
    return out


def crop(masks):
    return [m[4:-4, 4:-4] for m in masks]


def pixel_auroc(maps, masks):
    return mod.auroc(np.concatenate([m.ravel() for m in maps]), np.concatenate([g.ravel() for g in masks]).astype(int))


def pro_curve(maps, masks, n_thr=300):
    """PRO 曲线：横轴是正常像素的过检率，纵轴是各缺陷区域被盖住比例的平均（每个区域权重相同）。"""
    normal = np.sort(np.concatenate([m[~g] for m, g in zip(maps, masks)]))
    regions = []
    for m, g in zip(maps, masks):
        if g.any():
            n, lab = cv2.connectedComponents(g.astype(np.uint8), connectivity=8)
            regions += [np.sort(m[lab == i]) for i in range(1, n)]
    thr = np.quantile(np.concatenate([m.ravel() for m in maps]), np.linspace(0, 1, n_thr))[::-1]
    fpr = 1 - np.searchsorted(normal, thr, side="left") / len(normal)
    pro = np.mean([1 - np.searchsorted(r, thr, side="left") / len(r) for r in regions], axis=0)
    return fpr, pro


def aupro(maps, masks, limit=0.3):
    """PRO 曲线在过检率 0 到 limit 之间的面积，除以 limit 归一化到 0 到 1。"""
    fpr, pro = pro_curve(maps, masks)
    order = np.argsort(fpr, kind="mergesort")
    f, p = fpr[order], pro[order]
    keep = f <= limit
    f2 = np.concatenate([[0.0], f[keep], [limit]])
    p2 = np.concatenate([[0.0], p[keep], [np.interp(limit, f, p)]])
    return float(np.sum((f2[1:] - f2[:-1]) * (p2[1:] + p2[:-1]) / 2) / limit)


_MEM: dict = {}


def memory_order(key, bank, m=None):
    """贪心核心集的选点顺序（默认到记忆库的 10%）。k-center greedy 的前 m 个点就是大小为 m 的核心集，所以只算一次。"""
    m = m or max(1, len(bank) // 10)
    if key not in _MEM or len(_MEM[key]) < m:
        _MEM[key] = mod.greedy_coreset(bank, m)
    return _MEM[key][:m]


def cover_radius(bank, memory):
    return float(np.sqrt(max(mod.pairwise_sq_dists(bank[i:i + 2048], memory).min(axis=1).max() for i in range(0, len(bank), 2048))))


# ---------------------------------------------------------------- 图 1：三类方法，对齐与不对齐
def golden_maps(train, tests):
    g = np.stack(train)
    med = np.median(g, axis=0)
    sig = np.maximum(1.4826 * np.median(np.abs(g - med), axis=0), 0.01)
    return [np.abs(cv2.blur((im - med) / sig, (5, 5), borderType=cv2.BORDER_REFLECT))[4:-4, 4:-4] for im in tests]


def gaussian_maps(Ftrain, Ftest, eps=0.01):
    n, d = len(Ftrain), Ftrain.shape[-1]
    mu = Ftrain.mean(axis=0)
    X = Ftrain - mu
    inv = np.linalg.inv(np.einsum("nhwi,nhwj->hwij", X, X) / (n - 1) + eps * np.eye(d))
    return [upsample(np.sqrt(np.einsum("hwi,hwij,hwj->hw", f - mu, inv, f - mu)))[4:-4, 4:-4] for f in Ftest]


def fig_methods():
    names = ["金模板（逐像素）", "逐位置高斯", "记忆库最近邻"]
    res: dict = {}
    example = {}
    for aligned in (True, False):
        for amp in (0.2, 0.12):
            rows = {n: [] for n in names}
            for seed in SEEDS:
                train, tests, masks, labels = dataset(seed, aligned=aligned, amp=amp)
                Ftr, Fte = feats(train), feats(tests)
                bank = Ftr.reshape(-1, 16)
                s = score_maps(bank[memory_order((seed, aligned, 20, True), bank)], Fte)
                maps = {names[0]: golden_maps(train, tests), names[1]: gaussian_maps(Ftr, Fte), names[2]: heatmaps(s)}
                mk = crop(masks)
                for n in names:
                    img = image_score(s) if n == names[2] else np.array([m.max() for m in maps[n]])
                    rows[n].append([mod.auroc(img, labels), pixel_auroc(maps[n], mk), aupro(maps[n], mk)])
                if seed == 0 and amp == 0.2:
                    example[aligned] = (train[:2], tests[0], mk[0], {n: maps[n][0] for n in names})
            res[(aligned, amp)] = {n: np.mean(v, axis=0) for n, v in rows.items()}
    cond = {True: "对齐", False: "不对齐"}
    NUM["三类方法与对齐"] = {
        "说明": DATA_NOTE + "。20 张良品，200 张待检图（一半有缺陷），3 个种子取平均；对齐指良品条纹的相位只差 0 到 0.3 rad（约 0.4 px 以内），不对齐指相位完全随机；"
              "三种方法都不统计图像最外 4 个像素；记忆库用贪心核心集压到 10%，k=1。每项三个数依次是 image AUROC、pixel AUROC、AUPRO",
        **{f"{cond[a]}，缺陷幅度 {amp}": {n: [r3(x) for x in v] for n, v in res[(a, amp)].items()} for (a, amp) in res},
    }

    fig, ax = plt.subplots(1, 6, figsize=(16.5, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1.75]}, layout="constrained")
    tr, test, mask, maps = example[False]
    ax[0].imshow(np.concatenate([example[True][0][0][:32], example[True][0][1][32:]]), cmap="gray", vmin=0, vmax=1)
    ax[0].axhline(31.5, color=S.ORANGE, lw=1)
    ax[0].set_title("对齐：两张良品上下拼接")
    ax[1].imshow(np.concatenate([tr[0][:32], tr[1][32:]]), cmap="gray", vmin=0, vmax=1)
    ax[1].axhline(31.5, color=S.ORANGE, lw=1)
    ax[1].set_title("不对齐：条纹错开了")
    for a, n, title in zip(ax[2:5], names, ("金模板的分数图", "逐位置高斯的分数图", "记忆库的分数图")):
        a.imshow(maps[n], cmap="magma", extent=(4, 60, 60, 4))
        a.contour(np.arange(4, 60), np.arange(4, 60), mask, levels=[0.5], colors="cyan", linewidths=1.2)
        a.set_title(title)
    for a in ax[:5]:
        S.image_axes(a)
    x = np.arange(3)
    for i, (a, color) in enumerate(((True, S.MUTED), (False, S.BLUE))):
        vals = [res[(a, 0.2)][n][2] for n in names]
        bars = ax[5].bar(x + (i - 0.5) * 0.36, vals, 0.34, color=color, alpha=0.88, label=cond[a])
        for b, v in zip(bars, vals):
            ax[5].text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.5)
    ax[5].set_xticks(x, ["金模板", "逐位置高斯", "记忆库"])
    ax[5].set_ylim(0, 1.13)
    ax[5].set_ylabel("AUPRO（缺陷幅度 0.2）")
    ax[5].set_title("不对齐时金模板失效，记忆库不受影响")
    ax[5].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    S.clean(ax[5])
    ax[5].grid(axis="x", visible=False)
    S.save(fig, CH, "fig01_三类方法与对齐")


# ---------------------------------------------------------------- 图 2：流程、局部聚合与边界、上采样
def fig_pipeline():
    table: dict = {}
    ring_stats, post, example, mean_map = [], [], None, None
    for aligned, amp in ((False, 0.2), (False, 0.12), (True, 0.2)):
        rows = []
        for seed in SEEDS:
            train, tests, masks, labels = dataset(seed, aligned=aligned, amp=amp)
            row = []
            for agg in (True, False):
                bank = feats(train, agg).reshape(-1, 16)
                s = score_maps(bank[memory_order((seed, aligned, 20, agg), bank)], feats(tests, agg))
                row += [mod.auroc(image_score(s, inner=False), labels), mod.auroc(image_score(s), labels)]
                goods = s[labels == 0]
                on_ring = float(RING.ravel()[goods.reshape(len(goods), -1).argmax(axis=1)].mean())
                row.append(on_ring)
                if agg and not aligned and amp == 0.12:
                    ring_stats.append([np.median(goods[:, ~RING]), np.median(goods[:, RING & ~CORNER]), np.median(goods[:, CORNER])])
                    mk = crop(masks)
                    one = [mod.auroc(image_score(s), labels)]
                    for offset, sigma in ((1.5, 0), (0.0, 0), (1.5, 1), (1.5, 2), (1.5, 4)):
                        maps = heatmaps(s, sigma, offset)
                        one += [mod.auroc(np.array([m.max() for m in maps]), labels), pixel_auroc(maps, mk), aupro(maps, mk)]
                    post.append(one)
                if agg and not aligned and amp == 0.2 and seed == 0:
                    mean_map = goods.mean(axis=0)
                    example = (tests[0], masks[0], s[0])
            rows.append(row)
        table[(aligned, amp)] = np.mean(rows, axis=0)
    ring_stats, post = np.mean(ring_stats, axis=0), np.mean(post, axis=0)
    cond = {True: "对齐", False: "不对齐"}
    NUM["局部聚合与边界"] = {
        "说明": DATA_NOTE + "。20 张良品，200 张待检图，核心集 10%，k=1，3 个种子取平均。整图分数取 patch 分数图的最大值；“去掉最外一圈”指不统计 31x31 分数图最外面一圈 patch（占 "
              f"{RING.mean():.1%}）。聚合时边界用复制填充（和 Notebook 相同）",
        **{f"{cond[a]}，缺陷幅度 {amp}": {
            "聚合：image AUROC（全图）": r3(v[0]), "聚合：image AUROC（去掉最外一圈）": r3(v[1]), "聚合：良品图的最高分落在最外一圈的比例": r3(v[2]),
            "不聚合：image AUROC（全图）": r3(v[3]), "不聚合：image AUROC（去掉最外一圈）": r3(v[4]), "不聚合：良品图的最高分落在最外一圈的比例": r3(v[5]),
        } for (a, amp), v in table.items()},
        "最外一圈占 patch 总数的比例": r3(RING.mean()),
        "聚合后良品 patch 分数的中位数（不对齐）": {"内部": r3(ring_stats[0]), "边上（不含角）": r3(ring_stats[1]), "四个角": r3(ring_stats[2])},
        "聚合时邻域平均后的噪声方差相对单个 patch 的比例（按 9 个权重的平方和算）": {"内部": r3(9 / 81), "边上": r3(15 / 81), "角": r3(25 / 81)},
    }
    keys = ["按 patch 中心对齐，不平滑", "没按中心对齐（整体偏 1.5 px），不平滑", "对齐，高斯平滑 sigma=1", "对齐，高斯平滑 sigma=2", "对齐，高斯平滑 sigma=4"]
    NUM["上采样与平滑"] = {
        "说明": "不对齐、缺陷幅度 0.12 的主设置，聚合，核心集 10%，3 个种子取平均；分数图双线性插值回 64x64，不统计最外 4 个像素。每项三个数依次是：用热图最大值当整图分数的 image AUROC、pixel AUROC、AUPRO",
        "直接用 patch 分数最大值当整图分数的 image AUROC": r3(post[0]),
        **{k: [r3(x) for x in post[1 + 3 * i:4 + 3 * i]] for i, k in enumerate(keys)},
    }

    test, mask, s = example
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.5), gridspec_kw={"width_ratios": [1, 1, 1, 1.08, 1.5]}, layout="constrained")
    ax[0].imshow(test, cmap="gray", vmin=0, vmax=1)
    ax[0].contour(mask, levels=[0.5], colors="cyan", linewidths=1.2)
    ax[0].set_title("待检图（青色圈是缺陷）")
    ax[1].imshow(s, cmap="magma")
    ax[1].set_title("31×31 的 patch 分数图")
    ax[2].imshow(upsample(s), cmap="magma")
    ax[2].contour(mask, levels=[0.5], colors="cyan", linewidths=1.2)
    ax[2].set_title("上采样回 64×64 的热图")
    im = ax[3].imshow(mean_map, cmap="viridis")
    ax[3].set_title("100 张良品的平均分数图")
    fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.03).ax.tick_params(labelsize=8)
    for a in ax[:4]:
        S.image_axes(a)
    labels_x = ["幅度 0.2", "幅度 0.12"]
    x = np.arange(2)
    series = [("聚合，全图取最大", 0, S.RED), ("聚合，去掉最外一圈", 1, S.BLUE), ("不聚合，全图取最大", 3, S.MUTED)]
    for i, (name, col, color) in enumerate(series):
        vals = [table[(False, amp)][col] for amp in (0.2, 0.12)]
        bars = ax[4].bar(x + (i - 1) * 0.27, vals, 0.25, color=color, alpha=0.88, label=name)
        for b, v in zip(bars, vals):
            ax[4].text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5)
    ax[4].set_xticks(x, labels_x)
    ax[4].set_ylim(0.5, 1.07)
    ax[4].set_ylabel("image AUROC（不对齐）")
    ax[4].set_title("聚合的收益被边界一圈吃掉了")
    ax[4].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, columnspacing=1.0, handlelength=1.2)
    S.clean(ax[4])
    ax[4].grid(axis="x", visible=False)
    S.save(fig, CH, "fig02_流程与边界")


# ---------------------------------------------------------------- 图 3：核心集采样
def toy_points():
    rng = np.random.default_rng(7)
    return np.concatenate([rng.normal([3.0, 3.0], [0.9, 0.7], size=(300, 2)), rng.normal([7.2, 5.2], 0.3, size=(30, 2))])


def fig_coreset():
    # (a) 二维示意
    pts = toy_points()
    g_idx = mod.greedy_coreset(pts, 10)
    r_idx = np.random.default_rng(3).choice(len(pts), 10, replace=False)
    toy = {"贪心": (g_idx, cover_radius(pts, pts[g_idx])), "随机": (r_idx, cover_radius(pts, pts[r_idx]))}
    miss = np.mean([(np.random.default_rng(100 + i).choice(len(pts), 10, replace=False) >= 300).sum() == 0 for i in range(2000)])

    # (b) 主设置：均匀的良品加弱缺陷
    ratios = [1.0, 0.1, 0.03, 0.01, 0.003, 0.001]
    out = {"贪心": [], "随机": []}
    for seed in SEEDS:
        train, tests, masks, labels = dataset(seed)
        bank, Fte, mk = feats(train).reshape(-1, 16), feats(tests), crop(masks)
        order = memory_order((seed, False, 20, True), bank)
        rows = {"贪心": [], "随机": []}
        for ratio in ratios:
            m = max(1, round(len(bank) * ratio))
            picks = {"贪心": np.arange(len(bank)) if ratio >= 1 else order[:m],
                     "随机": np.arange(len(bank)) if ratio >= 1 else np.random.default_rng(seed).choice(len(bank), m, replace=False)}
            for name, idx in picks.items():
                if ratio >= 1 and name == "随机":
                    rows[name].append(rows["贪心"][-1])
                    continue
                s = score_maps(bank[idx], Fte)
                img = image_score(s)
                rows[name].append([mod.auroc(img, labels), aupro(heatmaps(s), mk), cover_radius(bank, bank[idx]) if ratio < 1 else 0.0,
                                   np.median(img[labels == 0]), np.median(img[labels == 1]), m])
                if ratio >= 1:
                    _MEM[("full-scores", seed)] = img
        for name in out:
            out[name].append(rows[name])
    mean = {name: np.mean(v, axis=0) for name, v in out.items()}
    NUM["核心集比例"] = {
        "说明": "不对齐、缺陷幅度 0.12 的主设置，20 张良品（记忆库 19220 个 patch），200 张待检图，k=1，整图分数去掉最外一圈，3 个种子取平均；随机指不放回地均匀抽同样多的点",
        "比例": ratios, "核心集大小": [int(v[5]) for v in mean["贪心"]],
        **{f"{name}：{label}": [r3(v[i]) for v in mean[name]] for name in mean for i, label in
           ((0, "image AUROC"), (1, "AUPRO"), (2, "覆盖半径"), (3, "良品图整图分数的中位数"), (4, "缺陷图整图分数的中位数"))},
        "各种子的 image AUROC（贪心）": [[r3(x[0]) for x in rows] for rows in out["贪心"]],
        "各种子的 image AUROC（随机）": [[r3(x[0]) for x in rows] for rows in out["随机"]],
    }
    easy = []
    for seed in SEEDS:
        train, tests, masks, labels = dataset(seed, aligned=True, amp=0.2)
        bank, Fte = feats(train).reshape(-1, 16), feats(tests)
        order = mod.greedy_coreset(bank, len(bank) // 10)
        easy.append([mod.auroc(image_score(score_maps(bank if r >= 1 else bank[order[:max(1, round(len(bank) * r))]], Fte)), labels) for r in ratios[:4]])
    NUM["核心集比例（容易的数据）"] = {
        "说明": "对齐、缺陷幅度 0.2（接近本章 Notebook 的设置），其余同上，贪心核心集，3 个种子取平均",
        "比例": ratios[:4], "image AUROC": [r3(x) for x in np.mean(easy, axis=0)],
    }
    NUM["核心集示意（二维）"] = {
        "说明": "330 个二维点：一团 300 个，另一小团 30 个（占 9%）。各选 10 个点；覆盖半径是所有点到被选点的最远距离",
        "覆盖半径": {k: r3(v[1]) for k, v in toy.items()},
        "选中的点里属于小团的个数": {k: int((v[0] >= 300).sum()) for k, v in toy.items()},
        "随机抽 10 个点完全没抽到小团的概率（2000 次模拟）": r3(miss),
        "同一概率的理论值": r3(math.comb(300, 10) / math.comb(330, 10)),
    }

    # (c) 稀有的正常模式
    rare_ratios = [1.0, 0.1, 0.03, 0.01, 0.003]
    rare = {"贪心": [], "随机": []}
    share = {"贪心": [], "随机": []}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        train = [good(rng, horizontal=(i < 2)) for i in range(20)]
        tests, labels, hz = [], [], []
        for i in range(240):
            img = good(rng, horizontal=(i >= 120))
            if i % 2 == 0:
                img, _ = add_defect(img, rng, 0.2)
            tests.append(img)
            labels.append(int(i % 2 == 0))
            hz.append(i >= 120)
        labels, hz = np.array(labels), np.array(hz)
        bank, Fte = feats(train).reshape(-1, 16), feats(tests)
        order = mod.greedy_coreset(bank, len(bank) // 10)
        rows = {"贪心": [], "随机": []}
        shr = {"贪心": [], "随机": []}
        for ratio in rare_ratios:
            m = max(1, round(len(bank) * ratio))
            for name in rows:
                idx = np.arange(len(bank)) if ratio >= 1 else order[:m] if name == "贪心" else np.random.default_rng(seed).choice(len(bank), m, replace=False)
                if ratio >= 1 and name == "随机":
                    rows[name].append(rows["贪心"][-1])
                    shr[name].append(shr["贪心"][-1])
                    continue
                img = image_score(score_maps(bank[idx], Fte))
                thr, _, ovk = mod.threshold_at_escape(img[~hz & (labels == 0)], img[~hz & (labels == 1)], 0.05)
                rows[name].append([ovk, float(np.mean(img[hz & (labels == 0)] >= thr)), mod.auroc(img[hz], labels[hz])])
                shr[name].append(float(np.mean(idx < 2 * G * G)))
        for name in rare:
            rare[name].append(rows[name])
            share[name].append(shr[name])
    rmean = {name: np.mean(v, axis=0) for name, v in rare.items()}
    NUM["稀有的正常模式"] = {
        "说明": "20 张良品里 18 张是竖条纹、2 张是横条纹（稀有但正常，占记忆库的 10%）。待检图：竖、横各 60 张良品和 60 张带缺陷（幅度 0.2）。"
              "阈值在竖条纹上按“漏检不超过 5%”选定，再看同一阈值下两种良品各有多少被判成缺陷；3 个种子取平均",
        "比例": rare_ratios,
        **{f"{name}：{label}": [r3(v[i]) for v in rmean[name]] for name in rmean for i, label in
           ((0, "常见良品的过检率"), (1, "稀有良品的过检率"), (2, "稀有模式内部的 image AUROC"))},
        **{f"{name}：记忆库里稀有模式的 patch 占比": [r3(x) for x in np.mean(share[name], axis=0)] for name in share},
    }

    fig, ax = plt.subplots(1, 4, figsize=(16.5, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.25, 1.25]}, layout="constrained")
    for a, (name, color) in zip(ax[:2], (("随机", S.ORANGE), ("贪心", S.BLUE))):
        idx, rad = toy[name]
        a.scatter(pts[:, 0], pts[:, 1], s=6, color="#c3c8d0")
        for p in pts[idx]:
            a.add_patch(plt.Circle(p, rad, fill=False, ec=color, lw=0.8, alpha=0.45))
        a.scatter(pts[idx, 0], pts[idx, 1], s=42, color=color, edgecolors="white", linewidths=0.8, zorder=3)
        a.set_xlim(0, 9.6)
        a.set_ylim(0, 7.4)
        a.set_aspect("equal")
        a.set_title(f"{name}选 10 个：覆盖半径 {rad:.2f}")
        S.image_axes(a)
    xs = [100 * r for r in ratios]
    for name, color, marker in (("贪心", S.BLUE, "o"), ("随机", S.ORANGE, "s")):
        ax[2].plot(xs, [v[1] for v in mean[name]], marker + "-", color=color, lw=1.8, ms=5, label=f"{name}：AUPRO")
        ax[2].plot(xs, [v[0] for v in mean[name]], marker + "--", color=color, lw=1.3, ms=4, alpha=0.75, label=f"{name}：image AUROC")
    ax[2].set_xscale("log")
    ax[2].set_xticks(xs, ["100", "10", "3", "1", "0.3", "0.1"])
    ax[2].minorticks_off()
    ax[2].invert_xaxis()
    ax[2].set_xlabel("记忆库保留的比例（%）")
    ax[2].set_ylim(0.5, 1.0)
    ax[2].set_title("均匀的良品加弱缺陷：贪心掉得更早")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    xr = [100 * r for r in rare_ratios]
    for name, color, marker in (("贪心", S.BLUE, "o"), ("随机", S.ORANGE, "s")):
        ax[3].plot(xr, [v[1] for v in rmean[name]], marker + "-", color=color, lw=1.8, ms=5, label=f"{name}：稀有良品")
        ax[3].plot(xr, [v[0] for v in rmean[name]], marker + "--", color=color, lw=1.3, ms=4, alpha=0.75, label=f"{name}：常见良品")
    ax[3].set_xscale("log")
    ax[3].set_xticks(xr, ["100", "10", "3", "1", "0.3"])
    ax[3].minorticks_off()
    ax[3].invert_xaxis()
    ax[3].set_xlabel("记忆库保留的比例（%）")
    ax[3].set_ylabel("良品被判成缺陷的比例")
    ax[3].set_ylim(-0.03, 1.05)
    ax[3].set_title("有稀有的正常模式：随机先丢掉它")
    ax[3].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[3])
    S.save(fig, CH, "fig03_核心集采样")


# ---------------------------------------------------------------- 图 4：良品里混进带缺陷的图，近邻个数 k
def fig_contamination():
    cs, ks = [0, 1, 2, 5], (1, 3, 9)
    res = {(name, c): [] for name in ("贪心", "随机") for c in cs}
    full, picked, example = {0: [], 5: []}, {c: [] for c in cs if c}, None
    for seed in SEEDS:
        train0, tests, masks, labels = dataset(seed)
        Q = feats(tests).reshape(-1, 16)
        full[0].append(mod.auroc(_MEM[("full-scores", seed)], labels))
        for c in cs:
            rng = np.random.default_rng([seed, 9])
            train, touch, dmask = list(train0), [], []
            for i in range(20):
                m = np.zeros((N, N), bool)
                if i < c:
                    for _ in range(3):
                        train[i], one = add_defect(train[i], rng, 0.12)
                        m |= one
                dmask.append(m)
                touch.append(np.lib.stride_tricks.sliding_window_view(m, (4, 4))[::2, ::2].any(axis=(2, 3)))
            touch = np.stack(touch).ravel()
            bank = feats(train).reshape(-1, 16)
            order = memory_order((seed, False, 20, True), bank) if c == 0 else mod.greedy_coreset(bank, len(bank) // 10)
            rnd = np.random.default_rng(seed).choice(len(bank), len(order), replace=False)
            for name, idx in (("贪心", order), ("随机", rnd)):
                sc = knn_multi(bank[idx], Q, ks)
                res[(name, c)].append([mod.auroc(image_score(sc[k].reshape(-1, G, G)), labels) for k in ks])
            if c:
                m1 = len(bank) // 100
                picked[c].append([touch.mean(), np.isin(np.flatnonzero(touch), order).mean(), touch[order].mean(), touch[order[:m1]].mean(), touch[rnd].mean(), touch.sum()])
            if c == 5:
                full[5].append(mod.auroc(image_score(score_maps(bank, Q.reshape(-1, G, G, 16))), labels))
            if c == 2 and seed == 0:
                sel = np.zeros(len(bank), bool)
                sel[order] = True
                example = (train[0], dmask[0], sel[:G * G].reshape(G, G))
    mean = {key: np.mean(v, axis=0) for key, v in res.items()}
    pk = {c: np.mean(v, axis=0) for c, v in picked.items()}
    NUM["良品里混进带缺陷的图"] = {
        "说明": "不对齐、缺陷幅度 0.12 的主设置，20 张良品里有 c 张各带 3 个缺陷（和待检缺陷同一类，位置随机）；核心集 10%，200 张待检图，整图分数去掉最外一圈，3 个种子取平均。"
              "“碰到缺陷的 patch”指 4x4 像素块里至少有一个缺陷像素",
        "混进的张数 c": cs, "近邻个数 k": list(ks),
        **{f"{name}核心集的 image AUROC（每行一个 c，每列一个 k）": [[r3(x) for x in mean[(name, c)]] for c in cs] for name in ("贪心", "随机")},
        "各种子的 image AUROC（贪心，k=1，每行一个 c）": [[r3(v[0]) for v in res[("贪心", c)]] for c in cs],
        "不压缩、k=1 的 image AUROC": {"c=0": r3(np.mean(full[0])), "c=5": r3(np.mean(full[5]))},
        "碰到缺陷的 patch 占记忆库的比例": {f"c={c}": r3(v[0]) for c, v in pk.items()},
        "这些 patch 被贪心核心集（10%）选中的比例": {f"c={c}": r3(v[1]) for c, v in pk.items()},
        "贪心核心集（10%）里这类 patch 的占比": {f"c={c}": r3(v[2]) for c, v in pk.items()},
        "贪心核心集（1%）里这类 patch 的占比": {f"c={c}": r3(v[3]) for c, v in pk.items()},
        "随机采样（10%）里这类 patch 的占比": {f"c={c}": r3(v[4]) for c, v in pk.items()},
        "碰到缺陷的 patch 个数（平均）": {f"c={c}": round(float(v[5]), 1) for c, v in pk.items()},
    }

    img, dm, sel = example
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw={"width_ratios": [1, 1.25, 1.45]}, layout="constrained")
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    ax[0].contour(dm, levels=[0.5], colors="cyan", linewidths=1.3)
    ys, xs = np.nonzero(sel)
    ax[0].scatter(2 * xs + 1.5, 2 * ys + 1.5, s=13, color=S.ORANGE, edgecolors="white", linewidths=0.4)
    ax[0].set_title("橙点：被贪心核心集选中的 patch")
    S.image_axes(ax[0])
    c = 2
    names = ["在整个\n记忆库里", "随机采样\n10%", "贪心核心集\n10%", "贪心核心集\n1%"]
    vals = [100 * pk[c][0], 100 * pk[c][4], 100 * pk[c][2], 100 * pk[c][3]]
    bars = ax[1].bar(names, vals, color=[S.MUTED, S.ORANGE, S.BLUE, S.BLUE], alpha=0.88, width=0.6)
    for b, v in zip(bars, vals):
        ax[1].text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.1f}%", ha="center", fontsize=9.5)
    ax[1].set_ylabel("碰到缺陷的 patch 所占比例（%）")
    ax[1].set_title("贪心把混进来的缺陷放大了")
    ax[1].tick_params(axis="x", labelsize=9)
    S.clean(ax[1])
    ax[1].grid(axis="x", visible=False)
    ax[2].plot(cs, [mean[("贪心", c)][0] for c in cs], "o-", color=S.BLUE, lw=1.8, ms=5, label="贪心核心集，k=1")
    ax[2].plot(cs, [mean[("贪心", c)][2] for c in cs], "o--", color=S.BLUE, lw=1.4, ms=4, alpha=0.7, label="贪心核心集，k=9")
    ax[2].plot(cs, [mean[("随机", c)][0] for c in cs], "s-", color=S.ORANGE, lw=1.8, ms=5, label="随机采样，k=1")
    ax[2].plot([0, 5], [np.mean(full[0]), np.mean(full[5])], "D:", color=S.GREEN, lw=1.5, ms=5, label="不压缩，k=1（只测了两个点）")
    ax[2].set_xticks(cs)
    ax[2].set_xlabel("20 张良品里带缺陷的张数")
    ax[2].set_ylabel("image AUROC")
    ax[2].set_ylim(0.85, 0.97)
    ax[2].set_title("混进缺陷后，贪心核心集掉得最多")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    S.save(fig, CH, "fig04_混进缺陷与近邻数")


# ---------------------------------------------------------------- 图 5：良品张数、错位
def fig_goods_shift():
    ns = [1, 2, 5, 10, 20, 40]
    curve = {True: [], False: []}
    for aligned in (True, False):
        for seed in SEEDS:
            row = []
            for n in ns:
                train, tests, masks, labels = dataset(seed, n_train=n, aligned=aligned)
                bank = feats(train).reshape(-1, 16)
                s = score_maps(bank[memory_order((seed, aligned, n, True), bank)], feats(tests))
                row.append([mod.auroc(image_score(s), labels), aupro(heatmaps(s), crop(masks))])
            curve[aligned].append(row)
    cm = {a: np.mean(v, axis=0) for a, v in curve.items()}
    NUM["良品张数"] = {
        "说明": "缺陷幅度 0.12，核心集取记忆库的 10%，k=1，200 张待检图，整图分数去掉最外一圈，3 个种子取平均。对齐指良品和待检图的条纹相位都只差 0 到 0.3 rad，不对齐指相位完全随机",
        "良品张数": ns,
        **{f"{'对齐' if a else '不对齐'}：{label}": [r3(v[i]) for v in cm[a]] for a in cm for i, label in ((0, "image AUROC"), (1, "AUPRO"))},
        "各种子的 image AUROC（对齐）": [[r3(x[0]) for x in row] for row in curve[True]],
        "各种子的 image AUROC（不对齐）": [[r3(x[0]) for x in row] for row in curve[False]],
    }

    # 绝对大小对比例
    sizes, table = [96, 384, 1922], {}
    for n in (10, 20, 40):
        rows = []
        for seed in SEEDS:
            train, tests, masks, labels = dataset(seed, n_train=n)
            bank, Fte = feats(train).reshape(-1, 16), feats(tests)
            order = memory_order((seed, False, n, True), bank, max(sizes))
            rows.append([mod.auroc(image_score(score_maps(bank[order[:m]], Fte)), labels) for m in sizes])
        table[n] = np.mean(rows, axis=0)
    NUM["核心集的绝对大小"] = {
        "说明": "不对齐、缺陷幅度 0.12 的主设置，贪心核心集，k=1，3 个种子取平均。良品张数不同，核心集保留同样多的点",
        "核心集大小 m": sizes,
        **{f"{n} 张良品（记忆库 {n * G * G} 个 patch）": {"image AUROC": [r3(x) for x in table[n]], "m 占记忆库的比例": [f"{100 * m / (n * G * G):.1f}%" for m in sizes]} for n in table},
    }

    # 错位
    shifts = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]
    names = ["金模板（逐像素）", "记忆库，只见过对齐的良品", "记忆库，不聚合", "记忆库，良品里有 0 到 1 px 的错位"]
    over = {n: [] for n in names}
    det = {n: [] for n in names}
    for seed in (0, 1):
        rng = np.random.default_rng([seed, 3])
        base = [good(rng, shift=0) for _ in range(20)]
        jit = [good(rng, shift=s) for s in rng.uniform(0, 1, size=20)]
        val = [good(rng, shift=0) for _ in range(200)]
        bad = [add_defect(good(rng, shift=0), rng, 0.2)[0] for _ in range(100)]

        def bank_scorer(train, agg=True):
            bank = feats(train, agg).reshape(-1, 16)
            mem = bank[mod.greedy_coreset(bank, len(bank) // 10)]
            return lambda imgs: image_score(score_maps(mem, feats(imgs, agg)))

        scorers = {names[0]: lambda imgs: np.array([m.max() for m in golden_maps(base, imgs)]), names[1]: bank_scorer(base),
                   names[2]: bank_scorer(base, False), names[3]: bank_scorer(jit)}
        thr = {n: np.quantile(f(val), 0.99) for n, f in scorers.items()}
        moved = {s: [good(rng, shift=s) for _ in range(100)] for s in shifts}
        for n, f in scorers.items():
            over[n].append([float(np.mean(f(moved[s]) > thr[n])) for s in shifts])
            det[n].append(float(np.mean(f(bad) > thr[n])))
    om = {n: np.mean(v, axis=0) for n, v in over.items()}
    NUM["错位与过检"] = {
        "说明": "20 张良品建模板或记忆库（核心集 10%，k=1）。阈值取 200 张对齐良品分数的 99% 分位，再让待检良品整体平移 s px（每个 s 100 张），统计被判成缺陷的比例；2 个种子取平均。"
              "最后一种记忆库的 20 张良品带 0 到 1 px 的随机错位。检出率用 100 张对齐的缺陷图（幅度 0.2）在同一阈值下统计",
        "平移（px）": shifts,
        **{f"过检率：{n}": [r3(x) for x in om[n]] for n in names},
        "同一阈值下对齐缺陷图的检出率": {n: r3(np.mean(det[n])) for n in names},
    }

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.0), layout="constrained")
    for a, color, marker, label in ((True, S.MUTED, "s", "对齐"), (False, S.BLUE, "o", "不对齐")):
        for row in curve[a]:
            ax[0].plot(ns, [x[0] for x in row], marker, color=color, ms=3.5, alpha=0.35)
        ax[0].plot(ns, [v[0] for v in cm[a]], marker + "-", color=color, lw=1.9, ms=5.5, label=f"{label}（粗线是 3 个种子的平均）")
    ax[0].set_xscale("log")
    ax[0].set_xticks(ns, [str(n) for n in ns])
    ax[0].minorticks_off()
    ax[0].set_xlabel("良品张数")
    ax[0].set_ylabel("image AUROC（缺陷幅度 0.12）")
    ax[0].set_ylim(0.6, 1.0)
    ax[0].set_title("对齐时 1 张良品已接近上限，不对齐时要 10 张左右")
    ax[0].legend(fontsize=9, loc="lower right")
    S.clean(ax[0])
    for n, color, style in zip(names, (S.MUTED, S.BLUE, S.PURPLE, S.GREEN), ("s--", "o-", "^:", "D-")):
        ax[1].plot(shifts, om[n], style, color=color, lw=1.7, ms=4.5, label=n)
    ax[1].set_xlabel("待检良品相对良品库的整体平移（px）")
    ax[1].set_ylabel("良品被判成缺陷的比例")
    ax[1].set_ylim(-0.04, 1.06)
    ax[1].set_title("半个像素的错位就让大部分良品报警")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[1])
    S.save(fig, CH, "fig05_良品张数与错位")


# ---------------------------------------------------------------- 图 6：指标与阈值
def fast_threshold(defect, max_escape):
    """和参考实现 threshold_at_escape 选出的阈值相同（main 里有核对）：把缺陷分数排序，取允许漏掉的个数对应的那一个。"""
    d = np.sort(defect)
    return float(d[int(math.floor(max_escape * len(d) + 1e-9))])


def fig_metrics():
    # (a) 大小缺陷混合：pixel AUROC 和 AUPRO
    sigmas = [0, 1, 2, 4, 6]
    rows, share = [], []
    for seed in SEEDS:
        rng = np.random.default_rng([seed, 4])
        train = dataset(seed)[0]
        tests, masks, size = [], [], []
        for i in range(200):
            img, mask, r = good(rng), np.zeros((N, N), bool), 0
            if i % 2 == 0:
                r = 2 if i % 4 == 0 else 7
                img, mask = add_defect(img, rng, 0.12, radius=r)
            tests.append(img)
            masks.append(mask)
            size.append(r)
        size, mk = np.array(size), crop(masks)
        bank = feats(train).reshape(-1, 16)
        s = score_maps(bank[memory_order((seed, False, 20, True), bank)], feats(tests))
        share.append(sum(g.sum() for g, r in zip(mk, size) if r == 7) / sum(g.sum() for g in mk))
        row = []
        for sigma in sigmas:
            maps = heatmaps(s, sigma)
            thr = np.quantile(np.concatenate([m[~g] for m, g in zip(maps, mk)]), 0.99)
            cov = {r: np.mean([(m[g] >= thr).mean() for m, g, rr in zip(maps, mk, size) if rr == r]) for r in (2, 7)}
            row.append([pixel_auroc(maps, mk), aupro(maps, mk), cov[2], cov[7]])
        rows.append(row)
    pm = np.mean(rows, axis=0)
    NUM["pixel AUROC 与 AUPRO"] = {
        "说明": "不对齐的主设置，缺陷幅度 0.12，一半是半径 2 px 的小缺陷、一半是半径 7 px 的大缺陷；对同一批分数图做不同强度的高斯平滑再评估，3 个种子取平均。"
              "覆盖率指像素过检率定在 1% 时，每个缺陷区域里超过阈值的像素比例，再按区域取平均",
        "缺陷像素里属于大缺陷的比例": r3(np.mean(share)),
        "高斯平滑的 sigma（px）": sigmas,
        "pixel AUROC": [r3(v[0]) for v in pm], "AUPRO": [r3(v[1]) for v in pm],
        "小缺陷的区域覆盖率": [r3(v[2]) for v in pm], "大缺陷的区域覆盖率": [r3(v[3]) for v in pm],
    }

    # (b) AUROC 约 0.99 时的工作点
    amp, n_each = 0.16, 1500
    train = dataset(0)[0]
    bank = feats(train).reshape(-1, 16)
    mem = bank[memory_order((0, False, 20, True), bank)]
    rng = np.random.default_rng([0, 5])
    normal, defect = [], []
    for start in range(0, n_each, 250):
        g = [good(rng) for _ in range(250)]
        b = [add_defect(good(rng), rng, amp)[0] for _ in range(250)]
        normal.append(image_score(score_maps(mem, feats(g))))
        defect.append(image_score(score_maps(mem, feats(b))))
    normal, defect = np.concatenate(normal), np.concatenate(defect)
    auc = mod.auroc(np.r_[normal, defect], np.r_[np.zeros(n_each), np.ones(n_each)].astype(int))
    points = {}
    for esc in (0.0, 0.01, 0.05, 0.10):
        thr, e, o = mod.threshold_at_escape(normal, defect, esc)
        assert abs(thr - fast_threshold(defect, esc)) < 1e-12
        prec = {f"缺陷率 {p:.1%}": r3(p * (1 - e) / (p * (1 - e) + (1 - p) * o)) if o > 0 else 1.0 for p in (0.01, 0.001)}
        points[f"漏检上限 {esc:.0%}"] = {"阈值": r3(thr), "实际漏检率": r3(e), "过检率": r3(o), "报警里真缺陷的比例": prec}
    NUM["从 AUROC 到工作点"] = {
        "说明": f"不对齐的主设置，缺陷幅度 {amp}，20 张良品、核心集 10%、k=1；{n_each} 张良品和 {n_each} 张缺陷图的整图分数。报警里真缺陷的比例按给定缺陷率换算",
        "image AUROC": r3(auc), **points,
    }

    # (c) 用多少个缺陷样本定阈值
    sizes, target = [20, 50, 100, 300], 0.05
    rng = np.random.default_rng([0, 6])
    val = {}
    for n in sizes:
        esc, ovk = [], []
        for _ in range(2000):
            pick = np.zeros(n_each, bool)
            pick[rng.choice(n_each, n, replace=False)] = True
            thr = fast_threshold(defect[pick], target)
            esc.append(float((defect[~pick] < thr).mean()))
            ovk.append(float((normal >= thr).mean()))
        esc = np.array(esc)
        val[n] = {"实际漏检率的中位数": r3(np.median(esc)), "实际漏检率的 5% 和 95% 分位": [r3(np.quantile(esc, 0.05)), r3(np.quantile(esc, 0.95))],
                  "实际漏检率超过 5% 的次数比例": r3(np.mean(esc > target)), "过检率的中位数": r3(np.median(ovk)), "_esc": esc}
    NUM["验证集大小与阈值"] = {
        "说明": f"同上一组分数。每次从 {n_each} 个缺陷分数里随机抽 n 个当验证集，按“漏检不超过 5%”定阈值，再用剩下的缺陷图算实际漏检率；每个 n 重复 2000 次",
        **{f"验证集有 {n} 个缺陷": {k: v for k, v in d.items() if not k.startswith("_")} for n, d in val.items()},
    }

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.0), gridspec_kw={"width_ratios": [1.1, 1.3, 1.1]}, layout="constrained")
    ax[0].plot(sigmas, [v[0] for v in pm], "o-", color=S.ORANGE, lw=1.8, ms=5, label="pixel AUROC")
    ax[0].plot(sigmas, [v[1] for v in pm], "s-", color=S.BLUE, lw=1.8, ms=5, label="AUPRO")
    ax[0].plot(sigmas, [v[2] for v in pm], "^--", color=S.RED, lw=1.5, ms=5, label="小缺陷的区域覆盖率")
    ax[0].plot(sigmas, [v[3] for v in pm], "v--", color=S.GREEN, lw=1.5, ms=5, label="大缺陷的区域覆盖率")
    ax[0].set_xlabel("对分数图做高斯平滑的 sigma（px）")
    ax[0].set_ylim(-0.04, 1.06)
    ax[0].set_title("小缺陷全丢了，pixel AUROC 几乎不动")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[0])
    bins = np.linspace(min(normal.min(), defect.min()), np.quantile(defect, 0.995), 60)
    ax[1].hist(normal, bins=bins, color=S.BLUE, alpha=0.55, label="良品图")
    ax[1].hist(defect, bins=bins, color=S.RED, alpha=0.45, label="缺陷图")
    top = ax[1].get_ylim()[1]
    for esc, color, yy in ((0.01, S.INK, 0.95), (0.05, S.MUTED, 0.72)):
        p = points[f"漏检上限 {esc:.0%}"]
        ax[1].axvline(p["阈值"], color=color, lw=1.3, ls="--")
        ax[1].annotate(f"漏检 ≤ {esc:.0%} 时\n过检 {100 * p['过检率']:.1f}%", xy=(p["阈值"], top * yy), xytext=(p["阈值"] + 0.03, top * yy),
                       fontsize=8.5, color=color, va="top", arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8},
                       bbox={"fc": "white", "ec": "none", "alpha": 0.9, "pad": 1.5})
    ax[1].set_xlabel("整图异常分数")
    ax[1].set_ylabel("图像张数")
    ax[1].set_title(f"AUROC {auc:.3f}，漏检压到 1% 时的过检率")
    ax[1].legend(fontsize=9, loc="center right")
    S.clean(ax[1])
    pos = np.arange(len(sizes))
    for i, n in enumerate(sizes):
        e = val[n]["_esc"]
        lo, mid, hi = np.quantile(e, [0.05, 0.5, 0.95])
        ax[2].plot([i, i], [100 * lo, 100 * hi], color=S.BLUE, lw=5, alpha=0.35, solid_capstyle="round")
        ax[2].plot(i, 100 * mid, "o", color=S.BLUE, ms=7)
    ax[2].axhline(100 * target, color=S.RED, lw=1.2, ls="--")
    ax[2].text(-0.45, 100 * target - 1.6, "要求的上限 5%", color=S.RED, fontsize=8.5, ha="left")
    ax[2].set_xticks(pos, [str(n) for n in sizes])
    ax[2].set_xlim(-0.5, len(sizes) - 0.5)
    ax[2].set_ylim(0, None)
    ax[2].set_xlabel("定阈值用的缺陷样本个数")
    ax[2].set_ylabel("上线后的实际漏检率（%）")
    ax[2].set_title("样本少时，实际漏检常常超过上限")
    S.clean(ax[2])
    S.save(fig, CH, "fig06_指标与阈值")


# ---------------------------------------------------------------- 表：重建类方法的容量
def table_reconstruction():
    dims = [1, 2, 4, 8, 16, 32, 48, 60]
    rows = []

    def patches(imgs):
        return np.stack([np.lib.stride_tricks.sliding_window_view(im, (8, 8))[::2, ::2].reshape(-1, 64) for im in imgs])

    for seed in SEEDS:
        train, tests, masks, labels = dataset(seed)
        X = patches(train).reshape(-1, 64)
        mu = X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
        T = patches(tests) - mu
        cum = np.cumsum((T @ Vt.T) ** 2, axis=-1)
        tot = (T ** 2).sum(axis=-1)
        row = []
        for k in dims:
            score = np.sqrt(np.maximum(tot - cum[..., k - 1], 0)).max(axis=1)
            row.append([mod.auroc(score, labels), np.median(score[labels == 0]), np.median(score[labels == 1])])
        rows.append(row)
    m = np.mean(rows, axis=0)
    NUM["重建类方法的容量"] = {
        "说明": "用主成分分析当线性自编码器：8x8 像素块（步长 2）共 64 维，只保留前 k 个主成分再重建，重建残差的范数当异常分数，整图分数取最大值。"
              "不对齐、缺陷幅度 0.12 的主设置，20 张良品，200 张待检图，3 个种子取平均",
        "保留的主成分个数 k": dims,
        "image AUROC": [r3(v[0]) for v in m],
        "良品图最大重建残差的中位数": [r3(v[1]) for v in m],
        "缺陷图最大重建残差的中位数": [r3(v[2]) for v in m],
    }


# ---------------------------------------------------------------- 表：预训练 ResNet-18 特征的对照
def table_resnet():
    try:
        import torch
        import torch.nn.functional as Fn
        import torchvision
        torch.set_num_threads(2)
        torch.manual_seed(0)
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()
    except Exception as exc:  # noqa: BLE001
        NUM["预训练特征对照"] = {"说明": f"没有运行：{type(exc).__name__}（需要 torch、torchvision 和 ResNet-18 的 ImageNet 权重）"}
        print("跳过 ResNet 对照：", exc)
        return
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def layers(imgs):
        x = torch.tensor(np.stack(imgs), dtype=torch.float32)[:, None]
        x = Fn.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False).clamp(0, 1).repeat(1, 3, 1, 1)
        x = (x - mean) / std
        out = {"layer1": [], "layer2": [], "layer3": []}
        for i in range(0, len(x), 50):
            h = net.maxpool(net.relu(net.bn1(net.conv1(x[i:i + 50]))))
            f1 = net.layer1(h)
            f2 = net.layer2(f1)
            f3 = net.layer3(f2)
            for k, f in (("layer1", f1), ("layer2", f2), ("layer3", f3)):
                out[k].append(f)
        return {k: torch.cat(v) for k, v in out.items()}

    def combine(fs, names, agg=True):
        sel = [fs[n] for n in names]
        if agg:
            sel = [Fn.avg_pool2d(f, 3, 1, 1) for f in sel]
        sel = [sel[0]] + [Fn.interpolate(f, size=sel[0].shape[-2:], mode="bilinear", align_corners=False) for f in sel[1:]]
        return torch.cat(sel, 1).permute(0, 2, 3, 1).numpy().astype(np.float64)

    def run(A, B, names, labels, masks=None, agg=True):
        a, b = combine(A, names, agg), combine(B, names, agg)
        h, d = a.shape[1], a.shape[-1]
        bank = a.reshape(-1, d)
        mem = bank[mod.greedy_coreset(bank, len(bank) // 10)]
        s = knn_multi(mem, b.reshape(-1, d), (1,))[1].reshape(len(b), h, h)
        out = [mod.auroc(s[:, 1:-1, 1:-1].reshape(len(s), -1).max(axis=1), labels), mod.auroc(s.reshape(len(s), -1).max(axis=1), labels)]
        if masks is not None:
            maps = [upsample(x, offset=(N / h - 1) / 2, stride=N / h)[4:-4, 4:-4] for x in s]
            out.append(aupro(maps, crop(masks)))
        return out, (h, d, len(bank))

    configs = [("layer1",), ("layer2",), ("layer3",), ("layer2", "layer3")]
    by_layer, by_radius, shape = {c: [] for c in configs}, {}, {}
    for radius, key in (((2, 4), "半径 2 到 3 px"), (5, "半径 5 px")):
        rows = []
        for seed in (0, 1):
            train, tests, masks, labels = dataset(seed, amp=0.2, radius=radius)
            A, B = layers(train), layers(tests)
            bank = feats(train).reshape(-1, 16)
            pix = mod.auroc(image_score(score_maps(bank[memory_order((seed, False, 20, True), bank)], feats(tests))), labels)
            for c in configs if radius == (2, 4) else configs[-1:]:
                res, shape[c] = run(A, B, c, labels, masks)
                if radius == (2, 4):
                    by_layer[c].append(res)
                if c == configs[-1]:
                    rows.append([pix, res[0]])
        by_radius[key] = np.mean(rows, axis=0)
    NUM["预训练特征对照"] = {
        "说明": "ResNet-18（ImageNet 权重，冻结）。把 64x64 的灰度图放大 2 倍到 128x128、复制成 3 通道，取中间层输出，做 3x3 平均池化；多层拼接时把深层上采样到浅层的尺寸。"
              "不对齐的条纹纹理，缺陷幅度 0.2，20 张良品，200 张待检图，核心集 10%，k=1，2 个种子取平均。整图分数和像素块一样去掉分数图最外一圈。"
              "合成纹理和真实工业图像差别很大，这组数字只说明这一组数据",
        "各层单独使用（缺陷半径 2 到 3 px）": {"+".join(c): {"特征图尺寸": f"{shape[c][0]}x{shape[c][0]}", "特征维度": shape[c][1], "记忆库大小": shape[c][2],
                                                   "image AUROC": r3(np.mean(v, axis=0)[0]), "image AUROC（不去掉最外一圈）": r3(np.mean(v, axis=0)[1]),
                                                   "AUPRO": r3(np.mean(v, axis=0)[2])} for c, v in by_layer.items()},
        "image AUROC 随缺陷大小（layer2+layer3 对 16 维像素块）": {k: {"16 维像素块": r3(v[0]), "ResNet-18 layer2+layer3": r3(v[1])} for k, v in by_radius.items()},
    }

    shifts, over = [0, 0.25, 0.5, 1, 2], {True: [], False: []}
    for seed in (0, 1):
        rng = np.random.default_rng([seed, 7])
        A = layers([good(rng, shift=0) for _ in range(20)])
        V = layers([good(rng, shift=0) for _ in range(200)])
        moved = {s: layers([good(rng, shift=s) for _ in range(100)]) for s in shifts}
        for agg in (True, False):
            a = combine(A, configs[-1], agg)
            bank = a.reshape(-1, a.shape[-1])
            mem = bank[mod.greedy_coreset(bank, len(bank) // 10)]
            score = lambda fs: knn_multi(mem, combine(fs, configs[-1], agg).reshape(-1, a.shape[-1]), (1,))[1].reshape(-1, a.shape[1], a.shape[2])[:, 1:-1, 1:-1].reshape(-1, (a.shape[1] - 2) ** 2).max(axis=1)  # noqa: E731
            thr = np.quantile(score(V), 0.99)
            over[agg].append([float(np.mean(score(moved[s]) > thr)) for s in shifts])
    NUM["预训练特征的错位容忍度"] = {
        "说明": "ResNet-18 layer2+layer3 特征（输入 128x128），记忆库只见过 20 张对齐的良品；阈值取 200 张对齐良品分数的 99% 分位，待检良品整体平移 s px（按原图 64x64 的像素算，每个 s 100 张），2 个种子取平均",
        "平移（px）": shifts,
        "过检率：做 3x3 平均池化": [r3(x) for x in np.mean(over[True], axis=0)],
        "过检率：不做池化": [r3(x) for x in np.mean(over[False], axis=0)],
        "各种子（做池化）": [[r3(x) for x in row] for row in over[True]],
        "各种子（不做池化）": [[r3(x) for x in row] for row in over[False]],
    }


def numbers_misc():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 16))
    sq = (x * x).sum(axis=1)
    raw = np.diag(sq[:, None] + sq[None, :] - 2.0 * x @ x.T)
    ref = np.diag(mod.pairwise_sq_dists(x, x))
    NUM["展开式的数值误差"] = {
        "说明": "500 个 16 维的标准正态随机点，用 ||a||^2 + ||b||^2 - 2a·b 算每个点到自己的平方距离（float64），真值都是 0",
        "不裁剪时小于 0 的个数": int((raw < 0).sum()), "不裁剪时的最小值": float(f"{raw.min():.2g}"),
        "参考实现（裁剪到 0）里小于 0 的个数": int((ref < 0).sum()),
    }
    NUM["AUROC 的分辨率"] = {
        "说明": "AUROC 等于“异常分数高于正常分数”的样本对所占的比例，所以最小变化量是 1/(异常个数×正常个数)",
        "Notebook 的 20 张良品加 20 张缺陷图": 1 / 400, "本章常用的 100 加 100": 1 / 10000,
    }
    NUM["理论常数"] = {
        "k-center greedy 的近似比（贪心解的覆盖半径不超过最优解的倍数，Gonzalez 1985 的经典结论）": 2,
        "本章 AUPRO 的积分上限（正常像素过检率）": 0.3,
    }
    q, mem = rng.normal(size=(40, 16)), rng.normal(size=(60, 16))
    for k in (1, 3):
        assert np.allclose(knn_multi(mem, q, (k,))[k], mod.knn_scores(mem, q, k)), "分块的 kNN 应和参考实现一致"


# ---------------------------------------------------------------- 表：聚合时边界换一种补法（Notebook 的“对照 1”）
def patch_features_valid(img):
    """和 patch_features 相同，但边界上只对真实存在的邻居平均（边上 6 个，角上 4 个），不做复制填充。"""
    pats = np.lib.stride_tricks.sliding_window_view(img, (4, 4))[::2, ::2]
    f = pats.reshape(pats.shape[0], pats.shape[1], -1).astype(np.float64)
    gh, gw = f.shape[:2]
    pad, ones = np.pad(f, ((1, 1), (1, 1), (0, 0))), np.pad(np.ones((gh, gw)), 1)
    count = sum(ones[dy:dy + gh, dx:dx + gw] for dy in range(3) for dx in range(3))
    return sum(pad[dy:dy + gh, dx:dx + gw] for dy in range(3) for dx in range(3)) / count[..., None]


def table_boundary_fill():
    rows, med = [], []
    for seed in SEEDS:
        train, tests, masks, labels = dataset(seed, aligned=False, amp=0.12)
        bank = np.stack([patch_features_valid(im) for im in train]).reshape(-1, 16)
        s = score_maps(bank[mod.greedy_coreset(bank, len(bank) // 10)], np.stack([patch_features_valid(im) for im in tests]))
        goods = s[labels == 0]
        rows.append([mod.auroc(image_score(s, inner=False), labels), mod.auroc(image_score(s), labels),
                     float(RING.ravel()[goods.reshape(len(goods), -1).argmax(axis=1)].mean())])
        med.append([np.median(goods[:, ~RING]), np.median(goods[:, RING & ~CORNER]), np.median(goods[:, CORNER])])
    v, m = np.mean(rows, axis=0), np.mean(med, axis=0)
    NUM["边界只对真实存在的邻居平均"] = {
        "说明": "不对齐、缺陷幅度 0.12 的主设置，其余条件和“局部聚合与边界”相同（20 张良品，200 张待检图，核心集 10%，k=1，3 个种子取平均），"
              "只把聚合时的边界处理从复制填充换成“只对真实存在的邻居平均”：边上的 patch 有 6 个邻居（含自己），角上有 4 个",
        "image AUROC（全图）": r3(v[0]), "image AUROC（去掉最外一圈）": r3(v[1]), "良品图的最高分落在最外一圈的比例": r3(v[2]),
        "各种子：image AUROC（全图）": [r3(x[0]) for x in rows], "各种子：image AUROC（去掉最外一圈）": [r3(x[1]) for x in rows],
        "良品 patch 分数的中位数": {"内部": r3(m[0]), "边上（不含角）": r3(m[1]), "四个角": r3(m[2])},
        "邻域平均后的噪声方差相对单个 patch 的比例（1 除以邻居个数）": {"内部": r3(1 / 9), "边上": r3(1 / 6), "角": r3(1 / 4)},
    }
    NUM["AUROC 的分辨率"]["Notebook 对照单元的 50 张良品加 50 张缺陷图"] = 1 / 2500


def main() -> None:
    S.setup()
    plt.rcParams["svg.hashsalt"] = "ch19"
    t_all = time.perf_counter()
    for fn in (numbers_misc, fig_methods, fig_pipeline, fig_coreset, fig_contamination, fig_goods_shift, fig_metrics, table_reconstruction, table_resnet, table_boundary_fill):
        t = time.perf_counter()
        fn()
        print(f"完成 {fn.__name__}（{time.perf_counter() - t:.0f} 秒）", flush=True)
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=2))
    print(f"总耗时 {time.perf_counter() - t_all:.0f} 秒")


if __name__ == "__main__":
    main()
