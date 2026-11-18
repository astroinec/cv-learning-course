"""第 18 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch18.py

输出：<主分支>/18_传统缺陷检测/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

CH = "18_传统缺陷检测"
mod = S.load_solution(CH, "18-1_金模板比对.py")
NUM: dict = {}
READ, GAIN, JITTER = 1.5, 0.25, 0.15


# ---------------------------------------------------------------- 数据与工具
def die_layout(n=96):
    """合成 die：左上是线阵列，右上是亮的平坦焊盘，下半是暗的平坦区和一块中灰区。"""
    img = np.full((n, n), 40.0)
    y, x = np.mgrid[0:n, 0:n]
    arr = (y >= 8) & (y < 44) & (x >= 8) & (x < 44)
    img[arr & ((x % 8) < 4)] = 120.0
    img[8:44, 52:88] = 180.0
    img[60:80, 50:86] = 100.0
    return img


def blur(img, sigma):
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.fftfreq(img.shape[1])[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2 * np.pi ** 2 * sigma ** 2 * (fy ** 2 + fx ** 2))))


def fshift(img, dy, dx):
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.fftfreq(img.shape[1])[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j * np.pi * (fy * dy + fx * dx))))


BASE = blur(die_layout(), 0.8)
REGIONS = {"暗的平坦区": np.s_[52:90, 6:46], "亮的平坦区": np.s_[12:40, 56:84], "线阵列区": np.s_[10:42, 10:42]}


def noise_sigma(img, read=READ, gain=GAIN):
    return np.sqrt(read ** 2 + gain * np.clip(img, 0, None))


def capture(rng, jitter=JITTER, read=READ, gain=GAIN):
    """拍一张良品：亚像素的对齐残差，加上随亮度变大的噪声。"""
    dy, dx = rng.normal(scale=jitter, size=2) if jitter else (0.0, 0.0)
    img = fshift(BASE, dy, dx) if jitter else BASE
    return img + rng.normal(size=img.shape) * noise_sigma(img, read, gain)


def add_defects(img, rng, n=5, amp=(15, 40)):
    out, truth = img.copy(), []
    for _ in range(n):
        cy, cx = rng.integers(8, 88, 2)
        out[cy - 1:cy + 2, cx - 1:cx + 2] += rng.choice([-1, 1]) * rng.uniform(*amp)
        truth.append((float(cy), float(cx)))
    return out, truth


def evaluate(tests, median, sigma, k, min_area=3, radius=3.0, truth_key=1):
    tp = fp = n = 0
    for item in tests:
        found = mod.detect_defects(item[0], median, sigma, k=k, min_area=min_area)
        m = mod.match_defects([d["centroid"] for d in found], item[truth_key], radius)
        tp += m["tp"]
        fp += m["fp"]
        n += len(item[truth_key])
    return tp / n, fp / len(tests)


def best_capture(curve, limit):
    ok = [c for c, f in curve if f <= limit]
    return round(max(ok), 3) if ok else 0.0


# ---------------------------------------------------------------- 图 1：固定阈值对逐像素 z
def fig_threshold():
    rng = np.random.default_rng(0)
    goods = np.stack([capture(rng) for _ in range(30)])
    median, sigma = mod.robust_golden(goods)
    tests = [add_defects(capture(rng), rng) for _ in range(60)]
    ones = np.ones_like(sigma)
    Ts = [8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 48]
    ks = [3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 10]
    fixed = [evaluate(tests, median, ones, T) for T in Ts]
    zs = [evaluate(tests, median, sigma, k) for k in ks]

    T0 = 4 * float(np.median(sigma))
    fresh = np.stack([capture(rng) for _ in range(200)])
    exceed = (np.abs(fresh - median) > T0).mean(axis=0)
    gy, gx = np.gradient(BASE)
    edge = np.hypot(gy, gx) > 8
    share = {"边缘像素占图像面积的比例": round(float(edge.mean()), 3),
             "超阈值事件里落在边缘像素上的比例": round(float(exceed[edge].sum() / exceed.sum()), 3),
             "边缘像素的平均超阈值概率": round(float(exceed[edge].mean()), 4),
             "亮的平坦区的平均超阈值概率": round(float(exceed[REGIONS["亮的平坦区"]].mean()), 4),
             "暗的平坦区的平均超阈值概率": round(float(exceed[REGIONS["暗的平坦区"]].mean()), 6)}
    NUM["固定阈值与逐像素阈值"] = {
        "说明": "96x96 合成 die，30 张良品建模板，60 张待检图各注入 5 个 3x3 缺陷（幅度 15 到 40，正负随机），min_area=3，匹配半径 3 px；"
              f"读出噪声 {READ}、散粒噪声系数 {GAIN}、对齐残差标准差 {JITTER} px",
        "各区域 sigma 的中位数": {name: round(float(np.median(sigma[sl])), 2) for name, sl in REGIONS.items()},
        "每张图误报不超过 0.1 个时的最高检出率": {"固定阈值": best_capture(fixed, 0.1), "逐像素 z": best_capture(zs, 0.1)},
        "每张图误报不超过 1 个时的最高检出率": {"固定阈值": best_capture(fixed, 1.0), "逐像素 z": best_capture(zs, 1.0)},
        "固定阈值曲线": {str(T): [round(c, 3), round(f, 2)] for T, (c, f) in zip(Ts, fixed)},
        "逐像素 z 曲线": {str(k): [round(c, 3), round(f, 2)] for k, (c, f) in zip(ks, zs)},
        "固定阈值取 4 倍 sigma 中位数时，良品上超阈值的分布（边缘指亮度梯度大于 8 的像素）": share,
        "上一项所用的固定阈值": round(T0, 1),
    }

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.9), gridspec_kw={"width_ratios": [1, 1.08, 1.08, 1.35]}, layout="constrained")
    ax[0].imshow(BASE, cmap="gray", vmin=0, vmax=200)
    box = {"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.5}
    for text, (ty, tx) in {"线阵列区": (49, 9), "亮的平坦区": (49, 53), "暗的平坦区": (70, 9)}.items():
        ax[0].text(tx, ty, text, fontsize=8.5, color=S.INK, bbox=box)
    ax[0].set_title("合成 die 的三种区域")
    im = ax[1].imshow(sigma, cmap="viridis")
    ax[1].set_title("逐像素 sigma（30 张良品）")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03).ax.tick_params(labelsize=8)
    im2 = ax[2].imshow(exceed, cmap="magma")
    ax[2].set_title(f"固定阈值 {T0:.0f}：良品上的超限频率")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.03).ax.tick_params(labelsize=8)
    for a in ax[:3]:
        S.image_axes(a)
    fx = [max(f, 0.02) for _, f in fixed]
    zx = [max(f, 0.02) for _, f in zs]
    ax[3].plot(fx, [c for c, _ in fixed], "s-", color=S.ORANGE, ms=4, lw=1.8, label="固定灰度差阈值")
    ax[3].plot(zx, [c for c, _ in zs], "o-", color=S.BLUE, ms=4, lw=1.8, label="逐像素 z 阈值")
    ax[3].set_xscale("log")
    ax[3].set_xticks([0.02, 0.1, 1, 10, 100], ["≤0.02", "0.1", "1", "10", "100"])
    ax[3].minorticks_off()
    ax[3].axvline(0.1, color=S.MUTED, lw=1, ls="--")
    ax[3].set_xlabel("每张图的误报个数")
    ax[3].set_ylabel("检出率")
    ax[3].set_ylim(0, 1.03)
    ax[3].set_title("同样的误报下谁检出更多")
    ax[3].legend(fontsize=9, loc="lower right")
    S.clean(ax[3])
    S.save(fig, CH, "fig02_固定阈值与逐像素阈值")


# ---------------------------------------------------------------- 图 2：良品张数与实际误报
def fig_sample_size():
    theory = math.erfc(5 / math.sqrt(2))
    Ns = [5, 9, 15, 30, 100]
    rng = np.random.default_rng(1)
    prob = {"中位数 + MAD": [], "均值 + 标准差": []}
    comps = {"中位数 + MAD": [], "均值 + 标准差": []}
    ratios = {}
    for N in Ns:
        cnt = dict.fromkeys(prob, 0)
        comp = dict.fromkeys(prob, 0)
        total = nimg = 0
        keep = []
        for _ in range(30):
            goods = 100 + rng.normal(scale=3.0, size=(N, 96, 96))
            med, sig = mod.robust_golden(goods, min_sigma=1.0)
            models = {"中位数 + MAD": (med, sig), "均值 + 标准差": (goods.mean(0), np.maximum(goods.std(0, ddof=1), 1.0))}
            fresh = 100 + rng.normal(scale=3.0, size=(20, 96, 96))
            for name, (c, s) in models.items():
                cnt[name] += int((np.abs(fresh - c) / s > 5).sum())
                comp[name] += sum(len(mod.detect_defects(img, c, s, k=5, min_area=3)) for img in fresh[:4])
            total += fresh.size
            nimg += 4
            keep.append(sig.ravel() / 3.0)
        for name in prob:
            prob[name].append(cnt[name] / total)
            comps[name].append(comp[name] / nimg)
        ratios[N] = np.concatenate(keep)
    floor = 1 / (30 * 20 * 96 * 96)
    NUM["良品张数与实际误报"] = {
        "说明": "纯高斯噪声（标准差 3），k=5，min_sigma=1.0；每个 N 重复 30 次建模板，每次用 20 张新的良品统计，共约 553 万个像素",
        "高斯理论值（sigma 已知时单像素超过 5 sigma 的概率）": float(f"{theory:.3g}"),
        "良品张数": Ns,
        "单像素误报概率": {name: [float(f"{max(p, 0):.3g}") for p in vals] for name, vals in prob.items()},
        "本实验能分辨的最小概率": float(f"{floor:.2g}"),
        "每张图面积不小于 3 的误报个数": {name: [round(v, 2) for v in vals] for name, vals in comps.items()},
        "MAD 估计的 sigma 与真值之比的 5% 和 95% 分位": {str(N): [round(float(np.percentile(r, 5)), 2), round(float(np.percentile(r, 95)), 2)] for N, r in ratios.items()},
    }

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9), layout="constrained")
    for name, color, marker in (("中位数 + MAD", S.BLUE, "o"), ("均值 + 标准差", S.ORANGE, "s")):
        ax[0].plot(Ns, [max(p, floor) for p in prob[name]], marker + "-", color=color, lw=1.8, ms=5, label=name)
    ax[0].axhline(theory, color=S.GREEN, lw=1.3, ls="--", label="sigma 已知时的理论值 5.7e-7")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xticks(Ns, [str(n) for n in Ns])
    ax[0].set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], ["1e-6", "1e-5", "1e-4", "0.001", "0.01", "0.1"])
    ax[0].minorticks_off()
    ax[0].set_xlabel("建模板用的良品张数 N")
    ax[0].set_ylabel("良品像素超过 k=5 的概率")
    ax[0].set_title("良品太少时，k=5 远没有理论上那么严")
    ax[0].legend(fontsize=9)
    S.clean(ax[0])
    bins = np.linspace(0, 2.5, 76)
    for N, color in ((5, S.RED), (15, S.ORANGE), (100, S.BLUE)):
        ax[1].hist(ratios[N], bins=bins, density=True, histtype="step", lw=1.8, color=color, label=f"N = {N}")
    ax[1].axvline(1, color=S.MUTED, lw=1, ls="--")
    ax[1].set_xlabel("估计的 sigma / 真实 sigma")
    ax[1].set_ylabel("像素占比（密度）")
    ax[1].set_title("原因：sigma 本身估不准，低估的像素会误报")
    ax[1].legend(fontsize=9)
    S.clean(ax[1])
    S.save(fig, CH, "fig04_良品张数与误报")


# ---------------------------------------------------------------- 图 3：良品里混进坏图
def fig_contamination():
    base = die_layout()
    rng = np.random.default_rng(2)
    names = ["中位数 + MAD", "均值 + 标准差"]
    ms = list(range(0, 7))
    inside = {n: [] for n in names}
    control = {n: [] for n in names}
    falarm = {n: [] for n in names}
    example = {}
    trials = 200
    for m in ms:
        hit = dict.fromkeys(names, 0)
        ctl = dict.fromkeys(names, 0)
        fa = dict.fromkeys(names, 0)
        for t in range(trials):
            goods = base + rng.normal(scale=3.0, size=(12, 96, 96))
            goods[:m, 56:72, 14:30] += 80.0
            test = base + rng.normal(scale=3.0, size=(96, 96))
            test[63:66, 21:24] += 40.0
            test[19:22, 69:72] += 40.0
            med, sig = mod.robust_golden(goods)
            models = {"中位数 + MAD": (med, sig), "均值 + 标准差": (goods.mean(0), np.maximum(goods.std(0, ddof=1), 1.0))}
            if m == 2 and t == 0:
                example = {k: v for k, v in models.items()}
            for name, (c, s) in models.items():
                cents = [d["centroid"] for d in mod.detect_defects(test, c, s, k=6, min_area=3)]
                near_in = [p for p in cents if math.hypot(p[0] - 64, p[1] - 22) <= 3]
                near_ct = [p for p in cents if math.hypot(p[0] - 20, p[1] - 70) <= 3]
                hit[name] += bool(near_in)
                ctl[name] += bool(near_ct)
                fa[name] += len(cents) - len(near_in) - len(near_ct)
        for name in names:
            inside[name].append(hit[name] / trials)
            control[name].append(ctl[name] / trials)
            falarm[name].append(fa[name] / trials)
    NUM["良品里混进坏图"] = {
        "说明": "12 张良品，其中 m 张在 16x16 的区域里整体偏亮 80；待检图在该区域内和区域外各有一个 3x3、幅度 40 的缺陷；噪声标准差 3，k=6，min_area=3，每个 m 做 200 次",
        "被污染的良品张数 m": ms,
        "污染区域内缺陷的检出率": {n: [round(v, 3) for v in inside[n]] for n in names},
        "区域外对照缺陷的检出率": {n: [round(v, 3) for v in control[n]] for n in names},
        "每张图的误报个数": {n: [round(v, 3) for v in falarm[n]] for n in names},
        "m=2 时污染区域内 sigma 的平均值": {n: round(float(example[n][1][56:72, 14:30].mean()), 1) for n in names},
        "m=2 时污染区域内模板相对真值的平均偏移": {n: round(float((example[n][0] - base)[56:72, 14:30].mean()), 1) for n in names},
    }

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1, 1.08, 1.08, 1.4]}, layout="constrained")
    ax[0].imshow(example["均值 + 标准差"][0], cmap="gray", vmin=0, vmax=200)
    ax[0].add_patch(plt.Rectangle((13.5, 55.5), 16, 16, fill=False, ec=S.RED, lw=1.5, ls="--"))
    ax[0].set_title("均值模板：坏图留下了“鬼影”")
    vmax = float(example["均值 + 标准差"][1].max())
    im = ax[1].imshow(example["均值 + 标准差"][1], cmap="viridis", vmin=0, vmax=vmax)
    ax[1].set_title("标准差：该区域被撑大")
    im2 = ax[2].imshow(example["中位数 + MAD"][1], cmap="viridis", vmin=0, vmax=vmax)
    ax[2].set_title("1.4826 × MAD：基本不受影响")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.03).ax.tick_params(labelsize=8)
    for a in ax[:3]:
        S.image_axes(a)
    ax[3].plot(ms, inside["中位数 + MAD"], "o-", color=S.BLUE, lw=1.8, ms=5, label="中位数 + MAD")
    ax[3].plot(ms, inside["均值 + 标准差"], "s-", color=S.ORANGE, lw=1.8, ms=5, label="均值 + 标准差")
    ax[3].axvline(6, color=S.MUTED, lw=1, ls="--")
    ax[3].text(5.9, 0.5, "一半良品被污染", fontsize=8.5, color=S.MUTED, ha="right")
    ax[3].set_xlabel("12 张良品里被污染的张数 m")
    ax[3].set_ylabel("污染区域内缺陷的检出率")
    ax[3].set_ylim(-0.03, 1.05)
    ax[3].set_title("混进 1 张坏图，均值法就漏检")
    ax[3].legend(fontsize=9, loc="center", bbox_to_anchor=(0.42, 0.5))
    S.clean(ax[3])
    S.save(fig, CH, "fig03_良品里混进坏图")


# ---------------------------------------------------------------- 图 4：k、面积过滤、匹配半径
def fig_operating_point():
    rng = np.random.default_rng(3)
    goods = np.stack([capture(rng) for _ in range(12)])
    median, sigma = mod.robust_golden(goods)
    tests = [add_defects(capture(rng), rng) for _ in range(60)]
    ks = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 9]
    curves = {a: [evaluate(tests, median, sigma, k, min_area=a) for k in ks] for a in (1, 3)}

    weak = []
    for _ in range(60):
        img, truth = add_defects(capture(rng), rng, amp=(10, 25))
        reported = [(cy + rng.normal(scale=1.5), cx + rng.normal(scale=1.5)) for cy, cx in truth]
        weak.append((img, truth, reported))
    radii = [1, 2, 3, 5, 8, 12, 20]
    by_radius = [evaluate(weak, median, sigma, 4.5, min_area=2, radius=r, truth_key=2) for r in radii]
    exact = evaluate(weak, median, sigma, 4.5, min_area=2, radius=3.0, truth_key=1)
    NUM["工作点与匹配规则"] = {
        "说明": "同图 2 的合成 die，12 张良品建模板，60 张待检图各 5 个缺陷；左图缺陷幅度 15 到 40，匹配半径 3 px",
        "k": ks,
        "min_area=1（检出率，每张图误报）": [[round(c, 3), round(f, 2)] for c, f in curves[1]],
        "min_area=3（检出率，每张图误报）": [[round(c, 3), round(f, 2)] for c, f in curves[3]],
        "匹配半径实验": {
            "说明": "弱缺陷（幅度 10 到 25），k=4.5，min_area=2；真值坐标带标准差 1.5 px 的定位误差，检测结果完全相同，只改匹配半径",
            "半径": radii,
            "检出率": [round(c, 3) for c, _ in by_radius],
            "每张图误报": [round(f, 2) for _, f in by_radius],
            "真值坐标没有误差、半径 3 时（检出率，每张图误报）": [round(exact[0], 3), round(exact[1], 2)],
        },
    }

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.9), layout="constrained")
    ax2 = ax[0].twinx()
    for a, ls in ((1, "--"), (3, "-")):
        ax[0].plot(ks, [c for c, _ in curves[a]], "o" + ls, color=S.BLUE, lw=1.8, ms=4, label=f"检出率，min_area={a}")
        ax2.plot(ks, [max(f, 0.02) for _, f in curves[a]], "s" + ls, color=S.RED, lw=1.8, ms=4, label=f"误报，min_area={a}")
    ax[0].set_xlabel("阈值 k")
    ax[0].set_ylabel("检出率", color=S.BLUE)
    ax[0].set_ylim(0, 1.03)
    ax2.set_yscale("log")
    ax2.set_yticks([0.02, 0.1, 1, 10, 100], ["≤0.02", "0.1", "1", "10", "100"])
    ax2.minorticks_off()
    ax2.set_ylabel("每张图的误报个数（对数轴）", color=S.RED)
    ax[0].set_title("k 和面积过滤怎样移动工作点")
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[0])
    ax2.spines["top"].set_visible(False)

    ax3 = ax[1].twinx()
    ax[1].plot(radii, [c for c, _ in by_radius], "o-", color=S.BLUE, lw=1.8, ms=5)
    ax3.plot(radii, [f for _, f in by_radius], "s-", color=S.RED, lw=1.8, ms=5)
    ax[1].axhline(exact[0], color=S.BLUE, lw=1, ls=":")
    ax[1].text(radii[-1], exact[0] - 0.075, "虚线：真值坐标无误差、半径 3 px 时的检出率", color=S.BLUE, fontsize=8.5, ha="right")
    ax[1].set_xlabel("匹配半径（px）")
    ax[1].set_ylabel("检出率", color=S.BLUE)
    ax[1].set_ylim(0, 1.03)
    ax3.set_ylabel("每张图的误报个数", color=S.RED)
    ax3.set_ylim(bottom=0)
    ax[1].set_title("检测结果不变，只改匹配半径")
    S.clean(ax[1])
    ax3.spines["top"].set_visible(False)
    S.save(fig, CH, "fig06_工作点与匹配规则")


# ---------------------------------------------------------------- 图 5：双比与三比
def fig_arbitration():
    rng = np.random.default_rng(4)
    s = noise_sigma(BASE)
    sd = np.sqrt(2) * s
    k = 4.0

    def die():
        return BASE + rng.normal(size=BASE.shape) * s

    def arbitrate(X, Y, Z, kk):
        d1, d2 = (X - Y) / sd, (X - Z) / sd
        return (np.abs(d1) > kk) & (np.abs(d2) > kk) & (np.sign(d1) == np.sign(d2))

    def blobs(mask):
        labels, n = mod.label_components(mask, 8)
        out = []
        for i in range(1, n + 1):
            ys, xs = np.nonzero(labels == i)
            if len(ys) >= 3:
                out.append((float(ys.mean()), float(xs.mean())))
        return out

    st = dict.fromkeys(["双比报出差异", "三比判给 B", "三比判给 A 或 C", "重复缺陷被双比报出", "重复缺陷被三比报出"], 0)
    noise_double = noise_triple = 0
    trials = 200
    example = None
    for t in range(trials):
        cy, cx = rng.integers(8, 88, 2)
        amp = rng.choice([-1, 1]) * 35.0
        A, B, C = die(), die(), die()
        clean = (A.copy(), B.copy(), C.copy())
        B[cy - 1:cy + 2, cx - 1:cx + 2] += amp
        near = lambda pts: any(math.hypot(p[0] - cy, p[1] - cx) <= 3 for p in pts)  # noqa: E731
        st["双比报出差异"] += near([d["centroid"] for d in mod.detect_defects(A, B, sd, k=k, min_area=3)])
        st["三比判给 B"] += near(blobs(arbitrate(B, A, C, k)))
        st["三比判给 A 或 C"] += near(blobs(arbitrate(A, B, C, k))) or near(blobs(arbitrate(C, A, B, k)))
        noise_double += int((np.abs(clean[0] - clean[1]) / sd > 3).sum())
        noise_triple += int(arbitrate(clean[0], clean[1], clean[2], 3).sum())
        R = [img.copy() for img in clean]
        for img in R:
            img[cy - 1:cy + 2, cx - 1:cx + 2] += amp
        st["重复缺陷被双比报出"] += near([d["centroid"] for d in mod.detect_defects(R[0], R[1], sd, k=k, min_area=3)])
        st["重复缺陷被三比报出"] += near(blobs(arbitrate(R[1], R[0], R[2], k)))
        if example is None and amp > 0:
            example = (A, B, C, int(cy), int(cx))
    NUM["双比与三比"] = {
        "说明": "三张相邻 die（A、B、C），缺陷在 B 上（3x3，幅度 35）；差值按 sqrt(2) 倍的像素噪声归一化，k=4，面积不小于 3，共 200 次",
        "次数（共 200）": {name: int(v) for name, v in st.items()},
        "无缺陷时每张图超过 k=3 的像素数": {"双比 |A-B|": round(noise_double / trials, 1), "三比（A 与 B、C 都不同且同号）": round(noise_triple / trials, 2)},
    }

    A, B, C, cy, cx = example
    zab, zcb = np.abs(A - B) / sd, np.abs(C - B) / sd
    arb = arbitrate(B, A, C, k)
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]}, layout="constrained")
    for a, img, title in ((ax[0], zab, "|A - B|：A 和 B 谁有缺陷？"), (ax[1], zcb, "|C - B|：同一位置又出现")):
        a.imshow(img, cmap="magma", vmin=0, vmax=8)
        a.plot(cx, cy, "o", mfc="none", mec="white", ms=16, mew=1.5)
        a.set_title(title)
        S.image_axes(a)
    ax[2].imshow(arb, cmap="gray_r")
    ax[2].plot(cx, cy, "o", mfc="none", mec=S.RED, ms=16, mew=1.5)
    ax[2].set_title("仲裁：B 和两边都不同，缺陷在 B")
    S.image_axes(ax[2])
    vals = [noise_double / trials, noise_triple / trials]
    bars = ax[3].bar(["双比\n|A - B|", "三比仲裁"], vals, color=[S.ORANGE, S.BLUE], alpha=0.85, width=0.5)
    for b, v in zip(bars, vals):
        ax[3].text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.1f}", ha="center", fontsize=10)
    ax[3].set_ylabel("无缺陷时超过 k=3 的像素数（每张图）")
    ax[3].set_title("仲裁顺带压低了噪声误报")
    S.clean(ax[3])
    ax[3].grid(axis="x", visible=False)
    S.save(fig, CH, "fig01_双比与三比")


# ---------------------------------------------------------------- 图 6：整体亮度变化
def fig_gain():
    cams = {"本章的噪声水平": (READ, GAIN), "噪声低一个档次的相机": (0.5, 0.02)}
    gains = [round(1 + 0.02 * i, 2) for i in range(11)]
    bright = REGIONS["亮的平坦区"]
    before, after, limit, snr = {}, {}, {}, {}
    example = None
    for name, (read, g) in cams.items():
        rng = np.random.default_rng(5)
        goods = np.stack([capture(rng, read=read, gain=g) for _ in range(30)])
        median, sigma = mod.robust_golden(goods, min_sigma=0.5)
        design = np.stack([median.ravel(), np.ones(median.size)], axis=1)
        level, noise = float(np.median(median[bright])), float(np.median(sigma[bright]))
        limit[name] = 5 * noise / level
        snr[name] = level / noise
        before[name], after[name] = [], []
        for gain in gains:
            f0 = f1 = 0.0
            for _ in range(20):
                img = capture(rng, read=read, gain=g) * gain
                z0 = np.abs(img - median) / sigma
                (a, b), *_ = np.linalg.lstsq(design, img.ravel(), rcond=None)
                z1 = np.abs((img - b) / a - median) / sigma
                f0 += float((z0[bright] > 5).mean())
                f1 += float((z1[bright] > 5).mean())
                if example is None and name != "本章的噪声水平" and gain == 1.08:
                    example = (z0, z1)
            before[name].append(f0 / 20)
            after[name].append(f1 / 20)
    NUM["整体亮度变化"] = {
        "说明": "同图 2 的合成 die，30 张良品建模板（min_sigma 取 0.5）；良品图整体乘一个增益，统计亮的平坦区（灰度约 180）里 z 超过 5 的像素比例，各 20 张；"
              "归一化指用最小二乘把待检图线性拟合到模板再比对",
        "增益": gains,
        "亮的平坦区的信噪比（灰度 / sigma）": {n: round(v, 1) for n, v in snr.items()},
        "按 k·sigma/灰度 估算的整片报警门槛": {n: f"{100 * v:.1f}%" for n, v in limit.items()},
        "归一化之前，区域内被报警的像素比例": {n: [round(v, 3) for v in vals] for n, vals in before.items()},
        "归一化之后，区域内被报警的像素比例": {n: [round(v, 4) for v in vals] for n, vals in after.items()},
    }

    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1, 1.08, 1.5]}, layout="constrained")
    ax[0].imshow(example[0], cmap="magma", vmin=0, vmax=10)
    ax[0].set_title("整体亮 8%：亮区整片报警")
    im = ax[1].imshow(example[1], cmap="magma", vmin=0, vmax=10)
    ax[1].set_title("亮度归一化之后的 z 图")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03).ax.tick_params(labelsize=8)
    for a in ax[:2]:
        S.image_axes(a)
    pct = [100 * (g - 1) for g in gains]
    for (name, color) in zip(cams, (S.ORANGE, S.RED)):
        ax[2].plot(pct, before[name], "o-", color=color, lw=1.8, ms=4, label=f"{name}，不归一化")
        ax[2].axvline(100 * limit[name], color=color, lw=1, ls=":")
    ax[2].plot(pct, after["噪声低一个档次的相机"], "s--", color=S.GREEN, lw=1.6, ms=4, label="归一化之后（两种相机都贴近 0）")
    ax[2].set_xlabel("待检图比模板整体亮多少（%）")
    ax[2].set_ylabel("亮区里 z > 5 的像素比例")
    ax[2].set_title("相机越好，越怕亮度漂移")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    S.save(fig, CH, "fig05_整体亮度变化")


def numbers_theory():
    """讲义里引用的两组理论数字：k 对应的高斯尾概率，和小样本检出率的置信区间。"""
    tail = {str(k): math.erfc(k / math.sqrt(2)) for k in (3, 4, 5, 6)}
    NUM["k 的理论含义"] = {
        "说明": "sigma 准确已知、噪声为高斯时，单个良品像素 |z| 超过 k 的概率，以及折算到 400 万像素的图上的期望个数",
        "概率": {k: float(f"{v:.3g}") for k, v in tail.items()},
        "400 万像素图上的期望超限像素数": {k: float(f"{4e6 * v:.3g}") for k, v in tail.items()},
    }
    NUM["参考图的噪声"] = {"说明": "差值噪声相对单张图噪声的倍数；金模板一项按高斯噪声下中位数的渐近方差 pi/(2N) 估算",
                      "双比（两张图相减）": round(math.sqrt(2), 3), "12 张良品的中位数做参考": round(math.sqrt(1 + math.pi / 24), 3)}
    hit, n, z = 28, 30, 1.96
    ph = hit / n
    center = (ph + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    NUM["置信区间示例"] = {"说明": "30 个真缺陷检出 28 个，Wilson 95% 置信区间", "检出率": round(ph, 3),
                     "区间": [round(center - half, 3), round(center + half, 3)]}


def main() -> None:
    S.setup()
    numbers_theory()
    for fn in (fig_arbitration, fig_threshold, fig_contamination, fig_sample_size, fig_gain, fig_operating_point):
        fn()
        print("完成", fn.__name__, flush=True)
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
