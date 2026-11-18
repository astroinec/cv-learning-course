"""第 16 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch16.py

输出：<主分支>/16_图像对齐与亚像素/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

CH = "16_图像对齐与亚像素"
mod = S.load_solution(CH, "16-1_相位相关与亚像素.py")
parabola = mod.parabola_peak_offset
NUM: dict = {}


# ---------------------------------------------------------------- 数据与工具
def texture(shape=(96, 96), seed=0, width=0.08):
    rng = np.random.default_rng(seed)
    f = np.fft.fft2(rng.normal(size=shape))
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    return np.real(np.fft.ifft2(f * np.exp(-(fy ** 2 + fx ** 2) / (2 * width ** 2))))


def fshift(img, dy, dx):
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.fftfreq(img.shape[1])[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j * np.pi * (fy * dy + fx * dx))))


def blur(img, sigma):
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.fftfreq(img.shape[1])[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2 * np.pi ** 2 * sigma ** 2 * (fy ** 2 + fx ** 2))))


def layout(n=128, period=None, mark=False, seed=4):
    """合成版图。period 给定时是严格周期结构，mark=True 时加一个十字对位标记。"""
    img = np.full((n, n), 0.15)
    if period:
        y, x = np.mgrid[0:n, 0:n]
        img[(x % period) < period // 3] = 0.8
        img[((y % 16) < 3) & ((x % period) >= period // 2)] = 0.55
    else:
        rng = np.random.default_rng(seed)
        for _ in range(14):
            y0, x0 = rng.integers(4, n - 30, size=2)
            if rng.random() < 0.5:
                img[y0:y0 + 3, x0:x0 + rng.integers(15, 60)] = 0.8
            else:
                img[y0:y0 + rng.integers(15, 60), x0:x0 + 3] = 0.8
        for _ in range(6):
            y0, x0 = rng.integers(8, n - 20, size=2)
            img[y0:y0 + 10, x0:x0 + 10] = 0.6
    if mark:
        img[20:23, 14:37] = 1.0
        img[10:33, 24:27] = 1.0
    return img


def surface(a, b, mode="phase", weight=None):
    """相关面。mode=phase 是相位相关，mode=cc 是不做白化的普通互相关。"""
    fa = np.fft.fft2(a - a.mean())
    fb = np.fft.fft2(b - b.mean())
    cross = fb * np.conj(fa)
    if mode == "phase":
        cross = cross / (np.abs(cross) + 1e-12)
    if weight is not None:
        cross = cross * weight
    return np.real(np.fft.ifft2(cross))


def peak(r, radius=None):
    n0, n1 = r.shape
    iy = np.fft.fftfreq(n0, 1 / n0)
    ix = np.fft.fftfreq(n1, 1 / n1)
    rr = r
    if radius is not None:
        keep = (np.abs(iy)[:, None] <= radius) & (np.abs(ix)[None, :] <= radius)
        rr = np.where(keep, r, -np.inf)
    py, px = np.unravel_index(np.argmax(rr), rr.shape)
    oy = parabola(r[(py - 1) % n0, px], r[py, px], r[(py + 1) % n0, px])
    ox = parabola(r[py, (px - 1) % n1], r[py, px], r[py, (px + 1) % n1])
    return float(iy[py] + oy), float(ix[px] + ox)


def magnitude_mask(a, ratio=0.1):
    mag = np.abs(np.fft.fft2(a - a.mean()))
    return (mag > ratio * mag.max()).astype(float)


# ---------------------------------------------------------------- 图 1：错位的代价
def fig_misalignment():
    base = blur(layout(128), 1.0)
    sigma = 0.01
    limit = 5 * sigma * np.sqrt(2)
    shifts = np.round(np.linspace(0, 1, 21), 2)
    counts = []
    for d in shifts:
        c = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            a = base + rng.normal(scale=sigma, size=base.shape)
            b = fshift(base, 0, d) + rng.normal(scale=sigma, size=base.shape)
            c.append(int((np.abs(b - a) > limit).sum()))
        counts.append(float(np.mean(c)))
    NUM["错位假差异"] = {
        "说明": "128x128 合成版图，高斯模糊 1 px，线条对比度 0.65，每张图噪声标准差 0.01，阈值为差值噪声的 5 倍标准差，20 次平均",
        "超过阈值的像素数": {f"{d:.2f}": round(c, 1) for d, c in zip(shifts, counts) if f"{d:.2f}" in ("0.00", "0.10", "0.20", "0.30", "0.50", "1.00")},
    }
    rng = np.random.default_rng(0)
    a = base + rng.normal(scale=sigma, size=base.shape)
    b = fshift(base, 0, 0.3) + rng.normal(scale=sigma, size=base.shape)
    diff = b - a

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.7), gridspec_kw={"width_ratios": [1, 1, 1.25]}, layout="constrained")
    ax[0].imshow(a, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("参考图（合成版图）")
    S.image_axes(ax[0])
    im = ax[1].imshow(diff, cmap="RdBu_r", vmin=-0.12, vmax=0.12)
    ax[1].set_title("只错开 0.3 px 的差值图")
    S.image_axes(ax[1])
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=8)
    ax[2].plot(shifts, counts, "o-", color=S.RED, ms=4, lw=1.8)
    ax[2].axvline(0.3, color=S.MUTED, lw=1, ls="--")
    k = int(np.argmin(np.abs(shifts - 0.3)))
    ax[2].annotate(f"0.3 px：约 {counts[k]:.0f} 个像素", (0.3, counts[k]), xytext=(0.42, counts[k] * 0.45),
                   arrowprops={"arrowstyle": "->", "color": S.MUTED}, fontsize=10)
    ax[2].set_xlabel("两张图的错位量（px）")
    ax[2].set_ylabel("超过 5σ 阈值的像素数")
    ax[2].set_title("没有任何缺陷，误报随错位增长")
    S.clean(ax[2])
    S.save(fig, CH, "fig01_错位的代价")


# ---------------------------------------------------------------- 图 2：相似度度量
def fig_similarity():
    base = blur(layout(128), 1.0) * 200
    yy, xx = np.mgrid[0:128, 0:128]
    cases = {
        "增益 0.5": base * 0.5,
        "增益 0.4、偏置 +20": base * 0.4 + 20,
        "增益 1.8": base * 1.8,
        "横向渐变照明": base * (0.4 + 1.2 * xx / 127),
    }
    spots = [(40, 30), (60, 70), (20, 80), (84, 40)]

    def ssd_map(image, t):
        win = np.lib.stride_tricks.sliding_window_view(image, t.shape)
        return ((win - t) ** 2).sum(axis=(2, 3))

    def ncc_plain(image, t):
        win = np.lib.stride_tricks.sliding_window_view(image, t.shape)
        return (win * t).sum(axis=(2, 3)) / (np.sqrt((win ** 2).sum(axis=(2, 3))) * np.sqrt((t ** 2).sum()) + 1e-12)

    table = {}
    for name, target in cases.items():
        hit = {"SSD": 0, "NCC": 0, "ZNCC": 0}
        zpeaks, npeaks = [], []
        for ty, tx in spots:
            tpl = base[ty:ty + 24, tx:tx + 24]
            s = ssd_map(target, tpl)
            n = ncc_plain(target, tpl)
            z = mod.ncc_map(target, tpl)
            hit["SSD"] += tuple(np.unravel_index(np.argmin(s), s.shape)) == (ty, tx)
            hit["NCC"] += tuple(np.unravel_index(np.argmax(n), n.shape)) == (ty, tx)
            hit["ZNCC"] += tuple(np.unravel_index(np.argmax(z), z.shape)) == (ty, tx)
            zpeaks.append(float(z.max()))
            npeaks.append(float(n.max()))
        table[name] = {"找对位置的次数（共 4 个模板）": {k: int(v) for k, v in hit.items()},
                       "NCC 峰值最小": round(min(npeaks), 4), "ZNCC 峰值最小": round(min(zpeaks), 4)}
    NUM["相似度度量"] = {"说明": "128x128 合成版图，24x24 模板取自 4 个位置，目标图做亮度变化后全图搜索", "结果": table}

    ty, tx = 60, 70
    target = cases["增益 1.8"]
    tpl = base[ty:ty + 24, tx:tx + 24]
    s = ssd_map(target, tpl)
    z = mod.ncc_map(target, tpl)
    sy, sx = np.unravel_index(np.argmin(s), s.shape)
    zy, zx = np.unravel_index(np.argmax(z), z.shape)

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.8), layout="constrained")
    ax[0].imshow(target, cmap="gray", vmin=0, vmax=360)
    ax[0].add_patch(plt.Rectangle((tx - 0.5, ty - 0.5), 24, 24, fill=False, ec=S.GREEN, lw=2))
    ax[0].add_patch(plt.Rectangle((sx - 0.5, sy - 0.5), 24, 24, fill=False, ec=S.RED, lw=2, ls="--"))
    box = {"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.5}
    ax[0].text(tx, ty + 32, "真实位置", color=S.GREEN, fontsize=9, fontweight="bold", bbox=box)
    ax[0].text(sx, sy - 5, "SSD 找到的位置", color=S.RED, fontsize=9, fontweight="bold", bbox=box)
    ax[0].set_title("目标图：亮度变成 1.8 倍")
    S.image_axes(ax[0])
    im = ax[1].imshow(s, cmap="viridis_r")
    ax[1].plot(tx, ty, "o", mfc="none", mec=S.GREEN, ms=11, mew=2)
    ax[1].plot(sx, sy, "x", color=S.RED, ms=10, mew=2.5)
    ax[1].set_title("SSD 响应图（越亮越像）")
    S.image_axes(ax[1])
    im2 = ax[2].imshow(z, cmap="magma", vmin=-1, vmax=1)
    ax[2].plot(tx, ty, "o", mfc="none", mec=S.GREEN, ms=11, mew=2)
    ax[2].plot(zx, zy, "x", color="white", ms=10, mew=2.5)
    ax[2].set_title(f"ZNCC 响应图（峰值 {z.max():.3f}）")
    S.image_axes(ax[2])
    cb = fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=8)
    S.save(fig, CH, "fig02_相似度度量")


# ---------------------------------------------------------------- 图 3：相位相关流程
def fig_pipeline():
    ref = texture(seed=2)
    true = (7.0, -11.0)
    mov = fshift(ref, *true)
    fa, fb = np.fft.fft2(ref), np.fft.fft2(mov)
    cross = fb * np.conj(fa)
    r_phase = np.fft.fftshift(surface(ref, mov))
    r_cc = np.fft.fftshift(surface(ref, mov, mode="cc"))
    c = 48
    py, px = c + 7, c - 11

    fig, ax = plt.subplots(1, 5, figsize=(14.5, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.35]}, layout="constrained")
    ax[0].imshow(ref, cmap="gray")
    ax[0].set_title("① 参考图 f")
    ax[1].imshow(mov, cmap="gray")
    ax[1].set_title("② 平移后的图 g")
    ax[2].imshow(np.fft.fftshift(np.angle(cross)), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax[2].set_title("③ 互功率谱的相位")
    ax[3].imshow(r_phase, cmap="magma")
    ax[3].plot(px, py, "o", mfc="none", mec="white", ms=13, mew=1.5)
    ax[3].set_title("④ 逆变换后的相关面")
    for a in ax[:4]:
        S.image_axes(a)
    xs = np.arange(96) - c
    ax[4].plot(xs, r_cc[py] / r_cc.max(), color=S.ORANGE, lw=1.8, label="普通互相关")
    ax[4].plot(xs, r_phase[py] / r_phase.max(), color=S.BLUE, lw=1.8, label="相位相关")
    ax[4].axvline(-11, color=S.MUTED, lw=1, ls="--")
    ax[4].set_xlim(-30, 10)
    ax[4].set_xlabel("dx（px），沿峰值所在行")
    ax[4].set_title("⑤ 白化让峰变尖")
    ax[4].legend(loc="upper right", fontsize=9)
    S.clean(ax[4])
    S.save(fig, CH, "fig03_相位相关流程")

    half = lambda row: int((row / row.max() > 0.5).sum())  # noqa: E731
    NUM["峰的宽度"] = {"说明": "96x96 随机纹理，沿峰值所在行统计高于半峰值的采样点数",
                    "相位相关": half(r_phase[py]), "普通互相关": half(r_cc[py])}


# ---------------------------------------------------------------- 图 4：亚像素偏差与噪声
def fig_subpixel():
    ref = texture(seed=2)
    fy = np.fft.fftfreq(96)[:, None]
    fx = np.fft.fftfreq(96)[None, :]
    lowpass = np.exp(-(fy ** 2 + fx ** 2) / (2 * 0.08 ** 2))
    fracs = np.linspace(0, 1, 41)

    def estimate(a, b, kind):
        if kind == "抛物线":
            return peak(surface(a, b))
        if kind == "幅度掩膜":
            return peak(surface(a, b, weight=magnitude_mask(a)))
        if kind == "低通加权":
            return peak(surface(a, b, weight=lowpass))
        if kind == "普通互相关":
            return peak(surface(a, b, mode="cc"))
        d = peak(surface(a, b))  # 迭代：按估计值移回去，再估一次残差
        for _ in range(2):
            res = peak(surface(a, fshift(b, -d[0], -d[1])))
            d = (d[0] + res[0], d[1] + res[1])
        return d

    kinds = ["抛物线", "迭代两轮", "幅度掩膜"]
    curves = {k: [] for k in kinds}
    for f in fracs:
        t = (3 + f, -2 - f)
        m = fshift(ref, *t)
        for k in kinds:
            d = estimate(ref, m, k)
            curves[k].append(d[0] - t[0])
    NUM["亚像素偏差"] = {"说明": "96x96 随机纹理，无噪声，真值小数部分从 0 扫到 1，dy 方向误差",
                     "最大绝对误差": {k: round(float(np.max(np.abs(v))), 4) for k, v in curves.items()}}

    white = texture(seed=5, width=10.0)  # 高斯低通宽度远大于奈奎斯特频率，相当于全频带白噪声纹理
    white_err = []
    for f in fracs:
        t = (3 + f, -2 - f)
        white_err.append(estimate(white, fshift(white, *t), "抛物线")[0] - t[0])
    NUM["亚像素偏差"]["全频带纹理上抛物线的最大绝对误差"] = round(float(np.max(np.abs(white_err))), 4)

    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5]
    noise_kinds = ["抛物线", "幅度掩膜", "低通加权", "普通互相关"]
    rms = {k: [] for k in noise_kinds}
    for lv in noise_levels:
        errs = {k: [] for k in noise_kinds}
        rng = np.random.default_rng(10)
        for _ in range(200):
            t = rng.uniform(-6, 6, size=2)
            a = ref + rng.normal(scale=lv * ref.std(), size=ref.shape)
            b = fshift(ref, *t) + rng.normal(scale=lv * ref.std(), size=ref.shape)
            for k in noise_kinds:
                d = estimate(a, b, k)
                errs[k].append(np.hypot(d[0] - t[0], d[1] - t[1]))
        for k in noise_kinds:
            rms[k].append(float(np.sqrt(np.mean(np.square(errs[k])))))
    NUM["噪声下的误差"] = {"说明": "同一张纹理，噪声标准差按纹理标准差的比例给，200 次随机平移的均方根误差（px）",
                      "噪声水平": noise_levels, "RMS": {k: [round(v, 3) for v in vals] for k, vals in rms.items()}}

    t = 3.35
    r = np.fft.fftshift(surface(ref, fshift(ref, t, 0.0)))
    col = r[:, 48]
    pk = int(np.argmax(col))
    l, c, rr = col[pk - 1], col[pk], col[pk + 1]
    delta = parabola(l, c, rr)
    a2 = (l - 2 * c + rr) / 2
    b1 = (rr - l) / 2
    xs = np.linspace(-1.4, 1.4, 100)

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8), layout="constrained")
    idx = np.arange(pk - 3, pk + 4)
    ax[0].plot(idx - 48, col[idx], "o", color=S.MUTED, ms=5, label="相关面的采样点")
    ax[0].plot(np.array([pk - 1, pk, pk + 1]) - 48, [l, c, rr], "o", color=S.BLUE, ms=8, label="参与拟合的三点")
    ax[0].plot(xs + pk - 48, a2 * xs ** 2 + b1 * xs + c, color=S.BLUE, lw=1.6)
    ax[0].axvline(t, color=S.GREEN, lw=1.4, ls="--", label=f"真值 {t}")
    NUM["亚像素偏差"]["图 4 左栏的例子"] = {"真值": t, "抛物线顶点": round(float(pk - 48 + delta), 3)}
    ax[0].axvline(pk - 48 + delta, color=S.RED, lw=1.4, label=f"抛物线顶点 {pk - 48 + delta:.3f}")
    ax[0].set_xlabel("dy（px）")
    ax[0].set_title("三点抛物线怎样取到小数")
    ax[0].set_xlim(-0.6, 8.6)
    ax[0].legend(fontsize=8.5, loc="upper right")
    S.clean(ax[0])

    colors = {"抛物线": S.RED, "迭代两轮": S.ORANGE, "幅度掩膜": S.BLUE}
    for k in kinds:
        ax[1].plot(fracs, curves[k], color=colors[k], lw=1.8, label=k)
    ax[1].axhline(0, color=S.INK, lw=0.6)
    ax[1].set_xlabel("真值的小数部分")
    ax[1].set_ylabel("估计误差（px）")
    ax[1].set_title("系统偏差随小数部分周期变化")
    ax[1].legend(fontsize=9)
    S.clean(ax[1])

    colors2 = {"抛物线": S.RED, "幅度掩膜": S.BLUE, "低通加权": S.PURPLE, "普通互相关": S.ORANGE}
    for k in noise_kinds:
        ax[2].plot(noise_levels, rms[k], "o-", color=colors2[k], lw=1.8, ms=4, label=k)
    ax[2].set_xlabel("噪声标准差 / 纹理标准差")
    ax[2].set_ylabel("RMS 误差（px）")
    ax[2].set_title("噪声变大时谁先垮")
    ax[2].legend(fontsize=9)
    S.clean(ax[2])
    S.save(fig, CH, "fig04_亚像素偏差")


# ---------------------------------------------------------------- 图 5：加窗
def fig_window():
    big = texture((160, 160), seed=3)
    results = {}
    for label, cyclic in [("裁剪视场", False), ("循环平移", True)]:
        for use_window in (False, True):
            rng = np.random.default_rng(1)
            errs = []
            pulled = 0
            for _ in range(200):
                dy, dx = rng.integers(-6, 7, size=2)
                a = big[30:126, 30:126]
                b = np.roll(a, (dy, dx), axis=(0, 1)) if cyclic else big[30 - dy:126 - dy, 30 - dx:126 - dx]
                b = b + rng.normal(scale=0.05 * big.std(), size=(96, 96))
                est = mod.phase_correlation(a, b, use_window=use_window)
                errs.append(max(abs(est[0] - dy), abs(est[1] - dx)))
                if errs[-1] > 1 and min(abs(est[0]), abs(est[1])) < 0.6:
                    pulled += 1
            errs = np.array(errs)
            results[(label, use_window)] = errs
            if not cyclic and not use_window:
                zero_pulled = pulled
    NUM["加窗"] = {"说明": "96x96 视场，整数平移 ±6 px，噪声 5%，200 次；误差取 dy、dx 中较大者",
                 "结果": {f"{k[0]}，{'加窗' if k[1] else '不加窗'}": {"中位数": round(float(np.median(v)), 3), "最大": round(float(v.max()), 2),
                                                               "误差超过 1 px 的次数": int((v > 1).sum())} for k, v in results.items()},
                 "裁剪视场不加窗的失败样本中，至少一个方向的估计值落在 0 附近（绝对值小于 0.6 px）的次数": zero_pulled}

    crop = big[30:126, 30:126]
    win = np.outer(np.hanning(96), np.hanning(96))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.5]}, layout="constrained")
    ax[0].imshow(np.tile(crop, (2, 2)), cmap="gray")
    ax[0].axhline(95.5, color=S.RED, lw=1.2)
    ax[0].axvline(95.5, color=S.RED, lw=1.2)
    ax[0].set_title("FFT 眼里的图：首尾相接处有断口")
    ax[1].imshow(np.tile(crop * win, (2, 2)), cmap="gray")
    ax[1].axhline(95.5, color=S.RED, lw=1.2)
    ax[1].axvline(95.5, color=S.RED, lw=1.2)
    ax[1].set_title("乘 Hanning 窗后断口消失")
    for a in ax[:2]:
        S.image_axes(a)
    labels, data, colors = [], [], []
    for (label, use_window), v in results.items():
        labels.append(f"{label}\n{'加窗' if use_window else '不加窗'}")
        data.append(np.clip(v, 1e-3, None))
        colors.append(S.BLUE if use_window else S.ORANGE)
    bp = ax[2].boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": S.INK},
                       flierprops={"marker": ".", "markersize": 3, "markeredgecolor": S.MUTED})
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax[2].set_xticks(range(1, 5), labels, fontsize=9)
    ax[2].set_yscale("log")
    ax[2].set_yticks([0.001, 0.01, 0.1, 1, 10], ["0.001", "0.01", "0.1", "1", "10"])
    ax[2].minorticks_off()
    ax[2].axhline(1, color=S.RED, lw=1, ls="--")
    ax[2].text(4.45, 1.15, "1 px", color=S.RED, fontsize=9, ha="right")
    ax[2].set_ylabel("对齐误差（px，对数轴）")
    ax[2].set_title("要不要加窗，取决于边界是否连续")
    S.clean(ax[2])
    S.save(fig, CH, "fig05_加窗")


# ---------------------------------------------------------------- 图 6：周期图案
def fig_periodic():
    configs = [
        ("相位相关\n全范围", "phase", None, False, None),
        ("相位相关\n限 ±5 px", "phase", 5, False, None),
        ("相位相关 + 幅度掩膜\n限 ±5 px", "phase", 5, False, "mask"),
        ("普通互相关\n全范围", "cc", None, False, None),
        ("普通互相关\n限 ±5 px", "cc", 5, False, None),
        ("相位相关\n加对位标记", "phase", None, True, None),
    ]
    fails, table = [], {}
    for name, mode, radius, mark, weight in configs:
        p = blur(layout(144, period=24, mark=mark), 1.0)
        rng = np.random.default_rng(3)
        errs = []
        for _ in range(200):
            t = rng.uniform(-5, 5, size=2)
            a = p + rng.normal(scale=0.01, size=p.shape)
            b = fshift(p, *t) + rng.normal(scale=0.01, size=p.shape)
            w = magnitude_mask(a) if weight == "mask" else None
            d = peak(surface(a, b, mode=mode, weight=w), radius)
            errs.append(max(abs(d[0] - t[0]), abs(d[1] - t[1])))
        errs = np.array(errs)
        bad = int((errs > 1).sum())
        fails.append(bad)
        ok = errs[errs <= 1]
        table[name.replace("\n", "，")] = {"误差超过 1 px 的次数（共 200）": bad,
                                         "其余样本的平均误差": round(float(ok.mean()), 3) if len(ok) >= 20 else "样本太少不统计"}
    p = blur(layout(144, period=24), 1.0)
    clean_surface = surface(p, fshift(p, 2.0, 3.0), mode="cc")
    NUM["周期图案"] = {"说明": "144x144 合成版图，x 方向周期 24 px、y 方向周期 16 px，真实平移在 ±5 px 内随机，每张图噪声标准差 0.01",
                   "无噪声时等高峰个数": int((clean_surface > 0.999 * clean_surface.max()).sum()), "结果": table}

    marked = blur(layout(144, period=24, mark=True), 1.0)
    row = np.fft.fftshift(clean_surface)[72 + 2]
    xs = np.arange(144) - 72
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0), gridspec_kw={"width_ratios": [1, 1.35, 1.5]}, layout="constrained")
    ax[0].imshow(marked, cmap="gray", vmin=0, vmax=1)
    ax[0].add_patch(plt.Rectangle((10, 6), 31, 31, fill=False, ec=S.GREEN, lw=1.8))
    ax[0].text(44, 14, "对位标记", color=S.GREEN, fontsize=9.5, fontweight="bold",
               bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 1.5})
    ax[0].set_title("周期版图（x 周期 24 px）")
    S.image_axes(ax[0])
    ax[1].plot(xs, row / row.max(), color=S.ORANGE, lw=1.6)
    ax[1].axvspan(-5, 5, color=S.GREEN, alpha=0.15, label="机台误差范围 ±5 px")
    ax[1].axvline(3, color=S.GREEN, lw=1.3, ls="--", label="真实平移 dx = 3")
    ax[1].set_xlabel("dx（px）")
    ax[1].set_ylabel("归一化相关值")
    ax[1].set_title("没有标记时：每隔 24 px 一个等高峰")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[1])
    ypos = np.arange(len(configs))[::-1]
    colors = [S.RED if f > 20 else S.GREEN for f in fails]
    ax[2].barh(ypos, [max(f, 1.5) for f in fails], color=colors, alpha=0.85, height=0.62)
    for y, f in zip(ypos, fails):
        ax[2].text(max(f, 1.5) + 3, y, f"{f}/200", va="center", fontsize=9.5)
    ax[2].set_yticks(ypos, [c[0] for c in configs], fontsize=8.5)
    ax[2].set_xlim(0, 235)
    ax[2].set_xlabel("对齐失败次数（误差 > 1 px）")
    ax[2].set_title("哪种做法救得回来")
    S.clean(ax[2])
    ax[2].grid(axis="y", visible=False)
    S.save(fig, CH, "fig06_周期图案")


def main() -> None:
    S.setup()
    for fn in (fig_misalignment, fig_similarity, fig_pipeline, fig_subpixel, fig_window, fig_periodic):
        fn()
        print("完成", fn.__name__)
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
