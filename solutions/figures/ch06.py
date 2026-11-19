"""第 6 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch06.py

输出：<主分支>/06_光流/figures/*.svg 和 实测数字.json。
前向-反向筛选全部调用本章作业的参考实现 filter_tracks；LK、Farneback、ECC 用 OpenCV；
相位相关调用第 16 章作业的参考实现。讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。

耗时会随机器负载波动：默认沿用 实测数字.json 里上次记下的值，设置 CH06_RETIME=1 才重测。
"""

from __future__ import annotations

import functools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

CH = "06_光流"
filter_tracks = S.load_solution(CH, "6-1_前向反向筛选.py").filter_tracks
phase_correlation = S.load_solution("16_图像对齐与亚像素", "16-1_相位相关与亚像素.py").phase_correlation
NUM: dict = {}
cv2.setNumThreads(1)  # 单线程：结果和耗时都更容易复现

CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
FB_ARGS = dict(pyr_scale=0.5, levels=4, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def r(x, k=3):
    return round(float(x), k)


def med(v):
    return float(np.median(np.asarray(v, float)))


def pct(v, q):
    return float(np.percentile(np.asarray(v, float), q))


# ---------------------------------------------------------------- 数据与工具
@functools.lru_cache(maxsize=64)
def field(n, sigma, seed):
    """周期的带限随机纹理：白噪声在频域乘高斯（相当于标准差 sigma 像素的高斯模糊），标准差归一到 1。"""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, n))
    fy, fx = np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :]
    out = np.real(np.fft.ifft2(np.fft.fft2(w) * np.exp(-2 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))))
    return out / out.std()


def fshift(f, dx, dy):
    """频域平移，out(x, y) = f(x - dx, y - dy)。纹理是周期、带限的，所以小数位移也没有插值误差。"""
    n = f.shape[0]
    fy, fx = np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(f) * np.exp(-2j * np.pi * (fx * dx + fy * dy))))


def q8(v, noise=0.0, rng=None):
    """灰度值（浮点）加高斯噪声后量化成 8 位。"""
    if noise > 0:
        v = v + rng.normal(0, noise, v.shape)
    return np.clip(np.rint(v), 0, 255).astype(np.uint8)


def pair(size=256, d=(3.0, 2.0), seed=0, sigma=2.0, noise=0.0, mean=128.0, std=40.0, gain=1.0, offset=0.0, ramp=0.0):
    """一对已知平移的帧：第二帧 = 第一帧的内容整体平移 d，再可选地乘增益、加偏移、乘一个左右方向的亮度斜坡。"""
    big = 2 * size
    f = field(big, sigma, seed)
    g = fshift(f, d[0], d[1])
    o = (big - size) // 2
    rng = np.random.default_rng(seed + 10_000)
    A = mean + std * f[o:o + size, o:o + size]
    B = (mean + std * g[o:o + size, o:o + size]) * gain + offset
    if ramp:
        B = B * (1 + ramp * (2 * np.arange(size)[None, :] / (size - 1) - 1))
    return q8(A, noise, rng), q8(B, noise, rng)


@functools.lru_cache(maxsize=16)
def f1d(n, sigma, seed):
    rng = np.random.default_rng(seed)
    fr = np.fft.fftfreq(n)
    out = np.real(np.fft.ifft(np.fft.fft(rng.standard_normal(n)) * np.exp(-2 * (np.pi * sigma) ** 2 * fr ** 2)))
    return out / out.std()


def stripe_value(xx, yy, theta, seed, sigma=2.0, up=16, n=8192):
    """只有一个方向有梯度的图案：I(x, y) = f(x cos + y sin)，f 是一维带限随机函数（按 1/up 像素采样后线性插值）。"""
    f = f1d(n, sigma * up, seed)
    t = (xx * np.cos(theta) + yy * np.sin(theta)) * up + n / 2
    i0 = np.floor(t).astype(int)
    fr = t - i0
    return f[i0 % n] * (1 - fr) + f[(i0 + 1) % n] * fr


def stripes(size, theta, d, seed=0, noise=2.0):
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    rng = np.random.default_rng(seed + 77)
    a = q8(128 + 40 * stripe_value(xx, yy, theta, seed), noise, rng)
    b = q8(128 + 40 * stripe_value(xx - d[0], yy - d[1], theta, seed), noise, rng)
    return a, b


def grid(size, margin, step):
    xs = np.arange(margin, size - margin + 1e-9, step)
    return np.stack(np.meshgrid(xs, xs), -1).reshape(-1, 1, 2).astype(np.float32)


def lk(a, b, pts, win=21, lev=3, crit=CRIT, flags=0, mineig=1e-4):
    p1, st, err = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(win, win), maxLevel=lev, criteria=crit, flags=flags, minEigThreshold=mineig)
    return p1, st.ravel() != 0, err.ravel()


def farneback(a, b, **kw):
    return cv2.calcOpticalFlowFarneback(a, b, None, **{**FB_ARGS, **kw})


def epe(p0, p1, d):
    """端点误差：估计的位移向量和真值之差的长度（px）。"""
    return np.linalg.norm((p1 - p0).reshape(-1, 2) - np.asarray(d, float), axis=1)


def flow_epe(flow, d, margin=40):
    return np.linalg.norm(flow[margin:-margin, margin:-margin] - np.asarray(d, np.float32), axis=2).ravel()


def save(fig, name):
    """constrained 布局解出的坐标轴位置，不同次运行之间会差 1e-13 的量级；SVG 里 clip-path 的编号是按坐标算的哈希，会跟着变。
    这里先让布局算完并定住，把每个坐标轴的位置取整到 1e-4 再存，这样重复运行得到逐字节相同的文件。"""
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for ax in fig.axes:
        ax.set_position([round(v, 4) for v in ax.get_position().bounds])
    S.save(fig, CH, name)


# ---------------------------------------------------------------- 表：基线
def table_baseline():
    out = {}
    for d in [(0.3, 0.2), (1.0, 0.5), (3.0, 2.0), (6.5, -4.25)]:
        for noise in (0, 5):
            el, ef = [], []
            for s in range(5):
                a, b = pair(d=d, seed=s, noise=noise)
                p0 = grid(256, 40, 16)
                p1, _, _ = lk(a, b, p0)
                el.append(epe(p0, p1, d))
                ef.append(flow_epe(farneback(a, b), d))
            el, ef = np.concatenate(el), np.concatenate(ef)
            out.setdefault(f"位移 ({d[0]}, {d[1]})", {})[f"噪声 {noise}"] = {
                "LK：EPE 中位数": r(med(el), 4), "LK：EPE 95% 分位": r(pct(el, 95), 4),
                "Farneback：EPE 中位数": r(med(ef), 4), "Farneback：EPE 95% 分位": r(pct(ef, 95), 4)}
    NUM["基线"] = {
        "说明": "256x256 的带限随机纹理（高斯模糊标准差 2 px，平均灰度 128，标准差 40 个灰度级，8 位），第二帧是第一帧整体平移，5 个随机种子；"
              "LK：winSize=21、maxLevel=3、迭代上限 30、EPS 0.01，点是离边界 40 px 以内的 16 px 网格，每个种子 121 个点；"
              "Farneback：pyr_scale=0.5、levels=4、winsize=15、iterations=3、poly_n=5、poly_sigma=1.2，统计离边界 40 px 以内的全部像素；"
              "EPE 是估计位移和真实位移之差的长度（px）；噪声的单位是灰度级",
        "按位移": out}


# ---------------------------------------------------------------- 图 1：三种区域
def region_scene(seed, noise=2.0, theta=np.deg2rad(75)):
    """360x200，内容整体平移 (3, 2)：纹理背景里嵌一个平坦方块和一个条纹方块（各 80x80），方块跟着一起移动。"""
    W, H, d = 360, 200, (3, 2)
    f = field(512, 2.0, seed)
    g = fshift(f, *d)
    A = 128 + 40 * f[100:100 + H, 60:60 + W]
    B = 128 + 40 * g[100:100 + H, 60:60 + W]
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    sa = 128 + 40 * stripe_value(xx, yy, theta, seed + 300)
    sb = 128 + 40 * stripe_value(xx - d[0], yy - d[1], theta, seed + 300)
    A[60:140, 40:120] = 128
    B[62:142, 43:123] = 128
    A[60:140, 240:320] = sa[60:140, 240:320]
    B[62:142, 243:323] = sb[62:142, 243:323]
    rng = np.random.default_rng(seed + 5)
    return q8(A, noise, rng), q8(B, noise, rng), d


REGIONS = {"二维纹理": (slice(20, 180), slice(150, 210)), "条纹": (slice(80, 120), slice(260, 300)), "平坦": (slice(80, 120), slice(60, 100))}


def fig_regions():
    stat = {k: {"fb": [], "lk": [], "st": [], "mag": []} for k in REGIONS}
    for seed in range(6):
        a, b, d = region_scene(seed)
        fl = farneback(a, b)
        E = np.linalg.norm(fl - np.float32(d), axis=2)
        for name, (ys, xs) in REGIONS.items():
            stat[name]["fb"].append(E[ys, xs].ravel())
            stat[name]["mag"].append(np.linalg.norm(fl[ys, xs], axis=2).ravel())
            px, py = np.meshgrid(np.arange(xs.start, xs.stop, 8), np.arange(ys.start, ys.stop, 8))
            p0 = np.stack([px, py], -1).reshape(-1, 1, 2).astype(np.float32)
            p1, st, _ = lk(a, b, p0)
            stat[name]["lk"].append(epe(p0, p1, d))
            stat[name]["st"].append(st)
        if seed == 0:
            show_a, show_E = a, E
    res = {}
    for name, s in stat.items():
        fbv, lkv, st = np.concatenate(s["fb"]), np.concatenate(s["lk"]), np.concatenate(s["st"])
        res[name] = {"Farneback：EPE 中位数": r(med(fbv)), "Farneback：输出向量长度的中位数": r(med(np.concatenate(s["mag"])), 2),
                     "LK：点数": int(len(st)), "LK：status=1 的比例（%）": r(100 * st.mean(), 1),
                     "LK：status=1 的点的 EPE 中位数": r(med(lkv[st])) if st.any() else None,
                     "LK：status=1 的点里 EPE 超过 1 px 的比例（%）": r(100 * (lkv[st] > 1).mean(), 1) if st.any() else None}
    NUM["三种区域"] = {
        "说明": "360x200 的图，内容整体平移 (3, 2)，位移长度 3.606 px，噪声 2 个灰度级，6 个随机种子；纹理背景里嵌一个 80x80 的平坦方块和一个 80x80 的条纹方块"
              "（条纹只在和 x 轴成 75 度的方向上有梯度），方块和里面的图案跟着一起平移；每个区域只统计离区域边界至少 20 px 的中间部分；"
              "LK 在每个区域里取 8 px 网格点，winSize=21、maxLevel=3；Farneback 用基线参数",
        "位移长度": r(np.hypot(3, 2)), "按区域": res}

    fig, ax = plt.subplots(1, 3, figsize=(14.2, 3.5), gridspec_kw={"width_ratios": [1.45, 1.6, 1.15]}, layout="constrained")
    ax[0].imshow(show_a, cmap="gray", vmin=0, vmax=255)
    for name, (x0, y0), col in [("平坦", (40, 60), S.ORANGE), ("条纹", (240, 60), S.BLUE)]:
        ax[0].add_patch(plt.Rectangle((x0, y0), 80, 80, fill=False, ec=col, lw=1.6))
        ax[0].text(x0 + 40, y0 + 40, name, color="white", ha="center", va="center", fontsize=10.5, fontweight="bold", bbox=dict(fc=col, ec="none", pad=2))
    ax[0].text(180, 14, "二维纹理", color="white", ha="center", va="center", fontsize=10.5, fontweight="bold", bbox=dict(fc=S.INK, ec="none", pad=2))
    ax[0].set_title("第一帧：三种区域一起平移 (3, 2)", fontsize=10.5)
    S.image_axes(ax[0])
    im = ax[1].imshow(show_E, cmap="magma", vmin=0, vmax=4)
    ax[1].set_title("Farneback 每个像素的端点误差", fontsize=10.5)
    S.image_axes(ax[1])
    cb = fig.colorbar(im, ax=ax[1], fraction=0.035, pad=0.02)
    cb.set_label("EPE（px）")
    names = list(REGIONS)
    x = np.arange(len(names))
    fbv = [res[n]["Farneback：EPE 中位数"] for n in names]
    lkv = [res[n]["LK：status=1 的点的 EPE 中位数"] or 0 for n in names]
    ax[2].bar(x - 0.19, fbv, 0.36, color=S.ORANGE, label="Farneback")
    ax[2].bar(x + 0.19, lkv, 0.36, color=S.BLUE, label="LK（status=1 的点）")
    for xi, v in zip(x - 0.19, fbv):
        ax[2].text(xi, v + 0.06, f"{v:.2f}", ha="center", fontsize=9)
    for xi, v in zip(x + 0.19, lkv):
        ax[2].text(xi, v + 0.06, f"{v:.2f}", ha="center", fontsize=9)
    ax[2].axhline(np.hypot(3, 2), color=S.MUTED, ls="--", lw=1)
    ax[2].text(-0.45, np.hypot(3, 2) + 0.08, "位移的长度 3.61 px", color=S.MUTED, fontsize=9)
    ax[2].set_xticks(x, names)
    ax[2].set_ylabel("EPE 中位数（px）")
    ax[2].set_ylim(0, 4.6)
    ax[2].set_title("只有二维纹理上的结果可信", fontsize=10.5)
    ax[2].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    S.clean(ax[2])
    save(fig, "fig01_三种区域的端点误差")


# ---------------------------------------------------------------- 图 2：孔径问题
def fig_aperture():
    th = np.deg2rad(75)
    nvec, tvec = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    d = np.array([3.0, 2.0])
    by_noise = {}
    keep_scatter = None
    for noise in (0, 1, 2, 5):
        acc = {k: [] for k in ("st", "en", "et", "fb", "both")}
        for seed in range(5):
            a, b = stripes(256, th, d, seed=seed, noise=noise)
            p0 = grid(256, 50, 12)
            p1, st, _ = lk(a, b, p0)
            pb, stb, _ = lk(b, a, p1)
            fl = (p1 - p0).reshape(-1, 2)
            acc["st"].append(st)
            acc["en"].append(fl @ nvec - d @ nvec)
            acc["et"].append(fl @ tvec - d @ tvec)
            acc["fb"].append(np.linalg.norm((pb - p0).reshape(-1, 2), axis=1))
            acc["both"].append(filter_tracks(p0, p1, pb, (st & stb).astype(np.uint8), 1.0))
            if noise == 2 and seed == 0:
                keep_scatter = fl[st].copy()
                show = a
        st, en, et, keep = (np.concatenate(acc[k]) for k in ("st", "en", "et", "both"))
        e = np.hypot(en, et)
        row = {"点数": int(len(st)), "status=1 的比例（%）": r(100 * st.mean(), 1)}
        if st.any():
            row.update({"法向误差绝对值的中位数（px）": r(med(np.abs(en[st]))), "切向误差绝对值的中位数（px）": r(med(np.abs(et[st]))),
                        "切向误差绝对值的 90% 分位（px）": r(pct(np.abs(et[st]), 90)), "EPE 小于 1 px 的比例（%）": r(100 * (e[st] < 1).mean(), 1),
                        "往返误差不超过 1 px 的比例（%）": r(100 * keep.mean(), 1),
                        "往返误差不超过 1 px 的点里 EPE 小于 1 px 的比例（%）": r(100 * (e[keep] < 1).mean(), 1) if keep.any() else None})
        by_noise[f"噪声 {noise}"] = row
    # 对照：二维纹理
    a, b = pair(d=d, seed=0, noise=2)
    p0 = grid(256, 50, 12)
    p1, st, _ = lk(a, b, p0)
    tex = (p1 - p0).reshape(-1, 2)[st]
    tex_row = {"法向误差绝对值的中位数（px）": r(med(np.abs(tex @ nvec - d @ nvec))), "切向误差绝对值的中位数（px）": r(med(np.abs(tex @ tvec - d @ tvec)))}

    # 最小特征值：三种图各取同一批点
    rng = np.random.default_rng(0)
    flat_a, flat_b = q8(np.full((256, 256), 128.0), 2, rng), q8(np.full((256, 256), 128.0), 2, rng)
    kinds = {"二维纹理": pair(d=d, seed=0, noise=2), "条纹": stripes(256, th, d, seed=0, noise=2), "平坦": (flat_a, flat_b)}
    eig, thr = {}, {}
    for name, (a, b) in kinds.items():
        _, _, ev = lk(a, b, p0, flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS)
        eig[name] = ev
        thr[name] = {}
        for t in (1e-4, 1e-3, 1e-2, 3e-2):
            p1, st, _ = lk(a, b, p0, mineig=t)
            thr[name][f"{t:g}"] = r(100 * st.mean(), 1)
    flat_rows = {}
    for nz in (0, 0.5, 1, 2, 5):
        fa, fb_ = q8(np.full((256, 256), 128.0), nz, rng), q8(np.full((256, 256), 128.0), nz, rng)
        p1, st, _ = lk(fa, fb_, p0)
        mag = np.linalg.norm((p1 - p0).reshape(-1, 2), axis=1)
        flat_rows[f"噪声 {nz:g}"] = {"status=1 的比例（%）": r(100 * st.mean(), 1), "输出位移长度的中位数（px，真值是 0）": r(med(mag[st]), 2) if st.any() else None}
    NUM["孔径"] = {
        "说明": "256x256 的条纹图，只在和 x 轴成 75 度的方向（法向）上有梯度，内容平移 (3, 2)；法向分量、切向分量是位移在梯度方向和条纹方向上的投影；"
              "LK：winSize=21、maxLevel=3，点是离边界 50 px 以内的 12 px 网格，每个种子 196 个点，5 个随机种子；噪声的单位是灰度级；"
              "往返误差用参考实现 filter_tracks 按阈值 1 px 筛选（前向和反向的 status 都为 1）",
        "真值的法向分量": r(d @ nvec), "真值的切向分量": r(d @ tvec),
        "条纹图": by_noise, "对照：二维纹理（噪声 2）": tex_row,
        "最小特征值（flags=OPTFLOW_LK_GET_MIN_EIGENVALS 时 err 返回的量，噪声 2，各 196 个点）": {
            n: {"最小": float(f"{v.min():.2g}"), "中位数": float(f"{np.median(v):.2g}"), "最大": float(f"{v.max():.2g}")} for n, v in eig.items()},
        "不同 minEigThreshold 下 status=1 的比例（%，噪声 2）": thr,
        "纯平坦图（两帧都是灰度 128 加独立噪声）用默认 minEigThreshold=1e-4": flat_rows}

    fig, ax = plt.subplots(1, 3, figsize=(13.6, 3.9), gridspec_kw={"width_ratios": [1, 1.2, 1.25]}, layout="constrained")
    crop = show[88:168, 88:168]
    ax[0].imshow(crop, cmap="gray", vmin=0, vmax=255, extent=(-40, 40, 40, -40))
    k = 8
    ax[0].annotate("", xy=(d[0] * k, d[1] * k), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=S.GREEN, lw=2.2))
    nn = (d @ nvec) * nvec
    ax[0].annotate("", xy=(nn[0] * k, nn[1] * k), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=S.ORANGE, lw=2.2))
    ax[0].plot([nn[0] * k - 30 * tvec[0], nn[0] * k + 30 * tvec[0]], [nn[1] * k - 30 * tvec[1], nn[1] * k + 30 * tvec[1]], "--", color="white", lw=1.2)
    ax[0].set_xlim(-40, 40)
    ax[0].set_ylim(40, -40)
    ax[0].text(-38, -37, "绿：真实位移\n橙：法向分量\n白虚线：约束直线", color="white", fontsize=9, va="top", bbox=dict(fc=S.INK, ec="none", alpha=0.75, pad=3))
    ax[0].set_title("条纹图局部（箭头放大 8 倍）", fontsize=10.5)
    S.image_axes(ax[0])
    ax[1].scatter(keep_scatter[:, 0], keep_scatter[:, 1], s=12, color=S.ORANGE, alpha=0.7, label="条纹图上的估计")
    t = np.array([-30, 30])
    ax[1].plot(nn[0] + t * tvec[0], nn[1] + t * tvec[1], "--", color=S.MUTED, lw=1, label="约束直线 $I_x u + I_y v + I_t = 0$", zorder=1)
    ax[1].plot(*d, "+", color=S.GREEN, ms=20, mew=2.2, label="真值 (3, 2)", zorder=3)
    ax[1].scatter(tex[:, 0], tex[:, 1], s=14, color=S.BLUE, alpha=0.8, label="二维纹理上的估计", zorder=4)
    ax[1].set_xlim(d[0] - 13, d[0] + 13)
    ax[1].set_ylim(d[1] + 7.5, d[1] - 7.5)
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("u（px）")
    ax[1].set_ylabel("v（px，向下为正）")
    ax[1].set_title("条纹上的估计沿约束直线散开", fontsize=10.5)
    ax[1].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[1])
    rs = np.random.default_rng(1)
    for i, (name, col) in enumerate([("二维纹理", S.BLUE), ("条纹", S.ORANGE), ("平坦", S.RED)]):
        v = eig[name]
        ax[2].scatter(i + rs.uniform(-0.22, 0.22, len(v)), v, s=9, color=col, alpha=0.55)
    ax[2].axhline(1e-4, color=S.MUTED, ls="--", lw=1.2)
    ax[2].text(2.45, 1.15e-4, "默认阈值 1e-4", color=S.MUTED, fontsize=9, ha="right", va="bottom")
    ax[2].axhline(1e-2, color=S.GREEN, ls=":", lw=1.4)
    ax[2].text(2.45, 1.15e-2, "阈值 1e-2", color=S.GREEN, fontsize=9, ha="right", va="bottom")
    ax[2].set_yscale("log")
    ax[2].set_ylim(2e-5, 1)
    ax[2].set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ["1e-4", "1e-3", "1e-2", "0.1", "1"])
    ax[2].minorticks_off()
    ax[2].set_xticks([0, 1, 2], ["二维纹理", "条纹", "平坦"])
    ax[2].set_xlim(-0.5, 2.5)
    ax[2].set_ylabel("窗口的最小特征值（OpenCV 归一化后）")
    ax[2].set_title("默认阈值分不开三种区域", fontsize=10.5)
    S.clean(ax[2])
    save(fig, "fig02_孔径问题")


# ---------------------------------------------------------------- 图 3：窗口、运动边界、迭代次数
def boundary_scene(seed, noise=2.0):
    """256x256：静止的纹理背景上有一个 120x120 的前景方块（另一张纹理），方块右移 4 px。"""
    rng = np.random.default_rng(seed)
    bg, ob = field(512, 2.0, seed)[:256, :256], field(512, 2.0, seed + 100)[:120, :120]
    A, B = 128 + 40 * bg, 128 + 40 * bg
    x0, y0 = 60, 68
    A[y0:y0 + 120, x0:x0 + 120] = 128 + 40 * ob
    B[y0:y0 + 120, x0 + 4:x0 + 124] = 128 + 40 * ob
    return q8(A, noise, rng), q8(B, noise, rng), x0, y0


def fig_window():
    wins, noises = [5, 9, 15, 21, 31, 41], [0, 5, 10, 20, 40]
    tab = {}
    for noise in noises:
        for win in wins:
            e = []
            for s in range(5):
                a, b = pair(d=(3, 2), seed=s, noise=noise)
                p0 = grid(256, 40, 16)
                p1, _, _ = lk(a, b, p0, win=win)
                e.append(epe(p0, p1, (3, 2)))
            e = np.concatenate(e)
            tab.setdefault(f"噪声 {noise}", {})[f"窗口 {win}"] = {"EPE 中位数": r(med(e)), "EPE 超过 1 px 的比例（%）": r(100 * (e > 1).mean(), 1)}
    NUM["窗口与噪声"] = {"说明": "256x256 纹理，平移 (3, 2)，LK 的 maxLevel=3，每个种子 121 个网格点，5 个随机种子；噪声的单位是灰度级，纹理自身的标准差是 40 个灰度级", "表": tab}

    dists = [2, 4, 6, 8, 10, 15, 20, 30, 60]
    bwins = [9, 15, 21, 31, 41]
    btab = {w: {dd: [] for dd in dists} for w in bwins}
    ftab = {w: {dd: [] for dd in dists} for w in (5, 15, 31)}
    for seed in range(6):
        a, b, x0, y0 = boundary_scene(seed)
        pts = np.array([[x0 + dd, y] for dd in dists for y in range(y0 + 40, y0 + 80, 4)], np.float32).reshape(-1, 1, 2)
        for w in bwins:
            p1, _, _ = lk(a, b, pts, win=w)
            e = epe(pts, p1, (4, 0))
            for i, dd in enumerate(dists):
                btab[w][dd].append(e[i * 10:(i + 1) * 10])
        for w in ftab:
            fl = farneback(a, b, winsize=w)
            for dd in dists:
                ftab[w][dd].append(np.linalg.norm(fl[y0 + 40:y0 + 80, x0 + dd] - np.float32([4, 0]), axis=1))
    NUM["运动边界"] = {
        "说明": "256x256，静止的纹理背景上有一个 120x120 的前景方块，方块右移 4 px，噪声 2 个灰度级，6 个随机种子；"
              "跟踪方块内部的点，按点到方块左边界的距离分组，每组 60 个点，报告 EPE 中位数（px）；LK 的 maxLevel=3；Farneback 除 winsize 外用基线参数",
        "LK：按窗口和到边界的距离（px）": {f"窗口 {w}": {str(dd): r(med(np.concatenate(v)), 2) for dd, v in row.items()} for w, row in btab.items()},
        "Farneback：按 winsize 和到边界的距离（px）": {f"winsize {w}": {str(dd): r(med(np.concatenate(v)), 2) for dd, v in row.items()} for w, row in ftab.items()}}

    counts = [1, 2, 3, 5, 10, 30]
    disp = [(0.5, 0.25), (1.5, 1.0), (3.3, -1.7)]
    itab = {}
    p0 = grid(256, 40, 16)
    for d in disp:
        a, b = pair(d=d, seed=0)
        for c in counts:
            p1, _, _ = lk(a, b, p0, lev=0, crit=(cv2.TERM_CRITERIA_COUNT, c, 0))
            itab.setdefault(f"位移 ({d[0]}, {d[1]})，长度 {np.hypot(*d):.2f} px", {})[f"迭代 {c} 次"] = r(med(epe(p0, p1, d)), 4)
    NUM["迭代次数"] = {"说明": "256x256 纹理，无噪声，LK 不用金字塔（maxLevel=0）、winSize=21，终止条件只设迭代次数上限，121 个网格点，EPE 中位数（px）", "表": itab}

    fig, ax = plt.subplots(1, 3, figsize=(13.8, 3.7), layout="constrained")
    cols = [S.MUTED, S.BLUE, S.GREEN, S.ORANGE, S.RED]
    for noise, col in zip(noises[1:], cols[1:]):
        ax[0].plot(wins, [tab[f"噪声 {noise}"][f"窗口 {w}"]["EPE 中位数"] for w in wins], "o-", color=col, lw=1.6, ms=4, label=f"噪声 {noise}")
    ax[0].set_yscale("log")
    ax[0].set_yticks([0.01, 0.03, 0.1, 0.3, 1], ["0.01", "0.03", "0.1", "0.3", "1"])
    ax[0].minorticks_off()
    ax[0].set_xticks(wins)
    ax[0].set_xlabel("LK 窗口边长（px）")
    ax[0].set_ylabel("EPE 中位数（px）")
    ax[0].set_title("窗口越大，噪声造成的误差越小")
    ax[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    S.clean(ax[0])
    for w, col in zip([9, 21, 41], [S.BLUE, S.ORANGE, S.RED]):
        ax[1].plot(dists, [med(np.concatenate(btab[w][dd])) for dd in dists], "o-", color=col, lw=1.6, ms=4, label=f"窗口 {w}")
        ax[1].axvline(w // 2, color=col, ls=":", lw=1)
    ax[1].set_xscale("log")
    ax[1].set_xticks(dists, [str(v) for v in dists])
    ax[1].minorticks_off()
    ax[1].set_xlabel("点到运动边界的距离（px），竖虚线是窗口的一半")
    ax[1].set_ylabel("EPE 中位数（px）")
    ax[1].set_title("离边界不到半个窗口的点被带偏")
    ax[1].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    S.clean(ax[1])
    for (name, row), col in zip(itab.items(), [S.BLUE, S.ORANGE, S.RED]):
        ax[2].plot(range(len(counts)), [max(row[f"迭代 {c} 次"], 1e-3) for c in counts], "o-", color=col, lw=1.6, ms=4, label=name.split("，")[1].replace("长度", "位移"))
    ax[2].set_yscale("log")
    ax[2].set_yticks([0.01, 0.1, 1], ["0.01", "0.1", "1"])
    ax[2].minorticks_off()
    ax[2].set_xticks(range(len(counts)), [str(c) for c in counts])
    ax[2].set_xlabel("迭代次数上限")
    ax[2].set_ylabel("EPE 中位数（px）")
    ax[2].set_title("位移越大，需要的迭代越多")
    ax[2].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    S.clean(ax[2])
    save(fig, "fig03_窗口与迭代")


# ---------------------------------------------------------------- 图 4：前向-反向误差
def fb_scene(seed, d, noise=4.0, size=384):
    """384x384 的纹理整体平移 d；第二帧里有一块 120x120 的区域被另一张纹理盖住（遮挡），落进去或移出画面的点没有正确答案。"""
    f = field(2 * size, 2.0, seed)
    g = fshift(f, *d)
    o = size // 2
    A = 128 + 40 * f[o:o + size, o:o + size]
    B = 128 + 40 * g[o:o + size, o:o + size]
    B[200:320, 220:340] = 128 + 40 * field(256, 2.0, seed + 500)[:120, :120]
    occ = np.zeros((size, size), bool)
    occ[200:320, 220:340] = True
    rng = np.random.default_rng(seed + 99)
    return q8(A, noise, rng), q8(B, noise, rng), occ


def fb_run(d, lev, seeds=10):
    rows = []
    for seed in range(seeds):
        a, b, occ = fb_scene(seed, d)
        size = a.shape[0]
        p0 = cv2.goodFeaturesToTrack(a, maxCorners=400, qualityLevel=0.01, minDistance=7, blockSize=7)
        p1, st, err = lk(a, b, p0, lev=lev)
        pb, stb, _ = lk(b, a, p1, lev=lev)
        q0, q1, qb = p0.reshape(-1, 2), p1.reshape(-1, 2), pb.reshape(-1, 2)
        true = q0 + np.asarray(d, float)
        inb = (true[:, 0] >= 0) & (true[:, 0] <= size - 1) & (true[:, 1] >= 0) & (true[:, 1] <= size - 1)
        ti = np.clip(np.rint(true).astype(int), 0, size - 1)
        visible = inb & ~occ[ti[:, 1], ti[:, 0]]
        e = np.linalg.norm(q1 - true, axis=1)
        rows.append(dict(p0=p0, p1=p1, pb=pb, st=st, both=st & stb, visible=visible, e=e, good=visible & (e < 1), fbe=np.linalg.norm(q0 - qb, axis=1), err=err))
    return rows


def fb_stats(rows, thresholds):
    cat = lambda k: np.concatenate([x[k] for x in rows])  # noqa: E731
    st, good, e, vis, err = cat("st"), cat("good"), cat("e"), cat("visible"), cat("err")
    n = len(st)
    out = {"点数": int(n), "真值可见（没被遮挡、没出界）的比例（%）": r(100 * vis.mean(), 1),
           "不筛选：正确率（%）": r(100 * good.mean(), 1),
           "只看前向 status": {"保留点数": int(st.sum()), "保留比例（%）": r(100 * st.mean(), 1), "正确率（%）": r(100 * good[st].mean(), 1), "留下的错点数": int((~good[st]).sum())},
           "按往返误差阈值（px）": {}}
    for th in thresholds:
        keep = np.concatenate([filter_tracks(x["p0"], x["p1"], x["pb"], x["both"].astype(np.uint8), th) for x in rows])
        out["按往返误差阈值（px）"][f"{th:g}"] = {
            "保留点数": int(keep.sum()), "保留比例（%）": r(100 * keep.mean(), 1), "正确率（%）": r(100 * good[keep].mean(), 2) if keep.any() else None,
            "留下的错点数": int((~good[keep]).sum()), "好点被保留的比例（%）": r(100 * (good & keep).sum() / max(good.sum(), 1), 1),
            "保留点的 EPE 中位数（px）": r(med(e[keep])) if keep.any() else None, "保留点的 EPE 95% 分位（px）": r(pct(e[keep], 95)) if keep.any() else None}
    keep = st & (err <= 12)
    out["status=1 且 err 不超过 12 个灰度级（噪声标准差的 3 倍）"] = {
        "保留点数": int(keep.sum()), "正确率（%）": r(100 * good[keep].mean(), 2) if keep.any() else None, "好点被保留的比例（%）": r(100 * (good & keep).sum() / max(good.sum(), 1), 1)}
    return out


def fig_fb():
    ths = [0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10]
    rows_a = fb_run((6.0, 4.0), 3)
    rows_b = fb_run((12.0, 8.0), 1)
    sa, sb = fb_stats(rows_a, ths), fb_stats(rows_b, ths)
    cat = lambda k: np.concatenate([x[k] for x in rows_a])  # noqa: E731
    both, good, vis, fbe, e = cat("both"), cat("good"), cat("visible"), cat("fbe"), cat("e")
    small_bad = both & ~good & (fbe <= 1)
    sa["往返误差不超过 1 px 却是错点的那些"] = {"个数": int(small_bad.sum()), "其中真值不可见（被遮挡或出界）": int((small_bad & ~vis).sum()), "它们的 EPE 中位数（px）": r(med(e[small_bad]), 2)}
    catb = lambda k: np.concatenate([x[k] for x in rows_b])  # noqa: E731
    bb, gb, fbb = catb("both"), catb("good"), catb("fbe")
    moved = np.linalg.norm(np.concatenate([(x["p1"] - x["p0"]).reshape(-1, 2) for x in rows_b]), axis=1)
    sel = bb & ~gb & (fbb <= 1)
    sb["往返误差不超过 1 px 却是错点的那些"] = {"个数": int(sel.sum()), "它们前向位移长度的中位数（px，真值是 14.42）": r(med(moved[sel]), 2)}
    NUM["前向反向"] = {
        "说明": "384x384 的纹理整体平移，噪声 4 个灰度级；第二帧里有一块 120x120 的区域被另一张纹理盖住，落进去或移出画面的点没有正确答案；"
              "每帧用 goodFeaturesToTrack 取 400 个角点（qualityLevel=0.01、minDistance=7、blockSize=7），10 个随机种子共 4000 个点；"
              "LK 的 winSize=21；前向跟踪后从结果位置反向跟踪回来；筛选调用参考实现 filter_tracks，传入的 status 是前向和反向 status 的与；"
              "正确 = 真值可见并且 EPE 小于 1 px",
        "常规场景：平移 (6, 4)，maxLevel=3": sa,
        "位移超出范围：平移 (12, 8)（长度 14.42 px），maxLevel=1": sb}

    stripe = NUM["孔径"]["条纹图"]["噪声 2"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1.15, 1.1, 1.1]}, layout="constrained")
    T = sa["按往返误差阈值（px）"]
    ax[0].plot(ths, [T[f"{t:g}"]["正确率（%）"] for t in ths], "o-", color=S.BLUE, lw=1.8, ms=4.5, label="保留点的正确率")
    ax[0].plot(ths, [T[f"{t:g}"]["好点被保留的比例（%）"] for t in ths], "s-", color=S.ORANGE, lw=1.8, ms=4.5, label="好点被保留的比例")
    ax[0].axhline(sa["只看前向 status"]["正确率（%）"], color=S.MUTED, ls="--", lw=1.1)
    ax[0].text(10, sa["只看前向 status"]["正确率（%）"] - 0.4, f"只看 status 的正确率 {sa['只看前向 status']['正确率（%）']}%", color=S.MUTED, fontsize=9, va="top", ha="right")
    ax[0].set_xscale("log")
    ax[0].set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10], ["0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10"])
    ax[0].minorticks_off()
    ax[0].set_ylim(80, 100.8)
    ax[0].set_xlabel("往返误差阈值（px）")
    ax[0].set_ylabel("比例（%）")
    ax[0].set_title("阈值越小越准，留下的好点越少")
    ax[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[0])
    lo = 3e-3
    for mask, col, lab in [(both & good, S.BLUE, "正确"), (both & ~vis, S.RED, "被遮挡或出界"), (both & vis & ~good, S.ORANGE, "可见但跟错")]:
        ax[1].scatter(np.maximum(fbe[mask], lo), np.maximum(e[mask], lo), s=6, color=col, alpha=0.45, label=lab, linewidths=0)
    ax[1].axvline(1, color=S.MUTED, ls="--", lw=1.1)
    ax[1].axhline(1, color=S.MUTED, ls=":", lw=1.1)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
    ax[1].set_yticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
    ax[1].set_xlim(lo * 0.8, 150)
    ax[1].set_ylim(lo * 0.8, 150)
    ax[1].minorticks_off()
    ax[1].set_xlabel("往返误差（px），竖虚线是阈值 1")
    ax[1].set_ylabel("真实的端点误差（px）")
    ax[1].set_title("阈值 1 px 的左上角还有红点")
    ax[1].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, markerscale=2)
    S.clean(ax[1])
    names = ["常规场景", "位移超出范围", "条纹（孔径）"]
    v_st = [sa["只看前向 status"]["正确率（%）"], sb["只看前向 status"]["正确率（%）"], stripe["EPE 小于 1 px 的比例（%）"]]
    v_fb = [sa["按往返误差阈值（px）"]["1"]["正确率（%）"], sb["按往返误差阈值（px）"]["1"]["正确率（%）"], stripe["往返误差不超过 1 px 的点里 EPE 小于 1 px 的比例（%）"]]
    x = np.arange(3)
    ax[2].bar(x - 0.19, v_st, 0.36, color=S.MUTED, label="只看 status")
    ax[2].bar(x + 0.19, v_fb, 0.36, color=S.BLUE, label="再加往返误差不超过 1 px")
    for xi, v in list(zip(x - 0.19, v_st)) + list(zip(x + 0.19, v_fb)):
        ax[2].text(xi, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    ax[2].set_xticks(x, names)
    ax[2].set_ylim(0, 112)
    ax[2].set_ylabel("保留点的正确率（%）")
    ax[2].set_title("后两种情况往返检查救不回来")
    ax[2].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    S.clean(ax[2])
    save(fig, "fig04_前向反向误差")


# ---------------------------------------------------------------- 图 5：金字塔
def small_object(seed, s, dx, noise=2.0, size=384):
    """静止纹理背景上一个边长 s 的方块（另一张纹理）右移 dx。"""
    rng = np.random.default_rng(seed)
    bg, ob = field(2 * size, 2.0, seed)[:size, :size], field(512, 2.0, seed + 100)[:s, :s]
    A, B = 128 + 40 * bg, 128 + 40 * bg
    x0 = y0 = (size - s) // 2 - 16
    A[y0:y0 + s, x0:x0 + s] = 128 + 40 * ob
    B[y0:y0 + s, x0 + dx:x0 + dx + s] = 128 + 40 * ob
    c = np.linspace(s * 0.3, s * 0.7, 5)
    pts = np.array([[x0 + u, y0 + v] for u in c for v in c], np.float32).reshape(-1, 1, 2)
    return q8(A, noise, rng), q8(B, noise, rng), pts


def fig_pyramid():
    mags = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    levs = [0, 1, 2, 3, 4, 5]
    size = 384
    p0 = grid(size, 90, 20)
    res = {}
    for sigma, wins in [(2.0, (9, 21, 41)), (1.0, (21,)), (4.0, (21,))]:
        for m in mags:
            pairs = []
            for s in range(3):
                th = 0.6 + s
                d = (m * np.cos(th), m * np.sin(th))
                pairs.append((d, pair(size=size, d=d, seed=s, sigma=sigma)))
            for win in wins:
                for lev in levs:
                    ok, st_all = [], []
                    for d, (a, b) in pairs:
                        p1, st, _ = lk(a, b, p0, win=win, lev=lev)
                        ok.append(epe(p0, p1, d) < 1)
                        st_all.append(st)
                    res[(sigma, win, lev, m)] = (100 * np.concatenate(ok).mean(), 100 * np.concatenate(st_all).mean())
    a0 = pair(size=size, d=(1, 0), seed=0)[0]
    built = {f"窗口 {w}": {f"maxLevel 设 {lv}": int(cv2.buildOpticalFlowPyramid(a0, (w, w), lv)[0]) for lv in (3, 4, 5)} for w in (9, 21, 41)}

    def table(sigma, win):
        return {f"maxLevel {lv}": {str(m): r(res[(sigma, win, lv, m)][0], 0) for m in mags} for lv in levs}

    NUM["金字塔"] = {
        "说明": "384x384 的纹理整体平移，无噪声；位移长度取 1 到 64 px，每个长度 3 个方向（3 个随机种子），点是离边界 90 px 以内的 20 px 网格，每格 363 个点；"
              "数值是 EPE 小于 1 px 的点所占百分比；纹理粗细用生成纹理时高斯模糊的标准差表示，默认 2 px",
        "窗口 21，纹理 2 px": table(2.0, 21), "窗口 9，纹理 2 px": table(2.0, 9), "窗口 41，纹理 2 px": table(2.0, 41),
        "窗口 21，纹理 1 px": table(1.0, 21), "窗口 21，纹理 4 px": table(4.0, 21),
        "status=1 的比例（%）：窗口 21、纹理 2 px、maxLevel 0": {str(m): r(res[(2.0, 21, 0, m)][1], 1) for m in mags},
        "所有组合里 status=1 比例的最小值（%）": r(min(v[1] for v in res.values()), 1),
        "buildOpticalFlowPyramid 在 384x384 的图上实际建到第几层": built}

    sizes = [16, 24, 32, 48, 64, 96, 128]
    obj = {}
    for dx in (4, 16):
        for s in sizes:
            ok = {lv: [] for lv in (0, 2, 4)}
            for seed in range(6):
                a, b, pts = small_object(seed, s, dx)
                for lv in ok:
                    p1, _, _ = lk(a, b, pts, lev=lv)
                    ok[lv].append(epe(pts, p1, (dx, 0)) < 1)
            obj.setdefault(f"右移 {dx} px", {})[f"边长 {s}"] = {f"maxLevel {lv}": r(100 * np.concatenate(v).mean(), 0) for lv, v in ok.items()}
    NUM["小物体"] = {
        "说明": "384x384，静止的纹理背景上一个方块（另一张纹理）向右平移，噪声 2 个灰度级，6 个随机种子；跟踪方块中间的 5x5 个点（共 150 个），LK 的 winSize=21；"
              "数值是 EPE 小于 1 px 的点所占百分比",
        "表": obj}

    fig, ax = plt.subplots(1, 3, figsize=(14.4, 3.9), gridspec_kw={"width_ratios": [1.45, 1.1, 1.0]}, layout="constrained")
    M = np.array([[res[(2.0, 21, lv, m)][0] for m in mags] for lv in levs])
    ax[0].imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    for i in range(len(levs)):
        for j in range(len(mags)):
            ax[0].text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=9, color="white" if (M[i, j] < 15 or M[i, j] > 85) else S.INK)
    ax[0].set_xticks(range(len(mags)), [str(m) for m in mags])
    ax[0].set_yticks(range(len(levs)), [str(v) for v in levs])
    ax[0].set_xlabel("位移长度（px）")
    ax[0].set_ylabel("maxLevel")
    ax[0].set_title("窗口 21：层数越多，能跟上的位移越大", fontsize=10.5)
    for (sigma, win), col, ls, lab in [((2.0, 9), S.BLUE, "-", "窗口 9"), ((2.0, 21), S.ORANGE, "-", "窗口 21"), ((2.0, 41), S.RED, "-", "窗口 41"),
                                       ((1.0, 21), S.MUTED, "--", "窗口 21，纹理更细（1 px）"), ((4.0, 21), S.GREEN, "--", "窗口 21，纹理更粗（4 px）")]:
        ax[1].plot(range(8), [res[(sigma, win, 0, m)][0] for m in mags[:8]], "o" + ls, color=col, lw=1.6, ms=4, label=lab)
    ax[1].set_xticks(range(8), [str(m) for m in mags[:8]])
    ax[1].set_xlabel("位移长度（px），不用金字塔")
    ax[1].set_ylabel("EPE 小于 1 px 的点（%）")
    ax[1].set_title("单层能跟多远，纹理粗细比窗口更要紧", fontsize=10.5)
    ax[1].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[1])
    for lv, col in [(0, S.MUTED), (2, S.BLUE), (4, S.RED)]:
        ax[2].plot(range(len(sizes)), [obj["右移 16 px"][f"边长 {s}"][f"maxLevel {lv}"] for s in sizes], "o-", color=col, lw=1.6, ms=4, label=f"右移 16 px，maxLevel {lv}")
    ax[2].plot(range(len(sizes)), [obj["右移 4 px"][f"边长 {s}"]["maxLevel 4"] for s in sizes], "s:", color=S.GREEN, lw=1.4, ms=4, label="右移 4 px，maxLevel 4")
    ax[2].set_xticks(range(len(sizes)), [str(s) for s in sizes])
    ax[2].set_xlabel("方块边长（px）")
    ax[2].set_ylabel("EPE 小于 1 px 的点（%）")
    ax[2].set_title("物体小、位移大，加层数也没用", fontsize=10.5)
    ax[2].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    save(fig, "fig05_金字塔与位移")


# ---------------------------------------------------------------- 表：整幅图对齐（和第 16 章的关系）
def lk_global(a, b, levels=1, iters=50, margin=0.2, eps=1e-4):
    """把整幅图当成一个窗口的 LK，只估平移：高斯-牛顿迭代，可选金字塔。返回 (dx, dy)。"""
    pa, pb = [a.astype(np.float32)], [b.astype(np.float32)]
    for _ in range(levels - 1):
        pa.append(cv2.pyrDown(pa[-1]))
        pb.append(cv2.pyrDown(pb[-1]))
    p = np.zeros(2)
    for i, (A, B) in enumerate(zip(pa[::-1], pb[::-1])):
        if i:
            p = p * 2
        h, w = A.shape
        m = int(round(margin * min(h, w)))
        sl = (slice(m, h - m), slice(m, w - m))
        gx = cv2.Sobel(A, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)[sl].astype(float)
        gy = cv2.Sobel(A, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)[sl].astype(float)
        G = np.array([[(gx * gx).sum(), (gx * gy).sum()], [(gx * gy).sum(), (gy * gy).sum()]])
        for _ in range(iters):
            Mw = np.float32([[1, 0, p[0]], [0, 1, p[1]]])
            Bw = cv2.warpAffine(B, Mw, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
            res = (Bw - A)[sl].astype(float)
            dp = -np.linalg.solve(G, np.array([(gx * res).sum(), (gy * res).sum()]))
            p = p + dp
            if np.hypot(*dp) < eps:
                break
    return p


def ecc_shift(a, b):
    try:
        W = np.eye(2, 3, dtype=np.float32)
        _, W = cv2.findTransformECC(a, b, W, cv2.MOTION_TRANSLATION, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-7), None, 5)
        return W[:, 2].astype(float)
    except cv2.error:
        return np.array([np.inf, np.inf])


def table_global():
    out = {}
    for mag in [0.3, 1, 2, 4, 8, 12, 16, 24, 32, 48]:
        errs = {k: [] for k in ("整图 LK，不用金字塔", "整图 LK，4 层金字塔", "相位相关（第 16 章参考实现，加窗）", "ECC（findTransformECC，只估平移）")}
        for seed in range(6):
            th = 0.5 + seed * 1.1
            d = np.array([mag * np.cos(th), mag * np.sin(th)])
            a, b = pair(d=d, seed=seed, noise=2)
            errs["整图 LK，不用金字塔"].append(np.hypot(*(lk_global(a, b, 1) - d)))
            errs["整图 LK，4 层金字塔"].append(np.hypot(*(lk_global(a, b, 4) - d)))
            dy, dx = phase_correlation(a, b, True)
            errs["相位相关（第 16 章参考实现，加窗）"].append(np.hypot(dx - d[0], dy - d[1]))
            errs["ECC（findTransformECC，只估平移）"].append(np.hypot(*(ecc_shift(a, b) - d)))
        out[f"位移 {mag:g} px"] = {k: {"误差中位数（px）": (r(med(v), 4) if np.isfinite(np.median(v)) and np.median(v) < 1e3 else None),
                                    "6 次里误差小于 0.5 px 的次数": int((np.asarray(v) < 0.5).sum())} for k, v in errs.items()}
    NUM["整图对齐"] = {
        "说明": "256x256 纹理整体平移，噪声 2 个灰度级，每个位移长度 6 个方向（6 个随机种子）；整图 LK 是配图脚本里写的高斯-牛顿迭代：把图像中间 60% 当成一个窗口，只估平移，"
              "每层最多迭代 50 次；相位相关调用第 16 章作业的参考实现（加 Hanning 窗、三点抛物线）；ECC 是 cv2.findTransformECC，从单位变换出发，最多 200 次迭代；"
              "误差是估计平移和真值之差的长度；ECC 抛出不收敛的异常时误差记为空",
        "表": out}


# ---------------------------------------------------------------- 表：Farneback 参数
def table_farneback():
    def run(d, noise, seeds=4, **kw):
        e = np.concatenate([flow_epe(farneback(*pair(d=d, seed=s, noise=noise), **kw), d) for s in range(seeds)])
        return {"EPE 中位数": r(med(e)), "EPE 小于 1 px 的像素（%）": r(100 * (e < 1).mean(), 1)}

    NUM["Farneback 参数"] = {
        "说明": "256x256 纹理整体平移，4 个随机种子，统计离边界 40 px 以内的全部像素；每次只改一个参数，其余用基线参数（levels=4、winsize=15、iterations=3、poly_n=5、poly_sigma=1.2）；噪声的单位是灰度级",
        "平移 (6, 4)，噪声 5：改 winsize": {str(w): run((6, 4), 5, winsize=w) for w in (5, 9, 15, 25, 35)},
        "平移 (6, 4)，噪声 5：改 iterations": {str(n): run((6, 4), 5, iterations=n) for n in (1, 3, 10)},
        "平移 (6, 4)，噪声 5：改 levels": {str(n): run((6, 4), 5, levels=n) for n in (1, 2, 4)},
        "平移 (6, 4)，噪声 5：poly_n=7、poly_sigma=1.5": run((6, 4), 5, poly_n=7, poly_sigma=1.5),
        "无噪声：位移和 levels": {f"平移 ({d[0]}, {d[1]})，长度 {np.hypot(*d):.1f} px": {f"levels={n}": run(d, 0, levels=n) for n in (1, 4, 6)} for d in [(6, 4), (12, 8), (24, 16), (36, 24)]},
        "平移 (12, 8)，无噪声，levels=1：改 iterations": {str(n): run((12, 8), 0, levels=1, iterations=n) for n in (1, 3, 10, 30)},
    }


# ---------------------------------------------------------------- 图 6：亮度、旋转场、耗时
UP = 4


def bilinear(hf, X, Y):
    x, y = np.clip(X * UP, 0, hf.shape[1] - 1.001), np.clip(Y * UP, 0, hf.shape[0] - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return hf[y0, x0] * (1 - fx) * (1 - fy) + hf[y0, x0 + 1] * fx * (1 - fy) + hf[y0 + 1, x0] * (1 - fx) * fy + hf[y0 + 1, x0 + 1] * fx * fy


def rotation_pair(size, deg, seed, noise=5.0, pad=64):
    """绕图像中心旋转 deg 度的一对帧：纹理按 4 倍分辨率生成，两帧都从它上面双线性取样。返回两帧和真实流场。"""
    hf = field((size + 2 * pad) * UP, 2.0 * UP, seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    c = (size - 1) / 2
    cs, sn = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    xs, ys = c + cs * (xx - c) + sn * (yy - c), c - sn * (xx - c) + cs * (yy - c)   # 第二帧像素在第一帧里的位置
    rng = np.random.default_rng(seed + 1)
    a = q8(128 + 40 * bilinear(hf, xx + pad, yy + pad), noise, rng)
    b = q8(128 + 40 * bilinear(hf, xs + pad, ys + pad), noise, rng)
    U, V = (c + cs * (xx - c) - sn * (yy - c)) - xx, (c + sn * (xx - c) + cs * (yy - c)) - yy
    return a, b, U, V


def local_norm(x):
    x = x.astype(np.float32)
    mu = cv2.GaussianBlur(x, (0, 0), 8)
    sd = np.sqrt(np.maximum(cv2.GaussianBlur((x - mu) ** 2, (0, 0), 8), 1e-6))
    return np.clip(np.rint(128 + 40 * (x - mu) / sd), 0, 255).astype(np.uint8)


def measure_time():
    a, b = pair(size=640, d=(3, 2), seed=0, noise=5)
    a, b = np.ascontiguousarray(a[:480]), np.ascontiguousarray(b[:480])

    def ms(fn, rep=7):
        ts = []
        for _ in range(rep):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        return float(f"{1000 * np.median(ts):.2g}")

    p0 = cv2.goodFeaturesToTrack(a, 500, 0.001, 3)
    yy, xx = np.mgrid[0:480, 0:640]
    pall = np.stack([xx, yy], -1).reshape(-1, 1, 2).astype(np.float32)
    return {
        "goodFeaturesToTrack 取 500 个角点（ms）": ms(lambda: cv2.goodFeaturesToTrack(a, 500, 0.001, 3)),
        "LK 跟踪 500 个点，单向（ms）": ms(lambda: lk(a, b, p0)),
        "LK 跟踪 500 个点，往返（ms）": ms(lambda: lk(b, a, lk(a, b, p0)[0])),
        "Farneback，iterations=1（ms）": ms(lambda: farneback(a, b, iterations=1)),
        "Farneback，iterations=3（ms）": ms(lambda: farneback(a, b)),
        "Farneback，iterations=10（ms）": ms(lambda: farneback(a, b, iterations=10)),
        "LK 每个像素一个点，共 307200 个点（ms）": ms(lambda: lk(a, b, pall), rep=1),
    }


def fig_dense():
    # 亮度
    offsets, gains = [0, 5, 10, 20, 40], [1.1, 1.2, 1.4]
    p0 = grid(256, 40, 16)

    def bright(norm=None, wins=(9, 21, 41), **kw):
        el, ef = {w: [] for w in wins}, []
        for s in range(5):
            a, b = pair(d=(3, 2), seed=s, mean=110.0, std=30.0, **kw)
            if norm == "全局":
                b = q8((b.astype(float) - b.mean()) / b.std() * a.std() + a.mean())
            elif norm == "局部":
                a, b = local_norm(a), local_norm(b)
            for w in wins:
                el[w].append(epe(p0, lk(a, b, p0, win=w)[0], (3, 2)))
            ef.append(flow_epe(farneback(a, b), (3, 2)))
        out = {f"LK 窗口 {w}：EPE 中位数": r(med(np.concatenate(v))) for w, v in el.items()}
        out["LK 窗口 21：EPE 超过 1 px 的比例（%）"] = r(100 * (np.concatenate(el[21]) > 1).mean(), 1) if 21 in el else None
        out["Farneback：EPE 中位数"] = r(med(np.concatenate(ef)))
        return out

    btab = {"加偏移（灰度级）": {str(o): bright(offset=o) for o in offsets},
            "乘增益": {str(g): bright(gain=g) for g in gains},
            "左右方向的亮度斜坡（左边乘 0.6，右边乘 1.4）": bright(ramp=0.4),
            "补救：偏移 40，全局匹配均值和标准差之后": bright("全局", offset=40),
            "补救：亮度斜坡，全局匹配均值和标准差之后": bright("全局", ramp=0.4),
            "补救：亮度斜坡，两帧都做局部归一化之后": bright("局部", ramp=0.4)}
    NUM["亮度变化"] = {
        "说明": "256x256 纹理（平均灰度 110，标准差 30 个灰度级），平移 (3, 2)，无噪声，只改第二帧的亮度，5 个随机种子，LK 每个种子 121 个网格点、maxLevel=3，Farneback 用基线参数；"
              "全局匹配 = 把第二帧线性拉伸到和第一帧相同的均值和标准差；局部归一化 = 减去标准差 8 px 的高斯局部均值，再除以同样尺度的局部标准差",
        "表": btab}

    # Notebook 的场景（和 实验_光流.ipynb 里的代码保持一致）
    def scene(scale=1):
        rng = np.random.default_rng(2026)
        tex = cv2.GaussianBlur(rng.normal(110, 32, (120 * scale, 160 * scale)).astype(np.float32), (5, 5), 0.8)
        cv2.circle(tex, (55 * scale, 58 * scale), 20 * scale, 220, -1)
        cv2.rectangle(tex, (95 * scale, 28 * scale), (135 * scale, 88 * scale), 40, -1)
        return np.clip(tex, 0, 255).astype(np.uint8)

    def flat_mask(scale=1):
        yy, xx = np.mgrid[0:120 * scale, 0:160 * scale]
        return ((((xx - 55 * scale) ** 2 + (yy - 58 * scale) ** 2) < (14 * scale) ** 2)
                | ((xx > 101 * scale) & (xx < 129 * scale) & (yy > 34 * scale) & (yy < 82 * scale)))

    f0, flat = scene(), flat_mask()

    def nb(dx=5, dy=3, winsize=15, add=0, gain=None, levels=4, scale=1):
        a0, fl = (f0, flat) if scale == 1 else (scene(scale), flat_mask(scale))
        f1 = cv2.warpAffine(a0, np.float32([[1, 0, dx], [0, 1, dy]]), (a0.shape[1], a0.shape[0]), borderMode=cv2.BORDER_REFLECT)
        if add:
            f1 = cv2.add(f1, add)
        if gain:
            f1 = cv2.convertScaleAbs(f1, alpha=gain)
        flow = cv2.calcOpticalFlowFarneback(a0, f1, None, 0.5, levels, winsize, 3, 5, 1.2, 0)
        m = max(abs(dx), abs(dy), 10) + 4
        E = np.linalg.norm(flow[m:-m, m:-m] - np.float32([dx, dy]), axis=2)
        fm = fl[m:-m, m:-m]
        out = {"Notebook 显示的 EPE（平均值）": r(E.mean()), "EPE 中位数": r(np.median(E)), "饱和到 255 的像素（%）": r(100 * (f1 == 255).mean(), 1)}
        if fm.any():
            out.update({"平坦的圆和矩形内部的 EPE 平均值": r(E[fm].mean()), "其余纹理区的 EPE 平均值": r(E[~fm].mean())})
        return out

    # Notebook 实验 2 和实验 3：同一对帧上的稀疏 LK
    nb_shift = np.float32([5, 3])
    nb_f1 = cv2.warpAffine(f0, np.float32([[1, 0, 5], [0, 1, 3]]), (160, 120), borderMode=cv2.BORDER_REFLECT)
    nb_pts = cv2.goodFeaturesToTrack(f0, 200, 0.01, 5)

    def nb_lk(second):
        out = {}
        for w in (9, 21, 31):
            moved, _, _ = cv2.calcOpticalFlowPyrLK(f0, second, nb_pts, None, winSize=(w, w), maxLevel=3)
            e = np.linalg.norm((moved - nb_pts).reshape(-1, 2) - nb_shift, axis=1)
            out[f"LK 窗口 {w}"] = {"EPE 中位数": r(np.median(e)), "EPE 超过 1 px 的点（%）": r(100 * (e > 1).mean(), 1)}
        return out

    def nb_round_trip(max_level):
        fwd, s1, _ = cv2.calcOpticalFlowPyrLK(f0, nb_f1, nb_pts, None, winSize=(21, 21), maxLevel=max_level)
        bwd, s2, _ = cv2.calcOpticalFlowPyrLK(nb_f1, f0, fwd, None, winSize=(21, 21), maxLevel=max_level)
        ok = (s1.ravel() != 0) & (s2.ravel() != 0)
        e = np.linalg.norm((fwd - nb_pts).reshape(-1, 2) - nb_shift, axis=1)
        out = {"前向和反向的 status 都是 1 的点数": int(ok.sum()), "EPE 小于 1 px 的点数": int((e < 1).sum())}
        for thr in (0.1, 1.0, 5.0):
            keep = filter_tracks(nb_pts, fwd, bwd, ok, thr)
            out[f"往返误差不超过 {thr:g} px"] = {"留下的点数": int(keep.sum()), "其中 EPE 不小于 1 px 的点数": int((keep & (e >= 1)).sum())}
        return out

    xy = np.rint(nb_pts.reshape(-1, 2)).astype(int)
    NUM["Notebook 场景"] = {
        "说明": "按 实验_光流.ipynb 的代码生成 160x120 的两帧（随机种子 2026，纹理上画了一个灰度 220 的圆和一个灰度 40 的矩形），用它的 Farneback 参数和它的 EPE 算法（去掉边缘后取平均）；"
              "平坦内部 = 圆心 14 px 以内和矩形向内收 6 px 的部分；变亮用 cv2.add（超过 255 的截断），乘增益用 cv2.convertScaleAbs；"
              "稀疏 LK 跟踪 goodFeaturesToTrack(frame0, 200, 0.01, 5) 取的角点，其余参数用 OpenCV 的默认值；"
              "放大的画面是把画布、圆和矩形都放大 4 倍（宽 640、高 480），纹理的粗细不变",
        "位移的长度（px）": {f"({dx}, {dy})": r(np.hypot(dx, dy), 1) for dx, dy in [(5, 3), (8, 5), (10, 6), (18, 12)]},
        "默认：位移 (5, 3)，winsize 15": nb(),
        "第二帧整体加 20": nb(add=20), "第二帧整体加 40": nb(add=40), "第二帧整体加 80": nb(add=80),
        "位移 (5, 3)，winsize 5": nb(winsize=5), "位移 (5, 3)，winsize 31": nb(winsize=31),
        "滑块最大位移 (18, 12)，winsize 5": nb(18, 12, 5), "滑块最大位移 (18, 12)，winsize 15": nb(18, 12, 15), "滑块最大位移 (18, 12)，winsize 31": nb(18, 12, 31),
        "实验 1：位移 (8, 5)": {f"winsize {w}": nb(8, 5, w) for w in (5, 15, 31)},
        "实验 1：位移 (10, 6)": {f"winsize {w}": nb(10, 6, w) for w in (5, 15, 31)},
        "实验 1：位移 (18, 12)、winsize 15，levels 取 1、2、4、8 时 Notebook 显示的 EPE": [nb(18, 12, 15, levels=n)["Notebook 显示的 EPE（平均值）"] for n in (1, 2, 4, 8)],
        "实验 1：画面放大到 640x480，位移 (18, 12)、winsize 15": {f"levels={n}": nb(18, 12, 15, levels=n, scale=4) for n in (1, 4)},
        "实验 2：位移 (5, 3)，Farneback（winsize 15）": {"第二帧整体加 10": nb(add=10), "第二帧整体乘 1.1": nb(gain=1.1)},
        "实验 2：位移 (5, 3)，稀疏 LK（maxLevel=3）": {
            "角点数": len(nb_pts),
            "不变": nb_lk(nb_f1), "第二帧整体加 10": nb_lk(cv2.add(nb_f1, 10)), "第二帧整体加 20": nb_lk(cv2.add(nb_f1, 20)),
            "第二帧整体乘 1.1": nb_lk(cv2.convertScaleAbs(nb_f1, alpha=1.1))},
        "实验 3：位移 (5, 3)，winSize=21，往返筛选（参考实现的 filter_tracks）": {
            "角点数": len(nb_pts), "落在平坦内部的角点数": int(flat[np.clip(xy[:, 1], 0, 119), np.clip(xy[:, 0], 0, 159)].sum()),
            "maxLevel=3": nb_round_trip(3), "maxLevel=0": nb_round_trip(0)},
    }

    # 旋转场
    rot = {}
    size = 384
    c = (size - 1) / 2
    for deg in (0.5, 1, 2, 4):
        el, ef, efp = [], [], []
        for seed in range(4):
            a, b, U, V = rotation_pair(size, deg, seed)
            q = cv2.goodFeaturesToTrack(a, 300, 0.01, 7, blockSize=7).reshape(-1, 2)
            q = q[np.abs(q - c).max(1) < c - 30]
            pts = q.reshape(-1, 1, 2)
            p1, _, _ = lk(a, b, pts)
            qi = np.rint(q).astype(int)
            el.append(np.linalg.norm((p1.reshape(-1, 2) - q) - np.stack([U[qi[:, 1], qi[:, 0]], V[qi[:, 1], qi[:, 0]]], 1), axis=1))
            fl = farneback(a, b)
            E = np.hypot(fl[..., 0] - U, fl[..., 1] - V)
            ef.append(E[30:-30, 30:-30].ravel())
            efp.append(E[qi[:, 1], qi[:, 0]])
        rot[f"{deg:g} 度"] = {"统计范围内的最大位移（px）": r(np.hypot(c - 30, c - 30) * 2 * np.sin(np.deg2rad(deg) / 2), 1),
                             "LK（角点处）：EPE 中位数": r(med(np.concatenate(el))), "Farneback（全部像素）：EPE 中位数": r(med(np.concatenate(ef))),
                             "Farneback（同一批角点处）：EPE 中位数": r(med(np.concatenate(efp)))}
    NUM["旋转场"] = {
        "说明": "384x384 纹理绕图像中心旋转，噪声 5 个灰度级，4 个随机种子；位移随位置变化，真值按旋转公式逐点算；LK 跟踪 goodFeaturesToTrack 取的角点（最多 300 个，去掉离边界 30 px 以内的），"
              "winSize=21、maxLevel=3；LK 的真值取角点四舍五入到整数像素处的值；Farneback 用基线参数，统计离边界 30 px 以内的全部像素",
        "表": rot}

    T = {k: v for k, v in S.pinned(CH, "耗时", measure_time, "CH06_RETIME").items() if k != "说明"}
    NUM["耗时"] = {
        "说明": "宽 640、高 480 的纹理图（640x480），平移 (3, 2)，噪声 5；在一台 Apple 芯片的 Mac 上、OpenCV 5.0.0 限制为单线程时测的一次结果，每项重复 7 次取中位数（逐像素 LK 只跑 1 次），"
              "保留两位有效数字；机器忙的时候会更慢，只看量级和倍数。这一组数默认沿用上次的测量，设置 CH06_RETIME=1 才重测",
        **T}

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.7), layout="constrained")
    top = 1.25
    for w, col in [(9, S.RED), (21, S.ORANGE), (41, S.GREEN)]:
        v = [btab["加偏移（灰度级）"][str(o)][f"LK 窗口 {w}：EPE 中位数"] for o in offsets]
        ax[0].plot(offsets, v, "o-", color=col, lw=1.6, ms=4, label=f"LK，窗口 {w}")
        if v[-1] > top:
            ax[0].text(39.5, top * 0.93, f"窗口 {w}，偏移 40：\n{v[-1]:.1f} px（画不下）", color=col, fontsize=9, ha="right", va="top")
    ax[0].set_ylim(-0.04, top)
    ax[0].plot(offsets, [btab["加偏移（灰度级）"][str(o)]["Farneback：EPE 中位数"] for o in offsets], "s-", color=S.BLUE, lw=1.8, ms=4.5, label="Farneback")
    ax[0].set_xticks(offsets)
    ax[0].set_xlabel("第二帧整体加的偏移（灰度级）")
    ax[0].set_ylabel("EPE 中位数（px）")
    ax[0].set_title("整体变亮：LK 的误差跟着涨，Farneback 不变")
    ax[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    S.clean(ax[0])
    degs = [0.5, 1, 2, 4]
    ax[1].plot(range(4), [rot[f"{g:g} 度"]["LK（角点处）：EPE 中位数"] for g in degs], "o-", color=S.ORANGE, lw=1.8, ms=5, label="LK（角点处）")
    ax[1].plot(range(4), [rot[f"{g:g} 度"]["Farneback（全部像素）：EPE 中位数"] for g in degs], "s-", color=S.BLUE, lw=1.8, ms=5, label="Farneback（全部像素）")
    ax[1].set_xticks(range(4), [f"{g:g}°" for g in degs])
    ax[1].set_ylim(0, None)
    ax[1].set_xlabel("两帧之间的旋转角")
    ax[1].set_ylabel("EPE 中位数（px）")
    ax[1].set_title("纹理充足时两者精度接近")
    ax[1].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[1])
    items = [("LK 500 点，单向", "LK 跟踪 500 个点，单向（ms）", S.ORANGE), ("LK 500 点，往返", "LK 跟踪 500 个点，往返（ms）", S.ORANGE),
             ("Farneback（iterations=3）", "Farneback，iterations=3（ms）", S.BLUE), ("LK 每个像素一个点", "LK 每个像素一个点，共 307200 个点（ms）", S.RED)]
    vals = [T[k] for _, k, _ in items]
    ax[2].barh(range(len(items)), vals, color=[c_ for _, _, c_ in items], height=0.6)
    for i, v in enumerate(vals):
        ax[2].text(v * 1.15, i, f"{v:g} ms", va="center", fontsize=9)
    ax[2].set_xscale("log")
    ax[2].set_xlim(1, max(vals) * 8)
    ax[2].set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    ax[2].minorticks_off()
    ax[2].set_yticks(range(len(items)), [n for n, _, _ in items])
    ax[2].invert_yaxis()
    ax[2].set_xlabel("耗时（ms，对数轴），640×480")
    ax[2].set_title("点少时 LK 快，逐像素做 LK 最慢")
    S.clean(ax[2])
    save(fig, "fig06_亮度与稀疏稠密")


# ---------------------------------------------------------------- 表：接口约定
def table_api():
    a, b = pair(d=(3, 2), seed=0)
    p = grid(256, 40, 32)

    def accepts(fn):
        try:
            fn()
            return True
        except cv2.error:
            return False

    lk_pts = {name: accepts(lambda q=q: cv2.calcOpticalFlowPyrLK(a, b, q, None)) for name, q in [
        ("float32 (N,1,2)", p), ("float32 (N,2)", p.reshape(-1, 2)), ("float64 (N,1,2)", p.astype(np.float64)), ("int32 (N,1,2)", p.astype(np.int32))]}
    out_shape = list(cv2.calcOpticalFlowPyrLK(a, b, p.reshape(-1, 2), None)[0].shape)
    imgs = {"float32 单通道": (a.astype(np.float32), b.astype(np.float32)), "uint16 单通道": (a.astype(np.uint16), b.astype(np.uint16)), "uint8 三通道": (cv2.merge([a] * 3), cv2.merge([b] * 3))}
    lk_img = {k: accepts(lambda v=v: cv2.calcOpticalFlowPyrLK(v[0], v[1], p, None)) for k, v in imgs.items()}
    fb_img = {k: accepts(lambda v=v: farneback(v[0], v[1])) for k, v in imgs.items()}
    p0 = grid(256, 40, 16)
    f_lk = np.median((lk(a, b, p0)[0] - p0).reshape(-1, 2), 0)
    r_lk = np.median((lk(b, a, p0)[0] - p0).reshape(-1, 2), 0)
    f_fb = np.median(farneback(a, b)[40:-40, 40:-40].reshape(-1, 2), 0)
    r_fb = np.median(farneback(b, a)[40:-40, 40:-40].reshape(-1, 2), 0)
    # err 的含义
    errs = {}
    for name, d, lev in [("跟对了：平移 (3, 2)，maxLevel=3", (3, 2), 3), ("跟丢了：平移 (30, 20)，maxLevel=0", (30, 20), 0)]:
        a2, b2 = pair(size=384, d=d, seed=1, noise=5)
        q0 = grid(384, 90, 20)
        q1, st, err = lk(a2, b2, q0, lev=lev)
        man = [np.abs(cv2.getRectSubPix(a2, (21, 21), (float(x0), float(y0))).astype(float) - cv2.getRectSubPix(b2, (21, 21), (float(x1), float(y1))).astype(float)).mean()
               for (x0, y0), (x1, y1) in zip(q0.reshape(-1, 2), q1.reshape(-1, 2))]
        _, _, ev = lk(a2, b2, q0, lev=lev, flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS)
        errs[name] = {"status=1 的比例（%）": r(100 * st.mean(), 1), "EPE 中位数（px）": r(med(epe(q0, q1, d)), 2), "err 中位数": r(med(err), 1),
                      "手算的窗口内平均绝对灰度差，中位数": r(med(man), 1), "err 和手算值之差的最大绝对值": r(np.abs(err - np.array(man)).max(), 2),
                      "flags 加 OPTFLOW_LK_GET_MIN_EIGENVALS 后 err 的中位数": r(med(ev), 3)}
    NUM["接口"] = {
        "说明": "OpenCV 5.0.0。输入是 256x256 的 8 位纹理、平移 (3, 2)；err 的两行用 384x384、噪声 5 个灰度级、winSize=21、121 个网格点",
        "calcOpticalFlowPyrLK 接受的点集": lk_pts, "点集传 (N,2) 时返回的 nextPts 的 shape": out_shape,
        "calcOpticalFlowPyrLK 接受的图像": lk_img, "calcOpticalFlowFarneback 接受的图像": fb_img,
        "前后帧传反": {"LK 正常顺序的位移中位数": [r(v, 2) for v in f_lk], "LK 两帧交换后的位移中位数": [r(v, 2) for v in r_lk],
                  "Farneback 正常顺序的位移中位数": [r(v, 2) for v in f_fb], "Farneback 两帧交换后的位移中位数": [r(v, 2) for v in r_fb]},
        "err 的含义": errs}


def main() -> None:
    S.setup()
    for fn in (table_baseline, fig_regions, fig_aperture, fig_window, fig_fb, fig_pyramid, table_global, table_farneback, fig_dense, table_api):
        t = time.perf_counter()
        fn()
        print(f"完成 {fn.__name__}（{time.perf_counter() - t:.1f} s）", flush=True)
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False))


if __name__ == "__main__":
    main()
