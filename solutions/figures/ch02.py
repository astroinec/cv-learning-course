"""第 2 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch02.py

输出：<主分支>/02_卷积与边缘/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。
相关运算一律调用本章作业的参考实现 correlate2d；Canny 和对照用 OpenCV。
“Canny 的耗时”“可分离卷积的耗时”两组是计时结果，用 S.pinned 钉住：默认沿用 实测数字.json 里上次记下的值，
所以重复运行脚本，figures/ 下所有文件逐字节不变。要重测时在机器空闲的时候加上 CH02_RETIME=1 运行，
重测后按新数字改讲义（讲义里耗时只写两位有效数字）。其余数字固定种子、重复运行不变。
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

CH = "02_卷积与边缘"
mod = S.load_solution(CH, "2-1_手写相关运算.py")
corr = mod.correlate2d
NUM: dict = {}
LO, HI, BLUR = 60.0, 160.0, 0.7          # 暗、亮两侧的灰度，边缘本身的模糊（高斯标准差，px）

DIFF_X = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], np.float32)
PREWITT_X = np.array([[-1, 0, 1]] * 3, np.float32)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
SCHARR_X = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], np.float32)
LAP4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
SOBEL_Y = SOBEL_X.T.copy()
OPS = {"中心差分": DIFF_X, "Prewitt": PREWITT_X, "Sobel": SOBEL_X, "Scharr": SCHARR_X}
MOTION = np.array([[0, 0, 0.2, 0.3, 0.5]], np.float32)   # 非对称的模糊核
_ERF = np.frompyfunc(math.erf, 1, 1)


# ---------------------------------------------------------------- 数据与工具
def phi(t):
    return 0.5 * (1.0 + _ERF(np.asarray(t, dtype=np.float64) / math.sqrt(2.0)).astype(np.float64))


def r3(v, n=3):
    return round(float(v), n)


def gauss1d(sigma):
    r = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-r, r + 1)
    g = np.exp(-x ** 2 / (2.0 * sigma ** 2))
    return (g / g.sum()).astype(np.float32)


def smooth2d(img, sigma):
    """可分离的高斯平滑：一行核、一列核各做一次相关（参考实现，reflect 填充）。"""
    if sigma <= 0:
        return np.asarray(img, np.float32)
    g = gauss1d(sigma)
    return corr(corr(img, g[None, :]), g[:, None])


def smooth_x(img, sigma):
    return np.asarray(img, np.float32) if sigma <= 0 else corr(img, gauss1d(sigma)[None, :])


def camera(img, rng=None, noise=0.0):
    """加高斯噪声并量化成 8 位，模拟相机输出。"""
    out = np.asarray(img, np.float64)
    if noise:
        out = out + rng.normal(scale=noise, size=out.shape)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def edge_img(h, w, x0, lo=LO, hi=HI, blur=BLUR):
    return np.tile(lo + (hi - lo) * phi((np.arange(w) - x0) / blur), (h, 1))


def line_img(h, w, xc, width, lo=LO, hi=HI, blur=BLUR):
    x = np.arange(w)
    return np.tile(lo + (hi - lo) * (phi((x - (xc - width / 2)) / blur) - phi((x - (xc + width / 2)) / blur)), (h, 1))


def oriented(theta, off=0.0, n=65, contrast=100.0, base=80.0, blur=BLUR):
    """法线方向为 theta（度）的直边，返回图像和每个像素到边的有符号距离。"""
    th = math.radians(theta)
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    c = (n - 1) / 2
    d = (x - c) * math.cos(th) + (y - c) * math.sin(th) - off
    return base + contrast * phi(d / blur), d


def peak(g, lo, hi, sign=1.0):
    """逐行在 [lo, hi) 列里找 sign*g 的最大值，再用三点抛物线取小数，返回每行的列坐标。"""
    lo, hi = max(lo, 0), min(hi, g.shape[1])
    i = (sign * g[:, lo:hi]).argmax(axis=1) + lo
    j = np.clip(i, 1, g.shape[1] - 2)
    rows = np.arange(g.shape[0])
    left, mid, right = sign * g[rows, j - 1], sign * g[rows, j], sign * g[rows, j + 1]
    den = left - 2 * mid + right
    ok = (np.abs(den) > 1e-9) & (i == j)
    delta = np.where(ok, 0.5 * (left - right) / np.where(ok, den, 1.0), 0.0)
    return i + np.clip(delta, -0.5, 0.5)


def corr_pad(img, k, mode):
    """和参考实现同样的相关运算，但填充方式可选（参考实现固定用 reflect）。"""
    img = np.asarray(img, np.float32)
    ry, rx = k.shape[0] // 2, k.shape[1] // 2
    p = np.pad(img, ((ry, ry), (rx, rx)), mode=mode)
    out = np.zeros(img.shape, np.float32)
    for dy in range(k.shape[0]):
        for dx in range(k.shape[1]):
            out += k[dy, dx] * p[dy:dy + img.shape[0], dx:dx + img.shape[1]]
    return out


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


# ---------------------------------------------------------------- 图 1：同一条线，三种做法量线宽
def fig_why():
    rng = np.random.default_rng(0)
    h, w, width, noise, trials = 40, 80, 5.0, 10.0, 40
    sigmas = [0, 1, 2, 3]
    ex = camera(line_img(h, w, 40.2, width), rng, noise)
    r_float = corr(ex, SOBEL_X)
    r_u8 = cv2.Sobel(ex, -1, 1, 0, ksize=3)
    same = float(np.abs(cv2.Sobel(ex, cv2.CV_32F, 1, 0, ksize=3) - r_float).max())
    err = {s: [] for s in sigmas}
    right_f, right_u, left_f, left_u = [], [], [], []
    for _ in range(trials):
        xc = 40 + rng.uniform(-0.5, 0.5)
        img = camera(line_img(h, w, xc, width), rng, noise)
        c = int(round(xc))
        for s in sigmas:
            g = corr(smooth2d(img, s), SOBEL_X)
            err[s].append((peak(g, c, c + 21, -1.0) - peak(g, c - 20, c + 1))[6:-6] - width)
        gf, g8 = corr(img, SOBEL_X)[6:-6], cv2.Sobel(img, -1, 1, 0, ksize=3)[6:-6]
        cr, cl = int(round(xc + width / 2)), int(round(xc - width / 2))
        right_f.append(gf[:, cr - 1:cr + 2].min(axis=1))
        right_u.append(g8[:, cr - 1:cr + 2].max(axis=1))
        left_f.append(gf[:, cl - 1:cl + 2].max(axis=1))
        left_u.append(g8[:, cl - 1:cl + 2].max(axis=1))
    err = {s: np.concatenate(v) for s, v in err.items()}
    right_f, right_u, left_f, left_u = (np.concatenate(v) for v in (right_f, right_u, left_f, left_u))
    shown = cv2.convertScaleAbs(r_float)[6:-6]
    NUM["线宽测量的三种做法"] = {
        "说明": f"{h}x{w} 的图，一条竖直亮线，真实线宽 {width:g} px，暗处灰度 {LO:g}、亮处 {HI:g}，边缘模糊 {BLUR} px，噪声标准差 {noise:g}，量化成 uint8；"
              f"{trials} 张图，每张取中间 {h - 12} 行，共 {err[0].size} 行；每行用 Sobel x 的正峰和负峰（三点抛物线取小数）之差作为线宽",
        "线宽真值": width,
        "参考实现与 cv2.Sobel(CV_32F) 的最大差": r3(same, 6),
        "高斯 sigma": sigmas,
        "线宽的平均偏差（px）": [r3(err[s].mean()) for s in sigmas],
        "线宽的标准差（px）": [r3(err[s].std()) for s in sigmas],
        "平均偏差占线宽的比例": [f"{100 * err[s].mean() / width:.1f}%" for s in sigmas],
        "输出存成 uint8（ddepth=-1）时": {
            "右边缘处浮点响应的平均值": r3(right_f.mean(), 1),
            "右边缘处 uint8 响应的最大值": int(right_u.max()),
            "左边缘处浮点响应峰值的平均": r3(left_f.mean(), 1),
            "左边缘峰值被削成 255 的行所占比例": r3((left_u == 255).mean()),
        },
        "对浮点响应做 cv2.convertScaleAbs（示例图中间 28 行）": {"左边缘附近的最大值": int(shown[:, 36:40].max()), "右边缘附近的最大值": int(shown[:, 41:45].max())},
    }

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [0.9, 1.35, 1.2]}, layout="constrained")
    ax[0].imshow(ex[:, 20:60], cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("5 px 宽的亮线，噪声 10")
    S.image_axes(ax[0])
    x = np.arange(w)
    ax[1].plot(x, r_float[20], color=S.BLUE, lw=1.6, label="输出 float32：一正一负两个峰")
    ax[1].plot(x, r_u8[20].astype(float), color=S.ORANGE, lw=2.2, label="输出 uint8：负峰是 0，正峰削平在 255")
    for xe in (40.2 - width / 2, 40.2 + width / 2):
        ax[1].axvline(xe, color=S.GREEN, lw=1, ls="--")
    ax[1].set_xlim(22, 58)
    ax[1].set_xlabel("列坐标（px），绿色虚线是真实边缘")
    ax[1].set_ylabel("Sobel x 的响应")
    ax[1].set_title("存成 uint8：右边缘消失了")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax[1])
    means, stds = [err[s].mean() for s in sigmas], [err[s].std() for s in sigmas]
    ax[2].errorbar(range(len(sigmas)), means, yerr=stds, fmt="o", color=S.BLUE, ms=6, capsize=6, lw=1.8)
    for i, (m, sd) in enumerate(zip(means, stds)):
        ax[2].text(i, m + sd + 0.09, f"{round(m, 2) + 0.0:+.2f} ± {sd:.2f}", fontsize=9, ha="center")
    ax[2].axhline(0, color=S.MUTED, lw=1, ls="--")
    ax[2].set_xticks(range(len(sigmas)), ["不平滑", "sigma=1", "sigma=2", "sigma=3"])
    ax[2].set_xlim(-0.5, len(sigmas) - 0.5)
    ax[2].set_ylim(-0.4, 2.5)
    ax[2].set_ylabel("线宽误差（px）：平均值 ± 标准差")
    ax[2].set_title("平滑越重，读数越集中，偏得越多")
    S.clean(ax[2])
    S.save(fig, CH, "fig02_线宽测量的三种做法")


# ---------------------------------------------------------------- 图 2：相关与卷积
def fig_corr_conv():
    K = np.arange(9, dtype=np.float32).reshape(3, 3)
    imp = np.zeros((7, 7), np.float32)
    imp[3, 3] = 1
    r_corr = corr(imp, K)[2:5, 2:5]
    r_conv = corr(imp, K[::-1, ::-1].copy())[2:5, 2:5]
    r_cv = cv2.filter2D(imp, cv2.CV_32F, K)[2:5, 2:5]
    try:
        import torch
        r_torch = torch.nn.functional.conv2d(torch.from_numpy(imp)[None, None], torch.from_numpy(K)[None, None], padding=1)[0, 0, 2:5, 2:5].numpy().astype(int).tolist()
    except Exception:  # noqa: BLE001
        r_torch = "没有测（torch 不可用）"
    rng = np.random.default_rng(1)
    tex = rng.uniform(0, 255, (64, 64)).astype(np.float32)
    g5 = cv2.getGaussianKernel(5, 1.0).astype(np.float32)
    kernels = {"高斯 5x5（sigma=1）": (g5 @ g5.T).astype(np.float32), "均值 3x3": np.full((3, 3), 1 / 9, np.float32),
               "Laplacian（4 邻域）": LAP4, "Sobel x": SOBEL_X, "非对称的模糊核 1x5": MOTION}
    table = {}
    for name, k in kernels.items():
        a, b = corr(tex, k), corr(tex, k[::-1, ::-1].copy())
        table[name] = {"相关与卷积的最大差": r3(np.abs(a - b).max(), 2), "最大差除以响应的标准差": r3(np.abs(a - b).max() / a.std(), 2),
                       "相关加卷积的最大绝对值（为 0 表示只差符号）": r3(np.abs(a + b).max(), 2)}
    patch = np.array([[10, 10, 50]] * 3, np.float32)
    small = {"左暗右亮": r3(corr(patch, PREWITT_X)[1, 1], 1), "左右对调": r3(corr(patch[:, ::-1].copy(), PREWITT_X)[1, 1], 1),
             "核翻转后（卷积）": r3(corr(patch, PREWITT_X[::-1, ::-1].copy())[1, 1], 1)}

    x0 = 32.3
    img = edge_img(24, 64, x0)
    blurred = {"相关": corr(img, MOTION), "卷积": corr(img, MOTION[:, ::-1].copy())}
    pos = {n: float(peak(corr(b, DIFF_X), 20, 45).mean()) for n, b in blurred.items()}
    fwd = np.array([[0, -1, 1]], np.float32)                     # 前向差分：I(x+1) - I(x)，不对称
    fpos = {"相关": float(peak(corr(img, fwd), 20, 45).mean()), "卷积": float(peak(corr(img, fwd[:, ::-1].copy()), 20, 45, -1.0).mean())}
    line = line_img(8, 64, 32.0, 8.0)
    s_corr, s_conv = corr(line, SOBEL_X)[4], corr(line, SOBEL_X[::-1, ::-1].copy())[4]
    NUM["相关与卷积"] = {
        "说明": "用参考实现 correlate2d 计算相关；卷积是先把核上下左右翻转再做相关。对比用 64x64 的均匀随机纹理（种子 1）",
        "冲激响应": {"核 K": K.astype(int).tolist(), "相关的输出（中心 3x3）": r_corr.astype(int).tolist(),
                 "卷积的输出（中心 3x3）": r_conv.astype(int).tolist(), "cv2.filter2D 的输出（中心 3x3）": r_cv.astype(int).tolist(),
                 "torch.nn.functional.conv2d 的输出（中心 3x3）": r_torch},
        "各种核": table,
        "3x3 小例子（核三行都是 -1 0 1，取中心像素）": small,
        "非对称核对边位置的影响": {
            "说明": f"竖直边真实位置 {x0}，对比度 {HI - LO:g}，无噪声；先用核 [0, 0, 0.2, 0.3, 0.5] 模糊，再用中心差分的峰加抛物线定位",
            "相关后测得的位置": r3(pos["相关"]), "卷积后测得的位置": r3(pos["卷积"]),
            "相关的位置偏差": r3(pos["相关"] - x0), "卷积的位置偏差": r3(pos["卷积"] - x0), "两者相差": r3(pos["卷积"] - pos["相关"]),
        },
        "前向差分核 [0, -1, 1]（同一条边，直接在原图上求导）": {"相关的响应峰位置减真值": r3(fpos["相关"] - x0), "卷积的响应峰位置减真值": r3(fpos["卷积"] - x0),
                                           "两者相差": r3(fpos["卷积"] - fpos["相关"])},
    }

    fig = plt.figure(figsize=(13.5, 3.7), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 1.15])
    sub = gs[0].subgridspec(1, 3, wspace=0.08)
    for i, (mat, title) in enumerate(((K, "核 K"), (r_corr, "相关：输出是翻转的 K"), (r_conv, "卷积：输出就是 K"))):
        a = fig.add_subplot(sub[i])
        a.imshow(mat, cmap="Blues", vmin=-2, vmax=12)
        for (yy, xx), v in np.ndenumerate(mat):
            a.text(xx, yy, f"{int(v)}", ha="center", va="center", fontsize=11, color=S.INK)
        a.set_title(title, fontsize=10)
        S.image_axes(a)
    a1 = fig.add_subplot(gs[1])
    xs = np.arange(64)
    a1.plot(xs, s_corr, color=S.BLUE, lw=1.8, label="相关")
    a1.plot(xs, s_conv, color=S.ORANGE, lw=1.8, ls="--", label="卷积（核翻转）")
    a1.set_xlim(20, 44)
    a1.set_xlabel("列坐标（px），8 px 宽的亮线")
    a1.set_ylabel("Sobel x 的响应")
    a1.set_title("Sobel：两者只差一个符号")
    a1.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(a1)
    a2 = fig.add_subplot(gs[2])
    a2.plot(xs, img[12], color=S.MUTED, lw=1.4, label="模糊前")
    a2.plot(xs, blurred["相关"][12], color=S.BLUE, lw=1.8, label=f"相关后，测得 {pos['相关']:.2f}")
    a2.plot(xs, blurred["卷积"][12], color=S.ORANGE, lw=1.8, label=f"卷积后，测得 {pos['卷积']:.2f}")
    a2.axvline(x0, color=S.GREEN, lw=1, ls="--")
    a2.set_xlim(25, 40)
    a2.set_xlabel(f"列坐标（px），绿色虚线是真实位置 {x0}")
    a2.set_ylabel("灰度")
    a2.set_title(f"非对称核：测出的边相差 {pos['卷积'] - pos['相关']:.1f} px")
    a2.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, columnspacing=1.0, handlelength=1.4)
    S.clean(a2)
    S.save(fig, CH, "fig01_相关与卷积")


# ---------------------------------------------------------------- 图 3：边界填充
MODES = {"复制边缘": "edge", "镜像，边界像素重复": "symmetric", "镜像，边界像素不重复": "reflect", "补零": "constant"}
CVNAME = {"edge": ("BORDER_REPLICATE", cv2.BORDER_REPLICATE), "symmetric": ("BORDER_REFLECT", cv2.BORDER_REFLECT),
          "reflect": ("BORDER_REFLECT_101（默认）", cv2.BORDER_REFLECT_101), "constant": ("BORDER_CONSTANT", cv2.BORDER_CONSTANT)}


def fig_border():
    rng = np.random.default_rng(2)
    tex = rng.uniform(0, 255, (64, 64)).astype(np.float32)
    k5 = rng.normal(size=(5, 5)).astype(np.float32)
    ref = corr(tex, k5)
    mapping = {}
    for label, mode in MODES.items():
        cvname, bt = CVNAME[mode]
        out = cv2.filter2D(tex, cv2.CV_32F, k5, borderType=bt)
        mapping[cvname] = {"中文叫法": label, "对应的 np.pad mode": mode,
                           "filter2D 与同名 np.pad 写法的最大差": r3(np.abs(out - corr_pad(tex, k5, mode)).max(), 4),
                           "与作业参考实现（np.pad reflect）的最大差": r3(np.abs(out - ref).max(), 3),
                           "与参考实现不同的像素占比（差大于 0.01）": r3((np.abs(out - ref) > 0.01).mean())}
    flat = np.full((16, 16), 100, np.float32)
    ramp = np.tile(100 + 2.0 * np.arange(16, dtype=np.float32), (16, 1))
    g2 = gauss1d(2.0)[None, :]
    at_border = {label: {"平场图最左一列的 Sobel x 响应": r3(corr_pad(flat, SOBEL_X, m)[8, 0], 1),
                         "斜坡图最左一列的 Sobel x 响应": r3(corr_pad(ramp, SOBEL_X, m)[8, 0], 1),
                         "斜坡图最左一列经 sigma=2 高斯平滑后的灰度误差": r3((corr_pad(ramp, g2, m) - ramp)[8, 0], 2)} for label, m in MODES.items()}
    ds, phases = [2, 3, 4, 5, 6, 8], np.linspace(-0.4, 0.4, 9)
    near = {}
    for sigma in (1.0, 2.0):
        g = gauss1d(sigma)
        near[sigma] = {}
        for label, m in MODES.items():
            row = []
            for d in ds:
                errs = []
                for f in phases:
                    img = edge_img(24, 64, d + f)
                    sm = corr_pad(corr_pad(img, g[None, :], m), g[:, None], m)
                    errs.append(float(peak(corr_pad(sm, SOBEL_X, m), 0, d + 10)[4:-4].mean()) - (d + f))
                row.append(float(np.mean(errs)))
            near[sigma][label] = row
    radius = int(math.ceil(3 * 2.0)) + 1
    NUM["边界填充"] = {
        "说明": "OpenCV 的 borderType 与 np.pad 的 mode 一一对照，用 64x64 随机纹理（种子 2）和一个 5x5 随机核检验；作业要求的是 np.pad 的 reflect",
        "名称对照": mapping,
        "边界一列的响应": {"说明": "16x16 的图。平场图灰度恒为 100；斜坡图每列亮 2，内部的 Sobel x 响应是 16", "斜坡图内部的 Sobel x 响应": r3(corr(ramp, SOBEL_X)[8, 8], 1),
                      "各填充方式": at_border},
        "靠近边界的边": {"说明": f"竖直边离图像左边界 d px（对比度 {HI - LO:g}，边缘模糊 {BLUR} px，无噪声），高斯平滑和 Sobel 都用同一种填充；"
                             "在第 0 列到第 d+9 列之间找最强的正响应并取小数，对 9 个亚像素位置取平均，数值是测得位置减真值（px）",
                   "d": ds, "sigma=1": {k: [r3(v) for v in vals] for k, vals in near[1.0].items()},
                   "sigma=2": {k: [r3(v) for v in vals] for k, vals in near[2.0].items()}},
        "受填充影响的范围": {"说明": "sigma=2 的高斯（半径 6）加 3x3 Sobel，最外面这么多圈像素的结果用到了填充出来的值", "圈数": radius,
                     "在 512x512 图上占的面积": f"{100 * (1 - ((512 - 2 * radius) / 512) ** 2):.1f}%"},
    }

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw={"width_ratios": [1.05, 1.15, 1.3]}, layout="constrained")
    scene = camera(np.tile(100 + 0.6 * np.arange(64), (48, 1)), rng, 2.0)
    mag = np.hypot(corr_pad(scene, SOBEL_X, "constant"), corr_pad(scene, SOBEL_Y, "constant"))
    im = ax[0].imshow(mag, cmap="magma")
    ax[0].set_title("补零：最强的“边”是图像的四周")
    fig.colorbar(im, ax=ax[0], fraction=0.035, pad=0.03).ax.tick_params(labelsize=8)
    S.image_axes(ax[0])
    labels = list(MODES)
    vals = [at_border[k]["斜坡图最左一列的 Sobel x 响应"] for k in labels]
    colors = [S.GREEN, S.ORANGE, S.BLUE, S.RED]
    ax[1].barh(range(4), [min(v, 26) for v in vals], color=colors, alpha=0.85, height=0.55)
    for i, v in enumerate(vals):
        ax[1].text(min(v, 26) + 0.6, i, f"{v:g}" + ("（画不下）" if v > 26 else ""), va="center", fontsize=9.5)
    ax[1].axvline(16, color=S.MUTED, lw=1.2, ls="--")
    ax[1].text(16.5, 2.0, "虚线：内部的响应 16", fontsize=8.5, color=S.MUTED, va="center")
    ax[1].set_yticks(range(4), ["复制边缘", "镜像\n（重复边界像素）", "镜像\n（不重复，默认）", "补零"], fontsize=9)
    ax[1].set_xlim(0, 36)
    ax[1].set_ylim(-0.5, 3.75)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("斜坡图最左一列的 Sobel x 响应")
    ax[1].set_title("最左一列：没有一种填充给出 16")
    S.clean(ax[1])
    ax[1].grid(axis="y", visible=False)
    for (label, vals2), color, marker in zip(near[2.0].items(), colors, "o^sD"):
        ax[2].plot(ds, [max(abs(v), 1e-3) for v in vals2], marker + "-", color=color, lw=1.8, ms=5, label=label)
    ax[2].set_yscale("log")
    ax[2].set_yticks([1e-3, 1e-2, 0.1, 1, 10], ["≤0.001", "0.01", "0.1", "1", "10"])
    ax[2].minorticks_off()
    ax[2].set_xticks(ds)
    ax[2].set_xlabel("边到图像边界的距离 d（px）")
    ax[2].set_ylabel("定位误差的绝对值（px）")
    ax[2].set_title("sigma=2 平滑后求梯度：靠近边界的边")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    S.save(fig, CH, "fig04_边界填充")


# ---------------------------------------------------------------- 图 4：几种 3x3 导数核
def fig_operators():
    thetas = [0, 5, 10, 15, 20, 22.5, 25, 30, 35, 40, 45]
    offs = np.linspace(-0.45, 0.45, 7)

    def bias(kx, theta, blur):
        ky, out = kx.T.copy(), []
        for off in offs:
            img, d = oriented(theta, off, blur=blur)
            gx, gy = corr(img, kx)[8:-8, 8:-8], corr(img, ky)[8:-8, 8:-8]
            on = np.abs(d[8:-8, 8:-8]) < 0.5
            out.append(math.degrees(math.atan2(float(gy[on].sum()), float(gx[on].sum()))) - theta)
        return float(np.mean(out))

    curves = {name: [bias(k, t, BLUR) for t in thetas] for name, k in OPS.items()}
    by_blur = {name: {str(b): r3(bias(k, 22.5, b), 2) for b in (0.5, 0.7, 1.5)} for name, k in OPS.items()}

    rng = np.random.default_rng(3)
    unit_noise = rng.normal(size=(400, 400)).astype(np.float32)
    snr = {}
    for name, k in {**OPS, "Laplacian（4 邻域）": LAP4}.items():
        sig = float(np.mean([np.abs(corr(edge_img(8, 64, 31.5 + f, lo=0.0, hi=1.0), k)[4]).max() for f in np.linspace(-0.5, 0.5, 11)]))
        gain = float(corr(unit_noise, k)[2:-2, 2:-2].std())
        snr[name] = {"单位对比度的边上响应的峰值": r3(sig), "单位噪声经过核之后的标准差": r3(gain), "理论值（核元素平方和开根号）": r3(np.sqrt((k ** 2).sum())),
                     "信噪比": r3(sig / gain)}
    for name in snr:
        snr[name]["相对中心差分的倍数"] = r3(snr[name]["信噪比"] / snr["中心差分"]["信噪比"], 2)
    roberts = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], np.float32)     # I(x+1, y+1) - I(x, y)，把 2x2 的核嵌进 3x3
    fr = np.linspace(-0.5, 0.5, 11)
    rob_sig = float(np.mean([np.abs(corr(edge_img(8, 64, 31.5 + f, lo=0.0, hi=1.0), roberts)[4]).max() for f in fr]))
    rob_gain = float(corr(unit_noise, roberts)[2:-2, 2:-2].std())
    rob_pos = float(np.mean([peak(corr(edge_img(8, 64, 32 + f), roberts), 20, 45).mean() - (32 + f) for f in fr]))
    imp = np.zeros((7, 7), np.float32)
    imp[3, 3] = 1
    yy, xx = np.mgrid[0:15, 0:15].astype(np.float32)
    quad, ramp1 = xx * xx + yy * yy, np.tile(np.arange(16, dtype=np.float32), (16, 1))
    lap = {f"ksize={k}": {"实际用的核": cv2.Laplacian(imp, cv2.CV_32F, ksize=k)[2:5, 2:5].astype(int).tolist(),
                          "在 x*x + y*y 上的响应（真实的二阶导之和是 4）": r3(cv2.Laplacian(quad, cv2.CV_32F, ksize=k)[7, 7], 1)} for k in (1, 3)}
    lap["不传 ksize"] = {"实际用的核": cv2.Laplacian(imp, cv2.CV_32F)[2:5, 2:5].astype(int).tolist()}
    ideal = {name: r3(float(np.abs(corr(np.tile((np.arange(16) >= 8).astype(np.float32), (8, 1)), k)[4]).max()) / math.sqrt(float((k ** 2).sum())) /
                      (1 / math.sqrt(2)), 2) for name, k in OPS.items()}

    noises = [0, 2, 5, 10]
    per_pixel = {name: [] for name in OPS}
    whole = {name: {} for name in OPS}
    for sn in noises:
        acc = {name: [] for name in OPS}
        agg = {name: [] for name in OPS}
        for _ in range(30):
            img, d = oriented(22.5, rng.uniform(-0.5, 0.5))
            noisy = img + rng.normal(scale=sn, size=img.shape) if sn else img
            on = np.abs(d[8:-8, 8:-8]) < 0.5
            for name, kx in OPS.items():
                gx, gy = corr(noisy, kx)[8:-8, 8:-8], corr(noisy, kx.T.copy())[8:-8, 8:-8]
                acc[name].append(np.degrees(np.arctan2(gy[on], gx[on])) - 22.5)
                agg[name].append(math.degrees(math.atan2(float(gy[on].sum()), float(gx[on].sum()))) - 22.5)
        for name in OPS:
            per_pixel[name].append(rms(np.concatenate(acc[name])))
            whole[name][str(sn)] = [r3(np.mean(agg[name]), 2), r3(np.std(agg[name]), 2)]
    NUM["几种导数核"] = {
        "说明": f"65x65 的图，一条法线方向为 theta 的直边，对比度 100，边缘模糊 {BLUR} px。方向偏差：把边中心线上（到边的距离小于 0.5 px）所有像素的 gx、gy 分别求和，"
              "再算方向减真值，对 7 个亚像素位置取平均，无噪声",
        "中心一行（列）的权重是两侧的几倍": {"中心差分": "只有中心一行", "Prewitt": 1, "Sobel": 2, "Scharr": r3(10 / 3, 2)},
        "theta（度）": thetas,
        "方向偏差（度）": {n: [r3(v, 2) for v in vals] for n, vals in curves.items()},
        "方向偏差绝对值的最大值（度）": {n: r3(max(abs(v) for v in vals), 2) for n, vals in curves.items()},
        "theta=22.5 度时，不同边缘模糊（px）下的方向偏差（度）": by_blur,
        "抗噪声": {"说明": "沿坐标轴方向的边（边缘模糊 0.7 px，对 11 个亚像素位置取平均）；噪声增益在 400x400 的单位方差白噪声上实测", "各核": snr,
                "理想阶跃边上相对中心差分的信噪比倍数": ideal},
        "逐像素的方向误差": {"说明": "theta=22.5 度，对比度 100，每个噪声水平 30 张图，统计边中心线上每个像素的方向误差的均方根（度）", "噪声标准差": noises,
                     "均方根（度）": {n: [r3(v, 2) for v in vals] for n, vals in per_pixel.items()},
                     "把边中心线上约 50 个像素的 gx、gy 分别求和后再算方向：30 张图的（平均偏差，标准差），按噪声标准差": whole},
        "Roberts（一个对角差分，嵌进 3x3 的核里）": {
            "说明": "核只有两个非零元素，算的是 I(x+1, y+1) - I(x, y)。竖直边、边缘模糊 0.7 px、11 个亚像素位置取平均",
            "信噪比": r3(rob_sig / rob_gain), "相对中心差分的倍数": r3(rob_sig / rob_gain / snr["中心差分"]["信噪比"], 2),
            "相对 Sobel 的倍数": r3(rob_sig / rob_gain / snr["Sobel"]["信噪比"], 2), "响应峰的位置减真实位置（px）": r3(rob_pos, 2)},
        "cv2.Laplacian 的 ksize": lap,
        "响应的尺度": {"说明": "亮度每列加 1 的斜坡图上各函数的输出，除以它才是每像素的灰度变化", "cv2.Sobel(ksize=3)": r3(cv2.Sobel(ramp1, cv2.CV_32F, 1, 0, ksize=3)[8, 8], 1),
                  "cv2.Scharr": r3(cv2.Scharr(ramp1, cv2.CV_32F, 1, 0)[8, 8], 1), "参考实现配 Sobel 核": r3(corr(ramp1, SOBEL_X)[8, 8], 1)},
    }

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9), layout="constrained")
    style = {"中心差分": (S.MUTED, "D"), "Prewitt": (S.ORANGE, "s"), "Sobel": (S.BLUE, "o"), "Scharr": (S.GREEN, "^")}
    for name, vals in curves.items():
        ax[0].plot(thetas, vals, style[name][1] + "-", color=style[name][0], lw=1.8, ms=4, label=name)
    ax[0].axhline(0, color=S.MUTED, lw=1, ls="--")
    ax[0].set_xlabel("边的法线方向 theta（度）")
    ax[0].set_ylabel("测得方向 - 真实方向（度）")
    ax[0].set_title("斜边的方向偏差：Scharr 最小")
    ax[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, columnspacing=1.0, handlelength=1.4)
    S.clean(ax[0])
    names = ["Prewitt", "Sobel", "Scharr", "中心差分", "Laplacian（4 邻域）"]
    vals = [snr[n]["相对中心差分的倍数"] for n in names]
    bars = ax[1].bar(range(5), vals, color=[S.ORANGE, S.BLUE, S.GREEN, S.MUTED, S.RED], alpha=0.85, width=0.55)
    for b, v in zip(bars, vals):
        ax[1].text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9.5)
    ax[1].set_xticks(range(5), ["Prewitt", "Sobel", "Scharr", "中心差分", "Laplacian"], fontsize=9)
    ax[1].set_ylabel("信噪比（中心差分为 1）")
    ax[1].set_title("抗噪声：三个 3x3 核的排序反过来")
    S.clean(ax[1])
    ax[1].grid(axis="x", visible=False)
    for name in ("中心差分", "Prewitt", "Sobel", "Scharr"):
        ax[2].plot(noises, per_pixel[name], style[name][1] + "-", color=style[name][0], lw=1.8, ms=4, label=name)
    ax[2].set_xlabel("噪声标准差（边的对比度 100）")
    ax[2].set_ylabel("逐像素方向误差的均方根（度）")
    ax[2].set_title("噪声到 5 以上，Scharr 逐像素不再更准")
    ax[2].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, columnspacing=1.0, handlelength=1.4)
    S.clean(ax[2])
    S.save(fig, CH, "fig03_几种导数核")


# ---------------------------------------------------------------- 图 5：平滑尺度与定位
def radius_bias(radius, sigma, n=96, cx=47.3, cy=48.6):
    """暗的圆孔：沿 90 个方向取梯度幅值的径向剖面，找峰（抛物线取小数），返回平均半径减真值。"""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    img = camera(HI + (LO - HI) * (1 - phi((np.hypot(x - cx, y - cy) - radius) / BLUR)))
    sm = smooth2d(img, sigma)
    mag = np.hypot(corr(sm, SOBEL_X), corr(sm, SOBEL_Y)).astype(np.float64)
    rs = np.arange(max(radius - 9.0, 0.5), radius + 9.0, 0.1)
    out = []
    for a in np.linspace(0, 2 * np.pi, 90, endpoint=False):
        xs, ys = cx + rs * np.cos(a), cy + rs * np.sin(a)
        x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
        fx, fy = xs - x0, ys - y0
        v = mag[y0, x0] * (1 - fx) * (1 - fy) + mag[y0, x0 + 1] * fx * (1 - fy) + mag[y0 + 1, x0] * (1 - fx) * fy + mag[y0 + 1, x0 + 1] * fx * fy
        i = int(np.clip(v.argmax(), 1, len(v) - 2))
        den = v[i - 1] - 2 * v[i] + v[i + 1]
        out.append(rs[i] + (0.05 * (v[i - 1] - v[i + 1]) / den if abs(den) > 1e-12 else 0.0))
    return float(np.mean(out)) - radius


def fig_sigma():
    sigmas = [0, 0.5, 1, 1.5, 2, 3, 4]
    noises = [5, 10, 20]
    h, w, trials = 48, 96, 60
    phases = np.linspace(-0.5, 0.5, 41)
    iso = {}
    for sn in [0] + noises:
        rng = np.random.default_rng(10)
        e2, e1, avg = ({s: [] for s in sigmas} for _ in range(3))
        for t in range(trials if sn else len(phases)):
            x0 = 48 + (rng.uniform(-0.5, 0.5) if sn else phases[t])
            img = camera(edge_img(h, w, x0), rng, sn)
            c = int(round(x0))
            for s in sigmas:
                a = peak(corr(smooth2d(img, s), SOBEL_X), c - 12, c + 13)[8:-8] - x0
                b = peak(corr(smooth_x(img, s), DIFF_X), c - 12, c + 13)[8:-8] - x0
                e2[s].append(a)
                e1[s].append(b)
                avg[s].append(a.mean())
        iso[sn] = {"二维": [np.concatenate(e2[s]) for s in sigmas], "一维": [np.concatenate(e1[s]) for s in sigmas], "行平均": [np.array(avg[s]) for s in sigmas]}

    widths = [3, 5, 8, 12]
    w_bias, w_std = {}, {}
    for wd in widths:
        rng = np.random.default_rng(11)
        w_bias[wd], w_std[wd] = [], []
        for s in sigmas:
            clean, noisy = [], []
            for _ in range(40):
                xc = 40 + rng.uniform(-0.5, 0.5)
                c = int(round(xc))
                for store, img in ((clean, camera(line_img(32, 80, xc, wd))), (noisy, camera(line_img(32, 80, xc, wd), rng, 10.0))):
                    g = corr(smooth2d(img, s), SOBEL_X)
                    store.append((peak(g, c, c + 21, -1.0) - peak(g, c - 20, c + 1))[6:-6] - wd)
            w_bias[wd].append(float(np.concatenate(clean).mean()))
            w_std[wd].append(float(np.concatenate(noisy).std()))

    radii, rsig = [6, 10, 20], [0, 1, 1.5, 2, 3]
    r_bias = {r: [radius_bias(r, s) for s in rsig] for r in radii}
    r_theory = {r: [-(s * s + BLUR ** 2) / (2 * r) for s in rsig] for r in radii}
    NUM["平滑尺度与定位"] = {
        "孤立直边": {
            "说明": f"{h}x{w} 的图，一条竖直边（对比度 {HI - LO:g}，边缘模糊 {BLUR} px，量化成 uint8），真实位置在 48 附近随机取亚像素；每个噪声水平 {trials} 张图，每张取中间 {h - 16} 行。"
                  "二维：二维高斯平滑加 Sobel x；一维：只沿行方向做一维高斯加中心差分。在真值左右 12 px 内找梯度峰并用三点抛物线取小数",
            "高斯 sigma": sigmas,
            "逐行定位误差的均方根（px），二维": {str(sn): [r3(rms(e), 4) for e in iso[sn]["二维"]] for sn in noises},
            "逐行定位误差的均方根（px），一维": {str(sn): [r3(rms(e), 4) for e in iso[sn]["一维"]] for sn in noises},
            "逐行误差超过 1 px 的比例，二维": {str(sn): [r3((np.abs(e) > 1).mean(), 4) for e in iso[sn]["二维"]] for sn in noises},
            "逐行误差超过 1 px 的比例，一维": {str(sn): [r3((np.abs(e) > 1).mean(), 4) for e in iso[sn]["一维"]] for sn in noises},
            f"{h - 16} 行取平均后的均方根（px），二维": {str(sn): [r3(rms(e), 4) for e in iso[sn]["行平均"]] for sn in noises},
            "无噪声时（41 个亚像素位置）误差绝对值的最大值（px），二维": [r3(np.abs(e).max(), 4) for e in iso[0]["二维"]],
            "无噪声时误差绝对值的最大值（px），一维": [r3(np.abs(e).max(), 4) for e in iso[0]["一维"]],
        },
        "线宽": {
            "说明": "32x80 的图，一条竖直亮线，二维高斯平滑加 Sobel x，正负峰之差为线宽；每组 40 张图、每张取中间 20 行。偏差在无噪声的图上统计，标准差在噪声 10 的图上统计",
            "高斯 sigma": sigmas,
            "线宽的平均偏差（px）": {str(wd): [r3(v) for v in vals] for wd, vals in w_bias.items()},
            "噪声 10 时线宽的标准差（px）": {str(wd): [r3(v) for v in vals] for wd, vals in w_std.items()},
        },
        "圆孔半径": {
            "说明": f"96x96 的图，亮背景上一个暗的圆孔（对比度 {HI - LO:g}，边缘模糊 {BLUR} px，无噪声），沿 90 个方向取梯度幅值的径向剖面找峰，取平均半径减真值。"
                  "近似公式是 -(sigma^2 + 边缘模糊^2) / (2R)。这种测法本身有约 0.05 px 的误差",
            "高斯 sigma": rsig,
            "半径偏差（px）": {str(r): [r3(v) for v in vals] for r, vals in r_bias.items()},
            "近似公式给出的偏差（px）": {str(r): [r3(v) for v in vals] for r, vals in r_theory.items()},
        },
    }

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0), layout="constrained")
    for sn, color in zip(noises, (S.GREEN, S.BLUE, S.RED)):
        ax[0].plot(sigmas, [rms(e) for e in iso[sn]["二维"]], "o-", color=color, lw=1.8, ms=4, label=f"噪声 {sn}，二维平滑")
        ax[0].plot(sigmas, [rms(e) for e in iso[sn]["一维"]], "s--", color=color, lw=1.4, ms=3.5, label=f"噪声 {sn}，只在行内平滑")
    ax[0].set_yscale("log")
    ax[0].set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2], ["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2"])
    ax[0].minorticks_off()
    ax[0].set_xlabel("高斯平滑的 sigma（px）")
    ax[0].set_ylabel("逐行定位误差的均方根（px）")
    ax[0].set_title("孤立直边：二维平滑越重，定位越稳")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[0])
    for wd, color, marker in zip(widths, (S.RED, S.ORANGE, S.BLUE, S.GREEN), "osD^"):
        ax[1].plot(sigmas, w_bias[wd], marker + "-", color=color, lw=1.8, ms=4, label=f"线宽 {wd} px")
    ax[1].axhline(0, color=S.MUTED, lw=1, ls="--")
    ax[1].set_xlabel("高斯平滑的 sigma（px）")
    ax[1].set_ylabel("测得线宽 - 真实线宽（px）")
    ax[1].set_title("两条边靠得近：线宽越量越宽")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[1])
    for r, color, marker in zip(radii, (S.RED, S.BLUE, S.GREEN), "os^"):
        ax[2].plot(rsig, r_bias[r], marker + "-", color=color, lw=1.8, ms=4, label=f"半径 {r} px，实测")
        ax[2].plot(rsig, r_theory[r], ":", color=color, lw=1.4, label=f"半径 {r} px，近似公式")
    ax[2].axhline(0, color=S.MUTED, lw=1, ls="--")
    ax[2].set_xlabel("高斯平滑的 sigma（px）")
    ax[2].set_ylabel("测得半径 - 真实半径（px）")
    ax[2].set_title("弯曲的边：圆孔的半径越量越小")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    S.save(fig, CH, "fig05_平滑尺度与定位")


# ---------------------------------------------------------------- 图 6：Canny 的三个细节
def make_scene(noise_std=10, seed=2026):
    """和本章 Notebook 里的 make_scene 相同：暗背景、亮矩形、一条 2 px 宽的斜线。"""
    image = np.full((128, 160), 35, dtype=np.float32)
    image[28:100, 35:125] = 170
    cv2.line(image, (12, 112), (148, 18), 235, 2)
    rng = np.random.default_rng(seed)
    image += rng.normal(0, noise_std, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def notebook_canny(noise_std=10, sigma=1.0, low=50, high=120, blur=True, aperture=3):
    """和 Notebook 的 show_edges 相同的处理，返回边缘图、边缘像素数、8 连通的段数。"""
    image = make_scene(noise_std)
    k = max(3, int(np.ceil(6 * sigma)) | 1)
    sm = cv2.GaussianBlur(image, (k, k), sigma) if blur else image
    e = cv2.Canny(sm, low, max(high, low + 1), apertureSize=aperture, L2gradient=True)
    n, _ = cv2.connectedComponents(e, connectivity=8)
    return e, int(np.count_nonzero(e)), int(n - 1)


def fading(seed, noise=12.0, h=96, w=240):
    """一条水平边，对比度从左到右由 60 线性降到 6。"""
    rng = np.random.default_rng(seed)
    con = np.linspace(60, 6, w)
    y = np.arange(h)[:, None]
    return camera(90.0 + con[None, :] * phi((y - 47.5) / BLUR), rng, noise)


def fig_canny():
    cv2.setNumThreads(1)
    e_raw, n_raw, seg_raw = notebook_canny(20, blur=False)
    e_blur, n_blur, seg_blur = notebook_canny(20)
    _, n_clean, seg_clean = notebook_canny(0)
    step = {}
    for c in (25, 26):
        im = np.full((40, 40), 50, np.uint8)
        im[:, 20:] = 50 + c
        step[str(c)] = {"Sobel 响应的峰值": r3(cv2.Sobel(im, cv2.CV_32F, 1, 0).max(), 1), "Canny(100, 100) 的边缘像素数": int(np.count_nonzero(cv2.Canny(im, 100, 100, L2gradient=True)))}
    weak = np.full((40, 60), 50, np.uint8)
    weak[20:, :30], weak[20:, 30:] = 90, 65
    low_test = {str(lo): int((cv2.Canny(weak, lo, 100, L2gradient=True)[:, 34:56] > 0).any(axis=0).sum()) for lo in (59, 60)}

    rules = {"只用高阈值 90": (90, 90), "只用低阈值 30": (30, 30), "滞后 30/90": (30, 90), "滞后 60/90": (60, 90)}
    hyst, example = {}, {}
    for name, (lo, hi) in rules.items():
        found, false = [], []
        for seed in range(40):
            e = cv2.Canny(cv2.GaussianBlur(fading(seed), (0, 0), 1.0), lo, hi, L2gradient=True)
            found.append((e[44:52, :] > 0).any(axis=0)[8:-8].mean())
            out = e.copy()
            out[38:58, :] = 0
            false.append(np.count_nonzero(out[4:-4, 4:-4]))
            if seed == 0:
                example[name] = e
        hyst[name] = {"检出的边长占比": r3(np.mean(found)), "每张图上边以外的假边像素": r3(np.mean(false), 1)}

    thetas = [0, 15, 30, 45]
    offs = np.linspace(-0.45, 0.45, 7)

    def min_contrast(theta, l2, thr=200):
        out = []
        for off in offs:
            for c in range(5, 150):
                im = camera(oriented(theta, off, n=96, contrast=c)[0])
                if np.count_nonzero(cv2.Canny(im, thr, thr, L2gradient=l2)[20:-20, 20:-20]) >= 40:
                    out.append(c)
                    break
        return float(np.mean(out))

    need = {"L1（默认）": [min_contrast(t, False) for t in thetas], "L2": [min_contrast(t, True) for t in thetas]}
    ratio = []
    for t in thetas:
        l1, l2 = [], []
        for off in offs:
            im = oriented(t, off, n=96)[0]
            gx, gy = corr(im, SOBEL_X)[20:-20, 20:-20], corr(im, SOBEL_Y)[20:-20, 20:-20]
            l1.append(float((np.abs(gx) + np.abs(gy)).max()))
            l2.append(float(np.hypot(gx, gy).max()))
        ratio.append(float(np.mean(l1) / np.mean(l2)))

    rng = np.random.default_rng(30)
    grid = np.full((1024, 1024), 60, np.uint8)
    for y0 in range(16, 1024, 64):
        for x0 in range(16, 1024, 64):
            grid[y0:y0 + 32, x0:x0 + 32] = 180
    tex = cv2.normalize(cv2.GaussianBlur(rng.uniform(0, 255, (1024, 1024)).astype(np.float32), (0, 0), 2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    def measure_timing():
        """L1 和 L2 两种幅值的耗时。会随机器忙闲波动，由 S.pinned 钉住，设置 CH02_RETIME=1 才会真的执行。"""
        timing = {}
        for name, im in (("方块阵列（只有横边和竖边）", grid), ("随机纹理", tex)):
            t1, t2 = [], []
            for _ in range(40):
                t = time.perf_counter()
                a = cv2.Canny(im, 60, 150, L2gradient=False)
                t1.append(time.perf_counter() - t)
                t = time.perf_counter()
                b = cv2.Canny(im, 60, 150, L2gradient=True)
                t2.append(time.perf_counter() - t)
            timing[name] = {"L1 耗时（ms）": float(f"{1e3 * min(t1):.2g}"), "L2 耗时（ms）": float(f"{1e3 * min(t2):.2g}"),
                            "L1 的边缘像素数": int(np.count_nonzero(a)), "L2 的边缘像素数": int(np.count_nonzero(b))}
        return {"说明": "1024x1024 的 uint8 图，阈值 60/150，OpenCV 单线程，两种设置交替各跑 40 次取最小值；每次运行会有波动", **timing}

    ramp = np.tile(np.arange(16, dtype=np.float32), (16, 1))
    NUM["Canny 的细节"] = {
        "cv2.Canny 不做高斯平滑": {
            "说明": "Notebook 的合成场景（128x160，种子 2026），阈值 50/120，L2gradient=True。直接调用是把带噪声的图直接传给 cv2.Canny；平滑是先做 7x7、sigma=1 的 GaussianBlur",
            "噪声 20，直接调用（边缘像素数，8 连通段数）": [n_raw, seg_raw], "噪声 20，先平滑（边缘像素数，段数）": [n_blur, seg_blur],
            "无噪声，先平滑（边缘像素数，段数）": [n_clean, seg_clean],
            "锐利阶跃边（不同对比度）": step,
            "低阈值测试：弱边的幅值恰好是 60，被保留的列数（共 22 列）": low_test,
        },
        "滞后阈值": {"说明": "96x240 的图，一条水平边，对比度从左到右由 60 线性降到 6，噪声标准差 12；先做 sigma=1 的高斯平滑再调用 cv2.Canny（L2）。40 张图取平均",
                 "各种阈值": hyst},
        "L1 与 L2 幅值": {
            "说明": f"96x96 的图，法线方向为 theta 的直边（边缘模糊 {BLUR} px），7 个亚像素位置取平均。刚好能检出的对比度：高低阈值都取 200，对比度从 5 逐级加 1，直到中心区域出现不少于 40 个边缘像素",
            "theta（度）": thetas, "刚好能检出的对比度": {k: [r3(v, 1) for v in vals] for k, vals in need.items()},
            "L1 幅值峰值与 L2 幅值峰值之比": [r3(v) for v in ratio],
            "不传 L2gradient 时的输出与 L2gradient=False 完全相同（随机纹理图）": bool(np.array_equal(cv2.Canny(tex, 60, 150), cv2.Canny(tex, 60, 150, L2gradient=False))),
        },
        "apertureSize": {"说明": "亮度每列加 1 的斜坡图上，cv2.Sobel 不同 ksize 的响应；以及 Notebook 默认场景（噪声 10、sigma=1、阈值 50/120）换 apertureSize 后的边缘像素数",
                         "斜坡上的响应": {str(k): r3(cv2.Sobel(ramp, cv2.CV_32F, 1, 0, ksize=k)[8, 8], 1) for k in (3, 5, 7)},
                         "边缘像素数": {str(k): notebook_canny(aperture=k)[1] for k in (3, 5)}},
    }
    NUM["Canny 的耗时"] = S.pinned(CH, "Canny 的耗时", measure_timing, "CH02_RETIME")

    fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.25, 1.1]}, layout="constrained")
    for a, e, title in ((ax[0], e_raw, f"噪声 20，直接 cv2.Canny：{n_raw} 个边缘点"), (ax[1], e_blur, f"先做 sigma=1 的高斯平滑：{n_blur} 个")):
        a.imshow(e, cmap="gray_r")
        a.set_title(title, fontsize=10.5)
        S.image_axes(a)
    shown = ["只用高阈值 90", "只用低阈值 30", "滞后 30/90"]
    gap = np.zeros((8, 240), np.uint8)
    ax[2].imshow(np.vstack([example[shown[0]], gap, example[shown[1]], gap, example[shown[2]]]), cmap="gray_r", aspect="auto")
    for i, name in enumerate(shown):
        ax[2].text(4, i * 104 + 4, f"{name}：检出 {hyst[name]['检出的边长占比']:.0%}，假边 {hyst[name]['每张图上边以外的假边像素']:.0f} 个", fontsize=8.5, va="top",
                   bbox={"fc": "white", "ec": "none", "alpha": 0.9, "pad": 1.5})
        if i:
            ax[2].axhline(i * 104 - 4, color="#b9bec7", lw=1)
    ax[2].set_title("对比度从左到右渐弱的一条边", fontsize=10.5)
    S.image_axes(ax[2])
    ax[3].plot(thetas, need["L1（默认）"], "s-", color=S.ORANGE, lw=1.8, ms=5, label="L1：|Gx| + |Gy|（默认）")
    ax[3].plot(thetas, need["L2"], "o-", color=S.BLUE, lw=1.8, ms=5, label="L2：L2gradient=True")
    ax[3].set_xticks(thetas)
    ax[3].set_ylim(0, 75)
    ax[3].set_xlabel("边的法线方向 theta（度）")
    ax[3].set_ylabel("刚好能检出的对比度（阈值 200）")
    ax[3].set_title("默认的 L1：斜边更容易过阈值", fontsize=10.5)
    ax[3].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax[3])
    S.save(fig, CH, "fig06_Canny的细节")


# ---------------------------------------------------------------- 只出数字的几组实验
def numbers_separable():
    NUM["可分离卷积的耗时"] = S.pinned(CH, "可分离卷积的耗时", measure_separable, "CH02_RETIME")


def measure_separable():
    """可分离卷积和逐像素循环的耗时。会随机器忙闲波动，由 S.pinned 钉住，设置 CH02_RETIME=1 才会真的执行。"""
    cv2.setNumThreads(1)
    rng = np.random.default_rng(20)
    img = rng.uniform(0, 255, (512, 512)).astype(np.float32)

    def best(fn, n):
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        return 1e3 * min(ts)

    def ms(v):
        return float(f"{v:.2g}")

    rows = {}
    for ks in (3, 5, 7, 9, 15, 31):
        g = cv2.getGaussianKernel(ks, ks / 6).astype(np.float32)
        k2 = (g @ g.T).astype(np.float32)
        t2, t1 = best(lambda: corr(img, k2), 7), best(lambda: corr(corr(img, g.T), g), 7)
        c2, c1 = best(lambda: cv2.filter2D(img, cv2.CV_32F, k2), 60), best(lambda: cv2.sepFilter2D(img, cv2.CV_32F, g, g), 60)
        rows[str(ks)] = {"乘法次数之比 k*k / 2k": r3(ks / 2, 1), "参考实现，二维核（ms）": ms(t2), "参考实现，拆成两次一维（ms）": ms(t1), "参考实现的加速倍数": r3(t2 / t1, 1),
                         "filter2D（ms）": ms(c2), "sepFilter2D（ms）": ms(c1), "OpenCV 的加速倍数": r3(c2 / c1, 1),
                         "二维与两次一维结果的最大差": r3(np.abs(corr(img, k2) - corr(corr(img, g.T), g)).max(), 4)}

    def naive(image, kernel):
        kh, kw = kernel.shape
        p = np.pad(image, ((kh // 2,) * 2, (kw // 2,) * 2), mode="reflect")
        out = np.zeros(image.shape, np.float32)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                out[y, x] = (p[y:y + kh, x:x + kw] * kernel).sum()
        return out

    small = img[:64, :64].copy()
    tn, tv = best(lambda: naive(small, SOBEL_X), 7), best(lambda: corr(small, SOBEL_X), 60)
    g5 = cv2.getGaussianKernel(5, 1.0)
    ranks = {}
    for name, k in (("Sobel x", SOBEL_X), ("高斯 5x5", g5 @ g5.T), ("均值 3x3", np.ones((3, 3)) / 9), ("Laplacian（4 邻域）", LAP4), ("锐化核", np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float))):
        sv = np.linalg.svd(np.asarray(k, float), compute_uv=False)
        ranks[name] = int((sv > 1e-9 * sv[0]).sum())
    return {
        "说明": "512x512 的 float32 图，k x k 的高斯核（sigma = k/6）。参考实现各跑 7 次、OpenCV 各跑 60 次取最小值，OpenCV 设为单线程。"
              "在一台 Apple 芯片的 Mac 上实测，耗时每次运行会有波动（机器忙的时候能差几成），倍数只看量级",
        "按核的边长 k": rows,
        "逐像素双重循环": {"说明": "64x64 的图、3x3 的核，每个输出像素取一次窗口再求和的写法，和参考实现（按核元素循环、整图相加）比", "逐像素循环（ms）": ms(tn), "参考实现（ms）": ms(tv),
                    "慢多少倍": int(round(tn / tv, -1)), "两种写法结果的最大差": r3(np.abs(naive(small, SOBEL_X) - corr(small, SOBEL_X)).max(), 5)},
        "核的秩（秩为 1 才能拆成一行乘一列）": ranks,
    }


def numbers_notebook():
    cases = {"默认（噪声 10，sigma 1，阈值 50/120）": {}, "噪声 40，sigma 0.5": {"noise_std": 40, "sigma": 0.5}, "噪声 40，sigma 3": {"noise_std": 40, "sigma": 3.0},
             "sigma 2": {"sigma": 2.0}, "sigma 3": {"sigma": 3.0}, "低阈值 10": {"low": 10}, "低阈值 110": {"low": 110}, "低阈值 119": {"low": 119},
             "噪声 40，sigma 1.5，低阈值 10": {"noise_std": 40, "sigma": 1.5, "low": 10}, "噪声 40，sigma 1.5，低阈值 70": {"noise_std": 40, "sigma": 1.5, "low": 70},
             "噪声 40，sigma 1.5，低阈值 110": {"noise_std": 40, "sigma": 1.5, "low": 110}}
    counts = {name: list(notebook_canny(**kw)[1:]) for name, kw in cases.items()}
    inside = {str(s): int(np.count_nonzero(notebook_canny(sigma=s)[0][34:94, 41:119])) for s in (1.0, 1.5, 2.0, 3.0)}
    image = make_scene(10)
    sm = cv2.GaussianBlur(image, (7, 7), 1.0)
    mag = cv2.magnitude(cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3))
    trunc = {}
    for ks, sg in ((3, 0.8), (5, 1.0), (7, 2.0), (13, 2.0)):
        g = cv2.getGaussianKernel(ks, sg).ravel()
        trunc[f"ksize={ks}, sigma={sg}"] = {"核的实际标准差": r3(math.sqrt(float((g * (np.arange(ks) - ks // 2) ** 2).sum())), 2), "平坦区噪声剩下的比例": f"{100 * float((g ** 2).sum()):.0f}%"}
    band = {}
    for s in (0, 1, 2):
        g = corr(smooth2d(edge_img(16, 64, 32.3), s), SOBEL_X)[8]
        band[str(s)] = int((g > 0.5 * g.max()).sum())
    NUM["Notebook 场景与 2.5 节"] = {
        "说明": "和 Notebook 的 show_edges 完全相同的处理（128x160，种子 2026，GaussianBlur 的核边长取 6 sigma 向上取奇数，L2gradient=True），数值是（边缘像素数，8 连通段数）",
        "各组参数": counts,
        "矩形内部（第 34 到 93 行、第 41 到 118 列，只有斜线穿过）的边缘像素数，按 sigma": inside,
        "默认参数下梯度幅值超过低阈值 50 的像素数": int((mag > 50).sum()),
        "高斯核被 ksize 截断": {"说明": "cv2.getGaussianKernel 给出的核。平坦区噪声剩下的比例指二维平滑后噪声标准差与原来之比（一维核元素的平方和）", "各组": trunc},
        "梯度亮带的宽度": {"说明": "一条孤立的竖直边，高斯平滑后 Sobel x 响应超过峰值一半的像素个数", "按 sigma": band},
    }


def numbers_noise():
    rng = np.random.default_rng(40)
    field = rng.normal(scale=10.0, size=(600, 600))
    gain, exceed = {}, {}
    for s in (0, 1, 2):
        gain[str(s)] = r3(float(corr(smooth2d(field, s), SOBEL_X)[12:-12, 12:-12].std()) / 10.0)
    mag = np.hypot(corr(field, SOBEL_X), corr(field, SOBEL_Y))[2:-2, 2:-2]
    for thr in (60, 100, 150):
        exceed[str(thr)] = {"实测": float(f"{(mag > thr).mean():.3g}"), "公式": float(f"{math.exp(-thr * thr / (24 * 100.0)):.3g}")}
    NUM["噪声与阈值"] = {
        "说明": "600x600 的平场图加标准差 10 的高斯噪声（种子 40），没有任何边。Sobel 的一个分量是 6 个权重的加权和，权重平方和是 12",
        "Sobel 一个分量的噪声标准差是像素噪声的多少倍（理论）": r3(math.sqrt(12)),
        "先做高斯平滑再做 Sobel，这个倍数的实测值（按 sigma）": gain,
        "梯度幅值超过阈值的像素比例（不平滑，噪声 10）": exceed,
        "单像素误报概率 1e-6 对应的阈值是像素噪声的多少倍（不平滑）": r3(math.sqrt(24 * math.log(1e6)), 1),
    }


TEMPLATE = '''import numpy as np


def correlate2d(image, kernel):
    img = np.asarray(image, dtype=np.float32)
    k = np.asarray(kernel, dtype=np.float32){flip}
    ry, rx = k.shape[0] // 2, k.shape[1] // 2
    pad = np.pad(img, ((ry, ry), (rx, rx)), mode="{mode}")
    out = np.zeros_like(img, dtype=np.float32)
    for dy in range(k.shape[0]):
        for dx in range(k.shape[1]):
            out += k[dy, dx] * pad[dy:dy + img.shape[0], dx:dx + img.shape[1]]
    return out.astype(np.float32)
'''


def numbers_homework():
    """把几种常见的错误写法交给本章的验收脚本，看它能不能分辨；再看和 cv2.filter2D 对照能不能分辨。"""
    variants = {"参考写法": (False, "reflect"), "把核翻转了（算成卷积）": (True, "reflect"), "填充用了 symmetric": (False, "symmetric"), "填充用了 edge": (False, "edge"),
                "填充用了 wrap": (False, "wrap"), "填充用了常数 0": (False, "constant")}
    script = S.MAIN / CH / "tests" / "check_progress.py"
    rng = np.random.default_rng(50)
    img = rng.uniform(0, 255, (64, 64)).astype(np.float32)
    k5 = rng.normal(size=(5, 5)).astype(np.float32)
    truth = cv2.filter2D(img, cv2.CV_32F, k5)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, (flip, mode)) in enumerate(variants.items()):
            d = Path(tmp) / f"v{i}" / CH
            d.mkdir(parents=True)
            (d / "2-1_手写相关运算.py").write_text(TEMPLATE.format(flip="[::-1, ::-1]" if flip else "", mode=mode), encoding="utf-8")
            env = dict(os.environ, COURSE_IMPL_DIR=str(d.parent), PYTHONDONTWRITEBYTECODE="1")
            res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=120)
            lines = [line for line in res.stdout.splitlines() if line.startswith("[")]
            status = "无输出" if not lines else ("PASS" if res.returncode == 0 else "FAIL")
            failed = [line[6:].split(" — ")[0].strip() for line in lines if line.startswith("[FAIL")]
            diff = np.abs(corr_pad(img, k5[::-1, ::-1].copy() if flip else k5, mode) - truth)
            out[name] = {"验收脚本的结论": status, "没通过的检查项": failed, "与 cv2.filter2D（默认填充）的最大差": r3(diff.max(), 4), "差大于 0.01 的像素占比": r3((diff > 0.01).mean())}
    NUM["验收脚本能分辨哪些错误"] = {
        "说明": "把每种写法存成作业文件，用 COURSE_IMPL_DIR 交给 tests/check_progress.py（加固后的版本：除恒等核和常量图外，还查冲激响应、手算的 reflect 边界、非方形核与 cv2.filter2D 的对照、整数输入）。"
              "结论按脚本的退出码定，有一项 FAIL 就算 FAIL。对照实验：64x64 随机图（种子 50）和一个 5x5 随机核，与 cv2.filter2D(img, cv2.CV_32F, k) 比较",
        "各种写法": out,
        "自查时建议的容差": {"说明": "灰度 0 到 255 的图、5x5 的核，参考实现与 filter2D 的差来自 float32 的舍入，取比它略大的值", "数值": 0.001},
    }


def main() -> None:
    S.setup()
    for fn in (fig_why, fig_corr_conv, fig_border, fig_operators, fig_sigma, fig_canny, numbers_separable, numbers_notebook, numbers_noise, numbers_homework):
        fn()
        print("完成", fn.__name__, flush=True)
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
