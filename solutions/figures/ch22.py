"""第 22 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch22.py

输出：<主分支>/22_成像与设备仿真/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。

长度单位统一用 λ/NA，这样结论不依赖具体的波长和物镜。
光学部分用标量、傍轴、非相干的模型：像的强度等于物的强度卷积 PSF。
传感器部分调用作业的参考实现 simulate_capture 和 estimate_gain_from_ptc。
simulate_capture 的黑电平参数默认是 0。只有专门研究暗处统计量的几组实验传了黑电平（BLACK），
它们用单独的随机数发生器，所以其余数字和加这个参数之前逐位相同。
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

CH = "22_成像与设备仿真"
mod = S.load_solution(CH, "22-1_成像仿真.py")
NUM: dict = {}
DX = 0.05          # 细网格间距，单位 λ/NA
U = "λ/NA"
U2 = "λ/NA$^2$"     # 上标用 mathtext，中文字体里没有“²”这个字形
_TAU = (np.arange(720) + 0.5) * (2 * np.pi / 720)
BLACK = 64.0        # 研究暗处统计量时用的黑电平（ADU），比这些实验里读出噪声换算成 ADU 的 5 倍还大


def r3(v, n=3):
    return round(float(v), n)


def sci(v, n=2):
    return float(f"{float(v):.{n}g}")


# ---------------------------------------------------------------- 光学工具
def bessel_j(order, x):
    """整数阶贝塞尔函数，用积分表示直接算（环境里没有 scipy）。"""
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    flat = x.ravel()
    out = np.empty(flat.size)
    for i in range(0, flat.size, 4096):
        seg = flat[i:i + 4096, None]
        out[i:i + 4096] = np.cos(order * _TAU[None, :] - seg * np.sin(_TAU)[None, :]).mean(axis=1)
    return out.reshape(x.shape)


def airy(r):
    """艾里斑强度，峰值为 1。r 以 λ/NA 为单位。"""
    v = 2 * np.pi * np.abs(np.atleast_1d(np.asarray(r, dtype=np.float64)))
    out = np.ones_like(v)
    nz = v > 1e-9
    out[nz] = (2 * bessel_j(1, v[nz]) / v[nz]) ** 2
    return out


def airy_encircled(r):
    """半径 r 以内的能量比例：1 - J0² - J1²。"""
    v = 2 * np.pi * np.atleast_1d(np.asarray(r, dtype=np.float64))
    return 1 - bessel_j(0, v) ** 2 - bessel_j(1, v) ** 2


def mtf_diffraction(nu):
    """圆孔径、非相干照明的衍射极限 MTF。nu 以 NA/λ 为单位，截止频率是 2。"""
    x = np.clip(np.abs(np.asarray(nu, dtype=np.float64)) / 2, 0, 1)
    return (2 / np.pi) * (np.arccos(x) - x * np.sqrt(1 - x * x))


def airy_psf(n, dx=DX):
    """对焦的 PSF：解析的艾里斑，中心在 n//2，每格的值是落在这一格里的能量比例。

    总能量归一的艾里斑密度是 π·[2J1(v)/v]²（长度单位 λ/NA）。窗口外的旁瓣能量没有计入，所以和略小于 1；
    做卷积时只卷“相对背景的增量”，背景另加，这样背景亮度不受截断影响。
    """
    c = n // 2
    y, x = np.mgrid[0:n, 0:n]
    return np.pi * airy(np.hypot(y - c, x - c) * dx) * dx * dx


def pupil_psf(w020=0.0, n_out=None, n=1600, dx=DX):
    """由光瞳函数算 PSF（硬边圆孔径，和为 1，中心在正中），可以带离焦。

    w020 是光瞳边缘的离焦波像差，单位是波长。离焦量 z 以 λ/NA² 为单位时 w020 = z / 2（傍轴近似）。
    n=1600 时光瞳半径占 80 个频率格点。n_out 给定时裁出中心的 n_out×n_out。
    """
    f = np.fft.fftfreq(n, d=dx)
    fx, fy = np.meshgrid(f, f)
    rho = np.hypot(fx, fy)
    psf = np.fft.fftshift(np.abs(np.fft.ifft2((rho <= 1) * np.exp(2j * np.pi * w020 * rho ** 2))) ** 2)
    psf /= psf.sum()
    if n_out is None:
        return psf
    a = n // 2 - n_out // 2
    return psf[a:a + n_out, a:a + n_out]


def gauss_psf_fine(sigma_u, n, dx=DX):
    c = n // 2
    y, x = np.mgrid[0:n, 0:n]
    g = np.exp(-((y - c) ** 2 + (x - c) ** 2) * dx * dx / (2 * sigma_u ** 2))
    return g / g.sum()


def fit_gauss_sigma(psf, dx=DX):
    """二维最小二乘意义下最接近 psf 的高斯宽度（单位 λ/NA）。"""
    n = psf.shape[0]
    c = n // 2
    y, x = np.mgrid[0:n, 0:n]
    r2 = ((y - c) ** 2 + (x - c) ** 2) * dx * dx
    best, best_err = None, np.inf
    for s in np.arange(0.150, 0.500, 0.001):
        g = np.exp(-r2 / (2 * s * s))
        err = float(((g / g.sum() - psf) ** 2).sum())
        if err < best_err:
            best, best_err = float(s), err
    return best


def fftconv(img, psf):
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(np.fft.ifftshift(psf))))


def fshift(img, dy, dx):
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.fftfreq(img.shape[1])[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j * np.pi * (fy * dy + fx * dx))))


def disk(d, cy, cx, n, dx=DX, sub=16):
    """直径 d（单位 λ/NA）的圆盘在细网格上的覆盖比例。中心 (cy, cx) 是细网格坐标，可以是小数。"""
    out = np.zeros((n, n))
    h = int(math.ceil(d / dx / 2)) + 2
    y0, x0 = int(round(cy)) - h, int(round(cx)) - h
    m = 2 * h + 1
    t = (np.arange(m * sub) + 0.5) / sub - 0.5
    cov = (np.hypot(y0 + t[:, None] - cy, x0 + t[None, :] - cx) * dx <= d / 2).astype(np.float64)
    out[y0:y0 + m, x0:x0 + m] = cov.reshape(m, sub, m, sub).mean(axis=(1, 3))
    return out


def binning(img, b, oy=0, ox=0):
    """像素积分：细网格按 b×b 求和成一个传感器像素。"""
    h = (img.shape[0] - oy) // b * b
    w = (img.shape[1] - ox) // b * b
    return img[oy:oy + h, ox:ox + w].reshape(h // b, b, w // b, b).sum(axis=(1, 3))


def bisect(fn, lo, hi, it=60):
    flo = fn(lo)
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = fn(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fwhm_of(profile, dx=DX):
    """过峰值的一维剖面的半高全宽，线性插值到亚格点。"""
    c = int(np.argmax(profile))
    half = profile[c] / 2
    i = c
    while profile[i + 1] > half:
        i += 1
    right = i + (profile[i] - half) / (profile[i] - profile[i + 1])
    i = c
    while profile[i - 1] > half:
        i -= 1
    left = i - (profile[i] - half) / (profile[i] - profile[i - 1])
    return (right - left) * dx


FIRST_ZERO = bisect(lambda r: float(bessel_j(1, 2 * np.pi * r)[0]), 0.5, 0.7)
SIG_FOCUS = fit_gauss_sigma(airy_psf(256))


# ---------------------------------------------------------------- 图 1：成像链路逐级
def fig_chain():
    n, b = 384, 6
    p = b * DX
    bg, hi = 0.4, 0.8
    E, read, gain, bits = 900.0, 5.0, 1.0, 10
    yy, xx = np.mgrid[0:n, 0:n]
    scene = np.full((n, n), bg)
    scene[(yy >= 36) & (yy < 156) & (xx >= 36) & (xx < 348) & (((xx - 36) % 24) < 12)] = hi
    diam = [0.2, 0.4, 0.8]
    pix = [(44, 12), (44, 32), (44, 52)]
    for d, (py, px) in zip(diam, pix):
        scene += (hi - bg) * disk(d, py * b + (b - 1) / 2, px * b + (b - 1) / 2, n)
    blurred = bg + fftconv(scene - bg, airy_psf(n))
    sampled = binning(blurred, b) / b ** 2
    mean_e = sampled * E
    rng = np.random.default_rng(1)
    shot = mod.simulate_capture(mean_e, read, gain, bits, rng)

    bg_e = bg * E
    rows = {}
    trials = 4000
    bgcap = mod.simulate_capture(np.full(trials * 16, bg_e), read, gain, bits, rng).astype(np.float64)
    for d, (py, px) in zip(diam, pix):
        fy, fx = py * b + (b - 1) // 2, px * b + (b - 1) // 2
        opt = float(blurred[fy - 8:fy + 10, fx - 8:fx + 10].max())
        cap = mod.simulate_capture(np.full(trials, mean_e[py, px]), read, gain, bits, rng).astype(np.float64)
        rows[f"{d}"] = {
            "物面对比度": 1.0,
            "光学模糊后的峰值对比度": r3(opt / bg - 1),
            "像素采样后的对比度": r3(sampled[py, px] / bg - 1),
            "该像素比背景多出的电子数": r3(mean_e[py, px] - bg_e, 1),
            "单像素信噪比（实测，4000 次）": r3((cap.mean() - bgcap.mean()) / bgcap.std(), 2),
            "单像素信噪比（按公式）": r3((mean_e[py, px] - bg_e) / math.sqrt(bg_e + read ** 2), 2),
        }
    line_obj, line_opt, line_pix = scene[96, 60:300], blurred[96, 60:300], sampled[16, 10:50]
    mich = lambda v: float((v.max() - v.min()) / (v.max() + v.min()))  # noqa: E731
    NUM["成像链路逐级"] = {
        "说明": f"物面 19.2×19.2 {U}，细网格间距 {DX} {U}；背景反射率 {bg}，线条和三个圆形颗粒的反射率 {hi}（比背景亮一倍，即对比度 1）；"
              f"线条周期 1.2 {U}（线宽 0.6）；对焦的艾里斑 PSF；像素 {p:.1f} {U}（64×64 像素），线条边缘正好落在像素边界上；"
              f"反射率 1 对应每像素 {E:.0f} e-；读出噪声 {read:.0f} e-，增益 {gain:.0f} e-/ADU，{bits} 位",
        "背景每像素电子数": r3(bg_e, 1),
        "背景噪声标准差（e-，按公式）": r3(math.sqrt(bg_e + read ** 2), 2),
        "按颗粒直径": rows,
        "线条的对比度 (max-min)/(max+min)": {"物面": r3(mich(line_obj)), "光学模糊后": r3(mich(line_opt)), "像素采样后": r3(mich(line_pix))},
    }

    fig, ax = plt.subplots(1, 4, figsize=(15, 4.1), layout="constrained")
    ax[0].imshow(scene, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("物面：三个颗粒都比背景亮一倍")
    for d, (py, px) in zip(diam, pix):
        ax[0].text(px * b + 3, py * b - 24, f"直径 {d}", ha="center", fontsize=8.5, color="white")
    ax[1].imshow(blurred, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("光学模糊后")
    ax[2].imshow(sampled, cmap="gray", vmin=0, vmax=1)
    ax[2].set_title(f"像素采样后（像素 {p:.1f} {U}）")
    ax[3].imshow(shot, cmap="gray", vmin=0, vmax=E)
    ax[3].set_title("加噪声和量化后的一次拍摄")
    for a, s in ((ax[1], b), (ax[2], 1), (ax[3], 1)):
        for d, (py, px) in zip(diam, pix):
            off = (b - 1) / 2 if s == b else 0
            a.plot(px * s + off, py * s + off, "o", mfc="none", mec=S.ORANGE, ms=17, mew=1.2)
    for a in ax:
        S.image_axes(a)
    S.save(fig, CH, "fig01_成像链路逐级")


# ---------------------------------------------------------------- 图 2：分辨率与小缺陷
def fig_resolution():
    infl = bisect(lambda r: float(airy(r + 1e-3)[0] + airy(r - 1e-3)[0] - 2 * airy(r)[0]), 0.15, 0.35)
    sparrow = 2 * infl
    x = np.linspace(-1.5, 1.5, 1201)
    seps = {"斯派罗极限": sparrow, "瑞利判据": FIRST_ZERO, "间距 1.0": 1.0}
    profs, dips = {}, {}
    for name, s in seps.items():
        prof = airy(x - s / 2) + airy(x + s / 2)
        profs[name] = prof / prof.max()
        dips[name] = float(prof[600] / prof.max())

    n = 512
    c = n // 2
    psf = airy_psf(n)
    gpsf = gauss_psf_fine(SIG_FOCUS, n)
    numeric = pupil_psf(0.0, n_out=n)
    psf_err = float(np.abs(numeric[c] / numeric[c, c] - psf[c] / psf[c, c]).max())
    diams = [0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.61, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    peak_a, peak_g, width = [], [], []
    for d in diams:
        dk = disk(d, c, c, n)
        spot = fftconv(dk, psf)
        peak_a.append(float(spot[c, c]))
        peak_g.append(float(fftconv(dk, gpsf)[c, c]))
        width.append(fwhm_of(spot[c]))
    exact = [float(airy_encircled(d / 2)[0]) for d in diams]
    psf_fwhm = fwhm_of(psf[c])
    coef = math.pi ** 2 / 4
    NUM["分辨率与小缺陷"] = {
        "说明": f"对焦的艾里斑 PSF，长度单位 {U}；圆形缺陷放在均匀背景上，峰值对比度指缺陷中心处“像的强度变化 ÷ 物的强度变化”；"
              f"细网格 {DX} {U}，512×512",
        "艾里斑第一暗环半径（瑞利判据的系数）": r3(FIRST_ZERO, 4),
        "艾里斑半高全宽": r3(psf_fwhm, 3),
        "两个等亮点光源连线中点的亮度 ÷ 峰值": {k: r3(v, 3) for k, v in dips.items()},
        "斯派罗极限（中间的凹陷刚好消失的间距）": r3(sparrow, 3),
        "FFT 光瞳法的 PSF 与解析艾里斑的最大偏差（峰值归一）": sci(psf_err),
        "缺陷直径": diams,
        "峰值对比度（艾里斑，数值卷积）": [r3(v, 4) for v in peak_a],
        "峰值对比度（艾里斑，解析值：半径 d/2 以内的能量比例）": [r3(v, 4) for v in exact],
        "数值卷积与解析值的最大偏差": sci(max(abs(a - e) for a, e in zip(peak_a, exact))),
        "峰值对比度（高斯近似）": [r3(v, 4) for v in peak_g],
        "像的半高全宽": [r3(v, 3) for v in width],
        "小缺陷极限：峰值对比度 = 系数 × 直径²，系数 π²/4": r3(coef, 3),
        "直径 0.2 与 0.1 的峰值对比度之比": r3(peak_a[4] / peak_a[2], 2),
        "直径 1.0 与 0.3 的峰值对比度之比": r3(peak_a[10] / peak_a[5], 2),
        "直径 1.0 时高斯近似比艾里斑高出的比例": r3(peak_g[10] / peak_a[10] - 1, 3),
    }

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0), layout="constrained")
    for (name, s), color in zip(seps.items(), (S.RED, S.BLUE, S.GREEN)):
        label = f"{name}（间距 {s:.2f}）" if "间距" not in name else f"间距 {s:.1f}"
        ax[0].plot(x, profs[name], color=color, lw=1.8, label=label)
    ax[0].set_xlabel(f"位置（{U}）")
    ax[0].set_ylabel("亮度（峰值归一）")
    ax[0].set_ylim(0, 1.08)
    ax[0].set_title("两个点光源：间距越小，中间的凹陷越浅")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    S.clean(ax[0])

    dd = np.array(diams)
    ax[1].plot(dd, peak_a, "o-", color=S.BLUE, lw=1.8, ms=4, label="艾里斑 PSF")
    ax[1].plot(dd, peak_g, "--", color=S.ORANGE, lw=1.5, label="高斯近似")
    ax[1].plot(dd[dd <= 0.8], coef * dd[dd <= 0.8] ** 2, ":", color=S.MUTED, lw=1.5, label=r"按面积：$(\pi^2/4)\,d^2$")
    ax[1].axvline(FIRST_ZERO, color=S.MUTED, lw=1, ls="--")
    ax[1].text(FIRST_ZERO * 1.08, 0.0062, "瑞利分辨率", fontsize=8.5, color=S.MUTED)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 3], ["0.05", "0.1", "0.2", "0.5", "1", "2", "3"])
    ax[1].set_yticks([0.01, 0.03, 0.1, 0.3, 1], ["0.01", "0.03", "0.1", "0.3", "1"])
    ax[1].minorticks_off()
    ax[1].set_xlabel(f"缺陷直径 d（{U}）")
    ax[1].set_ylabel("像的峰值对比度 / 物的对比度")
    ax[1].set_title("小于分辨率之后，峰值按面积下降")
    ax[1].legend(fontsize=8.5, loc="upper left")
    S.clean(ax[1])

    ax[2].plot(dd, width, "o-", color=S.BLUE, lw=1.8, ms=4, label="像的半高全宽")
    ax[2].plot(dd, dd, ":", color=S.MUTED, lw=1.5, label="等于缺陷直径")
    ax[2].axhline(psf_fwhm, color=S.ORANGE, lw=1.3, ls="--", label=f"PSF 的半高全宽 {psf_fwhm:.2f}")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 3], ["0.05", "0.1", "0.2", "0.5", "1", "2", "3"])
    ax[2].set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2, 3], ["0.05", "0.1", "0.2", "0.5", "1", "2", "3"])
    ax[2].minorticks_off()
    ax[2].set_xlabel(f"缺陷直径 d（{U}）")
    ax[2].set_ylabel(f"像的半高全宽（{U}）")
    ax[2].set_title("光斑的大小停在 PSF 的宽度")
    ax[2].legend(fontsize=8.5, loc="upper left")
    S.clean(ax[2])
    S.save(fig, CH, "fig02_分辨率与小缺陷")


# ---------------------------------------------------------------- 图 3：PSF、MTF 与离焦
def fig_psf_mtf():
    n = 1600                     # 频率间隔 1/(n·DX) = 0.0125 NA/λ
    c = n // 2
    f = np.fft.fftfreq(n, d=DX)
    kk = {0.5: 40, 1.0: 80, 1.5: 120, 2.0: 160}

    def otf_row(psf):
        return np.real(np.fft.fft(np.fft.ifftshift(psf.sum(axis=0))))

    focus = pupil_psf(0.0)
    otf0 = otf_row(focus)
    zs = [round(0.1 * i, 1) for i in range(21)]
    strehl, mt, slices = [], {0.5: [], 1.0: [], 1.5: []}, {}
    for z in zs:
        psf = focus if z == 0 else pupil_psf(z / 2)
        otf = otf_row(psf)
        strehl.append(float(psf[c, c] / focus[c, c]))
        for nu in mt:
            mt[nu].append(float(otf[kk[nu]]))
        if z in (0.5, 1.0):
            slices[z] = otf[:n // 2]
    theory = [1.0 if z == 0 else (math.sin(math.pi * z / 2) / (math.pi * z / 2)) ** 2 for z in zs]
    sig_half = fit_gauss_sigma(pupil_psf(0.25, n_out=256))
    sig_one = fit_gauss_sigma(pupil_psf(0.5, n_out=256))

    def rel(i):
        return {"斯特列尔比": r3(strehl[i], 3), **{f"频率 {nu} 的 OTF": r3(mt[nu][i] / mt[nu][0], 3) for nu in mt}}

    NUM["PSF、MTF 与离焦"] = {
        "说明": f"长度单位 {U}，空间频率单位 NA/λ，离焦量 z 的单位 λ/NA²（傍轴近似，光瞳边缘的波像差是 z/2 个波长）；"
              "PSF 由 1600×1600 的光瞳函数经 FFT 得到；PSF 对称，所以光学传递函数 OTF 是实数，MTF 是它的绝对值；OTF 为负表示这个频率的图案黑白反转",
        "最接近对焦艾里斑的高斯宽度 sigma（二维最小二乘）": r3(SIG_FOCUS, 3),
        "半径以内的能量比例": {
            "第一暗环以内": {"艾里斑": r3(airy_encircled(FIRST_ZERO)[0], 3), "高斯近似": r3(1 - math.exp(-FIRST_ZERO ** 2 / (2 * SIG_FOCUS ** 2)), 3)},
            "半径 1.0 以内": {"艾里斑": r3(airy_encircled(1.0)[0], 3), "高斯近似": r3(1 - math.exp(-1 / (2 * SIG_FOCUS ** 2)), 5)},
            "半径 2.0 以内": {"艾里斑": r3(airy_encircled(2.0)[0], 3), "高斯近似": 1.0},
        },
        "对焦时的 MTF": {
            "频率": [0.5, 1.0, 1.5, 2.0],
            "衍射极限公式": [r3(mtf_diffraction(v), 3) for v in kk],
            "由 PSF 数值计算": [r3(otf0[k], 3) for k in kk.values()],
            "高斯近似": [r3(math.exp(-2 * math.pi ** 2 * SIG_FOCUS ** 2 * v * v), 3) for v in kk],
        },
        "离焦量": zs,
        "中心亮度相对对焦时的比例（斯特列尔比）": [r3(v, 3) for v in strehl],
        "斯特列尔比的傍轴公式 sinc²(πz/2)": [r3(v, 3) for v in theory],
        "频率 0.5 处的 OTF": [r3(v, 3) for v in mt[0.5]],
        "频率 1.0 处的 OTF": [r3(v, 3) for v in mt[1.0]],
        "频率 1.5 处的 OTF": [r3(v, 3) for v in mt[1.5]],
        "离焦 0.25 时相对对焦的比例": {"斯特列尔比（傍轴公式）": r3((math.sin(math.pi / 8) / (math.pi / 8)) ** 2, 3)},
        "离焦 0.5 时相对对焦的比例": rel(5),
        "离焦 1.0 时相对对焦的比例": rel(10),
        "最接近离焦 PSF 的高斯宽度": {"离焦 0.5": r3(sig_half, 3), "离焦 1.0": r3(sig_one, 3),
                           "离焦 0.5 相对对焦的倍数": r3(sig_half / SIG_FOCUS, 2), "离焦 1.0 相对对焦的倍数": r3(sig_one / SIG_FOCUS, 2)},
    }

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0), layout="constrained")
    r = np.linspace(0, 2.2, 441)
    ax[0].plot(r, np.pi * airy(r), color=S.BLUE, lw=1.8, label="艾里斑")
    ax[0].plot(r, np.exp(-r * r / (2 * SIG_FOCUS ** 2)) / (2 * np.pi * SIG_FOCUS ** 2), "--", color=S.ORANGE, lw=1.6,
               label=f"高斯近似（sigma = {SIG_FOCUS:.3f}）")
    ax[0].axvline(FIRST_ZERO, color=S.MUTED, lw=1, ls=":")
    ax[0].text(FIRST_ZERO + 0.03, 2.2, "第一暗环", fontsize=8.5, color=S.MUTED)
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-4, 6)
    ax[0].set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ["0.0001", "0.001", "0.01", "0.1", "1"])
    ax[0].minorticks_off()
    ax[0].set_xlabel(f"到中心的距离（{U}）")
    ax[0].set_ylabel("PSF（总能量归一，对数轴）")
    ax[0].set_title("艾里斑在第一暗环外还有旁瓣")
    ax[0].legend(fontsize=8.5, loc="upper right")
    S.clean(ax[0])

    nu = np.linspace(0, 2.4, 241)
    half = slice(0, 193)
    ax[1].plot(nu, mtf_diffraction(nu), color=S.BLUE, lw=1.8, label="对焦：衍射极限公式")
    ax[1].plot(f[8:168:16], otf0[8:168:16], "o", color=S.BLUE, ms=4, label="对焦：由 PSF 数值计算")
    ax[1].plot(nu, np.exp(-2 * np.pi ** 2 * SIG_FOCUS ** 2 * nu ** 2), "--", color=S.ORANGE, lw=1.6, label="高斯近似")
    for z, color in ((0.5, S.GREEN), (1.0, S.RED)):
        ax[1].plot(f[half], slices[z][half], color=color, lw=1.6, label=f"离焦 {z} {U2}")
    ax[1].axhline(0, color=S.MUTED, lw=0.8)
    ax[1].axvline(2, color=S.MUTED, lw=1, ls=":")
    ax[1].text(1.97, 0.9, "截止频率", fontsize=8.5, color=S.MUTED, ha="right")
    ax[1].set_xlim(0, 2.4)
    ax[1].set_xlabel("空间频率（NA/λ）")
    ax[1].set_ylabel("OTF（绝对值是 MTF，负值是黑白反转）")
    ax[1].set_title("对焦时在 2 NA/λ 处截止，高斯近似没有截止")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[1])

    ax[2].plot(zs, strehl, "o-", color=S.INK, lw=1.8, ms=3.5, label="中心亮度（斯特列尔比）")
    ax[2].plot(zs, theory, ":", color=S.MUTED, lw=2.2, label=r"傍轴公式 $\mathrm{sinc}^2(\pi z/2)$")
    for nu0, color in ((0.5, S.BLUE), (1.0, S.ORANGE), (1.5, S.RED)):
        ax[2].plot(zs, np.array(mt[nu0]) / mt[nu0][0], color=color, lw=1.6, label=f"频率 {nu0} NA/λ 的 OTF")
    ax[2].axhline(0, color=S.MUTED, lw=0.8)
    ax[2].set_xlabel(f"离焦量 z（{U2}）")
    ax[2].set_ylabel("相对对焦时的比例")
    ax[2].set_title(f"离焦 1 {U2} 时中心亮度只剩四成")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    S.save(fig, CH, "fig03_PSF与MTF与离焦")


# ---------------------------------------------------------------- 图 4：像素尺寸
PIX_D, PIX_C, PIX_F = 0.3, 1.0, 12000.0


def defect_spot(n=256):
    """直径 PIX_D 的缺陷经过对焦 PSF 之后，每个细格里多出的电子数。缺陷中心在 (n/2, n/2)。"""
    c0 = n // 2
    return fftconv(disk(PIX_D, c0, c0, n), airy_psf(n)) * PIX_C * PIX_F * DX ** 2


def fig_pixel():
    n = 256
    c0 = n // 2
    spot = defect_spot(n)
    bs = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]
    reads = [0.0, 5.0, 20.0]
    grid = [i / 8 for i in range(8)]
    res = {rd: {k: [] for k in ("best", "worst", "mf", "frac", "mean")} for rd in reads}
    crops = {}
    for b in bs:
        j0 = (n // 2) // b
        peaks, energy = np.zeros((8, 8)), np.zeros((8, 8))
        for iy, fy in enumerate(grid):
            for ix, fx in enumerate(grid):
                cy, cx = j0 * b + (b - 1) / 2 + fy * b, j0 * b + (b - 1) / 2 + fx * b
                s = binning(fshift(spot, cy - c0, cx - c0), b)
                peaks[iy, ix] = s.max()
                energy[iy, ix] = (s ** 2).sum()
                if (iy, ix) in ((0, 0), (4, 4)):
                    crops[(b, iy)] = s
        for rd in reads:
            sd = math.sqrt(PIX_F * (b * DX) ** 2 + rd ** 2)
            res[rd]["best"].append(float(peaks[0, 0] / sd))
            res[rd]["worst"].append(float(peaks[4, 4] / sd))
            res[rd]["mean"].append(float(peaks.mean() / sd))
            res[rd]["frac"].append(float((peaks / sd >= 5).mean()))
            res[rd]["mf"].append(float(math.sqrt(energy[4, 4]) / sd))

    rng = np.random.default_rng(4)
    check = {}
    for b in (3, 6, 12, 24):
        s = crops[(b, 0)]
        py, px = np.unravel_index(s.argmax(), s.shape)
        bg_e = PIX_F * (b * DX) ** 2
        cap = mod.simulate_capture(np.full(6000, bg_e + s[py, px]), 5.0, 1.0, 16, rng).astype(np.float64)
        bgc = mod.simulate_capture(np.full(60000, bg_e), 5.0, 1.0, 16, rng).astype(np.float64)
        check[f"{b * DX:.2f}"] = {"按公式": r3(res[5.0]["best"][bs.index(b)], 2), "实测": r3((cap.mean() - bgc.mean()) / bgc.std(), 2)}

    ps = [round(b * DX, 2) for b in bs]
    table = {}
    for rd in reads:
        table[f"读出噪声 {rd:.0f} e-"] = {
            "单像素信噪比，缺陷在像素中心": [r3(v, 2) for v in res[rd]["best"]],
            "单像素信噪比，缺陷在四个像素的交点": [r3(v, 2) for v in res[rd]["worst"]],
            "单像素信噪比，对 64 个位置取平均": [r3(v, 2) for v in res[rd]["mean"]],
            "64 个位置里单像素信噪比不低于 5 的比例": [r3(v, 3) for v in res[rd]["frac"]],
            "光斑内所有像素按形状加权求和（匹配滤波）的信噪比，缺陷在像素交点": [r3(v, 2) for v in res[rd]["mf"]],
        }

    # 采样频率：严格的奈奎斯特条件和“分辨率的一半”
    p_half = FIRST_ZERO / 2
    nyq = 1 / (2 * p_half)
    # 混叠：一维线条，周期 0.7，先乘衍射极限 MTF，再做像素积分和采样
    period, cells = 0.7, 840
    xs = np.arange(cells) * DX
    bars = (np.mod(xs, period) < period / 2 - 1e-9).astype(np.float64)
    fr = np.fft.fftfreq(cells, d=DX)
    opt = np.real(np.fft.ifft(np.fft.fft(0.1 + 0.8 * bars) * mtf_diffraction(fr)))
    alias = {}
    for b in (2, 4, 6, 7, 8, 10, 12):
        smp = opt.reshape(cells // b, b).mean(axis=1)
        spec = np.abs(np.fft.rfft(smp - smp.mean()))
        k = int(spec.argmax())
        alias[f"{b * DX:.2f}"] = {"每个周期的像素数": r3(period / (b * DX), 2),
                                  "采样后测到的周期": r3(len(smp) * b * DX / k, 2),
                                  "对比度 (max-min)/(max+min)": r3((smp.max() - smp.min()) / (smp.max() + smp.min()), 3)}
    NUM["像素尺寸"] = {
        "说明": f"直径 {PIX_D} {U} 的圆形缺陷，比背景亮 {PIX_C:.0%}；对焦的艾里斑 PSF；背景每 ({U})² 面积 {PIX_F:.0f} 个电子，"
              "所以每像素背景电子数等于它乘以像素面积；噪声取背景处的散粒噪声加读出噪声；缺陷相对像素网格的位置取 8×8 共 64 种",
        "像素尺寸": ps,
        "每像素背景电子数": [r3(PIX_F * p * p, 0) for p in ps],
        "结果": table,
        "用 simulate_capture 核对（读出噪声 5 e-，缺陷在像素中心，6000 次）": check,
        "采样条件": {
            "截止频率（NA/λ）": 2.0,
            "严格奈奎斯特要求的像素尺寸": 0.25,
            "严格奈奎斯特像素 ÷ 瑞利分辨率": r3(0.25 / FIRST_ZERO, 2),
            "像素取瑞利分辨率的一半时的像素尺寸": r3(p_half, 3),
            "这时采样的奈奎斯特频率": r3(nyq, 2),
            "该频率处的衍射极限 MTF": r3(mtf_diffraction(nyq), 3),
        },
        "混叠实验": {
            "说明": f"一维线条，周期 {period} {U}（空间频率 {1 / period:.2f} NA/λ），反射率在 0.1 和 0.9 之间；"
                  "乘衍射极限 MTF 后做像素积分和采样，用采样后频谱的最高峰换算周期",
            "光学之后、采样之前的对比度": r3((opt.max() - opt.min()) / (opt.max() + opt.min()), 3),
            "按像素尺寸": alias,
        },
    }

    fig = plt.figure(figsize=(16, 4.4), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1])
    a0 = fig.add_subplot(gs[0])
    show = [3, 6, 12, 24]
    w = 72
    mosaic = np.full((2 * w + 4, 4 * w + 12), np.nan)
    for col, b in enumerate(show):
        for row, iy in enumerate((0, 4)):
            up = np.kron(crops[(b, iy)] / (PIX_F * (b * DX) ** 2), np.ones((b, b)))
            center = (n // 2) // b * b + (b - 1) / 2 + (0.5 * b if iy == 4 else 0.0)
            lo = int(round(center - w / 2 + 0.5))
            mosaic[row * (w + 4):row * (w + 4) + w, col * (w + 4):col * (w + 4) + w] = up[lo:lo + w, lo:lo + w]
    im = a0.imshow(mosaic, cmap="magma", vmin=0)
    for col, b in enumerate(show):
        a0.text(col * (w + 4) + w / 2, -4, f"像素 {b * DX:.2f}", ha="center", fontsize=9)
    a0.text(-4, w / 2, "缺陷在像素中心", rotation=90, va="center", ha="right", fontsize=9)
    a0.text(-4, w + 4 + w / 2, "缺陷在像素交点", rotation=90, va="center", ha="right", fontsize=9)
    a0.set_title("同一个缺陷在四种像素尺寸下的像\n（每个像素比背景多出的比例，无噪声）", pad=16)
    a0.set_xticks([])
    a0.set_yticks([])
    for side in a0.spines.values():
        side.set_visible(False)
    fig.colorbar(im, ax=a0, fraction=0.035, pad=0.02).ax.tick_params(labelsize=8)

    for a, rd in ((fig.add_subplot(gs[1]), 0.0), (fig.add_subplot(gs[2]), 20.0)):
        a.plot(ps, res[rd]["mf"], "^-", color=S.GREEN, lw=1.6, ms=4, label="光斑内像素加权求和，缺陷在像素交点")
        a.plot(ps, res[rd]["best"], "o-", color=S.BLUE, lw=1.8, ms=4, label="只看最亮的一个像素，缺陷在像素中心")
        a.plot(ps, res[rd]["worst"], "s-", color=S.RED, lw=1.8, ms=4, label="只看最亮的一个像素，缺陷在像素交点")
        a.axhline(5, color=S.MUTED, lw=1, ls="--")
        a.set_xscale("log")
        a.set_xticks([0.1, 0.25, 0.5, 1, 1.6], ["0.1", "0.25", "0.5", "1", "1.6"])
        a.minorticks_off()
        a.set_ylim(0, 9.8)
        a.set_xlabel(f"像素尺寸（{U}）")
        a.set_ylabel("缺陷信噪比")
        a.set_title("只有散粒噪声：像素越小，加权求和的信噪比越高" if rd == 0 else "再加 20 e- 读出噪声：像素太小，加权求和也变差")
        a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
        S.clean(a)
    S.save(fig, CH, "fig04_像素尺寸")


# ---------------------------------------------------------------- 图 5：噪声、PTC、曝光量、量化
def ptc_points(levels, read, gain, bits, rng, shape=(200, 200), black=0.0):
    """black 不为 0 时拍摄带黑电平，均值里把它减掉（方差不受黑电平影响）。"""
    m, v = [], []
    for lv in levels:
        mean_e = np.full(shape, float(lv))
        if black:
            f = mod.simulate_capture(mean_e, read, gain, bits, rng, black_level_adu=black)
        else:
            f = mod.simulate_capture(mean_e, read, gain, bits, rng)
        m.append(float(f.mean()) - black)
        v.append(float(f.var()))
    return np.array(m), np.array(v)


def ptc_repeat(levels, read, gain, bits, seed, reps=200, weighted=False, black=0.0):
    """重复做 PTC 拟合，返回每次的 (增益, 读出噪声, 截距)。weighted=True 时按方差的倒数加权。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        m, v = ptc_points(levels, read, gain, bits, rng, black=black)
        if weighted:
            a, b = np.polyfit(m, v, 1, w=1.0 / v)
            out.append((1 / a, math.sqrt(max(b, 0.0)) / a, b))
        else:
            g, r = mod.estimate_gain_from_ptc(m, v)
            out.append((g, r, np.polyfit(m, v, 1)[1]))
    return np.array(out)


def ptc_summary(est, read, gain):
    return {"增益的平均值": r3(est[:, 0].mean(), 3), "增益的标准差 ÷ 真值": f"{100 * est[:, 0].std() / gain:.1f}%",
            "读出噪声的平均值": r3(est[:, 1].mean(), 2), "读出噪声的标准差 ÷ 真值": f"{100 * est[:, 1].std() / read:.0f}%",
            "读出噪声偏离真值超过 25% 的次数比例": r3((np.abs(est[:, 1] - read) / read > 0.25).mean(), 3)}


def fig_noise():
    # 泊松噪声：方差等于均值
    rng = np.random.default_rng(10)
    poisson = {}
    for mu in (10, 100, 1000, 10000):
        f = mod.simulate_capture(np.full((400, 400), float(mu)), 0.0, 1.0, 16, rng)
        poisson[str(mu)] = {"均值": r3(f.mean(), 2), "方差": r3(f.var(), 1), "标准差": r3(f.std(), 2), "均值的平方根": r3(math.sqrt(mu), 2)}

    # PTC：从暗到饱和
    read, gain, bits = 8.0, 1.5, 12
    full = 2 ** bits - 1
    levels = [0, 20, 50, 100, 300, 800, 1500, 2200, 3000, 3800, 4500, 5200, 5600, 5900, 6050, 6150, 6250, 6400, 7000]
    rng = np.random.default_rng(11)
    pm, pv = ptc_points(levels, read, gain, bits, rng)
    ok = (pm < 0.7 * full) & (np.array(levels) >= 50)
    g_ok, r_ok = mod.estimate_gain_from_ptc(pm[ok], pv[ok])
    g_all, r_all = mod.estimate_gain_from_ptc(pm, pv)
    a_ok, b_ok = np.polyfit(pm[ok], pv[ok], 1)
    a_all, b_all = np.polyfit(pm, pv, 1)

    # Notebook 实验 2 的设置：全部点和只用未饱和点
    nb_levels = [20, 50, 100, 300, 800, 1500, 3000, 6000, 12000, 20000]
    rng = np.random.default_rng(12)
    nb_all, nb_ok = [], []
    for _ in range(50):
        m, v = ptc_points(nb_levels, read, gain, bits, rng, shape=(100, 100))
        nb_all.append(mod.estimate_gain_from_ptc(m, v))
        keep = m < 0.7 * full
        nb_ok.append(mod.estimate_gain_from_ptc(m[keep], v[keep]))
    nb_all, nb_ok = np.array(nb_all), np.array(nb_ok)

    # 重复 200 次：斜率和截距各有多准
    test_levels = [50, 200, 500, 1000, 2000, 4000]
    rep = {
        "验收脚本的设置（读出噪声 8 e-，增益 1.5，14 位）": ptc_summary(ptc_repeat(test_levels, 8.0, 1.5, 14, 20), 8.0, 1.5),
        "读出噪声换成 3 e-": ptc_summary(ptc_repeat(test_levels, 3.0, 1.5, 14, 21), 3.0, 1.5),
        "读出噪声 3 e-，改用按方差倒数加权的拟合": ptc_summary(ptc_repeat(test_levels, 3.0, 1.5, 14, 21, weighted=True), 3.0, 1.5),
        "读出噪声 8 e-，多加一档全暗（0 e-），不加黑电平": ptc_summary(ptc_repeat([0] + test_levels, 8.0, 1.5, 14, 22), 8.0, 1.5),
        f"读出噪声 8 e-，多加一档全暗（0 e-），加黑电平 {BLACK:.0f} ADU": ptc_summary(ptc_repeat([0] + test_levels, 8.0, 1.5, 14, 26, black=BLACK), 8.0, 1.5),
    }
    coarse = ptc_repeat([100, 200, 400, 800, 1600, 3200], 2.0, 8.0, 14, 23)
    b_mean, a_mean = coarse[:, 2].mean(), 1 / coarse[:, 0].mean()
    rng = np.random.default_rng(24)
    dark = {}
    for rd, gn in ((8.0, 1.5), (5.0, 2.0)):
        f = mod.simulate_capture(np.zeros((400, 400)), rd, gn, 12, rng).astype(np.float64)
        dark[f"读出噪声 {rd:.0f} e-、增益 {gn}"] = {"暗帧均值（ADU）": r3(f.mean(), 2), "暗帧标准差（ADU）": r3(f.std(), 2),
                                             "没有裁剪时应有的标准差（ADU）": r3(math.sqrt(rd ** 2 / gn ** 2 + 1 / 12), 2)}
    rng_b = np.random.default_rng(25)
    for rd, gn in ((8.0, 1.5), (5.0, 2.0)):
        f = mod.simulate_capture(np.zeros((400, 400)), rd, gn, 12, rng_b, black_level_adu=BLACK).astype(np.float64)
        dark[f"读出噪声 {rd:.0f} e-、增益 {gn}"].update({
            f"加黑电平 {BLACK:.0f} ADU：暗帧均值减去黑电平（ADU）": r3(f.mean() - BLACK, 2) + 0.0,   # 加 0.0 是为了不把 -0.0 写进 JSON
            f"加黑电平 {BLACK:.0f} ADU：暗帧标准差（ADU）": r3(f.std(), 2),
            f"加黑电平 {BLACK:.0f} ADU：被裁到 0 的像素个数": int((f == 0).sum())})

    # 信噪比随曝光量
    rd_s = 5.0
    mus = [1, 2, 4, 6, 10, 16, 25, 40, 64, 100, 160, 250, 400, 1000, 4000, 16000, 50000]
    rng = np.random.default_rng(13)
    snr_meas, mean_meas = [], []
    for mu in mus:
        f = mod.simulate_capture(np.full((300, 300), float(mu)), rd_s, 1.0, 16, rng).astype(np.float64)
        snr_meas.append(float(f.mean() / f.std()))
        mean_meas.append(float(f.mean()))
    snr_th = [mu / math.sqrt(mu + rd_s ** 2) for mu in mus]
    rng_b = np.random.default_rng(15)
    snr_black, mean_black = [], []
    for mu in mus:
        f = mod.simulate_capture(np.full((300, 300), float(mu)), rd_s, 1.0, 16, rng_b, black_level_adu=BLACK).astype(np.float64)
        mean_black.append(float(f.mean()) - BLACK)
        snr_black.append(mean_black[-1] / float(f.std()))
    at = lambda mu: mu / math.sqrt(mu + rd_s ** 2)  # noqa: E731

    # 量化位数
    fw, rd_q = 20000.0, 5.0
    bits_list = [6, 7, 8, 9, 10, 12, 14]
    bgs = [400.0, 2000.0, 16000.0]
    rng = np.random.default_rng(14)
    q_meas, q_th, q_ratio = {}, {}, {}
    for bg in bgs:
        ideal = math.sqrt(bg + rd_q ** 2)
        q_meas[bg], q_th[bg], q_ratio[bg] = [], [], []
        for nb in bits_list:
            step = fw / 2 ** nb
            mu = bg + rng.uniform(-step / 2, step / 2, size=200000)
            adu = mod.simulate_capture(mu, rd_q, step, nb, rng)
            q_meas[bg].append(float(np.sqrt(((adu * step - mu) ** 2).mean()) / ideal))
            q_th[bg].append(math.sqrt(1 + (step / ideal) ** 2 / 12))
            q_ratio[bg].append(step / ideal)
    weak = {}
    for nb in (6, 7, 8, 9, 10, 12):
        step = fw / 2 ** nb
        resp = []
        for bg in (400.0, 430.0, 470.0, 500.0, 550.0):
            a = mod.simulate_capture(np.full(400000, bg), rd_q, step, nb, rng) * step
            b = mod.simulate_capture(np.full(400000, bg * 1.03), rd_q, step, nb, rng) * step
            resp.append(float((b.mean() - a.mean()) / (bg * 0.03)))
        weak[str(nb)] = {"量化步长（e-）": r3(step, 1), "步长 ÷ 噪声": r3(step / math.sqrt(425.0), 2),
                         "测到的亮度差 ÷ 真实亮度差（5 个背景亮度下的最小和最大）": [r3(min(resp), 2), r3(max(resp), 2)]}

    NUM["噪声与 PTC"] = {
        "泊松噪声的方差等于均值": {"说明": "simulate_capture，读出噪声 0、增益 1 e-/ADU、16 位，400×400 像素", "按平均电子数": poisson},
        "PTC（图 5 左）": {
            "说明": f"读出噪声 {read:.0f} e-、增益 {gain} e-/ADU、{bits} 位（最大 {full} ADU，对应 {full * gain:.0f} e-），每档 200×200 像素；"
                  "“未饱和点”指均值低于最大值的 70% 且不低于 50 e- 的档",
            "平均电子数": levels,
            "均值（ADU）": [r3(v, 1) for v in pm],
            "方差（ADU²）": [r3(v, 1) for v in pv],
            "只用未饱和点拟合（增益，读出噪声）": [r3(g_ok, 3), r3(r_ok, 2)],
            "用全部点拟合（增益，读出噪声）": [r3(g_all, 2), r3(r_all, 1)],
        },
        "Notebook 实验 2 的设置，重复 50 次": {
            "说明": "电子数 20 到 20000 共 10 档，100×100 像素，真值是增益 1.5、读出噪声 8",
            "只用未饱和点：增益的平均值": r3(nb_ok[:, 0].mean(), 3), "只用未饱和点：读出噪声的平均值": r3(nb_ok[:, 1].mean(), 2),
            "用全部点：增益的平均值": r3(nb_all[:, 0].mean(), 2), "用全部点：读出噪声的平均值": r3(nb_all[:, 1].mean(), 1),
        },
        "斜率和截距各有多准（各重复 200 次）": {"说明": "电子数 50、200、500、1000、2000、4000 共 6 档，每档 200×200 像素，增益真值 1.5 e-/ADU", **rep},
        "截距里还有量化噪声": {
            "说明": "读出噪声 2 e-、增益 8 e-/ADU、14 位，电子数 100 到 3200 共 6 档，重复 200 次，用 200 次的平均截距和平均斜率换算",
            "换算出的读出噪声": r3(math.sqrt(b_mean) / a_mean, 2), "真值": 2.0, "按 sqrt(读出噪声² + 增益²/12) 算": r3(math.sqrt(4 + 64 / 12), 2),
        },
        "全暗时的裁剪": {"说明": f"平均电子数为 0，12 位，400×400 像素。不传黑电平（默认 0）时负值被裁成 0；传黑电平 {BLACK:.0f} ADU 时整个分布离开 0，不再被裁剪",
                   "只有读出噪声、负的一半被裁成 0 时（不算量化）：均值 ÷ 裁剪前的标准差": r3(1 / math.sqrt(2 * math.pi)),
                   "只有读出噪声、负的一半被裁成 0 时（不算量化）：标准差 ÷ 裁剪前的标准差": r3(math.sqrt(0.5 - 1 / (2 * math.pi))),
                   **dark},
    }
    NUM["信噪比随曝光量"] = {
        "说明": f"均匀场，读出噪声 {rd_s:.0f} e-、增益 1、16 位，每档 300×300 像素；信噪比 = 均值 ÷ 标准差。"
              f"实测有两组：一组不传黑电平（默认 0），一组传黑电平 {BLACK:.0f} ADU、算信噪比时先从均值里减掉它",
        "平均电子数": mus,
        "实测信噪比（不加黑电平）": [r3(v, 2) for v in snr_meas],
        f"实测信噪比（黑电平 {BLACK:.0f} ADU）": [r3(v, 2) for v in snr_black],
        "公式 μ/sqrt(μ+σ²)": [r3(v, 2) for v in snr_th],
        "实测均值（不加黑电平）": [r3(v, 2) for v in mean_meas],
        f"实测均值减去黑电平（黑电平 {BLACK:.0f} ADU）": [r3(v, 2) for v in mean_black],
        "两种噪声相等的电子数（读出噪声的平方）": rd_s ** 2,
        "曝光量乘 4 时信噪比的倍数（公式）": {"4 → 16 e-": r3(at(16) / at(4), 2), "100 → 400 e-": r3(at(400) / at(100), 2),
                                 "1000 → 4000 e-": r3(at(4000) / at(1000), 2), "4000 → 16000 e-": r3(at(16000) / at(4000), 2)},
        "曝光量乘 4 时信噪比的倍数（实测）": {"100 → 400 e-": r3(snr_meas[12] / snr_meas[9], 2), "1000 → 4000 e-": r3(snr_meas[14] / snr_meas[13], 2),
                                 "4000 → 16000 e-": r3(snr_meas[15] / snr_meas[14], 2)},
        f"曝光量乘 4 时信噪比的倍数（实测，黑电平 {BLACK:.0f} ADU）": {"4 → 16 e-": r3(snr_black[5] / snr_black[2], 2), "100 → 400 e-": r3(snr_black[12] / snr_black[9], 2),
                                                 "1000 → 4000 e-": r3(snr_black[14] / snr_black[13], 2)},
    }
    NUM["量化位数"] = {
        "说明": f"满量程 {fw:.0f} e-，量化步长 = 满量程 ÷ 2^位数，读出噪声 {rd_q:.0f} e-；每个像素的平均电子数在背景值上下半个步长内均匀分布，"
              "总误差指“ADU × 步长”与平均电子数之差的均方根，20 万个像素",
        "位数": bits_list,
        "量化步长（e-）": [r3(fw / 2 ** nb, 1) for nb in bits_list],
        "不量化时的噪声（e-）": {f"背景 {bg:.0f} e-": r3(math.sqrt(bg + rd_q ** 2), 1) for bg in bgs},
        "总误差 ÷ 不量化时的噪声（实测）": {f"背景 {bg:.0f} e-": [r3(v, 3) for v in q_meas[bg]] for bg in bgs},
        "总误差 ÷ 不量化时的噪声（按 sqrt(1 + (步长/噪声)²/12)）": {f"背景 {bg:.0f} e-": [r3(v, 3) for v in q_th[bg]] for bg in bgs},
        "步长 ÷ 噪声": {f"背景 {bg:.0f} e-": [r3(v, 2) for v in q_ratio[bg]] for bg in bgs},
        "步长等于噪声时总误差的倍数": r3(math.sqrt(1 + 1 / 12), 3),
        "步长等于 2 倍噪声时总误差的倍数": r3(math.sqrt(1 + 4 / 12), 3),
        "弱信号的响应": {"说明": "背景约 400 e-（噪声 20.6 e-）的均匀区，另一块比它亮 3%，各取 40 万个像素求平均后相减", "按位数": weak},
    }

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.1), layout="constrained")
    sat = pm > 0.97 * full
    ax[0].plot(pm[ok], pv[ok], "o", color=S.BLUE, ms=5, label="均值低于最大值 70% 的档（用于拟合）")
    ax[0].plot(pm[~ok & ~sat], pv[~ok & ~sat], "o", mfc="white", mec=S.BLUE, ms=5, label="没有用于拟合的档")
    ax[0].plot(pm[sat], pv[sat], "o", color=S.RED, ms=5, label="接近或达到饱和的档")
    xs = np.array([0, full])
    ax[0].plot(xs, a_ok * xs + b_ok, "-", color=S.BLUE, lw=1.5, label=f"只用蓝色实心点拟合：增益 {g_ok:.2f}")
    ax[0].plot(xs, a_all * xs + b_all, "--", color=S.RED, lw=1.5, label=f"用全部点拟合：增益 {g_all:.2f}")
    ax[0].axvline(full, color=S.MUTED, lw=1, ls=":")
    ax[0].set_xlabel("均值（ADU）")
    ax[0].set_ylabel("方差（ADU$^2$）")
    ax[0].set_ylim(bottom=-80)
    ax[0].set_title(f"PTC：未饱和时是直线，饱和后方差掉到 0（增益真值 {gain}）")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax[0])

    mm = np.array(mus, dtype=np.float64)
    ax[1].plot(mm, snr_th, "-", color=S.BLUE, lw=1.8, label=r"公式 $\mu/\sqrt{\mu+\sigma_r^2}$")
    ax[1].plot(mm, snr_meas, "o", mfc="white", mec=S.RED, ms=8, mew=1.3, label="实测，不加黑电平（负值被裁成 0）")
    ax[1].plot(mm, snr_black, "o", color=S.INK, ms=4, label=f"实测，黑电平 {BLACK:.0f} ADU（均值已减去黑电平）")
    ax[1].plot(mm[mm <= 200], mm[mm <= 200] / rd_s, ":", color=S.ORANGE, lw=1.6, label=r"只有读出噪声：$\mu/\sigma_r$")
    ax[1].plot(mm[mm >= 4], np.sqrt(mm[mm >= 4]), ":", color=S.GREEN, lw=1.6, label=r"只有散粒噪声：$\sqrt{\mu}$")
    ax[1].axvline(rd_s ** 2, color=S.MUTED, lw=1, ls="--")
    ax[1].text(rd_s ** 2 * 1.15, 0.25, "两种噪声相等", fontsize=8.5, color=S.MUTED)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xticks([1, 10, 100, 1000, 10000], ["1", "10", "100", "1000", "10000"])
    ax[1].set_yticks([0.3, 1, 3, 10, 30, 100], ["0.3", "1", "3", "10", "30", "100"])
    ax[1].minorticks_off()
    ax[1].set_xlabel("平均电子数（曝光量）")
    ax[1].set_ylabel("信噪比")
    ax[1].set_title("加黑电平后和公式一致；不加时最暗几档被裁剪抬高")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[1])

    for bg, color in zip(bgs, (S.RED, S.ORANGE, S.BLUE)):
        ax[2].plot(bits_list, q_meas[bg], "o", color=color, ms=5, label=f"背景 {bg:.0f} e-（噪声 {math.sqrt(bg + rd_q ** 2):.0f} e-）")
        ax[2].plot(bits_list, q_th[bg], "-", color=color, lw=1.4)
        ax[2].axvline(math.log2(fw / math.sqrt(bg + rd_q ** 2)), color=color, lw=1, ls=":")
    ax[2].plot([], [], "-", color=S.MUTED, lw=1.4, label=r"实线：$\sqrt{1+x^2/12}$，x 是步长与噪声之比")
    ax[2].plot([], [], ":", color=S.MUTED, lw=1, label="竖虚线：步长等于噪声的位数")
    ax[2].set_xticks(bits_list)
    ax[2].set_xlabel("量化位数")
    ax[2].set_ylabel("总误差 / 不量化时的噪声")
    ax[2].set_ylim(0.9, 4.8)
    ax[2].set_title("暗区先被量化卡住：8 位时多出近五成")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax[2])
    S.save(fig, CH, "fig05_噪声与PTC与量化")


# ---------------------------------------------------------------- 图 6：平场校正
def fig_flat():
    H = 128
    yy, xx = np.mgrid[0:H, 0:H]
    r2 = ((yy - 63.5) ** 2 + (xx - 63.5) ** 2) / (2 * 63.5 ** 2)
    V = (1 - 0.4 * r2) * (1 + 0.01 * np.random.default_rng(7).normal(size=(H, H)))
    for cy, cx, depth, s in ((30, 90, 0.15, 3.0), (95, 40, 0.10, 5.0), (70, 70, 0.20, 2.0)):
        V *= 1 - depth * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    FLAT, DARK, RD, GN, BITS = 6000.0, 100.0, 5.0, 2.0, 12
    rng = np.random.default_rng(8)

    def shoot(level, n, dark=DARK):
        base = level * V + dark
        return np.stack([mod.simulate_capture(base, RD, GN, BITS, rng).astype(np.float64) for _ in range(n)])

    def correct(raw, flat, dark):
        f = flat - dark
        return (raw - dark) / f * f.mean()

    def blocks(img, k=8):
        return img.reshape(H // k, k, H // k, k).mean(axis=(1, 3))

    cen, cor, mid = np.s_[56:72, 40:56], np.s_[2:14, 2:14], np.s_[32:96, 32:96]
    dark64 = shoot(0.0, 64).mean(0)
    flats = shoot(FLAT, 256)
    target = 3000.0
    raws = shoot(target, 256)
    net = raws.mean(0) - dark64
    c256 = correct(raws, flats.mean(0), dark64)
    cm = c256.mean(0)
    uniform = {
        "校正前": {"8×8 块均值的标准差 ÷ 平均值": f"{100 * blocks(net).std() / net.mean():.1f}%", "角落 ÷ 中心": r3(net[cor].mean() / net[cen].mean(), 3),
                "最深的尘点处 ÷ 周围": r3(net[69:72, 69:72].mean() / np.median(net[60:80, 60:80]), 3)},
        "校正后": {"8×8 块均值的标准差 ÷ 平均值": f"{100 * blocks(cm).std() / cm.mean():.2f}%", "角落 ÷ 中心": r3(cm[cor].mean() / cm[cen].mean(), 3),
                "最深的尘点处 ÷ 周围": r3(cm[69:72, 69:72].mean() / np.median(cm[60:80, 60:80]), 3)},
    }
    # 角落和中心的噪声
    tstd = c256.std(axis=0)
    corner = {"校正后的均值：角落 ÷ 中心": r3(cm[cor].mean() / cm[cen].mean(), 3),
              "校正后的时间噪声：角落 ÷ 中心": r3(tstd[cor].mean() / tstd[cen].mean(), 3),
              "角落的相对照度（相对中心）": r3(V[cor].mean() / V[cen].mean(), 3),
              "按 1/sqrt(相对照度) 估算的噪声比": r3(1 / math.sqrt(V[cor].mean() / V[cen].mean()), 3)}

    # 平场帧数对单帧噪声的影响（中间 64×64，目标和平场一样亮、目标是平场的一半两种）
    def var_adu(e):
        return (e + DARK + RD ** 2) / GN ** 2 + 1 / 12

    nfs = [1, 4, 16, 64]
    factor = {}
    for name, level in (("目标和平场一样亮", FLAT), ("目标亮度是平场的一半", target)):
        meas, th = [], []
        for nf in nfs:
            acc = []
            for _ in range(6):
                c = correct(shoot(level, 8), shoot(FLAT, nf).mean(0), dark64)
                acc.append(np.mean([(img[mid] - img[mid].mean()).var() for img in c]))
            s_adu, f_adu = level * V[mid] / GN, FLAT * V[mid] / GN
            scale = (FLAT * V).mean() / GN / f_adu
            temporal = float((var_adu(level * V[mid]) * scale ** 2).mean())
            fixed = float((s_adu ** 2 * var_adu(FLAT * V[mid]) / (nf * f_adu ** 2) * scale ** 2).mean())
            meas.append(math.sqrt(np.mean(acc) / temporal))
            th.append(math.sqrt(1 + fixed / temporal))
        factor[name] = {"实测": [r3(v, 3) for v in meas], "按公式": [r3(v, 3) for v in th]}

    # 多帧平均：平场带进来的噪声压不掉
    Ks = [1, 4, 16, 64, 256]
    resid = {}
    for nf in (1, 16, 256):
        c = correct(raws, flats[:nf].mean(0), dark64)
        resid[nf] = [float((c[:k].mean(0)[mid] - c[:k].mean(0)[mid].mean()).std()) for k in Ks]
    tn = float(np.sqrt((var_adu(target * V[mid]) * ((FLAT * V).mean() / (FLAT * V[mid])) ** 2).mean()))
    ideal = [tn / math.sqrt(k) for k in Ks]

    # 忘了减暗场
    nodark = {}
    for dk in (100.0, 400.0):
        d_frames = shoot(0.0, 64, dark=dk).mean(0)
        f_frames = shoot(FLAT, 64, dark=dk).mean(0)
        for level in (3000.0, 600.0):
            t = shoot(level, 64, dark=dk).mean(0)
            good = correct(t, f_frames, d_frames)
            bad = t / f_frames * f_frames.mean()
            nodark[f"暗信号 {dk:.0f} e-，目标 {level:.0f} e-"] = {"减暗场：角落 ÷ 中心": r3(good[cor].mean() / good[cen].mean(), 3),
                                                         "不减暗场：角落 ÷ 中心": r3(bad[cor].mean() / bad[cen].mean(), 3)}

    # 同一图形落在视场的两个位置上相减（相距 64 像素），超过 5 sigma 的像素比例
    pair = {}
    sub = raws[:64] - dark64
    cc = correct(raws[:64], flats[:64].mean(0), dark64)
    for name, stack in (("校正前（只减暗场）", sub), ("校正后", cc)):
        d = stack[:, :, :64] - stack[:, :, 64:]
        sd = np.sqrt(stack[:, :, :64].var(axis=0, ddof=1) + stack[:, :, 64:].var(axis=0, ddof=1))
        pair[name] = sci((np.abs(d) / sd > 5).mean(), 2)
    pair["高斯噪声下的理论值"] = sci(math.erfc(5 / math.sqrt(2)), 2)

    NUM["平场校正"] = {
        "说明": f"128×128 像素；视场响应 = 渐晕（角落约为中心的 0.6）× 1% 的像素响应不一致 × 3 个尘点（最深处暗 20%）；"
              f"平场帧在响应为 1 处是 {FLAT:.0f} e-，均匀目标是 {target:.0f} e-，暗信号 {DARK:.0f} e-；读出噪声 {RD:.0f} e-，增益 {GN:.0f} e-/ADU，{BITS} 位；"
              "校正公式：(原图 - 暗场) ÷ (平场 - 暗场) × 平均值",
        "视场响应的最小值和最大值": [r3(V.min(), 3), r3(V.max(), 3)],
        "均匀目标的不均匀度（256 帧平均；平场 256 帧、暗场 64 帧）": uniform,
        "平场校正之后噪声不平": corner,
        "平场帧数": nfs,
        "单帧噪声相对“平场没有噪声”时的倍数（中间 64×64 像素）": factor,
        "多帧平均后剩下的空间噪声（ADU，中间 64×64 像素，目标亮度是平场的一半）": {
            "平均的帧数": Ks, "只用 1 帧平场": [r3(v, 2) for v in resid[1]], "16 帧平场": [r3(v, 2) for v in resid[16]],
            "256 帧平场": [r3(v, 2) for v in resid[256]], "平场没有噪声时（按公式）": [r3(v, 2) for v in ideal]},
        "平均 256 帧之后：1 帧平场的残余 ÷ 256 帧平场的残余": r3(resid[1][-1] / resid[256][-1], 1),
        "忘了减暗场（平场 64 帧）": nodark,
        "相距 64 像素的两处相减，差值超过 5 sigma 的像素比例（均匀目标，64 帧）": pair,
    }

    fig, ax = plt.subplots(1, 4, figsize=(15.5, 4.0), gridspec_kw={"width_ratios": [1, 1, 1.25, 1.25]}, layout="constrained")
    lvl = float(np.median(net[cen]))
    ax[0].imshow(raws[0] - dark64, cmap="gray", vmin=0.5 * lvl, vmax=1.12 * lvl)
    ax[0].set_title("均匀目标的原图：渐晕和尘点")
    lvl2 = float(np.median(cm[cen]))
    ax[1].imshow(correct(raws[0], flats[:16].mean(0), dark64), cmap="gray", vmin=0.5 * lvl2, vmax=1.12 * lvl2)
    ax[1].set_title("平场校正后（16 帧平场）")
    for a in ax[:2]:
        a.axhline(70, color=S.ORANGE, lw=0.8, ls="--")
        S.image_axes(a)
    cols = np.arange(H)
    ax[2].plot(cols, net[70] / lvl, color=S.ORANGE, lw=1.6, label="校正前")
    ax[2].plot(cols, cm[70] / lvl2, color=S.BLUE, lw=1.6, label="校正后")
    ax[2].set_xlabel("列（像素）")
    ax[2].set_ylabel("亮度（相对中心）")
    ax[2].set_ylim(0.55, 1.1)
    ax[2].set_title("第 70 行的亮度（256 帧平均）")
    ax[2].legend(fontsize=8.5, loc="lower center")
    S.clean(ax[2])
    for nf, color, marker in ((1, S.RED, "s"), (16, S.ORANGE, "^"), (256, S.BLUE, "o")):
        ax[3].plot(Ks, resid[nf], marker + "-", color=color, lw=1.7, ms=5, label=f"平场用 {nf} 帧")
    ax[3].plot(Ks, ideal, ":", color=S.MUTED, lw=1.8, label="平场没有噪声时")
    ax[3].set_xscale("log")
    ax[3].set_yscale("log")
    ax[3].set_xticks(Ks, [str(k) for k in Ks])
    ax[3].set_yticks([1, 2, 5, 10, 20, 30], ["1", "2", "5", "10", "20", "30"])
    ax[3].minorticks_off()
    ax[3].set_xlabel("平均多少帧校正后的图")
    ax[3].set_ylabel("剩下的空间噪声（ADU）")
    ax[3].set_title("平场带进来的噪声，多帧平均压不掉")
    ax[3].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[3])
    S.save(fig, CH, "fig06_平场校正")


# ---------------------------------------------------------------- 图 7：仿真和“真实相机”的差距
def fig_gap():
    b, npx = 6, 64
    n = b * npx
    E, read, gain, bits = 4000.0, 5.0, 1.0, 12
    zreal, contr, k = 0.6, 0.6, 5.0
    nimg, ngood = 400, 100
    refl = np.full((n, n), 0.2)
    refl[: n // 2] = 0.8
    psfs = {"none": None, "focus": airy_psf(n), "real": pupil_psf(zreal / 2, n_out=n)}
    for key in ("focus", "real"):
        psfs[key] = psfs[key] / psfs[key].sum()

    def to_pixels(fine):
        return binning(fine, b) / b ** 2 * E

    bgs = {key: to_pixels(refl if p is None else fftconv(refl, p)) for key, p in psfs.items()}
    sig_const = float(np.sqrt((bgs["real"] + read ** 2).mean()))
    valid = np.zeros((npx, npx), bool)
    valid[3:29, 3:61] = True
    valid[35:61, 3:61] = True

    drng = np.random.default_rng(11)
    truths, incs = [], {key: [] for key in psfs}
    for _ in range(nimg):
        inc, truth = np.zeros((n, n)), []
        for region, lo, base in (("亮区", 6, 0.8), ("暗区", 38, 0.2)):
            used = []
            for _ in range(2):
                while True:
                    py, px = drng.uniform(lo, lo + 20), drng.uniform(6, 58)
                    if all(abs(px - q) > 10 for q in used):
                        break
                used.append(px)
                d = drng.uniform(0.3, 0.6)
                inc += contr * base * disk(d, py * b + (b - 1) / 2, px * b + (b - 1) / 2, n)
                truth.append((py, px, region))
        truths.append(truth)
        for key, p in psfs.items():
            incs[key].append(to_pixels(inc if p is None else fftconv(inc, p)))

    def capture(mean_e, kind, rng):
        if kind == "poisson":
            return mod.simulate_capture(mean_e, read, gain, bits, rng).astype(np.float64)
        return np.clip(np.round((mean_e + rng.normal(scale=sig_const, size=mean_e.shape)) / gain), 0, 2 ** bits - 1)

    def centroids(mask):
        cnt, _, _, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        return [(float(cen[i][1]), float(cen[i][0])) for i in range(1, cnt)]

    def template(bg, kind, rng, scale=1.0):
        goods = np.stack([capture(bg * scale, kind, rng) for _ in range(ngood)])
        return goods.mean(0), np.maximum(goods.std(0, ddof=1), 0.5)

    def score(images, ref, sig, kk, norm=False):
        hit, tot, fa, over = {"亮区": 0, "暗区": 0}, {"亮区": 0, "暗区": 0}, 0, 0
        for img, truth in images:
            if norm:
                img = img / ((img * ref).sum() / (ref * ref).sum())
            mask = (np.abs(img - ref) / sig > kk) & valid
            cents = centroids(mask)
            used = set()
            for py, px, region in truth:
                near = [i for i, c in enumerate(cents) if math.hypot(c[0] - py, c[1] - px) <= 2.0]
                tot[region] += 1
                hit[region] += bool(near)
                used.update(near)
            fa += len(cents) - len(used)
        return {r: hit[r] / tot[r] for r in hit}, fa / len(images)

    sources = {
        "不模糊 + 恒定的高斯噪声": ("none", "gauss"),
        "对焦模糊 + 恒定的高斯噪声": ("focus", "gauss"),
        "对焦模糊 + 泊松和读出噪声": ("focus", "poisson"),
        "离焦模糊 + 泊松和读出噪声": ("real", "poisson"),
        "“真实相机”": ("real", "poisson"),
    }
    result, crops, sig_at = {}, {}, {}
    y0, x0, _ = truths[0][0]
    cy0, cx0 = int(round(y0)), int(round(x0))
    for i, (name, (bk, nk)) in enumerate(sources.items()):
        rng = np.random.default_rng(100 + i)
        ref, sig = template(bgs[bk], nk, rng)
        images = [(capture(bgs[bk] + ie, nk, rng), t) for ie, t in zip(incs[bk], truths)]
        cap, fa = score(images, ref, sig, k)
        result[name] = {"亮区检出率": r3(cap["亮区"], 3), "暗区检出率": r3(cap["暗区"], 3),
                        "全部缺陷的检出率": r3((cap["亮区"] + cap["暗区"]) / 2, 3), "每张图误报": r3(fa, 3)}
        crops[name] = images[0][0][cy0 - 7:cy0 + 8, cx0 - 7:cx0 + 8]
        sig_at[name] = {"亮区": r3(np.median(sig[6:26, 6:58]), 1), "暗区": r3(np.median(sig[38:58, 6:58]), 1)}
    real = result["“真实相机”"]
    nreg = 2 * nimg
    se = {r: r3(math.sqrt(real[f"{r}检出率"] * (1 - real[f"{r}检出率"]) / nreg), 3) for r in ("亮区", "暗区")}
    peaks = {key: {"亮区": r3(np.mean([ie[6:26].max() for ie in incs[key]]), 0), "暗区": r3(np.mean([ie[38:58].max() for ie in incs[key]]), 0)} for key in psfs}

    # 灯变暗 30%
    rng = np.random.default_rng(55)
    bg = bgs["real"]
    ref, sig = template(bg, "poisson", rng)
    nv = int(valid.sum())
    lamp = {}

    def over_pixels(scale, r, s, kk, norm):
        cnt = 0
        for _ in range(200):
            img = capture(bg * scale, "poisson", rng)
            if norm:
                img = img / ((img * r).sum() / (r * r).sum())
            cnt += int(((np.abs(img - r) / s > kk) & valid).sum())
        return cnt / 200

    dim = 0.7
    ref_d, sig_d = template(bg, "poisson", rng, scale=dim)
    conds = {
        "灯没有变暗": (1.0, ref, sig, False),
        "变暗 30%，不做亮度归一化": (dim, ref, sig, False),
        "变暗 30%，归一化，sigma 沿用原来的": (dim, ref, sig, True),
        "变暗 30%，在新亮度下重建模板和 sigma": (dim, ref_d, sig_d, False),
    }
    for name, (scale, r, s, norm) in conds.items():
        images = [(capture((bg + ie) * scale, "poisson", rng), t) for ie, t in zip(incs["real"][:200], truths[:200])]
        cap, _ = score(images, r, s, 4.0, norm=norm)
        lamp[name] = {"良品图上每张图超过 k=4 的像素数": r3(over_pixels(scale, r, s, 4.0, norm), 2),
                      "亮区检出率": r3(cap["亮区"], 3), "暗区检出率": r3(cap["暗区"], 3)}

    NUM["仿真与真实的差距"] = {
        "说明": f"64×64 像素，像素 {b * DX:.1f} {U}；上半是亮区（反射率 0.8，约 {0.8 * E:.0f} e-），下半是暗区（0.2，约 {0.2 * E:.0f} e-）；"
              f"“真实相机”是一台离焦 {zreal} λ/NA² 的相机的完整仿真（艾里斑加离焦、像素积分、泊松噪声、读出噪声 {read:.0f} e-、{bits} 位）；"
              f"缺陷是直径 0.3 到 0.6 {U} 的圆，比所在区域亮 {contr:.0%}，位置随机；{nimg} 张图，每张亮区、暗区各 2 个，每个区域共 {nreg} 个缺陷；"
              f"检测方法固定：{ngood} 张良品求均值和标准差，z 超过 {k:.0f} 的连通域算一个报警，质心离真值 2 像素内算命中",
        "恒定高斯噪声的标准差（e-，取“真实相机”全图方差的平均）": r3(sig_const, 1),
        "各种仿真预测的检出率和“真实相机”上的检出率": result,
        "“真实相机”检出率的标准误差（二项分布）": se,
        "良品估出的 sigma 中位数（e-）": sig_at,
        "缺陷最亮像素平均多出的电子数": {"不模糊": peaks["none"], "对焦模糊": peaks["focus"], "离焦模糊": peaks["real"]},
        "灯变暗 30%": {
            "说明": f"同一台“真实相机”和同一批缺陷的前 200 张图，k=4；良品图上的超限像素数在 200 张图上统计，每张图参与统计的像素 {nv} 个；"
                  "归一化指用最小二乘求一个整体增益，把待检图除回模板的亮度",
            "按条件": lamp,
            "散粒噪声为主时，归一化后 z 被放大的倍数 1/sqrt(0.7)": r3(1 / math.sqrt(dim), 3),
        },
    }

    fig = plt.figure(figsize=(15, 4.2), layout="constrained")
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 2.6])
    short = {"不模糊 + 恒定的高斯噪声": "不模糊\n恒定高斯噪声", "对焦模糊 + 恒定的高斯噪声": "对焦模糊\n恒定高斯噪声",
             "对焦模糊 + 泊松和读出噪声": "对焦模糊\n泊松和读出噪声", "“真实相机”": "“真实相机”\n（离焦 0.6）"}
    vmax = float(max(c.max() for c in crops.values()))
    vmin = float(min(c.min() for c in crops.values()))
    for i, name in enumerate(short):
        a = fig.add_subplot(gs[i])
        a.imshow(crops[name], cmap="gray", vmin=vmin, vmax=vmax)
        a.set_title(short[name], fontsize=10)
        S.image_axes(a)
    a = fig.add_subplot(gs[4])
    names = list(sources)
    colors = [S.RED, S.ORANGE, S.PURPLE, S.BLUE, S.INK]
    width = 0.16
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = [result[name]["亮区检出率"], result[name]["暗区检出率"]]
        pos = [g + (i - 2) * width for g in (0, 1)]
        bars = a.bar(pos, vals, width=width * 0.92, color=color, alpha=0.9, label=name)
        for bar, v in zip(bars, vals):
            a.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{100 * v:.0f}", ha="center", fontsize=8.5)
    a.set_xticks([0, 1], ["亮区的缺陷", "暗区的缺陷"])
    a.set_ylabel("检出率")
    a.set_ylim(0, 1.12)
    a.set_title("仿真少一环，预测的检出率就偏一截（柱上数字是百分比）")
    a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    S.clean(a)
    a.grid(axis="x", visible=False)
    S.save(fig, CH, "fig07_仿真与真实的差距")


# ---------------------------------------------------------------- 没有配图的数字
NB_DIAMS = (0.5, 1, 1.5, 2, 3, 4, 6)


def nb_disk(diam_px, size=41, sub=16):
    """Notebook 实验 1 的画法：圆心在中间那个像素的中心，返回每个像素被圆覆盖的比例。"""
    c = (size - 1) / 2
    t = (np.arange(size * sub) + 0.5) / sub - 0.5
    inside = (t[:, None] - c) ** 2 + (t[None, :] - c) ** 2 <= (diam_px / 2) ** 2
    return inside.reshape(size, sub, size, sub).mean(axis=(1, 3))


def numbers_misc():
    """讲义里的算例、TDI 和设计问题用到的数字。"""
    ex = {f"{lam} nm、NA {na}": {"λ/NA（µm）": r3(lam / 1000 / na, 3),
                               "瑞利分辨率（µm）": r3(mod.rayleigh_resolution_um(lam, na), 3),
                               "分辨率的一半（µm）": r3(mod.rayleigh_resolution_um(lam, na) / 2, 3),
                               "严格奈奎斯特的像素 λ/(4NA)（µm）": r3(lam / 1000 / (4 * na), 3),
                               "λ/NA²（µm）": r3(lam / 1000 / na ** 2, 3)}
          for lam, na in ((532, 0.5), (532, 0.9), (266, 0.9), (193, 0.95))}
    d_u = 0.05 / (0.266 / 0.9)
    yy, xx = np.mgrid[0:41, 0:41]
    weak_e = NUM["成像链路逐级"]["按颗粒直径"]["0.2"]["该像素比背景多出的电子数"]
    NUM["讲义里的算例"] = {
        "说明": "瑞利分辨率调用作业的 rayleigh_resolution_um；其余按讲义里的公式直接计算",
        "瑞利分辨率": ex,
        "波长从 532 nm 换成 266 nm（NA 不变）": {"分辨率数值变为原来的": 0.5, "λ/NA² 变为原来的": 0.5},
        "NA 从 0.5 提到 0.9（波长不变）": {"分辨率数值变为原来的": r3(0.5 / 0.9, 2), "λ/NA² 变为原来的": r3((0.5 / 0.9) ** 2, 2)},
        "50 nm 的颗粒在 266 nm、NA 0.9 下": {"直径（λ/NA）": r3(d_u, 3), "峰值对比度 ÷ 物面对比度": r3(airy_encircled(d_u / 2)[0], 3)},
        "艾里斑第一暗环对应的 v = 2π·r·NA/λ（J1 的第一个零点）": r3(2 * math.pi * FIRST_ZERO, 4),
        "266 nm、NA 0.9 时离焦 0.5 λ/NA² 对应的距离（µm）": r3(0.5 * 0.266 / 0.9 ** 2, 3),
        "背景亮和背景暗的对照": {
            "说明": "图 1 里直径 0.2 的颗粒在最亮像素多出的电子数，分别放在图 1 的背景上和全暗的背景上（假设暗场下收到同样多的电子），读出噪声 5 e-；"
                  "信噪比 = 信号 ÷ sqrt(背景 + 读出噪声²)，背景为 0 时分母里换成信号自己的散粒噪声",
            "信号（e-）": weak_e, "背景 360 e- 时的信噪比": r3(weak_e / math.sqrt(360 + 25), 2), "背景为 0 时的信噪比": r3(weak_e / math.sqrt(weak_e + 25), 2),
        },
        "均值 1000 e- 的散粒噪声标准差": r3(math.sqrt(1000), 1),
        "均值 400 e-、读出噪声 5 e- 的信噪比": r3(400 / math.sqrt(425), 1),
        "作业验收里 1000 e-、读出噪声 5 e-、增益 2 的方差（ADU²，含量化项 1/12）": r3((1000 + 25) / 4 + 1 / 12, 2),
        "Notebook 实验 1 的 PSF sigma 1.5 像素，按对焦的高斯近似折算的像素尺寸（λ/NA）": r3(SIG_FOCUS / 1.5, 3),
        "Notebook 实验 1 的缺陷画法": {
            "说明": "41×41 像素，圆心在像素 (20, 20) 的中心，直径的单位是像素。只看像素中心：像素中心到圆心的距离不超过半径就整个置亮（Notebook 原来的画法）。"
                  "按覆盖比例：把每个像素切成 16×16 个小格，取落在圆里的小格所占的比例（Notebook 现在的画法，函数 nb_disk 和 Notebook 里的 disk_coverage 相同）",
            "只看像素中心：各直径置亮的像素数": {f"{d:g}": int((((yy - 20) ** 2 + (xx - 20) ** 2) <= (d / 2) ** 2).sum()) for d in NB_DIAMS},
            "按覆盖比例：各直径的覆盖面积（像素）": {f"{d:g}": r3(nb_disk(d).sum(), 3) for d in NB_DIAMS},
            "圆的面积 π·d²/4（像素）": {f"{d:g}": r3(math.pi * d * d / 4, 3) for d in NB_DIAMS},
            "按覆盖比例：圆心所在像素的覆盖比例": {f"{d:g}": r3(nb_disk(d)[20, 20], 3) for d in NB_DIAMS},
        },
        "颗粒远小于波长时的瑞利散射（教科书公式，强度正比于直径的 6 次方）：直径减半后散射强度剩下": f"1/{2 ** 6}",
    }

    # 顺序写反：先加噪声再模糊
    k = mod.gaussian_psf(1.5)
    rng = np.random.default_rng(30)
    m, v = [], []
    for lv in (50, 200, 500, 1000, 2000, 4000):
        f = mod.convolve_same(mod.simulate_capture(np.full((200, 200), float(lv)), 8.0, 1.5, 14, rng).astype(np.float64), k)
        m.append(float(f[10:-10, 10:-10].mean()))
        v.append(float(f[10:-10, 10:-10].var()))
    g_bad, _ = mod.estimate_gain_from_ptc(np.array(m), np.array(v))
    NUM["讲义里的算例"]["顺序写反：先加噪声再做 sigma 1.5 像素的模糊"] = {
        "说明": "验收脚本的 PTC 设置（读出噪声 8 e-、增益 1.5、14 位、6 档、200×200 像素），对加完噪声的图再卷积一次高斯 PSF",
        "PSF 各元素的平方和": r3(float((k ** 2).sum()), 4),
        "方差缩小到原来的（等于平方和）": r3(float((k ** 2).sum()), 4),
        "PTC 估出的增益": r3(g_bad, 1), "增益真值": 1.5, "估计值 ÷ 真值": r3(g_bad / 1.5, 0),
    }

    # TDI：级数和速度失配
    n, b = 256, 6
    spot = defect_spot(n)
    c0 = n // 2
    spot = fshift(spot, (n // 2 // b) * b + (b - 1) / 2 - c0, (n // 2 // b) * b + (b - 1) / 2 - c0)
    fx = np.fft.fftfreq(n)[None, :]
    bg_e, read = PIX_F * (b * DX) ** 2, 5.0
    base = float(binning(spot, b).max())
    snr1 = base / math.sqrt(bg_e + read ** 2)
    stages, mism = [1, 16, 64, 256], [0.0, 0.005, 0.01, 0.02]
    table, keep = {}, {}
    for eps in mism:
        row, kept = [], []
        for ns in stages:
            cells = ns * eps * b
            smeared = np.real(np.fft.ifft2(np.fft.fft2(spot) * np.sinc(fx * cells))) if cells > 0 else spot
            peak = float(binning(smeared, b).max())
            row.append(ns * peak / math.sqrt(ns * bg_e + read ** 2) / snr1)
            kept.append(peak / base)
        table[f"{100 * eps:g}%"] = [r3(v, 2) for v in row]
        keep[f"{100 * eps:g}%"] = [r3(v, 3) for v in kept]
    rng = np.random.default_rng(31)
    a = mod.simulate_capture(np.full(20000, 64 * (bg_e + base)), read, 4.0, 16, rng).astype(np.float64)
    c = mod.simulate_capture(np.full(200000, 64 * bg_e), read, 4.0, 16, rng).astype(np.float64)
    NUM["TDI 的级数和速度失配"] = {
        "说明": f"图 4 的缺陷（直径 {PIX_D} {U}，在像素中心），像素 {b * DX:.1f} {U}，每级背景 {bg_e:.0f} e-，读出噪声 {read:.0f} e- 只在最后加一次；"
              "速度失配 ε 时，N 级累加相当于沿扫描方向做长度 N·ε 个像素的均匀拖影；只算了这一项，没有算级间转移和振动",
        "单级的缺陷信噪比": r3(snr1, 2),
        "级数": stages,
        "拖影长度（像素）": {f"{100 * eps:g}%": [r3(ns * eps, 2) for ns in stages] for eps in mism},
        "缺陷信噪比相对单级的倍数（按速度失配）": table,
        "缺陷最亮像素保留的信号比例（按速度失配）": keep,
        "级数的平方根": [r3(math.sqrt(ns), 1) for ns in stages],
        "用 simulate_capture 核对 64 级、无失配的倍数": r3((a.mean() - c.mean()) / c.std() / snr1, 2),
    }


def main() -> None:
    S.setup()
    plt.rcParams["svg.hashsalt"] = "ch22"   # 固定 SVG 里的元素 id，重复运行逐字节相同
    for fn in (fig_chain, fig_resolution, fig_psf_mtf, fig_pixel, fig_noise, fig_flat, fig_gap):
        fn()
        print("完成", fn.__name__, flush=True)
    numbers_misc()
    order = ["成像链路逐级", "分辨率与小缺陷", "PSF、MTF 与离焦", "像素尺寸", "噪声与 PTC", "信噪比随曝光量", "量化位数",
             "平场校正", "TDI 的级数和速度失配", "仿真与真实的差距", "讲义里的算例"]
    data = {k: NUM[k] for k in order}
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False))


if __name__ == "__main__":
    main()
