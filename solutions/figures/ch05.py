"""第 5 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch05.py

输出：<主分支>/05_特征点/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。
Harris 响应用本章作业的参考实现，SIFT、ORB、匹配和 RANSAC 用 OpenCV。
随机种子全部固定，重复运行生成的 SVG 和 JSON 逐字节相同。“耗时”一组默认沿用上次的测量，
设置 CH05_RETIME=1 才重新计时（计时每次都会波动）。
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.dont_write_bytecode = True  # 脚本会加载主分支的验收脚本和参考实现，不在那些目录里留缓存文件
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

CH = "05_特征点"
mod = S.load_solution(CH, "5-1_Harris响应.py")
harris = mod.harris_response
NUM: dict = {}
VIEW = 400
V0 = (VIEW - 1) / 2.0
_CACHE: dict = {}


def r3(x, n=3):
    return round(float(x), n)


# ---------------------------------------------------------------- 合成场景
def make_master(seed=0, size=2048):
    """2 倍分辨率的母版：随机的矩形、多边形、圆和文字，叠在缓慢变化的明暗上。"""
    if ("poly", seed) in _CACHE:
        return _CACHE[("poly", seed)]
    rng = np.random.default_rng(seed)
    low = rng.normal(size=(size // 64, size // 64)).astype(np.float32)
    low = cv2.resize(low, (size, size), interpolation=cv2.INTER_CUBIC)
    img = np.clip(110 + 18 * low, 0, 255).astype(np.uint8)
    for _ in range(260):
        kind = rng.integers(0, 4)
        cx, cy = rng.integers(0, size, 2)
        g = int(rng.integers(20, 236))
        s = int(rng.choice([14, 20, 28, 40, 56, 80, 110, 150]))
        if kind == 0:
            box = cv2.boxPoints(((float(cx), float(cy)), (float(s), float(s * rng.uniform(0.4, 1.6))), float(rng.uniform(0, 180))))
            cv2.fillConvexPoly(img, np.round(box * 16).astype(np.int32), g, lineType=cv2.LINE_AA, shift=4)
        elif kind == 1:
            a = np.sort(rng.uniform(0, 2 * np.pi, int(rng.integers(3, 6))))
            pts = np.stack([cx + s * 0.7 * np.cos(a), cy + s * 0.7 * np.sin(a)], 1)
            cv2.fillConvexPoly(img, np.round(pts * 16).astype(np.int32), g, lineType=cv2.LINE_AA, shift=4)
        elif kind == 2:
            cv2.circle(img, (int(cx), int(cy)), int(s * 0.45), g, -1, lineType=cv2.LINE_AA)
        else:
            txt = "".join(rng.choice(list("ABCDEFGHKMNPRSTXYZ23456789"), 4))
            cv2.putText(img, txt, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, s / 28.0, g, max(2, s // 12), cv2.LINE_AA)
    _CACHE[("poly", seed)] = img
    return img


def make_texture_master(seed=0, size=2048, width=0.012):
    """对照用的平滑随机纹理：没有锐利的角，只有起伏。"""
    if ("tex", seed) in _CACHE:
        return _CACHE[("tex", seed)]
    rng = np.random.default_rng(seed)
    f = np.fft.fft2(rng.normal(size=(size, size)))
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    t = np.real(np.fft.ifft2(f * np.exp(-(fy ** 2 + fx ** 2) / (2 * width ** 2))))
    img = np.clip(128 + 45 * (t - t.mean()) / t.std(), 0, 255).astype(np.uint8)
    _CACHE[("tex", seed)] = img
    return img


def _rot(theta):
    t = math.radians(theta)
    return np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])


def render(master, s=1.0, theta=0.0, gain=1.0, bias=0.0, noise=0.0, blur=0.0, seed=0, shift=(0.0, 0.0), view=VIEW):
    """从母版拍一张 view x view 的图。s 是放大倍数，theta 是旋转角，shift 是亚像素平移。

    视图坐标 v 对应母版坐标 c + (2/s) Q (v - v0 - shift)。缩小时先按采样率做抗混叠，
    再统一加标准差 0.8 px 的镜头模糊，所以各个倍率下图像的锐度（以像素计）相同。
    """
    c = (master.shape[0] - 1) / 2.0
    f = 2.0 / s
    pre = 0.5 * math.sqrt(max(f * f - 1.0, 0.0))
    src = cv2.GaussianBlur(master, (0, 0), pre) if pre > 0.05 else master
    A = f * _rot(theta)
    v0 = (view - 1) / 2.0
    b = np.array([c, c]) - A @ (np.array([v0, v0]) + np.asarray(shift, float))
    out = cv2.warpAffine(src, np.hstack([A, b[:, None]]), (view, view), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    out = cv2.GaussianBlur(out, (0, 0), 0.8)
    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    out = out * gain + bias
    if noise > 0:
        out = out + np.random.default_rng(seed).normal(scale=noise, size=out.shape).astype(np.float32)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def to_b(pts, s=1.0, theta=0.0, shift=(0.0, 0.0)):
    """基准视图（s=1、theta=0）里的点在变换后视图里的坐标。"""
    return V0 + s * (np.asarray(pts, float).reshape(-1, 2) - V0) @ _rot(theta) + np.asarray(shift, float)


def from_b(pts, s=1.0, theta=0.0, shift=(0.0, 0.0)):
    return V0 + (np.asarray(pts, float).reshape(-1, 2) - V0 - np.asarray(shift, float)) @ _rot(theta).T / s


# ---------------------------------------------------------------- Harris 相关工具
def tensor(img, w=5, gauss=None):
    """结构张量的三个分量（float64）。gauss 给定时用高斯窗，否则用 w x w 方框窗。"""
    g = np.asarray(img, np.float64)
    ix = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    iy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    if gauss:
        f = lambda a: cv2.GaussianBlur(a, (0, 0), gauss)  # noqa: E731
    else:
        f = lambda a: cv2.boxFilter(a, cv2.CV_64F, (w, w), normalize=True)  # noqa: E731
    return f(ix * ix), f(iy * iy), f(ix * iy)


def eigs(sxx, syy, sxy):
    half, disc = (sxx + syy) / 2, np.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)
    return half + disc, np.maximum(half - disc, 0.0)


def harris_gauss(img, k, sigma):
    sxx, syy, sxy = tensor(img, gauss=sigma)
    return (sxx * syy - sxy * sxy - k * (sxx + syy) ** 2).astype(np.float32)


def nms_points(resp, n=100, radius=4, rel=None, border=8, thresh=None):
    """响应图 -> 角点 (x, y)。先取 (2r+1) 邻域内的正的局部极大，再按阈值或前 n 个筛选。"""
    size = 2 * radius + 1
    mask = (resp == cv2.dilate(resp, np.ones((size, size), np.uint8))) & (resp > 0)
    mask[:border] = mask[-border:] = False
    mask[:, :border] = mask[:, -border:] = False
    ys, xs = np.nonzero(mask)
    v = resp[ys, xs]
    order = np.argsort(-v, kind="stable")
    ys, xs, v = ys[order], xs[order], v[order]
    cut = rel * float(resp.max()) if rel is not None else thresh
    if cut is not None:
        keep = v > cut
        ys, xs, v = ys[keep], xs[keep], v[keep]
    if n is not None:
        ys, xs = ys[:n], xs[:n]
    return np.stack([xs, ys], 1).astype(float)


def corners(img, k=0.04, w=5, pre=0.0, gauss=None):
    """全部正的局部极大，按响应从大到小排。取前几个由 repeat_rate 在公共区域内决定。"""
    g = cv2.GaussianBlur(img, (0, 0), pre) if pre else img
    resp = harris_gauss(g, k, gauss) if gauss else harris(g, k, w)
    return nms_points(resp, n=None)


def repeat_rate(pa, pb, n=50, s=1.0, theta=0.0, shift=(0.0, 0.0), eps=2.0, margin=8, crop_first=True):
    """重复率：两张图各自只留公共区域里的点，再各取响应最强的 n 个，一对一配对，
    距离不超过 eps（按分辨率较低的那张图的像素算）。分母是两边参与点数里较少的一方。

    一定要先裁到公共区域再取前 n 个。如果每张图先在整幅视野里取前 n 个，视野更宽的那张图
    落在公共区域里的只剩最强的一小部分，拿它们去配另一张的 n 个点，重复率会被抬高。
    crop_first=False 就是这种有偏的做法，只用来量它把数字抬高了多少。"""
    if len(pa) == 0 or len(pb) == 0:
        return 0.0
    if not crop_first:
        pa, pb = np.asarray(pa, float)[:n], np.asarray(pb, float)[:n]
    ma = to_b(pa, s, theta, shift)
    ma = ma[np.all((ma >= margin) & (ma <= VIEW - 1 - margin), axis=1)]
    back = from_b(pb, s, theta, shift)
    pbv = np.asarray(pb, float)[np.all((back >= margin) & (back <= VIEW - 1 - margin), axis=1)]
    if crop_first:
        ma, pbv = ma[:n], pbv[:n]
    if len(ma) == 0 or len(pbv) == 0:
        return 0.0
    d = np.linalg.norm(ma[:, None, :] - pbv[None, :, :], axis=2)
    tol, used, hit = eps * max(1.0, s), set(), 0
    for i in np.argsort(d.min(1), kind="stable"):
        for j in np.argsort(d[i], kind="stable"):
            if d[i, j] > tol:
                break
            if j not in used:
                used.add(j)
                hit += 1
                break
    return hit / min(len(ma), len(pbv))


def wedge(phi, rot, n=64, dx=0.0, dy=0.0, fg=200.0, bg=60.0, blur=0.8, ss=8):
    """张角 phi、朝向 rot 的楔形，顶点在 ((n-1)/2 + dx, (n-1)/2 + dy)。8 倍超采样后按面积平均，真值精确。"""
    vx, vy = (n - 1) / 2 + dx, (n - 1) / 2 + dy
    c = (np.arange(n * ss) + 0.5) / ss - 0.5
    X, Y = np.meshgrid(c, c)
    d = (np.degrees(np.arctan2(Y - vy, X - vx)) - rot + 180) % 360 - 180
    hi = np.where(np.abs(d) <= phi / 2, fg, bg).astype(np.float32)
    lo = hi.reshape(n, ss, n, ss).mean(axis=(1, 3))
    return cv2.GaussianBlur(lo, (0, 0), blur) if blur > 0 else lo


def near_vertex(resp, n, dx, dy, radius=12):
    """只看顶点附近的圆内，避开楔形的边伸到图像边界处的假角点。"""
    y, x = np.mgrid[0:n, 0:n]
    return np.where(np.hypot(x - (n - 1) / 2 - dx, y - (n - 1) / 2 - dy) <= radius, resp, -np.inf)


PLACEMENTS = [(float(r), float(a), float(b)) for r, a, b in
              zip(np.random.default_rng(0).uniform(0, 360, 16), *np.random.default_rng(1).uniform(-0.5, 0.5, (2, 16)))]


# ---------------------------------------------------------------- 图 1：平坦、边缘、角点能不能定位
def fig_localize():
    img = np.full((120, 120), 60, np.float32)
    img[40:90, 40:90] = 180
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    spots = {"平坦": (65, 65), "边缘": (65, 40), "角点": (40, 40)}
    half, R, sigma, trials = 7, 8, 10.0, 200
    l1, l2 = eigs(*tensor(img, 2 * half + 1))
    resp = harris(img, 0.04, 2 * half + 1)
    rng = np.random.default_rng(3)
    errs = {name: [] for name in spots}
    for _ in range(trials):
        a = (img + rng.normal(scale=sigma, size=img.shape)).astype(np.float32)
        b = (img + rng.normal(scale=sigma, size=img.shape)).astype(np.float32)
        for name, (cy, cx) in spots.items():
            res = cv2.matchTemplate(b[cy - half - R:cy + half + R + 1, cx - half - R:cx + half + R + 1],
                                    a[cy - half:cy + half + 1, cx - half:cx + half + 1], cv2.TM_SQDIFF)
            y, x = np.unravel_index(np.argmin(res), res.shape)
            errs[name].append((x - R, y - R))
    errs = {k: np.array(v, float) for k, v in errs.items()}
    rms = lambda v: float(np.sqrt((v ** 2).mean()))  # noqa: E731
    NUM["平坦、边缘、角点能不能定位"] = {
        "说明": f"120x120 的图里一个亮方块（灰度 60 和 180，边缘模糊 1 px）；同一场景拍两次，各加标准差 {sigma:.0f} 的噪声；"
              f"从第一张取 {2 * half + 1}x{2 * half + 1} 的小块，在第二张的 ±{R} px 范围内找平方差最小的位置，真实位移为 0，共 {trials} 次。"
              "边缘那一块取在方块左侧的竖直边上，所以 x 是垂直于边的方向，y 是沿着边的方向",
        "在搜索范围内瞎猜时的均方根误差（px）": r3(math.sqrt(((2 * R + 1) ** 2 - 1) / 12), 2),
        "均方根误差（px），[x 方向, y 方向]": {k: [r3(rms(v[:, 0]), 2), r3(rms(v[:, 1]), 2)] for k, v in errs.items()},
        "两个方向都找对的比例": {k: r3((v == 0).all(1).mean()) for k, v in errs.items()},
        "小块的结构张量特征值 [大, 小]": {k: [round(float(l1[cy, cx])), round(float(l2[cy, cx]), 1)] for k, (cy, cx) in spots.items()},
        "小块中心的 Harris 响应（k=0.04，参考实现）": {k: float(f"{resp[cy, cx]:.3g}") for k, (cy, cx) in spots.items()},
    }
    _CACHE["spots"] = {k: (float(l1[cy, cx]), float(l2[cy, cx])) for k, (cy, cx) in spots.items()}

    fig, ax = plt.subplots(1, 4, figsize=(14.5, 3.9), gridspec_kw={"width_ratios": [1.05, 1, 1, 1]}, layout="constrained")
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    colors = {"平坦": S.MUTED, "边缘": S.ORANGE, "角点": S.BLUE}
    for name, (cy, cx) in spots.items():
        ax[0].add_patch(plt.Rectangle((cx - half - 0.5, cy - half - 0.5), 2 * half + 1, 2 * half + 1, fill=False, ec=colors[name], lw=2))
        ax[0].text(cx, cy - half - 3, name, color=colors[name], fontsize=10, ha="center", fontweight="bold",
                   bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.2})
    ax[0].set_title("同一场景里取三个小块")
    S.image_axes(ax[0])
    jit = np.random.default_rng(0)
    titles = {"平坦": "平坦：两个方向都定不住", "边缘": "边缘：沿着边的方向定不住", "角点": "角点：两个方向都定住了"}
    for a, name in zip(ax[1:], spots):
        e = errs[name] + jit.uniform(-0.3, 0.3, errs[name].shape)
        a.scatter(e[:, 0], e[:, 1], s=9, color=colors[name], alpha=0.55, linewidths=0)
        a.axhline(0, color=S.GRID, lw=0.8)
        a.axvline(0, color=S.GRID, lw=0.8)
        a.set_xlim(-R - 1, R + 1)
        a.set_ylim(R + 1, -R - 1)
        a.set_aspect("equal")
        a.set_xlabel("x 方向的定位误差（px）")
        a.set_title(titles[name])
        S.clean(a)
    ax[1].set_ylabel("y 方向的定位误差（px）")
    S.save(fig, CH, "fig01_平坦边缘角点的定位")


# ---------------------------------------------------------------- 图 2：k 和窗口大小
def r_max(k):
    return ((1 - 2 * k) + math.sqrt(1 - 4 * k)) / (2 * k)


def poly_image(pts, n, fg=200.0, bg=60.0, blur=0.8, ss=8):
    hi = np.zeros((n * ss, n * ss), np.uint8)
    cv2.fillConvexPoly(hi, np.round(((np.asarray(pts) + 0.5) * ss - 0.5) * 16).astype(np.int32), 1, shift=4)
    lo = bg + (fg - bg) * hi.reshape(n, ss, n, ss).mean(axis=(1, 3)).astype(np.float32)
    return cv2.GaussianBlur(lo, (0, 0), blur)


def seg_dist(p, a, b):
    t = np.clip(((p - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * (b - a)), axis=1)


def fig_k_window():
    ks = [0.02, 0.04, 0.06, 0.1, 0.15, 0.2, 0.24]
    NUM["k 的含义"] = {
        "说明": "R > 0 等价于两个特征值之比小于 r_max(k)，r_max 是 r + 1/r + 2 = 1/k 的大根。理想模型指窗口里只有两条等长直边、夹角为张角的情形，"
              "这时两个特征值正比于 1 ± |cos(张角)|，R > 0 等价于 sin(张角) > 2 sqrt(k)",
        "k": ks,
        "特征值之比的上限 r_max": [r3(r_max(k), 2) for k in ks],
        "理想模型下能得到正响应的最小张角（度）": [r3(math.degrees(math.asin(2 * math.sqrt(k))), 1) for k in ks],
        "k 趋于 0.25 时 r_max 趋于": 1.0,
    }

    # (b) 响应随张角
    phis = [15, 20, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
    kk = [0.04, 0.06, 0.1, 0.15]
    peak = np.zeros((len(phis), len(kk), len(PLACEMENTS)))
    for i, phi in enumerate(phis):
        for p, (rot, dx, dy) in enumerate(PLACEMENTS):
            img = wedge(phi, rot, dx=dx, dy=dy)
            for j, k in enumerate(kk):
                peak[i, j, p] = max(float(near_vertex(harris(img, k, 5), 64, dx, dy).max()), 0.0)
    ref = peak[phis.index(90), 0].mean()
    rel = peak.mean(2) / ref
    right = peak[phis.index(90), 0] / ref
    axis = float(near_vertex(harris(wedge(90, 45), 0.04, 5), 64, 0, 0).max()) / ref
    NUM["响应随张角的变化"] = {
        "说明": f"64x64 的楔形（灰度 60 和 200，模糊 0.8 px），窗口 5x5；每个张角取 {len(PLACEMENTS)} 种随机朝向和亚像素顶点位置，"
              "记录顶点附近 12 px 内的最大响应（负数按 0 计）再取平均；数值是相对“直角、k=0.04”这一格的比例",
        "张角（度）": phis,
        "相对响应": {f"k={k}": [r3(v) for v in rel[:, j]] for j, k in enumerate(kk)},
        "有正响应的摆放比例": {f"k={k}": [r3((peak[i, j] > 0).mean(), 2) for i in range(len(phis))] for j, k in enumerate(kk)},
        "同一个直角换朝向和亚像素位置，最大响应相对平均值的范围（k=0.04）": [r3(right.min(), 2), r3(right.max(), 2)],
        "两条边正好沿着像素行列、顶点落在像素角上时，直角的响应相对平均值": r3(axis, 2),
    }

    # (c) 带噪声的直边上的假角点
    sq = cv2.boxPoints(((63.5, 63.5), (70.0, 70.0), 20.0)).astype(float)
    clean_sq = poly_image(sq, 128)
    k_list = [0.01, 0.02, 0.04, 0.06, 0.1, 0.15, 0.2]
    false_edge = {5: [], 10: [], 20: []}
    false_flat = {5: [], 10: [], 20: []}
    found = {5: [], 10: [], 20: []}
    for nz in false_edge:
        for k in k_list:
            fe = ff = tp = 0
            for t in range(20):
                img = clean_sq + np.random.default_rng(100 + t).normal(scale=nz, size=clean_sq.shape).astype(np.float32)
                pts = nms_points(harris(img, k, 5), n=None, rel=0.01)
                if len(pts) == 0:
                    continue
                dc = np.linalg.norm(pts[:, None] - sq[None], axis=2)
                de = np.min([seg_dist(pts, sq[i], sq[(i + 1) % 4]) for i in range(4)], axis=0)
                far = dc.min(1) > 4
                fe += int((far & (de <= 3)).sum())
                ff += int((far & (de > 3)).sum())
                tp += int((dc.min(0) <= 4).sum())
            false_edge[nz].append(fe / 20)
            false_flat[nz].append(ff / 20)
            found[nz].append(tp / 20)
    NUM["k 与带噪声的直边"] = {
        "说明": "128x128 的图里一个转了 20 度的方块（边长 70，灰度 60 和 200，模糊 0.8 px），加高斯噪声，窗口 5x5；"
              "取响应大于最大值 1% 的局部极大（邻域半径 4），离四个真实顶点 4 px 以外的算假角点，按离边 3 px 以内和以外分开计数；每格 20 张图取平均",
        "k": k_list,
        "边上的假角点（每张图）": {f"噪声 {nz}": [r3(v, 2) for v in vals] for nz, vals in false_edge.items()},
        "平坦区的假角点（每张图）": {f"噪声 {nz}": [r3(v, 2) for v in vals] for nz, vals in false_flat.items()},
        "四个真实顶点里找到的个数": {f"噪声 {nz}": [r3(v, 2) for v in vals] for nz, vals in found.items()},
    }

    # (d) 窗口大小与定位偏移
    ws = [3, 5, 7, 9, 11, 15]
    off = {"方框窗": [], "方差相同的高斯窗": []}
    inside = []
    refine = {}
    for w in ws:
        db, dg, dr, din = [], [], [], []
        for rot, dx, dy in PLACEMENTS:
            img = wedge(90, rot, n=96, dx=dx, dy=dy)
            tx, ty = 47.5 + dx, 47.5 + dy
            for store, resp in ((db, harris(img, 0.04, w)), (dg, harris_gauss(img, 0.04, w / math.sqrt(12)))):
                y, x = np.unravel_index(np.argmax(near_vertex(resp, 96, dx, dy, 16)), resp.shape)
                store.append(math.hypot(x - tx, y - ty))
                if store is db:
                    ang = (math.degrees(math.atan2(y - ty, x - tx)) - rot + 180) % 360 - 180
                    din.append(abs(ang) <= 45)
                if store is db and w in (5, 9):
                    q = cv2.cornerSubPix(img, np.array([[[x, y]]], np.float32), (5, 5), (-1, -1),
                                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4))
                    dr.append(float(np.hypot(q[0, 0, 0] - tx, q[0, 0, 1] - ty)))
        off["方框窗"].append(float(np.mean(db)))
        off["方差相同的高斯窗"].append(float(np.mean(dg)))
        inside.append(float(np.mean(din)))
        if dr:
            refine[str(w)] = [r3(np.mean(dr), 2), r3(np.max(dr), 2)]
    scenes = [make_master(s) for s in range(6)]
    noisy = {"取最强的 50 个": {}, "取最强的 100 个": {}}
    for name, kw in [(f"方框窗 {w}x{w}", {"w": w}) for w in (3, 5, 7, 9, 11)] + \
                    [(f"先做标准差 {p} 的高斯平滑，再用方框窗 5x5", {"w": 5, "pre": p}) for p in (1.0, 1.5, 2.0)]:
        pairs = [(corners(render(m, noise=10, seed=10 + 10 * i + rep), **kw), corners(render(m, noise=10, seed=20 + 10 * i + rep + 5), **kw))
                 for i, m in enumerate(scenes) for rep in range(4)]
        for n in (50, 100):
            vals = [repeat_rate(a, b, n=n) for a, b in pairs]
            noisy[f"取最强的 {n} 个"][name] = [r3(np.mean(vals)), r3(np.std(vals) / math.sqrt(len(vals)))]
    NUM["窗口大小"] = {
        "说明": f"定位偏移：96x96 的直角楔形（模糊 0.8 px），{len(PLACEMENTS)} 种随机朝向和亚像素顶点位置，k=0.04，"
              "量响应最大的像素到真实顶点的距离再取平均；高斯窗的标准差取 w/sqrt(12)，和 w x w 方框窗的方差相同",
        "窗口边长 w": ws,
        "响应峰值到真实顶点的平均距离（px）": {k: [r3(v, 2) for v in vals] for k, vals in off.items()},
        "方框窗的峰值落在直角内部（顶点指向峰值的方向在两条边之间）的摆放比例": [r3(v, 2) for v in inside],
        "方框窗的峰值再用 cornerSubPix（搜索窗 11x11）修正后的误差 [平均, 最大]（px）": refine,
        "噪声下的重复率": {
            "说明": "6 个 400x400 的合成场景，同一场景拍两次各加标准差 10 的噪声，每个场景 4 组，共 24 对图；2 px 内算重复；每格是 [平均, 平均值的标准误差]",
            **noisy},
    }

    fig, ax = plt.subplots(1, 4, figsize=(15.5, 4.1), layout="constrained")
    lim = 18.0
    xs = np.linspace(0, lim, 50)
    for k, color in ((0.04, S.BLUE), (0.1, S.ORANGE), (0.2, S.RED)):
        ax[0].plot(xs, xs / r_max(k), color=color, lw=1.6, label=f"k={k}：比值上限 {r_max(k):.1f}")
        ax[0].plot(xs / r_max(k), xs, color=color, lw=1.6)
    ax[0].fill_between(xs, xs / r_max(0.04), np.minimum(xs * r_max(0.04), lim), color=S.BLUE, alpha=0.08)
    offsets = {"平坦": (30, 62), "边缘": (-34, 22), "角点": (6, 7)}
    for name, (a, b) in _CACHE["spots"].items():
        ax[0].plot(a / 1000, b / 1000, "o", color=S.INK, ms=6, clip_on=False, zorder=5)
        ax[0].annotate(f"图 1 的{name}", (a / 1000, b / 1000), textcoords="offset points", xytext=offsets[name], fontsize=9,
                       arrowprops={"arrowstyle": "-", "color": S.MUTED, "lw": 0.8})
    ax[0].text(11.5, 13.5, "R > 0\n角点", color=S.BLUE, fontsize=10, ha="center")
    ax[0].text(13.2, 3.3, "k=0.2 时这里\n已经 R < 0", color=S.RED, fontsize=8.8, ha="center")
    ax[0].set_xlim(0, lim)
    ax[0].set_ylim(0, lim)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("大的特征值（千）")
    ax[0].set_ylabel("小的特征值（千）")
    ax[0].set_title("k 决定“多扁还算角点”")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax[0])

    for j, (k, color) in enumerate(zip(kk, (S.BLUE, S.GREEN, S.ORANGE, S.RED))):
        ax[1].plot(phis, rel[:, j], "o-", color=color, lw=1.7, ms=4, label=f"k={k}")
    ax[1].set_xticks([15, 45, 90, 135, 165])
    ax[1].set_xlabel("楔形的张角（度）")
    ax[1].set_ylabel("顶点处的最大响应（直角、k=0.04 为 1）")
    ax[1].set_title("钝角比同样“扁”的锐角先丢")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    S.clean(ax[1])

    for nz, color, marker in ((5, S.GREEN, "o"), (10, S.ORANGE, "s"), (20, S.RED, "^")):
        ax[2].plot(range(len(k_list)), false_edge[nz], marker + "-", color=color, lw=1.7, ms=4.5, label=f"噪声 {nz}：边上")
    ax[2].plot(range(len(k_list)), false_flat[20], "^--", color=S.RED, lw=1.3, ms=4.5, label="噪声 20：平坦区")
    ax[2].set_xticks(range(len(k_list)), [str(k) for k in k_list])
    ax[2].set_xlabel("k")
    ax[2].set_ylabel("假角点个数（每张图）")
    ax[2].set_title("k 只管得住边上的假角点")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])

    ax[3].plot(ws, off["方框窗"], "s-", color=S.ORANGE, lw=1.8, ms=5, label="方框窗")
    ax[3].plot(ws, off["方差相同的高斯窗"], "o-", color=S.BLUE, lw=1.8, ms=5, label="方差相同的高斯窗")
    ax[3].set_xticks(ws)
    ax[3].set_xlabel("窗口边长 w（px）")
    ax[3].set_ylabel("响应峰值到真实顶点的距离（px）")
    ax[3].set_ylim(bottom=0)
    ax[3].set_title("方框窗越大，峰值离顶点越远")
    ax[3].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[3])
    S.save(fig, CH, "fig02_k与窗口大小")


# ---------------------------------------------------------------- 图 3：Harris 角点的重复率
def fig_repeat():
    N = 50
    scenes = [make_master(s) for s in range(6)]
    texes = [make_texture_master(s) for s in range(3)]
    base = [corners(render(m)) for m in scenes]
    tbase = [corners(render(m)) for m in texes]

    def sweep(masters, bases, key, values, **kw):
        out = []
        for v in values:
            out.append([repeat_rate(pa, corners(render(m, **{key: v})), n=N, **{key: v}, **kw) for m, pa in zip(masters, bases)])
        return np.array(out)

    thetas = [0, 5, 10, 15, 20, 30, 45, 60, 75, 90]
    scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
    rot, rot_t = sweep(scenes, base, "theta", thetas), sweep(texes, tbase, "theta", thetas)
    sca, sca_t = sweep(scenes, base, "s", scales), sweep(texes, tbase, "s", scales)
    biased = sweep(scenes, base, "s", scales, crop_first=False)
    biased_t = sweep(texes, tbase, "s", scales, crop_first=False)
    half_px = [repeat_rate(pa, corners(render(m, shift=(0.5, 0.5))), n=N, shift=(0.5, 0.5)) for m, pa in zip(scenes, base)]
    half_px_t = [repeat_rate(pa, corners(render(m, shift=(0.5, 0.5))), n=N, shift=(0.5, 0.5)) for m, pa in zip(texes, tbase)]
    chance = [repeat_rate(base[i], base[(i + 1) % len(base)], n=N) for i in range(len(base))]

    gains = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    n_fixed, n_rel, rep_top = [], [], []
    resp0 = [harris(render(m), 0.04, 5) for m in scenes]
    for g in gains:
        a = b = c = 0.0
        for m, r0, pa in zip(scenes, resp0, base):
            rg = harris(render(m, gain=g), 0.04, 5)
            a += len(nms_points(rg, n=None, thresh=0.01 * float(r0.max())))
            b += len(nms_points(rg, n=None, rel=0.01))
            c += repeat_rate(pa, nms_points(rg, n=None), n=N)
        n_fixed.append(a / len(scenes))
        n_rel.append(b / len(scenes))
        rep_top.append(c / len(scenes))
    m0 = scenes[0]
    ratio_half = float(harris(render(m0).astype(np.float32) * 0.5, 0.04, 5).max() / resp0[0].max())
    strong = render(m0).copy()
    strong[20:44, 20:44] = 0
    strong[20:32, 20:32] = 255
    strong[32:44, 32:44] = 255
    n_before = len(nms_points(resp0[0], n=None, rel=0.01))
    n_after = len(nms_points(harris(strong, 0.04, 5), n=None, rel=0.01))

    noises = [0, 2, 5, 10, 15, 20]
    noise_rows = {"取最强的 50 个": {"n": 50}, "取最强的 25 个": {"n": 25}, "取最强的 200 个": {"n": 200},
                  "取最强的 50 个，先做标准差 1.5 的高斯平滑": {"n": 50, "pre": 1.5}}
    noise_out = {name: [] for name in noise_rows}
    for nz in noises:
        for name, kw in noise_rows.items():
            vals = []
            for i, m in enumerate(scenes):
                for rep in range(2):
                    a = corners(render(m, noise=nz, seed=1000 + 10 * i + rep), pre=kw.get("pre", 0.0))
                    b = corners(render(m, noise=nz, seed=2000 + 10 * i + rep), pre=kw.get("pre", 0.0))
                    vals.append(repeat_rate(a, b, n=kw["n"]))
            noise_out[name].append(float(np.mean(vals)))

    stat = lambda arr: {"平均": [r3(v) for v in arr.mean(1)], "最小": [r3(v) for v in arr.min(1)], "最大": [r3(v) for v in arr.max(1)]}  # noqa: E731
    span = lambda v: {"平均": r3(np.mean(v)), "最小": r3(np.min(v)), "最大": r3(np.max(v))}  # noqa: E731
    NUM["Harris 角点的重复率"] = {
        "说明": f"400x400 的合成图，参考实现算响应（k=0.04，窗口 5x5），取邻域半径 4 内正的局部极大；两张图各自只留公共区域里的点，再各取响应最强的 {N} 个；"
              "重复率 = 一对一配上的点数 / 两边参与点数里较少的一方，容差 2 px（按分辨率较低的那张图的像素算）；"
              "多边形场景 6 个（矩形、多边形、圆和文字），平滑纹理场景 3 个（带限随机纹理，没有锐利的角）",
        "基准图里正的局部极大个数（多边形场景，6 个场景的最小和最大）": [min(len(b) for b in base), max(len(b) for b in base)],
        "旋转角（度）": thetas,
        "旋转：多边形场景": stat(rot),
        "旋转：平滑纹理场景的平均": [r3(v) for v in rot_t.mean(1)],
        "放大倍数": scales,
        "缩放：多边形场景": stat(sca),
        "缩放：平滑纹理场景的平均": [r3(v) for v in sca_t.mean(1)],
        "只平移半个像素时的重复率": {"多边形场景": span(half_px), "平滑纹理场景": span(half_px_t)},
        "两张不相干的图之间的“重复率”（碰巧重合的水平）": r3(np.mean(chance)),
        "统计口径的影响": {
            "说明": f"同样的检测结果，换一种有偏的统计方法：每张图先在整幅视野里取最强的 {N} 个，再裁到公共区域去配对。"
                  "放大或缩小之后两张图的视野不一样大，视野宽的那张留在公共区域里的只剩最强的少数几个点，重复率被抬高",
            "放大倍数": scales,
            "多边形场景，先取点再裁剪（有偏）": [r3(v) for v in biased.mean(1)],
            "多边形场景，先裁剪再取点（本章的做法）": [r3(v) for v in sca.mean(1)],
            "平滑纹理场景，先取点再裁剪（有偏）": [r3(v) for v in biased_t.mean(1)],
            "平滑纹理场景，先裁剪再取点（本章的做法）": [r3(v) for v in sca_t.mean(1)],
        },
        "亮度": {
            "说明": "图像整体乘一个增益（不加噪声）。固定阈值指用原图最大响应的 1% 作为绝对阈值；相对阈值指每张图各用自己最大响应的 1%。点数是 6 个场景的平均",
            "增益": gains,
            "固定阈值检出的角点数": [r3(v, 1) for v in n_fixed],
            "相对阈值检出的角点数": [r3(v, 1) for v in n_rel],
            f"取最强的 {N} 个时的重复率": [r3(v) for v in rep_top],
            "增益 0.5 时最大响应变成原来的多少倍（实测）": r3(ratio_half, 4),
            "按四次方算": 0.5 ** 4,
            "相对阈值 1% 相当于只留对比度不低于最强角点多少倍的角点": r3(0.01 ** 0.25, 3),
            "在场景 0 左上角贴一个黑白棋盘格交点之后，相对阈值检出的角点数 [之前, 之后]": [n_before, n_after],
        },
        "噪声": {
            "说明": "同一场景拍两次，各加同样标准差的高斯噪声（图像灰度范围 0 到 255），6 个场景各 2 组取平均",
            "噪声标准差": noises,
            **{name: [r3(v) for v in vals] for name, vals in noise_out.items()},
        },
    }

    fig, ax = plt.subplots(1, 4, figsize=(15.5, 4.1), layout="constrained")
    for a, xs, arr, arr_t, xlabel in ((ax[0], thetas, rot, rot_t, "旋转角（度）"), (ax[1], scales, sca, sca_t, "放大倍数")):
        a.fill_between(xs, arr.min(1), arr.max(1), color=S.BLUE, alpha=0.15)
        a.plot(xs, arr.mean(1), "o-", color=S.BLUE, lw=1.8, ms=4, label="多边形场景（带为 6 个场景的范围）")
        a.plot(xs, arr_t.mean(1), "s-", color=S.ORANGE, lw=1.6, ms=4, label="平滑纹理场景")
        a.axhline(np.mean(half_px), color=S.GREEN, lw=1.3, ls="--", label="只平移半个像素")
        a.axhline(np.mean(chance), color=S.MUTED, lw=1.2, ls=":", label="不相干的两张图")
        a.set_ylim(0, 1.04)
        a.set_xlabel(xlabel)
        a.legend(fontsize=8.3, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        S.clean(a)
    ax[0].set_ylabel("重复率")
    ax[0].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[0].set_title(TITLES_FIG3[0])
    ax[1].set_xscale("log")
    ax[1].set_xticks([0.5, 0.7, 1, 1.4, 2], ["0.5", "0.7", "1", "1.4", "2"])
    ax[1].minorticks_off()
    ax[1].set_title(TITLES_FIG3[1])

    ax[2].plot(gains, n_fixed, "s-", color=S.RED, lw=1.8, ms=4.5, label="固定的绝对阈值")
    ax[2].plot(gains, n_rel, "o-", color=S.BLUE, lw=1.8, ms=4.5, label="每张图最大响应的 1%")
    ax[2].set_xlabel("亮度增益")
    ax[2].set_ylabel("检出的角点数")
    ax[2].set_ylim(bottom=0)
    ax[2].set_title(TITLES_FIG3[2])
    ax[2].legend(fontsize=8.3, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax[2])

    styles = {"取最强的 25 个": (S.GREEN, "^-"), "取最强的 50 个": (S.BLUE, "o-"), "取最强的 200 个": (S.RED, "v-"),
              "取最强的 50 个，先做标准差 1.5 的高斯平滑": (S.PURPLE, "s--")}
    for name, (color, fmt) in styles.items():
        ax[3].plot(noises, noise_out[name], fmt, color=color, lw=1.7, ms=4.5, label=name.replace("，先做标准差 1.5 的高斯平滑", "，先平滑"))
    ax[3].set_xlabel("噪声标准差（灰度）")
    ax[3].set_ylabel("重复率")
    ax[3].set_ylim(0, 1.04)
    ax[3].set_title(TITLES_FIG3[3])
    ax[3].legend(fontsize=8.3, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[3])
    S.save(fig, CH, "fig03_Harris重复率")


# 图 3 四栏的标题。写的是结论，改实验之后要对着数字重新核对
TITLES_FIG3 = ["旋转：和只平移半个像素差不多", "缩放：离 1 倍越远掉得越多", "调暗：固定阈值下角点成片消失", "噪声：点取得越多，重复率越低"]


# ---------------------------------------------------------------- 图 4：SIFT、ORB 和固定小块
def kp_xy(kps):
    return np.array([k.pt for k in kps], float).reshape(-1, 2)


def feats(kind, img, **kw):
    """返回 (坐标, 描述子, 距离类型)。“固定小块”是没有尺度和方向的对照：Harris 角点 + 15x15 小块减均值再归一化。

    角点按“响应大于本图最大值的 1%”来取，不限个数。如果改成每张图固定取最强的若干个，缩放之后两张图视野不一样大，
    公共区域里两边的点数会差好几倍，找回率的上限被名额卡住，量到的就不是描述子本身的问题了。"""
    if kind == "SIFT":
        k, d = cv2.SIFT_create(nfeatures=1000, **kw).detectAndCompute(img, None)
        return kp_xy(k), d, cv2.NORM_L2
    if kind == "ORB":
        k, d = cv2.ORB_create(nfeatures=1000, **kw).detectAndCompute(img, None)
        return kp_xy(k), d, cv2.NORM_HAMMING
    g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0)
    pts, desc = [], []
    for x, y in nms_points(harris(img, 0.04, 5), n=None, rel=0.01).astype(int):
        p = g[y - 7:y + 8, x - 7:x + 8].ravel()
        p = p - p.mean()
        if np.linalg.norm(p) > 1e-6:
            pts.append((x, y))
            desc.append(p / np.linalg.norm(p))
    return np.array(pts, float).reshape(-1, 2), (np.array(desc, np.float32) if desc else None), cv2.NORM_L2


def knn_pairs(da, db, norm):
    if da is None or db is None or len(db) < 2:
        return []
    return [p for p in cv2.BFMatcher(norm).knnMatch(da, db, k=2) if len(p) == 2]


def match_stats(fa, fb, s=1.0, theta=0.0, ratio=0.75, tol=3.0, norm=None):
    """(第一张图里落在第二张图视野内的关键点数, 通过比值检验的匹配数, 其中正确的个数)。
    正确 = 按真实变换投影后离匹配到的点不超过 tol（按分辨率较低的那张图的像素算）。"""
    (pa, da, n0), (pb, db, _) = fa, fb
    proj = to_b(pa, s, theta)
    vis = int(np.all((proj >= 0) & (proj <= VIEW - 1), axis=1).sum())
    kept = correct = 0
    for m, n in knn_pairs(da, db, norm or n0):
        if m.distance < ratio * n.distance:
            kept += 1
            correct += bool(np.linalg.norm(proj[m.queryIdx] - pb[m.trainIdx]) <= tol * max(1.0, s))
    return np.array([vis, kept, correct])


METHODS = ["SIFT", "ORB", "固定小块"]


def fig_descriptors():
    scenes = [make_master(s) for s in range(4)]
    shot_a = [render(m, noise=3, seed=300 + i) for i, m in enumerate(scenes)]
    fa = {k: [feats(k, a) for a in shot_a] for k in METHODS}
    groups = {
        "旋转角（度）": ("theta", [0, 5, 10, 15, 30, 45, 60, 90, 135, 180]),
        "放大倍数": ("s", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]),
        "亮度增益": ("gain", [0.3, 0.4, 0.5, 0.7, 1.0]),
        "噪声标准差": ("noise", [3, 5, 10, 20, 30]),
        "额外模糊的标准差（px）": ("blur", [0, 1, 2, 3]),
    }
    res = {g: {k: [] for k in METHODS} for g in groups}
    adapt = {"SIFT，对比度阈值按增益同比降低": [], "ORB，FAST 阈值按增益同比降低": [],
             "SIFT，阈值同比降低，噪声也同比缩小": [], "ORB，阈值同比降低，噪声也同比缩小": []}
    nkp = {"SIFT": [], "ORB": []}
    for gname, (key, values) in groups.items():
        for v in values:
            tot = {k: np.zeros(3, int) for k in METHODS}
            ad = {k: np.zeros(3, int) for k in adapt}
            cnt = {"SIFT": 0, "ORB": 0}
            for i, m in enumerate(scenes):
                kw = {"noise": 3, "seed": 400 + i, key: v}
                geo = {k: kw[k] for k in ("s", "theta") if k in kw}
                b = render(m, **kw)
                a_feats = {k: fa[k][i] for k in METHODS}
                if key == "noise":
                    a_feats = {k: feats(k, render(m, noise=v, seed=300 + i)) for k in METHODS}
                for k in METHODS:
                    fb = feats(k, b)
                    tot[k] += match_stats(a_feats[k], fb, **geo)
                    if key == "gain" and k in cnt:
                        cnt[k] += len(fb[0])
                if key == "gain":
                    quiet = render(m, gain=v, noise=3 * v, seed=400 + i)
                    sift_kw, orb_kw = {"contrastThreshold": 0.04 * v}, {"fastThreshold": max(1, round(20 * v))}
                    ad["SIFT，对比度阈值按增益同比降低"] += match_stats(a_feats["SIFT"], feats("SIFT", b, **sift_kw))
                    ad["ORB，FAST 阈值按增益同比降低"] += match_stats(a_feats["ORB"], feats("ORB", b, **orb_kw))
                    ad["SIFT，阈值同比降低，噪声也同比缩小"] += match_stats(a_feats["SIFT"], feats("SIFT", quiet, **sift_kw))
                    ad["ORB，阈值同比降低，噪声也同比缩小"] += match_stats(a_feats["ORB"], feats("ORB", quiet, **orb_kw))
            for k in METHODS:
                res[gname][k].append(tot[k])
            if key == "gain":
                for k in adapt:
                    adapt[k].append(ad[k])
                for k in cnt:
                    nkp[k].append(round(cnt[k] / len(scenes)))
    bias_same = []
    for bias in (0, 40, 80):
        b = render(scenes[0], gain=0.6, bias=bias, noise=0)
        bias_same.append([int(v) for v in match_stats(fa["SIFT"][0], feats("SIFT", b))] + [int(v) for v in match_stats(fa["ORB"][0], feats("ORB", b))])

    def pack(rows):
        rows = np.array(rows)
        return {"视野内的关键点数": rows[:, 0].tolist(), "通过比值检验的匹配数": rows[:, 1].tolist(), "其中正确的个数": rows[:, 2].tolist(),
                "找回率（正确数 / 视野内的关键点数）": [r3(c / max(v, 1)) for v, _, c in rows],
                "正确率（正确数 / 通过比值检验的匹配数）": [r3(c / max(k, 1)) for _, k, c in rows]}

    NUM["SIFT、ORB 和固定小块"] = {
        "说明": "4 个 400x400 的多边形场景，每个场景拍两张，各加标准差 3 的噪声（噪声那一组两张都用表里的噪声），第二张再做表里的变换；"
              "SIFT 和 ORB 都设 nfeatures=1000，其余默认；“固定小块”是没有尺度和方向的对照：参考实现的 Harris 角点（响应大于本图最大值的 1%，不限个数）"
              "加 15x15 小块（减均值、归一化）；暴力匹配取两个最近邻，比值阈值 0.75；投影误差不超过 3 px（按分辨率较低的那张图的像素算）记为正确；表里是 4 个场景合计。"
              "三种方法检出的点数不同，找回率的分母是各自的关键点数，所以三条曲线的高低不能直接当成方法优劣，要看的是各自随变换下降的形状",
        "第一张图平均检出的关键点数": {k: round(float(np.mean([len(f[0]) for f in fa[k]]))) for k in METHODS},
        **{g: {"取值": groups[g][1], **{k: pack(res[g][k]) for k in METHODS}} for g in groups},
        "调暗之后同比降低检测阈值": {
            "说明": "“噪声也同比缩小”指第二张图的噪声标准差取 3 乘以增益，信噪比和原图相同，用来区分“阈值”和“信噪比”两个原因",
            "增益": groups["亮度增益"][1], **{k: pack(v) for k, v in adapt.items()},
            "默认阈值下第二张图平均检出的关键点数": nkp},
        "只加偏置（场景 0，增益 0.6，偏置 0、40、80，不加噪声）时 [SIFT 视野内, 保留, 正确, ORB 视野内, 保留, 正确]": bias_same,
    }

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.0), layout="constrained")
    style = {"SIFT": (S.BLUE, "o-"), "ORB": (S.ORANGE, "s-"), "固定小块": (S.MUTED, "^-")}
    recall = lambda rows: [c / max(v, 1) for v, _, c in rows]  # noqa: E731
    precision = lambda rows: [c / max(k, 1) for _, k, c in rows]  # noqa: E731
    for row, fn in ((0, recall), (1, precision)):
        for a, gname in zip(axes[row], list(groups)[:4]):
            xs = groups[gname][1]
            for k in METHODS:
                a.plot(xs, fn(res[gname][k]), style[k][1], color=style[k][0], lw=1.8, ms=4.5, label=k)
            a.set_ylim(0, 1.03)
            S.clean(a)
        for k, color in (("SIFT，对比度阈值按增益同比降低", S.BLUE), ("ORB，FAST 阈值按增益同比降低", S.ORANGE)):
            axes[row][2].plot(groups["亮度增益"][1], fn(adapt[k]), "--", color=color, lw=1.5, marker="D", ms=4, mfc="white",
                              label=k.split("，")[0] + "，检测阈值同比降低")
        axes[row][0].set_xticks([0, 45, 90, 135, 180])
        axes[row][1].set_xscale("log")
        axes[row][1].set_xticks([0.5, 0.7, 1, 1.4, 2], ["0.5", "0.7", "1", "1.4", "2"])
        axes[row][1].minorticks_off()
    axes[0][0].set_ylabel("找回率（正确匹配数 / 视野内的关键点数）")
    axes[1][0].set_ylabel("正确率（正确匹配数 / 保留的匹配数）")
    for a, title in zip(axes[0], TITLES_FIG4):
        a.set_title(title)
    for a, gname in zip(axes[1], list(groups)[:4]):
        a.set_xlabel(gname)
        a.legend(fontsize=8.3, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3 if a is not axes[1][2] else 2)
    S.save(fig, CH, "fig04_SIFT与ORB")


# 图 4 上面一行四栏的标题。写的是结论，改实验之后要对着数字重新核对
TITLES_FIG4 = ["旋转：没有方向的小块到 30 度几乎全丢", "缩放：缩到 0.5 倍时只有 SIFT 还剩三成", "调暗：降低检测阈值能找回大半", "噪声：三种方法都逐渐变差"]


def machine():
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        cpu = ""
    return " ".join(x for x in (platform.system(), platform.machine(), cpu) if x)


def numbers_timing():
    """耗时：取多次运行的中位数，保留两位有效数字。

    计时每次运行都会有波动。为了让重复运行脚本得到完全相同的 实测数字.json，默认沿用上次写下的测量值；
    设置环境变量 CH05_RETIME=1 才重新计时（换机器、换 OpenCV 版本之后要这样做，再按新数字改讲义）。"""
    TIMING_NOTE = ("在运行本脚本的这台机器上计时，每项先热身一次再取多次运行的中位数，保留两位有效数字。计时本身每次都会波动，"
                   "机器上同时跑别的任务时更大（编写本章时几次运行之间同一项最多差到约 1.8 倍），所以只在设置 CH05_RETIME=1 时重测，平时沿用这里的值；"
                   "只能用来看量级和相对快慢，换一台机器要重测")
    import os
    old = S.MAIN / CH / "figures" / "实测数字.json"
    if os.environ.get("CH05_RETIME") != "1" and old.exists():
        prev = json.loads(old.read_text(encoding="utf-8")).get("耗时")
        if prev:
            prev["说明"] = TIMING_NOTE
            NUM["耗时"] = prev
            print("耗时沿用上次的测量（设置 CH05_RETIME=1 重测）", flush=True)
            return

    def med(fn, n):
        fn()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)
        return float(f"{np.median(ts):.2g}")

    small = render(make_master(0), noise=3, seed=300)
    big = cv2.GaussianBlur(cv2.resize(make_master(0), (1024, 1024), interpolation=cv2.INTER_AREA), (0, 0), 0.8)
    sift, orb = cv2.SIFT_create(nfeatures=1000), cv2.ORB_create(nfeatures=1000)
    rng = np.random.default_rng(0)
    f1, f2 = rng.random((2000, 128), dtype=np.float32), rng.random((2000, 128), dtype=np.float32)
    u1, u2 = rng.integers(0, 256, (2000, 32), dtype=np.uint8), rng.integers(0, 256, (2000, 32), dtype=np.uint8)
    out = {}
    for label, threads in (("OpenCV 默认线程设置", -1), ("限制为单线程", 1)):
        cv2.setNumThreads(threads)
        out[label] = {
            "400x400：SIFT 检测加描述（ms）": med(lambda: sift.detectAndCompute(small, None), 30),
            "400x400：ORB 检测加描述（ms）": med(lambda: orb.detectAndCompute(small, None), 30),
            "1024x1024：SIFT 检测加描述（ms）": med(lambda: sift.detectAndCompute(big, None), 10),
            "1024x1024：ORB 检测加描述（ms）": med(lambda: orb.detectAndCompute(big, None), 10),
            "400x400：参考实现的 Harris 响应图（ms）": med(lambda: harris(small, 0.04, 5), 30),
            "2000 对 2000 暴力匹配取两个最近邻：128 维浮点、L2（ms）": med(lambda: cv2.BFMatcher(cv2.NORM_L2).knnMatch(f1, f2, k=2), 10),
            "2000 对 2000 暴力匹配取两个最近邻：32 字节、Hamming（ms）": med(lambda: cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(u1, u2, k=2), 10),
        }
    cv2.setNumThreads(-1)
    NUM["耗时"] = {
        "说明": TIMING_NOTE,
        "机器": machine(),
        "OpenCV 版本": cv2.__version__,
        "关键点数（400x400）[SIFT, ORB]": [len(sift.detect(small, None)), len(orb.detect(small, None))],
        "关键点数（1024x1024）[SIFT, ORB]": [len(sift.detect(big, None)), len(orb.detect(big, None))],
        **out,
        "每个描述子占的字节数 [SIFT, ORB]": [128 * 4, 32],
    }


# ---------------------------------------------------------------- 图 5：比值检验
def fig_ratio():
    scenes = [make_master(s) for s in range(4)]
    pairs = [{"theta": 30}, {"s": 0.7}, {"theta": 45, "s": 1.3}]
    recs = {"SIFT": [], "ORB": []}
    cross = {"SIFT": np.zeros(2, int), "ORB": np.zeros(2, int)}
    wrong_norm = {"Hamming": np.zeros(3, int), "L2": np.zeros(3, int)}
    for i, m in enumerate(scenes):
        a = render(m, noise=5, seed=500 + i)
        for j, geo in enumerate(pairs):
            b = render(m, noise=5, seed=600 + 10 * i + j, **geo)
            tol = 3.0 * max(1.0, geo.get("s", 1.0))
            for kind in recs:
                (pa, da, norm), (pb, db, _) = feats(kind, a), feats(kind, b)
                proj = to_b(pa, **geo)
                for mm, nn in knn_pairs(da, db, norm):
                    recs[kind].append((mm.distance / max(nn.distance, 1e-9), np.linalg.norm(proj[mm.queryIdx] - pb[mm.trainIdx]) <= tol))
                for mm in cv2.BFMatcher(norm, crossCheck=True).match(da, db):
                    cross[kind] += (1, int(np.linalg.norm(proj[mm.queryIdx] - pb[mm.trainIdx]) <= tol))
                if kind == "ORB":
                    wrong_norm["Hamming"] += match_stats((pa, da, norm), (pb, db, norm), **geo)
                    wrong_norm["L2"] += match_stats((pa, da, norm), (pb, db, norm), norm=cv2.NORM_L2, **geo)
    recs = {k: np.array(v, float) for k, v in recs.items()}
    ratios = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
    table = {}
    for kind, r in recs.items():
        total = r[:, 1].sum()
        rows = [(int((r[:, 0] < t).sum()), int(r[r[:, 0] < t, 1].sum())) for t in ratios]
        table[kind] = {"最近邻匹配总数": len(r), "其中正确的个数": int(total),
                       "错误匹配的个数": int(len(r) - total), "错误匹配里比值不小于 0.9 的个数": int(((r[:, 1] == 0) & (r[:, 0] >= 0.9)).sum()),
                       "错误匹配里比值不小于 0.9 的比例": r3(((r[:, 1] == 0) & (r[:, 0] >= 0.9)).sum() / max(len(r) - total, 1)),
                       "保留的匹配数": [k for k, _ in rows], "其中正确的个数（各阈值）": [c for _, c in rows],
                       "正确率": [r3(c / max(k, 1)) for k, c in rows], "保住的正确匹配比例": [r3(c / total) for _, c in rows],
                       "互选（crossCheck）[保留, 正确, 正确率]": [int(cross[kind][0]), int(cross[kind][1]), r3(cross[kind][1] / cross[kind][0])]}
    NUM["比值检验"] = {
        "说明": "4 个多边形场景，每个场景三组变换（转 30 度；缩小到 0.7 倍；转 45 度并放大 1.3 倍），两张图各加标准差 5 的噪声，共 12 对图合计；"
              "SIFT 和 ORB 的 nfeatures=1000；阈值 1.01 表示不做比值检验（所有最近邻都保留）",
        "比值阈值": ratios, **table,
        "ORB 描述子用错距离": {"说明": "同样 12 对图，比值阈值 0.75，[视野内的关键点数, 保留, 正确]", **{k: v.tolist() for k, v in wrong_norm.items()}},
    }

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9), layout="constrained")
    r = recs["SIFT"]
    bins = np.linspace(0, 1, 26)
    ax[0].hist(r[r[:, 1] == 1, 0], bins=bins, color=S.BLUE, alpha=0.75, label="正确的匹配")
    ax[0].hist(r[r[:, 1] == 0, 0], bins=bins, color=S.RED, alpha=0.6, label="错误的匹配")
    ax[0].axvline(0.75, color=S.INK, lw=1.2, ls="--")
    ax[0].set_xlabel("最近距离 / 次近距离")
    ax[0].set_ylabel("匹配个数")
    ax[0].set_title("SIFT：错误匹配的比值挤在 1 附近")
    ax[0].legend(fontsize=9, loc="upper center")
    S.clean(ax[0])
    for a, kind in ((ax[1], "SIFT"), (ax[2], "ORB")):
        xs = [min(t, 1.0) for t in ratios]
        a.plot(xs, table[kind]["正确率"], "o-", color=S.BLUE, lw=1.8, ms=4.5, label="保留下来的匹配里正确的比例")
        a.plot(xs, table[kind]["保住的正确匹配比例"], "s-", color=S.ORANGE, lw=1.8, ms=4.5, label="正确匹配里被保住的比例")
        a.axvline(0.75, color=S.INK, lw=1.2, ls="--")
        a.set_xlabel("比值阈值（最右一点是不做检验）")
        a.set_ylim(0, 1.04)
        a.set_title(f"{kind}：阈值放宽，正确率下降")
        a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
        S.clean(a)
    S.save(fig, CH, "fig05_比值检验")


# ---------------------------------------------------------------- 图 6：周期图案上的特征对齐
def periodic_canvas(n=420, mark=False):
    """和第 16 章图 6 同样的周期：x 方向 24 px，y 方向 16 px。mark=True 时加一个十字对位标记。"""
    y, x = np.mgrid[0:n, 0:n]
    img = np.full((n, n), 40, np.float32)
    img[(x % 24) < 8] = 200
    img[((y % 16) < 3) & ((x % 24) >= 12)] = 140
    if mark:
        c = n // 2
        img[c - 30:c - 24, c - 60:c - 14] = 255
        img[c - 50:c - 4, c - 40:c - 34] = 255
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def vote_shift(d, tol=2.0):
    """只估平移：每个匹配给出一个位移，取支持者最多的那个，再对支持者取平均。"""
    dist = np.linalg.norm(d[:, None] - d[None], axis=2)
    inl = dist[int(np.argmax((dist <= tol).sum(1)))] <= tol
    return d[inl].mean(0)


def fig_periodic():
    size, trials = 240, 100
    canvases = {"非周期的多边形场景": cv2.resize(make_master(3), (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32)[300:720, 300:720],
                "周期版图": periodic_canvas(), "周期版图加一个对位标记": periodic_canvas(mark=True)}
    dets = {"SIFT": (cv2.SIFT_create(), cv2.NORM_L2), "ORB": (cv2.ORB_create(nfeatures=1000), cv2.NORM_HAMMING)}
    out, cloud = {}, []
    for cname, canvas in canvases.items():
        out[cname] = {}
        o = (canvas.shape[0] - size) // 2
        for dname, (det, norm) in dets.items():
            rng = np.random.default_rng(5)
            cv2.setRNGSeed(0)
            st = {tag: {"fail": 0, "period": 0, "sim": 0, "kept": 0, "correct": 0} for tag in ("比值 0.75", "不做比值检验")}
            nk = 0
            for _ in range(trials):
                dx, dy = (int(v) for v in rng.integers(-5, 6, 2))
                a = np.clip(canvas[o:o + size, o:o + size] + rng.normal(scale=3.0, size=(size, size)), 0, 255).astype(np.uint8)
                b = np.clip(canvas[o - dy:o - dy + size, o - dx:o - dx + size] + rng.normal(scale=3.0, size=(size, size)), 0, 255).astype(np.uint8)
                ka, da = det.detectAndCompute(a, None)
                kb, db = det.detectAndCompute(b, None)
                nk += len(ka)
                knn = knn_pairs(da, db, norm)
                for tag, good in (("比值 0.75", [m for m, n in knn if m.distance < 0.75 * n.distance]), ("不做比值检验", [m for m, _ in knn])):
                    s = st[tag]
                    if len(good) < 3:
                        s["fail"] += 1
                        s["sim"] += 1
                        continue
                    src = np.float32([ka[m.queryIdx].pt for m in good])
                    dst = np.float32([kb[m.trainIdx].pt for m in good])
                    err = (dst - src) - np.array([dx, dy], np.float32)
                    s["kept"] += len(good)
                    s["correct"] += int((np.linalg.norm(err, axis=1) <= 2).sum())
                    e = vote_shift(dst - src) - np.array([dx, dy])
                    if np.linalg.norm(e) > 1:
                        s["fail"] += 1
                        s["period"] += int(abs(e[0] / 24 - round(e[0] / 24)) < 0.06 and abs(e[1] / 16 - round(e[1] / 16)) < 0.08)
                    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    box = np.array([[0, 0], [size, 0], [size, size], [0, size]], float)
                    if M is None or np.linalg.norm(box @ M[:, :2].T + M[:, 2] - (box + [dx, dy]), axis=1).max() > 1.5:
                        s["sim"] += 1
                    if cname == "周期版图" and dname == "SIFT" and tag == "比值 0.75":
                        cloud.append(err)
            out[cname][dname] = {"平均关键点数": round(nk / trials), **{tag: {
                "只估平移（位移投票）的失败次数": s["fail"], "其中误差是整数个周期的次数": s["period"],
                "估相似变换（平移、旋转、缩放，RANSAC）的失败次数": s["sim"],
                "平均每对图保留的匹配数": r3(s["kept"] / trials, 1), "其中位移正确的个数": r3(s["correct"] / trials, 1),
                "保留的匹配里位移正确的比例": r3(s["correct"] / max(s["kept"], 1))} for tag, s in st.items()}}
    NUM["周期图案上的特征对齐"] = {
        "说明": f"从 420x420 的底图上裁两个 {size}x{size} 的视场，真实平移在 ±5 px 内取整数，各加标准差 3 的噪声，每种组合 {trials} 次；"
              "周期版图 x 方向周期 24 px、y 方向周期 16 px，和第 16 章图 6 相同；SIFT 用默认参数，ORB 的 nfeatures=1000；"
              "位移误差不超过 2 px 的匹配记为正确；只估平移时估计误差超过 1 px 记为失败，估相似变换时视场四角的最大误差超过 1.5 px 记为失败",
        **out,
    }

    cloud = np.concatenate(cloud)
    bad = cloud[np.linalg.norm(cloud, axis=1) > 2]
    on_lattice = (np.abs(bad[:, 0] - 24 * np.round(bad[:, 0] / 24)) <= 2) & (np.abs(bad[:, 1] - 16 * np.round(bad[:, 1] / 16)) <= 2)
    NUM["周期图案上的特征对齐"]["周期版图上 SIFT 通过比值检验但位移错了的匹配"] = {
        "个数（100 对图合计）": int(len(bad)), "其中位移误差落在整数个周期 2 px 以内的比例": r3(on_lattice.mean())}

    # 对策：位移的范围已知时，只在这个范围内找最近邻（不做比值检验）
    limited = {}
    canvas = canvases["周期版图"]
    o = (canvas.shape[0] - size) // 2
    for dname, (det, norm) in dets.items():
        limited[dname] = {}
        for radius in (7, 13):
            rng = np.random.default_rng(5)
            cv2.setRNGSeed(0)
            fail = period = kept = correct = 0
            for _ in range(trials):
                dx, dy = (int(v) for v in rng.integers(-5, 6, 2))
                a = np.clip(canvas[o:o + size, o:o + size] + rng.normal(scale=3.0, size=(size, size)), 0, 255).astype(np.uint8)
                b = np.clip(canvas[o - dy:o - dy + size, o - dx:o - dx + size] + rng.normal(scale=3.0, size=(size, size)), 0, 255).astype(np.uint8)
                ka, da = det.detectAndCompute(a, None)
                kb, db = det.detectAndCompute(b, None)
                pa, pb = kp_xy(ka), kp_xy(kb)
                allowed = (np.abs(pa[:, None, :] - pb[None, :, :]).max(2) <= radius).astype(np.uint8)
                ms = cv2.BFMatcher(norm).match(da, db, mask=allowed)
                if len(ms) < 3:
                    fail += 1
                    continue
                d = np.array([pb[m.trainIdx] - pa[m.queryIdx] for m in ms])
                kept += len(ms)
                correct += int((np.linalg.norm(d - [dx, dy], axis=1) <= 2).sum())
                e = vote_shift(d) - np.array([dx, dy])
                if np.linalg.norm(e) > 1:
                    fail += 1
                    period += int(abs(e[0] / 24 - round(e[0] / 24)) < 0.06 and abs(e[1] / 16 - round(e[1] / 16)) < 0.08)
            limited[dname][f"只在 ±{radius} px 内找最近邻"] = {
                "只估平移（位移投票）的失败次数": fail, "其中误差是整数个周期的次数": period,
                "平均每对图的匹配数": r3(kept / trials, 1), "其中位移正确的个数": r3(correct / trials, 1),
                "匹配里位移正确的比例": r3(correct / max(kept, 1))}
    NUM["周期图案上的特征对齐"]["周期版图上限制匹配范围"] = {
        "说明": "同样的周期版图和 100 组平移（真实平移不超过 ±5 px）。每个关键点只和第二张图里 x、y 坐标差都不超过给定范围的关键点比较，取最近邻，不做比值检验。"
              "y 方向周期 16 px，真实位移加减一个周期的假位置离真实位置 16 px：范围取 ±7 时它们一定被排除在外，取 ±13 时有一部分会落进范围",
        **limited}

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1, 1.15, 1.35]}, layout="constrained")
    show = canvases["周期版图加一个对位标记"]
    o = (show.shape[0] - size) // 2
    ax[0].imshow(show[o:o + size, o:o + size], cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("周期版图（这一张带对位标记）")
    S.image_axes(ax[0])
    cloud = np.round(cloud).astype(int)
    uniq, cnt = np.unique(cloud, axis=0, return_counts=True)
    ok = (np.abs(uniq) <= 2).all(1)
    for gx in range(-216, 217, 24):
        ax[1].axvline(gx, color=S.GRID, lw=0.7)
    for gy in range(-208, 209, 16):
        ax[1].axhline(gy, color=S.GRID, lw=0.7)
    ax[1].scatter(uniq[~ok, 0], uniq[~ok, 1], s=6 + 5 * np.sqrt(cnt[~ok]), color=S.RED, alpha=0.6, linewidths=0, label="位移错了的匹配")
    ax[1].scatter(uniq[ok, 0], uniq[ok, 1], s=6 + 5 * np.sqrt(cnt[ok]), color=S.GREEN, alpha=0.9, linewidths=0, label="位移正确的匹配")
    ax[1].set_xlim(-130, 130)
    ax[1].set_ylim(110, -110)
    ax[1].set_xlabel("匹配给出的位移 - 真实位移，x（px）")
    ax[1].set_ylabel("y（px）")
    ax[1].set_title("错的匹配都错了整数个周期")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    names = list(canvases)
    width = 0.36
    for j, (dname, color) in enumerate((("SIFT", S.BLUE), ("ORB", S.ORANGE))):
        vals = [out[c][dname]["比值 0.75"]["只估平移（位移投票）的失败次数"] for c in names]
        bars = ax[2].bar(np.arange(len(names)) + (j - 0.5) * width, vals, width, color=color, alpha=0.88, label=dname)
        for bar, v in zip(bars, vals):
            ax[2].text(bar.get_x() + bar.get_width() / 2, v + 1.5, str(v), ha="center", fontsize=9.5)
    ax[2].set_xticks(np.arange(len(names)), ["非周期的\n多边形场景", "周期版图", "周期版图\n加对位标记"])
    ax[2].set_ylabel(f"{trials} 次对齐里失败的次数")
    ax[2].set_ylim(0, 108)
    ax[2].set_title("同一套流程，只换图案")
    ax[2].legend(fontsize=9, loc="upper right")
    S.clean(ax[2])
    ax[2].grid(axis="x", visible=False)
    S.save(fig, CH, "fig06_周期图案")


# ---------------------------------------------------------------- 不画图的数字：OpenCV 的行为、作业自查、完整流程
def raises(fn):
    try:
        fn()
        return False
    except cv2.error:
        return True


def wrong_variant(kind):
    """作业里容易写错的几种版本，只用来量它们的表现，供讲义的“常见错误”表使用。"""
    def f(gray, k=0.04, window_size=3):
        if kind == "梯度留在 uint8 里":
            g8 = np.asarray(gray, np.uint8)
            ix, iy = cv2.Sobel(g8, -1, 1, 0, ksize=3).astype(np.float32), cv2.Sobel(g8, -1, 0, 1, ksize=3).astype(np.float32)
        else:
            g = np.asarray(gray, np.float32)
            ix, iy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        if kind == "没有用上 k 和 window_size":
            k, window_size = 0.04, 3
        box = (lambda a: a) if kind == "没有做窗口求和" else (lambda a: cv2.boxFilter(a, cv2.CV_32F, (window_size, window_size), normalize=True))
        sxx, syy, sxy = box(ix * ix), box(iy * iy), box(ix * iy)
        det = sxx * syy - (0 if kind == "行列式漏了交叉项" else sxy * sxy)
        tr = sxx + syy
        return (det - k * (tr if kind == "迹没有平方" else tr * tr)).astype(np.float32)
    return f


def numbers_misc():
    import contextlib
    import importlib.util
    import io

    base = render(make_master(0))
    gf = base.astype(np.float32)
    ref = harris(base, 0.04, 3)
    cvr = cv2.cornerHarris(gf, 3, 3, 0.04)
    big = np.abs(cvr) > 1e-3 * np.abs(cvr).max()
    sxx, syy, sxy = tensor(base, 5)
    mine = sxx * syy - sxy ** 2 - 0.04 * (sxx + syy) ** 2
    snippet = cv2.cornerHarris(gf, blockSize=2, ksize=3, k=0.04)
    sift = cv2.SIFT_create()
    orb = cv2.ORB_create(nfeatures=1000)
    ks, ds = sift.detectAndCompute(base, None)
    ko, do = orb.detectAndCompute(base, None)
    flat = np.full((100, 100), 255, np.uint8)
    NUM["OpenCV 行为核对"] = {
        "说明": f"OpenCV {cv2.__version__}，图像是场景 0 的 400x400 基准视图。这里记录的是这个版本上的实际行为，换版本要重新确认",
        "参考实现的响应 / cv2.cornerHarris 的响应（float32 输入，blockSize=3）": {"中位数": r3(np.median(ref[big] / cvr[big]), 2), "相关系数": r3(np.corrcoef(ref.ravel(), cvr.ravel())[0, 1], 6)},
        "不同窗口下两者最大值之比": {str(w): r3(harris(base, 0.04, w).max() / cv2.cornerHarris(gf, w, 3, 0.04).max(), 1) for w in (3, 5, 7)},
        "cornerHarris 用 float32 输入和 uint8 输入的最大值之比": float(f"{cvr.max() / cv2.cornerHarris(base, 3, 3, 0.04).max():.4g}"),
        "255 的四次方": 255 ** 4,
        "本脚本自己算的结构张量和参考实现的最大相对差": float(f"{np.abs(mine - harris(base, 0.04, 5)).max() / np.abs(mine).max():.1g}"),
        "原讲义的标红写法（blockSize=2，响应大于最大值 1%）标出的像素数": int((snippet > 0.01 * snippet.max()).sum()),
        "同一张响应图做非极大值抑制（邻域半径 4）后剩下的点数": len(nms_points(snippet, n=None, rel=0.01)),
        "goodFeaturesToTrack（qualityLevel=0.01，minDistance=10）返回的点数": {
            "默认（Shi-Tomasi）": len(cv2.goodFeaturesToTrack(base, 300, 0.01, 10)),
            "useHarrisDetector=True": len(cv2.goodFeaturesToTrack(base, 300, 0.01, 10, useHarrisDetector=True, k=0.04))},
        "SIFT 默认参数 [nfeatures, contrastThreshold, edgeThreshold, sigma]": [sift.getNFeatures(), r3(sift.getContrastThreshold(), 3), sift.getEdgeThreshold(), r3(sift.getSigma(), 2)],
        "ORB 默认参数 [nfeatures, scaleFactor, nlevels, fastThreshold, WTA_K]": [cv2.ORB_create().getMaxFeatures(), r3(cv2.ORB_create().getScaleFactor(), 2),
                                                                             cv2.ORB_create().getNLevels(), cv2.ORB_create().getFastThreshold(), cv2.ORB_create().getWTA_K()],
        "SIFT 描述子的形状和类型": [list(ds.shape), str(ds.dtype)],
        "ORB 描述子的形状和类型（nfeatures=1000）": [list(do.shape), str(do.dtype)],
        "SIFT 关键点数随 edgeThreshold": {str(e): len(cv2.SIFT_create(edgeThreshold=e).detect(base, None)) for e in (2, 5, 10, 20, 50)},
        "SIFT 关键点数随 contrastThreshold": {str(c): len(cv2.SIFT_create(contrastThreshold=c).detect(base, None)) for c in (0.02, 0.04, 0.08)},
        "纯白图上 detectAndCompute 返回的描述子是 None": {"SIFT": sift.detectAndCompute(flat, None)[1] is None, "ORB": orb.detectAndCompute(flat, None)[1] is None},
        "SIFT 描述子配 NORM_HAMMING 调 knnMatch 是否直接报错": raises(lambda: cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(ds, ds, k=2)),
        "ORB 描述子配 NORM_L2 调 knnMatch 是否直接报错": raises(lambda: cv2.BFMatcher(cv2.NORM_L2).knnMatch(do, do, k=2)),
        "crossCheck=True 时调 knnMatch(k=2) 是否直接报错": raises(lambda: cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).knnMatch(ds, ds, k=2)),
        "FLANN 的 KD 树索引用在 ORB 描述子上是否直接报错": raises(lambda: cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50}).knnMatch(do, do, k=2)),
        "FLANN 的 LSH 索引用在 ORB 描述子上是否直接报错": raises(lambda: cv2.FlannBasedMatcher({"algorithm": 6, "table_number": 6, "key_size": 12, "multi_probe_level": 1}, {"checks": 50}).knnMatch(do, do, k=2)),
    }

    # 作业自查：几种写错的版本各自的表现，以及能不能通过验收脚本
    spec = importlib.util.spec_from_file_location("ch05_check", S.MAIN / CH / "tests" / "check_progress.py")
    test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test)
    diag = np.zeros((64, 64), np.uint8)
    diag[np.add.outer(np.arange(64), np.arange(64)) > 64] = 255
    diag = cv2.GaussianBlur(diag, (0, 0), 1.0)
    square = np.zeros((64, 64), np.uint8)
    square[18:46, 18:46] = 255
    scale_ref = float(harris(square).max())
    rows = {}
    for kind in ("参考实现", "行列式漏了交叉项", "迹没有平方", "没有做窗口求和", "梯度留在 uint8 里", "没有用上 k 和 window_size"):
        fn = harris if kind == "参考实现" else wrong_variant(kind)
        with contextlib.redirect_stdout(io.StringIO()):
            results = test.run_all(fn)
        passed = bool(all(results))
        failed = [name for (name, _), ok in zip(test.CHECKS, results) if not ok]
        rs = fn(square)
        quad = np.array([float(rs[y - 3:y + 4, x - 3:x + 4].max()) for y, x in ((18, 18), (18, 45), (45, 18), (45, 45))])
        half = square.astype(np.float32) * 0.5
        rows[kind] = {
            "能否通过验收脚本": passed,
            "验收里没通过的检查项": failed,
            "45 度直边中段的最大响应 / 参考实现在方块顶点的最大响应": r3(max(float(fn(diag)[20:44, 20:44].max()), 0.0) / scale_ref, 3),
            "方块上边中段的最小响应 / 参考实现在方块顶点的最大响应": float(f"{float(rs[14:22, 28:36].min()) / scale_ref:.2g}"),
            "方块四个顶点的响应（左上、右上、左下、右下，最大的记为 1）": [r3(v, 2) for v in quad / max(quad.max(), 1e-9)],
            "图像灰度减半后最大响应变成原来的多少倍": r3(float(fn(half).max()) / max(float(fn(square.astype(np.float32)).max()), 1e-9), 4),
            "改 k（0.04 改成 0.1）后输出是否变化": bool(np.abs(fn(square, 0.1, 3) - fn(square, 0.04, 3)).max() > 0),
        }
    NUM["作业自查"] = {"说明": "64x64 的测试图：验收脚本里的方块（18 到 45 行列为 255），和一条模糊 1 px 的 45 度直边。几种常见的写错的版本各自的表现；"
                             "“能否通过验收脚本”是把这个版本交给本章 tests/check_progress.py 的全部检查项跑一遍的结果", **rows}

    # 5.6 的完整流程
    cv2.setRNGSeed(0)
    full = cv2.SIFT_create()
    scene = render(make_master(0), theta=25, s=0.8, noise=3, seed=2)
    other = render(make_master(1), noise=3, seed=2)

    def pipeline(target, ratio):
        k1, d1 = full.detectAndCompute(base, None)
        k2, d2 = full.detectAndCompute(target, None)
        good = [m for m, n in cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2) if m.distance < ratio * n.distance]
        info = {"两张图的关键点数": [len(k1), len(k2)], "通过比值检验的匹配数": len(good)}
        if len(good) < 4:
            return {**info, "findHomography 是否返回了矩阵": False}, None
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None:
            return {**info, "findHomography 是否返回了矩阵": False}, None
        quad = cv2.perspectiveTransform(np.float32([[0, 0], [VIEW - 1, 0], [VIEW - 1, VIEW - 1], [0, VIEW - 1]]).reshape(-1, 1, 2), H).reshape(-1, 2)
        inl = int(mask.sum())
        return {**info, "findHomography 是否返回了矩阵": True, "RANSAC 内点数": inl, "内点比例": r3(inl / len(good)),
                "投影出的四边形是否为凸": bool(cv2.isContourConvex(quad.reshape(-1, 1, 2))),
                "四边形面积 / 模板面积": r3(abs(cv2.contourArea(quad.reshape(-1, 1, 2))) / (VIEW - 1) ** 2, 3)}, quad

    hit, quad = pipeline(scene, 0.75)
    truth = to_b([[0, 0], [VIEW - 1, 0], [VIEW - 1, VIEW - 1], [0, VIEW - 1]], 0.8, 25)
    hit["四个角的投影误差最大值（px）"] = r3(np.linalg.norm(quad - truth, axis=1).max(), 2)
    hit["真实的面积比（0.8 的平方）"] = r3(0.8 ** 2, 2)
    NUM["完整流程示例"] = {
        "说明": "按讲义 5.6 的代码运行：模板是场景 0 的基准视图，SIFT 默认参数，findHomography 用 RANSAC、阈值 4 px",
        "目标图是模板转 25 度、缩小到 0.8 倍、加标准差 3 的噪声": hit,
        "目标图是另一个不相干的场景，比值阈值 0.75": pipeline(other, 0.75)[0],
        "目标图是另一个不相干的场景，比值阈值放宽到 0.95": pipeline(other, 0.95)[0],
    }
    NUM["固定的数"] = {"SIFT 描述子维数 4x4x8": 4 * 4 * 8, "DoG 极值比较的邻居数 8+9+9": 8 + 9 + 9, "ORB 描述子位数 32x8": 32 * 8,
                    "SIFT 描述子占的字节数是 ORB 的多少倍": (128 * 4) // 32}


def main() -> None:
    S.setup()
    plt.rcParams["svg.hashsalt"] = "ch05"  # 固定 SVG 里的 id，重复运行生成的文件逐字节相同
    cv2.setRNGSeed(0)
    for fn in (fig_localize, fig_k_window, fig_repeat, fig_descriptors, fig_ratio, fig_periodic, numbers_misc, numbers_timing):
        t0 = time.perf_counter()
        fn()
        print(f"完成 {fn.__name__}（{time.perf_counter() - t0:.1f} s）", flush=True)
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
