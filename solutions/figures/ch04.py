"""第 4 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch04.py

输出：<主分支>/04_几何变换/figures/*.svg 和 实测数字.json。
点的投影全部调用本章作业的参考实现 transform_points；图像重采样和对照用 OpenCV。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。
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

CH = "04_几何变换"
mod = S.load_solution(CH, "4-1_齐次点变换.py")
tp = mod.transform_points
NUM: dict = {}

W, H = 640, 480
CORNERS = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=float)
GRID = np.stack(np.meshgrid(np.linspace(0, W - 1, 17), np.linspace(0, H - 1, 13)), -1).reshape(-1, 2)
INTERP = {"最近邻": cv2.INTER_NEAREST, "双线性": cv2.INTER_LINEAR, "双三次": cv2.INTER_CUBIC, "Lanczos4": cv2.INTER_LANCZOS4}
COLORS = {"最近邻": S.MUTED, "双线性": S.BLUE, "双三次": S.ORANGE, "Lanczos4": S.GREEN}


def r(x, k=3):
    return round(float(x), k)


# ---------------------------------------------------------------- 数据与工具
def scene(n=256, seed=0):
    """合成测试图：浅色底、细网格、实心圆和文字，8 位灰度。"""
    rng = np.random.default_rng(seed)
    img = np.full((n, n), 209, np.uint8)
    for x in range(16, n, 24):
        cv2.line(img, (x, 0), (x, n - 1), 90, 1)
    for y in range(16, n, 24):
        cv2.line(img, (0, y), (n - 1, y), 90, 1)
    for _ in range(max(4, n // 18)):
        c = tuple(int(v) for v in rng.integers(20, n - 20, 2))
        cv2.circle(img, c, int(rng.integers(4, 14)), int(rng.integers(25, 150)), -1)
    for i in range(max(2, n // 42)):
        p = tuple(int(v) for v in rng.integers(10, n - 60, 2))
        cv2.putText(img, f"CV{i}", p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 12, 1, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (0, 0), 0.8)


def norm_T(p):
    """Hartley 归一化：平移到质心，缩放到离原点的平均距离为 sqrt(2)。"""
    c = p.mean(axis=0)
    s = np.sqrt(2) / np.sqrt(((p - c) ** 2).sum(axis=1)).mean()
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])


def dlt(src, dst, mode="归一化", dtype=np.float64):
    """DLT 求单应，对方程矩阵做 SVD。mode：归一化、只平移到质心、不处理。"""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if mode == "归一化":
        Ts, Td = norm_T(src), norm_T(dst)
    elif mode == "只平移到质心":
        Ts, Td = np.eye(3), np.eye(3)
        Ts[:2, 2], Td[:2, 2] = -src.mean(axis=0), -dst.mean(axis=0)
    else:
        Ts, Td = np.eye(3), np.eye(3)
    s, d = tp(src, Ts).astype(dtype), tp(dst, Td).astype(dtype)
    x, y, u, v = s[:, 0], s[:, 1], d[:, 0], d[:, 1]
    z, o = np.zeros_like(x), np.ones_like(x)
    A = np.concatenate([np.stack([-x, -y, -o, z, z, z, u * x, u * y, u], 1), np.stack([z, z, z, -x, -y, -o, v * x, v * y, v], 1)])
    with np.errstate(all="ignore"):
        h = np.linalg.svd(A)[2][-1].astype(np.float64).reshape(3, 3)
        Hm = np.linalg.inv(Td) @ h @ Ts
        return Hm / Hm[2, 2] if abs(Hm[2, 2]) > 1e-12 else Hm


def affine_fit(src, dst):
    X, *_ = np.linalg.lstsq(np.hstack([src, np.ones((len(src), 1))]), dst, rcond=None)
    return np.vstack([X.T, [0, 0, 1]])


def rand_h(rng, strength=60.0):
    """随机单应：把画面四角各自随机挪动最多 strength 像素。"""
    dst = CORNERS + rng.uniform(-strength, strength, (4, 2))
    return cv2.getPerspectiveTransform(CORNERS.astype(np.float32), dst.astype(np.float32)).astype(float)


def h_err(Hest, Htrue):
    """估出的单应把全画面 17x13 个网格点映射后，和真值位置的均方根距离（px）。"""
    if Hest is None or not np.all(np.isfinite(Hest)):
        return float("inf")
    d = tp(GRID, Hest) - tp(GRID, Htrue)
    e = (d ** 2).sum(axis=1)
    return float(np.sqrt(e.mean())) if np.all(np.isfinite(e)) else float("inf")


def med(v):
    v = np.asarray(v, float)
    return float(np.median(np.where(np.isfinite(v), v, 1e9)))


def pct(v, q):
    v = np.asarray(v, float)
    return float(np.percentile(np.where(np.isfinite(v), v, 1e9), q))


# ---------------------------------------------------------------- RANSAC（批量版）
def batch_h(src4, dst4):
    """(K,4,2) 的最小样本批量求单应：坐标先缩到 1 左右，再对 8x9 的方程做 SVD。"""
    x, y, u, v = src4[..., 0] / 320 - 1, src4[..., 1] / 320 - 1, dst4[..., 0] / 320 - 1, dst4[..., 1] / 320 - 1
    z, o = np.zeros_like(x), np.ones_like(x)
    A = np.concatenate([np.stack([-x, -y, -o, z, z, z, u * x, u * y, u], -1), np.stack([z, z, z, -x, -y, -o, v * x, v * y, v], -1)], 1)
    h = np.linalg.svd(A)[2][:, -1, :].reshape(-1, 3, 3)
    T = np.array([[1 / 320, 0, -1], [0, 1 / 320, -1], [0, 0, 1.0]])
    return np.linalg.inv(T) @ h @ T


def hypotheses(src, dst, iters, rng):
    """抽 iters 组最小样本（每组 4 对，不重复），返回每个假设的单应、它对全部点的重投影误差和抽到的下标。"""
    n = len(src)
    idx = np.argsort(rng.random((iters, n)), axis=1)[:, :4]
    Hs = batch_h(src[idx], dst[idx])
    ph = np.concatenate([src, np.ones((n, 1))], 1) @ Hs.transpose(0, 2, 1)
    with np.errstate(all="ignore"):
        err = np.sqrt(((ph[..., :2] / ph[..., 2:3] - dst) ** 2).sum(-1))
    return Hs, np.where(np.isfinite(err), err, np.inf), idx


def finish(src, dst, Hs, err, thresh, upto=None):
    """在前 upto 个假设里取内点最多的，再用内点重新拟合两轮。"""
    cnt = (err[:upto] < thresh).sum(axis=1)
    b = int(np.argmax(cnt))
    mask = err[b] < thresh
    Hb = Hs[b]
    for _ in range(2):
        if mask.sum() < 4:
            break
        Hb = dlt(src[mask], dst[mask])
        with np.errstate(all="ignore"):
            e = np.sqrt(((tp(src, Hb) - dst) ** 2).sum(axis=1))
        mask = np.where(np.isfinite(e), e, np.inf) < thresh
    return Hb, mask


def make_matches(rng, n, out_ratio, sigma):
    """n 对匹配：内点按随机单应映射再加高斯噪声，外点的目标位置在画面内均匀随机。"""
    Ht = rand_h(rng)
    src = rng.uniform([0, 0], [W - 1, H - 1], (n, 2))
    dst = tp(src, Ht) + rng.normal(0, sigma, (n, 2))
    n_out = int(round(n * out_ratio))
    is_out = np.zeros(n, bool)
    is_out[rng.permutation(n)[:n_out]] = True
    dst[is_out] = rng.uniform([0, 0], [W - 1, H - 1], (n_out, 2))
    return src, dst, is_out, Ht


# ---------------------------------------------------------------- 图 1：前向映射的空洞
def forward_map(img, M3, out_shape):
    n = img.shape[0]
    ys, xs = np.mgrid[:n, :n]
    q = tp(np.stack([xs.ravel(), ys.ravel()], 1), M3)
    u, v = np.rint(q[:, 0]).astype(int), np.rint(q[:, 1]).astype(int)
    ok = (u >= 0) & (u < out_shape[1]) & (v >= 0) & (v < out_shape[0])
    hit = np.zeros(out_shape, int)
    np.add.at(hit, (v[ok], u[ok]), 1)
    out = np.zeros(out_shape, img.dtype)
    out[v[ok], u[ok]] = img.ravel()[ok]
    yt, xt = np.mgrid[:out_shape[0], :out_shape[1]]
    pre = tp(np.stack([xt.ravel(), yt.ravel()], 1), np.linalg.inv(M3))
    inside = ((pre[:, 0] >= 0) & (pre[:, 0] <= n - 1) & (pre[:, 1] >= 0) & (pre[:, 1] <= n - 1)).reshape(out_shape)
    return out, hit, inside


def fig_forward():
    n = 256
    img = scene(n)
    c = (n - 1) / 2
    angles = list(range(0, 91, 5))
    by_angle = {}
    for a in angles:
        M3 = np.vstack([cv2.getRotationMatrix2D((c, c), a, 1.0), [0, 0, 1]])
        _, hit, inside = forward_map(img, M3, (n, n))
        by_angle[a] = (100 * float(((hit == 0) & inside).sum() / inside.sum()), 100 * float(((hit > 1) & inside).sum() / inside.sum()))
    scales = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    by_scale = {}
    for s in scales:
        m = int(math.ceil(n * s))
        _, hit, inside = forward_map(img, np.diag([s, s, 1.0]), (m, m))
        by_scale[s] = (100 * float(((hit == 0) & inside).sum() / inside.sum()), 100 * (1 - 1 / s ** 2))
    NUM["前向映射的空洞"] = {
        "说明": "256x256 的图，把每个源像素按变换算出目标位置后四舍五入写进去；空洞比例 = 原像落在源图范围内却没有被写到的目标像素所占百分比；"
              "重复写入比例 = 被写了两次以上的目标像素所占百分比（后写的覆盖先写的）",
        "只旋转（不缩放）": {str(a): {"空洞（%）": r(by_angle[a][0], 1), "重复写入（%）": r(by_angle[a][1], 1)} for a in (0, 5, 10, 20, 30, 45, 60, 90)},
        "只放大": {str(s): {"空洞（%）": r(by_scale[s][0], 1), "理论值 1-1/s^2（%）": r(by_scale[s][1], 1)} for s in scales},
    }

    small = scene(128, seed=3)
    cs = (127) / 2
    M = cv2.getRotationMatrix2D((cs, cs), 30, 1.0)
    fwd, hit, inside = forward_map(small, np.vstack([M, [0, 0, 1]]), (128, 128))
    inv = cv2.warpAffine(small, M, (128, 128), flags=cv2.INTER_LINEAR)
    sl = (slice(40, 88), slice(40, 88))
    rgb = np.stack([fwd] * 3, -1)[sl].copy()
    rgb[((hit == 0) & inside)[sl]] = (214, 54, 47)

    fig, ax = plt.subplots(1, 4, figsize=(14.5, 3.7), gridspec_kw={"width_ratios": [1, 1, 1.25, 1.25]}, layout="constrained")
    ax[0].imshow(rgb, interpolation="nearest")
    ax[0].set_title("前向映射，旋转 30°（红色是空洞）", fontsize=10.5)
    ax[1].imshow(inv[sl], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax[1].set_title("反向映射 + 双线性，同一区域", fontsize=10.5)
    for a_ in ax[:2]:
        S.image_axes(a_)
    ax[2].plot(angles, [by_angle[a][0] for a in angles], "o-", color=S.RED, lw=1.8, ms=4, label="空洞")
    ax[2].plot(angles, [by_angle[a][1] for a in angles], "s--", color=S.BLUE, lw=1.4, ms=3.5, label="被重复写入")
    ax[2].set_xlabel("旋转角（度），不缩放")
    ax[2].set_ylabel("占目标像素的比例（%）")
    ax[2].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[2].set_title(f"只旋转也有空洞，45° 时 {by_angle[45][0]:.0f}%")
    ax[2].legend(fontsize=9)
    S.clean(ax[2])
    ax[3].plot(scales, [by_scale[s][1] for s in scales], "-", color=S.MUTED, lw=1.4, label="理论值 $1-1/s^2$")
    ax[3].plot(scales, [by_scale[s][0] for s in scales], "o", color=S.RED, ms=5.5, label="实测")
    ax[3].set_xlabel("放大倍数 s")
    ax[3].set_ylabel("空洞比例（%）")
    ax[3].set_ylim(0, 100)
    ax[3].set_title(f"放大 2 倍，{by_scale[2.0][0]:.0f}% 的像素是空的")
    ax[3].legend(fontsize=9, loc="lower right")
    S.clean(ax[3])
    S.save(fig, CH, "fig01_前向映射的空洞")


# ---------------------------------------------------------------- 图 2：插值误差累积
def fig_interp():
    n = 256
    img = scene(n)
    c = (n - 1) / 2
    yy, xx = np.mgrid[:n, :n]
    disk = (xx - c) ** 2 + (yy - c) ** 2 < (n * 0.42) ** 2
    base_std = float(img[disk].std())
    M10 = cv2.getRotationMatrix2D((c, c), 10, 1.0)
    one = cv2.warpAffine(img, cv2.getRotationMatrix2D((c, c), 90, 1.0), (n, n), flags=cv2.INTER_NEAREST)
    assert np.array_equal(one, np.rot90(img, 1)), "90 度旋转应与 np.rot90 逐像素相同"
    steps = list(range(9, 73, 9))
    rmse = {k: [] for k in INTERP}
    contrast = {k: [] for k in INTERP}
    after36 = {}
    for name, flag in INTERP.items():
        cur = img.copy()
        for k in range(1, 73):
            cur = cv2.warpAffine(cur, M10, (n, n), flags=flag, borderMode=cv2.BORDER_REFLECT)
            if k % 9 == 0:
                back = np.rot90(cur, -(k // 9) % 4).astype(float)
                rmse[name].append(float(np.sqrt(np.mean((back - img)[disk] ** 2))))
                contrast[name].append(float(back[disk].std() / base_std))
            if k == 36:
                after36[name] = cur.copy()
    Mf, Mb = cv2.getRotationMatrix2D((c, c), 17, 1.0), cv2.getRotationMatrix2D((c, c), -17, 1.0)
    round_trip = {}
    for name, flag in INTERP.items():
        t = cv2.warpAffine(cv2.warpAffine(img, Mf, (n, n), flags=flag, borderMode=cv2.BORDER_REFLECT), Mb, (n, n), flags=flag, borderMode=cv2.BORDER_REFLECT)
        round_trip[name] = float(np.sqrt(np.mean((t.astype(float) - img)[disk] ** 2)))
    gh, gw = 180, 240
    grid = np.full((gh, gw), 245, dtype=np.uint8)
    for x in range(0, gw, 20):
        cv2.line(grid, (x, 0), (x, gh - 1), 80, 1)
    for y in range(0, gh, 20):
        cv2.line(grid, (0, y), (gw - 1, y), 80, 1)
    cv2.putText(grid, "TOP", (85, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    quad = np.float32([[0, 0], [gw - 1, 0], [gw - 1, gh - 1], [0, gh - 1]])
    notebook = {}
    for tilt in (35, 90):
        dst = np.float32([[tilt, 8], [gw - 1 - tilt // 2, 0], [gw - 1, gh - 1], [0, gh - 1 - tilt // 3]])
        Hf, Hb = cv2.getPerspectiveTransform(quad, dst), cv2.getPerspectiveTransform(dst, quad)
        row = {}
        for name in ("最近邻", "双线性", "双三次"):
            w_ = cv2.warpPerspective(grid, Hf, (gw, gh), flags=INTERP[name], borderValue=255)
            b_ = cv2.warpPerspective(w_, Hb, (gw, gh), flags=INTERP[name], borderValue=255)
            row[name] = r(np.mean(np.abs(b_.astype(np.float32) - grid.astype(np.float32))), 2)
        notebook[f"tilt={tilt}"] = row
    M3 = np.vstack([M10, [0, 0, 1]])
    total = np.linalg.matrix_power(M3, 36)
    merged = cv2.warpAffine(img, total[:2], (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    NUM["插值误差累积"] = {
        "说明": "256x256 的 8 位合成图（网格线、实心圆、文字），每次绕中心旋转 10 度，每转满 90 度用 np.rot90 无损转回去和原图比；"
              "只统计中央半径 0.42 倍边长的圆内；RMSE 的单位是灰度级（0 到 255）；对比度比值 = 圆内标准差 / 原图圆内标准差",
        "原图圆内的标准差（灰度级）": r(base_std, 1),
        "旋转次数": steps,
        "RMSE（灰度级）": {k: [r(v, 1) for v in vals] for k, vals in rmse.items()},
        "对比度比值": {k: [r(v, 2) for v in vals] for k, vals in contrast.items()},
        "往返一次（+17 度再 -17 度）的 RMSE（灰度级）": {k: r(v, 2) for k, v in round_trip.items()},
        "Notebook 的网格图透视变换再变回来的平均绝对误差（灰度级）": {
            "说明": "和本章 Notebook 相同的 240x180 网格图（线宽 1 px，没有模糊）、相同的四角和 MAE 算法，只换插值方法", "结果": notebook},
        "把 36 个矩阵乘起来只变换一次": {"合成矩阵与单位阵的最大差": float(f"{np.abs(total - np.eye(3)).max():.1e}"),
                                "RMSE（灰度级）": r(np.sqrt(np.mean((merged.astype(float) - img)[disk] ** 2)), 3)},
    }

    imp = np.zeros((9, 9), np.float32)
    imp[4, 4] = 1
    om = np.linspace(0, np.pi, 721)
    kern = {}
    for name in ("双线性", "双三次", "Lanczos4"):
        w = cv2.warpAffine(imp, np.array([[1, 0, 0.5], [0, 1, 0.0]]), (9, 9), flags=INTERP[name])[4].astype(float)
        gain = np.abs((w[None, :] * np.exp(-1j * om[:, None] * (np.arange(9) - 4)[None, :])).sum(axis=1))
        kern[name] = {"权重": [r(v, 4) for v in w[np.abs(w) > 1e-9]], "最大增益": r(gain.max(), 3), "最大增益处的周期（px）": r(2 * np.pi / max(om[gain.argmax()], 1e-9), 1) if gain.max() > 1.0001 else None,
                      "周期 4 px 处的增益": r(gain[360], 3), "连乘 36 次的最大增益": r(gain.max() ** 36, 2), "周期 4 px 处连乘 36 次": float(f"{gain[360] ** 36:.2g}")}

    def keys(t, a):
        t = abs(t)
        return (a + 2) * t ** 3 - (a + 3) * t ** 2 + 1 if t <= 1 else a * t ** 3 - 5 * a * t ** 2 + 8 * a * t - 4 * a
    NUM["插值核"] = {"说明": "把一个单位脉冲平移 0.5 px，读出各插值方法在一维上的权重，再算它的频率响应；增益大于 1 表示这个频率被放大",
                   "平移半个像素时": kern, "三次卷积公式在 a=-0.75 时的权重": [r(keys(t, -0.75), 4) for t in (-1.5, -0.5, 0.5, 1.5)]}

    fig = plt.figure(figsize=(13.5, 6.4), layout="constrained")
    gs = fig.add_gridspec(2, 10, height_ratios=[1, 1.25])
    crop = (slice(96, 160), slice(96, 160))
    for i, (name, im) in enumerate([("原图", img)] + [(f"{k}，转 36 次", v) for k, v in after36.items()]):
        a_ = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
        a_.imshow(im[crop], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        a_.set_title(name, fontsize=10.5)
        S.image_axes(a_)
    a1, a2 = fig.add_subplot(gs[1, :5]), fig.add_subplot(gs[1, 5:])
    for name in INTERP:
        a1.plot(steps, rmse[name], "o-", color=COLORS[name], lw=1.8, ms=4.5, label=name)
        a2.plot(steps, contrast[name], "o-", color=COLORS[name], lw=1.8, ms=4.5, label=name)
    a1.set_xlabel("旋转次数（每次 10°）")
    a1.set_ylabel("和原图的 RMSE（灰度级）")
    a1.set_xticks(steps)
    a1.set_ylim(bottom=0)
    a1.set_title("误差随次数累积，双三次在 36 次之后超过双线性")
    a1.legend(fontsize=9, ncol=2)
    S.clean(a1)
    a2.axhline(1, color=S.INK, lw=0.6)
    a2.set_xlabel("旋转次数（每次 10°）")
    a2.set_ylabel("对比度比值（1 = 和原图相同）")
    a2.set_xticks(steps)
    a2.set_title("双线性越转越糊，双三次的对比度越转越高")
    S.clean(a2)
    S.save(fig, CH, "fig02_插值误差累积")


# ---------------------------------------------------------------- 表：缩小与掩码
def table_resize_mask():
    n = 480
    stripes = (((np.arange(n) % 3) < 1).astype(np.uint8) * 255)[None, :].repeat(n, 0)
    rows = {}
    for name, flag in (("最近邻", cv2.INTER_NEAREST), ("双线性", cv2.INTER_LINEAR), ("双三次", cv2.INTER_CUBIC), ("区域（INTER_AREA）", cv2.INTER_AREA)):
        o = cv2.resize(stripes, (120, 120), interpolation=flag).astype(float)
        rows[f"resize，{name}"] = {"平均值": r(o.mean(), 1), "标准差": r(o.std(), 1), "最小": r(o.min(), 0), "最大": r(o.max(), 0)}
    M = np.array([[0.25, 0, 0], [0, 0.25, 0.0]])
    wl = cv2.warpAffine(stripes, M, (120, 120), flags=cv2.INTER_LINEAR)
    wa = cv2.warpAffine(stripes, M, (120, 120), flags=cv2.INTER_AREA)
    pl = cv2.warpPerspective(stripes, np.diag([0.25, 0.25, 1.0]), (120, 120), flags=cv2.INTER_LINEAR)
    pa = cv2.warpPerspective(stripes, np.diag([0.25, 0.25, 1.0]), (120, 120), flags=cv2.INTER_AREA)
    for name, o in (("warpAffine，双线性", wl), ("warpAffine，传 INTER_AREA", wa)):
        o = o.astype(float)
        rows[name] = {"平均值": r(o.mean(), 1), "标准差": r(o.std(), 1), "最小": r(o.min(), 0), "最大": r(o.max(), 0)}
    img2 = scene(256)
    M2 = cv2.getRotationMatrix2D((127.5, 127.5), 17, 0.3)
    H2 = np.vstack([M2, [0.0004, -0.0003, 1.0]])
    general = {}
    for name, fn, mat in (("warpAffine", cv2.warpAffine, M2), ("warpPerspective", cv2.warpPerspective, H2)):
        out = {k: fn(img2, mat, (256, 256), flags=f).astype(int) for k, f in (("area", cv2.INTER_AREA), ("linear", cv2.INTER_LINEAR), ("nearest", cv2.INTER_NEAREST))}
        general[name] = {"传 INTER_AREA 与传 INTER_LINEAR 的最大像素差": int(np.abs(out["area"] - out["linear"]).max()),
                         "对照：INTER_LINEAR 与 INTER_NEAREST 的最大像素差": int(np.abs(out["linear"] - out["nearest"]).max())}
    NUM["缩小"] = {"说明": "480x480 的竖条纹，周期 3 px（1 px 亮 255、2 px 暗 0），缩小到四分之一（120x120）；缩小后条纹周期不到 1 px，理想结果是均匀的灰色",
                 "原图平均值": r(stripes.mean(), 1), "结果": rows,
                 "warpAffine 传 INTER_AREA 与传 INTER_LINEAR 的最大像素差": int(np.abs(wa.astype(int) - wl.astype(int)).max()),
                 "warpPerspective 传 INTER_AREA 与传 INTER_LINEAR 的最大像素差": int(np.abs(pa.astype(int) - pl.astype(int)).max()),
                 "第二组：256x256 合成图，缩小到 0.3 倍并旋转 17 度（warpPerspective 另加一点透视）": general,
                 "OpenCV 版本": cv2.__version__}

    lab = np.zeros((200, 200), np.uint8)
    cv2.circle(lab, (60, 70), 40, 1, -1)
    cv2.rectangle(lab, (90, 90), (180, 170), 3, -1)
    cv2.circle(lab, (140, 60), 30, 5, -1)
    M = cv2.getRotationMatrix2D((99.5, 99.5), 17, 1.0)
    Mi = cv2.invertAffineTransform(M)
    ys, xs = np.mgrid[:200, :200]
    sx, sy = Mi[0, 0] * xs + Mi[0, 1] * ys + Mi[0, 2], Mi[1, 0] * xs + Mi[1, 1] * ys + Mi[1, 2]
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)

    def at(yv, xv):
        ok = (yv >= 0) & (yv < 200) & (xv >= 0) & (xv < 200)
        return np.where(ok, lab[np.clip(yv, 0, 199), np.clip(xv, 0, 199)], 0)
    nb = np.stack([at(y0, x0), at(y0, x0 + 1), at(y0 + 1, x0), at(y0 + 1, x0 + 1)], 0)
    res = {}
    for name, flag in (("最近邻", cv2.INTER_NEAREST), ("双线性", cv2.INTER_LINEAR), ("双三次", cv2.INTER_CUBIC)):
        o = cv2.warpAffine(lab, M, (200, 200), flags=flag)
        foreign = ~(o[None] == nb).any(axis=0)
        illegal = ~np.isin(o, [0, 1, 3, 5])
        res[name] = {"输出里出现的类别编号": [int(v) for v in np.unique(o)], "编号不在原有类别里的像素数": int(illegal.sum()),
                     "编号合法、但周围 4 个源像素里没有这一类的像素数": int((foreign & ~illegal).sum())}
    NUM["掩码"] = {"说明": "200x200 的标签图，类别编号 0（背景）、1、3、5，8 位整数；绕中心旋转 17 度",
                 "四个源像素类别不全相同的目标像素数（类别边界）": int((nb.max(axis=0) != nb.min(axis=0)).sum()), "结果": res}


# ---------------------------------------------------------------- 图 3：仿射还是单应
def plane_views(f, z, tilt_deg):
    """800x600 mm 的平面，正对时刚好铺满 640x480；绕竖直轴倾斜 tilt_deg 后的投影。返回正对和倾斜两组像素坐标。"""
    K = np.array([[f, 0, 319.5], [0, f, 239.5], [0, 0, 1.0]])
    a = np.deg2rad(tilt_deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    plane = np.stack(np.meshgrid(np.linspace(-400, 400, 17), np.linspace(-300, 300, 13)), -1).reshape(-1, 2)
    front = tp(plane, K @ np.column_stack([[1, 0, 0], [0, 1, 0], [0, 0, z]]))
    tilted = tp(plane, K @ np.column_stack([R[:, 0], R[:, 1], [0, 0, z]]))
    return front, tilted


def fig_model():
    tilts = [0, 1, 2, 5, 10, 15, 20, 30, 45]
    cams = {"工作距离 1 m": (800.0, 1000.0), "工作距离 4 m（同样的放大倍率）": (3200.0, 4000.0)}
    bias = {name: {} for name in cams}
    for name, (f, z) in cams.items():
        for t in tilts:
            front, tilted = plane_views(f, z, t)
            d = np.sqrt(((tp(front, affine_fit(front, tilted)) - tilted) ** 2).sum(axis=1))
            three = [0, 16, 17 * 12]
            d3 = np.sqrt(((tp(front, affine_fit(front[three], tilted[three])) - tilted) ** 2).sum(axis=1))
            dh = np.sqrt(((tp(front, dlt(front, tilted)) - tilted) ** 2).sum(axis=1))
            bias[name][t] = {"仿射最小二乘：均方根（px）": r(np.sqrt((d ** 2).mean()), 2), "仿射最小二乘：最大（px）": r(d.max(), 1),
                             "用三个角求仿射：第四个角的误差（px）": r(d3[-1], 1), "单应：最大（px）": float(f"{dh.max():.1e}")}
    ns = [3, 4, 5, 6, 8, 12, 20, 50]
    var = {"仿射": [], "单应": []}
    for n in ns:
        rng = np.random.default_rng(1100 + n)
        ea, eh = [], []
        for _ in range(300):
            A = np.eye(3)
            A[:2, :2] += rng.uniform(-0.2, 0.2, (2, 2))
            A[:2, 2] = rng.uniform(-30, 30, 2)
            src = rng.uniform([0, 0], [W - 1, H - 1], (n, 2))
            dst = tp(src, A) + rng.normal(0, 1.0, (n, 2))
            ea.append(h_err(affine_fit(src, dst), A))
            if n >= 4:
                eh.append(h_err(dlt(src, dst), A))
        var["仿射"].append(med(ea))
        var["单应"].append(med(eh) if eh else None)
    NUM["仿射还是单应"] = {
        "模型不够的偏差": {"说明": "800x600 mm 的平面，正对相机时刚好铺满 640x480 的画面；绕竖直轴倾斜后，用仿射去拟合 17x13 个网格点从正对到倾斜的对应关系，无噪声；"
                              "两台相机焦距分别是 800 px 和 3200 px，工作距离 1 m 和 4 m，放大倍率相同", "倾斜角（度）": tilts,
                     "结果": {name: {str(t): v for t, v in d.items()} for name, d in bias.items()}},
        "自由度多的方差": {"说明": "真实变换就是仿射（线性部分在单位阵上随机加最多 0.2，平移最多 30 px），点在 640x480 内均匀随机，噪声 1 px，各 300 次；"
                              "误差是估出的变换把全画面 17x13 个网格点映射后和真值的均方根距离，取中位数", "点对数": ns,
                     "按仿射拟合（px）": [r(v, 2) for v in var["仿射"]], "按单应拟合（px）": [None if v is None else r(v, 2) for v in var["单应"]]},
    }

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 3.9), gridspec_kw={"width_ratios": [1.1, 1.2, 1.2]}, layout="constrained")
    front, tilted = plane_views(800.0, 1000.0, 20)
    pred = tp(front, affine_fit(front, tilted))
    for a_, b_ in zip(tilted, pred):
        ax[0].plot([a_[0], b_[0]], [a_[1], b_[1]], color=S.RED, lw=0.9)
    ax[0].plot(tilted[:, 0], tilted[:, 1], "o", color=S.BLUE, ms=3.2, label="真实位置（透视）")
    ax[0].plot(pred[:, 0], pred[:, 1], "x", color=S.ORANGE, ms=4.5, label="仿射拟合的位置")
    ax[0].set_xlim(-45, W + 45)
    ax[0].set_ylim(H + 70, -70)
    ax[0].set_aspect("equal")
    ax[0].set_title("平面倾斜 20°：仿射对不上的部分（红线）", fontsize=10.5)
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    S.image_axes(ax[0])
    for (name, d), color, marker in zip(bias.items(), (S.ORANGE, S.BLUE), ("s", "o")):
        ax[1].plot(tilts, [d[t]["仿射最小二乘：最大（px）"] for t in tilts], marker + "-", color=color, lw=1.8, ms=4.5, label=name)
    ax[1].set_xlabel("平面倾斜角（度）")
    ax[1].set_ylabel("仿射拟合的最大误差（px）")
    ax[1].set_title("模型不够：误差随倾斜角增长，离得远则小")
    ax[1].legend(fontsize=8.5)
    S.clean(ax[1])
    ax[2].plot(ns, var["仿射"], "s-", color=S.ORANGE, lw=1.8, ms=4.5, label="按仿射拟合（6 个参数）")
    ax[2].plot(ns[1:], var["单应"][1:], "o-", color=S.BLUE, lw=1.8, ms=4.5, label="按单应拟合（8 个参数）")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xticks(ns, [str(n) for n in ns])
    ax[2].set_yticks([0.3, 1, 3, 10], ["0.3", "1", "3", "10"])
    ax[2].minorticks_off()
    ax[2].set_xlabel("点对数（真实变换是仿射，噪声 1 px）")
    ax[2].set_ylabel("变换误差中位数（px）")
    ax[2].set_title("参数多了：同样的点，单应更怕噪声")
    ax[2].legend(fontsize=8.5)
    S.clean(ax[2])
    S.save(fig, CH, "fig03_仿射还是单应")


# ---------------------------------------------------------------- 图 4：点数、噪声和点的位置
def fig_points():
    ns = [4, 5, 6, 8, 12, 20, 50, 100, 200]
    sigmas = [0.5, 1.0, 2.0]
    err = {s: [] for s in sigmas}
    p90 = {s: [] for s in sigmas}
    resid = {s: [] for s in sigmas}
    for s in sigmas:
        for n in ns:
            rng = np.random.default_rng(1000 + n)
            e, res = [], []
            for _ in range(300):
                Ht = rand_h(rng)
                src = rng.uniform([0, 0], [W - 1, H - 1], (n, 2))
                dst = tp(src, Ht) + rng.normal(0, s, (n, 2))
                Hm = dlt(src, dst)
                e.append(h_err(Hm, Ht))
                res.append(float(np.sqrt(((tp(src, Hm) - dst) ** 2).sum(axis=1).mean())))
            err[s].append(med(e))
            p90[s].append(pct(e, 90))
            resid[s].append(med(res))
    NUM["点数与噪声"] = {
        "说明": "随机单应（画面四角各挪最多 60 px），点在 640x480 内均匀随机，归一化 DLT，各 300 次；真实误差是估出的单应把全画面 17x13 个网格点映射后和真值的均方根距离；"
              "拟合残差是估出的单应在参与拟合的那些点上的均方根重投影误差", "点对数": ns,
        "真实误差中位数（px）": {str(s): [r(v, 3) for v in err[s]] for s in sigmas},
        "真实误差 90% 分位（px）": {str(s): [r(v, 2) for v in p90[s]] for s in sigmas},
        "拟合残差中位数（px）": {str(s): [r(v, 3) for v in resid[s]] for s in sigmas},
        "噪声 1 px 时，真实误差 x sqrt(点对数)": [r(v * math.sqrt(n), 2) for v, n in zip(err[1.0], ns)],
    }

    inner = np.array([[240, 180], [400, 180], [400, 300], [240, 300.0]])
    place = {"画面四角": [], "画面内随机 4 点": [], "集中在中央 160x120": []}
    rng = np.random.default_rng(9)
    for _ in range(500):
        Ht = rand_h(rng)
        for name, src in (("画面四角", CORNERS), ("画面内随机 4 点", rng.uniform([0, 0], [W - 1, H - 1], (4, 2))), ("集中在中央 160x120", inner)):
            place[name].append(h_err(dlt(src, tp(src, Ht) + rng.normal(0, 1.0, (4, 2))), Ht))
    NUM["四个点的位置"] = {"说明": "只用 4 对点求单应，噪声 1 px，各 500 次，误差定义同上",
                     "结果": {k: {"中位数（px）": r(med(v), 2), "90% 分位（px）": r(pct(v, 90), 1)} for k, v in place.items()}}

    A, B, C, D = np.array([100, 100.0]), np.array([540, 100.0]), np.array([540, 380.0]), np.array([100, 380.0])
    center = (A + C) / 2
    d0 = abs((C[0] - A[0]) * (A[1] - D[1]) - (A[0] - D[0]) * (C[1] - A[1])) / np.linalg.norm(C - A)
    dists = [d0, 100, 50, 20, 10, 5, 2, 1]
    col = []
    for dist in dists:
        Dm = center + (D - center) * dist / d0
        src = np.stack([A, B, C, Dm])
        rng = np.random.default_rng(3)
        e = []
        for _ in range(300):
            Ht = rand_h(rng)
            e.append(h_err(dlt(src, tp(src, Ht) + rng.normal(0, 1.0, (4, 2))), Ht))
        col.append((med(e), pct(e, 90)))
    src = np.stack([A, B, C, center]).astype(np.float32)
    rng = np.random.default_rng(4)
    raised, on_pts, elsewhere = 0, [], []
    for _ in range(100):
        Ht = rand_h(rng)
        dst = tp(src, Ht).astype(np.float32)
        try:
            Hc = cv2.getPerspectiveTransform(src, dst).astype(float)
        except cv2.error:
            raised += 1
            continue
        with np.errstate(all="ignore"):
            back = np.sqrt(((tp(src, Hc) - dst) ** 2).sum(axis=1))
        on_pts.append(float(np.max(back)) if np.all(np.isfinite(back)) else np.inf)
        elsewhere.append(h_err(Hc, Ht))
    exact = {"说明": "第四个点正好放在第一、三个点连线的中点，目标点由真实单应算出、不加噪声，100 个随机单应",
             "抛出异常的次数": raised, "这 4 个点自己的重投影误差：最大（px）": float(f"{pct(on_pts, 100):.1e}"),
             "全画面网格点上的误差：中位数（px）": r(med(elsewhere), 1), "全画面网格点上的误差小于 1 px 的次数": int(sum(v < 1 for v in elsewhere))}
    with np.errstate(all="ignore"):
        A3 = cv2.getAffineTransform(np.float32([[0, 0], [100, 0], [200, 0]]), np.float32([[0, 0], [100, 10], [200, 30]]))
    NUM["接近共线"] = {"说明": "4 对点求单应，噪声 1 px，各 300 次；三个点固定在 (100,100)、(540,100)、(540,380)，第四个点从 (100,380) 沿对角线移向第一、三个点的连线",
                   "第四个点到连线的距离（px）": [r(d, 0) for d in dists], "误差中位数（px）": [r(v[0], 2) for v in col], "误差 90% 分位（px）": [r(v[1], 1) for v in col],
                   "三点完全共线时 getPerspectiveTransform": exact,
                   "三点完全共线时 getAffineTransform 返回的矩阵": [[r(v, 3) for v in row] for row in A3], "OpenCV 版本": cv2.__version__}

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0), gridspec_kw={"width_ratios": [1.3, 1, 1.15]}, layout="constrained")
    for s, color, marker in zip(sigmas, (S.GREEN, S.BLUE, S.ORANGE), ("^", "o", "s")):
        ax[0].plot(ns, err[s], marker + "-", color=color, lw=1.8, ms=4.5, label=f"真实误差，噪声 {s:g} px")
    ax[0].plot(ns, [max(v, 0.02) for v in resid[1.0]], "o--", color=S.MUTED, lw=1.4, ms=3.5, label="拟合残差，噪声 1 px")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xticks(ns, [str(n) for n in ns])
    ax[0].set_yticks([0.02, 0.1, 1, 10], ["0", "0.1", "1", "10"])
    ax[0].minorticks_off()
    ax[0].set_xlabel("点对数")
    ax[0].set_ylabel("误差中位数（px）")
    ax[0].set_title("点越多越准；残差却是点越少越小")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    S.clean(ax[0])
    names = list(place)
    vals = [med(place[k]) for k in names]
    bars = ax[1].bar(range(3), vals, 0.55, color=[S.GREEN, S.ORANGE, S.RED], alpha=0.85)
    for b, v in zip(bars, vals):
        ax[1].text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.1f}", ha="center", fontsize=10)
    ax[1].set_xticks(range(3), ["画面四角", "画面内\n随机 4 点", "集中在中央\n160x120"], fontsize=9.5)
    ax[1].set_ylabel("误差中位数（px），噪声 1 px")
    ax[1].set_ylim(0, max(vals) * 1.15)
    ax[1].set_title("同样 4 个点，位置决定精度")
    S.clean(ax[1])
    ax[1].grid(axis="x", visible=False)
    ax[2].plot(dists, [v[0] for v in col], "o-", color=S.RED, lw=1.8, ms=4.5)
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xticks([1, 2, 5, 10, 20, 50, 100, 236], ["1", "2", "5", "10", "20", "50", "100", "236"])
    ax[2].set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    ax[2].minorticks_off()
    ax[2].invert_xaxis()
    ax[2].set_xlabel("第四个点到另外两点连线的距离（px）")
    ax[2].set_ylabel("误差中位数（px），噪声 1 px")
    ax[2].set_title("越接近三点共线，误差越大")
    S.clean(ax[2])
    S.save(fig, CH, "fig04_点数噪声与点的位置")


# ---------------------------------------------------------------- 表：归一化
CV_LS = "cv2.findHomography（method=0，全部点最小二乘）"
BOARD = "不处理，但源点换成从 0 开始、宽 120 的坐标（模仿第 17 章的标定板坐标），目标点仍是像素坐标"


def table_normalization():
    modes = ("归一化", "只平移到质心", "不处理")
    dtypes = (("float64", np.float64), ("float32", np.float32))
    full = {}
    for n in (50, 200):
        for s in (0.5, 1.0, 2.0):
            rng = np.random.default_rng(77)
            e = {(m, dn): [] for m in modes for dn, _ in dtypes}
            ecv = []
            for _ in range(300):
                Ht = rand_h(rng)
                src = rng.uniform([0, 0], [W - 1, H - 1], (n, 2))
                dst = tp(src, Ht) + rng.normal(0, s, (n, 2))
                for m in modes:
                    for dn, dt in dtypes:
                        e[(m, dn)].append(h_err(dlt(src, dst, m, dt), Ht))
                ecv.append(h_err(cv2.findHomography(src, dst, 0)[0], Ht))
            full[f"{n} 对点，噪声 {s:g} px"] = {m: {dn: r(med(e[(m, dn)]), 3) for dn, _ in dtypes} for m in modes}
            full[f"{n} 对点，噪声 {s:g} px"][CV_LS] = r(med(ecv), 3)
    small = {}
    for frac in (1.0, 0.5, 0.25, 0.1):
        rng = np.random.default_rng(7)
        lo = np.array([(W - 1) * (1 - frac) / 2, (H - 1) * (1 - frac) / 2])
        hi = lo + np.array([(W - 1) * frac, (H - 1) * frac])
        g = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], 9), np.linspace(lo[1], hi[1], 9)), -1).reshape(-1, 2)
        e = {(m, dn): [] for m in modes for dn, _ in dtypes}
        ecv, eboard = [], []
        k120 = 120.0 / (hi[0] - lo[0])
        To = np.array([[k120, 0, -k120 * lo[0]], [0, k120, -k120 * lo[1]], [0, 0, 1.0]])
        for _ in range(300):
            Ht = rand_h(rng)
            src = rng.uniform(lo, hi, (20, 2))
            dst = tp(src, Ht) + rng.normal(0, 0.5, (20, 2))
            truth = tp(g, Ht)

            def local(Hm):
                with np.errstate(all="ignore"):
                    d = ((tp(g, Hm) - truth) ** 2).sum(axis=1)
                return float(np.sqrt(d.mean())) if np.all(np.isfinite(d)) else np.inf
            for m in modes:
                for dn, dt in dtypes:
                    e[(m, dn)].append(local(dlt(src, dst, m, dt)))
            ecv.append(local(cv2.findHomography(src, dst, 0)[0]))
            eboard.append(local(dlt(tp(src, To), dst, "不处理") @ To))
        key = f"点分布在画面中央 {round(639 * frac)}x{round(479 * frac)} px"
        small[key] = {m: {dn: r(med(e[(m, dn)]), 3) for dn, _ in dtypes} for m in modes}
        small[key][CV_LS] = r(med(ecv), 3)
        small[key][BOARD] = r(med(eboard), 3)
    NUM["归一化"] = {
        "说明": "DLT 用 SVD 求解，比较三种预处理：Hartley 归一化、只把两组点各自平移到质心、不处理；源点和目标点都是像素坐标；"
              "最后一项是把同样的点交给 OpenCV 的结果（float64 输入）",
        "点铺满画面": {"说明": "点在 640x480 内均匀随机，各 300 次；误差是全画面 17x13 网格点上的均方根（px），取中位数", "结果": full},
        "点集中在小区域": {"说明": "20 对点，噪声 0.5 px，各 300 次；误差只在点所在的区域内（9x9 网格）统计，均方根（px），取中位数", "结果": small},
    }


# ---------------------------------------------------------------- 图 5：RANSAC 的迭代次数
def fig_ransac():
    n, sigma, thresh = 100, 1.0, 3.0
    ratios_a = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    ls, ra, oracle = [], [], []
    for ratio in ratios_a:
        rng = np.random.default_rng(2000 + int(ratio * 100))
        a, b, c = [], [], []
        for _ in range(200):
            src, dst, is_out, Ht = make_matches(rng, n, ratio, sigma)
            a.append(h_err(dlt(src, dst), Ht))
            Hs, err, _ = hypotheses(src, dst, 500, rng)
            b.append(h_err(finish(src, dst, Hs, err, thresh)[0], Ht))
            c.append(h_err(dlt(src[~is_out], dst[~is_out]), Ht))
        ls.append(med(a))
        ra.append(med(b))
        oracle.append(med(c))
    NUM["最小二乘与 RANSAC"] = {
        "说明": "100 对匹配，内点噪声 1 px，外点的目标位置在画面内均匀随机，各 200 次；RANSAC 阈值 3 px、迭代 500 次、最后用内点重新拟合；误差是全画面网格点上的均方根（px），取中位数",
        "外点比例（%）": [r(v * 100, 0) for v in ratios_a], "全部点做最小二乘": [r(v, 2) for v in ls], "RANSAC": [r(v, 3) for v in ra],
        "只用真正的内点做最小二乘（理想情况）": [r(v, 3) for v in oracle],
    }

    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ks = [10, 50, 200, 1000]
    trials = 300
    succ = {k: [] for k in ks}
    clean = {k: [] for k in ks}
    formula = {k: [] for k in ks}
    exact = {k: [] for k in ks}
    cond = {"抽到过全内点样本": [0, 0], "没抽到过": [0, 0]}
    for ratio in ratios:
        rng = np.random.default_rng(3000 + int(ratio * 100))
        ok = {k: 0 for k in ks}
        cl = {k: 0 for k in ks}
        for _ in range(trials):
            src, dst, is_out, Ht = make_matches(rng, n, ratio, sigma)
            Hs, err, idx = hypotheses(src, dst, max(ks), rng)
            all_in = (~is_out[idx]).all(axis=1)
            for k in ks:
                good = h_err(finish(src, dst, Hs, err, thresh, k)[0], Ht) < 3.0
                has = bool(all_in[:k].any())
                ok[k] += good
                cl[k] += has
                key = "抽到过全内点样本" if has else "没抽到过"
                cond[key][0] += good
                cond[key][1] += 1
        w = 1 - ratio
        n_in = int(round(n * w))
        q = math.comb(n_in, 4) / math.comb(n, 4)
        for k in ks:
            succ[k].append(ok[k] / trials)
            clean[k].append(cl[k] / trials)
            formula[k].append(1 - (1 - w ** 4) ** k)
            exact[k].append(1 - (1 - q) ** k)
    NUM["迭代次数"] = {
        "说明": f"100 对匹配，内点噪声 1 px，阈值 3 px，各 {trials} 次；成功 = 最终单应在全画面网格点上的均方根误差小于 3 px；"
              "公式值 = 1-(1-w^4)^K，w 是内点比例；精确值按 100 个点里不放回抽 4 个算",
        "外点比例（%）": [r(v * 100, 0) for v in ratios],
        "结果": {str(k): {"公式值（%）": [r(v * 100, 1) for v in formula[k]], "不放回抽样的精确值（%）": [r(v * 100, 1) for v in exact[k]],
                         "实测：至少抽到一组全内点（%）": [r(v * 100, 1) for v in clean[k]], "实测：成功率（%）": [r(v * 100, 1) for v in succ[k]]} for k in ks},
        "上面所有格子合计": {k: {"次数": v[1], "其中成功（%）": r(100 * v[0] / max(v[1], 1), 1)} for k, v in cond.items()},
    }

    byk = {}
    mults = (1, 2, 3, 5)
    for ratio in (0.1, 0.3, 0.5, 0.7):
        w = 1 - ratio
        k99 = math.ceil(math.log(1 - 0.99) / math.log(1 - w ** 4))
        rng = np.random.default_rng(4000 + int(ratio * 100))
        ok = {m: 0 for m in mults}
        for _ in range(300):
            src, dst, is_out, Ht = make_matches(rng, n, ratio, sigma)
            Hs, err, _ = hypotheses(src, dst, k99 * max(mults), rng)
            for m in mults:
                ok[m] += h_err(finish(src, dst, Hs, err, thresh, k99 * m)[0], Ht) < 3.0
        byk[f"{ratio * 100:.0f}"] = {"公式给的迭代次数（置信度 0.99）": k99, "实测成功率（%）": {f"{m} 倍": r(100 * ok[m] / 300, 1) for m in mults}}
    table_k = {}
    for ratio in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        table_k[f"{ratio * 100:.0f}"] = {f"s={s}": math.ceil(math.log(1 - 0.99) / math.log(1 - (1 - ratio) ** s)) for s in (2, 3, 4)}
    NUM["按公式取迭代次数"] = {"说明": "同上的数据，迭代次数取公式值 K=log(1-p)/log(1-w^s) 的 1、2、3、5 倍，p=0.99，s=4，各 300 次", "按外点比例（%）": byk,
                       "公式值对照表（p=0.99；s=2、3、4 分别是相似、仿射、单应的最小样本）": table_k}

    quality = {}
    frac1 = None
    for s in (0.5, 1.0, 2.0):
        rng = np.random.default_rng(5)
        fr = []
        for _ in range(200):
            src, dst, _, _ = make_matches(rng, n, 0.0, s)
            _, err, _ = hypotheses(src, dst, 50, rng)
            fr.append((err < thresh).mean(axis=1))
        fr = np.concatenate(fr)
        if s == 1.0:
            frac1 = fr
        quality[f"噪声 {s:g} px"] = {"找回的内点比例中位数（%）": r(100 * np.median(fr), 0), "找回九成以上内点的样本占比（%）": r(100 * (fr >= 0.9).mean(), 1),
                                    "找回不到一半内点的样本占比（%）": r(100 * (fr < 0.5).mean(), 1),
                                    "找回不到两成内点的样本占比（%）": r(100 * (fr < 0.2).mean(), 1)}
    NUM["全内点样本的质量"] = {"说明": "100 对匹配全部是内点，随机抽 4 对求单应，数一数在 3 px 阈值下它能把多少比例的点判成内点；200 组数据，每组 50 个样本", "结果": quality}

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.1), gridspec_kw={"width_ratios": [1.05, 1.35, 1]}, layout="constrained")
    x = np.arange(len(ratios_a))
    ax[0].plot(x, ls, "s-", color=S.RED, lw=1.8, ms=4.5, label="全部点做最小二乘")
    ax[0].plot(x, ra, "o-", color=S.BLUE, lw=1.8, ms=4.5, label="RANSAC")
    ax[0].plot(x, oracle, "--", color=S.MUTED, lw=1.3, label="只用真正的内点（理想）")
    ax[0].set_yscale("log")
    ax[0].set_yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    ax[0].minorticks_off()
    ax[0].set_xticks(x, [f"{v * 100:.0f}" for v in ratios_a])
    ax[0].set_xlabel("外点比例（%）")
    ax[0].set_ylabel("单应误差中位数（px）")
    ax[0].set_title("100 对里错 1 对，最小二乘就差 20 倍")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    S.clean(ax[0])
    xr = [v * 100 for v in ratios]
    for k, color in zip(ks, (S.ORANGE, S.GREEN, S.BLUE, S.PURPLE)):
        ax[1].plot(xr, [v * 100 for v in formula[k]], "-", color=color, lw=1.2, alpha=0.75)
        ax[1].plot(xr, [v * 100 for v in succ[k]], "o", color=color, ms=5.5, label=f"K = {k}")
    ax[1].set_xlabel("外点比例（%）")
    ax[1].set_ylabel("成功率（%）")
    ax[1].set_ylim(-3, 103)
    ax[1].set_title("线是公式值，点是实测成功率：实测不高于公式")
    ax[1].legend(fontsize=8.5, loc="lower left")
    S.clean(ax[1])
    ax[2].hist(frac1 * 100, bins=np.arange(0, 101, 5), color=S.BLUE, alpha=0.85)
    ax[2].axvline(float(np.median(frac1) * 100), color=S.RED, lw=1.4, ls="--")
    ax[2].text(float(np.median(frac1) * 100) + 2, ax[2].get_ylim()[1] * 0.9, f"中位数 {np.median(frac1) * 100:.0f}%", color=S.RED, fontsize=9)
    ax[2].set_xlabel("一组全内点样本找回的内点比例（%）")
    ax[2].set_ylabel("样本数")
    ax[2].set_title("全内点的 4 对点，多数只找回小部分内点")
    S.clean(ax[2])
    S.save(fig, CH, "fig05_RANSAC迭代次数")


# ---------------------------------------------------------------- 图 6：RANSAC 的阈值
def fig_threshold():
    n, sigma = 100, 1.0
    ths = [0.5, 1, 1.5, 2, 3, 5, 8, 12, 20, 40]
    out = {}
    for ratio in (0.3, 0.5):
        rng = np.random.default_rng(6000 + int(ratio * 10))
        e = {t: [] for t in ths}
        rec = {t: [] for t in ths}
        fp = {t: [] for t in ths}
        for _ in range(300):
            src, dst, is_out, Ht = make_matches(rng, n, ratio, sigma)
            Hs, err, _ = hypotheses(src, dst, 1000, rng)
            for t in ths:
                Hb, mask = finish(src, dst, Hs, err, t)
                e[t].append(h_err(Hb, Ht))
                rec[t].append(float(mask[~is_out].mean()))
                fp[t].append(int(mask[is_out].sum()))
        out[ratio] = {"误差中位数（px）": [med(e[t]) for t in ths], "误差 90% 分位（px）": [pct(e[t], 90) for t in ths],
                      "内点被判为内点的比例（%）": [100 * float(np.mean(rec[t])) for t in ths], "每次混进来的外点数（平均）": [float(np.mean(fp[t])) for t in ths]}
    rayleigh = [100 * (1 - math.exp(-t * t / (2 * sigma ** 2))) for t in ths]
    NUM["阈值"] = {
        "说明": "100 对匹配，内点噪声 1 px（x、y 各 1 px），外点的目标位置在画面内均匀随机，迭代 1000 次，各 300 次；同一批假设换不同的阈值打分",
        "阈值（px）": ths,
        "结果": {f"外点 {ratio * 100:.0f}%": {k: [r(v, 3 if "中位数" in k else 2) for v in vals] for k, vals in d.items()} for ratio, d in out.items()},
        "理论值：真值模型下内点误差小于阈值的概率（%）": [r(v, 1) for v in rayleigh],
    }

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 3.9), layout="constrained")
    for ratio, color, marker in ((0.3, S.BLUE, "o"), (0.5, S.ORANGE, "s")):
        d = out[ratio]
        ax[0].plot(ths, d["误差中位数（px）"], marker + "-", color=color, lw=1.8, ms=4.5, label=f"外点 {ratio * 100:.0f}%，中位数")
        ax[0].plot(ths, d["误差 90% 分位（px）"], marker + ":", color=color, lw=1.2, ms=3, label=f"外点 {ratio * 100:.0f}%，90% 分位")
        ax[1].plot(ths, d["内点被判为内点的比例（%）"], marker + "-", color=color, lw=1.8, ms=4.5, label=f"实测，外点 {ratio * 100:.0f}%")
        ax[2].plot(ths, d["每次混进来的外点数（平均）"], marker + "-", color=color, lw=1.8, ms=4.5, label=f"外点 {ratio * 100:.0f}%")
    ax[1].plot(ths, rayleigh, "--", color=S.MUTED, lw=1.3, label="理论值（真值模型）")
    for a_ in ax:
        a_.set_xscale("log")
        a_.set_xticks([0.5, 1, 2, 3, 5, 12, 40], ["0.5", "1", "2", "3", "5", "12", "40"])
        a_.minorticks_off()
        a_.set_xlabel("内点阈值（px），内点噪声 1 px")
        S.clean(a_)
    ax[0].set_ylabel("单应误差（px）")
    ax[0].set_ylim(bottom=0)
    ax[0].set_title("阈值小于 2 倍噪声时误差明显变大")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax[1].set_ylabel("内点被判为内点的比例（%）")
    ax[1].set_ylim(0, 105)
    ax[1].set_title("太小的阈值把大部分内点也丢了")
    ax[1].legend(fontsize=8.5, loc="lower right")
    ax[2].set_ylabel("每次混进来的外点数（平均）")
    ax[2].set_ylim(bottom=0)
    ax[2].set_title("阈值放大到 12 px，混进来的外点仍不到 1 个")
    ax[2].legend(fontsize=8.5, loc="upper left")
    S.save(fig, CH, "fig06_RANSAC阈值")


# ---------------------------------------------------------------- 表：OpenCV 的 findHomography
def table_opencv():
    n = 100
    default = {}
    for ratio in (0.5, 0.7, 0.8, 0.85, 0.9):
        rng = np.random.default_rng(31 + int(ratio * 100))
        ok, inl = 0, []
        for _ in range(200):
            src, dst, _, Ht = make_matches(rng, n, ratio, 1.0)
            Hc, m = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            ok += h_err(Hc, Ht) < 3.0
            inl.append(int(m.sum()) if m is not None else 0)
        default[f"{ratio * 100:.0f}"] = {"真正的内点数": int(round(n * (1 - ratio))), "成功率（%）": r(100 * ok / 200, 1), "返回的内点数中位数": r(np.median(inl), 0)}
    rng = np.random.default_rng(5)
    inl = []
    for _ in range(200):
        src = rng.uniform([0, 0], [W - 1, H - 1], (n, 2))
        dst = rng.uniform([0, 0], [W - 1, H - 1], (n, 2))
        Hc, m = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        inl.append(int(m.sum()) if (Hc is not None and m is not None) else 0)
    same, shuffled = 0, []
    for i in range(20):
        src, dst, _, Ht = make_matches(np.random.default_rng(100 + i), n, 0.5, 1.0)
        H1, m1 = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        H2, m2 = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        same += bool(np.array_equal(H1, H2) and np.array_equal(m1, m2))
        perm = np.random.default_rng(i).permutation(n)
        H3, _ = cv2.findHomography(src[perm], dst[perm], cv2.RANSAC, 3.0)
        shuffled.append(h_err(H3, H1))
    lmeds = {}
    for ratio in (0.3, 0.45, 0.55, 0.7):
        rng = np.random.default_rng(12)
        ok = 0
        for _ in range(200):
            src, dst, _, Ht = make_matches(rng, n, ratio, 1.0)
            Hc, _ = cv2.findHomography(src, dst, cv2.LMEDS)
            ok += h_err(Hc, Ht) < 3.0
        lmeds[f"{ratio * 100:.0f}"] = r(100 * ok / 200, 1)
    capped = {}
    for k in (10, 50, 200):
        row = {}
        for ratio in (0.5, 0.6, 0.7):
            rng = np.random.default_rng(3000 + k + int(ratio * 100))
            ok = 0
            for _ in range(200):
                src, dst, _, Ht = make_matches(rng, n, ratio, 1.0)
                Hc, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=k)
                ok += h_err(Hc, Ht) < 3.0
            row[f"{ratio * 100:.0f}"] = {"成功率（%）": r(100 * ok / 200, 1), "公式值 1-(1-w^4)^K（%）": r(100 * (1 - (1 - (1 - ratio) ** 4) ** k), 1)}
        capped[str(k)] = row
    NUM["OpenCV 的 findHomography"] = {
        "说明": "cv2.findHomography(src, dst, cv2.RANSAC, 3.0)，其余参数用默认值；100 对匹配，内点噪声 1 px，各 200 次；成功的定义同上", "OpenCV 版本": cv2.__version__,
        "默认参数下按外点比例（%）": default,
        "把 maxIters 设成 K，按外点比例（%）": capped,
        "100 对全是随机匹配时返回的内点数": {"中位数": r(np.median(inl), 0), "最大": int(max(inl)), "返回了矩阵的次数": int(sum(v > 0 for v in inl)), "总次数": 200},
        "同一组输入调用两次，结果逐位相同的组数": f"{same}/20",
        "把点对的顺序打乱后再算，和原结果的差（全画面网格点均方根，px）": {"中位数": r(np.median(shuffled), 3), "最大": r(max(shuffled), 3)},
        "LMEDS 方法按外点比例（%）的成功率（%）": lmeds,
    }

    # 两套各自一致的结构：RANSAC 选哪一套
    rng = np.random.default_rng(8)
    pick_small = 0
    for _ in range(200):
        H1, H2 = rand_h(rng), rand_h(rng) @ np.array([[1, 0, 40], [0, 1, 25], [0, 0, 1.0]])
        src = rng.uniform([0, 0], [W - 1, H - 1], (n, 2))
        dst = rng.uniform([0, 0], [W - 1, H - 1], (n, 2))
        lab = rng.permutation(n)
        a, b = lab[:40], lab[40:85]
        dst[a] = tp(src[a], H1) + rng.normal(0, 1, (40, 2))
        dst[b] = tp(src[b], H2) + rng.normal(0, 1, (45, 2))
        Hs, err, _ = hypotheses(src, dst, 2000, rng)
        Hb, _ = finish(src, dst, Hs, err, 3.0)
        pick_small += h_err(Hb, H1) < h_err(Hb, H2)
    NUM["两套一致的结构"] = {"说明": "100 对匹配里 40 对服从单应 A、45 对服从另一个单应 B、15 对随机；本脚本的 RANSAC，阈值 3 px，迭代 2000 次，200 组数据",
                      "结果更接近 40 对那一套的比例（%）": r(100 * pick_small / 200, 1)}


# ---------------------------------------------------------------- 表：文档矫正
def table_document():
    f, z = 800.0, 700.0
    K = np.array([[f, 0, 319.5], [0, f, 239.5], [0, 0, 1.0]])
    rect = np.array([[-105, -148.5], [105, -148.5], [105, 148.5], [-105, 148.5]])

    def shot(R):
        return tp(rect, K @ np.column_stack([R[:, 0], R[:, 1], [0, 0, z]]))

    def cyclic(o):
        o = [int(v) for v in o]
        return len(set(o)) == 4 and any(o[k:] + o[:k] == [0, 1, 2, 3] for k in range(4))

    def by_sum_diff(p):
        s, d = p.sum(axis=1), p[:, 1] - p[:, 0]
        return [np.argmin(s), np.argmin(d), np.argmax(s), np.argmax(d)]

    def by_angle(p):
        c = p.mean(axis=0)
        return list(np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])))

    spins = [0, 10, 20, 30, 35, 40, 45]
    order = {}
    for tilt_max in (0, 30, 50):
        row_sd, row_an = [], []
        for spin in spins:
            rng = np.random.default_rng(700 + spin)
            bad_sd = bad_an = 0
            for _ in range(2000):
                a = np.deg2rad(spin)
                Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
                phi, t = rng.uniform(0, 2 * np.pi), np.deg2rad(rng.uniform(0, tilt_max))
                Rt, _ = cv2.Rodrigues(np.array([np.cos(phi), np.sin(phi), 0.0]) * t)
                p = shot(Rt @ Rz)
                bad_sd += not cyclic(by_sum_diff(p))
                bad_an += not cyclic(by_angle(p))
            row_sd.append(r(100 * bad_sd / 2000, 1))
            row_an.append(r(100 * bad_an / 2000, 1))
        order[f"倾斜上限 {tilt_max} 度"] = {"和差法出错（%）": row_sd, "按绕中心的角度排序出错（%）": row_an}
    NUM["角点排序"] = {"说明": "A4 纸（210x297 mm），焦距 800 px、距离 700 mm 的针孔相机；纸先在自己的平面内旋转给定角度，再朝随机方向倾斜（倾斜角在 0 到上限之间均匀取），各 2000 次；"
                         "出错 = 得到的四个角不是互不相同、按环形顺序排列的四个点", "纸面内旋转角（度）": spins, "结果": order}

    aspect = {}
    for tilt in (0, 10, 20, 30, 45):
        a = np.deg2rad(tilt)
        Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        p = shot(Rx)
        top, bottom = np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[3])
        left, right = np.linalg.norm(p[3] - p[0]), np.linalg.norm(p[2] - p[1])
        ratio = max(top, bottom) / max(left, right)
        aspect[str(tilt)] = {"上边（px）": r(top, 0), "下边（px）": r(bottom, 0), "左右边（px）": r(left, 0), "按最大边长算出的宽高比": r(ratio, 3),
                             "相对真实宽高比的偏差（%）": r(100 * (ratio / (210 / 297) - 1), 1)}
    NUM["输出尺寸的宽高比"] = {"说明": "同一张 A4 纸和相机，纸的上边向靠近相机的方向倾斜；按“上下边取较长的做宽、左右边取较长的做高”定输出尺寸",
                       "真实宽高比": r(210 / 297, 3), "按倾斜角（度）": aspect}


# ---------------------------------------------------------------- 表：零散的约定
def table_misc():
    M = cv2.getRotationMatrix2D((100, 100), 90, 1.0)
    th = np.deg2rad(90)
    book = np.array([[np.cos(th), -np.sin(th), 100 + 100 * np.sin(th) - 100 * np.cos(th)], [np.sin(th), np.cos(th), 100 - 100 * np.sin(th) - 100 * np.cos(th)], [0, 0, 1]])
    p = np.array([[150.0, 100.0]])
    keep = {}
    for ang in (10, 30, 45, 90):
        Mr = cv2.getRotationMatrix2D((319.5, 239.5), ang, 1.0)
        c = tp(CORNERS, np.vstack([Mr, [0, 0, 1]]))
        out = cv2.warpAffine(np.ones((H, W), np.uint8), Mr, (W, H), flags=cv2.INTER_NEAREST)
        keep[str(ang)] = {"留在原画布里的像素（%）": r(100 * out.sum() / (W * H), 1), "不裁切需要的画布": f"{math.ceil(np.ptp(c[:, 0]) + 1)}x{math.ceil(np.ptp(c[:, 1]) + 1)}"}
    img = scene(256).astype(np.float32)
    big = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    w0 = cv2.warpAffine(img, np.array([[2, 0, 0], [0, 2, 0.0]]), (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    w1 = cv2.warpAffine(img, np.array([[2, 0, 0.5], [0, 2, 0.5]]), (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    inner = (slice(16, -16), slice(16, -16))
    a = np.deg2rad(30)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    T = np.array([[1, 0, 100], [0, 1, 0], [0, 0, 1.0]])
    q = np.array([[50.0, 20.0]])
    Hd = np.array([[1, 0.2, 3], [0.1, 1, 2], [0.01, 0.02, 1]])
    line = tp(np.array([[-99.0, 0], [-100.0, 0], [-101.0, 0]]), Hd)
    NUM["约定"] = {
        "旋转方向": {"说明": "中心 (100,100)，角度 +90 度，测试点 (150,100) 在中心正右方",
                 "getRotationMatrix2D 把它映射到": [r(v, 1) for v in tp(p, np.vstack([M, [0, 0, 1]]))[0]],
                 "按 [[cos,-sin],[sin,cos]] 绕同一中心映射到": [r(v, 1) for v in tp(p, book)[0]],
                 "getRotationMatrix2D 的左上 2x2": [[r(v, 3) + 0.0 for v in row[:2]] for row in M]},
        "旋转后留在原画布里的比例（640x480，绕中心）": keep,
        "resize 与 warpAffine 放大 2 倍的差（256x256 合成图，灰度级，去掉边缘 16 px）": {
            "矩阵只写缩放 [[2,0,0],[0,2,0]]：最大差": r(np.abs(big - w0)[inner].max(), 1), "矩阵只写缩放：均方根差": r(np.sqrt(((big - w0)[inner] ** 2).mean()), 2),
            "矩阵补上 0.5 px 平移 [[2,0,0.5],[0,2,0.5]]：最大差": float(f"{np.abs(big - w1)[inner].max():.1e}")},
        "组合顺序": {"说明": "点 (50,20)，R 是绕原点转 30 度，T 是沿 x 平移 100 px", "T@R（先转后移）": [r(v, 1) for v in tp(q, T @ R)[0]], "R@T（先移后转）": [r(v, 1) for v in tp(q, R @ T)[0]],
                 "两个结果相距（px）": r(np.linalg.norm(tp(q, T @ R) - tp(q, R @ T)), 1)},
        "齐次第三分量过零": {"说明": "单应 [[1,0.2,3],[0.1,1,2],[0.01,0.02,1]]（作业文件演示用的矩阵），直线 y=0 上相邻的三个点；参考实现的输出",
                     "(-99,0)，第三分量 +0.01": [r(v, 0) for v in line[0]], "(-100,0)，第三分量 0": [None if np.isnan(v) else r(v, 0) for v in line[1]],
                     "(-101,0)，第三分量 -0.01": [r(v, 0) for v in line[2]]},
    }


def main() -> None:
    S.setup()
    for fn in (fig_forward, fig_interp, table_resize_mask, fig_model, fig_points, table_normalization, fig_ransac, fig_threshold,
               table_opencv, table_document, table_misc):
        fn()
        print("完成", fn.__name__, flush=True)
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False))


if __name__ == "__main__":
    main()
