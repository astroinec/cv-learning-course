"""第 17 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch17.py

输出：<主分支>/17_相机标定/figures/*.svg 和 实测数字.json。
线性解用本章的参考实现；带非线性优化的对照用 OpenCV 的 calibrateCamera（需要 opencv-python）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

CH = "17_相机标定"
mod = S.load_solution(CH, "17-1_张正友标定.py")
NUM: dict = {}

W, H = 640, 480
K_TRUE = np.array([[820.0, 0.0, 318.0], [0.0, 790.0, 247.0], [0.0, 0.0, 1.0]])
_xs, _ys = np.meshgrid(np.arange(9) * 15.0, np.arange(6) * 15.0)
BOARD = np.stack([_xs.ravel(), _ys.ravel()], axis=1)
BOARD = BOARD - BOARD.mean(axis=0)
BOARD3 = np.hstack([BOARD, np.zeros((len(BOARD), 1))])
OUTLINE = np.array([[-60, -37.5, 0], [60, -37.5, 0], [60, 37.5, 0], [-60, 37.5, 0], [-60, -37.5, 0]], dtype=float)


# ---------------------------------------------------------------- 合成相机
def rot_xyz(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def project(K, R, t, P, k1=0.0, k2=0.0):
    pc = P @ R.T + t
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    r2 = x * x + y * y
    f = 1 + k1 * r2 + k2 * r2 * r2
    return np.stack([K[0, 0] * x * f + K[0, 2], K[1, 1] * y * f + K[1, 2]], axis=1)


def make_view(rng, tilt_deg, spread, k1=0.0, k2=0.0):
    """随机摆一次标定板。tilt_deg 是绕 x、y 轴倾斜的上限；spread 是板中心偏离画面中心的范围（0 到 1）。"""
    for _ in range(2000):
        rx, ry = np.deg2rad(rng.uniform(-tilt_deg, tilt_deg, 2))
        R = rot_xyz(rx, ry, np.deg2rad(rng.uniform(-20, 20)))
        z = rng.uniform(380, 520)
        u0 = K_TRUE[0, 2] + spread * rng.uniform(-1, 1) * W / 2
        v0 = K_TRUE[1, 2] + spread * rng.uniform(-1, 1) * H / 2
        t = z * np.linalg.inv(K_TRUE) @ np.array([u0, v0, 1.0])
        uv = project(K_TRUE, R, t, BOARD3, k1, k2)
        if uv[:, 0].min() > 2 and uv[:, 0].max() < W - 2 and uv[:, 1].min() > 2 and uv[:, 1].max() < H - 2:
            return uv, R, t
    raise RuntimeError("找不到落在画面内的姿态")


def make_views(rng, n, tilt_deg, spread, noise, k1=0.0, k2=0.0):
    out = []
    for _ in range(n):
        uv, R, t = make_view(rng, tilt_deg, spread, k1, k2)
        out.append(uv + rng.normal(scale=noise, size=uv.shape))
    return out


# ---------------------------------------------------------------- 两种标定
def calib_linear(uvs):
    """本章作业的线性解：DLT 求单应，Zhang 闭式解求内参，再逐视图恢复外参。"""
    with np.errstate(all="ignore"):
        Hs = [mod.homography_dlt(BOARD, uv) for uv in uvs]
        K = mod.intrinsics_from_homographies(Hs)
    if not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
        return None
    sq = []
    for Hm, uv in zip(Hs, uvs):
        R, t = mod.extrinsics_from_homography(K, Hm)
        sq.append(mod.reprojection_rmse(K, R, t, BOARD3, uv) ** 2)
    return K, float(np.sqrt(np.mean(sq)))


def calib_cv(uvs, distortion=False):
    """OpenCV 的 calibrateCamera：线性初值加 Levenberg-Marquardt 优化。"""
    obj = [BOARD3.astype(np.float32)] * len(uvs)
    img = [uv.astype(np.float32).reshape(-1, 1, 2) for uv in uvs]
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    if not distortion:
        flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
    try:
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, (W, H), None, None, flags=flags)
    except cv2.error:
        return None
    if not np.all(np.isfinite(K)):
        return None
    return K, float(rms), dist.ravel(), rvecs, tvecs


def summarize(results):
    ok = [r for r in results if r is not None]
    fx = np.array([abs(r[0][0, 0] - 820) / 820 * 100 for r in ok])
    cx = np.array([abs(r[0][0, 2] - 318) for r in ok])
    rms = np.array([r[1] for r in ok])
    return {"失败次数": len(results) - len(ok), "fx 相对误差中位数（%）": round(float(np.median(fx)), 3),
            "fx 相对误差 90% 分位（%）": round(float(np.percentile(fx, 90)), 2), "cx 误差中位数（px）": round(float(np.median(cx)), 2),
            "重投影 RMS 中位数（px）": round(float(np.median(rms)), 3)}


# ---------------------------------------------------------------- 图 1：姿态多样性
def fig_tilt():
    tilts = [2, 5, 10, 20, 30, 45]
    trials = 200
    table = {"线性解": {}, "线性解 + 非线性优化": {}}
    for tilt in tilts:
        rng = np.random.default_rng(100 + tilt)
        lin, opt = [], []
        for _ in range(trials):
            uvs = make_views(rng, 8, tilt, 0.5, 0.3)
            lin.append(calib_linear(uvs))
            opt.append(calib_cv(uvs))
        table["线性解"][str(tilt)] = summarize(lin)
        table["线性解 + 非线性优化"][str(tilt)] = summarize(opt)
    NUM["姿态多样性"] = {"说明": "同图 1 的相机和标定板，每次 8 张视图，角点噪声 0.3 px，无畸变；绕 x、y 轴的倾斜角在正负上限之间均匀取，绕光轴另有正负 20 度的旋转，各 200 次", "倾斜上限（度）": tilts, "结果": table}

    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.7), gridspec_kw={"width_ratios": [1, 1, 1.25, 1.25]}, layout="constrained")
    for a, tilt, color in ((ax[0], 5, S.ORANGE), (ax[1], 30, S.BLUE)):
        for _ in range(8):
            _, R, t = make_view(rng, tilt, 0.5)
            o = project(K_TRUE, R, t, OUTLINE)
            a.plot(o[:, 0], o[:, 1], color=color, lw=1.4)
        a.set_xlim(0, W)
        a.set_ylim(H, 0)
        a.set_aspect("equal")
        a.set_title(f"8 张视图，倾斜上限 {tilt}°")
        S.image_axes(a)
    for name, color, marker in (("线性解", S.ORANGE, "s"), ("线性解 + 非线性优化", S.BLUE, "o")):
        med = [table[name][str(t)]["fx 相对误差中位数（%）"] for t in tilts]
        rms = [table[name][str(t)]["重投影 RMS 中位数（px）"] for t in tilts]
        ax[2].plot(tilts, med, marker + "-", color=color, lw=1.8, ms=5, label=name)
        ax[3].plot(tilts, rms, marker + "-", color=color, lw=1.8, ms=5, label=name)
    ax[2].set_yscale("log")
    ax[2].set_yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    ax[2].minorticks_off()
    ax[2].set_xlabel("倾斜角上限（度）")
    ax[2].set_ylabel("fx 相对误差中位数（%）")
    ax[2].set_title("焦距误差差了两个数量级")
    ax[2].legend(fontsize=8.5)
    S.clean(ax[2])
    ax[3].axhline(0.3 * np.sqrt(2), color=S.MUTED, lw=1, ls="--")
    ax[3].text(45, 0.33, "虚线：角点噪声对应的 0.42 px", color=S.MUTED, fontsize=8.5, ha="right")
    ax[3].set_ylim(0, 0.6)
    ax[3].set_xlabel("倾斜角上限（度）")
    ax[3].set_ylabel("重投影 RMS 中位数（px）")
    ax[3].set_title("重投影误差却几乎不变")
    S.clean(ax[3])
    S.save(fig, CH, "fig03_姿态多样性")


# ---------------------------------------------------------------- 图 2：视图数与噪声
def fig_views():
    ns = [3, 4, 6, 8, 12, 20]
    noises = [0.1, 0.3, 1.0]
    trials = 200
    by_n = {"线性解": {}, "线性解 + 非线性优化": {}}
    by_noise = {"线性解": {}, "线性解 + 非线性优化": {}}
    for n in ns:
        rng = np.random.default_rng(200 + n)
        lin, opt = [], []
        for _ in range(trials):
            uvs = make_views(rng, n, 30, 0.5, 0.3)
            lin.append(calib_linear(uvs))
            opt.append(calib_cv(uvs))
        by_n["线性解"][str(n)] = summarize(lin)
        by_n["线性解 + 非线性优化"][str(n)] = summarize(opt)
    for noise in noises:
        rng = np.random.default_rng(300)
        lin, opt = [], []
        for _ in range(trials):
            uvs = make_views(rng, 8, 30, 0.5, noise)
            lin.append(calib_linear(uvs))
            opt.append(calib_cv(uvs))
        by_noise["线性解"][str(noise)] = summarize(lin)
        by_noise["线性解 + 非线性优化"][str(noise)] = summarize(opt)
    NUM["视图数与噪声"] = {"说明": "640x480 合成相机，9x6 标定板，倾斜上限 30 度，无畸变，各 200 次；按视图数的一组噪声 0.3 px，按噪声的一组用 8 张视图",
                     "视图数": ns, "按视图数": by_n, "噪声（px）": noises, "按噪声": by_noise}

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    for name, color, marker in (("线性解", S.ORANGE, "s"), ("线性解 + 非线性优化", S.BLUE, "o")):
        ax[0].plot(ns, [by_n[name][str(n)]["fx 相对误差中位数（%）"] for n in ns], marker + "-", color=color, lw=1.8, ms=5, label=name)
        ax[1].plot(noises, [by_noise[name][str(v)]["fx 相对误差中位数（%）"] for v in noises], marker + "-", color=color, lw=1.8, ms=5, label=name)
    ax[0].set_xscale("log")
    ax[0].set_xticks(ns, [str(n) for n in ns])
    ax[0].minorticks_off()
    ax[0].set_xlabel("视图数")
    ax[0].set_ylabel("fx 相对误差中位数（%）")
    ax[0].set_ylim(bottom=0)
    ax[0].set_title("视图越多越准，但收益递减")
    ax[0].legend(fontsize=9)
    S.clean(ax[0])
    ax[1].set_xlabel("角点噪声（px）")
    ax[1].set_ylabel("fx 相对误差中位数（%）")
    ax[1].set_ylim(bottom=0)
    ax[1].set_xlim(left=0)
    ax[1].set_title("误差和角点噪声成正比")
    S.clean(ax[1])
    S.save(fig, CH, "fig01_视图数与噪声")


# ---------------------------------------------------------------- 图 3：归一化
def dlt(src, dst, normalize, dtype, solver="svd"):
    """DLT 求单应。solver="svd" 直接对方程矩阵做 SVD；solver="eig" 先乘成 A^T A 再做特征分解（法方程）。"""
    src, dst = src.astype(dtype), dst.astype(dtype)

    def norm(p):
        c = p.mean(axis=0)
        s = dtype(np.sqrt(2)) / np.sqrt(((p - c) ** 2).sum(axis=1)).mean()
        return (p - c) * s, np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]], dtype=dtype)

    if normalize:
        sn, Ts = norm(src)
        dn, Td = norm(dst)
    else:
        sn, dn, Ts, Td = src, dst, np.eye(3, dtype=dtype), np.eye(3, dtype=dtype)
    rows = []
    for (x, y), (u, v) in zip(sn, dn):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(rows, dtype=dtype)
    if solver == "svd":
        Hn = np.linalg.svd(A)[2][-1].reshape(3, 3)
    else:
        Hn = np.linalg.eigh(A.T @ A)[1][:, 0].reshape(3, 3)
    Hm = (np.linalg.inv(Td) @ Hn @ Ts).astype(np.float64)
    sv = np.linalg.svd(A.astype(np.float64), compute_uv=False)
    return Hm / Hm[2, 2], float(sv[0] / sv[-2])


def fig_normalization():
    scenes = {
        "常规：板 0 到 120 mm，像素 0 到 640": (BOARD - BOARD.min(axis=0), 0.0, 1.0),
        "点集中在小区域：图像缩小 8 倍": (BOARD - BOARD.min(axis=0), 0.0, 8.0),
        "大数值：板换成微米，像素加 4000": ((BOARD - BOARD.min(axis=0)) * 1000.0, 4000.0, 1.0),
    }
    solvers = [("SVD，float64", "svd", np.float64), ("SVD，float32", "svd", np.float32),
               ("法方程，float64", "eig", np.float64), ("法方程，float32", "eig", np.float32)]
    out = {}
    for sname, (src, offset, shrink) in scenes.items():
        rng = np.random.default_rng(400)
        errs = {(v[0], nm): [] for v in solvers for nm in (True, False)}
        ratio = {True: [], False: []}
        extent = []
        for _ in range(300):
            clean, _, _ = make_view(rng, 30, 0.5)
            center = clean.mean(axis=0)
            clean = (clean - center) / shrink + center + offset
            noisy = clean + rng.normal(scale=0.5, size=clean.shape)
            extent.append(np.ptp(clean, axis=0))
            for vname, solver, dtype in solvers:
                for nm in (True, False):
                    with np.errstate(all="ignore"):
                        try:
                            Hm, gap = dlt(src, noisy, nm, dtype, solver)
                            ph = np.hstack([src, np.ones((len(src), 1))]) @ Hm.T
                            e = float(np.sqrt(np.mean(np.sum((ph[:, :2] / ph[:, 2:3] - clean) ** 2, axis=1))))
                        except np.linalg.LinAlgError:
                            e, gap = np.inf, np.nan
                    errs[(vname, nm)].append(e if np.isfinite(e) else 1e9)
                    if solver == "svd" and dtype is np.float64:
                        ratio[nm].append(gap)
        out[sname] = {
            "角点在图像上占的范围（宽、高的中位数，px）": [round(float(v)) for v in np.median(extent, axis=0)],
            "单应的误差中位数（px）": {v[0]: {"归一化": float(f"{np.median(errs[(v[0], True)]):.3g}"),
                                          "不归一化": float(f"{np.median(errs[(v[0], False)]):.3g}")} for v in solvers},
            "方程矩阵最大奇异值与第二小奇异值之比的中位数": {"归一化": float(f"{np.nanmedian(ratio[True]):.3g}"), "不归一化": float(f"{np.nanmedian(ratio[False]):.3g}")},
        }
    NUM["归一化"] = {"说明": "54 个角点，噪声 0.5 px，300 个随机视图；误差是估出的单应把板上的点映射到图像后，和无噪声位置的均方根距离；"
                         "SVD 指直接分解方程矩阵 A，法方程指先算 A^T A 再做特征分解；缩小 8 倍指把角点的图像坐标绕各自的中心缩到八分之一", "结果": out}

    # 验收用的条件：两组坐标都带很大的整体偏移，无噪声，直接对 A 做 SVD（float64）。用单独的随机数发生器，不影响上面的结果和图
    base = BOARD - BOARD.min(axis=0)
    shifted = {}
    for offset in (1e4, 1e6):
        rng = np.random.default_rng(401)
        worst = {True: [], False: []}
        for _ in range(100):
            clean, _, _ = make_view(rng, 30, 0.5)
            src, dst = base + offset, clean + offset
            for nm in (True, False):
                Hm, _ = dlt(src, dst, nm, np.float64, "svd")
                ph = np.hstack([src, np.ones((len(src), 1))]) @ Hm.T
                worst[nm].append(float(np.abs(ph[:, :2] / ph[:, 2:3] - dst).max()))
        shifted[f"两组坐标都加 {int(offset)}"] = {"归一化": float(f"{np.median(worst[True]):.2g}"), "不归一化": float(f"{np.median(worst[False]):.2g}")}
    NUM["归一化"]["整体偏移很大时"] = {"说明": "54 个角点，无噪声，100 个随机视图，板坐标和像素坐标都加上同一个整体偏移；SVD，float64；"
                                              "数值是每个视图最大映射误差的中位数（px）", "结果": shifted}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.0), layout="constrained", sharey=True)
    x = np.arange(4)
    for a, (sname, res) in zip(ax, out.items()):
        table = res["单应的误差中位数（px）"]
        yes = [min(table[v[0]]["归一化"], 2e4) for v in solvers]
        no = [min(table[v[0]]["不归一化"], 2e4) for v in solvers]
        for vals, dx, color, label in ((yes, -0.2, S.BLUE, "归一化"), (no, 0.2, S.ORANGE, "不归一化")):
            bars = a.bar(x + dx, vals, 0.38, color=color, alpha=0.85, label=label)
            for b, v in zip(bars, vals):
                a.text(b.get_x() + b.get_width() / 2, v * 1.3, f"{v:.3f}" if v < 1 else f"{v:.0f}", ha="center", fontsize=8.5)
        a.set_xticks(x, ["SVD\nfloat64", "SVD\nfloat32", "法方程\nfloat64", "法方程\nfloat32"], fontsize=9)
        a.set_yscale("log")
        a.set_ylim(0.05, 2e5)
        a.set_yticks([0.1, 1, 10, 100, 1000, 10000], ["0.1", "1", "10", "100", "1000", "10000"])
        a.minorticks_off()
        a.set_title(sname, fontsize=10.5)
        S.clean(a)
        a.grid(axis="x", visible=False)
    ax[0].set_ylabel("单应误差中位数（px，对数轴）")
    ax[0].legend(fontsize=9, loc="upper left")
    S.save(fig, CH, "fig02_归一化")


# ---------------------------------------------------------------- 图 4：没建模的畸变
def fig_distortion():
    k1 = -0.1
    trials = 100
    stats = {"不建畸变模型": [], "建 k1、k2": [], "线性解（不含畸变）": []}
    signed = []
    for i in range(trials):
        rng = np.random.default_rng(500 + i)
        uvs = make_views(rng, 12, 30, 0.85, 0.2, k1=k1)
        a = calib_cv(uvs, distortion=False)
        b = calib_cv(uvs, distortion=True)
        c = calib_linear(uvs)
        if a is None or b is None or c is None:
            continue
        stats["不建畸变模型"].append((a[1], abs(a[0][0, 0] - 820) / 820 * 100, 0.0))
        signed.append((a[0][0, 0] - 820) / 820 * 100)
        stats["建 k1、k2"].append((b[1], abs(b[0][0, 0] - 820) / 820 * 100, b[2][0]))
        stats["线性解（不含畸变）"].append((c[1], abs(c[0][0, 0] - 820) / 820 * 100, 0.0))
    corner = project(K_TRUE, np.eye(3), np.zeros(3), np.array([[(W - 318) / 820, (H - 247) / 790, 1.0]]), k1)[0]
    NUM["没建模的畸变"] = {
        "说明": f"真实相机带径向畸变 k1={k1}（画面右下角的点被向内拉了约 {np.hypot(W - corner[0], H - corner[1]):.1f} px），12 张视图覆盖全画面，角点噪声 0.2 px，各 {trials} 次",
        "结果": {name: {"重投影 RMS 中位数（px）": round(float(np.median([v[0] for v in vals])), 3),
                       "fx 相对误差中位数（%）": round(float(np.median([v[1] for v in vals])), 3)} for name, vals in stats.items()},
        "建模时估出的 k1 中位数": round(float(np.median([v[2] for v in stats["建 k1、k2"]])), 4),
        "不建畸变模型时 fx 带符号偏差的 5%、50%、95% 分位（%）": [round(float(np.percentile(signed, q)), 2) for q in (5, 50, 95)],
    }

    # 画图用一次 40 张视图的标定：点多，按位置取平均后噪声被压低，系统偏差才看得见
    rng = np.random.default_rng(42)
    uvs = make_views(rng, 40, 30, 0.85, 0.2, k1=k1)
    obj = BOARD3.astype(np.float32)
    gy, gx = 6, 8
    edges = np.linspace(0, 350, 8)
    mids = (edges[:-1] + edges[1:]) / 2
    fields, profiles, one = {}, {}, {}
    for name, with_dist in (("不建畸变模型", False), ("建 k1、k2", True)):
        K, rms, dist, rvecs, tvecs = calib_cv(uvs, distortion=with_dist)
        pts, vec = [], []
        for uv, rv, tv in zip(uvs, rvecs, tvecs):
            pts.append(uv)
            vec.append(uv - cv2.projectPoints(obj, rv, tv, K, dist)[0].reshape(-1, 2))
        pts, vec = np.vstack(pts), np.vstack(vec)
        iy = np.clip((pts[:, 1] / H * gy).astype(int), 0, gy - 1)
        ix = np.clip((pts[:, 0] / W * gx).astype(int), 0, gx - 1)
        n, sx, sy = np.zeros((gy, gx)), np.zeros((gy, gx)), np.zeros((gy, gx))
        np.add.at(n, (iy, ix), 1)
        np.add.at(sx, (iy, ix), vec[:, 0])
        np.add.at(sy, (iy, ix), vec[:, 1])
        fields[name] = (np.round(sx / np.maximum(n, 1), 5), np.round(sy / np.maximum(n, 1), 5), n)  # 取整原因见 fig_coverage
        d = pts - np.array([K[0, 2], K[1, 2]])
        r = np.hypot(d[:, 0], d[:, 1])
        radial = (vec * d).sum(axis=1) / np.maximum(r, 1e-9)
        profiles[name] = [round(float(radial[(r >= lo) & (r < hi)].mean()), 5) if ((r >= lo) & (r < hi)).sum() >= 20 else np.nan for lo, hi in zip(edges[:-1], edges[1:])]
        mag = np.hypot(*fields[name][:2])[n >= 10]
        one[name] = {"重投影 RMS（px）": round(rms, 3), "fx": round(float(K[0, 0]), 1), "fy": round(float(K[1, 1]), 1),
                     "分格平均后残差的最大值（px）": round(float(mag.max()), 3), "分格平均后残差的中位数（px）": round(float(np.median(mag)), 3),
                     "径向残差按半径分档的平均（px）": [None if np.isnan(v) else round(v, 3) for v in profiles[name]]}
    NUM["没建模的畸变"]["单次 40 张视图的标定（画图用）"] = {"说明": "把 2160 个角点的残差按 8x6 的格子取平均；径向残差为正表示观测点比模型预测的更靠外",
                                                    "半径分档的中点（px）": [round(float(m)) for m in mids], "结果": one}

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw={"width_ratios": [1.15, 1.15, 1.2]}, layout="constrained")
    cy_, cx_ = (np.arange(gy) + 0.5) * H / gy, (np.arange(gx) + 0.5) * W / gx
    X, Y = np.meshgrid(cx_, cy_)
    for a_, name, color in ((ax[0], "不建畸变模型", S.RED), (ax[1], "建 k1、k2", S.BLUE)):
        fx_, fy_, n = fields[name]
        keep = n >= 10
        a_.quiver(X[keep], Y[keep], fx_[keep], fy_[keep], angles="xy", scale_units="xy", scale=1 / 120, color=color, width=0.006)
        a_.set_xlim(0, W)
        a_.set_ylim(H, 0)
        a_.set_aspect("equal")
        a_.set_title(f"{name}：分格平均的残差（RMS {one[name]['重投影 RMS（px）']:.2f} px）", fontsize=10.5)
        S.image_axes(a_)
    ax[0].text(8, H - 10, "箭头放大 120 倍", fontsize=8.5, color=S.MUTED)
    for name, color, marker in (("不建畸变模型", S.RED, "s"), ("建 k1、k2", S.BLUE, "o")):
        ax[2].plot(mids, profiles[name], marker + "-", color=color, lw=1.8, ms=5, label=name)
    ax[2].axhline(0, color=S.INK, lw=0.6)
    ax[2].set_xlabel("角点到主点的距离（px）")
    ax[2].set_ylabel("径向残差的平均值（px）")
    ax[2].set_title("系统偏差藏在平均值里")
    ax[2].legend(fontsize=9)
    S.clean(ax[2])
    S.save(fig, CH, "fig04_没建模的畸变")


# ---------------------------------------------------------------- 图 5：覆盖范围与留出验证
def fig_coverage():
    k1 = -0.1
    cases = {"只在画面中央": 0.15, "覆盖全画面": 0.85}
    gy, gx = 12, 16
    acc = {name: [np.zeros((gy, gx)), np.zeros((gy, gx))] for name in cases}
    train, held, k1s = {n: [] for n in cases}, {n: [] for n in cases}, {n: [] for n in cases}
    cover = {}
    obj = BOARD3.astype(np.float32)
    for i in range(30):
        rng = np.random.default_rng(600 + i)
        tests = make_views(rng, 40, 30, 0.85, 0.2, k1=k1)
        for name, spread in cases.items():
            uvs = make_views(rng, 12, 30, spread, 0.2, k1=k1)
            res = calib_cv(uvs, distortion=True)
            if res is None:
                continue
            K, rms, dist, _, _ = res
            if name not in cover:
                cover[name] = np.vstack(uvs)
            train[name].append(rms)
            k1s[name].append(float(dist[0]))
            sq = []
            for uv in tests:
                ok, rv, tv = cv2.solvePnP(obj, uv.astype(np.float32).reshape(-1, 1, 2), K, dist)
                e = np.hypot(*(uv - cv2.projectPoints(obj, rv, tv, K, dist)[0].reshape(-1, 2)).T)
                sq.append(e ** 2)
                iy = np.clip((uv[:, 1] / H * gy).astype(int), 0, gy - 1)
                ix = np.clip((uv[:, 0] / W * gx).astype(int), 0, gx - 1)
                np.add.at(acc[name][0], (iy, ix), e)
                np.add.at(acc[name][1], (iy, ix), 1)
            held[name].append(float(np.sqrt(np.mean(np.concatenate(sq)))))
    maps = {name: np.divide(a[0], a[1], out=np.full((gy, gx), np.nan), where=a[1] > 0) for name, a in acc.items()}
    NUM["覆盖范围与留出验证"] = {
        "说明": f"真实相机带径向畸变 k1={k1}；标定时建 k1、k2，12 张视图，角点噪声 0.2 px；留出的 40 张视图覆盖全画面，用标定结果求位姿后算重投影误差；各 30 次",
        "结果": {name: {"标定自己报告的 RMS 中位数（px）": round(float(np.median(train[name])), 3),
                       "留出视图的 RMS 中位数（px）": round(float(np.median(held[name])), 3),
                       "留出视图的 RMS 最大值（px）": round(float(np.max(held[name])), 2),
                       "估出的 k1 的 5% 到 95% 分位": [round(float(np.percentile(k1s[name], 5)), 3), round(float(np.percentile(k1s[name], 95)), 3)],
                       "画面中央格子的平均残差（px）": round(float(np.nanmean(maps[name][4:8, 5:11])), 3),
                       "画面四角格子的平均残差（px）": round(float(np.nanmean([maps[name][:2, :2], maps[name][:2, -2:], maps[name][-2:, :2], maps[name][-2:, -2:]])), 3)}
               for name in cases},
    }

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.08, 1.2]}, layout="constrained")
    for name, color in (("覆盖全画面", S.BLUE), ("只在画面中央", S.ORANGE)):
        ax[0].plot(cover[name][:, 0], cover[name][:, 1], ".", color=color, ms=2.5, label=name)
    ax[0].set_xlim(0, W)
    ax[0].set_ylim(H, 0)
    ax[0].set_aspect("equal")
    ax[0].set_title("两种拍法的角点分布")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2, markerscale=4)
    S.image_axes(ax[0])
    maps = {name: np.round(m, 5) for name, m in maps.items()}
    vmax = round(float(np.nanpercentile(maps["只在画面中央"], 98)), 4)
    for a, name in ((ax[1], "只在画面中央"), (ax[2], "覆盖全画面")):
        im = a.imshow(maps[name], cmap="magma", vmin=0, vmax=vmax, extent=(0, W, H, 0))
        a.set_title(f"留出残差（标定时{name}）", fontsize=10.5)
        S.image_axes(a)
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.03, label="px").ax.tick_params(labelsize=8)
    x = np.arange(2)
    # OpenCV 的优化结果在不同次运行之间会差到 1e-9 的量级，画图前取整，图才能逐字节复现
    tr = [round(float(np.median(train[n])), 5) for n in cases]
    he = [round(float(np.median(held[n])), 5) for n in cases]
    b1 = ax[3].bar(x - 0.2, tr, 0.38, color=S.MUTED, alpha=0.8, label="标定自己报告的 RMS")
    b2 = ax[3].bar(x + 0.2, he, 0.38, color=S.RED, alpha=0.85, label="留出视图的 RMS")
    for bars in (b1, b2):
        for b in bars:
            ax[3].text(b.get_x() + b.get_width() / 2, b.get_height() + max(he) * 0.02, f"{b.get_height():.2f}", ha="center", fontsize=9.5)
    ax[3].set_xticks(x, list(cases))
    ax[3].set_ylabel("RMS 中位数（px）")
    ax[3].set_title("只有留出验证看得出差别")
    ax[3].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    S.clean(ax[3])
    ax[3].grid(axis="x", visible=False)
    S.save(fig, CH, "fig05_覆盖范围与留出验证")


def main() -> None:
    S.setup()
    for fn in (fig_views, fig_normalization, fig_tilt, fig_distortion, fig_coverage):
        fn()
        print("完成", fn.__name__, flush=True)
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False))


if __name__ == "__main__":
    main()
