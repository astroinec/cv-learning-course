"""第 7 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch07.py

输出：<主分支>/07_传统CV综合/figures/*.svg 和 实测数字.json。
无畸变的投影全部调用本章的参考实现 project_points；PnP、三角化、本质矩阵用 OpenCV（需要 opencv-python）。
合成相机和第 17 章是同一台：640x480，fx=820，fy=790，主点 (318, 247)，长度单位毫米。
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

CH = "07_传统CV综合"
mod = S.load_solution(CH, "7-1_针孔投影.py")
NUM: dict = {}

W, H = 640, 480
K_TRUE = np.array([[820.0, 0.0, 318.0], [0.0, 790.0, 247.0], [0.0, 0.0, 1.0]])
D0 = np.zeros(5)
FLIP = np.diag([1.0, -1.0, -1.0])
proj = mod.project_points


# ---------------------------------------------------------------- 公共工具
def r(x, n=3):
    return round(float(x), n)


def rot_xyz(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rot_err(Ra, Rb):
    """两个旋转矩阵之间的夹角（度）。"""
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1))))


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def rms(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def rand_pose(rng, tilt=30.0):
    """物体相对相机的随机位姿：绕 x、y 轴倾斜不超过 tilt 度，绕光轴正负 20 度，距离 450 到 550 mm。"""
    R = rot_xyz(*np.deg2rad(rng.uniform(-tilt, tilt, 2)), np.deg2rad(rng.uniform(-20, 20)))
    t = np.array([rng.uniform(-30, 30), rng.uniform(-20, 20), rng.uniform(450, 550)])
    return R, t


def rand_points(rng, n, coplanar, half=60.0):
    obj = rng.uniform(-half, half, (n, 3))
    if coplanar:
        obj[:, 2] = 0.0
    return obj


def project_k1(P, K, R, t, k1):
    """带一个径向畸变系数的投影，公式和第 17 章相同。"""
    pc = P @ R.T + t
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    f = 1 + k1 * (x * x + y * y)
    return np.stack([K[0, 0] * x * f + K[0, 2], K[1, 1] * y * f + K[1, 2]], axis=1)


def pnp(obj, uv, flag=cv2.SOLVEPNP_ITERATIVE, K=K_TRUE, dist=D0):
    try:
        ok, rv, tv = cv2.solvePnP(np.asarray(obj, dtype=np.float64), np.asarray(uv, dtype=np.float64), K, dist, flags=flag)
    except cv2.error:
        return None
    return (cv2.Rodrigues(rv)[0], tv.ravel()) if ok else None


CUBE = np.array([[x, y, z] for z in (-50, 50) for y in (-50, 50) for x in (-50, 50)], dtype=float)
CUBE_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_cube(ax, uv, color, lw=1.8, label=None):
    for k, (a, b) in enumerate(CUBE_EDGES):
        ax.plot(uv[[a, b], 0], uv[[a, b], 1], color=color, lw=lw, label=label if k == 0 else None)


# ---------------------------------------------------------------- 图 1：约定弄反的症状
def fig_conventions():
    R0 = rot_xyz(np.deg2rad(20), np.deg2rad(-25), np.deg2rad(10))
    t0 = np.array([40.0, -25.0, 500.0])

    def variants(R, t):
        good = proj(CUBE, K_TRUE, R, t)
        return good, {
            "R 写成了转置": (proj(CUBE, K_TRUE, R.T, t), good),
            "把 t 当成相机位置": (proj(CUBE, K_TRUE, R, -R @ t), good),
            "先平移后旋转": (proj(CUBE, K_TRUE, R, R @ t), good),
            "OpenGL 相机轴没转换，用参考实现投影": (proj(CUBE, K_TRUE, FLIP @ R, FLIP @ t), good),
            "OpenGL 相机轴没转换，用 cv2.projectPoints 投影": (cv2.projectPoints(CUBE, cv2.Rodrigues(FLIP @ R)[0], FLIP @ t, K_TRUE, D0)[0].reshape(-1, 2), good),
            "把 (u, v) 当成 (行, 列)": (good[:, ::-1], good),
            "图像缩小一半，K 没有改": (good, (good + 0.5) / 2 - 0.5),
        }

    stats: dict = {}
    for i in range(300):
        rng = np.random.default_rng(1000 + i)
        R, t = rand_pose(rng)
        _, var = variants(R, t)
        for name, (bad, ref) in var.items():
            behind = np.isnan(bad).any(axis=1)
            d = np.linalg.norm(bad - ref, axis=1)
            stats.setdefault(name, []).append((behind.all(), np.nan if behind.all() else np.nanmean(d)))
    table = {}
    for name, vals in stats.items():
        vals = np.array(vals, dtype=float)
        allb = float(np.mean(vals[:, 0]))
        table[name] = {"8 个顶点全部落到相机后方的位姿比例": r(allb, 2),
                       "顶点偏差的中位数（px）": None if allb == 1 else r(np.nanmedian(vals[:, 1]), 1)}
    good, var = variants(R0, t0)
    one = {name: (None if np.isnan(bad).any() else r(np.mean(np.linalg.norm(bad - ref, axis=1)), 1)) for name, (bad, ref) in var.items()}
    gl = var["OpenGL 相机轴没转换，用 cv2.projectPoints 投影"][0]
    mirror = np.stack([2 * K_TRUE[0, 2] - good[:, 0], good[:, 1]], axis=1)
    C = -R0.T @ t0
    NUM["相机"] = {"图像尺寸": [W, H], "fx": 820, "fy": 790, "cx": 318, "cy": 247, "说明": "和第 17 章是同一台合成相机；长度单位毫米"}
    NUM["约定弄反"] = {
        "说明": "边长 100 mm 的立方体，中心在世界原点；300 个随机位姿（绕 x、y 轴倾斜不超过 30 度，绕光轴正负 20 度，距离 450 到 550 mm），每种写法和正确投影比较 8 个顶点；"
                "“图像缩小一半”一行的偏差在缩小后的图上量",
        "300 个随机位姿": table,
        "图 1 用的位姿": {"旋转（绕 x、y、z 轴，度）": [20, -25, 10], "t（mm）": [40, -25, 500], "t 的长度（mm）": r(np.linalg.norm(t0), 1),
                      "相机中心 C_w = -R^T t（mm）": [r(v, 1) for v in C], "把 t 当成相机位置时差了多远（mm）": r(np.linalg.norm(C - t0), 0),
                      "各种写法的顶点平均偏差（px）": one,
                      "OpenGL 轴加 cv2.projectPoints 的结果和“绕主点左右镜像”的最大差（px）": float(f"{np.abs(gl - mirror).max():.1e}")},
    }

    # 相机后方的点：参考实现和 cv2.projectPoints 的行为
    Pb = np.array([[30.0, 20.0, 500.0], [30.0, 20.0, -500.0]])
    ref = proj(Pb, K_TRUE, np.eye(3), np.zeros(3))
    cvp = cv2.projectPoints(Pb, np.zeros(3), np.zeros(3), K_TRUE, D0)[0].reshape(-1, 2)
    rng = np.random.default_rng(0)
    P = rng.uniform(-100, 100, (200, 3))
    Rr, tr = rot_xyz(0.3, -0.4, 0.2), np.array([20.0, -10.0, 600.0])
    diff = np.abs(proj(P, K_TRUE, Rr, tr) - cv2.projectPoints(P, cv2.Rodrigues(Rr)[0], tr, K_TRUE, D0)[0].reshape(-1, 2)).max()
    NUM["相机后方的点"] = {
        "说明": "相机在原点、不旋转；同一个点 (30, 20, Z) 分别放在相机前方 500 mm 和后方 500 mm",
        "前方的点": {"参考实现": [r(v, 1) for v in ref[0]], "cv2.projectPoints": [r(v, 1) for v in cvp[0]]},
        "后方的点": {"参考实现": "nan, nan", "cv2.projectPoints": [r(v, 1) for v in cvp[1]]},
        "后方的点相对主点的位置是前方的点取负": bool(np.allclose(cvp[1] - K_TRUE[:2, 2], -(cvp[0] - K_TRUE[:2, 2]))),
        "200 个前方随机点上参考实现和 cv2.projectPoints 的最大差（px）": float(f"{diff:.1e}"),
    }
    seg = np.array([[-50.0, 0, 0], [50.0, 0, 0]])
    NUM["近大远小"] = {"说明": "正对相机的一段 100 mm 线段在不同距离上的投影长度，用参考实现算",
                   "距离（mm）": [250, 500, 1000, 2000],
                   "投影长度（px）": [r(np.ptp(proj(seg, K_TRUE, np.eye(3), np.array([0, 0, float(z)]))[:, 0]), 1) for z in (250, 500, 1000, 2000)]}

    show = ["R 写成了转置", "OpenGL 相机轴没转换，用 cv2.projectPoints 投影", "把 (u, v) 当成 (行, 列)", "图像缩小一半，K 没有改"]
    titles = ["R 写成转置：姿态不对，位置差不多", "OpenGL 轴没转换：左右镜像", "u、v 对调：沿对角线翻转", "缩图不改 K：大一倍，大半在画面外"]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.5), layout="constrained")
    for a, name, title in zip(ax, show, titles):
        bad, refp = var[name]
        draw_cube(a, refp, "#9aa0a8", 2.2, "正确")
        draw_cube(a, bad, S.RED, 1.6, "写错")
        a.plot(*refp[0], "o", color="#9aa0a8", ms=5)
        a.plot(*bad[0], "o", color=S.RED, ms=5)
        a.set_xlim(0, W)
        a.set_ylim(H, 0)
        a.set_aspect("equal")
        a.set_title(f"{title}（平均偏 {one[name]:.0f} px）", fontsize=10.5)
        S.image_axes(a)
        a.set_xlabel("画面 640x480，圆点是同一个顶点", fontsize=9, color=S.MUTED)
    ax[3].plot([0, W / 2, W / 2, 0, 0], [0, 0, H / 2, H / 2, 0], "-", color=S.INK, lw=1)
    ax[3].text(6, H / 2 + 22, "黑框：320x240 的画面", fontsize=8.5, color=S.INK)
    ax[3].set_xlabel("黑框是实际画面，框外的部分看不到", fontsize=9, color=S.MUTED)
    ax[0].legend(fontsize=9, loc="lower left")
    S.save(fig, CH, "fig01_约定弄反的症状")


# ---------------------------------------------------------------- 图 2：PnP 的误差由什么决定
def pnp_stats(n, coplanar, noise, half=60.0, flag=cv2.SOLVEPNP_ITERATIVE, trials=300, seed=2000):
    out, fail, width = [], 0, []
    for i in range(trials):
        rng = np.random.default_rng(seed + i)
        obj = rand_points(rng, n, coplanar, half)
        R, t = rand_pose(rng)
        clean = proj(obj, K_TRUE, R, t)
        res = pnp(obj, clean + rng.normal(scale=noise, size=clean.shape), flag)
        width.append(np.ptp(clean[:, 0]))
        if res is None:
            fail += 1
            continue
        out.append((rot_err(res[0], R), abs(res[1][2] - t[2]), np.hypot(res[1][0] - t[0], res[1][1] - t[1])))
    if not out:
        return {"失败次数": fail}
    v = np.array(out)
    return {"旋转误差中位数（度）": r(np.median(v[:, 0]), 3), "旋转误差 90% 分位（度）": r(np.percentile(v[:, 0], 90), 2),
            "深度方向平移误差中位数（mm）": r(np.median(v[:, 1]), 3), "横向平移误差中位数（mm）": r(np.median(v[:, 2]), 3),
            "点在图上占的宽度中位数（px）": int(round(float(np.median(width)))), "失败次数": fail}


def fig_pnp():
    ns = [6, 8, 12, 20, 50, 100]
    noises = [0.1, 0.3, 0.5, 1.0, 2.0]
    halves = [7.5, 15, 30, 60, 100]
    shapes = {"共面": True, "不共面": False}
    by_n = {s: {str(n): pnp_stats(n, c, 0.5) for n in ns} for s, c in shapes.items()}
    by_noise = {s: {str(v): pnp_stats(20, c, v) for v in noises} for s, c in shapes.items()}
    by_half = {s: {str(h): pnp_stats(20, c, 0.5, half=h) for h in halves} for s, c in shapes.items()}
    methods = {"ITERATIVE": cv2.SOLVEPNP_ITERATIVE, "EPNP": cv2.SOLVEPNP_EPNP, "SQPNP": cv2.SOLVEPNP_SQPNP, "IPPE": cv2.SOLVEPNP_IPPE}
    by_method = {}
    for s, c in shapes.items():
        by_method[s] = {}
        for n in (4, 6, 20):
            by_method[s][str(n)] = {}
            for name, flag in methods.items():
                if name == "IPPE" and not c:
                    continue
                st = pnp_stats(n, c, 0.5, flag=flag)
                by_method[s][str(n)][name] = {k: st[k] for k in ("旋转误差中位数（度）", "旋转误差 90% 分位（度）", "失败次数") if k in st}
    NUM["PnP 误差"] = {
        "说明": "物体点在边长 120 mm 的正方形（共面，Z=0）或立方体（不共面）里均匀取；位姿随机（倾斜不超过 30 度，距离 450 到 550 mm）；"
                "像素噪声是每个坐标独立的高斯噪声；cv2.solvePnP 默认的 ITERATIVE 方法；各 300 次；失败指函数报错或返回失败",
        "按点数（噪声 0.5 px）": by_n, "按噪声（20 个点）": by_noise,
        "按点的范围（20 个点，噪声 0.5 px；键是范围的半宽 mm）": by_half,
        "按求解方法（噪声 0.5 px；键是点数）": by_method,
        "20 个点、噪声 0.5 px 时深度方向误差是横向误差的倍数": {s: r(by_noise[s]["0.5"]["深度方向平移误差中位数（mm）"] / by_noise[s]["0.5"]["横向平移误差中位数（mm）"], 1) for s in shapes},
        "20 个点、噪声 0.5 px 时共面的旋转误差是不共面的倍数": r(by_noise["共面"]["0.5"]["旋转误差中位数（度）"] / by_noise["不共面"]["0.5"]["旋转误差中位数（度）"], 1),
        "点数从 6 到 100 旋转误差缩小的倍数": {s: r(by_n[s]["6"]["旋转误差中位数（度）"] / by_n[s]["100"]["旋转误差中位数（度）"], 1) for s in shapes},
        "按 1/sqrt(N) 应缩小的倍数": r(np.sqrt(100 / 6), 1),
        "点数从 20 到 100 旋转误差缩小的倍数": {s: r(by_n[s]["20"]["旋转误差中位数（度）"] / by_n[s]["100"]["旋转误差中位数（度）"], 1) for s in shapes},
        "从 20 到 100 按 1/sqrt(N) 应缩小的倍数": r(np.sqrt(100 / 20), 1),
    }

    style = {"共面": (S.ORANGE, "s"), "不共面": (S.BLUE, "o")}
    key = "旋转误差中位数（度）"
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9), layout="constrained")
    for s, (color, marker) in style.items():
        ax[0].plot(ns, [by_n[s][str(n)][key] for n in ns], marker + "-", color=color, lw=1.8, ms=5, label=s)
        ax[1].plot(noises, [by_noise[s][str(v)][key] for v in noises], marker + "-", color=color, lw=1.8, ms=5, label=s)
        ax[2].plot([by_half[s][str(h)]["点在图上占的宽度中位数（px）"] for h in halves], [by_half[s][str(h)][key] for h in halves], marker + "-", color=color, lw=1.8, ms=5, label=s)
    for k, s in enumerate(style):
        guide = by_n[s]["6"][key] * np.sqrt(6 / np.array(ns, dtype=float))
        ax[0].plot(ns, guide, "--", color=S.MUTED, lw=1, label="从 6 个点起按 1/sqrt(N) 下降的参考线" if k == 0 else None)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xticks(ns, [str(n) for n in ns])
    ax[0].set_yticks([0.1, 0.2, 0.5, 1.0], ["0.1", "0.2", "0.5", "1"])
    ax[0].minorticks_off()
    ax[0].set_xlabel("点数（噪声 0.5 px）")
    ax[0].set_ylabel("旋转误差中位数（度）")
    ax[0].set_title("点越多越准，降得比 1/sqrt(N) 略快")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    S.clean(ax[0])
    ax[1].set_xlim(left=0)
    ax[1].set_ylim(bottom=0)
    ax[1].set_xlabel("像素噪声（px，20 个点）")
    ax[1].set_ylabel("旋转误差中位数（度）")
    ax[1].set_title("误差和像素噪声成正比")
    S.clean(ax[1])
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xticks([25, 50, 100, 200, 300], ["25", "50", "100", "200", "300"])
    ax[2].set_yticks([0.1, 0.3, 1, 3, 10], ["0.1", "0.3", "1", "3", "10"])
    ax[2].minorticks_off()
    ax[2].set_xlabel("点在图上占的宽度（px，20 个点，噪声 0.5 px）")
    ax[2].set_ylabel("旋转误差中位数（度）")
    ax[2].set_title("点占的画面越小，误差越大")
    S.clean(ax[2])
    S.save(fig, CH, "fig02_PnP误差的来源")


# ---------------------------------------------------------------- 图 3：方形标记的两个解
L_MARK = 50.0
SQUARE = np.array([[-L_MARK / 2, L_MARK / 2, 0], [L_MARK / 2, L_MARK / 2, 0], [L_MARK / 2, -L_MARK / 2, 0], [-L_MARK / 2, -L_MARK / 2, 0]])


def marker_pose(rng, tilt, z):
    """方形标记正面朝向相机，法线和视线夹角约为 tilt 度，倾斜方向随机。"""
    ax_ = rng.uniform(0, 2 * np.pi)
    rv = np.deg2rad(tilt) * np.array([np.cos(ax_), np.sin(ax_), 0.0])
    R = cv2.Rodrigues(rv)[0] @ rot_xyz(0, 0, rng.uniform(-np.pi, np.pi)) @ FLIP
    t = np.array([rng.uniform(-0.05, 0.05) * z, rng.uniform(-0.05, 0.05) * z, z])
    return R, t


def two_solutions(uv):
    """IPPE_SQUARE 的两个候选解，按重投影误差从小到大。"""
    _, rvs, tvs, errs = cv2.solvePnPGeneric(SQUARE, uv, K_TRUE, D0, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return [cv2.Rodrigues(v)[0] for v in rvs], [v.ravel() for v in tvs], np.ravel(errs)


def fig_ambiguity():
    sides = [200, 140, 100, 70, 50, 35, 25]
    tilts = [10, 30, 50]
    trials = 400
    table = {}
    for noise in (0.3, 1.0):
        table[str(noise)] = {}
        for tilt in tilts:
            table[str(noise)][str(tilt)] = {}
            for side in sides:
                z = K_TRUE[0, 0] * L_MARK / side
                flip_ippe = flip_iter = 0
                ratio, err1 = [], []
                for i in range(trials):
                    rng = np.random.default_rng(3000 + i)
                    R, t = marker_pose(rng, tilt, z)
                    uv = proj(SQUARE, K_TRUE, R, t) + rng.normal(scale=noise, size=(4, 2))
                    Rs, _, errs = two_solutions(uv)
                    if len(Rs) < 2:
                        continue
                    e = [rot_err(Rc, R) for Rc in Rs]
                    flip_ippe += e[0] > e[1]
                    ratio.append(errs[1] / max(errs[0], 1e-12))
                    err1.append(e[0])
                    it = pnp(SQUARE, uv)
                    good_, bad_ = (Rs[0], Rs[1]) if e[0] <= e[1] else (Rs[1], Rs[0])
                    flip_iter += rot_err(it[0], bad_) < rot_err(it[0], good_)
                table[str(noise)][str(tilt)][str(side)] = {
                    "IPPE_SQUARE 选错的比例": r(flip_ippe / trials, 3), "ITERATIVE 选错的比例": r(flip_iter / trials, 3),
                    "第二个解与第一个解的重投影误差之比（中位数）": r(np.median(ratio), 2),
                    "IPPE_SQUARE 旋转误差中位数（度）": r(np.median(err1), 2), "IPPE_SQUARE 旋转误差 90% 分位（度）": r(np.percentile(err1, 90), 1)}
    # 画图用的一个例子：边长约 35 px，倾斜 30 度
    rng = np.random.default_rng(12)
    z = K_TRUE[0, 0] * L_MARK / 35
    R, t = marker_pose(rng, 30, z)
    uv = proj(SQUARE, K_TRUE, R, t) + rng.normal(scale=0.3, size=(4, 2))
    Rs, ts, errs = two_solutions(uv)
    normal = np.array([[0, 0, 0], [0, 0, L_MARK]], dtype=float)
    NUM["方形标记的两个解"] = {
        "说明": "边长 50 mm 的方形标记，4 个角点，位置在画面中心附近，倾斜方向随机；图上的边长按 fx 折算；各 400 次；"
                "“选错”指函数返回的解离真值比另一个候选解更远。候选解由 cv2.solvePnPGeneric 加 SOLVEPNP_IPPE_SQUARE 给出",
        "图上的边长（px）": sides, "标记法线和视线的夹角（度）": tilts,
        "结果（第一层是像素噪声 px，第二层是夹角，第三层是边长）": table,
        "图 3 左的例子": {"图上的边长（px）": 35, "夹角（度）": 30, "噪声（px）": 0.3,
                      "两个候选解的重投影误差（px）": [r(v, 2) for v in errs],
                      "两个候选解各自的旋转误差（度）": [r(rot_err(Rc, R), 1) for Rc in Rs],
                      "两个候选解之间的夹角（度）": r(rot_err(Rs[0], Rs[1]), 1)},
    }

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0), gridspec_kw={"width_ratios": [1, 1.2, 1.2]}, layout="constrained")
    a = ax[0]
    quad = np.vstack([uv, uv[:1]])
    a.plot(quad[:, 0], quad[:, 1], "-", color=S.INK, lw=1.4)
    a.plot(uv[:, 0], uv[:, 1], "o", color=S.INK, ms=4, label="检测到的 4 个角点")
    for Rc, tc, color, name in zip(Rs, ts, (S.BLUE, S.RED), ("候选解 1 的法线", "候选解 2 的法线")):
        q = proj(SQUARE, K_TRUE, Rc, tc)
        a.plot(np.r_[q[:, 0], q[0, 0]], np.r_[q[:, 1], q[0, 1]], "--", color=color, lw=1)
        n2 = proj(normal, K_TRUE, Rc, tc)
        a.annotate("", xy=n2[1], xytext=n2[0], arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2})
        a.plot([], [], "-", color=color, lw=2, label=name)
    c = uv.mean(axis=0)
    a.set_xlim(c[0] - 55, c[0] + 55)
    a.set_ylim(c[1] + 55, c[1] - 55)
    a.set_aspect("equal")
    a.set_title("两个姿态，角点几乎重合", fontsize=10.5)
    a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=1)
    S.image_axes(a)
    colors = {10: S.RED, 30: S.ORANGE, 50: S.BLUE}
    xs = np.arange(len(sides))  # 边长从大到小等距排开（不用反向坐标轴，SVG 才能逐字节复现）
    for tilt in tilts:
        row = table["0.3"][str(tilt)]
        ax[1].plot(xs, [row[str(s)]["IPPE_SQUARE 选错的比例"] * 100 for s in sides], "o-", color=colors[tilt], lw=1.8, ms=5, label=f"夹角 {tilt}°，IPPE_SQUARE")
        ax[1].plot(xs, [row[str(s)]["ITERATIVE 选错的比例"] * 100 for s in sides], "x--", color=colors[tilt], lw=1.1, ms=6, label=f"夹角 {tilt}°，ITERATIVE")
        ax[2].plot(xs, [row[str(s)]["第二个解与第一个解的重投影误差之比（中位数）"] for s in sides], "o-", color=colors[tilt], lw=1.8, ms=5, label=f"夹角 {tilt}°")
    for a in ax[1:]:
        a.set_xticks(xs, [str(s) for s in sides])
        a.set_xlabel("标记在图上的边长（px，越往右越小；噪声 0.3 px）")
        S.clean(a)
    ax[1].axhline(50, color=S.MUTED, lw=1, ls=":")
    ax[1].text(0, 51.5, "50%：和抛硬币一样", fontsize=8.5, color=S.MUTED)
    ax[1].set_ylim(-2, 60)
    ax[1].set_ylabel("选错解的比例（%）")
    ax[1].set_title("标记越小、越正对，越容易选错；两种方法一样")
    ax[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax[2].set_yscale("log")
    ax[2].set_yticks([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"])
    ax[2].minorticks_off()
    ax[2].axhline(1, color=S.MUTED, lw=1, ls=":")
    ax[2].set_ylabel("第二个解与第一个解的重投影误差之比")
    ax[2].set_title("标记变小后，两个解的重投影误差分不开")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    S.save(fig, CH, "fig03_方形标记的两个解")


# ---------------------------------------------------------------- 图 4：内参不准时位姿被带偏多少
def intrinsic_stats(Kx, coplanar, trials=300, seed=5000):
    rows = []
    for i in range(trials):
        rng = np.random.default_rng(seed + i)
        obj = rand_points(rng, 20, coplanar)
        R, t = rand_pose(rng)
        uv = proj(obj, K_TRUE, R, t) + rng.normal(scale=0.3, size=(20, 2))
        Re, te = pnp(obj, uv, K=Kx)
        rows.append((rot_err(Re, R), (te[2] - t[2]) / t[2] * 100, te[0] - t[0], rms(proj(obj, Kx, Re, te), uv)))
    v = np.median(np.array(rows), axis=0)
    return {"旋转误差中位数（度）": r(v[0], 3), "距离 tz 的偏差中位数（%）": r(v[1], 2), "横向 tx 的偏差中位数（mm）": r(v[2], 2), "重投影 RMS 中位数（px）": r(v[3], 3)}


def fig_intrinsics():
    f_errs = [-10, -5, -2, 0, 2, 5, 10]
    c_errs = [0, 5, 10, 20, 40]
    shapes = {"共面": True, "不共面": False}
    by_f, by_c = {s: {} for s in shapes}, {s: {} for s in shapes}
    for s, c in shapes.items():
        for e in f_errs:
            Kx = K_TRUE.copy()
            Kx[0, 0] *= 1 + e / 100
            Kx[1, 1] *= 1 + e / 100
            by_f[s][str(e)] = intrinsic_stats(Kx, c)
        for e in c_errs:
            Kx = K_TRUE.copy()
            Kx[0, 2] += e
            by_c[s][str(e)] = intrinsic_stats(Kx, c)

    # 图像缩小一半之后内参怎么改
    scale_rows = {s: [] for s in shapes}
    for s, c in shapes.items():
        for i in range(300):
            rng = np.random.default_rng(5500 + i)
            obj = rand_points(rng, 20, c)
            R, t = rand_pose(rng)
            uvh = (proj(obj, K_TRUE, R, t) + rng.normal(scale=0.3, size=(20, 2)) + 0.5) / 2 - 0.5
            Kh = K_TRUE.copy()
            Kh[:2] *= 0.5
            Kc = Kh.copy()
            Kc[0, 2], Kc[1, 2] = (K_TRUE[0, 2] + 0.5) / 2 - 0.5, (K_TRUE[1, 2] + 0.5) / 2 - 0.5
            row = []
            for Kx in (K_TRUE, Kh, Kc):
                Re, te = pnp(obj, uvh, K=Kx)
                row += [rot_err(Re, R), te[2] / t[2], rms(proj(obj, Kx, Re, te), uvh), abs(te[0] - t[0])]
            scale_rows[s].append(row)
    scale = {}
    for s in shapes:
        v = np.median(np.array(scale_rows[s]), axis=0)
        scale[s] = {name: {"旋转误差中位数（度）": r(v[k], 2), "估出的距离是真值的倍数": r(v[k + 1], 3), "重投影 RMS 中位数（px）": r(v[k + 2], 2), "横向 tx 误差中位数（mm）": r(v[k + 3], 2)}
                    for name, k in (("K 不改", 0), ("fx、fy、cx、cy 都乘 0.5", 4), ("焦距乘 0.5，主点按 (c+0.5)/2-0.5 换算", 8))}

    # cv2.resize 缩小一半后，图上的点到底落在哪
    gx, gy = np.meshgrid(np.arange(40, W - 30, 40.0), np.arange(40, H - 30, 40.0))
    rng = np.random.default_rng(5)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1) + rng.uniform(-4, 4, (gx.size, 2))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    img = np.zeros((H, W))
    for u, v_ in pts:
        img += np.exp(-((xx - u) ** 2 + (yy - v_) ** 2) / (2 * 3.0 ** 2))
    ys, xs = np.mgrid[0:H // 2, 0:W // 2].astype(np.float64)
    resize = {}
    for name, interp in (("INTER_AREA", cv2.INTER_AREA), ("INTER_LINEAR", cv2.INTER_LINEAR), ("INTER_CUBIC", cv2.INTER_CUBIC), ("INTER_NEAREST", cv2.INTER_NEAREST)):
        small = cv2.resize(img, (W // 2, H // 2), interpolation=interp)
        meas = []
        for u, v_ in pts:
            w = small * ((np.abs(xs - u / 2) < 7) & (np.abs(ys - v_ / 2) < 7))
            meas.append((np.sum(w * xs) / w.sum(), np.sum(w * ys) / w.sum()))
        meas = np.array(meas)
        resize[name] = {"相对 u/2 的平均偏移（px）": r(np.mean(meas - pts / 2), 3), "相对 (u+0.5)/2-0.5 的平均偏移（px）": r(np.mean(meas - ((pts + 0.5) / 2 - 0.5)), 3)}

    # 忽略畸变
    dist_rows = {s: [] for s in shapes}
    for s, c in shapes.items():
        for i in range(300):
            rng = np.random.default_rng(5800 + i)
            obj = rand_points(rng, 20, c)
            R, t = rand_pose(rng)
            t[0], t[1] = rng.uniform(-120, 120), rng.uniform(-80, 80)
            uv = project_k1(obj, K_TRUE, R, t, -0.1) + rng.normal(scale=0.3, size=(20, 2))
            Ra, ta = pnp(obj, uv)
            Rb, tb = pnp(obj, uv, dist=np.array([-0.1, 0, 0, 0, 0.0]))
            dist_rows[s].append((rot_err(Ra, R), abs(ta[2] - t[2]) / t[2] * 100, rms(proj(obj, K_TRUE, Ra, ta), uv), rot_err(Rb, R), abs(tb[2] - t[2]) / t[2] * 100))
    ignore = {}
    for s in shapes:
        v = np.median(np.array(dist_rows[s]), axis=0)
        ignore[s] = {"畸变系数传零": {"旋转误差中位数（度）": r(v[0], 2), "距离误差中位数（%）": r(v[1], 2), "重投影 RMS 中位数（px）": r(v[2], 2)},
                     "畸变系数传真值": {"旋转误差中位数（度）": r(v[3], 2), "距离误差中位数（%）": r(v[4], 2)}}

    zbar = 500.0
    NUM["内参不准"] = {
        "说明": "20 个物体点（范围同“PnP 误差”），像素噪声 0.3 px，位姿随机，各 300 次；图像点用真内参生成，求位姿时故意传入有误差的内参；"
                "重投影 RMS 用传入的那组内参和估出的位姿算，也就是程序自己能看到的那个数",
        "焦距偏差（%，fx、fy 同时乘）": f_errs, "按焦距偏差": by_f,
        "主点 cx 的偏差（px）": c_errs, "按主点偏差": by_c,
        "按小角度估算，主点偏 20 px 对应的旋转（度）": r(np.degrees(np.arctan(20 / K_TRUE[0, 0])), 2),
        "按平均距离 500 mm 估算，主点偏 20 px 对应的横向位移（mm）": r(zbar * 20 / K_TRUE[0, 0], 1),
        "图像缩小一半": {"说明": "把像素坐标按像素中心对齐的方式缩小一半（u' = (u+0.5)/2-0.5），再用三种内参求位姿；条件同上", "结果": scale},
        "cv2.resize 缩小一半后点的位置": {"说明": "640x480 的图上放 165 个高斯圆斑（标准差 3 px），用 cv2.resize 缩到 320x240，在缩小后的图上取质心，和两种换算公式比较", "结果": resize},
        "忽略畸变": {"说明": "相机带 k1=-0.1 的径向畸变，物体中心在画面里横向正负 120 mm、纵向正负 80 mm 范围内随机；其余条件同上", "结果": ignore},
    }

    style = {"共面": (S.ORANGE, "s"), "不共面": (S.BLUE, "o")}
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9), layout="constrained")
    for s, (color, marker) in style.items():
        ax[0].plot(f_errs, [by_f[s][str(e)]["距离 tz 的偏差中位数（%）"] for e in f_errs], marker + "-", color=color, lw=1.8, ms=5, label=s)
        ax[1].plot(f_errs, [by_f[s][str(e)]["重投影 RMS 中位数（px）"] for e in f_errs], marker + "-", color=color, lw=1.8, ms=5, label=s)
        ax[2].plot(c_errs, [by_c[s][str(e)]["旋转误差中位数（度）"] for e in c_errs], marker + "-", color=color, lw=1.8, ms=5, label=s)
    ax[0].plot(f_errs, f_errs, ":", color=S.MUTED, lw=1, label="偏差相等的参考线")
    ax[0].set_xlabel("传入的焦距偏差（%）")
    ax[0].set_ylabel("估出的距离 tz 的偏差（%）")
    ax[0].set_title("焦距错百分之几，距离就错百分之几")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    S.clean(ax[0])
    ax[1].axhline(0.3 * np.sqrt(2), color=S.MUTED, lw=1, ls="--", label="0.3 px 噪声对应的 0.42 px")
    ax[1].set_ylim(0, 0.75)
    ax[1].set_xlabel("传入的焦距偏差（%）")
    ax[1].set_ylabel("重投影 RMS 中位数（px）")
    ax[1].set_title("共面点的重投影误差几乎不动")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    S.clean(ax[1])
    ax[2].plot(c_errs, np.degrees(np.arctan(np.array(c_errs) / K_TRUE[0, 0])), ":", color=S.MUTED, lw=1, label="arctan(偏差 / fx)")
    ax[2].set_xlabel("传入的主点 cx 的偏差（px）")
    ax[2].set_ylabel("旋转误差中位数（度）")
    ax[2].set_ylim(bottom=0)
    ax[2].set_title("主点偏了，姿态跟着转")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    S.clean(ax[2])
    S.save(fig, CH, "fig04_内参误差带偏位姿")


# ---------------------------------------------------------------- 图 5：三角化的深度误差
def spread(x):
    """16% 到 84% 分位之间宽度的一半；对高斯分布等于标准差，但不受少数极端值影响。"""
    return float(np.percentile(x, 84) - np.percentile(x, 16)) / 2


def triangulate(B, Z, sigma=0.5, n=4000, seed=6000):
    """两台平行放置、内参相同的相机，基线 B；在深度 Z 的平面上取 n 个点，加噪声后用 cv2.triangulatePoints 恢复。"""
    rng = np.random.default_rng(seed + int(B) * 100000 + int(Z))
    X = np.stack([rng.uniform(-0.25, 0.25, n) * Z + B / 2, rng.uniform(-0.2, 0.2, n) * Z, np.full(n, float(Z))], axis=1)
    t2 = np.array([-float(B), 0.0, 0.0])
    u1 = proj(X, K_TRUE, np.eye(3), np.zeros(3)) + rng.normal(scale=sigma, size=(n, 2))
    u2 = proj(X, K_TRUE, np.eye(3), t2) + rng.normal(scale=sigma, size=(n, 2))
    P1 = K_TRUE @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K_TRUE @ np.hstack([np.eye(3), t2.reshape(3, 1)])
    Xh = cv2.triangulatePoints(P1, P2, u1.T.copy(), u2.T.copy())
    Xe = (Xh[:3] / Xh[3]).T
    err = Xe - X
    ray = X - np.array([B / 2, 0.0, 0.0])
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    along = (err * ray).sum(axis=1)
    perp = np.linalg.norm(err - along[:, None] * ray, axis=1)
    return Xe[:, 2], perp


def fig_triangulation():
    sigma = 0.5
    Bs = [30, 60, 120, 240]
    Zs = [250, 500, 1000, 2000, 4000, 8000]
    fx, fy = K_TRUE[0, 0], K_TRUE[1, 1]
    table, zest = {}, {}
    for B in Bs:
        table[str(B)] = {}
        for Z in Zs:
            ze, perp = triangulate(B, Z, sigma)
            dz = ze - Z
            zest[(B, Z)] = ze
            table[str(B)][str(Z)] = {
                "视差（px）": r(fx * B / Z, 1),
                "深度误差的离散程度（mm）": float(f"{spread(dz):.3g}"),
                "公式 Z^2/(f B) 乘视差噪声（mm）": float(f"{Z * Z / (fx * B) * np.sqrt(2) * sigma:.3g}"),
                "深度误差的 RMS（mm）": float(f"{np.sqrt(np.mean(dz ** 2)):.3g}"),
                "深度的平均偏差（mm）": float(f"{np.mean(dz):.2g}"), "深度的中位数偏差（mm）": float(f"{np.median(dz):.2g}"),
                "垂直于视线方向的误差 RMS（mm）": float(f"{np.sqrt(np.mean(perp ** 2)):.3g}"),
                "深度小于等于 0 的点数": int(np.sum(ze <= 0)),
            }
    g = lambda B, Z, k: table[str(B)][str(Z)][k]  # noqa: E731
    key = "深度误差的离散程度（mm）"
    NUM["三角化"] = {
        "说明": "两台内参相同、平行放置的相机（基线沿 x 轴），点在深度 Z 的平面上取 4000 个，两幅图的像素坐标各加 0.5 px 高斯噪声，用真实的投影矩阵和 cv2.triangulatePoints 恢复；"
                "“离散程度”是误差的 16% 到 84% 分位之间宽度的一半，对高斯分布等于标准差；视差噪声是 sqrt(2) 乘 0.5 px",
        "基线（mm）": Bs, "距离（mm）": Zs, "结果（第一层是基线，第二层是距离）": table,
        "视差噪声（px）": r(np.sqrt(2) * sigma, 3),
        "实测离散程度与公式之比的范围": [r(min(g(B, Z, key) / g(B, Z, "公式 Z^2/(f B) 乘视差噪声（mm）") for B in Bs for Z in Zs), 2),
                              r(max(g(B, Z, key) / g(B, Z, "公式 Z^2/(f B) 乘视差噪声（mm）") for B in Bs for Z in Zs), 2)],
        "基线 60 mm：距离从 500 到 1000 mm，深度误差变成几倍": r(g(60, 1000, key) / g(60, 500, key), 1),
        "基线 60 mm：距离从 500 到 4000 mm，深度误差变成几倍": r(g(60, 4000, key) / g(60, 500, key), 0),
        "距离 1000 mm：基线从 60 到 240 mm，深度误差变成几分之一": r(g(60, 1000, key) / g(240, 1000, key), 1),
        "基线 60 mm、距离 1000 mm：深度误差是垂直视线方向误差的倍数": r(g(60, 1000, key) / g(60, 1000, "垂直于视线方向的误差 RMS（mm）"), 0),
        "垂直视线方向误差的估算 Z*sigma/sqrt(2)*sqrt(1/fx^2+1/fy^2)，距离 1000 mm（mm）": r(1000 * sigma / np.sqrt(2) * np.sqrt(1 / fx ** 2 + 1 / fy ** 2), 3),
        "基线 30 mm：估出的深度与真值之比的分位数": {
            "说明": "键是距离（mm）；视差越小，84% 和 95% 分位离 1 越远，16% 和 5% 分位离 1 的距离跟不上",
            **{str(Z): {"视差（px）": r(fx * 30 / Z, 1), **{f"{q}%": r(np.percentile(zest[(30, Z)] / Z, q), 3) for q in (5, 16, 50, 84, 95)}} for Z in (1000, 2000, 4000, 8000)}},
        "基线 30 mm、距离 8000 mm：估出的深度超过真值 1.5 倍的点的比例（%）": r(np.mean(zest[(30, 8000)] > 12000) * 100, 1),
        "基线 30 mm、距离 8000 mm：估出的深度不到真值 0.5 倍的点的比例（%）": r(np.mean(zest[(30, 8000)] < 4000) * 100, 1),
        "基线 30 mm、距离 8000 mm：去掉误差绝对值最大的 1% 的点之后的 RMS（mm）": float(f"{np.sqrt(np.mean(np.sort(np.abs(zest[(30, 8000)] - 8000))[:3960] ** 2)):.3g}"),
    }

    colors = {30: S.RED, 60: S.ORANGE, 120: S.GREEN, 240: S.BLUE}
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0), layout="constrained")
    zz = np.array(Zs, dtype=float)
    for B in Bs:
        ax[0].plot(Zs, [g(B, Z, key) for Z in Zs], "o", color=colors[B], ms=5.5, label=f"基线 {B} mm 实测")
        ax[0].plot(zz, zz ** 2 / (fx * B) * np.sqrt(2) * sigma, "-", color=colors[B], lw=1.1)
    ax[0].plot([], [], "-", color=S.MUTED, lw=1.1, label="实线：公式")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xticks(Zs, [str(z) for z in Zs])
    ax[0].set_yticks([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1000"])
    ax[0].minorticks_off()
    ax[0].set_xlabel("距离 Z（mm）")
    ax[0].set_ylabel("深度误差的离散程度（mm）")
    ax[0].set_title("距离翻倍误差变 4 倍，基线翻倍误差减半")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    S.clean(ax[0])
    for B, ls in ((60, "-"), (240, "--")):
        ax[1].plot(Zs, [g(B, Z, key) for Z in Zs], "o" + ls, color=S.RED, lw=1.6, ms=5, label=f"沿深度方向，基线 {B} mm")
        ax[1].plot(Zs, [g(B, Z, "垂直于视线方向的误差 RMS（mm）") for Z in Zs], "s" + ls, color=S.BLUE, lw=1.6, ms=5, label=f"垂直视线方向，基线 {B} mm")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xticks(Zs, [str(z) for z in Zs])
    ax[1].set_yticks([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1000"])
    ax[1].minorticks_off()
    ax[1].set_xlabel("距离 Z（mm）")
    ax[1].set_ylabel("误差（mm）")
    ax[1].set_title("横向误差小得多，而且和基线无关")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[1])
    zq = [1000, 2000, 4000, 8000]
    xq = np.arange(len(zq))
    q = {p_: np.array([np.percentile(zest[(30, Z)] / Z, p_) for Z in zq]) for p_ in (5, 16, 50, 84, 95)}
    ax[2].vlines(xq, q[5], q[95], color=S.RED, lw=1.4, label="5% 到 95% 分位")
    ax[2].vlines(xq, q[16], q[84], color=S.RED, lw=6, alpha=0.55, label="16% 到 84% 分位")
    ax[2].plot(xq, q[50], "o", color=S.INK, ms=5, label="中位数")
    ax[2].axhline(1, color=S.INK, lw=0.7)
    for x_, lo, hi in zip(xq, q[5], q[95]):
        ax[2].text(x_ + 0.12, hi, f"{hi:.2f}", fontsize=8.5, va="center", color=S.MUTED)
        ax[2].text(x_ + 0.12, lo, f"{lo:.2f}", fontsize=8.5, va="center", color=S.MUTED)
    ax[2].set_xticks(xq, [f"{fx * 30 / Z:.1f} px\n({Z} mm)" for Z in zq])
    ax[2].set_xlim(-0.5, len(zq) - 0.3)
    ax[2].set_xlabel("视差（括号里是距离；基线 30 mm）")
    ax[2].set_ylabel("估出的深度 / 真实深度")
    ax[2].set_title("视差越小，误差越往远处拖尾")
    ax[2].legend(fontsize=8.5, loc="upper left")
    S.clean(ax[2])
    ax[2].grid(axis="x", visible=False)
    S.save(fig, CH, "fig05_三角化的深度误差")


# ---------------------------------------------------------------- 图 6：两视图相对位姿什么时候不可信
DEPTH = 600.0


def two_views(i, baseline, relief, noise=0.5, outlier=0.0, n=100):
    """第一台相机在原点，第二台相机满足 x2 = R x1 + t。场景是深度 600 mm 附近一块倾斜的平面，再加正负 relief 的深度起伏。"""
    rng = np.random.default_rng(7000 + i)
    X = np.stack([rng.uniform(-150, 150, n), rng.uniform(-110, 110, n), np.zeros(n)], axis=1)
    slope = np.tan(np.deg2rad(rng.uniform(-20, 20, 2)))
    X[:, 2] = DEPTH + X[:, :2] @ slope + rng.uniform(-1, 1, n) * relief
    R = rot_xyz(*np.deg2rad(rng.uniform(-4, 4, 3)))
    d = rng.normal(size=3)
    d[2] *= 0.3
    d /= np.linalg.norm(d)
    p1 = proj(X, K_TRUE, np.eye(3), np.zeros(3)) + rng.normal(scale=noise, size=(n, 2))
    p2 = proj(X, K_TRUE, R, baseline * d) + rng.normal(scale=noise, size=(n, 2))
    nb = int(round(outlier * n))
    if nb:
        idx = rng.choice(n, nb, replace=False)
        p2[idx] = np.stack([rng.uniform(0, W, nb), rng.uniform(0, H, nb)], axis=1)
    return p1, p2, R, d


def essential(p1, p2, method, seed):
    """返回 3x3 的本质矩阵和内点掩码。"8 点" 是不做 RANSAC 的线性解；其余是 cv2.findEssentialMat 的三种鲁棒方法。"""
    if method == "8 点":
        F, _ = cv2.findFundamentalMat(p1, p2, cv2.FM_8POINT)
        return K_TRUE.T @ F @ K_TRUE, np.ones(len(p1), dtype=bool)
    cv2.setRNGSeed(seed)
    E, mask = cv2.findEssentialMat(p1, p2, K_TRUE, method={"RANSAC": cv2.RANSAC, "MAGSAC": cv2.USAC_MAGSAC}[method], prob=0.999, threshold=1.0)
    return E[:3], mask.ravel().astype(bool)


def pose_err(E, p1, p2, R, d, far=False):
    out = cv2.recoverPose(E, p1, p2, K_TRUE, distanceThresh=1e6) if far else cv2.recoverPose(E, p1, p2, K_TRUE)
    return rot_err(out[1], R), angle(out[2].ravel(), d), int(out[0])


def fig_relative():
    trials = 200
    baselines = [0, 1, 3, 10, 30, 100, 200]
    by_b = {}
    for b in baselines:
        rows = []
        for i in range(trials):
            p1, p2, R, d = two_views(i, b, 150.0)
            E, _ = essential(p1, p2, "RANSAC", i)
            a = pose_err(E, p1, p2, R, d)
            f = pose_err(E, p1, p2, R, d, far=True)
            rows.append((a[0], a[1], a[2], f[0], f[2]))
        v = np.array(rows)
        by_b[str(b)] = {"基线与深度之比（%）": r(b / DEPTH * 100, 2), "旋转误差中位数（度）": r(np.median(v[:, 0]), 2),
                        "旋转误差超过 90 度的比例": r(np.mean(v[:, 0] > 90), 3), "平移方向误差中位数（度）": r(np.median(v[:, 1]), 1),
                        "recoverPose 返回的点数中位数（共 100）": int(np.median(v[:, 2])),
                        "distanceThresh 调到 1e6 后旋转误差超过 90 度的比例": r(np.mean(v[:, 3] > 90), 3),
                        "distanceThresh 调到 1e6 后返回的点数中位数": int(np.median(v[:, 4]))}

    # 正深度检查的距离阈值：无噪声、真实的 E、所有点在同一深度
    Rt, tt = rot_xyz(0.02, -0.03, 0.01), np.array([1.0, 0.0, 0.0])
    Et = np.array([[0, -tt[2], tt[1]], [tt[2], 0, -tt[0]], [-tt[1], tt[0], 0]]) @ Rt
    rng = np.random.default_rng(0)
    thresh = {}
    for z in (45, 49, 51, 55):
        X = np.stack([rng.uniform(-0.2, 0.2, 200) * z, rng.uniform(-0.15, 0.15, 200) * z, np.full(200, float(z))], axis=1)
        q1, q2 = proj(X, K_TRUE, np.eye(3), np.zeros(3)), proj(X, K_TRUE, Rt, tt)
        thresh[str(z)] = {"默认": int(cv2.recoverPose(Et, q1, q2, K_TRUE)[0]), "distanceThresh=1e6": int(cv2.recoverPose(Et, q1, q2, K_TRUE, distanceThresh=1e6)[0])}

    reliefs = [0, 5, 20, 50, 150]
    by_relief = {}
    for noise in (0.0, 0.5):
        by_relief[str(noise)] = {}
        for relief in reliefs:
            res = {}
            for method in ("RANSAC", "8 点"):
                rows = []
                for i in range(trials):
                    p1, p2, R, d = two_views(i, 100.0, float(relief), noise=noise)
                    E, _ = essential(p1, p2, method, i)
                    rows.append(pose_err(E, p1, p2, R, d)[:2])
                v = np.array(rows)
                res[method] = {"成功率": r(np.mean((v[:, 0] < 2) & (v[:, 1] < 10)), 2), "旋转误差中位数（度）": r(np.median(v[:, 0]), 2), "平移方向误差中位数（度）": r(np.median(v[:, 1]), 1)}
            by_relief[str(noise)][str(relief)] = res

    by_out = {}
    for frac in (0.0, 0.05, 0.2, 0.4):
        rows = []
        for i in range(trials):
            p1, p2, R, d = two_views(i, 100.0, 150.0, outlier=frac)
            row = []
            E8, _ = essential(p1, p2, "8 点", i)
            row += pose_err(E8, p1, p2, R, d)[:2]
            Er, inl = essential(p1, p2, "RANSAC", i)
            row += pose_err(Er, p1, p2, R, d)[:2]
            E2, _ = essential(p1[inl], p2[inl], "8 点", i)
            row += pose_err(E2, p1[inl], p2[inl], R, d)[:2]
            Em, _ = essential(p1, p2, "MAGSAC", i)
            row += pose_err(Em, p1, p2, R, d)[:2]
            rows.append(row + [inl.sum()])
        v = np.median(np.array(rows, dtype=float), axis=0)
        by_out[str(frac)] = {name: {"旋转误差中位数（度）": r(v[k], 2), "平移方向误差中位数（度）": r(v[k + 1], 1)}
                             for name, k in (("8 点法，用全部点", 0), ("RANSAC（讲义里的两行流程）", 2), ("RANSAC 的内点再用 8 点法重估", 4), ("USAC_MAGSAC", 6))}
        by_out[str(frac)]["RANSAC 判为内点的个数中位数（共 100）"] = int(v[8])

    NUM["两视图相对位姿"] = {
        "说明": "100 对匹配点，像素噪声 0.5 px，场景在 600 mm 深度附近（一块倾斜不超过 20 度的平面，加上正负若干毫米的深度起伏），第二台相机相对第一台绕各轴转动不超过 4 度；各 200 次；"
                "默认流程是讲义里的 findEssentialMat（RANSAC，prob=0.999，threshold=1.0）加 recoverPose；平移方向误差是估出的单位向量和真实方向的夹角；"
                "成功指旋转误差小于 2 度并且平移方向误差小于 10 度",
        "按基线（mm；深度起伏正负 150 mm）": by_b,
        "recoverPose 的距离阈值": {"说明": "无噪声，传入真实的本质矩阵，200 个点都在同一深度，基线长度记为 1；键是深度与基线之比，值是通过正深度检查的点数", "结果": thresh},
        "按深度起伏（mm；基线 100 mm；第一层是像素噪声 px）": by_relief,
        "按误匹配比例（基线 100 mm，深度起伏正负 150 mm）": by_out,
    }

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0), layout="constrained")
    bs = [b for b in baselines if b > 0]
    ratio = [b / DEPTH * 100 for b in bs]
    ax[0].plot(ratio, [by_b[str(b)]["平移方向误差中位数（度）"] for b in bs], "s-", color=S.RED, lw=1.8, ms=5, label="平移方向误差")
    ax[0].plot(ratio, [by_b[str(b)]["旋转误差中位数（度）"] for b in bs], "o-", color=S.BLUE, lw=1.8, ms=5, label="旋转误差")
    ax[0].axhline(90, color=S.MUTED, lw=1, ls=":")
    ax[0].text(ratio[-1], 62, "90°：和随机猜一样", fontsize=8.5, color=S.MUTED, ha="right")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xticks(ratio, [f"{v:.2g}" for v in ratio])
    ax[0].set_yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    ax[0].minorticks_off()
    ax[0].set_xlabel("基线 / 场景深度（%）")
    ax[0].set_ylabel("误差中位数（度）")
    ax[0].set_title("基线太短：平移方向没法用，旋转还在")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[0])
    x = np.arange(len(baselines))
    ax[1].bar(x - 0.2, [by_b[str(b)]["recoverPose 返回的点数中位数（共 100）"] for b in baselines], 0.38, color=S.ORANGE, alpha=0.85, label="默认")
    ax[1].bar(x + 0.2, [by_b[str(b)]["distanceThresh 调到 1e6 后返回的点数中位数"] for b in baselines], 0.38, color=S.BLUE, alpha=0.85, label="distanceThresh=1e6")
    ax[1].set_xticks(x, [f"{b / DEPTH * 100:.2g}" for b in baselines])
    ax[1].set_xlabel("基线 / 场景深度（%）")
    ax[1].set_ylabel("recoverPose 返回的点数（中位数，共 100）")
    ax[1].set_title("深度超过基线 50 倍的点，默认不计数")
    ax[1].legend(fontsize=8.5, loc="upper left")
    S.clean(ax[1])
    ax[1].grid(axis="x", visible=False)
    xr = np.arange(len(reliefs))
    for method, color, marker, name in (("RANSAC", S.BLUE, "o", "RANSAC 流程"), ("8 点", S.ORANGE, "s", "8 点法")):
        ax[2].plot(xr, [by_relief["0.5"][str(v)][method]["成功率"] * 100 for v in reliefs], marker + "-", color=color, lw=1.8, ms=5, label=f"{name}，噪声 0.5 px")
        ax[2].plot(xr, [by_relief["0.0"][str(v)][method]["成功率"] * 100 for v in reliefs], marker + ":", color=color, lw=1.2, ms=4, label=f"{name}，无噪声")
    ax[2].set_xticks(xr, ["0"] + [f"±{v}" for v in reliefs[1:]])
    ax[2].set_ylim(-3, 105)
    ax[2].set_xlabel("场景的深度起伏（mm；0 是纯平面；场景深度 600 mm）")
    ax[2].set_ylabel("成功率（%）")
    ax[2].set_title("场景越接近平面，越容易失败")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    S.save(fig, CH, "fig06_相对位姿的退化")


# ---------------------------------------------------------------- 不画图的几组数字
def misc_numbers():
    # 旋转向量不是欧拉角
    rv = cv2.Rodrigues(rot_xyz(*np.deg2rad((20, 40, 30))))[0].ravel()
    r1 = cv2.Rodrigues(rot_xyz(np.deg2rad(20), 0, 0))[0]
    r2 = cv2.Rodrigues(rot_xyz(0, np.deg2rad(40), 0))[0]
    NUM["旋转向量"] = {
        "说明": "先绕 x 轴转 20 度、再绕 y 轴转 40 度、再绕 z 轴转 30 度（固定轴），用 cv2.Rodrigues 换成旋转向量",
        "旋转向量的三个分量（换算成度）": [r(v, 1) for v in np.degrees(rv)], "旋转向量的长度，也就是总转角（度）": r(np.degrees(np.linalg.norm(rv)), 1),
        "把“绕 x 轴 20 度”和“绕 y 轴 40 度”两个旋转向量直接相加，和真正先后做这两个旋转相差（度）": r(rot_err(cv2.Rodrigues(r1 + r2)[0], rot_xyz(0, np.deg2rad(40), 0) @ rot_xyz(np.deg2rad(20), 0, 0)), 1),
    }

    # 方形标记：角点顺序、单位、IPPE_SQUARE 的点序
    orders = {"顺序正确": [0, 1, 2, 3], "整体错开一位": [1, 2, 3, 0], "整体反向": [3, 2, 1, 0], "交换相邻两个": [1, 0, 2, 3]}
    res = {k: [] for k in orders}
    unit, square = [], {"角点在原点的写法 + IPPE_SQUARE": [], "角点在原点的写法 + IPPE": [], "文档规定的点序 + IPPE_SQUARE": []}
    corner0 = np.array([[0, 0, 0], [L_MARK, 0, 0], [L_MARK, L_MARK, 0], [0, L_MARK, 0]], dtype=float)
    shift = np.array([-L_MARK / 2, L_MARK / 2, 0.0])
    for i in range(300):
        rng = np.random.default_rng(8000 + i)
        R = rot_xyz(*np.deg2rad(rng.uniform(-30, 30, 2)), np.deg2rad(rng.uniform(-20, 20))) @ FLIP
        t = np.array([rng.uniform(-40, 40), rng.uniform(-30, 30), rng.uniform(300, 500)])
        uv = proj(SQUARE, K_TRUE, R, t) + rng.normal(scale=0.3, size=(4, 2))
        for name, order in orders.items():
            Re, te = pnp(SQUARE, uv[order])
            res[name].append((rms(cv2.projectPoints(SQUARE, cv2.Rodrigues(Re)[0], te, K_TRUE, D0)[0].reshape(-1, 2), uv[order]), rot_err(Re, R), te[2] < 0))
        Ra, ta = pnp(SQUARE, uv)
        Rb, tb = pnp(SQUARE * 100, uv)
        unit.append((rot_err(Ra, Rb), tb[2] / ta[2]))
        # 同一个标记换成“角点在原点、y 朝下”的坐标系：corner0 = FLIP @ (SQUARE - shift)
        Rc, tc = R @ FLIP, t + R @ shift
        for name, obj, flag, Rt, tt in (("角点在原点的写法 + IPPE_SQUARE", corner0, cv2.SOLVEPNP_IPPE_SQUARE, Rc, tc),
                                        ("角点在原点的写法 + IPPE", corner0, cv2.SOLVEPNP_IPPE, Rc, tc),
                                        ("文档规定的点序 + IPPE_SQUARE", SQUARE, cv2.SOLVEPNP_IPPE_SQUARE, R, t)):
            Re, te = pnp(obj, uv, flag)
            rp = cv2.projectPoints(obj, cv2.Rodrigues(Re)[0], te, K_TRUE, D0)[0].reshape(-1, 2)
            square[name].append((rot_err(Re, Rt), rms(rp, uv), te[2] < 0))
    assert np.allclose(proj(corner0, K_TRUE, Rc, tc), proj(SQUARE, K_TRUE, R, t))
    NUM["方形标记的角点顺序"] = {
        "说明": "边长 50 mm 的方形标记，300 个随机位姿（距离 300 到 500 mm），角点噪声 0.3 px，cv2.solvePnP 默认方法；物体点顺序不变，把图像点按不同方式排错",
        "结果": {k: {"重投影 RMS 中位数（px）": float(f"{np.median([x[0] for x in v]):.3g}"), "旋转误差中位数（度）": r(np.median([x[1] for x in v]), 1),
                    "估出的 tz 为负的比例": r(np.mean([x[2] for x in v]), 2)} for k, v in res.items()},
    }
    NUM["单位写错 100 倍"] = {"说明": "同上 300 个位姿，把物体点坐标乘 100 再求一次位姿",
                         "两次旋转之差的最大值（度）": float(f"{np.max([x[0] for x in unit]):.1g}"), "平移的倍数（最小、最大）": [r(np.min([x[1] for x in unit]), 2), r(np.max([x[1] for x in unit]), 2)]}
    NUM["IPPE_SQUARE 的点序"] = {
        "说明": "同上 300 个位姿。“角点在原点的写法”是 [0,0,0],[L,0,0],[L,L,0],[0,L,0]；“文档规定的点序”是 [-L/2,L/2,0],[L/2,L/2,0],[L/2,-L/2,0],[-L/2,-L/2,0]；重投影 RMS 用传入的物体点和返回的位姿算",
        "结果": {k: {"旋转误差中位数（度）": r(np.median([x[0] for x in v]), 1), "重投影 RMS 中位数（px）": float(f"{np.median([x[1] for x in v]):.3g}"),
                    "估出的 tz 为负的比例": r(np.mean([x[2] for x in v]), 2)} for k, v in square.items()},
    }
    # 讲义示例里的那组像素点
    img_pts = np.array([[312, 211], [441, 220], [430, 349], [301, 338]], dtype=np.float64)
    example = {}
    for name, obj, flag, top in (("角点在原点的写法 + IPPE_SQUARE", corner0, cv2.SOLVEPNP_IPPE_SQUARE, -L_MARK), ("角点在原点的写法 + IPPE", corner0, cv2.SOLVEPNP_IPPE, -L_MARK),
                                 ("文档规定的点序 + IPPE_SQUARE", SQUARE, cv2.SOLVEPNP_IPPE_SQUARE, L_MARK)):
        ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K_TRUE, D0, flags=flag)
        Re = cv2.Rodrigues(rvec)[0]
        rp = cv2.projectPoints(obj, rvec, tvec, K_TRUE, D0)[0].reshape(-1, 2)
        example[name] = {"函数返回值": bool(ok), "tz（mm）": r(tvec.ravel()[2], 1), "重投影 RMS（px）": float(f"{rms(rp, img_pts):.3g}"),
                         "立方体顶面比底面离相机近": bool((Re @ np.array([0, 0, top]))[2] < 0)}
    NUM["IPPE_SQUARE 的点序"]["讲义 7.2 节示例的那组像素点"] = {"说明": "边长 50 mm，内参用本章的合成相机（原讲义没有给 K）；顶面指示例里 Z=-L 的那一面，文档点序下改为 Z=+L", "结果": example}

    # 最少点数
    minimum = {}
    for n in (4, 5, 6):
        for name, coplanar in (("共面", True), ("不共面", False)):
            fail = 0
            for i in range(200):
                rng = np.random.default_rng(8500 + i)
                obj = rand_points(rng, n, coplanar)
                R, t = rand_pose(rng)
                fail += pnp(obj, proj(obj, K_TRUE, R, t)) is None
            minimum[f"{n} 个点，{name}"] = fail
    NUM["最少点数"] = {"说明": "cv2.solvePnP 默认的 ITERATIVE 方法，无噪声，各 200 次；值是报错或返回失败的次数。OpenCV 版本 " + cv2.__version__, "结果": minimum}

    # 离群点
    outl = {}
    for frac in (0.0, 0.02, 0.1, 0.3, 0.5, 0.7):
        rows = []
        for i in range(200):
            rng = np.random.default_rng(9000 + i)
            obj = rand_points(rng, 50, False)
            R, t = rand_pose(rng)
            uv = proj(obj, K_TRUE, R, t) + rng.normal(scale=0.5, size=(50, 2))
            nb = int(round(frac * 50))
            idx = rng.choice(50, nb, replace=False)
            uv[idx] = np.stack([rng.uniform(0, W, nb), rng.uniform(0, H, nb)], axis=1)
            Ra, _ = pnp(obj, uv)
            cv2.setRNGSeed(i)
            ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, uv, K_TRUE, D0, reprojectionError=2.0, iterationsCount=200, confidence=0.99)
            rows.append((rot_err(Ra, R), rot_err(cv2.Rodrigues(rvec)[0], R) if ok else np.nan, 0 if inl is None else len(inl)))
        v = np.array(rows, dtype=float)
        bad = int(np.isnan(v[:, 1]).sum())
        outl[str(frac)] = {"错误点个数（共 50）": int(round(frac * 50)), "solvePnP 旋转误差中位数（度）": r(np.median(v[:, 0]), 2),
                           "solvePnPRansac 旋转误差中位数（度）": None if bad > 100 else r(np.nanmedian(v[:, 1]), 2),
                           "solvePnPRansac 返回失败的次数（共 200）": bad, "solvePnPRansac 内点数中位数": int(np.median(v[:, 2]))}
    NUM["离群点"] = {"说明": "50 个不共面的点，噪声 0.5 px，把其中一部分图像点换成画面内的随机位置；solvePnPRansac 用 reprojectionError=2.0、iterationsCount=200、confidence=0.99；各 200 次", "结果": outl}

    # calibrateCamera 的 image_size 写反
    xs, ys = np.meshgrid(np.arange(9) * 15.0, np.arange(6) * 15.0)
    board = np.stack([xs.ravel(), ys.ravel(), np.zeros(54)], axis=1)
    board[:, :2] -= board[:, :2].mean(axis=0)
    rows = []
    for i in range(20):
        rng = np.random.default_rng(9500 + i)
        views = []
        while len(views) < 12:
            R = rot_xyz(*np.deg2rad(rng.uniform(-30, 30, 2)), np.deg2rad(rng.uniform(-20, 20)))
            z = rng.uniform(380, 520)
            t = z * np.linalg.inv(K_TRUE) @ np.array([K_TRUE[0, 2] + 0.85 * rng.uniform(-1, 1) * W / 2, K_TRUE[1, 2] + 0.85 * rng.uniform(-1, 1) * H / 2, 1.0])
            uv = project_k1(board, K_TRUE, R, t, -0.1)
            if uv[:, 0].min() > 2 and uv[:, 0].max() < W - 2 and uv[:, 1].min() > 2 and uv[:, 1].max() < H - 2:
                views.append((uv + rng.normal(scale=0.2, size=uv.shape)).astype(np.float32).reshape(-1, 1, 2))
        obj = [board.astype(np.float32)] * 12
        row = []
        base = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
        for flags in (base, base | cv2.CALIB_FIX_PRINCIPAL_POINT):
            for size in ((W, H), (H, W)):
                e, Kc, dc, _, _ = cv2.calibrateCamera(obj, views, size, None, None, flags=flags)
                row += [e, Kc[0, 0], Kc[0, 2], Kc[1, 2], dc.ravel()[0]]
        rows.append(row)
    a = np.array(rows)
    v = np.median(a, axis=0)
    pack = lambda k: {"重投影 RMS（px）": r(v[k], 2), "fx": r(v[k + 1], 1), "cx": r(v[k + 2], 1), "cy": r(v[k + 3], 1), "k1": r(v[k + 4], 3)}  # noqa: E731
    NUM["image_size 写反"] = {
        "说明": "和第 17 章相同的 9x6 标定板，12 张视图覆盖全画面，相机带 k1=-0.1 的畸变，角点噪声 0.2 px，20 次标定取中位数；真值 fx=820、cx=318、cy=247；"
                "写反指把 (640, 480) 传成 (480, 640)",
        "默认参数": {"尺寸正确": pack(0), "尺寸写反": pack(5), "20 次里两者的 fx 相差都小于 0.001": bool(np.abs(a[:, 1] - a[:, 6]).max() < 1e-3)},
        "加 CALIB_FIX_PRINCIPAL_POINT": {"尺寸正确": pack(10), "尺寸写反": pack(15)},
    }


# ---------------------------------------------------------------- 演示页里的几何，用参考实现重算一遍
def demo_numbers():
    """demos/相机投影与畸变.html 的棋盘和默认参数。讲义里“按顺序试”的数字从这里取，另用 node 执行页面脚本核对过。"""
    sq, cols, rows = 25.0, 10, 7
    gx, gy = np.meshgrid(np.arange(cols) * sq, np.arange(rows) * sq)
    board = np.stack([gx.ravel(), gy.ravel(), np.zeros(cols * rows)], axis=1)

    def corners(fx=800.0, fy=800.0, cx=320.0, cy=240.0, rx=20.0, ry=-15.0, tz=420.0, k1=0.0):
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        R = rot_xyz(np.deg2rad(rx), np.deg2rad(ry), 0.0)
        t = np.array([-(cols - 1) * sq / 2, -(rows - 1) * sq / 2, tz])
        return project_k1(board, K, R, t, k1) if k1 else proj(board, K, R, t)

    width = lambda uv: r(np.ptp(uv[:, 0]), 1)  # noqa: E731
    base = corners()
    far = corners(k1=-0.3)
    dist_c = np.linalg.norm(base - np.array([320.0, 240.0]), axis=1)
    move = np.linalg.norm(far - base, axis=1)
    tilt = corners(rx=0, ry=40)
    col_h = lambda uv, j: r(np.ptp(uv.reshape(rows, cols, 2)[:, j, 1]), 1)  # noqa: E731
    NUM["演示页"] = {
        "说明": "demos/相机投影与畸变.html 的默认参数：fx=fy=800，主点 (320, 240)，绕 x 轴 20 度、绕 y 轴 -15 度，tz=420 mm，10x7 个角点、格距 25 mm；下面每一项只改写明的滑块",
        "两个旋转都设 0：tz=420 和 tz=840 时棋盘的宽度（px）": [width(corners(rx=0, ry=0)), width(corners(rx=0, ry=0, tz=840))],
        "只把 cx 从 320 改到 380：所有角点的水平位移（px）": [r(np.min(corners(cx=380)[:, 0] - base[:, 0]), 1), r(np.max(corners(cx=380)[:, 0] - base[:, 0]), 1)],
        "fx、fy 改到 1200 同时 tz 改到 630，两个旋转都是 0：角点的最大位移（px）": r(np.abs(corners(rx=0, ry=0, fx=1200, fy=1200, tz=630) - corners(rx=0, ry=0)).max(), 6),
        "fx、fy 改到 1200 同时 tz 改到 630，旋转保持默认：角点的最大位移（px）": r(np.linalg.norm(corners(fx=1200, fy=1200, tz=630) - base, axis=1).max(), 1),
        "只把 fx、fy 改到 1200：棋盘宽度从多少到多少（px）": [width(base), width(corners(fx=1200, fy=1200))],
        "k1=-0.3：离主点最远的角点（距主点多少 px、移动多少 px）": [r(dist_c.max(), 1), r(move[np.argmax(dist_c)], 1)],
        "k1=-0.3：离主点最近的角点（距主点多少 px、移动多少 px）": [r(dist_c.min(), 1), r(move[np.argmin(dist_c)], 2)],
        "绕 x 轴 0 度、绕 y 轴 40 度：最左一列和最右一列角点的高度（px）": [col_h(tilt, 0), col_h(tilt, cols - 1)],
        "两个旋转都是 0：每一列角点的高度（px）": col_h(corners(rx=0, ry=0), 0),
    }


def main() -> None:
    S.setup()
    plt.rcParams["svg.hashsalt"] = "ch07"  # 固定 SVG 里的元素编号，重复运行逐字节相同
    for fn in (fig_conventions, fig_pnp, fig_ambiguity, fig_intrinsics, fig_triangulation, fig_relative, misc_numbers, demo_numbers):
        fn()
        print("完成", fn.__name__, flush=True)
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False))


if __name__ == "__main__":
    main()
