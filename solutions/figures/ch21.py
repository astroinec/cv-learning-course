"""第 21 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch21.py

输出：<主分支>/21_3D数学与渲染管线/figures/*.svg 和 实测数字.json。
视图矩阵、透视矩阵、屏幕映射和 z-buffer 都调用本章的参考实现（21-1_软光栅.py）。参考实现只输出深度，
需要面编号、覆盖次数、背面剔除和左上规则的地方，用本文件里的 raster()，它的“边上算在内”规则和参考实现逐像素核对过。
本机不运行真正的 OpenGL；和 OpenCV 约定的对照用的是第 7、17 章那台合成相机，投影公式自己写，不依赖 cv2。

耗时每次运行都会波动，用 S.pinned 钉住：默认沿用 实测数字.json 里上次的值，设置 CH21_RETIME=1 才重测。
参考实现的 render_depth 只遍历三角形的包围盒；耗时一节拿它和本文件里“每个三角形算整幅图”的 fullframe_render 对比。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

CH = "21_3D数学与渲染管线"
mod = S.load_solution(CH, "21-1_软光栅.py")
NUM: dict = {}

# 立方体：顶点和面表与演示页、Notebook 相同。CUBE_F 从外面看逆时针，按 cross(b - a, c - a) 算出的法线朝外。
# CW_FACES 把每个三角形的后两个下标对调（从外面看顺时针），用来展示“面表的绕序整体写反”时光照的症状。
CUBE_V = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
CUBE_F = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [3, 6, 2], [3, 7, 6], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]]
NB_FACES = [tuple(f) for f in CUBE_F]
CW_FACES = [(a, c, b) for a, b, c in CUBE_F]
CUBE_E = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
LIGHT = np.array([0.4, 0.8, 0.6]) / np.linalg.norm([0.4, 0.8, 0.6])
UP = np.array([0.0, 1.0, 0.0])


def r(x, n=3):
    return round(float(x), n)


def sig(x, n=3):
    return float(f"{float(x):.{n}g}")


def T4(t):
    M = np.eye(4)
    M[:3, 3] = t
    return M


def S4(s):
    return np.diag([s[0], s[1], s[2], 1.0])


def R4(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    M = np.eye(4)
    if axis == "x":
        M[1:3, 1:3] = [[c, -s], [s, c]]
    elif axis == "y":
        M[0, 0], M[0, 2], M[2, 0], M[2, 2] = c, s, -s, c
    else:
        M[:2, :2] = [[c, -s], [s, c]]
    return M


def orbit(az, el, dist):
    a, e = np.radians(az), np.radians(el)
    return dist * np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])


def look(eye, target=(0.0, 0.0, 0.0), up=UP):
    return mod.look_at(np.asarray(eye, dtype=float), np.asarray(target, dtype=float), np.asarray(up, dtype=float))


def homog(points):
    return np.hstack([np.asarray(points, dtype=float), np.ones((len(points), 1))])


def screen_nocheck(points, mvp, W, H):
    """和参考实现的 to_screen 相同，但不检查 clip w 的符号，照样做除法。用来展示“不处理相机后方的点”的症状。"""
    clip = homog(points) @ mvp.T
    ndc = clip[:, :3] / clip[:, 3:4]
    return np.stack([(ndc[:, 0] + 1) / 2 * W, (1 - ndc[:, 1]) / 2 * H, (ndc[:, 2] + 1) / 2], axis=1)


def raster(tris, W, H, rule="inclusive", cull=None):
    """z-buffer 光栅化，返回 (深度, 面编号, 每个像素被多少个三角形覆盖)。

    rule="inclusive"：边上算在内，算式和参考实现逐项相同；"exclusive"：边上不算；"topleft"：左上规则。
    cull=None 不剔除；"neg" 只画屏幕有向面积为负的三角形；"pos" 只画为正的。含 NaN 顶点的三角形跳过。
    """
    depth = np.full((H, W), np.inf)
    fid = np.full((H, W), -1)
    cnt = np.zeros((H, W), dtype=int)
    ys, xs = np.mgrid[0:H, 0:W]
    px, py = xs + 0.5, ys + 0.5
    for i, tri in enumerate(np.asarray(tris, dtype=float)):
        if not np.isfinite(tri).all():
            continue
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = tri
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue
        if (cull == "neg" and area > 0) or (cull == "pos" and area < 0):
            continue
        w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / area
        w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / area
        w2 = 1.0 - w0 - w1
        if rule == "inclusive":
            inside = (w0 >= -1e-12) & (w1 >= -1e-12) & (w2 >= -1e-12)
        else:
            sgn = 1.0 if area > 0 else -1.0
            inside = np.ones((H, W), dtype=bool)
            for (ax, ay), (bx, by) in (((x1, y1), (x2, y2)), ((x2, y2), (x0, y0)), ((x0, y0), (x1, y1))):
                e = sgn * ((ax - px) * (by - py) - (bx - px) * (ay - py))      # 边函数，内部为正；共享边在两个三角形里互为相反数
                gx, gy = -sgn * (by - ay), sgn * (bx - ax)                      # 边函数的梯度，指向三角形内部
                owns = rule == "topleft" and (gx > 0 or (gx == 0 and gy > 0))  # 左边：内部在右；上边：水平且内部在下（y 向下）
                inside &= (e > 0) | ((e == 0) & owns)
        z = w0 * z0 + w1 * z1 + w2 * z2
        cnt += inside
        upd = inside & (z < depth)
        depth[upd] = z[upd]
        fid[upd] = i
    return depth, fid, cnt


def frame(ax, W, H, pad=0.0):
    ax.set_xlim(-pad * W, W + pad * W)
    ax.set_ylim(H + pad * H, -pad * H)
    ax.set_aspect("equal")
    S.image_axes(ax)


def wire(ax, uv, edges, color, lw=1.8, **kw):
    for a, b in edges:
        if np.isfinite(uv[[a, b]]).all():
            ax.plot(uv[[a, b], 0], uv[[a, b], 1], color=color, lw=lw, solid_capstyle="round", **kw)


# ---------------------------------------------------------------- 图 1：矩阵顺序
def order_variants(P, V, Tm, Rm, Sm):
    M = Tm @ Rm @ Sm
    return {
        "正确：P V M，其中 M = T R S": P @ V @ M,
        "模型矩阵写成 S R T": P @ V @ (Sm @ Rm @ Tm),
        "模型矩阵写成 R T S": P @ V @ (Rm @ Tm @ Sm),
        "视图和模型换位，写成 P M V": P @ M @ V,
        "投影和视图换位，写成 V P M": V @ P @ M,
        "整个顺序倒过来，写成 M V P": M @ V @ P,
        "按行向量的习惯写成 p @ MVP": (P @ V @ M).T,
        "视图矩阵用成了它的逆": P @ np.linalg.inv(V) @ M,
    }


def fig_order():
    W, H = 240, 180
    P = mod.perspective(50.0, W / H, 0.5, 20.0)
    names = list(order_variants(P, np.eye(4), np.eye(4), np.eye(4), np.eye(4)))
    stats = {n: [] for n in names}
    for i in range(300):
        rng = np.random.default_rng(2100 + i)
        V = look(orbit(rng.uniform(-180, 180), rng.uniform(-40, 40), rng.uniform(5, 8)))
        var = order_variants(P, V, T4(rng.uniform(-1, 1, 3)), R4("y", rng.uniform(-90, 90)), S4(rng.uniform(0.5, 1.5, 3)))
        good = mod.to_screen(CUBE_V, var[names[0]], W, H)
        for name, mvp in var.items():
            s = mod.to_screen(CUBE_V, mvp, W, H)
            nan = np.isnan(s[:, 0])
            inside = (~nan) & (s[:, 0] >= 0) & (s[:, 0] <= W) & (s[:, 1] >= 0) & (s[:, 1] <= H) & (s[:, 2] >= 0) & (s[:, 2] <= 1)
            d = np.linalg.norm(s[:, :2] - good[:, :2], axis=1)
            stats[name].append((nan.mean(), inside.mean(), np.nan if nan.all() else np.nanmean(d)))
    table = {}
    for name, v in stats.items():
        v = np.array(v, dtype=float)
        table[name] = {"落到相机后方的顶点比例（%）": r(100 * v[:, 0].mean(), 1), "留在视锥内的顶点比例（%）": r(100 * v[:, 1].mean(), 1),
                       "顶点偏差的中位数（px）": r(np.nanmedian(v[:, 2]), 1)}

    eye = np.array([3.0, 2.5, 5.0])
    Tm, Rm, Sm = T4([0.8, 0.2, -0.5]), R4("y", 35.0), S4([1.4, 1.0, 0.6])
    var = order_variants(P, look(eye), Tm, Rm, Sm)
    scr = {name: mod.to_screen(CUBE_V, mvp, W, H) for name, mvp in var.items()}
    good = scr[names[0]]
    one = {}
    for name, s in scr.items():
        nan = np.isnan(s[:, 0])
        one[name] = {"落到相机后方的顶点数": int(nan.sum()),
                     "顶点平均偏差（px）": None if nan.all() else r(np.nanmean(np.linalg.norm(s[:, :2] - good[:, :2], axis=1)), 1)}
    NUM["矩阵顺序"] = {
        "说明": "画面 240x180，垂直视场角 50 度，near 0.5，far 20；边长 2 的立方体；300 个随机场景（种子 2100 到 2399）：相机在距原点 5 到 8、仰角正负 40 度以内的球面上看向原点，"
                "模型矩阵 M = T R S，平移每个分量在正负 1 之间，绕 y 轴旋转正负 90 度以内，三个方向的缩放各在 0.5 到 1.5 之间；"
                "每种写法都用参考实现的 to_screen 投影 8 个顶点，偏差是和正确写法的像素距离，先在一个场景里对能投影的顶点取平均，再对 300 个场景取中位数；"
                "留在视锥内指像素坐标在画面内并且深度在 0 到 1 之间",
        "300 个随机场景": table,
        "图 1 用的场景": {"相机位置": eye.tolist(), "平移": [0.8, 0.2, -0.5], "绕 y 轴旋转（度）": 35, "缩放": [1.4, 1.0, 0.6], "各种写法": one},
    }

    show = [("模型矩阵写成 S R T", "模型矩阵写成 S R T：还在画面里，偏了一点"),
            ("视图和模型换位，写成 P M V", "写成 P M V：大半跑到画面外"),
            ("按行向量的习惯写成 p @ MVP", "写成 p @ MVP：形状完全变了"),
            ("视图矩阵用成了它的逆", "视图矩阵用成逆：顶点落到相机后方")]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.5), layout="constrained")
    for a, (name, title) in zip(ax, show):
        s = scr[name]
        a.add_patch(plt.Rectangle((0, 0), W, H, fill=False, ec=S.INK, lw=1.2))
        wire(a, good, CUBE_E, "#9aa0a8", lw=2.6)
        wire(a, s, CUBE_E, S.RED, lw=1.5)
        ok = np.isfinite(s[:, 0])
        a.plot(s[ok, 0], s[ok, 1], "o", color=S.RED, ms=3.5)
        allp = np.vstack([s[ok, :2], [[0, 0], [W, H]]])
        lo, hi = np.maximum(allp.min(axis=0), [-1.2 * W, -1.2 * H]) - 12, np.minimum(allp.max(axis=0), [2.2 * W, 2.2 * H]) + 12
        (cx, cy), (bw, bh) = (lo + hi) / 2, hi - lo
        bw, bh = max(bw, bh * 4 / 3), max(bh, bw * 3 / 4)          # 四栏都撑成 4:3，标题才在同一高度
        a.set_xlim(cx - bw / 2, cx + bw / 2)
        a.set_ylim(cy + bh / 2, cy - bh / 2)
        a.set_aspect("equal")
        S.image_axes(a)
        for side in a.spines.values():
            side.set_visible(False)
        note = f"平均偏差 {one[name]['顶点平均偏差（px）']} px" if one[name]["顶点平均偏差（px）"] is not None else "没有顶点能投影"
        if one[name]["落到相机后方的顶点数"]:
            note = f"{one[name]['落到相机后方的顶点数']} 个顶点在相机后方，" + note
        a.set_xlabel(note, fontsize=9.5, color=S.MUTED)
        a.set_title(title, fontsize=10.5)
    ax[0].plot([], [], color="#9aa0a8", lw=2.6, label="正确的投影")
    ax[0].plot([], [], color=S.RED, lw=1.5, label="写错之后")
    ax[0].plot([], [], color=S.INK, lw=1.2, label="画面边框（240x180）")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=9.5)
    S.save(fig, CH, "fig01_矩阵顺序")


# ---------------------------------------------------------------- 图 2：法线变换
def sphere(nlat, nlon):
    verts = []
    for i in range(nlat + 1):
        th = np.pi * i / nlat
        for j in range(nlon):
            ph = 2 * np.pi * j / nlon
            verts.append([np.sin(th) * np.cos(ph), np.cos(th), np.sin(th) * np.sin(ph)])
    verts = np.array(verts)
    faces = []
    for i in range(nlat):
        for j in range(nlon):
            a, b = i * nlon + j, i * nlon + (j + 1) % nlon
            c, d = (i + 1) * nlon + (j + 1) % nlon, (i + 1) * nlon + j
            if i > 0:
                faces.append([a, b, c])
            if i < nlat - 1:
                faces.append([a, c, d])
    return verts, np.array(faces)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def fig_normal():
    # 解析部分：缩放 diag(s, 1, 1)，法线在 xy 平面内扫 0 到 90 度
    th = np.radians(np.linspace(0, 90, 9001))
    n2 = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], axis=1)
    curve_s = np.round(np.concatenate([np.linspace(1, 4, 31), np.linspace(4.5, 10, 12)]), 2)
    curve = []
    for s in curve_s:
        M = np.diag([s, 1.0, 1.0])
        ang = np.degrees(np.arccos(np.clip((unit(n2 @ M.T) * unit(n2 @ np.linalg.inv(M))).sum(axis=1), -1, 1)))
        curve.append((float(ang.max()), float(np.degrees(th[ang.argmax()]))))
    pick = [1.0, 1.5, 2.0, 3.0, 4.0, 10.0]
    by_s = {str(s): {"法线方向误差的最大值（度）": r(curve[list(curve_s).index(s)][0], 2), "出现在法线与缩放轴夹角（度）": r(curve[list(curve_s).index(s)][1], 1),
                     "按 arctan(s) - arctan(1/s) 算（度）": r(np.degrees(np.arctan(s) - np.arctan(1 / s)), 2)} for s in pick}

    # 渲染：单位球经 M = Ry(30) S(2, 1, 1) 变成椭球，逐面 Lambert（不加环境光）
    W, H = 240, 180
    verts, faces = sphere(20, 40)
    M = R4("y", 30.0) @ S4([2.0, 1.0, 1.0])
    M3 = M[:3, :3]
    eye = np.array([2.5, 3.0, 6.5])
    P = mod.perspective(40.0, W / H, 0.5, 30.0)
    s = mod.to_screen(verts, P @ look(eye) @ M, W, H)
    tris = s[faces]
    depth, fid, _ = raster(tris, W, H)
    assert np.array_equal(depth, mod.render_depth(tris, W, H)), "raster 的深度应与参考实现逐像素相同"
    rng = np.random.default_rng(2120)
    shuffled = sum(np.array_equal(depth, mod.render_depth(tris[rng.permutation(len(tris))], W, H)) for _ in range(20))
    n_obj = unit(verts[faces].mean(axis=1))
    n_right = unit(n_obj @ np.linalg.inv(M3))        # (M^-T n)^T = n^T M^-1
    n_wrong = unit(n_obj @ M3.T)
    assert np.allclose(np.linalg.inv(M).T[:3, :3], np.linalg.inv(M3).T)
    sh_right, sh_wrong = np.maximum(0, n_right @ LIGHT), np.maximum(0, n_wrong @ LIGHT)
    cov = fid >= 0
    img_r, img_w = np.where(cov, sh_right[fid], np.nan), np.where(cov, sh_wrong[fid], np.nan)
    diff = np.abs(img_r - img_w)
    ang_face = np.degrees(np.arccos(np.clip((n_right * n_wrong).sum(axis=1), -1, 1)))
    # 对照：均匀缩放加旋转、带平移的模型矩阵
    Mu = R4("y", 30.0) @ S4([1.7, 1.7, 1.7])
    uni = np.degrees(np.arccos(np.clip((unit(n_obj @ np.linalg.inv(Mu[:3, :3])) * unit(n_obj @ Mu[:3, :3].T)).sum(axis=1), -1, 1))).max()
    as_point = (T4([5.0, 0.0, 0.0]) @ np.array([0.0, 1.0, 0.0, 1.0]))[:3]
    NUM["法线变换"] = {
        "说明": "法线直接乘模型矩阵的 3x3 部分，和乘它的逆转置，两者归一化后的夹角。解析部分：缩放 diag(s, 1, 1)，法线在 xy 平面内从 0 扫到 90 度，取最大夹角。"
                "渲染部分：单位球（20x40 的经纬网格，1520 个三角形）经 M = Ry(30 度) S(2, 1, 1) 变成椭球，画面 240x180，垂直视场角 40 度，相机在 (2.5, 3.0, 6.5) 看向原点，"
                "光的方向 (0.4, 0.8, 0.6) 归一化，逐面 Lambert 亮度 max(0, n·l)，不加环境光；深度用参考实现的 render_depth 核对过",
        "按缩放比": by_s,
        "椭球渲染（缩放 2:1:1）": {
            "椭球占的像素数": int(cov.sum()),
            "可见像素上法线方向误差的平均（度）": r(ang_face[fid[cov]].mean(), 1), "可见像素上法线方向误差的最大值（度）": r(ang_face[fid[cov]].max(), 1),
            "亮度绝对误差的平均": r(np.nanmean(diff), 3), "亮度绝对误差的最大值": r(np.nanmax(diff), 3),
            "亮度误差超过 0.1 的像素比例（%）": r(100 * np.mean(diff[cov] > 0.1), 1),
            "正确的平均亮度": r(np.nanmean(img_r), 3), "错误的平均亮度": r(np.nanmean(img_w), 3),
            "错误的平均亮度比正确的低百分之几": r(100 * (1 - np.nanmean(img_w) / np.nanmean(img_r)), 1),
            "受光与背光判断相反的像素比例（%）": r(100 * np.mean((img_r[cov] > 0) != (img_w[cov] > 0)), 1),
            "把 1520 个三角形的绘制顺序随机打乱 20 次，深度图和原来逐像素相同的次数": int(shuffled),
        },
        "均匀缩放 1.7 倍加旋转 30 度时两种算法的最大夹角（度）": sig(uni, 2),
        "把法线 (0, 1, 0) 当成点（w=1）乘平移 (5, 0, 0) 的矩阵": {"结果": as_point.tolist(), "偏了多少度": r(np.degrees(np.arccos(as_point[1] / np.linalg.norm(as_point))), 1)},
    }

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.5), gridspec_kw={"width_ratios": [1, 1, 1.12, 1.15]}, layout="constrained")
    for a, img, title in ((ax[0], img_r, "法线乘逆转置：正确的明暗"), (ax[1], img_w, "法线直接乘 M：两头偏暗")):
        a.imshow(img, cmap="gray", vmin=0, vmax=1, extent=(0, W, H, 0))
        a.set_title(title, fontsize=10.5)
        frame(a, W, H)
    im = ax[2].imshow(diff, cmap="magma", vmin=0, vmax=float(np.nanmax(diff)), extent=(0, W, H, 0))
    ax[2].set_title("亮度差的绝对值：集中在两头", fontsize=10.5)
    frame(ax[2], W, H)
    fig.colorbar(im, ax=ax[2], fraction=0.035, pad=0.03).ax.tick_params(labelsize=8)
    ax[3].plot(curve_s, [c[0] for c in curve], color=S.BLUE, lw=1.8)
    ax[3].plot(pick[1:], [by_s[str(s)]["法线方向误差的最大值（度）"] for s in pick[1:]], "o", color=S.BLUE, ms=5)
    for s in pick[1:]:
        ax[3].annotate(f"{by_s[str(s)]['法线方向误差的最大值（度）']:.1f}°", (s, by_s[str(s)]["法线方向误差的最大值（度）"]), textcoords="offset points", xytext=(6, -12), fontsize=8.5, color=S.MUTED)
    ax[3].set_xlabel("一个方向的缩放比 s（另两个方向为 1）")
    ax[3].set_ylabel("法线方向误差的最大值（度）")
    ax[3].set_ylim(0, 90)
    ax[3].set_title("缩放越不均匀，法线偏得越多")
    S.clean(ax[3])
    S.save(fig, CH, "fig02_法线变换")


# ---------------------------------------------------------------- 表：旋转表示
def tab_rotation():
    E = lambda yaw, pitch, roll: (R4("z", yaw) @ R4("y", pitch) @ R4("x", roll))[:3, :3]   # noqa: E731
    ang_between = lambda A, B: float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))))   # noqa: E731

    drift = {}
    step64 = (R4("z", 1.0) @ R4("x", 0.7) @ R4("y", 0.3))[:3, :3]
    for dt in (np.float32, np.float64):
        R, step = np.eye(3, dtype=dt), step64.astype(dt)
        rec = {}
        for k in range(1, 1000001):
            R = R @ step
            if k in (1000, 100000, 1000000):
                R64 = R.astype(np.float64)
                rec[str(k)] = {"R^T R 与单位阵的最大差": sig(np.abs(R64.T @ R64 - np.eye(3)).max(), 2), "行列式与 1 的差": sig(abs(np.linalg.det(R64) - 1), 2)}
        drift[dt.__name__] = rec

    def quat(R):
        w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])

    def qmat(q):
        w, x, y, z = q / np.linalg.norm(q)
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    def slerp(q0, q1, t):
        th = np.arccos(min(1.0, q0 @ q1))
        return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)

    e0, e1 = (0.0, 0.0, 0.0), (90.0, 60.0, 90.0)
    R0, R1 = E(*e0), E(*e1)
    q0, q1 = quat(R0), quat(R1)
    assert q0 @ q1 > 0 and np.allclose(qmat(q1), R1)
    dev_e, dev_n, st_s, st_e, st_n, prev = [], [], [], [], [], None
    for t in np.linspace(0, 1, 101):
        cur = (qmat(slerp(q0, q1, t)), E(*[(1 - t) * a + t * b for a, b in zip(e0, e1)]), qmat((1 - t) * q0 + t * q1))
        dev_e.append(ang_between(cur[0], cur[1]))
        dev_n.append(ang_between(cur[0], cur[2]))
        if prev is not None:
            st_s.append(ang_between(prev[0], cur[0]))
            st_e.append(ang_between(prev[1], cur[1]))
            st_n.append(ang_between(prev[2], cur[2]))
        prev = cur
    axis_angle = 2 * np.arccos(q1[0]) * q1[1:] / np.linalg.norm(q1[1:])
    NUM["旋转表示"] = {
        "说明": "欧拉角按 R = Rz(偏航) Ry(俯仰) Rx(滚转) 合成；夹角指两个旋转之间的转角 arccos((tr(A^T B) - 1) / 2)",
        "连乘后的正交性": {"说明": "从单位阵出发，反复右乘同一个小旋转 Rz(1 度) Rx(0.7 度) Ry(0.3 度)，不做任何重新正交化；数字是乘了这么多次之后的结果", "结果": drift},
        "万向锁": {"说明": "俯仰角 90 度时，(偏航 30, 滚转 10) 和 (偏航 20, 滚转 0) 以及 (偏航 50, 滚转 30) 给出的旋转矩阵之差的最大元素；俯仰 89 度时前两组的差",
                  "俯仰 90 度：(30, 90, 10) 对 (20, 90, 0)": sig(np.abs(E(30, 90, 10) - E(20, 90, 0)).max(), 2),
                  "俯仰 90 度：(30, 90, 10) 对 (50, 90, 30)": sig(np.abs(E(30, 90, 10) - E(50, 90, 30)).max(), 2),
                  "俯仰 89 度：(30, 89, 10) 对 (20, 89, 0)": sig(np.abs(E(30, 89, 10) - E(20, 89, 0)).max(), 2)},
        "矩阵取平均": {"说明": "单位阵和绕 z 轴转 theta 的旋转矩阵逐元素取平均，得到的矩阵的行列式（旋转矩阵应为 1）",
                    "结果": {str(a): r(np.linalg.det((np.eye(3) + R4("z", a)[:3, :3]) / 2), 4) for a in (30, 90, 120, 180)}},
        "插值": {"说明": "从欧拉角 (0, 0, 0) 插值到 (偏航 90, 俯仰 60, 滚转 90)，t 从 0 到 1 取 101 个点；slerp 是四元数球面线性插值",
                "两个姿态之间的转角（度）": r(ang_between(R0, R1), 1),
                "终点姿态换成旋转向量（三个分量换算成度）": [r(np.degrees(v), 1) for v in axis_angle],
                "slerp 每一步的转角（度）最小和最大": [r(min(st_s), 3), r(max(st_s), 3)],
                "slerp 走过的总转角（度）": r(sum(st_s), 1),
                "四元数线性插值再归一化：走过的总转角（度）": r(sum(st_n), 1),
                "欧拉角逐分量线性插值：每一步的转角（度）最小和最大": [r(min(st_e), 3), r(max(st_e), 3)],
                "欧拉角逐分量线性插值：走过的总转角（度）": r(sum(st_e), 1),
                "欧拉角逐分量线性插值：中途偏离 slerp 最多（度）": r(max(dev_e), 1),
                "四元数线性插值再归一化：每一步的转角（度）最小和最大": [r(min(st_n), 3), r(max(st_n), 3)],
                "四元数线性插值再归一化：中途偏离 slerp 最多（度）": r(max(dev_n), 2)},
    }


# ---------------------------------------------------------------- 图 3：深度非线性
def fig_depth():
    far = 100.0
    nears = [0.001, 0.01, 0.1, 1.0]

    def depth_at(near, far_, dist):
        P = mod.perspective(50.0, 1.0, near, far_)
        d = np.atleast_1d(np.asarray(dist, dtype=float))
        return mod.to_screen(np.stack([np.zeros_like(d), np.zeros_like(d), -d], axis=1), P, 2, 2)[:, 2]

    table = {}
    for near in nears:
        row = {"深度到 0.5 时的距离": sig(2 * near * far / (far + near), 4),
               "距离 1、10、50 处的深度": [float(f"{v:.6f}") + 0.0 for v in depth_at(near, far, [max(1.0, near), 10.0, 50.0])],
               "距离 10 以外占用的深度范围（%）": sig(100 * (1 - depth_at(near, far, 10.0)[0]), 2)}
        for bits in (16, 24):
            step = 1 / (2 ** bits - 1)
            row[f"{bits} 位深度缓冲在距离 1、10、50 处能分开的最小间隔"] = [sig(d * d * (far - near) / (far * near) * step, 2) for d in (1.0, 10.0, 50.0)]
        table[str(near)] = row
    far_effect = {str(f): sig(50 * 50 * (f - 0.1) / (f * 0.1) / (2 ** 24 - 1), 4) for f in (100, 1000, 10000)}
    nb = {str(n): [r(v, 4) for v in depth_at(n, 20.0, [5.0, 10.0])] for n in (0.01, 0.5, 2.0)}
    NUM["深度非线性"] = {
        "说明": "屏幕深度 = (z_ndc + 1) / 2，用参考实现的 perspective 和 to_screen 计算，far = 100。能分开的最小间隔 = 一个量化台阶对应的距离变化 = d^2 (far - near) / (far near) / (2^位数 - 1)，"
                "是按整数深度缓冲算出的理论值，不是在显卡上实测的",
        "按 near": table,
        "near = 0.1 时，far 取 100、1000、10000，24 位深度缓冲在距离 50 处能分开的最小间隔": far_effect,
        "Notebook 的设置（far = 20）：near 取 0.01、0.5、2.0 时距离 5 和 10 处的深度": nb,
    }

    # 实测：地面上方 gap 处平放一块贴片，看量化之后贴片和地面的深度还分不分得开
    W, H = 320, 240
    V = look([0.0, 1.5, 0.0], [0.0, 0.0, -12.0])
    floor = np.array([[-40, 0, -1.0], [40, 0, -1.0], [40, 0, -90.0], [-40, 0, -90.0]])
    fight, maps = {}, {}
    for gap in (0.01, 0.001):
        decal = np.array([[-3, gap, -2.0], [3, gap, -2.0], [3, gap, -80.0], [-3, gap, -80.0]])
        rows = {}
        for near in nears:
            MVP = mod.perspective(50.0, W / H, near, far) @ V
            sf, sd = mod.to_screen(floor, MVP, W, H), mod.to_screen(decal, MVP, W, H)
            df = mod.render_depth(np.stack([sf[[0, 1, 2]], sf[[0, 2, 3]]]), W, H)
            dd = mod.render_depth(np.stack([sd[[0, 1, 2]], sd[[0, 2, 3]]]), W, H)
            m = np.isfinite(dd) & np.isfinite(df) & (dd >= 0) & (dd <= 1)
            dist = near * far / (far - dd * (far - near))          # 由深度反算沿视线方向的距离
            row = {"贴片占的像素数": int(m.sum())}
            for bits in (16, 24):
                q = 2 ** bits - 1
                tie = m & (np.round(dd * q) >= np.round(df * q))
                row[f"{bits} 位：分不开的像素比例（%）"] = r(100 * tie.sum() / m.sum(), 1)
                row[f"{bits} 位：最近的一处分不开的像素离相机多远"] = sig(dist[tie].min(), 2) if tie.any() else None
                maps[(gap, near, bits)] = (m, tie, np.isfinite(df))
            rows[str(near)] = row
        fight[str(gap)] = rows
    NUM["深度冲突"] = {
        "说明": "画面 320x240，垂直视场角 50 度，far = 100；相机在 (0, 1.5, 0) 看向 (0, 0, -12)；地面是 y = 0 的大矩形，贴片是地面上方 gap 处、宽 6、从距离 2 铺到 80 的矩形。"
                "地面和贴片各用参考实现的 render_depth 画一次（float64），再把深度四舍五入成 16 位或 24 位整数；贴片的量化深度不小于地面的，就算分不开（用“更近才写入”的深度测试时，谁先画谁赢）",
        "按贴片离地面的高度 gap": fight,
    }

    fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.7), gridspec_kw={"width_ratios": [1.15, 1.15, 1.15, 1.0]}, layout="constrained")
    dist = np.linspace(0.5, 20, 400)
    for near, color in ((0.01, S.RED), (0.5, S.ORANGE), (2.0, S.BLUE)):
        d = dist[dist >= near]
        ax[0].plot(d, depth_at(near, 20.0, d), color=color, lw=1.8, label=f"near = {near}")
    ax[0].set_xlabel("到相机的距离（far = 20）")
    ax[0].set_ylabel("屏幕深度（0 到 1）")
    ax[0].set_ylim(0, 1.03)
    ax[0].set_title("near 越小，深度越早挤到 1 附近")
    ax[0].legend(fontsize=9, loc="lower right")
    S.clean(ax[0])
    dd_ = np.logspace(0, 2, 60)
    for near, color in zip(nears, (S.RED, S.ORANGE, S.GREEN, S.BLUE)):
        ax[1].plot(dd_, dd_ ** 2 * (far - near) / (far * near) / (2 ** 24 - 1), color=color, lw=1.8, label=f"near = {near}")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xticks([1, 10, 100], ["1", "10", "100"])
    ax[1].set_yticks([1e-7, 1e-5, 1e-3, 1e-1], ["1e-7", "1e-5", "1e-3", "0.1"])
    ax[1].minorticks_off()
    ax[1].set_xlabel("到相机的距离（far = 100）")
    ax[1].set_ylabel("24 位深度能分开的最小间隔")
    ax[1].set_title("near 小 10 倍，间隔粗 10 倍")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[1])
    x = np.arange(len(nears))
    for (gap, bits), color, marker in (((0.01, 16), S.ORANGE, "s"), ((0.001, 16), S.RED, "o"), ((0.001, 24), S.BLUE, "^")):
        ax[2].plot(x, [fight[str(gap)][str(n)][f"{bits} 位：分不开的像素比例（%）"] for n in nears], marker + "-", color=color, lw=1.8, ms=5, label=f"{bits} 位，贴片高 {gap}")
    ax[2].set_xticks(x, [str(n) for n in nears])
    ax[2].set_xlabel("near（far = 100）")
    ax[2].set_ylabel("贴片上分不开的像素比例（%）")
    ax[2].set_ylim(-3, 103)
    ax[2].set_title("实测：near 越小，冲突的像素越多")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    m, tie, fl = maps[(0.01, 0.01, 16)]
    rgb = np.ones((H, W, 3))
    rgb[fl] = [0.88, 0.89, 0.91]
    rgb[m] = [0.18, 0.44, 0.93]
    rgb[tie] = [0.84, 0.21, 0.18]
    ax[3].imshow(rgb, extent=(0, W, H, 0))
    ax[3].set_title("near 0.01、16 位、贴片高 0.01", fontsize=10.5)
    ax[3].set_xlabel("蓝：贴片分得开；红：分不开；灰：地面", fontsize=9, color=S.MUTED)
    frame(ax[3], W, H)
    S.save(fig, CH, "fig03_深度非线性")


# ---------------------------------------------------------------- 图 4：OpenGL 和 OpenCV 两种相机约定
def rot_xyz(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rand_pose(rng, tilt=30.0):
    """和第 7 章配图脚本相同的随机位姿（同样的取数顺序），这样两章的数字可以直接对照。"""
    R = rot_xyz(*np.deg2rad(rng.uniform(-tilt, tilt, 2)), np.deg2rad(rng.uniform(-20, 20)))
    t = np.array([rng.uniform(-30, 30), rng.uniform(-20, 20), rng.uniform(450, 550)])
    return R, t


def fig_convention():
    W, H = 640, 480
    K = np.array([[820.0, 0.0, 318.0], [0.0, 790.0, 247.0], [0.0, 0.0, 1.0]])
    cube = np.array([[x, y, z] for z in (-50, 50) for y in (-50, 50) for x in (-50, 50)], dtype=float)
    edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
    near, far = 10.0, 5000.0

    def P_from_K(half=0.0):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2] + half, K[1, 2] + half
        return np.array([[2 * fx / W, 0, 1 - 2 * cx / W, 0], [0, 2 * fy / H, 2 * cy / H - 1, 0],
                         [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]])

    def cv_proj(P3, R, t):
        pc = P3 @ R.T + t
        return np.stack([K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]], axis=1)

    def pose4(R, t):
        E = np.eye(4)
        E[:3, :3], E[:3, 3] = R, t
        return E

    PK = P_from_K()
    fovy = float(np.degrees(2 * np.arctan(H / 2 / K[1, 1])))
    aspect = float((K[1, 1] / H) * W / K[0, 0])
    Psym, Pnaive = mod.perspective(fovy, aspect, near, far), mod.perspective(fovy, W / H, near, far)

    def cases(R, t):
        E = pose4(R, t)
        return {
            "转换后：视图矩阵 = diag(1, -1, -1, 1) [R|t]，投影矩阵由 K 构造": mod.to_screen(cube, PK @ FLIP @ E, W, H),
            "不转换，直接把 [R|t] 当视图矩阵（to_screen 检查 w）": mod.to_screen(cube, PK @ E, W, H),
            "不转换，也不检查 w，照样做除法": screen_nocheck(cube, PK @ E, W, H),
            "只把 z 取反，y 没有取反": mod.to_screen(cube, PK @ np.diag([1.0, 1.0, -1.0, 1.0]) @ E, W, H),
            "坐标轴转换了，投影用对称视锥（不管主点）": mod.to_screen(cube, Psym @ FLIP @ E, W, H),
            "坐标轴转换了，对称视锥并且宽高比直接取 W/H": mod.to_screen(cube, Pnaive @ FLIP @ E, W, H),
        }

    stats: dict = {}
    for i in range(300):
        R, t = rand_pose(np.random.default_rng(1000 + i))
        good = cv_proj(cube, R, t)
        for name, s in cases(R, t).items():
            nan = np.isnan(s[:, 0])
            d = np.linalg.norm(s[:, :2] - good, axis=1)
            stats.setdefault(name, []).append((nan.all(), np.nan if nan.all() else np.nanmean(d), np.nan if nan.all() else np.nanmax(d)))
    table = {}
    for name, v in stats.items():
        v = np.array(v, dtype=float)
        allb = float(v[:, 0].mean())
        table[name] = {"8 个顶点全部落到相机后方的位姿比例": r(allb, 2),
                       "顶点偏差的中位数（px）": None if allb == 1 else sig(np.nanmedian(v[:, 1]), 4),
                       "顶点偏差的最大值（px）": None if allb == 1 else sig(np.nanmax(v[:, 2]), 3)}

    R0, t0 = rot_xyz(np.deg2rad(20), np.deg2rad(-25), np.deg2rad(10)), np.array([40.0, -25.0, 500.0])
    good0, var0 = cv_proj(cube, R0, t0), cases(R0, t0)
    gl_ok = var0["转换后：视图矩阵 = diag(1, -1, -1, 1) [R|t]，投影矩阵由 K 构造"]
    mirror = np.stack([2 * K[0, 2] - good0[:, 0], good0[:, 1]], axis=1)
    vflip = np.stack([good0[:, 0], 2 * K[1, 2] - good0[:, 1]], axis=1)
    ch7_file = S.MAIN / "07_传统CV综合" / "figures" / "实测数字.json"
    ch7 = None
    if ch7_file.exists():
        ch7 = json.loads(ch7_file.read_text(encoding="utf-8"))["约定弄反"]["300 个随机位姿"]["OpenGL 相机轴没转换，用 cv2.projectPoints 投影"]["顶点偏差的中位数（px）"]

    # 像素中心：OpenCV 的整数坐标在像素中心，参考实现的屏幕坐标里像素中心在 +0.5
    rng = np.random.default_rng(2140)
    shift = {0.0: [], 0.5: []}
    for _ in range(200):
        x0, x1, y0, y1 = rng.uniform(-120, -20), rng.uniform(20, 120), rng.uniform(-90, -20), rng.uniform(20, 90)
        quad = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])
        t = np.array([0.0, 0.0, 500.0])
        uv = cv_proj(quad, np.eye(3), t)
        for half in shift:
            s = mod.to_screen(quad, P_from_K(half) @ FLIP @ pose4(np.eye(3), t), W, H)
            ys, xs = np.nonzero(np.isfinite(mod.render_depth(np.stack([s[[0, 1, 2]], s[[0, 2, 3]]]), W, H)))
            shift[half].append((xs.mean() - uv[:, 0].mean(), ys.mean() - uv[:, 1].mean()))
    NUM["相机约定"] = {
        "说明": "合成相机和第 7、17 章相同：640x480，fx = 820，fy = 790，主点 (318, 247)，长度单位毫米；边长 100 mm 的立方体；300 个随机位姿和第 7 章图 1 用的是同一批（种子 1000 到 1299）。"
                "正确答案是 OpenCV 约定下的针孔投影 u = fx Xc/Zc + cx，v = fy Yc/Zc + cy；被比较的是参考实现的 to_screen（OpenGL 约定）在各种写法下给出的像素坐标；"
                "偏差先在一个位姿里对 8 个顶点取平均，再对 300 个位姿取中位数；near = 10，far = 5000",
        "由 K 构造的投影矩阵": {"P[0][0] = 2 fx / W": r(PK[0, 0], 4), "P[1][1] = 2 fy / H": r(PK[1, 1], 4), "P[0][2] = 1 - 2 cx / W": r(PK[0, 2], 5), "P[1][2] = 2 cy / H - 1": r(PK[1, 2], 5),
                          "等效的垂直视场角（度）": r(fovy, 2), "等效的宽高比 (W / fx) / (H / fy)": r(aspect, 4), "图像的宽高比 W / H": r(W / H, 4),
                          "主点离画面中心（px）": [K[0, 2] - W / 2, K[1, 2] - H / 2], "主点离画面中心的距离（px）": r(np.hypot(K[0, 2] - W / 2, K[1, 2] - H / 2), 2)},
        "300 个随机位姿": table,
        "第 7 章 实测数字.json 里“OpenGL 相机轴没转换，用 cv2.projectPoints 投影”的偏差中位数（px）": ch7,
        "图 4 用的位姿（和第 7 章图 1 相同）": {
            "旋转（绕 x、y、z 轴，度）": [20, -25, 10], "t（mm）": [40, -25, 500],
            "转换后与 OpenCV 投影的最大差（px）": sig(np.abs(gl_ok[:, :2] - good0).max(), 2),
            "不转换也不检查 w 的结果与“绕主点左右镜像”的最大差（px）": sig(np.abs(var0["不转换，也不检查 w，照样做除法"][:, :2] - mirror).max(), 2),
            "只把 z 取反的结果与“绕主点上下翻转”的最大差（px）": sig(np.abs(var0["只把 z 取反，y 没有取反"][:, :2] - vflip).max(), 2),
            "各种写法的顶点平均偏差（px）": {name: (None if np.isnan(s).any() else sig(np.mean(np.linalg.norm(s[:, :2] - good0, axis=1)), 4)) for name, s in var0.items()},
            "转换后 8 个顶点的屏幕深度范围": [r(gl_ok[:, 2].min(), 4), r(gl_ok[:, 2].max(), 4)]},
        "像素中心": {"说明": "200 个正对相机、距离 500 mm 的随机矩形（种子 2140），用由 K 构造的投影矩阵和参考实现的 render_depth 光栅化；比较被覆盖像素的下标的质心和 OpenCV 投影出的四个角点的质心",
                  "直接用 cx、cy：质心之差的平均（x, y，px）": [r(v, 2) for v in np.mean(shift[0.0], axis=0)],
                  "用 cx + 0.5、cy + 0.5：质心之差的平均（x, y，px）": [r(v, 2) for v in np.mean(shift[0.5], axis=0)]},
    }

    show = [("转换后：视图矩阵 = diag(1, -1, -1, 1) [R|t]，投影矩阵由 K 构造", "转换后：和 OpenCV 的投影重合", S.BLUE),
            ("不转换，也不检查 w，照样做除法", "不转换也不检查 w：左右镜像", S.RED),
            ("只把 z 取反，y 没有取反", "只把 z 取反：上下翻转", S.RED),
            ("坐标轴转换了，投影用对称视锥（不管主点）", "对称视锥：每个顶点偏 7.3 px（局部放大）", S.RED)]
    assert abs(table[show[3][0]]["顶点偏差的中位数（px）"] - 7.3) < 0.05
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), layout="constrained")
    for k, (a, (name, title, color)) in enumerate(zip(ax, show)):
        s = var0[name]
        wire(a, good0, edges, "#9aa0a8", lw=3.0)
        wire(a, s, edges, color, lw=1.4)
        a.plot(*good0[0], "o", color="#6b7280", ms=6)
        a.plot(*s[0, :2], "o", color=color, ms=4)
        if k < 3:
            frame(a, W, H)
            a.plot(K[0, 2], K[1, 2], "+", color=S.INK, ms=9, mew=1.2)
        else:
            a.set_xlim(good0[0, 0] - 46, good0[0, 0] + 46)
            a.set_ylim(good0[0, 1] + 34.5, good0[0, 1] - 34.5)
            a.set_aspect("equal")
            S.image_axes(a)
            a.annotate("", xy=s[0, :2], xytext=good0[0], arrowprops=dict(arrowstyle="->", color=S.INK, lw=1.3))
            a.text(good0[0, 0] + 5, good0[0, 1] - 9, "同一个顶点，偏 7.3 px", fontsize=9.5, color=S.INK)
            a.set_xlabel("这一栏宽 92 px", fontsize=9, color=S.MUTED)
        a.set_title(title, fontsize=10.5)
    ax[0].plot([], [], color="#9aa0a8", lw=3.0, label="OpenCV 约定下的投影（正确答案）")
    ax[0].plot([], [], color=S.BLUE, lw=1.4, label="OpenGL 管线，转换正确")
    ax[0].plot([], [], color=S.RED, lw=1.4, label="OpenGL 管线，写错之后")
    ax[0].plot([], [], "+", color=S.INK, ms=9, mew=1.2, label="主点")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=9.5)
    S.save(fig, CH, "fig04_相机约定")


# ---------------------------------------------------------------- 图 5：共享边的像素归属
def grid_mesh(n, step, offset, jitter=0.0, rng=None):
    pts = np.array([[i * step + offset, j * step + offset] for j in range(n + 1) for i in range(n + 1)], dtype=float)
    if jitter:
        pts += rng.uniform(-jitter, jitter, pts.shape)
    tris = []
    for j in range(n):
        for i in range(n):
            a, b, c, d = j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i
            tris += [[a, b, c], [a, c, d]]
    return np.array([[[pts[k][0], pts[k][1], 0.5] for k in t] for t in tris])


def fig_edges():
    W = H = 80
    rules = [("inclusive", "边上算在内（作业的约定）"), ("exclusive", "边上不算"), ("topleft", "左上规则")]
    ys, xs = np.mgrid[0:H, 0:W]

    def region(lo, hi):
        return (xs + 0.5 > lo) & (xs + 0.5 < hi) & (ys + 0.5 > lo) & (ys + 0.5 < hi)

    def count(tris, reg):
        out = {}
        for rule, label in rules:
            _, _, cnt = raster(tris, W, H, rule)
            if rule == "inclusive":
                ref = sum(np.isfinite(mod.render_depth(t[None], W, H)).astype(int) for t in tris)
                assert np.array_equal(cnt, ref), "raster 的 inclusive 规则应与参考实现逐像素相同"
            out[label] = {"区域内的像素数": int(reg.sum()), "没有被任何三角形覆盖的像素": int(((cnt == 0) & reg).sum()), "被覆盖两次以上的像素": int(((cnt >= 2) & reg).sum()),
                          "单个像素最多被覆盖几次": int(cnt[reg].max()), "片元总数": int(cnt[reg].sum()), "_cnt": cnt}
        return out

    res = {"顶点在整数坐标": count(grid_mesh(8, 8, 8.0), region(8, 72)), "顶点在像素中心（半整数坐标）": count(grid_mesh(8, 8, 8.5), region(8.5, 72.5))}
    tot = {label: [0, 0, 0] for _, label in rules}
    for k in range(50):
        one = count(grid_mesh(8, 8, 8.0, 2.5, np.random.default_rng(2150 + k)), region(12, 68))
        for label, v in one.items():
            tot[label][0] += v["没有被任何三角形覆盖的像素"]
            tot[label][1] += v["被覆盖两次以上的像素"]
            tot[label][2] += v["区域内的像素数"]
    res["顶点是随机浮点坐标（50 个网格合计）"] = {label: {"区域内的像素数": v[2], "没有被任何三角形覆盖的像素": v[0], "被覆盖两次以上的像素": v[1], "片元总数": v[2] - v[0] + v[1]} for label, v in tot.items()}
    cnt_int = {label: v.pop("_cnt") for label, v in res["顶点在整数坐标"].items()}
    for v in res["顶点在像素中心（半整数坐标）"].values():
        v.pop("_cnt")
    tri = np.array([[[0.0, 0.0, 0.5], [8.0, 0.0, 0.5], [0.0, 8.0, 0.5]]])
    NUM["共享边规则"] = {
        "说明": "8x8 个边长 8 px 的方格，每格沿对角线切成两个三角形，共 128 个三角形，铺成一块 64x64 px 的平面；画面 80x80。统计区域是这块平面的内部（随机浮点坐标一组的顶点各自随机挪动正负 2.5 px 以内，"
                "区域取中间 56x56 px）。三种规则：边上算在内（三个重心坐标 >= 0，和作业、参考实现相同）；边上不算（> 0）；左上规则（像素中心正好在边上时，只有这条边是三角形的左边或上边才算）",
        "结果": res,
        "验收用的三角形 (0,0)、(8,0)、(0,8) 覆盖的像素数": {label: int(raster(tri, 10, 10, rule)[2].sum()) for rule, label in rules},
    }
    for group in res.values():
        for v in group.values():
            v["片元总数与区域内像素数之比"] = r(v["片元总数"] / v["区域内的像素数"], 3)
    NUM["共享边规则"]["顶点在整数坐标时，像素中心落在对角线上的像素占比（%）"] = r(100 * res["顶点在整数坐标"]["边上算在内（作业的约定）"]["被覆盖两次以上的像素"] / 4096, 1)
    NUM["共享边规则"]["每个方格的对角线经过的像素中心数"] = 512 // 64

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#d6362f", "#dfe6f3", "#e8a33a"])
    mesh = grid_mesh(8, 8, 8.0)
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.9), gridspec_kw={"width_ratios": [1, 1, 1, 1.35]}, layout="constrained")
    for a, (rule, label) in zip(ax, rules):
        # 只画左上角 24x24 px：色块用位图，像素之间的分隔线单独画，SVG 才不会有上万个小方块
        a.imshow(np.clip(cnt_int[label][8:32, 8:32], 0, 2), cmap=cmap, vmin=-0.5, vmax=2.5, extent=(8, 32, 32, 8), interpolation="nearest")
        a.vlines(np.arange(9, 32), 8, 32, color="white", lw=0.5)
        a.hlines(np.arange(9, 32), 8, 32, color="white", lw=0.5)
        for t in mesh:
            if t[:, 0].min() < 32 and t[:, 1].min() < 32:
                a.plot(*np.vstack([t[:, :2], t[:1, :2]]).T, color=S.INK, lw=0.9)
        a.set_xlim(8, 32)
        a.set_ylim(32, 8)
        a.set_aspect("equal")
        S.image_axes(a)
        v = res["顶点在整数坐标"][label]
        a.set_title(f"{label}：漏 {v['没有被任何三角形覆盖的像素']}，重复 {v['被覆盖两次以上的像素']}", fontsize=10.5)
    ax[0].set_xlabel("红：没画到；浅蓝：画了一次；橙：画了两次", fontsize=9, color=S.MUTED)
    ax[1].set_xlabel("左上角 24x24 px 的局部，黑线是三角形的边", fontsize=9, color=S.MUTED)
    groups = list(res)
    x = np.arange(len(groups))
    for k, ((rule, label), color) in enumerate(zip(rules, (S.ORANGE, S.RED, S.BLUE))):
        vals = [res[g][label]["片元总数与区域内像素数之比"] for g in groups]
        bars = ax[3].bar(x + (k - 1) * 0.27, vals, 0.25, color=color, alpha=0.85, label=label)
        for b, v in zip(bars, vals):
            ax[3].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}".rstrip("0").rstrip("."), ha="center", fontsize=8)
    ax[3].axhline(1, color=S.INK, lw=0.7)
    ax[3].set_xticks(x, ["整数坐标", "像素中心", "随机浮点"])
    ax[3].set_ylim(0, 1.75)
    ax[3].set_ylabel("片元总数 ÷ 应有的像素数")
    ax[3].set_title("只有左上规则在三种顶点位置下都是 1")
    ax[3].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    S.clean(ax[3])
    ax[3].grid(axis="x", visible=False)
    S.save(fig, CH, "fig05_共享边规则")


# ---------------------------------------------------------------- 图 6：绕序与背面剔除
COL = np.array([[214, 57, 49], [214, 57, 49], [46, 161, 33], [46, 161, 33], [47, 122, 229], [47, 122, 229],
                [222, 120, 2], [222, 120, 2], [143, 89, 214], [143, 89, 214], [20, 160, 170], [20, 160, 170]]) / 255.0


def face_normals(faces):
    return unit(np.array([np.cross(CUBE_V[b] - CUBE_V[a], CUBE_V[c] - CUBE_V[a]) for a, b, c in faces]))


def fig_cull():
    W, H = 240, 180
    P = mod.perspective(50.0, W / H, 0.5, 20.0)
    center = CUBE_V.mean(axis=0)
    outward = {name: int(sum(np.cross(CUBE_V[b] - CUBE_V[a], CUBE_V[c] - CUBE_V[a]) @ (CUBE_V[a] - center) > 0 for a, b, c in F))
               for name, F in (("演示页和 Notebook 的面表", NB_FACES), ("绕序整体写反的面表", CW_FACES))}
    nfront, same, flipped, holes, nocull_same = [], [], [], [], []
    for i in range(200):
        rng = np.random.default_rng(2160 + i)
        eye = orbit(rng.uniform(-180, 180), rng.uniform(-60, 60), rng.uniform(4, 8))
        s = mod.to_screen(CUBE_V, P @ look(eye), W, H)
        tris = s[np.array(CUBE_F)]
        area = (tris[:, 1, 0] - tris[:, 0, 0]) * (tris[:, 2, 1] - tris[:, 0, 1]) - (tris[:, 2, 0] - tris[:, 0, 0]) * (tris[:, 1, 1] - tris[:, 0, 1])
        facing = np.array([np.cross(CUBE_V[b] - CUBE_V[a], CUBE_V[c] - CUBE_V[a]) @ (eye - CUBE_V[a]) > 0 for a, b, c in CUBE_F])
        assert np.array_equal(area < 0, facing), "y 向下的屏幕坐标里，朝向相机的面有向面积为负"
        nfront.append(int(facing.sum()))
        full = mod.render_depth(tris, W, H)
        cov = np.isfinite(full)
        d_neg = raster(tris, W, H, cull="neg")[0]
        d_pos = raster(tris, W, H, cull="pos")[0]
        same.append(np.array_equal(full, d_neg))
        flipped.append((np.array_equal(cov, np.isfinite(d_pos)), float(np.mean(d_pos[cov] > full[cov])), float(np.mean(d_pos[cov] - full[cov]))))
        k = int(np.nonzero(facing)[0][0])
        bad = tris.copy()
        bad[k] = bad[k][[0, 2, 1]]
        holes.append(float((cov & ~np.isfinite(raster(bad, W, H, cull="neg")[0])).sum() / cov.sum()))
        d_bad = mod.render_depth(bad, W, H)                       # 顶点换了顺序，重心坐标的舍入不同，所以按 1e-12 的容差比较
        nocull_same.append(np.array_equal(np.isfinite(d_bad), cov) and float(np.abs(d_bad[cov] - full[cov]).max()) < 1e-12)

    eye = np.array([3.0, 2.5, 5.0])
    s = mod.to_screen(CUBE_V, P @ look(eye), W, H)
    tris = s[np.array(CUBE_F)]
    n_out = face_normals(CUBE_F)
    shade_out = 0.15 + 0.85 * np.maximum(0, n_out @ LIGHT)
    bad = tris.copy()
    bad[2] = bad[2][[0, 2, 1]]                                   # 把 z = +1 那个面的第一个三角形的绕序写反
    imgs, fids = {}, {}
    for key, (t, cull) in {"正确剔除": (tris, "neg"), "符号弄反": (tris, "pos"), "一个三角形绕序写反": (bad, "neg"), "不剔除": (tris, None)}.items():
        _, fid, _ = raster(t, W, H, cull=cull)
        rgb = np.ones((H, W, 3))
        rgb[fid >= 0] = (COL * shade_out[:, None])[fid[fid >= 0]]
        imgs[key], fids[key] = rgb, fid
    assert np.array_equal(fids["正确剔除"], fids["不剔除"])
    # 逐面 Lambert，按 Notebook 的写法：逐个三角形调用参考实现，深度最小的决定像素属于哪个面，法线 n = cross(V[b] - V[a], V[c] - V[a])
    def nb_lambert(faces):
        t = s[np.array(faces)]
        per_face = np.stack([mod.render_depth(t[i:i + 1], W, H) for i in range(len(faces))])
        fid, cov = np.argmin(per_face, axis=0), np.isfinite(per_face.min(axis=0))
        shade = 0.15 + 0.85 * np.maximum(0, face_normals(faces) @ LIGHT)
        return cov, shade[fid[cov]], per_face.min(axis=0)

    nb_cov, nb_vals, nb_depth = nb_lambert(NB_FACES)             # Notebook 的面表（逆时针）
    cw_cov, cw_vals, cw_depth = nb_lambert(CW_FACES)             # 绕序整体写反
    assert np.array_equal(nb_cov, cw_cov) and float(np.abs(nb_depth[nb_cov] - cw_depth[nb_cov]).max()) < 1e-12, "绕序不影响覆盖和深度"
    vals_nb = sorted({r(v, 3) for v in nb_vals})
    vals_cw = sorted({r(v, 3) for v in cw_vals})
    rgb_cw = np.ones((H, W, 3))
    rgb_cw[cw_cov] = np.repeat(cw_vals[:, None], 3, axis=1)
    fl = np.array(flipped)
    NUM["背面剔除"] = {
        "说明": "画面 240x180，垂直视场角 50 度；边长 2 的立方体用演示页的面表（12 个三角形，从外面看逆时针）；200 个随机视点（种子 2160 到 2359）：距离 4 到 8，仰角正负 60 度以内，看向原点。"
                "屏幕有向面积用作业里的公式 area = (x1-x0)(y2-y0) - (x2-x0)(y1-y0)，屏幕 y 向下。只画有向面积为负的三角形叫“剔除背面”",
        "面表里法线朝外的三角形数（共 12）": outward,
        "朝向相机的三角形数的分布（200 个视点）": {str(k): nfront.count(k) for k in sorted(set(nfront))},
        "朝向相机的三角形数的平均": r(np.mean(nfront), 2),
        "朝向相机的三角形的屏幕有向面积全为负的视点比例": 1.0,
        "剔除背面后的深度图和不剔除逐像素相同的视点比例": r(np.mean(same), 2),
        "判断符号弄反（只画有向面积为正的）": {"轮廓和正确结果相同的视点比例": r(fl[:, 0].mean(), 2), "轮廓内深度变大的像素比例": r(fl[:, 1].mean(), 3), "深度平均变大多少": r(fl[:, 2].mean(), 4)},
        "一个朝前的三角形绕序写反": {"开着剔除时，漏掉的像素占轮廓的比例（中位数，%）": r(100 * np.median(holes), 1), "不剔除时覆盖不变、深度差小于 1e-12 的视点比例": r(np.mean(nocull_same), 2)},
        "图 6 的视点 (3, 2.5, 5)": {"立方体占的像素数": int((fids["不剔除"] >= 0).sum()),
                               "三个可见面的亮度 0.15 + 0.85 max(0, n·l)": sorted({r(v, 3) for v in shade_out[fids["不剔除"][fids["不剔除"] >= 0]]}),
                               "一个三角形绕序写反时漏掉的像素数": int(((fids["不剔除"] >= 0) & (fids["一个三角形绕序写反"] < 0)).sum())},
        "Notebook 的逐面 Lambert（视点 (3, 2.5, 5)）": {"说明": "按 Notebook 的面表（和演示页相同，从外面看逆时针）和写法 n = cross(V[b] - V[a], V[c] - V[a]) 计算，深度用参考实现",
                                                "可见像素数": int(nb_cov.sum()), "可见像素上出现的亮度值": vals_nb},
        "面表的绕序整体写反时的逐面 Lambert（视点 (3, 2.5, 5)）": {"说明": "把面表里每个三角形的后两个下标对调（从外面看顺时针），其余同上；法线全部朝里。覆盖和深度与绕序正确时相同（深度差小于 1e-12）",
                                                      "可见像素数": int(cw_cov.sum()), "可见像素上出现的亮度值": vals_cw},
    }

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4), layout="constrained")
    for a, (img, title) in zip(ax, ((imgs["正确剔除"], "剔除背面：只画 6 个三角形，画面不变"), (imgs["符号弄反"], "符号弄反：看到的是三个背面"),
                                   (imgs["一个三角形绕序写反"], "一个三角形绕序写反：出现一个洞"), (rgb_cw, "面表的绕序整体写反：三个面一样暗"))):
        a.imshow(img, extent=(0, W, H, 0))
        a.set_title(title, fontsize=10.5)
        frame(a, W, H)
    ax[3].set_xlabel("逐面 Lambert 的灰度图，法线朝里，亮度都是 0.15", fontsize=9, color=S.MUTED)
    S.save(fig, CH, "fig06_绕序与背面剔除")


# ---------------------------------------------------------------- 图 7：透视校正插值
def fig_persp():
    W, H = 320, 240
    quad = np.array([[-2, 0, -1.0], [2, 0, -1.0], [2, 0, -9.0], [-2, 0, -9.0]])
    uv = np.array([[0, 0.0], [1, 0], [1, 1], [0, 1]])
    idx = [[0, 1, 2], [0, 2, 3]]
    checker = lambda u, v: (np.floor(np.clip(u, 0, 0.9999) * 8) + np.floor(np.clip(v, 0, 0.9999) * 8)) % 2   # noqa: E731

    def run(eye, target, up=UP):
        V = look(eye, target, up)
        MVP = mod.perspective(50.0, W / H, 0.1, 50.0) @ V
        s = mod.to_screen(quad, MVP, W, H)
        w = (homog(quad) @ MVP.T)[:, 3]
        zc = (homog(quad) @ V.T)[:, 2]

        def interp(attr):
            """屏幕空间线性插值：把属性放在第三列交给参考实现的 render_depth。共享边上的像素取第一个三角形的值。"""
            out = np.full((H, W), np.inf)
            for t in idx:
                d = mod.render_depth(np.column_stack([s[t, :2], attr[t]])[None], W, H)
                m = np.isfinite(d) & ~np.isfinite(out)
                out[m] = d[m]
            return out

        lin_u, lin_v, inv_w = interp(uv[:, 0]), interp(uv[:, 1]), interp(1 / w)
        with np.errstate(invalid="ignore"):
            pc_u, pc_v = interp(uv[:, 0] / w) / inv_w, interp(uv[:, 1] / w) / inv_w
        depth_lin, zc_lin = interp(s[:, 2]), interp(zc)
        cov = np.isfinite(lin_u)
        ys, xs = np.nonzero(cov)
        ndc = np.stack([(xs + 0.5) / W * 2 - 1, 1 - (ys + 0.5) / H * 2], axis=1)
        inv = np.linalg.inv(MVP)
        a = np.hstack([ndc, -np.ones((len(ndc), 1)), np.ones((len(ndc), 1))]) @ inv.T
        b = np.hstack([ndc, np.ones((len(ndc), 1)), np.ones((len(ndc), 1))]) @ inv.T
        a, b = a[:, :3] / a[:, 3:], b[:, :3] / b[:, 3:]
        hit = a + (a[:, 1] / (a[:, 1] - b[:, 1]))[:, None] * (b - a)          # 像素中心的视线和 y = 0 平面的交点
        tu, tv = (hit[:, 0] + 2) / 4, (-hit[:, 2] - 1) / 8
        ratio = float(w.max() / w.min())
        stat = {"顶点 w 的最大值与最小值之比": r(ratio, 2), "四边形占的像素数": int(cov.sum()),
                "屏幕空间线性插值：v 的最大误差": sig(np.abs(lin_v[cov] - tv).max(), 3), "屏幕空间线性插值：(u, v) 误差的平均": sig(np.hypot(lin_u[cov] - tu, lin_v[cov] - tv).mean(), 3),
                "按 (sqrt(r) - 1) / (sqrt(r) + 1) 算的最大误差": sig((np.sqrt(ratio) - 1) / (np.sqrt(ratio) + 1), 3),
                "屏幕空间线性插值：棋盘格颜色取错的像素比例（%）": r(100 * np.mean(checker(lin_u[cov], lin_v[cov]) != checker(tu, tv)), 1),
                "透视校正插值：(u, v) 的最大误差": sig(np.hypot(pc_u[cov] - tu, pc_v[cov] - tv).max(), 2),
                "屏幕深度线性插值的最大误差": sig(np.abs(depth_lin[cov] - mod.to_screen(hit, MVP, W, H)[:, 2]).max(), 2),
                "相机坐标 z 线性插值的最大误差": sig(np.abs(zc_lin[cov] - (homog(hit) @ V.T)[:, 2]).max(), 3), "相机坐标 z 在四个顶点间的跨度": r(zc.max() - zc.min(), 2)}
        tru = np.full((H, W), np.nan)
        tru[cov] = checker(tu, tv)
        err = np.full((H, W), np.nan)
        err[cov] = np.abs(lin_v[cov] - tv)
        return stat, (np.where(cov, checker(pc_u, pc_v), np.nan), np.where(cov, checker(lin_u, lin_v), np.nan), err, tru)

    poses = [("正上方俯视", [0, 12, -5.0], [0, 0, -5.0]), ("(0, 8, 1)", [0, 8, 1.0], [0, 0, -5.0]), ("(0, 4, 2)", [0, 4, 2.0], [0, 0, -5.0]),
             ("(0, 2, 1)", [0, 2, 1.0], [0, 0, -5.0]), ("(0, 1, 0.5)", [0, 1, 0.5], [0, 0, -5.0]), ("(0, 0.5, 0)", [0, 0.5, 0.0], [0, 0, -5.0])]
    res, imgs = {}, None
    for name, eye, tgt in poses:
        res[name], im = run(eye, tgt, up=(0.0, 0.0, -1.0) if name == "正上方俯视" else UP)
        if name == "(0, 2, 1)":
            imgs = im
    NUM["透视校正插值"] = {
        "说明": "画面 320x240，垂直视场角 50 度。y = 0 平面上的矩形（x 从 -2 到 2，z 从 -1 到 -9），切成两个三角形，四个角的纹理坐标 (u, v) 是 (0,0)、(1,0)、(1,1)、(0,1)。"
                "真值：每个像素中心的视线和平面求交。屏幕空间线性插值：直接用屏幕上的重心坐标插值属性（调用参考实现的 render_depth，把属性放在深度的位置）。"
                "透视校正：分别插值 属性/w 和 1/w 再相除。相机都看向 (0, 0, -5)，位置见各行；“正上方俯视”是相机在 (0, 12, -5) 垂直向下看（up 取 (0, 0, -1)）",
        "按相机位置": res,
    }

    from matplotlib.colors import ListedColormap
    cm = ListedColormap(["#2f3640", "#e9edf3"])
    fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.12, 1.1]}, layout="constrained")
    st = res["(0, 2, 1)"]
    for a, img, title in ((ax[0], imgs[0], "透视校正插值：格子沿直线排列"), (ax[1], imgs[1], "屏幕空间线性插值：沿对角线折了")):
        a.imshow(img, cmap=cm, vmin=0, vmax=1, extent=(0, W, H, 0), interpolation="nearest")
        a.set_title(title, fontsize=10.5)
        frame(a, W, H)
    ax[0].set_xlabel(f"相机在 (0, 2, 1)，顶点 w 之比 {st['顶点 w 的最大值与最小值之比']:.0f}", fontsize=9, color=S.MUTED)
    ax[1].set_xlabel(f"{st['屏幕空间线性插值：棋盘格颜色取错的像素比例（%）']}% 的像素颜色取错", fontsize=9, color=S.MUTED)
    im = ax[2].imshow(imgs[2], cmap="magma", vmin=0, extent=(0, W, H, 0))
    ax[2].set_title("线性插值的 v 误差（v 的范围是 0 到 1）", fontsize=10.5)
    frame(ax[2], W, H)
    fig.colorbar(im, ax=ax[2], fraction=0.035, pad=0.03).ax.tick_params(labelsize=8)
    rr = np.linspace(1, 9, 200)
    ax[3].plot(rr, (np.sqrt(rr) - 1) / (np.sqrt(rr) + 1), color=S.MUTED, lw=1.4, label="(sqrt(r) - 1) / (sqrt(r) + 1)")
    ax[3].plot([v["顶点 w 的最大值与最小值之比"] for v in res.values()], [v["屏幕空间线性插值：v 的最大误差"] for v in res.values()], "o", color=S.RED, ms=6, label="实测：线性插值")
    ax[3].plot([v["顶点 w 的最大值与最小值之比"] for v in res.values()], [v["透视校正插值：(u, v) 的最大误差"] for v in res.values()], "s", color=S.BLUE, ms=5, label="实测：透视校正")
    ax[3].set_xlabel("顶点 w 的最大值与最小值之比 r")
    ax[3].set_ylabel("纹理坐标的最大误差")
    ax[3].set_ylim(-0.03, 0.6)
    ax[3].set_title("误差随顶点的深度比增大，和公式重合")
    ax[3].legend(fontsize=8.5, loc="upper left")
    S.clean(ax[3])
    S.save(fig, CH, "fig07_透视校正插值")


# ---------------------------------------------------------------- 表：一个顶点走完整条管线
def tab_pipeline():
    W, H = 240, 180
    eye, p = np.array([3.0, 2.5, 5.0]), np.array([1.0, 1.0, 1.0, 1.0])
    V, P = look(eye), mod.perspective(50.0, W / H, 0.5, 20.0)
    cam = V @ p
    clip = P @ cam
    ndc = clip[:3] / clip[3]
    scr = mod.to_screen(p[None, :3], P @ V, W, H)[0]
    f = 1 / np.tan(np.radians(25.0))
    NUM["一个顶点走完整条管线"] = {
        "说明": "Notebook 的场景：立方体的顶点 (1, 1, 1)，没有模型变换；相机在 (3, 2.5, 5) 看向原点，up = (0, 1, 0)；垂直视场角 50 度，画面 240x180，near 0.5，far 20。全部用参考实现计算",
        "f = 1 / tan(25 度)": r(f, 4), "f / aspect": r(f / (W / H), 4),
        "相机到原点的距离": r(np.linalg.norm(eye), 3),
        "相机坐标": [r(v, 3) for v in cam[:3]], "裁剪坐标 (x, y, z, w)": [r(v, 3) for v in clip], "NDC": [r(v, 4) for v in ndc],
        "屏幕坐标 (x_pix, y_pix, 深度)": [r(scr[0], 1), r(scr[1], 1), r(scr[2], 4)],
        "所在的像素（列，行）": [int(np.floor(scr[0])), int(np.floor(scr[1]))],
    }


# ---------------------------------------------------------------- 表：相机后方的顶点
def tab_behind():
    W, H = 240, 180
    P = mod.perspective(50.0, W / H, 0.1, 20.0)
    ys, xs = np.mgrid[0:H, 0:W]
    ndc = np.stack([(xs.ravel() + 0.5) / W * 2 - 1, 1 - (ys.ravel() + 0.5) / H * 2], axis=1)
    rows = []
    for i in range(100):
        rng = np.random.default_rng(2170 + i)
        eye = rng.uniform(-0.6, 0.6, 3)
        while True:
            d = unit(rng.normal(size=3))
            if abs(d[1]) < 0.9:
                break
        MVP = P @ look(eye, eye + d)
        inv = np.linalg.inv(MVP)
        a = np.hstack([ndc, -np.ones((len(ndc), 1)), np.ones((len(ndc), 1))]) @ inv.T
        b = np.hstack([ndc, np.ones((len(ndc), 1)), np.ones((len(ndc), 1))]) @ inv.T
        ray = b[:, :3] / b[:, 3:] - a[:, :3] / a[:, 3:]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(ray != 0, (np.sign(ray) - eye) / ray, np.inf).min(axis=1)       # 相机在立方体内部，视线从哪个面穿出去
        truth = mod.to_screen(eye + t[:, None] * ray, MVP, W, H)[:, 2].reshape(H, W)
        s = mod.to_screen(CUBE_V, MVP, W, H)
        tris = s[np.array(CUBE_F)]
        d_ref = mod.render_depth(tris, W, H)
        d_raw = mod.render_depth(screen_nocheck(CUBE_V, MVP, W, H)[np.array(CUBE_F)], W, H)
        cov = np.isfinite(d_ref)
        drawn = np.isfinite(d_raw)
        rows.append((int(np.isnan(s[:, 0]).sum()), int(np.isnan(tris).any(axis=(1, 2)).sum()), 1 - cov.mean(),
                     float(np.abs(d_ref[cov] - truth[cov]).max()) if cov.any() else 0.0,
                     float(np.mean(drawn & ~(np.abs(d_raw - truth) < 1e-6))), float(np.mean(~drawn))))
    rows = np.array(rows, dtype=float)
    NUM["相机后方的顶点"] = {
        "说明": "相机放在边长 2 的立方体内部：位置的每个分量在正负 0.6 之间随机取，朝向随机，100 次（种子 2170 到 2269）；画面 240x180，垂直视场角 50 度，near 0.1，far 20。"
                "相机在封闭的立方体里，每个像素都应该看到内壁；真值是每个像素的视线和立方体内壁的交点的深度。"
                "作业的做法：to_screen 把 clip w <= 0 的顶点输出成 NaN，含 NaN 顶点的三角形在 render_depth 里画不出任何像素。不检查 w：照样做除法再光栅化",
        "相机后方的顶点数（8 个里）的平均": r(rows[:, 0].mean(), 2), "含相机后方顶点的三角形数（12 个里）的平均": r(rows[:, 1].mean(), 2),
        "作业的做法：没有画出来的像素比例（%）的平均": r(100 * rows[:, 2].mean(), 1),
        "作业的做法：没有画出来的像素比例（%）的最小和最大": [r(100 * rows[:, 2].min(), 1), r(100 * rows[:, 2].max(), 1)],
        "作业的做法：画面完整（一个像素都不缺）的次数": int((rows[:, 2] == 0).sum()),
        "作业的做法：画出来的像素与真值的最大深度差": sig(rows[:, 3].max(), 2),
        "不检查 w：画上了但深度不对的像素比例（%）的平均": r(100 * rows[:, 4].mean(), 1),
        "不检查 w：没有画出来的像素比例（%）的平均": r(100 * rows[:, 5].mean(), 1),
    }


# ---------------------------------------------------------------- 表：着色里的两个数
def tab_shading():
    lens = {}
    for ang in (10, 30, 60, 90, 120):
        a = np.radians(ang)
        mid = (np.array([np.sin(-a / 2), np.cos(a / 2)]) + np.array([np.sin(a / 2), np.cos(a / 2)])) / 2
        lens[str(ang)] = r(np.linalg.norm(mid), 4)
    spec = {}
    for sh in (8, 32, 128):
        half = np.degrees(np.arccos(0.5 ** (1 / sh)))
        need = np.log(0.5) / np.log(np.cos(np.radians(half / 2)))
        spec[str(sh)] = {"高光项降到一半时的夹角（度）": r(half, 2), "Blinn-Phong 用同样的指数时，换算成反射方向和视线的夹角（度）": r(2 * half, 2),
                         "要让 Blinn-Phong 和 Phong 一样宽，指数要取多少": r(need, 1), "是 Phong 指数的几倍": r(need / sh, 2)}
    NUM["着色"] = {
        "插值后的法线长度": {"说明": "两个顶点的单位法线夹角为 theta，在两个顶点正中间线性插值得到的向量的长度（等于 cos(theta/2)）；不重新归一化，n·l 就被缩小这么多倍", "按夹角（度）": lens},
        "Phong 和 Blinn-Phong 的高光宽度": {"说明": "Phong 的高光项是 (r·v)^s，r 是光线的反射方向，v 是视线方向；Blinn-Phong 是 (n·h)^s，h 是光线方向和视线方向的半程向量。"
                                              "光线、法线、视线在同一个平面内时，n 和 h 的夹角是 r 和 v 的夹角的一半。这里算的是高光项降到 0.5 时的夹角，纯公式计算", "按指数 s": spec},
    }


# ---------------------------------------------------------------- 表：演示页上能读到的数
def demo_render(spin=30.0, az=35.0, el=25.0, dist=6.0, fov=50.0, near=0.5, far=20.0, noz=False):
    """按 demos/渲染管线.html 的做法重算一遍：立方体绕 y 轴自转，相机在球面上看向原点；逐个三角形调用参考实现，
    深度不在 [0, 1] 内的像素丢掉；深度缓冲是 float32；含相机后方顶点的三角形整个丢掉。返回的各项和页面信息栏上的名字相同。"""
    W, H = 320, 240
    s = mod.to_screen(CUBE_V, mod.perspective(fov, W / H, near, far) @ look(orbit(az, el, dist)) @ R4("y", spin), W, H)
    zbuf = np.full((H, W), np.inf, dtype=np.float32)
    fid = np.full((H, W), -1)
    writes = dropped = zdrop = 0
    for i, f in enumerate(CUBE_F):
        tri = s[f]
        if np.isnan(tri).any():
            dropped += 1
            continue
        layer = mod.render_depth(tri[None], W, H)
        inside = np.isfinite(layer)
        ok = inside & (layer >= 0) & (layer <= 1)
        zdrop += int((inside & ~ok).sum())
        if not noz:
            ok &= layer < zbuf.astype(np.float64)
        zbuf[ok] = layer[ok]
        fid[ok] = i
        writes += int(ok.sum())
    cov = fid >= 0
    gray = np.rint(255 * (1 - zbuf[cov].astype(np.float64))) if cov.any() else np.array([np.nan])
    edges = np.array([np.linalg.norm(s[i, :2] - s[j, :2]) for i, j in ((0, 3), (1, 2), (4, 7), (5, 6))])
    return {"上色的像素": int(cov.sum()), "写入次数": writes, "含相机后方顶点、被整个丢掉的三角形": dropped, "深度不在 [0, 1] 被丢掉的片元": zdrop,
            "可见深度范围": [float(f"{zbuf[cov].min():.4f}"), float(f"{zbuf[cov].max():.4f}")] if cov.any() else None,
            "深度图的灰度范围（0 到 255）": [int(gray.min()), int(gray.max())] if cov.any() else None,
            "四条竖直棱在屏幕上的长度，最长比最短": r(edges.max() / edges.min(), 2) if np.isfinite(edges).all() else None}, fid


def tab_demo():
    base, fid0 = demo_render()
    noz, fid1 = demo_render(noz=True)
    out = {"默认值": base, "勾选“关闭深度测试”": noz}
    out["勾选“关闭深度测试”"]["和默认值相比颜色变了的像素数"] = int(np.sum((fid0 // 2) != (fid1 // 2)))
    for name, kw in (("near 0.05，far 30", dict(near=0.05, far=30.0)), ("near 3", dict(near=3.0)), ("相机距离 1.5", dict(dist=1.5)), ("相机距离 1", dict(dist=1.0)),
                     ("fovy 120", dict(fov=120.0)), ("fovy 120，相机距离 2.5", dict(fov=120.0, dist=2.5))):
        out[name] = demo_render(**kw)[0]
    NUM["演示页"] = {
        "说明": "按 demos/渲染管线.html 的参数和做法，用参考实现重算页面信息栏上能读到的数（各项的名字和信息栏相同）；画面 320x240，共 76800 个像素。"
                "默认值：自转角 30，方位角 35，仰角 25，相机距离 6，fovy 50，near 0.5，far 20。“上色的像素”一个像素只数一次；“写入次数”是通过深度测试并写入缓冲的次数。"
                "这些数和用 node 直接执行页面脚本的输出逐项核对过",
        "画面的像素总数": 320 * 240,
        "各种设置": out,
    }
    c = out["相机距离 1"]["上色的像素"]
    NUM["演示页"]["相机距离 1 时画上颜色的像素占画面的比例（%）"] = r(100 * c / (320 * 240), 1)
    NUM["演示页"]["默认值下立方体只用了深度范围的百分之几"] = r(100 * (base["可见深度范围"][1] - base["可见深度范围"][0]), 1)
    NUM["演示页"]["near 3 时立方体用了深度范围的百分之几"] = r(100 * (out["near 3"]["可见深度范围"][1] - out["near 3"]["可见深度范围"][0]), 1)
    NUM["演示页"]["关闭深度测试后颜色变了的像素占立方体的比例（%）"] = r(100 * noz["和默认值相比颜色变了的像素数"] / base["上色的像素"], 1)
    NUM["演示页"]["关闭深度测试后写入次数是上色像素数的几倍"] = r(noz["写入次数"] / noz["上色的像素"], 3)
    NUM["演示页"]["fovy 120、相机距离 2.5 时上色的像素是默认值的百分之几"] = r(100 * out["fovy 120，相机距离 2.5"]["上色的像素"] / base["上色的像素"], 0)


# ---------------------------------------------------------------- 表：耗时
def fullframe_render(triangles, W, H):
    """和参考实现同样的算式，但每个三角形都算整幅图的重心坐标，不用包围盒。用来和参考实现（只算包围盒）对比耗时。"""
    depth = np.full((H, W), np.inf)
    ys, xs = np.mgrid[0:H, 0:W]
    px, py = xs + 0.5, ys + 0.5
    for tri in triangles:
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = tri
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue
        w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / area
        w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / area
        w2 = 1.0 - w0 - w1
        z = w0 * z0 + w1 * z1 + w2 * z2
        upd = (w0 >= -1e-12) & (w1 >= -1e-12) & (w2 >= -1e-12) & (z < depth)
        depth[upd] = z[upd]
    return depth


def python_loop_render(triangles, W, H):
    """纯 Python 的逐像素两层循环，不用 NumPy 的数组运算。"""
    depth = [[float("inf")] * W for _ in range(H)]
    for (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) in triangles.tolist():
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue
        for y in range(H):
            py, row = y + 0.5, depth[y]
            for x in range(W):
                px = x + 0.5
                w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / area
                w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / area
                w2 = 1.0 - w0 - w1
                if w0 >= -1e-12 and w1 >= -1e-12 and w2 >= -1e-12:
                    z = w0 * z0 + w1 * z1 + w2 * z2
                    if z < row[x]:
                        row[x] = z
    return np.array(depth)


def random_tris(rng, T, W, H, size):
    c = rng.uniform([0, 0], [W, H], (T, 1, 2))
    return np.concatenate([c + rng.uniform(-size, size, (T, 3, 2)), rng.uniform(0.1, 0.9, (T, 3, 1))], axis=2)


def tab_time():
    sizes, counts = [(160, 120), (320, 240), (640, 480)], [12, 100, 1000]
    # 正确性核对每次都做，不属于计时
    rng = np.random.default_rng(2180)
    chk = random_tris(rng, 100, 160, 120, 0.06 * 160)
    ref = mod.render_depth(chk, 160, 120)
    assert np.array_equal(ref, fullframe_render(chk, 160, 120)), "参考实现（包围盒）应与整幅图版本逐像素相同"
    assert np.array_equal(mod.render_depth(chk[:12], 160, 120), python_loop_render(chk[:12], 160, 120)), "纯 Python 循环应与参考实现逐像素相同"
    big = np.array([[[0, 0, 0.5], [320, 0, 0.5], [0, 240, 0.5]], [[320, 0, 0.6], [320, 240, 0.6], [0, 240, 0.6]]] * 50, dtype=float)
    assert np.array_equal(mod.render_depth(big, 320, 240), fullframe_render(big, 320, 240))
    nb = mod.to_screen(CUBE_V, mod.perspective(50.0, 240 / 180, 0.5, 20.0) @ look([3.0, 2.5, 5.0]), 240, 180)[np.array(NB_FACES)]

    def measure():
        two = lambda v: float(f"{v:.2g}")   # noqa: E731

        def best(fn, *a, rep=3):
            ts = []
            for _ in range(rep):
                t0 = time.perf_counter()
                fn(*a)
                ts.append(time.perf_counter() - t0)
            return 1000 * min(ts)

        rng_ = np.random.default_rng(2181)
        cases = {(W, H, T): random_tris(rng_, T, W, H, 0.06 * W) for W, H in sizes for T in counts}
        small = random_tris(rng_, 12, 160, 120, 0.06 * 160)

        def one_round():
            out = {"full": {k: best(fullframe_render, tr, k[0], k[1]) for k, tr in cases.items()},
                   "ref": {k: best(mod.render_depth, tr, k[0], k[1]) for k, tr in cases.items()}}
            out["py"], out["np"] = best(python_loop_render, small, 160, 120, rep=1), best(fullframe_render, small, 160, 120)
            out["big"] = (best(fullframe_render, big, 320, 240), best(mod.render_depth, big, 320, 240))
            out["nb"] = best(mod.render_depth, nb, 240, 180, rep=5)
            return out

        rounds = [one_round() for _ in range(3)]
        med = lambda f: two(float(np.median([f(x) for x in rounds])))                    # noqa: E731
        span = lambda f: [two(min(f(x) for x in rounds)), two(max(f(x) for x in rounds))]   # noqa: E731
        table = lambda key: {f"{W}x{H}": {str(T): med(lambda x: x[key][(W, H, T)]) for T in counts} for W, H in sizes}   # noqa: E731
        return {
            "说明": "在一台 Apple 芯片 Mac 的 CPU 上测，机器上同时还有别的任务，绝对值只看量级。整套测量连做三轮，每轮每项取 3 次里最快的一次（纯 Python 循环每轮只跑 1 次）；"
                    "毫秒数给的是三轮的中位数，倍数给的是三轮里的最小值和最大值。随机三角形：中心在画面内均匀分布，三个顶点离中心不超过画面宽度的 6%。"
                    "“整幅图版本”对每个三角形都算整幅图的重心坐标，参考实现只算三角形包围盒里的像素，两者的输出逐像素相同",
            "整幅图版本（毫秒，三轮的中位数），按画面尺寸和三角形数": table("full"),
            "参考实现，只算包围盒（毫秒，三轮的中位数），按画面尺寸和三角形数": table("ref"),
            "1000 个小三角形时整幅图版本是参考实现的几倍（三轮的最小值和最大值）": {f"{W}x{H}": span(lambda x: x["full"][(W, H, 1000)] / x["ref"][(W, H, 1000)]) for W, H in sizes},
            "整幅图版本：320x240 时 1000 个三角形与 12 个的耗时之比（三轮的最小值和最大值）": span(lambda x: x["full"][(320, 240, 1000)] / x["full"][(320, 240, 12)]),
            "整幅图版本：100 个三角形时 640x480 与 160x120 的耗时之比（三轮的最小值和最大值）": span(lambda x: x["full"][(640, 480, 100)] / x["full"][(160, 120, 100)]),
            "100 个各盖住半个画面的大三角形（320x240）：参考实现是整幅图版本的几倍（三轮的最小值和最大值）": span(lambda x: x["big"][1] / x["big"][0]),
            "160x120、12 个三角形：纯 Python 逐像素循环是整幅图版本的几倍（三轮的最小值和最大值）": span(lambda x: x["py"] / x["np"]),
            "Notebook 的场景（240x180、立方体的 12 个三角形）：参考实现（毫秒，三轮的中位数）": med(lambda x: x["nb"]),
        }

    NUM["耗时"] = S.pinned(CH, "耗时", measure, "CH21_RETIME")
    NUM["耗时的规模"] = {"说明": "整幅图版本的计算量正比于 三角形数 x 画面像素数，下面是这个乘积的比值，用来和实测的耗时之比对照", "1000 个三角形与 12 个三角形之比": r(1000 / 12, 1),
                     "640x480 与 160x120 的像素数之比": (640 * 480) // (160 * 120)}


def main() -> None:
    S.setup()
    for fn in (fig_order, fig_normal, tab_rotation, tab_pipeline, fig_depth, fig_convention, fig_edges, fig_cull, fig_persp, tab_behind, tab_shading, tab_demo, tab_time):
        fn()
        print("完成", fn.__name__, flush=True)
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False))


if __name__ == "__main__":
    main()
