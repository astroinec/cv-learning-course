"""第 11 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch11.py

输出：<主分支>/11_目标检测/figures/*.svg 和 实测数字.json。
IoU 和 NMS 全部调用本章作业的参考实现 pairwise_iou、class_aware_nms；GIoU、DIoU、RoI Align 的对照用 torchvision。
本机没有检测模型和数据集，“检测器的输出”是模拟的：真值框加定位噪声得到主检测，再加重复框和假框。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。耗时用 S.pinned 钉住，设置 CH11_RETIME=1 才重测。
"""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import json
import math
import re
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torchvision
from matplotlib.patches import Rectangle
from torchvision import ops

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

torch.set_num_threads(2)
CH = "11_目标检测"
mod = S.load_solution(CH, "作业_批量IoU与类别NMS.py")
piou, cnms = mod.pairwise_iou, mod.class_aware_nms
NUM: dict = {}
IMG = 640
THS = [round(0.5 + 0.05 * i, 2) for i in range(10)]


def r(x, k=3):
    return round(float(x), k)


def T(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float64).reshape(-1, 4))


# ---------------------------------------------------------------- 工具：全部经过参考实现
def iou_np(a, b):
    """两组框的两两 IoU（参考实现）。"""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    return piou(T(a), T(b)).numpy()


def iou1(a, b):
    return float(iou_np(a, b)[0, 0])


def iou_diag(a, b, chunk=400):
    """a[i] 和 b[i] 一一对应的 IoU：分块调用参考实现后取对角线。"""
    return np.concatenate([np.diagonal(iou_np(a[i:i + chunk], b[i:i + chunk])) for i in range(0, len(a), chunk)])


def nms_np(boxes, scores, labels, thr):
    if len(boxes) == 0:
        return np.zeros(0, int)
    return cnms(T(boxes), torch.as_tensor(np.asarray(scores, float)), torch.as_tensor(np.asarray(labels, int)), thr).numpy()


def place(rng, n, size_fn, max_iou=0.05, tries=200):
    """在 640x640 的画面里随机放 n 个互相基本不重叠的真值框。"""
    out = []
    for _ in range(n):
        for _ in range(tries):
            w, h = size_fn(rng)
            x, y = rng.uniform(0, IMG - w), rng.uniform(0, IMG - h)
            b = np.array([x, y, x + w, y + h])
            if not out or iou_np(b[None], np.array(out)).max() <= max_iou:
                out.append(b)
                break
    return np.array(out).reshape(-1, 4)


def detect(rng, gt, gl, n_cls, miss=0.08, loc=0.05, loc_abs=0.0, dup=1.5, dup_loc=0.12, fp=3.0, confuse=0.05, fp_size=(16, 128)):
    """模拟检测器的输出。每个真值以 1-miss 的概率给一个主检测（四条边各加高斯噪声，标准差 loc*边长+loc_abs，分数 Beta(6,2)）；
    每个主检测再带 Poisson(dup) 个重复框（噪声 dup_loc*边长+loc_abs，分数是主检测的 0.4 到 0.9 倍）；
    每张图另有 Poisson(fp) 个位置随机的假框（分数 Beta(1.5,5)）；主检测以 confuse 的概率标错类别。"""
    B, Sc, L = [], [], []
    for b, c in zip(gt, gl):
        if rng.random() < miss:
            continue
        s = np.array([b[2] - b[0], b[3] - b[1], b[2] - b[0], b[3] - b[1]])
        sc = rng.beta(6, 2)
        lab = c if rng.random() >= confuse else (c + 1 + rng.integers(0, max(n_cls - 1, 1))) % n_cls
        B.append(b + rng.normal(0, 1, 4) * (loc * s + loc_abs)); Sc.append(sc); L.append(lab)
        for _ in range(rng.poisson(dup)):
            B.append(b + rng.normal(0, 1, 4) * (dup_loc * s + loc_abs)); Sc.append(sc * rng.uniform(0.4, 0.9)); L.append(lab)
    for _ in range(rng.poisson(fp)):
        w, h = np.exp(rng.uniform(np.log(fp_size[0]), np.log(fp_size[1]), 2))
        x, y = rng.uniform(0, IMG - w), rng.uniform(0, IMG - h)
        B.append([x, y, x + w, y + h]); Sc.append(rng.beta(1.5, 5)); L.append(rng.integers(0, n_cls))
    return np.array(B, float).reshape(-1, 4), np.array(Sc, float), np.array(L, int)


def flags(dets, gts, thr, n_cls, rule="coco"):
    """逐图逐类按分数从高到低贪心匹配。coco：在还没被占用的真值里找 IoU 最大的；voc：先找 IoU 最大的真值，它已被占用就算误报。"""
    out = {c: ([], []) for c in range(n_cls)}
    ngt = {c: 0 for c in range(n_cls)}
    for (b, s, l), (gb, gl) in zip(dets, gts):
        for c in range(n_cls):
            g = gb[gl == c]
            ngt[c] += len(g)
            m = l == c
            if not m.any():
                continue
            bb, ss = b[m], s[m]
            order = np.argsort(-ss, kind="stable")
            iou = iou_np(bb[order], g)
            used = np.zeros(len(g), bool)
            for i in range(len(order)):
                tp = False
                if len(g):
                    row = iou[i].copy()
                    if rule == "coco":
                        row[used] = -1
                    j = int(row.argmax())
                    if row[j] >= thr and not used[j]:
                        used[j] = True
                        tp = True
                out[c][0].append(ss[order[i]]); out[c][1].append(tp)
    return out, ngt


def pr_curve(scores, tp, ngt):
    order = np.argsort(-np.asarray(scores), kind="stable")
    tp = np.asarray(tp, float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    return ctp / max(ngt, 1), ctp / np.maximum(ctp + cfp, 1e-12)


def ap_of(rec, prec, how="全点"):
    """同一条 PR 曲线的四种面积算法：不插值、全点插值（VOC2010 起）、11 点（VOC2007）、101 点（COCO）。"""
    if len(rec) == 0:
        return 0.0
    if how == "不插值":
        return float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[0], prec, [0]])
    env = np.maximum.accumulate(mpre[::-1])[::-1]
    if how == "全点":
        return float(np.sum(np.diff(mrec) * env[1:]))
    pts = np.linspace(0, 1, 11 if how == "11 点" else 101)
    idx = np.minimum(np.searchsorted(mrec, pts, side="left"), len(env) - 1)
    return float(np.mean(env[idx] * (pts <= rec.max())))


def mean_ap(dets, gts, thr, n_cls, how="全点", rule="coco"):
    f, ngt = flags(dets, gts, thr, n_cls, rule)
    return float(np.mean([ap_of(*pr_curve(f[c][0], f[c][1], ngt[c]), how) for c in range(n_cls)]))


def run_nms(dets, thr, conf=0.0, agnostic=False):
    out = []
    for b, s, l in dets:
        m = s >= conf
        b, s, l = b[m], s[m], l[m]
        k = nms_np(b, s, np.zeros_like(l) if agnostic else l, thr)
        out.append((b[k], s[k], l[k]))
    return out


def prf(dets, gts, thr, n_cls, conf):
    """固定置信度阈值下的精确率、召回率、TP、FP、真值总数。"""
    d = [(b[s >= conf], s[s >= conf], l[s >= conf]) for b, s, l in dets]
    f, ngt = flags(d, gts, thr, n_cls)
    tp = sum(sum(f[c][1]) for c in f)
    n = sum(len(f[c][1]) for c in f)
    return tp / max(n, 1), tp / sum(ngt.values()), int(tp), int(n - tp), int(sum(ngt.values()))


def draw_boxes(ax, boxes, color, lw=1.4, ls="-", fill=None, alpha=1.0, z=2):
    for b in np.asarray(boxes, float).reshape(-1, 4):
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=fill is not None, facecolor=fill or "none", edgecolor=color, lw=lw, ls=ls, alpha=alpha, zorder=z))


# ---------------------------------------------------------------- 图 1：坐标格式弄混
def c2x(b):
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


def t2x(b):
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], 1)


def confusions(xyxy):
    """同一批框的数字被当成另一种格式来读，返回按 xyxy 解释后的结果。"""
    w, h = xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]
    cxcywh = np.stack([xyxy[:, 0] + w / 2, xyxy[:, 1] + h / 2, w, h], 1)
    xywh = np.stack([xyxy[:, 0], xyxy[:, 1], w, h], 1)
    return {
        "中心宽高 当成 左上右下": cxcywh,
        "左上宽高 当成 左上右下": xywh,
        "左上右下 当成 中心宽高": c2x(xyxy),
        "左上右下 当成 左上宽高": t2x(xyxy),
        "左上宽高 当成 中心宽高": c2x(xywh),
        "中心宽高 当成 左上宽高": t2x(cxcywh),
        "归一化的中心宽高 忘了乘图像宽高": c2x(cxcywh / IMG),
        "x 和 y 对调": xyxy[:, [1, 0, 3, 2]],
    }


def fig_formats():
    rng = np.random.default_rng(3)
    n = 5000
    w, h = rng.uniform(8, 200, n), rng.uniform(8, 200, n)
    x1, y1 = rng.uniform(0, IMG - w), rng.uniform(0, IMG - h)
    xyxy = np.stack([x1, y1, x1 + w, y1 + h], 1)
    res = {}
    for name, wrong in confusions(xyxy).items():
        iou = iou_diag(xyxy, wrong)
        deg = (wrong[:, 2] <= wrong[:, 0]) | (wrong[:, 3] <= wrong[:, 1])
        outside = (wrong[:, 0] < 0) | (wrong[:, 1] < 0) | (wrong[:, 2] > IMG) | (wrong[:, 3] > IMG)
        res[name] = {"平均 IoU": r(iou.mean()), "IoU 最小值": r(iou.min()), "IoU 最大值": r(iou.max()), "IoU 不低于 0.5 的比例（%）": r(100 * (iou >= 0.5).mean(), 1),
                     "IoU 等于 0 的比例（%）": r(100 * (iou == 0).mean(), 1), "右下角不在左上角右下方的比例（%）": r(100 * deg.mean(), 1), "超出画面的比例（%）": r(100 * outside.mean(), 1)}
    NUM["坐标格式弄混"] = {
        "说明": "640x640 的画面里 5000 个随机框，宽和高各自在 8 到 200 px 之间均匀取，位置随机且整个框在画面内；把同一组数字按另一种格式解释，再和正确的框用参考实现算 IoU。"
              "左上右下 = (x1,y1,x2,y2)；中心宽高 = (cx,cy,w,h)；左上宽高 = (x,y,w,h)",
        "结果": res, "七分之一": r(1 / 7, 4)}

    demo = np.array([[60, 70, 220, 190], [300, 60, 420, 300], [470, 330, 600, 420], [90, 300, 250, 520], [330, 400, 400, 470], [480, 80, 590, 180]], float)
    wrongs = confusions(demo)
    fig, ax = plt.subplots(1, 4, figsize=(14.5, 3.9), gridspec_kw={"width_ratios": [1, 1, 1, 1.45]}, layout="constrained")
    shows = [("左上宽高 当成 中心宽高", "左上宽高 当成 中心宽高：\n每个框偏半个框"), ("左上右下 当成 中心宽高", "左上右下 当成 中心宽高：\n框变大，中心落在原来的左上角"),
             ("中心宽高 当成 左上右下", "中心宽高 当成 左上右下：\n右下角跑到左上角的左上方")]
    for a_, (key, title) in zip(ax[:3], shows):
        draw_boxes(a_, demo, S.MUTED, lw=1.2, fill="#eef0f3")
        wb = wrongs[key]
        for b in wb:
            if b[2] > b[0] and b[3] > b[1]:
                draw_boxes(a_, b, S.RED, lw=1.5)
            else:
                a_.plot([b[0], b[2]], [b[1], b[3]], "--", color=S.RED, lw=1.3)
                a_.plot([b[0]], [b[1]], "o", color=S.RED, ms=4)
                a_.plot([b[2]], [b[3]], "x", color=S.RED, ms=6)
        lo, hi = (-170, 810) if key == "左上右下 当成 中心宽高" else (-60, IMG + 20)
        a_.set_xlim(lo, hi)
        a_.set_ylim(hi, lo)
        a_.set_aspect("equal")
        a_.add_patch(Rectangle((0, 0), IMG, IMG, fill=False, edgecolor="#b9bec7", lw=0.8, ls=":"))
        a_.set_title(title, fontsize=10.5)
        S.image_axes(a_)
    names = list(res)
    means = [res[k]["平均 IoU"] for k in names]
    y = np.arange(len(names))[::-1]
    ax[3].barh(y, means, color=S.BLUE, height=0.62)
    for yi, k, m in zip(y, names, means):
        ax[3].text(m + 0.006, yi, f"{m:.3f}（IoU=0 占 {res[k]['IoU 等于 0 的比例（%）']:.0f}%）", va="center", fontsize=8.5, color=S.INK)
    ax[3].set_yticks(y)
    ax[3].set_yticklabels(names, fontsize=8.5)
    ax[3].set_xlim(0, 0.33)
    ax[3].set_xlabel("和正确的框的平均 IoU")
    ax[3].set_title("8 种弄混，平均 IoU 都不到 0.15")
    S.clean(ax[3])
    S.save(fig, CH, "fig01_坐标格式弄混")


# ---------------------------------------------------------------- 表：RoI Pooling 的取整
def table_roi():
    stride, wf = 16, 40
    xs = (torch.arange(wf, dtype=torch.float32) + 0.5) * stride
    feat = xs[None, None, None, :].repeat(1, 1, wf, 1)
    rows = {}
    for off in (0, 4, 8, 12, 16, 20):
        box = torch.tensor([[0, 200.0 + off, 200.0, 232.0 + off, 232.0]], dtype=torch.float32)
        rows[str(off)] = {"RoI 的真实中心 x（px）": 216.0 + off,
                          "roi_align，aligned=True": r(ops.roi_align(feat, box, 1, spatial_scale=1 / stride, sampling_ratio=4, aligned=True).item(), 2),
                          "roi_align，aligned=False（默认值）": r(ops.roi_align(feat, box, 1, spatial_scale=1 / stride, sampling_ratio=4, aligned=False).item(), 2),
                          "roi_pool（区域内最大值）": r(ops.roi_pool(feat, box, 1, spatial_scale=1 / stride).item(), 2)}
    NUM["RoI 取整"] = {
        "说明": f"步长 16 的特征图，每个格子的值等于它的中心在原图上的 x 坐标（一个线性的斜坡）；一个 32x32 px 的 RoI 从 x=200 开始每次右移 4 px，输出尺寸 1x1；torchvision {torchvision.__version__}。"
              "RoI Align 在斜坡上的输出应该等于 RoI 中心的 x 坐标",
        "按右移量（px）": rows, "aligned=False 的固定偏差（px）": r(rows["0"]["roi_align，aligned=False（默认值）"] - 216.0, 2), "半个格子（px）": stride / 2,
        "roi_align 的 aligned 参数在这个版本里的默认值": str(inspect.signature(ops.roi_align).parameters["aligned"].default)}


# ---------------------------------------------------------------- 图 2：小目标的 IoU
def shift_needed(s, target, diag):
    """边长 s 的框平移多少 px 时 IoU 掉到 target（对参考实现做二分）。"""
    lo, hi = 0.0, float(s)
    for _ in range(50):
        mid = (lo + hi) / 2
        v = iou1([0, 0, s, s], [mid, mid if diag else 0, s + mid, s + (mid if diag else 0)])
        lo, hi = (mid, hi) if v > target else (lo, mid)
    return (lo + hi) / 2


def fig_small():
    sizes = [5, 10, 20, 50, 100, 200]
    shifts = [0.5, 1, 2, 3, 5]
    one = {str(s): {str(d): r(iou1([0, 0, s, s], [d, 0, s + d, s])) for d in shifts} for s in sizes}
    dia = {str(s): {str(d): r(iou1([0, 0, s, s], [d, d, s + d, s + d])) for d in shifts} for s in sizes}
    grow = {str(s): r(iou1([0, 0, s, s], [-1, -1, s + 1, s + 1])) for s in sizes}
    need = {str(s): {f"IoU 掉到 {t}": {"只沿 x": r(shift_needed(s, t, False), 2), "x 和 y 同时": r(shift_needed(s, t, True), 2)} for t in (0.5, 0.75, 0.9)} for s in (10, 20, 50, 100)}
    NUM["偏移与 IoU"] = {
        "说明": "两个同样大小的正方形框，一个相对另一个平移 d px，参考实现算出的 IoU；连续坐标，面积不加 1",
        "只沿 x 平移": one, "x 和 y 同时平移": dia, "位置不变、四条边各向外多 1 px（边长多 2 px）": grow, "IoU 掉到给定值需要平移多少（px）": need,
        "同样偏 2 px 时 IoU 掉了多少": {"10 px 的框，只沿 x": r(1 - one["10"]["2"]), "100 px 的框，只沿 x": r(1 - one["100"]["2"]),
                               "10 px 的框，x 和 y 同时": r(1 - dia["10"]["2"]), "100 px 的框，x 和 y 同时": r(1 - dia["100"]["2"])},
        "讲义 11.4 的手算例子 G=(10,10,50,50)、P=(30,20,70,60)": r(iou1([10, 10, 50, 50], [30, 20, 70, 60]))}

    rng = np.random.default_rng(0)
    msizes = [6, 8, 10, 12, 16, 24, 32, 48, 64, 128]
    mc = {}
    for sig in (0.5, 1.0, 2.0):
        row = {}
        for s in msizes:
            g = np.tile([[100.0, 100.0, 100.0 + s, 100.0 + s]], (20000, 1))
            v = iou_diag(g, g + rng.normal(0, sig, g.shape))
            row[str(s)] = {"平均 IoU": r(v.mean()), "IoU 不低于 0.5 的比例（%）": r(100 * (v >= 0.5).mean(), 1), "IoU 不低于 0.75 的比例（%）": r(100 * (v >= 0.75).mean(), 1)}
        mc[str(sig)] = row
    NUM["定位噪声与 IoU"] = {"说明": "正方形真值框，预测框的四条边各自加独立的高斯噪声（标准差见键名，单位 px），每格 20000 次，参考实现算 IoU", "按噪声标准差（px）再按边长（px）": mc}

    def inc(b):
        b = np.array(b, float)
        b[2:] += 1
        return b
    csizes = [3, 5, 6, 10, 20, 50, 100]
    conv = {}
    for s in csizes:
        a, b = [0, 0, s, s], [2, 0, s + 2, s]
        conv[str(s)] = {"不加 1": r(iou1(a, b)), "加 1": r(iou1(inc(a), inc(b))),
                        "两个完全相同的框：面积加 1、交集不加 1": r(s * s / (2 * (s + 1) ** 2 - s * s)),
                        "两个完全相同的框：交集加 1、面积不加 1": r((s + 1) ** 2 / (2 * s * s - (s + 1) ** 2)),
                        "只有一条边相接的两个框：加 1": r(iou1(inc([0, 0, s, s]), inc([s, 0, 2 * s, s])), 4), "只有一条边相接的两个框：不加 1": r(iou1([0, 0, s, s], [s, 0, 2 * s, s]), 4)}
    rng = np.random.default_rng(7)
    s_int = rng.integers(6, 17, 20000)
    g = np.stack([np.full(20000, 100), np.full(20000, 100), 100 + s_int, 100 + s_int], 1).astype(float)
    p = g + np.rint(rng.normal(0, 1.5, g.shape))
    v0, v1 = iou_diag(g, p), iou_diag(g + [0, 0, 1, 1], p + [0, 0, 1, 1])
    NUM["端点约定"] = {
        "说明": "坐标数字相同，面积按 x2-x1（不加 1，连续坐标）或 x2-x1+1（加 1，把 x2 当成最后一个像素的下标）来算；“加 1”等价于把右下角各加 1 再交给参考实现。键是 x2-x1 的值（px），第二个框沿 x 平移 2 px",
        "按 x2-x1（px）": conv,
        "小框上判定翻转的比例": {"说明": "x2-x1 在 6 到 16 之间均匀取整数，预测框四条边各加标准差 1.5 px 的噪声后取整，20000 个；看 IoU 是否不低于 0.5",
                       "不加 1 时达标（%）": r(100 * (v0 >= 0.5).mean(), 1), "加 1 时达标（%）": r(100 * (v1 >= 0.5).mean(), 1), "两种约定判定不同的比例（%）": r(100 * ((v0 >= 0.5) != (v1 >= 0.5)).mean(), 1)}}

    eps = {}
    for width in (640, 1280, 4096):
        eps[str(width)] = {str(side): r((side / width) ** 2 / ((side / width) ** 2 + 1e-7), 4) for side in (2, 4, 8, 16, 32)}
    b32 = torch.tensor([[0.5, 0.5, 0.5 + 4 / 4096, 0.5 + 4 / 4096]], dtype=torch.float32)
    NUM["分母加 1e-7"] = {"说明": "两个完全相同的框按 inter/(union+1e-7) 算出的 IoU（应该是 1）；坐标归一化到 0 到 1，键是图像宽度（px）和框的边长（px）",
                      "按图像宽度再按框的边长": eps, "4096 宽的图像里 4 px 的框归一化之后的面积": float(f"{(4 / 4096) ** 2:.1e}"), "坐标用像素时 4 px 的框：算出的 IoU 比 1 小多少": float(f"{1 - 16 / (16 + 1e-7):.1e}"),
                      "参考实现（并集大于 0 才相除，不加 eps）在 4096 宽图像里 4 px 的框上（float32）": r(piou(b32, b32)[0, 0].item(), 6)}

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.9), gridspec_kw={"width_ratios": [1.05, 1.2, 1.2, 1.2]}, layout="constrained")
    a0 = ax[0]
    draw_boxes(a0, [0, 0, 100, 100], S.MUTED, lw=1.3, fill="#eef0f3")
    draw_boxes(a0, [2, 2, 102, 102], S.BLUE, lw=1.3)
    draw_boxes(a0, [120, 45, 130, 55], S.MUTED, lw=1.3, fill="#eef0f3")
    draw_boxes(a0, [122, 47, 132, 57], S.RED, lw=1.3)
    a0.text(50, 50, f"100 px 的框\nIoU {dia['100']['2']:.2f}", ha="center", va="center", fontsize=10, color=S.BLUE)
    a0.text(126, 70, f"10 px 的框\nIoU {dia['10']['2']:.2f}", ha="center", va="top", fontsize=10, color=S.RED)
    a0.set_xlim(-6, 150); a0.set_ylim(112, -8); a0.set_aspect("equal")
    a0.set_title("同样向右下偏 2 px")
    S.image_axes(a0)
    dd = np.linspace(0, 5, 41)
    for s, c in zip((10, 20, 50, 100), (S.RED, S.ORANGE, S.GREEN, S.BLUE)):
        ax[1].plot(dd, [iou1([0, 0, s, s], [d, d, s + d, s + d]) for d in dd], "-", color=c, lw=1.8, label=f"{s} px，x 和 y 同时偏")
        ax[1].plot(dd, [iou1([0, 0, s, s], [d, 0, s + d, s]) for d in dd], "--", color=c, lw=1.1, label=f"{s} px，只沿 x 偏")
    ax[1].axhline(0.5, color=S.MUTED, lw=0.9, ls=":")
    ax[1].text(0.05, 0.51, "0.5", color=S.MUTED, fontsize=8.5)
    ax[1].set_xlabel("偏移（px）"); ax[1].set_ylabel("IoU"); ax[1].set_ylim(0, 1.02)
    ax[1].set_title(f"10 px 的框偏 2 px：{dia['10']['2']:.2f}，已低于 0.5")
    ax[1].legend(fontsize=7.8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[1])
    for sig, c in ((1.0, S.BLUE), (2.0, S.RED)):
        ax[2].plot(msizes, [mc[str(sig)][str(s)]["IoU 不低于 0.5 的比例（%）"] for s in msizes], "o-", color=c, lw=1.8, ms=4, label=f"噪声 {sig:g} px：IoU 不低于 0.5")
        ax[2].plot(msizes, [mc[str(sig)][str(s)]["IoU 不低于 0.75 的比例（%）"] for s in msizes], "s--", color=c, lw=1.2, ms=3.5, label=f"噪声 {sig:g} px：IoU 不低于 0.75")
    ax[2].set_xscale("log"); ax[2].set_xticks([6, 12, 24, 48, 128]); ax[2].set_xticklabels(["6", "12", "24", "48", "128"]); ax[2].minorticks_off()
    ax[2].set_xlabel("框的边长（px）"); ax[2].set_ylabel("达标的比例（%）"); ax[2].set_ylim(-3, 103)
    ax[2].set_title("四条边各有噪声时，小框先不达标")
    ax[2].legend(fontsize=7.8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax[2])
    ax[3].plot(csizes, [conv[str(s)]["不加 1"] for s in csizes], "o-", color=S.BLUE, lw=1.8, ms=4, label="偏 2 px：不加 1")
    ax[3].plot(csizes, [conv[str(s)]["加 1"] for s in csizes], "s-", color=S.ORANGE, lw=1.8, ms=4, label="偏 2 px：加 1")
    ax[3].plot(csizes, [conv[str(s)]["两个完全相同的框：面积加 1、交集不加 1"] for s in csizes], "^--", color=S.RED, lw=1.3, ms=4, label="同一个框和自己：面积加 1、交集不加 1")
    ax[3].set_xscale("log"); ax[3].set_xticks([3, 6, 10, 20, 50, 100]); ax[3].set_xticklabels(["3", "6", "10", "20", "50", "100"]); ax[3].minorticks_off()
    ax[3].set_xlabel("x2 - x1（px）"); ax[3].set_ylabel("IoU"); ax[3].set_ylim(0, 1.02)
    ax[3].set_title("加不加 1，只在小框上差得多")
    ax[3].legend(fontsize=7.8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax[3])
    S.save(fig, CH, "fig02_小目标的IoU")


# ---------------------------------------------------------------- 图 3：IoU、GIoU、DIoU
def fig_losses():
    fns = {"IoU": piou, "GIoU": ops.generalized_box_iou, "DIoU": ops.distance_box_iou}
    cols = {"IoU": S.BLUE, "GIoU": S.ORANGE, "DIoU": S.GREEN}

    def val_grad(fn, p, g):
        p = T(p).clone().requires_grad_(True)
        v = fn(p, T(g))[0, 0]
        (1 - v).backward()
        return float(v.detach()), float(p.grad[0, 0] + p.grad[0, 2])

    offs = list(range(0, 101, 2))
    sep = {k: [val_grad(fn, [100 + o, 100, 140 + o, 140], [100, 100, 140, 140]) for o in offs] for k, fn in fns.items()}
    cen = list(range(0, 41, 2))
    inside = {k: [val_grad(fn, [40 + c, 40, 60 + c, 60], [0, 0, 100, 100]) for c in cen] for k, fn in fns.items()}
    pick = lambda d, xs, x: d[xs.index(x)]
    NUM["IoU 与 GIoU、DIoU"] = {
        "说明": f"值由参考实现（IoU）和 torchvision {torchvision.__version__} 的 generalized_box_iou、distance_box_iou 算出；梯度是损失 1-指标 对预测框整体沿 x 平移的导数，用自动求导得到，单位是每像素",
        "两个 40x40 的框，预测框向右偏": {str(o): {k: {"值": r(pick(sep[k], offs, o)[0]), "对平移的梯度": r(pick(sep[k], offs, o)[1], 4)} for k in fns} for o in (0, 10, 20, 30, 40, 50, 80)},
        "20x20 的预测框完全落在 100x100 的真值框里，中心向右偏": {str(c): {k: {"值": r(pick(inside[k], cen, c)[0]), "对平移的梯度": r(pick(inside[k], cen, c)[1], 4)} for k in fns} for c in (0, 10, 20, 30, 40)}}

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.7), layout="constrained")
    for k in fns:
        ax[0].plot(offs, [v for v, _ in sep[k]], "-", color=cols[k], lw=1.8, label=k)
        ax[1].plot(offs[1:], [abs(g) for _, g in sep[k]][1:], "-", color=cols[k], lw=1.8, label=k)
        ax[2].plot(cen, [v for v, _ in inside[k]], "-" if k != "GIoU" else "--", color=cols[k], lw=1.8, label=k)
    for a_ in ax[:2]:
        a_.axvline(40, color=S.MUTED, lw=0.9, ls=":")
        a_.set_xlabel("预测框向右偏移（px），框宽 40 px")
    ax[0].text(42, 0.86, "从这里起\n不再相交", fontsize=8.5, color=S.MUTED)
    ax[0].set_ylabel("指标的值"); ax[0].set_title("不相交之后 IoU 恒为 0，GIoU、DIoU 继续下降")
    ax[1].set_ylabel("损失对平移的梯度大小（每像素）"); ax[1].set_title("不相交时 IoU 的梯度是 0")
    ax[2].set_xlabel("小框中心相对大框中心的偏移（px）"); ax[2].set_ylabel("指标的值"); ax[2].set_ylim(-0.06, 0.08)
    ax[2].set_title("小框在大框内移动：IoU、GIoU 都不变")
    for a_ in ax:
        a_.legend(fontsize=9)
        S.clean(a_)
    S.save(fig, CH, "fig03_IoU与GIoU和DIoU")


# ---------------------------------------------------------------- 图 4：置信度阈值与 AP 的算法
def baseline():
    rng = np.random.default_rng(11)
    gts, raw = [], []
    for _ in range(200):
        gb = place(rng, int(rng.integers(6, 11)), lambda q: np.exp(q.uniform(np.log(16), np.log(128), 2)))
        gl = rng.integers(0, 3, len(gb))
        gts.append((gb, gl))
        raw.append(detect(rng, gb, gl, 3))
    return gts, raw


def fig_pr_ap():
    nc = 3
    gts, raw = baseline()
    post = run_nms(raw, 0.5)
    n_gt, n_raw, n_post = sum(len(g[0]) for g in gts), sum(len(d[0]) for d in raw), sum(len(d[0]) for d in post)
    NUM["模拟检测器"] = {
        "说明": "本机没有检测模型，检测器的输出是模拟的。200 张 640x640 的图，每张 6 到 10 个互相基本不重叠的真值框（宽高各自在 16 到 128 px 之间按对数均匀取），3 个类别。"
              "每个真值以 92% 的概率得到一个主检测：四条边各加标准差为边长 5% 的高斯噪声，分数服从 Beta(6,2)，5% 的主检测类别标错；"
              "每个主检测带 Poisson(1.5) 个重复框（噪声为边长的 12%，分数是主检测的 0.4 到 0.9 倍）；每张图另有 Poisson(3) 个位置随机的假框，分数服从 Beta(1.5,5)。随机种子 11",
        "真值框总数": n_gt, "NMS 之前的检测框总数": n_raw, "按类别做 NMS（阈值 0.5）之后的检测框总数": n_post}
    confs = [0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    by_conf = {}
    for c in confs:
        p, rc, tp, fp, _ = prf(post, gts, 0.5, nc, c)
        by_conf[str(c)] = {"精确率（%）": r(100 * p, 1), "召回率（%）": r(100 * rc, 1), "F1（%）": r(200 * p * rc / (p + rc), 1), "每张图的误报数": r(fp / 200, 2)}
    NUM["置信度阈值"] = {"说明": "模拟检测器的输出先按类别做 NMS（阈值 0.5），再只保留分数不低于置信度阈值的框；匹配的 IoU 阈值 0.5，一个真值只能被匹配一次", "按置信度阈值": by_conf}

    hows = ["不插值", "全点", "11 点", "101 点"]
    f, ngt = flags(post, gts, 0.5, nc)
    rec0, pre0 = pr_curve(f[0][0], f[0][1], ngt[0])
    per_thr = [mean_ap(post, gts, t, nc, "101 点") for t in THS]
    cut = {str(c): r(100 * mean_ap(run_nms(raw, 0.5, c), gts, 0.5, nc), 1) for c in (0.001, 0.1, 0.25, 0.5)}
    NUM["AP 的算法"] = {
        "说明": "同一份 NMS 之后的检测结果（见“模拟检测器”），只改变 AP 的算法；mAP 是 3 个类别的平均，单位 %",
        "IoU 阈值 0.5 时的 mAP，四种面积算法": {h: r(100 * mean_ap(post, gts, 0.5, nc, h), 1) for h in hows},
        "类别 0 的 AP，四种面积算法": {h: r(100 * ap_of(rec0, pre0, h), 1) for h in hows},
        "类别 0 能达到的最高召回率（%）": r(100 * rec0.max(), 1),
        "不同 IoU 阈值下的 mAP（101 点）": {str(t): r(100 * a, 1) for t, a in zip(THS, per_thr)},
        "IoU 阈值从 0.5 到 0.95 十档的平均（101 点）": r(100 * np.mean(per_thr), 1),
        "匹配规则换成 VOC 式（IoU 最大的真值已被占用就算误报），IoU 0.5，全点": r(100 * mean_ap(post, gts, 0.5, nc, "全点", "voc"), 1),
        "不做 NMS 直接算，IoU 0.5，全点": r(100 * mean_ap(raw, gts, 0.5, nc), 1),
        "先按置信度截断再做 NMS、再算 mAP（IoU 0.5，全点）": cut}
    A = NUM["AP 的算法"]
    A["全点比 11 点高多少（mAP）"] = r(A["IoU 阈值 0.5 时的 mAP，四种面积算法"]["全点"] - A["IoU 阈值 0.5 时的 mAP，四种面积算法"]["11 点"], 1)
    A["置信度 0.5 截断比不截断低多少（mAP）"] = r(cut["0.001"] - cut["0.5"], 1)
    A["mAP@0.5 比 mAP@0.5:0.95 高多少"] = r(A["不同 IoU 阈值下的 mAP（101 点）"]["0.5"] - A["IoU 阈值从 0.5 到 0.95 十档的平均（101 点）"], 1)

    sizes = [8, 12, 16, 24, 32, 64, 128]
    by_size, curves = {}, {}
    for sig in (1.0, 2.0):
        row = {}
        for s in sizes:
            rng = np.random.default_rng(1000 + s)
            g2, d2 = [], []
            for _ in range(150):
                gb = place(rng, 8, lambda q: (s, s))
                g2.append((gb, np.zeros(len(gb), int)))
                d2.append(detect(rng, gb, g2[-1][1], 1, miss=0.05, loc=0, loc_abs=sig, dup=0, fp=2.0, confuse=0, fp_size=(s, s * 1.001)))
            a = [mean_ap(d2, g2, t, 1, "101 点") for t in THS]
            curves[(sig, s)] = a
            row[str(s)] = {"AP@0.5": r(100 * a[0], 1), "AP@0.75": r(100 * a[5], 1), "AP@0.5:0.95": r(100 * np.mean(a), 1)}
        by_size[str(sig)] = row
    NUM["按目标尺寸"] = {"说明": "单一类别，每张图 8 个同样大小的正方形目标，150 张图；每个目标以 95% 的概率得到一个检测框，四条边各加给定标准差（px）的高斯噪声，没有重复框，每张图另有 Poisson(2) 个同样大小的假框；AP 用 101 点，单位 %",
                    "按噪声标准差（px）再按边长（px）": by_size}

    rng = np.random.default_rng(112)
    g3, d3 = [], []
    for _ in range(150):
        gb = place(rng, 8, lambda q: (12, 12))
        g3.append((gb, np.zeros(len(gb), int)))
        d3.append(detect(rng, gb, g3[-1][1], 1, miss=0.05, loc=0, loc_abs=1.5, dup=0, fp=2.0, confuse=0, fp_size=(12, 12.01)))
    by_iou = {}
    for t in (0.1, 0.3, 0.5, 0.75, 0.9):
        p, rc, tp, fp, _ = prf(d3, g3, t, 1, 0.3)
        by_iou[str(t)] = {"召回率（%）": r(100 * rc, 1), "精确率（%）": r(100 * p, 1), "每张图的误报数": r(fp / 150, 2)}

    def center_match(radius):
        tp = n = ng = 0
        for (b, s, _), (gb, _) in zip(d3, g3):
            b = b[s >= 0.3]
            s_ = s[s >= 0.3]
            used = np.zeros(len(gb), bool)
            gc = (gb[:, :2] + gb[:, 2:]) / 2
            ng += len(gb)
            n += len(b)
            for i in np.argsort(-s_, kind="stable"):
                dist = np.hypot(*(gc - (b[i, :2] + b[i, 2:]) / 2).T)
                dist[used] = 1e9
                j = int(dist.argmin())
                if dist[j] <= radius:
                    used[j] = True
                    tp += 1
        return {"召回率（%）": r(100 * tp / ng, 1), "精确率（%）": r(100 * tp / n, 1), "每张图的误报数": r((n - tp) / 150, 2)}
    NUM["匹配规则"] = {"说明": "同一份检测结果：单一类别，每张图 8 个 12x12 px 的目标，150 张图，检测框四条边各有标准差 1.5 px 的噪声，95% 的目标有检测框，每张图 Poisson(2) 个假框；置信度阈值固定 0.3，只改变“怎样算命中”",
                   "按 IoU 阈值": by_iou, "按中心距离（px）": {str(k): center_match(k) for k in (1, 2, 3, 6)}}

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9), layout="constrained")
    cs = np.linspace(0.02, 0.95, 48)
    P, R_ = zip(*[prf(post, gts, 0.5, nc, c)[:2] for c in cs])
    ax[0].plot(cs, 100 * np.array(P), "-", color=S.BLUE, lw=1.8, label="精确率")
    ax[0].plot(cs, 100 * np.array(R_), "-", color=S.ORANGE, lw=1.8, label="召回率")
    ax[0].plot(cs, 200 * np.array(P) * np.array(R_) / (np.array(P) + np.array(R_)), "--", color=S.MUTED, lw=1.3, label="F1")
    ax[0].set_xlabel("置信度阈值"); ax[0].set_ylabel("%"); ax[0].set_ylim(0, 102)
    ax[0].set_title("阈值越高，精确率升、召回率降")
    ax[0].legend(fontsize=9, loc="lower left")
    S.clean(ax[0])
    mrec, mpre = np.concatenate([[0], rec0, [1]]), np.concatenate([[0], pre0, [0]])
    env = np.maximum.accumulate(mpre[::-1])[::-1]
    ax[1].plot(rec0, pre0, "-", color=S.BLUE, lw=1.1, label="原始的 PR 曲线")
    ax[1].step(mrec, env, where="pre", color=S.ORANGE, lw=1.6, label="插值后（取右侧的最大精确率）")
    pts = np.linspace(0, 1, 11)
    ax[1].plot(pts, env[np.minimum(np.searchsorted(mrec, pts, side="left"), len(env) - 1)] * (pts <= rec0.max()), "o", color=S.RED, ms=5, label="11 点插值的采样点")
    ax[1].set_xlabel("召回率"); ax[1].set_ylabel("精确率"); ax[1].set_xlim(0, 1.02); ax[1].set_ylim(0, 1.04)
    a11, aall = NUM["AP 的算法"]["类别 0 的 AP，四种面积算法"]["11 点"], NUM["AP 的算法"]["类别 0 的 AP，四种面积算法"]["全点"]
    ax[1].set_title(f"同一条曲线：11 点 {a11:.1f}，全点 {aall:.1f}")
    ax[1].legend(fontsize=8.5, loc="lower left")
    S.clean(ax[1])
    for s, c in ((12, S.RED), (24, S.ORANGE), (64, S.BLUE)):
        ax[2].plot(THS, [100 * v for v in curves[(1.0, s)]], "o-", color=c, lw=1.8, ms=4,
                   label=f"{s} px：AP@0.5 {by_size['1.0'][str(s)]['AP@0.5']:.0f}，AP@0.5:0.95 {by_size['1.0'][str(s)]['AP@0.5:0.95']:.0f}")
    ax[2].set_xlabel("匹配用的 IoU 阈值"); ax[2].set_ylabel("AP（%）"); ax[2].set_ylim(-3, 103)
    ax[2].set_title("每条边 1 px 噪声：AP@0.5 接近，往后差得远")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax[2])
    S.save(fig, CH, "fig04_置信度阈值与AP的算法")


# ---------------------------------------------------------------- 图 5：NMS 阈值与密集目标
def dense_scene(rng, n_obj, s, nb):
    """一排 n_obj 个边长 s 的同类目标，相邻两个真值框的 IoU 是 nb；每个目标一个主检测加 Poisson(3) 个重复框，没有假框。"""
    step = s * (1 - nb) / (1 + nb)
    x0 = 50 + step * np.arange(n_obj)
    gb = np.stack([x0, np.full(n_obj, 100.0), x0 + s, np.full(n_obj, 100.0 + s)], 1)
    gl = np.zeros(n_obj, int)
    return (gb, gl), detect(rng, gb, gl, 1, miss=0, loc=0.03, dup=3.0, dup_loc=0.08, fp=0, confuse=0)


def soft_nms(b, s, l, sigma=0.5, floor=0.001):
    """高斯 Soft-NMS：不删框，把和当前最高分框同类的框的分数乘上 exp(-IoU^2/sigma)；分数低于 floor 才丢掉。"""
    b, s, l = b.copy(), s.copy(), l.copy()
    kb, ks, kl = [], [], []
    alive = np.ones(len(b), bool)
    while alive.any():
        i = int(np.argmax(np.where(alive, s, -1)))
        kb.append(b[i]); ks.append(s[i]); kl.append(l[i])
        alive[i] = False
        m = alive & (l == l[i])
        s[m] *= np.exp(-iou_np(b[i:i + 1], b)[0][m] ** 2 / sigma)
        alive &= s >= floor
    return np.array(kb).reshape(-1, 4), np.array(ks), np.array(kl, int)


def matched_gt(b, s, gb, thr=0.5):
    used = np.zeros(len(gb), bool)
    iou = iou_np(b, gb)
    for i in np.argsort(-s, kind="stable"):
        row = np.where(used, -1, iou[i])
        if len(row) and row.max() >= thr:
            used[int(row.argmax())] = True
    return used


def fig_dense():
    nms_ths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    nbs = [0.0, 0.2, 0.4, 0.6]
    table, keep_scene, soft = {}, None, {}
    for nb in nbs + [0.7]:
        rng = np.random.default_rng(int(nb * 100) + 5)
        scenes = [dense_scene(rng, 6, 40, nb) for _ in range(300)]
        g, d = [x[0] for x in scenes], [x[1] for x in scenes]
        if nb == 0.4:
            keep_scene = scenes[0]
        if nb in nbs:
            row = {}
            for t in nms_ths:
                p, rc, tp, fp, n = prf(run_nms(d, t), g, 0.5, 1, 0.0)
                row[str(t)] = {"召回率（%）": r(100 * rc, 1), "每个目标多出来的框": r(fp / n, 2), "F1（%）": r(200 * p * rc / (p + rc), 1)}
            table[str(nb)] = {"NMS 之前每个目标的框数": r(sum(len(x[0]) for x in d) / 1800, 2), "按 NMS 阈值": row}
        if nb in (0.0, 0.4, 0.6, 0.7):
            row = {}
            for name, out in (("硬 NMS，阈值 0.5", run_nms(d, 0.5)), ("硬 NMS，阈值 0.7", run_nms(d, 0.7)), ("Soft-NMS（高斯，sigma=0.5）", [soft_nms(*x) for x in d])):
                p, rc, *_ = prf(out, g, 0.5, 1, 0.3)
                row[name] = {"AP@0.5（%）": r(100 * mean_ap(out, g, 0.5, 1), 1), "置信度阈值 0.3 时的召回率（%）": r(100 * rc, 1), "置信度阈值 0.3 时的精确率（%）": r(100 * p, 1)}
            row["硬 NMS 阈值 0.7，匹配规则换成 VOC 式：AP@0.5（%）"] = r(100 * mean_ap(run_nms(d, 0.7), g, 0.5, 1, "全点", "voc"), 1)
            soft[str(nb)] = row
    rng = np.random.default_rng(0)
    s = 40
    base = np.array([0.0, 0, s, s])
    main = base + rng.normal(0, 1, (20000, 4)) * 0.03 * s
    dup_iou = iou_diag(main, base + rng.normal(0, 1, (20000, 4)) * 0.08 * s)
    nb_iou = {}
    for nb in (0.2, 0.4, 0.6, 0.7):
        step = s * (1 - nb) / (1 + nb)
        nb_iou[nb] = iou_diag(main, base + [step, 0, step, 0] + rng.normal(0, 1, (20000, 4)) * 0.03 * s)
    q = lambda v, ps: [r(x) for x in np.percentile(v, ps)]
    NUM["NMS 阈值与密集目标"] = {
        "说明": "单一类别，一排 6 个 40x40 px 的目标，相邻两个真值框的 IoU 见键名；每个目标一个主检测（四条边噪声 3% 边长，分数 Beta(6,2)）加 Poisson(3) 个重复框（噪声 8% 边长，分数是主检测的 0.4 到 0.9 倍），没有漏检和假框；"
              "每种密度 300 组；NMS 之后不做置信度过滤，匹配的 IoU 阈值 0.5。“每个目标多出来的框”= 没有匹配上真值的框数 / 真值数",
        "按相邻真值框的 IoU": table,
        "两种 IoU 的分布（各 20000 次）": {"重复框和自己的主检测框之间：1%、5%、10%、50%、90% 分位": q(dup_iou, [1, 5, 10, 50, 90]),
                                 "相邻两个目标的主检测框之间：5%、50%、95% 分位": {str(k): q(v, [5, 50, 95]) for k, v in nb_iou.items()}}}
    NUM["Soft-NMS"] = {"说明": "和“NMS 阈值与密集目标”相同的数据。Soft-NMS 是脚本里按高斯降分公式写的一个朴素版本，分数低于 0.001 才丢弃；AP 用全点插值", "按相邻真值框的 IoU": soft}

    (gb, _), (b, sc, lb) = keep_scene
    fig = plt.figure(figsize=(14, 6.9), layout="constrained")
    gs = fig.add_gridspec(2, 12, height_ratios=[0.8, 1])
    panels = [("NMS 之前", None), ("阈值 0.3", 0.3), ("阈值 0.5", 0.5), ("阈值 0.8", 0.8)]
    for i, (name, t) in enumerate(panels):
        a_ = fig.add_subplot(gs[0, 3 * i:3 * i + 3])
        k = np.arange(len(b)) if t is None else nms_np(b, sc, lb, t)
        draw_boxes(a_, gb, "#b9bec7", lw=0.8, fill="#eef0f3", alpha=0.6, z=1)
        draw_boxes(a_, b[k], S.BLUE if t is not None else S.MUTED, lw=1.2 if t is not None else 0.8)
        used = matched_gt(b[k], sc[k], gb)
        cx = (gb[:, 0] + gb[:, 2]) / 2
        a_.plot(cx[used], np.full(used.sum(), 152), "o", color=S.GREEN, ms=6)
        a_.plot(cx[~used], np.full((~used).sum(), 152), "x", color=S.RED, ms=8, mew=2)
        a_.set_xlim(gb[0, 0] - 10, gb[-1, 2] + 10); a_.set_ylim(160, 88); a_.set_aspect("equal")
        a_.set_title(f"{name}：{len(k)} 个框" if t is None else f"{name}：剩 {len(k)} 个框，找到 {int(used.sum())}/6", fontsize=10.5)
        S.image_axes(a_)
    a1, a2, a3 = fig.add_subplot(gs[1, :4]), fig.add_subplot(gs[1, 4:8]), fig.add_subplot(gs[1, 8:])
    for nb, c in zip(nbs, (S.GREEN, S.BLUE, S.ORANGE, S.RED)):
        row = table[str(nb)]["按 NMS 阈值"]
        a1.plot(nms_ths, [row[str(t)]["召回率（%）"] for t in nms_ths], "o-", color=c, lw=1.8, ms=4, label=f"相邻目标 IoU {nb:g}")
        a2.plot(nms_ths, [row[str(t)]["每个目标多出来的框"] for t in nms_ths], "o-", color=c, lw=1.8, ms=4, label=f"相邻目标 IoU {nb:g}")
    a1.set_xlabel("NMS 阈值"); a1.set_ylabel("召回率（%）"); a1.set_ylim(0, 104)
    a1.set_title("阈值低于相邻目标的 IoU，真目标被删")
    a1.legend(fontsize=8.5, loc="lower right")
    a2.set_xlabel("NMS 阈值"); a2.set_ylabel("每个目标多出来的框")
    a2.set_title("阈值超过 0.6，重复框开始留下")
    a2.legend(fontsize=8.5, loc="upper left")
    bins = np.linspace(0, 1, 51)
    a3.hist(nb_iou[0.6], bins=bins, color=S.RED, alpha=0.55, label="相邻两个目标的主检测框（真值 IoU 0.6）")
    a3.hist(dup_iou, bins=bins, color=S.BLUE, alpha=0.55, label="重复框和自己的主检测框")
    a3.set_xlabel("两个框的 IoU"); a3.set_ylabel("次数")
    a3.set_title("两种 IoU 的范围重叠时，没有两全的阈值")
    a3.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    for a_ in (a1, a2, a3):
        S.clean(a_)
    S.save(fig, CH, "fig05_NMS阈值与密集目标")


# ---------------------------------------------------------------- 图 6：按类别还是不分类别
def fig_class():
    s = 48

    def pair_scene(rng, u):
        step = s * (1 - u) / (1 + u)
        gb = np.array([[100, 100, 100 + s, 100 + s], [100 + step, 100, 100 + step + s, 100 + s]], float)
        gl = np.array([0, 1])
        return (gb, gl), detect(rng, gb, gl, 2, miss=0, loc=0.03, dup=2.0, dup_loc=0.08, fp=0, confuse=0)

    def confuse_scene(rng, q):
        gb = place(rng, 6, lambda _: (s, s))
        gl = rng.integers(0, 2, len(gb))
        b, sc, l = detect(rng, gb, gl, 2, miss=0, loc=0.03, dup=2.0, dup_loc=0.08, fp=0, confuse=0)
        eb, es, el = [], [], []
        for bb, c in zip(gb, gl):
            if rng.random() < q:
                eb.append(bb + rng.normal(0, 1, 4) * 0.05 * s); es.append(rng.beta(3, 4)); el.append(1 - c)
        if eb:
            b, sc, l = np.vstack([b, eb]), np.concatenate([sc, es]), np.concatenate([l, el])
        return (gb, gl), (b, sc, l)

    us, qs = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.0, 0.1, 0.2, 0.3, 0.5]
    A, B = {}, {}
    for u in us:
        rng = np.random.default_rng(int(u * 100))
        sc_ = [pair_scene(rng, u) for _ in range(500)]
        g, d = [x[0] for x in sc_], [x[1] for x in sc_]
        A[str(u)] = {name: {"召回率（%）": r(100 * prf(run_nms(d, 0.5, agnostic=ag), g, 0.5, 2, 0)[1], 1)} for name, ag in (("按类别", False), ("不分类别", True))}
    for q in qs:
        rng = np.random.default_rng(int(q * 100) + 77)
        sc_ = [confuse_scene(rng, q) for _ in range(300)]
        g, d = [x[0] for x in sc_], [x[1] for x in sc_]
        row = {}
        for name, ag in (("按类别", False), ("不分类别", True)):
            out = run_nms(d, 0.5, agnostic=ag)
            p, rc, tp, fp, _ = prf(out, g, 0.5, 2, 0)
            row[name] = {"精确率（%）": r(100 * p, 1), "召回率（%）": r(100 * rc, 1), "每张图的误报数": r(fp / 300, 2), "mAP@0.5（%）": r(100 * mean_ap(out, g, 0.5, 2), 1)}
        B[str(q)] = row
    NUM["按类别还是不分类别"] = {
        "说明": "NMS 阈值 0.5，匹配的 IoU 阈值 0.5，不做置信度过滤；“不分类别”是把所有框的类别都设成 0 再交给参考实现。目标都是 48x48 px，每个目标一个主检测加 Poisson(2) 个重复框",
        "情形 A：两个不同类别的真目标互相重叠": {"说明": "每组两个目标，类别 0 和类别 1 各一个，键是两个真值框的 IoU，各 500 组", "按两个真值框的 IoU": A},
        "情形 B：同一个目标被报成两个类别": {"说明": "每张图 6 个互不重叠的目标，各 300 张；每个目标以给定概率多出一个类别报错的框（位置噪声 5% 边长，分数 Beta(3,4)，均值比正确的主检测低）；键是这个概率", "按报错概率": B}}

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8), gridspec_kw={"width_ratios": [0.9, 1.1, 1.1]}, layout="constrained")
    a0 = ax[0]
    draw_boxes(a0, [10, 30, 58, 78], S.BLUE, lw=1.8)
    draw_boxes(a0, [22, 30, 70, 78], S.ORANGE, lw=1.8)
    a0.text(40, 18, "情形 A", ha="center", fontsize=10.5, fontweight="bold")
    a0.text(40, 86, "两个真目标，类别不同\n两个框都该留下", ha="center", va="top", fontsize=9, color=S.MUTED)
    draw_boxes(a0, [120, 30, 168, 78], S.BLUE, lw=1.8)
    draw_boxes(a0, [122, 32, 171, 80], S.ORANGE, lw=1.8, ls="--")
    a0.text(145, 18, "情形 B", ha="center", fontsize=10.5, fontweight="bold")
    a0.text(145, 86, "一个目标，被报成两类\n虚线框是误报", ha="center", va="top", fontsize=9, color=S.MUTED)
    a0.set_xlim(-22, 207); a0.set_ylim(118, 0); a0.set_aspect("equal")
    a0.set_title("蓝色是类别 0，橙色是类别 1")
    S.image_axes(a0)
    ax[1].plot(us, [A[str(u)]["按类别"]["召回率（%）"] for u in us], "o-", color=S.BLUE, lw=1.8, ms=4, label="按类别做 NMS")
    ax[1].plot(us, [A[str(u)]["不分类别"]["召回率（%）"] for u in us], "s-", color=S.RED, lw=1.8, ms=4, label="不分类别做 NMS")
    ax[1].axvline(0.5, color=S.MUTED, lw=0.9, ls=":")
    ax[1].set_xlabel("两个真值框的 IoU"); ax[1].set_ylabel("召回率（%）"); ax[1].set_ylim(0, 104)
    ax[1].set_title("情形 A：不分类别时，重叠超过阈值就丢一个")
    ax[1].legend(fontsize=9, loc="lower left")
    for name, c in (("按类别", S.BLUE), ("不分类别", S.RED)):
        ax[2].plot(qs, [B[str(q)][name]["精确率（%）"] for q in qs], "o-", color=c, lw=1.8, ms=4, label=f"{name}：精确率")
        ax[2].plot(qs, [B[str(q)][name]["召回率（%）"] for q in qs], "s--", color=c, lw=1.2, ms=3.5, label=f"{name}：召回率")
    ax[2].set_xlabel("一个目标被多报成另一类的概率"); ax[2].set_ylabel("%"); ax[2].set_ylim(50, 102)
    ax[2].set_title("情形 B：按类别时，报错的框删不掉")
    ax[2].legend(fontsize=8.5, loc="lower left")
    for a_ in ax[1:]:
        S.clean(a_)
    S.save(fig, CH, "fig06_按类别还是不分类别")


# ---------------------------------------------------------------- 表：错误实现、对照、算例、耗时
def bug_no_clamp(b1, b2):
    wh = torch.minimum(b1[:, None, 2:], b2[None, :, 2:]) - torch.maximum(b1[:, None, :2], b2[None, :, :2])
    inter = wh[..., 0] * wh[..., 1]
    area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area(b1)[:, None] + area(b2)[None, :] - inter)


def bug_nan(b1, b2):
    wh = (torch.minimum(b1[:, None, 2:], b2[None, :, 2:]) - torch.maximum(b1[:, None, :2], b2[None, :, :2])).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area = lambda b: (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (area(b1)[:, None] + area(b2)[None, :] - inter)


def bug_sorted_index(boxes, scores, labels, thr):
    order = torch.argsort(scores, descending=True)
    return cnms(boxes[order], scores[order], labels[order], thr)


def bug_not_greedy(boxes, scores, labels, thr):
    order = torch.argsort(scores, descending=True).tolist()
    iou = piou(boxes, boxes)
    keep = [i for n, i in enumerate(order) if not any(labels[j] == labels[i] and iou[i, j] > thr for j in order[:n])]
    return torch.tensor(keep, dtype=torch.long)


def bug_ge(boxes, scores, labels, thr):
    keep, dead = [], torch.zeros(len(boxes), dtype=torch.bool)
    for i in torch.argsort(scores, descending=True).tolist():
        if not dead[i]:
            keep.append(i)
            dead |= (labels == labels[i]) & (piou(boxes[i:i + 1], boxes)[0] >= thr)
    return torch.tensor(keep, dtype=torch.long)


def run_checker(iou_fn, nms_fn):
    """把验收脚本里的 load 换成给定的两个函数，返回“通过 a/b”。"""
    spec = importlib.util.spec_from_file_location("ch11_check", S.MAIN / CH / "tests" / "check_progress.py")
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    chk.load = lambda: types.SimpleNamespace(pairwise_iou=iou_fn, class_aware_nms=nms_fn)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        chk.main()
    return re.findall(r"通过 (\d+/\d+)", buf.getvalue())[-1]


def table_misc():
    rng = np.random.default_rng(1)
    sets = []
    for _ in range(300):
        n = int(rng.integers(2, 60))
        xy, wh = rng.uniform(0, 100, (n, 2)), rng.uniform(5, 60, (n, 2))
        sets.append((T(np.hstack([xy, xy + wh])), torch.as_tensor(rng.random(n)), torch.as_tensor(rng.integers(0, 3, n)), float(rng.choice([0.3, 0.5, 0.7]))))
    same_tv = sum(cnms(b, s, l, t).tolist() == ops.batched_nms(b.float(), s.float(), l, t).tolist() for b, s, l, t in sets)
    max_diff = max(float((piou(b, b) - ops.box_iou(b, b)).abs().max()) for b, *_ in sets)
    eq = T([[0, 0, 2, 2], [0, 0, 2, 1]])
    eq_s, eq_l = torch.tensor([0.9, 0.8], dtype=torch.float64), torch.tensor([0, 0])
    zero = T([[5, 5, 5, 5]])
    NUM["和 torchvision 对照"] = {
        "说明": f"300 组随机输入（每组 2 到 59 个框，位置在 100x100 的范围内，宽高 5 到 60，3 个类别，阈值在 0.3、0.5、0.7 里随机取），参考实现对比 torchvision {torchvision.__version__} 的 batched_nms 和 box_iou",
        "NMS 保留的下标完全相同的组数": int(same_tv), "IoU 矩阵的最大差": float(f"{max_diff:.1e}"),
        "IoU 正好等于阈值": {"说明": "框 (0,0,2,2) 和 (0,0,2,1) 的 IoU 正好是 0.5，阈值也取 0.5", "IoU": r(piou(eq, eq)[0, 1].item(), 4),
                       "参考实现保留": cnms(eq, eq_s, eq_l, 0.5).tolist(), "torchvision.ops.nms 保留": ops.nms(eq.float(), eq_s.float(), 0.5).tolist(), "写成大于等于就抑制时保留": bug_ge(eq, eq_s, eq_l, 0.5).tolist()},
        "面积为 0 的框和它自己": {"参考实现": r(piou(zero, zero)[0, 0].item(), 3), "torchvision.ops.box_iou": "nan" if torch.isnan(ops.box_iou(zero, zero)[0, 0]) else r(ops.box_iou(zero, zero)[0, 0].item(), 3)},
        "只有一条边相接的两个框 (0,0,1,1) 和 (1,0,2,1)": r(iou1([0, 0, 1, 1], [1, 0, 2, 1]), 3)}

    def differ(nms_fn):
        return r(100 * np.mean([nms_fn(b, s, l, t).tolist() != cnms(b, s, l, t).tolist() for b, s, l, t in sets]), 1)
    pairs = T(np.hstack([(xy := np.random.default_rng(2).uniform(0, 100, (4000, 2))), xy + np.random.default_rng(3).uniform(5, 30, (4000, 2))]))
    ref_m, bad_m = piou(pairs[:2000], pairs[2000:]).diagonal(), bug_no_clamp(pairs[:2000], pairs[2000:]).diagonal()
    NUM["常见的错误实现"] = {
        "说明": "把作业里常见的五种写错的实现交给本章的验收脚本 tests/check_progress.py（共 2 项），并在上面那 300 组随机输入上和参考实现比较。这些错误实现只存在于配图脚本里",
        "交集的宽高没有截断到 0": {"验收": run_checker(bug_no_clamp, cnms), "2000 对随机框里本来不相交却算出正 IoU 的比例（%）": r(100 * ((ref_m == 0) & (bad_m > 0)).double().mean().item(), 1),
                         "不相交的框对里算出的 IoU 最大值": r(bad_m[ref_m == 0].max().item(), 3)},
        "并集为 0 时直接相除": {"验收": run_checker(bug_nan, cnms), "面积为 0 的框和它自己的 IoU": "nan" if torch.isnan(bug_nan(zero, zero)[0, 0]) else "不是 nan"},
        "返回的是排序之后的位置，不是原始下标": {"验收": run_checker(piou, bug_sorted_index), "300 组随机输入里结果和参考实现不同的比例（%）": differ(bug_sorted_index)},
        "已经被删掉的框继续去删别的框": {"验收": run_checker(piou, bug_not_greedy), "300 组随机输入里结果和参考实现不同的比例（%）": differ(bug_not_greedy)},
        "IoU 大于等于阈值就抑制": {"验收": run_checker(piou, bug_ge), "300 组随机输入里结果和参考实现不同的比例（%）": differ(bug_ge)},
        "参考实现": {"验收": run_checker(piou, cnms)}}

    chain = np.array([[0, 0, 10, 10], [4, 0, 14, 10], [8, 0, 18, 10]], float)
    ci = iou_np(chain, chain)
    nb_boxes = np.array([[1, 1, 6, 6], [1.5, 1.4, 6.2, 6.1], [5, 2, 9, 6], [1, 7, 4, 9]], float)
    nb_scores = np.array([0.95, 0.82, 0.75, 0.60])
    ni = iou_np(nb_boxes, nb_boxes)
    NUM["链式重叠"] = {"说明": "三个同类的框 (0,0,10,10)、(4,0,14,10)、(8,0,18,10)，NMS 阈值 0.3；只改变三个框的分数",
                   "第 0 和第 1 个框的 IoU": r(ci[0, 1]), "第 1 和第 2 个框的 IoU": r(ci[1, 2]), "第 0 和第 2 个框的 IoU": r(ci[0, 2]),
                   "分数 0.9、0.8、0.7 时保留": nms_np(chain, [0.9, 0.8, 0.7], [0, 0, 0], 0.3).tolist(), "分数 0.8、0.9、0.7 时保留": nms_np(chain, [0.8, 0.9, 0.7], [0, 0, 0], 0.3).tolist(),
                   "分数 0.9、0.8、0.7 时“被删的框继续删别人”的错误实现保留": bug_not_greedy(T(chain), torch.tensor([0.9, 0.8, 0.7]), torch.zeros(3, dtype=torch.long), 0.3).tolist()}
    NUM["Notebook 里的四个框"] = {"说明": "本章 Notebook 的 boxes 和 scores，用参考实现重算（都当成同一类）",
                            "第 0 和第 1 个框的 IoU": r(ni[0, 1]), "第 0 和第 2 个框的 IoU": r(ni[0, 2]), "第 1 和第 2 个框的 IoU": r(ni[1, 2]),
                            "按 NMS 阈值保留的下标": {str(t): nms_np(nb_boxes, nb_scores, np.zeros(4, int), t).tolist() for t in (0.1, 0.45, 0.9)}}

    npos, nneg, ppos, pneg, gamma = 10, sum((640 // st) ** 2 for st in (8, 16, 32)) - 10, 0.3, 0.02, 2
    bce_p, bce_n = npos * -math.log(ppos), nneg * -math.log(1 - pneg)
    foc_p, foc_n = bce_p * (1 - ppos) ** gamma, bce_n * pneg ** gamma
    NUM["Focal Loss 算例"] = {
        "说明": "纯算术的例子，不是训练实测。640x640 的输入在步长 8、16、32 的三层特征图上共有 80x80+40x40+20x20 个位置；假设其中 10 个是正样本、模型给它们的概率是 0.3，其余都是负样本、模型给它们的目标概率是 0.02；gamma=2，不加 alpha",
        "三层特征图的位置总数": npos + nneg, "负样本个数": nneg, "步长 8、16、32 的特征图边长": [640 // st for st in (8, 16, 32)],
        "一个 10 px 的目标在三层特征图上占几个格子（边长）": {str(st): r(10 / st, 2) for st in (8, 16, 32)},
        "权重 (1-p_t)^2": {str(p): float(f"{(1 - p) ** 2:.2g}") for p in (0.3, 0.5, 0.9, 0.98, 0.99)},
        "交叉熵：正样本的损失之和": r(bce_p, 2), "交叉熵：负样本的损失之和": r(bce_n, 1), "交叉熵：负样本占总损失的比例（%）": r(100 * bce_n / (bce_p + bce_n), 1),
        "Focal：正样本的损失之和": r(foc_p, 2), "Focal：负样本的损失之和": r(foc_n, 3), "Focal：负样本占总损失的比例（%）": r(100 * foc_n / (foc_p + foc_n), 1)}

    def timing():
        rng_ = np.random.default_rng(5)
        xy, wh = rng_.uniform(0, 600, (1000, 2)), rng_.uniform(5, 60, (1000, 2))
        b = T(np.hstack([xy, xy + wh]))
        fast = []
        for _ in range(5):
            t0 = time.perf_counter()
            piou(b, b)
            fast.append(time.perf_counter() - t0)
        bl = b.tolist()
        t0 = time.perf_counter()
        for p in bl:
            for q_ in bl:
                iw, ih = max(0.0, min(p[2], q_[2]) - max(p[0], q_[0])), max(0.0, min(p[3], q_[3]) - max(p[1], q_[1]))
                u = (p[2] - p[0]) * (p[3] - p[1]) + (q_[2] - q_[0]) * (q_[3] - q_[1]) - iw * ih
                _ = iw * ih / u if u > 0 else 0.0
        slow = time.perf_counter() - t0
        two = lambda v: float(f"{v:.2g}")
        return {"说明": "1000 个框和它们自己的两两 IoU（100 万对），在一台 Apple 芯片 Mac 的 CPU 上测一次，torch 限制为 2 个线程，机器上同时还有别的任务；只看量级",
                "参考实现（广播）：5 次里最快的一次（毫秒）": two(1000 * min(fast)), "Python 两层循环逐对计算（秒）": two(slow), "倍数（约）": two(slow / min(fast))}
    NUM["向量化的耗时"] = S.pinned(CH, "向量化的耗时", timing, "CH11_RETIME")


def main() -> None:
    S.setup()
    for fn in (fig_formats, table_roi, fig_small, fig_losses, fig_pr_ap, fig_dense, fig_class, table_misc):
        fn()
        print("完成", fn.__name__, flush=True)
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False))


if __name__ == "__main__":
    main()
