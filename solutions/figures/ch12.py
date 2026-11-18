"""第 12 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch12.py

输出：<主分支>/12_图像分割/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。

数据是代码生成的 64x64 纹理加圆斑缺陷，网络是两次下采样的小 U-Net（约 2.8 万个参数），
损失和跳跃连接的拼接调用本章作业的参考实现。全部在 CPU 上跑，固定种子，不下载任何数据和权重。
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

CH = "12_图像分割"
mod = S.load_solution(CH, "作业_Dice与跳跃连接.py")
NUM: dict = {}
N = 64
STEPS, BS, LR = 300, 8, 3e-3
SEEDS = (0, 1, 2)
AMP = 0.18
DATA_NOTE = (f"{N}x{N} 的合成纹理（高斯模糊的随机场，标准差 0.06，再加标准差 0.03 的白噪声，平均灰度 0.5）；"
             f"缺陷是圆斑，比周围亮或暗 {AMP}，圆心位置带小数")
NET_NOTE = (f"小 U-Net：两次 2 倍下采样，通道数 8、16、32，每级两个 3x3 卷积加 BatchNorm 和 ReLU，双线性上采样后和编码特征拼接；"
            f"Adam，学习率 {LR}，批大小 {BS}，每个种子训练 {STEPS} 步，CPU；每次评估前先用全部训练图把 BatchNorm 的均值和方差重新统计一遍")
STATE: dict = {}


def r3(v):
    return round(float(v), 3)


def r4(v):
    return round(float(v), 4)


def sig2(v):
    return float(f"{float(v):.2g}")


# ---------------------------------------------------------------- 数据
def texture(rng, n=N):
    base = cv2.GaussianBlur(rng.normal(size=(n, n)), (0, 0), 3.0)
    return 0.5 + 0.06 * base / base.std() + rng.normal(scale=0.03, size=(n, n))


def add_spots(img, rng, radii, gap=None):
    """往 img 上加圆斑。返回掩膜和“每个像素属于半径多大的缺陷”的图。gap 不为 None 时缺陷之间至少隔 gap 个像素。"""
    n = img.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    mask, rmap, placed = np.zeros((n, n), bool), np.zeros((n, n)), []
    for r in radii:
        for _ in range(200):
            cy, cx = rng.uniform(r + 2, n - 2 - r, 2)
            if gap is None or all((cy - a) ** 2 + (cx - b) ** 2 > (r + c + gap) ** 2 for a, b, c in placed):
                break
        placed.append((cy, cx, r))
        one = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        img[one] += rng.choice([-1.0, 1.0]) * AMP
        mask |= one
        rmap[one] = r
    return mask, rmap


def make_set(tag, n_img, radii_fn, size=N, gap=None):
    rng = np.random.default_rng(tag)
    X, M, R = [], [], []
    for i in range(n_img):
        img = texture(rng, size)
        mask, rmap = add_spots(img, rng, radii_fn(i), gap)
        X.append(img)
        M.append(mask)
        R.append(rmap)
    return torch.tensor(np.stack(X), dtype=torch.float32)[:, None], torch.tensor(np.stack(M), dtype=torch.float32)[:, None], np.stack(R)


# ---------------------------------------------------------------- 网络与训练
def norm_layer(kind, c):
    return nn.GroupNorm(4, c) if kind == "gn" else nn.BatchNorm2d(c)


def block(cin, cout, norm="bn"):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), norm_layer(norm, cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), norm_layer(norm, cout), nn.ReLU(inplace=True))


class TinyUNet(nn.Module):
    def __init__(self, base=8, skip=True, norm="bn"):
        super().__init__()
        self.skip = skip
        self.e1, self.e2, self.mid = block(1, base, norm), block(base, 2 * base, norm), block(2 * base, 4 * base, norm)
        self.r2 = nn.Conv2d(4 * base, 2 * base, 1)
        self.d2 = block(4 * base if skip else 2 * base, 2 * base, norm)
        self.r1 = nn.Conv2d(2 * base, base, 1)
        self.d1 = block(2 * base if skip else base, base, norm)
        self.head = nn.Conv2d(base, 1, 1)

    def features(self, x, rec=None):
        """输出层之前的特征。rec 是一个列表时，把池化、上采样、拼接产生的张量也记下来（统计激活量用）。"""
        def keep(t):
            if rec is not None:
                rec.append(t)
            return t

        s1 = self.e1(x)
        s2 = self.e2(keep(F.max_pool2d(s1, 2)))
        h = self.mid(keep(F.max_pool2d(s2, 2)))
        h = self.r2(keep(F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)))
        h = self.d2(keep(mod.align_and_concat(h, s2)) if self.skip else h)
        h = self.r1(keep(F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)))
        return self.d1(keep(mod.align_and_concat(h, s1)) if self.skip else h)

    def forward(self, x):
        return self.head(self.features(x))


def bce(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)


LOSSES = {
    "交叉熵": bce,
    "Dice": lambda lg, t: mod.binary_dice_loss(lg, t),
    "交叉熵 + Dice": lambda lg, t: bce(lg, t) + mod.binary_dice_loss(lg, t),
}


def new_net(seed, skip=True, norm="bn"):
    torch.manual_seed(seed)
    net = TinyUNet(skip=skip, norm=norm)
    return net, torch.optim.Adam(net.parameters(), lr=LR), torch.Generator().manual_seed(seed)


@torch.no_grad()
def predict(net, X):
    net.eval()
    return torch.cat([torch.sigmoid(net(X[i:i + 50])) for i in range(0, len(X), 50)])[:, 0].numpy()


@torch.no_grad()
def calibrate(net, X):
    """用 X 把 BatchNorm 的均值和方差重新统计一遍（各批等权平均），再切到推理模式。批只有 8 时训练中的滑动平均很不稳，见 exp_skip 里的对照。"""
    bns = [m for m in net.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    net.train()
    for i in range(0, len(X), BS):
        net(X[i:i + BS])
    for m in bns:
        m.momentum = 0.1
    net.eval()
    return net


def train(net, opt, gen, loss, Xtr, Mtr, steps, log=None, calib=True):
    """训练 steps 步。log=(每多少步, 评估用的图, 掩膜) 时返回阈值 0.5 下的 Dice 曲线。calib=False 时结束后不重新统计 BatchNorm。"""
    curve = []
    net.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(Xtr), (BS,), generator=gen)
        opt.zero_grad()
        loss(net(Xtr[idx]), Mtr[idx]).backward()
        opt.step()
        if log and step % log[0] == 0:
            curve.append(dice(predict(calibrate(net, Xtr), log[1]) > 0.5, log[2]))
            net.train()
    if calib:
        calibrate(net, Xtr)
    net.eval()
    return curve


def dice(a, b):
    """两个布尔数组合起来算的 Dice；两边都空时返回 1。"""
    s = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / s if s else 1.0


def iou(a, b):
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else 1.0


def disk(n, cy, cx, r):
    yy, xx = np.mgrid[0:n, 0:n]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def overlay(img, pred, gt):
    """灰度图上叠颜色：命中绿，多报橙，漏报蓝。"""
    g = np.clip((img - 0.2) / 0.6, 0, 1)
    rgb = np.stack([g, g, g], axis=-1)
    for sel, color in ((pred & gt, (0.10, 0.61, 0.29)), (pred & ~gt, (0.91, 0.44, 0.04)), (~pred & gt, (0.18, 0.44, 0.93))):
        rgb[sel] = 0.25 * rgb[sel] + 0.75 * np.array(color)
    return rgb


# ---------------------------------------------------------------- 图 1：框和掩膜
def scratch(n, angle, length=40.0, half_width=1.0):
    """一条划痕：到线段的距离不超过 half_width 的像素。"""
    yy, xx = np.mgrid[0:n, 0:n]
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    mid = (n - 1) / 2
    u = (xx - mid) * c + (yy - mid) * s
    v = -(xx - mid) * s + (yy - mid) * c
    du = np.clip(np.abs(u) - length / 2, 0, None)
    return np.hypot(du, v) <= half_width


def bbox_area(mask):
    ys, xs = np.nonzero(mask)
    return int((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)), (xs.min(), ys.min(), xs.max(), ys.max())


def fig_box_vs_mask():
    angles = list(range(0, 91, 5))
    ratio, area = [], []
    for a in angles:
        m = scratch(96, a)
        ratio.append(bbox_area(m)[0] / m.sum())
        area.append(int(m.sum()))
    disks = {}
    for r in (2, 5, 10, 20):
        vals = []
        rng = np.random.default_rng([12, 1, r])
        for _ in range(200):
            m = disk(96, *(48 + rng.uniform(-0.5, 0.5, 2)), r)
            vals.append(bbox_area(m)[0] / m.sum())
        disks[r] = float(np.mean(vals))
    show = [0, 5, 10, 20, 30, 45]
    NUM["外接框和掩膜的面积"] = {
        "说明": "划痕是到一条 40 px 长的线段距离不超过 1 px 的像素（宽约 2 px），画在 96x96 的图中央，转不同角度；圆斑的圆心位置带随机小数，每个半径 200 次取平均。"
              "比值是外接矩形框的像素数除以掩膜的像素数",
        "划痕角度（度）": show,
        "划痕的掩膜像素数": [area[angles.index(a)] for a in show],
        "划痕：框面积除以掩膜面积": [round(ratio[angles.index(a)], 2) for a in show],
        "圆斑：框面积除以掩膜面积": {f"半径 {r} px": round(v, 2) for r, v in disks.items()},
        "圆的外接正方形面积除以圆面积的理论值 4/pi": round(4 / math.pi, 3),
    }
    NUM["前景占比举例"] = {
        "说明": "一个缺陷的像素数除以整张图的像素数，圆斑按 pi*r^2 算",
        "64x64 图里一个半径 2 px 的圆斑": f"{100 * math.pi * 4 / 64 ** 2:.2f}%",
        "64x64 图里十六个半径 2 px 的圆斑（不重叠时）": f"{100 * 16 * math.pi * 4 / 64 ** 2:.1f}%",
        "1024x1024 图里一个 10x10 的缺陷": f"{100 * 100 / 1024 ** 2:.4f}%",
    }

    rng = np.random.default_rng([12, 1])
    img = texture(rng, 96)
    m = scratch(96, 45)
    img[m] -= AMP
    _, (x0, y0, x1, y1) = bbox_area(m)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0), gridspec_kw={"width_ratios": [1, 1, 1.5]}, layout="constrained")
    ax[0].imshow(img, cmap="gray", vmin=0.2, vmax=0.8)
    ax[0].add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0 + 1, y1 - y0 + 1, fill=False, ec=S.ORANGE, lw=1.6))
    ax[0].set_title("一条斜 45 度的划痕和它的外接框")
    box = np.zeros((96, 96), bool)
    box[y0:y1 + 1, x0:x1 + 1] = True
    ax[1].imshow(overlay(np.full((96, 96), 0.8), box, m))
    ax[1].set_title(f"绿：划痕 {int(m.sum())} px，橙：框里多算的 {int(box.sum() - m.sum())} px")
    for a in ax[:2]:
        a.set_xlim(x0 - 8, x1 + 8)
        a.set_ylim(y1 + 8, y0 - 8)
        S.image_axes(a)
    ax[2].plot(angles, ratio, "o-", color=S.BLUE, lw=1.8, ms=4, label="划痕（长 40 px，宽约 2 px）")
    ax[2].axhline(disks[10], color=S.GREEN, lw=1.4, ls="--", label=f"圆斑（半径 2 到 20 px 都在 {min(disks.values()):.2f} 到 {max(disks.values()):.2f} 倍之间）")
    ax[2].set_xticks(range(0, 91, 15))
    ax[2].set_xlabel("划痕和水平方向的夹角（度）")
    ax[2].set_ylabel("框面积 / 掩膜面积")
    ax[2].set_ylim(0, None)
    ax[2].set_title(f"斜 45 度的划痕，框面积是真实面积的 {ratio[angles.index(45)]:.0f} 倍")
    ax[2].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax[2])
    S.save(fig, CH, "fig01_外接框与掩膜")


# ---------------------------------------------------------------- 图 3：三种损失，两种前景占比
def exp_losses():
    levels = {"每张图 16 个缺陷": 16, "每张图 1 个缺陷": 1}
    n_steps = {16: STEPS // 2, 1: STEPS}  # 缺陷多的那组收敛快，只训练一半的步数，省时间
    thr_grid = [round(0.05 * i, 2) for i in range(1, 20)]
    res, curves, sweeps, share = {}, {}, {}, {}
    plateau, longer = [], []
    for lname, cnt in levels.items():
        Xtr, Mtr, _ = make_set([12, 3, cnt, 0], 96, lambda i: [2.0] * cnt)
        Xte, Mte, _ = make_set([12, 3, cnt, 1], 100, lambda i: [2.0] * cnt)
        gt = Mte[:, 0].numpy() > 0.5
        share[lname] = float(Mtr.mean())
        for kind, loss in LOSSES.items():
            for seed in SEEDS:
                net, opt, gen = new_net(seed)
                curves[(lname, kind, seed)] = train(net, opt, gen, loss, Xtr, Mtr, n_steps[cnt], log=(n_steps[cnt] // 6, Xte[:40], gt[:40]))
                p = predict(net, Xte)
                pred = p > 0.5
                tp = int((pred & gt).sum())
                sw = [dice(p > t, gt) for t in thr_grid]
                sweeps[(lname, kind, seed)] = sw
                res[(lname, kind, seed)] = {
                    "Dice": dice(pred, gt), "查全": tp / int(gt.sum()), "查准": tp / max(int(pred.sum()), 1),
                    "最好的阈值": thr_grid[int(np.argmax(sw))], "最好阈值下的 Dice": max(sw),
                    "最大概率": float(p.max()), "缺陷像素上概率的中位数": float(np.median(p[gt])),
                    "像素准确率": float((pred == gt).mean()),
                }
                if kind == "交叉熵" and cnt == 1:
                    with torch.no_grad():
                        f = net.features(Xte)
                        below = (net.head(f)[:, 0] < net.head.bias - 1e-4).numpy()
                    dead = (f == 0).all(dim=1).numpy()
                    plateau.append({"种子": seed, "阈值 0.5 下的 Dice": r3(dice(pred, gt)), "最大概率": r3(p.max()),
                                    "输出层偏置过 sigmoid 的值": r3(torch.sigmoid(net.head.bias.detach())),
                                    "缺陷像素里最后一层特征全为 0 的比例": r3(dead[gt].mean()), "背景像素里的同一比例": r4(dead[~gt].mean()),
                                    "输出层 8 个权重里负数的个数": int((net.head.weight.detach() < 0).sum()),
                                    "背景像素里 logit 低于偏置的比例": r4(below[~gt].mean()), "缺陷像素里 logit 低于偏置的比例": r3(below[gt].mean())})
                    train(net, opt, gen, loss, Xtr, Mtr, STEPS)
                    p2 = predict(net, Xte)
                    longer.append({"种子": seed, "阈值 0.5 下的 Dice": r3(dice(p2 > 0.5, gt)), "最大概率": r3(p2.max())})
        print(f"  {lname} 完成", flush=True)

    def cell(lname, kind, key, fmt=r3):
        return [fmt(res[(lname, kind, s)][key]) for s in SEEDS]

    NUM["三种损失与前景占比"] = {
        "说明": DATA_NOTE + "，半径 2 px（约 13 个像素）。两种数据只差每张图里缺陷的个数；各 96 张训练图、100 张测试图。" + NET_NOTE
              + f"（每张图 16 个缺陷的那一组只训练 {STEPS // 2} 步）。Dice 损失的 eps 用作业默认的 1e-6。每项三个数是种子 0、1、2 的结果，指标在测试集上把全部像素合起来算，阈值 0.5",
        "训练集的前景占比": {k: f"{100 * v:.2f}%" for k, v in share.items()},
        **{lname: {kind: {key: cell(lname, kind, key) for key in ("Dice", "查全", "查准", "最好的阈值", "最好阈值下的 Dice", "最大概率", "缺陷像素上概率的中位数")}
                   for kind in LOSSES} for lname in levels},
        "每 50 步的 Dice（每张图 1 个缺陷，在前 40 张测试图上算）": {kind: [[r3(v) for v in curves[("每张图 1 个缺陷", kind, s)]] for s in SEEDS] for kind in LOSSES},
        "每 25 步的 Dice（每张图 16 个缺陷，共 150 步）": {kind: [[r3(v) for v in curves[("每张图 16 个缺陷", kind, s)]] for s in SEEDS] for kind in LOSSES},
        "像素准确率（每张图 1 个缺陷，交叉熵）": cell("每张图 1 个缺陷", "交叉熵", "像素准确率", r4),
    }
    NUM["交叉熵卡在阈值以下"] = {
        "说明": "每张图 1 个缺陷、交叉熵训练的三个网络。先看第 300 步：最大概率、输出层偏置过 sigmoid 的值，以及输出层之前的 8 个特征（ReLU 之后）全为 0 的像素占多少。"
              "“logit 低于偏置”指比偏置低 1e-4 以上，说明这个像素上被激活的特征合起来给了负的贡献。再接着训练 300 步到第 600 步",
        "第 300 步": plateau, "第 600 步": longer,
    }
    low = "每张图 1 个缺陷"
    NUM["阈值与 Dice"] = {
        "说明": "每张图 1 个缺陷、训练 300 步的网络，在测试集上换不同阈值得到的 Dice（全部像素合起来算），三个种子取平均",
        "阈值": thr_grid,
        **{kind: [r3(np.mean([sweeps[(low, kind, s)][i] for s in SEEDS])) for i in range(len(thr_grid))] for kind in LOSSES},
    }
    dice_sweep = NUM["阈值与 Dice"]["Dice"][1:-1]
    NUM["阈值与 Dice"]["Dice 训练的网络在阈值 0.1 到 0.9 之间的最大值减最小值"] = r3(max(dice_sweep) - min(dice_sweep))

    colors = {"交叉熵": S.BLUE, "Dice": S.ORANGE, "交叉熵 + Dice": S.GREEN}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.1), layout="constrained")
    for a, lname in zip(ax[:2], ("每张图 16 个缺陷", low)):
        total = n_steps[levels[lname]]
        steps = list(range(total // 6, total + 1, total // 6))
        for kind, color in colors.items():
            for s in SEEDS:
                a.plot(steps, curves[(lname, kind, s)], "-", color=color, lw=0.9, alpha=0.45)
            a.plot(steps, np.mean([curves[(lname, kind, s)] for s in SEEDS], axis=0), "o-", color=color, lw=2.0, ms=4.5, label=kind)
        a.set_xticks(steps)
        a.set_xlabel("训练步数")
        a.set_ylabel("测试图上的 Dice（阈值 0.5）")
        a.set_ylim(-0.04, 1.04)
        a.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
        S.clean(a)
    worst = min(res[("每张图 16 个缺陷", kind, s)]["Dice"] for kind in LOSSES for s in SEEDS)
    ax[0].set_title(f"前景占 {100 * share['每张图 16 个缺陷']:.1f}%：三种损失 {STEPS // 2} 步后都不低于 {math.floor(worst * 100) / 100:.2f}")
    stuck = sum(res[(low, "交叉熵", s)]["Dice"] < 0.5 for s in SEEDS)
    ax[1].set_title(f"前景占 {100 * share[low]:.2f}%：交叉熵有 {stuck} 个种子到 300 步还不到 0.5")
    for kind, color in colors.items():
        for s in SEEDS:
            ax[2].plot(thr_grid, sweeps[(low, kind, s)], "-", color=color, lw=0.9, alpha=0.45)
        ax[2].plot(thr_grid, NUM["阈值与 Dice"][kind], "o-", color=color, lw=2.0, ms=4, label=kind)
    ax[2].axvline(0.5, color=S.MUTED, lw=1, ls=":")
    ax[2].set_xlabel("阈值")
    ax[2].set_ylabel(f"Dice（前景占 {100 * share[low]:.2f}%，第 300 步）")
    ax[2].set_ylim(-0.04, 1.04)
    ax[2].set_title("交叉熵训练的网络要把阈值降下来才有结果")
    ax[2].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    S.clean(ax[2])
    S.save(fig, CH, "fig03_三种损失")


# ---------------------------------------------------------------- 图 2：有没有跳跃连接
RADII = (1.0, 1.5, 2.0, 3.0, 5.0)


def per_target(pred, gt_r, r):
    """一个目标的 Dice：只在它周围 2 px 的窗口里看预测。返回 (Dice, 是否至少碰到一个像素)。"""
    reg = gt_r == r
    win = cv2.dilate(reg.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    a = pred & win
    return 2.0 * int((a & reg).sum()) / (int(a.sum()) + int(reg.sum())), bool((pred & reg).any())


def exp_skip():
    Xtr, Mtr, _ = make_set([12, 2, 0], 96, lambda i: RADII, gap=3)
    Xte, Mte, Rte = make_set([12, 2, 1], 100, lambda i: RADII, gap=3)
    gt = Mte[:, 0].numpy() > 0.5
    k3 = np.ones((3, 3), np.uint8)
    band = np.stack([cv2.dilate(g.astype(np.uint8), k3) - cv2.erode(g.astype(np.uint8), k3) for g in gt]).astype(bool)
    rows, nets, preds, bn_probe = {True: [], False: []}, {}, {}, []
    for skip in (True, False):
        for seed in SEEDS:
            net, opt, gen = new_net(seed, skip)
            if skip:
                train(net, opt, gen, LOSSES["交叉熵 + Dice"], Xtr, Mtr, STEPS - 100, calib=False)
                probe = []
                for k in range(5):
                    if k:
                        train(net, opt, gen, LOSSES["交叉熵 + Dice"], Xtr, Mtr, 25, calib=False)
                    tmp = copy.deepcopy(net).train()
                    with torch.no_grad():
                        ptrain = torch.cat([torch.sigmoid(tmp(Xte[i:i + BS])) for i in range(0, 96, BS)])[:, 0].numpy()
                    probe.append((dice(predict(net, Xte) > 0.5, gt), dice(ptrain > 0.5, gt[:96]), dice(predict(calibrate(copy.deepcopy(net), Xtr), Xte) > 0.5, gt)))
                bn_probe.append(probe)
                calibrate(net, Xtr)
            else:
                train(net, opt, gen, LOSSES["交叉熵 + Dice"], Xtr, Mtr, STEPS)
            pred = predict(net, Xte) > 0.5
            nets[(skip, seed)], preds[(skip, seed)] = net, pred
            by_r = {r: np.mean([per_target(pred[i], Rte[i], r) for i in range(len(gt))], axis=0) for r in RADII}
            err = pred != gt
            rows[skip].append({"全部像素合起来的 Dice": dice(pred, gt), "按半径": by_r, "边界带里的错误率": float(err[band].mean()),
                               "边界带外的错误率": float(err[~band].mean()), "参数量": sum(q.numel() for q in net.parameters())})
        print(f"  跳跃连接={skip} 完成", flush=True)
    STATE.update(nets=nets)
    name = {True: "有跳跃连接", False: "没有跳跃连接"}
    NUM["跳跃连接"] = {
        "说明": DATA_NOTE + "。每张图 5 个缺陷，半径 1、1.5、2、3、5 px 各一个，互不相连；96 张训练图、100 张测试图，前景占 "
              f"{100 * float(Mtr.mean()):.1f}%。" + NET_NOTE + "，损失是交叉熵加 Dice。“没有跳跃连接”指解码器不拼接编码特征，其余相同。"
              "按半径的 Dice 是逐个目标算的：只看目标周围 2 px 以内的预测，再对 100 个目标取平均。边界带指真值边界内外各 1 px。每项三个数是种子 0、1、2",
        "半径（px）": list(RADII),
        "各半径圆斑的平均像素数": [round(float(np.mean([(Rte[i] == r).sum() for i in range(len(gt))])), 1) for r in RADII],
        **{name[k]: {
            "参数量": v[0]["参数量"],
            "全部像素合起来的 Dice": [r3(x["全部像素合起来的 Dice"]) for x in v],
            "全部像素合起来的 Dice（三个种子平均）": r3(np.mean([x["全部像素合起来的 Dice"] for x in v])),
            **{f"半径 {r} px 的目标：Dice": [r3(x["按半径"][r][0]) for x in v] for r in RADII},
            "各半径目标的 Dice（三个种子平均）": [r3(np.mean([x["按半径"][r][0] for x in v])) for r in RADII],
            "半径 1 px 的目标至少被碰到一个像素的比例": [r3(x["按半径"][1.0][1]) for x in v],
            "边界带里的错误率": [r3(x["边界带里的错误率"]) for x in v],
            "边界带里的错误率（三个种子平均）": r3(np.mean([x["边界带里的错误率"] for x in v])),
            "边界带外的错误率": [sig2(x["边界带外的错误率"]) for x in v],
        } for k, v in rows.items()},
    }

    NUM["批很小时的 BatchNorm"] = {
        "说明": "上面有跳跃连接的三次训练（批大小 8）。从第 200 步到第 300 步每 25 步把同一份权重用三种方式评估："
              "推理模式（用训练中滑动平均得到的均值和方差，momentum 0.1）；训练模式（用测试批自己的统计量，每批 8 张，只用前 96 张测试图）；"
              "先用全部 96 张训练图重新统计一遍均值和方差，再用推理模式。每行一个种子，每列一个评估时刻",
        "评估时刻（步）": [200, 225, 250, 275, 300],
        "推理模式（滑动平均）的 Dice": [[r3(v[0]) for v in pr] for pr in bn_probe],
        "训练模式的 Dice": [[r3(v[1]) for v in pr] for pr in bn_probe],
        "重新统计后推理模式的 Dice": [[r3(v[2]) for v in pr] for pr in bn_probe],
        "推理模式（滑动平均）的最小值和最大值": [r3(min(v[0] for pr in bn_probe for v in pr)), r3(max(v[0] for pr in bn_probe for v in pr))],
        "训练模式的最小值和最大值": [r3(min(v[1] for pr in bn_probe for v in pr)), r3(max(v[1] for pr in bn_probe for v in pr))],
        "重新统计后的最小值和最大值": [r3(min(v[2] for pr in bn_probe for v in pr)), r3(max(v[2] for pr in bn_probe for v in pr))],
        "同一份权重两种推理模式结果的最大差": r3(max(abs(v[0] - v[2]) for pr in bn_probe for v in pr)),
    }

    with_, without = NUM["跳跃连接"]["有跳跃连接"], NUM["跳跃连接"]["没有跳跃连接"]
    NUM["跳跃连接"]["去掉跳跃连接后掉了多少（三个种子平均）"] = {
        "全部像素合起来的 Dice": r3(with_["全部像素合起来的 Dice（三个种子平均）"] - without["全部像素合起来的 Dice（三个种子平均）"]),
        "各半径目标的 Dice": [r3(a - b) for a, b in zip(with_["各半径目标的 Dice（三个种子平均）"], without["各半径目标的 Dice（三个种子平均）"])],
    }

    # 后处理对小目标的影响（有跳跃连接的三个网络的预测）
    k_sq, k_cr = np.ones((3, 3), np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    def area_filter(m, min_area):
        _, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= min_area
        keep[0] = False
        return keep[lab]

    posts = {
        "不做后处理": lambda m: m,
        "开运算，3x3 方形": lambda m: cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k_sq).astype(bool),
        "开运算，3x3 十字": lambda m: cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k_cr).astype(bool),
        "面积过滤，不小于 3": lambda m: area_filter(m, 3),
        "面积过滤，不小于 5": lambda m: area_filter(m, 5),
    }
    post_rows = {}
    for pname, fn in posts.items():
        per_seed = []
        for seed in SEEDS:
            q = np.stack([fn(m) for m in preds[(True, seed)]])
            per_seed.append([np.mean([per_target(q[i], Rte[i], r)[0] for i in range(len(gt))]) for r in RADII] + [dice(q, gt)])
        post_rows[pname] = [r3(v) for v in np.mean(per_seed, axis=0)]
    NUM["后处理对小目标的影响"] = {
        "说明": "把有跳跃连接的三个网络在测试集上的二值预测分别做几种后处理，再算逐个目标的 Dice（算法同上）和全部像素合起来的 Dice，三个种子取平均。连通域按 8 连通。"
              "每项六个数依次是半径 1、1.5、2、3、5 px 的目标和全部像素合起来",
        **post_rows,
    }

    # 配图用哪一张测试图：两种网络差别较大的一张，免得挑到一张看不出差别的
    gap = [dice(preds[(True, 0)][i], gt[i]) - dice(preds[(False, 0)][i], gt[i]) for i in range(len(gt))]
    i = int(np.argsort(gap)[-10])
    img = Xte[i, 0].numpy()
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.0), gridspec_kw={"width_ratios": [1, 1, 1, 1.75]}, layout="constrained")
    ax[0].imshow(img, cmap="gray", vmin=0.2, vmax=0.8)
    ax[0].contour(gt[i], levels=[0.5], colors="cyan", linewidths=1.0)
    ax[0].set_title("测试图（青色圈是 5 个缺陷）")
    for a, skip in zip(ax[1:3], (True, False)):
        a.imshow(overlay(img, preds[(skip, 0)][i], gt[i]), interpolation="nearest")
        a.set_title(f"{name[skip]}：这张图 Dice {dice(preds[(skip, 0)][i], gt[i]):.2f}")
    for a in ax[:3]:
        S.image_axes(a)
    x = np.arange(len(RADII))
    for j, (skip, color) in enumerate(((True, S.BLUE), (False, S.MUTED))):
        vals = NUM["跳跃连接"][name[skip]]["各半径目标的 Dice（三个种子平均）"]
        bars = ax[3].bar(x + (j - 0.5) * 0.38, vals, 0.36, color=color, alpha=0.88, label=name[skip])
        for s in range(len(SEEDS)):
            ax[3].plot(x + (j - 0.5) * 0.38, [rows[skip][s]["按半径"][r][0] for r in RADII], ".", color=S.INK, ms=5, alpha=0.6)
        for b, v in zip(bars, vals):
            ax[3].text(b.get_x() + b.get_width() / 2, 1.02, f"{v:.2f}", ha="center", fontsize=8.5, color=color if skip else S.INK)
    ax[3].set_xticks(x, [f"{r:g} px" for r in RADII])
    ax[3].set_xlabel("缺陷半径（黑点是每个种子）")
    ax[3].set_ylabel("逐个目标的 Dice")
    ax[3].set_ylim(0, 1.1)
    ax[3].set_title("去掉跳跃连接，越小的目标掉得越多")
    ax[3].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[3])
    ax[3].grid(axis="x", visible=False)
    S.save(fig, CH, "fig02_跳跃连接")

    # 同一批预测：逐图平均和全部像素合起来
    Xb, Mb, Rb = make_set([12, 2, 2], 140, lambda i: [RADII[i % 5]] if i < 100 else [])
    gb = Mb[:, 0].numpy() > 0.5
    has = gb.reshape(len(gb), -1).any(axis=1)
    agg = {True: [], False: []}
    for (skip, seed), net in nets.items():
        pred = predict(net, Xb) > 0.5
        per = np.array([dice(pred[i], gb[i]) for i in range(len(gb))])
        fp_empty = pred[~has].reshape(int((~has).sum()), -1).any(axis=1)
        agg[skip].append({
            "合起来": dice(pred[has], gb[has]), "逐图": per[has].mean(),
            "按半径逐图": [per[:100][np.arange(100) % 5 == j].mean() for j in range(5)],
            "含空图合起来": dice(pred, gb), "含空图逐图": per.mean(), "有误报的空图张数": int(fp_empty.sum()),
        })
    STATE.update(agg=agg, Xb=Xb, gb=gb, has=has, Xtr=Xtr, Mtr=Mtr, Xte=Xte, gt=gt)
    zero = np.zeros_like(gb)
    drop = {key: r3(np.mean([x[key] for x in agg[True]]).round(3) - np.mean([x[key] for x in agg[False]]).round(3)) for key in ("合起来", "逐图")}
    NUM["逐图平均与合起来算"] = {
        "说明": "用上面训练好的网络预测另一批图，共 140 张：100 张各有 1 个缺陷（半径 1、1.5、2、3、5 px 各 20 张），另有 40 张没有缺陷。同一批预测用两种办法算 Dice："
              "把全部像素合起来算一次，或者每张图算一个再取平均。空图的约定：预测也为空记 1，有任何误报记 0。每项三个数是种子 0、1、2",
        "各半径的缺陷像素占全部缺陷像素的比例": [r3((Rb == r).sum() / gb.sum()) for r in RADII],
        **{name[k]: {
            "100 张有缺陷的图：合起来算": [r3(x["合起来"]) for x in v], "100 张有缺陷的图：逐图平均": [r3(x["逐图"]) for x in v],
            "合起来算（三个种子平均）": r3(np.mean([x["合起来"] for x in v])), "逐图平均（三个种子平均）": r3(np.mean([x["逐图"] for x in v])),
            "两种算法之差（三个种子平均）": r3(np.mean([x["合起来"] - x["逐图"] for x in v])),
            "按半径的逐图平均（三个种子平均）": [r3(y) for y in np.mean([x["按半径逐图"] for x in v], axis=0)],
            "加上 40 张空图：合起来算": [r3(x["含空图合起来"]) for x in v], "加上 40 张空图：逐图平均": [r3(x["含空图逐图"]) for x in v],
            "40 张空图里有误报的张数": [x["有误报的空图张数"] for x in v],
        } for k, v in agg.items()},
        "去掉跳跃连接后掉了多少（三个种子平均）": {"合起来算": drop["合起来"], "逐图平均": drop["逐图"]},
        "什么都不预测（全背景）": {
            "100 张有缺陷的图：合起来算": 0.0, "100 张有缺陷的图：逐图平均": 0.0,
            "加上 40 张空图：合起来算": r3(dice(zero, gb)), "加上 40 张空图：逐图平均": r3(np.mean([dice(zero[i], gb[i]) for i in range(len(gb))])),
            "像素准确率": r4((zero == gb).mean()),
        },
    }


# ---------------------------------------------------------------- 表：换成 GroupNorm 会怎样
def exp_groupnorm():
    Xtr, Mtr, Xte, gt, Xb, gb, has = (STATE[k] for k in ("Xtr", "Mtr", "Xte", "gt", "Xb", "gb", "has"))
    nets, rows = {}, []
    for seed in SEEDS[:2]:
        net, opt, gen = new_net(seed, norm="gn")
        train(net, opt, gen, LOSSES["交叉熵 + Dice"], Xtr, Mtr, STEPS)
        nets[seed] = net
        pb = predict(net, Xb) > 0.5
        rows.append({"测试集 Dice": dice(predict(net, Xte) > 0.5, gt), "有误报的空图张数": int(pb[~has].reshape(int((~has).sum()), -1).any(axis=1).sum()),
                     "单缺陷图合起来的 Dice": dice(pb[has], gb[has])})
    STATE.update(gn_nets=nets)
    bn = NUM["逐图平均与合起来算"]["有跳跃连接"]
    NUM["换成 GroupNorm"] = {
        "说明": "把有跳跃连接的网络里的 BatchNorm 换成 GroupNorm（4 组），其余和跳跃连接实验相同：训练图每张 5 个缺陷，在同一个测试集（每张 5 个缺陷）上评估；"
              "再拿去预测“逐图平均与合起来算”那一批图（100 张各 1 个缺陷，40 张没有缺陷）。GroupNorm 的均值和方差来自当前这一张输入。GroupNorm 只跑了种子 0 和 1，BatchNorm 是种子 0、1、2",
        "GroupNorm：每张 5 个缺陷的测试集上的 Dice": [r3(x["测试集 Dice"]) for x in rows],
        "GroupNorm：40 张空图里有误报的张数": [x["有误报的空图张数"] for x in rows],
        "GroupNorm：100 张单缺陷图合起来算的 Dice": [r3(x["单缺陷图合起来的 Dice"]) for x in rows],
        "BatchNorm：每张 5 个缺陷的测试集上的 Dice": NUM["跳跃连接"]["有跳跃连接"]["全部像素合起来的 Dice"],
        "BatchNorm：40 张空图里有误报的张数": bn["40 张空图里有误报的张数"],
        "BatchNorm：100 张单缺陷图合起来算的 Dice": bn["100 张有缺陷的图：合起来算"],
    }


# ---------------------------------------------------------------- 图 4：Dice 里的平滑项，以及不用训练就能算的损失数值
def dloss(logits, target, eps=1e-6):
    return float(mod.binary_dice_loss(logits, target, eps))


def const_logits(z, target_px=0, hit=None, size=N):
    """一张 size x size 的图：背景 logit 是 z，前 target_px 个像素是目标；hit 是目标像素上的 logit（None 表示和背景一样）。"""
    logits = torch.full((1, 1, size, size), float(z))
    target = torch.zeros(1, 1, size, size)
    target.view(-1)[:target_px] = 1
    if hit is not None:
        logits.view(-1)[:target_px] = float(hit)
    return logits, target


def fig_eps():
    eps_list = [1e-6, 1e-3, 1.0, 100.0]
    eps_name = {1e-6: "1e-6", 1e-3: "0.001", 1.0: "1", 100.0: "100"}
    zs = np.linspace(-20, 0, 81)
    empty_curve = {e: [dloss(*const_logits(z), e) for z in zs] for e in eps_list}
    sizes = [1, 4, 13, 50, 200, 1000]
    miss = {e: [dloss(*const_logits(-30, g), e) for g in sizes] for e in eps_list}
    fps = [0, 1, 4, 13]
    fp = {}
    for e in eps_list:
        row = []
        for k in fps:
            logits, target = const_logits(-30)
            logits.view(-1)[:k] = 30.0
            row.append(dloss(logits, target, e))
        fp[e] = row
    grad = {}
    for e in eps_list:
        logits, target = const_logits(-4)
        logits.requires_grad_(True)
        mod.binary_dice_loss(logits, target, e).backward()
        grad[e] = float(logits.grad.abs().sum())
    bgs = [-4, -6, -8, -12]
    perfect = {g: [dloss(*const_logits(z, g, hit=12), 1e-6) for z in bgs] for g in (13, 1300)}
    perfect_curve = {g: [dloss(*const_logits(z, g, hit=12), 1e-6) for z in zs[zs <= -2]] for g in (13, 1300)}
    chk = torch.full((2, 1, 4, 4), -12.0)
    NUM["Dice 的平滑项"] = {
        "说明": "全部用作业参考实现的 binary_dice_loss 直接算出，不涉及训练。图像 64x64（4096 个像素），一个样本",
        "eps": [eps_name[e] for e in eps_list],
        "验收脚本里的例子（2 张 4x4 的空掩膜，logit 全是 -12，eps=1e-6）的损失": r4(dloss(chk, torch.zeros_like(chk))),
        "logit=-12 时单个像素的概率": float(f"{1 / (1 + math.exp(12)):.3g}"),
        "空掩膜、整张图的 logit 是同一个常数时的损失（每行一个 eps，每列一个 logit）": {
            "logit": [-4, -8, -12, -20], **{eps_name[e]: [r4(dloss(*const_logits(z), e)) for z in (-4, -8, -12, -20)] for e in eps_list}},
        "空掩膜上有 k 个像素被判成前景、其余干净时的损失": {"k": fps, **{eps_name[e]: [r4(v) for v in fp[e]] for e in eps_list}},
        "g 个像素的目标被完全漏掉时的损失": {"g": sizes, **{eps_name[e]: [r4(v) for v in miss[e]] for e in eps_list}},
        "空掩膜、logit 全是 -4 时，损失对 logit 的梯度绝对值之和": {eps_name[e]: sig2(grad[e]) for e in eps_list},
        "目标像素 logit=12（硬判定完全正确）、背景 logit 取不同值时的损失（eps=1e-6）": {
            "背景 logit": bgs, "背景每个像素的概率": [float(f"{1 / (1 + math.exp(-z)):.2g}") for z in bgs],
            "目标 13 个像素": [r4(v) for v in perfect[13]], "目标 1300 个像素": [r4(v) for v in perfect[1300]]},
    }

    # 做作业时容易写错的几种 Dice，各自在同一个例子上的数值
    logits, target = const_logits(-12, 1300, hit=12)
    small_l, small_t = const_logits(-12, 13)
    both_l, both_t = torch.cat([logits, small_l]), torch.cat([target, small_t])
    p = torch.sigmoid(both_l)
    pooled = 1 - float((2 * (p * both_t).sum() + 1e-6) / (p.sum() + both_t.sum() + 1e-6))
    NUM["Dice 损失的几种写法"] = {
        "说明": "两张 64x64 的图组成一个批：第一张有 1300 个像素的目标，预测完全正确；第二张有 13 个像素的目标，完全漏掉。背景 logit 都是 -12",
        "参考实现（每张图算一个 Dice 再平均）": r4(dloss(both_l, both_t)),
        "把整个批的像素合起来算一个 Dice": r4(pooled),
        "掩膜取值是 0 和 255、预测完全正确时参考实现的损失": r4(dloss(logits, target * 255)),
        "同一情况下交叉熵的值": r3(bce(logits, target * 255)),
    }

    # 交叉熵：全背景预测有多便宜；sigmoid 做了两次会怎样
    shares = [0.05, 0.01, 13 / 4096, 0.001, 0.0001]
    ent = [-(q * math.log(q) + (1 - q) * math.log(1 - q)) for q in shares]
    NUM["全背景预测的交叉熵"] = {
        "说明": "每个像素都输出同一个概率 p 时，交叉熵最小的 p 就是前景占比，这时的损失是下面的值（自然对数）。初始化后输出概率在 0.5 附近时损失是 ln 2",
        "ln 2": r4(math.log(2)),
        "前景占比": [f"{100 * q:.2f}%" for q in shares],
        "全背景常数预测能达到的最小交叉熵": [r4(v) for v in ent],
        "它占初始损失 ln 2 的比例": [f"{100 * v / math.log(2):.1f}%" for v in ent],
        "阈值 0.5 下的像素准确率": [f"{100 * (1 - q):.2f}%" for q in shares],
        "阈值 0.5 下的 Dice": 0,
    }
    lo, hi = 1 / (1 + math.exp(-0.0)), 1 / (1 + math.exp(-1.0))
    q = 13 / 4096
    NUM["sigmoid 做了两次"] = {
        "说明": "先手动做一次 sigmoid 再交给 BCEWithLogitsLoss，等于把 0 到 1 之间的数当成 logit 再做一次 sigmoid",
        "这时最终概率能取到的范围": [r3(lo), r3(hi)],
        "背景像素的损失下限 ln 2": r4(-math.log(1 - lo)),
        "前景像素的损失下限": r4(-math.log(hi)),
        "前景占 0.32% 时整张图的损失下限": r4(q * -math.log(hi) + (1 - q) * -math.log(1 - lo)),
    }

    colors = {1e-6: S.RED, 1e-3: S.ORANGE, 1.0: S.BLUE, 100.0: S.GREEN}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.1), layout="constrained")
    for e in eps_list:
        ax[0].plot(zs, empty_curve[e], "-", color=colors[e], lw=1.9, label=f"eps = {eps_name[e]}")
    ax[0].axvline(-12, color=S.MUTED, lw=1, ls=":")
    ax[0].text(-11.7, 0.5, "验收脚本用的\nlogit = -12", fontsize=8.5, color=S.MUTED, va="center")
    ax[0].set_xlabel("整张图的 logit（越往左预测越干净）")
    ax[0].set_ylabel("Dice 损失")
    ax[0].set_title("空掩膜：eps=1e-6 时预测再干净损失也接近 1")
    ax[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    S.clean(ax[0])
    for e in (1e-6, 1.0, 100.0):
        ax[1].plot(sizes, miss[e], "o-", color=colors[e], lw=1.9, ms=5, label=f"eps = {eps_name[e]}")
    ax[1].set_xscale("log")
    ax[1].set_xticks(sizes, [str(g) for g in sizes])
    ax[1].minorticks_off()
    ax[1].set_xlabel("被完全漏掉的目标有多少个像素")
    ax[1].set_ylabel("Dice 损失")
    ax[1].set_ylim(-0.04, 1.06)
    ax[1].set_title("漏掉小目标：eps=100 时几乎不罚")
    ax[1].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    S.clean(ax[1])
    zz = zs[zs <= -2]
    ax[2].plot(zz, perfect_curve[13], "-", color=S.RED, lw=1.9, label="目标 13 个像素")
    ax[2].plot(zz, perfect_curve[1300], "-", color=S.BLUE, lw=1.9, label="目标 1300 个像素")
    ax[2].set_xlabel("背景像素的 logit（目标像素的 logit 固定为 12）")
    ax[2].set_ylabel("Dice 损失（eps = 1e-6）")
    ax[2].set_ylim(-0.04, 1.06)
    ax[2].set_title("硬判定全对，小目标的软 Dice 损失仍可以很大")
    ax[2].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    S.save(fig, CH, "fig04_平滑项")


# ---------------------------------------------------------------- 图 5：Dice 与 IoU，偏移，逐图与合起来
def fig_metrics():
    ious = [0.1, 0.3, 0.5, 0.7, 0.9]
    x = np.linspace(0, 1, 100001)
    gap = 2 * x / (1 + x) - x
    # 用随机掩膜核对换算式
    rng = np.random.default_rng([12, 5])
    worst = 0.0
    for _ in range(200):
        a, b = rng.random((32, 32)) < rng.uniform(0.05, 0.6), rng.random((32, 32)) < rng.uniform(0.05, 0.6)
        worst = max(worst, abs(dice(a, b) - 2 * iou(a, b) / (1 + iou(a, b))))
    NUM["Dice 与 IoU"] = {
        "说明": "同一对掩膜上 Dice = 2*IoU/(1+IoU)，IoU = Dice/(2-Dice)。用 200 对随机掩膜核对过",
        "IoU": ious, "对应的 Dice": [r3(2 * i / (1 + i)) for i in ious],
        "Dice": ious, "对应的 IoU": [r3(d / (2 - d)) for d in ious],
        "Dice 减 IoU 的最大值": r3(gap.max()), "最大值出现在 IoU 等于": r3(x[gap.argmax()]), "这时的 Dice": r3(2 * x[gap.argmax()] / (1 + x[gap.argmax()])),
        "200 对随机掩膜上换算式和直接计算的最大差": float(f"{worst:.1g}"),
        "平均之后排序可以反过来": {
            "说明": "两个模型各在两张图上评估，先逐图算指标再平均。单张图上 Dice 和 IoU 的排序总是一致的，平均之后不一定",
            "模型甲两张图的 IoU": [0.92, 0.1], "模型乙两张图的 IoU": [0.5, 0.5],
            "模型甲的平均 IoU": r3(np.mean([0.92, 0.1])), "模型乙的平均 IoU": r3(np.mean([0.5, 0.5])),
            "模型甲的平均 Dice": r3(np.mean([2 * i / (1 + i) for i in (0.92, 0.1)])), "模型乙的平均 Dice": r3(np.mean([2 * i / (1 + i) for i in (0.5, 0.5)])),
        },
    }

    radii = [1, 1.5, 2, 3, 5, 8, 12, 20]
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    ops = {
        "预测整体偏 1 px": lambda g: np.roll(g, 1, 1),
        "预测整体偏 2 px": lambda g: np.roll(g, 2, 1),
        "预测比真值胖一圈（膨胀 1 px）": lambda g: cv2.dilate(g.astype(np.uint8), cross).astype(bool),
        "预测比真值瘦一圈（腐蚀 1 px）": lambda g: cv2.erode(g.astype(np.uint8), cross).astype(bool),
    }
    table = {k: [] for k in ops}
    table_iou = {k: [] for k in ops}
    areas = []
    for r in radii:
        rng = np.random.default_rng([12, 6, int(r * 10)])
        gs = [disk(96, *(48 + rng.uniform(-0.5, 0.5, 2)), r) for _ in range(200)]
        areas.append(float(np.mean([g.sum() for g in gs])))
        for k, op in ops.items():
            table[k].append(float(np.mean([dice(g, op(g)) for g in gs])))
            table_iou[k].append(float(np.mean([iou(g, op(g)) for g in gs])))
    NUM["偏移和粗细对 Dice 的影响"] = {
        "说明": "真值是一个圆斑，圆心位置带随机小数，每个半径 200 次取平均；预测是把真值整体平移，或者用 3x3 十字结构元膨胀、腐蚀一次。没有任何别的误差",
        "半径（px）": radii, "平均像素数": [round(a, 1) for a in areas],
        **{f"{k}：Dice": [r3(v) for v in vals] for k, vals in table.items()},
        **{f"{k}：IoU": [r3(v) for v in vals] for k, vals in table_iou.items()},
    }
    # 演示页用的设置：圆心在整数像素上，胖瘦用半径加减
    demo = {}
    for r in (2, 20):
        g = disk(96, 48, 48, r)
        demo[f"半径 {r} px，偏 1 px"] = [r3(dice(g, np.roll(g, 1, 1))), r3(iou(g, np.roll(g, 1, 1)))]
    g = disk(96, 48, 48, 2)
    demo["半径 2 px，预测半径 3 px"] = [r3(dice(g, disk(96, 48, 48, 3))), r3(iou(g, disk(96, 48, 48, 3)))]
    far = np.roll(g, 6, 1)
    demo["半径 2 px，偏 6 px（完全错开）"] = [r3(dice(g, far)), r3(iou(g, far))]
    demo["同上，96x96 图上的像素准确率"] = r4((g == far).mean())
    s0 = scratch(96, 0, 40, 1.0)
    demo["划痕（长 40 px、宽约 2 px），垂直于划痕偏 1 px"] = [r3(dice(s0, np.roll(s0, 1, 0))), r3(iou(s0, np.roll(s0, 1, 0)))]
    demo["划痕，沿着划痕偏 1 px"] = [r3(dice(s0, np.roll(s0, 1, 1))), r3(iou(s0, np.roll(s0, 1, 1)))]
    NUM["演示页的同款设置"] = {"说明": "演示页里圆心固定在整数像素上、画布 96x96，每项两个数是 Dice 和 IoU。和上一组（圆心带随机小数、取平均）略有出入", **demo}

    agg = STATE["agg"]
    both = NUM["逐图平均与合起来算"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.1), gridspec_kw={"width_ratios": [1, 1.25, 1.15]}, layout="constrained")
    ax[0].plot(x, 2 * x / (1 + x), "-", color=S.BLUE, lw=2, label="Dice = 2·IoU / (1 + IoU)")
    ax[0].plot([0, 1], [0, 1], "--", color=S.MUTED, lw=1.2, label="Dice = IoU（对角线）")
    xm = x[gap.argmax()]
    ax[0].plot([xm, xm], [xm, 2 * xm / (1 + xm)], "-", color=S.RED, lw=1.6)
    ax[0].text(xm + 0.06, xm + 0.03, f"最大相差 {gap.max():.3f}\n（IoU = {xm:.3f}）", color=S.RED, fontsize=9, va="center")
    ax[0].set_xlabel("IoU")
    ax[0].set_ylabel("Dice")
    ax[0].set_aspect("equal")
    ax[0].set_title("同一个结果，Dice 总是不小于 IoU")
    ax[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax[0])
    for (k, vals), color, marker in zip(table.items(), (S.BLUE, S.RED, S.ORANGE, S.GREEN), ("o-", "s-", "^--", "v--")):
        ax[1].plot(radii, vals, marker, color=color, lw=1.7, ms=5, label=k)
    ax[1].set_xscale("log")
    ax[1].set_xticks(radii, [f"{r:g}" for r in radii])
    ax[1].minorticks_off()
    ax[1].set_xlabel("目标半径（px）")
    ax[1].set_ylabel("Dice")
    ax[1].set_ylim(-0.04, 1.04)
    i2 = radii.index(2)
    ax[1].set_title(f"同样偏 1 px：半径 2 px 的目标 Dice {table['预测整体偏 1 px'][i2]:.2f}，半径 20 px 的 {table['预测整体偏 1 px'][-1]:.2f}")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[1])
    xs = np.arange(2)
    for j, (key, color, label) in enumerate((("合起来", S.MUTED, "全部像素合起来算"), ("逐图", S.BLUE, "每张图算一个再平均"))):
        vals = [np.mean([v[key] for v in agg[k]]) for k in (True, False)]
        bars = ax[2].bar(xs + (j - 0.5) * 0.38, vals, 0.36, color=color, alpha=0.88, label=label)
        for k_i, k in enumerate((True, False)):
            ax[2].plot([xs[k_i] + (j - 0.5) * 0.38] * len(SEEDS), [v[key] for v in agg[k]], ".", color=S.INK, ms=5, alpha=0.6)
        for b, v in zip(bars, vals):
            ax[2].text(b.get_x() + b.get_width() / 2, 1.02, f"{v:.2f}", ha="center", fontsize=9)
    ax[2].set_xticks(xs, ["有跳跃连接", "没有跳跃连接"])
    ax[2].set_ylim(0, 1.1)
    ax[2].set_ylabel("100 张有缺陷的图上的 Dice（黑点是每个种子）")
    d_with, d_without = both["有跳跃连接"]["两种算法之差（三个种子平均）"], both["没有跳跃连接"]["两种算法之差（三个种子平均）"]
    ax[2].set_title(f"同一批预测，两种算法相差 {d_with:.3f} 和 {d_without:.3f}")
    ax[2].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    ax[2].grid(axis="x", visible=False)
    S.save(fig, CH, "fig05_指标")


# ---------------------------------------------------------------- 图 6：大图切块，激活量
def fig_tiling():
    net = STATE["nets"][(True, 0)]
    big, bigm, _ = make_set([12, 6, 0], 2, lambda i: RADII * 16, size=256, gap=3)
    gt = bigm[:, 0].numpy() > 0.5
    core = 64

    @torch.no_grad()
    def prob(x, model=None):
        return torch.sigmoid((model or net)(x))[:, 0].numpy()

    whole = prob(big)

    def tiled(halo, how, model=None):
        out, wsum = np.zeros_like(whole), np.zeros_like(whole)
        for y in range(0, 256, core):
            for x in range(0, 256, core):
                y0, y1, x0, x1 = max(y - halo, 0), min(y + core + halo, 256), max(x - halo, 0), min(x + core + halo, 256)
                p = prob(big[:, :, y0:y1, x0:x1], model)
                if how == "中央":
                    out[:, y:y + core, x:x + core] = p[:, y - y0:y - y0 + core, x - x0:x - x0 + core]
                    continue
                if how == "平均":
                    w = np.ones(p.shape[1:])
                else:  # 加权：离块的边越近权重越小；贴着整图边界的那一侧不降权
                    wy = np.minimum(np.where(y0 > 0, np.arange(y1 - y0) + 1, 1e9), np.where(y1 < 256, (y1 - y0) - np.arange(y1 - y0), 1e9))
                    wx = np.minimum(np.where(x0 > 0, np.arange(x1 - x0) + 1, 1e9), np.where(x1 < 256, (x1 - x0) - np.arange(x1 - x0), 1e9))
                    w = np.minimum(wy[:, None], wx[None, :])
                out[:, y0:y1, x0:x1] += p * w
                wsum[:, y0:y1, x0:x1] += w
        return out if how == "中央" else out / wsum

    halos = [0, 4, 8, 12, 16, 20, 24, 28, 32]
    stats = {how: [] for how in ("中央", "平均", "加权")}
    for h in halos:
        for how in stats:
            t = tiled(h, how)
            stats[how].append((float(np.abs(t - whole).max()), int(((t > 0.5) != (whole > 0.5)).sum()), dice(t > 0.5, gt)))
    nohalo = tiled(0, "中央")

    # 感受野：一个输出像素对哪些输入像素有梯度
    def reach_of(model):
        far = 0
        xin = torch.randn(1, 1, 128, 128, generator=torch.Generator().manual_seed(0))
        for oy in range(64, 68):
            for ox in range(64, 68):
                xi = xin.clone().requires_grad_(True)
                model(xi)[0, 0, oy, ox].backward()
                ys, xs = torch.nonzero(xi.grad[0, 0] != 0, as_tuple=True)
                far = max(far, int((ys - oy).abs().max()), int((xs - ox).abs().max()))
        return far

    reach = reach_of(net)
    gn = STATE["gn_nets"][0]
    gn_whole = prob(big, gn)
    gn_rows = []
    for h in (0, 16, 32):
        t = tiled(h, "中央", gn)
        gn_rows.append([sig2(np.abs(t - gn_whole).max()), int(((t > 0.5) != (gn_whole > 0.5)).sum())])
    NUM["换成 GroupNorm"]["切块推理（种子 0，只保留每块中央）"] = {
        "说明": "和“大图切块推理”同样的两张 256x256 的图和切法。每项两个数是和整图推理的最大概率差、阈值 0.5 下判定不同的像素数",
        "多读的像素数 h": [0, 16, 32], "GroupNorm": gn_rows,
        "一个输出像素最远受多远的输入像素影响（px，输入 128x128 时）": reach_of(gn),
    }
    off = {}
    for o in (32, 34):
        p = prob(big[:, :, o:o + 128, o:o + 128])
        off[o] = float(np.abs(p[:, 48:80, 48:80] - whole[:, o + 48:o + 80, o + 48:o + 80]).max())
    NUM["大图切块推理"] = {
        "说明": "用上面有跳跃连接、种子 0 的网络（在 64x64 的图上训练）预测 2 张 256x256 的图（每张 80 个缺陷）。先整张图一次推理作为参照，再切成 4x4 个 64x64 的块，"
              "每块向四周多读 h 个像素的上下文（到图像边界为止）。三种拼法：只保留每块中央的 64x64；把重叠处的概率直接平均；按离块边的距离加权平均。"
              "每项三个数依次是：和整图推理的最大概率差、阈值 0.5 下判定不同的像素数（共 131072 个像素）、对真值的 Dice",
        "整图推理的 Dice": r4(dice(whole > 0.5, gt)),
        "多读的像素数 h": halos,
        **{f"拼法：{'只保留中央' if how == '中央' else '重叠处直接平均' if how == '平均' else '按距离加权平均'}":
           [[sig2(a), b, r4(c)] for a, b, c in v] for how, v in stats.items()},
        "一个输出像素最远受多远的输入像素影响（px，按梯度是否为 0 实测）": reach,
        "块的起点不是 4 的倍数时": {
            "说明": "从大图里切一个 128x128 的块单独推理，只比较块中央的 32x32（四周各留 48 px，超过上面的影响范围），和整图推理的最大概率差",
            "起点 (32, 32)": sig2(off[32]), "起点 (34, 34)": sig2(off[34])},
    }

    # 激活量和参数量
    npar = sum(q.numel() for q in net.parameters())
    act = {}
    for size in (64, 128, 256, 512):
        by_res: dict = {}

        def hook(_m, _i, out):
            by_res[out.shape[-1]] = by_res.get(out.shape[-1], 0) + out.numel()

        hooks = [m.register_forward_hook(hook) for m in net.modules() if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))]
        rec: list = []
        with torch.no_grad():
            net.features(torch.zeros(1, 1, size, size), rec)
        for h in hooks:
            h.remove()
        by_res[size] += size * size  # 输出层的 1 个通道（features 不经过输出层）
        for t in rec:
            by_res[t.shape[-1]] = by_res.get(t.shape[-1], 0) + t.numel()
        act[size] = [by_res[size], by_res[size // 2], by_res[size // 4]]
    NUM["激活量和参数量"] = {
        "说明": "同一个小 U-Net，输入 1 张图。激活量是一次前向里每个卷积、BatchNorm、池化、上采样、拼接输出的张量元素个数之和（ReLU 是原地操作，不另算）。"
              "训练时反向传播要用到这些中间结果，所以显存大致随它线性增长；具体占多少字节取决于实现，这里只比较个数",
        "参数量": npar,
        "输入边长": list(act),
        "激活元素个数": [sum(v) for v in act.values()],
        "激活元素个数除以参数量": [round(sum(v) / npar, 1) for v in act.values()],
        "各分辨率所占比例（原分辨率、1/2、1/4）": [f"{100 * a / sum(act[64]):.1f}%" for a in act[64]],
        "边长加倍后激活量变成几倍": round(sum(act[128]) / sum(act[64]), 2),
    }

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3), gridspec_kw={"width_ratios": [1, 1.35, 1.2]}, layout="constrained")
    d = np.abs(nohalo - whole)[0]
    im = ax[0].imshow(d, cmap="magma", vmin=0, vmax=1)
    for k in range(64, 256, 64):
        ax[0].axhline(k - 0.5, color="cyan", lw=0.6, alpha=0.7)
        ax[0].axvline(k - 0.5, color="cyan", lw=0.6, alpha=0.7)
    ax[0].set_title("不多读上下文：和整图推理的概率差")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03).ax.tick_params(labelsize=8)
    S.image_axes(ax[0])
    for how, color, marker, label in (("中央", S.BLUE, "o-", "只保留每块中央"), ("平均", S.RED, "s--", "重叠处直接平均"), ("加权", S.GREEN, "^-", "按距离加权平均")):
        ax[1].plot(halos, [max(v[0], 1e-7) for v in stats[how]], marker, color=color, lw=1.8, ms=5, label=label)
    ax[1].set_yscale("log")
    ax[1].set_yticks([1e-6, 1e-4, 1e-2, 1], ["1e-6", "1e-4", "0.01", "1"])
    ax[1].minorticks_off()
    ax[1].set_xticks(halos)
    ax[1].axvline(reach, color=S.MUTED, lw=1, ls=":")
    ax[1].text(reach - 0.6, 1.3e-6, f"输入像素的影响最远到 {reach} px", fontsize=8.5, color=S.MUTED, ha="right", va="bottom")
    ax[1].set_xlabel("每块向四周多读的像素数")
    ax[1].set_ylabel("和整图推理的最大概率差")
    ax[1].set_title("多读够了并且只留中央，才和整图推理一致")
    ax[1].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    S.clean(ax[1])
    sides = list(act)
    for j, (label, color, marker) in enumerate((("原分辨率的层", S.RED, "o-"), ("1/2 分辨率的层", S.ORANGE, "s-"), ("1/4 分辨率的层", S.GREEN, "^-"))):
        ax[2].plot(sides, [act[s][j] / npar for s in sides], marker, color=color, lw=1.8, ms=5, label=label)
    ax[2].axhline(1, color=S.MUTED, lw=1.3, ls="--", label="参数量（不随输入变）")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xticks(sides, [str(s) for s in sides])
    ax[2].set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    ax[2].minorticks_off()
    ax[2].set_xlabel("输入边长（px，1 张图）")
    ax[2].set_ylabel("激活元素个数 / 参数量")
    ax[2].set_title("激活量随面积涨，大头在原分辨率那一级")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    S.save(fig, CH, "fig06_切块与激活量")


# ---------------------------------------------------------------- 不配图的几张表
def tables():
    # 掩膜缩小到 1/4：小目标还在不在
    sides = [1, 2, 3, 4, 6, 8]
    ways = ["最近邻", "双线性后取 > 0.5", "INTER_AREA 后取 > 0.5", "块里有前景就算前景"]
    keep = {w: [] for w in ways}
    area = {w: [] for w in ways}
    for s in sides:
        rng = np.random.default_rng([12, 7, s])
        rows = {w: [] for w in ways}
        for _ in range(500):
            m = np.zeros((128, 128), np.uint8)
            y, x = rng.integers(8, 112, 2)
            m[y:y + s, x:x + s] = 1
            f = m.astype(np.float32)
            outs = {
                ways[0]: cv2.resize(m, (32, 32), interpolation=cv2.INTER_NEAREST) > 0,
                ways[1]: cv2.resize(f, (32, 32), interpolation=cv2.INTER_LINEAR) > 0.5,
                ways[2]: cv2.resize(f, (32, 32), interpolation=cv2.INTER_AREA) > 0.5,
                ways[3]: m.reshape(32, 4, 32, 4).max(axis=(1, 3)) > 0,
            }
            for w, o in outs.items():
                rows[w].append((o.any(), o.sum() * 16 / m.sum()))
        for w in ways:
            k, a = np.mean(rows[w], axis=0)
            keep[w].append(float(k))
            area[w].append(float(a))
    NUM["掩膜缩小后小目标还在不在"] = {
        "说明": "128x128 的二值掩膜里放一个 s x s 的方块，位置随机，用 cv2.resize 缩小到 32x32（边长变成 1/4），每种边长 500 次。"
              "保留比例是缩小后还剩至少一个前景像素的次数占比；面积比是缩小后的前景像素数乘 16 再除以原来的像素数",
        "方块边长 s（px）": sides,
        "随机位置下最近邻保留比例的理论值 min(1, (s/4)^2)": [r3(min(1.0, (s / 4) ** 2)) for s in sides],
        **{f"{w}：保留比例": [r3(v) for v in keep[w]] for w in ways},
        **{f"{w}：面积比": [round(v, 2) for v in area[w]] for w in ways},
    }

    # 转置卷积：覆盖次数，和随机初始化时的周期图案
    cover = {}
    for k, s, p in ((2, 2, 0), (3, 2, 1), (4, 2, 1), (5, 2, 2), (4, 4, 0), (6, 4, 1)):
        ct = nn.ConvTranspose2d(1, 1, k, s, p, bias=False)
        nn.init.ones_(ct.weight)
        with torch.no_grad():
            inner = ct(torch.ones(1, 1, 8, 8))[0, 0, 4:-4, 4:-4]
        cover[f"kernel={k}, stride={s}"] = sorted({int(v) for v in inner.flatten().tolist()})

    def phase_spread(make, trials=200):
        torch.manual_seed(12)
        vals = []
        for _ in range(trials):
            with torch.no_grad():
                o = make()(torch.ones(1, 8, 8, 8))[0][:, 4:-4, 4:-4]
            ph = torch.stack([o[:, i::2, j::2].mean(dim=(1, 2)) for i in (0, 1) for j in (0, 1)])
            vals.append(float((ph.max(0).values - ph.min(0).values).mean() / o.abs().mean()))
        return float(np.mean(vals))

    class UpConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(8, 8, 3, padding=1, bias=False)

        def forward(self, x):
            return self.conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))

    NUM["转置卷积的棋盘格"] = {
        "说明": "覆盖次数：权重全设成 1、输入全 1 时输出内部各位置的取值，也就是每个输出像素被多少个卷积核位置覆盖。"
              "周期图案：8 通道进 8 通道出，PyTorch 默认随机初始化 200 次，输入是全 1 的常数图，把输出按 2x2 的四个位置分组求平均，"
              "四组平均值的极差除以输出的平均绝对值；输入没有任何结构时这个数应该是 0",
        "输出内部的覆盖次数": cover,
        "随机初始化时的周期图案强度": {
            "转置卷积 kernel=2, stride=2": r3(phase_spread(lambda: nn.ConvTranspose2d(8, 8, 2, 2, 0, bias=False))),
            "转置卷积 kernel=3, stride=2": r3(phase_spread(lambda: nn.ConvTranspose2d(8, 8, 3, 2, 1, bias=False))),
            "转置卷积 kernel=4, stride=2": r3(phase_spread(lambda: nn.ConvTranspose2d(8, 8, 4, 2, 1, bias=False))),
            "双线性上采样后接 3x3 卷积": r3(phase_spread(UpConv)),
        },
    }

    # 下采样之后小缺陷占一个特征点的多少；尺寸对不上的情况
    NUM["下采样与小缺陷"] = {
        "说明": "总下采样倍率为 s 时，特征图上一个点对应原图 s x s 个像素。半径 2 px 的圆斑约 13 个像素",
        "总下采样倍率": [2, 4, 8, 16, 32], "一个特征点对应的原图像素数": [s * s for s in (2, 4, 8, 16, 32)],
        "13 个像素占一个特征点的比例": [f"{100 * 13 / (s * s):.1f}%" for s in (2, 4, 8, 16, 32)],
    }
    net = STATE["nets"][(True, 0)]
    sizes = {}
    for n in (64, 65, 66, 67):
        half, quarter = n // 2, n // 2 // 2
        with torch.no_grad():
            out = net(torch.zeros(1, 1, n, n))
        sizes[f"输入边长 {n}"] = {"编码器三级的边长": [n, half, quarter], "两次上采样后的边长": [quarter * 2, half * 2],
                              "比对应的编码特征少几个像素": [half - quarter * 2, n - half * 2], "网络输出的边长（拼接前用参考实现对齐）": int(out.shape[-1])}
    NUM["尺寸对不上"] = {
        "说明": "两次 2 倍池化（向下取整）再两次 2 倍上采样。第一个差值是 1/4 分辨率上采样后和 1/2 分辨率编码特征的差，第二个是 1/2 分辨率特征上采样后和原分辨率编码特征的差",
        **sizes,
    }


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(0)
    S.setup()
    t_all = time.perf_counter()
    for fn in (fig_box_vs_mask, exp_skip, exp_groupnorm, exp_losses, fig_eps, fig_metrics, fig_tiling, tables):
        t = time.perf_counter()
        fn()
        print(f"完成 {fn.__name__}（{time.perf_counter() - t:.0f} 秒）", flush=True)
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=2))
    print(f"总耗时 {time.perf_counter() - t_all:.0f} 秒")


if __name__ == "__main__":
    main()
