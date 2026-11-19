"""第 13 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch13.py

输出：<主分支>/13_生成与自监督/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。

全部实验用代码生成的 16x16 条纹小图和很小的全连接网络，在 CPU 上跑，不下载任何数据和权重。
重参数化和 NT-Xent 调用本章作业的参考实现。耗时默认沿用上次的记录，设置 CH13_RETIME=1 才重测。
"""

from __future__ import annotations

import contextlib
import importlib.util
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
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

torch.set_num_threads(2)
torch.manual_seed(0)

CH = "13_生成与自监督"
hw = S.load_solution(CH, "作业_重参数与对比损失.py")
NUM: dict = {}
TIMES: dict = {}
SEEDS = (0, 1, 2)
SIZE, PIX = 16, 256
# 16 类条纹：每类是 16x16 画面里沿 x、y 方向各有几个整周期。前 4 类是四个方向（竖、横、两条对角线）
FREQS = [(2, 0), (0, 2), (2, 2), (2, -2), (3, 0), (0, 3), (3, 3), (3, -3), (2, 1), (1, -2), (1, 2), (2, -1), (3, 1), (1, -3), (1, 3), (3, -1)]
FREQ_T = torch.tensor(FREQS, dtype=torch.float32)
YY, XX = torch.meshgrid(torch.arange(SIZE, dtype=torch.float32), torch.arange(SIZE, dtype=torch.float32), indexing="ij")
DATA_NOTE = "16x16 的条纹小图：类别决定条纹的方向和疏密（每类在画面里沿 x、y 各有固定个数的整周期），相位随机，幅度在 0.1 到 0.3、平均亮度在 0.4 到 0.6 之间均匀取"


def r3(v):
    return round(float(v), 3)


def r4(v):
    return round(float(v), 4)


def gen(seed):
    return torch.Generator().manual_seed(int(seed))


def stripes(n, g, classes=16, noise=0.2, amp=(0.1, 0.3), bright=(0.4, 0.6), y=None):
    """返回带噪声的图、不带噪声的图、类别标签。y 给定时按它生成，否则随机取类别。"""
    y = torch.randint(0, classes, (n,), generator=g) if y is None else y
    f = FREQ_T[y]
    phase = torch.rand(n, generator=g) * 2 * math.pi
    a = amp[0] + (amp[1] - amp[0]) * torch.rand(n, generator=g)
    b = bright[0] + (bright[1] - bright[0]) * torch.rand(n, generator=g)
    arg = 2 * math.pi * (f[:, 0, None, None] * XX + f[:, 1, None, None] * YY) / SIZE + phase[:, None, None]
    clean = b[:, None, None] + a[:, None, None] * torch.sin(arg)
    return clean + noise * torch.randn(n, SIZE, SIZE, generator=g), clean, y


def auroc(score, label):
    """按秩次算 AUROC，并列取平均秩。"""
    score, label = np.asarray(score, float), np.asarray(label).astype(bool)
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    first = np.r_[True, s[1:] != s[:-1]]
    start = np.flatnonzero(first)
    end = np.r_[start[1:], len(s)]
    ranks = np.empty(len(s))
    ranks[order] = np.repeat((start + end + 1) / 2, end - start)
    n1, n0 = label.sum(), (~label).sum()
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mlp(sizes, last_relu=False):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2 or last_relu:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def n_params(net):
    return int(sum(p.numel() for p in net.parameters()))


# ---------------------------------------------------------------- 图 1：自编码器的瓶颈宽度与“把异常也重建出来”
def fig_bottleneck():
    pca_ks, ae_ks, checks = [1, 2, 4, 8, 16, 32, 64, 128], [1, 2, 8, 32, 128], (300, 1500)
    amp_d, rad, n_te = 0.12, 2.0, 500
    rows = {"pca": [], **{("ae", c): [] for c in checks}, **{("dae", c): [] for c in checks}}
    example = None
    for seed in SEEDS:
        g = gen(100 + seed)
        kw = dict(classes=1, noise=0.06, amp=(0.3, 0.3), bright=(0.5, 0.5))
        X = stripes(2000, g, **kw)[0].flatten(1) - 0.5
        A = stripes(n_te, g, **kw)[0].flatten(1) - 0.5
        c = torch.randint(3, SIZE - 3, (n_te, 2), generator=g).float()
        mask = ((YY - c[:, 0, None, None]) ** 2 + (XX - c[:, 1, None, None]) ** 2) <= rad * rad
        sign = torch.where(torch.rand(n_te, generator=g) < 0.5, -1.0, 1.0)
        D = (mask.float() * sign[:, None, None] * amp_d).flatten(1)
        B = A + D
        label = np.r_[np.zeros(n_te), np.ones(n_te)]

        def metrics(rec):
            ya, yb = rec(A), rec(B)
            score = torch.cat([((A - ya) ** 2).mean(1), ((B - yb) ** 2).mean(1)]).numpy()
            return [auroc(score, label), float(((A - ya) ** 2).mean().sqrt()), float((((yb - ya) * D).sum(1) / (D * D).sum(1)).mean())]

        mu = X.mean(0)
        Vt = torch.linalg.svd(X - mu, full_matrices=False)[2]
        rows["pca"].append([metrics(lambda Z, P=Vt[:k]: (Z - mu) @ P.T @ P + mu) for k in pca_ks])

        def train(k, in_noise):
            torch.manual_seed(seed)
            gg = gen(500 + seed)
            net = nn.Sequential(nn.Linear(PIX, 128), nn.ReLU(), nn.Linear(128, k), nn.Linear(k, 128), nn.ReLU(), nn.Linear(128, PIX))
            opt = torch.optim.Adam(net.parameters(), lr=2e-3)
            out = {}
            for step in range(1, max(checks) + 1):
                xb = X[torch.randint(0, len(X), (256,), generator=gg)]
                inp = xb + in_noise * torch.randn(xb.shape, generator=gg) if in_noise else xb
                loss = ((net(inp) - xb) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if step in checks:
                    with torch.no_grad():
                        out[step] = metrics(net)
            return out, net

        ae = [train(k, 0.0)[0] for k in ae_ks]
        dae = train(128, 0.3)[0]
        for c_ in checks:
            rows[("ae", c_)].append([m[c_] for m in ae])
            rows[("dae", c_)].append(dae[c_])
        if seed == 0:
            example = (A[0].reshape(SIZE, SIZE) + 0.5, B[0].reshape(SIZE, SIZE) + 0.5, mask[0].numpy())
    mean = {k: np.mean(v, axis=0) for k, v in rows.items()}
    NUM["瓶颈宽度与异常重建"] = {
        "说明": "良品是 16x16 的竖条纹（周期 8 px，幅度 0.3，平均亮度 0.5，相位随机，像素噪声标准差 0.06，和第 19 章 2.3 节用的条纹相同），2000 张训练。"
              f"待检图 {n_te} 张良品加 {n_te} 张带缺陷的图，缺陷是同一批良品上加一个半径 {rad:.0f} px、幅度 {amp_d} 的圆斑（亮暗随机）。"
              "异常分数是整图重建误差的均方值。自编码器是 256-128-k-128-256 的全连接网络（ReLU），Adam，学习率 0.002，批大小 256，在同一次训练的第 300 步和第 1500 步各评估一次；"
              "去噪自编码器给输入加标准差 0.3 的噪声、目标是不加噪声的图。主成分分析保留前 k 个主成分，相当于最优的线性自编码器。3 个种子取平均。"
              "“缺陷被重建出来的比例”：同一张图加缺陷前后各重建一次，两次重建之差在缺陷方向上的投影占缺陷本身的比例，0 表示缺陷完全留在残差里，1 表示缺陷被原样画了出来",
        "主成分分析的 k": pca_ks,
        "主成分分析：image AUROC": [r3(v[0]) for v in mean["pca"]],
        "主成分分析：良品重建的均方根误差": [r4(v[1]) for v in mean["pca"]],
        "主成分分析：缺陷被重建出来的比例": [r3(v[2]) for v in mean["pca"]],
        "自编码器的 k": ae_ks,
        **{f"自编码器训练 {c} 步：{label}": [(r4 if i == 1 else r3)(v[i]) for v in mean[("ae", c)]] for c in checks for i, label in
           ((0, "image AUROC"), (1, "良品重建的均方根误差"), (2, "缺陷被重建出来的比例"))},
        **{f"去噪自编码器（k=128）训练 {c} 步": {"image AUROC": r3(mean[("dae", c)][0]), "良品重建的均方根误差": r4(mean[("dae", c)][1]),
                                         "缺陷被重建出来的比例": r3(mean[("dae", c)][2])} for c in checks},
        "各种子的 image AUROC（自编码器 1500 步，每行一个种子）": [[r3(m[0]) for m in row] for row in rows[("ae", 1500)]],
        "各种子的 image AUROC（主成分分析，每行一个种子）": [[r3(m[0]) for m in row] for row in rows["pca"]],
        "像素噪声的标准差（重建误差低于它说明连噪声也在重建）": 0.06,
        "自编码器 k=128 的参数量": n_params(nn.Sequential(nn.Linear(PIX, 128), nn.Linear(128, 128), nn.Linear(128, 128), nn.Linear(128, PIX))),
    }

    fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.9), gridspec_kw={"width_ratios": [0.62, 0.62, 1.35, 1.35]}, layout="constrained")
    ax[0].imshow(example[0], cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("一张良品")
    ax[1].imshow(example[1], cmap="gray", vmin=0, vmax=1)
    ax[1].contour(example[2], levels=[0.5], colors="cyan", linewidths=1.3)
    ax[1].set_title("加了缺陷（青圈）")
    for a in ax[:2]:
        S.image_axes(a)
    series = [("pca", pca_ks, "主成分分析（线性）", S.MUTED, "s--"), (("ae", 300), ae_ks, "自编码器，训练 300 步", S.BLUE, "o-"),
              (("ae", 1500), ae_ks, "自编码器，训练 1500 步", S.RED, "o-")]
    for col, a, ylabel in ((0, ax[2], "image AUROC"), (2, ax[3], "缺陷被重建出来的比例")):
        for key, ks, name, color, style in series:
            a.plot(ks, [v[col] for v in mean[key]], style, color=color, lw=1.7, ms=5, label=name)
        a.plot([128], [mean[("dae", 1500)][col]], "D", color=S.GREEN, ms=8, label="去噪自编码器，k=128，1500 步")
        a.set_xscale("log", base=2)
        a.set_xticks(pca_ks, [str(k) for k in pca_ks])
        a.minorticks_off()
        a.set_xlabel("瓶颈宽度 k（输入是 256 维）")
        a.set_ylabel(ylabel)
        a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        S.clean(a)
    ax[2].set_ylim(0.45, 1.0)
    ax[2].set_title("k=1 不够；k 很宽时训练越久越差")
    ax[3].set_ylim(0, 0.6)
    ax[3].set_title("宽瓶颈训练久了，缺陷被一起画出来")
    S.save(fig, CH, "fig01_瓶颈宽度与异常重建")


# ---------------------------------------------------------------- 图 2：VAE 的 KL 权重与后验坍缩
class VAE(nn.Module):
    def __init__(self, d=16, h=128):
        super().__init__()
        self.enc = mlp([PIX, h, h], last_relu=True)
        self.mu, self.lv = nn.Linear(h, d), nn.Linear(h, d)
        self.dec = mlp([d, h, h, PIX])

    def encode(self, x):
        e = self.enc(x.flatten(1) - 0.5)
        return self.mu(e), self.lv(e)

    def decode(self, z):
        return self.dec(z) + 0.5


def kl_per_dim(mu, lv):
    return 0.5 * (mu.pow(2) + lv.exp() - 1 - lv)


VAE_KW = dict(classes=4, noise=0.06)
_FIT = []
for _fx, _fy in FREQS[:4]:
    _arg = (2 * math.pi * (_fx * XX + _fy * YY) / SIZE).flatten()
    _b = torch.stack([torch.ones(PIX), torch.sin(_arg), torch.cos(_arg)], 1)
    _FIT.append((_b, torch.linalg.pinv(_b)))


def family_fit(img):
    """每张图（n,256）在 4 类条纹里拟合得最好的那一类：返回残差的均方根、拟合出的幅度、类别。"""
    res, amp, cls = None, None, None
    for c, (b, pinv) in enumerate(_FIT):
        coef = img @ pinv.T
        r = (img - coef @ b.T).pow(2).mean(1).sqrt()
        a = coef[:, 1:].pow(2).sum(1).sqrt()
        if res is None:
            res, amp, cls = r.clone(), a.clone(), torch.zeros(len(img), dtype=torch.long)
        else:
            m = r < res
            res[m], amp[m], cls[m] = r[m], a[m], c
    return res, amp, cls


def train_vae(beta, seed, steps=600, plain=False, mean_reduction=False):
    g = gen(2000 + seed)
    torch.manual_seed(seed)
    data = stripes(4096, g, **VAE_KW)[0]
    net = VAE()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(steps):
        x = data[torch.randint(0, len(data), (256,), generator=g)]
        mu, lv = net.encode(x)
        z = mu if plain else hw.reparameterize(mu, lv, torch.randn(mu.shape, generator=g))
        se = (net.decode(z) - x.flatten(1)).pow(2)
        rec = se.mean() if mean_reduction else se.sum(1).mean()
        loss = rec if plain else rec + beta * kl_per_dim(mu, lv).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return net


VAE_KEEP: dict = {}


def fig_vae():
    betas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    g = gen(77)
    xte, cte, _ = stripes(2000, g, **VAE_KW)
    res_r, amp_r, _ = family_fit(cte.flatten(1))
    thr_res, amp_lo, amp_hi = 0.03, 0.08, 0.32

    @torch.no_grad()
    def evaluate(net, seed):
        mu, lv = net.encode(xte)
        mse = float((net.decode(mu) - cte.flatten(1)).pow(2).mean())
        kld = kl_per_dim(mu, lv).mean(0)
        samp = net.decode(torch.randn(1000, mu.shape[1], generator=gen(9000 + seed)))
        res, amp, cls = family_fit(samp)
        ok = (res < thr_res) & (amp > amp_lo) & (amp < amp_hi)
        return [mse, float(kld.sum()), int((kld > 0.01).sum()), float(ok.float().mean()), int(len(torch.unique(cls[ok]))), float(amp.median())], samp[:6]

    keys = ["普通自编码器"] + betas + ["重建项取平均，β=0.03"]
    rows, grids = {k: [] for k in keys}, {}
    for seed in SEEDS:
        for k in keys:
            if k == keys[0]:
                net = train_vae(0.0, seed, plain=True)
            elif k == keys[-1]:
                net = train_vae(0.03, seed, mean_reduction=True)
            else:
                net = train_vae(k, seed)
            m, samp = evaluate(net, seed)
            rows[k].append(m)
            if seed == 0:
                grids[k] = samp
                if k == 0.03:
                    VAE_KEEP["net"] = net
    mean = {k: np.mean(v, axis=0) for k, v in rows.items()}
    labels = ("重建误差（和不带噪声的图比，均方）", "KL（每张图，单位 nat）", "用到的隐变量维数", "先验采样里像真条纹的比例", "合格样本覆盖的类别数", "采样图里条纹幅度的中位数")
    best = max(betas, key=lambda b: mean[b][3])
    NUM["KL 权重与后验坍缩"] = {
        "说明": DATA_NOTE + "。这里只用前 4 类（四个方向），像素噪声标准差 0.06，4096 张训练。VAE 是 256-128-128 的编码器、16 维隐变量、16-128-128-256 的解码器，"
              "损失是“每张图 256 个像素的平方误差之和 + β×KL”，Adam，学习率 0.002，批大小 256，训练 600 步，3 个种子取平均。"
              "重建误差用隐变量的均值解码，在 2000 张测试图上和不带噪声的图比。“用到的维数”指平均 KL 大于 0.01 nat 的维度个数（共 16 维）。"
              f"“像真条纹”：从标准高斯采 1000 个 z 解码，解码图用 4 类条纹各拟合一次，取最好的一类，残差均方根小于 {thr_res} 且幅度在 {amp_lo} 到 {amp_hi} 之间算合格"
              "（真实的不带噪声的图按这个标准全部合格）。普通自编码器是同一个网络去掉采样和 KL 项。"
              "“重建项取平均”指把平方误差对 256 个像素取平均（F.mse_loss 的默认行为）而 KL 仍然求和，相当于 β 放大 256 倍",
        "β": betas,
        **{label: [(r4 if i == 0 else r3)(mean[b][i]) for b in betas] for i, label in enumerate(labels)},
        "普通自编码器": {label: (r4 if i == 0 else r3)(mean[keys[0]][i]) for i, label in enumerate(labels) if i not in (1, 2)},
        "重建项取平均、β=0.03": {label: (r4 if i == 0 else r3)(mean[keys[-1]][i]) for i, label in enumerate(labels)},
        "重建项取平均时 β=0.03 相当于求和写法下的 β": r3(0.03 * PIX),
        "各种子的“像真条纹的比例”（每行一个种子）": [[r3(rows[b][s][3]) for b in betas] for s in range(len(SEEDS))],
        "各种子的“用到的维数”（每行一个种子）": [[int(rows[b][s][2]) for b in betas] for s in range(len(SEEDS))],
        "采样质量最好的 β": best,
        "真实图的检验": {"不带噪声的真图合格比例": r3(((res_r < thr_res) & (amp_r > amp_lo) & (amp_r < amp_hi)).float().mean()), "隐变量全被忽略时的重建误差（只输出平均图）": r4((cte - cte.mean(0)).pow(2).mean())},
        "VAE 的参数量": n_params(VAE()),
    }

    fig = plt.figure(figsize=(16, 3.9), layout="constrained")
    gs = fig.add_gridspec(3, 4, width_ratios=[1.15, 1, 1, 1.25])
    a0, a1, a2 = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1]), fig.add_subplot(gs[:, 2])
    a0.plot(betas, [mean[b][0] for b in betas], "o-", color=S.BLUE, lw=1.8, ms=5, label="重建误差（左轴）")
    a0.set_xscale("log")
    a0.set_xlabel("KL 项的权重 β")
    a0.set_ylabel("重建误差（均方）")
    a0b = a0.twinx()
    a0b.plot(betas, [mean[b][1] for b in betas], "s--", color=S.ORANGE, lw=1.6, ms=5, label="KL（右轴）")
    a0b.set_ylabel("KL（nat）")
    a0b.spines["top"].set_visible(False)
    h1, l1 = a0.get_legend_handles_labels()
    h2, l2 = a0b.get_legend_handles_labels()
    a0.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    a0.set_title("β 加大：KL 一路降，重建误差先平后陡升")
    S.clean(a0)
    for s in range(len(SEEDS)):
        a1.plot(betas, [rows[b][s][2] for b in betas], "o", color=S.PURPLE, ms=3.5, alpha=0.35)
    a1.plot(betas, [mean[b][2] for b in betas], "o-", color=S.PURPLE, lw=1.8, ms=5)
    a1.set_xscale("log")
    a1.set_ylim(-0.6, 17)
    a1.set_xlabel("KL 项的权重 β")
    a1.set_ylabel("用到的隐变量维数（共 16）")
    a1.set_title("用到的维度随 β 减少，最后归零")
    S.clean(a1)
    for s in range(len(SEEDS)):
        a2.plot(betas, [rows[b][s][3] for b in betas], "o", color=S.GREEN, ms=3.5, alpha=0.35)
    a2.plot(betas, [mean[b][3] for b in betas], "o-", color=S.GREEN, lw=1.8, ms=5, label="VAE（浅色点是各种子）")
    a2.axhline(mean[keys[0]][3], color=S.RED, lw=1.3, ls="--", label="普通自编码器")
    a2.set_xscale("log")
    a2.set_ylim(-0.03, 0.6)
    a2.set_xlabel("KL 项的权重 β")
    a2.set_ylabel("先验采样里像真条纹的比例")
    a2.set_title("坍缩前采样最像样，也只有两成合格")
    a2.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(a2)
    for a in (a0, a1, a2):
        a.set_xticks(betas, ["0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1"], fontsize=8.5)
        a.minorticks_off()
    for i, (k, title) in enumerate(((keys[0], "普通自编码器：从标准高斯采样再解码"), (best, f"VAE，β={best}"), (1.0, "VAE，β=1（已坍缩）"))):
        a = fig.add_subplot(gs[i, 3])
        a.imshow(torch.cat([im.reshape(SIZE, SIZE) for im in grids[k]], 1).numpy(), cmap="gray", vmin=0, vmax=1)
        a.set_title(title, fontsize=9.5)
        S.image_axes(a)
    S.save(fig, CH, "fig02_KL权重与后验坍缩")


# ---------------------------------------------------------------- 图 3：重参数化和得分函数两种梯度估计的方差
def fig_gradient():
    dims, n = [1, 2, 4, 8, 16, 32, 64], 100000
    sigma, offset = 1.0, 1.0
    var = {"重参数化": [], "得分函数": [], "得分函数减基线": []}
    bias, hist = [], None
    for d in dims:
        g = gen(300 + d)
        a = torch.full((d,), offset)
        mu = torch.zeros(n, d, requires_grad=True)
        logvar = torch.full((n, d), 2 * math.log(sigma))
        eps = torch.randn(n, d, generator=g)
        z = hw.reparameterize(mu, logvar, eps)
        f = ((z - a) ** 2).sum(1)
        g_rp = torch.autograd.grad(f.sum(), mu)[0]
        zd, fd = z.detach(), f.detach()
        score = (zd - mu.detach()) / sigma ** 2
        g_sf = fd[:, None] * score
        g_bl = (fd - (d * sigma ** 2 + d * offset ** 2))[:, None] * score
        for name, est in (("重参数化", g_rp), ("得分函数", g_sf), ("得分函数减基线", g_bl)):
            var[name].append(float(est.var(0).mean()))
        bias.append([float(est.mean()) for est in (g_rp, g_sf, g_bl)])
        if d == 16:
            hist = (g_rp[:, 0].numpy().copy(), g_sf[:, 0].numpy().copy())
    # mu 等于目标时得分函数估计的方差有闭式：sigma^2 (D+2)(D+4)
    d0, g0 = 16, gen(999)
    e0 = torch.randn(n, d0, generator=g0)
    check = float(((sigma * e0).pow(2).sum(1, keepdim=True) * e0 / sigma).var(0).mean())

    # 优化：只更新 mu，最小化 E||z - a||^2
    d, steps, lr, trials = 16, 1000, 0.002, 20
    a = torch.full((d,), offset)

    def descend(name, lr_, steps_):
        g = gen(42)
        mu = torch.zeros(trials, d)
        k = 64 if "64" in name else 1
        path, blow = [float((mu - a).pow(2).mean(1).sqrt().median())], None
        for step in range(1, steps_ + 1):
            eps = torch.randn(trials, k, d, generator=g)
            z = mu[:, None] + sigma * eps
            f = ((z - a) ** 2).sum(2, keepdim=True)
            if name == "重参数化":
                grad = (2 * (z - a)).mean(1)
            elif name == "得分函数减基线":
                grad = ((f - (d * sigma ** 2 + ((mu - a) ** 2).sum(1)[:, None, None])) * eps / sigma).mean(1)
            else:
                grad = (f * eps / sigma).mean(1)
            mu = mu - lr_ * grad
            err = float((mu - a).pow(2).mean(1).sqrt().median())
            if blow is None and not (err < 100):
                blow = step
            path.append(err)
        return path, blow

    curves = {name: descend(name, lr, steps)[0] for name in ("重参数化", "得分函数", "得分函数减基线", "得分函数，64 个样本平均")}
    big_lr = 0.02
    big = {name: descend(name, big_lr, 300) for name in ("重参数化", "得分函数")}

    # 在训练好的 VAE 上量一次
    net, g = VAE_KEEP["net"], gen(55)
    x = stripes(64, g, **VAE_KW)[0]
    with torch.no_grad():
        mu0, lv0 = net.encode(x)
    rp, sf = [], []
    for _ in range(400):
        mu = mu0.clone().requires_grad_(True)
        eps = torch.randn(mu.shape, generator=g)
        z = hw.reparameterize(mu, lv0, eps)
        rec = (net.decode(z) - x.flatten(1)).pow(2).sum(1)
        rp.append(torch.autograd.grad(rec.sum(), mu)[0])
        sf.append(rec.detach()[:, None] * eps / torch.exp(0.5 * lv0))
    rp, sf = torch.stack(rp), torch.stack(sf)
    truth = rp.mean(0)
    NUM["两种梯度估计的方差"] = {
        "说明": f"玩具问题：z 服从 D 维高斯 N(mu, sigma^2 I)，目标是 E||z - a||^2 对 mu 的梯度，mu=0，a 的每个分量都是 {offset:.0f}，sigma={sigma:.0f}，真值每个分量都是 {-2 * offset:.0f}。"
              f"每个 D 采 {n} 个样本，每个样本各给出一个估计，统计这些估计在每个分量上的方差再对分量取平均。重参数化的梯度由 autograd 穿过参考实现的 reparameterize 得到；"
              "得分函数估计是 f(z)·(z-mu)/sigma^2；“减基线”是把 f(z) 换成 f(z) 减去它的期望",
        "维度 D": dims,
        **{f"{k}：单样本估计的方差": [float(f"{v:.4g}") for v in vs] for k, vs in var.items()},
        "方差之比（得分函数 / 重参数化）": [round(a / b, 1) for a, b in zip(var["得分函数"], var["重参数化"])],
        "方差之比（得分函数减基线 / 重参数化）": [round(a / b, 1) for a, b in zip(var["得分函数减基线"], var["重参数化"])],
        "重参数化方差的理论值 4·sigma^2": 4 * sigma ** 2,
        "三种估计的平均值（D=16，依次是重参数化、得分函数、减基线；真值 -2）": [r3(v) for v in bias[dims.index(16)]],
        "核对闭式：mu 等于 a、D=16 时得分函数估计的方差": {"实测": round(check, 1), "理论值 sigma^2·(D+2)(D+4)": sigma ** 2 * (d0 + 2) * (d0 + 4)},
        "用梯度下降只更新 mu": {
            "说明": f"D={d}，学习率 {lr}，从 mu=0 出发走 {steps} 步，每步 1 个样本（另注明的除外），{trials} 次独立重复取中位数；误差是 mu 到 a 的均方根距离，起点是 {offset:.0f}。"
                  "“减基线”这里用的是 f 的真实期望",
            **{f"{k}：第 {s_} 步的误差": float(f"{v[s_]:.3g}") for k, v in curves.items() for s_ in (300, 1000)},
            f"学习率加大到 {big_lr} 时": {"重参数化：第 300 步的误差": float(f"{big['重参数化'][0][300]:.3g}"), "得分函数：误差超过 100（发散）的步数": big["得分函数"][1]},
        },
        "在训练好的 VAE 上": {
            "说明": "图 2 里 β=0.03、种子 0 的 VAE，固定 64 张图，重复采样 400 次噪声，比较重建项对编码器输出的均值 mu（64x16 个数）的两种梯度估计；信噪比是 |平均梯度| 除以单次估计的标准差，对所有分量取中位数",
            "方差之比（得分函数 / 重参数化，各分量方差之和的比）": round(float(sf.var(0).sum() / rp.var(0).sum()), 0),
            "重参数化：单次估计的信噪比": r3((truth.abs() / rp.std(0)).median()),
            "得分函数：单次估计的信噪比": r4((truth.abs() / sf.std(0)).median()),
        },
    }

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9), layout="constrained")
    bins = np.linspace(-60, 56, 117)
    ax[0].hist(hist[1], bins=bins, color=S.ORANGE, alpha=0.6, label="得分函数")
    ax[0].hist(hist[0], bins=bins, color=S.BLUE, alpha=0.75, label="重参数化")
    ax[0].axvline(-2 * offset, color=S.INK, lw=1.2, ls="--", label="真值 -2")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("单个样本给出的梯度估计（第一个分量，D=16）")
    ax[0].set_ylabel("样本数（对数）")
    ax[0].set_title("两种估计都以真值为中心，宽度差很多")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    S.clean(ax[0])
    for name, color, style in (("得分函数", S.ORANGE, "s-"), ("得分函数减基线", S.PURPLE, "^-"), ("重参数化", S.BLUE, "o-")):
        ax[1].plot(dims, var[name], style, color=color, lw=1.8, ms=5, label=name)
    ax[1].set_xscale("log", base=2)
    ax[1].set_yscale("log")
    ax[1].set_xticks(dims, [str(d_) for d_ in dims])
    ax[1].minorticks_off()
    ax[1].set_xlabel("隐变量维度 D")
    ax[1].set_ylabel("单样本梯度估计的方差")
    ax[1].set_title("得分函数的方差随维度涨，重参数化不变")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    S.clean(ax[1])
    for name, color, style in (("得分函数", S.ORANGE, "-"), ("得分函数减基线", S.PURPLE, "-"), ("得分函数，64 个样本平均", S.ORANGE, ":"), ("重参数化", S.BLUE, "-")):
        ax[2].plot(curves[name], style, color=color, lw=1.8, label=name)
    ax[2].set_yscale("log")
    ax[2].set_yticks([0.05, 0.1, 0.2, 0.5, 1.0], ["0.05", "0.1", "0.2", "0.5", "1"])
    ax[2].minorticks_off()
    ax[2].set_xlabel("更新步数（每步 1 个样本，另注明的除外）")
    ax[2].set_ylabel("mu 到目标的均方根距离")
    ax[2].set_title("同样的学习率，得分函数估计降得慢")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    S.save(fig, CH, "fig03_两种梯度估计的方差")


# ---------------------------------------------------------------- 图 4：二维数据上的 GAN，模式覆盖
GAN_K, GAN_R, GAN_SD = 8, 2.0, 0.1
GAN_C = torch.tensor([[GAN_R * math.cos(2 * math.pi * i / GAN_K), GAN_R * math.sin(2 * math.pi * i / GAN_K)] for i in range(GAN_K)])


def gan_real(n, g):
    return GAN_C[torch.randint(0, GAN_K, (n,), generator=g)] + GAN_SD * torch.randn(n, 2, generator=g)


def gan_stats(x):
    d, idx = torch.cdist(x, GAN_C).min(1)
    good = d < 3 * GAN_SD
    share = torch.bincount(idx[good], minlength=GAN_K).float() / len(x)
    return float(good.float().mean()), int((share > 0.02).sum())


def train_gan(seed, lr_g, lr_d, steps=2000, every=100):
    g = gen(seed)
    torch.manual_seed(seed)
    G, D = mlp([2, 64, 64, 2]), mlp([2, 64, 64, 1])
    og = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    zfix = torch.randn(2000, 2, generator=g)
    ones, zeros = torch.ones(256, 1), torch.zeros(256, 1)
    hist, lg_tail = [], []
    for step in range(1, steps + 1):
        fake = G(torch.randn(256, 2, generator=g)).detach()
        ld = F.binary_cross_entropy_with_logits(D(gan_real(256, g)), ones) + F.binary_cross_entropy_with_logits(D(fake), zeros)
        od.zero_grad(set_to_none=True)
        ld.backward()
        od.step()
        lg = F.binary_cross_entropy_with_logits(D(G(torch.randn(256, 2, generator=g))), ones)   # 非饱和的生成器损失 -log D(G(z))
        og.zero_grad(set_to_none=True)
        lg.backward()
        og.step()
        if step > steps - 100:
            lg_tail.append(float(lg.detach()))
        if step % every == 0:
            with torch.no_grad():
                hist.append(gan_stats(G(zfix)))
    with torch.no_grad():
        return hist, G(zfix).numpy(), float(np.mean(lg_tail))


def fig_gan():
    settings = {"两边学习率都是 0.001": (1e-3, 1e-3, range(5)), "判别器学习率降到 0.0001": (1e-3, 1e-4, range(5)), "生成器学习率降到 0.0001": (1e-4, 1e-3, range(3))}
    runs = {name: [train_gan(seed, lg, ld) for seed in seeds] for name, (lg, ld, seeds) in settings.items()}
    steps = list(range(100, 2001, 100))
    NUM["GAN 的模式覆盖"] = {
        "说明": f"二维数据：{GAN_K} 个高斯团均匀排在半径 {GAN_R:.0f} 的圆上，每团标准差 {GAN_SD}。生成器 2-64-64-2，判别器 2-64-64-1（ReLU），噪声是 2 维标准高斯；"
              "生成器用非饱和损失 -log D(G(z))，Adam（beta1=0.5），批大小 256，判别器和生成器每步各更新一次，训练 2000 步。"
              "每 100 步用同一批 2000 个噪声生成样本：落在某个团中心 3 倍标准差以内算“高质量”；一个团拿到的高质量样本超过全部样本的 2% 算“被覆盖”",
        **{name: {
            "各种子最终覆盖的模式数": [r[0][-1][1] for r in rs],
            "各种子最终的高质量样本比例": [r3(r[0][-1][0]) for r in rs],
            "各种子在第 200 步覆盖的模式数": [r[0][1][1] for r in rs],
            "各种子最后 100 步的生成器损失平均": [r3(r[2]) for r in rs],
            "最后 1000 步里覆盖数的最小值和最大值（所有种子）": [int(min(h[1] for r in rs for h in r[0][10:])), int(max(h[1] for r in rs for h in r[0][10:]))],
        } for name, rs in runs.items()},
    }
    # 两种生成器损失对判别器输出（logit）的梯度大小
    probs = [0.5, 0.1, 0.01, 0.001]
    grads = []
    for p in probs:
        logit = torch.tensor(math.log(p / (1 - p)), requires_grad=True)
        sat = torch.autograd.grad(torch.log(1 - torch.sigmoid(logit)), logit)[0]
        non = torch.autograd.grad(-torch.log(torch.sigmoid(logit)), logit)[0]
        grads.append([abs(float(sat)), abs(float(non))])
    NUM["生成器损失的梯度"] = {
        "说明": "判别器对一个假样本输出 D，两种生成器损失对判别器 logit 的梯度大小（由 autograd 算出）：原始的 log(1-D) 是 D，非饱和的 -log D 是 1-D",
        "D": probs, "原始损失 log(1-D)": [float(f"{v[0]:.3g}") for v in grads], "非饱和损失 -log D": [float(f"{v[1]:.3g}") for v in grads],
        "D=0.01 时两者之比": round(grads[2][1] / grads[2][0], 0),
        "判别器对真假样本都输出 0.5 时的生成器损失 ln 2": r3(math.log(2)),
    }

    names = list(settings)
    fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.9), gridspec_kw={"width_ratios": [1, 1, 1, 1.5]}, layout="constrained")
    real = gan_real(500, gen(5)).numpy()
    for a, pts, title, color in ((ax[0], real, "真实数据：8 个团", S.MUTED), (ax[1], runs[names[0]][0][1], "两边学习率相同，2000 步后", S.BLUE),
                                 (ax[2], runs[names[1]][0][1], "判别器学习率降到 1/10，2000 步后", S.RED)):
        a.scatter(real[:, 0], real[:, 1], s=4, color="#c3c8d0")
        if pts is not real:
            a.scatter(pts[:600, 0], pts[:600, 1], s=5, color=color, alpha=0.6)
        a.set_xlim(-3.2, 3.2)
        a.set_ylim(-3.2, 3.2)
        a.set_aspect("equal")
        a.set_title(title)
        S.image_axes(a)
    for name, color, style in zip(names, (S.BLUE, S.RED, S.GREEN), ("-", "-", "--")):
        for i, r in enumerate(runs[name]):
            ax[3].plot(steps, [h[1] + (names.index(name) - 1) * 0.07 for h in r[0]], style, color=color, lw=1.2, alpha=0.75, label=name if i == 0 else None)
    ax[3].set_xlabel("训练步数")
    ax[3].set_ylabel("覆盖的模式数（共 8 个）")
    ax[3].set_ylim(-0.5, 8.6)
    ax[3].set_title("每条线一个种子：判别器学得慢时反复崩塌")
    ax[3].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[3])
    S.save(fig, CH, "fig04_GAN的模式覆盖")


# ---------------------------------------------------------------- 对比学习的公共部分
CON_KW = dict(classes=16, noise=0.2)
AUG0 = dict(shift=4, noise=0.1, jitter=0.1, rot=0.0)
N_LABEL, N_TEST, POOL = 320, 2000, 4096


def roll_each(x, dx, dy):
    """每张图各自循环平移 (dx, dy) 个像素。"""
    n, idx = x.shape[0], torch.arange(SIZE)
    ix = (idx[None, :] - dx[:, None]) % SIZE
    iy = (idx[None, :] - dy[:, None]) % SIZE
    x = torch.gather(x, 2, ix[:, None, :].expand(n, SIZE, SIZE))
    return torch.gather(x, 1, iy[:, :, None].expand(n, SIZE, SIZE))


def augment(x, g, shift=4, noise=0.1, jitter=0.1, rot=0.0):
    n = x.shape[0]
    if shift > 0:
        d = torch.randint(-shift, shift + 1, (n, 2), generator=g)
        x = roll_each(x, d[:, 0], d[:, 1])
    if rot > 0:
        x = torch.where((torch.rand(n, generator=g) < rot)[:, None, None], torch.rot90(x, 1, (1, 2)), x)
    if jitter > 0:
        x = x + (torch.rand(n, 1, 1, generator=g) * 2 - 1) * jitter
    if noise > 0:
        x = x + noise * torch.randn(x.shape, generator=g)
    return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = mlp([PIX, 128, 64], last_relu=True)   # 输出 h：下游用的表示
        self.head = mlp([64, 64, 32])                      # 投影头，输出 z：只给对比损失用

    def forward(self, x):
        h = self.body(x.flatten(1) - 0.5)
        return h, self.head(h)


def probe(ftr, ytr, fte, yte, lam=1e-3, classes=16):
    """线性探测：特征标准化后，用带 L2 的最小二乘拟合 one-hot 标签，返回测试精度。"""
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    a = torch.cat([(ftr - mu) / sd, torch.ones(len(ftr), 1)], 1).double()
    b = torch.cat([(fte - mu) / sd, torch.ones(len(fte), 1)], 1).double()
    w = torch.linalg.solve(a.T @ a + lam * len(a) * torch.eye(a.shape[1], dtype=torch.float64), a.T @ F.one_hot(ytr, classes).double())
    return float(((b @ w).argmax(1) == yte).float().mean())


_g = gen(77)
XTR, _, YTR = stripes(N_LABEL, _g, **CON_KW)
XTE, _, YTE = stripes(N_TEST, _g, **CON_KW)


@torch.no_grad()
def con_eval(net):
    htr, ztr = net(XTR)
    hte, zte = net(XTE)
    return probe(htr, YTR, hte, YTE), probe(ztr, YTR, zte, YTE)


def simclr(seed, tau=0.2, batch=256, chunk=None, steps=300, aug=None, loss_fn=None):
    """chunk 给定时，把一批拆成每 chunk 对一组，各组内部算 NT-Xent 再平均：批大小不变，每个锚点的负样本数变成 2*chunk-2。"""
    aug, loss_fn = aug or AUG0, loss_fn or hw.nt_xent_loss
    g = gen(1000 + seed)
    torch.manual_seed(seed)
    data = stripes(POOL, g, **CON_KW)[0]
    net = Encoder()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    t0, last = time.perf_counter(), 0.0
    for _ in range(steps):
        x = data[torch.randint(0, POOL, (batch,), generator=g)]
        z1, z2 = net(augment(x, g, **aug))[1], net(augment(x, g, **aug))[1]
        if chunk and chunk < batch:
            loss = torch.stack([loss_fn(z1[i:i + chunk], z2[i:i + chunk], tau) for i in range(0, batch, chunk)]).mean()
        else:
            loss = loss_fn(z1, z2, tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach())
    return net, last, time.perf_counter() - t0


_RUNS: dict = {}


def run_default(seed):
    if seed not in _RUNS:
        net, loss, sec = simclr(seed)
        _RUNS[seed] = (net, loss, con_eval(net))
        TIMES.setdefault("simclr", []).append(sec)
    return _RUNS[seed]


@torch.no_grad()
def geometry(net, tau):
    g = gen(4242)
    x, y = XTE[:256], YTE[:256]
    z1 = F.normalize(net(augment(x, g, **AUG0))[1], dim=1)
    z2 = F.normalize(net(augment(x, g, **AUG0))[1], dim=1)
    same = (y[:, None] == y[None, :])
    eye = torch.eye(256, dtype=torch.bool)
    s11, s12 = z1 @ z1.T, z1 @ z2.T
    neg = torch.cat([s11.masked_fill(eye, float("-inf")), s12.masked_fill(eye, float("-inf"))], 1) / tau
    w = torch.softmax(neg, 1)
    return [float(s12.diag().mean()), float(s11[same & ~eye].mean()), float(s11[~same].mean()), float(w.sort(1, descending=True)[0][:, :10].sum(1).mean())]


# ---------------------------------------------------------------- 图 5：温度、负样本个数、批大小
def fig_temperature():
    taus = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    tau_rows = {}
    for tau in taus:
        rows = []
        for seed in SEEDS:
            net, loss, acc = run_default(seed) if tau == 0.2 else (lambda r: (r[0], r[1], con_eval(r[0])))(simclr(seed, tau=tau))
            rows.append([acc[0], acc[1], loss, *geometry(net, tau)])
        tau_rows[tau] = np.array(rows)
    chunks = [2, 8, 32, 256]
    neg_rows = {}
    for c in chunks:
        neg_rows[c] = np.array([run_default(s)[2][0] if c == 256 else con_eval(simclr(s, chunk=c)[0])[0] for s in SEEDS])
    combos = [(16, 300), (64, 300), (256, 300), (16, 4800)]
    bat_rows = {}
    for b, steps in combos:
        bat_rows[(b, steps)] = np.array([run_default(s)[2][0] if (b, steps) == (256, 300) else con_eval(simclr(s, batch=b, steps=steps)[0])[0] for s in SEEDS])
    cond = (DATA_NOTE + f"。16 类，像素噪声标准差 0.2（特意加大，让各种设置能拉开差距）。{POOL} 张无标签图做对比学习：编码器是 256-128-64 的全连接网络（输出 h），"
            "投影头 64-64-32（输出 z），损失是参考实现的 nt_xent_loss，Adam，学习率 0.002。默认设置：温度 0.2，批大小 256，训练 300 步，"
            "两个视图各自做循环平移（x、y 各在 -4 到 4 px 内随机）、亮度抖动（整图加 -0.1 到 0.1 的常数）、加标准差 0.1 的噪声。"
            f"线性探测：冻结编码器，用 {N_LABEL} 张带标签的图在 h 上拟合线性分类器（带 L2 的最小二乘），在另外 {N_TEST} 张图上算精度；3 个种子")
    m = {t: v.mean(0) for t, v in tau_rows.items()}
    NUM["温度"] = {
        "说明": cond + "。后四项在 256 张测试图的两个增强视图上统计，相似度是归一化后的 z 的余弦相似度；“最难的 10 个”指每个锚点的 510 个负样本里相似度最高的 10 个，按 softmax(相似度/温度) 算它们占全部负样本权重的比例",
        "温度": taus,
        "线性探测精度（投影前 h）": [r3(m[t][0]) for t in taus],
        "线性探测精度（投影后 z）": [r3(m[t][1]) for t in taus],
        "各种子的精度（h，每行一个温度）": [[r3(x) for x in tau_rows[t][:, 0]] for t in taus],
        "训练结束时的损失": [r3(m[t][2]) for t in taus],
        "正样本对的平均相似度": [r3(m[t][3]) for t in taus],
        "同类不同图的平均相似度": [r3(m[t][4]) for t in taus],
        "不同类的平均相似度": [r3(m[t][5]) for t in taus],
        "最难的 10 个负样本占全部负样本权重的比例": [r3(m[t][6]) for t in taus],
        "10 个负样本在 510 个里按个数占的比例": r3(10 / 510),
        "所有向量都相同时的损失 ln(2B-1)，B=256": r3(math.log(511)),
    }
    NUM["负样本个数"] = {
        "说明": cond + "。批大小保持 256 不变，把每批拆成若干组，NT-Xent 只在组内计算再对各组取平均：每组 c 对图时，每个锚点有 2c-2 个负样本。这样每步用到的图和更新次数都不变，只改负样本个数",
        "每组的图对数 c": chunks, "每个锚点的负样本个数": [2 * c - 2 for c in chunks],
        "线性探测精度（h）": [r3(neg_rows[c].mean()) for c in chunks],
        "各种子的精度（每行一个 c）": [[r3(x) for x in neg_rows[c]] for c in chunks],
    }
    NUM["批大小与训练步数"] = {
        "说明": cond + "。这里整批一起算 NT-Xent（负样本个数是 2B-2），改批大小 B 和训练步数",
        **{f"批大小 {b}，训练 {s} 步（共 {b * s} 对图）": {"线性探测精度（h）": r3(bat_rows[(b, s)].mean()), "各种子": [r3(x) for x in bat_rows[(b, s)]]} for b, s in combos},
    }

    fig, ax = plt.subplots(1, 4, figsize=(16.5, 3.9), layout="constrained")
    for t in taus:
        ax[0].plot([t] * 3, tau_rows[t][:, 0], "o", color=S.BLUE, ms=3.5, alpha=0.35)
    ax[0].plot(taus, [m[t][0] for t in taus], "o-", color=S.BLUE, lw=1.8, ms=5, label="投影前的 h（浅色点是各种子）")
    ax[0].plot(taus, [m[t][1] for t in taus], "s--", color=S.ORANGE, lw=1.4, ms=4, label="投影后的 z")
    ax[0].set_xscale("log")
    ax[0].set_xticks(taus, [str(t) for t in taus])
    ax[0].minorticks_off()
    ax[0].set_xlabel("温度")
    ax[0].set_ylabel("线性探测精度")
    ax[0].set_ylim(0, 1.03)
    ax[0].set_title("温度 0.05 到 0.2 都行，到 1 以上学不好")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[0])
    ax[1].plot(taus, [m[t][6] for t in taus], "o-", color=S.RED, lw=1.8, ms=5, label="训练后的模型上实测")
    ax[1].axhline(10 / 510, color=S.MUTED, lw=1.2, ls="--", label="权重平均分时（10/510）")
    ax[1].set_xscale("log")
    ax[1].set_xticks(taus, [str(t) for t in taus])
    ax[1].minorticks_off()
    ax[1].set_xlabel("温度")
    ax[1].set_ylabel("最难的 10 个负样本拿到的权重占比")
    ax[1].set_ylim(0, 1.03)
    ax[1].set_title("温度越低，权重越集中在最像的负样本上")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[1])
    negs = [2 * c - 2 for c in chunks]
    for c, n_ in zip(chunks, negs):
        ax[2].plot([n_] * 3, neg_rows[c], "o", color=S.GREEN, ms=3.5, alpha=0.35)
    ax[2].plot(negs, [neg_rows[c].mean() for c in chunks], "o-", color=S.GREEN, lw=1.8, ms=5)
    ax[2].set_xscale("log")
    ax[2].set_xticks(negs, [str(n_) for n_ in negs])
    ax[2].minorticks_off()
    ax[2].set_xlabel("每个锚点的负样本个数（批大小固定 256）")
    ax[2].set_ylabel("线性探测精度")
    ax[2].set_ylim(0, 1.03)
    ax[2].set_title("负样本从 2 个加到 62 个有用，再多差别很小")
    S.clean(ax[2])
    names = ["批 16\n300 步", "批 64\n300 步", "批 256\n300 步", "批 16\n4800 步"]
    vals = [bat_rows[c].mean() for c in combos]
    bars = ax[3].bar(names, vals, color=[S.MUTED, S.MUTED, S.BLUE, S.ORANGE], alpha=0.88, width=0.6)
    for i, (b, c) in enumerate(zip(bars, combos)):
        ax[3].plot([i] * 3, bat_rows[c], "o", color=S.INK, ms=3, alpha=0.5)
        ax[3].text(b.get_x() + b.get_width() / 2, vals[i] + 0.035, f"{vals[i]:.2f}", ha="center", fontsize=9)
    ax[3].set_ylim(0, 1.12)
    ax[3].set_ylabel("线性探测精度")
    ax[3].set_title("批 16 多走 16 倍的步数，不比批 256 差")
    S.clean(ax[3])
    ax[3].grid(axis="x", visible=False)
    S.save(fig, CH, "fig05_温度与负样本个数")


# ---------------------------------------------------------------- 图 6：数据增强的强弱、投影头、和别的做法比
def bug_loss(kind):
    def fn(z1, z2, temperature=0.2):
        n = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        if kind != "不做归一化":
            z = F.normalize(z, dim=1)
        sim = z @ z.T / temperature
        if kind != "不去掉对角线":
            sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool), float("-inf"))
        a, b = torch.arange(n, 2 * n), torch.arange(0, n)
        if kind == "正样本错位一行":
            a, b = a.roll(1), b.roll(1)
        return F.cross_entropy(sim, torch.cat([a, b]))
    return fn


def run_checker(reparam_fn, loss_fn):
    """把两个函数交给本章的验收脚本 tests/check_progress.py，返回“通过的项数/总项数”。

    验收脚本自己会设随机种子，这里跑完把全局随机数状态恢复原样，不影响后面的实验。
    """
    state = torch.get_rng_state()
    spec = importlib.util.spec_from_file_location("ch13_check", S.MAIN / CH / "tests" / "check_progress.py")
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    chk.load = lambda: types.SimpleNamespace(reparameterize=reparam_fn, nt_xent_loss=loss_fn)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        chk.main()
    torch.set_rng_state(state)
    return re.findall(r"通过 (\d+/\d+)", buf.getvalue())[-1]


def passes_check(fn):
    """把一个 NT-Xent 的写法（配上参考实现的 reparameterize）交给验收脚本，返回“通过的项数/总项数”。"""
    result = run_checker(hw.reparameterize, fn)
    # 这个函数以前在这里设过一次种子、取过两次随机数。保留这一行，后面各组实验的随机数状态和以前相同，讲义里的数字不用跟着变
    torch.manual_seed(13); torch.randn(4, 6); torch.randn(4, 6)
    return result


# ---------------------------------------------------------------- 图 6（续）
def fig_augment():
    configs = [
        ("不做增强（两个视图相同）", "太弱", dict(shift=0, noise=0.0, jitter=0.0)),
        ("不平移，只加噪声和亮度抖动", "太弱", dict(shift=0, noise=0.1, jitter=0.1)),
        ("去掉亮度抖动", "太弱", dict(shift=4, noise=0.1, jitter=0.0)),
        ("平移只有 ±1 px", "合适", dict(shift=1, noise=0.1, jitter=0.1)),
        ("默认（平移 ±4 px，噪声 0.1，亮度 ±0.1）", "合适", AUG0),
        ("噪声加到 0.2", "太强", dict(shift=4, noise=0.2, jitter=0.1)),
        ("噪声加到 0.4", "太强", dict(shift=4, noise=0.4, jitter=0.1)),
        ("亮度抖动加到 ±1", "太强", dict(shift=4, noise=0.1, jitter=1.0)),
        ("再加随机旋转 90°（一半的图）", "改变类别", dict(shift=4, noise=0.1, jitter=0.1, rot=0.5)),
    ]
    res = {}
    for name, _, aug in configs:
        rows = []
        for seed in SEEDS:
            if aug is AUG0:
                _, loss, acc = run_default(seed)
            else:
                net, loss, _ = simclr(seed, aug={**AUG0, **aug})
                acc = con_eval(net)
            rows.append([acc[0], acc[1], loss])
        res[name] = np.array(rows)

    # 对照：像素、随机初始化的编码器、自编码器的编码
    base = {"原始像素": [probe(XTR.flatten(1), YTR, XTE.flatten(1), YTE)] * 3}
    rnd, aec, ae_mse = [], [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        rnd.append(con_eval(Encoder())[0])
        g = gen(1000 + seed)
        torch.manual_seed(seed)
        data, clean, _ = stripes(POOL, g, **CON_KW)
        enc, dec = mlp([PIX, 128, 64], last_relu=True), mlp([64, 128, PIX])
        opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=2e-3)
        for _ in range(600):
            x = data[torch.randint(0, POOL, (256,), generator=g)].flatten(1) - 0.5
            loss = ((dec(enc(x)) - x) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            aec.append(probe(enc(XTR.flatten(1) - 0.5), YTR, enc(XTE.flatten(1) - 0.5), YTE))
            ae_mse.append(float(((dec(enc(data.flatten(1) - 0.5)) + 0.5 - clean.flatten(1)) ** 2).mean()))
    base.update({"随机初始化的编码器": rnd, "自编码器的编码（64 维）": aec, "对比学习（默认设置）": [run_default(s)[2][0] for s in SEEDS]})

    # 标签个数：每类 2、4、10、20 张带标签的图
    per_class = [2, 4, 10, 20]
    yb = torch.arange(16).repeat_interleave(max(per_class))
    xb = stripes(len(yb), gen(78), y=yb, **CON_KW)[0]

    def ncm(ftr, ytr, fte):
        ftr, fte = F.normalize(ftr, dim=1), F.normalize(fte, dim=1)
        cents = F.normalize(torch.stack([ftr[ytr == c].mean(0) for c in range(16)]), dim=1)
        return float(((fte @ cents.T).argmax(1) == YTE).float().mean())

    def supervised(seed, xs, ys, use_aug):
        g = gen(3000 + seed)
        torch.manual_seed(seed)
        net, head = Encoder(), nn.Linear(64, 16)
        opt = torch.optim.Adam([*net.body.parameters(), *head.parameters()], lr=2e-3)
        for _ in range(300):
            idx = torch.randint(0, len(xs), (min(64, len(xs)),), generator=g)
            inp = augment(xs[idx], g, **AUG0) if use_aug else xs[idx]
            loss = F.cross_entropy(head(net(inp)[0]), ys[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            return float((head(net(XTE)[0]).argmax(1) == YTE).float().mean())

    lab_names = ["对比学习的 h，最近类中心", "对比学习的 h，线性探测（最小二乘）", "直接监督训练，用同样的增强", "直接监督训练，不做增强"]
    lab = {n: [] for n in lab_names}
    for n_per in per_class:
        idx = torch.cat([torch.arange(c * max(per_class), c * max(per_class) + n_per) for c in range(16)])
        xs, ys = xb[idx], yb[idx]
        rows = {n: [] for n in lab_names}
        for seed in SEEDS:
            net = run_default(seed)[0]
            with torch.no_grad():
                hs, ht = net(xs)[0], net(XTE)[0]
            rows[lab_names[0]].append(ncm(hs, ys, ht))
            rows[lab_names[1]].append(probe(hs, ys, ht, YTE))
            rows[lab_names[2]].append(supervised(seed, xs, ys, True))
            rows[lab_names[3]].append(supervised(seed, xs, ys, False))
        for n in lab_names:
            lab[n].append(rows[n])
    flat = stripes(POOL, gen(1000), **CON_KW)[1]

    bugs = {}
    for kind in ("不去掉对角线", "不做归一化", "正样本错位一行"):
        fn = bug_loss(kind)
        ok = passes_check(fn)
        rows = [(lambda r: [con_eval(r[0])[0], r[1]])(simclr(seed, loss_fn=fn)) for seed in SEEDS[:2]]
        bugs[kind] = {"验收": ok, "线性探测精度（h，2 个种子）": [r3(r[0]) for r in rows], "训练结束时的损失": [r3(r[1]) for r in rows]}

    cond = NUM["温度"]["说明"].split("。后四项")[0]
    NUM["数据增强的强弱"] = {
        "说明": cond + "。每一行只改增强，其余同默认设置。每项依次是线性探测精度（h）、线性探测精度（z）、训练结束时的损失，3 个种子取平均。"
              "随机旋转 90° 会把一类条纹变成另一类（竖变横，两条对角线互换），16 类两两配成 8 对",
        **{name: {"分组": grp, "平均": [r3(x) for x in res[name].mean(0)], "各种子的精度（h）": [r3(x) for x in res[name][:, 0]]} for name, grp, _ in configs},
        "16 类随机猜的精度": r4(1 / 16),
        "旋转把 16 类配成 8 对后，完全分不出对内两类时的精度上限": 0.5,
    }
    NUM["和别的做法比"] = {
        "说明": cond + "。都是线性探测。自编码器用同样的 256-128-64 编码器加 64-128-256 解码器，在同一批无标签图上训练 600 步",
        **{k: {"平均": r3(np.mean(v)), "各种子": [r3(x) for x in v]} for k, v in base.items()},
        "自编码器的重建误差（和不带噪声的图比，均方）": r4(np.mean(ae_mse)),
        "只输出平均图时的重建误差": r4((flat - flat.mean(0)).pow(2).mean()),
    }
    NUM["标签个数"] = {
        "说明": cond + "。这里换一批带标签的图，每类正好 n 张（16 类）。前两行用默认设置训练好的对比学习编码器（没用到标签），只在 h 上放一个分类器："
              "“最近类中心”是把每类带标签图的 h 归一化后取平均当类中心，测试图归到余弦相似度最高的类；线性探测同上。后两行不做对比学习，"
              "用同样结构的编码器加一个线性层，只用这些带标签的图从头训练（Adam，学习率 0.002，批大小不超过 64，300 步），“同样的增强”指默认设置里的平移、亮度抖动和噪声。3 个种子取平均",
        "每类带标签的图数": per_class, "带标签的图总数": [16 * n for n in per_class],
        **{n: [r3(np.mean(v)) for v in lab[n]] for n in lab_names},
        **{f"各种子：{n}（每行一个标签数）": [[r3(x) for x in v] for v in lab[n]] for n in lab_names},
        "h 的维数（带标签的图数接近它时，最小二乘的线性探测最差）": 64,
    }
    ref_pass = passes_check(hw.nt_xent_loss)
    NUM["NT-Xent 的三种写错"] = {"说明": f"把参考实现改错一处再训练，其余同默认设置。“验收”是把这种写法交给本章的验收脚本 tests/check_progress.py（共 {ref_pass.split('/')[1]} 项）之后通过的项数",
                            **bugs, "参考实现": {"验收": ref_pass, "线性探测精度（h，3 个种子）": [r3(run_default(s)[2][0]) for s in SEEDS]}}

    colors = {"太弱": S.ORANGE, "合适": S.GREEN, "太强": S.RED, "改变类别": S.PURPLE}
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.3), gridspec_kw={"width_ratios": [1.5, 0.9, 1.1]}, layout="constrained")
    names = [c[0] for c in configs][::-1]
    for i, name in enumerate(names):
        grp = next(c[1] for c in configs if c[0] == name)
        v = res[name][:, 0]
        ax[0].barh(i, v.mean(), color=colors[grp], alpha=0.85, height=0.62)
        ax[0].plot(v, [i] * 3, "o", color=S.INK, ms=3, alpha=0.55)
        ax[0].text(max(v.max(), v.mean()) + 0.02, i, f"{v.mean():.2f}", va="center", fontsize=9)
    ax[0].axvline(np.mean(rnd), color=S.MUTED, lw=1.2, ls="--")
    ax[0].text(np.mean(rnd) + 0.01, len(names) - 0.42, "随机初始化的编码器", color=S.MUTED, fontsize=8.5)
    ax[0].set_yticks(range(len(names)), names, fontsize=9)
    ax[0].set_xlim(0, 1.12)
    ax[0].set_ylim(-0.6, len(names) - 0.1)
    ax[0].set_xlabel("线性探测精度（h；黑点是各种子）")
    ax[0].set_title("增强太弱、太强、改变类别，都会掉下来")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85) for c in colors.values()]
    ax[0].legend(handles, list(colors), fontsize=8.5, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=4)
    S.clean(ax[0])
    ax[0].grid(axis="y", visible=False)
    pick = [("默认", configs[4][0]), ("不平移", configs[1][0]), ("不做增强", configs[0][0]), ("加旋转 90°", configs[8][0])]
    x = np.arange(len(pick))
    for j, (label, color) in enumerate((("投影前的 h", S.BLUE), ("投影后的 z", S.ORANGE))):
        vals = [res[n][:, j].mean() for _, n in pick]
        bars = ax[1].bar(x + (j - 0.5) * 0.38, vals, 0.36, color=color, alpha=0.88, label=label)
        for b, v in zip(bars, vals):
            ax[1].text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.5)
    ax[1].set_xticks(x, [p[0] for p in pick], fontsize=9)
    ax[1].set_ylim(0, 1.12)
    ax[1].set_ylabel("线性探测精度")
    ax[1].set_title("增强有问题时，投影前的 h 明显比 z 好")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[1])
    ax[1].grid(axis="x", visible=False)
    xs_ = [16 * n for n in per_class]
    short = ["对比学习 + 最近类中心", "对比学习 + 线性探测", "监督训练，同样的增强", "监督训练，不做增强"]
    for n, name, color, style in zip(lab_names, short, (S.GREEN, S.GREEN, S.PURPLE, S.MUTED), ("o-", "o:", "s-", "s--")):
        ax[2].plot(xs_, [np.mean(v) for v in lab[n]], style, color=color, lw=1.7, ms=5, label=name)
    ax[2].set_xscale("log")
    ax[2].set_xticks(xs_, [str(x_) for x_ in xs_])
    ax[2].minorticks_off()
    ax[2].set_ylim(0, 1.05)
    ax[2].set_xlabel("带标签的图数（16 类，每类 2、4、10、20 张）")
    ax[2].set_ylabel("测试精度")
    ax[2].set_title("带同样增强的监督训练最高，不做增强的最低")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    S.save(fig, CH, "fig06_增强强弱与线性探测")


# ---------------------------------------------------------------- 不画图的数字：自检用的数值、显存估算、扩散的前向过程
def numbers_misc():
    eye = torch.eye(6)[:4]
    tau, b = 0.2, 4
    ref = float(hw.nt_xent_loss(eye, eye, tau))
    g = gen(1)
    z1, z2 = torch.randn(8, 16, generator=g), torch.randn(8, 16, generator=g)
    same = torch.ones(256, 32)
    NUM["NT-Xent 的自检数值"] = {
        "说明": f"用参考实现算出来、可以拿来核对自己实现的几个数。4 对正样本，z1 和 z2 都取 6 维单位矩阵的前 4 行（正样本相似度 1，负样本相似度 0），温度 {tau}",
        "参考实现的损失": r4(ref),
        "理论值 ln(1 + (2B-2)·exp(-1/τ))": r4(math.log(1 + (2 * b - 2) * math.exp(-1 / tau))),
        "忘了去掉对角线会得到": r4(bug_loss("不去掉对角线")(eye, eye, tau)),
        "它的理论值 ln(2 + (2B-2)·exp(-1/τ))": r4(math.log(2 + (2 * b - 2) * math.exp(-1 / tau))),
        "正样本错位一行会得到": r4(bug_loss("正样本错位一行")(eye, eye, tau)),
        "把 z1、z2 都乘以 10：参考实现的损失变化": float(f"{abs(float(hw.nt_xent_loss(10 * z1, 10 * z2, tau) - hw.nt_xent_loss(z1, z2, tau))):.1g}"),
        "把 z1、z2 都乘以 10：不做归一化的损失从多少变到多少": [r3(bug_loss("不做归一化")(z1, z2, tau)), r3(bug_loss("不做归一化")(10 * z1, 10 * z2, tau))],
        "所有向量都相同（B=256）时参考实现的损失": r3(hw.nt_xent_loss(same, same, tau)),
        "对应的理论值 ln(2B-1)": r3(math.log(2 * 256 - 1)),
    }
    n = 200000
    mu, logvar = torch.full((n, 1), 1.0), torch.full((n, 1), math.log(4.0))
    z = hw.reparameterize(mu, logvar, torch.randn(n, 1, generator=g))
    wrong = mu + torch.exp(logvar) * torch.randn(n, 1, generator=g)

    def wrong_fn(m, lv, eps=None):
        return m + torch.exp(lv) * (torch.randn_like(m) if eps is None else eps)

    sig, m1 = 0.5, 1.0
    q = m1 + sig * torch.randn(n, generator=g)
    mc = (-0.5 * ((q - m1) / sig) ** 2 - math.log(sig) + 0.5 * q ** 2).mean()
    d = torch.distributions.Normal(torch.zeros(3, requires_grad=True), torch.ones(3))
    NUM["重参数化的自检数值"] = {
        "说明": f"mu=1、logvar=ln 4（标准差应为 2），用参考实现的 reparameterize 采 {n} 个样本",
        "样本均值": r3(z.mean()), "样本标准差": r3(z.std()),
        "把 exp(0.5*logvar) 写成 exp(logvar) 时的样本标准差": r3(wrong.std()),
        "把这个写错的版本交给验收脚本，通过的项数": run_checker(wrong_fn, hw.nt_xent_loss),
        "参考实现交给验收脚本，通过的项数": run_checker(hw.reparameterize, hw.nt_xent_loss),
        "torch.distributions.Normal 的 sample() 结果是否带梯度": bool(d.sample().requires_grad),
        "rsample() 结果是否带梯度": bool(d.rsample().requires_grad),
        "KL 闭式核对（一维，mu=1，sigma=0.5）": {"闭式 0.5·(mu^2 + sigma^2 - 1 - ln sigma^2)": r4(0.5 * (m1 ** 2 + sig ** 2 - 1 - math.log(sig ** 2))), f"{n} 个样本的蒙特卡洛估计": r4(mc)},
    }
    try:
        import torchvision
        net = torchvision.models.resnet18(weights=None).eval()
        count = [0]
        hooks = [mod.register_forward_hook(lambda _m, _i, o: count.__setitem__(0, count[0] + o.numel()))
                 for mod in net.modules() if isinstance(mod, (nn.Conv2d, nn.BatchNorm2d))]
        with torch.no_grad():
            net(torch.zeros(1, 3, 224, 224))
        for h in hooks:
            h.remove()
        act_mb = count[0] * 4 / 2 ** 20
        res_params = n_params(net)
    except Exception as exc:  # noqa: BLE001
        print("跳过 ResNet-18 的激活量估算：", exc)
        act_mb, res_params = None, None
    batches = [256, 1024, 4096, 16384]
    NUM["显存估算"] = {
        "说明": "按 float32 每个数 4 字节算，1 MB 取 2^20 字节。相似度矩阵是 2B x 2B。ResNet-18 的数字是 224x224 输入、一张图前向时所有卷积层和归一化层输出的元素数之和，"
              "只是量级估算（没有算反向传播的额外开销，也没有在显卡上实测）",
        "批大小 B": batches,
        "相似度矩阵（MB）": [round((2 * b_) ** 2 * 4 / 2 ** 20, 1) for b_ in batches],
        "ResNet-18 一张图的中间输出（MB）": None if act_mb is None else round(act_mb, 1),
        "ResNet-18 一批两个视图的中间输出（MB）": None if act_mb is None else [round(act_mb * 2 * b_) for b_ in batches],
        "ResNet-18 一批两个视图的中间输出（GB）": None if act_mb is None else [round(act_mb * 2 * b_ / 1024, 1) for b_ in batches],
        "ResNet-18 的参数量": res_params,
        "本章对比学习编码器加投影头的参数量": n_params(Encoder()),
    }
    side, patch, ratio = 224, 16, 0.75
    n_patch = (side // patch) ** 2
    NUM["MAE 的计算量"] = {
        "说明": f"{side}x{side} 的图切成 {patch}x{patch} 的 patch；掩码率取 MAE 论文默认的 75%。自注意力的计算量按 patch 数的平方估算",
        "patch 总数": n_patch, "编码器实际处理的 patch 数": round(n_patch * (1 - ratio)),
        "编码器处理的 patch 占比": 1 - ratio, "编码器自注意力的计算量相对不遮挡时的比例": (1 - ratio) ** 2,
    }
    T = 1000
    beta = np.linspace(1e-4, 0.02, T)
    abar = np.cumprod(1 - beta)
    rng = np.random.default_rng(0)
    x = np.ones(20000)
    snap = {}
    for t in range(1, T + 1):
        x = math.sqrt(1 - beta[t - 1]) * x + math.sqrt(beta[t - 1]) * rng.normal(size=x.shape)
        if t in (500, 1000):
            snap[t] = [float(x.mean()), float(x.std())]
    ts = [1, 100, 250, 500, 750, 1000]
    NUM["扩散的前向过程"] = {
        "说明": f"DDPM 论文里的线性噪声调度：T={T}，beta 从 0.0001 线性加到 0.02。x_t = sqrt(abar_t)·x_0 + sqrt(1-abar_t)·eps",
        "t": ts,
        "信号系数 sqrt(abar_t)": [float(f"{math.sqrt(abar[t - 1]):.3g}") for t in ts],
        "噪声系数 sqrt(1-abar_t)": [float(f"{math.sqrt(1 - abar[t - 1]):.3g}") for t in ts],
        "逐步加噪 20000 个 x_0=1 的样本，核对闭式（每项是均值、标准差）": {f"t={t}：逐步模拟": [r3(v) for v in snap[t]] for t in snap}
        | {f"t={t}：闭式": [r3(math.sqrt(abar[t - 1])), r3(math.sqrt(1 - abar[t - 1]))] for t in snap},
        "训练一步要调用网络的次数": 1, "采样一张图要调用网络的次数（不做加速）": T,
    }


def numbers_demo():
    """演示页 demos/对比损失与温度.html 的引导语里出现的数字，用参考实现再算一遍（演示页自己的计算已用 node 核对过）。"""
    def loss_of(th, tau):
        z = torch.stack([th.cos(), th.sin()], 1)
        return hw.nt_xent_loss(z[0::2], z[1::2], tau)

    def shares(deg, tau, anchor=0):
        th = torch.tensor(deg, dtype=torch.float64) * math.pi / 180
        s = torch.cos(th[anchor] - th) / tau
        s[anchor] = float("-inf")
        p = torch.softmax(s, 0)
        neg = p.clone()
        neg[anchor ^ 1] = 0
        return neg / neg.sum()

    def descend(deg, tau, steps=900, lr=0.5):
        th = (torch.tensor(deg, dtype=torch.float64) * math.pi / 180).requires_grad_(True)
        for _ in range(steps):
            g = torch.autograd.grad(loss_of(th, tau), th)[0]
            th = (th - lr * tau * g).detach().requires_grad_(True)
        return float(loss_of(th, tau))

    a1, a4 = [0, 20, 40, 50, 150, 165, 250, 265], [0, 10, 25, 35, 180, 190, 270, 280]
    same = [60.0] * 8
    near = [60.0 + k for k in range(8)]
    best = torch.tensor([0, 0, 90, 90, 180, 180, 270, 270], dtype=torch.float64) * math.pi / 180
    NUM["演示页的数字"] = {
        "说明": "单位圆上的 4 对点（第 5 步是 2、4、8 对），角度见演示页各步的“载入”按钮；损失由参考实现的 nt_xent_loss 算出，权重占比是 softmax(相似度/温度) 在负样本上归一化",
        "第 1 步：最像的负样本占负样本权重的比例": {f"温度 {t}": r3(shares(a1, t).max()) for t in (0.05, 0.2, 2.0)},
        "第 1 步：6 个负样本平均分时每个占": r3(1 / 6),
        "第 2 步：8 个点全部重合时的损失": {f"温度 {t}": r3(loss_of(torch.tensor(same, dtype=torch.float64) * math.pi / 180, t)) for t in (0.05, 0.2, 2.0)},
        "第 2 步：相邻的点各差 1° 时，优化 900 步前后的损失": [r3(loss_of(torch.tensor(near, dtype=torch.float64) * math.pi / 180, 0.2)), r3(descend(near, 0.2))],
        "第 3 步：打散的 4 对优化前后的损失": [r3(loss_of(torch.tensor([10, 200, 95, 300, 150, 40, 250, 120], dtype=torch.float64) * math.pi / 180, 0.2)), r3(descend([10, 200, 95, 300, 150, 40, 250, 120], 0.2))],
        "第 3 步：4 对各自重合、互相隔 90° 时的损失": {f"温度 {t}": r3(loss_of(best, t)) for t in (0.2, 1.0, 2.0)},
        "第 4 步：相邻那一对的两个点合计占负样本权重的比例": {f"温度 {t}": r3(shares(a4, t)[2:4].sum()) for t in (0.05, 0.2, 1.0)},
        "第 5 步：全部重合时的损失": {f"{n} 对": r3(loss_of(torch.full((2 * n,), 1.0, dtype=torch.float64), 0.2)) for n in (2, 4, 8)},
        "第 5 步：理论值 ln(2N-1)": {f"{n} 对": r3(math.log(2 * n - 1)) for n in (2, 4, 8)},
    }


def main() -> None:
    S.setup()
    t_all = time.perf_counter()
    for fn in (numbers_misc, numbers_demo, fig_bottleneck, fig_vae, fig_gradient, fig_gan, fig_temperature, fig_augment):
        t = time.perf_counter()
        fn()
        TIMES[fn.__name__] = time.perf_counter() - t
        print(f"完成 {fn.__name__}（{TIMES[fn.__name__]:.0f} 秒）", flush=True)
    total = time.perf_counter() - t_all

    def measure():
        two = lambda v: float(f"{v:.2g}")  # noqa: E731
        return {"说明": "在一台 Apple 芯片的 Mac 上用 CPU（2 个线程）测的一次结果，同时还有别的程序在跑，只能看量级。默认沿用这次记录，设置 CH13_RETIME=1 才重测",
                "对比学习训练一次（批大小 256，300 步）的秒数": two(float(np.median(TIMES["simclr"]))),
                "整个配图脚本的秒数": two(total)}

    NUM["耗时"] = S.pinned(CH, "耗时", measure, "CH13_RETIME")
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=2))
    print(f"总耗时 {total:.0f} 秒")


if __name__ == "__main__":
    main()
