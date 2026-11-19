"""第 29 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch29.py

输出：<主分支>/29_扩散模型实战/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。

全部实验用代码生成的数据和很小的网络，在 CPU 上跑，不下载任何数据和权重：
- 二维的 6 团点分布加一个小 MLP：采样器、三种参数化、无分类器引导。这个分布的最优去噪器有解析式，用来把“采样器的误差”和“网络的误差”分开。
- 16x16 的条纹纹理加缺陷和一个 4 万参数的小 UNet：空间条件（控制旁路）、低秩微调、修复式生成、闭环实验。
噪声表和前向过程沿用第 20 章作业的参考实现（linear_beta_schedule、alpha_bars，1000 步，下标从 0 开始）；
参数化换算、DDIM 一步、引导的组合、LoRA 的前向与合并、修复时已知区域的替换调用本章作业的参考实现。
耗时默认沿用上次的记录，设置 CH29_RETIME=1 才重测。
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
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

CH = "29_扩散模型实战"
hw = S.load_solution(CH, "29-1_采样引导与低秩微调.py")
hw20 = S.load_solution("20_合成缺陷与生成模型", "20-1_缺陷合成与扩散前向.py")
NUM: dict = {}
TIMES: dict = {}
T = 1000
ABAR = hw20.alpha_bars(hw20.linear_beta_schedule(T))          # 和第 20 章同一张线性噪声表，abar[0] 是加过一次噪声之后的值
AB = torch.tensor(ABAR, dtype=torch.float32)
SCHED_NOTE = f"噪声表是第 20 章的线性表（{T} 步，beta 从 0.0001 到 0.02，下标从 0 开始）"


def r3(v):
    return round(float(v), 3)


def r2(v):
    return round(float(v), 2)


def r4(v):
    return round(float(v), 4)


def gen(*seed):
    return torch.Generator().manual_seed(int(sum(s * 1000 ** i for i, s in enumerate(seed))))


def temb(t, dim=32):
    """步数的正弦编码。t 是 0 到 T-1 的整数下标。"""
    f = torch.exp(-math.log(1000.0) * torch.arange(dim // 2) / (dim // 2))
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


def timesteps(steps):
    """采样时用到的步数下标：从 T-1 到 0 等间隔取 steps 个。"""
    return np.round(np.linspace(T - 1, 0, steps)).astype(int)


def one_step(x, eps, a_t, a_prev, sampler, g):
    """三种采样器都写成作业的 ddim_step：
    ddim 是 eta=0；post 是 eta=1（DDPM，方差取后验方差）；large 也是 eta=1，但把噪声放大到方差等于 1 - abar_t / abar_prev（DDPM 的另一种常见取法）。"""
    if sampler == "ddim":
        return hw.ddim_step(x, eps, a_t, a_prev)
    noise = torch.randn(x.shape, generator=g)
    if sampler == "large" and a_prev < 1.0:
        noise = noise * math.sqrt((1.0 - a_t) / (1.0 - a_prev))
    return hw.ddim_step(x, eps, a_t, a_prev, 1.0, noise)


@torch.no_grad()
def run_sampler(eps_fn, x, steps, sampler, g, after=None):
    """eps_fn(x, t) 返回噪声预测。after(x, abar) 在每一步之后调用（修复时用来替换已知区域）。"""
    ts = timesteps(steps)
    for i, t in enumerate(ts):
        a_t = float(ABAR[t])
        a_prev = float(ABAR[ts[i + 1]]) if i + 1 < len(ts) else 1.0
        x = one_step(x, eps_fn(x, int(t)), a_t, a_prev, sampler, g)
        if after is not None:
            x = after(x, a_prev)
    return x


# ================================================================ 第一部分：二维点分布
R2D, N2D = 2.0, 4000
ANG = np.deg2rad(np.arange(6) * 60.0)
MU = torch.tensor(np.c_[R2D * np.cos(ANG), R2D * np.sin(ANG)], dtype=torch.float32)
CLS = torch.tensor([0, 1, 2, 0, 1, 2])
NULL = 3                                                            # “没有条件”用的类别编号
DIRS = np.stack([np.cos(np.pi * np.arange(32) / 32), np.sin(np.pi * np.arange(32) / 32)], 1)
NOTE_2D = (f"二维数据：6 个高斯团，团心在半径 {R2D:.0f} 的圆上每隔 60 度一个，团的标准差 0.6；团按 0、1、2、0、1、2 的顺序分成 3 类，每类是相对的两团。"
           f"这个分布的最优去噪器 E[x0 | x_t] 有解析式（表里叫“解析去噪器”）。每次采 {N2D} 个点。" + SCHED_NOTE)
NET_NOTE_2D = ("网络是 3 个隐藏层、宽 128 的 MLP，输入是 x_t、步数的正弦编码和类别的嵌入；Adam，学习率 0.002 余弦下降，批大小 512，训练 2500 步；"
               "训练时每个样本的类别有 15% 的概率换成“没有条件”")


class Mix:
    """6 团高斯混合，以及它在任意噪声水平下的解析量。"""

    def __init__(self, sigma):
        self.sig = sigma

    def sample(self, n, g, cls=None):
        k = torch.randint(0, 6, (n,), generator=g) if cls is None else cls + 3 * torch.randint(0, 2, (n,), generator=g)
        return MU[k] + self.sig * torch.randn(n, 2, generator=g), CLS[k]

    def _logw(self, x, a, cls):
        var = a * self.sig ** 2 + 1.0 - a
        lw = -((x[:, None, :] - math.sqrt(a) * MU[None]) ** 2).sum(-1) / (2 * var)
        if cls is not None:
            lw = lw.masked_fill(CLS[None] != cls[:, None], -float("inf"))
        return lw, var

    def x0(self, x, a, cls=None):
        """最优去噪：E[x0 | x_t]。cls 给定时是条件分布（只有该类的两团）下的。"""
        lw, var = self._logw(x, a, cls)
        post = MU[None] + (math.sqrt(a) * self.sig ** 2 / var) * (x[:, None, :] - math.sqrt(a) * MU[None])
        return (torch.softmax(lw, 1)[..., None] * post).sum(1)

    def eps(self, x, a, cls=None):
        return (x - math.sqrt(a) * self.x0(x, a, cls)) / math.sqrt(1.0 - a)

    def log_p(self, x, cls=None):
        return torch.logsumexp(self._logw(x, 1.0, cls)[0], 1)

    def bayes(self, x):
        lw = self._logw(x, 1.0, None)[0]
        return torch.stack([torch.logsumexp(lw[:, CLS == c], 1) for c in range(3)], 1).argmax(1)

    def blob_stats(self, x, cls=None):
        """把每个点分给最近的团心（给了 cls 就只在该类的两团里找），各团分别统计后取平均。
        返回 [团内标准差（两个坐标平均）, 质心沿半径方向外移的距离, 沿半径方向的标准差, 垂直于半径方向的标准差]，单位是数据坐标。
        相邻的团有重叠时，分给最近的团心会把尾巴切掉，真实样本这样统计出来也小于团的标准差，所以要和真实样本按同样办法统计的值比。"""
        d = ((x[:, None, :] - MU[None]) ** 2).sum(-1)
        if cls is not None:
            d = d.masked_fill(CLS[None] != cls[:, None], float("inf"))
        k = d.argmin(1)
        rows = []
        for b in range(6):
            pts = x[k == b]
            if len(pts) >= 5:
                u = MU[b] / R2D
                c = pts - pts.mean(0)
                rows.append([float(pts.std(0).mean()), float(((pts.mean(0) - MU[b]) * u).sum()), float((c @ u).std()), float((c @ torch.stack([-u[1], u[0]])).std())])
        return np.mean(rows, axis=0)


MIX = Mix(0.6)


def sliced_w(a, b):
    """切片 Wasserstein 距离：往 32 个方向投影，各自排序后逐个相减取绝对值，再平均。两组点数要相同。"""
    a, b = np.asarray(a, np.float64) @ DIRS.T, np.asarray(b, np.float64) @ DIRS.T
    return float(np.abs(np.sort(a, 0) - np.sort(b, 0)).mean())


class MLP(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.cls = nn.Embedding(4, 16)
        self.net = nn.Sequential(nn.Linear(2 + 32 + 16, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))

    def forward(self, x, t, c):
        return self.net(torch.cat([x, temb(t), self.cls(c)], 1))


def train_2d(kind, seed=0, steps=2500, batch=512, p_drop=0.15):
    """kind 是网络预测的量：eps、x0 或 v。损失都是对各自目标的均方误差，每个步数权重相同。"""
    torch.manual_seed(seed)
    g = gen(seed, 29)
    net = MLP()
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x0, c = MIX.sample(batch, g)
        t = torch.randint(0, T, (batch,), generator=g)
        a = AB[t][:, None]
        eps = torch.randn(batch, 2, generator=g)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        c = torch.where(torch.rand(batch, generator=g) < p_drop, torch.full_like(c, NULL), c)
        target = {"eps": eps, "x0": x0, "v": a.sqrt() * eps - (1 - a).sqrt() * x0}[kind]
        loss = ((net(xt, t, c) - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return net.eval()


_CACHE: dict = {}


def cached(key, make):
    if key not in _CACHE:
        _CACHE[key] = make()
    return _CACHE[key]


def net2d(kind, **kw):
    return cached(("2d", kind, tuple(sorted(kw.items()))), lambda: train_2d(kind, **kw))


def net_eps_fn(net, kind, cls=None, w=1.0):
    """把网络的输出换算成噪声预测。cls 为 None 时不给条件；否则按引导强度 w 把无条件和有条件的预测组合起来。"""
    def fn(x, t):
        tt = torch.full((len(x),), t)
        a = float(ABAR[t])
        eu = hw.convert_prediction(x, net(x, tt, torch.full((len(x),), NULL)), a, kind)[1]
        if cls is None:
            return eu
        return hw.cfg_combine(eu, hw.convert_prediction(x, net(x, tt, cls), a, kind)[1], w)
    return fn


def exact_eps_fn(mix, cls=None, w=1.0):
    def fn(x, t):
        a = float(ABAR[t])
        if cls is None:
            return mix.eps(x, a)
        return hw.cfg_combine(mix.eps(x, a), mix.eps(x, a, cls), w)
    return fn


def sample_2d(eps_fn, steps, sampler, seed, n=N2D):
    g = gen(seed, 31)
    return run_sampler(eps_fn, torch.randn(n, 2, generator=g), steps, sampler, g)


SAMPLERS = [("ddim", "DDIM（确定性，eta=0）", S.BLUE, "o-"), ("post", "DDPM，后验方差（eta=1）", S.ORANGE, "s--"), ("large", "DDPM，方差取 beta", S.GREEN, "^:")]
STEP_LIST = [1000, 200, 50, 20, 10, 5]


# ---------------------------------------------------------------- 图 2：采样器与步数
def fig_samplers():
    real = MIX.sample(N2D, gen(7))[0]
    floor = [sliced_w(MIX.sample(N2D, gen(100 + s))[0], real) for s in range(8)]
    seeds = (0, 1, 2)
    ref_std = MIX.blob_stats(real)[0]
    res = {}
    for key, _, _, _ in SAMPLERS:
        sw, sd = [], []
        for steps in STEP_LIST:
            xs = [sample_2d(exact_eps_fn(MIX), steps, key, 20 + s) for s in seeds]
            sw.append([sliced_w(x, real) for x in xs])
            sd.append([MIX.blob_stats(x)[0] / ref_std for x in xs])
        res[key] = (np.array(sw), np.array(sd))
    net = net2d("eps")
    net_sw = {key: [sliced_w(sample_2d(net_eps_fn(net, "eps"), steps, key, 20), real) for steps in STEP_LIST] for key in ("ddim", "post")}
    narrow = Mix(0.15)
    real_n = narrow.sample(N2D, gen(7))[0]
    ref_n = narrow.blob_stats(real_n)[0]
    nar = {key: [(sliced_w(x, real_n), narrow.blob_stats(x)[0] / ref_n) for x in (sample_2d(exact_eps_fn(narrow), steps, key, 20) for steps in STEP_LIST)] for key, _, _, _ in SAMPLERS}
    NUM["采样器与步数"] = {
        "说明": NOTE_2D + "。三种采样器都从同样的标准高斯噪声出发，步数下标从 999 到 0 等间隔取；这张表用解析去噪器，没有网络误差，剩下的全是采样器的误差。"
              "距离是切片 Wasserstein 距离（往 32 个方向投影后的一维距离取平均），和另外 4000 个真实点比；3 个采样种子取平均。"
              "团内标准差比：把每个样本分给最近的团心，各团的标准差取平均，再除以 4000 个真实样本按同样办法统计出的值；小于 1 表示样本比真实数据更集中。"
              "不直接除以 0.6，是因为相邻的团有重叠，分给最近的团心会把尾巴切掉，真实样本这样统计出来也到不了 0.6",
        "真实样本按同样办法统计的团内标准差": r3(ref_std), "它相对团的标准差 0.6 的比例": r2(ref_std / 0.6),
        "步数": STEP_LIST, "每个样本调用去噪器的次数": STEP_LIST,
        "两批真实样本之间的距离（8 次）": {"平均": r3(np.mean(floor)), "最小": r3(np.min(floor)), "最大": r3(np.max(floor))},
        **{name: {"到真实分布的距离": [r3(v) for v in res[key][0].mean(1)], "各种子的距离": [[r3(v) for v in row] for row in res[key][0]],
                  "团内标准差比": [r2(v) for v in res[key][1].mean(1)]} for key, name, _, _ in SAMPLERS},
        "换成训练出来的网络（预测噪声，1 个采样种子）": {"说明": NET_NOTE_2D, "DDIM 到真实分布的距离": [r3(v) for v in net_sw["ddim"]], "DDPM 后验方差 到真实分布的距离": [r3(v) for v in net_sw["post"]]},
    }
    w = world()

    def bg_level(x):
        r = residual(x).abs().flatten(1)
        return float(r.sort(1).values[:, :int(0.8 * r.shape[1])].pow(2).mean(1).sqrt().mean())

    ref_bg = bg_level(w["blob"][2000:2500])
    img_steps = [50, 20, 10, 5]
    img = {}
    for key, name, _, _ in SAMPLERS:
        rows = []
        for st in img_steps:
            g = gen(2, 37)
            n_img = 192

            def eps_fn(x, t):
                return w["base"](x, torch.full((n_img,), t))

            x = run_sampler(eps_fn, torch.randn(n_img, 1, SIZE, SIZE, generator=g), st, key, g)
            rt = rates(x)
            rows.append([rt[1], rt[0], bg_level(x) / ref_bg])
        img[name] = np.array(rows)
    g = gen(2, 37)
    n_ref = 96

    def eps_ref(x, t):
        return w["base"](x, torch.full((n_ref,), t))

    x_ref = run_sampler(eps_ref, torch.randn(n_ref, 1, SIZE, SIZE, generator=g), T, "post", g)      # 一步不跳的 DDPM
    rt_ref = rates(x_ref)
    NUM["采样器与步数：16x16 的图像"] = {
        "说明": NOTE_IMG + "。" + NOTE_UNET + "。同一个基础模型换三种采样器和四种步数，各采 192 张。" + JUDGE_NOTE + "。"
              "背景起伏：每张图减去各列的中位数，去掉绝对值最大的 20% 像素（缺陷所在的地方），剩下的像素的均方根；表里是它相对 500 张真实斑点图的倍数，"
              "1 表示背景和真实图一样干净，大于 1 表示背景里有没去干净的噪声，小于 1 表示比真实图还平滑",
        "步数": img_steps,
        **{name: {"判成斑点的比例": [r3(v) for v in rows[:, 0]], "判成良品的比例": [r3(v) for v in rows[:, 1]], "背景起伏相对真实图的倍数": [r2(v) for v in rows[:, 2]]} for name, rows in img.items()},
        "参照：DDPM 走满 1000 步（后验方差，每一步都不跳）": {
            "说明": "只采了 96 张（每张要调用网络 1000 次），比例的抽样标准误差约 0.05",
            "张数": n_ref, "判成斑点的比例": r3(rt_ref[1]), "判成良品的比例": r3(rt_ref[0]), "背景起伏相对真实图的倍数": r2(bg_level(x_ref) / ref_bg)},
        "真实斑点图按同样办法统计": {"判成斑点的比例": r3(rates(w["blob"][2000:2500])[1]), "背景起伏相对真实图的倍数": 1.0},
    }
    NUM["采样器与步数：团更窄的数据"] = {
        "说明": "同样的 6 个团心，团的标准差从 0.6 改成 0.15，其余不变；解析去噪器，1 个采样种子。团内标准差比的分母是真实样本按同样办法统计的值（这时团之间不重叠，它就是 0.15 左右）",
        "真实样本按同样办法统计的团内标准差": r3(ref_n),
        "步数": STEP_LIST,
        **{name: {"到真实分布的距离": [r3(v[0]) for v in nar[key]], "团内标准差比": [r2(v[1]) for v in nar[key]]} for key, name, _, _ in SAMPLERS},
    }

    fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.3), gridspec_kw={"width_ratios": [1, 1, 1.25, 1.25]}, layout="constrained", dpi=150)
    pts, cl = MIX.sample(1500, gen(8))
    for c, color in enumerate((S.BLUE, S.ORANGE, S.GREEN)):
        ax[0].scatter(pts[cl == c, 0], pts[cl == c, 1], s=4, color=color, alpha=0.6, label=f"类别 {c}", rasterized=True)
    ax[0].set_title("真实数据：6 团，3 类")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, markerscale=2.5, columnspacing=1.0, handletextpad=0.2)
    few = sample_2d(exact_eps_fn(MIX), 5, "ddim", 20, n=1500)
    ax[1].scatter(pts[:, 0], pts[:, 1], s=4, color="#c9cdd4", alpha=0.6, label="真实", rasterized=True)
    ax[1].scatter(few[:, 0], few[:, 1], s=4, color=S.RED, alpha=0.6, label="DDIM，5 步", rasterized=True)
    ax[1].set_title("DDIM 只走 5 步：团缩小了")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, markerscale=2.5, columnspacing=1.0, handletextpad=0.2)
    for a in ax[:2]:
        a.set_xlim(-4.2, 4.2)
        a.set_ylim(-4.2, 4.2)
        a.set_aspect("equal")
        S.image_axes(a)
    xs = np.arange(len(STEP_LIST))
    for key, name, color, style in SAMPLERS:
        ax[2].plot(xs, res[key][0].mean(1), style, color=color, lw=1.8, ms=5.5, label=name)
        ax[3].plot(xs, res[key][1].mean(1), style, color=color, lw=1.8, ms=5.5, label=name)
    ax[2].axhspan(np.min(floor), np.max(floor), color=S.GRID, alpha=0.9, lw=0)
    ax[2].text(0, np.max(floor) + 0.015, "灰带：两批真实样本之间的距离", ha="left", fontsize=8.5, color=S.MUTED)
    ax[2].set_ylabel("到真实分布的距离（切片 Wasserstein）")
    ax[2].set_title("步数减少时三种采样器都变差，快慢不同")
    ax[3].axhline(1.0, color=S.MUTED, lw=1, ls=":")
    ax[3].set_ylabel("团内标准差 / 真实样本的团内标准差")
    ax[3].set_title("少步时 DDIM 和后验方差的样本偏集中")
    for a in ax[2:]:
        a.set_xticks(xs, [str(s) for s in STEP_LIST])
        a.set_xlabel("采样步数（每个样本调用去噪器的次数）")
        a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        S.clean(a)
    S.save(fig, CH, "fig02_采样器与步数")


# ---------------------------------------------------------------- 图 3：三种参数化
KINDS = [("eps", "预测噪声", S.BLUE, "o-"), ("x0", "预测原图", S.ORANGE, "s--"), ("v", "预测 v", S.GREEN, "^:")]


def fig_param():
    nets = {k: net2d(k) for k, _, _, _ in KINDS}
    ts = [0, 10, 50, 200, 500, 800, 950, 990, 999]
    g = gen(8, 1)
    x0, _ = MIX.sample(20000, g)
    err = {k: {"x0": [], "eps": []} for k in nets}
    best = {"x0": [], "eps": []}
    for t in ts:
        a = float(ABAR[t])
        eps = torch.randn(20000, 2, generator=g)
        xt = math.sqrt(a) * x0 + math.sqrt(1 - a) * eps
        best["x0"].append(float(((MIX.x0(xt, a) - x0) ** 2).mean()))
        best["eps"].append(float(((MIX.eps(xt, a) - eps) ** 2).mean()))
        with torch.no_grad():
            for k, net in nets.items():
                px0, peps, _ = hw.convert_prediction(xt, net(xt, torch.full((20000,), t), torch.full((20000,), NULL)), a, k)
                err[k]["x0"].append(float(((px0 - x0) ** 2).mean()))
                err[k]["eps"].append(float(((peps - eps) ** 2).mean()))
    real = MIX.sample(N2D, gen(7))[0]
    few_steps = [1, 2, 5, 10, 20, 50]
    few = {k: [sliced_w(sample_2d(net_eps_fn(nets[k], k), s, "ddim", 12), real) for s in few_steps] for k in nets}
    few_exact = [sliced_w(sample_2d(exact_eps_fn(MIX), s, "ddim", 12), real) for s in few_steps]

    def sig2(v):
        return float(f"{v:.2g}")

    NUM["三种参数化"] = {
        "说明": NOTE_2D + "。" + NET_NOTE_2D + "；三个网络结构和训练设置相同，只是预测的量不同（噪声、原图、v），损失都是对各自目标的均方误差。"
              "评估时不给条件。把每个网络的输出用作业的 convert_prediction 换算成对原图和对噪声的估计，在 20000 个新样本上量均方误差（两个坐标平均）；"
              "“最优去噪器”那一行是这个分布上任何网络都不可能再低的误差",
        "步数下标 t": ts, "各步的 abar": [float(f"{ABAR[t]:.2g}") for t in ts],
        "换算倍数 (1 - abar) / abar（噪声误差乘它得到原图误差）": [sig2((1 - ABAR[t]) / ABAR[t]) for t in ts],
        "最优去噪器：原图的均方误差": [r3(v) for v in best["x0"]], "最优去噪器：噪声的均方误差": [r4(v) for v in best["eps"]],
        **{name: {"换算成原图的均方误差": [r3(v) for v in err[k]["x0"]], "换算成噪声的均方误差": [r4(v) for v in err[k]["eps"]]} for k, name, _, _ in KINDS},
        "t=999 时原图误差相对最优的倍数": {name: r2(err[k]["x0"][-1] / best["x0"][-1]) for k, name, _, _ in KINDS},
        "t=0 时噪声误差相对最优的倍数": {name: r2(err[k]["eps"][0] / best["eps"][0]) for k, name, _, _ in KINDS},
    }
    NUM["三种参数化：少步 DDIM"] = {
        "说明": "同上的三个网络，DDIM 采样，步数从 1 到 50；到真实分布的切片 Wasserstein 距离，1 个采样种子。1 步就是在 t=999 上做一次预测、直接换算成原图",
        "步数": few_steps, **{name: [r3(v) for v in few[k]] for k, name, _, _ in KINDS}, "解析去噪器": [r3(v) for v in few_exact],
    }

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), layout="constrained")
    xs = np.arange(len(ts))
    for a, key, title, ylabel in ((ax[0], "x0", "换算成原图：预测噪声的网络在高噪声端误差放大", "对原图的均方误差"),
                                 (ax[1], "eps", "换算成噪声：预测原图的网络在低噪声端误差放大", "对噪声的均方误差")):
        a.plot(xs, best[key], color=S.MUTED, lw=4, alpha=0.35, label="最优去噪器")
        for k, name, color, style in KINDS:
            a.plot(xs, err[k][key], style, color=color, lw=1.7, ms=5, label=name)
        a.set_yscale("log")
        a.set_xticks(xs, [str(t) for t in ts])
        a.set_xlabel("步数下标 t（越大噪声越多）")
        a.set_ylabel(ylabel)
        a.set_title(title, fontsize=10.5)
        a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, columnspacing=1.0)
        S.clean(a)
    ax[0].set_yticks([0.001, 0.01, 0.1, 1, 10], ["0.001", "0.01", "0.1", "1", "10"])
    ax[1].set_yticks([0.0001, 0.001, 0.01, 0.1, 1, 10], ["0.0001", "0.001", "0.01", "0.1", "1", "10"])
    for a in ax[:2]:
        a.minorticks_off()
    xs = np.arange(len(few_steps))
    ax[2].plot(xs, few_exact, color=S.MUTED, lw=4, alpha=0.35, label="解析去噪器")
    for k, name, color, style in KINDS:
        ax[2].plot(xs, few[k], style, color=color, lw=1.7, ms=5, label=name)
    ax[2].set_xticks(xs, [str(s) for s in few_steps])
    ax[2].set_xlabel("DDIM 采样步数")
    ax[2].set_ylabel("到真实分布的距离")
    ax[2].set_title("步数很少时预测噪声的网络最差", fontsize=10.5)
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, columnspacing=1.0)
    S.clean(ax[2])
    S.save(fig, CH, "fig03_三种参数化")


# ---------------------------------------------------------------- 图 4：无分类器引导
W_LIST = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]


def fig_cfg():
    cls = torch.arange(3).repeat_interleave(N2D // 3 + 1)[:N2D]
    g = gen(9)
    real_c = MIX.sample(N2D, g, cls)[0]
    thr = torch.quantile(MIX.log_p(MIX.sample(N2D, g, cls)[0], cls), 0.01)

    ref = MIX.blob_stats(real_c, cls)

    def measure(x):
        st = MIX.blob_stats(x, cls)
        return [float((MIX.bayes(x) == cls).float().mean()), st[0] / ref[0], st[1] / MIX.sig, float((MIX.log_p(x, cls) < thr).float().mean()),
                float(np.mean([sliced_w(x[cls == c], real_c[cls == c]) for c in range(3)])), st[2] / ref[2], st[3] / ref[3]]

    net = net2d("eps")
    rows = {"网络": [], "解析去噪器": []}
    shown = {}
    for w in W_LIST:
        x = sample_2d(net_eps_fn(net, "eps", cls, w), 50, "ddim", 13)
        rows["网络"].append(measure(x))
        rows["解析去噪器"].append(measure(sample_2d(exact_eps_fn(MIX, cls, w), 50, "ddim", 13)))
        if w in (0.0, 1.0, 3.0, 8.0):
            shown[w] = x[cls == 0][:500]
    names = ["条件符合率", "团内标准差比", "质心外移（单位：团的标准差）", "离群比例", "到条件分布的距离", "沿半径方向的标准差比", "垂直于半径方向的标准差比"]
    nodrop = net2d("eps", p_drop=0.0)
    nd = {f"w={w:g}": dict(zip(names, [r3(v) for v in measure(sample_2d(net_eps_fn(nodrop, "eps", cls, w), 50, "ddim", 13))])) for w in (1.0, 3.0)}
    real_row = measure(real_c)
    NUM["无分类器引导"] = {
        "说明": NOTE_2D + "。" + NET_NOTE_2D + "（预测噪声）。采样用 DDIM 50 步，每一步把无条件和有条件的噪声预测用作业的 cfg_combine 组合："
              "无条件 + w x（有条件 - 无条件），w=0 是不给条件，w=1 是普通的条件生成。4000 个样本，3 个类别各三分之一。"
              "条件符合率：用真实分布的贝叶斯分类器判断样本属于哪一类，和要求的类别一致的比例。团内标准差比、质心外移和两个方向的标准差比只在要求的那一类的两团里统计，"
              "标准差比的分母都是真实条件样本按同样办法统计的值；“沿半径方向”指从原点指向团心的方向。"
              "离群比例：样本在要求的条件分布下的对数密度，低于真实条件样本 1% 分位的比例（真实样本按定义约为 0.01）。"
              "到条件分布的距离：每一类各自和真实条件样本算切片 Wasserstein 距离再平均",
        "引导强度 w": W_LIST,
        **{src: {n: [r2(row[i]) if i in (1, 2, 5, 6) else r3(row[i]) for row in rows[src]] for i, n in enumerate(names)} for src in rows},
        "真实的条件样本": dict(zip(names, [r3(v) for v in real_row])),
        "训练时从不丢掉条件的网络（丢弃概率 0）": {"说明": "其余设置相同。它没有学过“没有条件”这个输入，无条件那一支的输出没有意义", **nd},
    }

    fig, ax = plt.subplots(1, 5, figsize=(18, 4.0), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.9]}, layout="constrained", dpi=150)
    bg = MIX.sample(1500, gen(8))[0]
    for a, w in zip(ax[:4], (0.0, 1.0, 3.0, 8.0)):
        a.scatter(bg[:, 0], bg[:, 1], s=3, color="#d5d8dd", alpha=0.6, rasterized=True)
        a.scatter(shown[w][:, 0], shown[w][:, 1], s=4, color=S.BLUE, alpha=0.7, rasterized=True)
        a.scatter(MU[[0, 3], 0], MU[[0, 3], 1], marker="+", s=90, color=S.RED, lw=1.5)
        a.set_xlim(-5.2, 5.2)
        a.set_ylim(-5.2, 5.2)
        a.set_aspect("equal")
        S.image_axes(a)
        a.set_title(f"要求类别 0，w = {w:g}", fontsize=10.5)
    xs = np.arange(len(W_LIST))
    arr = np.array(rows["网络"])
    ax[4].plot(xs, arr[:, 0], "o-", color=S.BLUE, lw=1.8, ms=5.5, label="条件符合率")
    ax[4].plot(xs, arr[:, 1], "s-", color=S.ORANGE, lw=1.8, ms=5.5, label="团内标准差比（多样性）")
    ax[4].plot(xs, arr[:, 3], "^-", color=S.RED, lw=1.8, ms=5.5, label="离群比例（失真）")
    ax[4].axhline(1.0, color=S.MUTED, lw=1, ls=":")
    ax[4].set_xticks(xs, [f"{w:g}" for w in W_LIST])
    ax[4].set_ylim(-0.04, 2.1)
    ax[4].set_xlabel("引导强度 w")
    ax[4].set_title("w 越大符合率越高、团越小；w 到 5 以上离群的样本又多起来", fontsize=10.5)
    ax[4].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    S.clean(ax[4])
    S.save(fig, CH, "fig04_无分类器引导")


# ================================================================ 第二部分：16x16 的纹理加缺陷
SIZE, PIX_NOISE, PB = 16, 0.02, 8
YY, XX = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
NOTE_IMG = (f"图像数据：{SIZE}x{SIZE} 的竖条纹（周期 4 px，每张图的横向平移随机），增益在 0.9 到 1.1 之间随机，像素噪声标准差 {PIX_NOISE}（灰度 0 到 1）；"
            "送进网络前减 0.5 再除以 0.2。斑点缺陷是椭圆高斯（短轴标准差 1.0 到 1.6 px，长短轴比 1 到 1.8，方向随机），横向划痕是长 5 到 9 px、和水平方向夹角 20 度以内的线段（横截面是标准差 0.7 px 的高斯）；"
            "两种缺陷的幅度都是 0.2 到 0.4，亮暗各半。缺陷掩膜取轮廓大于峰值 0.3 倍的像素")
NOTE_UNET = ("去噪网络是一个小 UNet：16x16 和 8x8 两个尺度，通道数 16 和 32，8 个 3x3 卷积，步数用正弦编码加两层 MLP 注入，预测噪声；"
             "在 4000 张斑点缺陷图上训练 2000 步（Adam，学习率 0.002 余弦下降，批大小 32）。采样没有特别说明时都是 DDIM 20 步")


def texture(n, rng):
    return 0.5 + 0.15 * np.cos(2 * np.pi * (XX[None] + rng.uniform(0, 4, size=(n, 1, 1))) / 4.0)


def _uv(n, rng, lo, hi, th):
    cy, cx = rng.uniform(lo, hi, size=(2, n, 1, 1))
    return (YY[None] - cy) * np.sin(th) + (XX[None] - cx) * np.cos(th), -(YY[None] - cy) * np.cos(th) + (XX[None] - cx) * np.sin(th)


def blob_prof(n, rng):
    s, ratio, th = rng.uniform(1.0, 1.6, size=(n, 1, 1)), rng.uniform(1.0, 1.8, size=(n, 1, 1)), rng.uniform(0, np.pi, size=(n, 1, 1))
    u, v = _uv(n, rng, 3.5, SIZE - 4.5, th)
    return np.exp(-(u ** 2 / (2 * (s * ratio) ** 2) + v ** 2 / (2 * s ** 2)))


def scratch_prof(n, rng):
    length, th = rng.uniform(5, 9, size=(n, 1, 1)), np.deg2rad(rng.uniform(-20, 20, size=(n, 1, 1)))
    u, v = _uv(n, rng, 4.5, SIZE - 5.5, th)
    return np.exp(-np.hypot(np.maximum(np.abs(u) - length / 2, 0), v) ** 2 / (2 * 0.7 ** 2))


def finish(clean, rng):
    img = clean * rng.uniform(0.9, 1.1, size=(len(clean), 1, 1)) + rng.normal(0, PIX_NOISE, size=clean.shape)
    return torch.tensor((img - 0.5) / 0.2, dtype=torch.float32)[:, None]


def make(n, rng, kind):
    """返回 (图, 掩膜)，都是 (n, 1, 16, 16)。kind：good 良品、blob 斑点、scratch 横向划痕。"""
    clean = texture(n, rng)
    if kind == "good":
        return finish(clean, rng), torch.zeros(n, 1, SIZE, SIZE)
    prof = blob_prof(n, rng) if kind == "blob" else scratch_prof(n, rng)
    amp = rng.choice([-1.0, 1.0], size=(n, 1, 1)) * rng.uniform(0.2, 0.4, size=(n, 1, 1))
    return finish(clean + amp * prof, rng), torch.tensor(prof > 0.3, dtype=torch.float32)[:, None]


def residual(x):
    """条纹沿竖直方向不变，减掉每一列的中位数，剩下的就是缺陷和噪声。"""
    return x - x.median(dim=2, keepdim=True).values


class LConv(nn.Module):
    """3x3 卷积，可以挂一个低秩增量。挂上之后的前向把输入展开成 3x3 邻域，用作业的 lora_forward 计算（卷积就是对邻域做线性层）。"""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride, 1)
        self.A = self.B = None
        self.alpha = 1.0

    def add_lora(self, r, g):
        cout, fan = self.conv.weight.shape[0], self.conv.weight[0].numel()
        r = min(r, cout, fan)                                  # 秩不可能超过权重矩阵较短的那一边
        self.A = nn.Parameter(torch.randn(r, fan, generator=g) / math.sqrt(fan))
        self.B = nn.Parameter(torch.zeros(cout, r))
        self.alpha = float(r)                                  # alpha 取 r，缩放系数 alpha / r 恒为 1

    def merged(self):
        w = self.conv.weight
        return w if self.A is None else hw.lora_merge(w.flatten(1), self.A, self.B, self.alpha).view_as(w)

    def forward(self, x):
        if self.A is None:
            return self.conv(x)
        b, _, h, w = x.shape
        st = self.conv.stride[0]
        y = hw.lora_forward(F.unfold(x, 3, padding=1, stride=st).transpose(1, 2), self.conv.weight.flatten(1), self.conv.bias, self.A, self.B, self.alpha)
        return y.transpose(1, 2).reshape(b, -1, h // st, w // st)


class UNet(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.c = c
        self.t = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 64))
        self.c1, self.c2, self.d, self.m1, self.m2 = LConv(1, c), LConv(c, c), LConv(c, 2 * c, 2), LConv(2 * c, 2 * c), LConv(2 * c, 2 * c)
        self.u, self.o1, self.o2 = LConv(2 * c, c), LConv(c, c), LConv(c, 1)
        self.f1, self.f2 = nn.Linear(64, c), nn.Linear(64, 2 * c)

    def convs(self):
        return [self.c1, self.c2, self.d, self.m1, self.m2, self.u, self.o1, self.o2]

    def encode(self, x, e, hint=None):
        h = F.silu(self.c1(x))
        if hint is not None:
            h = h + hint
        h1 = F.silu(self.c2(h + self.f1(e)[:, :, None, None]))
        return h1, F.silu(self.m1(F.silu(self.d(h1)) + self.f2(e)[:, :, None, None]))

    def forward(self, x, t, ctrl=None):
        h1, h2 = self.encode(x, self.t(temb(t)))
        if ctrl is not None:                                   # 控制旁路给出的两个残差，加在两个尺度的特征上
            h1, h2 = h1 + ctrl[0], h2 + ctrl[1]
        u = F.interpolate(F.silu(self.u(F.silu(self.m2(h2)))), scale_factor=2, mode="nearest")
        return self.o2(F.silu(self.o1(u + h1)))


class Control(nn.Module):
    """ControlNet 式的旁路：原模型编码器的一份可训练拷贝，输入处加上掩膜的特征；输出经过两个 1x1 卷积加回原模型。
    zero_init=True 时这两个 1x1 卷积的权重和偏置都从 0 开始，旁路在训练开始时不改变原模型的输出。"""

    def __init__(self, base, zero_init=True):
        super().__init__()
        self.enc = copy.deepcopy(base)
        for name in ("m2", "u", "o1", "o2"):
            delattr(self.enc, name)
        for p in self.enc.parameters():
            p.requires_grad_(True)
        self.hint = nn.Sequential(nn.Conv2d(1, base.c, 3, padding=1), nn.SiLU(), nn.Conv2d(base.c, base.c, 3, padding=1))
        self.z1, self.z2 = nn.Conv2d(base.c, base.c, 1), nn.Conv2d(2 * base.c, 2 * base.c, 1)
        if zero_init:
            for z in (self.z1, self.z2):
                nn.init.zeros_(z.weight)
                nn.init.zeros_(z.bias)

    def forward(self, x, t, mask):
        h1, h2 = self.enc.encode(x, self.enc.t(temb(t)), self.hint(mask))
        return self.z1(h1), self.z2(h2)


def n_params(params):
    return sum(p.numel() for p in params)


def train_diffusion(net, data, steps, batch, lr, seed, params=None, ctrl=None, masks=None, cosine=True, hook=None):
    """噪声预测损失。params 为 None 时训练 net 的全部参数；给了 ctrl 时把掩膜当条件，通常只训练旁路。返回每一步的损失。
    hook(i) 在已经更新了 i 步之后调用（i 从 0 到 steps），用来在训练中途做评估。"""
    g = gen(seed, 33)
    params = list(net.parameters()) if params is None else params
    opt = torch.optim.Adam(params, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps) if cosine else None
    log = []
    for i in range(steps):
        if hook is not None:
            hook(i)
        idx = torch.randint(0, len(data), (batch,), generator=g)
        x0 = data[idx]
        t = torch.randint(0, T, (batch,), generator=g)
        a = AB[t][:, None, None, None]
        eps = torch.randn(x0.shape, generator=g)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        loss = ((net(xt, t, ctrl(xt, t, masks[idx]) if ctrl is not None else None) - eps) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        log.append(float(loss.detach()))
    if hook is not None:
        hook(steps)
    return log


def sample_img(net, n, seed, steps=20, ctrl=None, mask=None, known=None, region=None, level="prev", noised=True):
    """DDIM 采样。known、region 给定时做修复：每一步之后用作业的 inpaint_replace 把 region 为 0 的地方换成加噪到当前噪声水平的 known。
    level、noised 只在演示两种写错的做法时改动：level="t" 是替换时用了这一步之前的噪声水平，noised=False 是直接贴回没加噪的原图。"""
    g = gen(seed, 37)
    x = torch.randn(n, 1, SIZE, SIZE, generator=g)

    def eps_fn(x, t):
        tt = torch.full((n,), t)
        return net(x, tt, ctrl(x, tt, mask) if ctrl is not None else None)

    if known is None:
        return run_sampler(eps_fn, x, steps, "ddim", g)
    ts = timesteps(steps)
    with torch.no_grad():
        for i, t in enumerate(ts):
            a_t = float(ABAR[t])
            a_prev = float(ABAR[ts[i + 1]]) if i + 1 < len(ts) else 1.0
            x = hw.ddim_step(x, eps_fn(x, int(t)), a_t, a_prev)
            lv = a_prev if level == "prev" or a_prev == 1.0 else a_t
            x = hw.inpaint_replace(x, known, region, lv if noised else 1.0, torch.randn(x.shape, generator=g))
    return x


class Clf(nn.Module):
    """三层卷积加全局最大池化的小分类器（和第 20 章的结构相同，输入小一半）。"""

    def __init__(self, out=1):
        super().__init__()
        self.c1, self.c2, self.c3 = nn.Conv2d(1, 12, 3, padding=1), nn.Conv2d(12, 24, 3, stride=2, padding=1), nn.Conv2d(24, 24, 3, padding=1)
        self.fc = nn.Linear(24, out)

    def forward(self, x):
        return self.fc(F.relu(self.c3(F.relu(self.c2(F.relu(self.c1(x)))))).amax(dim=(2, 3)))


def fit_clf(batch_fn, out, steps, seed, lr=3e-3):
    """batch_fn(g) 返回一个批的 (图, 标签)。out 大于 1 时是多分类，等于 1 时是二分类。"""
    torch.manual_seed(seed)
    g = gen(seed, 77)
    net = Clf(out)
    opt = torch.optim.Adam(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x, y = batch_fn(g)
        o = net(x)
        loss = F.cross_entropy(o, y) if out > 1 else F.binary_cross_entropy_with_logits(o[:, 0], y.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return net.eval()


def auroc(neg, pos):
    """按秩次算 AUROC，并列取平均秩。"""
    s = np.concatenate([neg, pos])
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    end = np.cumsum(cnt)
    ranks = ((end - cnt + 1 + end) / 2.0)[inv]
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def world():
    """图像部分共用的东西：数据、判别器、残差阈值、只见过斑点的基础模型。"""
    def build():
        rng = np.random.default_rng(0)
        w = {}
        w["blob"], w["blob_m"] = make(4000, rng, "blob")
        w["good"], _ = make(2000, rng, "good")
        w["scr"], w["scr_m"] = make(3000, rng, "scratch")
        xs = torch.cat([w["good"][:1500], w["blob"][:1500], w["scr"][:1500]])
        ys = torch.arange(3).repeat_interleave(1500)

        def batch(g):
            idx = torch.randint(0, len(xs), (32,), generator=g)
            return xs[idx], ys[idx]

        w["judge"] = fit_clf(batch, 3, 600, 0)
        w["thr"] = float(torch.quantile(residual(w["good"]).abs().amax(dim=(1, 2, 3)), 0.99))
        torch.manual_seed(0)
        base = UNet()
        t0 = time.perf_counter()
        w["base_log"] = train_diffusion(base, w["blob"], 2000, 32, 2e-3, seed=0)
        TIMES["base_step"] = (time.perf_counter() - t0) / 2000
        for p in base.parameters():
            p.requires_grad_(False)
        w["base"] = base.eval()
        w["req_x"], w["req_m"] = make(500, np.random.default_rng(5), "blob")      # “指定的掩膜”：500 个斑点形状的掩膜
        return w
    return cached("world", build)


def rates(x):
    """判别器把一批图判成 良品 / 斑点 / 横向划痕 的比例。"""
    with torch.no_grad():
        p = world()["judge"](x).argmax(1)
    return [float((p == c).float().mean()) for c in range(3)]


def mask_metrics(x, m):
    """生成的缺陷和指定掩膜对得上的程度，返回 [掩膜内能量占比, IoU, 掩膜内有缺陷的比例]。"""
    thr = world()["thr"]
    r = residual(x)
    e = (r * (r.abs() > thr)) ** 2
    tot, ins = e.sum((1, 2, 3)), (e * m).sum((1, 2, 3))
    has = tot > 0
    region = (r.abs() > 0.3 * r.abs().amax(dim=(1, 2, 3), keepdim=True)) & (r.abs() > thr)
    inter, union = (region & (m > 0.5)).sum((1, 2, 3)).float(), (region | (m > 0.5)).sum((1, 2, 3)).float()
    return [float((ins[has] / tot[has]).mean()), float((inter / union.clamp_min(1)).mean()), float(((r.abs() * m).amax(dim=(1, 2, 3)) > thr).float().mean())]


METRIC_NOTE = ("残差指图减去每一列的中位数（条纹沿竖直方向不变，剩下缺陷和噪声）；阈值取良品图残差最大绝对值的 99% 分位。"
               "掩膜内能量占比：超过阈值的残差的平方和里，落在指定掩膜内的比例。IoU：残差超过阈值且超过自身峰值 0.3 倍的区域，和指定掩膜的交并比。"
               "掩膜内有缺陷：掩膜内残差的最大绝对值超过阈值的图所占的比例")
JUDGE_NOTE = "判别器是三层卷积的小分类器，用良品、斑点、横向划痕各 1500 张真实图训练 600 步，用来判断一张生成图像哪一类"


def show(ax, img, mask=None, color="cyan", box=None):
    ax.imshow(img, cmap="gray", vmin=-2.2, vmax=2.2)
    if mask is not None and float(mask.max()) > 0:
        ax.contour(np.asarray(mask), levels=[0.5], colors=color, linewidths=1.1)
    if box is not None:
        ax.add_patch(plt.Rectangle((box[1] - 0.5, box[0] - 0.5), PB, PB, fill=False, ec="#ff5a4f", lw=1.1))
    S.image_axes(ax)


# ---------------------------------------------------------------- 图 1：第 20 章停下的地方
def fig_gaps():
    w = world()
    base, req_m = w["base"], w["req_m"]
    xb = sample_img(base, 500, 3)
    real_r, gen_r = rates(w["blob"][2000:2500]), rates(xb)
    with torch.no_grad():
        acc = [float((w["judge"](x).argmax(1) == c).float().mean()) for c, x in enumerate((w["good"][1500:], w["blob"][1500:2000], w["scr"][1500:2000]))]
    NUM["基础模型"] = {
        "说明": NOTE_IMG + "。" + NOTE_UNET + "。" + JUDGE_NOTE + "。" + METRIC_NOTE + "。“指定的掩膜”是另外生成的 500 个斑点形状的掩膜，和无条件模型的样本一一配对",
        "可训练参数个数": n_params(base.parameters()), "训练最后 100 步的平均损失": r3(np.mean(w["base_log"][-100:])),
        "残差阈值（归一化后的灰度）": r2(w["thr"]),
        "判别器在留出真实图上的精度（良品，斑点，横向划痕）": [r3(v) for v in acc],
        "真实斑点图被判成（良品，斑点，横向划痕）": [r3(v) for v in real_r],
        "基础模型的 500 个样本被判成（良品，斑点，横向划痕）": [r3(v) for v in gen_r],
        "逐步采样每张图调用网络的次数": T, "DDIM 20 步每张图调用网络的次数": 20, "调用次数之比": T // 20,
        "基础模型的样本对指定掩膜（掩膜内能量占比，IoU，掩膜内有缺陷）": [r3(v) for v in mask_metrics(xb, req_m)],
        "真实图对自己的掩膜（同上三个数）": [r3(v) for v in mask_metrics(w["req_x"], req_m)],
        "掩膜的平均面积（像素）": r2(float(req_m.sum((1, 2, 3)).mean())), "掩膜面积占整张图的比例": r3(float(req_m.mean())),
    }
    fig, ax = plt.subplots(3, 8, figsize=(12.5, 5.1), layout="constrained")
    for i in range(8):
        show(ax[0, i], w["blob"][i, 0], w["blob_m"][i, 0])
        show(ax[1, i], xb[i, 0])
        show(ax[2, i], xb[i, 0], req_m[i, 0], color="#ff5a4f")
    for a, name in zip(ax[:, 0], ("真实斑点图\n青线是缺陷掩膜", "无条件模型的样本", "同一批样本\n红线是想要的位置")):
        a.set_ylabel(name, fontsize=9.5)
    fig.suptitle("只见过斑点的无条件模型能生成带缺陷的图，但缺陷落在哪里不由你定", fontsize=12, fontweight="bold")
    S.save(fig, CH, "fig01_无条件生成的缺口")


# ---------------------------------------------------------------- 图 5：空间条件（控制旁路）
CTRL_STEPS, CTRL_SEEDS = 400, (1, 2)
CTRL_CHECK = [0, 10, 30, 100, 200, 400]


def ctrl_eval_set():
    """固定的一批评估样本（图、步数、噪声都固定），用来在训练前后比较损失。"""
    def build():
        w = world()
        g = gen(0, 5)
        x0, m = w["blob"][:1000], w["blob_m"][:1000]
        t = torch.randint(0, T, (1000,), generator=g)
        a = AB[t][:, None, None, None]
        eps = torch.randn(x0.shape, generator=g)
        return a.sqrt() * x0 + (1 - a).sqrt() * eps, t, m, eps
    return cached("ctrl_eval", build)


def get_ctrl(zero=True, seed=1, cosine=True):
    """训练一个控制旁路，返回 (旁路, 训练前的几个量, 固定评估集上各检查点的损失)。后两项只在 seed=1、余弦下降时记录。"""
    def build():
        w = world()
        base = w["base"]
        xt, t, m, eps = ctrl_eval_set()
        torch.manual_seed(seed)
        ctrl = Control(base, zero_init=zero)
        main = seed == 1 and cosine
        init, curve = {}, []
        if main:
            with torch.no_grad():
                p0, p1 = base(xt, t), base(xt, t, ctrl(xt, t, m))
            init = {"训练前旁路让输出改变的最大值": float(f"{float((p1 - p0).abs().max()):.3g}"), "不接旁路时的损失": r4(((p0 - eps) ** 2).mean()),
                    "训练前带旁路采样，样本判成斑点的比例": r3(rates(sample_img(base, 256, 3, ctrl=ctrl, mask=w["req_m"][:256]))[1])}

        def hook(i):
            if i in CTRL_CHECK:
                with torch.no_grad():
                    curve.append(float(((base(xt, t, ctrl(xt, t, m)) - eps) ** 2).mean()))

        train_diffusion(base, w["blob"], CTRL_STEPS, 32, 2e-3, seed=seed, params=list(ctrl.parameters()), ctrl=ctrl, masks=w["blob_m"], cosine=cosine, hook=hook if main else None)
        return ctrl.eval(), init, curve
    return cached(("ctrl", zero, seed, cosine), build)


def fig_control():
    w = world()
    base, req_m = w["base"], w["req_m"]
    names = ["掩膜内能量占比", "IoU", "掩膜内有缺陷的比例"]
    xb = sample_img(base, 500, 3)
    res, xc = {}, None
    for cosine in (True, False):
        for zero in (True, False):
            rows = []
            for seed in (CTRL_SEEDS if cosine else CTRL_SEEDS[:1]):
                x = sample_img(base, 500, 3, ctrl=get_ctrl(zero, seed, cosine)[0], mask=req_m)
                rows.append(mask_metrics(x, req_m))
                if cosine and zero and seed == 1:
                    xc = x
            res[(cosine, zero)] = np.array(rows)
    bars = {"不加条件": np.array([mask_metrics(xb, req_m)]), "旁路随机初始化": res[(True, False)], "旁路零初始化": res[(True, True)],
            "真实图对自己的掩膜": np.array([mask_metrics(w["req_x"], req_m)])}
    ctrl, init_z, curve_z = get_ctrl(True)
    _, init_r, curve_r = get_ctrl(False)
    r = residual(xc)
    both_m = torch.clamp(req_m[:250] + req_m[250:], 0, 1)
    sep = (req_m[:250] * req_m[250:]).sum((1, 2, 3)) == 0
    x2 = sample_img(base, 250, 4, ctrl=ctrl, mask=both_m)
    r2_ = residual(x2).abs()
    both = ((r2_ * req_m[:250]).amax(dim=(1, 2, 3)) > w["thr"]) & ((r2_ * req_m[250:]).amax(dim=(1, 2, 3)) > w["thr"])
    empty = rates(sample_img(base, 250, 4, ctrl=ctrl, mask=torch.zeros(250, 1, SIZE, SIZE)))

    def pack(arr):
        return {**{n: r3(v) for n, v in zip(names, arr.mean(0))}, **{f"各种子的{n}": [r3(v) for v in arr[:, i]] for i, n in enumerate(names)}}

    NUM["空间条件"] = {
        "说明": NOTE_IMG + "。" + NOTE_UNET + "。控制旁路是基础模型编码器（前 4 个卷积和步数的 MLP）的一份可训练拷贝，掩膜经过两层卷积后加在拷贝的第一层特征上；"
              "拷贝在两个尺度上的输出各经过一个 1x1 卷积，加到基础模型对应尺度的特征上。基础模型冻结，只训练旁路："
              f"在同样的 4000 张斑点图和它们的掩膜上训练 {CTRL_STEPS} 步（Adam，学习率 0.002 余弦下降到 0，批大小 32）。零初始化指那两个 1x1 卷积的权重和偏置从 0 开始；随机初始化指用 PyTorch 默认的初始化，其余完全相同。"
              f"两种初始化各训练 {len(CTRL_SEEDS)} 个种子（旁路的初始化和训练时抽到的样本都跟着换），表里是平均。评估：500 个指定的掩膜各生成一张图。" + METRIC_NOTE,
        "旁路的可训练参数个数": n_params(ctrl.parameters()), "旁路参数占基础模型的比例": r3(n_params(ctrl.parameters()) / n_params(base.parameters())),
        "不加条件": dict(zip(names, [r3(v) for v in bars["不加条件"][0]])),
        "旁路随机初始化": pack(res[(True, False)]), "旁路零初始化": pack(res[(True, True)]),
        "真实图对自己的掩膜": dict(zip(names, [r3(v) for v in bars["真实图对自己的掩膜"][0]])),
        "训练前": {"说明": "种子 1 的那两个旁路", "零初始化": init_z, "随机初始化": init_r},
        "固定评估集上的损失": {
            "说明": "种子 1 的那两次训练。1000 个固定的评估样本（图、步数、噪声都固定），在更新了若干步之后各算一次噪声预测损失；“不接旁路”是基础模型自己在同一批样本上的损失",
            "已经更新的步数": CTRL_CHECK, "零初始化": [r4(v) for v in curve_z], "随机初始化": [r4(v) for v in curve_r], "不接旁路": init_z["不接旁路时的损失"]},
        "换一种学习率设置：全程 0.002 不下降，其余相同": {
            "说明": "两种初始化各只训练了 1 次（种子 1）。用来看两种初始化在训练结束时的差别是不是只在余弦下降时才有",
            "旁路随机初始化": pack(res[(False, False)]), "旁路零初始化": pack(res[(False, True)])},
        "零初始化的旁路（种子 1）：生成的缺陷里亮缺陷的比例": r3(((r * req_m).sum((1, 2, 3)) > 0).float().mean()),
        "零初始化的旁路（种子 1）：给空掩膜时样本被判成（良品，斑点，横向划痕）": [r3(v) for v in empty],
        "零初始化的旁路（种子 1）：一张掩膜里放两个不重叠的斑点（训练时每张只有一个）": {
            "张数": int(sep.sum()), "两处都有缺陷的比例": r3(both[sep].float().mean()), **dict(zip(names, [r3(v) for v in mask_metrics(x2[sep], both_m[sep])]))},
    }

    fig = plt.figure(figsize=(17.5, 4.6), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 1.15, 1.0])
    left = gs[0].subgridspec(3, 6)
    for i in range(6):
        ax = fig.add_subplot(left[0, i])
        ax.imshow(req_m[i, 0], cmap="gray", vmin=0, vmax=1.6)
        S.image_axes(ax)
        if i == 0:
            ax.set_ylabel("指定的掩膜", fontsize=9.5)
        for row, data, name in ((1, xb, "不加条件"), (2, xc, "加控制旁路")):
            ax = fig.add_subplot(left[row, i])
            show(ax, data[i, 0], req_m[i, 0], color="#ff5a4f")
            if i == 0:
                ax.set_ylabel(name, fontsize=9.5)
    ax = fig.add_subplot(gs[1])
    gx = np.arange(3)
    for i, (name, color) in enumerate(zip(bars, (S.MUTED, S.ORANGE, S.BLUE, "#b9bec7"))):
        vals = bars[name].mean(0)
        rects = ax.bar(gx + (i - 1.5) * 0.2, vals, 0.19, color=color, alpha=0.9, label=name)
        for j, (b, v) in enumerate(zip(rects, vals)):
            ax.text(b.get_x() + b.get_width() / 2, bars[name][:, j].max() + 0.015, f"{v:.2f}", ha="center", fontsize=7.5)
            if len(bars[name]) > 1:
                ax.plot([b.get_x() + b.get_width() / 2] * len(bars[name]), bars[name][:, j], ".", color=S.INK, ms=4)
    ax.set_xticks(gx, ["掩膜内\n能量占比", "IoU", "掩膜内\n有缺陷"])
    ax.set_ylim(0, 1.12)
    ax.set_title("加了旁路，缺陷才落进指定的掩膜（黑点是各个种子）", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    S.clean(ax)
    ax.grid(axis="x", visible=False)
    ax = fig.add_subplot(gs[2])
    xs = np.arange(len(CTRL_CHECK))
    ax.plot(xs, curve_r, "s--", color=S.ORANGE, lw=1.8, ms=5.5, label="旁路随机初始化")
    ax.plot(xs, curve_z, "o-", color=S.BLUE, lw=1.8, ms=5.5, label="旁路零初始化")
    ax.axhline(init_z["不接旁路时的损失"], color=S.MUTED, lw=1, ls=":")
    ax.text(len(xs) - 1, init_z["不接旁路时的损失"] * 1.04, "不接旁路", ha="right", va="bottom", fontsize=8.5, color=S.MUTED)
    ax.set_yscale("log")
    ax.set_yticks([0.06, 0.1, 0.2], ["0.06", "0.1", "0.2"])
    ax.minorticks_off()
    ax.set_xticks(xs, [str(v) for v in CTRL_CHECK])
    ax.set_xlabel("旁路已经训练的步数")
    ax.set_ylabel("固定评估集上的噪声预测损失")
    ax.set_title("零初始化从原模型的损失起步，随机初始化先把它抬高", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    S.clean(ax)
    S.save(fig, CH, "fig05_空间条件")


# ---------------------------------------------------------------- 图 6：低秩微调
LORA_LR, FULL_LR, FT_BATCH = 5e-3, 1e-3, 16


def new_finetune(r, seed):
    """复制一份基础模型准备微调。r 是整数时给 8 个卷积各挂一个秩 r 的增量，只训练增量；r 为 "full" 时直接训练 8 个卷积的权重。"""
    net = copy.deepcopy(world()["base"])
    if r == "full":
        params = [c.conv.weight for c in net.convs()]
        for p in params:
            p.requires_grad_(True)
    else:
        g = gen(seed, 41)
        for c in net.convs():
            c.add_lora(r, g)
        params = [p for c in net.convs() for p in (c.A, c.B)]
    return net, params


def merged_copy(net):
    """用作业的 lora_merge 把每个卷积的增量并进权重，返回一个不带旁路的普通网络。采样时用它，没有额外开销。"""
    out = copy.deepcopy(net)
    with torch.no_grad():
        for c in out.convs():
            if c.A is not None:
                c.conv.weight.copy_(c.merged())
                c.A = c.B = None
    return out.eval()


def finetune(train, r, steps, seed):
    """返回 (合并后的网络, 可训练参数个数, 合并前的网络)。"""
    net, params = new_finetune(r, seed)
    train_diffusion(net, train, steps, FT_BATCH, FULL_LR if r == "full" else LORA_LR, seed, params=params, cosine=False)
    return merged_copy(net), n_params(params), net.eval()


def nn_dist(a, b):
    """a 里每张图到 b 里最近的一张的距离（逐像素差的均方根）。"""
    return (torch.cdist(a.flatten(1), b.flatten(1)) / math.sqrt(a[0].numel())).min(1).values


def gen_report(x, train):
    """生成样本的几个量：[判成新类型（横向划痕）的比例, 到训练集最近邻距离的中位数, 复制比例, 留出真实样本的最近邻距离中位数, 判成良品的比例]。
    复制比例 = 生成样本里，到训练集的最近邻距离比“留出的真实样本到训练集的最近邻距离的 1% 分位”还小的比例。
    距离是整张图逐像素比的，背景的条纹对得上也会让距离变小。"""
    held = world()["scr"][2000:2500]
    d, dr = nn_dist(x, train), nn_dist(held, train)
    rt = rates(x)
    return [rt[2], float(d.median()), float((d < torch.quantile(dr, 0.01)).float().mean()), float(dr.median()), rt[0]]


def fig_lora():
    w = world()
    scr = w["scr"]
    base_x = sample_img(w["base"], 256, 5)
    base_scr = rates(base_x)[2]
    many, _, _ = finetune(scr[:2000], "full", 600, 0)
    many_rep = gen_report(sample_img(many, 256, 5), scr[:2000])
    ranks = [1, 4, 8, 32, "full"]
    sweep, npar = {}, {}
    for r in ranks:
        out = []
        for seed in (0, 1):
            merged, npar[r], net = finetune(scr[:16], r, 300, seed)
            out.append(gen_report(sample_img(merged, 256, 5), scr[:16]))
            if r == 4 and seed == 0:
                g = gen(0, 6)
                xt, t = torch.randn(64, 1, SIZE, SIZE, generator=g), torch.randint(0, T, (64,), generator=g)
                with torch.no_grad():
                    merge = {"合并前后输出的最大差": float(f"{float((merged(xt, t) - net(xt, t)).abs().max()):.1g}"), "微调后和基础模型输出的最大差": r2((w["base"](xt, t) - net(xt, t)).abs().max()),
                             "输出的典型大小（标准差）": r2(net(xt, t).std())}
        sweep[r] = np.array(out)
    over, examples = {}, None
    cps = [100, 300, 800]
    configs = [("K=4，秩 4", scr[:4], 4, 0), ("K=4，秩 4，换另外 4 张训练图", scr[4:8], 4, 1), ("K=16，秩 4", scr[:16], 4, 0)]
    before = {}
    for label, train, r, seed in configs:
        before[label] = gen_report(base_x, train)[2]
        net, params = new_finetune(r, seed)
        rows, last = [], []

        def hook(i):
            if i in cps:
                x = sample_img(merged_copy(net), 256, 5)
                rows.append(gen_report(x, train))
                last[:] = [x]

        train_diffusion(net, train, cps[-1], FT_BATCH, FULL_LR if r == "full" else LORA_LR, 10 + seed, params=params, cosine=False, hook=hook)
        over[label] = np.array(rows)
        if label == configs[1][0]:
            x = last[0]
            d = torch.cdist(x.flatten(1), train.flatten(1)) / SIZE
            order = d.min(1).values.argsort()[:8]
            examples = (train, x[order], d[order].min(1).values, d[order].argmin(1))

    def name(r):
        return "直接微调全部卷积权重" if r == "full" else f"秩 {r}"

    NUM["低秩微调：秩和参数量"] = {
        "说明": NOTE_IMG + "。" + NOTE_UNET + "。基础模型只见过斑点，现在给 16 张横向划痕图微调 300 步（Adam，批大小 16，学习率不变）。"
              "低秩微调：冻结全部原有权重，给 8 个卷积各加一个秩 r 的增量 B A（把 3x3 卷积核展平成 输出通道 x (输入通道 x 9) 的矩阵；A 随机初始化，B 从 0 开始，alpha 取 r），"
              f"只训练 A 和 B，学习率 {LORA_LR}；每层的秩不超过矩阵较短的一边，所以秩 32 时每层都是满秩。直接微调：训练 8 个卷积的权重本身，学习率 {FULL_LR}。"
              "每种设置 2 个种子，各生成 256 个样本取平均。采样前先把增量并回权重。" + JUDGE_NOTE + "。基础模型的样本被判成横向划痕的比例见“基础模型”一组",
        "微调之前基础模型的样本被判成横向划痕的比例（同样的 256 个起点）": r3(base_scr),
        "参照：给 2000 张真实横向划痕、直接微调全部卷积权重 600 步": {
            "说明": "同一个小 UNet 在样本不缺的时候能学到什么程度，1 个种子。低秩微调的结果要和它比，不要和 1 比",
            "样本被判成横向划痕的比例": r3(many_rep[0]), "样本被判成良品的比例": r3(many_rep[4])},
        "设置": [name(r) for r in ranks], "可训练参数个数": [npar[r] for r in ranks],
        "可训练参数占 8 个卷积权重的比例": [r3(npar[r] / npar["full"]) for r in ranks],
        "样本被判成横向划痕的比例": [r3(sweep[r][:, 0].mean()) for r in ranks], "各种子": [[r3(v) for v in sweep[r][:, 0]] for r in ranks],
        "到训练集最近邻距离的中位数": [r3(sweep[r][:, 1].mean()) for r in ranks], "复制比例": [r3(sweep[r][:, 2].mean()) for r in ranks],
        "留出的真实划痕到这 16 张训练图的最近邻距离中位数": r3(sweep[4][0, 3]),
        "合并": {"说明": "秩 4、种子 0 的那个模型：用作业的 lora_merge 把增量并进卷积权重、去掉旁路之后，和合并前（用 lora_forward 分两路计算）在 64 个随机输入上比较输出", **merge},
    }
    NUM["低秩微调：样本数、步数和复制"] = {
        "说明": "同上的微调，改训练图的张数 K 和步数，同一次训练在中途取三个检查点，每个检查点生成 256 个样本。每一行只跑了 1 次；K=4 用两组不同的训练图各做了一次。“微调之前”一项是把同样 256 个起点交给没微调的基础模型，按同样办法统计，用来看这个比例里有多少和微调无关。复制比例：生成样本到训练集的最近邻距离，比“500 张留出的真实划痕到同一个训练集的最近邻距离的 1% 分位”还小的比例；"
              "真实的新样本按定义约为 0.01，明显超过它说明生成器在复制训练图。距离是逐像素差的均方根（归一化后的灰度）",
        "微调步数": cps,
        **{label: {"样本被判成横向划痕的比例": [r3(v) for v in rows[:, 0]], "到训练集最近邻距离的中位数": [r3(v) for v in rows[:, 1]],
                   "复制比例": [r3(v) for v in rows[:, 2]], "微调之前基础模型的样本按同样办法统计的复制比例": r3(before[label]),
                   "留出的真实划痕到训练集的最近邻距离中位数": r3(rows[0, 3])} for label, rows in over.items()},
    }

    fig = plt.figure(figsize=(17.5, 4.3), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.3])
    ax = fig.add_subplot(gs[0])
    xs = [npar[r] for r in ranks[:-1]]
    ax.plot(xs, [sweep[r][:, 0].mean() for r in ranks[:-1]], "o-", color=S.BLUE, lw=1.8, ms=6, label="低秩微调（数字是秩）")
    for r, x in zip(ranks[:-1], xs):
        ax.plot([x, x], sweep[r][:, 0], ".", color=S.INK, ms=4)
        if r == 32:                                            # 和旁边“直接微调”的点挨得太近，标在下面
            ax.text(x, sweep[r][:, 0].min() - 0.03, str(r), ha="center", va="top", fontsize=9, color=S.BLUE)
        else:
            ax.text(x, sweep[r][:, 0].max() + 0.025, str(r), ha="center", fontsize=9, color=S.BLUE)
    ax.plot([npar["full"]], [sweep["full"][:, 0].mean()], "D", color=S.ORANGE, ms=7, label="直接微调全部卷积权重")
    ax.plot([npar["full"]] * 2, sweep["full"][:, 0], ".", color=S.INK, ms=4)
    ax.axhline(base_scr, color=S.MUTED, lw=1, ls=":")
    ax.text(1100, base_scr + 0.02, "微调之前的基础模型", ha="left", fontsize=8.5, color=S.MUTED)
    ax.axhline(many_rep[0], color=S.MUTED, lw=1, ls=":")
    ax.text(1100, many_rep[0] + 0.02, "2000 张真实划痕直接微调", ha="left", fontsize=8.5, color=S.MUTED)
    ax.set_xscale("log")
    ax.set_xticks([1000, 3000, 10000, 30000], ["1000", "3000", "10000", "30000"])
    ax.minorticks_off()
    ax.set_ylim(-0.04, 1.0)
    ax.set_xlabel("可训练参数个数")
    ax.set_ylabel("样本被判成横向划痕的比例")
    ax.set_title("16 张图微调 300 步：秩 1 最低，直接微调最高（黑点是各个种子）", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax)
    ax = fig.add_subplot(gs[1])
    for (label, rows), color, style in zip(over.items(), (S.RED, S.PURPLE, S.BLUE), ("o-", "v-", "s-")):
        ax.plot(np.arange(len(cps)), rows[:, 2], style, color=color, lw=1.8, ms=5.5, label=label)
    ax.axhline(0.01, color=S.MUTED, lw=1, ls=":")
    ax.text(len(cps) - 1, 0.015, "真实新样本的水平", ha="right", fontsize=8.5, color=S.MUTED)
    ax.set_xticks(np.arange(len(cps)), [str(c) for c in cps])
    ax.set_xlabel("微调步数")
    ax.set_ylabel("复制比例")
    ax.set_title("只有 4 张训练图时训练越久复制得越多，16 张时没有", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax)
    sub = gs[2].subgridspec(3, 4)
    tr, gx, gd, gi = examples
    for i in range(4):
        a = fig.add_subplot(sub[0, i])
        show(a, tr[i, 0])
        a.set_title(f"训练图 {i + 1}", fontsize=9)
        for row in (1, 2):
            j = (row - 1) * 4 + i
            a = fig.add_subplot(sub[row, i])
            show(a, gx[j, 0])
            a.set_title(f"最像训练图 {int(gi[j]) + 1}，距离 {float(gd[j]):.2f}", fontsize=8.5)
    S.save(fig, CH, "fig06_低秩微调")


# ---------------------------------------------------------------- 图 7：修复式生成，以及闭环实验
_ramp = np.minimum.reduce([np.arange(1, PB + 1)[:, None] * np.ones((1, PB)), np.arange(PB, 0, -1)[:, None] * np.ones((1, PB)),
                           np.arange(1, PB + 1)[None, :] * np.ones((PB, 1)), np.arange(PB, 0, -1)[None, :] * np.ones((PB, 1))])
FEATHER = np.clip(_ramp / 3.0, 0, 1)                    # 从图块边缘向里 3 px 线性过渡到 1
_n = PB - 2
_S = np.sin(np.pi * np.outer(np.arange(1, _n + 1), np.arange(1, _n + 1)) / (_n + 1))
_lam = 2 - 2 * np.cos(np.pi * np.arange(1, _n + 1) / (_n + 1))
_LAM = _lam[:, None] + _lam[None, :]


def poisson_patch(tpatch, spatch):
    """泊松融合（矩形图块），和第 20 章配图脚本的做法相同：最外一圈取目标图，内部保持源图块的拉普拉斯。"""
    diff = tpatch - spatch
    b = np.zeros((_n, _n))
    b[0, :] += diff[0, 1:-1]
    b[-1, :] += diff[-1, 1:-1]
    b[:, 0] += diff[1:-1, 0]
    b[:, -1] += diff[1:-1, -1]
    out = tpatch.copy()
    out[1:-1, 1:-1] = spatch[1:-1, 1:-1] + _S @ ((_S @ b @ _S) / _LAM) @ _S * (2.0 / (_n + 1)) ** 2
    return out


def seam_stat(img, y0, x0, k):
    """跨图块边界的平均绝对差（第 20 章 4.2 节的统计量）。k=0：图块最外一圈和它外面的邻居；k=1：最外一圈和它里面的邻居。"""
    a, b = (y0 + k, x0 + k), (y0 + PB - 1 - k, x0 + PB - 1 - k)
    rows, cols = slice(y0, y0 + PB), slice(x0, x0 + PB)
    parts = [np.abs(img[a[0], cols] - img[a[0] - 1, cols]), np.abs(img[b[0], cols] - img[b[0] + 1, cols]),
             np.abs(img[rows, a[1]] - img[rows, a[1] - 1]), np.abs(img[rows, b[1]] - img[rows, b[1] + 1])]
    return float(np.mean(np.concatenate(parts)))


def centroid_box(m):
    """以掩膜质心为中心的 PB x PB 方框的左上角，四周至少留 1 px。"""
    ys, xs = torch.tensor(YY, dtype=torch.float32), torch.tensor(XX, dtype=torch.float32)
    area = m.sum((1, 2, 3))
    cy, cx = (m[:, 0] * ys).sum((1, 2)) / area, (m[:, 0] * xs).sum((1, 2)) / area
    f = lambda c: torch.clamp(torch.round(c).long() - PB // 2 + 1, 1, SIZE - PB - 1)
    return f(cy).numpy(), f(cx).numpy()


INPAINT: dict = {}


def run_inpaint():
    w = world()
    base, ctrl, n = w["base"], get_ctrl(True)[0], 300
    tgt, ref, req_m = w["good"][:n], w["good"][n:2 * n], w["req_m"][:n]
    y0, x0 = centroid_box(req_m)
    region = torch.zeros(n, 1, SIZE, SIZE)
    for i in range(n):
        region[i, 0, y0[i]:y0[i] + PB, x0[i]:x0[i] + PB] = 1
    src, src_m = w["blob"][2000:2000 + n], w["blob_m"][2000:2000 + n]
    sy, sx = centroid_box(src_m)
    outs = {"扩散修复，不加条件": sample_img(base, n, 7, known=tgt, region=region),
            "扩散修复，加掩膜条件": sample_img(base, n, 7, ctrl=ctrl, mask=req_m, known=tgt, region=region)}
    for key, name in (("direct", "直接粘贴"), ("feather", "羽化融合"), ("poisson", "泊松融合")):
        out = tgt.clone().numpy().astype(np.float64)
        for i in range(n):
            sp = src[i, 0, sy[i]:sy[i] + PB, sx[i]:sx[i] + PB].numpy().astype(np.float64)
            tp = out[i, 0, y0[i]:y0[i] + PB, x0[i]:x0[i] + PB].copy()
            new = hw20.blend(tp, sp, np.ones((PB, PB))) if key == "direct" else hw20.blend(tp, sp, FEATHER) if key == "feather" else poisson_patch(tp, sp)
            out[i, 0, y0[i]:y0[i] + PB, x0[i]:x0[i] + PB] = new
        outs[name] = torch.tensor(out, dtype=torch.float32)
    free = sample_img(base, n, 7, ctrl=ctrl, mask=req_m)                      # 采样过程中完全不管良品图
    bugs = {"已知区域直接贴回没加噪的原图": sample_img(base, n, 7, ctrl=ctrl, mask=req_m, known=tgt, region=region, noised=False),
            "替换时用了这一步之前的噪声水平": sample_img(base, n, 7, ctrl=ctrl, mask=req_m, known=tgt, region=region, level="t"),
            "中间各步都不替换，只在最后把方框外贴回良品图": region * free + (1 - region) * tgt}
    rf = ref[:, 0].numpy()
    base_stat = [np.array([seam_stat(rf[i], y0[i], x0[i], k) for i in range(n)]) for k in (0, 1)]

    def report(out):
        o = out[:, 0].numpy()
        seam = [auroc(base_stat[k], np.array([seam_stat(o[i], y0[i], x0[i], k) for i in range(n)])) for k in (0, 1)]
        res = residual(out).abs()
        rt = rates(out)
        return {"方框外的最大改动": float(f"{float(((out - tgt).abs() * (1 - region)).max()):.2g}"),
                "跨边界差的 AUROC：对外": r3(seam[0]), "跨边界差的 AUROC：对内": r3(seam[1]),
                "方框内有缺陷的比例": r3(((res * region).amax(dim=(1, 2, 3)) > w["thr"]).float().mean()),
                "判成（良品，斑点，横向划痕）": [r3(v) for v in rt]}

    whole = free
    INPAINT.update(outs=outs, tgt=tgt, req_m=req_m, y0=y0, x0=x0)
    cond = outs["扩散修复，加掩膜条件"]
    NUM["修复式生成"] = {
        "说明": NOTE_IMG + "。" + NOTE_UNET + f"。300 张良品图，每张指定一个斑点形状的掩膜，修复区域是以掩膜质心为中心的 {PB}x{PB} 方框（四周至少留 1 px）。"
              "扩散修复：DDIM 20 步，每一步之后用作业的 inpaint_replace 把方框外换成“良品图加噪到当前噪声水平”的结果，最后一步的噪声水平是 0，所以方框外和良品图逐像素相同。"
              "“加掩膜条件”指同时接上图 5 那个零初始化训练的控制旁路。三种贴图方法和第 20 章 4.1 节相同：从另外 300 张真实斑点图上各剪下以缺陷为中心的方框，贴到同样的位置；羽化的过渡带 3 px。"
              "跨边界差的 AUROC 是第 20 章 4.2 节的统计量：方框四条边上相邻像素的平均绝对差，和另外 300 张没动过的良品图在同样位置上的值比；0.5 是分不开，大于 0.5 是边界处的差比正常的大，小于 0.5 是比正常的小。" + METRIC_NOTE,
        "方框面积（像素）": PB * PB, "方框面积占整张图的比例": PB * PB / SIZE ** 2,
        "AUROC 的抽样标准差（两组各 300 张、两组没有差别时，约为 sqrt(1 / (6 x 300))）": r3(math.sqrt(1 / (6 * n))),
        **{name: report(out) for name, out in outs.items()},
        "扩散修复，加掩膜条件：缺陷和指定掩膜（掩膜内能量占比，IoU，掩膜内有缺陷）": [r3(v) for v in mask_metrics(cond, req_m)],
        "扩散修复，不加条件：缺陷和指定掩膜（同上三个数）": [r3(v) for v in mask_metrics(outs["扩散修复，不加条件"], req_m)],
        "对照：只加掩膜条件、整张图重新生成，方框外和良品图的平均绝对差": r3(((whole - tgt).abs() * (1 - region)).sum() / (1 - region).sum()),
        "对照：两张不同的良品图之间的平均绝对差": r3((ref - tgt).abs().mean()),
        "三种写错的替换": {"说明": "都用加掩膜条件的那一组设置，只改替换这一步", **{name: report(out) for name, out in bugs.items()},
                     **{name + "：缺陷和指定掩膜（三个数）": [r3(v) for v in mask_metrics(out, req_m)] for name, out in bugs.items()}},
    }


LOOP_K, LOOP_SEEDS, CLF_STEPS, N_SYN = (4, 16), (0, 1, 2), 150, 300
ALT_STEPS, ALT_SYN = 200, 400          # 第二组设置：生成样本多用 100 张，分类器多训练 50 步，只重复“只用真实”和“真实加生成”两行
LOOP_CONDS = ["只用真实的 K 张", "真实 + 残差移植 300 张", "真实 + 程序化合成 300 张", "真实 + 微调后的生成器采样 300 张", "真实 + 生成样本里被判成划痕的那部分（诊断用）"]
LOOP: dict = {}


def run_loop():
    w = world()
    rt = np.random.default_rng(99)
    te_neg = torch.cat([make(700, rt, "good")[0], make(700, rt, "blob")[0]])
    te_pos = make(1000, rt, "scratch")[0]

    def evaluate(pos, neg, seed, steps=CLF_STEPS):
        def batch(g):
            return (torch.cat([pos[torch.randint(0, len(pos), (16,), generator=g)], neg[torch.randint(0, len(neg), (16,), generator=g)]]),
                    torch.cat([torch.ones(16), torch.zeros(16)]))
        net = fit_clf(batch, 1, steps, seed)
        with torch.no_grad():
            sn, sp = net(te_neg)[:, 0].numpy().astype(np.float64), net(te_pos)[:, 0].numpy().astype(np.float64)
        return [auroc(sn, sp), float((sp > np.quantile(sn, 0.95)).mean())]

    def transplant(real, n, rng):
        res, bg = residual(real), make(n, rng, "good")[0]
        idx = rng.integers(0, len(real), n)
        return torch.stack([bg[i] + torch.roll(res[j], shifts=(int(rng.integers(-3, 4)), int(rng.integers(-3, 4))), dims=(1, 2)) for i, j in enumerate(idx)])

    def procedural(n, rng):
        clean = texture(n, rng)
        for i in range(n):
            cy, cx = rng.uniform(4.5, SIZE - 5.5, size=2)
            length, th = rng.uniform(3, 8), np.deg2rad(rng.uniform(-30, 30))
            p0, p1 = (cy - length / 2 * np.sin(th), cx - length / 2 * np.cos(th)), (cy + length / 2 * np.sin(th), cx + length / 2 * np.cos(th))
            clean[i] = hw20.blend(clean[i], clean[i] + rng.choice([-1.0, 1.0]) * rng.uniform(0.2, 0.4), hw20.scratch_mask((SIZE, SIZE), p0, p1, rng.uniform(1.0, 2.0), 1.0))
        return finish(clean, rng)

    res, gen_stats, alt = {}, {}, {}
    for k in LOOP_K:
        rows, gs, alt_rows = {c: [] for c in LOOP_CONDS}, [], []
        for seed in LOOP_SEEDS:
            rng = np.random.default_rng([seed, k])
            real = make(k, rng, "scratch")[0]
            neg = torch.cat([make(300, rng, "good")[0], make(300, rng, "blob")[0]])
            net = finetune(real, 4, 300, seed)[0]
            fake_all = sample_img(net, ALT_SYN, 50 + seed)
            fake = fake_all[:N_SYN]
            alt_rows.append(evaluate(real, neg, seed, ALT_STEPS) + evaluate(torch.cat([real, fake_all]), neg, seed, ALT_STEPS))
            with torch.no_grad():
                looks = w["judge"](fake).argmax(1)
            rep = gen_report(fake, real)
            gs.append([float((looks == 2).float().mean()), float((looks == 1).float().mean()), float((looks == 0).float().mean()), rep[2]])
            picked = fake[looks == 2]
            pools = [real, torch.cat([real, transplant(real, N_SYN, rng)]), torch.cat([real, procedural(N_SYN, rng)]), torch.cat([real, fake]),
                     torch.cat([real, picked]) if len(picked) else real]
            for c, pos in zip(LOOP_CONDS, pools):
                rows[c].append(evaluate(pos, neg, seed))
        res[k] = {c: np.array(v) for c, v in rows.items()}
        gen_stats[k] = np.array(gs)
        alt[k] = np.array(alt_rows)
    upper = np.array([evaluate(make(512, np.random.default_rng([s, 512]), "scratch")[0],
                               torch.cat([make(300, np.random.default_rng([s, 513]), "good")[0], make(300, np.random.default_rng([s, 514]), "blob")[0]]), s) for s in LOOP_SEEDS])
    LOOP.update(res=res, upper=upper)
    NUM["闭环实验"] = {
        "说明": NOTE_IMG + "。" + NOTE_UNET + "。任务：新出现了一类缺陷（横向划痕），手上只有 K 张真实的；老的缺陷（斑点）和良品都很多。要训练一个“是不是横向划痕”的小分类器，"
              "负样本是 300 张良品加 300 张斑点图，测试集是留出的 1000 张真实横向划痕、700 张良品、700 张斑点图。检出率指阈值定在测试负样本分数的 95% 分位（误报 5%）时的检出率。"
              f"分类器是三层卷积加全局最大池化，训练 {CLF_STEPS} 步，每步正负样本各 16 张；K 取 4 和 16，每个 K 做 3 个种子（真实样本、负样本、生成器的微调都跟着换）。"
              "微调后的生成器：只见过斑点的基础模型，用这 K 张真实划痕做秩 4 的低秩微调 300 步，再用 DDIM 20 步采 300 张，全部当作划痕的正样本。"
              "残差移植：把 K 张真实划痕图减去各列中位数得到的残差，随机平移 3 px 以内，加到 300 张新的良品图上（相当于第 20 章里相位对齐的剪贴）。"
              "程序化合成：用第 20 章作业的 scratch_mask 和 blend 在成像前画线段（长 3 到 8 px，夹角 30 度以内，宽 1 到 2 px，幅度 0.2 到 0.4），形状参数和真实划痕不同。"
              "“诊断用”一行：用图 1 那个见过 1500 张真实划痕的判别器，把生成样本里判成划痕的挑出来当正样本；实际工作里没有这个判别器，这一行只用来看问题出在哪",
        "K": list(LOOP_K),
        **{c: {"AUROC": [r3(res[k][c][:, 0].mean()) for k in LOOP_K], "检出率": [r3(res[k][c][:, 1].mean()) for k in LOOP_K],
               "各种子的 AUROC": [[r3(v) for v in res[k][c][:, 0]] for k in LOOP_K]} for c in LOOP_CONDS},
        "512 张真实划痕（参照）": {"AUROC": r3(upper[:, 0].mean()), "检出率": r3(upper[:, 1].mean()), "各种子的 AUROC": [r3(v) for v in upper[:, 0]]},
        f"换一组设置：生成样本 {ALT_SYN} 张、分类器训练 {ALT_STEPS} 步": {
            "说明": "真实样本、负样本、微调出的生成器都和上面相同，只是生成样本多用 100 张、分类器多训练 50 步。用来看“生成样本有没有帮助”这个结论对设置敏不敏感",
            "只用真实的 K 张": {"AUROC": [r3(alt[k][:, 0].mean()) for k in LOOP_K], "检出率": [r3(alt[k][:, 1].mean()) for k in LOOP_K],
                           "各种子的 AUROC": [[r3(v) for v in alt[k][:, 0]] for k in LOOP_K]},
            f"真实 + 微调后的生成器采样 {ALT_SYN} 张": {"AUROC": [r3(alt[k][:, 2].mean()) for k in LOOP_K], "检出率": [r3(alt[k][:, 3].mean()) for k in LOOP_K],
                                            "各种子的 AUROC": [[r3(v) for v in alt[k][:, 2]] for k in LOOP_K]}},
        "微调后的生成器采出的 300 张": {
            "被判成横向划痕的比例": [r3(gen_stats[k][:, 0].mean()) for k in LOOP_K], "被判成斑点的比例": [r3(gen_stats[k][:, 1].mean()) for k in LOOP_K],
            "被判成良品的比例": [r3(gen_stats[k][:, 2].mean()) for k in LOOP_K], "复制比例": [r3(gen_stats[k][:, 3].mean()) for k in LOOP_K],
            "各种子被判成横向划痕的比例": [[r3(v) for v in gen_stats[k][:, 0]] for k in LOOP_K]},
    }


def fig_inpaint_loop():
    run_inpaint()
    run_loop()
    o, tgt, req_m, y0, x0 = INPAINT["outs"], INPAINT["tgt"], INPAINT["req_m"], INPAINT["y0"], INPAINT["x0"]
    fig = plt.figure(figsize=(18, 4.8), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.2, 1.25])
    left = gs[0].subgridspec(4, 6)
    rows = [("良品图\n红框是修复区域", tgt, False), ("扩散修复\n不加条件", o["扩散修复，不加条件"], False), ("扩散修复\n加掩膜条件", o["扩散修复，加掩膜条件"], True), ("直接粘贴", o["直接粘贴"], False)]
    for r, (name, data, with_mask) in enumerate(rows):
        for i in range(6):
            ax = fig.add_subplot(left[r, i])
            show(ax, data[i, 0], req_m[i, 0] if with_mask else None, color="cyan", box=(y0[i], x0[i]))
            if i == 0:
                ax.set_ylabel(name, fontsize=8.5)
    ax = fig.add_subplot(gs[1])
    names = list(o)
    gx = np.arange(len(names))
    for i, (key, label, color) in enumerate((("跨边界差的 AUROC：对外", "最外一圈和它外面的邻居", S.RED), ("跨边界差的 AUROC：对内", "最外一圈和它里面的邻居", S.BLUE))):
        vals = [NUM["修复式生成"][n][key] for n in names]
        bars = ax.bar(gx + (i - 0.5) * 0.38, [v - 0.5 for v in vals], 0.36, bottom=0.5, color=color, alpha=0.88, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.012 if v >= 0.5 else -0.012), f"{v:.2f}", ha="center", va="bottom" if v >= 0.5 else "top", fontsize=8)
    ax.axhline(0.5, color=S.INK, lw=1)
    ax.set_xticks(gx, [n.replace("，", "\n") for n in names], fontsize=8.5)
    ax.set_ylim(0.25, 1.0)
    ax.set_ylabel("跨边界差能认出改动的 AUROC（0.5 是认不出）")
    ax.set_title("扩散修复在方框边界上的痕迹比三种贴图都轻", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax)
    ax.grid(axis="x", visible=False)
    ax = fig.add_subplot(gs[2])
    xs = np.arange(len(LOOP_K))
    colors = [S.MUTED, S.GREEN, S.PURPLE, S.RED, S.ORANGE]
    for i, (c, color) in enumerate(zip(LOOP_CONDS, colors)):
        vals = [LOOP["res"][k][c][:, 0].mean() for k in LOOP_K]
        rects = ax.bar(xs + (i - 2) * 0.17, vals, 0.16, color=color, alpha=0.9, label=c)
        for b, v, k in zip(rects, vals, LOOP_K):
            pts = LOOP["res"][k][c][:, 0]
            ax.plot([b.get_x() + b.get_width() / 2] * len(pts), pts, ".", color=S.INK, ms=4)
            ax.text(b.get_x() + b.get_width() / 2, pts.max() + 0.012, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.axhline(0.5, color=S.MUTED, lw=1, ls=":")
    ax.set_xlim(-0.55, 1.85)
    ax.text(1.83, 0.51, "随机猜", ha="right", fontsize=8.5, color=S.MUTED)
    ax.set_xticks(xs, [f"真实划痕 {k} 张" for k in LOOP_K])
    ax.set_ylim(0.4, 1.08)
    ax.set_ylabel("留出的真实划痕上的 AUROC")
    ax.set_title("闭环：加进生成样本，没有一档赶上两种简单的办法", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
    S.clean(ax)
    ax.grid(axis="x", visible=False)
    S.save(fig, CH, "fig07_修复与闭环")


# ---------------------------------------------------------------- 几种写错的实现各是什么后果
def numbers_bugs():
    real = MIX.sample(N2D, gen(7))[0]
    cls = torch.arange(3).repeat_interleave(N2D // 3 + 1)[:N2D]
    ref_u = MIX.blob_stats(real)[0]
    ref_c = MIX.blob_stats(MIX.sample(N2D, gen(9), cls)[0], cls)[0]

    def adherence(fn):
        return r3((MIX.bayes(sample_2d(fn, 50, "ddim", 13)) == cls).float().mean())

    def swapped(x, t):
        a = float(ABAR[t])
        return hw.cfg_combine(MIX.eps(x, a, cls), MIX.eps(x, a), 3.0)

    def plus_one(x, t):                                       # 论文里的写法是 (1 + w) 有条件 - w 无条件；把库里的 w 直接代进去，等于多加了 1
        a = float(ABAR[t])
        return hw.cfg_combine(MIX.eps(x, a), MIX.eps(x, a, cls), 1.0 + 1.0)

    g = gen(21, 31)
    x = torch.randn(N2D, 2, generator=g)
    ts = timesteps(50)
    for i, t in enumerate(ts):                                # eta=1，但每一步都没有加随机噪声
        a_t, a_prev = float(ABAR[t]), float(ABAR[ts[i + 1]]) if i + 1 < len(ts) else 1.0
        x = hw.ddim_step(x, MIX.eps(x, a_t), a_t, a_prev, 1.0, torch.zeros_like(x))
    good = sample_2d(exact_eps_fn(MIX), 50, "post", 21)
    w = world()
    xt, t = torch.randn(64, 1, SIZE, SIZE, generator=gen(0, 6)), torch.randint(0, T, (64,), generator=gen(0, 7))
    net, params = new_finetune(4, 0)
    with torch.no_grad():
        for c in net.convs():
            c.B.copy_(torch.randn(c.B.shape, generator=gen(1, 8)) * 0.1)
        d_rand = float((net(xt, t) - w["base"](xt, t)).abs().max())
    net0, params0 = new_finetune(4, 0)
    with torch.no_grad():
        for c in net0.convs():
            c.A.zero_()
    eps = torch.randn(xt.shape, generator=gen(0, 9))
    ((net0(xt, t) - eps) ** 2).mean().backward()
    NUM["写错的实现"] = {
        "说明": "把几个零件故意写错，看结果变成什么样。二维的几项用解析去噪器、50 步；引导的几项要求 3 个类别各三分之一；团内标准差比的分母是真实样本按同样办法统计的值。修复的三种错误见“修复式生成”一组",
        "引导：w=3 正确写法的条件符合率": adherence(exact_eps_fn(MIX, cls, 3.0)),
        "引导：有条件和无条件的位置写反（w=3）的条件符合率": adherence(swapped),
        "引导：随机猜的条件符合率": r3(1 / 3),
        "引导：想要 w=1（普通条件生成）却按论文的 (1+w) 写法多加了 1，团内标准差比": r2(MIX.blob_stats(sample_2d(plus_one, 50, "ddim", 13), cls)[0] / ref_c),
        "引导：w=1 正确写法的团内标准差比": r2(MIX.blob_stats(sample_2d(exact_eps_fn(MIX, cls, 1.0), 50, "ddim", 13), cls)[0] / ref_c),
        "DDPM 50 步（eta=1）：正确写法到真实分布的距离": r3(sliced_w(good, real)), "DDPM 50 步（eta=1）：正确写法的团内标准差比": r2(MIX.blob_stats(good)[0] / ref_u),
        "DDPM 50 步（eta=1）：忘了加随机噪声，到真实分布的距离": r3(sliced_w(x, real)), "DDPM 50 步（eta=1）：忘了加随机噪声，团内标准差比": r2(MIX.blob_stats(x)[0] / ref_u),
        "LoRA：B 没有从 0 开始（标准差 0.1 的随机数），训练前输出和基础模型的最大差": r2(d_rand),
        "LoRA：A 和 B 都从 0 开始，第一次反传时 A 和 B 的梯度的最大绝对值": float(max(float(p.grad.abs().max()) for p in params0)),
    }


# ---------------------------------------------------------------- 潜空间扩散的动机：计算量的核算
def conv_macs(size, c=16):
    """本章小 UNet 的 8 个 3x3 卷积在输入边长 size 时的乘加次数（每个输出位置 9 x 输入通道 x 输出通道 次）。"""
    full, half = size * size, (size // 2) ** 2
    return 9 * (full * (1 * c + c * c) + half * (c * 2 * c + 2 * (2 * c) * (2 * c)) + half * (2 * c * c) + full * (c * c + c * 1))


def numbers_latent():
    sizes = [16, 32, 64, 128]
    macs = [conv_macs(s) for s in sizes]
    NUM["潜空间扩散的计算量"] = {
        "说明": "两部分。一是本章小 UNet 的卷积乘加次数随输入边长的变化，按公式数出来的（全卷积网络，和像素数成正比）；实测的前向耗时见“耗时”一组。"
              "二是 Stable Diffusion 1.x 的公开设计：自编码器把图像的边长压到八分之一、通道变成 4 个，扩散在这个潜变量上做。这一部分是按尺寸做的算术，不是实测",
        "输入边长": sizes, "小 UNet 一次前向的卷积乘加次数": macs, "相对 16x16 的倍数": [m // macs[0] for m in macs],
        "边长每加倍一次，像素数和卷积乘加次数的倍数": 4, "边长每加倍一次，全图自注意力里“位置对”的个数的倍数": 16,
        "Stable Diffusion 1.x": {
            "图像边长": 512, "潜变量边长": 512 // 8, "下采样倍数": 8, "潜变量通道数": 4,
            "图像的空间位置数": 512 * 512, "潜变量的空间位置数": (512 // 8) ** 2, "空间位置数之比": 8 * 8,
            "图像的数值个数（3 通道）": 512 * 512 * 3, "潜变量的数值个数": (512 // 8) ** 2 * 4, "数值个数之比": 512 * 512 * 3 // ((512 // 8) ** 2 * 4),
            "同一种自注意力放在像素上和放在潜变量上，“位置对”个数之比": (8 * 8) ** 2,
            "一个潜变量位置对应的像素块": "8x8", "一个 8x8 像素块的像素数": 64,
            "图像边长 256 时的潜变量边长": 256 // 8, "图像边长 768 时的潜变量边长": 768 // 8,
        },
    }


# ---------------------------------------------------------------- 其余数字
def numbers_misc():
    a_t, a_prev = 0.25, 0.64
    x_t, eps = torch.tensor([1.0]), torch.tensor([0.5])
    x0_hat = hw.convert_prediction(x_t, eps, a_t, "eps")[0]
    v = hw.convert_prediction(x_t, eps, a_t, "eps")[2]
    det = hw.ddim_step(x_t, eps, a_t, a_prev)
    sto0 = hw.ddim_step(x_t, eps, a_t, a_prev, 1.0, torch.zeros(1))
    sto1 = hw.ddim_step(x_t, eps, a_t, a_prev, 1.0, torch.ones(1))
    ts20 = timesteps(20)
    NUM["作业自检"] = {
        "说明": "用参考实现算的一个小例子，可以拿来核对自己的实现：x_t = 1，预测的噪声 = 0.5，abar_t = 0.25，abar_prev = 0.64",
        "换算出的原图估计": r3(x0_hat), "换算出的 v": r3(v),
        "DDIM 一步（eta=0）的结果": r3(det), "eta=1、噪声取 0 时的结果": r3(sto0), "eta=1 时噪声的系数（噪声取 1 和取 0 的结果之差）": r3(sto1 - sto0),
        "引导的组合：无条件 0.2、有条件 0.5、w=3 时的结果": r3(hw.cfg_combine(torch.tensor(0.2), torch.tensor(0.5), 3.0)),
    }
    NUM["理论数值"] = {
        "说明": "由脚本算出的常数，讲义里直接引用。" + SCHED_NOTE,
        "总步数": T, "abar[0]": r4(ABAR[0]), "abar[999]": float(f"{ABAR[-1]:.1g}"),
        "t=999 时 1 / sqrt(abar)（噪声预测的误差换算成原图时放大的倍数）": float(f"{1 / math.sqrt(ABAR[-1]):.3g}"),
        "t=0 时 sqrt(abar) / sqrt(1 - abar)（原图预测的误差换算成噪声时放大的倍数）": float(f"{math.sqrt(ABAR[0] / (1 - ABAR[0])):.3g}"),
        "DDIM 20 步用到的步数下标": [int(t) for t in ts20], "相邻两个下标的间隔": [int(ts20[0] - ts20[1]), int(ts20[1] - ts20[2])],
        "DDIM 5 步用到的步数下标": [int(t) for t in timesteps(5)],
        "1000 步和 20 步的网络调用次数之比": T // 20, "1000 步和 50 步的网络调用次数之比": T // 50,
        "引导每一步要算的预测个数（无条件加有条件）": 2,
        "引导的两种写法": {"论文里的 w 和库里的 guidance_scale 的关系": "guidance_scale = 1 + w", "库里 7.5 对应论文的 w": 6.5},
        "二维 MLP 的参数个数": n_params(MLP().parameters()),
        "16x16 的图的像素数": SIZE * SIZE, "小 UNet 的 8 个卷积的权重个数": n_params(c.conv.weight for c in UNet().convs()),
    }
    NUM["论文和库里的设定（引用，不是实测）"] = {
        "说明": "讲义里引用的几个常数，出自论文和 diffusers 的文档，本章没有验证，以原文和当前文档为准。列在这里只是为了让讲义里的每个数都有出处",
        "Stable Diffusion 1.x 的默认输出边长": 512, "自编码器的下采样倍数": 8, "潜变量通道数": 4,
        "diffusers 里 StableDiffusionPipeline 的默认值": {"num_inference_steps": 50, "guidance_scale": 7.5},
        "半精度浮点的位数": 16, "单精度浮点的位数": 32,
    }
    NUM["演示页"] = {
        "说明": "demos/采样与引导.html 用的就是本章的二维数据（6 个团心在半径 2 的圆上，团的标准差 0.6，3 类）和同一张噪声表，去噪器是解析式，每次 600 个点；页面里的数值在浏览器里实时算，随机数和本脚本不同，讲义只引用方向",
        "点数": 600, "团心半径": R2D, "团的标准差": 0.6, "可选的步数": [5, 10, 20, 50, 200], "引导强度的范围": [0, 12],
    }


def time_forward():
    out = {}
    net = UNet().eval()
    for size in (16, 32, 64):
        x, t = torch.randn(16, 1, size, size), torch.zeros(16, dtype=torch.long)
        best = float("inf")
        with torch.no_grad():
            for _ in range(5):
                t0 = time.perf_counter()
                for _ in range(5):
                    net(x, t)
                best = min(best, (time.perf_counter() - t0) / 5)
        out[size] = best
    return out


def main() -> None:
    S.setup()
    plt.rcParams["svg.hashsalt"] = "ch29"
    t_all = time.perf_counter()
    for fn in (numbers_misc, fig_gaps, fig_samplers, fig_param, fig_cfg, fig_control, fig_lora, fig_inpaint_loop, numbers_bugs, numbers_latent):
        t = time.perf_counter()
        fn()
        TIMES[fn.__name__] = time.perf_counter() - t
        print(f"完成 {fn.__name__}（{TIMES[fn.__name__]:.0f} 秒）", flush=True)
    total = time.perf_counter() - t_all

    def measure():
        fwd = time_forward()
        return {
            "说明": "在一台 Mac 的 CPU 上（2 个线程，同时还有别的程序在跑）测的一次结果，只看量级和倍数关系，不是显卡上的数字。设置 CH29_RETIME=1 重测。"
                  "前向耗时是小 UNet 在批大小 16 时一次前向的时间，每个尺寸测 5 组、每组 5 次，取最快的一组",
            "基础模型每步训练耗时（毫秒，批大小 32）": float(f"{1000 * TIMES['base_step']:.2g}"),
            "小 UNet 一次前向（毫秒，批大小 16），输入边长 16、32、64": [float(f"{1000 * fwd[s]:.2g}") for s in (16, 32, 64)],
            "前向耗时相对 16x16 的倍数，输入边长 32、64": [float(f"{fwd[s] / fwd[16]:.2g}") for s in (32, 64)],
            "整个配图脚本（秒）": float(f"{total:.2g}"),
        }

    NUM["耗时"] = S.pinned(CH, "耗时", measure, "CH29_RETIME")
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=2))
    print("各部分用时：", {k: round(v, 1) for k, v in TIMES.items()})
    print(f"总耗时 {total:.0f} 秒")


if __name__ == "__main__":
    main()
