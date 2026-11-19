"""第 25 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch25.py

输出：<主分支>/25_Transformer与ViT/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。

全部实验用代码生成的小图（4 种 3x3 的图案加噪声）和从零训练的很小的 ViT、CNN，在 CPU 上跑，不下载任何数据和权重。
切块、多头注意力、位置编码调用本章作业的参考实现。torchvision 的 vit_b_16 只用不带权重的结构数参数量。
耗时默认沿用上次的记录，设置 CH25_RETIME=1 才重测。
"""

from __future__ import annotations

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

CH = "25_Transformer与ViT"
mod = S.load_solution(CH, "25-1_ViT核心零件.py")
NUM: dict = {}
KEEP: dict = {}          # 训练好的模型，后面的图接着用
SEEDS = (0, 1, 2)
NOISE, STEPS, LR, BATCH = 0.15, 500, 3e-3, 32
LN4, LN2 = math.log(4), math.log(2)
DATA_NOTE = (f"图案是 3x3 的横线、竖线、斜线、反斜线四种，幅度 1，叠在标准差 {NOISE} 的高斯噪声上；"
             f"训练用 Adam（学习率 {LR}，批 {BATCH}），除非另外说明每步新生成一批数据、共 {STEPS} 步；测试集 1000 张")
VIT_NOTE = "小 ViT：块大小 4，嵌入 32 维，2 层，4 个头，MLP 隐藏层 64 维，LayerNorm 放在前面，二维正余弦位置编码，全局平均汇聚"


def r3(v):
    return round(float(v), 3)


def r4(v):
    return round(float(v), 4)


def sig2(v):
    return float(f"{float(v):.2g}")


# ---------------------------------------------------------------- 数据
GLYPHS = np.zeros((4, 3, 3), np.float32)
GLYPHS[0, 1, :] = 1
GLYPHS[1, :, 1] = 1
GLYPHS[2, [0, 1, 2], [0, 1, 2]] = 1
GLYPHS[3, [0, 1, 2], [2, 1, 0]] = 1
GLYPH_NAMES = ["横线", "竖线", "斜线", "反斜线"]


def data_type(n, rng, size=16, region=None, align=False):
    """一个图案，类别是图案的种类。region=(x0, x1) 限制左上角的列范围；align=True 时左上角对齐到 4 px 的网格（图案落在一个块里）。"""
    xs = NOISE * rng.standard_normal((n, 1, size, size)).astype(np.float32)
    y = rng.integers(0, 4, n)
    pos = np.zeros((n, 2), np.int64)
    lo, hi = (0, size - 3) if region is None else region
    for i in range(n):
        px, py = int(rng.integers(lo, hi + 1)), int(rng.integers(0, size - 2))
        if align:
            px, py = px // 4 * 4, py // 4 * 4
        xs[i, 0, py:py + 3, px:px + 3] += GLYPHS[y[i]]
        pos[i] = py, px
    return torch.from_numpy(xs), torch.from_numpy(y), pos


def data_quadrant(n, rng, size=16):
    """一个图案（种类随机），类别是它在哪个象限（左上、右上、左下、右下）。图案完整地落在一个象限里。"""
    xs = NOISE * rng.standard_normal((n, 1, size, size)).astype(np.float32)
    y = np.zeros(n, np.int64)
    h = size // 2
    for i in range(n):
        qy, qx = rng.integers(0, 2, 2)
        py, px = int(qy * h + rng.integers(0, h - 2)), int(qx * h + rng.integers(0, h - 2))
        xs[i, 0, py:py + 3, px:px + 3] += GLYPHS[rng.integers(0, 4)]
        y[i] = qy * 2 + qx
    return torch.from_numpy(xs), torch.from_numpy(y), None


def data_pair(n, rng, d=None):
    """4x32 的窄条切成 8 个块，其中两个块里各有一个图案，类别是两个图案是否同一种。d 给定时两个图案固定相距 d 个像素。"""
    xs = NOISE * rng.standard_normal((n, 1, 4, 32)).astype(np.float32)
    y = rng.integers(0, 2, n)
    tok = np.zeros((n, 2), np.int64)
    for i in range(n):
        while True:
            a, b = rng.integers(0, 8, 2)
            if a != b and (d is None or abs(a - b) * 4 == d):
                break
        ka = int(rng.integers(0, 4))
        kb = ka if y[i] else (ka + int(rng.integers(1, 4))) % 4
        xs[i, 0, 0:3, a * 4:a * 4 + 3] += GLYPHS[ka]
        xs[i, 0, 0:3, b * 4:b * 4 + 3] += GLYPHS[kb]
        tok[i] = a, b
    return torch.from_numpy(xs), torch.from_numpy(y), tok


def data_defect(n, rng, size=32, blob=2, amp=0.6):
    """噪声背景上有没有一个 blob x blob 像素的亮斑。返回图像、标签、亮斑左上角（没有时为 -1）。"""
    xs = NOISE * rng.standard_normal((n, 1, size, size)).astype(np.float32)
    y = rng.integers(0, 2, n)
    pos = -np.ones((n, 2), np.int64)
    for i in range(n):
        if y[i]:
            py, px = rng.integers(0, size - blob + 1, 2)
            xs[i, 0, py:py + blob, px:px + blob] += amp
            pos[i] = py, px
    return torch.from_numpy(xs), torch.from_numpy(y), pos


# ---------------------------------------------------------------- 模型
class Block(nn.Module):
    def __init__(self, dim, heads, pre=True, ratio=2):
        super().__init__()
        self.heads, self.pre = heads, pre
        self.w_qkv = nn.Parameter(torch.empty(3 * dim, dim))
        nn.init.xavier_uniform_(self.w_qkv)
        self.b_qkv = nn.Parameter(torch.zeros(3 * dim))
        self.out = nn.Linear(dim, dim)
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, ratio * dim), nn.GELU(), nn.Linear(ratio * dim, dim))
        self.attn = None

    def attend(self, x, mask):
        out, self.attn = mod.multi_head_attention(x, self.w_qkv, self.b_qkv, self.out.weight, self.out.bias, self.heads, mask)
        return out

    def forward(self, x, mask=None):
        if self.pre:
            x = x + self.attend(self.ln1(x), mask)
            return x + self.mlp(self.ln2(x))
        x = self.ln1(x + self.attend(x, mask))
        return self.ln2(x + self.mlp(x))


class ViT(nn.Module):
    """很小的 ViT。pos: sincos / learned / none；pool: mean / cls；masks: 每层一个注意力掩码（True 表示允许）或 None。"""

    def __init__(self, hw=(16, 16), patch=4, dim=32, depth=2, heads=4, classes=4, pos="sincos", pool="mean", pre=True, masks=None, pos_scale=1.0):
        super().__init__()
        self.patch, self.pool, self.masks = patch, pool, masks
        self.grid = (hw[0] // patch, hw[1] // patch)
        n = self.grid[0] * self.grid[1]
        self.embed = nn.Linear(patch * patch, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim)) if pool == "cls" else None
        if pos == "learned":
            self.pos = nn.Parameter(0.02 * torch.randn(1, n, dim))
        elif pos == "sincos":
            self.register_buffer("pos", pos_scale * mod.sincos_2d_position_embedding(dim, *self.grid)[None])
        else:
            self.pos = None
        self.blocks = nn.ModuleList([Block(dim, heads, pre) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim) if pre else nn.Identity()
        self.head = nn.Linear(dim, classes)

    def features(self, x, perm=None):
        t = mod.patchify(x, self.patch)
        if perm is not None:
            t = t[:, perm]
        t = self.embed(t)
        if self.pos is not None:
            t = t + self.pos
        if self.cls is not None:
            t = torch.cat([self.cls.expand(len(t), -1, -1), t], dim=1)
        for i, blk in enumerate(self.blocks):
            t = blk(t, None if self.masks is None else self.masks[i])
        t = self.ln(t)
        return t[:, 0] if self.pool == "cls" else t.mean(1)

    def forward(self, x, perm=None):
        return self.head(self.features(x, perm))


class Conv(nn.Module):
    """补零的二维卷积，用 unfold 加一个线性层实现，输出和 nn.Conv2d(cin, cout, k, padding=k // 2) 相同（numbers_mechanics 里有核对）。

    本章的张量很小，这样写在这台机器的 CPU 上比 nn.Conv2d 快，整个脚本才能在 5 分钟内跑完。学员的 Notebook 里用的是 nn.Conv2d。
    """

    def __init__(self, cin, cout, k):
        super().__init__()
        self.k = k
        self.lin = nn.Linear(cin * k * k, cout)

    def forward(self, x):
        b, _, h, w = x.shape
        u = x.flatten(2) if self.k == 1 else F.unfold(x, self.k, padding=self.k // 2)
        return self.lin(u.transpose(1, 2)).transpose(1, 2).reshape(b, -1, h, w)


class SmallCNN(nn.Module):
    """和小 ViT 参数量相近的 CNN：3x3 卷积 16 通道、2x2 平均池化、两层 3x3 卷积 36 通道、全局平均、线性层。"""

    def __init__(self, classes=4, c1=16, c2=36):
        super().__init__()
        self.body = nn.Sequential(Conv(1, c1, 3), nn.ReLU(), nn.AvgPool2d(2), Conv(c1, c2, 3), nn.ReLU(), Conv(c2, c2, 3), nn.ReLU())
        self.head = nn.Linear(c2, classes)

    def forward(self, x):
        return self.head(self.body(x).mean((2, 3)))


class StripCNN(nn.Module):
    """感受野只有 7x7 的 CNN：一层 7x7 卷积加两层 1x1 卷积，全局平均，线性层。"""

    def __init__(self, ch=48, classes=2):
        super().__init__()
        self.body = nn.Sequential(Conv(1, ch, 7), nn.ReLU(), Conv(ch, ch, 1), nn.ReLU(), Conv(ch, ch, 1), nn.ReLU())
        self.head = nn.Linear(ch, classes)

    def forward(self, x):
        return self.head(self.body(x).mean((2, 3)))


def nparams(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def plain_log_ticks(axis):
    """对数轴的刻度写成普通小数。默认的 10 的负几次方会用到字体里没有的减号。"""
    from matplotlib.ticker import FuncFormatter
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def train(model, seed, fresh=None, data=None, steps=STEPS, lr=LR, warmup=0, probe=None):
    """fresh(rng, n) 每步新生成一批；data=(x, y) 是固定的训练集。probe(step, model) 在指定的步数被调用。返回每步的损失。"""
    rng = np.random.default_rng([seed, 77])
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses, order, ptr = [], None, 0
    model.train()
    for s in range(steps):
        if probe is not None:
            probe(s, model)
        if data is None:
            x, y = fresh(rng, BATCH)[:2]
        else:
            if order is None or ptr + BATCH > len(data[0]):
                order, ptr = torch.randperm(len(data[0]), generator=g), 0
            idx = order[ptr:ptr + BATCH]
            ptr += BATCH
            x, y = data[0][idx], data[1][idx]
        if warmup:
            for group in opt.param_groups:
                group["lr"] = lr * min(1.0, (s + 1) / warmup)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    if probe is not None:
        probe(steps, model)
    model.eval()
    return losses


@torch.no_grad()
def predict(model, x, **kw):
    model.eval()
    return model(x, **kw)


def acc_of(model, x, y, **kw):
    return float((predict(model, x, **kw).argmax(1) == y).float().mean())


def build(seed, make):
    torch.manual_seed(1000 + seed)
    return make()


def test_set(seed, fn, n=1000, **kw):
    return fn(n, np.random.default_rng([seed, 5]), **kw)


# ---------------------------------------------------------------- 图 1：相距很远的两处要合在一起判断
def fig_longrange():
    dists = [4, 8, 16, 28]
    eye = torch.eye(8, dtype=torch.bool)
    makers = {
        "ViT，1 层": lambda: ViT(hw=(4, 32), depth=1, classes=2),
        "ViT，1 层，每个标记只能看自己": lambda: ViT(hw=(4, 32), depth=1, classes=2, masks=[eye]),
        "CNN，感受野 7x7": lambda: StripCNN(),
    }
    res = {name: [] for name in makers}
    loss = {name: [] for name in makers}
    for name, make in makers.items():
        for d in dists:
            accs, ls = [], []
            for seed in SEEDS[:2]:
                xt, yt, tok = test_set(seed, data_pair, d=d)
                m = build(seed, make)
                hist = train(m, seed, fresh=lambda rng, n, d=d: data_pair(n, rng, d))
                accs.append(acc_of(m, xt, yt))
                ls.append(float(np.mean(hist[-30:])))
                if name == "ViT，1 层" and d == 16 and seed == 0:
                    KEEP["pair-vit"] = (m, xt, yt, tok)
            res[name].append(accs)
            loss[name].append(ls)
    counts = {name: nparams(make()) for name, make in makers.items()}

    # 注意力落在哪：图案 A 所在的标记当查询，看它给图案 B 所在标记的权重
    m, xt, yt, tok = KEEP["pair-vit"]
    predict(m, xt)
    attn = m.blocks[0].attn                                   # (B, 4, 8, 8)
    idx = torch.arange(len(xt))
    a, b = torch.from_numpy(tok[:, 0]), torch.from_numpy(tok[:, 1])
    w_ab = torch.stack([attn[idx, :, a, b], attn[idx, :, b, a]]).mean(0)          # (B, 4)
    w_self = torch.stack([attn[idx, :, a, a], attn[idx, :, b, b]]).mean(0)
    other = torch.ones(len(xt), 8, dtype=torch.bool)
    other[idx, a] = False
    other[idx, b] = False
    bg_rows = attn.mean(1)[other].reshape(len(xt), 6, 8)                           # 背景标记当查询的那几行
    bg_to_glyph = (bg_rows[idx, :, a] + bg_rows[idx, :, b]).mean()
    NUM["相距很远的两处"] = {
        "说明": "4x32 的窄条切成 8 个 4x4 的块，其中两个块里各放一个图案（左上角对齐块的左上角），类别是两个图案是否同一种，随机猜是 0.5。"
              "每个间距单独训练、单独测试，2 个种子。" + DATA_NOTE + "。三个模型最后都是全局平均加一个线性层。"
              "ViT 是块大小 4、嵌入 32 维、1 层、4 个头；“只能看自己”是给同一个 ViT 加一个只有对角线为 True 的注意力掩码；"
              "CNN 是一层 7x7 卷积加两层 1x1 卷积（48 通道），一个输出位置只依赖输入的 7x7 区域。图案宽 3 px：相距 4 px 时两个图案一共占 7 列，一个窗口刚好全看到；"
              "相距 8 px 时一个窗口最多同时看到左边图案的最后一列和右边图案的第一列（横线、斜线、反斜线的这一列各不相同，竖线的这一列是空的）；"
              "相距 16 px 以上时任何一个窗口都只能看到其中一个",
        "两个图案相距（px）": dists,
        "参数量": counts,
        **{f"准确率：{name}": [r3(np.mean(v)) for v in res[name]] for name in res},
        **{f"各种子的准确率：{name}": [[r3(x) for x in v] for v in res[name]] for name in res},
        **{f"最后 30 步的训练损失：{name}": [r3(np.mean(v)) for v in loss[name]] for name in loss},
        "随机猜时的损失 ln 2": r3(LN2),
        "注意力落在哪（相距 16 px 的那个 ViT，种子 0，1000 张测试图）": {
            "说明": "一个图案所在的标记当查询时，给另一个图案所在标记的注意力权重。8 个标记平均分是 0.125",
            "4 个头各自的平均权重": [r3(x) for x in w_ab.mean(0)],
            "4 个头取平均": r3(w_ab.mean()),
            "每张图里权重最大的那个头，平均": r3(w_ab.max(1).values.mean()),
            "给自己的权重（4 个头取平均）": r3(w_self.mean()),
            "背景标记当查询时，给两个图案标记的权重之和（4 个头取平均）": r3(bg_to_glyph),
        },
    }

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.0), gridspec_kw={"width_ratios": [1.5, 1.2, 0.95]}, layout="constrained")
    rng = np.random.default_rng(3)
    rows = []
    for d, same in ((4, 1), (16, 1), (16, 0), (28, 0)):
        while True:
            x, y, _ = data_pair(1, rng, d)
            if int(y[0]) == same:
                break
        rows.append((x[0, 0].numpy(), f"相距 {d} px，{'同一种' if same else '不同种'}"))
    canvas = np.full((4 * 4 + 3 * 3, 32), np.nan)
    for i, (img, _) in enumerate(rows):
        canvas[i * 7:i * 7 + 4] = img
    ax[0].imshow(canvas, cmap="gray", vmin=-0.3, vmax=1.1, aspect="equal")
    for i, (_, label) in enumerate(rows):
        ax[0].text(0, i * 7 - 0.9, label, fontsize=9.5, color=S.INK, va="bottom")
        for k in range(1, 8):
            ax[0].plot([k * 4 - 0.5] * 2, [i * 7 - 0.5, i * 7 + 3.5], color=S.ORANGE, lw=0.6, alpha=0.7)
    ax[0].set_ylim(4 * 7 - 3.5, -2.6)
    ax[0].axis("off")
    styles = {"ViT，1 层": (S.BLUE, "o-"), "ViT，1 层，每个标记只能看自己": (S.PURPLE, "^--"), "CNN，感受野 7x7": (S.ORANGE, "s-")}
    xs = np.arange(len(dists))
    for name, (color, style) in styles.items():
        for j in range(2):
            ax[1].plot(xs, [v[j] for v in res[name]], style[0], color=color, ms=3.5, alpha=0.35)
        ax[1].plot(xs, [np.mean(v) for v in res[name]], style, color=color, lw=1.8, ms=5.5, label=name)
    ax[1].axhline(0.5, color=S.RED, lw=1, ls=":")
    ax[1].text(len(dists) - 1, 0.515, "随机猜", color=S.RED, fontsize=8.5, ha="right")
    ax[1].set_xticks(xs, [str(d) for d in dists])
    ax[1].set_xlabel("两个图案相距（px）")
    ax[1].set_ylabel("测试准确率")
    ax[1].set_ylim(0.4, 1.04)
    ax[1].set_title("超出感受野 CNN 退回随机猜，ViT 不受距离影响")
    ax[0].set_title("任务：两个图案是不是同一种（橙线是块的边界）")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax[1])
    k = next(i for i in range(len(xt)) if int(yt[i]) == 1)
    head = int(w_ab[k].argmax())
    im = ax[2].imshow(attn[k, head].numpy(), cmap="Blues", vmin=0, vmax=1)
    for t in tok[k]:
        ax[2].add_patch(plt.Rectangle((-0.5, t - 0.5), 8, 1, fill=False, ec=S.ORANGE, lw=1.4))
    ax[2].set_xticks(range(8))
    ax[2].set_yticks(range(8))
    ax[2].set_xlabel("键（被看的标记）")
    ax[2].set_ylabel("查询（去看的标记）")
    ax[2].set_title(f"一张图的注意力权重（第 {head + 1} 个头）")
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.03).ax.tick_params(labelsize=8)
    NUM["相距很远的两处"]["图 1 右栏画的那张图"] = {"两个图案所在的标记": [int(t) for t in tok[k]], "画的是第几个头": head + 1,
                                           "两个图案标记互相给的权重": [r3(attn[k, head, tok[k][0], tok[k][1]]), r3(attn[k, head, tok[k][1], tok[k][0]])]}
    S.save(fig, CH, "fig01_远距离关系")


# ---------------------------------------------------------------- 表：切块、缩放、置换（不训练）
def numbers_mechanics():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 3, 32, 48, generator=g, dtype=torch.float64)
    conv = nn.Conv2d(3, 20, 8, stride=8).double()
    with torch.no_grad():
        a = mod.patchify(x, 8) @ conv.weight.reshape(20, -1).T + conv.bias
        b = conv(x).flatten(2).transpose(1, 2)
    back = mod.unpatchify(mod.patchify(x, 8), 8, 3, 32, 48)
    ref_conv, my_conv = nn.Conv2d(3, 5, 3, padding=1), Conv(3, 5, 3)
    with torch.no_grad():
        my_conv.lin.weight.copy_(ref_conv.weight.reshape(5, -1))
        my_conv.lin.bias.copy_(ref_conv.bias)
        z = torch.randn(2, 3, 6, 7, generator=g)
        conv_diff = float((ref_conv(z) - my_conv(z)).abs().max())
    assert conv_diff < 1e-5
    NUM["切块等于卷积"] = {
        "说明": "4 张 3x32x48 的随机图（float64），块大小 8：先切块再乘同一组权重，和步长等于核大小的 nn.Conv2d(3, 20, 8, stride=8) 的输出比较",
        "标记个数": int(a.shape[1]), "每个标记的长度": int(mod.patchify(x, 8).shape[2]),
        "两种算法输出的最大差": sig2((a - b).abs().max()),
        "切块再还原和原图的最大差": float((back - x).abs().max()),
        "本脚本里用 unfold 实现的卷积和 nn.Conv2d 输出的最大差（float32）": sig2(conv_diff),
    }

    dims, n_keys, trials = [4, 16, 64, 256, 1024], 16, 2000
    table = {"不缩放": [], "除以根号 d": [], "除以 d": []}
    for d in dims:
        q = torch.randn(trials, 1, d, generator=g, dtype=torch.float64)
        k = torch.randn(trials, n_keys, d, generator=g, dtype=torch.float64)
        _, w_ref = mod.scaled_dot_product_attention(q, k, k)
        raw = q @ k.transpose(1, 2)
        ws = {"不缩放": torch.softmax(raw, -1), "除以根号 d": w_ref, "除以 d": torch.softmax(raw / d, -1)}
        assert torch.allclose(w_ref, torch.softmax(raw / math.sqrt(d), -1))
        for name, w in ws.items():
            table[name].append([float(w.max(-1).values.mean()), float((1 - (w ** 2).sum(-1)).mean()), float((raw / {"不缩放": 1, "除以根号 d": math.sqrt(d), "除以 d": d}[name]).std())])
    NUM["缩放点积"] = {
        "说明": f"查询和 {n_keys} 个键都是 d 维的标准正态随机向量，{trials} 次取平均。“最大权重”是 softmax 之后最大的那个权重；"
              "“1 减权重平方和”是 softmax 雅可比矩阵的迹，权重集中到一个键上时它趋于 0，梯度也就传不回查询和键；均匀分给 16 个键时最大权重是 0.0625，1 减平方和是 0.9375",
        "维数 d": dims,
        **{f"{name}：最大权重": [r3(v[0]) for v in rows] for name, rows in table.items()},
        **{f"{name}：1 减权重平方和": [r3(v[1]) for v in rows] for name, rows in table.items()},
        **{f"{name}：分数的标准差": [r3(v[2]) for v in rows] for name, rows in table.items()},
        "均匀时的最大权重": 1 / n_keys, "均匀时的 1 减平方和": 1 - 1 / n_keys,
    }

    torch.manual_seed(0)
    d = 32
    xx = torch.randn(2, 16, d, dtype=torch.float64)
    wq, bq = torch.randn(3 * d, d, dtype=torch.float64) / math.sqrt(d), torch.zeros(3 * d, dtype=torch.float64)
    wo, bo = torch.randn(d, d, dtype=torch.float64) / math.sqrt(d), torch.zeros(d, dtype=torch.float64)
    perm = torch.randperm(16)
    out, attn = mod.multi_head_attention(xx, wq, bq, wo, bo, 4)
    out_p, _ = mod.multi_head_attention(xx[:, perm], wq, bq, wo, bo, 4)
    NUM["置换与头数"] = {
        "说明": "2 组 16 个 32 维的随机标记过一层多头自注意力（随机权重，不加位置编码）",
        "把标记打乱后的输出，和原输出按同样方式打乱，最大差": sig2((out_p - out[:, perm]).abs().max()),
        "对所有标记取平均后，打乱前后的最大差": sig2((out_p.mean(1) - out.mean(1)).abs().max()),
        "多头注意力的参数量（D=32，与头数无关）": int(wq.numel() + bq.numel() + wo.numel() + bo.numel()),
    }


# ---------------------------------------------------------------- 图 2：位置编码
def fig_position():
    tasks = {"图案是哪一种": data_type, "图案在哪个象限": data_quadrant}
    kinds = {"不加位置编码": dict(pos="none"), "正余弦": dict(pos="sincos"), "可学习": dict(pos="learned"), "正余弦，类别标记": dict(pos="sincos", pool="cls")}
    perm = torch.from_numpy(np.random.default_rng(7).permutation(16))
    res: dict = {}
    for tname, fn in tasks.items():
        for kname, kw in list(kinds.items()) + ([("正余弦，乘 0.1", dict(pos="sincos", pos_scale=0.1))] if tname == "图案是哪一种" else []):
            rows = []
            for seed in SEEDS:
                xt, yt, pos = test_set(seed, fn)
                m = build(seed, lambda: ViT(**kw))
                hist = train(m, seed, fresh=lambda rng, n: fn(n, rng))
                a, b = predict(m, xt), predict(m, xt, perm=perm)
                rows.append([float((a.argmax(1) == yt).float().mean()), float((b.argmax(1) == yt).float().mean()), float((a - b).abs().max()),
                             float((a.argmax(1) != b.argmax(1)).float().mean()), float(np.mean(hist[-30:]))])
                KEEP[(tname, kname, seed)] = m
            res[(tname, kname)] = np.array(rows)

    learned = KEEP[("图案在哪个象限", "可学习", 0)].pos.detach()[0]
    ln = F.normalize(learned, dim=1)
    sim = (ln @ ln.T).numpy()
    quad = np.array([(i // 4 // 2) * 2 + (i % 4) // 2 for i in range(16)])
    same = quad[:, None] == quad[None, :]
    off = ~np.eye(16, dtype=bool)
    sims_all = []
    for seed in SEEDS:
        p = F.normalize(KEEP[("图案在哪个象限", "可学习", seed)].pos.detach()[0], dim=1)
        s = (p @ p.T).numpy()
        sims_all.append([s[same & off].mean(), s[~same].mean()])
    pe = mod.sincos_2d_position_embedding(32, 8, 8)
    pn = F.normalize(pe, dim=1)
    ref = 3 * 8 + 3
    simmap = (pn @ pn[ref]).reshape(8, 8).numpy()
    pe4 = mod.sincos_2d_position_embedding(32, 4, 4)
    quad_of = lambda i: (i // 4 // 2) * 2 + (i % 4) // 2  # noqa: E731
    stay = float(np.mean([quad_of(i) == quad_of(int(perm[i])) for i in range(16)]))
    emb_norm = []
    for seed in SEEDS:
        m0 = build(seed, lambda: ViT())
        with torch.no_grad():
            emb_norm.append(float(m0.embed(mod.patchify(test_set(seed, data_type)[0], 4)).norm(dim=-1).mean()))
    NUM["位置编码"] = {
        "说明": "16x16 的图里放一个图案（位置随机，不对齐块的网格），切成 16 个 4x4 的块。两个任务：分图案的种类（和位置无关）；分图案在哪个象限（只和位置有关）。"
              + VIT_NOTE + "；每种设置 3 个种子取平均。" + DATA_NOTE + "。“打乱”指测试时把 16 个块按同一个固定的随机顺序重排（位置编码不跟着动），训练时没有打乱。随机猜是 0.25",
        "参数量": {k: nparams(ViT(**kw)) for k, kw in kinds.items()},
        **{f"{t}｜{k}": {"准确率": r3(v[:, 0].mean()), "各种子": [r3(x) for x in v[:, 0]], "打乱块之后的准确率": r3(v[:, 1].mean()),
                        "打乱前后输出的最大差": sig2(v[:, 2].max()), "打乱后预测变了的比例": r3(v[:, 3].mean()), "最后 30 步的训练损失": r3(v[:, 4].mean())}
           for (t, k), v in res.items()},
        "学到的位置编码（象限任务，可学习）": {
            "说明": "训练后 16 个位置编码两两之间的余弦相似度，按两个位置是否在同一个象限分组取平均，3 个种子",
            "同一象限": r3(np.mean([s[0] for s in sims_all])), "不同象限": r3(np.mean([s[1] for s in sims_all])),
            "各种子（同一象限，不同象限）": [[r3(x) for x in s] for s in sims_all],
        },
        "正余弦编码的相似度（32 维，8x8 网格，和第 3 行第 3 列那个位置的余弦相似度）": {
            "自己": r3(simmap[3, 3]), "左右相邻": r3(simmap[3, 4]), "上下相邻": r3(simmap[4, 3]), "隔 2 格（同一行）": r3(simmap[3, 5]),
            "隔 4 格（同一行）": r3(simmap[3, 7]), "斜对角相邻": r3(simmap[4, 4]), "整张图里的最小值": r3(simmap.min()),
        },
        "正余弦编码每一行的平方和（32 维）": r3((pe4 ** 2).sum(1).mean()),
        "正余弦编码每一行的范数（32 维）": r3(pe4.norm(dim=1).mean()),
        "初始化时块嵌入的平均范数（3 组初始化）": r3(np.mean(emb_norm)),
        "位置编码的范数是块嵌入的几倍（初始化时）": r3(float(pe4.norm(dim=1).mean()) / np.mean(emb_norm)),
        "可学习位置编码的初始标准差": 0.02,
        "固定的打乱顺序里，仍留在原来象限的块所占的比例": r3(stay),
    }

    fig, ax = plt.subplots(1, 4, figsize=(17, 3.9), gridspec_kw={"width_ratios": [1, 1.5, 1.3, 1]}, layout="constrained")
    im = ax[0].imshow(simmap, cmap="viridis", vmin=min(0.0, float(simmap.min())), vmax=1)
    ax[0].plot(3, 3, "x", color="white", ms=8, mew=2)
    ax[0].set_title("正余弦编码：和 × 处的余弦相似度")
    ax[0].set_xlabel("列")
    ax[0].set_ylabel("行")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03).ax.tick_params(labelsize=8)
    names = ["不加位置编码", "正余弦", "可学习"]
    colors = [S.MUTED, S.BLUE, S.GREEN]
    xs = np.arange(2)
    for i, (k, color) in enumerate(zip(names, colors)):
        vals = [res[(t, k)][:, 0].mean() for t in tasks]
        bars = ax[1].bar(xs + (i - 1) * 0.27, vals, 0.25, color=color, alpha=0.88, label=k)
        for j, t in enumerate(tasks):
            ax[1].plot([xs[j] + (i - 1) * 0.27] * 3, res[(t, k)][:, 0], ".", color=S.INK, ms=4)
        for bar, v in zip(bars, vals):
            ax[1].text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=8.5)
    ax[1].axhline(0.25, color=S.RED, lw=1, ls=":")
    ax[1].set_xticks(xs, list(tasks))
    ax[1].set_ylim(0, 1.12)
    ax[1].set_ylabel("测试准确率")
    ax[1].set_title("不加位置编码：分种类没问题，分象限等于随机猜")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    S.clean(ax[1])
    ax[1].grid(axis="x", visible=False)
    labels, before, after = [], [], []
    for t in tasks:
        for k in ("不加位置编码", "正余弦"):
            labels.append(f"{'种类' if '种' in t else '象限'}\n{k}")
            before.append(res[(t, k)][:, 0].mean())
            after.append(res[(t, k)][:, 1].mean())
    xs = np.arange(len(labels))
    ax[2].bar(xs - 0.2, before, 0.38, color=S.BLUE, alpha=0.88, label="原样")
    ax[2].bar(xs + 0.2, after, 0.38, color=S.ORANGE, alpha=0.88, label="把块打乱")
    for x0, v0, v1 in zip(xs, before, after):
        ax[2].text(x0 - 0.2, v0 + 0.03, f"{v0:.2f}", ha="center", fontsize=8.5)
        ax[2].text(x0 + 0.2, v1 + 0.03, f"{v1:.2f}", ha="center", fontsize=8.5)
    ax[2].axhline(0.25, color=S.RED, lw=1, ls=":")
    ax[2].set_xticks(xs, labels, fontsize=8.5)
    ax[2].set_ylim(0, 1.12)
    ax[2].set_title("打乱块：只有靠位置的任务掉下来")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    ax[2].grid(axis="x", visible=False)
    order = np.argsort(quad, kind="stable")
    im = ax[3].imshow(sim[order][:, order], cmap="RdBu_r", vmin=-1, vmax=1)
    for k in range(1, 4):
        ax[3].axhline(k * 4 - 0.5, color="white", lw=1.2)
        ax[3].axvline(k * 4 - 0.5, color="white", lw=1.2)
    ax[3].set_xticks([1.5, 5.5, 9.5, 13.5], ["左上", "右上", "左下", "右下"], fontsize=8.5)
    ax[3].set_yticks([1.5, 5.5, 9.5, 13.5], ["左上", "右上", "左下", "右下"], fontsize=8.5)
    ax[3].set_title("学到的位置编码两两的相似度")
    fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.03).ax.tick_params(labelsize=8)
    S.save(fig, CH, "fig02_位置编码")


# ---------------------------------------------------------------- 图 3：LayerNorm 放前面还是后面
def spread_of(model, x):
    """不同输入的汇聚特征彼此差多少：到平均特征的距离除以特征的范数。接近 0 表示输出几乎不随输入变。"""
    was = model.training
    model.eval()
    with torch.no_grad():
        f = model.features(x)
    model.train(was)
    return float((f - f.mean(0)).norm(dim=1).mean() / f.norm(dim=1).mean())


def fig_layernorm():
    probe_steps = [0, 5, 20, 50, 100, 200, 300, 500, 800]
    grid, curves, spreads = {}, {}, {}
    configs = [(2, 3e-3, True, 0, 300, 3), (2, 3e-3, False, 0, 300, 3), (2, 1e-2, True, 0, 300, 3), (2, 1e-2, False, 0, 300, 3), (2, 1e-2, False, 100, 300, 3),
               (6, 3e-3, True, 0, 300, 2), (6, 3e-3, False, 0, 300, 2), (6, 1e-3, False, 200, 800, 2)]
    for depth, lr, pre, warm, steps, n_seed in configs:
        rows, cs, sp = [], [], []
        for seed in SEEDS[:n_seed]:
            xt, yt, _ = test_set(seed, data_type)
            m = build(seed, lambda: ViT(depth=depth, pre=pre))
            log = {}

            def probe(s, model, log=log, xt=xt):
                if s in probe_steps:
                    log[s] = spread_of(model, xt[:256])

            hist = train(m, seed, fresh=lambda rng, n: data_type(n, rng), lr=lr, warmup=warm, steps=steps, probe=probe)
            rows.append([float(np.mean(hist[-30:])), acc_of(m, xt, yt)])
            cs.append(hist)
            sp.append(log)
        key = (depth, lr, pre, warm, steps)
        grid[key], curves[key], spreads[key] = np.array(rows), cs, sp

    grads = {}
    for depth in (2, 6):
        for pre in (True, False):
            ratios = []
            for seed in range(5):
                m = build(seed, lambda: ViT(depth=depth, pre=pre))
                x, y, _ = data_type(64, np.random.default_rng([seed, 9]))
                m.train()
                F.cross_entropy(m(x), y).backward()
                g = [float(b.w_qkv.grad.norm()) for b in m.blocks]
                ratios.append(g[0] / g[-1])
            grads[(depth, pre)] = float(np.exp(np.mean(np.log(ratios))))

    def label(key):
        depth, lr, pre, warm, steps = key
        return f"{depth} 层，{'前置' if pre else '后置'}，学习率 {lr:g}" + (f"，预热 {warm} 步" if warm else "") + f"，{steps} 步"

    NUM["LayerNorm 的位置"] = {
        "说明": "分图案种类的任务（16x16 的图，图案位置随机）。小 ViT 只改两样：层数，以及 LayerNorm 放在注意力和 MLP 的前面（前置）还是残差相加之后（后置）。"
              + DATA_NOTE + "。学习率、预热步数和训练步数见每一项的名字；2 层的设置 3 个种子，6 层的 2 个种子。随机猜的损失是 ln 4。"
              "“特征区分度”是 256 张测试图的汇聚特征到它们平均值的距离除以特征的范数，接近 0 表示不管输入什么输出都一样",
        "随机猜的损失 ln 4": r3(LN4),
        **{label(k): {"最后 30 步的训练损失": [r3(x) for x in v[:, 0]], "测试准确率": [r3(x) for x in v[:, 1]],
                      "特征区分度随训练步数（种子 0）": {str(s): r4(x) for s, x in spreads[k][0].items()}} for k, v in grid.items()},
        "初始化时第一层和最后一层注意力权重的梯度范数之比（5 组初始化的几何平均）": {f"{d} 层，{'前置' if p else '后置'}": r3(v) for (d, p), v in grads.items()},
    }

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.0), gridspec_kw={"width_ratios": [1.15, 1.15, 1.3]}, layout="constrained")
    kern = np.ones(10) / 10
    k_pre, k_post, k_fix = (6, 3e-3, True, 0, 300), (6, 3e-3, False, 0, 300), (6, 1e-3, False, 200, 800)
    for key, color, name in ((k_pre, S.BLUE, "前置"), (k_post, S.RED, "后置")):
        for j, c in enumerate(curves[key]):
            ax[0].plot(np.arange(9, len(c)), np.convolve(c, kern, mode="valid"), color=color, lw=1.5, alpha=0.9 if j == 0 else 0.5, label=name if j == 0 else None)
    ax[0].axhline(LN4, color=S.MUTED, lw=1, ls=":")
    ax[0].text(295, LN4 + 0.03, "随机猜 ln 4", color=S.MUTED, fontsize=8.5, ha="right")
    ax[0].set_xlabel("训练步数")
    ax[0].set_ylabel("训练损失（10 步滑动平均）")
    ax[0].set_ylim(0, 1.75)
    ax[0].set_title("6 层、学习率 0.003：后置停在随机猜")
    ax[0].legend(fontsize=9, loc="lower left")
    S.clean(ax[0])
    for key, color, style, name in ((k_pre, S.BLUE, "o-", "6 层，前置，学习率 0.003"), (k_post, S.RED, "s-", "6 层，后置，学习率 0.003"),
                                    (k_fix, S.GREEN, "^--", "6 层，后置，学习率 0.001，预热 200 步")):
        log = spreads[key][0]
        ax[1].plot([s + 1 for s in log], list(log.values()), style, color=color, lw=1.6, ms=4.5, label=name)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xticks([1, 6, 21, 51, 101, 301, 801], ["0", "5", "20", "50", "100", "300", "800"])
    ax[1].minorticks_off()
    plain_log_ticks(ax[1].yaxis)
    ax[1].set_xlabel("训练步数")
    ax[1].set_ylabel("特征区分度")
    ax[1].set_title("前置掉下去又回升，后置（红）一路掉下去")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax[1])
    order = [(2, 3e-3, True, 0, 300), (2, 3e-3, False, 0, 300), (2, 1e-2, True, 0, 300), (2, 1e-2, False, 0, 300), (2, 1e-2, False, 100, 300), k_pre, k_post, k_fix]
    names = ["2 层\n0.003", "2 层\n0.003", "2 层\n0.01", "2 层\n0.01", "2 层\n0.01\n预热", "6 层\n0.003", "6 层\n0.003", "6 层\n0.001\n预热\n800 步"]
    for i, key in enumerate(order):
        color = S.BLUE if key[2] else (S.GREEN if key[3] else S.RED)
        ax[2].bar(i, grid[key][:, 0].mean(), 0.62, color=color, alpha=0.85)
        ax[2].plot([i] * len(grid[key]), grid[key][:, 0], ".", color=S.INK, ms=5)
    ax[2].axhline(LN4, color=S.MUTED, lw=1, ls=":")
    ax[2].set_xticks(range(len(order)), names, fontsize=8)
    ax[2].set_ylabel("最后 30 步的训练损失")
    ax[2].set_ylim(0, 1.6)
    ax[2].set_title("后置在学习率大或层数多时训练不动")
    from matplotlib.patches import Patch
    ax[2].legend(handles=[Patch(color=S.BLUE, alpha=0.85, label="前置"), Patch(color=S.RED, alpha=0.85, label="后置"), Patch(color=S.GREEN, alpha=0.85, label="后置，加预热")],
                 fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    S.clean(ax[2])
    ax[2].grid(axis="x", visible=False)
    S.save(fig, CH, "fig03_LayerNorm位置")


# ---------------------------------------------------------------- 图 4：小 ViT 对小 CNN
def fig_vit_vs_cnn():
    sizes = [50, 200, 800]
    makers = {"ViT": lambda: ViT(), "CNN": lambda: SmallCNN()}
    curve = {name: [] for name in makers}
    train_loss = {name: [] for name in makers}
    for name, make in makers.items():
        for n in sizes:
            accs, ls = [], []
            for seed in SEEDS:
                xt, yt, _ = test_set(seed, data_type)
                xs, ys, _ = data_type(n, np.random.default_rng([seed, 11]))
                m = build(seed, make)
                hist = train(m, seed, data=(xs, ys))
                accs.append(acc_of(m, xt, yt))
                ls.append(float(np.mean(hist[-30:])))
            curve[name].append(accs)
            train_loss[name].append(ls)
    # 每步新数据：ViT 直接用图 2 训练好的（正余弦、全局平均），CNN 在这里训练
    inf = {"ViT": [], "CNN": []}
    for seed in SEEDS:
        xt, yt, _ = test_set(seed, data_type)
        inf["ViT"].append(acc_of(KEEP[("图案是哪一种", "正余弦", seed)], xt, yt))
        m = build(seed, makers["CNN"])
        train(m, seed, fresh=lambda rng, k: data_type(k, rng))
        KEEP[("cnn", seed)] = m
        inf["CNN"].append(acc_of(m, xt, yt))
    longer = []
    for seed in SEEDS[:2]:
        xt, yt, _ = test_set(seed, data_type)
        m = build(seed, makers["ViT"])
        train(m, seed, fresh=lambda rng, k: data_type(k, rng), steps=3 * STEPS)
        KEEP[("vit-long", seed)] = m
        longer.append(acc_of(m, xt, yt))
    NUM["训练样本数"] = {
        "说明": "分图案种类的任务（16x16 的图，图案位置随机）。" + VIT_NOTE + "。小 CNN：3x3 卷积 16 通道、2x2 平均池化、两层 3x3 卷积 36 通道、全局平均、线性层。"
              + DATA_NOTE + "。固定训练集时反复用同样的 n 张图训练同样的步数；“每步新数据”相当于 16000 张不重复的图。3 个种子",
        "参数量": {name: nparams(make()) for name, make in makers.items()},
        "训练集张数": sizes + ["每步新数据"],
        **{f"{name}：测试准确率": [r3(np.mean(v)) for v in curve[name]] + [r3(np.mean(inf[name]))] for name in makers},
        **{f"{name}：各种子": [[r3(x) for x in v] for v in curve[name]] + [[r3(x) for x in inf[name]]] for name in makers},
        **{f"{name}：最后 30 步的训练损失（固定训练集）": [r3(np.mean(v)) for v in train_loss[name]] for name in makers},
        "ViT 每步新数据训练 3 倍步数（1500 步，2 个种子）的测试准确率": [r3(x) for x in longer],
        "CNN 减 ViT 的测试准确率": [r3(np.mean(c) - np.mean(v)) for c, v in zip(curve["CNN"] + [inf["CNN"]], curve["ViT"] + [inf["ViT"]])],
    }

    # 没见过的位置：训练时图案只出现在左边，测试时在右边
    regions = {"左（见过）": (0, 5), "右（没见过）": (8, 13)}
    gen_makers = {"ViT，不加位置编码": lambda: ViT(pos="none"), "ViT，正余弦": lambda: ViT(), "CNN": lambda: SmallCNN()}
    gen = {}
    for name, make in gen_makers.items():
        rows = []
        for seed in SEEDS:
            m = build(seed, make)
            train(m, seed, fresh=lambda rng, k: data_type(k, rng, region=regions["左（见过）"]))
            rows.append([acc_of(m, *test_set(seed, data_type, region=r)[:2]) for r in regions.values()])
        gen[name] = np.array(rows)
    NUM["没见过的位置"] = {
        "说明": "同一个任务，训练时图案的左上角只出现在第 0 到 5 列（图案完全在图的左半边），测试时分两组：仍在第 0 到 5 列，或者在第 8 到 13 列（完全在右半边）。3 个种子取平均",
        **{name: {"图案在左边": r3(v[:, 0].mean()), "图案在右边": r3(v[:, 1].mean()), "各种子（右边）": [r3(x) for x in v[:, 1]]} for name, v in gen.items()},
    }

    # 平移：把测试图循环平移 t 个像素，预测变不变
    shifts = [1, 2, 3, 4, 8]
    models = {"ViT，正余弦": [KEEP[("图案是哪一种", "正余弦", s)] for s in SEEDS], "ViT，不加位置编码": [KEEP[("图案是哪一种", "不加位置编码", s)] for s in SEEDS],
              "ViT，正余弦，训练 1500 步": [KEEP[("vit-long", s)] for s in SEEDS[:2]], "CNN": [KEEP[("cnn", s)] for s in SEEDS]}
    flip = {name: [] for name in models}
    offs = {name: [] for name in models}
    base_acc = {name: [] for name in models}
    for name, ms in models.items():
        for seed, m in zip(SEEDS, ms):
            xt, yt, pos = data_type(3000, np.random.default_rng([seed, 13]), region=(0, 5))   # 图案在左边，向右平移 8 px 以内不会绕到另一边
            base = predict(m, xt).argmax(1)
            flip[name].append([float((predict(m, torch.roll(xt, t, dims=3)).argmax(1) != base).float().mean()) for t in shifts])
            base_acc[name].append(float((base == yt).float().mean()))
            xa, ya, pa = data_type(6000, np.random.default_rng([seed, 14]))
            ok = (predict(m, xa).argmax(1) == ya).numpy()
            offs[name].append([[ok[(pa[:, 0] % 4 == r) & (pa[:, 1] % 4 == c)].mean() for c in range(4)] for r in range(4)])
    off_mean = {name: np.mean(v, axis=0) for name, v in offs.items()}
    inside = np.zeros((4, 4), bool)
    inside[:2, :2] = True
    NUM["平移"] = {
        "说明": "用上面“每步新数据”训练出的模型（500 步的各 3 个种子，1500 步的 2 个种子）。测试图里图案的左上角在第 0 到 5 列，把整张图沿水平方向向右循环平移 t 个像素"
              "（图案不会绕到另一边，绕过去的只有噪声），统计预测类别和平移前不一样的比例。"
              "块大小是 4，所以平移 4 和 8 px 相当于图案整块整块地挪。“按块内位置”是把 6000 张测试图按图案左上角除以 4 的余数分组，"
              "余数是 0 或 1 时 3x3 的图案完整地落在一个块里，是 2 或 3 时被块的边界切开",
        "平移（px）": shifts,
        **{f"预测变了的比例：{name}": [r3(x) for x in np.mean(v, axis=0)] for name, v in flip.items()},
        **{f"平移前的准确率：{name}": r3(np.mean(v)) for name, v in base_acc.items()},
        **{f"按块内位置的准确率：{name}": {"图案完整落在一个块里（行、列余数都是 0 或 1）": r3(m[inside].mean()), "行和列都被块边界切开（余数都是 2 或 3）": r3(m[2:, 2:].mean()),
                                    "最高": r3(m.max()), "最低": r3(m.min()), "4x4 的表（行余数 x 列余数）": [[r3(x) for x in row] for row in m]} for name, m in off_mean.items()},
    }

    fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.0), gridspec_kw={"width_ratios": [1.2, 1.1, 1.2, 0.95]}, layout="constrained")
    xs = np.arange(len(sizes) + 1)
    for name, color, marker in (("ViT", S.BLUE, "o"), ("CNN", S.ORANGE, "s")):
        allv = curve[name] + [inf[name]]
        for j in range(3):
            ax[0].plot(xs, [v[j] for v in allv], marker, color=color, ms=3.5, alpha=0.35)
        ax[0].plot(xs, [np.mean(v) for v in allv], marker + "-", color=color, lw=1.9, ms=5.5, label=f"{name}（{nparams(makers[name]())} 个参数）")
    ax[0].axhline(0.25, color=S.RED, lw=1, ls=":")
    ax[0].set_xticks(xs, [str(n) for n in sizes] + ["每步\n新数据"])
    ax[0].set_xlabel("训练集张数")
    ax[0].set_ylabel("测试准确率")
    ax[0].set_ylim(0.2, 1.03)
    ax[0].set_title("样本越少，CNN 的优势越大")
    ax[0].legend(fontsize=8.5, loc="lower right", bbox_to_anchor=(1.0, 0.1))
    S.clean(ax[0])
    names = list(gen)
    xs = np.arange(len(names))
    for i, (col, color, lab) in enumerate(((0, S.MUTED, "图案在左边（训练时见过）"), (1, S.BLUE, "图案在右边（没见过）"))):
        vals = [gen[n][:, col].mean() for n in names]
        bars = ax[1].bar(xs + (i - 0.5) * 0.38, vals, 0.36, color=color, alpha=0.88, label=lab)
        for b, v in zip(bars, vals):
            ax[1].text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5)
    ax[1].set_xticks(xs, ["ViT\n不加位置编码", "ViT\n正余弦", "CNN"], fontsize=8.5)
    ax[1].set_ylim(0.5, 1.07)
    ax[1].set_ylabel("测试准确率")
    ax[1].set_title("只在左边训练：换到右边都没有退回随机猜")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    S.clean(ax[1])
    ax[1].grid(axis="x", visible=False)
    for name, color, style in (("ViT，正余弦", S.BLUE, "o-"), ("ViT，正余弦，训练 1500 步", S.GREEN, "D-."), ("ViT，不加位置编码", S.PURPLE, "^--"), ("CNN", S.ORANGE, "s-")):
        ax[2].plot(range(len(shifts)), np.mean(flip[name], axis=0), style, color=color, lw=1.8, ms=5.5, label=name)
    ax[2].set_xticks(range(len(shifts)), [str(t) for t in shifts])
    ax[2].set_xlabel("测试图水平循环平移（px）")
    ax[2].set_ylabel("预测类别变了的比例")
    ax[2].set_ylim(-0.01, None)
    ax[2].set_title("平移不是整块时，ViT 的预测变得多")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0)
    S.clean(ax[2])
    m = off_mean["ViT，正余弦"]
    im = ax[3].imshow(m, cmap="viridis", vmin=min(0.7, float(m.min())), vmax=1)
    for r in range(4):
        for c in range(4):
            ax[3].text(c, r, f"{m[r, c]:.2f}", ha="center", va="center", fontsize=9, color="white" if m[r, c] < 0.9 else S.INK)
    ax[3].add_patch(plt.Rectangle((-0.5, -0.5), 2, 2, fill=False, ec=S.ORANGE, lw=1.8))
    ax[3].set_xticks(range(4))
    ax[3].set_yticks(range(4))
    ax[3].set_xlabel("图案左上角的列除以 4 的余数")
    ax[3].set_ylabel("行除以 4 的余数")
    ax[3].set_title("ViT（500 步）的准确率：按块内位置")
    S.save(fig, CH, "fig04_ViT对CNN")


# ---------------------------------------------------------------- 图 5：注意力图与注意力展开
def rollout(attns, residual=True):
    """注意力展开：每层先对头取平均，和单位阵各占一半（代表残差连接）再按行归一化，然后逐层相乘。attns 的每一项是 (B, H, L, L)。"""
    out = None
    for a in attns:
        a = a.mean(1)
        if residual:
            a = 0.5 * a + 0.5 * torch.eye(a.shape[-1])
            a = a / a.sum(-1, keepdim=True)
        out = a if out is None else a @ out
    return out


def fig_attention_maps():
    depth = 3
    names = ["第 1 层", "最后一层", "注意力展开", "遮挡"]
    rows, example = [], None
    for seed in SEEDS:
        xt, yt, pos = test_set(seed, data_type, align=True)
        m = build(seed, lambda: ViT(depth=depth, pool="cls"))
        train(m, seed, fresh=lambda rng, n: data_type(n, rng, align=True))
        logits = predict(m, xt)
        attns = [b.attn for b in m.blocks]
        target = torch.from_numpy(pos[:, 0] // 4 * 4 + pos[:, 1] // 4)
        maps = {"第 1 层": attns[0].mean(1)[:, 0, 1:], "最后一层": attns[-1].mean(1)[:, 0, 1:], "注意力展开": rollout(attns)[:, 0, 1:]}
        base = logits.gather(1, yt[:, None])[:, 0]
        drop = torch.zeros(len(xt), 16)
        g = torch.Generator().manual_seed(seed)
        for p in range(16):
            x2 = xt.clone()
            r, c = divmod(p, 4)
            x2[:, :, r * 4:r * 4 + 4, c * 4:c * 4 + 4] = NOISE * torch.randn(len(xt), 1, 4, 4, generator=g)
            drop[:, p] = base - predict(m, x2).gather(1, yt[:, None])[:, 0]
        maps["遮挡"] = drop
        one = [acc_of(m, xt, yt)]
        for k in names:
            v = maps[k]
            one.append(float((v.argmax(1) == target).float().mean()))
            one.append(float((v.gather(1, target[:, None])[:, 0] / v.sum(1)).mean()) if k != "遮挡" else float("nan"))
        cls_self = [float(a.mean(1)[:, 0, 0].mean()) for a in attns]
        rows.append(one + cls_self)
        if seed == 0:
            pick = next(i for i in range(len(xt)) if int(maps["最后一层"][i].argmax()) != int(target[i]) and int(maps["注意力展开"][i].argmax()) == int(target[i]))
            example = (xt[pick, 0].numpy(), int(target[pick]), {k: maps[k][pick].reshape(4, 4).numpy() for k in names}, int(yt[pick]))
    r = np.array(rows)
    NUM["注意力图"] = {
        "说明": "分图案种类的任务，图案对齐到块的网格上（每张图只有一个块里有图案，其余 15 个块是噪声）。ViT 用类别标记汇聚，3 层，其余同前；3 个种子，每个 1000 张测试图。"
              "对每张图取类别标记那一行的注意力（对 4 个头取平均，去掉类别标记自己那一列），看权重最大的块是不是图案所在的块（命中率，随机猜是 0.0625），以及图案所在的块占 16 个块权重的比例。"
              "注意力展开按 Abnar 和 Zuidema 2020 年的做法：每层对头取平均，和单位阵各占一半再按行归一化，逐层相乘。"
              "遮挡是对照：把 16 个块逐个换成噪声，看正确类别的输出掉得最多的是哪一块",
        "测试准确率": r3(r[:, 0].mean()),
        **{f"{k}：命中率": r3(r[:, 1 + 2 * i].mean()) for i, k in enumerate(names)},
        **{f"{k}：各种子的命中率": [r3(x) for x in r[:, 1 + 2 * i]] for i, k in enumerate(names)},
        **{f"{k}：图案所在块占的权重": r3(r[:, 2 + 2 * i].mean()) for i, k in enumerate(names[:3])},
        "类别标记给自己的权重（第 1、2、3 层，3 个种子的平均）": [r3(x) for x in r[:, 9:12].mean(0)],
        "随机猜的命中率": 1 / 16,
    }

    img, tgt, maps, label = example
    fig, ax = plt.subplots(1, 6, figsize=(17, 3.5), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1.45]}, layout="constrained")
    ax[0].imshow(img, cmap="gray", vmin=-0.3, vmax=1.1)
    for k in range(1, 4):
        ax[0].axhline(k * 4 - 0.5, color=S.ORANGE, lw=0.6, alpha=0.7)
        ax[0].axvline(k * 4 - 0.5, color=S.ORANGE, lw=0.6, alpha=0.7)
    ax[0].set_title(f"输入（{GLYPH_NAMES[label]}）")
    S.image_axes(ax[0])
    for a, k, title in zip(ax[1:5], names, ("第 1 层的注意力", "最后一层的注意力", "注意力展开", "遮挡后输出掉多少")):
        a.imshow(maps[k], cmap="magma")
        r0, c0 = divmod(tgt, 4)
        a.add_patch(plt.Rectangle((c0 - 0.5, r0 - 0.5), 1, 1, fill=False, ec="cyan", lw=1.6))
        a.set_title(title)
        S.image_axes(a)
    vals = [r[:, 1 + 2 * i].mean() for i in range(4)]
    bars = ax[5].bar(range(4), vals, 0.6, color=[S.BLUE, S.RED, S.GREEN, S.MUTED], alpha=0.88)
    for i in range(4):
        ax[5].plot([i] * 3, r[:, 1 + 2 * i], ".", color=S.INK, ms=5)
    for b, v in zip(bars, vals):
        ax[5].text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    ax[5].axhline(1 / 16, color=S.RED, lw=1, ls=":")
    ax[5].set_xticks(range(4), ["第 1 层", "最后一层", "注意力\n展开", "遮挡"], fontsize=9)
    ax[5].set_ylim(0, 1.12)
    ax[5].set_ylabel("最亮的块就是图案所在块的比例")
    ax[5].set_title("最后一层的注意力图常常不指向图案")
    S.clean(ax[5])
    ax[5].grid(axis="x", visible=False)
    S.save(fig, CH, "fig05_注意力图与展开")


# ---------------------------------------------------------------- 图 6：分辨率的代价、块大小与小缺陷
def measure_time():
    out = {}
    d, heads = 64, 4
    g = torch.Generator().manual_seed(0)
    wq, bq = torch.randn(3 * d, d, generator=g) / math.sqrt(d), torch.zeros(3 * d)
    wo, bo = torch.randn(d, d, generator=g) / math.sqrt(d), torch.zeros(d)
    mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    for n, reps in ((64, 30), (256, 20), (1024, 8), (4096, 3)):
        x = torch.randn(1, n, d, generator=g)
        ta, tm = [], []
        with torch.no_grad():
            mod.multi_head_attention(x, wq, bq, wo, bo, heads)
            for _ in range(reps):
                t = time.perf_counter()
                mod.multi_head_attention(x, wq, bq, wo, bo, heads)
                ta.append(time.perf_counter() - t)
                t = time.perf_counter()
                mlp(x)
                tm.append(time.perf_counter() - t)
        out[str(n)] = {"多头注意力（ms）": sig2(1000 * np.median(ta)), "MLP（ms）": sig2(1000 * np.median(tm))}
    return out


def fig_cost():
    # (a) ViT-B/16 的形状：标记数、注意力矩阵、两类计算量
    dim, heads, layers, patch = 768, 12, 12, 16
    sides = [224, 448, 896, 1792]
    shape = {}
    for s in sides:
        n = (s // patch) ** 2
        quad, lin = 2 * n * n * dim, 12 * n * dim * dim
        elems = heads * n * n
        shape[s] = {"标记数（不计类别标记）": n, "一层注意力权重的元素个数（百万）": r3(elems / 1e6), "一层注意力权重占的内存（MiB，float32）": round(elems * 4 / 2 ** 20, 1),
                    "12 层都留着时（MiB，float32）": round(layers * elems * 4 / 2 ** 20, 1), "半精度时一层（MiB）": round(elems * 2 / 2 ** 20, 1),
                    "每层和标记数平方成正比的乘加（十亿）": r3(quad / 1e9), "每层和标记数成正比的乘加（十亿）": r3(lin / 1e9), "平方项占的比例": r3(quad / (quad + lin))}
    NUM["分辨率的代价（按 ViT-B/16 的形状算）"] = {
        "说明": "块大小 16、嵌入 768 维、12 个头、12 层，每张图、不计类别标记。注意力权重是 (头数, N, N) 的张量。乘加次数只数矩阵乘法："
              "和 N 的平方成正比的是 q 乘 k 转置、权重乘 v 两步，合计 2*N*N*D；和 N 成正比的是 q、k、v 三个投影、输出投影和 MLP（隐藏层 4D），合计 12*N*D*D。"
              "这些是按公式数出来的张量大小和乘加次数，不是在任何显卡上量到的显存或耗时",
        **{f"输入 {s}x{s}": v for s, v in shape.items()},
        "边长加倍时：标记数的倍数": 4, "边长加倍时：注意力权重元素个数的倍数": 16,
        "相对 224 的倍数：标记数": [(s // patch) ** 2 // 196 for s in sides], "相对 224 的倍数：注意力权重": [((s // patch) ** 2 // 196) ** 2 for s in sides],
        "每层乘加合计（十亿）": [r3((2 * (s // patch) ** 4 * dim + 12 * (s // patch) ** 2 * dim * dim) / 1e9) for s in sides],
        "相对 224 的倍数：每层乘加合计": [round((2 * (s // patch) ** 4 * dim + 12 * (s // patch) ** 2 * dim * dim) / (2 * 196 ** 2 * dim + 12 * 196 * dim * dim), 1) for s in sides],
        "一个 16x16 的彩色块有多少个数（3*16*16）": 3 * patch * patch,
        "把 1792x1792 的图切成 224x224 的小图": {"块数": (1792 // 224) ** 2, "每块一层注意力权重（MiB）": round(heads * 196 * 196 * 4 / 2 ** 20, 1),
                                        "64 块的注意力权重加起来（MiB）": round(64 * heads * 196 * 196 * 4 / 2 ** 20, 1), "是整图一次算的几分之一": 64},
        "8 GiB 折合 MiB": 8192,
        "平方项和线性项相等时的标记数（6 倍嵌入维数）": 6 * dim, "对应的输入边长（块大小 16）": int(math.sqrt(6 * dim) * patch),
    }
    try:
        import torchvision
        net = torchvision.models.vit_b_16(weights=None)
        blk = net.encoder.layers[0]
        NUM["torchvision 的 vit_b_16（只数参数，不带权重）"] = {
            "说明": "torchvision.models.vit_b_16(weights=None) 随机初始化的结构，只数参数个数",
            "总参数量": nparams(net), "切块嵌入（16x16 步长 16 的卷积）": nparams(net.conv_proj), "位置编码": int(net.encoder.pos_embedding.numel()),
            "类别标记": int(net.class_token.numel()), "一个编码块": nparams(blk), "其中多头注意力": nparams(blk.self_attention), "其中 MLP": nparams(blk.mlp),
            "编码块个数": len(net.encoder.layers), "分类头": nparams(net.heads), "位置编码的形状": list(net.encoder.pos_embedding.shape),
            "12 个编码块占总参数的比例": r3(12 * nparams(blk) / nparams(net)),
            "嵌入维数": int(net.hidden_dim), "头数": int(blk.self_attention.num_heads), "每个头的维数": int(net.hidden_dim // blk.self_attention.num_heads),
            "MLP 隐藏层的维数": int(blk.mlp[0].out_features), "块大小": int(net.patch_size), "默认输入边长": int(net.image_size),
            "标记数（196 个块加 1 个类别标记）": int(net.encoder.pos_embedding.shape[1]),
            "输入边长改成 384 时位置编码的形状": list(torchvision.models.vit_b_16(weights=None, image_size=384).encoder.pos_embedding.shape),
            "torchvision 版本": torchvision.__version__, "torch 版本": torch.__version__,
        }
    except Exception as exc:  # noqa: BLE001
        NUM["torchvision 的 vit_b_16（只数参数，不带权重）"] = {"说明": f"没有运行：{type(exc).__name__}"}

    timing = S.pinned(CH, "耗时", lambda: {
        "说明": "一层多头自注意力（参考实现，64 维，4 个头）和一个 MLP（64 到 256 到 64）的前向耗时，批大小 1，不求梯度，取多次的中位数。"
              "在一台 Apple 芯片的 Mac 上用 CPU（2 线程）测的一次结果，测的时候机器上还有别的任务，只看量级和倍数",
        **measure_time()}, "CH25_RETIME")
    NUM["耗时"] = timing
    ns = [64, 256, 1024, 4096]
    ta = [timing[str(n)]["多头注意力（ms）"] for n in ns]
    tm = [timing[str(n)]["MLP（ms）"] for n in ns]
    NUM["耗时的倍数（由上面的耗时算出）"] = {
        "标记数从 1024 到 4096（4 倍）：注意力耗时的倍数": sig2(ta[3] / ta[2]), "标记数从 1024 到 4096：MLP 耗时的倍数": sig2(tm[3] / tm[2]),
        "标记数从 64 到 256：注意力耗时的倍数": sig2(ta[1] / ta[0]), "4096 个标记时注意力耗时是 MLP 的几倍": sig2(ta[3] / tm[3]),
        "64 维、4 个头时 4096 个标记的注意力权重元素个数（百万）": r3(4 * 4096 * 4096 / 1e6),
    }

    # (b) 块大小与小缺陷
    patches = [4, 8, 16]
    res, res_long = {}, {}
    for p in patches:
        rows = []
        for seed in SEEDS[:2]:
            xt, yt, _ = test_set(seed, data_defect)
            m = build(seed, lambda: ViT(hw=(32, 32), patch=p, classes=2, pos="none"))
            train(m, seed, fresh=lambda rng, n: data_defect(n, rng))
            rows.append(acc_of(m, xt, yt))
            if p == 4:
                KEEP[("defect", seed)] = m
        res[p] = rows
    for p in patches[1:]:
        rows = []
        for seed in SEEDS[:2]:
            xt, yt, _ = test_set(seed, data_defect)
            m = build(seed, lambda: ViT(hw=(32, 32), patch=p, classes=2, pos="none"))
            train(m, seed, fresh=lambda rng, n: data_defect(n, rng), steps=3 * STEPS)
            rows.append(acc_of(m, xt, yt))
        res_long[p] = rows
    NUM["块大小与小缺陷"] = {
        "说明": "32x32 的噪声图（标准差 0.15）里有没有一个 2x2 像素、幅度 0.6 的亮斑，位置随机，随机猜是 0.5。ViT 嵌入 32 维、2 层、4 个头、不加位置编码、全局平均，只改块大小。"
              "每步新生成一批（批 32），2 个种子，测试集 1000 张",
        "块大小": patches, "标记数": [(32 // p) ** 2 for p in patches], "一个块的像素数": [p * p for p in patches],
        "亮斑占一个块的面积比例": [r3(4 / (p * p)) for p in patches],
        "只看整块像素之和时的信噪比（亮斑完整落在块里：4 个像素共多出 2.4，除以块内噪声之和的标准差 0.15*P）": [r3(4 * 0.6 / (NOISE * p)) for p in patches],
        "亮斑的 4 个像素一共多出的亮度": r3(4 * 0.6),
        "参数量": [nparams(ViT(hw=(32, 32), patch=p, classes=2, pos="none")) for p in patches],
        "500 步的测试准确率": [r3(np.mean(res[p])) for p in patches], "500 步各种子": [[r3(x) for x in res[p]] for p in patches],
        "1500 步的测试准确率（只测了块大小 8 和 16）": {str(p): r3(np.mean(v)) for p, v in res_long.items()},
        "1500 步各种子": {str(p): [r3(x) for x in v] for p, v in res_long.items()},
    }

    # (c) 切块推理：小图上训练的模型拿去看大图
    tile_rows = []
    for seed in SEEDS:
        m = build(seed, lambda: ViT(hw=(16, 16), patch=4, classes=2, pos="none"))
        train(m, seed, fresh=lambda rng, n: data_defect(n, rng, size=16))
        xt, yt, pos = data_defect(4000, np.random.default_rng([seed, 21]), size=32)

        def run(starts):
            tiles = torch.stack([xt[:, :, r:r + 16, c:c + 16] for r in starts for c in starts], 1).reshape(-1, 1, 16, 16)
            return predict(m, tiles).softmax(1)[:, 1].reshape(len(xt), -1).max(1).values > 0.5

        cut = torch.from_numpy((pos[:, 0] == 15) | (pos[:, 1] == 15))            # 亮斑被不重叠的 16x16 块的边界切开
        has = yt == 1
        one = []
        for starts in ((0, 16), (0, 8, 16)):
            hit = run(starts)
            one += [float(hit[has & ~cut].float().mean()), float(hit[has & cut].float().mean()), float(hit[~has].float().mean())]
        whole = predict(m, xt).softmax(1)[:, 1] > 0.5                              # 不加位置编码、全局平均的模型可以直接吃 32x32（64 个标记）
        one += [float(whole[has].float().mean()), float(whole[~has].float().mean()), float((has & cut).sum()), float(has.sum())]
        xs16, ys16, _ = data_defect(4000, np.random.default_rng([seed, 22]), size=16)
        p16 = predict(m, xs16).softmax(1)[:, 1] > 0.5
        one += [float(p16[ys16 == 1].float().mean()), float(p16[ys16 == 0].float().mean())]
        tile_rows.append(one)
    t = np.mean(tile_rows, axis=0)
    NUM["切块推理"] = {
        "说明": "在 16x16 的图上训练“有没有亮斑”（块大小 4，16 个标记，500 步），再拿去判断 32x32 的图：切成若干个 16x16 的小图分别推理，任何一块报警就算整张图报警（阈值 0.5）。"
              "不重叠是 4 块；重叠一半（步长 8）是 9 块。“被边界切开”指 2x2 的亮斑正好跨在不重叠切法的块边界上。3 个种子取平均，每个 4000 张 32x32 的测试图",
        "单张 16x16 小图上：检出率": r3(t[10]), "单张 16x16 小图上：误报率": r3(t[11]),
        "不重叠 4 块：亮斑完整时的检出率": r3(t[0]), "不重叠 4 块：亮斑被边界切开时的检出率": r3(t[1]), "不重叠 4 块：没有亮斑的图报警的比例": r3(t[2]),
        "重叠 9 块：亮斑完整时的检出率": r3(t[3]), "重叠 9 块：同一批被切开的亮斑的检出率": r3(t[4]), "重叠 9 块：没有亮斑的图报警的比例": r3(t[5]),
        "整张 32x32 直接送进去（64 个标记）：检出率": r3(t[6]), "整张直接送进去：没有亮斑的图报警的比例": r3(t[7]),
        "有亮斑的图里，亮斑被边界切开的比例": r3(t[8] / t[9]),
        "按每块独立、单块误报率同上估计的整图误报率": {"4 块": r3(1 - (1 - t[11]) ** 4), "9 块": r3(1 - (1 - t[11]) ** 9)},
        "重叠 9 块相对不重叠 4 块：推理的块数是几倍": 9 / 4,
        "每张图的注意力权重元素个数（4 个头）": {"不重叠 4 块": 4 * 4 * 16 * 16, "重叠 9 块": 9 * 4 * 16 * 16, "整张 64 个标记": 4 * 64 * 64},
    }

    # (d) 窗口注意力：用掩码把注意力限制在窗口里
    def window(shift):
        idx = (torch.arange(8) + shift) % 8 // 2
        return idx[:, None] == idx[None, :]

    win = {}
    for name, masks, d in (("两层都用同样的窗口", [window(0), window(0)], 4), ("第二层把窗口错开一个标记", [window(0), window(1)], 4),
                           ("第二层把窗口错开一个标记", [window(0), window(1)], 8), ("第二层把窗口错开一个标记", [window(0), window(1)], 16)):
        rows = []
        for seed in SEEDS[:2]:
            xt, yt, tok = test_set(seed, data_pair, d=d)
            m = build(seed, lambda: ViT(hw=(4, 32), depth=2, classes=2, masks=masks))
            train(m, seed, fresh=lambda rng, n, d=d: data_pair(n, rng, d))
            ok = (predict(m, xt).argmax(1) == yt).numpy()
            same = (tok[:, 0] // 2) == (tok[:, 1] // 2)
            rows.append([ok.mean(), ok[same].mean() if same.any() else float("nan"), ok[~same].mean()])
        win[(name, d)] = np.array(rows)
    NUM["窗口注意力"] = {
        "说明": "图 1 的窄条任务（8 个标记，判断两个图案是否同一种），ViT 改成 2 层，用注意力掩码把每个标记能看的范围限制在 2 个标记（8 px）的窗口里：标记 0 和 1 一组，2 和 3 一组，依此类推。"
              "“错开”指第二层的窗口整体挪一个标记（1 和 2 一组，3 和 4 一组，7 和 0 一组）。每个间距单独训练，2 个种子，随机猜是 0.5",
        **{f"{name}，相距 {d} px": {"准确率": r3(v[:, 0].mean()), "各种子": [r3(x) for x in v[:, 0]],
                                  "两个图案在第一层的同一个窗口里": (r3(np.nanmean(v[:, 1])) if not np.isnan(v[:, 1]).all() else "没有这种情况"),
                                  "两个图案在第一层的不同窗口里": r3(v[:, 2].mean())} for (name, d), v in win.items()},
        "8 个标记时注意力权重的元素个数（每个头）": {"全局": 64, "窗口（4 个窗口，每个 2x2）": 16},
        "按 ViT-B/16 的形状、输入 1792x1792（12544 个标记）时每个头的元素个数（百万）": {"全局": r3(12544 ** 2 / 1e6), "7x7 个标记的窗口": r3(12544 * 49 / 1e6)},
        "全局是 7x7 窗口的几倍（12544 除以 49）": 12544 // 49, "7x7 的窗口里有几个标记": 49,
        "两层 2 个标记的窗口（第二层错开）最远能把相距几个标记的信息接起来": 3, "折合像素": 12,
    }

    fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.0), gridspec_kw={"width_ratios": [1.15, 1.1, 1.0, 1.25]}, layout="constrained")
    mem = [shape[s]["一层注意力权重占的内存（MiB，float32）"] for s in sides]
    rel_n = [(s // patch) ** 2 / 196 for s in sides]
    ax[0].plot(range(len(sides)), [v * v for v in rel_n], "o-", color=S.RED, lw=1.8, ms=5.5, label="注意力权重的元素个数")
    ax[0].plot(range(len(sides)), rel_n, "s--", color=S.BLUE, lw=1.6, ms=5, label="标记数")
    for i, v in enumerate(mem):
        ax[0].text(i - 0.08 if i else 0.05, rel_n[i] ** 2 * (1.5 if i else 2.4), f"{v:g} MiB", ha="right", fontsize=8.5, color=S.RED)
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(len(sides)), [str(s) for s in sides])
    ax[0].set_xlabel("输入边长（px，块大小 16）")
    ax[0].set_ylabel("相对 224 输入的倍数")
    ax[0].set_title("边长加倍：标记数 4 倍，注意力权重 16 倍")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax[0].set_xlim(-0.6, len(sides) - 0.7)
    ax[0].set_ylim(0.6, 3e4)
    ax[0].minorticks_off()
    plain_log_ticks(ax[0].yaxis)
    S.clean(ax[0])
    ax[1].plot(range(4), ta, "o-", color=S.RED, lw=1.8, ms=5.5, label="多头注意力")
    ax[1].plot(range(4), tm, "s--", color=S.BLUE, lw=1.6, ms=5, label="MLP")
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(4), [str(n) for n in ns])
    ax[1].set_xlabel("标记数（每一格是 4 倍）")
    ax[1].set_ylabel("一次前向的 CPU 耗时（ms）")
    ax[1].set_title("实测耗时：注意力比 MLP 涨得快")
    ax[1].minorticks_off()
    plain_log_ticks(ax[1].yaxis)
    ax[1].legend(fontsize=8.5, loc="upper left")
    S.clean(ax[1])
    vals = [np.mean(res[p]) for p in patches]
    bars = ax[2].bar(range(3), vals, 0.6, color=S.BLUE, alpha=0.88)
    for i, p in enumerate(patches):
        ax[2].plot([i] * 2, res[p], ".", color=S.INK, ms=5)
        ax[2].text(i, vals[i] + 0.03, f"{vals[i]:.2f}", ha="center", fontsize=9)
    ax[2].axhline(0.5, color=S.RED, lw=1, ls=":")
    ax[2].set_xticks(range(3), [f"块 {p}x{p}\n{(32 // p) ** 2} 个标记" for p in patches], fontsize=8.5)
    ax[2].set_ylim(0.4, 1.08)
    ax[2].set_ylabel("测试准确率（500 步）")
    ax[2].set_title("2x2 的亮斑：块越大越难检出")
    S.clean(ax[2])
    ax[2].grid(axis="x", visible=False)
    groups = ["亮斑完整", "亮斑被边界切开", "没有亮斑\n（误报）"]
    xs = np.arange(3)
    for i, (off, color, lab) in enumerate(((0, S.MUTED, "不重叠 4 块"), (3, S.GREEN, "重叠一半，9 块"))):
        v = [t[off], t[off + 1], t[off + 2]]
        bars = ax[3].bar(xs + (i - 0.5) * 0.38, v, 0.36, color=color, alpha=0.88, label=lab)
        for b, x in zip(bars, v):
            ax[3].text(b.get_x() + b.get_width() / 2, x + 0.02, f"{x:.3f}", ha="center", fontsize=8)
    ax[3].set_xticks(xs, groups, fontsize=8.5)
    ax[3].set_ylim(0, 1.12)
    ax[3].set_ylabel("整张图报警的比例")
    ax[3].set_title("切块推理：重叠把被切开的亮斑找回一部分")
    ax[3].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    S.clean(ax[3])
    ax[3].grid(axis="x", visible=False)
    S.save(fig, CH, "fig06_分辨率的代价")


def numbers_model():
    m = ViT()
    NUM["小 ViT 的参数量"] = {
        "说明": VIT_NOTE + "，输入 1x16x16，4 类",
        "合计": nparams(m), "切块嵌入": nparams(m.embed), "一个编码块": nparams(m.blocks[0]),
        "其中注意力（q、k、v 和输出投影）": int(m.blocks[0].w_qkv.numel() + m.blocks[0].b_qkv.numel() + nparams(m.blocks[0].out)),
        "其中 MLP": nparams(m.blocks[0].mlp), "其中两个 LayerNorm": nparams(m.blocks[0].ln1) + nparams(m.blocks[0].ln2),
        "最后的 LayerNorm 和分类头": nparams(m.ln) + nparams(m.head),
        "换成可学习位置编码多出的参数": nparams(ViT(pos="learned")) - nparams(m), "换成类别标记多出的参数": nparams(ViT(pool="cls")) - nparams(m),
        "标记数": 16, "每个标记的像素数": 16,
    }
    NUM["理论常数"] = {"ln 2": r3(LN2), "ln 4": r3(LN4), "16 个标记均匀注意力的权重": 1 / 16, "8 个标记均匀注意力的权重": 1 / 8}


def numbers_heads():
    """头数：嵌入维数不变，把 32 维切成 1、2、4、8 个头。"""
    res = {}
    for heads in (1, 2, 4, 8):
        rows = []
        for seed in SEEDS:
            xt, yt, _ = test_set(seed, data_type)
            if heads == 4:
                m = KEEP[("图案是哪一种", "正余弦", seed)]
            else:
                m = build(seed, lambda: ViT(heads=heads))
                train(m, seed, fresh=lambda rng, n: data_type(n, rng))
            rows.append(acc_of(m, xt, yt))
        res[heads] = rows
    NUM["头数"] = {
        "说明": "分图案种类的任务（16x16 的图，图案位置随机）。" + VIT_NOTE + "，只改头数，嵌入维数保持 32，所以每个头的维数是 32 除以头数。" + DATA_NOTE + "。3 个种子",
        "头数": [1, 2, 4, 8], "每个头的维数": [32, 16, 8, 4],
        "参数量": [nparams(ViT(heads=h)) for h in (1, 2, 4, 8)],
        "测试准确率": [r3(np.mean(res[h])) for h in (1, 2, 4, 8)],
        "各种子": [[r3(x) for x in res[h]] for h in (1, 2, 4, 8)],
    }


def numbers_demo():
    """演示页“按顺序试”各步的数字，用参考实现重算一遍。预设直接从演示页里读，保证两边用的是同一组坐标。"""
    import re
    page = S.MAIN / "demos" / "自注意力.html"
    if not page.exists():
        NUM["演示页的数字"] = {"说明": "没有运行：找不到 demos/自注意力.html"}
        return
    presets = [json.loads(m) for m in re.findall(r"data-preset='([^']+)'", page.read_text(encoding="utf-8"))]

    def weights(preset, query=None, tau=None, heads=None, mask=None, pts=None):
        pts_t = torch.tensor(preset["pts"] if pts is None else pts, dtype=torch.float64)
        qi = preset["q"] if query is None else query
        tau = preset["tau"] if tau is None else tau
        heads = preset["heads"] if heads is None else heads
        mk = torch.tensor(preset["mask"] if mask is None else mask, dtype=torch.bool)[None]
        outs, ws = [], []
        for h in range(heads):
            cols = [0, 1] if heads == 1 else [h]
            k = pts_t[:, cols]
            o, w = mod.scaled_dot_product_attention(k[qi:qi + 1] / tau, k, k, mk)      # 温度写成把查询除以 tau
            outs.append(o[0])
            ws.append(w[0])
        return ws, torch.cat(outs)

    def eff(w):
        return float(1 / (w ** 2).sum()) if float(w.sum()) > 0 else 0.0

    names = "ABCDEF"
    w_a, out_a = weights(presets[0])
    w_c, _ = weights(presets[0], query=2)
    step2 = {str(tau): weights(presets[1], tau=tau)[0][0] for tau in (1, 0.1, 10)}
    far = [list(v) for v in presets[1]["pts"]]
    far[2] = [2.2, 0.0]
    w_far, _ = weights(presets[1], pts=far)
    w3 = {str(tau): weights(presets[2], tau=tau) for tau in (0.1, 1, 10)}
    mean3 = torch.tensor(presets[2]["pts"], dtype=torch.float64).mean(0)
    w4, _ = weights(presets[3])
    w4h, _ = weights(presets[3], heads=2)
    w5, _ = weights(presets[4])
    w5z, out5z = weights(presets[4], mask=[0] * 6)
    order = sorted(range(6), key=lambda j: -float(w_a[0][j]))
    NUM["演示页的数字"] = {
        "说明": "demos/自注意力.html 里“按顺序试”五步的预设，用本章作业的参考实现重算（6 个二维的点同时当查询、键、值；温度 tau 等价于把查询除以 tau；"
              "2 个头时第 1 个头只用横坐标、第 2 个头只用纵坐标）。“相当于平均分给几个键”是权重平方和的倒数",
        "第 1 步：查询 A 给 A 到 F 的权重": [r3(x) for x in w_a[0]], "第 1 步：A 的权重从大到小": "".join(names[j] for j in order),
        "第 1 步：查询 A 的输出": [r3(x) for x in out_a],
        "第 1 步：换成查询 C，给 A 到 F 的权重": [r3(x) for x in w_c[0]],
        "第 2 步：查询 C 相当于平均分给几个键（温度 1、0.1、10）": [round(eff(step2[k]), 2) for k in ("1", "0.1", "10")],
        "第 2 步：温度 1 时 C 给自己的权重": r3(step2["1"][2]), "第 2 步：温度 10 时最大和最小的权重": [r3(step2["10"].max()), r3(step2["10"].min())],
        "第 2 步：把 C 拖到 (2.2, 0) 后给自己的权重和相当于几个键": [r3(w_far[0][2]), round(eff(w_far[0]), 2)],
        "第 3 步：查询在原点时每个键的权重（温度 0.1、1、10 都一样）": r3(w3["1"][0][0][0]),
        "第 3 步：三种温度下权重的最大差": sig2(max(float((w3[k][0][0] - 1 / 6).abs().max()) for k in w3)),
        "第 3 步：输出和 6 个点的平均": [[r3(x) for x in w3["1"][1]], [r3(x) for x in mean3]],
        "第 4 步：1 个头时 A 给自己、B、C 的权重": [r3(w4[0][0]), r3(w4[0][1]), r3(w4[0][2])],
        "第 4 步：2 个头时第 1 个头 A 给自己、B、C 的权重": [r3(w4h[0][0]), r3(w4h[0][1]), r3(w4h[0][2])],
        "第 4 步：2 个头时第 2 个头 A 给自己、B、C 的权重": [r3(w4h[1][0]), r3(w4h[1][1]), r3(w4h[1][2])],
        "第 5 步：屏蔽 C 之后查询 A 给 A 到 F 的权重": [r3(x) for x in w5[0]],
        "第 5 步：全部屏蔽后权重之和与输出": [float(w5z[0].sum()), [float(x) for x in out5z]],
        "点的个数": 6, "均匀时每个键的权重": r3(1 / 6),
        "各步预设的坐标（A 到 F）": {f"第 {i + 1} 步": q["pts"] for i, q in enumerate(presets)}, "第 2 步把 C 拖到的位置": far[2],
    }


def main() -> None:
    S.setup()
    t_all = time.perf_counter()
    for fn in (numbers_model, numbers_mechanics, fig_longrange, fig_position, fig_layernorm, fig_vit_vs_cnn, fig_attention_maps, fig_cost, numbers_heads, numbers_demo):
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
