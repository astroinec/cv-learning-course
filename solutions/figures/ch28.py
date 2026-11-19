"""第 28 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch28.py

输出：<主分支>/28_视觉语言模型/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。

全部实验用代码生成的 32x32 彩色小图和配套的短句（“图文对”），以及一个从零训练的很小的双塔模型，在 CPU 上跑，
不下载任何数据和权重。零样本分类器、零样本预测、Recall@K、模态间隙和区域打分调用本章作业的参考实现。
二十几个小模型分给 3 个进程训练，每个进程 1 个线程；结果和进程数无关（CH28_WORKERS 可以改进程数）。
耗时默认沿用上次的记录，设置 CH28_RETIME=1 才重测。
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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

CH = "28_视觉语言模型"
hw = S.load_solution(CH, "28-1_零样本分类与图文检索.py")
hw19 = S.load_solution("19_异常检测与深度缺陷检测", "19-1_PatchCore核心.py")  # 只用它的 auroc
NUM: dict = {}
TIMES: dict = {}

SIZE, CELL, P = 32, 16, 8
G = SIZE // P
L = G * G
DIM, OUT = 64, 32
STEPS, BATCH, LR = 1200, 64, 2e-3
LONG_STEPS = 2400                                     # 属性绑定一节把训练加长一倍
NOISE = 0.05
SEEDS = (0, 1, 2)
N_TEST = 1000

SHAPES = ["方块", "圆", "三角", "十字"]
COLORS = ["红色", "绿色", "蓝色", "黄色"]
SIZES = ["大", "小"]
TEXS = ["实心", "条纹"]
COUNTS = ["一个", "两个", "三个"]
CELLS = ["左上", "右上", "左下", "右下"]
UNSEEN = ["一张", "的", "照片"]                      # 在词表里，但训练句子里从不出现
VOCAB = ["<pad>"] + COUNTS + SIZES + TEXS + COLORS + SHAPES + CELLS + ["在", "左边", "右边", "图", "里", "有", "这", "是", "空", "背景", "图形"] + UNSEEN
TOK = {w: i for i, w in enumerate(VOCAB)}
MAXLEN = 12
RGB = torch.tensor([[0.90, 0.22, 0.20], [0.20, 0.78, 0.30], [0.25, 0.42, 0.95], [0.92, 0.84, 0.20]])
EXT = (12, 8)                                         # “大”和“小”的边长（px）
KINDS = (0.45, 0.2, 0.3, 0.05)                        # 单个目标、多个相同目标、左右两个不同目标、空图
HOLD = ((0, 1), (1, 2), (2, 3), (3, 0))               # 训练时拿掉的（颜色，形状）：红圆、绿三角、蓝十字、黄方块
HOLD_NAMES = [COLORS[c] + SHAPES[s] for c, s in HOLD]
PREFIX = ([], ["图", "里", "有"], ["这", "是"])
COMBOS = [f"{c} {s}" for c in COLORS for s in SHAPES]  # 类别编号 = 颜色 x 4 + 形状

DATA_NOTE = (f"{SIZE}x{SIZE} 的彩色小图分成 2x2 个 {CELL}x{CELL} 的格子，每格至多一个目标。目标有 4 种形状（方块、圆、三角、十字）、4 种颜色（红、绿、蓝、黄）、"
             f"2 种大小（边长 {EXT[0]} px 和 {EXT[1]} px）、2 种纹理（实心，或者每隔 2 行压暗到 0.3 倍的条纹），在格子里的位置有正负 1 px 的抖动，亮度在 0.8 到 1 倍之间随机；"
             f"背景是 0.15 到 0.3 之间的灰度，像素噪声标准差 {NOISE}。一张图是四种场景之一：单个目标（概率 {KINDS[0]}）、2 到 3 个完全相同的目标（{KINDS[1]}）、"
             f"同一行左右两个颜色和形状都不同的目标（{KINDS[2]}）、空图（{KINDS[3]}）。配套的句子由场景的属性拼出来：前缀随机取“（无）”“图 里 有”“这 是”之一；"
             "单个目标写成“一个 [大小] [纹理] [颜色] 形状”，大小和纹理各以 0.5 的概率写出，颜色以 0.85 的概率写出，形状以 0.85 的概率写出、否则写成“图形”，"
             "另有 0.4 的概率加上“在 左上”这样的位置；多个相同目标把“一个”换成“两个”“三个”；左右两个目标写成“红色 方块 在 蓝色 圆 左边”或者等价的“右边”说法；空图写成“空 背景”。"
             f"词表 {len(VOCAB)} 个词（含填充符和 3 个训练句子里从不出现的词），句子最长 {MAXLEN} 个词")
MODEL_NOTE = (f"图像塔是一个很小的 ViT：{P}x{P} 的块（每张图 {L} 个），宽 {DIM}，2 层，4 个头，固定的正弦位置编码，对 {L} 个块的输出取平均后线性投影到 {OUT} 维；"
              f"文本塔是词嵌入加可学习的位置编码、2 层同样的编码块，对有效词的输出取平均后线性投影到 {OUT} 维。两边的向量做 L2 归一化后算余弦相似度，"
              "损失是第 13 章 5.3 节的对称交叉熵，温度是可学习的参数（初值 0.07，相似度的放大倍数上限 100）。"
              f"优化器 AdamW（学习率 {LR}，权重衰减 0.05，学习率先升后降），批大小 {BATCH}，每步新生成数据，训练 {STEPS} 步")


def r3(v):
    return round(float(v), 3)


def r2(v):
    return round(float(v), 2)


def gen(*seed):
    return torch.Generator().manual_seed(int(sum(s * 1000 ** i for i, s in enumerate(seed))))


# ---------------------------------------------------------------- 数据
def _bank():
    """BANK[形状, 大小, jx, jy] 是一个格子大小的 0/1 掩码；jx、jy 取 0 到 2，对应目标中心偏移 -1、0、+1 px。"""
    yy, xx = torch.meshgrid(torch.arange(CELL, dtype=torch.float32), torch.arange(CELL, dtype=torch.float32), indexing="ij")
    out = torch.zeros(4, 2, 3, 3, CELL, CELL)
    for z, ext in enumerate(EXT):
        half = ext / 2
        for jx in range(3):
            for jy in range(3):
                dx, dy = xx - (7.5 + jx - 1), yy - (7.5 + jy - 1)
                square = (dx.abs() < half) & (dy.abs() < half)
                circle = dx ** 2 + dy ** 2 < (half + 0.2) ** 2
                tri = (dy.abs() < half) & (dx.abs() < (dy + half) / 2 + 0.5)
                arm = ext / 6 + 0.3
                cross = ((dx.abs() < arm) & (dy.abs() < half)) | ((dy.abs() < arm) & (dx.abs() < half))
                for s, m in enumerate((square, circle, tri, cross)):
                    out[s, z, jx, jy] = m.float()
    return out


BANK = _bank()
ROWS = torch.arange(CELL)[:, None].expand(CELL, CELL)
STRIPE = torch.where((ROWS // 2) % 2 == 0, 1.0, 0.3)
CY, CX = torch.meshgrid(torch.arange(CELL, dtype=torch.float32), torch.arange(CELL, dtype=torch.float32), indexing="ij")


def _objects(n, g, combos):
    """n 个目标的属性 (形状, 颜色, 大小, 纹理)；（颜色, 形状）只从 combos 里取。"""
    pick = combos[torch.randint(0, len(combos), (n,), generator=g)]
    return torch.stack([pick[:, 1], pick[:, 0], torch.randint(0, 2, (n,), generator=g), torch.randint(0, 2, (n,), generator=g)], 1)


def sample_scenes(n, g, kinds=KINDS, hold=None, only_hold=False):
    """随机场景。kind：0 单个目标，1 多个相同目标，2 左右两个不同目标，3 空图。hold 给出要排除的（颜色，形状）组合；only_hold=True 时反过来只用这些组合。"""
    all_c = [(c, s) for c in range(4) for s in range(4)]
    if hold is not None:
        all_c = [cs for cs in all_c if (cs in hold) == only_hold]
    combos = torch.tensor(all_c)
    kind = torch.multinomial(torch.tensor(kinds), n, replacement=True, generator=g)
    a1, a2 = _objects(n, g, combos), _objects(n, g, combos)
    for _ in range(30):                                # 第二个目标的颜色和形状都要和第一个不同
        clash = (a1[:, 0] == a2[:, 0]) | (a1[:, 1] == a2[:, 1])
        a2 = torch.where(clash[:, None], _objects(n, g, combos), a2)
    clash = (a1[:, 0] == a2[:, 0]) | (a1[:, 1] == a2[:, 1])
    kind = torch.where((kind == 2) & clash, torch.zeros_like(kind), kind)
    rank = torch.rand(n, 4, generator=g).argsort(1).argsort(1)          # 每个格子的随机名次
    cnt = torch.randint(2, 4, (n,), generator=g)
    row = torch.randint(0, 2, (n,), generator=g)
    cell = torch.arange(4)[None]
    in_row = (cell // 2) == row[:, None]
    occ = torch.where((kind == 0)[:, None], rank == 0, torch.zeros(n, 4, dtype=torch.bool))
    occ = torch.where((kind == 1)[:, None], rank < cnt[:, None], occ)
    occ = torch.where((kind == 2)[:, None], in_row, occ)
    attr = a1[:, None, :].expand(n, 4, 4).clone()
    right = (kind == 2)[:, None] & in_row & (cell % 2 == 1)
    attr = torch.where(right[..., None], a2[:, None, :], attr)
    return {"kind": kind, "occ": occ, "attr": attr * occ[..., None]}


def render(sc, g, noise=NOISE, notch=None):
    """把场景画成 (N, 3, 32, 32) 的图。notch 是 (N, 4) 的整数，大于 0 时在该格目标的中心挖一个 notch x notch 的暗坑（训练里从不出现）。"""
    n = len(sc["kind"])
    occ, attr = sc["occ"], sc["attr"]
    img = (0.15 + 0.15 * torch.rand(n, 1, 1, 1, generator=g)).expand(n, 3, SIZE, SIZE).clone()
    jx = torch.randint(0, 3, (n, 4), generator=g)
    jy = torch.randint(0, 3, (n, 4), generator=g)
    bright = 0.8 + 0.2 * torch.rand(n, 4, generator=g)
    for c in range(4):
        m = BANK[attr[:, c, 0], attr[:, c, 2], jx[:, c], jy[:, c]] * occ[:, c, None, None].float()
        tex = torch.where(attr[:, c, 3, None, None] == 1, STRIPE[None], torch.ones(1, CELL, CELL))
        obj = (RGB[attr[:, c, 1]] * bright[:, c, None])[:, :, None, None] * tex[:, None]
        if notch is not None:
            half = notch[:, c, None, None].float() / 2
            hole = ((CX[None] - (6.5 + jx[:, c, None, None])).abs() < half) & ((CY[None] - (6.5 + jy[:, c, None, None])).abs() < half)
            obj = torch.where(hole[:, None], torch.full_like(obj, 0.05), obj)
        y0, x0 = (c // 2) * CELL, (c % 2) * CELL
        img[:, :, y0:y0 + CELL, x0:x0 + CELL] = img[:, :, y0:y0 + CELL, x0:x0 + CELL] * (1 - m[:, None]) + obj * m[:, None]
    return img + noise * torch.randn(n, 3, SIZE, SIZE, generator=g)


def describe(a, rng, count="一个", tex_words=True):
    s, c, z, t = a
    w = [count]
    if rng.random() < 0.5:
        w.append(SIZES[z])
    if rng.random() < 0.5 and tex_words:
        w.append(TEXS[t])
    use_c, use_s = rng.random() < 0.85, rng.random() < 0.85
    if not use_c and not use_s:
        use_c = use_s = True
    if use_c:
        w.append(COLORS[c])
    w.append(SHAPES[s] if use_s else "图形")
    return w


def pair_words(sc, i):
    """左右两个目标各自的（颜色词，形状词）。"""
    l, r = sorted(sc["occ"][i].nonzero().flatten().tolist())
    a, b = sc["attr"][i, l].tolist(), sc["attr"][i, r].tolist()
    return (COLORS[a[1]], SHAPES[a[0]]), (COLORS[b[1]], SHAPES[b[0]])


def caption(sc, i, rng, tex_words=True):
    """训练用的句子（词的列表）。tex_words=False 时句子里永远不出现“实心”“条纹”。"""
    kind = int(sc["kind"][i])
    cells = sc["occ"][i].nonzero().flatten().tolist()
    pre = list(PREFIX[int(rng.integers(0, 3))])
    if kind == 3:
        return pre + ["空", "背景"]
    if kind == 0:
        w = pre + describe(sc["attr"][i, cells[0]].tolist(), rng, tex_words=tex_words)
        return w + ["在", CELLS[cells[0]]] if rng.random() < 0.4 else w
    if kind == 1:
        return pre + describe(sc["attr"][i, cells[0]].tolist(), rng, count=COUNTS[len(cells) - 1], tex_words=tex_words)
    a, b = pair_words(sc, i)
    return pre + ([a[0], a[1], "在", b[0], b[1], "左边"] if rng.random() < 0.5 else [b[0], b[1], "在", a[0], a[1], "右边"])


def swapped_caption(sc, i, rng):
    """左右两个目标的句子的“难负样本”：把两个颜色词对调，或者把两个目标的先后对调。词还是那几个，描述的却是另一张图。"""
    a, b = pair_words(sc, i)
    rel = "左边" if rng.random() < 0.5 else "右边"
    first, second = (a, b) if rel == "左边" else (b, a)
    if rng.random() < 0.5:
        return [second[0], first[1], "在", first[0], second[1], rel]
    return [second[0], second[1], "在", first[0], first[1], rel]


def full_caption(sc, i):
    """把场景的全部属性都写出来的句子，检索实验用。"""
    kind = int(sc["kind"][i])
    cells = sc["occ"][i].nonzero().flatten().tolist()
    if kind == 2:
        a, b = pair_words(sc, i)
        return [a[0], a[1], "在", b[0], b[1], "左边"]
    s, c, z, t = sc["attr"][i, cells[0]].tolist()
    if kind == 0:
        return ["一个", SIZES[z], TEXS[t], COLORS[c], SHAPES[s], "在", CELLS[cells[0]]]
    return [COUNTS[len(cells) - 1], SIZES[z], TEXS[t], COLORS[c], SHAPES[s]]


def encode(sents):
    out = torch.zeros(len(sents), MAXLEN, dtype=torch.long)
    for i, s in enumerate(sents):
        ids = [TOK[w] for w in (s.split() if isinstance(s, str) else s)]
        out[i, :len(ids)] = torch.tensor(ids)
    return out


def single_test(n, *seed, hold=None, only_hold=False, noise=NOISE):
    """n 张只有一个目标的图，以及每张图的各种标签。"""
    g = gen(*seed)
    sc = sample_scenes(n, g, kinds=(1.0, 0.0, 0.0, 0.0), hold=hold, only_hold=only_hold)
    img = render(sc, g, noise=noise)
    cell = sc["occ"].float().argmax(1)
    a = sc["attr"][torch.arange(n), cell]
    return img, {"形状": a[:, 0].numpy(), "颜色": a[:, 1].numpy(), "大小": a[:, 2].numpy(), "纹理": a[:, 3].numpy(),
                 "位置": cell.numpy(), "颜色x形状": (a[:, 1] * 4 + a[:, 0]).numpy()}


# ---------------------------------------------------------------- 模型
def sincos_2d(dim, grid):
    def one(d, pos):
        omega = 1.0 / 100 ** (torch.arange(d // 2, dtype=torch.float32) / (d / 2))
        out = pos[:, None] * omega[None]
        return torch.cat([out.sin(), out.cos()], 1)
    ys, xs = torch.meshgrid(torch.arange(grid, dtype=torch.float32), torch.arange(grid, dtype=torch.float32), indexing="ij")
    return torch.cat([one(dim // 2, ys.reshape(-1)), one(dim // 2, xs.reshape(-1))], 1)


class Block(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x, mask=None):
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.n2(x))


def patchify(img):
    n = img.shape[0]
    return img.reshape(n, 3, G, P, G, P).permute(0, 2, 4, 1, 3, 5).reshape(n, L, 3 * P * P)


class ImageTower(nn.Module):
    def __init__(self, out=OUT):
        super().__init__()
        self.embed = nn.Linear(3 * P * P, DIM)
        self.register_buffer("pos", sincos_2d(DIM, G)[None])
        self.blocks = nn.ModuleList([Block(DIM) for _ in range(2)])
        self.norm = nn.LayerNorm(DIM)
        self.proj = nn.Linear(DIM, out, bias=False)

    def tokens(self, img):
        x = self.embed(patchify((img - 0.4) / 0.3)) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.norm(x)                           # (B, 16, 64)

    def forward(self, img):
        return self.proj(self.tokens(img).mean(1))    # (B, 32)，没有归一化


class TextTower(nn.Module):
    def __init__(self, bow=False):
        super().__init__()
        self.tok = nn.Embedding(len(VOCAB), DIM, padding_idx=0)
        nn.init.normal_(self.tok.weight, std=0.5)
        with torch.no_grad():
            self.tok.weight[0].zero_()
        self.bow = bow
        if bow:                                       # 词袋：词嵌入取平均再过一个小 MLP，结构上就不知道词序
            self.mlp = nn.Sequential(nn.Linear(DIM, 2 * DIM), nn.GELU(), nn.Linear(2 * DIM, DIM))
        else:
            self.pos = nn.Parameter(torch.zeros(1, MAXLEN, DIM))
            nn.init.normal_(self.pos, std=0.5)
            self.blocks = nn.ModuleList([Block(DIM) for _ in range(2)])
        self.norm = nn.LayerNorm(DIM)
        self.proj = nn.Linear(DIM, OUT, bias=False)

    def forward(self, ids):
        valid = ids != 0
        w = valid.float()[..., None]
        x = self.tok(ids)
        if self.bow:
            return self.proj(self.norm(self.mlp((x * w).sum(1) / w.sum(1))))
        x = x + self.pos
        for b in self.blocks:
            x = b(x, valid[:, None, None, :])
        return self.proj((self.norm(x) * w).sum(1) / w.sum(1))


class TwoTower(nn.Module):
    def __init__(self, bow=False, temp=0.07, learn_temp=True):
        super().__init__()
        self.image, self.text = ImageTower(), TextTower(bow)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / temp)), requires_grad=learn_temp)

    def scale(self):
        return self.logit_scale.exp().clamp(max=100.0)


def n_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


@torch.no_grad()
def img_emb(net, img):
    return net.image(img).numpy().astype(np.float64)


@torch.no_grad()
def txt_emb(net, sents):
    return net.text(encode(sents)).numpy().astype(np.float64)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


# ---------------------------------------------------------------- 训练（在子进程里跑）
def probe_pairs():
    """量模态间隙用的固定图文对：1000 个随机场景和按训练时的写法生成的句子。"""
    g = gen(4, 4)
    rng = np.random.default_rng(44)
    sc = sample_scenes(1000, g)
    return render(sc, g), [caption(sc, i, rng) for i in range(1000)]


def train_job(spec):
    """训练一个模型，返回它的参数（NumPy 数组）和训练记录。spec = (名字, 种子, 选项)。"""
    name, seed, kw = spec
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    g = gen(seed, 7)
    rng = np.random.default_rng(1000 + seed)
    hold = HOLD if kw.get("hold") else None
    sup = bool(kw.get("sup"))
    net = ImageTower(out=16) if sup else TwoTower(bow=bool(kw.get("bow")), temp=kw.get("temp", 0.07), learn_temp=kw.get("learn_temp", True))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.05)
    steps = kw.get("steps", STEPS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)
    log = {"step": [], "gap": [], "spread_img": [], "spread_txt": [], "scale": [], "loss": []}
    probe = probe_pairs() if kw.get("log") else None
    bind_set = (pair_set(600, 0, 55), single_test(N_TEST, 0, 99)) if kw.get("log_bind") else None
    bind_log = {"step": [], "两个颜色词对调": [], "两个目标的先后对调": [], "单目标的零样本精度": []}
    tgt = torch.arange(BATCH)
    t0 = time.perf_counter()
    for step in range(steps + 1):
        if probe is not None and step % 100 == 0:
            net.eval()
            gap = hw.modality_gap(img_emb(net, probe[0]), txt_emb(net, probe[1]))
            net.train()
            for key, val in zip(("step", "gap", "spread_img", "spread_txt", "scale"), (step, *gap, float(net.scale().detach()))):
                log[key].append(val)
        if bind_set is not None and step % 200 == 0:
            net.eval()
            acc, _ = binding_scores(net, *bind_set[0])
            img_, y_ = bind_set[1]
            bind_log["step"].append(step)
            bind_log["两个颜色词对调"].append(acc["两个颜色词对调"])
            bind_log["两个目标的先后对调"].append(acc["两个目标的先后对调"])
            bind_log["单目标的零样本精度"].append(zs_acc(net, img_emb(net, img_), y_["颜色x形状"], classifier(net, COMBOS)))
            net.train()
        if step == steps:
            break
        if sup:                                       # 有监督对照：只用单个目标的图，标签是 16 种“颜色 x 形状”
            sc = sample_scenes(BATCH, g, kinds=(1.0, 0.0, 0.0, 0.0))
            cell = sc["occ"].float().argmax(1)
            a = sc["attr"][torch.arange(BATCH), cell]
            loss = F.cross_entropy(net(render(sc, g)), a[:, 1] * 4 + a[:, 0])
        else:
            sc = sample_scenes(BATCH, g, hold=hold)
            img = render(sc, g)
            caps = [caption(sc, i, rng, kw.get("tex_words", True)) for i in range(BATCH)]
            u, v = F.normalize(net.image(img), dim=-1), F.normalize(net.text(encode(caps)), dim=-1)
            logits = net.scale() * u @ v.t()
            i2t = logits
            if kw.get("hard"):                        # 给每个“左右两个目标”的场景加一句词相同、意思不同的句子当负样本（只加在图找文这一侧）
                idx = (sc["kind"] == 2).nonzero().flatten().tolist()
                extra = F.normalize(net.text(encode([swapped_caption(sc, i, rng) for i in idx])), dim=-1)
                i2t = torch.cat([logits, net.scale() * u @ extra.t()], 1)
            loss = 0.5 * (F.cross_entropy(i2t, tgt) + F.cross_entropy(logits.t(), tgt))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step >= steps - 50:
            log["loss"].append(float(loss.detach()))
    log["bind"] = bind_log
    return {"state": {k: v.numpy() for k, v in net.state_dict().items()}, "log": log, "seconds": time.perf_counter() - t0}


JOBS = ([("main", s, {"log": True}) for s in SEEDS] + [("hard", s, {"hard": True}) for s in SEEDS] + [("hold", s, {"hold": True}) for s in (0, 1)]
        + [("notex", s, {"tex_words": False}) for s in (0, 1)] + [("bow", s, {"bow": True}) for s in (0, 1)]
        + [(f"temp{t}", 0, {"temp": t, "learn_temp": False, "log": True}) for t in (0.01, 0.3, 1.0)] + [("sup", s, {"sup": True}) for s in (0, 1)]
        + [("long", s, {"steps": LONG_STEPS, "log_bind": True}) for s in SEEDS] + [("longhard", s, {"steps": LONG_STEPS, "hard": True, "log_bind": True}) for s in SEEDS])
MODELS: dict = {}
LOGS: dict = {}


def train_all():
    workers = int(os.environ.get("CH28_WORKERS", "3"))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        results = list(ex.map(train_job, JOBS))
    for (name, seed, kw), res in zip(JOBS, results):
        net = ImageTower(out=16) if kw.get("sup") else TwoTower(bow=bool(kw.get("bow")), temp=kw.get("temp", 0.07), learn_temp=kw.get("learn_temp", True))
        net.load_state_dict({k: torch.from_numpy(np.array(v)) for k, v in res["state"].items()})
        MODELS[(name, seed)] = net.eval()
        LOGS[(name, seed)] = dict(res["log"], seconds=res["seconds"])
    TIMES["train_wall"] = time.perf_counter() - t0


def M(name, seed=0):
    return MODELS[(name, seed)]


def rand_model(seed):
    torch.manual_seed(100 + seed)
    return TwoTower().eval()


# ---------------------------------------------------------------- 评价用的小工具
def scale_of(net):
    return float(net.scale().detach())


def classifier(net, phrases, templates=("一个 {}",)):
    """把每个类别的短语套进若干模板，过文本塔，交给作业的 build_zeroshot_classifier。返回 (C, D)。"""
    t = np.stack([txt_emb(net, [tpl.format(p) for p in phrases]) for tpl in templates], 1)
    return hw.build_zeroshot_classifier(t)


def zs_acc(net, feats, y, clf):
    return float((hw.zeroshot_predict(feats, clf, scale_of(net))[1] == y).mean())


TASKS = {                                             # 任务名：(类别短语, 模板)
    "形状": (SHAPES, "一个 {}"),
    "颜色": ([f"{c} 图形" for c in COLORS], "一个 {}"),
    "大小": ([f"{z} 图形" for z in SIZES], "一个 {}"),
    "纹理": ([f"{t} 图形" for t in TEXS], "一个 {}"),
    "位置": (CELLS, "一个 图形 在 {}"),
    "数量": (COUNTS, "{} 图形"),
    "颜色x形状": (COMBOS, "一个 {}"),
}


def fit_head(x, y, w_init, lam, scale, steps=100, lr=0.05):
    """在归一化的图像向量上训练一个线性分类头：交叉熵加 lam 乘“离初始权重的距离的平方”，全批量 Adam。
    w_init 全 0、lam 很小就是线性探测；w_init 取零样本分类器就是“从零样本分类器出发的微调”。"""
    x = torch.from_numpy(unit(x)).float()
    y = torch.from_numpy(np.asarray(y)).long()
    w0 = torch.from_numpy(np.asarray(w_init)).float()
    w = w0.clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(scale * x @ w.t(), y) + lam * ((w - w0) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach().numpy().astype(np.float64)


def head_acc(w, feats, y, scale):
    return float((hw.zeroshot_predict(feats, w, scale)[1] == y).mean())


def k_shot(y_pool, classes, k, g):
    """从带标签的池子里每类随机取 k 个，返回下标。"""
    y_pool = torch.from_numpy(np.asarray(y_pool))
    idx = []
    for c in range(classes):
        cand = (y_pool == c).nonzero().flatten()
        idx.append(cand[torch.randperm(len(cand), generator=g)[:k]])
    return torch.cat(idx).numpy()


def stats(vals):
    return {"平均": r3(np.mean(vals)), "各次": [r3(v) for v in vals]}


def dots(ax, x, vals, color="black", size=9):
    ax.scatter(np.full(len(vals), x) + np.linspace(-0.06, 0.06, len(vals)) if len(vals) > 1 else [x], vals, s=size, color=color, zorder=5, linewidths=0)


# ---------------------------------------------------------------- 图 1：写一句话就是一个分类器
def fig_why():
    img, y = single_test(N_TEST, 0, 99)
    tr_img, ytr = single_test(2000, 5, 99)
    g = gen(3, 3)
    sc = sample_scenes(900, g, kinds=(0.34, 0.66, 0.0, 0.0))
    cnt_img, y_cnt = render(sc, g), (sc["occ"].sum(1) - 1).numpy()
    sc = sample_scenes(2000, g, kinds=(0.34, 0.66, 0.0, 0.0))
    cnt_tr, ytr_cnt = render(sc, g), (sc["occ"].sum(1) - 1).numpy()
    pool_img, pool_y = single_test(4000, 9, 99)
    ks = [1, 2, 4, 8, 16, 32, 64]
    zs = {t: [] for t in TASKS}
    lin = {t: [] for t in TASKS}
    by_size = {"大": [], "小": []}
    curve = {k: [] for k in ks}
    for s in SEEDS:
        net = M("main", s)
        sc_ = scale_of(net)
        f, ftr, fc, fctr = (img_emb(net, x) for x in (img, tr_img, cnt_img, cnt_tr))
        for task, (phr, tpl) in TASKS.items():
            a, b, ya, yb = (fc, fctr, y_cnt, ytr_cnt) if task == "数量" else (f, ftr, y[task], ytr[task])
            zs[task].append(zs_acc(net, a, ya, classifier(net, phr, (tpl,))))
            lin[task].append(head_acc(fit_head(b, yb, np.zeros((len(phr), OUT)), 0.01, sc_), a, ya, sc_))
        pred = hw.zeroshot_predict(f, classifier(net, SHAPES), sc_)[1]
        for z, name in enumerate(SIZES):
            by_size[name].append(float((pred[y["大小"] == z] == y["形状"][y["大小"] == z]).mean()))
        fpool = img_emb(net, pool_img)
        for k in ks:
            for rep in range(3):
                idx = k_shot(pool_y["颜色x形状"], 16, k, gen(s, k, rep, 11))
                curve[k].append(head_acc(fit_head(fpool[idx], pool_y["颜色x形状"][idx], np.zeros((16, OUT)), 0.01, sc_), f, y["颜色x形状"], sc_))
    sup = {"颜色x形状": [], "形状": [], "颜色": [], "形状，大目标": [], "形状，小目标": []}
    for s in (0, 1):
        with torch.no_grad():
            pred = M("sup", s)(img).argmax(1).numpy()
        sup["颜色x形状"].append(float((pred == y["颜色x形状"]).mean()))
        sup["形状"].append(float((pred % 4 == y["形状"]).mean()))
        sup["颜色"].append(float((pred // 4 == y["颜色"]).mean()))
        for z, name in enumerate(SIZES):
            sup[f"形状，{name}目标"].append(float((pred[y["大小"] == z] % 4 == y["形状"][y["大小"] == z]).mean()))
    chance = {t: 1 / len(TASKS[t][0]) for t in TASKS}
    NUM["零样本分类和有监督线性头"] = {
        "说明": DATA_NOTE + "。" + MODEL_NOTE + f"。双塔模型 3 个种子。测试图 {N_TEST} 张，都只有一个目标（“数量”一项用 900 张含 1 到 3 个相同目标的图）。"
              "零样本：把每个类别写成一句话（模板见“各任务的提示词”），过文本塔得到分类器，不用任何带标签的图。"
              "有监督线性头：图像塔冻结，在 2000 张带标签的图的图像向量上训练一个线性分类头（交叉熵，全批量 Adam 100 步，L2 系数 0.01）。"
              f"有监督对照：同样结构的图像塔后面直接接 16 类输出，用“颜色 x 形状”的标签从头训练 {STEPS} 步（每步 {BATCH} 张新图，标签不限量），2 个种子",
        "各任务的提示词": {t: [TASKS[t][1].format(p) for p in TASKS[t][0]] for t in TASKS},
        "随机猜": {t: r3(chance[t]) for t in TASKS},
        "零样本": {t: stats(zs[t]) for t in TASKS},
        "有监督线性头（2000 张标签）": {t: stats(lin[t]) for t in TASKS},
        "零样本分形状，按目标大小分开": {k: stats(v) for k, v in by_size.items()},
        "有监督对照（端到端训练，标签不限量）": {k: stats(v) for k, v in sup.items()},
        "标签数和线性头": {
            "说明": "“颜色 x 形状”16 类。从 4000 张带标签的图里每类随机取 k 张训练线性头，3 个种子的模型各取 3 次，共 9 次的平均",
            "每类标签数": ks, "带标签的图总数": [16 * k for k in ks],
            "线性头": [r3(np.mean(curve[k])) for k in ks], "各次的最小和最大": [[r3(min(curve[k])), r3(max(curve[k]))] for k in ks],
            "零样本（不用标签）": r3(np.mean(zs["颜色x形状"])),
        },
    }

    fig = plt.figure(figsize=(16.5, 4.9), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 1.0])
    sub = gs[0].subgridspec(2, 4)
    g = gen(28)
    sc = sample_scenes(200, g)
    rng = np.random.default_rng(28)
    pics = render(sc, g)
    order = [int((sc["kind"] == k).nonzero().flatten()[j]) for k, j in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0))]
    for n, i in enumerate(order):
        a = fig.add_subplot(sub[n // 4, n % 4])
        a.imshow(pics[i].permute(1, 2, 0).clamp(0, 1).numpy(), interpolation="nearest")
        words = caption(sc, i, rng)
        half = (len(words) + 1) // 2 if len(words) > 4 else len(words)
        a.set_title(" ".join(words[:half]) + ("\n" + " ".join(words[half:]) if half < len(words) else "\n"), fontsize=8.5, fontweight="normal")
        S.image_axes(a)
    ax = fig.add_subplot(gs[1])
    names = list(TASKS)
    x = np.arange(len(names))
    ax.bar(x - 0.2, [np.mean(zs[t]) for t in names], 0.38, color=S.BLUE, label="零样本（一句话当分类器，0 张标签）")
    ax.bar(x + 0.2, [np.mean(lin[t]) for t in names], 0.38, color=S.ORANGE, label="有监督线性头（2000 张标签）")
    for i, t in enumerate(names):
        dots(ax, i - 0.2, zs[t])
        dots(ax, i + 0.2, lin[t])
        ax.plot([i - 0.42, i + 0.42], [chance[t]] * 2, color=S.RED, lw=1.3, ls="--", label="随机猜" if i == 0 else None)
        ax.text(i - 0.2, np.mean(zs[t]) + 0.015, f"{np.mean(zs[t]):.2f}", ha="center", fontsize=7.5)
        ax.text(i + 0.2, np.mean(lin[t]) + 0.015, f"{np.mean(lin[t]):.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, [f"{t}\n{len(TASKS[t][0])} 类" for t in names], fontsize=8.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("测试精度")
    ax.set_title("七个任务上，零样本和 2000 张标签的线性头差不多")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    S.clean(ax)
    ax.grid(axis="x", visible=False)
    ax = fig.add_subplot(gs[2])
    ax.plot(ks, [np.mean(curve[k]) for k in ks], "o-", color=S.ORANGE, lw=1.8, ms=5.5, label="有监督线性头")
    ax.fill_between(ks, [min(curve[k]) for k in ks], [max(curve[k]) for k in ks], color=S.ORANGE, alpha=0.15, lw=0)
    ax.axhline(np.mean(zs["颜色x形状"]), color=S.BLUE, lw=1.8, label="零样本（不用标签）")
    ax.axhline(1 / 16, color=S.RED, lw=1.2, ls="--", label="随机猜")
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.minorticks_off()
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("每类带标签的图数（“颜色 x 形状”16 类）")
    ax.set_ylabel("测试精度")
    ax.set_title("线性头每类要几十张标签才追上零样本")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    S.clean(ax)
    S.save(fig, CH, "fig01_一句话当分类器")


# ---------------------------------------------------------------- 图 2：提示词模板
TEMPLATES = {
    "只写类别名": "{}",
    "一个 X": "一个 {}",
    "图 里 有 一个 X": "图 里 有 一个 {}",
    "这 是 一个 X": "这 是 一个 {}",
    "一张 X 的 照片（三个词没训练过）": "一张 {} 的 照片",
    "三个 X（数量说错了）": "三个 {}",
}
SEEN_TPL = ("一个 X", "图 里 有 一个 X", "这 是 一个 X")


def fig_templates():
    img, y = single_test(N_TEST, 0, 99)
    tasks = {"形状": SHAPES, "颜色x形状": COMBOS}
    rows = list(TEMPLATES) + ["三个训练时见过的模板取平均", "六个模板全部取平均"]
    res = {t: {r: [] for r in rows} for t in tasks}
    extra = {"把四种颜色各写一遍再平均": [], "把颜色、大小、纹理的 16 种写法各写一遍再平均": []}
    conf = {r: [] for r in rows}
    for s in SEEDS:
        net = M("main", s)
        f = img_emb(net, img)
        for t, phr in tasks.items():
            for r, tpl in TEMPLATES.items():
                res[t][r].append(zs_acc(net, f, y[t], classifier(net, phr, (tpl,))))
            res[t][rows[-2]].append(zs_acc(net, f, y[t], classifier(net, phr, tuple(TEMPLATES[r] for r in SEEN_TPL))))
            res[t][rows[-1]].append(zs_acc(net, f, y[t], classifier(net, phr, tuple(TEMPLATES.values()))))
        extra["把四种颜色各写一遍再平均"].append(zs_acc(net, f, y["形状"], classifier(net, SHAPES, tuple(f"一个 {c} {{}}" for c in COLORS))))
        extra["把颜色、大小、纹理的 16 种写法各写一遍再平均"].append(
            zs_acc(net, f, y["形状"], classifier(net, SHAPES, tuple(f"一个 {z} {t} {c} {{}}" for z in SIZES for t in TEXS for c in COLORS))))
        for r in rows:                                # 同一组预测，各模板下“最大概率”的平均，说明模板也改变置信度
            tpls = tuple(TEMPLATES[k] for k in SEEN_TPL) if r == rows[-2] else tuple(TEMPLATES.values()) if r == rows[-1] else (TEMPLATES[r],)
            conf[r].append(float(hw.zeroshot_predict(f, classifier(net, SHAPES, tpls), scale_of(net))[0].max(1).mean()))
    best_single = {t: max(np.mean(res[t][r]) for r in SEEN_TPL) for t in tasks}
    NUM["提示词模板"] = {
        "说明": "同图 1 的 3 个双塔模型和 1000 张单目标测试图，只改提示词的写法。X 是类别名：形状任务是“方块”这样的一个词，“颜色 x 形状”任务是“红色 方块”这样的两个词。"
              "训练句子里出现过的写法是“一个 X”“图 里 有 一个 X”“这 是 一个 X”；“只写类别名”这种不带数量词的单目标句子训练时没有出现过；"
              "“一张 X 的 照片”里“一张”“的”“照片”三个词的词嵌入从来没有被训练过；“三个 X”是见过的写法，但测试图里只有一个目标。"
              "多个模板取平均用作业的 build_zeroshot_classifier：每个模板的文本向量先各自归一化，再平均，再归一化",
        **{t: {r: stats(v) for r, v in res[t].items()} for t in tasks},
        "形状任务的另外两种平均": {k: stats(v) for k, v in extra.items()},
        "三个见过的模板里单个最好的（平均精度）": {t: r3(v) for t, v in best_single.items()},
        "六个单模板里最好和最差的差（平均精度）": {t: r3(max(np.mean(res[t][r]) for r in TEMPLATES) - min(np.mean(res[t][r]) for r in TEMPLATES)) for t in tasks},
        "形状任务上最大概率的平均（温度用模型学到的）": {r: r3(np.mean(v)) for r, v in conf.items()},
        "模型学到的相似度放大倍数": [r2(scale_of(M("main", s))) for s in SEEDS],
    }
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6), layout="constrained", sharey=True)
    colors = [S.MUTED, S.BLUE, S.BLUE, S.BLUE, S.RED, S.RED, S.GREEN, S.GREEN]
    for ax, (t, title) in zip(axes, (("形状", "分形状（4 类）"), ("颜色x形状", "分“颜色 x 形状”（16 类）"))):
        ypos = np.arange(len(rows))[::-1]
        vals = [np.mean(res[t][r]) for r in rows]
        ax.barh(ypos, vals, 0.62, color=colors, alpha=0.9)
        for yy, r, v in zip(ypos, rows, vals):
            ax.scatter(res[t][r], np.full(len(SEEDS), yy) + np.linspace(-0.15, 0.15, len(SEEDS)), s=10, color="black", zorder=5, linewidths=0)
            ax.text(1.005, yy, f"{v:.3f}", va="center", fontsize=8.5)
        ax.set_yticks(ypos, rows, fontsize=9)
        ax.set_xlim(0.5, 1.06)
        ax.set_xlabel("零样本精度（黑点是 3 个种子）")
        ax.set_title(title)
        S.clean(ax)
        ax.grid(axis="y", visible=False)
    fig.suptitle("写法不对的模板掉几个点到十几个点；多模板平均接近最好的单个模板，把坏模板的影响抹平", fontsize=11.5, fontweight="bold")
    S.save(fig, CH, "fig02_提示词模板")


# ---------------------------------------------------------------- 图 3：组合泛化、属性绑定、词序
def pair_set(n, *seed):
    g = gen(*seed)
    sc = sample_scenes(n, g, kinds=(0.0, 0.0, 1.0, 0.0))
    keep = sc["kind"] == 2
    sc = {k: v[keep] for k, v in sc.items()}
    return sc, render(sc, g)


TIE = 1e-5                                            # 两句话的余弦相差不到这个数就算相同（float32 的舍入误差在 1e-7 量级）


def binding_scores(net, sc, img):
    """每张“左右两个目标”的图，在正确的句子和三种改动过的句子之间二选一。相似度相同记半对。"""
    u = unit(img_emb(net, img))
    caps = {"正确": [], "两个颜色词对调": [], "两个目标的先后对调": [], "只把左边换成右边": []}
    for i in range(len(u)):
        a, b = pair_words(sc, i)
        caps["正确"].append([a[0], a[1], "在", b[0], b[1], "左边"])
        caps["两个颜色词对调"].append([b[0], a[1], "在", a[0], b[1], "左边"])
        caps["两个目标的先后对调"].append([b[0], b[1], "在", a[0], a[1], "左边"])
        caps["只把左边换成右边"].append([a[0], a[1], "在", b[0], b[1], "右边"])
    cos = {k: (u * unit(txt_emb(net, v))).sum(1) for k, v in caps.items()}
    out = {}
    for k in list(caps)[1:]:
        d = cos["正确"] - cos[k]
        out[k] = float(((d > TIE) + 0.5 * (np.abs(d) <= TIE)).mean())
    return out, {k: float(v.mean()) for k, v in cos.items()}


def fig_compose():
    # A：训练时拿掉 4 种（颜色，形状）组合
    img_h, yh = single_test(600, 1, 99, hold=HOLD, only_hold=True)
    img_s, ys = single_test(1200, 2, 99, hold=HOLD)
    tr_img, ytr = single_test(2400, 5, 98, hold=HOLD)
    held_ids = np.array([c * 4 + s for c, s in HOLD])
    comp = {}
    for name in ("hold", "main"):
        rows = {k: [] for k in ("零样本：见过的组合", "零样本：没见过的组合", "没见过的组合：颜色判对", "没见过的组合：形状判对", "没见过的组合：被判成某个见过的组合",
                                "没见过的组合：正确答案排进前 3", "没见过的组合：前 4 名全是同一种颜色", "只问形状（4 类）：见过的组合", "只问形状（4 类）：没见过的组合",
                                "平铺的 16 类线性头：见过的组合", "平铺的 16 类线性头：没见过的组合", "颜色、形状各一个线性头：见过的组合", "颜色、形状各一个线性头：没见过的组合",
                                "文本向量和同一组合图像向量平均的余弦：见过的组合", "文本向量和同一组合图像向量平均的余弦：没见过的组合")}
        for s in (0, 1):
            net = M(name, s)
            sc_ = scale_of(net)
            fh, fs, ftr = (img_emb(net, x) for x in (img_h, img_s, tr_img))
            clf = classifier(net, COMBOS)
            ph, pred_h = hw.zeroshot_predict(fh, clf, sc_)
            pred_s = hw.zeroshot_predict(fs, clf, sc_)[1]
            rows["零样本：见过的组合"].append(float((pred_s == ys["颜色x形状"]).mean()))
            rows["零样本：没见过的组合"].append(float((pred_h == yh["颜色x形状"]).mean()))
            rows["没见过的组合：颜色判对"].append(float((pred_h // 4 == yh["颜色"]).mean()))
            rows["没见过的组合：形状判对"].append(float((pred_h % 4 == yh["形状"]).mean()))
            rows["没见过的组合：被判成某个见过的组合"].append(float((~np.isin(pred_h, held_ids)).mean()))
            rows["没见过的组合：正确答案排进前 3"].append(float((np.argsort(-ph, axis=1)[:, :3] == yh["颜色x形状"][:, None]).any(1).mean()))
            rows["没见过的组合：前 4 名全是同一种颜色"].append(float((np.argsort(-ph, axis=1)[:, :4] // 4 == yh["颜色"][:, None]).all(1).mean()))
            shape_clf = classifier(net, SHAPES)
            rows["只问形状（4 类）：见过的组合"].append(zs_acc(net, fs, ys["形状"], shape_clf))
            rows["只问形状（4 类）：没见过的组合"].append(zs_acc(net, fh, yh["形状"], shape_clf))
            flat = fit_head(ftr, ytr["颜色x形状"], np.zeros((16, OUT)), 0.01, sc_)
            rows["平铺的 16 类线性头：见过的组合"].append(head_acc(flat, fs, ys["颜色x形状"], sc_))
            rows["平铺的 16 类线性头：没见过的组合"].append(head_acc(flat, fh, yh["颜色x形状"], sc_))
            wc = fit_head(ftr, ytr["颜色"], np.zeros((4, OUT)), 0.01, sc_)
            ws = fit_head(ftr, ytr["形状"], np.zeros((4, OUT)), 0.01, sc_)
            for key, f, yy in (("见过的组合", fs, ys), ("没见过的组合", fh, yh)):
                both = hw.zeroshot_predict(f, wc, sc_)[1] * 4 + hw.zeroshot_predict(f, ws, sc_)[1]
                rows[f"颜色、形状各一个线性头：{key}"].append(float((both == yy["颜色x形状"]).mean()))
            for key, f, yy in (("见过的组合", fs, ys), ("没见过的组合", fh, yh)):
                ids = np.unique(yy["颜色x形状"])
                proto = unit(np.stack([unit(f)[yy["颜色x形状"] == k].mean(0) for k in ids]))
                rows[f"文本向量和同一组合图像向量平均的余弦：{key}"].append(float((proto * clf[ids]).sum(1).mean()))
        comp[name] = rows
    NUM["组合泛化"] = {
        "说明": f"训练时把 4 种（颜色，形状）组合从所有场景里拿掉：{'、'.join(HOLD_NAMES)}。每种颜色、每种形状仍然各在另外 3 种组合里出现。2 个种子，其余设置同图 1。"
              "测试：1200 张“见过的组合”的单目标图，600 张“没见过的组合”的单目标图，都在 16 个“一个 颜色 形状”的句子里选。"
              "两种线性头都在 2400 张带标签的图上训练，这些图只含见过的 12 种组合：“平铺的”是一个 16 类的头；“各一个”是颜色 4 类、形状 4 类两个头，预测时拼起来。"
              "对照是图 1 的模型（训练时 16 种组合都有），用同样的测试图。随机猜：16 选 1 是 0.062，形状 4 选 1 是 0.25",
        "训练时拿掉 4 种组合": {k: stats(v) for k, v in comp["hold"].items()},
        "对照：训练时 16 种组合都有": {k: stats(v) for k, v in comp["main"].items()},
    }

    # B：属性绑定
    sc, pimg = pair_set(600, 0, 55)
    img, y = single_test(N_TEST, 0, 99)
    groups = [("词袋文本塔，1200 步", "bow", (0, 1)), ("Transformer 文本塔，1200 步（图 1 的模型）", "main", SEEDS), ("加难负样本，1200 步", "hard", SEEDS),
              ("Transformer 文本塔，2400 步", "long", SEEDS), ("加难负样本，2400 步", "longhard", SEEDS)]
    bind = {}
    for label, name, seeds in groups:
        rows = {"两个颜色词对调": [], "两个目标的先后对调": [], "只把左边换成右边": [], "单目标图上“颜色 x 形状”的零样本精度": [],
                "正确句子的平均余弦": [], "颜色词对调后的平均余弦": []}
        for s in seeds:
            net = M(name, s)
            acc, cos = binding_scores(net, sc, pimg)
            for k, v in acc.items():
                rows[k].append(v)
            rows["单目标图上“颜色 x 形状”的零样本精度"].append(zs_acc(net, img_emb(net, img), y["颜色x形状"], classifier(net, COMBOS)))
            rows["正确句子的平均余弦"].append(cos["正确"])
            rows["颜色词对调后的平均余弦"].append(cos["两个颜色词对调"])
        bind[label] = rows
    # 一个训练批里，“词相同、意思不同”的句子自然出现得有多频繁
    g = gen(0, 91)
    met, pairs, hard_pairs, neg_pairs = 0, 0, 0, 0
    for _ in range(300):
        bsc = sample_scenes(BATCH, g)
        idx = (bsc["kind"] == 2).nonzero().flatten().tolist()
        words = [pair_words(bsc, i) for i in idx]
        bags = [tuple(sorted(a + b)) for a, b in words]
        for i in range(len(idx)):
            same = [j for j in range(len(idx)) if j != i and bags[j] == bags[i] and words[j] != words[i]]
            met += bool(same)
            hard_pairs += len(same)
        pairs += len(idx)
        neg_pairs += BATCH * (BATCH - 1)
    traj = {name: {"步数": LOGS[(name, 0)]["bind"]["step"], **{key: [[r3(v) for v in LOGS[(name, s)]["bind"][key]] for s in SEEDS] for key in ("两个颜色词对调", "两个目标的先后对调", "单目标的零样本精度")}}
            for name in ("long", "longhard")}
    NUM["属性绑定"] = {
        "说明": f"{len(pimg)} 张“左右两个不同目标”的图（颜色、形状都不同）。每张图在正确的句子（例如“红色 方块 在 蓝色 圆 左边”）和一句改动过的句子之间二选一，看图像向量和哪一句更像，随机猜 0.5，"
              f"两句话的余弦相差不到 {TIE} 记半对。“两个颜色词对调”是“蓝色 方块 在 红色 圆 左边”，“两个目标的先后对调”是“蓝色 圆 在 红色 方块 左边”，这两种句子用的词和正确句子完全相同；“只把左边换成右边”改了一个词。"
              "词袋文本塔：词嵌入取平均后过一个小 MLP，结构上不含词序，2 个种子。Transformer 文本塔 1200 步就是图 1 的模型，3 个种子。"
              "加难负样本：训练时给每个“左右两个目标”的场景额外加一句词相同、意思不同的句子（颜色词对调或者先后对调，各一半）当作图找文一侧的负样本，3 个种子，其余不变。"
              f"2400 步：其余设置不变，只把训练步数加倍（学习率同样先升后降，所以它的前 1200 步和 1200 步的模型不是同一条轨迹），3 个种子",
        **{label: {k: stats(v) for k, v in rows.items()} for label, rows in bind.items()},
        "训练过程（2400 步的两组，每 200 步量一次，每个种子一行）": {"不加难负样本": traj["long"], "加难负样本": traj["longhard"]},
        "一个批里自然出现的难负样本": {
            "说明": f"按训练时的场景分布抽 300 个批（每批 {BATCH} 个场景）统计。对一个“左右两个目标”的场景，如果同一批里另有一个场景用了同样的两个颜色词和两个形状词、但不是同一种摆法，那个场景的句子对它来说就是“词相同、意思不同”的负样本",
            "每批里“左右两个目标”的场景个数（平均）": r2(pairs / 300),
            "这些场景里，同批至少遇到一句这种句子的比例": r3(met / pairs),
            "这种（图，句子）对占一个批里全部负样本对的比例": float(f"{hard_pairs / neg_pairs:.2g}"),
            "加难负样本之后每个这样的场景固定多出的句子数": 1,
        },
    }

    # C：检索，以及把句子里的词打乱
    g = gen(0, 71)
    rsc = sample_scenes(500, g, kinds=(0.5, 0.2, 0.3, 0.0))
    rimg = render(rsc, g)
    caps = [full_caption(rsc, i) for i in range(500)]
    keys = np.array([" ".join(c) for c in caps])
    pos = keys[:, None] == keys[None, :]
    rng = np.random.default_rng(5)
    shuf = [list(rng.permutation(c)) for c in caps]
    kinds = rsc["kind"].numpy()
    kind_names = {0: "单个目标", 1: "多个相同目标", 2: "左右两个不同目标"}
    rgroups = [("1200 步（图 1 的模型）", "main"), ("2400 步", "long"), ("2400 步，加难负样本", "longhard")]
    retr = {label: {k: [] for k in ("图找文 R@1", "图找文 R@5", "文找图 R@1", "文找图 R@5", "打乱词序后：图找文 R@1", "打乱词序后：图找文 R@5", "原句和打乱后的句子的余弦", "只认对角线的图找文 R@1")}
            for label, _ in rgroups}
    by_kind = {label: {name: {"原句": [], "打乱": []} for name in kind_names.values()} for label, _ in rgroups}
    for label, name in rgroups:
        for s in SEEDS:
            net = M(name, s)
            u, v, vs = unit(img_emb(net, rimg)), unit(txt_emb(net, caps)), unit(txt_emb(net, shuf))
            sim, sims = u @ v.T, u @ vs.T
            for k in (1, 5):
                retr[label][f"图找文 R@{k}"].append(hw.recall_at_k(sim, pos, k))
                retr[label][f"文找图 R@{k}"].append(hw.recall_at_k(sim.T, pos.T, k))
                retr[label][f"打乱词序后：图找文 R@{k}"].append(hw.recall_at_k(sims, pos, k))
            retr[label]["原句和打乱后的句子的余弦"].append(float((v * vs).sum(1).mean()))
            retr[label]["只认对角线的图找文 R@1"].append(hw.recall_at_k(sim, np.eye(500, dtype=bool), 1))
            for kd, kname in kind_names.items():
                by_kind[label][kname]["原句"].append(hw.recall_at_k(sim[kinds == kd], pos[kinds == kd], 1))
                by_kind[label][kname]["打乱"].append(hw.recall_at_k(sims[kinds == kd], pos[kinds == kd], 1))
    NUM["图文检索和打乱词序"] = {
        "说明": "500 个场景（单个目标 0.5、多个相同目标 0.2、左右两个不同目标 0.3）。每个场景配一句把全部属性都写出来的句子，例如“一个 大 条纹 红色 方块 在 左上”“两个 小 实心 蓝色 圆”“红色 方块 在 蓝色 圆 左边”。"
              "两张图的句子完全相同时互为正确答案。Recall@K 用作业的 recall_at_k：前 K 个里有一个正确答案就算命中。三组模型各 3 个种子（同“属性绑定”）。"
              "打乱词序：把每句话里的词随机排列（固定种子），图像不动，正确答案的定义不变。“只认对角线”是一种常见的写错：只把第 i 句当成第 i 张图的正确答案，句子相同的其他图不算",
        "场景数": 500, "各类场景的个数": {name: int((kinds == kd).sum()) for kd, name in kind_names.items()},
        "句子不止对应一张图的场景比例": r3((pos.sum(1) > 1).mean()),
        **{label: {k: stats(v) for k, v in d.items()} for label, d in retr.items()},
        "按场景分的图找文 R@1": {label: {kname: {k: stats(v) for k, v in d.items()} for kname, d in dd.items()} for label, dd in by_kind.items()},
    }

    fig, ax = plt.subplots(1, 4, figsize=(19.5, 4.9), gridspec_kw={"width_ratios": [1.2, 0.7, 1.35, 1.15]}, layout="constrained")
    series = [("零样本，训练时缺这 4 种组合", "hold", "零样本", S.BLUE), ("平铺的 16 类线性头（标签只有见过的组合）", "hold", "平铺的 16 类线性头", S.ORANGE),
              ("颜色、形状各一个线性头", "hold", "颜色、形状各一个线性头", S.GREEN), ("零样本，训练时 16 种组合都有（对照）", "main", "零样本", S.MUTED)]
    for i, (label, name, key, color) in enumerate(series):
        vals = [comp[name][f"{key}：见过的组合"], comp[name][f"{key}：没见过的组合"]]
        xs = np.arange(2) + (i - 1.5) * 0.2
        ax[0].bar(xs, [np.mean(v) for v in vals], 0.19, color=color, label=label)
        for xx, v in zip(xs, vals):
            dots(ax[0], xx, v)
            ax[0].text(xx, max(v) + 0.02, f"{np.mean(v):.2f}", ha="center", fontsize=7.5)
    ax[0].axhline(1 / 16, color=S.RED, lw=1.1, ls="--")
    ax[0].set_xticks([0, 1], ["见过的 12 种组合", "没见过的 4 种组合"])
    ax[0].set_ylim(0, 1.1)
    ax[0].set_ylabel("16 选 1 的精度")
    ax[0].set_title("没见过的组合：零样本和两种线性头都很低", fontsize=10.5)
    ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
    keys3 = ["没见过的组合：颜色判对", "没见过的组合：形状判对", "没见过的组合：被判成某个见过的组合"]
    for i, k in enumerate(keys3):
        v = comp["hold"][k]
        ax[1].bar(i, np.mean(v), 0.6, color=[S.BLUE, S.RED, S.PURPLE][i])
        dots(ax[1], i, v)
        ax[1].text(i, max(max(v) + 0.02, 0.27) if i == 1 else max(v) + 0.02, f"{np.mean(v):.2f}", ha="center", fontsize=8.5)
    ax[1].plot([0.6, 1.4], [0.25, 0.25], color=S.RED, lw=1.1, ls="--")
    ax[1].set_xticks(range(3), ["颜色\n判对", "形状\n判对", "判成见过\n的组合"], fontsize=9)
    ax[1].set_ylim(0, 1.1)
    ax[1].set_title("错在哪：颜色对，形状错", fontsize=10.5)
    steps = traj["long"]["步数"]
    for s in range(len(SEEDS)):
        ax[2].plot(steps, traj["long"]["两个颜色词对调"][s], "-", color=S.BLUE, lw=1.5, alpha=0.9, label="二选一：不加难负样本（每个种子一条）" if s == 0 else None)
        ax[2].plot(steps, traj["longhard"]["两个颜色词对调"][s], "--", color=S.GREEN, lw=1.5, alpha=0.9, label="二选一：加难负样本（每个种子一条）" if s == 0 else None)
    ax[2].plot(steps, np.mean(traj["long"]["单目标的零样本精度"], 0), "o-", color=S.MUTED, lw=1.6, ms=4, label="单目标“颜色 x 形状”的零样本精度（3 个种子平均）")
    ax[2].axhline(0.5, color=S.RED, lw=1.1, ls=":")
    ax[2].text(steps[-1], 0.455, "二选一的随机猜", ha="right", fontsize=8, color=S.RED)
    ax[2].set_ylim(0, 1.05)
    ax[2].set_xlabel("训练步数（一共 2400 步的那几次训练）")
    ax[2].set_ylabel("精度")
    ax[2].set_title("和“两个颜色词对调”的句子二选一：\n单目标的精度先上去，绑定晚得多，早晚看种子", fontsize=10.5)
    ax[2].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    pair_name = "左右两个不同目标"
    bars = [("单个目标\n1200 步", by_kind["1200 步（图 1 的模型）"]["单个目标"]), ("左右两个目标\n1200 步", by_kind["1200 步（图 1 的模型）"][pair_name]),
            ("左右两个目标\n2400 步", by_kind["2400 步"][pair_name]), ("左右两个目标\n2400 步\n加难负样本", by_kind["2400 步，加难负样本"][pair_name])]
    for i, (_, d) in enumerate(bars):
        for off, key, color in ((-0.2, "原句", S.BLUE), (0.2, "打乱", S.ORANGE)):
            ax[3].bar(i + off, np.mean(d[key]), 0.38, color=color, label={"原句": "原句", "打乱": "词序随机打乱"}[key] if i == 0 else None)
            dots(ax[3], i + off, d[key])
            ax[3].text(i + off, max(d[key]) + 0.02, f"{np.mean(d[key]):.2f}", ha="center", fontsize=8)
    ax[3].set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=8.5)
    ax[3].set_ylim(0, 1.12)
    ax[3].set_ylabel("图找文 R@1")
    ax[3].set_title("还像词袋的模型，打乱词序没有代价；\n学会绑定之后才有", fontsize=10.5)
    ax[3].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    for a in ax:
        S.clean(a)
        a.grid(axis="x", visible=False)
    S.save(fig, CH, "fig03_组合泛化与词袋")


# ---------------------------------------------------------------- 图 4：模态间隙
def gap_row(net, probe, test):
    u, v = img_emb(net, probe[0]), txt_emb(net, probe[1])
    gap, su, sv = hw.modality_gap(u, v)
    un, vn = unit(u), unit(v)
    img, y = test
    f, clf = img_emb(net, img), classifier(net, COMBOS)
    acc = zs_acc(net, f, y["颜色x形状"], clf)
    centered = hw.build_zeroshot_classifier(clf - vn.mean(0))          # 两边各自减去自己的中心再归一化，间隙被平移掉
    acc_c = float((hw.zeroshot_predict(unit(f) - un.mean(0), centered, scale_of(net))[1] == y["颜色x形状"]).mean())
    return {"间隙": gap, "图像向量的散布": su, "文本向量的散布": sv, "间隙除以两个散布的平均": gap / (0.5 * (su + sv)),
            "配对的图文余弦": float((un * vn).sum(1).mean()), "不配对的图文余弦": float((un @ vn.T).mean()),
            "零样本精度（颜色 x 形状）": acc, "两边各自减去中心后的零样本精度": acc_c}, (un, vn)


def fig_gap():
    probe = probe_pairs()
    test = single_test(N_TEST, 0, 99)
    groups = [("没有训练（随机初始化）", [rand_model(s) for s in SEEDS]), ("温度可学习（图 1 的模型）", [M("main", s) for s in SEEDS]),
              ("温度固定 0.01", [M("temp0.01")]), ("温度固定 0.3", [M("temp0.3")]), ("温度固定 1.0", [M("temp1.0")])]
    table, clouds = {}, {}
    for name, nets in groups:
        rows = [gap_row(n, probe, test) for n in nets]
        table[name] = {k: stats([r[0][k] for r in rows]) for k in rows[0][0]}
        clouds[name] = rows[0][1]
    table["温度可学习（图 1 的模型）"]["训练结束时的温度"] = [r3(1 / scale_of(M("main", s))) for s in SEEDS]
    for t in (0.01, 0.3, 1.0):
        table[f"温度固定 {t}"]["最后 50 步的平均损失"] = r3(np.mean(LOGS[(f"temp{t}", 0)]["loss"]))
    table["温度可学习（图 1 的模型）"]["最后 50 步的平均损失"] = stats([np.mean(LOGS[("main", s)]["loss"]) for s in SEEDS])
    traj = {"温度可学习": LOGS[("main", 0)], "温度固定 0.01": LOGS[("temp0.01", 0)], "温度固定 0.3": LOGS[("temp0.3", 0)], "温度固定 1.0": LOGS[("temp1.0", 0)]}
    NUM["模态间隙"] = {
        "说明": "1000 个随机场景的图像向量和它们各自句子的文本向量（句子按训练时的写法生成），都归一化到单位球面上。用作业的 modality_gap 量三个数："
              "间隙是两团向量各自的中心之间的距离；散布是每个向量到本团中心的距离的均方根（向量全部重合时是 0，均匀铺满球面时接近 1）。"
              "没有训练的模型和温度可学习的模型各 3 个种子，三个固定温度各 1 个种子。零样本精度是 1000 张单目标图上“颜色 x 形状”16 类。"
              "“各自减去中心”：图像向量减去图像团的中心、分类器的文本向量减去文本团的中心，再各自归一化，然后照常做零样本分类",
        **table,
        "训练过程中的间隙（种子 0，每 100 步量一次）": {"步数": traj["温度可学习"]["step"], **{k: [r3(v) for v in d["gap"]] for k, d in traj.items()}},
        "训练过程中图像向量的散布（种子 0）": {k: [r3(v) for v in d["spread_img"]] for k, d in traj.items()},
        "理论值": {"两个单位向量方向相反时的距离": 2.0, "两团中心都在原点（各自均匀铺开）时的间隙": 0.0},
    }

    fig, ax = plt.subplots(1, 5, figsize=(19.0, 4.2), gridspec_kw={"width_ratios": [0.8, 0.8, 0.8, 1.15, 1.25]}, layout="constrained")
    pick = np.random.default_rng(0).choice(1000, 300, replace=False)
    for a, name, title in zip(ax[:3], ("没有训练（随机初始化）", "温度可学习（图 1 的模型）", "温度固定 0.01"), ("没有训练", "训练后（温度可学习）", "训练后（温度固定 0.01）")):
        un, vn = clouds[name]
        x = np.concatenate([un, vn])
        mean = x.mean(0)
        _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
        pu, pv = (un - mean) @ vt[:2].T, (vn - mean) @ vt[:2].T
        for j in range(2):                            # 固定主成分的正负号，重复运行得到同一张图
            if pu[:, j].mean() < pv[:, j].mean():
                pu[:, j], pv[:, j] = -pu[:, j], -pv[:, j]
        a.scatter(pu[pick, 0], pu[pick, 1], s=7, color=S.BLUE, alpha=0.6, linewidths=0, label="图像向量")
        a.scatter(pv[pick, 0], pv[pick, 1], s=7, color=S.ORANGE, alpha=0.6, linewidths=0, label="文本向量")
        a.set_xlim(-1.15, 1.15)
        a.set_ylim(-1.15, 1.15)
        a.set_aspect("equal")
        a.set_title(f"{title}\n种子 0，间隙 {table[name]['间隙']['各次'][0]:.2f}", fontsize=10.5)
        a.set_xlabel("第 1 主成分")
        S.clean(a)
    ax[0].set_ylabel("第 2 主成分")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, markerscale=2)
    for (k, d), color, style in zip(traj.items(), (S.BLUE, S.RED, S.GREEN, S.PURPLE), ("o-", "s-", "^-", "v-")):
        ax[3].plot(d["step"], d["gap"], style, color=color, lw=1.6, ms=4, label=k)
    ax[3].set_xlabel("训练步数")
    ax[3].set_ylabel("间隙（两团中心的距离）")
    ax[3].set_ylim(0, 1.4)
    ax[3].set_title("温度越低，初始化带来的间隙留得越多", fontsize=10.5)
    ax[3].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[3])
    names = ["温度固定 0.01", "温度可学习（图 1 的模型）", "温度固定 0.3", "温度固定 1.0"]
    xs = np.arange(len(names))
    ax[4].bar(xs - 0.2, [table[n]["间隙"]["平均"] for n in names], 0.38, color=S.PURPLE, label="间隙")
    ax[4].bar(xs + 0.2, [table[n]["零样本精度（颜色 x 形状）"]["平均"] for n in names], 0.38, color=S.BLUE, label="零样本精度")
    for i, n in enumerate(names):
        ax[4].text(i - 0.2, table[n]["间隙"]["平均"] + 0.02, f"{table[n]['间隙']['平均']:.2f}", ha="center", fontsize=8.5)
        ax[4].text(i + 0.2, table[n]["零样本精度（颜色 x 形状）"]["平均"] + 0.02, f"{table[n]['零样本精度（颜色 x 形状）']['平均']:.2f}", ha="center", fontsize=8.5)
    learned = np.mean(table["温度可学习（图 1 的模型）"]["训练结束时的温度"])
    ax[4].set_xticks(xs, ["温度固定\n0.01", f"温度可学习\n（最后约 {learned:.2f}）", "温度固定\n0.3", "温度固定\n1.0"], fontsize=9)
    ax[4].set_ylim(0, 1.1)
    ax[4].set_title("间隙最大的那个模型精度并不差，\n间隙最小的那个精度最差", fontsize=10.5)
    ax[4].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    S.clean(ax[4])
    ax[4].grid(axis="x", visible=False)
    S.save(fig, CH, "fig04_模态间隙")


# ---------------------------------------------------------------- 图 5：文字查询找目标
REGION_HOWS = ["裁剪后贴回原位", "裁剪后放大到整图", "块特征", "块特征加区域头"]


def det_scenes(n, g):
    """3 个目标 1 个空格：要找的目标、一个同色不同形状的干扰、一个同形状不同色的干扰。"""
    occ = torch.zeros(n, 4, dtype=torch.bool)
    attr = torch.zeros(n, 4, 4, dtype=torch.long)
    tgt = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        cells = torch.randperm(4, generator=g)[:3].tolist()
        c, c2 = torch.randperm(4, generator=g)[:2].tolist()
        s, s2 = torch.randperm(4, generator=g)[:2].tolist()
        for cell, (ss, cc) in zip(cells, ((s, c), (s2, c), (s, c2))):
            occ[i, cell] = True
            attr[i, cell] = torch.tensor([ss, cc, int(torch.randint(0, 2, (1,), generator=g)), int(torch.randint(0, 2, (1,), generator=g))])
        tgt[i] = cells[0]
    return {"kind": torch.full((n,), 9), "occ": occ, "attr": attr}, tgt


def diff_scenes(n, g):
    """3 个目标 1 个空格，三个目标的颜色各不相同、形状也各不相同。返回场景、一句“在图里”的查询和一句“把两个目标的属性重新组合”的查询。"""
    occ = torch.zeros(n, 4, dtype=torch.bool)
    attr = torch.zeros(n, 4, 4, dtype=torch.long)
    present, absent = [], []
    for i in range(n):
        cells = torch.randperm(4, generator=g)[:3].tolist()
        cols = torch.randperm(4, generator=g)[:3].tolist()
        shps = torch.randperm(4, generator=g)[:3].tolist()
        for cell, cc, ss in zip(cells, cols, shps):
            occ[i, cell] = True
            attr[i, cell] = torch.tensor([ss, cc, int(torch.randint(0, 2, (1,), generator=g)), int(torch.randint(0, 2, (1,), generator=g))])
        present.append(f"一个 {COLORS[cols[0]]} {SHAPES[shps[0]]}")
        absent.append(f"一个 {COLORS[cols[0]]} {SHAPES[shps[1]]}")
    return {"kind": torch.full((n,), 9), "occ": occ, "attr": attr}, present, absent


def cell_mean(e):
    """(B, 4, 4, D) 的块特征，按 2x2 个块一个格子取平均，得到 (B, 4, D)。"""
    return torch.stack([e[:, (c // 2) * 2:(c // 2) * 2 + 2, (c % 2) * 2:(c % 2) * 2 + 2].mean((1, 2)) for c in range(4)], 1)


@torch.no_grad()
def region_feats(net, img, how, head=None):
    """四个格子各自的区域特征 (N, 4, D)，没有归一化。"""
    n = len(img)
    if how.startswith("块特征"):
        t = net.image.tokens(img)
        e = head(t) if how == "块特征加区域头" else net.image.proj(t)
        return cell_mean(e.reshape(n, G, G, -1)).numpy().astype(np.float64)
    out = []
    for c in range(4):
        y0, x0 = (c // 2) * CELL, (c % 2) * CELL
        crop = img[:, :, y0:y0 + CELL, x0:x0 + CELL]
        if how == "裁剪后放大到整图":
            x = F.interpolate(crop, scale_factor=2, mode="bilinear", align_corners=False)
        else:                                         # 其余位置填成这一格四个角的平均灰度，目标的大小和位置都不变
            fill = torch.stack([crop[:, :, 0, 0], crop[:, :, 0, -1], crop[:, :, -1, 0], crop[:, :, -1, -1]]).mean((0, 2))
            x = fill[:, None, None, None].expand(n, 3, SIZE, SIZE).clone()
            x[:, :, y0:y0 + CELL, x0:x0 + CELL] = crop
        out.append(img_emb(net, x))
    return np.stack(out, 1)


def train_region_head(net, seed, steps=300):
    """双塔冻结，只训练一个作用在块标记上的线性头：每个格子的特征要和“这一格里的目标”的句子对上，空格子和“空 背景”对上。需要格子一级的标注。"""
    torch.manual_seed(seed)
    g = gen(seed, 81)
    head = nn.Linear(DIM, OUT, bias=False)
    with torch.no_grad():
        head.weight.copy_(net.image.proj.weight)
    opt = torch.optim.Adam(head.parameters(), lr=2e-3)
    wall = torch.from_numpy(np.concatenate([classifier(net, COMBOS), classifier(net, ["空 背景"], ("{}",))])).float()
    sc_ = scale_of(net)
    for _ in range(steps):
        sc = sample_scenes(BATCH, g)
        with torch.no_grad():
            t = net.image.tokens(render(sc, g))
        r = F.normalize(cell_mean(head(t).reshape(BATCH, G, G, -1)), dim=-1)
        lab = torch.where(sc["occ"], sc["attr"][..., 1] * 4 + sc["attr"][..., 0], torch.full((BATCH, 4), 16))
        loss = F.cross_entropy((sc_ * r @ wall.t()).reshape(-1, 17), lab.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head.eval()


def fig_detect():
    n = 600
    loc = {h: {"全部": [], "大目标": [], "小目标": []} for h in REGION_HOWS}
    cosm = {h: {"要找的目标所在的格子": [], "别的目标所在的格子": [], "空格子": []} for h in REGION_HOWS}
    pres = {"整图向量": [], "区域最高分（裁剪后贴回原位）": []}
    top_empty = {h: [] for h in REGION_HOWS}
    cls = {k: [] for k in ("目标格子的 16 类精度", "其中大目标", "其中小目标", "空格子上最大类别概率的平均", "空格子上最大类别概率超过 0.5 的比例", "空格子上最大类别概率超过 0.3 的比例",
                           "加背景提示：目标格子的精度", "加背景提示：空格子上最大类别概率的平均", "加背景提示：空格子上最大类别概率超过 0.5 的比例", "加背景提示：空格子上最大类别概率超过 0.3 的比例")}
    example = None
    for s in SEEDS:
        net = M("main", s)
        sc_ = scale_of(net)
        head = train_region_head(net, s)
        sc, tgt = det_scenes(n, gen(s, 41))
        img = render(sc, gen(s, 42))
        ta = sc["attr"][torch.arange(n), tgt]
        queries = [f"一个 {COLORS[c]} {SHAPES[sh]}" for sh, c in zip(ta[:, 0].tolist(), ta[:, 1].tolist())]
        q = txt_emb(net, queries)
        big = (ta[:, 2] == 0).numpy()
        occ = sc["occ"].numpy()
        is_tgt = np.eye(4, dtype=bool)[tgt.numpy()]
        feats = {}
        for how in REGION_HOWS:
            r = region_feats(net, img, how, head)
            feats[how] = r
            scores = np.stack([hw.score_regions(r[i], q[i:i + 1])[0][:, 0] for i in range(n)])
            top = np.array([hw.score_regions(r[i], q[i:i + 1])[1][0] for i in range(n)])
            hit = top == tgt.numpy()
            top_empty[how].append(float((~occ[np.arange(n), top]).mean()))
            for key, mask in (("全部", np.ones(n, bool)), ("大目标", big), ("小目标", ~big)):
                loc[how][key].append(float(hit[mask].mean()))
            for key, mask in (("要找的目标所在的格子", is_tgt), ("别的目标所在的格子", occ & ~is_tgt), ("空格子", ~occ)):
                cosm[how][key].append(float(scores[mask].mean()))
            if s == 0:
                solid_big = ((sc["attr"][..., 2] == 0) & (sc["attr"][..., 3] == 0)) | ~sc["occ"]
                e = int(solid_big.all(1).nonzero().flatten()[0])              # 第一张三个目标都是大的、实心的图，看得清
                example = example or {"img": img[e], "tgt": int(tgt[e]), "query": queries[e], "maps": {}}
                example["maps"][how] = scores[e].reshape(2, 2)
        r = feats["裁剪后贴回原位"].reshape(-1, OUT)
        clf16 = classifier(net, COMBOS)
        clf17 = np.concatenate([clf16, classifier(net, ["空 背景"], ("{}",))])
        truth = (sc["attr"][..., 1] * 4 + sc["attr"][..., 0]).numpy().reshape(-1)
        occf, bigf = occ.reshape(-1), (sc["attr"][..., 2] == 0).numpy().reshape(-1)
        p16, pred16 = hw.zeroshot_predict(r, clf16, sc_)
        p17, pred17 = hw.zeroshot_predict(r, clf17, sc_)
        cls["目标格子的 16 类精度"].append(float((pred16[occf] == truth[occf]).mean()))
        cls["其中大目标"].append(float((pred16[occf & bigf] == truth[occf & bigf]).mean()))
        cls["其中小目标"].append(float((pred16[occf & ~bigf] == truth[occf & ~bigf]).mean()))
        cls["加背景提示：目标格子的精度"].append(float((pred17[occf] == truth[occf]).mean()))
        for pre, p in (("", p16.max(1)), ("加背景提示：", p17[:, :16].max(1))):
            cls[pre + "空格子上最大类别概率的平均"].append(float(p[~occf].mean()))
            cls[pre + "空格子上最大类别概率超过 0.5 的比例"].append(float((p[~occf] > 0.5).mean()))
            cls[pre + "空格子上最大类别概率超过 0.3 的比例"].append(float((p[~occf] > 0.3).mean()))
        # 查询在不在图里：整图向量和区域最高分各能不能分开
        sc2, present, absent = diff_scenes(n, gen(s, 43))
        img2 = render(sc2, gen(s, 44))
        qp, qa = unit(txt_emb(net, present)), unit(txt_emb(net, absent))
        whole = unit(img_emb(net, img2))
        r2_ = region_feats(net, img2, "裁剪后贴回原位")
        lab = np.r_[np.ones(n), np.zeros(n)]
        pres["整图向量"].append(hw19.auroc(np.r_[(whole * qp).sum(1), (whole * qa).sum(1)], lab))
        best = [hw.score_regions(r2_[i], np.stack([qp[i], qa[i]]))[2] for i in range(n)]
        pres["区域最高分（裁剪后贴回原位）"].append(hw19.auroc(np.r_[[b[0] for b in best], [b[1] for b in best]], lab))
    NUM["文字查询找目标"] = {
        "说明": f"图 1 的 3 个模型。每个种子 {n} 张图，每张图 3 个目标 1 个空格：要找的目标、一个同色不同形状的干扰、一个同形状不同色的干扰，大小和纹理随机。查询是“一个 颜色 形状”。"
              "候选区域就是 4 个格子，每个区域一个特征，用作业的 score_regions 和查询的文本向量算余弦，取分数最高的格子，随机猜 0.25。区域特征的四种来源："
              "“裁剪后贴回原位”是把这一格的像素贴到一张填了背景灰度的空图的同一位置再过图像塔（目标的大小不变）；“裁剪后放大到整图”是把 16x16 的格子双线性放大到 32x32 再过图像塔（目标变成原来的 2 倍大）；"
              "“块特征”是整图过一次图像塔，把这一格的 2x2 个块标记各自投影后取平均；“块特征加区域头”是双塔冻结，另外训练一个作用在块标记上的线性头，"
              f"用 {300 * BATCH} 张带格子标注的图训练 300 步，让每个格子的特征和这一格目标的句子对上",
        "定位精度": {h: {k: stats(v) for k, v in d.items()} for h, d in loc.items()},
        "分数最高的格子是空格子的比例": {h: stats(v) for h, v in top_empty.items()},
        "查询和各类格子的平均余弦": {h: {k: stats(v) for k, v in d.items()} for h, d in cosm.items()},
        "查询在不在图里（AUROC）": {
            "说明": f"另外 {n} 张图，3 个目标的颜色各不相同、形状也各不相同。“在图里”的查询是其中一个目标；“不在图里”的查询把一个目标的颜色和另一个目标的形状拼在一起，"
                  "这两个词图里都有，但没有一个目标同时是这个颜色和这个形状。整图向量：直接用整张图的图像向量和查询算余弦。区域最高分：4 个格子里最高的余弦。0.5 是分不开",
            **{k: stats(v) for k, v in pres.items()},
        },
        "区域分类和背景提示": {
            "说明": "同样的图，区域特征用“裁剪后贴回原位”。把每个格子当成一张小图做 16 类零样本分类（作业的 zeroshot_predict，温度用模型学到的）。"
                  "空格子里什么都没有，但 16 个类别的概率加起来必须是 1。加背景提示：分类器里多加一行“空 背景”的文本向量，变成 17 类，再看前 16 类里最大的概率",
            **{k: stats(v) for k, v in cls.items()},
        },
    }

    fig, ax = plt.subplots(1, 5, figsize=(20.5, 4.3), gridspec_kw={"width_ratios": [0.72, 1.9, 1.5, 0.75, 0.9]}, layout="constrained")
    ax[0].imshow(example["img"].permute(1, 2, 0).clamp(0, 1).numpy(), interpolation="nearest")
    ax[0].plot([15.5, 15.5], [-0.5, 31.5], color="white", lw=0.8, alpha=0.7)
    ax[0].plot([-0.5, 31.5], [15.5, 15.5], color="white", lw=0.8, alpha=0.7)
    ty, tx = (example["tgt"] // 2) * CELL, (example["tgt"] % 2) * CELL
    ax[0].add_patch(plt.Rectangle((tx - 0.5, ty - 0.5), CELL, CELL, fill=False, edgecolor="cyan", lw=2))
    ax[0].set_title(f"查询：“{example['query']}”\n青色框是它所在的格子", fontsize=10.5)
    S.image_axes(ax[0])
    tile = np.full((2, 4 * 2 + 3), np.nan)
    for j, how in enumerate(REGION_HOWS):
        tile[:, j * 3:j * 3 + 2] = example["maps"][how]
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("white")
    ax[1].imshow(tile, cmap=cmap, vmin=-0.6, vmax=1.0, aspect="equal")
    for j, how in enumerate(REGION_HOWS):
        for cy in range(2):
            for cx in range(2):
                v = example["maps"][how][cy, cx]
                ax[1].text(j * 3 + cx, cy, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color="black" if v > 0.45 else "white")
        tcy, tcx = example["tgt"] // 2, example["tgt"] % 2
        ax[1].add_patch(plt.Rectangle((j * 3 + tcx - 0.5, tcy - 0.5), 1, 1, fill=False, edgecolor="cyan", lw=2))
    ax[1].set_xticks([0.5 + 3 * j for j in range(4)], ["裁剪后\n贴回原位", "裁剪后\n放大到整图", "块特征", "块特征\n加区域头"], fontsize=9)
    ax[1].set_yticks([])
    ax[1].tick_params(length=0)
    for side in ax[1].spines.values():
        side.set_visible(False)
    ax[1].set_title("这张图上四个格子和查询的余弦", fontsize=10.5)
    xs = np.arange(len(REGION_HOWS))
    for i, (key, color) in enumerate((("全部", S.BLUE), ("大目标", S.GREEN), ("小目标", S.ORANGE))):
        vals = [loc[h][key] for h in REGION_HOWS]
        ax[2].bar(xs + (i - 1) * 0.26, [np.mean(v) for v in vals], 0.24, color=color, label=key)
        for xx, v in zip(xs + (i - 1) * 0.26, vals):
            dots(ax[2], xx, v)
            ax[2].text(xx, max(v) + 0.02, f"{np.mean(v):.2f}", ha="center", fontsize=7.5)
    ax[2].axhline(0.25, color=S.RED, lw=1.1, ls="--")
    ax[2].set_xticks(xs, ["裁剪后\n贴回原位", "裁剪后\n放大到整图", "块特征", "块特征\n加区域头"], fontsize=9)
    ax[2].set_ylim(0, 1.12)
    ax[2].set_ylabel("分数最高的格子就是目标的比例")
    ax[2].set_title("区域特征从哪来，决定找不找得到", fontsize=10.5)
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    for i, (k, color) in enumerate(zip(pres, (S.MUTED, S.BLUE))):
        ax[3].bar(i, np.mean(pres[k]), 0.6, color=color)
        dots(ax[3], i, pres[k])
        ax[3].text(i, max(pres[k]) + 0.02, f"{np.mean(pres[k]):.2f}", ha="center", fontsize=8.5)
    ax[3].axhline(0.5, color=S.RED, lw=1.1, ls="--")
    ax[3].set_xticks([0, 1], ["整图\n向量", "区域\n最高分"], fontsize=9)
    ax[3].set_ylim(0, 1.12)
    ax[3].set_ylabel("AUROC")
    ax[3].set_title("查询在不在图里\n（属性重新组合的查询）", fontsize=10.5)
    keys = [("空格子上最大类别概率的平均", "加背景提示：空格子上最大类别概率的平均"), ("空格子上最大类别概率超过 0.3 的比例", "加背景提示：空格子上最大类别概率超过 0.3 的比例")]
    for i, (a_, b_) in enumerate(keys):
        ax[4].bar(i - 0.2, np.mean(cls[a_]), 0.38, color=S.ORANGE, label="16 个类别" if i == 0 else None)
        ax[4].bar(i + 0.2, np.mean(cls[b_]), 0.38, color=S.GREEN, label="加一行“空 背景”" if i == 0 else None)
        dots(ax[4], i - 0.2, cls[a_])
        dots(ax[4], i + 0.2, cls[b_])
        ax[4].text(i - 0.2, max(cls[a_]) + 0.02, f"{np.mean(cls[a_]):.2f}", ha="center", fontsize=8.5)
        ax[4].text(i + 0.2, max(cls[b_]) + 0.02, f"{np.mean(cls[b_]):.2f}", ha="center", fontsize=8.5)
    ax[4].set_xticks([0, 1], ["最大类别\n概率的平均", "超过 0.3\n的比例"], fontsize=9)
    ax[4].set_ylim(0, 1.12)
    ax[4].set_title("空格子被判成某一类\n的把握有多大", fontsize=10.5)
    ax[4].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    for a in ax[2:]:
        S.clean(a)
        a.grid(axis="x", visible=False)
    S.save(fig, CH, "fig05_文字查询找目标")


# ---------------------------------------------------------------- 图 6：少样本适配；训练文字里没有的属性
def low_contrast(img):
    return 0.4 + 0.5 * (img - 0.4)


def product_set(n, a, g, tex=0, notch=0):
    """同一种“产品”的 n 张图：形状、颜色、位置固定的一个大目标。tex=1 是条纹，notch>0 是中心有一个暗坑。"""
    shape, color, cell = a
    occ = torch.zeros(n, 4, dtype=torch.bool)
    attr = torch.zeros(n, 4, 4, dtype=torch.long)
    occ[:, cell] = True
    attr[:, cell] = torch.tensor([shape, color, 0, tex])
    nt = torch.zeros(n, 4, dtype=torch.long)
    nt[:, cell] = notch
    return render({"kind": torch.zeros(n, dtype=torch.long), "occ": occ, "attr": attr}, g, notch=nt if notch else None)


def fig_adapt():
    # A：换了成像条件之后的少样本适配
    img, y = single_test(N_TEST, 0, 99)
    pool_img, pool_y = single_test(4000, 9, 99)
    yte, ypool = y["颜色x形状"], pool_y["颜色x形状"]
    ks = [1, 2, 4, 8, 16]
    lam = 0.3
    few = {m: {k: [] for k in ks} for m in ("线性探测（从 0 开始）", "从零样本分类器出发微调")}
    zs_shift, zs_orig, ceiling = [], [], []
    for s in SEEDS:
        net = M("main", s)
        sc_ = scale_of(net)
        f, fpool = img_emb(net, low_contrast(img)), img_emb(net, low_contrast(pool_img))
        clf = classifier(net, COMBOS)
        zs_shift.append(zs_acc(net, f, yte, clf))
        zs_orig.append(zs_acc(net, img_emb(net, img), yte, clf))
        ceiling.append(head_acc(fit_head(fpool[:1600], ypool[:1600], np.zeros((16, OUT)), 0.01, sc_), f, yte, sc_))
        for k in ks:
            for rep in range(4):
                idx = k_shot(ypool, 16, k, gen(s, k, rep, 5))
                few["线性探测（从 0 开始）"][k].append(head_acc(fit_head(fpool[idx], ypool[idx], np.zeros((16, OUT)), 0.01, sc_), f, yte, sc_))
                few["从零样本分类器出发微调"][k].append(head_acc(fit_head(fpool[idx], ypool[idx], clf, lam, sc_), f, yte, sc_))
    NUM["少样本适配"] = {
        "说明": "图 1 的 3 个模型，双塔都冻结。目标域是“对比度减半”的图：每个像素换成 0.4 + 0.5 x (原值 - 0.4)，模型训练时没见过这种图。任务是“颜色 x 形状”16 类，测试图 1000 张。"
              "每类从 4000 张带标签的目标域图里随机取 k 张，每个模型取 4 次，共 12 次的平均。两种做法都是在归一化的图像向量上训练线性分类头（交叉熵，全批量 Adam 100 步，学习率 0.05）："
              f"线性探测的权重从 0 开始，L2 系数 0.01；从零样本分类器出发微调的权重从文本向量开始，惩罚的是离文本向量的距离，系数 {lam}",
        "每类样本数": ks,
        "零样本（原来的图）": stats(zs_orig), "零样本（对比度减半的图）": stats(zs_shift),
        "线性探测，每类 100 张标签（1600 张）": stats(ceiling),
        **{m: [r3(np.mean(few[m][k])) for k in ks] for m in few},
        **{f"{m}：各次的最小和最大": [[r3(min(few[m][k])), r3(max(few[m][k]))] for k in ks] for m in few},
    }

    # B：训练句子里从不提纹理
    ns = [4, 16, 64, 256]
    tex_models = {"句子里写纹理（图 1 的模型）": [M("main", s) for s in (0, 1)], "句子里从不写纹理": [M("notex", s) for s in (0, 1)], "没有训练（随机初始化）": [rand_model(s) for s in (0, 1)]}
    tex = {name: {"零样本": [], "线性头": {n_: [] for n_ in ns}, "形状的零样本": [], "投影前": []} for name in tex_models}
    for name, nets in tex_models.items():
        for s, net in enumerate(nets):
            sc_ = scale_of(net)
            f, fpool = img_emb(net, img), img_emb(net, pool_img)
            tex[name]["零样本"].append(zs_acc(net, f, y["纹理"], classifier(net, TASKS["纹理"][0])))
            tex[name]["形状的零样本"].append(zs_acc(net, f, y["形状"], classifier(net, SHAPES)))
            with torch.no_grad():                    # 投影之前的 64 维特征（16 个块标记的平均）
                h, hpool = (net.image.tokens(x).mean(1).numpy().astype(np.float64) for x in (img, pool_img))
            for n_ in ns:
                for rep in range(3):
                    idx = k_shot(pool_y["纹理"], 2, n_ // 2, gen(s, n_, rep, 6))
                    tex[name]["线性头"][n_].append(head_acc(fit_head(fpool[idx], pool_y["纹理"][idx], np.zeros((2, OUT)), 0.01, sc_), f, y["纹理"], sc_))
                    if n_ == ns[-1]:
                        tex[name]["投影前"].append(head_acc(fit_head(hpool[idx], pool_y["纹理"][idx], np.zeros((2, DIM)), 0.01, sc_), h, y["纹理"], sc_))
    NUM["训练句子里没有的属性"] = {
        "说明": "两组双塔模型，训练图像完全一样（一半目标是条纹的），只差句子：一组照常写“实心”“条纹”，另一组的句子里永远不出现这两个词。各 2 个种子，另加 2 个没有训练的模型作参照。"
              "任务：1000 张单目标图分“实心”和“条纹”，随机猜 0.5。零样本用“一个 实心 图形”“一个 条纹 图形”两句话。"
              "线性头：图像塔冻结，在图像向量上用带标签的图训练（两类各一半，每个模型取 3 次）。“投影之前”指图像塔最后那个线性投影之前、16 个块标记取平均得到的 64 维特征",
        "带标签的图数": ns,
        **{name: {"零样本": stats(d["零样本"]), "线性头": [r3(np.mean(d["线性头"][n_])) for n_ in ns], "形状的零样本（对照）": stats(d["形状的零样本"]),
                  "线性头换到投影之前的 64 维特征上（256 张标签）": r3(np.mean(d["投影前"]))} for name, d in tex.items()},
    }

    # C：用文字，或者用几张良品，检出两类“缺陷”
    cases = [("条纹", {"tex": 1}), ("暗坑 2 px", {"notch": 2}), ("暗坑 4 px", {"notch": 4}), ("暗坑 6 px", {"notch": 6})]
    methods = ["文字（句子里写纹理的模型）", "文字（句子里从不写纹理的模型）", "8 张良品的近邻，图像向量（句子里写纹理的模型）",
               "8 张良品的近邻，图像向量（句子里从不写纹理的模型）", "8 张良品的近邻，图像向量（没有训练的模型）", "8 张良品的近邻，原始像素"]
    ad = {c: {m: [] for m in methods} for c, _ in cases}
    lab = np.r_[np.zeros(50), np.ones(50)]
    for s in (0, 1):
        g = gen(s, 61)
        nets = {"句子里写纹理的模型": M("main", s), "句子里从不写纹理的模型": M("notex", s), "没有训练的模型": rand_model(s)}
        vals = {c: {m: [] for m in methods} for c, _ in cases}
        for _ in range(12):
            a = [int(torch.randint(0, 4, (1,), generator=g)) for _ in range(3)]
            ref, good = product_set(8, a, g), product_set(50, a, g)
            for cname, kw in cases:
                query = torch.cat([good, product_set(50, a, g, **kw)])
                for mname, net in nets.items():
                    fr, fq = unit(img_emb(net, ref)), unit(img_emb(net, query))
                    vals[cname][f"8 张良品的近邻，图像向量（{mname}）"].append(hw19.auroc(1 - (fq @ fr.T).max(1), lab))
                    if mname != "没有训练的模型" and cname == "条纹":
                        t = unit(txt_emb(net, [f"一个 条纹 {COLORS[a[1]]} {SHAPES[a[0]]}", f"一个 实心 {COLORS[a[1]]} {SHAPES[a[0]]}"]))
                        vals[cname][f"文字（{mname}）"].append(hw19.auroc(fq @ t[0] - fq @ t[1], lab))
                d = ((query.reshape(100, 1, -1) - ref.reshape(1, 8, -1)) ** 2).sum(-1).min(1).values.numpy()
                vals[cname]["8 张良品的近邻，原始像素"].append(hw19.auroc(d, lab))
        for c, _ in cases:
            for m in methods:
                if vals[c][m]:
                    ad[c][m].append(float(np.mean(vals[c][m])))
    NUM["用文字或几张良品检出缺陷"] = {
        "说明": "把“同一种产品”定义成形状、颜色、位置都固定的一个大目标（实心）。每个种子随机取 12 种产品，每种产品 50 张良品和 50 张缺陷图，算 image AUROC（第 19 章作业的 auroc），再对 12 种产品取平均；2 个种子。"
              "缺陷有两类：整个目标变成条纹（训练图像里有这种图，句子里写不写纹理看模型）；目标中心有一个 2、4、6 px 见方的暗坑（任何训练图像和句子里都没有）。"
              "文字：分数是图像向量和“一个 条纹 颜色 形状”的余弦减去和“一个 实心 颜色 形状”的余弦，不用任何参考图。暗坑没有对应的词，所以没有这一项。"
              "8 张良品的近邻：分数是 1 减去图像向量和 8 张良品图像向量的最大余弦；原始像素那一行是到 8 张良品的最小欧氏距离",
        **{c: {m: stats(v) for m, v in d.items() if v} for c, d in ad.items()},
    }

    fig, ax = plt.subplots(1, 3, figsize=(17.5, 4.7), gridspec_kw={"width_ratios": [1.0, 1.0, 1.45]}, layout="constrained")
    for m, color, style in zip(few, (S.ORANGE, S.GREEN), ("o-", "D-")):
        ax[0].plot(ks, [np.mean(few[m][k]) for k in ks], style, color=color, lw=1.8, ms=5.5, label=m)
        ax[0].fill_between(ks, [min(few[m][k]) for k in ks], [max(few[m][k]) for k in ks], color=color, alpha=0.13, lw=0)
    ax[0].axhline(np.mean(zs_shift), color=S.BLUE, lw=1.8, label="零样本（不用标签）")
    ax[0].axhline(np.mean(ceiling), color=S.MUTED, lw=1.1, ls=":", label="线性探测，每类 100 张")
    ax[0].set_xscale("log")
    ax[0].set_xticks(ks, [str(k) for k in ks])
    ax[0].minorticks_off()
    ax[0].set_ylim(0.2, 1.0)
    ax[0].set_xlabel("目标域里每类带标签的图数")
    ax[0].set_ylabel("对比度减半的图上的精度（16 类）")
    ax[0].set_title("样本少时，从零样本分类器出发最稳")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[0])
    for (name, d), color, style in zip(tex.items(), (S.BLUE, S.RED, S.MUTED), ("o-", "s-", "v:")):
        ax[1].plot(ns, [np.mean(d["线性头"][n_]) for n_ in ns], style, color=color, lw=1.8, ms=5.5, label=name)
    ax[1].axhline(0.5, color=S.RED, lw=1.0, ls="--")
    ax[1].set_xscale("log")
    ax[1].set_xticks(ns, [str(n_) for n_ in ns])
    ax[1].minorticks_off()
    ax[1].set_ylim(0.4, 1.03)
    ax[1].set_xlabel("带标签的图数（实心、条纹各一半）")
    ax[1].set_ylabel("冻结的图像向量上线性头分纹理的精度")
    ax[1].set_title("句子里从不写的属性，图像向量里也留不住")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax[1])
    colors = [S.BLUE, S.RED, S.BLUE, S.RED, S.MUTED, "#111111"]
    hatches = ["", "", "//", "//", "//", ".."]
    short = ["文字，句子里写纹理的模型", "文字，句子里从不写纹理的模型", "8 张良品近邻，句子里写纹理的模型", "8 张良品近邻，句子里从不写纹理的模型", "8 张良品近邻，没有训练的模型", "8 张良品近邻，原始像素"]
    for i, (c, _) in enumerate(cases):
        for j, m in enumerate(methods):
            if not ad[c][m]:
                continue
            v = float(np.mean(ad[c][m]))
            xx = i + (j - 2.5) * 0.14
            ax[2].bar(xx, v, 0.13, color=colors[j], alpha=0.85 if j < 2 else 0.55, hatch=hatches[j], edgecolor="white", label=short[j] if i == 0 else None)
            ax[2].text(xx, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.8, rotation=90, va="bottom")
    ax[2].axhline(0.5, color=S.RED, lw=1.0, ls="--")
    ax[2].set_xticks(range(len(cases)), ["整个目标变成条纹", "暗坑 2 px", "暗坑 4 px", "暗坑 6 px"])
    ax[2].set_ylim(0.3, 1.16)
    ax[2].set_ylabel("image AUROC（12 种产品、2 个种子的平均）")
    ax[2].set_title("文字只认训练句子里写过的那种变化；暗坑没有词可用，2 px 时谁都不行")
    ax[2].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    S.clean(ax[2])
    ax[2].grid(axis="x", visible=False)
    S.save(fig, CH, "fig06_少样本适配与没见过的属性")


# ---------------------------------------------------------------- 表：作业里几种写错的后果
def numbers_bugs():
    """把作业函数的几种典型错误写法放到同一批数据上，量出后果。讲义的常见错误表引用。错误写法只在这里出现，不在讲义里。"""
    img, y = single_test(N_TEST, 0, 99)
    out = {k: [] for k in ("正确：六个模板先各自归一化再平均", "模板不归一化直接平均", "各模板文本向量长度的最大值除以最小值",
                           "正确的最大概率平均", "忘了乘放大倍数时的最大概率平均", "忘了乘放大倍数时预测变了的比例",
                           "图像向量不归一化时预测变了的比例", "图像向量不归一化时最大概率的平均", "图像向量长度的最大值除以最小值")}
    loc = {"正确": [], "区域特征和查询都不归一化": [], "区域特征长度的最大值除以最小值": []}
    gap = {"正确": [], "不归一化": []}
    for s in SEEDS:
        net = M("main", s)
        sc_ = scale_of(net)
        f = img_emb(net, img)
        t = np.stack([txt_emb(net, [tpl.format(p) for p in SHAPES]) for tpl in TEMPLATES.values()], 1)      # (4, 6, D)
        out["正确：六个模板先各自归一化再平均"].append(zs_acc(net, f, y["形状"], hw.build_zeroshot_classifier(t)))
        out["模板不归一化直接平均"].append(zs_acc(net, f, y["形状"], unit(t.mean(1))))
        norms = np.linalg.norm(t, axis=-1)
        out["各模板文本向量长度的最大值除以最小值"].append(float(norms.max() / norms.min()))
        clf = classifier(net, SHAPES)
        probs, pred = hw.zeroshot_predict(f, clf, sc_)
        p1, pred1 = hw.zeroshot_predict(f, clf, 1.0)
        out["正确的最大概率平均"].append(float(probs.max(1).mean()))
        out["忘了乘放大倍数时的最大概率平均"].append(float(p1.max(1).mean()))
        out["忘了乘放大倍数时预测变了的比例"].append(float((pred1 != pred).mean()))
        raw = sc_ * f @ clf.T
        e = np.exp(raw - raw.max(1, keepdims=True))
        out["图像向量不归一化时预测变了的比例"].append(float((raw.argmax(1) != pred).mean()))
        out["图像向量不归一化时最大概率的平均"].append(float((e / e.sum(1, keepdims=True)).max(1).mean()))
        fn = np.linalg.norm(f, axis=1)
        out["图像向量长度的最大值除以最小值"].append(float(fn.max() / fn.min()))
        sc, tgt = det_scenes(600, gen(s, 41))
        dimg = render(sc, gen(s, 42))
        ta = sc["attr"][torch.arange(600), tgt]
        q = txt_emb(net, [f"一个 {COLORS[c]} {SHAPES[sh]}" for sh, c in zip(ta[:, 0].tolist(), ta[:, 1].tolist())])
        r = region_feats(net, dimg, "裁剪后贴回原位")
        loc["正确"].append(float(np.mean([hw.score_regions(r[i], q[i:i + 1])[1][0] == int(tgt[i]) for i in range(600)])))
        loc["区域特征和查询都不归一化"].append(float(((r * q[:, None]).sum(-1).argmax(1) == tgt.numpy()).mean()))
        rn = np.linalg.norm(r, axis=-1)
        loc["区域特征长度的最大值除以最小值"].append(float((rn.max(1) / rn.min(1)).mean()))
        probe = probe_pairs()
        u, v = img_emb(net, probe[0]), txt_emb(net, probe[1])
        gap["正确"].append(hw.modality_gap(u, v)[0])
        gap["不归一化"].append(float(np.linalg.norm(u.mean(0) - v.mean(0))))
    NUM["几种写错的后果"] = {
        "说明": "图 1 的 3 个模型，数据同各自的实验。每一项都是把作业函数的一步故意写错后量出来的，用来说明这一步为什么要这样规定。"
              "检索里“只认对角线”的后果见“图文检索和打乱词序”",
        "形状任务，六个模板取平均": {k: stats(out[k]) for k in ("正确：六个模板先各自归一化再平均", "模板不归一化直接平均", "各模板文本向量长度的最大值除以最小值")},
        "形状任务，零样本预测": {k: stats(out[k]) for k in list(out)[3:]},
        "文字查询找目标（裁剪后贴回原位）": {k: stats(v) for k, v in loc.items()},
        "模态间隙": {k: stats(v) for k, v in gap.items()},
    }


# ---------------------------------------------------------------- 其余数字
def numbers_misc():
    net = TwoTower()
    NUM["模型规模"] = {
        "说明": "本章小模型的可训练参数个数（固定的正弦位置编码不算）。真模型的参数量以各自的模型卡为准，本章不引用",
        "图像塔": n_params(net.image), "文本塔": n_params(net.text), "双塔合计（含 1 个温度参数）": n_params(net),
        "词袋文本塔": n_params(TextTower(bow=True)), "词表大小": len(VOCAB), "句子最长的词数": MAXLEN, "共同空间的维数": OUT,
        "一张图的块数": L, "一个块的像素数": P * P,
    }
    NUM["理论数值"] = {
        "说明": "由脚本算出的常数，讲义里直接引用",
        f"什么都没学到时的损失 ln {BATCH}": r3(math.log(BATCH)),
        "温度 0.07 对应的相似度放大倍数": r2(1 / 0.07),
        "16 选 1 的随机猜": r3(1 / 16), "4 选 1 的随机猜": 0.25,
        "“颜色 x 形状”的组合数": 16, "训练时拿掉的组合数": len(HOLD), "剩下的组合数": 16 - len(HOLD),
        "左右两个目标（颜色、形状都不同）的写法种数": 16 * 9,
        "16 句里随便挑 3 句，包含正确答案的概率": r3(3 / 16),
        "把句子里的 6 个词随机排列，恰好排回原样的概率": r3(1 / math.factorial(6)),
        "放大倍数 100、余弦差 0.1 时两类的概率比 e^10": float(f"{math.exp(10):.3g}"),
        "放大倍数 20、余弦差 0.1 时两类的概率比 e^2": r2(math.exp(2)),
        "“小”目标的面积占一个格子的比例": r3(EXT[1] ** 2 / CELL ** 2), "“大”目标放大 2 倍后的边长（px）": 2 * EXT[0], "“小”目标放大 2 倍后的边长（px）": 2 * EXT[1],
        "2 px 暗坑的像素数": 4, "4 px 暗坑的像素数": 16, "6 px 暗坑的像素数": 36, "2 px 暗坑占一个 8x8 块的比例": r3(4 / (P * P)),
        "区域头用到的带格子标注的图数": 300 * BATCH,
        "一个批里每张图面对的负样本句子数": BATCH - 1,
        "输入 224 px 时的块数": {"块 32 px": (224 // 32) ** 2, "块 16 px": (224 // 16) ** 2, "块 14 px": (224 // 14) ** 2},
        "一个块的像素数": {"块 32 px": 32 * 32, "块 16 px": 16 * 16, "块 14 px": 14 * 14},
        "3x3 px 的缺陷占一个 32x32 px 块的比例": float(f"{9 / 1024:.2g}"),
        "把边长 2048 px 的图缩到 224 px 的倍数": r2(2048 / 224), "原图上 10 px 的缺陷缩小后的边长（px）": r2(10 * 224 / 2048),
    }
    NUM["论文和官方实现里的设定（引用，不是实测）"] = {
        "说明": "讲义里引用的几个常数，出自论文和官方仓库，本章没有验证，以原文和仓库为准。列在这里只是为了让讲义里的每个数都有出处",
        "CLIP 论文的训练数据规模（亿对图文）": 4,
        "CLIP 论文里温度的初值": 0.07, "CLIP 论文里相似度放大倍数的上限": 100,
        "CLIP 文本编码器的最长标记数": 77,
        "CLIP 论文在 ImageNet 上用的提示词模板个数": 80,
    }
    NUM["演示页"] = {
        "说明": "demos/零样本分类与提示词.html 里的二维例子：3 个类别（方块、圆、三角），每类 40 个图像向量、A、B、C 三个模板各一个文本向量，另有 20 个不属于任何类别的“空区域”图像向量。"
              "页面里的数值在浏览器里实时算，讲义只引用这里列的规模",
        "类别数": 3, "每类图像向量个数": 40, "图像向量总数": 120, "每类的模板个数": 3, "空区域的图像向量个数": 20, "按顺序试的步数": 5, "放大倍数的默认值": 20,
    }
    # 作业文档字符串里的几个性质，用参考实现核对一遍，讲义“作业里的零件”一节引用
    rng = np.random.default_rng(28)
    u = rng.normal(size=(50, 6))
    gap, su, sv = hw.modality_gap(u, 3.0 * u)
    cu = unit(u).mean(0)
    assert gap < 1e-12 and abs(su - sv) < 1e-12 and abs(su - math.sqrt(1 - cu @ cu)) < 1e-12
    probs, pred = hw.zeroshot_predict(np.eye(3)[[0, 1]], np.eye(3), 100.0)
    assert pred.tolist() == [0, 1] and abs(probs[0, 0] - 1 / (1 + 2 * math.exp(-100))) < 1e-12
    NUM["作业自检"] = {
        "说明": "用参考实现核对的性质：同一组向量和它自己乘 3 之后的向量之间，间隙是 0、两个散布相等；单位向量的散布等于根号下（1 减去中心长度的平方）",
        "同一组向量和它的 3 倍之间的间隙": r3(gap), "这组向量的散布": r3(su),
    }


def main() -> None:
    S.setup()
    plt.rcParams["svg.hashsalt"] = "ch28"
    t_all = time.perf_counter()
    train_all()
    print(f"训练完成：{len(JOBS)} 个模型，{TIMES['train_wall']:.0f} 秒", flush=True)
    for fn in (numbers_misc, fig_why, fig_templates, fig_compose, fig_gap, fig_detect, fig_adapt, numbers_bugs):
        t = time.perf_counter()
        fn()
        TIMES[fn.__name__] = time.perf_counter() - t
        print(f"完成 {fn.__name__}（{TIMES[fn.__name__]:.0f} 秒）", flush=True)
    total = time.perf_counter() - t_all

    NUM["训练规模"] = {
        "说明": "整个脚本训练的模型个数。除“有监督对照”只有图像塔外，都是上面那个双塔小模型",
        "图 1 的模型（温度可学习）": len(SEEDS), "加难负样本": len(SEEDS), "训练时拿掉 4 种组合": 2, "句子里从不写纹理": 2, "词袋文本塔": 2, "固定温度": 3, "有监督对照": 2,
        "训练 2400 步": len(SEEDS), "训练 2400 步并加难负样本": len(SEEDS),
        "合计": len(JOBS), "区域头（只训练一个线性层）": len(SEEDS),
        "每个模型的训练步数（除两组 2400 步的）": STEPS, "批大小": BATCH, "每个模型见过的图文对数": STEPS * BATCH,
    }

    def measure():
        two = [LOGS[(n, s)]["seconds"] for n, s, kw in JOBS if n in ("main", "hold", "notex")]
        return {
            "说明": "在一台 Mac 的 CPU 上（训练分给 3 个进程，每个进程 1 个线程，同时还有别的程序在跑）测的一次结果，只看量级，不是显卡上的数字。设置 CH28_RETIME=1 重测",
            "双塔模型每步训练耗时（毫秒，批大小 64，含生成数据）": float(f"{1000 * np.mean(two) / STEPS:.2g}"),
            "训练一个双塔模型 1200 步（秒）": float(f"{np.mean(two):.2g}"),
            "整个配图脚本（秒）": float(f"{total:.2g}"),
        }

    NUM["耗时"] = S.pinned(CH, "耗时", measure, "CH28_RETIME")
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=2))
    print(f"总耗时 {total:.0f} 秒")


if __name__ == "__main__":
    main()
