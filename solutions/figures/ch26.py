"""第 26 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch26.py

输出：<主分支>/26_自监督基础模型/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。

全部实验用代码生成的 32x32 条纹小图和一个很小的 ViT（8x8 的块，16 个块，宽 64，2 层），在 CPU 上跑，
不下载任何数据和权重。随机遮挡、重建损失、教师的滑动平均、自蒸馏损失和 k 近邻分类调用本章作业的参考实现。
耗时默认沿用上次的记录，设置 CH26_RETIME=1 才重测。
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

CH = "26_自监督基础模型"
hw = S.load_solution(CH, "26-1_掩码建模与自蒸馏.py")
hw19 = S.load_solution("19_异常检测与深度缺陷检测", "19-1_PatchCore核心.py")  # 第 5 节的异常检测沿用第 19 章的 knn_scores 和 auroc
NUM: dict = {}
TIMES: dict = {}
SEEDS = (0, 1)
SIZE, P = 32, 8
G = SIZE // P
L = G * G
ANGLES = (0.0, 45.0, 90.0, 135.0)
PERIODS = (6.0, 9.0, 14.0)
NA, NB = len(ANGLES), len(PERIODS)
NOISE = 0.05
K_OUT = 32
MAE_STEPS, DINO_STEPS, SUP_STEPS = 800, 600, 400
N_PER, N_TEST, KNN_K = 3, 600, 5
YY, XX = torch.meshgrid(torch.arange(SIZE, dtype=torch.float32), torch.arange(SIZE, dtype=torch.float32), indexing="ij")
DATA_NOTE = (f"{SIZE}x{SIZE} 的条纹小图：方向 4 种（0、45、90、135 度），周期 3 种（6、9、14 px），一共 12 类；每张图的相位随机，"
             f"条纹幅度在 0.15 到 0.3、平均亮度在 0.4 到 0.6 之间均匀取，像素噪声标准差 {NOISE}")
MODEL_NOTE = (f"编码器是一个很小的 ViT：{P}x{P} 的块（每张图 {L} 个），宽 64，2 层，4 个头，固定的正弦位置编码，带一个类别标记；"
              "输入先减 0.5 再除以 0.25。优化器 AdamW（权重衰减 0.05），学习率先升后降（OneCycle），批大小 32")
EVAL_NOTE = (f"评价时编码器冻结。带标签的图每个“方向 x 周期”组合 {N_PER} 张（共 {N_PER * NA * NB} 张），测试图 {N_TEST} 张；"
             f"k 近邻用作业的 knn_classify（k={KNN_K}，温度 0.07），线性探测是特征标准化后带 L2 的最小二乘；每个模型换 2 套带标签的图取平均")


def r3(v):
    return round(float(v), 3)


def r2(v):
    return round(float(v), 2)


def gen(*seed):
    return torch.Generator().manual_seed(int(sum(s * 1000 ** i for i, s in enumerate(seed))))


# ---------------------------------------------------------------- 数据
def latent(n, g, ya=None, yb=None):
    """每张图的隐变量：方向类别、周期类别、相位、条纹幅度、平均亮度。"""
    ya = torch.randint(0, NA, (n,), generator=g) if ya is None else ya
    yb = torch.randint(0, NB, (n,), generator=g) if yb is None else yb
    return {"ya": ya, "yb": yb, "ph": torch.rand(n, generator=g) * 2 * math.pi,
            "amp": 0.15 + 0.15 * torch.rand(n, generator=g), "br": 0.4 + 0.2 * torch.rand(n, generator=g)}


def draw(z, g, shift=0.0, jitter=0.0, noise=NOISE, angles=ANGLES, periods=PERIODS):
    """按隐变量画图。shift、jitter 大于 0 时是一次增强：整体平移 shift 像素以内、亮度整体加减 jitter 以内的常数。"""
    n = len(z["ya"])
    th = torch.tensor(angles)[z["ya"]] * math.pi / 180
    per = torch.tensor(periods)[z["yb"]]
    dx = shift * (2 * torch.rand(n, generator=g) - 1)
    dy = shift * (2 * torch.rand(n, generator=g) - 1)
    arg = 2 * math.pi * (torch.cos(th)[:, None, None] * (XX + dx[:, None, None]) + torch.sin(th)[:, None, None] * (YY + dy[:, None, None])) / per[:, None, None]
    img = z["br"][:, None, None] + z["amp"][:, None, None] * torch.sin(arg + z["ph"][:, None, None])
    img = img + jitter * (2 * torch.rand(n, 1, 1, generator=g) - 1)
    return img + noise * torch.randn(n, SIZE, SIZE, generator=g)


def patchify(img):
    n = img.shape[0]
    return img.reshape(n, G, P, G, P).permute(0, 1, 3, 2, 4).reshape(n, L, P * P)


def unpatchify(tok):
    n = tok.shape[0]
    return tok.reshape(n, G, G, P, P).permute(0, 1, 3, 2, 4).reshape(n, SIZE, SIZE)


def normed(img):
    return (img - 0.5) / 0.25


# ---------------------------------------------------------------- 模型
def sincos_2d(dim, grid):
    def one(d, pos):
        omega = 1.0 / 100 ** (torch.arange(d // 2, dtype=torch.float32) / (d / 2))
        out = pos[:, None] * omega[None]
        return torch.cat([out.sin(), out.cos()], 1)
    ys, xs = torch.meshgrid(torch.arange(grid, dtype=torch.float32), torch.arange(grid, dtype=torch.float32), indexing="ij")
    return torch.cat([one(dim // 2, ys.reshape(-1)), one(dim // 2, xs.reshape(-1))], 1)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        return self.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, dim, heads, ratio=2):
        super().__init__()
        self.n1, self.attn, self.n2 = nn.LayerNorm(dim), Attention(dim, heads), nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, ratio * dim), nn.GELU(), nn.Linear(ratio * dim, dim))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class Encoder(nn.Module):
    """很小的 ViT：块的线性嵌入，固定的正弦位置编码，一个类别标记，depth 个预归一化的 Transformer 块。"""

    def __init__(self, dim=64, depth=2, heads=4):
        super().__init__()
        self.embed = nn.Linear(P * P, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls, std=0.02)
        self.register_buffer("pos", sincos_2d(dim, G)[None])
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def tokens(self, img):
        return self.embed(patchify(normed(img))) + self.pos

    def run(self, tok):
        x = torch.cat([self.cls.expand(len(tok), -1, -1), tok], 1)
        for b in self.blocks:
            x = b(x)
        return self.norm(x)

    def forward(self, img):
        return self.run(self.tokens(img))


class MAE(nn.Module):
    """编码器只处理可见块；解码器更窄、只有 1 层，输入是编码结果加掩码标记，按 ids_restore 还原顺序后加位置编码。"""

    def __init__(self, dim=64, depth=2, heads=4, ddim=48, ddepth=1):
        super().__init__()
        self.enc = Encoder(dim, depth, heads)
        self.proj = nn.Linear(dim, ddim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, ddim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer("dpos", sincos_2d(ddim, G)[None])
        self.dblocks = nn.ModuleList([Block(ddim, heads) for _ in range(ddepth)])
        self.dnorm = nn.LayerNorm(ddim)
        self.out = nn.Linear(ddim, P * P)

    def forward(self, img, ratio, noise):
        tok = self.enc.tokens(img)
        keep, mask, restore = hw.random_masking(tok, ratio, noise)
        z = self.proj(self.enc.run(keep))
        full = torch.cat([z[:, 1:], self.mask_token.expand(len(z), L - keep.shape[1], -1)], 1)
        full = torch.gather(full, 1, restore[..., None].expand(-1, -1, full.shape[-1])) + self.dpos
        x = torch.cat([z[:, :1], full], 1)
        for b in self.dblocks:
            x = b(x)
        return self.out(self.dnorm(x))[:, 1:], mask


class Head(nn.Module):
    """DINO 式的投影头：MLP、L2 归一化、再和 K 个归一化的原型做内积，输出落在 -1 到 1 之间。"""

    def __init__(self, dim, k=K_OUT, hidden=64, bott=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, bott))
        self.last = nn.Linear(bott, k, bias=False)

    def forward(self, x):
        return F.normalize(self.mlp(x), dim=-1) @ F.normalize(self.last.weight, dim=-1).t()


class DinoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.head = Head(self.enc.dim)

    def forward(self, img):
        return self.head(self.enc(img)[:, 0])


def n_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def optim(params, lr, steps):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
    return opt, torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)


# ---------------------------------------------------------------- 训练
def train_mae(seed, ratio=0.75, steps=MAE_STEPS, batch=32, lr=3e-3, loss_all=False, norm_pix=False):
    torch.manual_seed(seed)
    g = gen(seed, 1)
    net = MAE()
    opt, sched = optim(net.parameters(), lr, steps)
    for _ in range(steps):
        img = draw(latent(batch, g), g)
        pred, mask = net(img, ratio, torch.rand(batch, L, generator=g))
        loss = hw.masked_mse_loss(pred, patchify(normed(img)), torch.ones_like(mask) if loss_all else mask, norm_pix)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return net.eval()


@torch.no_grad()
def mae_errors(net, ratio, seed, n=1000, norm_pix=False):
    """在新图上量重建误差：被遮块和可见块分开算，单位是“每个像素都预测成全体的平均值”的误差的倍数。"""
    g = gen(seed, 7)
    img = draw(latent(n, g), g)
    pred, mask = net(img, ratio, torch.rand(n, L, generator=g))
    t = patchify(normed(img))
    if norm_pix:
        t = (t - t.mean(-1, keepdim=True)) / (t.var(-1, keepdim=True) + 1e-6).sqrt()
    err = ((pred - t) ** 2).mean(-1)
    base = float(((t - t.mean()) ** 2).mean())
    vis = float((err * (1 - mask)).sum() / (1 - mask).sum())
    return float((err * mask).sum() / mask.sum()) / base, vis / base


def entropy(p):
    return float(-(p * (p + 1e-12).log()).sum(-1).mean())


def train_dino(seed, steps=DINO_STEPS, batch=32, lr=2e-3, momentum=0.99, t_temp=0.04, s_temp=0.1, center=True, c_mom=0.9, shift=2.0, jitter=0.1, every=10):
    """返回学生、教师和训练过程的记录。两个视图是同一张图的两次增强；教师的输出不参与求导，每一步用学生的参数做滑动平均。"""
    torch.manual_seed(seed)
    g = gen(seed, 2)
    student, teacher = DinoNet(), DinoNet()
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt, sched = optim(student.parameters(), lr, steps)
    c = torch.zeros(1, K_OUT)
    log = {"step": [], "loss": [], "H": [], "Hbar": [], "std": [], "top": []}
    t0 = time.perf_counter()
    for step in range(steps):
        z = latent(batch, g)
        v1, v2 = draw(z, g, shift, jitter), draw(z, g, shift, jitter)
        s1, s2 = student(v1), student(v2)
        with torch.no_grad():
            t1, t2 = teacher(v1), teacher(v2)
        loss = 0.5 * (hw.dino_loss(s1, t2, c, s_temp, t_temp) + hw.dino_loss(s2, t1, c, s_temp, t_temp))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        hw.ema_update(teacher, student, momentum)
        tt = torch.cat([t1, t2])
        if step % every == 0 or step == steps - 1:
            q = F.softmax((tt - c) / t_temp, -1)
            for key, val in (("step", step), ("loss", float(loss.detach())), ("H", entropy(q)), ("Hbar", entropy(q.mean(0, keepdim=True))),
                             ("std", float(tt.std(0).mean())), ("top", float(q.mean(0).max()))):
                log[key].append(val)
        if center:
            c = c_mom * c + (1 - c_mom) * tt.mean(0, keepdim=True)
    log["seconds"] = time.perf_counter() - t0
    log["mean_dist"] = np.sort(F.softmax((tt - c) / t_temp, -1).mean(0).numpy())[::-1].copy()
    return student.eval(), teacher.eval(), log


def train_sup(seed, task="a", steps=SUP_STEPS, batch=32, lr=2e-3, shift=2.0, jitter=0.1, pool=None):
    """有监督训练。pool 为 None 时每一步都是新图（上游预训练，标签不限量）；给了 pool 就只在这些带标签的图上训练（从头训练的基线）。
    两种情况都用和自蒸馏相同的增强。"""
    torch.manual_seed(seed)
    g = gen(seed, 3)
    enc = Encoder()
    head = nn.Linear(enc.dim, {"a": NA, "b": NB, "ab": NA * NB}[task])
    opt, sched = optim([*enc.parameters(), *head.parameters()], lr, steps)
    for _ in range(steps):
        if pool is None:
            z = latent(batch, g)
        else:
            idx = torch.randint(0, len(pool["ya"]), (batch,), generator=g)
            z = {k: v[idx] for k, v in pool.items()}
        y = {"a": z["ya"], "b": z["yb"], "ab": z["ya"] * NB + z["yb"]}[task]
        loss = F.cross_entropy(head(enc(draw(z, g, shift, jitter))[:, 0]), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return enc.eval(), head.eval()


_CACHE: dict = {}


def cached(key, make):
    if key not in _CACHE:
        _CACHE[key] = make()
    return _CACHE[key]


def get_mae(seed, ratio=0.75, **kw):
    return cached(("mae", seed, ratio, tuple(sorted(kw.items()))), lambda: train_mae(seed, ratio, **kw))


def get_dino(seed, **kw):
    return cached(("dino", seed, tuple(sorted(kw.items()))), lambda: train_dino(seed, **kw))


def get_sup(seed, task):
    return cached(("sup", seed, task), lambda: train_sup(seed, task)[0])


def get_rand(seed):
    def make():
        torch.manual_seed(100 + seed)
        return Encoder().eval()
    return cached(("rand", seed), make)


# ---------------------------------------------------------------- 评价
@torch.no_grad()
def feats(enc, img, kind="cls"):
    """冻结的编码器给出的整图特征：类别标记（cls）或者全部块特征的平均（mean）。enc 为 None 时直接用展平的像素。"""
    if enc is None:
        return img.reshape(len(img), -1).numpy().astype(np.float64)
    out = torch.cat([enc(img[i:i + 500]) for i in range(0, len(img), 500)])
    return (out[:, 0] if kind == "cls" else out[:, 1:].mean(1)).numpy().astype(np.float64)


def linear_probe(ftr, ytr, fte, yte, classes, lam=1e-2):
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    a = np.c_[(ftr - mu) / sd, np.ones(len(ftr))]
    b = np.c_[(fte - mu) / sd, np.ones(len(fte))]
    w = np.linalg.solve(a.T @ a + lam * len(a) * np.eye(a.shape[1]), a.T @ np.eye(classes)[ytr])
    return float(((b @ w).argmax(1) == yte).mean())


def knn_acc(ftr, ytr, fte, yte, classes, k=KNN_K):
    return float((hw.knn_classify(ftr, ytr, fte, min(k, len(ftr)), classes) == yte).mean())


def retrieval_p5(f, y, k=5):
    """检索：每张测试图在其余测试图里找最像的 5 张，同类的比例。"""
    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
    sim = f @ f.T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    return float((y[idx] == y[:, None]).mean())


def labeled_set(seed, n_per):
    def make():
        g = gen(seed, 5, n_per)
        cls = torch.arange(NA * NB).repeat_interleave(n_per)
        z = latent(len(cls), g, cls // NB, cls % NB)
        return z, draw(z, g)
    return cached(("labeled", seed, n_per), make)


def test_set(seed):
    def make():
        g = gen(seed, 6)
        z = latent(N_TEST, g)
        return z, draw(z, g)
    return cached(("test", seed), make)


TASKS = {"a": ("方向（4 类）", NA), "b": ("周期（3 类）", NB), "ab": ("方向 x 周期（12 类）", NA * NB)}


def labels_of(z, task):
    return {"a": z["ya"], "b": z["yb"], "ab": z["ya"] * NB + z["yb"]}[task].numpy()


def evaluate(enc, kind="cls", n_per=N_PER, tasks=("a", "b", "ab"), label_seeds=SEEDS):
    """返回 {任务: [线性探测, k 近邻, 检索 P@5]}，对 label_seeds 取平均。"""
    out = {t: [] for t in tasks}
    for s in label_seeds:
        ztr, xtr = labeled_set(s, n_per)
        zte, xte = test_set(s)
        a, b = feats(enc, xtr, kind), feats(enc, xte, kind)
        for t in tasks:
            ytr, yte, nc = labels_of(ztr, t), labels_of(zte, t), TASKS[t][1]
            out[t].append([linear_probe(a, ytr, b, yte, nc), knn_acc(a, ytr, b, yte, nc), retrieval_p5(b, yte)])
    return {t: np.mean(v, axis=0) for t, v in out.items()}


def eval_models(encs, **kw):
    """对几个种子训练出来的编码器分别评价。返回 (平均, 各种子)，结构同 evaluate。"""
    each = [evaluate(e, **kw) for e in encs]
    return {t: np.mean([e[t] for e in each], axis=0) for t in each[0]}, each


# ---------------------------------------------------------------- 图 1：标签很少时的几种做法
def fig_few_labels():
    ns = [1, 2, 5, 10, 20]
    names = ["原始像素，k 近邻", "随机初始化的编码器，k 近邻", "只用这些标签从头训练（要训练，带增强）", "MAE 的编码器，k 近邻",
             "自蒸馏的编码器，k 近邻", "自蒸馏的编码器，线性头", "有监督预训练（只有方向标签）的编码器，k 近邻"]
    res = {n: [] for n in names}
    spread = {n: [] for n in names}
    for n_per in ns:
        rows = {n: [] for n in names}
        for s in SEEDS:
            ztr, xtr = labeled_set(s, n_per)
            zte, xte = test_set(s)
            ytr, yte = labels_of(ztr, "ab"), labels_of(zte, "ab")
            rows[names[0]].append(knn_acc(feats(None, xtr), ytr, feats(None, xte), yte, 12))
            enc, head = train_sup(s, "ab", steps=300, pool=ztr)
            with torch.no_grad():
                rows[names[2]].append(float((head(enc(xte)[:, 0]).argmax(1).numpy() == yte).mean()))
            for ms in SEEDS:
                for name, e in ((names[1], get_rand(ms)), (names[3], get_mae(ms).enc), (names[4], get_dino(ms)[0].enc), (names[6], get_sup(ms, "a"))):
                    rows[name].append(knn_acc(feats(e, xtr), ytr, feats(e, xte), yte, 12))
                e = get_dino(ms)[0].enc
                rows[names[5]].append(linear_probe(feats(e, xtr), ytr, feats(e, xte), yte, 12))
        for n in names:
            res[n].append(float(np.mean(rows[n])))
            spread[n].append([r3(min(rows[n])), r3(max(rows[n]))])
    NUM["标签很少时的几种做法"] = {
        "说明": DATA_NOTE + "。任务是 12 类分类（方向 x 周期），测试图 600 张，随机猜是 1/12。" + MODEL_NOTE + "。"
              f"MAE（遮挡比例 0.75，{MAE_STEPS} 步）和自蒸馏（{DINO_STEPS} 步）的预训练只用无标签的图，各 2 个种子；带标签的图换 2 套；冻结特征的每个数是 2x2 次的平均。"
              "从头训练：同样的 ViT 加线性头，只在这些带标签的图上训练 300 步，用和自蒸馏相同的增强（平移 2 px 以内、亮度抖动 0.1 以内），2 套标签各训练一次。"
              f"有监督预训练（只有方向标签）：同样的编码器在不限量的带标签新图上训练 {SUP_STEPS} 步，标签只有 4 种方向，然后冻结。除“从头训练”外，其余各行的编码器都是冻结的。"
              f"k 近邻的 k 取 {KNN_K} 和带标签图数里较小的那个，特征用类别标记",
        "每类带标签的图数": ns, "带标签的图总数": [12 * n for n in ns], "随机猜": r3(1 / 12),
        **{name: [r3(v) for v in res[name]] for name in names},
        **{f"{name}：各次的最小和最大": spread[name] for name in names},
    }

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.0, 1.55]}, layout="constrained")
    g = gen(26)
    cls = torch.arange(12)
    z = latent(12, g, cls // NB, cls % NB)
    imgs = draw(z, g).numpy()
    tile = np.full((NB * (SIZE + 2) + 2, NA * (SIZE + 2) + 2), 1.0)
    for i in range(12):
        a, b = i // NB, i % NB
        tile[2 + b * (SIZE + 2):2 + b * (SIZE + 2) + SIZE, 2 + a * (SIZE + 2):2 + a * (SIZE + 2) + SIZE] = imgs[i]
    ax[0].imshow(tile, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("12 类条纹：每列一个方向，每行一个周期")
    ax[0].set_xticks([2 + SIZE / 2 + a * (SIZE + 2) for a in range(NA)], [f"{int(v)} 度" for v in ANGLES])
    ax[0].set_yticks([2 + SIZE / 2 + b * (SIZE + 2) for b in range(NB)], [f"周期 {int(v)} px" for v in PERIODS])
    ax[0].tick_params(length=0)
    for side in ax[0].spines.values():
        side.set_visible(False)
    styles = [("s--", S.MUTED), ("v:", "#9aa0a8"), ("x:", S.RED), ("^-", S.GREEN), ("o-", S.BLUE), ("o--", S.BLUE), ("D-", S.ORANGE)]
    for name, (style, color) in zip(names, styles):
        ax[1].plot(ns, res[name], style, color=color, lw=1.2 if name == names[2] else 1.8, ms=5.5, alpha=0.7 if name in (names[2], names[5]) else 1.0, label=name)
    ax[1].axhline(1 / 12, color=S.MUTED, lw=1, ls=":")
    ax[1].text(ns[-1], 1 / 12 + 0.015, "随机猜", ha="right", fontsize=8.5, color=S.MUTED)
    ax[1].set_xscale("log")
    ax[1].set_xticks(ns, [str(n) for n in ns])
    ax[1].minorticks_off()
    ax[1].set_ylim(0, 1.03)
    ax[1].set_xlabel("每类带标签的图数")
    ax[1].set_ylabel("12 类分类的测试精度")
    ax[1].set_title("同样是冻结特征加 k 近邻，特征是怎么来的差别很大")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    S.clean(ax[1])
    S.save(fig, CH, "fig01_标签很少时的几种做法")


# ---------------------------------------------------------------- 图 2：遮挡、重建，损失算在哪些块上
def fig_mae_recon():
    g = gen(27)
    z = latent(3, g, torch.tensor([0, 1, 2]), torch.tensor([2, 1, 0]))
    img = draw(z, g)
    noise = torch.rand(3, L, generator=g)
    rows = []
    for net in (get_mae(0), get_mae(0, loss_all=True)):
        with torch.no_grad():
            pred, mask = net(img, 0.75, noise)
        rows.append((unpatchify(pred) * 0.25 + 0.5, unpatchify(mask[..., None].expand(-1, -1, P * P))))
    (rec_m, m), (rec_a, _) = rows
    seen = torch.where(m > 0.5, torch.full_like(img, float("nan")), img)
    pasted = torch.where(m > 0.5, rec_m, img)
    stats = {}
    for name, kw in (("只在被遮块上算", {}), ("在全部块上算", {"loss_all": True})):
        e = np.array([mae_errors(get_mae(s, **kw), 0.75, s) for s in SEEDS])
        stats[name] = {"被遮块的重建误差": r3(e[:, 0].mean()), "可见块的重建误差": r3(e[:, 1].mean()),
                       "各种子（被遮块，可见块）": [[r3(v) for v in row] for row in e]}
    NUM["损失算在哪些块上：重建误差"] = {
        "说明": DATA_NOTE + "。" + MODEL_NOTE + f"；解码器宽 48、1 层。遮挡比例 0.75（16 个块遮 12 个），训练 {MAE_STEPS} 步，2 个种子取平均。"
              "重建误差在 1000 张新图上量，单位是“每个像素都预测成全体像素的平均值”时的误差的倍数：1 表示和只猜平均值一样，0 表示完全重建。"
              "可见块的误差指解码器在没有被遮住的位置上的输出和原图的差",
        **stats,
    }

    cols = [("原图", img), ("编码器看到的 4 个块", seen), ("解码器的输出\n（损失只算被遮块）", rec_m), ("同左，把可见块贴回去", pasted),
            ("解码器的输出\n（损失算全部块）", rec_a)]
    fig, ax = plt.subplots(3, 5, figsize=(11.5, 7.0), layout="constrained")
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#f3b98f")
    for j, (title, data) in enumerate(cols):
        for i in range(3):
            ax[i, j].imshow(data[i].numpy(), cmap=cmap, vmin=0, vmax=1)
            for k in range(1, G):
                ax[i, j].axhline(k * P - 0.5, color="white", lw=0.4, alpha=0.5)
                ax[i, j].axvline(k * P - 0.5, color="white", lw=0.4, alpha=0.5)
            S.image_axes(ax[i, j])
        ax[0, j].set_title(title, fontsize=10.5)
    fig.suptitle("损失只算被遮块时，解码器在可见位置上的输出是乱的；贴回可见块才是常见的那种重建图", fontsize=12, fontweight="bold")
    S.save(fig, CH, "fig02_遮挡与重建")


# ---------------------------------------------------------------- 图 3：遮挡比例、损失算在哪、目标是否按块归一化
def fig_mae_ratio():
    ratios = [0.25, 0.5, 0.75, 0.875]
    err, acc = [], {"cls": [], "mean": []}
    for ratio in ratios:
        nets = [get_mae(s, ratio) for s in SEEDS]
        err.append([mae_errors(n, ratio, s) for n, s in zip(nets, SEEDS)])
        for kind in acc:
            acc[kind].append(eval_models([n.enc for n in nets], kind=kind))
    err = np.array(err)

    def table(kind, col):
        return {TASKS[t][0]: [r3(a[0][t][col]) for a in acc[kind]] for t in ("a", "b", "ab")}

    def seeds(kind, col, t):
        return [[r3(e[t][col]) for e in a[1]] for a in acc[kind]]

    NUM["遮挡比例"] = {
        "说明": DATA_NOTE + "。" + MODEL_NOTE + f"；解码器宽 48、1 层，损失只算被遮块，目标不做按块归一化。每个比例训练 {MAE_STEPS} 步，2 个种子取平均。"
              "重建误差的单位同上（只猜平均值为 1）。" + EVAL_NOTE,
        "遮挡比例": ratios, "被遮住的块数": [int(round(L * r)) for r in ratios], "编码器处理的块数": [L - int(round(L * r)) for r in ratios],
        "自注意力的计算量相对不遮挡（按块数的平方算）": [r3(((L - round(L * r)) / L) ** 2) for r in ratios],
        "被遮块的重建误差": [r3(v) for v in err[:, :, 0].mean(1)],
        "被遮块的重建误差（各种子）": [[r3(v) for v in row] for row in err[:, :, 0]],
        "类别标记，k 近邻": table("cls", 1), "类别标记，线性探测": table("cls", 0),
        "块特征平均，k 近邻": table("mean", 1), "块特征平均，线性探测": table("mean", 0),
        "各种子：类别标记的 k 近邻，方向": seeds("cls", 1, "a"), "各种子：类别标记的 k 近邻，周期": seeds("cls", 1, "b"),
        "各种子：块特征平均的 k 近邻，方向": seeds("mean", 1, "a"), "各种子：块特征平均的 k 近邻，周期": seeds("mean", 1, "b"),
        "各种子：类别标记的 k 近邻，方向 x 周期": seeds("cls", 1, "ab"), "各种子：类别标记的线性探测，方向": seeds("cls", 0, "a"),
        "各种子：类别标记的线性探测，周期": seeds("cls", 0, "b"),
    }

    variants = [("只算被遮块（默认）", {}), ("算全部块", {"loss_all": True}), ("目标按块归一化", {"norm_pix": True})]
    var = {}
    for name, kw in variants:
        nets = [get_mae(s, **kw) for s in SEEDS]
        e = np.array([mae_errors(n, 0.75, s, norm_pix=kw.get("norm_pix", False)) for n, s in zip(nets, SEEDS)])
        var[name] = {"err": e, **{kind: eval_models([n.enc for n in nets], kind=kind) for kind in ("cls", "mean")}}
    NUM["损失设计的两个选择"] = {
        "说明": f"遮挡比例 0.75，其余同上，2 个种子取平均。“算全部块”指可见块和被遮块一起算均方误差；“目标按块归一化”指每个块的目标各自减均值、除以标准差（作业的 norm_pix=True），"
              "它的重建误差是在归一化后的目标上量的，和另外两行不能直接比。精度是 k 近邻 / 线性探测",
        **{name: {
            "被遮块的重建误差": r3(v["err"][:, 0].mean()), "可见块的重建误差": r3(v["err"][:, 1].mean()),
            **{f"{label}，{TASKS[t][0]}：k 近邻": r3(v[kind][0][t][1]) for kind, label in (("cls", "类别标记"), ("mean", "块特征平均")) for t in ("a", "b")},
            **{f"{label}，{TASKS[t][0]}：线性探测": r3(v[kind][0][t][0]) for kind, label in (("cls", "类别标记"), ("mean", "块特征平均")) for t in ("a", "b")},
            **{f"各种子：{label}的 k 近邻，{TASKS[t][0]}": [r3(e[t][1]) for e in v[kind][1]] for kind, label in (("cls", "类别标记"), ("mean", "块特征平均")) for t in ("a", "b")},
        } for name, v in var.items()},
    }

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1, 1.25, 1.25]}, layout="constrained")
    xs = np.arange(len(ratios))
    for j in range(len(SEEDS)):
        ax[0].plot(xs, err[:, j, 0], "o", color=S.BLUE, ms=3.5, alpha=0.35)
    ax[0].plot(xs, err[:, :, 0].mean(1), "o-", color=S.BLUE, lw=1.9, ms=6)
    ax[0].axhline(1.0, color=S.MUTED, lw=1, ls=":")
    ax[0].text(0.02, 1.02, "只猜平均值", fontsize=8.5, color=S.MUTED, va="bottom")
    ax[0].set_xticks(xs, [f"{r}\n看 {L - int(round(L * r))} 块" for r in ratios])
    ax[0].set_ylim(0, 1.12)
    ax[0].set_xlabel("遮挡比例")
    ax[0].set_ylabel("被遮块的重建误差（只猜平均值 = 1）")
    ax[0].set_title("遮得越多，被遮块越难重建")
    S.clean(ax[0])
    for kind, ls, label in (("cls", "-", "类别标记"), ("mean", "--", "块特征平均")):
        for t, color, marker in (("a", S.BLUE, "o"), ("b", S.ORANGE, "s")):
            ax[1].plot(xs, [a[0][t][1] for a in acc[kind]], marker + ls, color=color, lw=1.8, ms=5.5, label=f"{label}：{TASKS[t][0]}")
            for j in range(len(SEEDS)):
                ax[1].plot(xs, [a[1][j][t][1] for a in acc[kind]], marker, color=color, ms=3, alpha=0.3)
    ax[1].set_xticks(xs, [str(r) for r in ratios])
    ax[1].set_ylim(0.3, 1.0)
    ax[1].set_xlabel("遮挡比例")
    ax[1].set_ylabel(f"冻结特征的 k 近邻精度（每类 {N_PER} 张标签）")
    ax[1].set_title("表示质量：0.75 最好或并列最好，只剩 2 块时方向的精度明显下降", fontsize=10.5)
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[1])
    groups = [("类别标记\n方向", "cls", "a"), ("类别标记\n周期", "cls", "b"), ("块特征平均\n方向", "mean", "a"), ("块特征平均\n周期", "mean", "b")]
    gx = np.arange(len(groups))
    for i, ((name, _), color) in enumerate(zip(variants, (S.BLUE, S.MUTED, S.GREEN))):
        vals = [var[name][kind][0][t][1] for _, kind, t in groups]
        bars = ax[2].bar(gx + (i - 1) * 0.27, vals, 0.25, color=color, alpha=0.88, label=name)
        for b, v, (_, kind, t) in zip(bars, vals, groups):
            pts = [e[t][1] for e in var[name][kind][1]]
            ax[2].text(b.get_x() + b.get_width() / 2, max(pts) + 0.015, f"{v:.2f}", ha="center", fontsize=8)
            ax[2].plot([b.get_x() + b.get_width() / 2] * len(pts), pts, ".", color=S.INK, ms=4)
    ax[2].set_xticks(gx, [g[0] for g in groups])
    ax[2].set_ylim(0.3, 1.0)
    ax[2].set_ylabel("k 近邻精度（遮挡比例 0.75）")
    ax[2].set_title("按块归一化：块特征平均变好，类别标记变差")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    S.clean(ax[2])
    ax[2].grid(axis="x", visible=False)
    S.save(fig, CH, "fig03_遮挡比例与损失设计")


# ---------------------------------------------------------------- 表：两种实现错误训练出来是什么样
def numbers_bugs():
    """把 random_masking 换成两种写错的版本各训练一次 MAE，看训练时显示的损失、真正被遮块上的误差和表示质量。"""
    orig = hw.random_masking

    def wrong_restore(x, ratio, noise):
        keep, mask, _ = orig(x, ratio, noise)
        return keep, mask, torch.argsort(noise, dim=1)        # 返回的是排序下标本身，不是每个块的名次

    def flipped(x, ratio, noise):
        keep, mask, restore = orig(x, ratio, noise)
        return keep, 1 - mask, restore                         # 1 表示可见

    rows = {}
    for name, fn in (("正确的实现", orig), ("还原下标返回了排序下标本身", wrong_restore), ("掩码反了（1 表示可见）", flipped)):
        errs, accs = [], []
        for seed in SEEDS:
            hw.random_masking = fn
            try:
                net = get_mae(seed) if fn is orig else train_mae(seed)
                g = gen(seed, 7)
                img = draw(latent(1000, g), g)
                noise = torch.rand(1000, L, generator=g)
                with torch.no_grad():
                    pred, shown = net(img, 0.75, noise)
            finally:
                hw.random_masking = orig
            mask = orig(torch.zeros(1000, L, 1), 0.75, noise)[1]
            t = patchify(normed(img))
            err = ((pred - t) ** 2).mean(-1)
            base = float(((t - t.mean()) ** 2).mean())
            errs.append([float((err * m).sum() / m.sum()) / base for m in (shown, mask, 1 - mask)])
            ev = evaluate(net.enc, tasks=("a", "b"))
            accs.append([ev["a"][1], ev["b"][1]])
        e, a = np.mean(errs, axis=0), np.mean(accs, axis=0)
        rows[name] = {"训练时显示的损失（按这份实现自己的掩码算）": r3(e[0]), "真正被遮住的块上的误差": r3(e[1]), "真正可见的块上的误差": r3(e[2]),
                      "类别标记的 k 近邻：方向": r3(a[0]), "类别标记的 k 近邻：周期": r3(a[1]),
                      "各种子（真正被遮块的误差，方向 k 近邻，周期 k 近邻）": [[r3(x[1]), r3(y[0]), r3(y[1])] for x, y in zip(errs, accs)]}
    NUM["两种实现错误的后果"] = {
        "说明": f"把作业的 random_masking 换成写错的版本训练 MAE（遮挡比例 0.75，{MAE_STEPS} 步，2 个种子取平均），其余代码不变。误差的单位同上（只猜平均值为 1），在 1000 张新图上量；"
              "“真正被遮住”按正确的掩码定义统计。" + EVAL_NOTE,
        **rows,
    }


# ---------------------------------------------------------------- 图 4：自蒸馏的坍缩
COLLAPSE = [("默认（中心化 + 锐化）", {}, SEEDS, S.BLUE), ("去掉中心化", {"center": False}, SEEDS, S.RED),
            ("去掉锐化（教师温度 0.1）", {"t_temp": 0.1}, SEEDS, S.ORANGE), ("两个都去掉", {"center": False, "t_temp": 0.1}, (0,), S.PURPLE)]


def dino_summary(runs):
    """runs: [(student, teacher, log), ...]。最后 5 次记录（约 50 步）取平均，再对种子取平均。"""
    tail = {k: [float(np.mean(log[k][-5:])) for _, _, log in runs] for k in ("H", "Hbar", "std", "top", "loss")}
    ev, each = eval_models([st.enc for st, _, _ in runs], tasks=("a", "b"))
    return {
        "单个样本的熵 H": r3(np.mean(tail["H"])), "平均分布的熵": r3(np.mean(tail["Hbar"])),
        "两者之差（输出里关于输入的信息）": r3(np.mean(tail["Hbar"]) - np.mean(tail["H"])),
        "平均分布里最大的一维": r3(np.mean(tail["top"])), "教师输出各维在批内的标准差": r3(np.mean(tail["std"])),
        "损失": r3(np.mean(tail["loss"])),
        "学生编码器的 k 近邻：方向": r3(ev["a"][1]), "学生编码器的 k 近邻：周期": r3(ev["b"][1]),
        "各种子（H，平均分布的熵，方向 k 近邻，周期 k 近邻）": [[r3(h), r3(hb), r3(e["a"][1]), r3(e["b"][1])] for h, hb, e in zip(tail["H"], tail["Hbar"], each)],
    }, ev


def fig_dino_collapse():
    out, evs, logs = {}, {}, {}
    for name, kw, seeds, _ in COLLAPSE:
        runs = [get_dino(s, **kw) for s in seeds]
        out[name], evs[name] = dino_summary(runs)
        logs[name] = runs[0][2]
    rand_ev, _ = eval_models([get_rand(s) for s in SEEDS], tasks=("a", "b"))
    NUM["自蒸馏的坍缩"] = {
        "说明": DATA_NOTE + "。" + MODEL_NOTE + f"；投影头是两层 MLP 加 L2 归一化，再和 {K_OUT} 个归一化的原型做内积（输出 {K_OUT} 维，取值在 -1 到 1 之间）。"
              "两个视图是同一张图的两次增强（整体平移 2 px 以内、亮度加减 0.1 以内、各自的噪声）。学生温度 0.1，默认教师温度 0.04、教师动量 0.99、中心的动量 0.9，"
              f"训练 {DINO_STEPS} 步，损失、滑动平均都用作业的参考实现。熵按教师的目标分布（减中心、除以教师温度、softmax）在每个批的 64 个视图上算，单位是奈特；"
              "表里是最后约 50 步的平均，前三种设置 2 个种子取平均，“两个都去掉”只跑了 1 个种子。" + EVAL_NOTE,
        "输出维数 K": K_OUT, "ln K（均匀分布的熵）": r3(math.log(K_OUT)),
        **out,
        "对照：随机初始化的编码器的 k 近邻": {"方向": r3(rand_ev["a"][1]), "周期": r3(rand_ev["b"][1])},
    }

    fig, ax = plt.subplots(1, 4, figsize=(17, 4.1), gridspec_kw={"width_ratios": [1.15, 1.15, 1.0, 1.1]}, layout="constrained")
    for name, _, _, color in COLLAPSE:
        log = logs[name]
        ax[0].plot(log["step"], log["H"], color=color, lw=1.7, label=name)
        ax[1].plot(log["step"], np.array(log["Hbar"]) - np.array(log["H"]), color=color, lw=1.7, label=name)
    for a, title, ylabel in ((ax[0], "单个样本的熵：去掉中心化降到 0，去掉锐化升到 ln K", "单个样本的熵（奈特）"),
                             (ax[1], "两个熵的差：只有默认设置离开了 0", "平均分布的熵 - 单个样本的熵（奈特）")):
        a.axhline(math.log(K_OUT), color=S.MUTED, lw=1, ls=":")
        a.text(DINO_STEPS, math.log(K_OUT) + 0.05, "ln K", ha="right", fontsize=8.5, color=S.MUTED)
        a.set_ylim(-0.1, 3.85)
        a.set_xlabel("训练步数")
        a.set_ylabel(ylabel)
        a.set_title(title, fontsize=10.5)
        S.clean(a)
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    dims = np.arange(1, K_OUT + 1)
    for (name, _, _, color), off in zip(COLLAPSE[:3], (-0.28, 0.0, 0.28)):
        ax[2].bar(dims + off, np.maximum(logs[name]["mean_dist"], 1e-4), 0.28, color=color, alpha=0.9, label=name)
    ax[2].set_yscale("log")
    ax[2].set_ylim(1e-4, 1.5)
    ax[2].set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ["0.0001", "0.001", "0.01", "0.1", "1"])
    ax[2].minorticks_off()
    ax[2].set_xlabel(f"输出的 {K_OUT} 维，按占比从大到小排")
    ax[2].set_ylabel("训练结束时平均分布里各维的占比")
    ax[2].set_title("坍缩的两种样子：一维独占，或各维均分", fontsize=10.5)
    S.clean(ax[2])
    names = [c[0] for c in COLLAPSE]
    short = ["默认", "去掉\n中心化", "去掉\n锐化", "都去掉"]
    gx = np.arange(len(names))
    for i, (t, color) in enumerate((("a", S.BLUE), ("b", S.ORANGE))):
        vals = [evs[n][t][1] for n in names]
        bars = ax[3].bar(gx + (i - 0.5) * 0.36, vals, 0.34, color=color, alpha=0.88, label=TASKS[t][0])
        for b, v in zip(bars, vals):
            ax[3].text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5)
        ax[3].axhline(rand_ev[t][1], color=color, lw=1.1, ls=":")
    ax[3].plot([], [], color=S.MUTED, lw=1.1, ls=":", label="虚线：随机初始化的编码器")
    ax[3].set_xticks(gx, short)
    ax[3].set_ylim(0.3, 1.08)
    ax[3].set_ylabel("学生编码器的 k 近邻精度")
    ax[3].set_title("坍缩之后特征明显变差，但仍高于随机初始化", fontsize=10.5)
    ax[3].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0)
    S.clean(ax[3])
    ax[3].grid(axis="x", visible=False)
    S.save(fig, CH, "fig04_自蒸馏的两种坍缩")


# ---------------------------------------------------------------- 图 5：教师动量；通用特征的评价
def fig_momentum_eval():
    moms = [(0.0, (0,)), (0.9, SEEDS), (0.99, SEEDS), (0.999, SEEDS)]
    table, logs, acc = {}, {}, {}
    for m, seeds in moms:
        runs = [get_dino(s) if m == 0.99 else get_dino(s, momentum=m) for s in seeds]
        summ, _ = dino_summary(runs)
        ev_s, each_s = eval_models([st.enc for st, _, _ in runs], tasks=("a", "b"))
        ev_t, each_t = eval_models([te.enc for _, te, _ in runs], tasks=("a", "b"))
        peak = [float(np.max(log["top"])) for _, _, log in runs]
        acc[m] = (np.mean([ev_s["a"][1], ev_s["b"][1]]), np.mean([ev_t["a"][1], ev_t["b"][1]]),
                  [np.mean([e["a"][1], e["b"][1]]) for e in each_s], [np.mean([e["a"][1], e["b"][1]]) for e in each_t])
        logs[m] = runs[0][2]
        table[f"动量 {m}"] = {
            "种子数": len(seeds), "训练结束时单个样本的熵": summ["单个样本的熵 H"], "训练结束时平均分布的熵": summ["平均分布的熵"],
            "训练过程中平均分布里最大一维的峰值": r3(np.mean(peak)), "训练结束时的损失": summ["损失"],
            "学生的 k 近邻（方向，周期）": [r3(ev_s["a"][1]), r3(ev_s["b"][1])], "教师的 k 近邻（方向，周期）": [r3(ev_t["a"][1]), r3(ev_t["b"][1])],
            "学生的 k 近邻（两个任务平均）": r3(acc[m][0]), "教师的 k 近邻（两个任务平均）": r3(acc[m][1]),
            f"教师里初始权重还剩的比例（动量的 {DINO_STEPS} 次方）": float(f"{m ** DINO_STEPS:.2g}"),
        }
    NUM["教师动量"] = {
        "说明": f"自蒸馏的默认设置，只改教师的动量，训练 {DINO_STEPS} 步。动量 0 就是教师每一步都直接等于学生（只跑了 1 个种子），其余 2 个种子取平均。"
              "“平均分布里最大一维的峰值”是整个训练过程里出现过的最大值，接近 1 说明中途有过一维独占的坍缩。" + EVAL_NOTE,
        **table,
    }

    temps = [0.02, 0.04, 0.07, 0.1]
    trow = {}
    for tt in temps:
        runs = [get_dino(s) for s in SEEDS] if tt == 0.04 else [get_dino(s, t_temp=tt) for s in (SEEDS if tt == 0.1 else (0,))]
        summ, _ = dino_summary(runs)
        trow[f"教师温度 {tt}"] = {"种子数": len(runs), "单个样本的熵": summ["单个样本的熵 H"], "平均分布的熵": summ["平均分布的熵"],
                                 "学生的 k 近邻（方向，周期）": [summ["学生编码器的 k 近邻：方向"], summ["学生编码器的 k 近邻：周期"]]}
    NUM["教师温度"] = {
        "说明": f"自蒸馏的默认设置（中心化开，教师动量 0.99，学生温度 0.1），只改教师温度，训练 {DINO_STEPS} 步。0.02 和 0.07 只跑了 1 个种子，0.04 和 0.1 是 2 个种子的平均。"
              "熵是最后约 50 步的平均，单位奈特，上限 ln 32。" + EVAL_NOTE,
        **trow,
    }

    methods = [("原始像素", [None]), ("随机初始化", [get_rand(s) for s in SEEDS]), ("有监督预训练\n（方向标签）", [get_sup(s, "a") for s in SEEDS]),
               ("有监督预训练\n（周期标签）", [get_sup(s, "b") for s in SEEDS]), ("MAE", [get_mae(s).enc for s in SEEDS]), ("自蒸馏", [get_dino(s)[0].enc for s in SEEDS])]
    ev = {name: eval_models(encs) for name, encs in methods}
    extra = {"MAE，块特征平均": eval_models([get_mae(s).enc for s in SEEDS], kind="mean"), "自蒸馏，块特征平均": eval_models([get_dino(s)[0].enc for s in SEEDS], kind="mean"),
             "自蒸馏的教师": eval_models([get_dino(s)[1].enc for s in SEEDS])}
    both = eval_models([get_sup(s, "ab") for s in SEEDS])

    def rows(e):
        return {TASKS[t][0]: {"线性探测": r3(e[0][t][0]), "k 近邻": r3(e[0][t][1]), "检索 P@5": r3(e[0][t][2]),
                              "各种子的 k 近邻": [r3(x[t][1]) for x in e[1]]} for t in ("a", "b", "ab")}

    NUM["通用特征的评价"] = {
        "说明": DATA_NOTE + "。" + MODEL_NOTE + f"。有监督预训练：同样的编码器加线性头，每一步都是带标签的新图，训练 {SUP_STEPS} 步，用和自蒸馏相同的增强，标签只用方向或者只用周期。"
              f"MAE 是遮挡比例 0.75、{MAE_STEPS} 步，自蒸馏是默认设置、{DINO_STEPS} 步，都没有用标签。每种预训练 2 个种子。没有特别说明时特征取类别标记。"
              "检索 P@5：每张测试图在其余 599 张测试图里按余弦相似度找最像的 5 张，其中同类的比例。" + EVAL_NOTE,
        **{name.replace("\n", ""): rows(e) for name, e in ev.items()},
        **{name: rows(e) for name, e in extra.items()},
        "有监督预训练（方向和周期的标签都用，12 类）": rows(both),
        "随机猜": {"方向（4 类）": r3(1 / NA), "周期（3 类）": r3(1 / NB), "方向 x 周期（12 类）": r3(1 / (NA * NB))},
    }

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.3), gridspec_kw={"width_ratios": [1.1, 0.9, 1.7]}, layout="constrained")
    for (m, _), color in zip(moms, (S.MUTED, S.ORANGE, S.BLUE, S.GREEN)):
        ax[0].plot(logs[m]["step"], logs[m]["top"], color=color, lw=1.6, label=f"动量 {m}")
    ax[0].set_ylim(-0.03, 1.05)
    ax[0].set_xlabel("训练步数")
    ax[0].set_ylabel("平均分布里最大一维的占比")
    ax[0].set_title("动量 0 和 0.9：中途冲到一维独占再崩开", fontsize=10.5)
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, columnspacing=1.0, handlelength=1.4)
    S.clean(ax[0])
    gx = np.arange(len(moms))
    for i, (label, color) in enumerate((("学生", S.BLUE), ("教师", S.GREEN))):
        vals = [acc[m][i] for m, _ in moms]
        bars = ax[1].bar(gx + (i - 0.5) * 0.36, vals, 0.34, color=color, alpha=0.88, label=label)
        for b, v, (m, _) in zip(bars, vals, moms):
            ax[1].text(b.get_x() + b.get_width() / 2, max(acc[m][2 + i]) + 0.015, f"{v:.2f}", ha="center", fontsize=8.5)
            ax[1].plot([b.get_x() + b.get_width() / 2] * len(acc[m][2 + i]), acc[m][2 + i], ".", color=S.INK, ms=4)
    ax[1].set_xticks(gx, [str(m) for m, _ in moms])
    ax[1].set_ylim(0.3, 1.08)
    ax[1].set_xlabel("教师动量")
    ax[1].set_ylabel("k 近邻精度（方向和周期两个任务平均）")
    ax[1].set_title("动量太小不稳，太大教师跟不上", fontsize=10.5)
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[1])
    ax[1].grid(axis="x", visible=False)
    gx = np.arange(len(methods))
    for i, (t, color) in enumerate((("a", S.BLUE), ("b", S.ORANGE))):
        vals = [ev[name][0][t][1] for name, _ in methods]
        bars = ax[2].bar(gx + (i - 0.5) * 0.36, vals, 0.34, color=color, alpha=0.88, label=f"下游任务：{TASKS[t][0]}")
        for b, v, (name, _) in zip(bars, vals, methods):
            pts = [e[t][1] for e in ev[name][1]]
            ax[2].text(b.get_x() + b.get_width() / 2, max(pts) + 0.015, f"{v:.2f}", ha="center", fontsize=8.5)
            ax[2].plot([b.get_x() + b.get_width() / 2] * len(pts), pts, ".", color=S.INK, ms=4)
    ax[2].set_xticks(gx, [name for name, _ in methods])
    ax[2].tick_params(axis="x", labelsize=9)
    ax[2].set_ylim(0.3, 1.1)
    ax[2].set_ylabel(f"冻结特征的 k 近邻精度（每类 {N_PER} 张标签）")
    ax[2].set_title("有监督预训练只在自己那个任务上最好，换一个任务不如自监督", fontsize=10.5)
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    ax[2].grid(axis="x", visible=False)
    S.save(fig, CH, "fig05_教师动量与特征评价")


# ---------------------------------------------------------------- 图 6：冻结的块特征加最近邻做异常检测
def texture(n, g, angle, per, noise):
    z = latent(n, g, torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long))
    return z, draw(z, g, angles=(angle,), periods=(per,), noise=noise)


def blob_defect(img, g, radius, amp=0.2):
    """半径 radius 的圆斑，整体加亮或减暗 amp（和第 19 章的缺陷同一种）。"""
    n = len(img)
    m = int(radius) + 2
    cy = torch.randint(m, SIZE - m, (n,), generator=g)
    cx = torch.randint(m, SIZE - m, (n,), generator=g)
    sign = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
    mask = (YY[None] - cy[:, None, None]) ** 2 + (XX[None] - cx[:, None, None]) ** 2 <= radius ** 2
    return img + mask * (sign[:, None, None] * amp), mask


def pattern_defect(z, img, g, angle, per, size, noise):
    """size x size 的方块里条纹换成另一种周期（幅度、亮度、相位参数不变），方块位置随机。"""
    n = len(img)
    alt = draw(z, g, angles=(angle,), periods=(per,), noise=noise)
    cy = torch.randint(0, SIZE - size + 1, (n,), generator=g)
    cx = torch.randint(0, SIZE - size + 1, (n,), generator=g)
    mask = (YY[None] >= cy[:, None, None]) & (YY[None] < cy[:, None, None] + size) & (XX[None] >= cx[:, None, None]) & (XX[None] < cx[:, None, None] + size)
    return torch.where(mask, alt, img), mask


@torch.no_grad()
def patch_tokens(enc, img):
    """每张图 16 个块特征。enc 为 None 时直接用归一化后的 8x8 像素块（64 维）。"""
    if enc is None:
        return patchify(normed(img)).numpy().astype(np.float64)
    return torch.cat([enc(img[i:i + 500]) for i in range(0, len(img), 500)])[:, 1:].numpy().astype(np.float64)


def ad_data(seed, kind, angle=0.0, per=9.0, noise=NOISE, radius=2.5, d_per=6.0, size=10, n_good=20, n_test=200):
    g = gen(seed, 11)
    _, good = texture(n_good, g, angle, per, noise)
    zt, test = texture(n_test, g, angle, per, noise)
    if kind == "blob":
        bad, mask = blob_defect(test[::2], g, radius)
    else:
        bad, mask = pattern_defect({k: v[::2] for k, v in zt.items()}, test[::2], g, angle, d_per, size, noise)
    test = test.clone()
    test[::2] = bad
    labels = np.zeros(n_test, int)
    labels[::2] = 1
    return good, test, mask, labels


def ad_scores(enc, good, test):
    mem = patch_tokens(enc, good)
    q = patch_tokens(enc, test)
    return hw19.knn_scores(mem.reshape(-1, mem.shape[-1]), q.reshape(-1, q.shape[-1]), k=1).reshape(len(test), L)


AD_FEATS = ["8x8 像素块", "随机初始化的块特征", "MAE 的块特征", "自蒸馏的块特征"]
AD_SEEDS = (0, 1, 2)


def ad_encoders():
    return {AD_FEATS[0]: [None], AD_FEATS[1]: [get_rand(s) for s in SEEDS], AD_FEATS[2]: [get_mae(s).enc for s in SEEDS], AD_FEATS[3]: [get_dino(s)[0].enc for s in SEEDS]}


def ad_run(kind, **kw):
    """返回 {特征: (image AUROC, 定位命中率)}，对数据种子和模型种子取平均。命中指分数最高的块碰到了缺陷。"""
    out = {}
    encs = ad_encoders()
    for name in AD_FEATS:
        rows = []
        for s in AD_SEEDS:
            good, test, mask, labels = ad_data(s, kind, **kw)
            touched = patchify(mask.float()).sum(-1).numpy() > 0
            for enc in encs[name]:
                sc = ad_scores(enc, good, test)
                rows.append([hw19.auroc(sc.max(1), labels), float(np.mean(touched[np.arange(len(touched)), sc[::2].argmax(1)]))])
        out[name] = np.mean(rows, axis=0)
    return out


def fig_anomaly():
    def pack(res):
        return {name: {"image AUROC": r3(v[0]), "定位命中率": r3(v[1])} for name, v in res.items()}

    radii = [1.5, 2.5, 4.0]
    blob = {r: ad_run("blob", radius=r) for r in radii}
    noises = [0.05, 0.1, 0.15, 0.2, 0.3]
    pat = {nz: ad_run("pattern", noise=nz) for nz in noises}
    sizes = [6, 10, 16]
    by_size = {sz: ad_run("pattern", size=sz) for sz in sizes}
    domain = {
        "见过的纹理：0 度，周期 9 px，缺陷处周期 6 px": pat[0.2],
        "没见过的周期：0 度，周期 20 px，缺陷处周期 30 px": ad_run("pattern", per=20.0, d_per=30.0, noise=0.2),
        "没见过的方向：22.5 度，周期 9 px，缺陷处周期 6 px": ad_run("pattern", angle=22.5, noise=0.2),
    }
    goods = [5, 20, 80]
    by_good = {n: ad_run("pattern", noise=0.2, n_good=n) for n in goods}
    common = ("良品是同一种条纹（方向 0 度、周期 9 px，相位、幅度、亮度随机）。每次 20 张良品建记忆库（每张 16 个块，共 320 个块特征，不做核心集采样），200 张待检图里一半有缺陷；"
              "每个块的分数是它到记忆库的最近邻欧氏距离（第 19 章作业的 knn_scores，k=1），整图分数取 16 个块的最大值，image AUROC 用第 19 章作业的 auroc 算。定位命中率：缺陷图里分数最高的块碰到缺陷的比例。"
              "3 套数据；ViT 的三种特征各有 2 个种子的模型，共 6 次取平均，像素块 3 次取平均。" + MODEL_NOTE.split("；输入")[0] +
              "；MAE 和自蒸馏是在 12 类条纹上预训练的（噪声 0.05，没见过任何缺陷），块特征取最后一层除类别标记外的 16 个输出")
    NUM["冻结块特征做异常检测：圆斑缺陷"] = {
        "说明": "缺陷是圆斑，整体加亮或减暗 0.2（和第 19 章的缺陷同一种），像素噪声 0.05。" + common,
        "缺陷半径（px）": radii, "缺陷面积占一个 8x8 块的比例": [r3(math.pi * r * r / (P * P)) for r in radii],
        **{f"半径 {r} px": pack(blob[r]) for r in radii},
    }
    NUM["冻结块特征做异常检测：图形缺陷与噪声"] = {
        "说明": "缺陷是一个 10x10 px 的方块，里面的条纹周期从 9 px 变成 6 px，幅度和亮度不变（图形错了，灰度统计没变）。像素噪声从 0.05 加到 0.3，良品和待检图的噪声相同。" + common,
        "像素噪声标准差": noises,
        **{name: {"image AUROC": [r3(pat[nz][name][0]) for nz in noises], "定位命中率": [r3(pat[nz][name][1]) for nz in noises]} for name in AD_FEATS},
    }
    NUM["冻结块特征做异常检测：缺陷方块的大小"] = {
        "说明": "同上的图形缺陷，像素噪声 0.05，只改缺陷方块的边长。" + common,
        "方块边长（px）": sizes, **{name: {"image AUROC": [r3(by_size[sz][name][0]) for sz in sizes]} for name in AD_FEATS},
    }
    NUM["冻结块特征做异常检测：没见过的纹理"] = {
        "说明": "图形缺陷，像素噪声 0.2。预训练只见过方向 0、45、90、135 度和周期 6、9、14 px。三行的良品纹理不同，缺陷都是 10x10 px 方块里周期变成 1.5 倍或 2/3。" + common,
        **{k: pack(v) for k, v in domain.items()},
    }
    NUM["冻结块特征做异常检测：良品张数"] = {
        "说明": "图形缺陷（周期 9 px 变 6 px），像素噪声 0.2，只改良品张数。" + common,
        "良品张数": goods, **{name: {"image AUROC": [r3(by_good[n][name][0]) for n in goods]} for name in AD_FEATS},
    }

    good, test, mask, labels = ad_data(0, "pattern", noise=0.2)
    pick = 0
    maps = {name: ad_scores(encs[0], good, test)[pick].reshape(G, G) for name, encs in ad_encoders().items() if name in (AD_FEATS[0], AD_FEATS[3])}
    fig, ax = plt.subplots(1, 5, figsize=(17.5, 4.0), gridspec_kw={"width_ratios": [0.8, 0.8, 0.8, 1.35, 1.35]}, layout="constrained")
    ax[0].imshow(test[pick].numpy(), cmap="gray", vmin=0, vmax=1)
    ax[0].contour(mask[pick // 2].numpy(), levels=[0.5], colors="cyan", linewidths=1.3)
    ax[0].set_title("一张待检图（噪声 0.2）\n青色框里周期变了", fontsize=10.5)
    for a, name in zip(ax[1:3], (AD_FEATS[0], AD_FEATS[3])):
        a.imshow(maps[name], cmap="magma", extent=(-0.5, SIZE - 0.5, SIZE - 0.5, -0.5))
        a.contour(mask[pick // 2].numpy(), levels=[0.5], colors="cyan", linewidths=1.3)
        a.set_title(f"{name}\n4x4 的分数图", fontsize=10.5)
    for a in ax[:3]:
        S.image_axes(a)
    styles = [("s--", S.MUTED), ("v:", "#9aa0a8"), ("^-", S.GREEN), ("o-", S.BLUE)]
    for name, (style, color) in zip(AD_FEATS, styles):
        ax[3].plot(noises, [pat[nz][name][0] for nz in noises], style, color=color, lw=1.8, ms=5.5, label=name)
    ax[3].set_xticks(noises, [str(v) for v in noises])
    ax[3].set_ylim(0.5, 1.02)
    ax[3].set_xlabel("像素噪声标准差")
    ax[3].set_ylabel("image AUROC（图形缺陷）")
    ax[3].set_title("噪声小时像素块最好，噪声到 0.3 时预训练的块特征反超", fontsize=10.5)
    ax[3].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[3])
    gx = np.arange(len(radii))
    for i, (name, (_, color)) in enumerate(zip(AD_FEATS, styles)):
        vals = [blob[r][name][0] for r in radii]
        bars = ax[4].bar(gx + (i - 1.5) * 0.2, vals, 0.19, color=color, alpha=0.9, label=name)
        for b, v in zip(bars, vals):
            ax[4].text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", fontsize=7.5)
    ax[4].set_xticks(gx, [f"半径 {r} px" for r in radii])
    ax[4].set_ylim(0.5, 1.02)
    ax[4].set_ylabel("image AUROC（圆斑缺陷，噪声 0.05）")
    ax[4].set_title("圆斑缺陷：三种 ViT 块特征差不多，都远不如像素块", fontsize=10.5)
    ax[4].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[4])
    ax[4].grid(axis="x", visible=False)
    S.save(fig, CH, "fig06_块特征做异常检测")


# ---------------------------------------------------------------- 其余数字
def numbers_misc():
    mae, dino = MAE(), DinoNet()
    NUM["模型规模"] = {
        "说明": "本章小模型的可训练参数个数（固定的正弦位置编码不算）。真模型的参数量以各自的模型卡为准，本章不引用",
        "编码器（小 ViT）": n_params(mae.enc), "MAE 的解码器部分": n_params(mae) - n_params(mae.enc), "MAE 合计": n_params(mae),
        "自蒸馏的投影头": n_params(dino.head), "自蒸馏的学生（编码器加投影头）": n_params(dino),
        "解码器占 MAE 的比例": r3((n_params(mae) - n_params(mae.enc)) / n_params(mae)),
    }
    NUM["理论数值"] = {
        "说明": "由脚本算出的常数，讲义里直接引用",
        f"ln {K_OUT}（{K_OUT} 维均匀分布的熵，也是学生输出均匀时的自蒸馏损失）": r3(math.log(K_OUT)),
        "动量 m 的滑动平均大致相当于对最近 1/(1-m) 步取平均": {"0.9": 10, "0.99": 100, "0.999": 1000},
        f"训练 {DINO_STEPS} 步后教师里初始权重还剩的比例（m 的 {DINO_STEPS} 次方）": {"0.9": float(f"{0.9 ** DINO_STEPS:.2g}"), "0.99": float(f"{0.99 ** DINO_STEPS:.2g}"), "0.999": r3(0.999 ** DINO_STEPS)},
        "学生温度除以教师温度（默认 0.1 / 0.04）": 2.5,
        "输入 224 px、块 14 px 时的块数": (224 // 14) ** 2, "输入 448 px、块 14 px 时的块数": (448 // 14) ** 2,
        "边长加倍后自注意力的计算量倍数（块数的平方）": ((448 // 14) ** 2 / (224 // 14) ** 2) ** 2,
        "输入 224 px、块 16 px 时的块数": (224 // 16) ** 2, "其中遮挡 75% 后编码器处理的块数": (224 // 16) ** 2 // 4,
        "块 14 px 的特征图上一个位置对应的像素数": 14 * 14,
        "输入 518 px、块 14 px 时每边的块数": 518 // 14, "输入 518 px、块 14 px 时的块数": (518 // 14) ** 2,
        "本章小模型：一个 8x8 的块对应的像素数": P * P, "本章小模型：编码器在遮挡 0.75 时处理的标记数（4 个块加类别标记）": 5, "解码器处理的标记数（16 个块加类别标记）": 17,
        "AUROC 的最小变化量（100 张良品加 100 张缺陷图）": 1 / (100 * 100),
    }
    # 作业文档字符串里的两个性质，用参考实现核对一遍，讲义第 8 节引用
    g = gen(99)
    t = torch.randn(4, K_OUT, generator=g)
    uniform = float(hw.dino_loss(torch.zeros(4, K_OUT), t, torch.zeros(K_OUT)))
    tok = torch.randn(2, L, 5, generator=g)
    keep, mask, restore = hw.random_masking(tok, 0.75, torch.rand(2, L, generator=g))
    back = torch.gather(torch.cat([keep, torch.zeros(2, L - keep.shape[1], 5)], 1), 1, restore[..., None].expand(-1, -1, 5))
    assert abs(uniform - math.log(K_OUT)) < 1e-5 and torch.equal(back, tok * (1 - mask)[..., None])
    NUM["论文和官方实现里的设定（引用，不是实测）"] = {
        "说明": "讲义里引用的几个常数，出自论文和官方仓库，本章没有验证，以原文和仓库为准。列在这里只是为了让讲义里的每个数都有出处",
        "MAE 论文默认的遮挡比例": 0.75,
        "DINO 官方实现的默认值": {"学生温度": 0.1, "教师温度（起始值）": 0.04, "教师动量（起始值，训练中按余弦规律升到 1）": 0.996, "投影头输出维数": 65536, "中心的动量": 0.9},
        "DINO 论文做 k 近邻评价时的设置": {"k": 20, "温度": 0.07},
        "DINOv2 官方权重的块大小（px）": 14,
        "DINOv2 四种骨干的特征维数": {"ViT-S/14": 384, "ViT-B/14": 768, "ViT-L/14": 1024, "ViT-g/14": 1536},
        "ImageNet 归一化的均值和标准差": {"均值": [0.485, 0.456, 0.406], "标准差": [0.229, 0.224, 0.225]},
    }
    NUM["演示页"] = {
        "说明": "demos/自蒸馏与坍缩.html 里那个小模型的规模：样本是单位圆上的点，分成 4 团，每团 6 个；输出是 6 个原型的打分。熵的上限是 ln 6。页面里的数值在浏览器里实时算，讲义不引用",
        "样本个数": 24, "原型个数（输出维数）": 6, "ln 6": r3(math.log(6)), "学生温度": 0.1, "中心的动量": 0.9,
    }
    NUM["作业自检"] = {
        "说明": "用参考实现核对的两个性质：学生输出各维相同时，自蒸馏损失等于 ln K；随机遮挡后用 ids_restore 能把可见块放回原位",
        f"学生输出全 0、K={K_OUT} 时的自蒸馏损失": r3(uniform), "16 个块遮挡 0.75 后保留的块数": int(keep.shape[1]),
    }


def time_mae_steps():
    out = {}
    for ratio in (0.0, 0.25, 0.5, 0.75, 0.875):
        torch.manual_seed(0)
        g = gen(5, 9)
        net = MAE()
        opt, _ = optim(net.parameters(), 1e-3, 100)
        img = draw(latent(32, g), g)
        best = float("inf")
        for _ in range(4):
            t = time.perf_counter()
            for _ in range(20):
                pred, mask = net(img, ratio, torch.rand(32, L, generator=g))
                loss = ((pred - patchify(normed(img))) ** 2).mean() if ratio == 0 else hw.masked_mse_loss(pred, patchify(normed(img)), mask)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            best = min(best, (time.perf_counter() - t) / 20)
        out[str(ratio)] = best
    return {k: float(f"{1000 * v:.2g}") for k, v in out.items()}, {k: float(f"{v / out['0.0']:.2g}") for k, v in out.items()}


def main() -> None:
    S.setup()
    plt.rcParams["svg.hashsalt"] = "ch26"
    t_all = time.perf_counter()
    for fn in (numbers_misc, fig_few_labels, fig_mae_recon, fig_mae_ratio, numbers_bugs, fig_dino_collapse, fig_momentum_eval, fig_anomaly):
        t = time.perf_counter()
        fn()
        TIMES[fn.__name__] = time.perf_counter() - t
        print(f"完成 {fn.__name__}（{TIMES[fn.__name__]:.0f} 秒）", flush=True)
    total = time.perf_counter() - t_all

    NUM["训练规模"] = {
        "说明": "整个脚本一共训练了多少个模型（都是上面那个小 ViT）",
        "MAE": sum(1 for k in _CACHE if k[0] == "mae") + 4, "自蒸馏": sum(1 for k in _CACHE if k[0] == "dino"),
        "有监督预训练": sum(1 for k in _CACHE if k[0] == "sup"), "只用少量标签从头训练": 10,
    }

    def measure():
        dino = [v[2]["seconds"] for k, v in _CACHE.items() if k[0] == "dino"]
        step_ms, rel = time_mae_steps()
        return {
            "说明": "在一台 Mac 的 CPU 上（2 个线程，同时还有别的程序在跑）测的一次结果，只看量级和倍数关系，不是显卡上的数字。设置 CH26_RETIME=1 重测。"
                  "MAE 每步耗时是前向加反向加参数更新，每个比例测 4 组、每组 20 步，取最快的一组；比例 0 表示不遮挡（编码器处理全部 16 个块）",
            "MAE 每步训练耗时（毫秒，批大小 32），各遮挡比例": step_ms,
            "MAE 每步训练耗时相对不遮挡的倍数": rel,
            "自蒸馏每步训练耗时（毫秒，批大小 32，两个视图）": float(f"{1000 * np.mean(dino) / DINO_STEPS:.2g}"),
            "整个配图脚本（秒）": float(f"{total:.2g}"),
        }

    NUM["耗时"] = S.pinned(CH, "耗时", measure, "CH26_RETIME")
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=2))
    print(f"总耗时 {total:.0f} 秒")


if __name__ == "__main__":
    main()
