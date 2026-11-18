"""第 20 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch20.py

输出：<主分支>/20_合成缺陷与生成模型/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。

数据全部由代码生成：32x32 的线阵列版图，经过光学模糊、增益波动、散粒噪声和读出噪声得到“相机图”。
“真实”缺陷和合成缺陷用两套不同的代码生成（形状、轮廓、插入方式都不同），用来量合成和真实之间的差距。
程序化合成、羽化粘贴调用本章作业的参考实现（scratch_mask、blend），扩散前向调用 linear_beta_schedule、alpha_bars、q_sample。
分类器是一个 8 千多参数的小卷积网络，在 CPU 上训练；全部实验共训练 84 次。
类 FID 距离用 torchvision 的 ResNet-18（只在本机已有缓存权重时计算，不联网下载）。
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

torch.set_num_threads(2)
torch.manual_seed(0)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

CH = "20_合成缺陷与生成模型"
mod = S.load_solution(CH, "20-1_缺陷合成与扩散前向.py")
NUM: dict = {}
SEEDS = (0, 1, 2)

N = 32                       # 图像边长
FS, RN = 600.0, 5.0          # 灰度 1.0 对应的电子数、读出噪声（电子）
BLUR = 1.0                   # 光学模糊的高斯 sigma（px）
PB = 12                      # 剪贴图块的边长
STEPS, BATCH, LR = 400, 32, 3e-3
YY, XX = np.mgrid[0:N, 0:N].astype(np.float64)
FY = np.fft.fftfreq(N)[:, None]
FX = np.fft.fftfreq(N)[None, :]
OTF = np.exp(-2 * (np.pi * BLUR) ** 2 * (FY ** 2 + FX ** 2))
DATA_NOTE = (f"{N}x{N} 的线阵列版图（线周期 8 px，加一条横向总线，每张图的平移在 ±4 px 内随机），高斯光学模糊 sigma {BLUR:.0f} px，"
             f"增益在 0.9 到 1.1 之间随机，散粒噪声（灰度 1.0 对应 {FS:.0f} 个电子）加读出噪声 {RN:.0f} 个电子")
REAL_NOTE = ("“真实”缺陷在成像前插入，三类各占三分之一：椭圆高斯颗粒、带拐弯且亮度沿长度起伏的划痕、图形缺失或桥接；"
             "幅度 0.12 到 0.45，亮暗各半，中心离图像边缘至少 7 px")
SYN_NOTE = ("程序化合成用作业的 scratch_mask 和 blend：直线划痕（长 6 到 12 px，宽 1 到 2.5 px）和圆点（直径 2 到 4.5 px）各半，"
            "过渡带 0.8 到 1.5 px，幅度 0.12 到 0.45，亮暗各半，位置随机")
TRAIN_NOTE = (f"分类器是三层卷积加全局最大池化的小网络，Adam，学习率 {LR} 余弦下降，{STEPS} 步，每步 {BATCH // 2} 个正样本加 {BATCH // 2} 个负样本；"
              "负样本是 400 张良品图。测试集固定：1000 张良品图和 1500 张“真实”缺陷图。检出率指把阈值定在测试良品分数的 95% 分位（误报 5%）时的检出率；"
              f"每个条件 {len(SEEDS)} 个种子取平均")


def r3(v):
    return round(float(v), 3)


def r3s(vs):
    return [r3(v) for v in vs]


# ---------------------------------------------------------------- 版图和成像
def canonical_layout():
    lines = ((XX % 8) < 4).astype(np.float64)
    bus = ((YY >= 13) & (YY < 18)).astype(np.float64)
    lay = 0.28 + 0.34 * lines + 0.22 * bus
    return np.real(np.fft.ifft2(np.fft.fft2(lay) * np.exp(-2 * (np.pi * 0.7) ** 2 * (FY ** 2 + FX ** 2))))


LAY_HAT = np.fft.fft2(canonical_layout())


def clean_shifted(d):
    """按给定平移量 d (n, 2) 生成成像前的版图。"""
    ph = np.exp(-2j * np.pi * (FY[None] * d[:, 0, None, None] + FX[None] * d[:, 1, None, None]))
    return np.real(np.fft.ifft2(LAY_HAT[None] * ph))


def clean_batch(rng, n):
    return clean_shifted(rng.uniform(-4, 4, size=(n, 2)))


def blur(x):
    return np.real(np.fft.ifft2(np.fft.fft2(x) * OTF))


def capture(x, rng):
    g = rng.uniform(0.9, 1.1, size=(len(x), 1, 1))
    sig = np.clip(x * g, 0, None) * FS
    return (rng.poisson(sig) + rng.normal(0, RN, size=sig.shape)) / FS


def image_chain(clean, rng):
    """成像前的版图 → 光学模糊 → 增益 → 散粒噪声和读出噪声。"""
    return capture(blur(clean), rng)


def good_set(rng, n):
    return image_chain(clean_batch(rng, n), rng)


# ---------------------------------------------------------------- “真实”缺陷：成像前插入，形状和程序化合成的不同
def seg_dist(p0, p1):
    ay, ax = p0
    by, bx = p1
    vy, vx = by - ay, bx - ax
    t = np.clip(((YY - ay) * vy + (XX - ax) * vx) / max(vy * vy + vx * vx, 1e-12), 0, 1)
    return np.hypot(YY - (ay + t * vy), XX - (ax + t * vx)), t


def real_defect(img, rng, kind=None, amp_range=(0.12, 0.45), margin=7):
    """在一张成像前的版图上原地加一个缺陷，返回 (类型, 幅度, 极性, 中心, 掩膜)。"""
    kind = kind or rng.choice(["particle", "scratch", "pattern"])
    cy, cx = rng.uniform(margin, N - margin, size=2)
    s = rng.choice([-1.0, 1.0])
    amp = rng.uniform(*amp_range)
    if kind == "particle":
        r, ratio, th = rng.uniform(1.0, 2.0), rng.uniform(1.0, 2.0), rng.uniform(0, np.pi)
        u = (YY - cy) * np.sin(th) + (XX - cx) * np.cos(th)
        v = -(YY - cy) * np.cos(th) + (XX - cx) * np.sin(th)
        prof = np.exp(-(u ** 2 / (2 * (r * ratio) ** 2) + v ** 2 / (2 * r ** 2)))
        img += s * amp * prof
        mask = prof > 0.3
    elif kind == "scratch":
        length, th = rng.uniform(6, 12), rng.uniform(0, np.pi)
        bend, w = np.deg2rad(rng.uniform(-35, 35)), rng.uniform(0.5, 0.9)
        a = (cy - length / 2 * np.sin(th), cx - length / 2 * np.cos(th))
        b = (cy + length / 2 * np.sin(th + bend), cx + length / 2 * np.cos(th + bend))
        d1, t1 = seg_dist(a, (cy, cx))
        d2, t2 = seg_dist((cy, cx), b)
        d = np.minimum(d1, d2)
        along = np.where(d1 <= d2, 0.5 * t1, 0.5 + 0.5 * t2)
        wave = 0.65 + 0.35 * np.cos(2 * np.pi * (along * rng.uniform(0.5, 1.5) + rng.uniform()))
        img += s * amp * np.exp(-d ** 2 / (2 * w * w)) * wave
        mask = np.exp(-d ** 2 / (2 * w * w)) > 0.3
    else:
        h, w = rng.integers(3, 7), rng.integers(3, 7)
        y0, x0 = int(round(cy - h / 2)), int(round(cx - w / 2))
        level = 0.25 if s < 0 else 0.70
        k = 1.0 - min(1.0, amp / 0.45) * 0.9
        img[y0:y0 + h, x0:x0 + w] = img[y0:y0 + h, x0:x0 + w] * k + level * (1 - k)
        mask = np.zeros((N, N), bool)
        mask[y0:y0 + h, x0:x0 + w] = True
    return str(kind), float(amp), float(s), (float(cy), float(cx)), mask


def real_sources(rng, n, **kw):
    """n 张“真实缺陷图”，连同版图平移量和缺陷信息（剪贴时要用）。"""
    d = rng.uniform(-4, 4, size=(n, 2))
    clean = clean_shifted(d)
    meta = [real_defect(clean[i], rng, **kw) for i in range(n)]
    return image_chain(clean, rng), d, meta


def real_set(rng, n, **kw):
    return real_sources(rng, n, **kw)[0]


# ---------------------------------------------------------------- 程序化合成：作业的 scratch_mask 和 blend
def synth_mask(rng, shapes, center=None, softness=None, margin=7):
    kind = rng.choice(list(shapes))
    cy, cx = center if center is not None else rng.uniform(margin, N - margin, size=2)
    soft = rng.uniform(0.8, 1.5) if softness is None else softness
    if kind == "scratch":
        length, th = rng.uniform(6, 12), rng.uniform(0, np.pi)
        p0 = (cy - length / 2 * np.sin(th), cx - length / 2 * np.cos(th))
        p1 = (cy + length / 2 * np.sin(th), cx + length / 2 * np.cos(th))
        return mod.scratch_mask((N, N), p0, p1, rng.uniform(1.0, 2.5), soft)
    return mod.scratch_mask((N, N), (cy, cx), (cy, cx), rng.uniform(2.0, 4.5), soft)


def synth_set(rng, n, stage="pre", mode="add", softness=None, shapes=("scratch", "dot"),
              amp_range=(0.12, 0.45), signs=(-1.0, 1.0), center_box=None):
    """程序化合成 n 张缺陷图。
    stage：'pre' 在成像前的版图上画，再模糊加噪；'post' 在成像后的良品图上画；'post_blur' 在成像后的图上画，掩膜先按光学模糊的 sigma 模糊。
    mode：'add' 是 blend(bg, bg + 幅度, 掩膜)；'replace' 是 blend(bg, 固定灰度, 掩膜)，固定灰度取掩膜下的平均灰度加减幅度。"""
    clean = clean_batch(rng, n)
    base = clean if stage == "pre" else image_chain(clean, rng)
    out = np.empty_like(base)
    for i in range(n):
        c = rng.uniform(center_box[0], center_box[1], size=2) if center_box is not None else None
        m = synth_mask(rng, shapes, center=c, softness=softness)
        if stage == "post_blur":
            m = np.clip(blur(m), 0, 1)
        amp = rng.choice(list(signs)) * rng.uniform(*amp_range)
        bg = base[i]
        if mode == "replace":
            out[i] = mod.blend(bg, float((bg * m).sum() / max(m.sum(), 1e-9)) + amp, m)
        else:
            out[i] = mod.blend(bg, bg + amp, m)
    return image_chain(out, rng) if stage == "pre" else out


NARROW = dict(shapes=("scratch",), signs=(1.0,), amp_range=(0.3, 0.45))


# ---------------------------------------------------------------- 剪贴：直接粘贴、羽化、泊松
_ramp = np.minimum.reduce([np.arange(1, PB + 1)[:, None] * np.ones((1, PB)), np.arange(PB, 0, -1)[:, None] * np.ones((1, PB)),
                           np.arange(1, PB + 1)[None, :] * np.ones((PB, 1)), np.arange(PB, 0, -1)[None, :] * np.ones((PB, 1))])
FEATHER = np.clip(_ramp / 4.0, 0, 1)        # 从图块边缘向里 4 px 线性过渡到 1
_n = PB - 2
_S = np.sin(np.pi * np.outer(np.arange(1, _n + 1), np.arange(1, _n + 1)) / (_n + 1))
_lam = 2 - 2 * np.cos(np.pi * np.arange(1, _n + 1) / (_n + 1))
_LAM = _lam[:, None] + _lam[None, :]


def poisson_patch(tpatch, spatch):
    """泊松融合（矩形图块）：最外一圈取目标图的值，内部保持源图块的拉普拉斯。
    等价于 源 + u，u 是调和函数，在最外一圈上等于 目标 - 源。用离散正弦变换直接求解，不迭代。"""
    diff = tpatch - spatch
    b = np.zeros((_n, _n))
    b[0, :] += diff[0, 1:-1]
    b[-1, :] += diff[-1, 1:-1]
    b[:, 0] += diff[1:-1, 0]
    b[:, -1] += diff[1:-1, -1]
    u = _S @ ((_S @ b @ _S) / _LAM) @ _S * (2.0 / (_n + 1)) ** 2
    out = tpatch.copy()
    out[1:-1, 1:-1] = spatch[1:-1, 1:-1] + u
    return out


def paste(target, spatch, y0, x0, method):
    out = target.copy()
    tp = target[y0:y0 + PB, x0:x0 + PB]
    if method == "direct":
        out[y0:y0 + PB, x0:x0 + PB] = mod.blend(tp, spatch, np.ones((PB, PB)))
    elif method == "feather":
        out[y0:y0 + PB, x0:x0 + PB] = mod.blend(tp, spatch, FEATHER)
    else:
        out[y0:y0 + PB, x0:x0 + PB] = poisson_patch(tp, spatch)
    return out


METHOD_CN = {"direct": "直接粘贴", "feather": "羽化融合", "poisson": "泊松融合"}


def box_of(center):
    return (int(np.clip(round(center[0]) - PB // 2, 1, N - PB - 1)), int(np.clip(round(center[1]) - PB // 2, 1, N - PB - 1)))


def paste_set(rng, n, src_imgs, src_shift, centers, method, aligned=False):
    """把源图里以 centers 为中心的 PB×PB 图块贴到 n 张新的良品图上。
    aligned=True：目标良品和源图的版图相位相同，贴回同一位置（只有增益和噪声不同）；否则目标相位随机、位置随机。"""
    idx = rng.integers(0, len(src_imgs), n)
    targets = image_chain(clean_shifted(src_shift[idx]), rng) if aligned else good_set(rng, n)
    out, boxes = np.empty_like(targets), []
    for i, j in enumerate(idx):
        sy, sx = box_of(centers[j])
        ty, tx = (sy, sx) if aligned else (int(rng.integers(1, N - PB)), int(rng.integers(1, N - PB)))
        out[i] = paste(targets[i], src_imgs[j][sy:sy + PB, sx:sx + PB], ty, tx, method)
        boxes.append((ty, tx))
    return out, boxes


def blank_sources(rng, n=200):
    """没有缺陷的源图，用来做“无缺陷粘贴”：贴上去的图块里没有任何缺陷，只留下粘贴本身的痕迹。"""
    d = rng.uniform(-4, 4, size=(n, 2))
    return image_chain(clean_shifted(d), rng), d, [tuple(rng.uniform(7, N - 7, size=2)) for _ in range(n)]


def seam_stat(img, y0, x0, k):
    """跨图块边界的平均绝对差。k=0：图块最外一圈和它外面的邻居；k=1：最外一圈和它里面的邻居。"""
    a, b = (y0 + k, x0 + k), (y0 + PB - 1 - k, x0 + PB - 1 - k)
    rows, cols = slice(y0, y0 + PB), slice(x0, x0 + PB)
    parts = [np.abs(img[a[0], cols] - img[a[0] - 1, cols]), np.abs(img[b[0], cols] - img[b[0] + 1, cols]),
             np.abs(img[rows, a[1]] - img[rows, a[1] - 1]), np.abs(img[rows, b[1]] - img[rows, b[1] + 1])]
    return float(np.mean(np.concatenate(parts)))


# ---------------------------------------------------------------- 小分类器
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 12, 5, stride=2, padding=2)
        self.c2 = nn.Conv2d(12, 24, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(24, 24, 3, padding=1)
        self.fc = nn.Linear(24, 1)

    def forward(self, x):
        x = Fn.relu(self.c3(Fn.relu(self.c2(Fn.relu(self.c1(x))))))
        return self.fc(x.amax(dim=(2, 3))).squeeze(1)


def to_t(x):
    return torch.tensor((np.asarray(x) - 0.5) / 0.25, dtype=torch.float32)[:, None]


N_FITS = 0


def fit(pos, neg, seed, pos2=None):
    """pos2 不为 None 时，每个 batch 的正样本一半来自 pos、一半来自 pos2（按来源配平）。"""
    global N_FITS
    N_FITS += 1
    torch.manual_seed(seed)
    rng = np.random.default_rng([seed, 77])
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    P, Q = to_t(pos), to_t(neg)
    P2 = to_t(pos2) if pos2 is not None else None
    h = BATCH // 2
    y = torch.cat([torch.ones(h), torch.zeros(h)])
    for _ in range(STEPS):
        if P2 is None:
            xp = P[rng.integers(0, len(P), h)]
        else:
            xp = torch.cat([P[rng.integers(0, len(P), h // 2)], P2[rng.integers(0, len(P2), h - h // 2)]])
        loss = Fn.binary_cross_entropy_with_logits(net(torch.cat([xp, Q[rng.integers(0, len(Q), h)]])), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net.eval()


@torch.no_grad()
def score(net, x):
    return net(to_t(x)).numpy().astype(np.float64)


def auroc(neg, pos):
    """按秩次算 AUROC，并列取平均秩。"""
    s = np.concatenate([neg, pos])
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    end = np.cumsum(cnt)
    ranks = ((end - cnt + 1 + end) / 2.0)[inv]
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------- 固定的测试集和分组
_rng = np.random.default_rng([999, 0])
TEST_GOOD = good_set(_rng, 1000)
TEST_REAL, _, TEST_META = real_sources(_rng, 1500)
_kind = np.array([m[0] for m in TEST_META])
_amp = np.array([m[1] for m in TEST_META])
_sign = np.array([m[2] for m in TEST_META])
_cen = np.array([m[3] for m in TEST_META])
_central = (np.abs(_cen[:, 0] - N / 2) < 4) & (np.abs(_cen[:, 1] - N / 2) < 4)
GROUPS = {"亮": _sign > 0, "暗": _sign < 0, "中心": _central, "非中心": ~_central,
          "弱（幅度低于 0.2）": _amp < 0.2, "中（0.2 到 0.3）": (_amp >= 0.2) & (_amp < 0.3), "强（0.3 以上）": _amp >= 0.3,
          "颗粒": _kind == "particle", "划痕": _kind == "scratch", "图形": _kind == "pattern"}


def evaluate(net, extra=None):
    """返回 [AUROC, 检出率, 各分组检出率..., 额外探针集被报警的比例...]。"""
    sg, sp = score(net, TEST_GOOD), score(net, TEST_REAL)
    thr = np.quantile(sg, 0.95)
    row = [auroc(sg, sp), float(np.mean(sp > thr))] + [float(np.mean(sp[g] > thr)) for g in GROUPS.values()]
    for probe in (extra or []):
        row.append(float(np.mean(score(net, probe) > thr)))
    return row


def summary(rows, extra_names=()):
    m = np.mean(rows, axis=0)
    names = list(GROUPS)
    out = {"AUROC": r3(m[0]), "各种子的 AUROC": r3s([r[0] for r in rows]), "检出率": r3(m[1]), "各种子的检出率": r3s([r[1] for r in rows]),
           "分组检出率": {k: r3(v) for k, v in zip(GROUPS, m[2:2 + len(GROUPS)])},
           "各种子的分组检出率": {k: r3s([r[2 + names.index(k)] for r in rows]) for k in ("暗", "弱（幅度低于 0.2）", "非中心")}}
    for name, v in zip(extra_names, m[2 + len(GROUPS):]):
        out[name] = r3(v)
    return out


def neg_of(seed):
    return good_set(np.random.default_rng([seed, 500]), 400)


RES: dict = {}       # 各组实验的原始结果，画图时用


def run_real_count():
    """真实缺陷样本数从 8 到 1024。8 个和 256 个的那两组真实样本，在剪贴和混合实验里继续用。"""
    rows = {}
    for k in (8, 64, 256, 1024):
        nets = [fit(real_set(np.random.default_rng([seed, k]), k), neg_of(seed), seed) for seed in SEEDS]
        rows[k] = [evaluate(net) for net in nets]
        if k == 1024:
            RES["真实 1024 的网络"] = nets
    RES["真实数量"] = rows
    NUM["真实缺陷样本数量"] = {
        "说明": DATA_NOTE + "。" + REAL_NOTE + "。" + TRAIN_NOTE + "。只用真实缺陷图训练",
        **{f"{k} 张真实缺陷图": summary(v) for k, v in rows.items()},
        "各分组的缺陷个数": {k: int(v.sum()) for k, v in GROUPS.items()},
        "分类器的参数个数": int(sum(p.numel() for p in Net().parameters())),
        "灰度 0.5 处的噪声标准差（散粒噪声加读出噪声，按公式）": r3(math.sqrt(0.5 * FS + RN ** 2) / FS),
        "共训练了多少次": None,
    }


def run_stage():
    """程序化合成：在成像链的哪一步插入、掩膜软硬、加法还是替换。"""
    variants = {
        "成像前，软掩膜": dict(stage="pre"),
        "成像前，硬掩膜": dict(stage="pre", softness=0.05),
        "成像后，直接加": dict(stage="post"),
        "成像后，硬掩膜，直接加": dict(stage="post", softness=0.05),
        "成像后，替换成固定灰度": dict(stage="post", mode="replace"),
        "成像后，掩膜先模糊再加": dict(stage="post_blur"),
    }
    pr = np.random.default_rng([998, 1])
    probes = [synth_set(pr, 400, stage="pre"), synth_set(pr, 400, stage="post"), synth_set(pr, 400, stage="post", mode="replace")]
    names = ("合成探针检出率：成像前插入", "合成探针检出率：成像后直接加", "合成探针检出率：成像后替换成固定灰度")
    rows = {}
    for name, kw in variants.items():
        rows[name] = [evaluate(fit(synth_set(np.random.default_rng([seed, 20]), 400, **kw), neg_of(seed), seed), probes) for seed in SEEDS]
    RES["插入阶段"] = rows
    # 反过来：把合成缺陷当测试集，去预测“用 1024 张真实缺陷图训练的分类器”的检出率
    as_test = np.mean([evaluate(net, probes) for net in RES["真实 1024 的网络"]], axis=0)
    # 同一条划痕在两种做法下的峰值：成像前插入的会被光学模糊压低
    peak = {}
    for w in (1.0, 1.5, 2.5):
        m = mod.scratch_mask((N, N), (16.0, 8.0), (16.0, 24.0), w, 1.0)
        peak[f"划痕宽 {w} px"] = r3(blur(m).max() / m.max())
    for dia in (2.0, 3.0, 4.5):
        m = mod.scratch_mask((N, N), (16.0, 16.0), (16.0, 16.0), dia, 1.0)
        peak[f"圆点直径 {dia} px"] = r3(blur(m).max() / m.max())
    NUM["插入阶段"] = {
        "说明": DATA_NOTE + "。" + SYN_NOTE + "。只用 400 张合成缺陷图训练，在“真实”缺陷上测试。" + TRAIN_NOTE
              + "。硬掩膜指过渡带 0.05 px；“替换成固定灰度”是 blend(图, v, 掩膜)，v 取掩膜下的平均灰度加减幅度（本章 Notebook 的写法）；"
              "“掩膜先模糊”指掩膜先做一次和光学模糊相同的高斯模糊再相加。合成探针是 400 张同样形状的合成缺陷图，只是插入方式不同，用同一个阈值看检出率",
        **{k: summary(v, names) for k, v in rows.items()},
        "成像前插入时峰值幅度剩下的比例（过渡带 1 px，无噪声）": peak,
        "把合成缺陷当测试集": {
            "说明": "分类器固定为用 1024 张真实缺陷图训练的那 3 个网络，阈值不变（测试良品分数的 95% 分位）。三组合成探针各 400 张，形状、幅度范围相同，只是插入方式不同",
            "在真实缺陷上的检出率": r3(as_test[1]),
            "在合成探针上的检出率：成像前插入": r3(as_test[-3]), "在合成探针上的检出率：成像后直接加": r3(as_test[-2]),
            "在合成探针上的检出率：成像后替换成固定灰度": r3(as_test[-1]),
        },
    }


def run_paste():
    """剪贴：三种贴法的痕迹，以及分类器学到的是缺陷还是痕迹。源图是“真实缺陷样本数量”实验里那 8 张真实缺陷图。"""
    conds = {
        "直接粘贴，相位随机": ("direct", False, False),
        "羽化融合，相位随机": ("feather", False, False),
        "泊松融合，相位随机": ("poisson", False, False),
        "泊松融合，相位对齐": ("poisson", True, False),
        "直接粘贴，相位随机，良品也贴一块": ("direct", False, True),
    }
    rows = {k: [] for k in conds}
    for name, (method, aligned, negpaste) in conds.items():
        for seed in SEEDS:
            src, shift, meta = real_sources(np.random.default_rng([seed, 8]), 8)
            r = np.random.default_rng([seed, 30])
            pos, _ = paste_set(r, 400, src, shift, [m[3] for m in meta], method, aligned)
            neg = paste_set(r, 400, *blank_sources(r), method, aligned)[0] if negpaste else neg_of(seed)
            probe = paste_set(r, 400, *blank_sources(r), method, aligned)[0]
            rows[name].append(evaluate(fit(pos, neg, seed), [probe]))
    RES["剪贴"] = rows

    # 不训练网络的痕迹统计：无缺陷粘贴和没动过的良品，在同一批图块位置上比较跨边界的平均绝对差
    seam, bg_err, example = {}, {}, {}
    for aligned in (False, True):
        for method in ("direct", "feather", "poisson"):
            r = np.random.default_rng([997, 3])
            imgs, boxes = paste_set(r, 600, *blank_sources(r), method, aligned)
            plain = good_set(r, 600)
            au, means = [], []
            for k in (0, 1):
                a = np.array([seam_stat(im, *bx, k) for im, bx in zip(imgs, boxes)])
                b = np.array([seam_stat(im, *bx, k) for im, bx in zip(plain, boxes)])
                au.append(auroc(b, a))
                means.append([a.mean(), b.mean()])
            key = f"{METHOD_CN[method]}，{'相位对齐' if aligned else '相位随机'}"
            seam[key] = {"AUROC：最外一圈对外面的邻居": r3(au[0]), "AUROC：最外一圈对里面的邻居": r3(au[1]),
                         "可分性（两处取 max(A, 1-A) 的较大者）": r3(max(max(a, 1 - a) for a in au)),
                         "跨边界平均绝对差（粘贴，没动过）：对外": r3s(means[0]), "对内": r3s(means[1])}
            # 无噪声时贴进去的背景和目标背景差多少（没有缺陷时理想结果应该和目标完全一样）
            r2 = np.random.default_rng([996, 4])
            d_s = r2.uniform(-4, 4, size=(300, 2))
            d_t = d_s if aligned else r2.uniform(-4, 4, size=(300, 2))
            g_s, g_t = r2.uniform(0.9, 1.1, size=(300, 1, 1)), r2.uniform(0.9, 1.1, size=(300, 1, 1))
            Ssrc, Tgt = blur(clean_shifted(d_s)) * g_s, blur(clean_shifted(d_t)) * g_t
            errs = []
            for i in range(300):
                y0, x0 = int(r2.integers(1, N - PB)), int(r2.integers(1, N - PB))
                ty, tx = (y0, x0) if aligned else (int(r2.integers(1, N - PB)), int(r2.integers(1, N - PB)))
                out = paste(Tgt[i], Ssrc[i][y0:y0 + PB, x0:x0 + PB], ty, tx, method)
                errs.append(np.sqrt(np.mean((out - Tgt[i])[ty:ty + PB, tx:tx + PB] ** 2)))
            bg_err[key] = r3(np.mean(errs))
    for seed in (0,):
        src, shift, meta = real_sources(np.random.default_rng([seed, 8]), 8)
        j = int(np.argmax([m[1] for m in meta]))
        tgt = good_set(np.random.default_rng([995, 5]), 1)[0]
        sy, sx = box_of(meta[j][3])
        for method in ("direct", "feather", "poisson"):
            example[method] = paste(tgt, src[j][sy:sy + PB, sx:sx + PB], 10, 9, method)
        example["source"], example["box"] = src[j], (sy, sx)
    RES["剪贴示例"] = example
    a = np.random.default_rng(0).normal(size=(PB, PB))
    lap = lambda z: 4 * z[1:-1, 1:-1] - z[:-2, 1:-1] - z[2:, 1:-1] - z[1:-1, :-2] - z[1:-1, 2:]  # noqa: E731
    other = np.random.default_rng(1).normal(size=(PB, PB))
    out = poisson_patch(other, a)
    areas = [m[4][box_of(m[3])[0]:box_of(m[3])[0] + PB, box_of(m[3])[1]:box_of(m[3])[1] + PB].mean() for m in TEST_META]
    NUM["剪贴融合"] = {
        "说明": DATA_NOTE + f"。手上只有 8 张“真实”缺陷图，把以缺陷为中心的 {PB}x{PB} 图块贴到新的良品图上，造 400 张缺陷图来训练，在“真实”缺陷上测试。" + TRAIN_NOTE
              + "。相位随机指目标良品的版图平移和粘贴位置都随机；相位对齐指目标良品和源图的版图平移相同、贴回同一位置，只有增益和噪声不同。"
              "羽化是从图块边缘向里 4 px 线性过渡；泊松融合让图块最外一圈等于目标图、内部保持源图块的拉普拉斯。"
              "“无缺陷粘贴被报警的比例”：用同一种贴法把不含缺陷的图块贴到良品上（400 张），看同一阈值下有多少被判成缺陷。"
              "“良品也贴一块”指训练用的负样本也用同一种贴法贴了一块不含缺陷的图块",
        **{k: summary(v, ("无缺陷粘贴被报警的比例",)) for k, v in rows.items()},
        "痕迹统计（不训练网络）": {"说明": "600 张无缺陷粘贴和 600 张没动过的良品，在同一批图块位置上算跨图块边界的平均绝对差，用它当分数算 AUROC。"
                                    "大于 0.5 表示粘贴处的跨边界差更大，小于 0.5 表示更小（比正常的还平滑，同样是痕迹）", **seam},
        "无噪声时贴进去的背景与目标背景的均方根误差（图块内，300 对）": bg_err,
        "泊松求解器自检": {"源和目标只差一个常数时，结果与目标的最大差": float(f"{np.abs(poisson_patch(a + 0.3, a) - (a + 0.3)).max():.1e}"),
                     "图块内部（离边界 2 px 以上）拉普拉斯与源图块的最大差": float(f"{np.abs(lap(out)[1:-1, 1:-1] - lap(a)[1:-1, 1:-1]).max():.1e}"),
                     "最外一圈与目标图的最大差": float(max(np.abs(out - other)[[0, -1], :].max(), np.abs(out - other)[:, [0, -1]].max()))},
        "图块里属于缺陷掩膜的像素比例（1500 个测试缺陷的平均）": r3(np.mean(areas)),
    }


def run_diversity():
    variants = {
        "只有划痕（没有圆点）": dict(shapes=("scratch",)),
        "只有亮缺陷": dict(signs=(1.0,)),
        "只有强缺陷（幅度 0.3 到 0.45）": dict(amp_range=(0.3, 0.45)),
        "只有弱缺陷（幅度 0.12 到 0.2）": dict(amp_range=(0.12, 0.2)),
        "只在图像中心 6x6 px 内": dict(center_box=(13, 19)),
    }
    rows = {"全部随机": RES["插入阶段"]["成像前，软掩膜"]}
    for name, kw in variants.items():
        rows[name] = [evaluate(fit(synth_set(np.random.default_rng([seed, 40]), 400, **kw), neg_of(seed), seed)) for seed in SEEDS]
    RES["多样性"] = rows
    NUM["合成缺陷的多样性"] = {
        "说明": DATA_NOTE + "。" + SYN_NOTE + "（这是“全部随机”）。每次只收窄一个维度，其余不变，都在成像前插入；只用 400 张合成缺陷图训练，在“真实”缺陷上测试。" + TRAIN_NOTE
              + "。“全部随机”和“插入阶段”实验里的“成像前，软掩膜”是同一组网络。“中心”指缺陷中心离图像中心不超过 4 px",
        **{k: summary(v) for k, v in rows.items()},
    }


def run_mixing():
    rows = {}
    for k, plan in ((8, [("好", {}, 64, False), ("好", {}, 512, False), ("窄", NARROW, 512, False)]),
                    (256, [("窄", NARROW, m, False) for m in (64, 512, 2048)] + [("窄", NARROW, 2048, True), ("好", {}, 2048, False)])):
        rows[(k, "无", 0, False)] = RES["真实数量"][k]
        for qname, kw, m, balanced in plan:
            out = []
            for seed in SEEDS:
                real = real_set(np.random.default_rng([seed, k]), k)
                syn = synth_set(np.random.default_rng([seed, 50, m]), m, **kw)
                net = fit(real, neg_of(seed), seed, pos2=syn) if balanced else fit(np.concatenate([real, syn]), neg_of(seed), seed)
                out.append(evaluate(net))
            rows[(k, qname, m, balanced)] = out
    RES["混合"] = rows

    def label(key):
        k, q, m, bal = key
        if m == 0:
            return f"真实 {k} 张，不加合成"
        return f"真实 {k} 张 + {q}的合成 {m} 张" + ("，按来源配平" if bal else "")

    NUM["真实加合成"] = {
        "说明": DATA_NOTE + "。" + SYN_NOTE + "。“好的合成”是成像前插入、全部随机；“窄的合成”也在成像前插入，但只有亮的、幅度 0.3 到 0.45 的划痕。"
              "默认把真实和合成的缺陷图放在一起均匀抽样；“按来源配平”指每个 batch 的正样本一半来自真实、一半来自合成。" + TRAIN_NOTE
              + "。真实样本和“真实缺陷样本数量”实验里同样张数的那一组相同",
        **{label(key): summary(v) for key, v in rows.items()},
    }


# ---------------------------------------------------------------- 类 FID 距离（Fréchet 距离）
def frechet(a, b):
    m1, m2 = a.mean(0), b.mean(0)
    c1, c2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    w, v = np.linalg.eigh(c1)
    s1 = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    ev = np.linalg.eigvalsh(s1 @ c2 @ s1)
    return float(((m1 - m2) ** 2).sum() + np.trace(c1) + np.trace(c2) - 2 * np.sqrt(np.clip(ev, 0, None)).sum())


def table_fd():
    ck = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
    if not ck.exists():
        NUM["类 FID 距离"] = {"说明": "没有运行：本机没有 ResNet-18 的 ImageNet 权重缓存（脚本不联网下载）"}
        print("跳过类 FID 距离：没有缓存的 ResNet-18 权重")
        return
    import torchvision
    net = torchvision.models.resnet18()
    net.load_state_dict(torch.load(ck, map_location="cpu"))
    net.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def feats(imgs):
        x = torch.tensor(np.clip(imgs, 0, 1), dtype=torch.float32)[:, None]
        x = (Fn.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1) - mean) / std
        out = []
        for i in range(0, len(x), 100):
            h = net.maxpool(net.relu(net.bn1(net.conv1(x[i:i + 100]))))
            out.append(net.layer2(net.layer1(h)).mean(dim=(2, 3)))
        return torch.cat(out).numpy().astype(np.float64)

    n = 400
    r = np.random.default_rng([994, 6])
    ref = feats(real_set(r, n))
    src, shift, meta = real_sources(np.random.default_rng([0, 8]), 8)
    cen = [m[3] for m in meta]
    sets = {
        "另一批真实缺陷图": (real_set(r, n), None),
        "良品图（一个缺陷都没有）": (good_set(r, n), None),
        "程序化合成，成像前插入，全部随机": (synth_set(r, n), ("插入阶段", "成像前，软掩膜")),
        "程序化合成，只有弱缺陷": (synth_set(r, n, amp_range=(0.12, 0.2)), ("多样性", "只有弱缺陷（幅度 0.12 到 0.2）")),
        "程序化合成，只有亮缺陷": (synth_set(r, n, signs=(1.0,)), ("多样性", "只有亮缺陷")),
        "程序化合成，成像后替换成固定灰度": (synth_set(r, n, stage="post", mode="replace"), ("插入阶段", "成像后，替换成固定灰度")),
        "剪贴，泊松融合，相位对齐": (paste_set(r, n, src, shift, cen, "poisson", True)[0], ("剪贴", "泊松融合，相位对齐")),
        "剪贴，直接粘贴，相位随机": (paste_set(r, n, src, shift, cen, "direct", False)[0], ("剪贴", "直接粘贴，相位随机")),
    }
    out = {}
    for name, (imgs, key) in sets.items():
        down = r3(np.mean([row[0] for row in RES[key[0]][key[1]]])) if key else None
        out[name] = {"到真实缺陷图的 Fréchet 距离": round(frechet(ref, feats(imgs)), 4),
                     "只用它当缺陷样本训练，在真实缺陷上的 AUROC": down if down is not None else ("不适用（基准）" if "另一批" in name else "不适用（没有缺陷，当不了正样本）")}
    NUM["类 FID 距离"] = {
        "说明": f"每组 {n} 张图，和另外 {n} 张“真实”缺陷图比。特征是 ResNet-18（ImageNet 权重）layer2 输出的全局平均，128 维；图像放大 2 倍到 64x64 输入。"
              "距离按 FID 的公式算（两组特征各拟合一个高斯分布，求 Fréchet 距离）。这不是标准的 FID：标准做法用 Inception-v3 的 2048 维特征和上万张图，数值不能和论文里的 FID 比。"
              "右边一列是同一种合成数据在前面各实验里的下游结果（3 个种子平均）",
        **out,
    }


# ---------------------------------------------------------------- 扩散前向过程
T = 1000


def cosine_abar(steps, s=0.008):
    """余弦噪声表（Nichol 和 Dhariwal 2021）：abar_t = f(t)/f(0)，f(t) = cos^2(((t/T + s)/(1 + s))·pi/2)，beta 截到 0.999。"""
    t = np.arange(steps + 1) / steps
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    betas = np.clip(1 - f[1:] / f[:-1], 0, 0.999)
    return mod.alpha_bars(betas), betas


def scene(res):
    """同一个场景按不同分辨率采样：先在 256x256 上画，再块平均到 res x res。取值缩放到 [-1, 1] 附近。"""
    y, x = np.mgrid[0:256, 0:256].astype(np.float64)
    img = 0.28 + 0.34 * ((x % 64) < 32) + 0.22 * ((y >= 104) & (y < 144))
    img += 0.45 * np.exp(-((y - 184) ** 2 + (x - 80) ** 2) / (2 * 10.0 ** 2))
    k = 256 // res
    return (img.reshape(res, k, res, k).mean(axis=(1, 3)) - 0.5) * 2


def block(a, k):
    return a if k == 1 else a.reshape(a.shape[0] // k, k, a.shape[1] // k, k).mean(axis=(1, 3))


def first_below(arr, level):
    return int(np.argmax(arr < level)) if (arr < level).any() else None


def fig_forward():
    betas = mod.linear_beta_schedule(T)
    ab = mod.alpha_bars(betas)
    abc, betas_c = cosine_abar(T)

    # (a) 闭式和逐步加噪是否一致
    x0 = scene(32)
    reps = 100
    rng = np.random.default_rng([993, 7])
    x = np.repeat(x0[None], reps, 0)
    marks = (0, 99, 299, 599, 999)
    it, cf = {}, {}

    def coef(xs):
        m = xs.mean(0)
        return float((m * x0).sum() / (x0 * x0).sum()), float((xs - m).std())

    for t in range(T):
        x = np.sqrt(1 - betas[t]) * x + np.sqrt(betas[t]) * rng.normal(size=x.shape)
        if t in marks:
            it[t] = coef(x)
            cf[t] = coef(np.stack([mod.q_sample(x0, t, ab, rng.normal(size=x0.shape)) for _ in range(reps)]))
    var, v = np.zeros(T), 0.0
    for t in range(T):
        v = (1 - betas[t]) * v + betas[t]
        var[t] = v
    noise = rng.normal(size=x0.shape)
    ok = mod.q_sample(x0, T - 1, ab, noise)
    wrong_beta = np.sqrt(1 - betas[-1]) * x0 + np.sqrt(betas[-1]) * noise
    xs, e = x0.copy(), rng.normal(size=x0.shape)
    for t in range(T):
        xs = np.sqrt(1 - betas[t]) * xs + np.sqrt(betas[t]) * e
    cum_wrong = 1 - np.cumsum(betas)
    NUM["闭式与逐步加噪"] = {
        "说明": f"线性噪声表，{T} 步，beta 从 1e-4 到 0.02。x0 是 32x32 的条纹图（取值 {x0.min():.2f} 到 {x0.max():.2f}）。逐步：从 x0 出发每步乘 sqrt(1-beta_t) 再加 sqrt(beta_t) 倍的新噪声，{reps} 条独立的链；"
              f"闭式：直接调用 q_sample，同样 {reps} 次。“信号系数”是多次平均后的 x_t 对 x0 的回归斜率，“噪声标准差”是去掉平均后的标准差。下标 t 从 0 开始，x_t 已经加了 t+1 次噪声",
        "步数 t": list(marks),
        "公式给的信号系数 sqrt(abar_t)": r3s([np.sqrt(ab[t]) for t in marks]),
        "逐步加噪量出的信号系数": r3s([it[t][0] for t in marks]),
        "q_sample 量出的信号系数": r3s([cf[t][0] for t in marks]),
        "公式给的噪声标准差 sqrt(1-abar_t)": r3s([np.sqrt(1 - ab[t]) for t in marks]),
        "逐步加噪量出的噪声标准差": r3s([it[t][1] for t in marks]),
        "q_sample 量出的噪声标准差": r3s([cf[t][1] for t in marks]),
        "把每一步的系数连乘起来，和 sqrt(abar_t) 的最大差（1000 步）": float(f"{np.abs(np.cumprod(np.sqrt(1 - betas)) - np.sqrt(ab)).max():.1e}"),
        "按递推算出的噪声方差，和 1-abar_t 的最大差": float(f"{np.abs(var - (1 - ab)).max():.1e}"),
        "x0 的标准差": r3(x0.std()),
        "t=300 时 x_t 和 x0 的逐像素相关系数（按公式）": r3(np.sqrt(ab[300] * x0.var() / (ab[300] * x0.var() + 1 - ab[300]))),
        "abar 在 t=0、300、999 的值": [float(f"{ab[0]:.4f}"), r3(ab[300]), float(f"{ab[-1]:.1e}")],
        "本章 Notebook 的条纹图（96x96，条纹宽 5 px，x0 取 -0.4 和 0.4）": {
            "t=300 时逐像素的相关系数（按公式）": r3(np.sqrt(ab[300] * 0.16 / (ab[300] * 0.16 + 1 - ab[300]))),
            "一条条纹的像素个数": 5 * 96,
            "t=300 时把每条条纹里的像素平均之后的相关系数（按公式）": r3(np.sqrt(ab[300] * 0.16 / (ab[300] * 0.16 + (1 - ab[300]) / (5 * 96)))),
        },
    }
    NUM["前向过程的几种写错"] = {
        "说明": "同一张 x0、同一份噪声，线性噪声表 1000 步，看最后一步或中间一步的结果",
        "正确：最后一步 x_T 和噪声的相关系数": float(f"{np.corrcoef(ok.ravel(), noise.ravel())[0, 1]:.5f}"),
        "把 abar_t 写成 1-beta_t（没有累乘）：最后一步和噪声的相关系数": r3(np.corrcoef(wrong_beta.ravel(), noise.ravel())[0, 1]),
        "同上：最后一步和 x0 的相关系数": r3(np.corrcoef(wrong_beta.ravel(), x0.ravel())[0, 1]),
        "忘了开方（x_t = abar·x0 + (1-abar)·噪声），t=500：abar": r3(ab[500]),
        "同上：噪声系数写成了": r3(1 - ab[500]),
        "同上：噪声系数应为": r3(np.sqrt(1 - ab[500])),
        "同上：对方差为 1 的数据，x_t 的方差变成": r3(ab[500] ** 2 + (1 - ab[500]) ** 2),
        "把累乘写成 1-累加（1 - cumsum(beta)）：最后一步的“abar”": r3(cum_wrong[-1]),
        "同上：第一次变成负数的步数": first_below(cum_wrong, 0),
        "逐步加噪时每一步都用同一份噪声：最后 x_T 的标准差（应约为 1）": r3(xs.std()),
    }

    # (b) 线性和余弦噪声表
    info = {}
    for name, a in (("线性", ab), ("余弦", abc)):
        snr = a / (1 - a)
        info[name] = {
            "abar 在 t=100、300、500、700、900 的值": [float(f"{a[t]:.4f}") for t in (100, 300, 500, 700, 900)],
            "最后一步的 abar": float(f"{a[-1]:.1e}"),
            "信噪比在 t=300、500、700 的值": [float(f"{snr[t]:.3g}") for t in (300, 500, 700)],
            "信噪比第一次低于 1 的步数": first_below(snr, 1.0),
            "信噪比第一次低于 0.01 的步数": first_below(snr, 0.01),
            "信噪比低于 0.01 的步数占全部步数的比例": r3(np.mean(snr < 0.01)),
            "信噪比高于 100 的步数": int((snr > 100).sum()),
        }
    short = {}
    for steps in (100, 200, 500, 1000):
        a1 = mod.alpha_bars(mod.linear_beta_schedule(steps))
        a2 = mod.alpha_bars(mod.linear_beta_schedule(steps, 1e-4 * 1000 / steps, 0.02 * 1000 / steps))
        short[f"T={steps}"] = {"beta 范围不变时最后一步的 abar": float(f"{a1[-1]:.2g}"), "beta 按 1000/T 放大后最后一步的 abar": float(f"{a2[-1]:.1e}")}
    NUM["线性和余弦噪声表"] = {
        "说明": f"都是 {T} 步。线性表 beta 从 1e-4 到 0.02；余弦表按 Nichol 和 Dhariwal（2021）的公式，偏移 s=0.008，beta 截到 0.999。信噪比指 abar/(1-abar)，"
              "即信号系数和噪声系数的平方比，没有乘图像本身的方差",
        **info,
        "线性表换总步数": short,
    }

    # (c) 分辨率
    res_list, t_list = (32, 64, 128, 256), (100, 300, 500, 700)
    res_out, curves = {}, {}
    ref = scene(32)
    for res in res_list:
        k = res // 32
        xr = scene(res)
        s2 = block(xr, k).var()
        theory = np.sqrt(ab * s2 / (ab * s2 + (1 - ab) / k ** 2))
        theory_c = np.sqrt(abc * s2 / (abc * s2 + (1 - abc) / k ** 2))
        meas = []
        for t in t_list:
            cs = [np.corrcoef(block(mod.q_sample(xr, t, ab, np.random.default_rng([992, res, t, rep]).normal(size=xr.shape)), k).ravel(), ref.ravel())[0, 1] for rep in range(20)]
            meas.append(float(np.mean(cs)))
        curves[res] = (theory, meas)
        res_out[f"{res}x{res}"] = {"实测相关系数（t=100、300、500、700）": r3s(meas), "按公式算的相关系数": r3s([theory[t] for t in t_list]),
                                   "相关系数降到 0.5 的步数（线性表）": first_below(theory, 0.5), "相关系数降到 0.5 的步数（余弦表）": first_below(theory_c, 0.5)}
    NUM["同一步数在不同分辨率下"] = {
        "说明": "同一个场景（条纹、一条横带、一个亮斑）按 32、64、128、256 四种分辨率采样，高分辨率的图块平均后和 32x32 的完全相同。各自按线性噪声表加噪到第 t 步，"
              "再块平均回 32x32，和 32x32 的 x0 算相关系数（20 次平均）。公式：块平均 k×k 个像素后噪声标准差除以 k，相关系数 = sqrt(abar·s² / (abar·s² + (1-abar)/k²))，s² 是 32x32 的 x0 的方差",
        "32x32 的 x0 的方差": r3(ref.var()),
        **res_out,
    }

    # ---- 图 5
    fig = plt.figure(figsize=(16.5, 4.3), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.75, 1, 1.1])
    ax0, ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    show_t = (0, 100, 300, 500, 700)
    x64 = scene(64)
    nz = np.random.default_rng([991, 8]).normal(size=x64.shape)
    gap = 4
    rows = []
    for a in (ab, abc):
        tiles = []
        for t in show_t:
            tiles += [np.clip(mod.q_sample(x64, t, a, nz), -1.6, 1.6), np.full((64, gap), np.nan)]
        rows.append(np.concatenate(tiles[:-1], axis=1))
    grid = np.concatenate([rows[0], np.full((gap + 6, rows[0].shape[1]), np.nan), rows[1]], axis=0)
    ax0.imshow(grid, cmap="gray", vmin=-1.6, vmax=1.6, interpolation="nearest")
    for i, t in enumerate(show_t):
        ax0.text(i * (64 + gap) + 32, -4, f"t={t}", ha="center", va="bottom", fontsize=9.5)
    ax0.text(-4, 32, "线性表", ha="right", va="center", rotation=90, fontsize=10)
    ax0.text(-4, 64 + gap + 6 + 32, "余弦表", ha="right", va="center", rotation=90, fontsize=10)
    ax0.set_title("同一张图、同一份噪声：t=500 时线性表已看不出条纹", pad=18)
    ax0.axis("off")
    ts = np.arange(T)
    ax1.plot(ts, np.sqrt(ab), color=S.BLUE, lw=1.8, label="闭式：信号系数")
    ax1.plot(ts, np.sqrt(1 - ab), color=S.ORANGE, lw=1.8, label="闭式：噪声标准差")
    ax1.plot(marks, [it[t][0] for t in marks], "o", color=S.BLUE, ms=7, mfc="white", mew=1.6, label="逐步加噪量出的信号系数")
    ax1.plot(marks, [it[t][1] for t in marks], "s", color=S.ORANGE, ms=7, mfc="white", mew=1.6, label="逐步加噪量出的噪声标准差")
    ax1.set_xlabel("步数 t（线性表）")
    ax1.set_title("逐步加噪的结果落在闭式公式的曲线上")
    ax1.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax1)
    for name, a, color in (("线性表", ab, S.BLUE), ("余弦表", abc, S.GREEN)):
        ax2.plot(ts, a / (1 - a), color=color, lw=1.9, label=name)
    for level, text in ((1.0, "信噪比 1"), (0.01, "信噪比 0.01")):
        ax2.axhline(level, color=S.MUTED, lw=1, ls="--")
        ax2.text(T * 0.99, level * 1.35, text, ha="right", fontsize=8.5, color=S.MUTED)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-5, 1e4)
    ax2.set_yticks([1e-4, 1e-2, 1, 1e2, 1e4], ["0.0001", "0.01", "1", "100", "10000"])
    ax2.minorticks_off()
    ax2.set_xlabel("步数 t")
    ax2.set_ylabel("信噪比 abar/(1-abar)")
    ax2.set_title("线性表后三分之一的步数信噪比低于 0.01")
    ax2.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax2)
    S.save(fig, CH, "fig05_前向过程与噪声表")

    # ---- 图 6
    fig, ax = plt.subplots(1, 5, figsize=(16.5, 3.7), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.9]}, layout="constrained")
    for a, res in zip(ax[:4], res_list):
        xt = mod.q_sample(scene(res), 300, ab, np.random.default_rng([990, res]).normal(size=(res, res)))
        a.imshow(xt, cmap="gray", vmin=-1.6, vmax=1.6, interpolation="nearest")
        a.set_title(f"{res}×{res}，t=300")
        S.image_axes(a)
    for res, color, marker in zip(res_list, (S.MUTED, S.BLUE, S.ORANGE, S.RED), ("o", "s", "^", "D")):
        theory, meas = curves[res]
        ax[4].plot(ts, theory, color=color, lw=1.7, label=f"{res}×{res}")
        ax[4].plot(t_list, meas, marker, color=color, ms=6, mfc="white", mew=1.5)
    ax[4].axhline(0.5, color=S.MUTED, lw=1, ls="--")
    ax[4].set_xlabel("步数 t（线性表）；线是公式，空心点是实测")
    ax[4].set_ylabel("块平均到 32×32 后和 x0 的相关系数")
    ax[4].set_title("同一步数，分辨率越高，粗结构留得越多")
    ax[4].legend(fontsize=9, loc="upper right")
    S.clean(ax[4])
    S.save(fig, CH, "fig06_分辨率")


def table_denoiser():
    """训练目标的含义：一个很小的去噪网络预测噪声，按步数分段看损失，并和“把输入原样输出”比较。"""
    ab = mod.alpha_bars(mod.linear_beta_schedule(T))
    AB = torch.tensor(ab, dtype=torch.float32)
    X0 = torch.tensor((good_set(np.random.default_rng([989, 9]), 600) - 0.5) * 2, dtype=torch.float32)[:, None]
    torch.manual_seed(0)
    den = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 3, padding=1))
    opt = torch.optim.Adam(den.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    steps, batch = 300, 32

    def make(idx, t):
        eps = torch.randn(len(idx), 1, N, N, generator=g)
        a = AB[t].view(-1, 1, 1, 1)
        xt = a.sqrt() * X0[idx] + (1 - a).sqrt() * eps
        return xt, eps, a

    for _ in range(steps):
        xt, eps, a = make(torch.randint(0, 500, (batch,), generator=g), torch.randint(0, T, (batch,), generator=g))
        loss = ((den(torch.cat([xt, (1 - a).sqrt().expand(-1, 1, N, N)], 1)) - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    s2 = float((X0[500:] ** 2).mean())
    marks = (10, 50, 100, 200, 300, 500, 700, 900)
    rows = {"步数 t": list(marks), "网络的损失": [], "把 x_t 原样当作噪声输出的损失": [], "同上，按公式": [], "网络预测换算成 x0 之后的均方误差": [], "换算倍数 (1-abar)/abar": []}
    worst = 0.0
    with torch.no_grad():
        idx = torch.arange(500, 600)
        for t in marks:
            xt, eps, a = make(idx, torch.full((100,), t))
            pred = den(torch.cat([xt, (1 - a).sqrt().expand(-1, 1, N, N)], 1))
            l_net = float(((pred - eps) ** 2).mean())
            x0_hat = (xt - (1 - a).sqrt() * pred) / a.sqrt()
            mse0 = float(((x0_hat - X0[idx]) ** 2).mean())
            ratio = (1 - ab[t]) / ab[t]
            worst = max(worst, abs(mse0 / (l_net * ratio) - 1))
            rows["网络的损失"].append(float(f"{l_net:.3g}"))
            rows["把 x_t 原样当作噪声输出的损失"].append(float(f"{float(((xt - eps) ** 2).mean()):.2g}"))
            rows["同上，按公式"].append(float(f"{ab[t] * s2 + (1 - np.sqrt(1 - ab[t])) ** 2:.2g}"))
            rows["网络预测换算成 x0 之后的均方误差"].append(float(f"{mse0:.3g}"))
            rows["换算倍数 (1-abar)/abar"].append(float(f"{ratio:.3g}"))
        tt = torch.randint(0, T, (400,), generator=g)
        xt, eps, a = make(torch.randint(500, 600, (400,), generator=g), tt)
        pred = den(torch.cat([xt, (1 - a).sqrt().expand(-1, 1, N, N)], 1))
        per = ((pred - eps) ** 2).mean(dim=(1, 2, 3)).numpy()
        per_copy = ((xt - eps) ** 2).mean(dim=(1, 2, 3)).numpy()
        late = tt.numpy() >= 500
    NUM["训练目标：预测噪声"] = {
        "说明": f"x0 是 500 张良品图（缩放到约 [-1, 1]，均方 {s2:.3f}），线性噪声表 {T} 步。去噪网络是三层 3x3 卷积（16 个通道），输入 x_t 和一个填满 sqrt(1-abar_t) 的通道，输出对噪声的预测；"
              f"Adam，学习率 3e-3，{steps} 步，每步 {batch} 张图、步数随机。在另外 100 张良品图上按步数评估。“输出全 0”的损失恒为 1。网络很小、训练很短，数字只说明这一个网络",
        **rows,
        "换算关系 x0 误差 = 损失 × (1-abar)/abar 的最大相对偏差": float(f"{worst:.1e}"),
        "对 t 均匀取平均的损失（400 个样本）": {"网络": r3(per.mean()), "把 x_t 原样输出": r3(per_copy.mean())},
        "只看 t 不小于 500 的那一半": {"网络": float(f"{per[late].mean():.2g}"), "把 x_t 原样输出": float(f"{per_copy[late].mean():.2g}")},
        "只看 t 小于 500 的那一半": {"网络": r3(per[~late].mean()), "把 x_t 原样输出": r3(per_copy[~late].mean())},
        "去噪网络的参数个数": int(sum(p.numel() for p in den.parameters())),
    }


def numbers_misc():
    """VAE 重参数化的几个数。"""
    rng = np.random.default_rng([988, 10])
    n = 200000
    mu, sigma = 2.0, 3.0
    eps = rng.normal(size=n)
    z = mod.reparameterize(np.full(n, mu), np.full(n, np.log(sigma ** 2)), eps)
    wrong = mu + np.exp(np.log(sigma ** 2)) * eps
    logq = -0.5 * np.log(2 * np.pi * sigma ** 2) - (z - mu) ** 2 / (2 * sigma ** 2)
    logp = -0.5 * np.log(2 * np.pi) - z ** 2 / 2
    path = 2 * z                                   # 重参数化（路径）梯度：d(z^2)/dmu = 2z
    scoref = z ** 2 * (z - mu) / sigma ** 2        # 打分函数（REINFORCE）梯度
    NUM["VAE 重参数化"] = {
        "说明": f"mu={mu:.0f}、方差 {sigma ** 2:.0f}（logvar=log 9）、{n} 个标准正态样本，调用作业的 reparameterize。梯度一项估计的是 E[z^2] 对 mu 的导数，真值是 2·mu",
        "z 的均值": r3(z.mean()), "z 的标准差": r3(z.std()),
        "把 exp(0.5·logvar) 写成 exp(logvar) 时 z 的标准差": r3(wrong.std()),
        "KL 散度的闭式值 0.5·(mu^2 + sigma^2 - 1 - log sigma^2)": r3(0.5 * (mu ** 2 + sigma ** 2 - 1 - np.log(sigma ** 2))),
        "用重参数化采样估计的 KL 散度": r3((logq - logp).mean()),
        "梯度的真值": 2 * mu,
        "重参数化估计的梯度：均值": r3(path.mean()), "重参数化估计的梯度：单个样本的标准差": r3(path.std()),
        "打分函数估计的梯度：均值": r3(scoref.mean()), "打分函数估计的梯度：单个样本的标准差": r3(scoref.std()),
        "两种估计的方差之比": round(float(scoref.var() / path.var()), 1),
    }


def numbers_mask():
    """作业里划痕掩膜的几个取值，和两种常见写错的后果。"""
    shape, p0, p1 = (20, 30), (10.0, 5.0), (10.0, 25.0)
    m1 = mod.scratch_mask(shape, p0, p1, 3.0, 1.0)
    m2 = mod.scratch_mask(shape, p0, p1, 3.0, 2.0)
    y, x = np.mgrid[0:20, 0:30].astype(np.float64)
    d_line = np.abs(y - 10.0)                                   # 写错一：按无限长直线算距离
    wrong_line = np.clip((1.5 + 0.5 - d_line) / 1.0, 0, 1)
    t = np.clip((x - 5.0) / 20.0, 0, 1)
    d_seg = np.hypot(y - 10.0, x - (5.0 + 20.0 * t))
    wrong_half = np.clip((1.5 - d_seg) / 1.0, 0, 1)            # 写错二：漏了 softness/2
    NUM["划痕掩膜的取值"] = {
        "说明": "20x30 的图上一条水平划痕，从 (10, 5) 到 (10, 25)，宽 3 px。调用作业的 scratch_mask，取离中心线 0、1、2、3 px 处的值",
        "过渡带 1 px": [float(m1[10 + k, 15]) for k in range(4)],
        "过渡带 2 px": [float(m2[10 + k, 15]) for k in range(4)],
        "掩膜取值之和（过渡带 1 px）": r3(m1.sum()),
        "掩膜取值之和（过渡带 2 px）": r3(m2.sum()),
        "写错一（按无限长直线算距离）：掩膜取值之和": r3(wrong_line.sum()),
        "写错一：端点外 3 px 处（第 28 列）中心线上的值，应为 0": float(wrong_line[10, 28]),
        "写错二（漏了 softness/2）：离中心线 1 px 处的值，应为 1": float(wrong_half[11, 15]),
        "写错二：掩膜取值之和": r3(wrong_half.sum()),
    }
    n = {k: int(v.sum()) for k, v in GROUPS.items()}
    NUM["检出率的抽样误差"] = {
        "说明": "测试集的缺陷个数有限，检出率 p 的标准误差是 sqrt(p(1-p)/n)。这里只算抽样误差，不含换训练种子带来的波动",
        "全部 1500 个缺陷，检出率 0.8 时": r3(math.sqrt(0.8 * 0.2 / 1500)),
        f"暗缺陷 {n['暗']} 个，检出率 0.8 时": r3(math.sqrt(0.8 * 0.2 / n["暗"])),
        f"弱缺陷 {n['弱（幅度低于 0.2）']} 个，检出率 0.6 时": r3(math.sqrt(0.6 * 0.4 / n["弱（幅度低于 0.2）"])),
        f"中心的缺陷 {n['中心']} 个，检出率 0.8 时": r3(math.sqrt(0.8 * 0.2 / n["中心"])),
    }


# ---------------------------------------------------------------- 图 1 到图 4，图 7
def bar_labels(ax, bars, vals, fmt="{:.2f}", dy=0.012, fs=8.5):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + dy, fmt.format(v), ha="center", fontsize=fs)


def fig_real_count():
    rows = RES["真实数量"]
    rng = np.random.default_rng([987, 11])
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 1, 2.2]}, layout="constrained")
    clean = clean_shifted(rng.uniform(-4, 4, size=(4, 2)))
    metas = [None] + [real_defect(clean[i + 1], rng, kind=k, amp_range=(0.36, 0.42)) for i, k in enumerate(("particle", "scratch", "pattern"))]
    imgs = image_chain(clean, rng)
    for a, im, meta, title in zip(ax[:4], imgs, metas, ("良品", "颗粒", "划痕", "图形缺失或桥接")):
        a.imshow(im, cmap="gray", vmin=0.1, vmax=1.0, interpolation="nearest")
        if meta is not None:
            a.contour(meta[4], levels=[0.5], colors="cyan", linewidths=1.1)
        a.set_title(title)
        S.image_axes(a)
    ks = list(rows)
    x = np.arange(len(ks))
    for col, color, marker, label in ((0, S.BLUE, "o", "AUROC"), (1, S.ORANGE, "s", "误报 5% 时的检出率")):
        for i, k in enumerate(ks):
            ax[4].plot([i] * len(rows[k]), [r[col] for r in rows[k]], marker, color=color, ms=3.5, alpha=0.35)
        vals = [np.mean([r[col] for r in rows[k]]) for k in ks]
        ax[4].plot(x, vals, marker + "-", color=color, lw=1.9, ms=6, label=label + "（浅色点是各个种子）")
        for i, v in enumerate(vals):
            ax[4].text(i + (0.07 if col == 1 and i == 0 else 0), v + (0.045 if col == 0 else -0.02 if i == 0 else -0.095), f"{v:.2f}",
                       ha="left" if col == 1 and i == 0 else "center", fontsize=8.5, color=color)
    ax[4].axhline(0.5, color=S.MUTED, lw=1, ls="--")
    ax[4].text(len(ks) - 1.02, 0.515, "AUROC 0.5 = 瞎猜", ha="right", fontsize=8.5, color=S.MUTED)
    ax[4].set_xticks(x, [str(k) for k in ks])
    ax[4].set_xlabel("训练用的真实缺陷图张数")
    ax[4].set_ylim(0, 1.05)
    ax[4].set_title("真实缺陷只有 8 张时，分类器接近瞎猜")
    ax[4].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[4])
    S.save(fig, CH, "fig01_真实样本数量")


def fig_stage():
    rows = RES["插入阶段"]
    rng = np.random.default_rng([986, 12])
    clean = clean_shifted(np.array([[1.3, 2.1]]))[0]
    m = mod.scratch_mask((N, N), (11.0, 9.0), (21.0, 23.0), 1.5, 1.0)
    amp = 0.3
    final = image_chain(clean[None], rng)[0]
    shots = [image_chain(mod.blend(clean, clean + amp, m)[None], rng)[0],
             mod.blend(final, final + amp, m),
             mod.blend(final, float((final * m).sum() / m.sum()) + amp, m),
             mod.blend(final, final + amp, np.clip(blur(m), 0, 1))]
    titles = ("成像前插入（正确）", "成像后直接加", "成像后替换成固定灰度", "成像后，掩膜先模糊再加")
    fig, ax = plt.subplots(1, 6, figsize=(17, 3.5), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25, 2.6]}, layout="constrained")
    for a, im, title in zip(ax[:4], shots, titles):
        a.imshow(im[4:28, 4:28], cmap="gray", vmin=0.1, vmax=1.0, interpolation="nearest")
        a.set_title(title, fontsize=10.5)
        S.image_axes(a)
    mh = mod.scratch_mask((N, N), (16.0, 6.0), (16.0, 26.0), 1.5, 1.0)
    d = np.arange(N) - 16
    ax[4].plot(d, amp * mh[:, 16], "s-", color=S.RED, lw=1.5, ms=4, label="成像后直接画")
    ax[4].plot(d, amp * blur(mh)[:, 16], "o-", color=S.BLUE, lw=1.5, ms=4, label="成像前插入")
    ax[4].set_xlim(-6, 6)
    ax[4].set_xlabel("离划痕中心线的距离（px）")
    ax[4].set_ylabel("缺陷带来的灰度变化")
    ax[4].set_title("同一条划痕的横截面", fontsize=10.5)
    ax[4].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
    S.clean(ax[4])
    names = list(rows)
    weak = 2 + list(GROUPS).index("弱（幅度低于 0.2）")
    x = np.arange(len(names))
    for i, (col, color, label) in enumerate(((0, S.BLUE, "真实缺陷上的 AUROC"), (weak, S.ORANGE, "弱的真实缺陷的检出率"))):
        vals = [np.mean([r[col] for r in rows[n]]) for n in names]
        bar_labels(ax[5], ax[5].bar(x + (i - 0.5) * 0.38, vals, 0.36, color=color, alpha=0.88, label=label), vals)
    ax[5].set_xticks(x, ["成像前\n软掩膜", "成像前\n硬掩膜", "成像后\n直接加", "成像后\n硬掩膜", "成像后\n固定灰度", "成像后\n掩膜先模糊"], fontsize=9)
    ax[5].set_ylim(0, 1.12)
    ax[5].set_title("在成像后直接画，弱缺陷的检出率掉得最多")
    ax[5].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[5])
    ax[5].grid(axis="x", visible=False)
    S.save(fig, CH, "fig02_插入阶段")


def fig_paste():
    rows, ex = RES["剪贴"], RES["剪贴示例"]
    seam = NUM["剪贴融合"]["痕迹统计（不训练网络）"]
    fig, ax = plt.subplots(1, 6, figsize=(17.5, 3.6), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.5, 2.2]}, layout="constrained")
    sy, sx = ex["box"]
    ax[0].imshow(ex["source"], cmap="gray", vmin=0.1, vmax=1.0, interpolation="nearest")
    ax[0].add_patch(plt.Rectangle((sx - 0.5, sy - 0.5), PB, PB, fill=False, ec="cyan", lw=1.2))
    ax[0].set_title("源图：框里是要剪的图块", fontsize=10.5)
    for a, method, title in zip(ax[1:4], ("direct", "feather", "poisson"), ("直接粘贴", "羽化融合", "泊松融合")):
        a.imshow(ex[method], cmap="gray", vmin=0.1, vmax=1.0, interpolation="nearest")
        for cy, cx, dy, dx in ((10, 9, -1, -1), (10, 9 + PB, -1, 1), (10 + PB, 9, 1, -1), (10 + PB, 9 + PB, 1, 1)):
            a.plot([cx - 0.5, cx - 0.5 + 2.5 * dx], [cy - 0.5, cy - 0.5], color="cyan", lw=1.2)
            a.plot([cx - 0.5, cx - 0.5], [cy - 0.5, cy - 0.5 + 2.5 * dy], color="cyan", lw=1.2)
        a.set_title(title + "（相位随机）", fontsize=10.5)
    for a in ax[:4]:
        S.image_axes(a)
        a.set_xlim(-0.5, N - 0.5)
        a.set_ylim(N - 0.5, -0.5)
    x = np.arange(3)
    key = "可分性（两处取 max(A, 1-A) 的较大者）"
    for i, (tag, color) in enumerate((("相位随机", S.RED), ("相位对齐", S.MUTED))):
        vals = [seam[f"{m}，{tag}"][key] for m in ("直接粘贴", "羽化融合", "泊松融合")]
        bar_labels(ax[4], ax[4].bar(x + (i - 0.5) * 0.38, vals, 0.36, color=color, alpha=0.88, label=tag), vals)
    ax[4].axhline(0.5, color=S.MUTED, lw=1, ls="--")
    ax[4].set_xticks(x, ["直接", "羽化", "泊松"])
    ax[4].set_ylim(0.4, 1.08)
    ax[4].set_ylabel("跨边界差的可分性（0.5 = 看不出）")
    ax[4].set_title("图块边界上的痕迹", fontsize=10.5)
    ax[4].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    S.clean(ax[4])
    ax[4].grid(axis="x", visible=False)
    names = list(rows)
    x = np.arange(len(names))
    probe = 2 + len(GROUPS)
    for i, (col, color, label) in enumerate(((0, S.BLUE, "真实缺陷上的 AUROC"), (probe, S.RED, "无缺陷粘贴被报警的比例"))):
        vals = [np.mean([r[col] for r in rows[n]]) for n in names]
        bar_labels(ax[5], ax[5].bar(x + (i - 0.5) * 0.38, vals, 0.36, color=color, alpha=0.88, label=label), vals)
    ax[5].set_xticks(x, ["直接\n相位随机", "羽化\n相位随机", "泊松\n相位随机", "泊松\n相位对齐", "直接，相位随机\n良品也贴一块"], fontsize=9)
    ax[5].set_ylim(0, 1.14)
    ax[5].set_title("相位随机时，贴了一块没有缺陷的图块也会报警")
    ax[5].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[5])
    ax[5].grid(axis="x", visible=False)
    S.save(fig, CH, "fig04_三种贴法")


def fig_diversity():
    rows = RES["多样性"]
    names = list(rows)
    cols = ["全部"] + list(GROUPS)
    mat = np.array([np.mean(rows[n], axis=0)[1:2 + len(GROUPS)] for n in names])
    fig, ax = plt.subplots(figsize=(13.5, 3.9), layout="constrained")
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{100 * mat[i, j]:.0f}", ha="center", va="center", fontsize=10, color=S.INK)
    short = ["全部", "亮", "暗", "中心", "非中心", "弱", "中", "强", "颗粒", "划痕", "图形"]
    ax.set_xticks(np.arange(len(cols)), short)
    ax.set_yticks(np.arange(len(names)), [f"{n}（AUROC {np.mean([r[0] for r in rows[n]]):.3f}）" for n in names])
    for xline in (0.5, 2.5, 4.5, 7.5):
        ax.axvline(xline, color="white", lw=3)
    ax.set_xlabel("真实缺陷的分组（格子里是误报 5% 时的检出率，%）")
    ax.set_ylabel("训练用的合成缺陷")
    ax.set_title("只合成亮缺陷，暗缺陷大半检不出；只合成强缺陷，弱缺陷没有少检")
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01).ax.tick_params(labelsize=8)
    S.save(fig, CH, "fig03_多样性")


def fig_mixing():
    rows = RES["混合"]
    dark = 2 + list(GROUPS).index("暗")
    bright = 2 + list(GROUPS).index("亮")

    def val(key, col=0):
        return float(np.mean([r[col] for r in rows[key]]))

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.1), gridspec_kw={"width_ratios": [1, 1.45]}, layout="constrained")
    good = [val((8, "无", 0, False)), val((8, "好", 64, False)), val((8, "好", 512, False))]
    ax[0].plot(range(3), good, "o-", color=S.BLUE, lw=1.9, ms=6, label="加好的合成（全部随机）")
    nv = val((8, "窄", 512, False))
    ax[0].plot([2], [nv], "s", color=S.ORANGE, ms=8, label="加窄的合成（只有亮的强划痕）")
    for i, v in enumerate(good):
        ax[0].text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=8.5, color=S.BLUE)
    ax[0].text(2, nv - 0.06, f"{nv:.2f}", ha="center", fontsize=8.5, color=S.ORANGE)
    ax[0].axhline(0.5, color=S.MUTED, lw=1, ls="--")
    ax[0].set_xticks(range(3), ["0", "64", "512"])
    ax[0].set_xlim(-0.3, 2.3)
    ax[0].set_xlabel("加进去的合成缺陷图张数（真实缺陷图 8 张）")
    ax[0].set_ylabel("真实缺陷上的 AUROC")
    ax[0].set_ylim(0.4, 1.02)
    ax[0].set_title("真实只有 8 张：加合成就有用，窄的也有用")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax[0])
    keys = [(256, "无", 0, False)] + [(256, "窄", m, False) for m in (64, 512, 2048)]
    extra = [(256, "窄", 2048, True), (256, "好", 2048, False)]
    for col, color, marker, label, dy in ((1, S.INK, "o", "全部真实缺陷", 0.0), (bright, S.ORANGE, "^", "亮缺陷", 0.035), (dark, S.PURPLE, "v", "暗缺陷", -0.05)):
        vals = [val(k, col) for k in keys]
        ax[1].plot(range(4), vals, marker + "-", color=color, lw=1.8, ms=6, label=label)
        for xpos, key in zip((4, 5), extra):
            v = val(key, col)
            ax[1].plot([xpos], [v], marker, color=color, ms=8, mfc="white", mew=1.7)
            ax[1].text(xpos + 0.12, v, f"{v:.2f}", va="center", fontsize=8.5, color=color)
        if col != 1:
            ax[1].text(-0.12, vals[0] - 0.6 * dy, f"{vals[0]:.2f}", ha="right", va="center", fontsize=8.5, color=color)
            ax[1].text(3, vals[3] + dy, f"{vals[3]:.2f}", ha="center", va="center", fontsize=8.5, color=color)
    ax[1].axvline(3.5, color=S.GRID, lw=1.2)
    ax[1].set_xticks(range(6), ["0", "64", "512", "2048", "窄的 2048\n按来源配平", "换成好的\n合成 2048"])
    ax[1].set_xlim(-0.55, 5.6)
    ax[1].set_xlabel("加进去的窄的合成缺陷图张数（真实缺陷图 256 张）；右边两组空心点是对照")
    ax[1].set_ylabel("误报 5% 时的检出率")
    ax[1].set_ylim(0.4, 1.0)
    ax[1].set_title("真实 256 张：窄的合成加到 512 张以上，暗缺陷的检出率明显下降")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3)
    S.clean(ax[1])
    S.save(fig, CH, "fig07_真实加合成")


def main() -> None:
    S.setup()
    t_all = time.perf_counter()
    for fn in (numbers_misc, numbers_mask, run_real_count, run_stage, run_paste, run_diversity, run_mixing, table_fd, fig_forward, table_denoiser,
               fig_real_count, fig_stage, fig_diversity, fig_paste, fig_mixing):
        t = time.perf_counter()
        fn()
        print(f"完成 {fn.__name__}（{time.perf_counter() - t:.0f} 秒，累计训练 {N_FITS} 次）", flush=True)
    NUM["真实缺陷样本数量"]["共训练了多少次"] = N_FITS

    def one_fit():
        global N_FITS
        t = time.perf_counter()
        fit(real_set(np.random.default_rng([0, 64]), 64), neg_of(0), 0)
        N_FITS -= 1
        return {"说明": f"在一台 Mac 的 CPU（2 个线程）上训练一次分类器（{STEPS} 步，每步 {BATCH} 张 {N}x{N} 的图）的耗时，只测了一次；同时有别的任务在跑，换机器会变",
                "秒": float(f"{time.perf_counter() - t:.2g}")}

    NUM["训练一次分类器的耗时"] = S.pinned(CH, "训练一次分类器的耗时", one_fit, "CH20_RETIME")
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=2))
    print(f"总耗时 {time.perf_counter() - t_all:.0f} 秒")


if __name__ == "__main__":
    main()
