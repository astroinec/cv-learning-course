"""第 27 章的小模型：一个能在 CPU 上一两分钟训练完的“可提示分割器”。

这个文件不是作业，可以直接读。它包含三部分：

1. 合成数据：64x64 的灰度图，上面有两三个物体，物体里可能套着部件，部件里可能再套一个小部件，
   所以点一下常常有两三种合理的答案（整体、部件、小部件）。
2. 模型：图像编码器（只跑一次）、提示编码（点和框变成向量）、掩码解码器（很轻，每个提示跑一次）。
   结构照 SAM 的思路缩小：解码器里有 1 个“预测 IoU”的输出向量和 4 个“掩码”输出向量，
   第 0 个掩码是单输出（任何时候都被监督），第 1 到 3 个是多输出（训练时只回传损失最小的那一个）。
3. 训练和几个评估时反复用到的小工具（按点提示跑解码器、模拟修正点击）。

点提示的位置编码和“由掩码求外接框”是本章作业里的两个函数，这里不实现，由调用方传进来（参数 pe_fn 和 box_fn）。
所以 Notebook 要等你写完 `point_position_encoding` 和 `mask_to_box` 才能训练这个模型。

和真 SAM 不一样的地方（为了小和快）：
- 编码器是几层卷积，不是 ViT；输入 64x64，特征图 16x16。
- 点提示的向量除了“位置编码 + 类型向量”之外，还加上了该点位置的图像特征（在三个分辨率的特征图上各做一次双线性采样）。
  真 SAM 的提示编码器不看图像，靠解码器里的交叉注意力自己去图像特征里查；
  小模型只训练一千多步，把“去点击处读特征”这一步直接接好，学得快很多。
- “预测 IoU”的分支除了看 IoU 输出向量，还看这张掩码自己的输出向量和几个直接从掩码上算出来的量
  （前景比例、边界有多糊、掩码里面的平均特征）。真 SAM 的这个分支只看 IoU 输出向量；
  小模型训练步数太少，只靠那一个向量排不好候选的先后（iou_sees_mask=False 就是只看向量的写法，讲义第 4 节有对照）。
- 没有“上一轮的掩码”这个稠密提示，交互时只把所有点一起重新送进解码器。
- 为了代码简单，这个小模型固定在 CPU 上跑；用显卡的写法见讲义“在自己的机器上用真模型”一节。
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N = 64            # 图像边长
G = 16            # 特征图边长（下采样 4 倍）
DIM = 32          # 特征维度
N_OUT = 4         # 掩码输出个数：第 0 个是单输出，第 1 到 3 个是多输出
NEG, POS, BOX_TL, BOX_BR, PAD = 0, 1, 2, 3, 4   # 提示的类型：负点、正点、框的左上角、框的右下角、占位

_YY, _XX = np.mgrid[0:N, 0:N]


# ---------------------------------------------------------------- 合成数据
def _smooth_noise(rng, sigma=2.0):
    f = np.fft.fftfreq(N)
    h = np.exp(-2 * (np.pi * sigma) ** 2 * (f[:, None] ** 2 + f[None, :] ** 2))
    x = np.fft.ifft2(np.fft.fft2(rng.normal(size=(N, N))) * h).real
    return x / x.std()


def _shape(kind, cy, cx, r, ratio, ang):
    """kind 0 圆，1 旋转的矩形，2 旋转的椭圆。"""
    dy, dx = _YY - cy, _XX - cx
    u = dx * np.cos(ang) + dy * np.sin(ang)
    v = -dx * np.sin(ang) + dy * np.cos(ang)
    if kind == 0:
        return u * u + v * v <= r * r
    if kind == 1:
        return (np.abs(u) <= r) & (np.abs(v) <= r * ratio)
    return (u / r) ** 2 + (v / (r * ratio)) ** 2 <= 1.0


def _gray(rng, parent, lo, hi):
    """取一个和 parent 相差 lo 到 hi 的灰度，亮暗随机；超出 0.05 到 0.95 就换方向。"""
    c = rng.uniform(lo, hi) * rng.choice([-1.0, 1.0])
    g = parent + c
    if not 0.05 <= g <= 0.95:
        g = parent - c
    return float(np.clip(g, 0.05, 0.95))


def make_scene(rng, contrast=(0.15, 0.4), texture=0.04, noise=0.03, n_obj=None, min_sub_r=2.5):
    """生成一张图和它的全部掩码。

    返回 img (64, 64) float32，masks (M, 64, 64) bool，level (M,) int（0 整体，1 部件，2 小部件），
    parent (M,) int（上一级掩码的下标，整体是 -1）。
    后画的物体会挡住先画的；掩码只含看得见的部分。整体的掩码包含它的部件，部件的掩码包含它的小部件。
    contrast 是相邻两级之间灰度差的范围。
    """
    bg = rng.uniform(0.3, 0.7)
    img = np.full((N, N), bg)
    masks, level, parent = [], [], []
    for _ in range(n_obj or int(rng.integers(2, 4))):
        r = rng.uniform(10, 15)
        cy, cx = rng.uniform(r * 0.7, N - r * 0.7, 2)
        whole = _shape(int(rng.integers(0, 3)), cy, cx, r, rng.uniform(0.7, 1.0), rng.uniform(0, np.pi))
        for i in range(len(masks)):
            masks[i] = masks[i] & ~whole
        g1 = _gray(rng, bg, *contrast)
        img[whole] = g1
        masks.append(whole); level.append(0); parent.append(-1)
        w_idx = len(masks) - 1
        if rng.random() < 0.75:
            r2 = rng.uniform(5.0, 0.6 * r)
            a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0, max(0.0, 0.65 * r - r2))
            cy2, cx2 = cy + d * np.sin(a), cx + d * np.cos(a)
            part = _shape(int(rng.integers(0, 3)), cy2, cx2, r2, rng.uniform(0.75, 1.0), rng.uniform(0, np.pi)) & whole
            g2 = _gray(rng, g1, *contrast)
            img[part] = g2
            masks.append(part); level.append(1); parent.append(w_idx)
            p_idx = len(masks) - 1
            if rng.random() < 0.65:
                r3 = rng.uniform(min_sub_r, max(min_sub_r + 0.1, 0.55 * r2))
                a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0, max(0.0, 0.7 * r2 - r3))
                sub = _shape(0, cy2 + d * np.sin(a), cx2 + d * np.cos(a), r3, 1.0, 0.0) & part
                img[sub] = _gray(rng, g2, *contrast)
                masks.append(sub); level.append(2); parent.append(p_idx)
    img = img + texture * _smooth_noise(rng) + rng.normal(scale=noise, size=(N, N))
    # 被挡得只剩很少像素的掩码不要；它的下级跟着去掉
    keep = [i for i, m in enumerate(masks) if m.sum() >= 12]
    keep = [i for i in keep if parent[i] < 0 or parent[i] in keep]
    keep = [i for i in keep if parent[i] < 0 or parent[i] in keep]
    remap = {old: new for new, old in enumerate(keep)}
    return (img.astype(np.float32), np.stack([masks[i] for i in keep]), np.array([level[i] for i in keep]),
            np.array([remap.get(parent[i], -1) for i in keep]))


class SceneSet:
    """一批场景，掩码补齐成同样的个数方便批量取用。"""

    def __init__(self, seed, n, **kw):
        rng = np.random.default_rng(seed)
        scenes = [make_scene(rng, **kw) for _ in range(n)]
        self.count = np.array([len(s[1]) for s in scenes])
        m_max = int(self.count.max())
        self.images = torch.from_numpy(np.stack([s[0] for s in scenes]))[:, None]          # (n, 1, 64, 64)
        self.masks = torch.zeros(n, m_max, N, N, dtype=torch.bool)                         # (n, m_max, 64, 64)
        self.level = torch.full((n, m_max), -1, dtype=torch.long)
        self.parent = torch.full((n, m_max), -1, dtype=torch.long)
        for i, (_, m, lv, pa) in enumerate(scenes):
            self.masks[i, :len(m)] = torch.from_numpy(m)
            self.level[i, :len(m)] = torch.from_numpy(lv)
            self.parent[i, :len(m)] = torch.from_numpy(pa)
        self.depth = self.masks.sum(1)                                                    # 每个像素被几个掩码包含（0 是背景）

    def __len__(self):
        return len(self.images)


# ---------------------------------------------------------------- 模型
class Attention(nn.Module):
    """多头注意力。down 是内部降维的倍数（SAM 的解码器里交叉注意力降一半）。"""

    def __init__(self, dim, heads=2, down=1):
        super().__init__()
        self.heads, self.inner = heads, dim // down
        self.q, self.k, self.v = nn.Linear(dim, self.inner), nn.Linear(dim, self.inner), nn.Linear(dim, self.inner)
        self.out = nn.Linear(self.inner, dim)

    def forward(self, q, k, v, key_pad=None):
        b = q.shape[0]

        def split(x):
            return x.view(b, x.shape[1], self.heads, self.inner // self.heads).transpose(1, 2)

        q, k, v = split(self.q(q)), split(self.k(k)), split(self.v(v))
        a = (q @ k.transpose(-1, -2)) / math.sqrt(self.inner // self.heads)
        if key_pad is not None:                      # 占位的提示不参与
            a = a.masked_fill(key_pad[:, None, None, :], float("-inf"))
        x = (a.softmax(-1) @ v).transpose(1, 2).reshape(b, -1, self.inner)
        return self.out(x)


class TwoWayBlock(nn.Module):
    """一层“双向”解码：提示之间自注意力，提示查图像，MLP，图像查提示。"""

    def __init__(self, dim):
        super().__init__()
        self.self_attn = Attention(dim)
        self.tok2img = Attention(dim, down=2)
        self.img2tok = Attention(dim, down=2)
        self.n1, self.n2, self.n3, self.n4 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))

    def forward(self, tok, src, tok_pe, pos, pad):
        q = tok + tok_pe
        tok = self.n1(tok + self.self_attn(q, q, tok, pad))
        tok = self.n2(tok + self.tok2img(tok + tok_pe, src + pos, src))
        tok = self.n3(tok + self.mlp(tok))
        src = self.n4(src + self.img2tok(src + pos, tok + tok_pe, tok, pad))
        return tok, src


class ImageEncoder(nn.Module):
    """图像编码器：每张图只跑一次。

    输出三样东西：16x16 的特征图（解码器的注意力在它上面做），以及 32x32 和 64x64 两张浅层特征（解码器放大掩码时用来找回边界）。
    """

    def __init__(self, dim=DIM, width=48, depth=4):
        super().__init__()
        self.stem0 = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1), nn.ReLU())                                         # 64x64
        self.stem1 = nn.Sequential(nn.Conv2d(8, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU())  # 32x32
        layers = [nn.Conv2d(16, width, 3, 2, 1), nn.ReLU()]                                                     # 16x16
        for i in range(depth):                      # 空洞卷积把感受野撑到能盖住整个物体
            d = (1, 2, 3, 1)[i % 4]
            layers += [nn.Conv2d(width, width, 3, 1, d, dilation=d), nn.ReLU()]
        self.body = nn.Sequential(*layers, nn.Conv2d(width, dim, 1))
        self.skip1 = nn.Conv2d(16, dim // 2, 1)
        self.skip0 = nn.Conv2d(8, dim // 4, 1)

    def forward(self, x):
        h0 = self.stem0((x - 0.5) / 0.25)
        h1 = self.stem1(h0)
        return self.body(h1), self.skip1(h1), self.skip0(h0)


class MaskDecoder(nn.Module):
    """提示编码加掩码解码：每个提示跑一次，很轻。"""

    def __init__(self, pe_fn, dim=DIM, depth=2, seed=0, iou_sees_mask=True):
        super().__init__()
        self.pe_fn = pe_fn
        self.iou_sees_mask = iou_sees_mask
        gen = torch.Generator().manual_seed(1000 + seed)
        self.register_buffer("freqs", torch.randn(2, dim // 2, generator=gen))    # 位置编码用的随机频率，不训练
        self.type_embed = nn.Embedding(5, dim)             # 负点、正点、框左上、框右下、占位
        self.point_feat = nn.ModuleList([nn.Linear(dim, dim), nn.Linear(dim // 2, dim), nn.Linear(dim // 4, dim)])   # 点击处的图像特征（见文件开头的说明）
        self.out_tokens = nn.Embedding(1 + N_OUT, dim)     # 第 0 个预测 IoU，后面 4 个各出一张掩码
        self.blocks = nn.ModuleList(TwoWayBlock(dim) for _ in range(depth))
        self.final_attn = Attention(dim, down=2)
        self.final_norm = nn.LayerNorm(dim)
        self.up1 = nn.ConvTranspose2d(dim, dim // 2, 2, 2)
        self.mix = nn.Conv2d(dim // 2, dim // 2, 3, 1, 1)
        self.up2 = nn.ConvTranspose2d(dim // 2, dim // 4, 2, 2)
        self.hyper = nn.ModuleList(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim // 4)) for _ in range(N_OUT))
        if iou_sees_mask:
            self.iou_head = nn.Sequential(nn.Linear(2 * dim + dim // 4 + 2, dim), nn.ReLU(), nn.Linear(dim, 1))
        else:                                              # 真 SAM 的写法：只看 IoU 输出向量，一次给出 4 个分数
            self.iou_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, N_OUT))
        g = torch.arange(G, dtype=torch.float32) * (N / G) + (N / G - 1) / 2      # 每个特征格子中心的像素坐标
        self.register_buffer("grid", torch.stack(torch.meshgrid(g, g, indexing="xy"), -1).reshape(-1, 2))

    def encode_prompts(self, feats, points, labels):
        """points (B, P, 2) 是像素坐标 (x, y)，labels (B, P) 取 0 到 4。返回 (B, P, dim)。"""
        real = (labels != PAD)[..., None]
        pe = self.pe_fn(points, (N, N), self.freqs)
        grid = ((points + 0.5) / N * 2 - 1)[:, :, None]
        feat = 0
        for f, proj in zip(feats, self.point_feat):          # 三个分辨率的特征图各在点击处采样一次
            feat = feat + proj(F.grid_sample(f, grid, mode="bilinear", padding_mode="border", align_corners=False)[..., 0].transpose(1, 2))
        return (pe + feat) * real + self.type_embed(labels)

    def forward(self, feats, points, labels):
        """feats 是编码器的三个输出，形状 (B, dim, 16, 16)、(B, dim/2, 32, 32)、(B, dim/4, 64, 64)。

        返回掩码 logits (B, 4, 64, 64) 和预测的 IoU (B, 4)。
        """
        emb, skip1, skip0 = feats
        b = emb.shape[0]
        pad = labels == PAD
        prompts = self.encode_prompts(feats, points, labels)
        tok = torch.cat([self.out_tokens.weight[None].expand(b, -1, -1), prompts], 1)
        pad = torch.cat([torch.zeros(b, 1 + N_OUT, dtype=torch.bool, device=pad.device), pad], 1)
        tok_pe = tok
        src = emb.flatten(2).transpose(1, 2)
        pos = self.pe_fn(self.grid, (N, N), self.freqs)[None]
        for blk in self.blocks:
            tok, src = blk(tok, src, tok_pe, pos, pad)
        tok = self.final_norm(tok + self.final_attn(tok + tok_pe, src + pos, src))
        up = F.relu(self.up1(src.transpose(1, 2).reshape(b, -1, G, G)) + skip1)
        up = F.relu(self.up2(up + F.relu(self.mix(up))) + skip0)
        weights = torch.stack([h(tok[:, 1 + i]) for i, h in enumerate(self.hyper)], 1)
        masks = torch.einsum("bmc,bchw->bmhw", weights, up)
        if not self.iou_sees_mask:
            return masks, self.iou_head(tok[:, 0])
        # IoU 分支的输入：IoU 向量、这张掩码的输出向量，再加上“看一眼这张掩码”得到的几个量（见文件开头的说明）
        prob = masks.detach().sigmoid()
        area = prob.flatten(2).mean(-1, keepdim=True)                                     # 前景占整张图的比例
        fuzzy = (prob * (1 - prob)).flatten(2).sum(-1, keepdim=True) / (prob.flatten(2).sum(-1, keepdim=True) + 1)   # 边界有多糊
        pooled = torch.einsum("bmhw,bchw->bmc", prob, up.detach()) / (prob.flatten(2).sum(-1, keepdim=True) + 1)     # 掩码里面的平均特征
        iou_in = torch.cat([tok[:, :1].expand(-1, N_OUT, -1), tok[:, 1:1 + N_OUT], pooled, area, fuzzy], -1)
        return masks, self.iou_head(iou_in)[..., 0]


class MiniSAM(nn.Module):
    def __init__(self, pe_fn, seed=0, iou_sees_mask=True):
        super().__init__()
        self.encoder = ImageEncoder()
        self.decoder = MaskDecoder(pe_fn, seed=seed, iou_sees_mask=iou_sees_mask)

    def encode(self, images):
        return self.encoder(images)

    def decode(self, feats, points, labels):
        return self.decoder(feats, points, labels)


def repeat_feats(feats, index):
    """按下标取出（或重复）若干张图的特征，一张图上的多个提示共用同一份特征。"""
    return tuple(f[index] for f in feats)


# ---------------------------------------------------------------- 训练和评估用的小工具
def mask_iou(a, b):
    """两组布尔掩码逐个算 IoU，形状可广播，最后两维是 H、W。并集为空时记 0。"""
    inter = (a & b).flatten(-2).sum(-1).float()
    union = (a | b).flatten(-2).sum(-1).float()
    return inter / union.clamp(min=1)


def mask_loss(logits, target):
    """logits (B, M, H, W)，target (B, H, W) 布尔 → 每个输出一个损失 (B, M)：5 倍交叉熵加 Dice。"""
    t = target[:, None].float().expand_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none").flatten(2).mean(-1)
    p = logits.sigmoid()
    dice = 1 - (2 * (p * t).flatten(2).sum(-1) + 1) / (p.flatten(2).sum(-1) + t.flatten(2).sum(-1) + 1)
    return 5 * bce + dice


def sample_pixel(region, gen):
    """从每张布尔图里均匀抽一个为真的像素。返回 (B, 2) 的 (x, y) 和 (B,) 的“这张图有没有真像素”。"""
    w = region.flatten(1).float()
    ok = w.sum(1) > 0
    idx = torch.multinomial(w + (~ok)[:, None].float(), 1, generator=gen)[:, 0]
    return torch.stack([idx % N, idx // N], -1).float(), ok


def point_prompt(points, labels, capacity=None):
    """把 (B, k, 2) 的点和 (B, k) 的标签补成固定长度（多出来的位置是占位）。"""
    b, k = labels.shape
    capacity = capacity or k
    pts = torch.zeros(b, capacity, 2)
    lab = torch.full((b, capacity), PAD, dtype=torch.long)
    pts[:, :k], lab[:, :k] = points, labels
    return pts, lab


def box_prompt(boxes, capacity=2):
    """(B, 4) 的框 → 两个角点提示。"""
    b = len(boxes)
    pts = torch.zeros(b, capacity, 2)
    lab = torch.full((b, capacity), PAD, dtype=torch.long)
    pts[:, 0], pts[:, 1] = boxes[:, :2], boxes[:, 2:]
    lab[:, 0], lab[:, 1] = BOX_TL, BOX_BR
    return pts, lab


def sample_targets(data, img_idx, n_prompt, gen, depth_weights=(0.15, 0.25, 0.6)):
    """每张图抽 n_prompt 个前景点；每个点从“包含它的掩码”里等概率抽一个当目标。

    抽点分两步：先在“被 1 个、2 个、3 个掩码包含”这三类像素里按 depth_weights 的比例选一类（图里没有的类不选），
    再在这一类像素里均匀抽一个。默认的比例偏向被包含得多的像素（落在部件和小部件里的点）：
    整张图里这类像素很少，按面积均匀抽的话，部件和小部件几乎轮不到当目标。
    点在小部件里时，小部件、部件、整体各有三分之一的机会是标准答案，歧义是数据里真实存在的。
    返回 点 (B, 2)、目标掩码 (B, H, W)、目标在该图里的下标 (B,)、包含该点的掩码个数 (B,)。
    """
    rep = img_idx.repeat_interleave(n_prompt)
    depth = data.depth[rep]                                                        # (B, H, W)，每个像素被几个掩码包含
    has = torch.stack([(depth == k).flatten(1).any(1) for k in (1, 2, 3)], 1).float() * torch.tensor(depth_weights)
    k = torch.multinomial(has, 1, generator=gen)[:, 0] + 1
    pts, _ = sample_pixel(depth == k[:, None, None], gen)
    inside = data.masks[rep, :, pts[:, 1].long(), pts[:, 0].long()]               # (B, m_max)
    pick = torch.multinomial(inside.float(), 1, generator=gen)[:, 0]
    return pts, data.masks[rep, pick], pick, inside.sum(1)


def _make_batch(model, data, n_img, n_prompt, gen, box_fn):
    """抽一批训练用的提示。返回 特征、点、标签、目标掩码。box_fn 是作业里的 mask_to_box。"""
    b = n_img * n_prompt
    ar = torch.arange(b)
    rep = torch.arange(n_img).repeat_interleave(n_prompt)
    img_idx = torch.randint(0, len(data), (n_img,), generator=gen)
    p0, target, _, _ = sample_targets(data, img_idx, n_prompt, gen)
    feats = repeat_feats(model.encode(data.images[img_idx]), rep)
    pts, lab = point_prompt(p0[:, None], torch.full((b, 1), POS), capacity=5)
    use_box = torch.rand(b, generator=gen) < 0.25
    box = torch.from_numpy(np.asarray(box_fn(target.numpy()))).float()          # (B, 4) 的 [x0, y0, x1, y1]
    side = (box[:, 2:] - box[:, :2] + 1).repeat(1, 2)
    box = (box + torch.randn(b, 4, generator=gen) * 0.05 * side).clamp(0, N - 1)
    bp, bl = box_prompt(box, capacity=5)
    pts, lab = torch.where(use_box[:, None, None], bp, pts), torch.where(use_box[:, None], bl, lab)
    with torch.no_grad():                       # 模拟用户：在选中的掩码错的地方补点
        logits, iou_pred = model.decode(feats, pts, lab)
        chosen = logits[ar, 1 + iou_pred[:, 1:].argmax(1)] > 0
        err = chosen ^ target
        more = torch.rand(b, generator=gen) < 0.5
        used = (lab != PAD).sum(1)
        for j in range(1 + int(torch.randint(0, 3, (1,), generator=gen))):
            p, ok = sample_pixel(err, gen)
            ok = ok & more
            pts[ar[ok], used[ok] + j] = p[ok]
            lab[ar[ok], used[ok] + j] = target[ar, p[:, 1].long(), p[:, 0].long()].long()[ok]
    return feats, pts, lab, target


def train_mini_sam(pe_fn, box_fn, data, steps=1500, n_img=8, n_prompt=4, lr=3e-3, seed=0, multimask=True, iou_weight=5.0,
                   iou_sees_mask=True, log=None):
    """训练小模型。pe_fn、box_fn 是作业里的 point_position_encoding 和 mask_to_box。

    每一步取 n_img 张图，各编码一次；每张图抽 n_prompt 个提示（四分之一是框，其余是一个正点）。
    先不算梯度地解码一遍，对其中一半的提示按“当前选中的掩码错在哪”补 1 到 3 个修正点，再带梯度地解码一遍算损失。

    掩码损失：单输出总是被监督；三个多输出只回传损失最小的那一个。
    multimask=False 时三个多输出各自都被监督（不取最小），用来做对照。
    “预测 IoU”的分支用均方误差拟合每张输出掩码和目标的真实 IoU；iou_sees_mask=False 时它只看 IoU 输出向量（真 SAM 的写法）。
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = MiniSAM(pe_fn, seed=seed, iou_sees_mask=iou_sees_mask)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    for step in range(steps):
        feats, pts, lab, target = _make_batch(model, data, n_img, n_prompt, gen, box_fn)
        logits, iou_pred = model.decode(feats, pts, lab)
        per = mask_loss(logits, target)
        with torch.no_grad():
            true_iou = mask_iou(logits > 0, target[:, None])
        multi = per[:, 1:].min(1).values if multimask else per[:, 1:].mean(1)
        loss = (per[:, 0] + multi).mean() + iou_weight * F.mse_loss(iou_pred, true_iou)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and (step % log == 0 or step == steps - 1):
            print(f"第 {step:4d} 步  损失 {loss.item():.3f}  三个候选里最好的 IoU {true_iou[:, 1:].max(1).values.mean():.3f}", flush=True)
    return model.eval()
