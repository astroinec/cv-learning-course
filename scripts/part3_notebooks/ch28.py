"""生成第 28 章的互动实验 Notebook。

用法（在仓库根目录运行）：
    python scripts/part3_notebooks/ch28.py
修改实验内容时改本文件再重新生成，不要直接手改 .ipynb，避免两边不一致。
Notebook 只调用学员自己在作业文件里的实现，不包含参考答案；
设置环境变量 COURSE_IMPL_DIR 可以让 Notebook 改用另一份实现（维护者验证用）。

本文件自成一体：HELPER 和 notebook() 抄自 scripts/build_track2_notebooks.py，不 import 那个文件。
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = "28_视觉语言模型"
ASSIGNMENT = "28-1_零样本分类与图文检索.py"
OUTPUT = "实验_视觉语言模型.ipynb"

HELPER = '''%matplotlib inline
import importlib.util, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CHAPTER_NAME = "{chapter}"
ASSIGNMENT = "{assignment}"

def find_chapter(name):
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if p.name == name:
            return p
        if (p / name).is_dir():
            return p / name
    raise FileNotFoundError("请在课程目录内启动 Jupyter")

CHAPTER = find_chapter(CHAPTER_NAME)
COURSE_ROOT = CHAPTER.parent

def load_module(chapter_dir, filename):
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / chapter_dir.name if override else chapter_dir
    spec = importlib.util.spec_from_file_location(Path(filename).stem, base / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod = load_module(CHAPTER, ASSIGNMENT)

def attempt(fn, *args, **kwargs):
    """调用作业函数；还没实现时打印提示并返回 None，而不是中断整个 Notebook。"""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        print(f"先完成作业里的 {{fn.__name__}}：{{exc}}")
        return None

try:
    import cv2  # 可选，用于和 OpenCV 对照
except ImportError:
    cv2 = None
print("作业文件：", CHAPTER / ASSIGNMENT, "｜OpenCV：", "可用" if cv2 else "未安装（对照实验会跳过）")
'''


def notebook(title, intro, chapter, assignment, cells, outro):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3 (cv-learning)", "language": "python", "name": "cv-learning"},
        "language_info": {"name": "python", "version": "3"},
    }
    nb.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}")]
    nb.cells.append(nbf.v4.new_code_cell(HELPER.format(chapter=chapter, assignment=assignment)))
    for kind, src in cells:
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nb.cells.append(nbf.v4.new_markdown_cell(outro))
    return nb


SETUP = '''import math, time
import torch
import torch.nn.functional as F
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
print("设备：", device, "｜", torch.cuda.get_device_name(0) if device.type == "cuda" else "没有检测到 CUDA，用 CPU（这里的小模型在 CPU 上一两分钟能跑完）")

PROBES = {   # 用一个最小的输入试一下每个作业函数有没有实现
    "build_zeroshot_classifier": lambda: mod.build_zeroshot_classifier(np.ones((2, 1, 3))),
    "zeroshot_predict": lambda: mod.zeroshot_predict(np.eye(2), np.eye(2), 10.0),
    "recall_at_k": lambda: mod.recall_at_k(np.eye(2), np.eye(2, dtype=bool), 1),
    "modality_gap": lambda: mod.modality_gap(np.eye(2), np.eye(2)),
    "score_regions": lambda: mod.score_regions(np.eye(2), np.eye(2)),
}

def ready(*names):
    """这些作业函数都已经实现才返回 True；没实现的打印提示，不中断 Notebook。"""
    ok = True
    for name in names:
        try:
            PROBES[name]()
        except NotImplementedError as exc:
            print(f"先完成作业里的 {name}：{exc}")
            ok = False
        except Exception as exc:
            print(f"作业里的 {name} 运行出错（{type(exc).__name__}: {exc}），先跑一遍 tests/check_progress.py")
            ok = False
    return ok

SIZE, CELL, P = 32, 16, 8                 # 32x32 的图分成 2x2 个格子，每格至多一个目标；图像塔把图切成 8x8 的块
G, L = SIZE // P, (SIZE // P) ** 2
SHAPES = ["方块", "圆", "三角", "十字"]
COLORS = ["红色", "绿色", "蓝色", "黄色"]
SIZES, TEXS = ["大", "小"], ["实心", "条纹"]
COUNTS, CELLS = ["一个", "两个", "三个"], ["左上", "右上", "左下", "右下"]
VOCAB = ["<pad>"] + COUNTS + SIZES + TEXS + COLORS + SHAPES + CELLS + ["在", "左边", "右边", "图", "里", "有", "这", "是", "空", "背景", "图形", "一张", "的", "照片"]
TOK = {w: i for i, w in enumerate(VOCAB)}
MAXLEN = 12
RGB = torch.tensor([[0.90, 0.22, 0.20], [0.20, 0.78, 0.30], [0.25, 0.42, 0.95], [0.92, 0.84, 0.20]])
EXT = (12, 8)                             # “大”和“小”的边长
KINDS = (0.45, 0.2, 0.3, 0.05)            # 单个目标、多个相同目标、左右两个不同目标、空图
HOLD = ((0, 1), (1, 2), (2, 3), (3, 0))   # 实验 3 训练时拿掉的（颜色，形状）：红圆、绿三角、蓝十字、黄方块
COMBOS = [f"{c} {s}" for c in COLORS for s in SHAPES]      # 类别编号 = 颜色 x 4 + 形状

def gen(*seed):
    return torch.Generator().manual_seed(int(sum(s * 1000 ** i for i, s in enumerate(seed))))

def _bank():
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
STRIPE = torch.where((torch.arange(CELL)[:, None].expand(CELL, CELL) // 2) % 2 == 0, 1.0, 0.3)
CY, CX = torch.meshgrid(torch.arange(CELL, dtype=torch.float32), torch.arange(CELL, dtype=torch.float32), indexing="ij")

def _objects(n, g, combos):
    pick = combos[torch.randint(0, len(combos), (n,), generator=g)]
    return torch.stack([pick[:, 1], pick[:, 0], torch.randint(0, 2, (n,), generator=g), torch.randint(0, 2, (n,), generator=g)], 1)

def sample_scenes(n, g, kinds=KINDS, hold=None, only_hold=False):
    """随机场景。occ[i, c] 表示第 c 个格子里有没有目标，attr[i, c] 是它的（形状，颜色，大小，纹理）。"""
    all_c = [(c, s) for c in range(4) for s in range(4)]
    if hold is not None:
        all_c = [cs for cs in all_c if (cs in hold) == only_hold]
    combos = torch.tensor(all_c)
    kind = torch.multinomial(torch.tensor(kinds), n, replacement=True, generator=g)
    a1, a2 = _objects(n, g, combos), _objects(n, g, combos)
    for _ in range(30):
        clash = (a1[:, 0] == a2[:, 0]) | (a1[:, 1] == a2[:, 1])
        a2 = torch.where(clash[:, None], _objects(n, g, combos), a2)
    clash = (a1[:, 0] == a2[:, 0]) | (a1[:, 1] == a2[:, 1])
    kind = torch.where((kind == 2) & clash, torch.zeros_like(kind), kind)
    rank = torch.rand(n, 4, generator=g).argsort(1).argsort(1)
    cnt, row, cell = torch.randint(2, 4, (n,), generator=g), torch.randint(0, 2, (n,), generator=g), torch.arange(4)[None]
    in_row = (cell // 2) == row[:, None]
    occ = torch.where((kind == 0)[:, None], rank == 0, torch.zeros(n, 4, dtype=torch.bool))
    occ = torch.where((kind == 1)[:, None], rank < cnt[:, None], occ)
    occ = torch.where((kind == 2)[:, None], in_row, occ)
    attr = a1[:, None, :].expand(n, 4, 4).clone()
    attr = torch.where(((kind == 2)[:, None] & in_row & (cell % 2 == 1))[..., None], a2[:, None, :], attr)
    return {"kind": kind, "occ": occ, "attr": attr * occ[..., None]}

def render(sc, g, noise=0.05, notch=None):
    """把场景画成 (N, 3, 32, 32)。notch 是 (N, 4) 的整数，大于 0 时在目标中心挖一个暗坑（训练里从不出现）。"""
    n = len(sc["kind"])
    occ, attr = sc["occ"], sc["attr"]
    img = (0.15 + 0.15 * torch.rand(n, 1, 1, 1, generator=g)).expand(n, 3, SIZE, SIZE).clone()
    jx, jy = torch.randint(0, 3, (n, 4), generator=g), torch.randint(0, 3, (n, 4), generator=g)
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
    if rng.random() < 0.5: w.append(SIZES[z])
    if rng.random() < 0.5 and tex_words: w.append(TEXS[t])
    use_c, use_s = rng.random() < 0.85, rng.random() < 0.85
    if not use_c and not use_s: use_c = use_s = True
    if use_c: w.append(COLORS[c])
    w.append(SHAPES[s] if use_s else "图形")
    return w

def pair_words(sc, i):
    l, r = sorted(sc["occ"][i].nonzero().flatten().tolist())
    a, b = sc["attr"][i, l].tolist(), sc["attr"][i, r].tolist()
    return (COLORS[a[1]], SHAPES[a[0]]), (COLORS[b[1]], SHAPES[b[0]])

def caption(sc, i, rng, tex_words=True):
    """训练用的句子：由场景的属性拼出来，每次随机省略一些属性。tex_words=False 时永远不写“实心”“条纹”。"""
    kind, cells = int(sc["kind"][i]), sc["occ"][i].nonzero().flatten().tolist()
    pre = list(([], ["图", "里", "有"], ["这", "是"])[int(rng.integers(0, 3))])
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
    """“左右两个目标”的难负样本：把两个颜色词对调，或者把两个目标的先后对调。词一样，意思不一样。"""
    a, b = pair_words(sc, i)
    rel = "左边" if rng.random() < 0.5 else "右边"
    first, second = (a, b) if rel == "左边" else (b, a)
    return [second[0], first[1], "在", first[0], second[1], rel] if rng.random() < 0.5 else [second[0], second[1], "在", first[0], first[1], rel]

def encode(sents):
    out = torch.zeros(len(sents), MAXLEN, dtype=torch.long)
    for i, s in enumerate(sents):
        ids = [TOK[w] for w in (s.split() if isinstance(s, str) else s)]
        out[i, :len(ids)] = torch.tensor(ids)
    return out

def single_test(n, *seed, hold=None, only_hold=False, noise=0.05):
    """n 张只有一个目标的图和它们的标签。"""
    g = gen(*seed)
    sc = sample_scenes(n, g, kinds=(1.0, 0.0, 0.0, 0.0), hold=hold, only_hold=only_hold)
    img = render(sc, g, noise=noise)
    cell = sc["occ"].float().argmax(1)
    a = sc["attr"][torch.arange(n), cell]
    return img, {"形状": a[:, 0].numpy(), "颜色": a[:, 1].numpy(), "大小": a[:, 2].numpy(), "纹理": a[:, 3].numpy(), "位置": cell.numpy(), "颜色x形状": (a[:, 1] * 4 + a[:, 0]).numpy()}

g = gen(28)
sc = sample_scenes(200, g)
rng = np.random.default_rng(28)
pics = render(sc, g)
order = [int((sc["kind"] == k).nonzero().flatten()[j]) for k, j in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0))]
fig, ax = plt.subplots(2, 4, figsize=(9, 5.2))
for a, i in zip(ax.ravel(), order):
    a.imshow(pics[i].permute(1, 2, 0).clamp(0, 1).numpy(), interpolation="nearest"); a.set_xticks([]); a.set_yticks([])
    a.set_title(" ".join(caption(sc, i, rng)), fontsize=8)
plt.tight_layout(); plt.show()'''

MODEL = '''def sincos_2d(dim, grid):
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
    """很小的 ViT：16 个块，宽 64，2 层；16 个块的输出取平均，再线性投影到 32 维的共同空间。"""
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(3 * P * P, 64)
        self.register_buffer("pos", sincos_2d(64, G)[None])
        self.blocks = nn.ModuleList([Block(64) for _ in range(2)])
        self.norm, self.proj = nn.LayerNorm(64), nn.Linear(64, 32, bias=False)
    def tokens(self, img):                       # (B, 16, 64)：每个块一个标记
        x = self.embed(patchify((img - 0.4) / 0.3)) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.norm(x)
    def forward(self, img):                      # (B, 32)，没有归一化
        return self.proj(self.tokens(img).mean(1))

class TextTower(nn.Module):
    """词嵌入加可学习的位置编码，2 层同样的编码块，有效词的输出取平均，再线性投影到 32 维。"""
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(len(VOCAB), 64, padding_idx=0); nn.init.normal_(self.tok.weight, std=0.5)
        with torch.no_grad(): self.tok.weight[0].zero_()
        self.pos = nn.Parameter(torch.zeros(1, MAXLEN, 64)); nn.init.normal_(self.pos, std=0.5)
        self.blocks = nn.ModuleList([Block(64) for _ in range(2)])
        self.norm, self.proj = nn.LayerNorm(64), nn.Linear(64, 32, bias=False)
    def forward(self, ids):
        valid = ids != 0
        w = valid.float()[..., None]
        x = self.tok(ids) + self.pos
        for b in self.blocks:
            x = b(x, valid[:, None, None, :])
        return self.proj((self.norm(x) * w).sum(1) / w.sum(1))

class TwoTower(nn.Module):
    def __init__(self, temp=0.07, learn_temp=True):
        super().__init__()
        self.image, self.text = ImageTower(), TextTower()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / temp)), requires_grad=learn_temp)   # 温度的倒数取对数
    def scale(self):
        return self.logit_scale.exp().clamp(max=100.0)

def train_two_tower(seed=0, steps=1200, batch=64, lr=2e-3, hold=None, tex_words=True, hard=False, temp=0.07, learn_temp=True, quiet=False):
    """在随机生成的图文对上训练双塔。损失是第 13 章 5.3 节的对称交叉熵：一批 64 对，每张图在 64 句话里认出自己那一句，反过来也一样。"""
    torch.manual_seed(seed)
    g, rng = gen(seed, 7), np.random.default_rng(1000 + seed)
    net = TwoTower(temp, learn_temp).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    tgt, losses, t0 = torch.arange(batch, device=device), [], time.perf_counter()
    for step in range(steps):
        sc = sample_scenes(batch, g, hold=hold)
        img = render(sc, g).to(device)
        caps = [caption(sc, i, rng, tex_words) for i in range(batch)]
        u, v = F.normalize(net.image(img), dim=-1), F.normalize(net.text(encode(caps).to(device)), dim=-1)
        logits = net.scale() * u @ v.t()
        i2t = logits
        if hard:      # 每个“左右两个目标”的场景再加一句词相同、意思不同的句子当负样本
            idx = (sc["kind"] == 2).nonzero().flatten().tolist()
            extra = F.normalize(net.text(encode([swapped_caption(sc, i, rng) for i in idx]).to(device)), dim=-1)
            i2t = torch.cat([logits, net.scale() * u @ extra.t()], 1)
        loss = 0.5 * (F.cross_entropy(i2t, tgt) + F.cross_entropy(logits.t(), tgt))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        losses.append(float(loss.detach()))
        if not quiet and (step % 300 == 0 or step == steps - 1):
            print(f"  第 {step:4d} 步，损失 {losses[-1]:.3f}，温度 {1 / float(net.scale().detach()):.3f}")
    if not quiet:
        print(f"  训练用时 {time.perf_counter() - t0:.1f} 秒")
    net.losses = losses
    return net.eval()

@torch.no_grad()
def img_emb(net, img):
    """图像向量，(N, 32) 的 NumPy 数组，没有归一化。"""
    return net.image(img.to(device)).cpu().numpy().astype(np.float64)

@torch.no_grad()
def txt_emb(net, sents):
    return net.text(encode(sents).to(device)).cpu().numpy().astype(np.float64)

def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

def scale_of(net):
    return float(net.scale().detach())

def text_bank(net, phrases, templates=("一个 {}",)):
    """每个类别的短语套进每个模板，过文本塔。返回 (C, T, D)，正好是作业 build_zeroshot_classifier 的输入。"""
    return np.stack([txt_emb(net, [t.format(p) for p in phrases]) for t in templates], 1)

def fit_head(x, y, w_init, lam, scale, steps=100, lr=0.05):
    """在归一化的图像向量上训练线性分类头：交叉熵加 lam 乘“离初始权重的距离的平方”。w_init 全 0 就是线性探测。"""
    x, y = torch.from_numpy(unit(x)).float(), torch.from_numpy(np.asarray(y)).long()
    w0 = torch.from_numpy(np.asarray(w_init)).float()
    w = w0.clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(scale * x @ w.t(), y) + lam * ((w - w0) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return w.detach().numpy().astype(np.float64)

def k_shot(y_pool, classes, k, g):
    y_pool = torch.from_numpy(np.asarray(y_pool))
    return torch.cat([(y_pool == c).nonzero().flatten()[torch.randperm(int((y_pool == c).sum()), generator=g)[:k]] for c in range(classes)]).numpy()

print("图像塔参数：", sum(p.numel() for p in ImageTower().parameters()), "｜文本塔参数：", sum(p.numel() for p in TextTower().parameters()), "｜词表：", len(VOCAB), "个词")
print(f"训练双塔模型（1200 步，每步 64 对新生成的图文）。什么都没学到时损失是 ln 64 = {math.log(64):.3f}")
net = train_two_tower()
test_img, test_y = single_test(1000, 0, 99)        # 后面几个实验共用的 1000 张单目标测试图
test_f = img_emb(net, test_img)'''

ZEROSHOT = '''TASKS = {   # 任务：(类别短语, 模板)
    "形状": (SHAPES, "一个 {}"), "颜色": ([f"{c} 图形" for c in COLORS], "一个 {}"), "大小": ([f"{z} 图形" for z in SIZES], "一个 {}"),
    "纹理": ([f"{t} 图形" for t in TEXS], "一个 {}"), "位置": (CELLS, "一个 图形 在 {}"), "颜色x形状": (COMBOS, "一个 {}"),
}
if ready("build_zeroshot_classifier", "zeroshot_predict"):
    tr_img, tr_y = single_test(2000, 5, 99)
    tr_f = img_emb(net, tr_img)
    print(f"{'任务':8s} 类别数  随机猜   零样本（0 张标签）  线性头（2000 张标签）   提示词举例")
    for task, (phrases, tpl) in TASKS.items():
        clf = mod.build_zeroshot_classifier(text_bank(net, phrases, (tpl,)))
        probs, pred = mod.zeroshot_predict(test_f, clf, scale_of(net))
        head = fit_head(tr_f, tr_y[task], np.zeros((len(phrases), 32)), 0.01, scale_of(net))
        acc_head = (mod.zeroshot_predict(test_f, head, scale_of(net))[1] == test_y[task]).mean()
        print(f"{task:8s} {len(phrases):4d}   {1 / len(phrases):.3f}      {(pred == test_y[task]).mean():.3f}              {acc_head:.3f}           “{tpl.format(phrases[0])}”")
    clf = mod.build_zeroshot_classifier(text_bank(net, COMBOS))
    pool_img, pool_y = single_test(4000, 9, 99)
    pool_f = img_emb(net, pool_img)
    print("\\n“颜色 x 形状”16 类，线性头每类给 k 张标签：")
    for k in (1, 4, 16, 64):
        idx = k_shot(pool_y["颜色x形状"], 16, k, gen(0, k, 11))
        head = fit_head(pool_f[idx], pool_y["颜色x形状"][idx], np.zeros((16, 32)), 0.01, scale_of(net))
        print(f"  每类 {k:2d} 张：{(mod.zeroshot_predict(test_f, head, scale_of(net))[1] == test_y['颜色x形状']).mean():.3f}")
    print(f"  零样本：  {(mod.zeroshot_predict(test_f, clf, scale_of(net))[1] == test_y['颜色x形状']).mean():.3f}")'''

TEMPLATES = '''TEMPLATE_SET = {
    "只写类别名": "{}", "一个 X": "一个 {}", "图 里 有 一个 X": "图 里 有 一个 {}", "这 是 一个 X": "这 是 一个 {}",
    "一张 X 的 照片（三个词没训练过）": "一张 {} 的 照片", "三个 X（数量说错了）": "三个 {}",
}
if ready("build_zeroshot_classifier", "zeroshot_predict"):
    def acc_with(templates, phrases=SHAPES, task="形状"):
        clf = mod.build_zeroshot_classifier(text_bank(net, phrases, templates))
        probs, pred = mod.zeroshot_predict(test_f, clf, scale_of(net))
        return (pred == test_y[task]).mean(), probs.max(1).mean()
    names, accs = [], []
    for name, tpl in TEMPLATE_SET.items():
        a, conf = acc_with((tpl,))
        names.append(name); accs.append(a)
        print(f"{name:28s} 精度 {a:.3f}   最大概率的平均 {conf:.3f}")
    for name, tpls in (("三个训练时见过的模板取平均", ("一个 {}", "图 里 有 一个 {}", "这 是 一个 {}")), ("六个模板全部取平均", tuple(TEMPLATE_SET.values()))):
        a, conf = acc_with(tpls)
        names.append(name); accs.append(a)
        print(f"{name:28s} 精度 {a:.3f}   最大概率的平均 {conf:.3f}")
    plt.figure(figsize=(7.5, 3.4))
    plt.barh(range(len(names))[::-1], accs, color=["gray"] + ["C0"] * 3 + ["C3"] * 2 + ["C2"] * 2)
    plt.yticks(range(len(names))[::-1], names, fontsize=9); plt.xlim(0.5, 1.0); plt.xlabel("分形状的零样本精度"); plt.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.show()'''

COMPOSE = '''def pair_set(n, *seed):
    g = gen(*seed)
    sc = sample_scenes(n, g, kinds=(0.0, 0.0, 1.0, 0.0))
    keep = sc["kind"] == 2
    sc = {k: v[keep] for k, v in sc.items()}
    return sc, render(sc, g)

def binding_accuracy(model, sc, img):
    """每张“左右两个目标”的图，在正确的句子和改动过的句子之间二选一。两句话的余弦一样就记半对。"""
    u = unit(img_emb(model, img))
    caps = {"正确": [], "两个颜色词对调": [], "两个目标的先后对调": []}
    for i in range(len(u)):
        a, b = pair_words(sc, i)
        caps["正确"].append([a[0], a[1], "在", b[0], b[1], "左边"])
        caps["两个颜色词对调"].append([b[0], a[1], "在", a[0], b[1], "左边"])
        caps["两个目标的先后对调"].append([b[0], b[1], "在", a[0], a[1], "左边"])
    cos = {k: (u * unit(txt_emb(model, v))).sum(1) for k, v in caps.items()}
    return {k: float(((cos["正确"] - cos[k] > 1e-5) + 0.5 * (np.abs(cos["正确"] - cos[k]) <= 1e-5)).mean()) for k in list(caps)[1:]}

if ready("build_zeroshot_classifier", "zeroshot_predict"):
    print("训练第二个模型：所有场景里都不出现 红圆、绿三角、蓝十字、黄方块")
    net_hold = train_two_tower(hold=HOLD, quiet=True)
    img_seen, y_seen = single_test(1200, 2, 99, hold=HOLD)
    img_held, y_held = single_test(600, 1, 99, hold=HOLD, only_hold=True)
    print(f"{'':30s}见过的 12 种组合   没见过的 4 种组合")
    for name, model in (("训练时缺这 4 种组合", net_hold), ("对照：训练时 16 种都有", net)):
        clf = mod.build_zeroshot_classifier(text_bank(model, COMBOS))
        pred_s = mod.zeroshot_predict(img_emb(model, img_seen), clf, scale_of(model))[1]
        pred_h = mod.zeroshot_predict(img_emb(model, img_held), clf, scale_of(model))[1]
        print(f"{name:26s}    {(pred_s == y_seen['颜色x形状']).mean():.3f}             {(pred_h == y_held['颜色x形状']).mean():.3f}")
        if model is net_hold:
            print(f"    没见过的组合上：颜色判对 {(pred_h // 4 == y_held['颜色']).mean():.3f}，形状判对 {(pred_h % 4 == y_held['形状']).mean():.3f}")
            wrong = pred_h[pred_h != y_held["颜色x形状"]]
            top = np.bincount(wrong, minlength=16).argsort()[::-1][:3]
            print("    最常被错判成：", "、".join(COMBOS[k].replace(" ", "") for k in top))

pair_sc, pair_img = pair_set(600, 0, 55)
print("\\n属性绑定：每张“左右两个目标”的图，在正确的句子和改动过的句子之间二选一（随机猜 0.5）")
print("  1200 步的模型：", {k: round(v, 3) for k, v in binding_accuracy(net, pair_sc, pair_img).items()})
TRAIN_LONG = False        # 改成 True：再训练一个 2400 步、加难负样本的模型（多花一倍多的时间），看绑定能不能学会
if TRAIN_LONG:
    net_long = train_two_tower(steps=2400, hard=True, quiet=True)
    print("  2400 步，加难负样本：", {k: round(v, 3) for k, v in binding_accuracy(net_long, pair_sc, pair_img).items()})'''

RETRIEVAL = '''def full_caption(sc, i):
    """把全部属性都写出来的句子。"""
    kind, cells = int(sc["kind"][i]), sc["occ"][i].nonzero().flatten().tolist()
    if kind == 2:
        a, b = pair_words(sc, i)
        return [a[0], a[1], "在", b[0], b[1], "左边"]
    s, c, z, t = sc["attr"][i, cells[0]].tolist()
    return ["一个", SIZES[z], TEXS[t], COLORS[c], SHAPES[s], "在", CELLS[cells[0]]] if kind == 0 else [COUNTS[len(cells) - 1], SIZES[z], TEXS[t], COLORS[c], SHAPES[s]]

if ready("recall_at_k"):
    g = gen(0, 71)
    rsc = sample_scenes(500, g, kinds=(0.5, 0.2, 0.3, 0.0))
    rimg = render(rsc, g)
    caps = [full_caption(rsc, i) for i in range(500)]
    keys = np.array([" ".join(c) for c in caps])
    positives = keys[:, None] == keys[None, :]            # 句子完全相同的图互为正确答案
    print(f"500 个场景，其中 {(positives.sum(1) > 1).mean():.0%} 的句子不止对应一张图。例句：", " ｜ ".join(keys[:3]))
    u, v = unit(img_emb(net, rimg)), unit(txt_emb(net, caps))
    sim = u @ v.T
    for k in (1, 5, 10):
        print(f"  图找文 R@{k}: {mod.recall_at_k(sim, positives, k):.3f}   文找图 R@{k}: {mod.recall_at_k(sim.T, positives.T, k):.3f}")
    print(f"  如果只把对角线当正确答案，图找文 R@1 会被算成 {mod.recall_at_k(sim, np.eye(500, dtype=bool), 1):.3f}")
    rng = np.random.default_rng(5)
    shuffled = [list(rng.permutation(c)) for c in caps]
    sim_s = u @ unit(txt_emb(net, shuffled)).T
    print("\\n把每句话里的词随机打乱（例如：", " ".join(shuffled[0]), "）")
    kinds = rsc["kind"].numpy()
    for kd, name in ((0, "单个目标"), (1, "多个相同目标"), (2, "左右两个不同目标")):
        m = kinds == kd
        print(f"  {name:10s} 图找文 R@1：原句 {mod.recall_at_k(sim[m], positives[m], 1):.3f}   打乱 {mod.recall_at_k(sim_s[m], positives[m], 1):.3f}")'''

GAP = '''if ready("modality_gap"):
    g, rng = gen(4, 4), np.random.default_rng(44)
    psc = sample_scenes(1000, g)
    pimg, pcaps = render(psc, g), [caption(psc, i, rng) for i in range(1000)]
    torch.manual_seed(100)
    untrained = TwoTower().to(device).eval()
    rows = {}
    for name, model in (("没有训练", untrained), ("训练后", net)):
        u, v = img_emb(model, pimg), txt_emb(model, pcaps)
        gap, s_img, s_txt = mod.modality_gap(u, v)
        un, vn = unit(u), unit(v)
        rows[name] = (un, vn)
        print(f"{name}：间隙 {gap:.3f}，图像向量的散布 {s_img:.3f}，文本向量的散布 {s_txt:.3f}｜配对的图文余弦 {(un * vn).sum(1).mean():.3f}，不配对的 {(un @ vn.T).mean():.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.9))
    for a, (name, (un, vn)) in zip(ax, rows.items()):
        x = np.concatenate([un, vn]); mean = x.mean(0)
        _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
        pu, pv = (un - mean) @ vt[:2].T, (vn - mean) @ vt[:2].T
        a.scatter(pu[::4, 0], pu[::4, 1], s=6, label="图像向量"); a.scatter(pv[::4, 0], pv[::4, 1], s=6, label="文本向量")
        a.set_xlim(-1.15, 1.15); a.set_ylim(-1.15, 1.15); a.set_aspect("equal"); a.set_title(name + "（前两个主成分）"); a.legend(fontsize=8)
    plt.tight_layout(); plt.show()'''

DETECT = '''def det_scenes(n, g):
    """3 个目标 1 个空格：要找的目标、一个同色不同形状的干扰、一个同形状不同色的干扰。"""
    occ, attr, tgt = torch.zeros(n, 4, dtype=torch.bool), torch.zeros(n, 4, 4, dtype=torch.long), torch.zeros(n, dtype=torch.long)
    for i in range(n):
        cells = torch.randperm(4, generator=g)[:3].tolist()
        c, c2 = torch.randperm(4, generator=g)[:2].tolist()
        s, s2 = torch.randperm(4, generator=g)[:2].tolist()
        for cell, (ss, cc) in zip(cells, ((s, c), (s2, c), (s, c2))):
            occ[i, cell] = True
            attr[i, cell] = torch.tensor([ss, cc, int(torch.randint(0, 2, (1,), generator=g)), int(torch.randint(0, 2, (1,), generator=g))])
        tgt[i] = cells[0]
    return {"kind": torch.full((n,), 9), "occ": occ, "attr": attr}, tgt

@torch.no_grad()
def region_feats(model, img, how):
    """四个格子各自的区域特征，(N, 4, 32)。三种来源见上面的说明。"""
    n = len(img)
    if how == "块特征":
        e = model.image.proj(model.image.tokens(img.to(device))).reshape(n, G, G, -1)     # 每个块标记各自投影到共同空间
        return torch.stack([e[:, (c // 2) * 2:(c // 2) * 2 + 2, (c % 2) * 2:(c % 2) * 2 + 2].mean((1, 2)) for c in range(4)], 1).cpu().numpy().astype(np.float64)
    out = []
    for c in range(4):
        y0, x0 = (c // 2) * CELL, (c % 2) * CELL
        crop = img[:, :, y0:y0 + CELL, x0:x0 + CELL]
        if how == "裁剪后放大到整图":
            x = F.interpolate(crop, scale_factor=2, mode="bilinear", align_corners=False)
        else:                                    # 裁剪后贴回原位：其余位置填成这一格四个角的平均灰度
            fill = torch.stack([crop[:, :, 0, 0], crop[:, :, 0, -1], crop[:, :, -1, 0], crop[:, :, -1, -1]]).mean((0, 2))
            x = fill[:, None, None, None].expand(n, 3, SIZE, SIZE).clone()
            x[:, :, y0:y0 + CELL, x0:x0 + CELL] = crop
        out.append(img_emb(model, x))
    return np.stack(out, 1)

if ready("score_regions"):
    n = 400
    dsc, tgt = det_scenes(n, gen(0, 41))
    dimg = render(dsc, gen(0, 42))
    ta = dsc["attr"][torch.arange(n), tgt]
    queries = [f"一个 {COLORS[c]} {SHAPES[s]}" for s, c in zip(ta[:, 0].tolist(), ta[:, 1].tolist())]
    q = txt_emb(net, queries)
    big = (ta[:, 2] == 0).numpy()
    maps = {}
    print("分数最高的格子就是目标的比例（4 个格子，随机猜 0.25）：")
    for how in ("裁剪后贴回原位", "裁剪后放大到整图", "块特征"):
        r = region_feats(net, dimg, how)
        best = np.array([mod.score_regions(r[i], q[i:i + 1])[1][0] for i in range(n)])
        hit = best == tgt.numpy()
        maps[how] = mod.score_regions(r[0], q[:1])[0][:, 0].reshape(2, 2)
        print(f"  {how:10s} 全部 {hit.mean():.3f}   大目标 {hit[big].mean():.3f}   小目标 {hit[~big].mean():.3f}   最高分落在空格子上的比例 {(~dsc['occ'].numpy()[np.arange(n), best]).mean():.3f}")
    fig, ax = plt.subplots(1, 4, figsize=(11, 3))
    ax[0].imshow(dimg[0].permute(1, 2, 0).clamp(0, 1).numpy(), interpolation="nearest"); ax[0].set_title("查询：" + queries[0], fontsize=9)
    for a, (how, m) in zip(ax[1:], maps.items()):
        a.imshow(m, cmap="magma", vmin=-0.6, vmax=1); a.set_title(how, fontsize=9)
        for cy in range(2):
            for cx in range(2):
                a.text(cx, cy, f"{m[cy, cx]:.2f}", ha="center", va="center", color="white" if m[cy, cx] < 0.45 else "black")
    for a in ax: a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.show()

    if ready("build_zeroshot_classifier", "zeroshot_predict"):
        r = region_feats(net, dimg, "裁剪后贴回原位").reshape(-1, 32)
        occ = dsc["occ"].numpy().reshape(-1)
        truth = (dsc["attr"][..., 1] * 4 + dsc["attr"][..., 0]).numpy().reshape(-1)
        clf16 = mod.build_zeroshot_classifier(text_bank(net, COMBOS))
        clf17 = np.concatenate([clf16, mod.build_zeroshot_classifier(text_bank(net, ["空 背景"], ("{}",)))])
        print("\\n把每个格子当成一张小图做 16 类零样本分类；空格子里什么都没有：")
        for name, clf in (("16 个类别", clf16), ("再加一行“空 背景”", clf17)):
            probs, pred = mod.zeroshot_predict(r, clf, scale_of(net))
            top = probs[:, :16].max(1)
            print(f"  {name:14s} 目标格子的精度 {(pred[occ] == truth[occ]).mean():.3f}   空格子上最大类别概率的平均 {top[~occ].mean():.3f}，超过 0.3 的比例 {(top[~occ] > 0.3).mean():.3f}")'''

FEWSHOT = '''def low_contrast(img):
    """一种训练时没见过的成像条件：对比度减半。"""
    return 0.4 + 0.5 * (img - 0.4)

if ready("build_zeroshot_classifier", "zeroshot_predict"):
    y_te = test_y["颜色x形状"]
    f_shift = img_emb(net, low_contrast(test_img))
    pool_img, pool_y = single_test(4000, 9, 99)
    pool_f, y_pool = img_emb(net, low_contrast(pool_img)), pool_y["颜色x形状"]
    clf = mod.build_zeroshot_classifier(text_bank(net, COMBOS))
    acc = lambda w, f=f_shift: (mod.zeroshot_predict(f, w, scale_of(net))[1] == y_te).mean()
    print(f"零样本：原来的图 {acc(clf, test_f):.3f}，对比度减半的图 {acc(clf):.3f}")
    ks, probe, tuned = [1, 2, 4, 8, 16], [], []
    for k in ks:
        a, b = [], []
        for rep in range(3):
            idx = k_shot(y_pool, 16, k, gen(0, k, rep, 5))
            a.append(acc(fit_head(pool_f[idx], y_pool[idx], np.zeros((16, 32)), 0.01, scale_of(net))))     # 线性探测：权重从 0 开始
            b.append(acc(fit_head(pool_f[idx], y_pool[idx], clf, 0.3, scale_of(net))))                     # 从零样本分类器出发，不许走远
        probe.append(np.mean(a)); tuned.append(np.mean(b))
        print(f"  每类 {k:2d} 张：线性探测 {probe[-1]:.3f}   从零样本分类器出发微调 {tuned[-1]:.3f}")
    plt.figure(figsize=(5.6, 3.6))
    plt.plot(ks, probe, "o-", label="线性探测（从 0 开始）"); plt.plot(ks, tuned, "D-", label="从零样本分类器出发微调")
    plt.axhline(acc(clf), color="gray", ls="--", label="零样本")
    plt.xscale("log"); plt.xticks(ks, [str(k) for k in ks]); plt.xlabel("目标域里每类带标签的图数"); plt.ylabel("对比度减半的图上的精度"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()'''

UNSEEN = '''def product_set(n, a, g, tex=0, notch=0):
    """同一种“产品”的 n 张图：形状、颜色、位置固定的一个大目标。tex=1 是条纹，notch>0 是中心有一个暗坑。"""
    shape, color, cell = a
    occ, attr, nt = torch.zeros(n, 4, dtype=torch.bool), torch.zeros(n, 4, 4, dtype=torch.long), torch.zeros(n, 4, dtype=torch.long)
    occ[:, cell] = True
    attr[:, cell] = torch.tensor([shape, color, 0, tex])
    nt[:, cell] = notch
    return render({"kind": torch.zeros(n, dtype=torch.long), "occ": occ, "attr": attr}, g, notch=nt if notch else None)

def auroc_simple(score, label):
    """异常分数的 AUROC：随机取一张缺陷图和一张良品图，缺陷图分数更高的概率（相同算一半）。"""
    pos, neg = score[label == 1], score[label == 0]
    return float(((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean())

if ready("build_zeroshot_classifier", "zeroshot_predict"):
    print("训练第三个模型：图像和第一个模型完全一样，但句子里永远不出现“实心”“条纹”")
    net_notex = train_two_tower(tex_words=False, quiet=True)
    pool_img, pool_y = single_test(4000, 9, 99)
    print(f"{'':24s}零样本分纹理   零样本分形状   线性头分纹理（16 张标签 / 256 张标签）")
    for name, model in (("句子里写纹理", net), ("句子里从不写纹理", net_notex)):
        f, sc_ = img_emb(model, test_img), scale_of(model)
        zs = lambda phrases, task: (mod.zeroshot_predict(f, mod.build_zeroshot_classifier(text_bank(model, phrases)), sc_)[1] == test_y[task]).mean()
        heads = []
        for n_lab in (16, 256):
            idx = k_shot(pool_y["纹理"], 2, n_lab // 2, gen(0, n_lab, 6))
            head = fit_head(img_emb(model, pool_img[idx]), pool_y["纹理"][idx], np.zeros((2, 32)), 0.01, sc_)
            heads.append((mod.zeroshot_predict(f, head, sc_)[1] == test_y["纹理"]).mean())
        print(f"{name:20s}   {zs([t + ' 图形' for t in TEXS], '纹理'):.3f}          {zs(SHAPES, '形状'):.3f}          {heads[0]:.3f} / {heads[1]:.3f}")

    g = gen(0, 61)
    a = (0, 2, 1)                                   # 产品：右上角的一个大的蓝色方块
    ref, good = product_set(8, a, g), product_set(50, a, g)
    label = np.r_[np.zeros(50), np.ones(50)]
    print("\\n同一种产品（大的蓝色方块，实心），50 张良品对 50 张缺陷图的 AUROC：")
    for cname, kw in (("整个目标变成条纹", {"tex": 1}), ("中心有 4 px 的暗坑", {"notch": 4}), ("中心有 2 px 的暗坑", {"notch": 2})):
        query = torch.cat([good, product_set(50, a, g, **kw)])
        line = f"  {cname:12s}"
        for name, model in (("写纹理的模型", net), ("不写纹理的模型", net_notex)):
            fq, fr = unit(img_emb(model, query)), unit(img_emb(model, ref))
            line += f"   {name}：8 张良品的近邻 {auroc_simple(1 - (fq @ fr.T).max(1), label):.3f}"
            if "tex" in kw:
                t = unit(txt_emb(model, ["一个 条纹 蓝色 方块", "一个 实心 蓝色 方块"]))
                line += f"，文字 {auroc_simple(fq @ t[0] - fq @ t[1], label):.3f}"
        print(line)
    fig, ax = plt.subplots(1, 4, figsize=(8, 2.3))
    for a_, (title, im) in zip(ax, (("良品", good[0]), ("条纹", product_set(1, a, g, tex=1)[0]), ("4 px 暗坑", product_set(1, a, g, notch=4)[0]), ("2 px 暗坑", product_set(1, a, g, notch=2)[0]))):
        a_.imshow(im.permute(1, 2, 0).clamp(0, 1).numpy(), interpolation="nearest"); a_.set_title(title, fontsize=9); a_.set_xticks([]); a_.set_yticks([])
    plt.tight_layout(); plt.show()'''

REAL = '''RUN_REAL = False          # 改成 True 再运行这个单元；要先 pip install open_clip_torch，第一次运行会联网下载权重
if not RUN_REAL:
    print("跳过。把 RUN_REAL 改成 True 再运行；做法、显存不够时怎么办、应该看到什么，见教程第 9 节。")
else:
    try:
        import open_clip
        from PIL import Image
        clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        clip_model = clip_model.to(device).eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        names = ["cat", "dog", "car"]                                   # 换成你自己的类别
        templates = ["a photo of a {}.", "a close-up photo of a {}."]   # 换成适合你的图像的句式
        paths = []                                                      # 填上几张图片的路径
        with torch.inference_mode():
            bank = torch.stack([clip_model.encode_text(tokenizer([t.format(n) for n in names]).to(device)) for t in templates], 1)     # (C, T, D)
            feats = clip_model.encode_image(torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device)) if paths else None
        print("文本向量：", tuple(bank.shape))
        if feats is not None and ready("build_zeroshot_classifier", "zeroshot_predict"):
            clf = mod.build_zeroshot_classifier(bank.float().cpu().numpy())     # 真模型的向量直接交给你写的函数
            probs, pred = mod.zeroshot_predict(feats.float().cpu().numpy(), clf, 100.0)
            for p, row, k in zip(paths, probs, pred):
                print(p, "->", names[k], np.round(row, 3))
        if device.type == "cuda":
            print("显存峰值（MiB）：", round(torch.cuda.max_memory_allocated() / 2 ** 20))
    except Exception as exc:
        print("没有跑通：", type(exc).__name__, exc)'''

CELLS = [
    ("md", "## 准备：图文对和双塔小模型\n\n数据全部由代码生成：32×32 的彩色小图分成 2×2 个格子，每格至多一个目标，目标有形状、颜色、大小、纹理四种属性；配套的句子由这些属性拼出来，每次随机省略一些。下面先画 8 个例子。\n\n第二个单元定义双塔模型并训练 1200 步。这两个单元不涉及作业，直接运行。训练损失用的是第 13 章 5.3 节的对称交叉熵，这里直接写在训练函数里。"),
    ("code", SETUP),
    ("code", MODEL),
    ("md", "## 实验 1：写一句话就是一个分类器\n\n把每个类别写成一句话，过文本塔，用你写的 `build_zeroshot_classifier` 得到分类器，再用 `zeroshot_predict` 给 1000 张测试图分类，一张带标签的图都不用。旁边放一个对照：图像塔冻结，用 2000 张带标签的图训练一个线性头。\n\n先预测：线性头每类要多少张标签，才追得上零样本？对照教程图 1。"),
    ("code", ZEROSHOT),
    ("md", "## 实验 2：提示词模板和多模板平均\n\n同一批图，只改句子的写法。训练时见过的写法有三种；“只写类别名”和“一张 X 的 照片”是没见过的；“三个 X”见过，但图里只有一个目标。\n\n先预测哪一种掉得最多，再运行。对照教程图 2。"),
    ("code", TEMPLATES),
    ("md", "## 实验 3：没见过的属性组合，和属性绑定\n\n再训练一个模型，训练时所有场景里都不出现红圆、绿三角、蓝十字、黄方块；每种颜色、每种形状仍然各在另外 3 种组合里出现。然后拿这 4 种组合的图去测。\n\n先预测：模型见过“红色”也见过“圆”，它认得出“红色 圆”吗？认不出的话，是颜色错还是形状错？对照教程图 3。\n\n单元的后半部分量属性绑定：一张“红方块在蓝圆左边”的图，和“蓝方块在红圆左边”这句话的相似度，比正确的那句低吗？"),
    ("code", COMPOSE),
    ("md", "## 实验 4：图文检索，把句子里的词打乱\n\n500 个场景，每个配一句把全部属性都写出来的句子。用你写的 `recall_at_k` 算图找文和文找图的 Recall@K。注意很多图的句子是一样的，正确答案不止一个，也不在对角线上。\n\n然后把每句话里的词随机打乱再检索一次。先预测 R@1 会掉多少。"),
    ("code", RETRIEVAL),
    ("md", "## 实验 5：模态间隙\n\n1000 个图文对，图像向量和文本向量都归一化到单位球面上。用你写的 `modality_gap` 量两团向量的中心相距多远、各自散得多开，比较没有训练的模型和训练后的模型。对照教程图 4。"),
    ("code", GAP),
    ("md", "## 实验 6：用文字查询找目标\n\n每张图里有 3 个目标：要找的那个、一个同色不同形状的、一个同形状不同色的。候选区域就是 4 个格子，每个格子一个特征，用你写的 `score_regions` 和查询的文本向量打分，取分数最高的格子。\n\n区域特征有三种来源：\n\n- **裁剪后贴回原位**：把这一格的像素贴到一张空图的同一位置，再过图像塔。\n- **裁剪后放大到整图**：把 16×16 的格子放大到 32×32，再过图像塔。\n- **块特征**：整图只过一次图像塔，把这一格的 2×2 个块标记各自投影到共同空间后取平均。\n\n先预测三种来源的排序，再运行。对照教程图 5。最后一段把每个格子当成小图做 16 类分类，看空格子会被判成什么，加一句“空 背景”之后又怎样。"),
    ("code", DETECT),
    ("md", "## 实验 7：少样本适配\n\n把测试图的对比度减半，模型训练时没见过这种图，零样本精度会掉。现在每类给 1 到 16 张目标域的带标签图，双塔都冻结，比较两种做法：线性探测（分类头的权重从 0 开始），和从零样本分类器出发微调（权重从文本向量开始，并且惩罚离开文本向量的距离）。\n\n先预测两条曲线在哪里交叉。对照教程图 6 左栏。"),
    ("code", FEWSHOT),
    ("md", "## 实验 8：训练句子里没有的属性\n\n训练第三个模型：图像和第一个模型完全一样，一半目标是条纹的，但句子里永远不出现“实心”“条纹”。然后问它纹理。\n\n后半部分模拟缺陷检测：同一种产品，缺陷是“整个目标变成条纹”或者“中心有一个暗坑”（暗坑在任何训练图像和句子里都没有）。比较用文字描述异常，和用 8 张良品做近邻。对照教程图 6 的中栏、右栏和第 8 节。"),
    ("code", UNSEEN),
    ("md", "## 选做：在自己的机器上加载 CLIP\n\n需要联网，并且要先安装 `open_clip_torch`。这个单元默认跳过，写课程的机器上没有运行过，接口以官方仓库 `mlfoundations/open_clip` 的说明为准。真模型给出的文本向量和图像向量，可以原样交给你写的 `build_zeroshot_classifier` 和 `zeroshot_predict`。"),
    ("code", REAL),
]

OUTRO = ("## 破坏实验与记录\n\n"
         "- 实验 2 里自己再写两个模板加进 `TEMPLATE_SET`（只能用词表里的词），先猜它们的精度。\n"
         "- 实验 3 里把 `TRAIN_LONG` 改成 `True`，看训练加长、加上难负样本之后二选一的精度；再用这个模型重跑实验 4 的打乱词序。\n"
         "- 实验 5 里用 `train_two_tower(temp=0.01, learn_temp=False, quiet=True)` 和 `temp=1.0` 各训练一个模型，量间隙和零样本精度。\n"
         "- 实验 6 里把“空 背景”换成别的句子（例如“一个 图形”），看空格子的误报怎么变。\n"
         "- 把每个实验的数字填进 `实验记录.md`。")


def main():
    nb = notebook(
        "第 28 章互动实验：从零训练的双塔小模型",
        "用你在作业里写的五个零件，在一个从零训练的很小的图文双塔模型上做零样本分类、提示词、组合泛化、检索、模态间隙、文字查询找目标和少样本适配。全部数据由代码生成，不需要下载任何权重；作业还没写完时，对应的单元只打印提示。",
        CHAPTER, ASSIGNMENT, CELLS, OUTRO,
    )
    path = ROOT / CHAPTER / OUTPUT
    path.parent.mkdir(parents=True, exist_ok=True)
    for i, cell in enumerate(nb.cells):
        cell["id"] = f"c{i:03d}"
    nbf.write(nb, path)
    print("写入", f"{CHAPTER}/{OUTPUT}")


if __name__ == "__main__":
    main()
