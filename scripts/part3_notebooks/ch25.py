"""生成第 25 章的互动实验 Notebook。

用法（在仓库根目录运行）：
    python scripts/part3_notebooks/ch25.py
修改实验内容时改本文件再重新生成，不要直接手改 .ipynb，避免两边不一致。
Notebook 只调用学员自己在作业文件里的实现，不包含参考答案；
设置环境变量 COURSE_IMPL_DIR 可以让 Notebook 改用另一份实现（维护者验证用）。
HELPER 和 notebook() 抄自 scripts/build_track2_notebooks.py，本文件不依赖它。
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]

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

import math, time
import torch
import torch.nn.functional as F
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
print("作业文件：", CHAPTER / ASSIGNMENT, "｜PyTorch：", torch.__version__, "｜设备：", device)
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


CELLS = [
    ("md", "## 0. 数据：四种小图案\n\n全部数据由代码生成：16×16 的噪声图上放一个 3×3 的图案（横线、竖线、斜线、反斜线）。"
           "`make_type` 的类别是图案的种类，`make_quadrant` 的类别是图案在哪个象限。不需要下载任何数据。"),
    ("code", '''NOISE = 0.15
GLYPHS = np.zeros((4, 3, 3), np.float32)
GLYPHS[0, 1, :] = 1; GLYPHS[1, :, 1] = 1
GLYPHS[2, [0, 1, 2], [0, 1, 2]] = 1; GLYPHS[3, [0, 1, 2], [2, 1, 0]] = 1
GLYPH_NAMES = ["横线", "竖线", "斜线", "反斜线"]

def make_type(n, rng, size=16, align=False):
    """类别是图案的种类。align=True 时图案的左上角对齐到 4 px 的网格（落在一个块里）。"""
    xs = NOISE * rng.standard_normal((n, 1, size, size)).astype(np.float32)
    y = rng.integers(0, 4, n); pos = np.zeros((n, 2), np.int64)
    for i in range(n):
        py, px = rng.integers(0, size - 2, 2)
        if align: py, px = py // 4 * 4, px // 4 * 4
        xs[i, 0, py:py + 3, px:px + 3] += GLYPHS[y[i]]; pos[i] = py, px
    return torch.from_numpy(xs), torch.from_numpy(y), pos

def make_quadrant(n, rng, size=16):
    """类别是图案在哪个象限：0 左上，1 右上，2 左下，3 右下。"""
    xs = NOISE * rng.standard_normal((n, 1, size, size)).astype(np.float32)
    y = np.zeros(n, np.int64); h = size // 2
    for i in range(n):
        qy, qx = rng.integers(0, 2, 2)
        py, px = qy * h + rng.integers(0, h - 2), qx * h + rng.integers(0, h - 2)
        xs[i, 0, py:py + 3, px:px + 3] += GLYPHS[rng.integers(0, 4)]; y[i] = qy * 2 + qx
    return torch.from_numpy(xs), torch.from_numpy(y), None

x_demo, y_demo, _ = make_type(4, np.random.default_rng(0))
fig, ax = plt.subplots(1, 4, figsize=(10, 2.8))
for a, im, lab in zip(ax, x_demo, y_demo):
    a.imshow(im[0], cmap="gray"); a.set_title(GLYPH_NAMES[int(lab)]); a.set_xticks(range(0, 17, 4)); a.set_yticks(range(0, 17, 4)); a.grid(color="orange", lw=0.5)
plt.show()'''),
    ("md", "## 1. 切块：一张图变成一串标记\n\n先看形状，再做两项核对：切块再还原应该逐元素相等；“切块后接线性层”应该和“步长等于核大小的卷积”算出同样的数。"
           "第二项对不上，说明块内的展平顺序不是 (C, P, P)。"),
    ("code", '''tokens = attempt(mod.patchify, x_demo, 4)
if tokens is not None:
    print("图像", tuple(x_demo.shape), "切成标记", tuple(tokens.shape))
    back = attempt(mod.unpatchify, tokens, 4, 1, 16, 16)
    if back is not None:
        print("切块再还原，最大差：", float((back - x_demo).abs().max()))
    conv = nn.Conv2d(1, 8, 4, stride=4)
    with torch.no_grad():
        a = tokens @ conv.weight.reshape(8, -1).T + conv.bias
        b = conv(x_demo).flatten(2).transpose(1, 2)
    print("切块加线性层 对 步长 4 的卷积，最大差：", float((a - b).abs().max()))
    fig, ax = plt.subplots(4, 4, figsize=(3.6, 3.6))
    for n, a in enumerate(ax.ravel()):
        a.imshow(tokens[0, n].reshape(4, 4), cmap="gray", vmin=-0.3, vmax=1.1); a.axis("off")
    fig.suptitle("第 0 张图的 16 个标记，各自还原成 4x4"); plt.show()'''),
    ("md", "## 2. 缩放点积注意力：为什么除以根号 d\n\n查询和 16 个键都是随机向量。维数越高，点积的波动越大，softmax 越接近只选一个。"
           "下面用你的函数算三种情况：正常缩放；把缩放抵消掉（给查询乘上根号 d）；再多除一个根号 d。先预测维数从 4 到 1024 时“最大权重”各怎么变。"),
    ("code", '''g = torch.Generator().manual_seed(0)
print(f"{'维数 d':>8} {'不缩放':>10} {'除以根号 d':>10} {'除以 d':>10}    （16 个键里最大的权重，2000 次平均；均匀是 {1 / 16:.4f}）")
for d in (4, 16, 64, 256, 1024):
    q = torch.randn(2000, 1, d, generator=g, dtype=torch.float64); k = torch.randn(2000, 16, d, generator=g, dtype=torch.float64)
    res = attempt(mod.scaled_dot_product_attention, q, k, k)
    if res is None:
        break
    w_std = res[1]
    w_raw = mod.scaled_dot_product_attention(q * math.sqrt(d), k, k)[1]
    w_over = mod.scaled_dot_product_attention(q / math.sqrt(d), k, k)[1]
    print(f"{d:>8} {float(w_raw.max(-1).values.mean()):>10.3f} {float(w_std.max(-1).values.mean()):>10.3f} {float(w_over.max(-1).values.mean()):>10.3f}")'''),
    ("md", "掩码：`True` 表示允许关注。下面让第 i 个查询只能看第 0 到 i 个键（下三角），再看一行全部被屏蔽时你的函数返回什么。"),
    ("code", '''q = torch.randn(1, 5, 8, generator=g); mask = torch.ones(5, 5).tril().bool()
res = attempt(mod.scaled_dot_product_attention, q, q, q, mask)
if res is not None:
    np.set_printoptions(precision=3, suppress=True)
    print("下三角掩码下的权重：\\n", res[1][0].numpy())
    mask[2] = False
    w = mod.scaled_dot_product_attention(q, q, q, mask)[1][0]
    print("第 2 行全部屏蔽后，这一行的权重：", w[2].numpy(), "（应全为 0，不能是 nan）")'''),
    ("md", "## 3. 二维正余弦位置编码\n\n画出 8×8 网格里每个位置和第 3 行第 3 列那个位置的余弦相似度。离得近的位置编码应该更像，而且行和列的方向对称。"),
    ("code", '''pe = attempt(mod.sincos_2d_position_embedding, 32, 8, 8)
if pe is not None:
    pn = F.normalize(pe, dim=1); sim = (pn @ pn[3 * 8 + 3]).reshape(8, 8)
    print("形状", tuple(pe.shape), "｜每一行的平方和（应为 dim/2 = 16）：", float((pe ** 2).sum(1).mean()))
    print("和 (3, 3) 的相似度：左右相邻 %.3f，上下相邻 %.3f，隔 4 格 %.3f" % (sim[3, 4], sim[4, 3], sim[3, 7]))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    im = ax[0].imshow(sim, cmap="viridis"); ax[0].plot(3, 3, "wx"); ax[0].set_title("和 x 处的余弦相似度"); fig.colorbar(im, ax=ax[0])
    ax[1].imshow(pe.T, cmap="RdBu_r", aspect="auto"); ax[1].set_xlabel("位置（行优先）"); ax[1].set_ylabel("维度"); ax[1].set_title("编码本身：前一半管行，后一半管列")
    plt.show()'''),
    ("md", "## 4. 用你的零件拼一个小 ViT\n\n下面的模型只做三件事：`patchify` 后接线性层；加位置编码；过几层“多头注意力加 MLP”的编码块。"
           "多头注意力调用的是你的 `multi_head_attention`，权重由模型自己保存。`pre=True` 是 LayerNorm 放在前面，`pre=False` 是放在残差相加之后。\n\n"
           "先通过验收再跑后面的单元：作业只完成了一部分时，后面的单元会提示还缺哪个函数。"),
    ("code", '''def missing_functions():
    """返回还没实现的作业函数名。只看是不是 TODO，不检查对错；对错由 tests/check_progress.py 负责。"""
    z = torch.zeros
    calls = [(mod.patchify, (z(1, 1, 4, 4), 2)), (mod.unpatchify, (z(1, 4, 4), 2, 1, 4, 4)), (mod.scaled_dot_product_attention, (z(1, 2, 4), z(1, 2, 4), z(1, 2, 4))),
             (mod.multi_head_attention, (z(1, 2, 4), z(12, 4), z(12), z(4, 4), z(4), 2)), (mod.sincos_2d_position_embedding, (8, 2, 2))]
    out = []
    for fn, args in calls:
        try:
            fn(*args)
        except NotImplementedError:
            out.append(fn.__name__)
    return out

MISSING = missing_functions()
READY = not MISSING
print("作业五个函数都已实现，可以往下跑。" if READY else "先完成作业里的 " + "、".join(MISSING) + "，再回来运行后面的单元。")

class Block(nn.Module):
    def __init__(self, dim, heads, pre=True):
        super().__init__()
        self.heads, self.pre = heads, pre
        self.w_qkv = nn.Parameter(torch.empty(3 * dim, dim)); nn.init.xavier_uniform_(self.w_qkv)
        self.b_qkv = nn.Parameter(torch.zeros(3 * dim))
        self.out = nn.Linear(dim, dim)
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.attn = None                      # 最近一次前向的注意力权重 (B, H, L, L)，画注意力图时用

    def attend(self, x):
        out, self.attn = mod.multi_head_attention(x, self.w_qkv, self.b_qkv, self.out.weight, self.out.bias, self.heads)
        return out

    def forward(self, x):
        if self.pre:
            x = x + self.attend(self.ln1(x))
            return x + self.mlp(self.ln2(x))
        x = self.ln1(x + self.attend(x))
        return self.ln2(x + self.mlp(x))

class TinyViT(nn.Module):
    """pos: "sincos"、"learned" 或 "none"；pool: "mean"（全局平均）或 "cls"（类别标记）。"""
    def __init__(self, size=16, patch=4, dim=32, depth=2, heads=4, classes=4, pos="sincos", pool="mean", pre=True):
        super().__init__()
        self.patch, self.pool = patch, pool
        g = size // patch
        self.embed = nn.Linear(patch * patch, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim)) if pool == "cls" else None
        if pos == "learned":
            self.pos = nn.Parameter(0.02 * torch.randn(1, g * g, dim))
        elif pos == "sincos":
            self.register_buffer("pos", mod.sincos_2d_position_embedding(dim, g, g)[None])
        else:
            self.pos = None
        self.blocks = nn.ModuleList([Block(dim, heads, pre) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim) if pre else nn.Identity()
        self.head = nn.Linear(dim, classes)

    def forward(self, x, perm=None):
        t = mod.patchify(x, self.patch)
        if perm is not None:
            t = t[:, perm]                    # 把块打乱：位置编码不跟着动
        t = self.embed(t)
        if self.pos is not None:
            t = t + self.pos
        if self.cls is not None:
            t = torch.cat([self.cls.expand(len(t), -1, -1), t], dim=1)
        for blk in self.blocks:
            t = blk(t)
        t = self.ln(t)
        return self.head(t[:, 0] if self.pool == "cls" else t.mean(1))

class SmallCNN(nn.Module):
    """参数量和小 ViT 相近的 CNN：3x3 卷积、2x2 平均池化、两层 3x3 卷积、全局平均、线性层。"""
    def __init__(self, classes=4, c1=16, c2=36):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(1, c1, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
                                  nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.Conv2d(c2, c2, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(c2, classes)
    def forward(self, x):
        return self.head(self.body(x).mean((2, 3)))

def count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def train(model, make, steps=500, lr=3e-3, batch=32, seed=0, data=None, warmup=0):
    """make(n, rng) 每步新生成一批；data=(x, y) 时改用固定的训练集。返回每步的损失。"""
    model.to(device).train()
    rng = np.random.default_rng(seed); gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr); losses = []
    for s in range(steps):
        if data is None:
            x, y = make(batch, rng)[:2]
        else:
            idx = torch.randint(0, len(data[0]), (batch,), generator=gen); x, y = data[0][idx], data[1][idx]
        if warmup:
            for group in opt.param_groups:
                group["lr"] = lr * min(1.0, (s + 1) / warmup)
        loss = F.cross_entropy(model(x.to(device)), y.to(device))
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
    model.eval()
    return losses

@torch.no_grad()
def accuracy(model, x, y, **kw):
    model.eval()
    return float((model(x.to(device), **kw).argmax(1).cpu() == y).float().mean())

if READY:
    print("小 ViT 参数量：", count(TinyViT()), "｜小 CNN 参数量：", count(SmallCNN()))'''),
    ("md", "## 5. 小 ViT 对小 CNN：训练样本从少到多\n\n同一个任务（分图案的种类），固定训练集的张数从 50 到 800，再加一组每步都生成新数据。"
           "先预测：样本少的时候谁好？差距随样本数怎么变？这一格在 CPU 上大约半分钟。"),
    ("code", '''if not READY:
    print("先完成作业里的 " + "、".join(MISSING) + "，再回来运行这一格。")
else:
    xt, yt, _ = make_type(1000, np.random.default_rng(100))
    sizes = [50, 200, 800, None]
    curve = {"ViT": [], "CNN": []}
    for n in sizes:
        fixed = None if n is None else make_type(n, np.random.default_rng(1))[:2]
        for name, make_model in (("ViT", TinyViT), ("CNN", SmallCNN)):
            torch.manual_seed(0); m = make_model()
            train(m, make_type, data=fixed)
            curve[name].append(accuracy(m, xt, yt))
        print(f"训练集 {'每步新数据' if n is None else n:>6}：ViT {curve['ViT'][-1]:.3f}   CNN {curve['CNN'][-1]:.3f}")
    plt.figure(figsize=(5.2, 3.4))
    for name, c in curve.items():
        plt.plot(range(4), c, "o-", label=name)
    plt.xticks(range(4), ["50", "200", "800", "每步新数据"]); plt.xlabel("训练集张数"); plt.ylabel("测试准确率"); plt.legend(); plt.grid(alpha=0.3); plt.show()'''),
    ("md", "## 6. 位置编码管什么：两个任务，再把块打乱\n\n“分种类”和位置无关，“分象限”只和位置有关。分别用不加位置编码和正余弦位置编码训练，"
           "再在测试时把 16 个块按一个固定的随机顺序重排。先预测 2×2 的表里哪一格会是随机猜（0.25），打乱之后哪一格会掉。"),
    ("code", '''if not READY:
    print("先完成作业里的 " + "、".join(MISSING) + "，再回来运行这一格。")
else:
    perm = torch.from_numpy(np.random.default_rng(7).permutation(16))
    for task, make in (("分种类", make_type), ("分象限", make_quadrant)):
        xt, yt, _ = make(1000, np.random.default_rng(100))
        for pos in ("none", "sincos"):
            torch.manual_seed(0); m = TinyViT(pos=pos)
            train(m, make)
            with torch.no_grad():
                diff = float((m(xt.to(device)) - m(xt.to(device), perm=perm.to(device))).abs().max())
            print(f"{task}，位置编码 {pos:>6}：准确率 {accuracy(m, xt, yt):.3f}｜打乱块之后 {accuracy(m, xt, yt, perm=perm.to(device)):.3f}｜打乱前后输出的最大差 {diff:.2g}")'''),
    ("md", "## 7. 注意力图和注意力展开\n\n这一组用类别标记汇聚、3 层，图案对齐到块的网格（每张图只有一个块里有图案）。"
           "对每张测试图取类别标记那一行的注意力，看最亮的块是不是图案所在的块。比较三种看法：第 1 层、最后一层、注意力展开（每层对头取平均，和单位阵各占一半再归一化，逐层相乘）。"),
    ("code", '''def rollout(attns):
    out = None
    for a in attns:
        a = a.mean(1)                                             # 对头取平均，得到 (B, L, L)
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)   # 残差连接算一半
        a = a / a.sum(-1, keepdim=True)
        out = a if out is None else a @ out
    return out

if not READY:
    print("先完成作业里的 " + "、".join(MISSING) + "，再回来运行这一格。")
else:
    make_aligned = lambda n, rng: make_type(n, rng, align=True)
    torch.manual_seed(0); m = TinyViT(depth=3, pool="cls")
    train(m, make_aligned)
    xt, yt, pos = make_aligned(1000, np.random.default_rng(100))
    with torch.no_grad():
        m(xt.to(device))
    attns = [b.attn.cpu() for b in m.blocks]
    target = torch.from_numpy(pos[:, 0] // 4 * 4 + pos[:, 1] // 4)
    maps = {"第 1 层": attns[0].mean(1)[:, 0, 1:], "最后一层": attns[-1].mean(1)[:, 0, 1:], "注意力展开": rollout(attns)[:, 0, 1:]}
    print("测试准确率：", accuracy(m, xt, yt))
    for name, v in maps.items():
        print(f"{name}：最亮的块就是图案所在块的比例 {float((v.argmax(1) == target).float().mean()):.3f}（随机猜 {1 / 16:.4f}）")
    fig, ax = plt.subplots(3, 4, figsize=(9, 6.6))
    for i in range(3):
        ax[i, 0].imshow(xt[i, 0], cmap="gray"); ax[i, 0].set_title("输入：" + GLYPH_NAMES[int(yt[i])])
        for j, (name, v) in enumerate(maps.items()):
            ax[i, j + 1].imshow(v[i].reshape(4, 4), cmap="magma"); ax[i, j + 1].set_title(name)
            r, c = divmod(int(target[i]), 4); ax[i, j + 1].add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec="cyan", lw=1.5))
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    plt.show()'''),
    ("md", "## 8. 标记数、耗时和显存\n\n一层多头自注意力（64 维，4 个头）在不同标记数下的前向耗时。注意力权重是 (头数, N, N) 的张量，N 变成 4 倍，它变成 16 倍。"
           "在有 CUDA 的机器上，这一格还会打印显存峰值，把结果填进实验记录。读带 `torch.cuda` 的那几行时注意：它们没有在写课程的机器上运行过。"),
    ("code", '''if not READY:
    print("先完成作业里的 " + "、".join(MISSING) + "，再回来运行这一格。")
else:
    d, heads = 64, 4
    wq, bq = (torch.randn(3 * d, d) / math.sqrt(d)).to(device), torch.zeros(3 * d, device=device)
    wo, bo = (torch.randn(d, d) / math.sqrt(d)).to(device), torch.zeros(d, device=device)
    print(f"{'标记数':>8} {'注意力权重元素（百万）':>20} {'耗时（ms）':>12}" + ("   显存峰值（MiB）" if device.type == "cuda" else ""))
    for n in (64, 256, 1024, 2048):
        x = torch.randn(1, n, d, device=device)
        with torch.no_grad():
            mod.multi_head_attention(x, wq, bq, wo, bo, heads)          # 预热
            if device.type == "cuda":                                    # 这个分支没有在写课程的机器上运行过
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            ts = []
            for _ in range(5):
                t0 = time.perf_counter(); mod.multi_head_attention(x, wq, bq, wo, bo, heads)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
        line = f"{n:>8} {heads * n * n / 1e6:>20.3f} {1000 * sorted(ts)[2]:>12.2f}"
        if device.type == "cuda":
            line += f"   {torch.cuda.max_memory_allocated() / 2 ** 20:.1f}"
        print(line)'''),
]

OUTRO = ("## 破坏实验与记录\n\n每个实验先写下预测，再运行，把结果和原因记到 `实验记录.md`。\n\n"
         "- 第 6 节里把位置编码换成 `\"learned\"`，分象限的准确率变不变？训练后把 `m.pos` 取出来，同一象限的位置编码是不是更像？\n"
         "- 第 5 节里把 `TinyViT` 换成 `lambda: TinyViT(pos=\"none\")`，样本少的时候和 CNN 的差距变大还是变小？\n"
         "- 用 `TinyViT(depth=6, pre=False)` 和 `TinyViT(depth=6, pre=True)` 各训练 300 步（学习率 0.003），比较最后的损失；"
         "再给后置的那个加上 `warmup=200`、学习率降到 0.001、训练 800 步。\n"
         "- 把 `TinyViT` 的 `heads` 改成 1、2、8，其余不变，参数量变不变？准确率有没有稳定的差别（至少跑 3 个种子）？\n"
         "- 第 7 节里把 `depth` 改成 1。这时“最后一层”和“第 1 层”是同一层，三种看法的命中率各是多少？")


def main():
    nb = notebook(
        "第 25 章互动实验：用自己的零件拼一个小 ViT",
        "作业里的五个函数（切块、还原、缩放点积注意力、多头注意力、二维正余弦位置编码）在这里拼成一个很小的 ViT，"
        "在代码生成的小图上训练，全部在 CPU 上几分钟内跑完，不需要下载任何权重。有 CUDA 时会自动用显卡。\n\n"
        "作业还没完成时，每个单元会提示还缺哪个函数，不会报错。建议先让 `tests/check_progress.py` 全部 PASS 再来跑训练的单元。",
        "25_Transformer与ViT", "25-1_ViT核心零件.py", CELLS, OUTRO,
    )
    rel = "25_Transformer与ViT/实验_小ViT.ipynb"
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    for i, cell in enumerate(nb.cells):
        cell["id"] = f"c{i:03d}"
    nbf.write(nb, path)
    print("写入", rel)


if __name__ == "__main__":
    main()
