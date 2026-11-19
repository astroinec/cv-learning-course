"""生成第 26 章的互动实验 Notebook。

用法（在仓库根目录运行）：
    python scripts/part3_notebooks/ch26.py
修改实验内容时改本文件再重新生成，不要直接手改 .ipynb，避免两边不一致。
Notebook 只调用学员自己在作业文件里的实现，不包含参考答案；
设置环境变量 COURSE_IMPL_DIR 可以让 Notebook 改用另一份实现（维护者验证用）。

本文件自成一体：HELPER 和 notebook() 抄自 scripts/build_track2_notebooks.py，不 import 那个文件。
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = "26_自监督基础模型"
ASSIGNMENT = "26-1_掩码建模与自蒸馏.py"
OUTPUT = "实验_自监督基础模型.ipynb"

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

def implemented(fn, *args, **kwargs):
    """试着调用一次作业函数，返回它是否已经实现。有些作业函数没有返回值，所以不能靠 attempt 的返回值判断。"""
    try:
        fn(*args, **kwargs)
        return True
    except NotImplementedError as exc:
        print(f"先完成作业里的 {fn.__name__}：{exc}")
        return False

SIZE, P = 32, 8                      # 32x32 的图切成 8x8 的块，一共 16 个
G, L = SIZE // P, (SIZE // P) ** 2
ANGLES, PERIODS = (0.0, 45.0, 90.0, 135.0), (6.0, 9.0, 14.0)
YY, XX = torch.meshgrid(torch.arange(SIZE, dtype=torch.float32), torch.arange(SIZE, dtype=torch.float32), indexing="ij")

def latent(n, g, ya=None, yb=None):
    """每张图的隐变量：方向类别、周期类别、相位、条纹幅度、平均亮度。"""
    ya = torch.randint(0, len(ANGLES), (n,), generator=g) if ya is None else ya
    yb = torch.randint(0, len(PERIODS), (n,), generator=g) if yb is None else yb
    return {"ya": ya, "yb": yb, "ph": torch.rand(n, generator=g) * 2 * math.pi,
            "amp": 0.15 + 0.15 * torch.rand(n, generator=g), "br": 0.4 + 0.2 * torch.rand(n, generator=g)}

def draw(z, g, shift=0.0, jitter=0.0, noise=0.05, angles=ANGLES, periods=PERIODS):
    """按隐变量画图。shift、jitter 大于 0 时是一次增强：整体平移、亮度整体加减一个常数。"""
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

g = torch.Generator().manual_seed(26)
cls = torch.arange(12)
demo = draw(latent(12, g, cls // 3, cls % 3), g)
fig, ax = plt.subplots(3, 4, figsize=(7, 5.4))
for i in range(12):
    a = ax[i % 3, i // 3]
    a.imshow(demo[i], cmap="gray", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"{int(ANGLES[i // 3])} 度，周期 {int(PERIODS[i % 3])} px", fontsize=9)
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
    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.n2(x))

class Encoder(nn.Module):
    """很小的 ViT：块的线性嵌入、固定的正弦位置编码、一个类别标记、2 个 Transformer 块。"""
    def __init__(self, dim=64, depth=2):
        super().__init__()
        self.embed = nn.Linear(P * P, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim)); nn.init.normal_(self.cls, std=0.02)
        self.register_buffer("pos", sincos_2d(dim, G)[None])
        self.blocks = nn.ModuleList([Block(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.dim = dim
    def tokens(self, img):
        return self.embed(patchify(normed(img))) + self.pos
    def run(self, tok):
        x = torch.cat([self.cls.expand(len(tok), -1, -1), tok], 1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
    def forward(self, img):                      # 输出 (B, 1 + 16, 64)：第 0 个是类别标记，后面是 16 个块特征
        return self.run(self.tokens(img))

class MAE(nn.Module):
    """编码器只处理可见块；解码器更窄、只有 1 层。遮挡用你写的 random_masking。"""
    def __init__(self, dim=64, ddim=48):
        super().__init__()
        self.enc = Encoder(dim)
        self.proj = nn.Linear(dim, ddim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, ddim)); nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer("dpos", sincos_2d(ddim, G)[None])
        self.dblock, self.dnorm, self.out = Block(ddim), nn.LayerNorm(ddim), nn.Linear(ddim, P * P)
    def forward(self, img, ratio, noise):
        keep, mask, restore = mod.random_masking(self.enc.tokens(img), ratio, noise)
        z = self.proj(self.enc.run(keep))                                   # 第 0 个是类别标记
        full = torch.cat([z[:, 1:], self.mask_token.expand(len(z), L - keep.shape[1], -1)], 1)
        full = torch.gather(full, 1, restore[..., None].expand(-1, -1, full.shape[-1])) + self.dpos
        x = self.dnorm(self.dblock(torch.cat([z[:, :1], full], 1)))
        return self.out(x)[:, 1:], mask

class DinoNet(nn.Module):
    """编码器加投影头。投影头最后做 L2 归一化，再和 32 个归一化的原型做内积，输出在 -1 到 1 之间。"""
    def __init__(self, k=32):
        super().__init__()
        self.enc = Encoder()
        self.mlp = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 32))
        self.last = nn.Linear(32, k, bias=False)
    def forward(self, img):
        return F.normalize(self.mlp(self.enc(img)[:, 0]), dim=-1) @ F.normalize(self.last.weight, dim=-1).t()

def optimizer(params, lr, steps):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
    return opt, torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)

print("编码器参数个数：", sum(p.numel() for p in Encoder().parameters()), "｜MAE 合计：", sum(p.numel() for p in MAE().parameters()))'''

MASKING = '''g = torch.Generator().manual_seed(1)
img = draw(latent(4, g), g)
tok = patchify(img)                          # (4, 16, 64)：这里直接拿像素块当“块向量”，方便看遮了哪些
out = attempt(mod.random_masking, tok, 0.75, torch.rand(4, L, generator=g))
if out is not None:
    keep, mask, restore = out
    print("可见块：", tuple(keep.shape), "｜每张图被遮住的块数：", mask.sum(1).tolist())
    print("第 1 张图的 ids_restore：", restore[0].tolist())
    filler = torch.full((4, L - keep.shape[1], P * P), float("nan"))        # 被遮住的位置先放 nan，画出来是橙色
    back = torch.gather(torch.cat([keep, filler], 1), 1, restore[..., None].expand(-1, -1, P * P))
    cmap = plt.get_cmap("gray").copy(); cmap.set_bad("#f3b98f")
    fig, ax = plt.subplots(2, 4, figsize=(9, 4.6))
    for i in range(4):
        ax[0, i].imshow(img[i], cmap="gray", vmin=0, vmax=1); ax[0, i].set_title("原图")
        ax[1, i].imshow(unpatchify(back)[i], cmap=cmap, vmin=0, vmax=1); ax[1, i].set_title("用 ids_restore 放回原位")
        for a in ax[:, i]: a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.show()'''

MAE_TRAIN = '''def train_mae(ratio=0.75, steps=600, loss_all=False, norm_pix=False, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(100 + seed)
    net = MAE().to(device)
    opt, sched = optimizer(net.parameters(), 3e-3, steps)
    for step in range(steps):
        img = draw(latent(32, g), g).to(device)
        pred, mask = net(img, ratio, torch.rand(32, L, generator=g).to(device))
        loss = mod.masked_mse_loss(pred, patchify(normed(img)), torch.ones_like(mask) if loss_all else mask, norm_pix)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 150 == 0 or step == steps - 1:
            print(f"  第 {step:4d} 步，损失 {loss.item():.3f}")
    return net.eval()

mae_ready = (implemented(mod.random_masking, torch.zeros(1, L, 4), 0.75, torch.rand(1, L))
             and implemented(mod.masked_mse_loss, torch.zeros(1, 2, 4), torch.ones(1, 2, 4), torch.ones(1, 2)))
mae = None
if mae_ready:
    t0 = time.perf_counter()
    mae = train_mae()
    print(f"训练用时 {time.perf_counter() - t0:.1f} 秒。目标是归一化后的像素，全部预测成平均值时损失大约是 0.5。")
    g = torch.Generator().manual_seed(2)
    img = draw(latent(4, g), g)
    with torch.no_grad():
        pred, mask = mae(img.to(device), 0.75, torch.rand(4, L, generator=g).to(device))
    rec = unpatchify(pred.cpu()) * 0.25 + 0.5
    m = unpatchify(mask.cpu()[..., None].expand(-1, -1, P * P)) > 0.5
    cmap = plt.get_cmap("gray").copy(); cmap.set_bad("#f3b98f")
    rows = [("原图", img), ("编码器看到的", torch.where(m, torch.full_like(img, float("nan")), img)), ("解码器的原始输出", rec), ("把可见块贴回去", torch.where(m, rec, img))]
    fig, ax = plt.subplots(4, 4, figsize=(8, 8.4))
    for r, (title, data) in enumerate(rows):
        for i in range(4):
            ax[r, i].imshow(data[i], cmap=cmap, vmin=0, vmax=1); ax[r, i].set_xticks([]); ax[r, i].set_yticks([])
        ax[r, 0].set_ylabel(title)
    plt.tight_layout(); plt.show()'''

DINO_TRAIN = '''def train_dino(steps=400, momentum=0.99, t_temp=0.04, center=True, seed=0):
    """返回学生的编码器和训练记录。两个视图是同一张图的两次增强（平移 2 px 以内、亮度加减 0.1 以内、各自的噪声）。"""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(200 + seed)
    student, teacher = DinoNet().to(device), DinoNet().to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt, sched = optimizer(student.parameters(), 2e-3, steps)
    c = torch.zeros(1, 32, device=device)
    log = {"step": [], "H": [], "Hbar": []}
    for step in range(steps):
        z = latent(32, g)
        v1, v2 = draw(z, g, 2.0, 0.1).to(device), draw(z, g, 2.0, 0.1).to(device)
        s1, s2 = student(v1), student(v2)
        with torch.no_grad():
            t1, t2 = teacher(v1), teacher(v2)
        loss = 0.5 * (mod.dino_loss(s1, t2, c, 0.1, t_temp) + mod.dino_loss(s2, t1, c, 0.1, t_temp))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        mod.ema_update(teacher, student, momentum)
        tt = torch.cat([t1, t2])
        if step % 10 == 0 or step == steps - 1:
            q = F.softmax((tt - c) / t_temp, -1)                       # 教师的目标分布
            ent = lambda p: float(-(p * (p + 1e-12).log()).sum(-1).mean())
            log["step"].append(step); log["H"].append(ent(q)); log["Hbar"].append(ent(q.mean(0, keepdim=True)))
        if center:
            c = 0.9 * c + 0.1 * tt.mean(0, keepdim=True)               # 中心：教师输出的滑动平均
    return student.enc.eval(), log

dino_ready = (implemented(mod.dino_loss, torch.zeros(2, 4), torch.zeros(2, 4), torch.zeros(4))
              and implemented(mod.ema_update, nn.Linear(2, 2), nn.Linear(2, 2), 0.9))
dino_runs = {}
if dino_ready:
    for name, kw in (("默认", {}), ("去掉中心化", {"center": False}), ("去掉锐化（教师温度 0.1）", {"t_temp": 0.1})):
        t0 = time.perf_counter()
        dino_runs[name] = train_dino(**kw)
        log = dino_runs[name][1]
        print(f"{name}：{time.perf_counter() - t0:.1f} 秒，最后单个样本的熵 {log['H'][-1]:.2f}，平均分布的熵 {log['Hbar'][-1]:.2f}（ln 32 = {math.log(32):.2f}）")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for name, (_, log) in dino_runs.items():
        ax[0].plot(log["step"], log["H"], label=name)
        ax[1].plot(log["step"], np.array(log["Hbar"]) - np.array(log["H"]), label=name)
    ax[0].axhline(math.log(32), color="gray", ls=":"); ax[0].set_title("单个样本的熵"); ax[0].set_xlabel("训练步数"); ax[0].legend()
    ax[1].set_title("平均分布的熵 - 单个样本的熵"); ax[1].set_xlabel("训练步数")
    plt.tight_layout(); plt.show()'''

EVAL = '''@torch.no_grad()
def features(enc, img, kind="cls"):
    """冻结的编码器给出的整图特征：类别标记，或者 16 个块特征的平均。enc 为 None 时直接用像素。"""
    if enc is None:
        return img.reshape(len(img), -1).numpy().astype(np.float64)
    out = enc(img.to(device)).cpu()
    return (out[:, 0] if kind == "cls" else out[:, 1:].mean(1)).numpy().astype(np.float64)

g = torch.Generator().manual_seed(5)
cls = torch.arange(12).repeat_interleave(3)                 # 每个“方向 x 周期”组合 3 张带标签的图，共 36 张
z_tr = latent(36, g, cls // 3, cls % 3); x_tr = draw(z_tr, g)
z_te = latent(600, g); x_te = draw(z_te, g)
torch.manual_seed(100)
encoders = {"原始像素": None, "随机初始化的编码器": Encoder().to(device).eval()}
if mae is not None: encoders["MAE 的编码器"] = mae.enc
if "默认" in dino_runs: encoders["自蒸馏的编码器"] = dino_runs["默认"][0]
if "去掉中心化" in dino_runs: encoders["自蒸馏（去掉中心化，坍缩了）"] = dino_runs["去掉中心化"][0]

if implemented(mod.knn_classify, np.eye(2), np.array([0, 1]), np.eye(2), 1, 2):
    print(f"{'k 近邻精度（k=5，36 张标签）':28s}  方向（4 类）  周期（3 类）")
    for name, enc in encoders.items():
        a, b = features(enc, x_tr), features(enc, x_te)
        acc = [float((mod.knn_classify(a, z_tr[key].numpy(), b, 5, n) == z_te[key].numpy()).mean()) for key, n in (("ya", 4), ("yb", 3))]
        print(f"{name:28s}  {acc[0]:.3f}       {acc[1]:.3f}")'''

ANOMALY = '''def pattern_defect(z, img, g, per, size, noise):
    """size x size 的方块里条纹换成另一种周期，其余不变。返回带缺陷的图和缺陷掩码。"""
    n = len(img)
    alt = draw(z, g, angles=(0.0,), periods=(per,), noise=noise)
    cy = torch.randint(0, SIZE - size + 1, (n,), generator=g); cx = torch.randint(0, SIZE - size + 1, (n,), generator=g)
    m = (YY[None] >= cy[:, None, None]) & (YY[None] < cy[:, None, None] + size) & (XX[None] >= cx[:, None, None]) & (XX[None] < cx[:, None, None] + size)
    return torch.where(m, alt, img), m

@torch.no_grad()
def patch_tokens(enc, img):
    """每张图 16 个块特征。enc 为 None 时用归一化后的 8x8 像素块。"""
    if enc is None:
        return patchify(normed(img)).numpy().astype(np.float64)
    return enc(img.to(device))[:, 1:].cpu().numpy().astype(np.float64)

NOISE_AD = 0.2                                               # 改成 0.05 再跑一次，看排序怎么变
g = torch.Generator().manual_seed(11)
zeros = torch.zeros(220, dtype=torch.long)
z = latent(220, g, zeros, zeros)
imgs = draw(z, g, angles=(0.0,), periods=(9.0,), noise=NOISE_AD)             # 良品：0 度、周期 9 px 的条纹
good, test = imgs[:20], imgs[20:].clone()
bad, dmask = pattern_defect({k: v[20::2] for k, v in z.items()}, test[::2], g, 6.0, 10, NOISE_AD)
test[::2] = bad                                              # 一半待检图里有一个 10x10 的方块周期变成了 6 px
labels = np.zeros(200, dtype=int); labels[::2] = 1

mod19 = None
try:
    mod19 = load_module(COURSE_ROOT / "19_异常检测与深度缺陷检测", "19-1_PatchCore核心.py")
except Exception as exc:
    print("没有找到第 19 章的作业文件：", type(exc).__name__, exc)
if mod19 is not None and implemented(mod19.knn_scores, np.eye(2), np.eye(2), 1) and implemented(mod19.auroc, np.array([0.1, 0.9]), np.array([0, 1])):
    print("这里用的是你在第 19 章写的 knn_scores 和 auroc。")
    shown = None
    for name, enc in encoders.items():
        memory = patch_tokens(enc, good); query = patch_tokens(enc, test)
        s = mod19.knn_scores(memory.reshape(-1, memory.shape[-1]), query.reshape(-1, query.shape[-1]), k=1).reshape(200, L)
        print(f"{name:28s}  image AUROC {mod19.auroc(s.max(1), labels):.3f}")
        shown = (name, s[0].reshape(G, G))
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.3))
    ax[0].imshow(test[0], cmap="gray", vmin=0, vmax=1); ax[0].contour(dmask[0].numpy(), levels=[0.5], colors="c"); ax[0].set_title("待检图，青色框是缺陷")
    ax[1].imshow(shown[1], cmap="magma"); ax[1].set_title(f"4x4 的分数图\\n（{shown[0]}）", fontsize=9)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.show()'''

REAL = '''RUN_REAL = False          # 改成 True 再运行这个单元；第一次会联网下载官方仓库的代码和权重
if not RUN_REAL:
    print("跳过。把 RUN_REAL 改成 True 再运行；做法、显存不够时怎么办、应该看到什么，见教程第 6 节。")
else:
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
        x = torch.rand(2, 3, 224, 224, device=device)      # 换成你自己的图：边长是 14 的倍数，按 ImageNet 的均值和标准差归一化
        with torch.inference_mode():
            out = model.forward_features(x)
        print("类别标记：", tuple(out["x_norm_clstoken"].shape), "｜块特征：", tuple(out["x_norm_patchtokens"].shape))
        if device.type == "cuda":
            print("显存峰值（MiB）：", round(torch.cuda.max_memory_allocated() / 2 ** 20))
    except Exception as exc:
        print("没有跑通：", type(exc).__name__, exc)'''

CELLS = [
    ("md", "## 准备：数据和小模型\n\n数据是代码生成的 32×32 条纹：方向 4 种、周期 3 种，一共 12 类，每张图的相位、幅度、亮度随机。模型是一个很小的 ViT（8×8 的块，一张图 16 个块）。这两个单元不涉及作业，直接运行。"),
    ("code", SETUP),
    ("code", MODEL),
    ("md", "## 实验 1：随机遮挡和还原下标\n\n用你写的 `random_masking` 遮住 75% 的块，再按文档字符串里的办法用 `ids_restore` 把可见块放回原位。每张图遮住的位置应该各不相同，放回去之后可见块应该在原来的位置上。"),
    ("code", MASKING),
    ("md", "## 实验 2：迷你 MAE\n\n编码器只看 4 个可见块，解码器补上掩码标记重建全部 16 个块，损失用你写的 `masked_mse_loss`，只算被遮住的 12 个块。训练 600 步。\n\n先预测：最后一行“把可见块贴回去”和第三行“解码器的原始输出”，在可见块的位置上会有什么不同？对照教程图 2。"),
    ("code", MAE_TRAIN),
    ("md", "## 实验 3：自蒸馏，和它的两种坍缩\n\n学生和教师是同一个结构，教师的参数用你写的 `ema_update` 跟着学生走，损失是你写的 `dino_loss`。训练三次：默认设置、去掉中心化、去掉锐化（教师温度调到和学生一样的 0.1），每次 400 步。\n\n先预测三次训练里“单个样本的熵”各会走到哪里（接近 0、接近 ln 32、还是中间），再运行。对照教程图 4。"),
    ("code", DINO_TRAIN),
    ("md", "## 实验 4：评价冻结的特征\n\n每个“方向 × 周期”组合只给 3 张带标签的图，用你写的 `knn_classify` 在冻结的特征上分类。比较原始像素、随机初始化的编码器，和上面训练出来的几个编码器。注意坍缩了的那个：它的输出已经不含输入的信息，编码器的特征还剩多少？"),
    ("code", EVAL),
    ("md", "## 实验 5：冻结的块特征加最近邻做异常检测\n\n和第 19 章的 PatchCore 同一个流程，只是把特征换成 ViT 的 16 个块特征：20 张良品的块特征放进记忆库，待检图每个块到记忆库的最近邻距离是它的分数，整图分数取最大值。良品是同一种条纹，缺陷是一个 10×10 的方块里周期从 9 px 变成 6 px。\n\n这个单元调用你在第 19 章写的 `knn_scores` 和 `auroc`。先用噪声 0.2 跑，再把 `NOISE_AD` 改成 0.05 跑一次，对照教程图 6 和第 5 节的表。"),
    ("code", ANOMALY),
    ("md", "## 选做：在自己的机器上加载 DINOv2\n\n需要联网。这个单元默认跳过，写课程的机器上没有运行过，接口以官方仓库 `facebookresearch/dinov2` 的说明为准。"),
    ("code", REAL),
]

OUTRO = ("## 破坏实验与记录\n\n"
         "- 把实验 2 的遮挡比例改成 0.25 和 0.875 各训练一次，比较损失和实验 4 的 k 近邻精度。\n"
         "- 实验 2 里把 `loss_all=True` 传给 `train_mae`，再看“解码器的原始输出”在可见块上的样子。\n"
         "- 实验 3 里把教师动量改成 0 和 0.9 各训练一次，看熵的曲线。\n"
         "- 把两种坍缩各自的两个熵、实验 4 和实验 5 的结果填进 `实验记录.md`。")


def main():
    nb = notebook(
        "第 26 章互动实验：迷你 MAE、迷你自蒸馏和冻结特征",
        "用你在作业里写的五个零件，在一个很小的 ViT 上把掩码图像建模和自蒸馏各训练一遍，再评价冻结的特征。全部数据由代码生成，不需要下载任何权重；作业还没写完时，对应的单元只打印提示。",
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
