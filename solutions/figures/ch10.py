"""第 10 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch10.py

输出：<主分支>/10_迁移学习/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。

数据是代码生成的 5 类缺陷复查小图（良品、划痕、颗粒、偏红污渍、偏蓝污渍），模型是 torchvision 的 ResNet-18，
ImageNet 权重用本机缓存。全部在 CPU 上跑，torch.set_num_threads(2)，固定随机种子。
参数冻结用本章作业的参考实现 configure_trainable_parameters，单张热图归一化用 normalize_heatmap。
耗时用 S.pinned 钉住，设置 CH10_RETIME=1 才重测。
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

torch.set_num_threads(2)
warnings.filterwarnings("ignore", category=UserWarning)

CH = "10_迁移学习"
mod = S.load_solution(CH, "作业_冻结策略与热图归一化.py")
NUM: dict = {}
CLASSES = ["良品", "划痕", "颗粒", "偏红污渍", "偏蓝污渍"]
C = len(CLASSES)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
WEIGHTS = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
SEEDS = (0, 1, 2)
N_TEST = 40          # 测试集每类张数
ITERS = 300          # 一次微调累计看过的训练图张数
BS = 10
LR_HEAD, LR_BODY = 1e-3, 1e-5
ALL_PREFIXES = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc")
DATA_NOTE = ("代码生成的缺陷复查小图，RGB，64x64：灰色拉丝纹理背景（亮度、色偏、明暗渐变都随机，像素噪声标准差 0.03），"
             "5 类各占五分之一：良品、划痕（亮细线）、颗粒（暗圆点，半径 2 到 3.5 px）、偏红污渍、偏蓝污渍（两种污渍只有颜色不同），缺陷位置随机。"
             "模型是 torchvision 的 ResNet-18，ImageNet 权重（IMAGENET1K_V1），输入按 ImageNet 的均值方差归一化。"
             f"测试集固定 {N_TEST * C} 张（每类 {N_TEST} 张），训练集按种子另外生成。全部在 CPU 上跑")
STORE: dict = {}


def save(fig, name):
    """先让 constrained 布局算一次，再把每个坐标轴的位置四舍五入到 1e-5 并关掉布局引擎，然后保存。

    SVG 里 clip-path 的编号是对坐标轴裁剪框的浮点坐标取哈希得到的。实测同一份代码连续运行，图 4 中间那栏的
    裁剪框打印出来的 6 位小数完全相同，编号却时而不同，说明布局结果在最后几位上不稳定。位置取整之后编号只由
    取整后的数决定，文件可以逐字节复现。"""
    fig.canvas.draw()
    if fig.subfigs:   # 子图组的位置还要靠布局引擎排，这种图只能整体保留布局
        return S.save(fig, CH, name)
    for ax in fig.axes:
        b = ax.get_position()
        ax.set_position([round(v, 5) for v in (b.x0, b.y0, b.width, b.height)])
    fig.set_layout_engine("none")
    return S.save(fig, CH, name)


def r3(v):
    return round(float(v), 3)


def mean_and_seeds(rows):
    """rows: [种子][...]。返回均值和各种子的值，都保留三位小数。"""
    a = np.asarray(rows, float)
    return {"均值": r3(a.mean()), "各种子": [r3(x) for x in a]}


# ---------------------------------------------------------------- 数据
def make_image(rng, cls, size=64):
    """一张合成的缺陷复查小图，RGB，取值 0 到 1。返回 (图, 缺陷掩码)。cls=5 是训练时没有的环形缺陷。"""
    s = size / 64.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = rng.uniform(0.40, 0.60)
    tex = rng.normal(0, 1, (size, size)).astype(np.float32)
    k = int(round(15 * s)) | 1
    tex = cv2.blur(tex, (k, 1), borderType=cv2.BORDER_REFLECT) * np.sqrt(k)
    gx, gy = rng.uniform(-0.10, 0.10, 2)
    gray = base + 0.035 * tex + gx * (xx / size - 0.5) + gy * (yy / size - 0.5)
    img = gray[..., None] + rng.normal(0, 0.015, 3).astype(np.float32)
    mask = np.zeros((size, size), bool)
    m = 10 * s
    cy, cx = rng.uniform(m, size - m, 2)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    if cls == 1:
        th = rng.uniform(0, np.pi)
        length = rng.uniform(18, 34) * s
        w = 0.8 * s
        u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        prof = np.exp(-v * v / (2 * w * w)) * (np.abs(u) <= length / 2)
        img = img + rng.uniform(0.10, 0.20) * prof[..., None]
        mask = prof > 0.3
    elif cls == 2:
        r = rng.uniform(2.0, 3.5) * s
        prof = np.clip((r - d) / s + 0.5, 0, 1)
        img = img - rng.uniform(0.12, 0.24) * prof[..., None]
        mask = d <= r
    elif cls in (3, 4):
        sg = rng.uniform(4.0, 7.0) * s
        prof = np.exp(-d * d / (2 * sg * sg))
        col = np.array([1.0, 0.0, -1.0], np.float32) * (1 if cls == 3 else -1)
        img = img + rng.uniform(0.05, 0.10) * prof[..., None] * col
        mask = prof > 0.5
    elif cls == 5:
        r, w = rng.uniform(4.0, 6.0) * s, 1.0 * s
        prof = np.exp(-(d - r) ** 2 / (2 * w * w))
        img = img + rng.uniform(0.10, 0.20) * prof[..., None]
        mask = prof > 0.3
    img = img + rng.normal(0, 0.03, img.shape).astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32), mask


def make_set(seed, per_class, size=64, classes=range(C)):
    """每类 per_class 张，按“每轮各类一张”的顺序排，所以前 k*C 张正好是每类 k 张。"""
    rng = np.random.default_rng(seed)
    X, Y, M = [], [], []
    for _ in range(per_class):
        for c in classes:
            im, mk = make_image(rng, c, size)
            X.append(im)
            Y.append(c)
            M.append(mk)
    return np.stack(X), np.array(Y), np.stack(M)


def to_tensor(X, mean=MEAN, std=STD, bgr=False, scale=1.0):
    """HWC 的 0 到 1 浮点图 -> NCHW 张量。mean=None 表示不减均值不除方差。"""
    X = X[..., ::-1] if bgr else X
    X = X * np.float32(scale)
    if mean is not None:
        X = (X - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return torch.from_numpy(np.ascontiguousarray(X.transpose(0, 3, 1, 2))).float()


# ---------------------------------------------------------------- 模型、训练、评估
def resnet(pretrained=True, seed=0):
    torch.manual_seed(seed)
    net = torchvision.models.resnet18(weights=WEIGHTS if pretrained else None)
    torch.manual_seed(seed + 1000)
    net.fc = nn.Linear(512, C)
    return net


@torch.no_grad()
def trunk(net, x, bs=50):
    """conv1 到 layer3 的输出。net 的模式（train 或 eval）由调用者决定。"""
    out = []
    for i in range(0, len(x), bs):
        h = net.maxpool(net.relu(net.bn1(net.conv1(x[i:i + bs]))))
        out.append(net.layer3(net.layer2(net.layer1(h))))
    return torch.cat(out)


def pooled(net, l3):
    """layer3 的输出 -> 分类头之前的 512 维特征。"""
    return torch.flatten(net.avgpool(net.layer4(l3)), 1)


@torch.no_grad()
def features(net, x, bs=50):
    return torch.cat([pooled(net, trunk(net, x[i:i + bs])) for i in range(0, len(x), bs)])


def accuracy(logits, y):
    return float((logits.argmax(1).numpy() == y).mean())


def set_mode(net, bn_eval):
    """net.train()，然后按需要把所有 BatchNorm 切回 eval（不用批统计量，也不更新 running stats）。"""
    net.train()
    if bn_eval:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def loop(forward, groups, y, iters, bs, seed, wd=1e-4, hook=None, hook_at=()):
    """小批量训练。iters 是累计看过的图像张数；每一轮把训练集打乱后顺序取批。返回每一步的损失。

    hook_at 里的张数一到就调用 hook(张数)，用来在同一次训练的中途评估，省掉一次重复的训练。
    """
    opt = torch.optim.AdamW(groups, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    yt = torch.from_numpy(y)
    n, seen, losses = len(y), 0, []
    while seen < iters:
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = F.cross_entropy(forward(idx), yt[idx])
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            seen += len(idx)
            for t in hook_at:
                if seen - len(idx) < t <= seen and t < iters:
                    hook(t)
            if seen >= iters:
                break
    return losses


def train_head(f, y, steps=300, lr=1e-2, seed=0):
    """在缓存好的特征上训练线性分类头，全批量。骨干完全冻结且 BN 在 eval 时，这和带着骨干训练是一回事。"""
    torch.manual_seed(seed + 2000)
    head = nn.Linear(f.shape[1], C)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    yt = torch.from_numpy(y)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(head(f), yt).backward()
        opt.step()
    return head


def param_groups(net, lr_head, lr_body):
    head = [p for n, p in net.named_parameters() if n.startswith("fc.") and p.requires_grad]
    body = [p for n, p in net.named_parameters() if not n.startswith("fc.") and p.requires_grad]
    return [{"params": head, "lr": lr_head}] + ([{"params": body, "lr": lr_body}] if body else [])


def body_vector(net):
    return torch.cat([p.detach().flatten() for n, p in net.named_parameters() if not n.startswith("fc.")])


def finetune_full(seed, k, lr_head=LR_HEAD, lr_body=LR_BODY, bn_eval=True, iters=ITERS, warm=True, pretrained=True, eval_at=(), train_acc=False):
    """全部微调。warm=True 时分类头先用“只训练分类头”的结果初始化。

    返回训练结束时的测试精度等；eval_at 里的张数一到就在中途评估一次，结果放在返回值的 "at" 里。
    中途评估只做 eval 模式的前向，不改变之后的训练，所以 at[300] 和单独训练 300 张图的结果相同。
    """
    tr, te = STORE["train"][seed], STORE["test"]
    n = k * C
    net = resnet(pretrained, seed)
    if warm:
        net.fc.load_state_dict(STORE["heads"][(seed, k)].state_dict())
    mod.configure_trainable_parameters(net, ALL_PREFIXES)
    w0 = body_vector(net)
    losses, at = [], {}

    def evaluate():
        net.eval()
        f = features(net, te["x"])
        with torch.no_grad():
            res = {"acc": accuracy(net.fc(f), te["Y"]), "骨干权重相对变化": float((body_vector(net) - w0).norm() / w0.norm()),
                   "特征相对变化": float((f - te["f"]).norm() / te["f"].norm()) if pretrained else None}
            if train_acc:
                res["train_acc"] = accuracy(net(tr["x"][:n]), tr["Y"][:n])
        set_mode(net, bn_eval)
        return res

    def snap(t):
        at[t] = evaluate()

    set_mode(net, bn_eval)
    losses = loop(lambda idx: net(tr["x"][idx]), param_groups(net, lr_head, lr_body), tr["Y"][:n], iters, BS, seed, hook=snap, hook_at=eval_at)
    res = evaluate()
    for t, r in at.items():
        r["losses"] = losses[:t // BS]
    return {**res, "losses": losses, "at": at}


def finetune_layer4(seed, k, lr_head=LR_HEAD, lr_body=LR_BODY, iters=ITERS):
    """只解冻 layer4 和分类头。conv1 到 layer3 冻结且 BN 在 eval，它们的输出可以缓存。"""
    tr, te = STORE["train"][seed], STORE["test"]
    n = k * C
    net = resnet(True, seed)
    net.fc.load_state_dict(STORE["heads"][(seed, k)].state_dict())
    names = mod.configure_trainable_parameters(net, ("layer4", "fc"))
    set_mode(net, True)
    loop(lambda idx: net.fc(pooled(net, tr["l3"][idx])), param_groups(net, lr_head, lr_body), tr["Y"][:n], iters, BS, seed)
    net.eval()
    with torch.no_grad():
        logits = net.fc(pooled(net, te["l3"]))
    return accuracy(logits, te["Y"]), len(names)


def auroc(score, positive):
    """按秩次算 AUROC，并列取平均秩。"""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    npos, nneg = positive.sum(), (~positive).sum()
    return float((r[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg))


# ---------------------------------------------------------------- 准备：数据、缓存特征、分类头
KS_DENSE = (1, 2, 3, 5, 10, 20, 30)
KS_FT = (3, 10, 30)


def prepare():
    pre = resnet(True, 0).eval()
    rnd = resnet(False, 0).eval()
    Xte, Yte, _ = make_set(999, N_TEST)
    xte = to_tensor(Xte)
    l3 = trunk(pre, xte)
    STORE["pre"], STORE["rnd"] = pre, rnd
    STORE["test"] = {"X": Xte, "Y": Yte, "x": xte, "l3": l3, "f": pooled(pre, l3).detach(), "f_rnd": features(rnd, xte)}
    STORE["train"], STORE["heads"] = {}, {}
    for seed in SEEDS:
        Xtr, Ytr, _ = make_set(100 + seed, max(KS_DENSE))
        xtr = to_tensor(Xtr)
        l3 = trunk(pre, xtr)
        STORE["train"][seed] = {"X": Xtr, "Y": Ytr, "x": xtr, "l3": l3, "f": pooled(pre, l3).detach(), "f_rnd": features(rnd, xtr)}
        for k in KS_DENSE:
            STORE["heads"][(seed, k)] = train_head(STORE["train"][seed]["f"][:k * C], Ytr[:k * C], seed=seed)
    net = resnet(True, 0)
    counts = {}
    for name, prefixes in (("只训练分类头", ("fc",)), ("解冻 layer4 和分类头", ("layer4", "fc")), ("全部微调", ALL_PREFIXES)):
        names = mod.configure_trainable_parameters(net, prefixes)
        counts[name] = {"可训练的参数张量个数": len(names), "可训练的参数个数": int(sum(p.numel() for p in net.parameters() if p.requires_grad))}
    total = sum(p.numel() for p in net.parameters())
    bn_buffers = int(sum(b.numel() for n, b in net.named_buffers() if "running" in n))
    with torch.no_grad():
        shapes = {}
        for size in (64, 128, 224):
            h = net.layer3(net.layer2(net.layer1(net.maxpool(net.relu(net.bn1(net.conv1(torch.zeros(1, 3, size, size))))))))
            shapes[f"输入 {size}x{size}"] = {"layer3 输出": f"{h.shape[2]}x{h.shape[3]}", "layer4 输出": f"{net.layer4(h).shape[2]}x{net.layer4(h).shape[3]}"}
    NUM["数据与模型"] = {
        "说明": DATA_NOTE,
        "类别": CLASSES,
        "ResNet-18 换成 5 类分类头之后的参数总数": int(total),
        "参数总数（百万）": round(total / 1e6, 2),
        "三种冻结策略": counts,
        "只训练分类头时可训练参数占全部参数的比例": float(f"{counts['只训练分类头']['可训练的参数个数'] / total:.2g}"),
        "解冻 layer4 和分类头时可训练参数占全部参数的比例": r3(counts["解冻 layer4 和分类头"]["可训练的参数个数"] / total),
        "BatchNorm 的 running_mean 和 running_var 一共多少个数（它们不是参数，requires_grad 管不到）": bn_buffers,
        "BatchNorm 层数": int(sum(isinstance(m, nn.BatchNorm2d) for m in net.modules())),
        "预训练的卷积权重绝对值的中位数": float(f"{torch.cat([p.detach().abs().flatten() for p in pre.parameters() if p.dim() == 4]).median():.2g}"),
        "特征图尺寸": shapes,
    }


def fig_samples_vs_k():
    """图 1：每类样本数从 1 到 30，从头训练、只训练分类头、解冻 layer4、全部微调的测试精度。"""
    te = STORE["test"]
    head = {k: [] for k in KS_DENSE}
    rnd_head = {k: [] for k in KS_DENSE}
    for seed in SEEDS:
        tr = STORE["train"][seed]
        for k in KS_DENSE:
            n = k * C
            head[k].append(accuracy(STORE["heads"][(seed, k)](te["f"]).detach(), te["Y"]))
            rnd_head[k].append(accuracy(train_head(tr["f_rnd"][:n], tr["Y"][:n], seed=seed)(te["f_rnd"]).detach(), te["Y"]))
    l4 = {k: [finetune_layer4(seed, k)[0] for seed in SEEDS] for k in KS_FT}
    l4_lr = {lr: (l4[10] if lr == LR_BODY else [finetune_layer4(seed, 10, lr_body=lr)[0] for seed in SEEDS]) for lr in (1e-5, 1e-4, 1e-3)}
    STORE["l4_lr"] = l4_lr
    full = {k: [] for k in KS_FT}
    for k in KS_FT:
        for seed in SEEDS:
            if k == 10:   # 图 3 还要用两倍预算的结果，这里一次跑到 600 张，在 300 张时评估
                res = finetune_full(seed, k, iters=2 * ITERS, eval_at=(ITERS,))
                STORE[("full", seed)] = res["at"][ITERS]
                STORE[("full2x", seed)] = {key: res[key] for key in ("acc", "losses", "骨干权重相对变化", "特征相对变化")}
                full[k].append(res["at"][ITERS]["acc"])
            else:
                full[k].append(finetune_full(seed, k)["acc"])
    scratch_seeds = SEEDS[:2]
    ks_scratch = KS_FT
    scratch = {k: [] for k in ks_scratch}
    scratch_train = {k: [] for k in ks_scratch}
    long_run = []
    for k in ks_scratch:
        for seed in scratch_seeds:
            kw = dict(lr_head=1e-3, lr_body=1e-3, bn_eval=False, warm=False, pretrained=False, train_acc=True)
            if k == 30:
                res = finetune_full(seed, k, iters=2 * ITERS, eval_at=(ITERS,), **kw)
                long_run.append([res["acc"], res["train_acc"], float(np.mean(res["losses"][-5:]))])
                res = res["at"][ITERS]
            else:
                res = finetune_full(seed, k, **kw)
            scratch[k].append(res["acc"])
            scratch_train[k].append(res["train_acc"])
    long_run = np.array(long_run)

    NUM["样本数与做法"] = {
        "说明": DATA_NOTE + f"。每类样本数 k 从 1 到 30，{len(SEEDS)} 个种子（训练集不同、测试集相同）取平均，表里是测试精度；随机猜是 {1 / C:.2f}。"
              "只训练分类头：骨干冻结、BN 在 eval，在缓存的 512 维特征上全批量训练 300 步（AdamW，学习率 1e-2，权重衰减 1e-4）。"
              f"解冻 layer4 和全部微调：分类头先按上一种做法训练好，再用 AdamW 训练，累计看 {ITERS} 张训练图（批大小 {BS}，也就是 {ITERS // BS} 步），"
              f"骨干学习率 {LR_BODY:g}、分类头 {LR_HEAD:g}，BN 保持 eval。"
              f"从头训练：ResNet-18 随机初始化，全部参数学习率 1e-3，BN 用训练模式，同样看 {ITERS} 张图，只跑了 {len(scratch_seeds)} 个种子。"
              "随机骨干只训练分类头：骨干是没有训练过的随机权重，其余和只训练分类头相同",
        "每类样本数（只训练分类头的两行）": list(KS_DENSE),
        "预训练骨干，只训练分类头": {f"k={k}": mean_and_seeds(head[k]) for k in KS_DENSE},
        "随机骨干，只训练分类头": {f"k={k}": mean_and_seeds(rnd_head[k]) for k in KS_DENSE},
        "每类样本数（解冻 layer4 和全部微调）": list(KS_FT),
        "每类样本数（从头训练）": list(ks_scratch),
        "预训练，解冻 layer4 和分类头": {f"k={k}": mean_and_seeds(l4[k]) for k in KS_FT},
        "预训练，全部微调": {f"k={k}": mean_and_seeds(full[k]) for k in KS_FT},
        "从头训练": {f"k={k}": mean_and_seeds(scratch[k]) for k in ks_scratch},
        "从头训练在训练集上的精度（eval 模式）": {f"k={k}": mean_and_seeds(scratch_train[k]) for k in ks_scratch},
        f"从头训练，k=30，看 {2 * ITERS} 张图（两倍预算）": {
            "测试精度": mean_and_seeds(long_run[:, 0]), "训练集精度（eval 模式）": mean_and_seeds(long_run[:, 1]), "最后 5 步的平均训练损失": mean_and_seeds(long_run[:, 2])},
        "全部微调比只训练分类头高多少": {f"k={k}": r3(np.mean(full[k]) - np.mean(head[k])) for k in KS_FT},
        "解冻 layer4 比只训练分类头高多少": {f"k={k}": r3(np.mean(l4[k]) - np.mean(head[k])) for k in KS_FT},
        "只训练分类头：种子之间最大相差": {f"k={k}": r3(max(head[k]) - min(head[k])) for k in KS_DENSE},
        "全部微调：种子之间最大相差": {f"k={k}": r3(max(full[k]) - min(full[k])) for k in KS_FT},
    }

    fig, ax = plt.subplots(1, 2, figsize=(13.2, 4.3), gridspec_kw={"width_ratios": [1.05, 1.25]}, layout="constrained")
    rng = np.random.default_rng(3)
    tiles = [[make_image(rng, c)[0] for c in range(C)] for _ in range(3)]
    gap = 3
    sheet = np.ones((3 * 64 + 2 * gap, C * 64 + (C - 1) * gap, 3), np.float32)
    for i, row in enumerate(tiles):
        for j, im in enumerate(row):
            sheet[i * (64 + gap):i * (64 + gap) + 64, j * (64 + gap):j * (64 + gap) + 64] = im
    ax[0].imshow(np.round(sheet * 255).astype(np.uint8), interpolation="nearest")
    ax[0].set_xticks([j * (64 + gap) + 32 for j in range(C)], CLASSES, fontsize=10)
    ax[0].tick_params(axis="x", length=0, colors=S.INK)
    ax[0].set_yticks([])
    for side in ax[0].spines.values():
        side.set_visible(False)
    ax[0].set_title("5 类缺陷复查小图，每类 3 张（64×64）")

    a = ax[1]
    xs = list(KS_DENSE)
    a.plot(xs, [np.mean(head[k]) for k in xs], "-o", color=S.BLUE, lw=2, ms=5, label="预训练，只训练分类头")
    for k in xs:
        a.plot([k] * len(SEEDS), head[k], ".", color=S.BLUE, alpha=0.35, ms=5)
    a.plot(list(KS_FT), [np.mean(l4[k]) for k in KS_FT], "--s", color=S.GREEN, lw=1.6, ms=6, label="预训练，解冻 layer4 和分类头")
    a.plot(list(KS_FT), [np.mean(full[k]) for k in KS_FT], "--D", color=S.ORANGE, lw=1.6, ms=6, label="预训练，全部微调")
    a.plot(xs, [np.mean(rnd_head[k]) for k in xs], "-o", color=S.MUTED, lw=1.4, ms=4, label="随机骨干，只训练分类头")
    a.plot(list(ks_scratch), [np.mean(scratch[k]) for k in ks_scratch], ":^", color=S.RED, lw=1.6, ms=7, label="从头训练")
    a.axhline(1 / C, color=S.MUTED, lw=1, ls=(0, (2, 3)))
    a.text(1.0, 1 / C - 0.06, "随机猜 0.20", color=S.MUTED, fontsize=9)
    a.set_xscale("log")
    a.set_xticks(xs, [str(k) for k in xs])
    a.minorticks_off()
    a.set_ylim(0.1, 1.0)
    a.set_xlabel("每类训练样本数")
    a.set_ylabel("测试精度（200 张）")
    a.set_title("预训练特征加线性头远好于从头训练；这个预算下微调只多几个点")
    a.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    S.clean(a)
    save(fig, "fig01_样本数与三种做法")


# ---------------------------------------------------------------- 图 3：骨干和分类头的学习率
def bar_with_seeds(ax, x, rows, color, width=0.62, fmt="{:.2f}", label=None, dy=0.02):
    """柱子加各种子的点和数值标注。画图用的数先四舍五入到 4 位：训练结果在最后几位上的浮点差别
    会改变标注文字的位置，进而改变 constrained 布局和 SVG 里 clip-path 的编号，让文件不能逐字节复现。"""
    rows = [[round(float(v), 4) for v in r] for r in rows]
    vals = [round(float(np.mean(r)), 4) for r in rows]
    bars = ax.bar(x, vals, width, color=color, alpha=0.88, label=label)
    for b, r, v in zip(bars, rows, vals):
        cx = b.get_x() + b.get_width() / 2
        ax.plot([cx] * len(r), r, ".", color=S.INK, ms=5, alpha=0.7)
        ax.text(cx, max(max(r), v) + dy, fmt.format(v), ha="center", fontsize=9)
    return bars


def fig_learning_rate():
    k = 10
    configs = [("同一个 1e-3", 1e-3, 1e-3), ("同一个 1e-4", 1e-4, 1e-4), ("骨干 1e-4\n分类头 1e-3", 1e-3, 1e-4), ("骨干 1e-5\n分类头 1e-3", LR_HEAD, LR_BODY)]
    out, longer = {}, {}
    for name, lh, lb in configs:
        if (lh, lb) == (LR_HEAD, LR_BODY):
            out[name] = [STORE[("full", seed)] for seed in SEEDS]
            longer[name] = [STORE[("full2x", seed)] for seed in SEEDS]
        elif lh != lb:
            runs = [finetune_full(seed, k, lr_head=lh, lr_body=lb, iters=2 * ITERS, eval_at=(ITERS,)) for seed in SEEDS]
            out[name] = [r["at"][ITERS] for r in runs]
            longer[name] = runs
        else:
            out[name] = [finetune_full(seed, k, lr_head=lh, lr_body=lb) for seed in SEEDS]
    head_only = [accuracy(STORE["heads"][(seed, k)](STORE["test"]["f"]).detach(), STORE["test"]["Y"]) for seed in SEEDS]
    key = lambda name: name.replace("\n", "，")  # noqa: E731
    steps = ITERS // BS
    NUM["学习率"] = {
        "说明": DATA_NOTE + f"。每类 {k} 张，{len(SEEDS)} 个种子。分类头先按“只训练分类头”训练好（这时训练损失已经接近 0），再解冻全部参数用 AdamW 微调，"
              f"累计看 {ITERS} 张图（{steps} 步，批大小 {BS}），BN 保持 eval，权重衰减 1e-4。只改骨干和分类头的学习率。"
              "骨干权重相对变化是微调前后骨干全部参数之差的范数除以微调前的范数；特征相对变化是测试集 512 维特征之差的范数除以微调前的范数",
        "只训练分类头的测试精度（对照）": mean_and_seeds(head_only),
        **{key(name): {
            "测试精度": mean_and_seeds([r["acc"] for r in runs]),
            "微调第 1 步的训练损失": mean_and_seeds([r["losses"][0] for r in runs]),
            "前 10 步里训练损失的最大值": mean_and_seeds([max(r["losses"][:10]) for r in runs]),
            "最后 5 步的平均训练损失": mean_and_seeds([np.mean(r["losses"][-5:]) for r in runs]),
            "骨干权重相对变化": mean_and_seeds([r["骨干权重相对变化"] for r in runs]),
            "测试集特征相对变化": mean_and_seeds([r["特征相对变化"] for r in runs]),
        } for name, runs in out.items()},
        "只解冻 layer4 和分类头时 layer4 的学习率（分类头 1e-3，其余和上面相同，conv1 到 layer3 冻结）": {
            f"layer4 学习率 {lr:g}": mean_and_seeds(v) for lr, v in STORE["l4_lr"].items()},
        f"两倍预算（累计看 {2 * ITERS} 张图，{2 * ITERS // BS} 步），其余不变": {key(name): {
            "测试精度": mean_and_seeds([r["acc"] for r in runs]),
            "最后 5 步的平均训练损失": mean_and_seeds([np.mean(r["losses"][-5:]) for r in runs]),
            "骨干权重相对变化": mean_and_seeds([r["骨干权重相对变化"] for r in runs]),
            "比只训练分类头高多少": r3(np.mean([r["acc"] for r in runs]) - np.mean(head_only)),
        } for name, runs in longer.items()},
        "骨干学习率 1e-4 与 1e-5 的权重相对变化之比（分类头都是 1e-3）": r3(np.mean([r["骨干权重相对变化"] for r in out[configs[2][0]]]) / np.mean([r["骨干权重相对变化"] for r in out[configs[3][0]]])),
        "骨干学习率 1e-3 与 1e-4 的权重相对变化之比（同一个学习率的两组）": r3(np.mean([r["骨干权重相对变化"] for r in out[configs[0][0]]]) / np.mean([r["骨干权重相对变化"] for r in out[configs[1][0]]])),
    }

    colors = [S.RED, S.PURPLE, S.GREEN, S.ORANGE]
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.4), gridspec_kw={"width_ratios": [1.55, 1.15, 0.95]}, layout="constrained")
    x = np.arange(len(configs))
    for i, (name, _, _) in enumerate(configs):
        bar_with_seeds(ax[0], [i], [[r["acc"] for r in out[name]]], colors[i])
    xl = [4.5, 5.5]
    for xi, (name, _, _), color in zip(xl, configs[2:], colors[2:]):
        bars = bar_with_seeds(ax[0], [xi], [[r["acc"] for r in longer[name]]], color)
        for b in bars:
            b.set_hatch("//")
            b.set_edgecolor("white")
    ax[0].axhline(np.mean(head_only), color=S.BLUE, lw=1.4, ls="--", label=f"只训练分类头 {np.mean(head_only):.2f}")
    ax[0].axhline(1 / C, color=S.MUTED, lw=1, ls=(0, (2, 3)), label="随机猜 0.20")
    ax[0].axvline(3.75, color=S.GRID, lw=1)
    ax[0].text(1.5, 1.07, f"累计看 {ITERS} 张图", ha="center", fontsize=9, color=S.MUTED)
    ax[0].text(5.0, 1.07, f"累计看 {2 * ITERS} 张图", ha="center", fontsize=9, color=S.MUTED)
    ax[0].set_xticks(list(x) + xl, [n for n, _, _ in configs] + [n for n, _, _ in configs[2:]], fontsize=9)
    ax[0].set_ylim(0, 1.13)
    ax[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax[0].set_ylabel("测试精度（每类 10 张，黑点是各种子）")
    ax[0].set_title("骨干和分类头都用 1e-3，精度掉回随机猜")
    S.clean(ax[0])
    ax[0].grid(axis="x", visible=False)

    for i, (name, _, _) in enumerate(configs):
        med = np.median(np.array([r["losses"] for r in out[name]]), axis=0)
        ax[1].plot(np.arange(1, len(med) + 1), np.maximum(med, 1e-4), "-", color=colors[i], lw=1.8, label=key(name))
    ax[1].set_yscale("log")
    ax[1].set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], ["≤0.0001", "0.001", "0.01", "0.1", "1", "10", "100"])
    ax[1].minorticks_off()
    ax[1].set_xlabel("微调的步数（每步 10 张图）")
    ax[1].set_ylabel("这一步的训练损失（3 个种子的中位数）")
    ax[1].set_title("骨干学习率 1e-4 以上，损失先从接近 0 窜上去")
    ax[1].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[1])

    vals = [np.mean([r["骨干权重相对变化"] for r in out[name]]) for name, _, _ in configs]
    bars = ax[2].bar(x, vals, 0.62, color=colors, alpha=0.88)
    for b, v in zip(bars, vals):
        ax[2].text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.2g}", ha="center", fontsize=9)
    ax[2].set_yscale("log")
    ax[2].set_ylim(min(vals) / 3, max(vals) * 4)
    ax[2].set_yticks([1e-3, 1e-2, 1e-1], ["0.001", "0.01", "0.1"])
    ax[2].minorticks_off()
    ax[2].set_xticks(x, [n for n, _, _ in configs], fontsize=8.5)
    ax[2].set_ylabel("骨干权重的相对变化")
    ax[2].set_title("骨干学习率大十倍，权重也多挪约十倍")
    S.clean(ax[2])
    ax[2].grid(axis="x", visible=False)
    save(fig, "fig03_学习率")


# ---------------------------------------------------------------- 图 4：BatchNorm 留在训练模式
def fig_batchnorm():
    k = 10
    n = k * C
    te, pre = STORE["test"], STORE["pre"]
    rows = {"eval,4": [], "eval,10": [], "train,4": [], "train,10": [], "重估": [], "imagenet": []}
    same_params, drift, buf_change, tr_vs_ev = [], [], [], []
    for seed in SEEDS:
        tr = STORE["train"][seed]
        y = tr["Y"][:n]
        rows["imagenet"].append(accuracy(STORE["heads"][(seed, k)](te["f"]).detach(), te["Y"]))
        for bs in (4, 10):
            # BN 在 eval：骨干的输出不变，直接在缓存特征上跑同样的小批量循环
            net = resnet(True, seed)
            loop(lambda idx: net.fc(tr["f"][idx]), [{"params": net.fc.parameters(), "lr": 1e-2}], y, ITERS, bs, seed)
            rows[f"eval,{bs}"].append(accuracy(net.fc(te["f"]).detach(), te["Y"]))
            # BN 在 train：参数冻结，但每个批都用批统计量，running_mean 和 running_var 跟着更新
            net = resnet(True, seed)
            mod.configure_trainable_parameters(net, ("fc",))
            before = {name: p.detach().clone() for name, p in net.named_parameters() if not name.startswith("fc.")}
            buf0 = torch.cat([b.flatten().clone() for name, b in net.named_buffers() if "running" in name])
            net.train()
            loop(lambda idx: net(tr["x"][idx]), [{"params": net.fc.parameters(), "lr": 1e-2}], y, ITERS, bs, seed)
            net.eval()
            f = features(net, te["x"])
            rows[f"train,{bs}"].append(accuracy(net.fc(f).detach(), te["Y"]))
            if bs == 10:
                same_params.append(all(torch.equal(before[name], p) for name, p in net.named_parameters() if name in before))
                buf1 = torch.cat([b.flatten() for name, b in net.named_buffers() if "running" in name])
                buf_change.append(float((buf1 - buf0).norm() / buf0.norm()))
                drift.append(float((f - te["f"]).norm() / te["f"].norm()))
        # 不带噪声的版本：用训练集把 BN 统计量整体重估一遍（momentum=None 是累计平均），再冻结、提特征、训练分类头
        net = resnet(True, seed)
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                m.momentum = None
        net.train()
        f_train_mode = torch.cat([features(net, tr["x"][i:i + BS]) for i in range(0, n, BS)])
        tr_vs_ev.append(float((f_train_mode - tr["f"][:n]).norm() / tr["f"][:n].norm()))
        net.eval()
        head = train_head(features(net, tr["x"][:n]), y, seed=seed)
        rows["重估"].append(accuracy(head(features(net, te["x"])).detach(), te["Y"]))
    ft_eval = [STORE[("full", seed)]["acc"] for seed in SEEDS]
    ft_train = [finetune_full(seed, k, bn_eval=False)["acc"] for seed in SEEDS]
    with torch.no_grad():
        a = pre.conv1(te["x"])
        ratio = np.sort((a.var(dim=(0, 2, 3)) / pre.bn1.running_var).numpy())
        shift = ((a.mean(dim=(0, 2, 3)) - pre.bn1.running_mean) / pre.bn1.running_var.sqrt()).abs().numpy()
        x_in = te["x"]
    steps = ITERS // BS
    NUM["BatchNorm 模式"] = {
        "说明": DATA_NOTE + f"。每类 {k} 张，{len(SEEDS)} 个种子。冻结骨干只训练分类头：骨干全部参数 requires_grad=False，分类头随机初始化，AdamW 学习率 1e-2，"
              f"小批量训练累计看 {ITERS} 张图。BN 在 eval 指训练时骨干的 BatchNorm 用 ImageNet 的 running 统计量；BN 在 train 指直接 model.train()，"
              "BatchNorm 用当前批的统计量并更新 running 统计量（momentum 0.1）。两种情况的批次划分和分类头初值相同，测试时都是 model.eval()",
        "冻结骨干，BN 在 eval，批大小 4": mean_and_seeds(rows["eval,4"]),
        "冻结骨干，BN 在 train，批大小 4": mean_and_seeds(rows["train,4"]),
        "冻结骨干，BN 在 eval，批大小 10": mean_and_seeds(rows["eval,10"]),
        "冻结骨干，BN 在 train，批大小 10": mean_and_seeds(rows["train,10"]),
        "BN 留在 train 掉了多少（批大小 4）": r3(np.mean(rows["eval,4"]) - np.mean(rows["train,4"])),
        "BN 留在 train 掉了多少（批大小 10）": r3(np.mean(rows["eval,10"]) - np.mean(rows["train,10"])),
        f"BN 在 train 训练 {steps} 步之后（批大小 10）": {
            "骨干参数是否逐位不变（各种子）": [bool(v) for v in same_params],
            "running_mean 和 running_var 的相对变化": mean_and_seeds(buf_change),
            "同一批测试图的 512 维特征相对变化": mean_and_seeds(drift),
            "running 统计量向新数据靠拢的比例（1 - 0.9 的步数次方）": r3(1 - 0.9 ** steps),
        },
        "同一批训练图，train 模式的特征和 eval 模式的特征相差多少（相对范数，批大小 10）": mean_and_seeds(tr_vs_ev),
        "不带批噪声的对照：用训练集把 BN 统计量整体重估后冻结，再在缓存特征上全批量训练 300 步": {
            "用 ImageNet 统计量": mean_and_seeds(rows["imagenet"]), "用训练集重估的统计量": mean_and_seeds(rows["重估"]),
            "掉了多少": r3(np.mean(rows["imagenet"]) - np.mean(rows["重估"]))},
        "全部微调（骨干 1e-5，分类头 1e-3，分类头先训练好）": {"BN 在 eval": mean_and_seeds(ft_eval), "BN 在 train": mean_and_seeds(ft_train),
                                           "掉了多少": r3(np.mean(ft_eval) - np.mean(ft_train))},
        "这组数据和 ImageNet 差多远（第一层卷积输出，64 个通道，200 张测试图）": {
            "逐通道方差除以 ImageNet 的 running_var：中位数": r3(np.median(ratio)), "最小": r3(ratio[0]), "最大": r3(ratio[-1]),
            "比值小于 0.5 的通道个数": int((ratio < 0.5).sum()), "通道总数": int(len(ratio)),
            "逐通道均值偏移除以 running 标准差的中位数": r3(np.median(shift)),
            "归一化之后输入张量的均值": r3(x_in.mean()), "归一化之后输入张量的标准差": r3(x_in.std()),
        },
    }

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.1), gridspec_kw={"width_ratios": [1, 1.5, 0.9]}, layout="constrained")
    ax[0].plot(np.arange(1, len(ratio) + 1), ratio, "o", color=S.BLUE, ms=4)
    ax[0].axhline(1.0, color=S.MUTED, lw=1.2, ls="--")
    ax[0].text(2, 1.25, "和 ImageNet 一样", color=S.MUTED, fontsize=9)
    ax[0].set_yscale("log")
    ax[0].set_yticks([0.01, 0.1, 1], ["0.01", "0.1", "1"])
    ax[0].minorticks_off()
    ax[0].set_ylim(min(0.008, ratio[0] / 1.5), 3)
    ax[0].set_xlabel("第一层卷积的 64 个通道（按比值从小到大排）")
    ax[0].set_ylabel("本数据的方差 / ImageNet 的 running_var")
    ax[0].set_title("这组图的激活起伏比 ImageNet 小得多")
    S.clean(ax[0])

    labels = ["BN eval\n批 4", "BN train\n批 4", "BN eval\n批 10", "BN train\n批 10", "ImageNet\n统计量", "重估的\n统计量"]
    series = [rows["eval,4"], rows["train,4"], rows["eval,10"], rows["train,10"], rows["imagenet"], rows["重估"]]
    cols = [S.BLUE, S.RED, S.BLUE, S.RED, S.BLUE, S.RED]
    xs = [0, 1, 2.4, 3.4, 4.8, 5.8]
    for xi, r, c in zip(xs, series, cols):
        bar_with_seeds(ax[1], [xi], [r], c, width=0.8)
    ax[1].set_xticks(xs, labels, fontsize=9)
    ax[1].set_ylim(0, 1.0)
    ax[1].set_ylabel("测试精度（每类 10 张，黑点是各种子）")
    ax[1].set_title("骨干冻结：BN 用了新数据的统计量（红），精度都更低")
    ax[1].text(0.5, 0.93, "小批量训练 300 张图", ha="center", fontsize=9, color=S.MUTED)
    ax[1].text(2.9, 0.93, "小批量训练 300 张图", ha="center", fontsize=9, color=S.MUTED)
    ax[1].text(5.3, 0.93, "全批量训练 300 步", ha="center", fontsize=9, color=S.MUTED)
    S.clean(ax[1])
    ax[1].grid(axis="x", visible=False)

    bar_with_seeds(ax[2], [0], [ft_eval], S.BLUE, width=0.6)
    bar_with_seeds(ax[2], [1], [ft_train], S.RED, width=0.6)
    ax[2].set_xticks([0, 1], ["BN eval", "BN train"], fontsize=9)
    ax[2].set_ylim(0, 1.0)
    ax[2].set_xlim(-0.6, 1.6)
    ax[2].set_ylabel("测试精度")
    ax[2].set_title("全部微调也一样")
    S.clean(ax[2])
    ax[2].grid(axis="x", visible=False)
    save(fig, "fig04_BatchNorm模式")


# ---------------------------------------------------------------- 图 2：预处理和预训练时不一致
def fig_preprocess():
    te, pre = STORE["test"], STORE["pre"]
    variants = [
        ("和预训练一致", {}),
        ("通道顺序\n写成 BGR", {"bgr": True}),
        ("均值方差\n都用 0.5", {"mean": [0.5] * 3, "std": [0.5] * 3}),
        ("不减均值\n不除方差", {"mean": None}),
        ("像素没有\n除以 255", {"scale": 255.0}),
    ]
    f_te = {name: (te["f"] if not kw else features(pre, to_tensor(te["X"], **kw))) for name, kw in variants}
    test_only = {k: {name: [] for name, _ in variants} for k in (10, 30)}
    both = {name: [] for name, _ in variants}
    recall = {name: [] for name, _ in variants}
    swapped = []   # 推理时写成 BGR：偏红判成偏蓝、偏蓝判成偏红的比例
    for seed in SEEDS:
        tr = STORE["train"][seed]
        for k in (10, 30):
            head = STORE["heads"][(seed, k)]
            for name, _ in variants:
                logits = head(f_te[name]).detach()
                test_only[k][name].append(accuracy(logits, te["Y"]))
                if k == 30:
                    pred = logits.argmax(1).numpy()
                    recall[name].append([float((pred[te["Y"] == c] == c).mean()) for c in range(C)])
                    if name == variants[1][0]:
                        swapped.append([float((pred[te["Y"] == 3] == 4).mean()), float((pred[te["Y"] == 4] == 3).mean())])
        n = 10 * C
        for name, kw in variants:
            f_tr = tr["f"][:n] if not kw else features(pre, to_tensor(tr["X"][:n], **kw))
            both[name].append(accuracy(train_head(f_tr, tr["Y"][:n], seed=seed)(f_te[name]).detach(), te["Y"]))
    key = lambda name: name.replace("\n", "")  # noqa: E731
    bgr = variants[1][0]
    NUM["预处理不一致"] = {
        "说明": DATA_NOTE + f"。预训练骨干冻结，只训练分类头，{len(SEEDS)} 个种子。“只有推理时错”：训练用正确的预处理，测试图换一种预处理；"
              "“训练和推理都这样”：训练图和测试图用同一种（和预训练不一致的）预处理，分类头重新训练。"
              "正确的预处理是：像素除以 255 到 0 到 1，RGB 顺序，减 ImageNet 均值、除以 ImageNet 标准差",
        "只有推理时错，每类 10 张": {key(name): mean_and_seeds(v) for name, v in test_only[10].items()},
        "只有推理时错，每类 30 张": {key(name): mean_and_seeds(v) for name, v in test_only[30].items()},
        "训练和推理都这样，每类 10 张": {key(name): mean_and_seeds(v) for name, v in both.items()},
        "只有推理时错，掉了多少（每类 30 张）": {key(name): r3(np.mean(test_only[30][variants[0][0]]) - np.mean(v)) for name, v in test_only[30].items() if name != variants[0][0]},
        "训练和推理都这样，比正确预处理低多少（每类 10 张）": {key(name): r3(np.mean(both[variants[0][0]]) - np.mean(v)) for name, v in both.items() if name != variants[0][0]},
        "推理时写成 BGR（每类 30 张）：偏红污渍被判成偏蓝污渍的比例": r3(np.mean(swapped, axis=0)[0]),
        "推理时写成 BGR（每类 30 张）：偏蓝污渍被判成偏红污渍的比例": r3(np.mean(swapped, axis=0)[1]),
        "各类召回率（每类 30 张，只有推理时错）": {key(name): dict(zip(CLASSES, [r3(x) for x in np.mean(v, axis=0)])) for name, v in recall.items() if name in (variants[0][0], bgr)},
    }

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained")
    x = np.arange(len(variants))
    for i, (rows, color, label) in enumerate(((test_only[10], S.RED, "只有推理时错"), (both, S.BLUE, "训练和推理都这样"))):
        vals = [np.mean(rows[name]) for name, _ in variants]
        bars = ax[0].bar(x + (i - 0.5) * 0.38, vals, 0.36, color=color, alpha=0.88, label=label)
        for b, v in zip(bars, vals):
            ax[0].text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    ax[0].axhline(1 / C, color=S.MUTED, lw=1, ls=(0, (2, 3)))
    ax[0].set_xticks(x, [name for name, _ in variants], fontsize=9.5)
    ax[0].set_ylim(0, 1.0)
    ax[0].set_ylabel("测试精度（每类 10 张，3 个种子平均）")
    ax[0].set_title("训练和推理不一致掉得多；两边一致地“错”，多数情况掉得少")
    ax[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[0])
    ax[0].grid(axis="x", visible=False)

    xc = np.arange(C)
    for i, (name, color, label) in enumerate(((variants[0][0], S.BLUE, "推理时 RGB（正确）"), (bgr, S.RED, "推理时 BGR"))):
        vals = np.mean(recall[name], axis=0)
        bars = ax[1].bar(xc + (i - 0.5) * 0.38, vals, 0.36, color=color, alpha=0.88, label=label)
        for b, v in zip(bars, vals):
            ax[1].text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.5)
    ax[1].set_xticks(xc, CLASSES, fontsize=9.5)
    ax[1].set_ylim(0, 1.12)
    ax[1].set_ylabel("各类召回率（每类 30 张训练）")
    ax[1].set_title("通道顺序反了，只有靠颜色区分的两类垮掉")
    ax[1].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    S.clean(ax[1])
    ax[1].grid(axis="x", visible=False)
    save(fig, "fig02_预处理不一致")


# ---------------------------------------------------------------- 图 5、图 6：Grad-CAM 的归一化和随机化检查
SIZE_CAM = 128
TOL = 8  # 指向命中的容差（像素）


def gradcam(net, x, target, layers, use_prob=False, bs=50):
    """Grad-CAM：各层的原始热图（ReLU 之后、归一化之前，N x 1 x h x w）和 logits。target 是每张图的目标类别。"""
    out = {name: [] for name in layers}
    logits_all = []
    for i in range(0, len(x), bs):
        acts, hooks = {}, []
        for name, layer in layers.items():
            def fwd(m, inp, o, name=name):
                o.retain_grad()
                acts[name] = o
            hooks.append(layer.register_forward_hook(fwd))
        net.zero_grad()
        logits = net(x[i:i + bs])
        t = target[i:i + bs]
        score = (F.softmax(logits, 1) if use_prob else logits)[torch.arange(len(t)), t].sum()
        score.backward()
        for name in layers:
            a, g = acts[name], acts[name].grad
            w = g.mean(dim=(2, 3), keepdim=True)
            out[name].append((w * a).sum(1, keepdim=True).relu().detach())
        for h in hooks:
            h.remove()
        logits_all.append(logits.detach())
    return {k: torch.cat(v) for k, v in out.items()}, torch.cat(logits_all)


def upsample(cam):
    return F.interpolate(cam, size=(SIZE_CAM, SIZE_CAM), mode="bilinear", align_corners=False)[:, 0].numpy()


def hit_rate(cam, masks):
    """指向命中率：上采样后的热图峰值落在缺陷掩码膨胀 TOL 像素的范围内算命中；全零的热图算没命中。"""
    m = upsample(cam)
    kernel = np.ones((2 * TOL + 1, 2 * TOL + 1), np.uint8)
    hits = []
    for heat, mask in zip(m, masks):
        if heat.max() <= 0:
            hits.append(0)
            continue
        py, px = np.unravel_index(heat.argmax(), heat.shape)
        hits.append(int(cv2.dilate(mask.astype(np.uint8), kernel)[py, px]))
    return float(np.mean(hits))


def rank_corr(a, b):
    """秩相关（并列取平均秩）。有一边全相同就没有定义，返回 None。"""
    def rank(v):
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        return (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    ra, rb = rank(a.ravel()), rank(b.ravel())
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def overlay(ax, img, heat, mask=None):
    ax.imshow(np.round(img * 255).astype(np.uint8), interpolation="nearest")
    ax.imshow(np.round(heat * 255).astype(np.uint8), cmap="jet", alpha=0.5, vmin=0, vmax=255, interpolation="nearest")
    if mask is not None and mask.any():
        ax.contour(mask, levels=[0.5], colors="white", linewidths=1.0)
    S.image_axes(ax)


def fig_gradcam():
    pre = STORE["pre"]
    k = 30
    Xtr, Ytr, _ = make_set(200, k, SIZE_CAM)
    head = train_head(features(pre, to_tensor(Xtr)), Ytr, seed=0)
    net = resnet(True, 0)
    net.fc.load_state_dict(head.state_dict())
    net.eval()
    Xd, Yd, Md = make_set(777, 25, SIZE_CAM, classes=(1, 2, 3, 4))    # 100 张有缺陷
    Xg, Yg, _ = make_set(778, 100, SIZE_CAM, classes=(0,))            # 100 张良品
    Xc, _, _ = make_set(779, 10, SIZE_CAM, classes=(1, 2, 3, 4))      # 40 张有缺陷，只用来定固定范围
    xd, xg, xc = to_tensor(Xd), to_tensor(Xg), to_tensor(Xc)

    def best_defect(x):
        with torch.no_grad():
            lg = torch.cat([net(x[i:i + 50]) for i in range(0, len(x), 50)])
        return lg, lg[:, 1:].argmax(1) + 1

    lg_d, t_d = best_defect(xd)
    lg_g, t_g = best_defect(xg)
    _, t_c = best_defect(xc)
    layers = {"layer4 的输出": net.layer4, "layer4 最后一个 conv2 的输出": net.layer4[-1].conv2, "layer3 的输出": net.layer3}
    cams_d, _ = gradcam(net, xd, t_d, layers)
    raw_d = cams_d["layer4 的输出"]
    raw_g = gradcam(net, xg, t_g, {"layer4": net.layer4})[0]["layer4"]
    raw_c = gradcam(net, xc, t_c, {"layer4": net.layer4})[0]["layer4"]
    # Grad-CAM 在全局平均池化前的那一层上等于 CAM：权重就是分类头的权重除以位置数
    with torch.no_grad():
        h = net.layer4(trunk(net, xd[:20]))
        cam_w = torch.einsum("nk,nkhw->nhw", net.fc.weight[t_d[:20]], h).relu() / (h.shape[2] * h.shape[3])
    cam_diff = float((raw_d[:20, 0] - cam_w).abs().max())

    max_d, max_g, max_c = (r.flatten(1).max(1).values.numpy() for r in (raw_d, raw_g, raw_c))
    vmax = float(np.median(max_c))
    per_d, per_g = upsample(mod.normalize_heatmap(raw_d)), upsample(mod.normalize_heatmap(raw_g))
    fix_d, fix_g = np.clip(upsample(raw_d) / vmax, 0, 1), np.clip(upsample(raw_g) / vmax, 0, 1)
    hot = lambda m: (m >= 0.5).reshape(len(m), -1)  # noqa: E731
    labels = np.r_[np.ones(len(max_d), bool), np.zeros(len(max_g), bool)]
    area_per = np.r_[hot(per_d).mean(1), hot(per_g).mean(1)]
    pred_d, pred_g = lg_d.argmax(1).numpy(), lg_g.argmax(1).numpy()
    conf_g = F.softmax(lg_g, 1).max(1).values.numpy()
    chance = float(np.mean([cv2.dilate(m.astype(np.uint8), np.ones((2 * TOL + 1, 2 * TOL + 1), np.uint8)).mean() for m in Md]))
    hits = {name: hit_rate(c, Md) for name, c in cams_d.items()}
    hits_by_class = {name: {CLASSES[c]: r3(hit_rate(cam[Yd == c], Md[Yd == c])) for c in range(1, C)} for name, cam in cams_d.items()}
    shape = {name: f"{c.shape[2]}x{c.shape[3]}" for name, c in cams_d.items()}

    NUM["热图归一化"] = {
        "说明": f"输入放大到 {SIZE_CAM}x{SIZE_CAM}（缺陷尺寸同比例放大），这时 layer4 的输出是 {shape['layer4 的输出']}。模型是预训练骨干加线性分类头（每类 {k} 张训练，只训练分类头）。"
              "对每张图取“最像的那一类缺陷”（4 个缺陷类里 logit 最大的）做 Grad-CAM，目标层是 layer4 的输出。测试用 100 张有缺陷的图（4 类各 25 张）和 100 张良品。"
              "单张归一化用本章作业的 normalize_heatmap（先 ReLU，再按这一张图的最小最大值拉到 0 到 1）；"
              "固定范围是除以一个固定的上限再截断到 0 到 1，上限取另外 40 张有缺陷的图原始热图最大值的中位数。“有热点”指上采样后的热图里有像素不小于 0.5",
        "这个模型在这 200 张图上的 5 类精度": r3(np.mean(np.r_[pred_d == Yd, pred_g == 0])),
        "良品被判成某一类缺陷的比例": r3(np.mean(pred_g != 0)),
        "原始热图最大值的中位数": {"有缺陷的图": r3(np.median(max_d)), "良品": r3(np.median(max_g)), "两者之比": r3(np.median(max_d) / np.median(max_g))},
        "原始热图最大值的 5% 到 95% 分位": {"有缺陷的图": [r3(v) for v in np.quantile(max_d, [0.05, 0.95])], "良品": [r3(v) for v in np.quantile(max_g, [0.05, 0.95])]},
        "固定范围的上限": r3(vmax),
        "ReLU 之后整张热图全是 0 的比例": {"有缺陷的图": r3(np.mean(max_d == 0)), "良品": r3(np.mean(max_g == 0))},
        "有热点的图所占比例": {
            "单张归一化": {"有缺陷的图": r3(hot(per_d).any(1).mean()), "良品": r3(hot(per_g).any(1).mean())},
            "固定范围": {"有缺陷的图": r3(hot(fix_d).any(1).mean()), "良品": r3(hot(fix_g).any(1).mean())},
        },
        "单张归一化后热点面积占整张图的比例（中位数）": {"有缺陷的图": r3(np.median(hot(per_d).mean(1))), "良品": r3(np.median(hot(per_g).mean(1)))},
        "只凭热图分辨“有没有缺陷”的 AUROC": {
            "用原始热图的最大值": r3(auroc(np.r_[max_d, max_g], labels)),
            "用单张归一化后的热点面积（面积大算有缺陷）": r3(auroc(area_per, labels)),
        },
        "良品图上分类器的最大 softmax 概率：中位数": r3(np.median(conf_g)),
        "Grad-CAM（layer4 的输出）和 CAM（分类头权重加权）的最大差": float(f"{cam_diff:.2g}"),
    }
    NUM["目标层与指向命中率"] = {
        "说明": f"同一个模型、同样 100 张有缺陷的图。指向命中：把热图双线性上采样到 {SIZE_CAM}x{SIZE_CAM}，峰值落在缺陷掩码向外扩 {TOL} px 的范围内算命中。"
              "随机落点的命中率是这个范围占整张图面积的平均比例",
        "热图分辨率": shape,
        "指向命中率": {name: r3(v) for name, v in hits.items()},
        "分缺陷类别的指向命中率": hits_by_class,
        "随机落点的命中率": r3(chance),
    }

    # 用 softmax 概率而不是 logit 反传
    cam_p = gradcam(net, xd, t_d, {"layer4": net.layer4}, use_prob=True)[0]["layer4"]
    max_p = cam_p.flatten(1).max(1).values.numpy()
    conf_d = F.softmax(lg_d, 1).max(1).values.numpy()
    NUM["用 logit 还是用概率反传"] = {
        "说明": "同样 100 张有缺陷的图、目标层 layer4 的输出。把反传的起点从目标类别的 logit 换成 softmax 之后的概率",
        "原始热图最大值的中位数": {"logit": r3(np.median(max_d)), "概率": float(f"{np.median(max_p):.2g}"), "logit 是概率的多少倍": float(f"{np.median(max_d) / np.median(max_p):.2g}")},
        "指向命中率": {"logit": r3(hits["layer4 的输出"]), "概率": r3(hit_rate(cam_p, Md))},
        "这些图上最大 softmax 概率的中位数": r3(np.median(conf_d)),
    }

    # 讲义里的 hook 写法能不能跑；full backward hook 挂在后面紧跟原地操作的层上会报错
    hook_ok = {}
    for name, layer in (("layer4[-1].conv2（讲义原代码）", net.layer4[-1].conv2), ("layer4[-1]（整个残差块）", net.layer4[-1]), ("layer4[-1].bn2", net.layer4[-1].bn2)):
        handle = layer.register_full_backward_hook(lambda m, gi, go: None)
        try:
            net.zero_grad()
            net(xd[:1])[0, 1].backward()
            hook_ok[name] = "正常"
        except RuntimeError as exc:
            hook_ok[name] = "RuntimeError：" + str(exc).split(".")[0]
        handle.remove()
    NUM["register_full_backward_hook 挂在哪一层"] = {"说明": f"torch {torch.__version__.split('+')[0]}，torchvision 的 ResNet-18。挂上之后做一次前向和反向", **hook_ok}

    # 随机化检查：从分类头往前逐级把权重换成随机初始化，热图应该跟着变
    levels = [("原模型", []), ("随机化分类头", ["fc"]), ("再随机化 layer4", ["fc", "layer4"]), ("全部随机化", None)]
    rand = {}
    example_maps = {}
    ex = int(np.where(Yd == 3)[0][0])
    base = raw_d[:, 0].numpy()

    def same_cell(cam):
        peak = cam.flatten(1).argmax(1).numpy()
        return float(np.bincount(peak, minlength=cam.shape[2] * cam.shape[3]).max() / len(peak))
    for name, tops in levels[1:]:
        net2 = resnet(True, 0)
        net2.fc.load_state_dict(head.state_dict())
        torch.manual_seed(7)
        for mname, m in net2.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)) and (tops is None or mname.split(".")[0] in tops):
                m.reset_parameters()
        net2.eval()
        cam2 = gradcam(net2, xd, t_d, {"layer4": net2.layer4})[0]["layer4"]
        rho = [rank_corr(base[i], cam2[i, 0].numpy()) for i in range(len(base))]
        valid = [v for v in rho if v is not None]
        per2 = upsample(mod.normalize_heatmap(cam2))
        rand[name] = {"和原模型热图的秩相关（均值）": r3(np.mean(valid)), "秩相关（中位数）": r3(np.median(valid)), "算不出秩相关的图（热图全零）所占比例": r3(1 - len(valid) / len(rho)),
                      "指向命中率": r3(hit_rate(cam2, Md)), "单张归一化后有热点的比例": r3(hot(per2).any(1).mean()),
                      "峰值落在最常见的那个格子里的图所占比例": r3(same_cell(cam2))}
        example_maps[name] = per2[ex]
    NUM["随机化检查"] = {
        "说明": f"同样 100 张有缺陷的图，目标类别保持原模型选的那一类，目标层 layer4 的输出。从分类头开始往前，把权重换成随机初始化（reset_parameters，BatchNorm 的统计量也重置），"
              f"再算 Grad-CAM，和原模型的热图比。秩相关在 {shape['layer4 的输出']} 的原始热图（16 个值）上算，并列取平均秩",
        "原模型的指向命中率": r3(hits["layer4 的输出"]),
        "随机落点的命中率": r3(chance),
        "原模型：峰值落在最常见的那个格子里的图所占比例": r3(same_cell(raw_d)),
        "16 个格子均匀落点时，一个格子的占比": r3(1 / (raw_d.shape[2] * raw_d.shape[3])),
        **rand,
    }

    # 图 5
    fig = plt.figure(figsize=(15.5, 6.2), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[1.75, 1])
    gs = left.add_gridspec(3, 7, width_ratios=[1, 1, 1, 0.12, 1, 1, 1])
    d_idx = [int(np.where(Yd == c)[0][0]) for c in (1, 2, 3)]
    g_idx = [0, 1, 2]
    row_names = ["输入（白圈是缺陷）", "单张归一化", f"固定范围（上限 {vmax:.2f}）"]
    for r in range(3):
        for j, i in enumerate(d_idx):
            ax = left.add_subplot(gs[r, j])
            if r == 0:
                ax.imshow(np.round(Xd[i] * 255).astype(np.uint8), interpolation="nearest")
                ax.contour(Md[i], levels=[0.5], colors="white", linewidths=1.0)
                S.image_axes(ax)
                ax.set_title(f"有缺陷：{CLASSES[Yd[i]]}", fontsize=10)
            else:
                overlay(ax, Xd[i], per_d[i] if r == 1 else fix_d[i], Md[i])
            if j == 0:
                ax.set_ylabel(row_names[r], fontsize=10)
        for j, i in enumerate(g_idx):
            ax = left.add_subplot(gs[r, 4 + j])
            if r == 0:
                ax.imshow(np.round(Xg[i] * 255).astype(np.uint8), interpolation="nearest")
                S.image_axes(ax)
                ax.set_title("良品", fontsize=10)
            else:
                overlay(ax, Xg[i], per_g[i] if r == 1 else fix_g[i])
    axr = right.subplots(2, 1, gridspec_kw={"height_ratios": [1.7, 1]})
    ax = axr[0]
    bins = np.linspace(0, max(max_d.max(), max_g.max()) * 1.02, 30)
    ax.hist(max_g, bins=bins, color=S.BLUE, alpha=0.75, label="良品（100 张）")
    ax.hist(max_d, bins=bins, color=S.ORANGE, alpha=0.75, label="有缺陷（100 张）")
    ax.axvline(0.5 * vmax, color=S.RED, lw=1.4, ls="--")
    ax.text(0.5 * vmax, ax.get_ylim()[1] * 0.97, " 固定范围下\n 显示成 0.5 的位置", color=S.RED, fontsize=9, va="top")
    ax.set_xlabel("原始热图的最大值（归一化之前）")
    ax.set_ylabel("图的张数")
    ax.set_title("良品的原始热图本来弱，单张归一化把它拉满")
    ax.legend(fontsize=9, loc="center right")
    S.clean(ax)
    ax = axr[1]
    vals = [hot(per_d).any(1).mean(), hot(per_g).any(1).mean(), hot(fix_d).any(1).mean(), hot(fix_g).any(1).mean()]
    bars = ax.barh([3, 2, 1, 0], vals, 0.62, color=[S.ORANGE, S.BLUE, S.ORANGE, S.BLUE], alpha=0.88)
    for b, v in zip(bars, vals):
        ax.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.0%}", va="center", fontsize=9)
    ax.set_yticks([3, 2, 1, 0], ["单张归一化：有缺陷", "单张归一化：良品", "固定范围：有缺陷", "固定范围：良品"], fontsize=9)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("出现热点（有像素不小于 0.5）的图所占比例")
    S.clean(ax)
    ax.grid(axis="y", visible=False)
    save(fig, "fig05_热图归一化")

    # 图 6
    fig = plt.figure(figsize=(16, 3.6), layout="constrained")
    gs = fig.add_gridspec(1, 7, width_ratios=[1.15, 1.15, 1.15, 1.15, 0.12, 1.9, 1.6])
    example_maps = {"原模型": per_d[ex], **example_maps}
    for j, (name, _) in enumerate(levels):
        ax = fig.add_subplot(gs[0, j])
        overlay(ax, Xd[ex], example_maps[name], Md[ex])
        ax.set_title(name, fontsize=10)
    ax = fig.add_subplot(gs[0, 5])
    names = [n for n, _ in levels[1:]]
    x = np.arange(len(names))
    v1 = [rand[n]["和原模型热图的秩相关（均值）"] for n in names]
    v2 = [rand[n]["指向命中率"] for n in names]
    for i, (vals, color, label) in enumerate(((v1, S.PURPLE, "和原模型热图的秩相关"), (v2, S.GREEN, "指向命中率"))):
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.36, color=color, alpha=0.88, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(hits["layer4 的输出"], color=S.GREEN, lw=1.2, ls="--")
    ax.text(len(names) - 0.5, hits["layer4 的输出"] + 0.015, f"原模型的命中率 {hits['layer4 的输出']:.2f}", color=S.GREEN, fontsize=9, ha="right")
    ax.axhline(0, color=S.MUTED, lw=0.8)
    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylim(min(-0.1, min(v1) - 0.08), 1.0)
    ax.set_title("随机化到 layer4，热图不再像原来的，命中率到 0")
    ax.legend(fontsize=9, loc="upper right")
    S.clean(ax)
    ax.grid(axis="x", visible=False)
    ax = fig.add_subplot(gs[0, 6])
    order = ["layer4 最后一个 conv2 的输出", "layer4 的输出", "layer3 的输出"]
    short = [f"layer4[-1].conv2\n{shape[order[0]]}", f"layer4 的输出\n{shape[order[1]]}", f"layer3 的输出\n{shape[order[2]]}"]
    bars = ax.bar(np.arange(3), [hits[n] for n in order], 0.6, color=[S.MUTED, S.BLUE, S.ORANGE], alpha=0.88)
    for b, n in zip(bars, order):
        ax.text(b.get_x() + b.get_width() / 2, hits[n] + 0.015, f"{hits[n]:.2f}", ha="center", fontsize=9)
    ax.axhline(chance, color=S.MUTED, lw=1, ls=(0, (2, 3)))
    ax.text(-0.45, chance + 0.6, f"虚线：随机落点 {chance:.2f}", color=S.MUTED, fontsize=9, ha="left")
    ax.set_xticks(np.arange(3), short, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("指向命中率")
    ax.set_title("目标层换成 layer3，命中率更高")
    S.clean(ax)
    ax.grid(axis="x", visible=False)
    save(fig, "fig06_随机化检查")


# ---------------------------------------------------------------- 没有配图的数字
def numbers_misc():
    te, pre = STORE["test"], STORE["pre"]
    # 数据增强：翻转不改变标签；红蓝通道互换会把偏红污渍变成偏蓝污渍
    aug = {}
    for k in (3, 10):
        n = k * C
        rows = {"不增强": [], "加三种翻转": [], "加红蓝通道互换": []}
        red_blue = {name: [] for name in rows}
        for seed in SEEDS:
            tr = STORE["train"][seed]
            X, y = tr["X"][:n], tr["Y"][:n]
            flips = np.concatenate([X[:, :, ::-1], X[:, ::-1], X[:, ::-1, ::-1]])
            f_flip = features(pre, to_tensor(flips))
            f_swap = features(pre, to_tensor(X[..., ::-1]))
            sets = {"不增强": (tr["f"][:n], y), "加三种翻转": (torch.cat([tr["f"][:n], f_flip]), np.tile(y, 4)), "加红蓝通道互换": (torch.cat([tr["f"][:n], f_swap]), np.tile(y, 2))}
            for name, (f, yy) in sets.items():
                pred = train_head(f, yy, seed=seed)(te["f"]).detach().argmax(1).numpy()
                rows[name].append(float((pred == te["Y"]).mean()))
                idx = (te["Y"] == 3) | (te["Y"] == 4)
                red_blue[name].append(float((pred[idx] == te["Y"][idx]).mean()))
        aug[f"每类 {k} 张"] = {name: {"测试精度": mean_and_seeds(v), "两类污渍的召回率": r3(np.mean(red_blue[name]))} for name, v in rows.items()}
    NUM["数据增强"] = {
        "说明": DATA_NOTE + "。预训练骨干冻结，只训练分类头（全批量 300 步），3 个种子。“加三种翻转”把每张训练图的左右翻转、上下翻转、旋转 180 度都加进训练集（4 倍数据），"
              "这组数据里缺陷的方向和位置都是随机的，翻转不改变标签。“加红蓝通道互换”把每张训练图交换 R 和 B 通道后按原标签加进训练集（2 倍数据），偏红污渍会因此变成偏蓝污渍，标签就错了",
        **aug,
    }

    # 置信度：分箱的正确率，以及训练时没有的类别
    conf_rows = {}
    rng = np.random.default_rng(555)
    ring = np.stack([make_image(rng, 5)[0] for _ in range(100)])
    f_ring = features(pre, to_tensor(ring))
    for k in (10, 30):
        conf, correct, ood = [], [], []
        for seed in SEEDS:
            head = STORE["heads"][(seed, k)]
            p = F.softmax(head(te["f"]).detach(), 1)
            conf.append(p.max(1).values.numpy())
            correct.append(p.argmax(1).numpy() == te["Y"])
            ood.append(F.softmax(head(f_ring).detach(), 1).numpy())
        conf, correct, ood = np.concatenate(conf), np.concatenate(correct), np.concatenate(ood)
        hi = conf >= 0.9
        conf_rows[f"每类 {k} 张"] = {
            "测试精度": r3(correct.mean()), "平均置信度（最大 softmax 概率）": r3(conf.mean()),
            "置信度不低于 0.9 的预测所占比例": r3(hi.mean()), "这些预测里实际正确的比例": r3(correct[hi].mean()),
            "训练时没有的环形缺陷：平均置信度": r3(ood.max(1).mean()), "环形缺陷：置信度不低于 0.9 的比例": r3((ood.max(1) >= 0.9).mean()),
            "环形缺陷被分到各类的比例": dict(zip(CLASSES, [r3(v) for v in np.bincount(ood.argmax(1), minlength=C) / len(ood)])),
        }
    NUM["置信度"] = {
        "说明": DATA_NOTE + "。预训练骨干冻结，只训练分类头，3 个种子的预测合在一起统计（600 个预测）。环形缺陷是另外生成的 100 张亮圆环，训练集里没有这一类，分类器只能在已有的 5 类里选一个",
        **conf_rows,
    }

    # 10.7 节的例子：80 张杯子、10 张手机、10 张书，模型永远猜杯子
    counts = np.array([80, 10, 10])
    tp = np.array([80, 0, 0])
    pred_count = np.array([100, 0, 0])
    recall = tp / counts
    precision = np.divide(tp, pred_count, out=np.zeros(3), where=pred_count > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=(precision + recall) > 0)
    NUM["永远猜大类的例子"] = {
        "说明": "100 张验证图：80 张杯子、10 张手机、10 张书，模型把所有图都判成杯子。没有被预测到的类别，精确率按 0 算",
        "准确率": r3(tp.sum() / counts.sum()), "各类召回率": [r3(v) for v in recall], "宏平均召回率": r3(recall.mean()),
        "杯子的精确率": r3(precision[0]), "杯子的 F1": r3(f1[0]), "宏平均 F1": r3(f1.mean()),
    }

    # 改了 requires_grad 之后不重建优化器会怎样
    tr = STORE["train"][0]
    xb, yb = tr["x"][:BS], torch.from_numpy(tr["Y"][:BS])

    def steps(net, opt, n, set_to_none=True):
        set_mode(net, True)
        for _ in range(n):
            opt.zero_grad(set_to_none=set_to_none)
            F.cross_entropy(net(xb), yb).backward()
            opt.step()

    def rel_change(net, before, prefix):
        a = torch.cat([p.detach().flatten() for n, p in net.named_parameters() if n.startswith(prefix)])
        return float((a - before).norm() / before.norm())

    def snapshot(net, prefix):
        return torch.cat([p.detach().flatten().clone() for n, p in net.named_parameters() if n.startswith(prefix)])

    net = resnet(True, 0)
    mod.configure_trainable_parameters(net, ("fc",))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)   # 把冻结的参数也传给优化器
    before = snapshot(net, "layer")
    steps(net, opt, 3)
    passed_all = {"传给优化器的参数张量个数": len(opt.param_groups[0]["params"]), "3 步之后优化器里有状态的参数张量个数": len(opt.state),
                  "冻结的 layer1 到 layer4 的相对变化": rel_change(net, before, "layer")}
    stale = {}
    for name, set_to_none in (("zero_grad() 的默认行为（梯度置为 None）", True), ("zero_grad(set_to_none=False)（梯度清成 0）", False)):
        net = resnet(True, 0)
        mod.configure_trainable_parameters(net, ALL_PREFIXES)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
        steps(net, opt, 3, set_to_none)
        mod.configure_trainable_parameters(net, ("fc",))                      # 中途冻结骨干，不重建优化器
        before = snapshot(net, "layer")
        steps(net, opt, 3, set_to_none)
        stale[name] = float(f"{rel_change(net, before, 'layer'):.2g}")
    net = resnet(True, 0)
    mod.configure_trainable_parameters(net, ("fc",))
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    mod.configure_trainable_parameters(net, ("layer4", "fc"))                 # 解冻 layer4，不重建优化器
    before = snapshot(net, "layer4")
    steps(net, opt, 3)
    missed = {"layer4 的参数有没有梯度": bool(all(p.grad is not None for p in net.layer4.parameters())), "3 步之后 layer4 的相对变化": rel_change(net, before, "layer4")}
    seq = nn.Sequential(*[nn.Linear(2, 2) for _ in range(12)])
    matched = mod.configure_trainable_parameters(seq, ("1",))
    NUM["优化器与冻结"] = {
        "说明": f"torch {torch.__version__.split('+')[0]}，ResNet-18，AdamW（学习率 1e-4，权重衰减 1e-4），同一个批（10 张图）连续训练 3 步，BN 在 eval。相对变化是参数之差的范数除以原来的范数",
        "骨干冻结，但把全部参数都传给优化器": passed_all,
        "先全部训练 3 步，再冻结骨干且不重建优化器，又训练 3 步：layer1 到 layer4 的相对变化": stale,
        "优化器只有分类头，之后解冻 layer4 且不重建优化器": missed,
        "按前缀匹配的一个坑：12 层的 nn.Sequential，前缀写 \"1\"，被解冻的参数": matched,
    }

    NUM["原讲义保留的举例数字"] = {
        "说明": "这些数是原讲义举例时用的假设值，不是实测结果。放在这里只是为了讲义里出现的数都能查到出处",
        "10.6 节错误分析表里两行的置信度": [0.72, 0.95],
        "10.7 节 softmax 最大值的例子": 0.99,
        "10.7 节 0.8 到 0.9 这一箱实际的正确率": 0.6,
        "Flowers102 的类别数": 102,
        "10.7 节按切片评估的例子：整体、明亮场景、暗光场景": [0.92, 0.98, 0.55],
        "10.9 节小样本过拟合测试的样本数": [16, 32],
    }

    def timing():
        x = STORE["train"][0]["x"][:BS]
        y = torch.from_numpy(STORE["train"][0]["Y"][:BS])
        net = resnet(True, 0)

        def per_image(step, repeat=5):
            step()
            t = time.perf_counter()
            for _ in range(repeat):
                step()
            return (time.perf_counter() - t) / repeat / BS * 1e3

        opt = torch.optim.AdamW(net.parameters(), lr=1e-5)

        def full_step():
            opt.zero_grad()
            F.cross_entropy(net(x), y).backward()
            opt.step()

        set_mode(net, True)
        t_full = per_image(full_step)
        net.eval()
        with torch.no_grad():
            t_fwd = per_image(lambda: net(x))
        return {"说明": f"在一台 Apple 芯片的 Mac 上用 CPU（2 个线程）测的一次结果，ResNet-18，输入 64x64，批大小 {BS}，每张图的毫秒数，保留两位有效数字。"
                      "机器忙的时候会更慢，只看两者的比例",
                "只做前向（冻结骨干提特征）每张图毫秒": float(f"{t_fwd:.2g}"), "前向加反向加更新（全部微调）每张图毫秒": float(f"{t_full:.2g}"),
                "全部微调是只做前向的多少倍": float(f"{t_full / t_fwd:.2g}")}

    NUM["耗时"] = S.pinned(CH, "耗时", timing, "CH10_RETIME")


def main() -> None:
    S.setup()
    t_all = time.perf_counter()
    for fn in (prepare, fig_samples_vs_k, fig_learning_rate, fig_batchnorm, fig_preprocess, fig_gradcam, numbers_misc):
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
