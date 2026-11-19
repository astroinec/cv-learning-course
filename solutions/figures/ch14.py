"""第 14 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch14.py

输出：<主分支>/14_DL综合项目/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。

实验分两类：
1. 代码生成的“桌面物体”小任务（120x160 的 RGB 图，8 类），在 CPU 上训练一个带 BatchNorm 和 Dropout 的小卷积网络；
2. torchvision 的 ResNet-18（ImageNet 权重，本机有缓存），只做前向。
letterbox 的元数据和检测框回映射调用本章作业的参考实现。

会波动的测量（耗时）用 _style.pinned 钉住：默认沿用 实测数字.json 里上次的值，设置 CH14_RETIME=1 才重测。
本机没有 onnx 和 onnxruntime，脚本不导出也不运行 ONNX。
"""

from __future__ import annotations

import copy
import io
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import warnings
import zlib
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

CH = "14_DL综合项目"
mod = S.load_solution(CH, "作业_预处理与部署契约.py")
NUM: dict = {}
RETIME = "CH14_RETIME"

SRC_HW = (120, 160)
DST = 48
PAD = 114
CLASSES = ["空桌面", "红杯", "蓝杯", "方便签", "横卡片", "竖卡片", "键盘", "鼠标垫"]
K = len(CLASSES)
SEEDS = (0, 1, 2)
IMN_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMN_STD = np.array([0.229, 0.224, 0.225], np.float32)
TASK_NOTE = (f"代码生成的桌面物体图，{SRC_HW[0]}x{SRC_HW[1]}（高x宽）的 8 位 RGB，{K} 类：" + "、".join(CLASSES)
             + "。红杯和蓝杯只差颜色，方便签（1:1）、横卡片（3:2）、竖卡片（2:3）只差长宽比，键盘和鼠标垫只差有没有细格纹。"
             "每张图的桌面灰度、白平衡、亮度、光照方向和噪声都随机")


def r3(v):
    return round(float(v), 3)


def r4(v):
    return round(float(v), 4)


def sig(v, n=2):
    """保留 n 位有效数字。耗时一律存两位。"""
    return float(f"{float(v):.{n}g}")


def save(fig, name):
    """先让 constrained 布局在 SVG 渲染器下算完，把坐标轴位置取整后固定，再保存。

    布局求解器给出的位置在不同进程里可能差最后一两位浮点数（打印到 SVG 里的 6 位小数相同），
    而 SVG 的裁剪编号是按没有取整的位置算哈希的，长脚本连跑两遍时个别图会因此不能逐字节相同。
    """
    fig.savefig(io.BytesIO(), format="svg")
    fig.set_layout_engine("none")
    for ax in fig.axes:
        ax.set_position([round(v, 5) for v in ax.get_position().bounds])
    return S.save(fig, CH, name)


# ---------------------------------------------------------------- 合成数据
def make_env(rng):
    g = rng.uniform(95, 195)
    return {
        "base": np.clip(g + rng.uniform(-18, 18, 3), 0, 255),
        "tint": rng.uniform(0.88, 1.12, 3),
        "gain": rng.uniform(0.75, 1.15),
        "ang": rng.uniform(0, 2 * np.pi),
        "ramp": rng.uniform(0.05, 0.22),
        "grain": rng.uniform(2, 7),
        "noise": rng.uniform(2, 5),
    }


def sample_obj(rng, cls):
    H, W = SRC_HW
    if cls == 0:
        return None
    if cls in (1, 2):
        r = rng.uniform(16, 26)
        w = h = 2 * r
    elif cls == 3:
        w = h = rng.uniform(36, 52)
    elif cls == 4:
        u = rng.uniform(15, 20)
        w, h = 3 * u, 2 * u
    elif cls == 5:
        u = rng.uniform(15, 20)
        w, h = 2 * u, 3 * u
    else:
        h = rng.uniform(30, 40)
        w = 2 * h
    x0 = rng.uniform(4, W - 4 - w)
    y0 = rng.uniform(4, H - 4 - h)
    return {"box": (x0, y0, x0 + w, y0 + h), "jit": rng.uniform(-14, 14, 3)}


def render(rng, cls, env=None, obj="auto", dxy=(0.0, 0.0)):
    """一张桌面图。返回 (uint8 RGB HxWx3, 物体的框 xyxy 或 None)。"""
    H, W = SRC_HW
    env = env or make_env(rng)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    img = np.empty((H, W, 3))
    img[:] = env["base"]
    grain = cv2.GaussianBlur(rng.normal(size=(H, W)), (0, 0), sigmaX=6, sigmaY=1.2)
    img += (env["grain"] * grain / (grain.std() + 1e-9))[..., None]
    if isinstance(obj, str):
        obj = sample_obj(rng, cls)
    box = None
    if obj is not None:
        x0, y0, x1, y1 = obj["box"]
        x0, x1, y0, y1 = x0 + dxy[0], x1 + dxy[0], y0 + dxy[1], y1 + dxy[1]
        box = (x0, y0, x1, y1)
        cx, cy, w = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0
        if cls in (1, 2):
            d = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy)
            m = np.clip(w / 2 - d + 0.5, 0, 1)
            col = np.array([200, 45, 40.0] if cls == 1 else [40, 60, 200.0]) + obj["jit"]
            inner = np.clip(w / 2 * 0.72 - d + 0.5, 0, 1)
            layer = col[None, None] * (1 - 0.35 * inner[..., None])
        else:
            m = np.clip(np.minimum(np.minimum(xx + 1 - x0, x1 - xx), np.minimum(yy + 1 - y0, y1 - yy)), 0, 1)
            if cls in (3, 4, 5):
                layer = np.broadcast_to(np.array([228, 198, 62.0]) + obj["jit"], (H, W, 3)).copy()
            else:
                layer = np.broadcast_to(np.array([72, 72, 78.0]) + obj["jit"] * 0.5, (H, W, 3)).copy()
                if cls == 6:
                    layer[(((xx - x0) % 8) < 2) | (((yy - y0) % 8) < 2)] += 85
        img = img * (1 - m[..., None]) + layer * m[..., None]
    ramp = 1 + env["ramp"] * ((xx - W / 2) * np.cos(env["ang"]) + (yy - H / 2) * np.sin(env["ang"])) / (W / 2)
    img = img * ramp[..., None] * env["gain"] * env["tint"][None, None]
    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    img += rng.normal(scale=env["noise"], size=img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8), box


def make_set(seed, n, classes=None):
    rng = np.random.default_rng(seed)
    classes = list(range(K)) if classes is None else classes
    labels = np.array([classes[i % len(classes)] for i in range(n)])
    imgs, boxes = [], []
    for c in labels:
        im, b = render(rng, int(c))
        imgs.append(im)
        boxes.append(b)
    return imgs, labels, boxes


# ---------------------------------------------------------------- 预处理
CV_FLAGS = {"linear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST, "area": cv2.INTER_AREA, "cubic": cv2.INTER_CUBIC}


def resize(img, wh, how="pil"):
    """how='pil' 是 PIL 的双线性（缩小时自带抗锯齿，torchvision 对 PIL 图的默认行为）；其余是 cv2.resize 的四种标志。"""
    if how == "pil":
        return np.asarray(Image.fromarray(img).resize(wh, Image.BILINEAR))
    return cv2.resize(img, wh, interpolation=CV_FLAGS[how])


def letterbox(img, dst=DST, how="pil", pad=PAD):
    meta = mod.letterbox_metadata(img.shape[:2], (dst, dst))
    nh, nw = meta["resized_hw"]
    px, py = meta["pad_xy"]
    out = np.full((dst, dst, 3), pad, np.uint8)
    out[py:py + nh, px:px + nw] = resize(img, (nw, nh), how)
    return out, meta


def stretch(img, dst=DST, how="pil"):
    return resize(img, (dst, dst), how)


def to_tensor(canvases, mean, std, order="rgb", scale=255.0, layout="transpose", normalize=True):
    x = np.stack(canvases).astype(np.float32)
    if order == "bgr":
        x = x[..., ::-1]
    x = x / np.float32(scale)
    if normalize:
        x = (x - mean) / std
    x = x.transpose(0, 3, 1, 2) if layout == "transpose" else x.reshape(len(x), 3, x.shape[1], x.shape[2])
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def channel_stats(imgs):
    arr = np.stack([letterbox(i)[0] for i in imgs]).astype(np.float32) / 255
    return arr.reshape(-1, 3).mean(0), arr.reshape(-1, 3).std(0)


# ---------------------------------------------------------------- 小模型
def conv_block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU())


def backbone():
    return nn.Sequential(conv_block(3, 16), nn.MaxPool2d(2), conv_block(16, 32), nn.MaxPool2d(2),
                         conv_block(32, 64), nn.MaxPool2d(2), conv_block(64, 64))


class SmallNet(nn.Module):
    """4 个 3x3 卷积块（每块 Conv-BN-ReLU），全局平均池化，Dropout，全连接。"""

    def __init__(self, out=K, p=0.3):
        super().__init__()
        self.features = backbone()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(64, out)

    def forward(self, x):
        return self.fc(self.drop(self.pool(self.features(x)).flatten(1)))


class BoxNet(nn.Module):
    """同样的卷积主干，后面接展平和两层全连接，回归一个框（画布上的 xyxy 除以画布边长）。"""

    def __init__(self):
        super().__init__()
        self.features = backbone()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 6 * 6, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        return self.head(self.features(x))


def train(net, x, y, seed, epochs, lr=4e-3, loss="ce"):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    steps = epochs * ((len(x) + 63) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), 64):
            idx = perm[i:i + 64]
            out = net(x[idx])
            val = nn.functional.cross_entropy(out, y[idx]) if loss == "ce" else nn.functional.smooth_l1_loss(out, y[idx], beta=0.02)
            opt.zero_grad()
            val.backward()
            opt.step()
            sched.step()
    return net.eval()


def new_classifier(seed):
    torch.manual_seed(seed)
    return SmallNet()


@torch.inference_mode()
def predict(net, x, chunk=256):
    return torch.cat([net(x[i:i + chunk]) for i in range(0, len(x), chunk)])


def accuracy(logits, y):
    return float((logits.argmax(1) == y).float().mean())


def count_cost(net, x):
    """参数量，以及一次前向里卷积和全连接的乘加次数（一个输出值对应一次“核大小 x 输入通道数”的乘加）。"""
    macs = [0]

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            macs[0] += out[0].numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        else:
            macs[0] += out[0].numel() * m.in_features

    hs = [m.register_forward_hook(hook) for m in net.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.inference_mode():
        net(x)
    for h in hs:
        h.remove()
    return sum(p.numel() for p in net.parameters()), macs[0]


_CTX: dict = {}


def ctx():
    """小任务的数据和 3 个种子的分类模型，多组实验共用。"""
    if _CTX:
        return _CTX
    tr_imgs, tr_y, _ = make_set(100, 1280)
    te_imgs, te_y, _ = make_set(200, 1200)
    mean, std = channel_stats(tr_imgs)
    xtr = to_tensor([letterbox(i)[0] for i in tr_imgs], mean, std)
    xte = to_tensor([letterbox(i)[0] for i in te_imgs], mean, std)
    ytr, yte = torch.from_numpy(tr_y), torch.from_numpy(te_y)
    nets = [train(new_classifier(s), xtr, ytr, s, epochs=12) for s in SEEDS]
    _CTX.update(tr_imgs=tr_imgs, te_imgs=te_imgs, mean=mean, std=std, xtr=xtr, xte=xte, ytr=ytr, yte=yte, nets=nets)
    params, macs = count_cost(nets[0], xte[:1])
    NUM["数据与小模型"] = {
        "说明": TASK_NOTE + f"。训练集 {len(tr_imgs)} 张，测试集 {len(te_imgs)} 张，各类张数相同。"
                f"训练时的预处理契约：RGB，letterbox 到 {DST}x{DST}（PIL 双线性缩小，加边值 {PAD}），除以 255，减训练集的通道均值、除以通道标准差，HWC 转成 CHW。"
                "模型是 4 个 Conv-BN-ReLU 块（16、32、64、64 通道）加全局平均池化、Dropout(0.3) 和全连接；Adam，学习率单周期调度（最大 0.004），批大小 64，12 轮；3 个随机种子",
        "letterbox 元数据（参考实现）": {"scale": mod.letterbox_metadata(SRC_HW, (DST, DST))["scale"],
                                "resized_hw": list(mod.letterbox_metadata(SRC_HW, (DST, DST))["resized_hw"]),
                                "pad_xy": list(mod.letterbox_metadata(SRC_HW, (DST, DST))["pad_xy"])},
        "原图尺寸（高, 宽）": list(SRC_HW), "模型输入边长": DST, "训练集张数": len(tr_imgs), "测试集张数": len(te_imgs),
        "训练集通道均值（除以 255 之后）": [r3(v) for v in mean],
        "训练集通道标准差": [r3(v) for v in std],
        "参数量": params,
        "一次前向的乘加次数（百万）": r3(macs / 1e6),
        "预处理一致时的测试精度（各种子）": [r3(accuracy(predict(n, xte), yte)) for n in nets],
    }
    return _CTX


# ---------------------------------------------------------------- ResNet-18 和它用的图
_RN: dict = {}


def resnet():
    if not _RN:
        W = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        _RN["net"] = torchvision.models.resnet18(weights=W).eval()
        _RN["weights"] = W
        _RN["photo"] = np.asarray(Image.open(os.path.join(matplotlib.get_data_path(), "sample_data", "grace_hopper.jpg")).convert("RGB"))
    return _RN["net"]


def image_sets(seed, n_syn, n_real):
    """ResNet-18 用的两组图：合成桌面图放大到 240x320；matplotlib 自带的一张真实照片（600x512）的随机裁剪。"""
    resnet()
    photo = _RN["photo"]
    rng = np.random.default_rng(seed)
    syn = [cv2.resize(render(rng, i % K)[0], (320, 240), interpolation=cv2.INTER_CUBIC) for i in range(n_syn)]
    real = []
    for _ in range(n_real):
        h = int(rng.integers(260, 520))
        w = min(int(h * rng.choice([0.75, 1.0, 4 / 3])), photo.shape[1])
        y0 = int(rng.integers(0, photo.shape[0] - h + 1))
        x0 = int(rng.integers(0, photo.shape[1] - w + 1))
        real.append(photo[y0:y0 + h, x0:x0 + w].copy())
    return syn, real


def imagenet_geom(img, how="pil"):
    """torchvision 给 ResNet-18 的标准几何：短边缩到 256，中心裁 224。"""
    h, w = img.shape[:2]
    s = 256 / min(h, w)
    nh, nw = (256, int(w * s)) if h <= w else (int(h * s), 256)
    r = resize(img, (nw, nh), how)
    y0, x0 = int(round((nh - 224) / 2.0)), int(round((nw - 224) / 2.0))
    return r[y0:y0 + 224, x0:x0 + 224]


@torch.inference_mode()
def resnet_run(net, x, chunk=8):
    """返回 (logits, 全局平均池化后的 512 维特征)。"""
    body = nn.Sequential(*list(net.children())[:-1])
    feats = torch.cat([body(x[i:i + chunk]).flatten(1) for i in range(0, len(x), chunk)])
    return net.fc(feats), feats


def compare_logits(out, ref):
    return {
        "top-1 一致率": r3((out.argmax(1) == ref.argmax(1)).float().mean()),
        "logits 相对变化（中位数）": r3(((out - ref).norm(dim=1) / ref.norm(dim=1)).median()),
        "平均 top-1 概率": r3(out.softmax(1).max(1).values.mean()),
    }


# ---------------------------------------------------------------- 图 1：只改预处理的一处
VARIANTS = ["一致", "通道顺序 BGR", "没有除以 255", "除了两次 255", "没有减均值除方差", "用了 ImageNet 的均值方差",
            "直接拉伸，不加边", "加边值用 0", "cv2 双线性", "cv2 最近邻", "cv2 双三次", "cv2 INTER_AREA", "reshape 代替 transpose"]


def small_variants(imgs, mean, std):
    lb = lambda **kw: [letterbox(i, **kw)[0] for i in imgs]  # noqa: E731
    base = lb()
    return {
        "一致": to_tensor(base, mean, std),
        "通道顺序 BGR": to_tensor(base, mean, std, order="bgr"),
        "没有除以 255": to_tensor(base, mean, std, scale=1.0),
        "除了两次 255": to_tensor(base, mean, std, scale=255.0 * 255.0),
        "没有减均值除方差": to_tensor(base, mean, std, normalize=False),
        "用了 ImageNet 的均值方差": to_tensor(base, IMN_MEAN, IMN_STD),
        "直接拉伸，不加边": to_tensor([stretch(i) for i in imgs], mean, std),
        "加边值用 0": to_tensor(lb(pad=0), mean, std),
        "cv2 双线性": to_tensor(lb(how="linear"), mean, std),
        "cv2 最近邻": to_tensor(lb(how="nearest"), mean, std),
        "cv2 双三次": to_tensor(lb(how="cubic"), mean, std),
        "cv2 INTER_AREA": to_tensor(lb(how="area"), mean, std),
        "reshape 代替 transpose": to_tensor(base, mean, std, layout="reshape"),
    }


def resnet_variants(imgs):
    geo = lambda how="pil": [imagenet_geom(i, how) for i in imgs]  # noqa: E731
    base = geo()
    t = lambda c, **kw: to_tensor(c, IMN_MEAN, IMN_STD, **kw)  # noqa: E731
    return {
        "一致": t(base),
        "通道顺序 BGR": t(base, order="bgr"),
        "没有除以 255": t(base, scale=1.0),
        "除了两次 255": t(base, scale=255.0 * 255.0),
        "没有减均值除方差": t(base, normalize=False),
        "均值方差都用 0.5": to_tensor(base, np.float32(0.5), np.float32(0.5)),
        "直接拉伸到 224x224": t([resize(i, (224, 224)) for i in imgs]),
        "letterbox 到 224x224": t([letterbox(i, 224)[0] for i in imgs]),
        "cv2 双线性": t(geo("linear")),
        "cv2 最近邻": t(geo("nearest")),
        "cv2 双三次": t(geo("cubic")),
        "cv2 INTER_AREA": t(geo("area")),
        "reshape 代替 transpose": t(base, layout="reshape"),
    }


def fig_preprocess():
    c = ctx()
    xs = small_variants(c["te_imgs"], c["mean"], c["std"])
    acc = {k: [accuracy(predict(n, x), c["yte"]) for n in c["nets"]] for k, x in xs.items()}
    conf = {k: float(np.mean([predict(n, x).softmax(1).max(1).values.mean() for n in c["nets"]])) for k, x in xs.items()}
    per_cls = {}
    for k in ("通道顺序 BGR", "直接拉伸，不加边", "cv2 最近邻", "cv2 双线性"):
        p = [predict(n, xs[k]).argmax(1) for n in c["nets"]]
        per_cls[k] = {CLASSES[j]: r3(np.mean([float((q[c["yte"] == j] == j).float().mean()) for q in p])) for j in range(K)}
    NUM["预处理不一致（小模型）"] = {
        "说明": f"同一批 {len(c['te_imgs'])} 张测试图、同样的 3 个模型，部署时只把预处理改一处，其余和训练契约相同。数值是测试精度，随机猜是 {1 / K}",
        "改动的种数": len(xs) - 1,
        "精度低于 0.9 的改动种数": int(sum(np.mean(v) < 0.9 for k, v in acc.items() if k != "一致")),
        "精度（3 个种子平均）": {k: r3(np.mean(v)) for k, v in acc.items()},
        "精度（各种子）": {k: [r3(a) for a in v] for k, v in acc.items()},
        "平均 top-1 概率（3 个种子平均）": {k: r3(v) for k, v in conf.items()},
        "分类别的精度（3 个种子平均）": per_cls,
    }

    net = resnet()
    syn, real = image_sets(7, 48, 48)
    tf = _RN["weights"].transforms()
    check = float((tf(Image.fromarray(real[0])) - to_tensor([imagenet_geom(real[0])], IMN_MEAN, IMN_STD)[0]).abs().max())
    res = {}
    for name, imgs in (("合成桌面图", syn), ("真实照片的裁剪", real)):
        xv = resnet_variants(imgs)
        ref, fref = resnet_run(net, xv["一致"])
        rows = {}
        for k, x in xv.items():
            out, f = resnet_run(net, x)
            rows[k] = dict(compare_logits(out, ref), **{"特征余弦相似度（中位数）": r3(nn.functional.cosine_similarity(f, fref).median())})
        res[name] = rows
    params, macs = count_cost(net, torch.zeros(1, 3, 224, 224))
    NUM["预处理不一致（ResNet-18）"] = {
        "说明": "torchvision 的 ResNet-18（ImageNet 权重）。参考管线是权重自带的预处理：RGB，短边用 PIL 双线性缩到 256，中心裁 224，除以 255，"
                "减 ImageNet 均值除以标准差。部署时只改一处，比较 1000 类输出和参考管线是否一致。"
                "两组图各 48 张：合成桌面图（放大到 240x320）；matplotlib 自带的一张真实照片（grace_hopper.jpg，600x512）的 48 个随机裁剪，不是 48 张不同的照片。"
                "这些图都不是 ImageNet 的验证图，模型本来就不确定（看“一致”一行的平均 top-1 概率），所以一致率对小改动偏敏感，只看相对大小",
        "参考管线和 torchvision 自带 transforms 的最大差": check,
        "ImageNet 的通道均值": [r3(v) for v in IMN_MEAN], "ImageNet 的通道标准差": [r3(v) for v in IMN_STD],
        "真实照片的尺寸（高, 宽）": list(_RN["photo"].shape[:2]), "每组图的张数": len(syn),
        "合成桌面图": res["合成桌面图"],
        "真实照片的裁剪": res["真实照片的裁剪"],
    }
    NUM["模型规模"] = {
        "说明": "参数量，以及一次前向里卷积和全连接的乘加次数（不含 BatchNorm、池化、激活）。ResNet-18 按 224x224 输入，小模型按 48x48 输入",
        "ResNet-18 参数量": params, "ResNet-18 参数量（百万）": round(params / 1e6, 2), "ResNet-18 乘加次数（十亿）": r3(macs / 1e9),
        "小模型参数量": NUM["数据与小模型"]["参数量"], "小模型参数量（千）": round(NUM["数据与小模型"]["参数量"] / 1e3, 1),
        "小模型乘加次数（百万）": NUM["数据与小模型"]["一次前向的乘加次数（百万）"],
    }

    # ---- 画图
    show = ["一致", "通道顺序 BGR", "没有除以 255", "直接拉伸，不加边", "cv2 最近邻", "reshape 代替 transpose"]
    idx = int(np.flatnonzero(c["yte"].numpy() == 1)[3])
    fig = plt.figure(figsize=(14.4, 5.2), layout="constrained")
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 3.3, 3.3])
    mean, std = c["mean"], c["std"]
    for j, k in enumerate(show):
        ax = fig.add_subplot(gs[j // 3, j % 3])
        seen = xs[k][idx].numpy().transpose(1, 2, 0) * std + mean
        ax.imshow(np.clip(seen, 0, 1), interpolation="nearest")
        S.image_axes(ax)
        ax.set_title(k.replace("，不加边", "").replace(" 代替 transpose", "").replace("通道顺序 ", ""), fontsize=9.5, fontweight="normal")
    ax = fig.add_subplot(gs[:, 3])
    order = sorted(VARIANTS, key=lambda k: -np.mean(acc[k]))[::-1]
    ys = np.arange(len(order))
    vals = [np.mean(acc[k]) for k in order]
    ax.barh(ys, vals, color=[S.GREEN if k == "一致" else (S.BLUE if v > 0.9 else S.RED) for k, v in zip(order, vals)], height=0.68)
    for y, k in zip(ys, order):
        ax.plot(acc[k], [y] * len(acc[k]), "o", color=S.INK, ms=2.6, zorder=3)
        ax.text(max(acc[k]) + 0.02, y, f"{np.mean(acc[k]):.3f}", va="center", fontsize=8.5, color=S.INK)
    ax.axvline(1 / K, color=S.MUTED, lw=1, ls="--")
    ax.text(1 / K + 0.01, len(order) - 0.45, "随机猜", color=S.MUTED, fontsize=8.5, va="bottom")
    ax.set_yticks(ys)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlim(0, 1.16)
    ax.set_ylim(-0.6, len(order) + 0.1)
    ax.set_xlabel("小模型的测试精度（黑点是 3 个种子）")
    n_bad = sum(v < 0.9 for k, v in zip(order, vals) if k != "一致")
    ax.set_title("小任务：%d 种改动里 %d 种让精度低于 0.9" % (len(order) - 1, n_bad))
    S.clean(ax)
    ax = fig.add_subplot(gs[:, 4])
    names = [k for k in res["合成桌面图"] if k != "一致"]
    names = sorted(names, key=lambda k: res["合成桌面图"][k]["top-1 一致率"])
    ys = np.arange(len(names))
    ax.barh(ys + 0.19, [res["合成桌面图"][k]["top-1 一致率"] for k in names], height=0.36, color=S.BLUE, label="合成桌面图")
    ax.barh(ys - 0.19, [res["真实照片的裁剪"][k]["top-1 一致率"] for k in names], height=0.36, color=S.ORANGE, label="真实照片的裁剪")
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("ResNet-18 的 top-1 和参考管线一致的比例")
    ax.set_title("ResNet-18：同样的改动，一致率从 0 到 %.2f" % max(max(r[k]["top-1 一致率"] for k in names) for r in res.values()))
    ax.legend(loc="lower right", fontsize=9)
    S.clean(ax)
    fig.suptitle("同一个模型、同一批图，部署时只把预处理改一处", fontsize=12.5, fontweight="bold")
    save(fig, "fig01_预处理不一致")


# ---------------------------------------------------------------- 图 2：数据覆盖与相邻帧泄漏
def fig_data():
    te_imgs, te_y, _ = make_set(900, 800)
    yte = torch.from_numpy(te_y)

    def fit_eval(imgs, ys, seed, epochs, extra=None):
        mean, std = channel_stats(imgs)
        prep = lambda ims: to_tensor([letterbox(i)[0] for i in ims], mean, std)  # noqa: E731
        net = train(new_classifier(seed), prep(imgs), torch.tensor(ys), seed, epochs)
        out = [accuracy(predict(net, prep(te_imgs)), yte)]
        if extra is not None:
            out.append(accuracy(predict(net, prep(extra[0])), torch.tensor(extra[1])))
        return out

    n_img, envs_list = 320, (1, 5, 20, 320)
    cover = {}
    for E in envs_list:
        rows = []
        for seed in SEEDS:
            rng = np.random.default_rng(1000 + seed)
            envs = [make_env(rng) for _ in range(E)]
            imgs, ys = [], []
            for i in range(n_img):
                imgs.append(render(rng, i % K, env=envs[(i // K) % E])[0])
                ys.append(i % K)
            rows.append(fit_eval(imgs, ys, seed, 12)[0])
        cover[E] = rows
    NUM["数据覆盖"] = {
        "说明": f"训练集固定 {n_img} 张（每类 {n_img // K} 张），来自 E 种环境（一种环境是一组桌面灰度、白平衡、亮度、光照方向、噪声水平），每种环境张数相同；"
                f"测试集 {len(te_imgs)} 张，每张都是没见过的新环境。小模型训练 12 轮，3 个种子",
        "环境数": list(envs_list),
        "新环境上的精度（3 个种子平均）": [r3(np.mean(cover[E])) for E in envs_list],
        "各种子": {str(E): [r3(v) for v in cover[E]] for E in envs_list},
    }

    V, F = 80, 8
    leak = []
    for seed in SEEDS:
        rng = np.random.default_rng(2000 + seed)
        frames = []
        for v in range(V):
            cls, env = v % K, make_env(rng)
            obj = sample_obj(rng, cls)
            for _ in range(F):
                frames.append((cls, render(rng, cls, env=env, obj=obj, dxy=tuple(rng.uniform(-1.5, 1.5, 2)))[0], v))
        n = len(frames)
        perm = np.random.default_rng(seed).permutation(n)
        rand_val, rand_tr = perm[:n // 5], perm[n // 5:]
        vperm = np.random.default_rng(seed).permutation(V)
        val_v = set()
        for cls in range(K):
            cand = [int(v) for v in vperm if v % K == cls]
            val_v.update(cand[:len(cand) // 5])
        grp_val = [i for i in range(n) if frames[i][2] in val_v]
        grp_tr = [i for i in range(n) if frames[i][2] not in val_v]
        row = []
        for tr, va in ((rand_tr, rand_val), (grp_tr, grp_val)):
            te, va_acc = fit_eval([frames[i][1] for i in tr], [frames[i][0] for i in tr], seed, 12,
                                  extra=([frames[i][1] for i in va], [frames[i][0] for i in va]))
            row += [va_acc, te]
        leak.append(row)
    leak = np.array(leak)
    NUM["相邻帧泄漏"] = {
        "说明": f"{V} 段“视频”，每段 {F} 帧：同一段里环境和物体都相同，物体每帧只随机挪动不超过 1.5 px，噪声每帧重新生成。共 {V * F} 帧，每类 {V // K} 段。"
                f"两种切分都留 20% 做验证：按帧随机切（同一段的帧会同时出现在训练和验证里）；按视频切（每类留 {V // K // 5} 段，整段进验证）。"
                f"“新视频上的精度”用 {len(te_imgs)} 张全新环境、全新物体的图测。小模型训练 12 轮，3 个种子",
        "按帧随机切：验证精度（各种子）": [r3(v) for v in leak[:, 0]],
        "按帧随机切：新视频上的精度（各种子）": [r3(v) for v in leak[:, 1]],
        "按视频切：验证精度（各种子）": [r3(v) for v in leak[:, 2]],
        "按视频切：新视频上的精度（各种子）": [r3(v) for v in leak[:, 3]],
        "平均": {"按帧随机切：验证": r3(leak[:, 0].mean()), "按帧随机切：新视频": r3(leak[:, 1].mean()),
               "按视频切：验证": r3(leak[:, 2].mean()), "按视频切：新视频": r3(leak[:, 3].mean())},
        "验证精度虚高了多少（验证减新视频，平均）": {"按帧随机切": r3((leak[:, 0] - leak[:, 1]).mean()), "按视频切": r3((leak[:, 2] - leak[:, 3]).mean())},
        "按视频切时验证集的视频段数": len(val_v), "按视频切时训练集的视频段数": V - len(val_v),
        "两种切分的训练帧数": [int(len(rand_tr)), len(grp_tr)],
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), layout="constrained")
    ax = axes[0]
    xs_ = np.arange(len(envs_list))
    for i, E in enumerate(envs_list):
        ax.plot([i] * len(cover[E]), cover[E], "o", color=S.BLUE, alpha=0.35, ms=5)
    ax.plot(xs_, [np.mean(cover[E]) for E in envs_list], "o-", color=S.BLUE, lw=2, ms=6)
    for i, E in enumerate(envs_list):
        ax.annotate(f"{np.mean(cover[E]):.3f}", (i, np.mean(cover[E])), textcoords="offset points", xytext=(0, -17), ha="center", fontsize=9)
    ax.set_xticks(xs_)
    ax.set_xticklabels([str(E) for E in envs_list])
    ax.set_xlabel(f"训练集的环境数（总张数固定 {n_img} 张）")
    ax.set_ylabel("新环境上的精度")
    ax.set_xlim(-0.35, len(envs_list) - 0.65)
    ax.set_ylim(0.2, 1.0)
    ax.set_title("同样 %d 张图：1 种环境 %.2f，20 种环境 %.2f" % (n_img, np.mean(cover[1]), np.mean(cover[20])))
    S.clean(ax)
    ax = axes[1]
    w = 0.36
    for i, (a, b, name) in enumerate(((0, 1, "按帧随机切"), (2, 3, "按视频切"))):
        ax.bar(i - w / 2, leak[:, a].mean(), w, color=S.ORANGE, label="验证精度" if i == 0 else None)
        ax.bar(i + w / 2, leak[:, b].mean(), w, color=S.GREEN, label="新视频上的精度" if i == 0 else None)
        ax.plot([i - w / 2] * 3, leak[:, a], "o", color=S.INK, ms=3)
        ax.plot([i + w / 2] * 3, leak[:, b], "o", color=S.INK, ms=3)
        ax.text(i - w / 2, leak[:, a].max() + 0.02, f"{leak[:, a].mean():.3f}", ha="center", fontsize=9)
        ax.text(i + w / 2, leak[:, b].max() + 0.02, f"{leak[:, b].mean():.3f}", ha="center", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["按帧随机切", "按视频切"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("精度（黑点是 3 个种子）")
    ax.set_title("按帧随机切：验证 %.2f，新视频上只有 %.2f" % (leak[:, 0].mean(), leak[:, 1].mean()))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    S.clean(ax)
    save(fig, "fig02_数据覆盖与泄漏")


# ---------------------------------------------------------------- 图 3：计时
COLD_SCRIPT = r'''
import sys, time, json, warnings
warnings.simplefilter("ignore")
import torch, torchvision
torch.set_num_threads(2)
torch.manual_seed(0)
x = torch.randn(1, 3, 224, 224)
if sys.argv[1] == "eager":
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()
else:
    m = torch.jit.load(sys.argv[2]).eval()
ts = []
with torch.inference_mode():
    for _ in range(30):
        t = time.perf_counter(); m(x); ts.append((time.perf_counter() - t) * 1e3)
print(json.dumps(ts))
'''


def timeit(fn, n, warm=2):
    with torch.inference_mode():
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t) * 1e3)
    return np.array(ts)


def machine_note():
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        cpu = ""
    return (f"{cpu or platform.processor()}，{platform.system()} {platform.machine()}，只用 CPU，torch.set_num_threads(2)；"
            f"torch {torch.__version__.split('+')[0]}，torchvision {torchvision.__version__.split('+')[0]}，OpenCV {cv2.__version__}。"
            "测量时机器上同时有别的任务在跑，绝对值偏大，只看量级和倍数")


def measure_cold():
    net = resnet()
    x = torch.randn(1, 3, 224, 224)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "resnet18.pt")
        torch.jit.trace(net, x).save(path)
        size = os.path.getsize(path)
        out = {}
        for kind in ("eager", "torchscript"):
            runs = []
            for _ in range(3):
                res = subprocess.run([sys.executable, "-c", COLD_SCRIPT, kind, path], capture_output=True, text=True, timeout=300)
                runs.append(json.loads(res.stdout.strip().splitlines()[-1]))
            runs = np.array(runs)
            out[kind] = {"前 10 次调用（ms，3 个进程的中位数）": [sig(v) for v in np.median(runs[:, :10], axis=0)],
                         "第 1 次（ms）": sig(np.median(runs[:, 0])), "第 2 次（ms）": sig(np.median(runs[:, 1])),
                         "第 11 到 30 次的中位数（ms）": sig(np.median(runs[:, 10:]))}
            out[kind]["第 1 次是稳定后的几倍"] = sig(out[kind]["第 1 次（ms）"] / out[kind]["第 11 到 30 次的中位数（ms）"])
    return {"说明": "ResNet-18，输入 1x3x224x224。每次新开一个 Python 进程，载入模型后连续调用 30 次，记下每次的耗时；eager 和 TorchScript（trace 后存盘再载入）各开 3 个进程。" + machine_note(),
            "eager": out["eager"], "TorchScript": out["torchscript"], "TorchScript 文件大小（MB）": sig(size / 1e6, 3)}


def measure_repeat():
    net = resnet()
    x = torch.randn(1, 3, 224, 224)
    ts = timeit(lambda: net(x), 200, warm=5)
    return {"说明": "同一个进程里，预热 5 次后对同一个输入连续调用 200 次 ResNet-18（批大小 1）。" + machine_note(),
            "最小（ms）": sig(ts.min()), "P50（ms）": sig(np.percentile(ts, 50)), "P95（ms）": sig(np.percentile(ts, 95)),
            "P99（ms）": sig(np.percentile(ts, 99)), "最大（ms）": sig(ts.max()),
            "最大是最小的几倍": sig(ts.max() / ts.min()),
            "只取前 5 次的中位数（ms）": sig(np.median(ts[:5])),
            "全部 200 次（ms，按测量顺序，只用来画图）": [sig(v, 3) for v in ts]}


def measure_batch():
    net = resnet()
    sizes = (1, 2, 4, 8, 12, 16, 24, 32)
    out = {}
    for on in (True, False):
        per, whole = [], []
        with torch.backends.nnpack.flags(enabled=on):
            for b in sizes:
                xb = torch.randn(b, 3, 224, 224)
                ts = timeit(lambda: net(xb), 7 if b <= 8 else 4, warm=1)
                whole.append(sig(np.median(ts)))
                per.append(sig(np.median(ts) / b))
        out["NNPACK 开（PyTorch 在这台机器上的默认）" if on else "NNPACK 关"] = {"整批耗时（ms）": whole, "折合每张（ms）": per}
    a = out["NNPACK 开（PyTorch 在这台机器上的默认）"]["折合每张（ms）"]
    return dict({"说明": "ResNet-18，224x224，批大小从 1 到 32，每个批大小预热 1 次后测 4 到 7 次取中位数。"
                         "torch.backends.nnpack.flags(enabled=...) 切换卷积是否走 NNPACK。" + machine_note(),
                 "批大小": list(sizes)}, **out,
                **{"默认设置下批大小 8 的整批耗时是批大小 1 的几倍": sig(out["NNPACK 开（PyTorch 在这台机器上的默认）"]["整批耗时（ms）"][sizes.index(8)]
                                                      / out["NNPACK 开（PyTorch 在这台机器上的默认）"]["整批耗时（ms）"][0]),
                   "默认设置下批大小 1 的每张耗时是批大小 8 的几倍": sig(a[0] / a[sizes.index(8)]),
                   "默认设置下批大小 16 每张耗时是批大小 8 的几倍": sig(a[sizes.index(16)] / a[sizes.index(8)]),
                   "默认设置下批大小 32 的整批耗时是批大小 1 的几倍": sig(out["NNPACK 开（PyTorch 在这台机器上的默认）"]["整批耗时（ms）"][-1]
                                                       / out["NNPACK 开（PyTorch 在这台机器上的默认）"]["整批耗时（ms）"][0])})


def measure_threads():
    net = resnet()
    x = torch.randn(1, 3, 224, 224)
    out = {}
    for th in (1, 2, 4):
        torch.set_num_threads(th)
        out[str(th)] = sig(np.median(timeit(lambda: net(x), 30)))
    torch.set_num_threads(2)
    return {"说明": "ResNet-18，批大小 1，224x224，改 torch.set_num_threads，每种预热 2 次后测 30 次取中位数。" + machine_note(),
            "线程数对应的耗时（ms）": out}


def measure_pipeline():
    net = resnet()
    rng = np.random.default_rng(0)
    big = cv2.resize(render(rng, 6)[0], (1920, 1440), interpolation=cv2.INTER_CUBIC)[180:1260]
    big = np.clip(big.astype(np.float32) + rng.normal(scale=4, size=big.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(big, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    out = {}
    for flag, name in ((cv2.INTER_AREA, "缩放用 INTER_AREA"), (cv2.INTER_LINEAR, "缩放用 INTER_LINEAR")):
        st = {"JPEG 解码和通道转换": [], "缩放和裁剪": [], "归一化和转张量": [], "模型前向": [], "softmax 和取前 5": []}
        with torch.inference_mode():
            for i in range(45):
                t0 = time.perf_counter()
                im = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                t1 = time.perf_counter()
                h, w = im.shape[:2]
                s = 256 / min(h, w)
                r = cv2.resize(im, (int(w * s), 256), interpolation=flag)
                x0 = (r.shape[1] - 224) // 2
                crop = r[16:240, x0:x0 + 224]
                t2 = time.perf_counter()
                xt = torch.from_numpy(np.ascontiguousarray(((crop.astype(np.float32) / 255 - IMN_MEAN) / IMN_STD).transpose(2, 0, 1)))[None]
                t3 = time.perf_counter()
                o = net(xt)
                t4 = time.perf_counter()
                top = o.softmax(1).topk(5)
                _ = [(int(j), float(p)) for p, j in zip(top.values[0], top.indices[0])]
                t5 = time.perf_counter()
                if i >= 5:
                    for k, v in zip(st, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
                        st[k].append(v * 1e3)
        med = {k: sig(np.median(v)) for k, v in st.items()}
        total = sum(np.median(v) for v in st.values())
        out[name] = dict(med, **{"合计（ms）": sig(total), "模型前向占合计的比例": sig(np.median(st["模型前向"]) / total)})
    out["缩放和裁剪：INTER_AREA 是 INTER_LINEAR 的几倍"] = sig(out["缩放用 INTER_AREA"]["缩放和裁剪"] / out["缩放用 INTER_LINEAR"]["缩放和裁剪"])
    return {"说明": f"一张宽 1920、高 1080 的 JPEG（质量 90，{len(buf) // 1000} KB，内容是放大并加噪声的合成桌面图）走完整条分类流水线：cv2.imdecode 解码并转 RGB，"
                    "短边缩到 256 再中心裁 224，除以 255 并归一化成张量，ResNet-18 前向，softmax 取前 5。预热 5 次后测 40 次，各阶段取中位数（ms）。" + machine_note(),
            **out}


def fig_timing():
    cold = NUM["计时：冷启动"] = S.pinned(CH, "计时：冷启动", measure_cold, RETIME)
    rep = NUM["计时：重复 200 次"] = S.pinned(CH, "计时：重复 200 次", measure_repeat, RETIME)
    bat = NUM["计时：批大小"] = S.pinned(CH, "计时：批大小", measure_batch, RETIME)
    NUM["计时：线程数"] = S.pinned(CH, "计时：线程数", measure_threads, RETIME)
    pipe = NUM["计时：端到端"] = S.pinned(CH, "计时：端到端", measure_pipeline, RETIME)

    fig, axes = plt.subplots(1, 4, figsize=(14.6, 4.1), gridspec_kw={"width_ratios": [1.0, 1.15, 1.15, 1.0]}, layout="constrained")
    ax = axes[0]
    calls = np.arange(1, 11)
    ax.plot(calls, cold["eager"]["前 10 次调用（ms，3 个进程的中位数）"], "o-", color=S.BLUE, label="eager")
    ax.plot(calls, cold["TorchScript"]["前 10 次调用（ms，3 个进程的中位数）"], "s-", color=S.ORANGE, label="TorchScript")
    ax.set_xticks(calls)
    ax.set_xlabel("新进程里的第几次调用")
    ax.set_ylabel("耗时（ms）")
    ax.set_ylim(0, None)
    ax.set_title("TorchScript 第 1 次调用\n是稳定后的 %.2g 倍" % cold["TorchScript"]["第 1 次是稳定后的几倍"])
    ax.legend(loc="upper right")
    S.clean(ax)
    ax = axes[1]
    ts = np.array(rep["全部 200 次（ms，按测量顺序，只用来画图）"])
    ax.plot(np.arange(1, len(ts) + 1), ts, ".", color=S.BLUE, ms=4)
    for key, col in (("P50（ms）", S.GREEN), ("P95（ms）", S.RED)):
        ax.axhline(rep[key], color=col, lw=1.2, ls="--")
        ax.text(len(ts), rep[key], key[:3] + " ", color=col, fontsize=9, va="bottom" if key.startswith("P95") else "top", ha="right")
    ax.set_xlabel("同一进程里的第几次调用")
    ax.set_ylabel("耗时（ms）")
    ax.set_ylim(0, ts.max() * 1.15)
    ax.set_title("同样的调用 200 次：\n最小 %.2g ms，最大 %.2g ms" % (rep["最小（ms）"], rep["最大（ms）"]))
    S.clean(ax)
    ax = axes[2]
    sizes = bat["批大小"]
    xs_ = np.arange(len(sizes))
    ax.plot(xs_, bat["NNPACK 开（PyTorch 在这台机器上的默认）"]["折合每张（ms）"], "o-", color=S.RED, label="默认设置")
    ax.plot(xs_, bat["NNPACK 关"]["折合每张（ms）"], "s--", color=S.MUTED, label="关掉 NNPACK")
    ax.set_xticks(xs_)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("批大小")
    ax.set_ylabel("折合每张的耗时（ms）")
    ax.set_ylim(0, None)
    ax.set_title("批大小 16 时，每张耗时\n是批大小 8 的 %.2g 倍（这台机器）" % bat["默认设置下批大小 16 每张耗时是批大小 8 的几倍"])
    ax.legend(loc="upper left")
    S.clean(ax)
    ax = axes[3]
    stages = ["JPEG 解码和通道转换", "缩放和裁剪", "归一化和转张量", "模型前向", "softmax 和取前 5"]
    cols = [S.PURPLE, S.ORANGE, S.GREEN, S.BLUE, S.MUTED]
    for i, name in enumerate(("缩放用 INTER_AREA", "缩放用 INTER_LINEAR")):
        bottom = 0.0
        for st, col in zip(stages, cols):
            v = pipe[name][st]
            ax.bar(i, v, 0.55, bottom=bottom, color=col, label=st if i == 0 else None)
            if v > 1.5:
                ax.text(i, bottom + v / 2, f"{v:g}", ha="center", va="center", color="white", fontsize=9)
            bottom += v
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["INTER_AREA", "INTER_LINEAR"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("一张图的耗时（ms）")
    ax.set_title("端到端：模型前向只占\n%.0f%% 到 %.0f%%" % (100 * pipe["缩放用 INTER_AREA"]["模型前向占合计的比例"], 100 * pipe["缩放用 INTER_LINEAR"]["模型前向占合计的比例"]))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=8.5)
    S.clean(ax)
    save(fig, "fig03_计时")


# ---------------------------------------------------------------- 图 4：缩放、插值和 letterbox 回映射
def iou(a, b):
    ix = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = ix * iy
    union = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(union, 1e-12)


def to_canvas(boxes, meta, src_hw):
    """原图上的框变到 letterbox 画布上。图像内容实际被缩放到 resized_hw，所以两个方向各用实际的比例。"""
    nh, nw = meta["resized_hw"]
    px, py = meta["pad_xy"]
    sh, sw = src_hw
    b = np.array(boxes, dtype=np.float64)
    b[:, [0, 2]] = b[:, [0, 2]] * nw / sw + px
    b[:, [1, 3]] = b[:, [1, 3]] * nh / sh + py
    return b


def clip_boxes(b, hw):
    b = b.copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, hw[1])
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, hw[0])
    return b


def wrong_inverses(dst):
    """几种常见的错误反算，和参考实现 boxes_to_original 对照。"""
    def no_pad(b, m, hw):
        return clip_boxes(b / m["scale"], hw)

    def as_stretch(b, m, hw):
        o = b.copy()
        o[:, [0, 2]] *= hw[1] / dst
        o[:, [1, 3]] *= hw[0] / dst
        return clip_boxes(o, hw)

    def pad_swapped(b, m, hw):
        return mod.boxes_to_original(b, dict(m, pad_xy=(m["pad_xy"][1], m["pad_xy"][0])), hw)

    def order_wrong(b, m, hw):
        o = b / m["scale"]
        o[:, [0, 2]] -= m["pad_xy"][0]
        o[:, [1, 3]] -= m["pad_xy"][1]
        return clip_boxes(o, hw)

    return {"正确（参考实现）": mod.boxes_to_original, "漏减加边": no_pad, "按拉伸反算": as_stretch,
            "加边的 x 和 y 写反": pad_swapped, "先除比例再减加边": order_wrong}


def c_round(x):
    """C++ std::round 对正数的行为：.5 远离零。"""
    f = math.floor(x)
    return f + (1 if x - f >= 0.5 else 0)


def fig_resize_letterbox():
    # ---- 插值：cv2 的四种标志和 PIL 双线性差多少
    rng = np.random.default_rng(3)
    imgs = [render(rng, i % K)[0] for i in range(64)]
    H, W = SRC_HW
    factors = (1.25, 1.5, 2.0, 2.5, 10 / 3, 4.0, 5.0)
    hows = ("linear", "area", "nearest", "cubic")
    mae = {h: [] for h in hows}
    t_aa, t_noaa = [], []
    xt = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float()
    for f in factors:
        wh = (int(round(W / f)), int(round(H / f)))
        ref = np.stack([resize(i, wh) for i in imgs]).astype(np.float64)
        for h in hows:
            mae[h].append(float(np.abs(np.stack([resize(i, wh, h) for i in imgs]) - ref).mean()))
        for aa, store in ((True, t_aa), (False, t_noaa)):
            o = nn.functional.interpolate(xt, size=(wh[1], wh[0]), mode="bilinear", antialias=aa, align_corners=False).permute(0, 2, 3, 1).numpy()
            store.append(float(np.abs(o - ref).mean()))
    up = (W * 2, H * 2)
    ref = np.stack([resize(i, up) for i in imgs]).astype(np.float64)
    up_mae = {h: float(np.abs(np.stack([resize(i, up, h) for i in imgs]) - ref).mean()) for h in hows}
    area_vs_nearest = int(np.abs(np.stack([resize(i, up, "area") for i in imgs]).astype(int) - np.stack([resize(i, up, "nearest") for i in imgs])).max())
    name = {"linear": "cv2 双线性（INTER_LINEAR）", "area": "cv2 INTER_AREA", "nearest": "cv2 最近邻", "cubic": "cv2 双三次"}
    NUM["缩放与插值"] = {
        "说明": f"64 张 {H}x{W} 的合成桌面图，缩小到 1/f。参考是 PIL 的双线性（Image.resize，缩小时自带抗锯齿）。数值是和参考的平均绝对差，单位是灰度级（0 到 255）。"
                "torch 的两行是 F.interpolate(mode='bilinear') 开和不开 antialias（输出是浮点，没有取整）",
        "缩小倍数 f": [r3(f) for f in factors],
        "和 PIL 双线性的平均绝对差": dict({name[h]: [r3(v) for v in mae[h]] for h in hows},
                               **{"torch 双线性，antialias=True": [r3(v) for v in t_aa], "torch 双线性，antialias=False": [r3(v) for v in t_noaa]}),
        "放大 2 倍时和 PIL 双线性的平均绝对差": {name[h]: r3(v) for h, v in up_mae.items()},
        "放大 2 倍时 INTER_AREA 和最近邻的最大差": area_vs_nearest,
    }

    # ---- 回映射：只算几何，不用模型
    dst = 640
    inv = wrong_inverses(dst)
    geo = {}
    rng = np.random.default_rng(0)
    for hw in ((1080, 1920), (480, 640), (2048, 2448)):
        h, w = hw
        meta = mod.letterbox_metadata(hw, (dst, dst))
        n = 2000
        bw, bh = rng.uniform(0.05, 0.4, n) * w, rng.uniform(0.05, 0.4, n) * h
        x0, y0 = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
        gt = np.stack([x0, y0, x0 + bw, y0 + bh], 1)
        canvas = to_canvas(gt, meta, hw)
        rows = {}
        for k, f in inv.items():
            out = f(canvas.copy(), meta, hw)
            v = iou(out, gt)
            ctr = np.hypot((out[:, 0] + out[:, 2] - gt[:, 0] - gt[:, 2]) / 2, (out[:, 1] + out[:, 3] - gt[:, 1] - gt[:, 3]) / 2)
            rows[k] = {"IoU 平均": r3(v.mean()), "IoU 不低于 0.5 的比例": r3((v >= 0.5).mean()), "框中心偏移的中位数（px）": round(float(np.median(ctr)), 1),
                       "角点最大误差（px）": round(float(np.abs(out - gt).max()), 2)}
        geo[f"{h}x{w}"] = {"原图（高, 宽）": [h, w], "缩放后没有取整的尺寸（高, 宽）": [round(h * meta["scale"], 1), round(w * meta["scale"], 1)], "元数据": {"scale": r4(meta["scale"]), "resized_hw": list(meta["resized_hw"]), "pad_xy": list(meta["pad_xy"])}, "结果": rows}
    NUM["letterbox 回映射（只算几何）"] = {
        "说明": f"原图（高x宽）letterbox 到 {dst}x{dst}。每种尺寸随机 2000 个真值框（宽高各占原图的 5% 到 40%），先精确变到画布上（当作模型的理想输出），"
                "再用不同的方法反算回原图，和真值比。元数据和“正确”一行用本章作业的参考实现",
        **geo,
    }

    # ---- 取整带来的误差，以及 Python 和 C++ 的取整方式
    rng = np.random.default_rng(1)
    worst = (0.0, None)
    errs = []
    for _ in range(3000):
        h, w = int(rng.integers(200, 4001)), int(rng.integers(200, 4001))
        meta = mod.letterbox_metadata((h, w), (dst, dst))
        gt = np.array([[0.05 * w, 0.05 * h, 0.95 * w, 0.95 * h]])
        e = float(np.abs(mod.boxes_to_original(to_canvas(gt, meta, (h, w)), meta, (h, w)) - gt).max())
        errs.append(e)
        if e > worst[0]:
            worst = (e, (h, w, meta["scale"]))
    NUM["回映射的取整误差"] = {
        "说明": f"参考实现把缩放后的尺寸四舍五入成整数（resized_hw），图像内容按这个整数尺寸缩放，反算时却除以没有取整的 scale，两者差出来的就是这个误差。"
                f"3000 种随机尺寸（高、宽各在 200 到 4000 之间），letterbox 到 {dst}x{dst}。每种尺寸取一个四边各离图像边界 5% 的框（不会碰到裁剪），看反算后的最大角点误差",
        "误差的中位数（px）": r3(np.median(errs)), "最大误差（px）": r3(worst[0]),
        "最大误差出现的尺寸（高, 宽）": [worst[1][0], worst[1][1]], "该尺寸的 scale": r4(worst[1][2]),
        "该尺寸在图像边界处的理论上界 0.5/scale（px）": r3(0.5 / worst[1][2]),
        "误差超过 1 px 的尺寸占比": r3(np.mean(np.array(errs) > 1.0)),
        "2048x2448 的最大角点误差（px）": geo["2048x2448"]["结果"]["正确（参考实现）"]["角点最大误差（px）"],
    }
    lo, hi = 200, 2000
    hs = np.arange(lo, hi + 1, dtype=np.int64)
    hh, ww = np.meshgrid(hs, hs, indexing="ij")
    long_side, short_side = np.maximum(hh, ww), np.minimum(hh, ww)
    twice = 2 * short_side * dst
    tie = (twice % long_side == 0) & ((twice // long_side) % 2 == 1)
    diff, example = 0, None
    for h, w in zip(hh[tie].tolist(), ww[tie].tolist()):
        meta = mod.letterbox_metadata((h, w), (dst, dst))
        s = min(dst / h, dst / w)
        c_hw = (min(dst, c_round(h * s)), min(dst, c_round(w * s)))
        if c_hw != tuple(meta["resized_hw"]):
            diff += 1
            if (h, w) == (721, 1280):
                example = (meta, c_hw)
    meta_py, c_hw = example
    meta_c = {"scale": meta_py["scale"], "resized_hw": c_hw, "pad_xy": ((dst - c_hw[1]) // 2, (dst - c_hw[0]) // 2)}
    box = np.array([[100.0, 200.0, 300.0, 400.0]])
    shift = np.abs(mod.boxes_to_original(box, meta_py, (721, 1280)) - mod.boxes_to_original(box, meta_c, (721, 1280))).max()
    NUM["取整方式：Python 和 C++"] = {
        "说明": f"Python 的 round 遇到正好 .5 时取偶数，C++ 的 std::round 遇到 .5 时远离零。letterbox 到 {dst}x{dst} 时，"
                f"高、宽各从 {lo} 到 {hi} 的全部尺寸里，找出短边缩放后正好落在 .5 上的，再看两种取整给出的 resized_hw 是否相同",
        "尺寸总数": int(hh.size), "缩放后正好落在 .5 上的尺寸数": int(tie.sum()), "其中两种取整结果不同的尺寸数": diff,
        "两种取整结果不同的尺寸占全部尺寸的比例": round(diff / hh.size, 5),
        "例子": {"原图（高, 宽）": [721, 1280], "Python round 的 resized_hw": list(meta_py["resized_hw"]), "Python 的 pad_xy": list(meta_py["pad_xy"]),
               "std::round 的 resized_hw": list(c_hw), "std::round 的 pad_xy": list(meta_c["pad_xy"]),
               "同一个画布框按两套元数据反算，原图上相差（px）": r3(shift)},
    }

    # ---- 回归小模型：训练用 letterbox，部署换一种做法
    cls = list(range(1, K))
    tr_imgs, _, tr_b = make_set(300, 1280, cls)
    te_imgs, _, te_b = make_set(400, 700, cls)
    tr_b, te_b = np.array(tr_b), np.array(te_b)
    meta = mod.letterbox_metadata(SRC_HW, (DST, DST))
    mean, std = channel_stats(tr_imgs)
    xtr = to_tensor([letterbox(i)[0] for i in tr_imgs], mean, std)
    ytr = torch.from_numpy((to_canvas(tr_b, meta, SRC_HW) / DST).astype(np.float32))
    x_lb = to_tensor([letterbox(i)[0] for i in te_imgs], mean, std)
    x_st = to_tensor([stretch(i) for i in te_imgs], mean, std)
    inv48 = wrong_inverses(DST)
    pipes = ["letterbox 输入，正确反算", "letterbox 输入，漏减加边", "letterbox 输入，按拉伸反算", "拉伸输入，按拉伸反算", "拉伸输入，按 letterbox 反算"]
    reg = {p: [] for p in pipes}
    example_boxes = None
    for seed in (0, 1):
        torch.manual_seed(seed)
        net = train(BoxNet(), xtr, ytr, seed, epochs=12, lr=3e-3, loss="box")
        p_lb = predict(net, x_lb).numpy().astype(np.float64) * DST
        p_st = predict(net, x_st).numpy().astype(np.float64) * DST
        outs = {
            pipes[0]: inv48["正确（参考实现）"](p_lb, meta, SRC_HW), pipes[1]: inv48["漏减加边"](p_lb, meta, SRC_HW),
            pipes[2]: inv48["按拉伸反算"](p_lb, meta, SRC_HW), pipes[3]: inv48["按拉伸反算"](p_st, meta, SRC_HW),
            pipes[4]: inv48["正确（参考实现）"](p_st, meta, SRC_HW),
        }
        for p in pipes:
            v = iou(outs[p], te_b)
            reg[p].append([float(v.mean()), float((v >= 0.5).mean()), float((v >= 0.75).mean())])
        if seed == 0:
            example_boxes = {p: outs[p][4] for p in pipes}
    NUM["回归小模型"] = {
        "说明": f"和分类小模型同样的卷积主干，后面接展平和两层全连接，回归物体的框。训练集 {len(tr_imgs)} 张、测试集 {len(te_imgs)} 张（每张一个物体，7 类物体）。"
                f"训练时输入是 letterbox 到 {DST}x{DST} 的图，标签是画布上的框；smooth L1 损失，Adam，12 轮，2 个种子。"
                "测试时换不同的输入做法和反算做法，在原图坐标里和真值比",
        "IoU 平均": {p: r3(np.mean([r[0] for r in reg[p]])) for p in pipes},
        "IoU 不低于 0.5 的比例": {p: r3(np.mean([r[1] for r in reg[p]])) for p in pipes},
        "IoU 不低于 0.75 的比例": {p: r3(np.mean([r[2] for r in reg[p]])) for p in pipes},
        "各种子（IoU 平均、不低于 0.5、不低于 0.75）": {p: [[r3(v) for v in r] for r in reg[p]] for p in pipes},
    }

    # ---- 画图
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.1), gridspec_kw={"width_ratios": [1.1, 1.0, 1.15, 1.25]}, layout="constrained")
    ax = axes[0]
    cols = {"linear": S.RED, "area": S.GREEN, "nearest": S.MUTED, "cubic": S.PURPLE}
    xs_ = np.arange(len(factors))
    for h in hows:
        ax.plot(xs_, mae[h], "o-", color=cols[h], label=name[h].replace("（INTER_LINEAR）", ""), ms=4)
    ax.set_xticks(xs_)
    ax.set_xticklabels([f"{f:.3g}" for f in factors])
    ax.set_xlabel("缩小倍数")
    ax.set_ylabel("和 PIL 双线性的平均绝对差（灰度级）")
    ax.set_ylim(0, None)
    ax.set_title("缩小时最接近 PIL 双线性的\n是 INTER_AREA，不是 INTER_LINEAR")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5)
    S.clean(ax)
    ax = axes[1]
    ax.imshow(te_imgs[4])
    S.image_axes(ax)
    style = {pipes[0]: (S.GREEN, "-"), pipes[1]: (S.RED, "--"), pipes[2]: (S.ORANGE, "-.")}
    gx0, gy0, gx1, gy1 = te_b[4]
    ax.add_patch(plt.Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, fill=False, ec="white", lw=2.4))
    for p, (col, ls) in style.items():
        b = example_boxes[p]
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=col, lw=1.8, ls=ls, label=p.replace("letterbox 输入，", "")))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=1, fontsize=8.5)
    ax.set_title("同一个预测框的三种反算\n（白框是真值）")
    ax = axes[2]
    kinds = list(inv)
    sizes = list(geo)
    wbar = 0.26
    for i, sz in enumerate(sizes):
        ax.bar(np.arange(len(kinds)) + (i - 1) * wbar, [geo[sz]["结果"][k]["IoU 平均"] for k in kinds], wbar, color=[S.BLUE, S.ORANGE, S.PURPLE][i], label=sz)
    ax.set_xticks(np.arange(len(kinds)))
    ax.set_xticklabels(["正确", "漏减\n加边", "按拉伸\n反算", "加边\nx、y 反", "先除\n再减"], fontsize=8.5)
    ax.set_ylabel("和真值的平均 IoU")
    ax.set_ylim(0, 1.08)
    miss = [geo[sz]["结果"]["漏减加边"]["IoU 平均"] for sz in sizes]
    ax.set_title("只算几何：漏减加边时，\n平均 IoU 只剩 %.2f 到 %.2f" % (min(miss), max(miss)))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8.5, title="原图尺寸（高x宽）", title_fontsize=8.5)
    S.clean(ax)
    ax = axes[3]
    ys = np.arange(len(pipes))[::-1]
    v5 = [NUM["回归小模型"]["IoU 不低于 0.5 的比例"][p] for p in pipes]
    v75 = [NUM["回归小模型"]["IoU 不低于 0.75 的比例"][p] for p in pipes]
    ax.barh(ys + 0.19, v5, 0.36, color=S.BLUE, label="IoU 不低于 0.5")
    ax.barh(ys - 0.19, v75, 0.36, color=S.ORANGE, label="IoU 不低于 0.75")
    for y, a, b in zip(ys, v5, v75):
        ax.text(a + 0.015, y + 0.19, f"{a:.2f}", va="center", fontsize=8.5)
        ax.text(b + 0.015, y - 0.19, f"{b:.2f}", va="center", fontsize=8.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([p.replace("，", "\n") for p in pipes], fontsize=8.5)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("测试框的比例")
    ax.set_title("回归小模型：部署改成拉伸，IoU\n不低于 0.75 的比例从 %.2f 到 %.2f" % (v75[0], v75[3]))
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2, fontsize=8.5)
    S.clean(ax)
    save(fig, "fig04_缩放与回映射")


# ---------------------------------------------------------------- 图 5：忘了切到推理模式
def set_mode(net, bn_train, drop_train):
    net.eval()
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train(bn_train)
        if isinstance(m, nn.Dropout):
            m.train(drop_train)
    return net


def fig_eval_mode():
    c = ctx()
    base, xte, yte = c["nets"][0], c["xte"], c["yte"]
    eval_acc = accuracy(predict(base, xte), yte)

    def acc_mode(bn, dp, bs, order):
        m = set_mode(copy.deepcopy(base), bn, dp)
        torch.manual_seed(0)
        preds = torch.empty(len(xte), dtype=torch.long)
        with torch.no_grad():
            for i in range(0, len(xte), bs):
                idx = order[i:i + bs]
                preds[idx] = m(xte[idx]).argmax(1)
        return float((preds == yte).float().mean())

    sizes = (1, 2, 4, 8, 16, 32, 64)
    shuffled = torch.randperm(len(xte), generator=torch.Generator().manual_seed(0))
    by_class = torch.argsort(yte, stable=True)
    modes = {"只有 BatchNorm 在训练模式": (True, False), "只有 Dropout 在训练模式": (False, True), "整个模型 train()": (True, True)}
    curves = {k: [acc_mode(bn, dp, b, shuffled) for b in sizes] for k, (bn, dp) in modes.items()}
    sorted_acc = {str(b): r3(acc_mode(True, True, b, by_class)) for b in (8, 64)}
    NUM["忘了切推理模式：精度随批大小"] = {
        "说明": f"分类小模型（种子 0，eval 模式下测试精度 {eval_acc:.3f}），{len(xte)} 张测试图打乱后按批大小分批推理，全程在 torch.no_grad() 里。"
                "三种情况分别只把 BatchNorm、只把 Dropout、把整个模型留在训练模式",
        "批大小": list(sizes), "eval 模式的精度": r3(eval_acc),
        **{k: [r3(v) for v in vs] for k, vs in curves.items()},
        "整个模型 train()，测试图按类别排好再分批（同一批里几乎都是同一类）": sorted_acc,
    }

    # 同一张图放进不同的批
    g = torch.Generator().manual_seed(1)
    n_target, n_rep, bs = 24, 40, 8
    spread = {}
    strip = None
    for k, (bn, dp) in (("eval", (False, False)), ("只有 BatchNorm 在训练模式", (True, False)), ("整个模型 train()", (True, True))):
        m = set_mode(copy.deepcopy(base), bn, dp)
        torch.manual_seed(5)
        flips, ranges, probs_all = [], [], []
        for t in range(n_target):
            with torch.no_grad():
                ref = int(base(xte[t:t + 1]).argmax())
                probs, tops = [], []
                for _ in range(n_rep):
                    others = torch.randint(0, len(xte), (bs - 1,), generator=g)
                    o = m(torch.cat([xte[t:t + 1], xte[others]]))[0]
                    probs.append(float(o.softmax(0)[int(yte[t])]))
                    tops.append(int(o.argmax()))
            flips.append(np.mean(np.array(tops) != ref))
            ranges.append(max(probs) - min(probs))
            probs_all.append(probs)
        spread[k] = {"预测和 eval 模式不同的比例": r3(np.mean(flips)), "正确类概率的最大值减最小值（各图的中位数）": r3(np.median(ranges)),
                     "正确类概率的最大值减最小值（最大的一张图）": r3(np.max(ranges))}
        if k == "只有 BatchNorm 在训练模式":
            strip = np.array(probs_all)
    with torch.no_grad():
        alone = torch.cat([base(xte[i:i + 1]) for i in range(64)])
        batched = base(xte[:64])
        with torch.backends.nnpack.flags(enabled=True):
            batched_nnpack = base(xte[:64])
    NUM["同一张图放进不同的批"] = {
        "说明": f"取 {n_target} 张测试图，每张和随机抽的 {bs - 1} 张别的图拼成一批，重复 {n_rep} 次，看这张图自己的输出怎么变",
        **spread,
        "eval 模式下单张推理和放进 64 张的批里，logits 的最大差": float(f"{float((alone - batched).abs().max()):.2g}"),
        "同上，但成批那次让卷积走 NNPACK（另一种实现）": float(f"{float((alone - batched_nnpack).abs().max()):.2g}"),
    }

    # 运行统计量被改掉
    x_bgr = to_tensor([letterbox(i)[0] for i in c["te_imgs"]], c["mean"], c["std"], order="bgr")
    feeds = {"和训练同分布的图": xte, "只有空桌面": xte[yte == 0], "只有红杯": xte[yte == 1], "通道顺序是 BGR 的图": x_bgr}
    checkpoints = (1, 5, 20, 100)
    drift = {}
    for k, feed in feeds.items():
        m = copy.deepcopy(base).train()
        gg = torch.Generator().manual_seed(0)
        torch.manual_seed(0)
        row, done = [], 0
        for nb in checkpoints:
            with torch.no_grad():
                for _ in range(nb - done):
                    m(feed[torch.randint(0, len(feed), (8,), generator=gg)])
            done = nb
            row.append(accuracy(predict(copy.deepcopy(m).eval(), xte), yte))
        drift[k] = row
    NUM["运行统计量被改掉"] = {
        "说明": "分类小模型在训练模式下（torch.no_grad() 里）对 n 批图做前向，每批 8 张，然后切回 eval，在原来的测试集上测精度。"
                "BatchNorm 的 momentum 是默认的 0.1。没有反向传播，也没有优化器",
        "前向的批数 n": list(checkpoints), "切回 eval 之前的精度": r3(eval_acc),
        **{k: [r3(v) for v in vs] for k, vs in drift.items()},
    }

    m = copy.deepcopy(base).train()
    torch.manual_seed(0)
    with torch.no_grad():
        a = m(xte[:8])
    torch.manual_seed(0)
    with torch.inference_mode():
        b = m(xte[:8])
    with torch.no_grad():
        ref = base(xte[:8])
    NUM["no_grad 不等于 eval"] = {
        "说明": "同一批 8 张图：模型在训练模式下分别放在 torch.no_grad() 和 torch.inference_mode() 里前向（随机种子相同），和 eval 模式的输出比",
        "no_grad 和 inference_mode 两次输出的最大差": float((a - b).abs().max()),
        "训练模式加 no_grad 的输出和 eval 模式的最大差": r3((a - ref).abs().max()),
    }

    # ResNet-18
    net = resnet()
    syn, _ = image_sets(7, 48, 0)
    xr = to_tensor([imagenet_geom(i) for i in syn], IMN_MEAN, IMN_STD)
    ref, _ = resnet_run(net, xr)
    rn = {}
    for b in (1, 8, 48):
        m = copy.deepcopy(net).train()
        with torch.no_grad():
            out = torch.cat([m(xr[i:i + b]) for i in range(0, len(xr), b)])
        after, _ = resnet_run(m.eval(), xr)
        rn[str(b)] = {"训练模式下的 top-1 一致率": r3((out.argmax(1) == ref.argmax(1)).float().mean()),
                      "之后切回 eval 的 top-1 一致率": r3((after.argmax(1) == ref.argmax(1)).float().mean()),
                      "之后切回 eval 的 logits 相对变化（中位数）": r3(((after - ref).norm(dim=1) / ref.norm(dim=1)).median()),
                      "训练模式下一共前向了几批": math.ceil(len(xr) / b)}
    try:
        with torch.no_grad():
            copy.deepcopy(net).train()(torch.zeros(1, 3, 32, 32))
        err = "没有报错"
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    NUM["ResNet-18 留在训练模式"] = {
        "说明": "ResNet-18（只有 BatchNorm，没有 Dropout）对 48 张合成桌面图按批大小 b 在训练模式下前向一遍（no_grad），和 eval 模式的输出比 top-1；"
                "然后把这个模型切回 eval 再跑一遍。合成图的像素统计和 ImageNet 差得远，所以这是偏极端的情况",
        "批大小": rn,
        "训练模式、批大小 1、输入 32x32 时": err,
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), gridspec_kw={"width_ratios": [1.1, 1.3, 1.1]}, layout="constrained")
    ax = axes[0]
    xs_ = np.arange(len(sizes))
    for (k, vs), col, mk in zip(curves.items(), (S.RED, S.BLUE, S.PURPLE), ("o-", "s-", "^--")):
        ax.plot(xs_, vs, mk, color=col, label=k, ms=5)
    ax.axhline(eval_acc, color=S.GREEN, lw=1.2, ls="--")
    ax.text(len(sizes) - 1, eval_acc + 0.015, "eval 模式", color=S.GREEN, ha="right", fontsize=9)
    ax.set_xticks(xs_)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("推理时的批大小")
    ax.set_ylabel("测试精度")
    ax.set_ylim(0, 1.08)
    ax.set_title("BatchNorm 留在训练模式：\n批大小 1 时精度 %.3f，64 时 %.3f" % (curves["只有 BatchNorm 在训练模式"][0], curves["只有 BatchNorm 在训练模式"][-1]))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=8.5)
    S.clean(ax)
    ax = axes[1]
    for t in range(n_target):
        ax.plot([t] * n_rep, strip[t], ".", color=S.RED, ms=3.5, alpha=0.6)
    with torch.no_grad():
        p_eval = [float(base(xte[t:t + 1]).softmax(1)[0, int(yte[t])]) for t in range(n_target)]
    ax.plot(range(n_target), p_eval, "_", color=S.GREEN, ms=11, mew=2.2, label="eval 模式（只有一个值）")
    ax.plot([], [], ".", color=S.RED, label=f"BatchNorm 在训练模式，{n_rep} 个不同的批")
    ax.set_xlabel(f"{n_target} 张测试图")
    ax.set_xticks([])
    ax.set_ylabel("正确类的概率")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("同一张图，换一批同伴，\n正确类的概率能差 %.2f（中位数）" % spread["只有 BatchNorm 在训练模式"]["正确类概率的最大值减最小值（各图的中位数）"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1, fontsize=8.5)
    S.clean(ax)
    ax = axes[2]
    xs_ = np.arange(len(checkpoints))
    for (k, vs), col, mk in zip(drift.items(), (S.GREEN, S.RED, S.ORANGE, S.PURPLE), ("o-", "s-", "^-", "d-")):
        ax.plot(xs_, vs, mk, color=col, label=k, ms=5)
    ax.set_xticks(xs_)
    ax.set_xticklabels([str(n) for n in checkpoints])
    ax.set_xlabel("训练模式下前向了几批（每批 8 张）")
    ax.set_ylabel("切回 eval 之后的测试精度")
    ax.set_ylim(0, 1.08)
    ax.set_title("没有训练，模型也被改了：\n喂 100 批空桌面后精度 %.3f" % drift["只有空桌面"][-1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5)
    S.clean(ax)
    save(fig, "fig05_推理模式")


# ---------------------------------------------------------------- 表：TorchScript 导出
class AutoRange(nn.Module):
    """把预处理包进模型，并且“自动判断取值范围”：最大值大于 1 就当成 0 到 255。"""

    def __init__(self, net, mean, std):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        if x.max() > 1.0:
            x = x / 255.0
        return self.net((x - self.mean) / self.std)


def table_export():
    c = ctx()
    base, xte, yte = c["nets"][0], c["xte"], c["yte"]
    canv = np.stack([letterbox(i)[0] for i in c["te_imgs"]])
    x255 = torch.from_numpy(canv.transpose(0, 3, 1, 2).astype(np.float32))
    x01 = x255 / 255
    out: dict = {}

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        traced = torch.jit.trace(base, xte[:1])
    out["trace 时的弃用提示"] = sorted({str(r.message) for r in rec if issubclass(r.category, FutureWarning)})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "small.pt")
        traced.save(path)
        loaded = torch.jit.load(path).eval()
        size = os.path.getsize(path)
    ref = predict(base, xte)
    out["小模型 eval 模式下 trace，存盘再载入"] = {
        "文件大小（MB，一位有效数字）": sig(size / 1e6, 1), "trace 用的那一张图上的最大差": float((loaded(xte[:1]) - base(xte[:1])).abs().max()),
        f"全部 {len(xte)} 张测试图上的最大差": float((predict(loaded, xte) - ref).abs().max()),
        "测试精度": r3(accuracy(predict(loaded, xte), yte)),
    }
    net = resnet()
    syn, real = image_sets(7, 16, 16)
    xr = to_tensor([imagenet_geom(i) for i in syn + real], IMN_MEAN, IMN_STD)
    tr_rn = torch.jit.trace(net, xr[:1])
    with torch.inference_mode():
        other = torch.randn(2, 3, 160, 192)
        out["ResNet-18 eval 模式下 trace"] = {
            "32 张图上的最大差": float((torch.cat([tr_rn(xr[i:i + 8]) for i in range(0, 32, 8)]) - resnet_run(net, xr)[0]).abs().max()),
            "换一个输入尺寸（2x3x160x192）的最大差": float((tr_rn(other) - net(other)).abs().max()),
        }

    wrap = AutoRange(base, c["mean"], c["std"]).eval()
    rows = {"不导出，直接运行": {"0 到 255 的输入": r3(accuracy(predict(wrap, x255), yte)), "0 到 1 的输入": r3(accuracy(predict(wrap, x01), yte))}}
    for name, ex in (("用 0 到 255 的样例 trace", x255[:1]), ("用 0 到 1 的样例 trace", x01[:1])):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            tw = torch.jit.trace(wrap, ex)
        rows[name] = {"样例上和原模型的最大差": float((tw(ex) - wrap(ex)).abs().max()),
                      "0 到 255 的输入": r3(accuracy(predict(tw, x255), yte)), "0 到 1 的输入": r3(accuracy(predict(tw, x01), yte)),
                      "trace 时有没有 TracerWarning": any(r.category.__name__ == "TracerWarning" for r in rec)}
    sw = torch.jit.script(wrap)
    rows["torch.jit.script"] = {"0 到 255 的输入": r3(accuracy(predict(sw, x255), yte)), "0 到 1 的输入": r3(accuracy(predict(sw, x01), yte))}
    try:
        torch.export.export(wrap, (x255[:4],))
        rows["torch.export"] = "导出成功"
    except Exception as exc:  # noqa: BLE001
        rows["torch.export"] = f"导出时报错 {type(exc).__name__}"
    try:
        ep = torch.export.export(base, (xte[:4],))
        with torch.inference_mode():
            rows["torch.export（不带分支的小模型）"] = {"4 张图上的最大差": float((ep.module()(xte[:4]) - base(xte[:4])).abs().max())}
    except Exception as exc:  # noqa: BLE001
        rows["torch.export（不带分支的小模型）"] = f"导出时报错 {type(exc).__name__}"
    out["带数据相关分支的模型"] = dict({"说明": "把预处理包进模型，forward 里写了 if x.max() > 1: x = x / 255。数值是测试精度"}, **rows)

    m = copy.deepcopy(base).train()
    torch.manual_seed(0)
    tt = torch.jit.trace(m, xte[:8], check_trace=False).eval()
    with torch.inference_mode():
        o8 = torch.cat([tt(xte[i:i + 8]) for i in range(0, len(xte), 8)])
        o1 = torch.cat([tt(xte[i:i + 1]) for i in range(64)])
        torch.manual_seed(1)
        a = tt(xte[:8])
        torch.manual_seed(2)
        b = tt(xte[:8])
    out["训练模式下 trace，之后再调用 .eval()"] = {
        "批大小 8 的测试精度": r3(accuracy(o8, yte)), "批大小 8 时和原模型 eval 输出的最大差": r3((o8 - ref).abs().max()),
        "批大小 1 的精度（前 64 张）": r3(accuracy(o1, yte[:64])), "同一批调用两次，输出的最大差": r3((a - b).abs().max()),
    }
    NUM["TorchScript 导出"] = dict({"说明": f"torch {torch.__version__.split('+')[0]}。分类小模型（种子 0）和 ResNet-18。torch.jit.trace 用一个样例输入把模型跑一遍，记下执行过的运算"}, **out)


# ---------------------------------------------------------------- 图 6：量化、半精度、剪枝
def state_size(m, half=False):
    sd = {k: (v.half() if half and v.is_floating_point() else v) for k, v in m.state_dict().items()}
    buf = io.BytesIO()
    torch.save(sd, buf)
    raw = buf.getvalue()
    return len(raw), len(zlib.compress(raw, 6))


def fig_compress():
    import torch.nn.utils.prune as prune
    from torchvision.models.quantization import resnet18 as qresnet18

    net = resnet()
    syn, real = image_sets(7, 48, 48)
    xe = to_tensor([imagenet_geom(i) for i in syn + real], IMN_MEAN, IMN_STD)
    cs, cr = image_sets(8, 16, 16)
    xc = to_tensor([imagenet_geom(i) for i in cs + cr], IMN_MEAN, IMN_STD)
    torch.manual_seed(0)
    xnoise = torch.randn(32, 3, 224, 224)
    ref, _ = resnet_run(net, xe)
    torch.backends.quantized.engine = "qnnpack"

    def run(m):
        with torch.inference_mode():
            return torch.cat([m(xe[i:i + 8]) for i in range(0, len(xe), 8)])

    def static_q(cal):
        qm = qresnet18(weights=None, quantize=False)
        qm.load_state_dict(net.state_dict())
        qm.eval()
        qm.fuse_model(is_qat=False)
        qm.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
        torch.ao.quantization.prepare(qm, inplace=True)
        with torch.inference_mode():
            for i in range(0, len(cal), 8):
                qm(cal[i:i + 8])
        torch.ao.quantization.convert(qm, inplace=True)
        return qm

    def pruned(amount):
        pm = copy.deepcopy(net)
        params = [(m, "weight") for m in pm.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
        for m, n in params:
            prune.remove(m, n)
        return pm

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        torch.ao.quantization.quantize_dynamic(nn.Sequential(nn.Linear(4, 4)), {nn.Linear}, dtype=torch.qint8)
    hints = sorted({str(r.message).strip().splitlines()[0] for r in rec if "quantiz" in str(r.message).lower() or "torchao" in str(r.message).lower()})
    models = {
        "float32（原模型）": net,
        "半精度存储（FP16）": copy.deepcopy(net).half().float(),
        "动态量化": torch.ao.quantization.quantize_dynamic(copy.deepcopy(net), {nn.Linear}, dtype=torch.qint8),
        "静态量化，用 32 张同类图校准": static_q(xc),
        "静态量化，用 32 张随机噪声校准": static_q(xnoise),
        "剪掉 50% 的权重": pruned(0.5),
        "剪掉 80% 的权重": pruned(0.8),
    }
    rows = {}
    for k, m in models.items():
        out = run(m)
        raw, zipped = state_size(m, half=(k == "半精度存储（FP16）"))
        agree = (out.argmax(1) == ref.argmax(1)).float()
        rows[k] = {"文件大小（MB）": r3(raw / 1e6), "再用 zlib 压缩后（MB）": r3(zipped / 1e6),
                   "top-1 一致率（合成桌面图）": r3(agree[:48].mean()), "top-1 一致率（真实照片的裁剪）": r3(agree[48:].mean()),
                   "logits 相对变化（中位数）": r4(((out - ref).norm(dim=1) / ref.norm(dim=1)).median()),
                   "logits 最大差": r3((out - ref).abs().max())}
    n_fc = sum(p.numel() for p in net.fc.parameters())
    n_all = sum(p.numel() for p in net.parameters())
    NUM["压缩：大小和输出差别"] = {
        "说明": "ResNet-18（ImageNet 权重）。文件大小是 state_dict 用 torch.save 存下来的字节数。输出差别在 96 张图上和 float32 原模型比（48 张合成桌面图加 48 个真实照片的裁剪，"
                "和“预处理不一致（ResNet-18）”用的是同一批图，一致率同样偏敏感）。"
                "半精度存储：权重转成 FP16 保存，载入后转回 float32 计算。动态量化：torch.ao.quantization.quantize_dynamic，只作用于 nn.Linear。"
                "静态量化：torchvision.models.quantization.resnet18 的结构，融合 Conv-BN-ReLU，qnnpack 默认配置，用 32 张图校准后转换；校准图和测试图不重复。"
                "剪枝：全部卷积和全连接的权重按绝对值全局排序，把最小的一部分置零，不做微调",
        **rows,
        "调用 quantize_dynamic 时的提示（每条的第一行）": hints,
        "全连接层占全部参数的比例": r3(n_fc / n_all),
        "动态量化后文件小了多少（比例）": r3(1 - rows["动态量化"]["文件大小（MB）"] / rows["float32（原模型）"]["文件大小（MB）"]),
        "半精度存储后文件是原来的比例": r3(rows["半精度存储（FP16）"]["文件大小（MB）"] / rows["float32（原模型）"]["文件大小（MB）"]),
        "静态量化后文件是原来的几分之一": round(rows["float32（原模型）"]["文件大小（MB）"] / rows["静态量化，用 32 张同类图校准"]["文件大小（MB）"], 2),
    }

    mlp = nn.Sequential(nn.Flatten(), nn.Linear(3 * DST * DST, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, K)).eval()
    torch.manual_seed(0)
    for p in mlp.parameters():
        nn.init.normal_(p, std=0.02)
    mlp_q = torch.ao.quantization.quantize_dynamic(copy.deepcopy(mlp), {nn.Linear}, dtype=torch.qint8)
    xm = torch.randn(64, 3, DST, DST)
    with torch.inference_mode():
        a, b = mlp(xm), mlp_q(xm)
    NUM["压缩：全连接模型的动态量化"] = {
        "说明": f"一个只有三层全连接的模型（{3 * DST * DST}-1024-1024-{K}，随机权重，没有训练），只用来看动态量化对大小的作用。输出差别在 64 个随机输入上比",
        "float32 文件大小（MB）": r3(state_size(mlp)[0] / 1e6), "动态量化后（MB）": r3(state_size(mlp_q)[0] / 1e6),
        "缩小到原来的几分之一": r3(state_size(mlp)[0] / state_size(mlp_q)[0]),
        "输出相对变化（中位数）": r4(((a - b).norm(dim=1) / a.norm(dim=1)).median()),
    }

    def measure():
        x1 = xe[:1]
        names = list(models)
        ts = {k: [] for k in names}
        with torch.inference_mode():
            for k in names:
                models[k](x1)
                models[k](x1)
            for _ in range(15):
                for k in names:
                    t = time.perf_counter()
                    models[k](x1)
                    ts[k].append((time.perf_counter() - t) * 1e3)
            tm = {"float32": [], "动态量化": []}
            for _ in range(200):
                for k, m in (("float32", mlp), ("动态量化", mlp_q)):
                    t = time.perf_counter()
                    m(xm[:1])
                    tm[k].append((time.perf_counter() - t) * 1e3)
        med = {k: sig(np.median(v)) for k, v in ts.items()}
        base_t = np.median(ts["float32（原模型）"])
        return {"说明": "ResNet-18 各版本，批大小 1，224x224。各版本轮流测，每个 15 次取中位数（ms）。全连接模型轮流测 200 次。" + machine_note(),
                "ResNet-18 单张耗时（ms）": med,
                "是 float32 的几倍": {k: sig(np.median(v) / base_t) for k, v in ts.items()},
                "全连接模型单次耗时（ms）": {k: sig(np.median(v)) for k, v in tm.items()},
                "全连接模型动态量化后是原来的几倍": sig(np.median(tm["动态量化"]) / np.median(tm["float32"]))}

    tim = NUM["计时：压缩后的耗时"] = S.pinned(CH, "计时：压缩后的耗时", measure, RETIME)

    names = list(models)
    short = ["float32", "FP16\n存储", "动态\n量化", "静态量化\n同类图\n校准", "静态量化\n噪声\n校准", "剪枝\n50%", "剪枝\n80%"]
    cols = [S.GREEN, S.BLUE, S.BLUE, S.ORANGE, S.ORANGE, S.PURPLE, S.PURPLE]
    xs_ = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), layout="constrained")
    ax = axes[0]
    vals = [rows[k]["文件大小（MB）"] for k in names]
    ax.bar(xs_, vals, 0.62, color=cols)
    for x, v in zip(xs_, vals):
        ax.text(x, v + 0.8, f"{v:.1f}", ha="center", fontsize=8.5)
    ax.set_ylabel("文件大小（MB）")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_title("大小：动态量化 %.1f MB，静态量化 %.1f MB，\n剪枝后还是 %.1f MB" % (vals[2], vals[3], vals[5]))
    ax = axes[1]
    w = 0.36
    ax.bar(xs_ - w / 2, [rows[k]["top-1 一致率（合成桌面图）"] for k in names], w, color=S.BLUE, label="合成桌面图")
    ax.bar(xs_ + w / 2, [rows[k]["top-1 一致率（真实照片的裁剪）"] for k in names], w, color=S.ORANGE, label="真实照片的裁剪")
    ax.set_ylabel("top-1 和原模型一致的比例")
    ax.set_ylim(0, 1.12)
    ax.set_title("输出：校准用噪声时，照片裁剪的一致率 %.2f；\n用同类图校准是 %.2f" % (rows[names[4]]["top-1 一致率（真实照片的裁剪）"], rows[names[3]]["top-1 一致率（真实照片的裁剪）"]))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=8.5, ncol=2)
    ax = axes[2]
    vals = [tim["ResNet-18 单张耗时（ms）"][k] for k in names]
    ax.bar(xs_, vals, 0.62, color=cols)
    for x, v in zip(xs_, vals):
        ax.text(x, v + max(vals) * 0.015, f"{v:g}", ha="center", fontsize=8.5)
    ax.set_ylabel("单张耗时（ms，这台 Mac 的 CPU）")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_title("耗时：静态量化是 float32 的 %.2g 倍，\n剪枝 %.2g 倍（这台机器）" % (tim["是 float32 的几倍"][names[3]], tim["是 float32 的几倍"][names[5]]))
    for ax in axes:
        ax.set_xticks(xs_)
        ax.set_xticklabels(short, fontsize=8.5)
        S.clean(ax)
    save(fig, "fig06_量化与压缩")


# ---------------------------------------------------------------- 其余数字
def numbers_misc():
    logits = torch.zeros(4, 3)
    try:
        nn.functional.cross_entropy(logits, torch.tensor([1, 2, 3, 1]))
        msg = "没有报错"
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
    c = ctx()
    pred = predict(c["nets"][0], c["xte"]).argmax(1)
    NUM["标签编号错一位"] = {
        "说明": "两种情况。训练时：3 类的输出配上从 1 开始编号的标签（出现了标签 3），在 CPU 上调用 cross_entropy。"
                "部署时：模型输出的编号从 0 开始，类别表却按从 1 开始去查（等于把预测编号加 1 再和标签比），用分类小模型（种子 0）在测试集上算精度",
        "训练时 CPU 上的表现": msg,
        "部署时类别表错一位的精度": r3(float(((pred + 1) == c["yte"]).float().mean())),
    }
    meta = mod.letterbox_metadata((100, 200), (320, 320))
    def meta_of(src, dst):
        m = mod.letterbox_metadata(src, dst)
        return {"scale": r4(m["scale"]), "resized_hw": list(m["resized_hw"]), "pad_xy": list(m["pad_xy"])}

    NUM["作业自检用例"] = {
        "说明": "tests/check_progress.py 里的几个用例，用参考实现算出来的结果。作业的文档字符串规定取整用 Python 内置的 round（正好落在 .5 时取偶数），"
                "多出来的 1 个像素放右侧或下侧；后四个用例分别是缩放后带小数并且加边为奇数、缩放后的高正好是 360.5、361.5、缩放后的宽正好是 362.5",
        "letterbox_metadata((100, 200), (320, 320))": {"scale": meta["scale"], "resized_hw": list(meta["resized_hw"]), "pad_xy": list(meta["pad_xy"])},
        "letterbox_metadata((100, 300), (326, 320))": meta_of((100, 300), (326, 320)),
        "letterbox_metadata((721, 1280), (640, 640))": meta_of((721, 1280), (640, 640)),
        "letterbox_metadata((723, 1280), (640, 640))": meta_of((723, 1280), (640, 640)),
        "letterbox_metadata((1280, 725), (640, 640))": meta_of((1280, 725), (640, 640)),
        "boxes_to_original 用例的输出": mod.boxes_to_original(np.array([[10.0, 20.0, 310.0, 220.0], [-5.0, 0.0, 400.0, 300.0]]),
                                                    {"scale": 2.0, "resized_hw": (200, 300), "pad_xy": (10, 20)}, (100, 150)).tolist(),
    }
    NUM["运行环境"] = {
        "说明": "生成本文件时的库版本。除了“计时”开头的各组，其余数字重复运行不变",
        "torch": torch.__version__.split("+")[0], "torchvision": torchvision.__version__.split("+")[0], "OpenCV": cv2.__version__,
        "numpy": np.__version__, "Pillow": Image.__version__, "线程数": torch.get_num_threads(),
        "卷积是否走 NNPACK": "除了“计时：批大小”和“同一张图放进不同的批”里注明的地方，全程关闭（torch.backends.nnpack.set_flags(False)）。"
                         "这台机器上 PyTorch 默认开着 NNPACK，批大小 16 以上时比关掉慢（见“计时：批大小”），所以关掉它来缩短训练小模型的时间",
        "onnx 和 onnxruntime": "本机没有安装，脚本没有导出也没有运行 ONNX",
    }


def main() -> None:
    S.setup()
    warnings.simplefilter("ignore")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.nnpack.set_flags(False)
    t_all = time.perf_counter()
    for fn in (fig_preprocess, fig_data, fig_timing, fig_resize_letterbox, fig_eval_mode, table_export, fig_compress, numbers_misc):
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
