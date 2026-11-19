"""第 31 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch31.py

输出：<主分支>/31_部署与加速/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。

全部实验用代码生成的 32x32 缺陷小图和一个很小的卷积网络，在 CPU 上用 PyTorch 和 NumPy 跑，不下载任何数据和权重，
也不需要 onnx、onnxruntime、TensorRT。量化参数、量化与反量化、逐通道量化、BN 折叠、按节拍选批大小调用本章作业的参考实现；
整数量化是用“量化再反量化”在浮点里模拟的，不是真的整数推理引擎。
耗时默认沿用上次的记录，设置 CH31_RETIME=1 才重测。
"""

from __future__ import annotations

import copy
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import warnings
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

CH = "31_部署与加速"
hw = S.load_solution(CH, "31-1_量化与部署零件.py")
hw19 = S.load_solution("19_异常检测与深度缺陷检测", "19-1_PatchCore核心.py")  # 第 19 章的 auroc 和 threshold_at_escape
NUM: dict = {}
TIMES: dict = {}
RETIME = "CH31_RETIME"
ONLY = os.environ.get("CH31_ONLY")          # 调试用：只跑名字里含这个字符串的函数（不写 JSON）
SEEDS = (0, 1)
SIZE = 32
CLASSES = ["良品", "亮点", "暗点", "横划痕", "竖划痕"]
K = len(CLASSES)
AMP = (0.06, 0.30)
WEAK = 0.10
NOISE = 0.02
WIDTHS = (8, 16, 32, 32)
STEPS, BATCH = 600, 64
N_VAL, N_TEST, N_CAL = 1500, 3000, 256
MAX_ESCAPE = 0.02
YY, XX = torch.meshgrid(torch.arange(SIZE, dtype=torch.float32), torch.arange(SIZE, dtype=torch.float32), indexing="ij")
DATA_NOTE = (f"{SIZE}x{SIZE} 的 8 位灰度小图：背景是竖直的线条（周期 8 px，明暗相差 0.12，相位随机，整体亮度乘 0.9 到 1.1 之间的随机数），像素噪声标准差 {NOISE}。"
             f"一半是良品，另一半带一个缺陷，4 种缺陷各占四分之一：亮点、暗点（标准差 1.5 px 的高斯斑）、横划痕、竖划痕（长 7 到 11 px 的细线，亮暗随机）；"
             f"缺陷幅度在 {AMP[0]} 到 {AMP[1]} 之间均匀取，幅度小于 {WEAK} 的叫弱缺陷（占缺陷的六分之一左右）。图像存成 8 位（0 到 255 的整数再除以 255）")
MODEL_NOTE = (f"模型是 4 个 Conv-BN-ReLU（3x3，通道数 {WIDTHS[0]}、{WIDTHS[1]}、{WIDTHS[2]}、{WIDTHS[3]}，第 2、3 个步长 2）、全局平均池化、全连接，输出 {K} 类；"
              f"输入先减 0.5 再除以 0.25。AdamW，学习率先升后降，批大小 {BATCH}，训练 {STEPS} 步，每一步都是新生成的图")
EVAL_NOTE = (f"缺陷分数取 1 减去“良品”这一类的概率。阈值按第 19 章的 threshold_at_escape 在验证集（{N_VAL} 张）上定：漏检率不超过 {MAX_ESCAPE:.0%} 的最大阈值；"
             f"漏检率和过检率在测试集（{N_TEST} 张）上量。“沿用原阈值”指用浮点模型定下的阈值去判压缩后模型的分数，“重新定阈值”指用压缩后的模型在验证集上重定")
QUANT_NOTE = ("量化是在浮点里模拟的：权重量化再反量化后写回卷积，激活值在每个 ReLU 之后量化再反量化；BN 先折叠进卷积；输入图像本身是 8 位的，不另外量化；偏置不量化。"
              "“仔细”指权重逐通道对称量化、激活值按均方误差最小定范围；“直接”指权重逐张量对称量化、激活值按最小最大值定范围。"
              f"激活值的范围用 {N_CAL} 张校准图统计（和训练、验证、测试都不重复）")


def r3(v):
    return round(float(v), 3)


def r4(v):
    return round(float(v), 4)


def sig(v, n=2):
    """保留 n 位有效数字。耗时一律存两位。"""
    return float(f"{float(v):.{n}g}")


def gen(*seed):
    return torch.Generator().manual_seed(int(sum(s * 1000 ** i for i, s in enumerate(seed))))


# ---------------------------------------------------------------- 数据
def make_data(n, g, p_good=0.5, amp=AMP):
    """返回 (图像 (n,1,32,32)，类别 (n,)，缺陷幅度 (n,)，良品的幅度记 0)。"""
    ph = torch.rand(n, generator=g) * 8
    bg = 0.5 + 0.06 * torch.sign(torch.sin(2 * math.pi * (XX[None] + ph[:, None, None]) / 8))
    bright = 0.9 + 0.2 * torch.rand(n, 1, 1, generator=g)
    y = torch.where(torch.rand(n, generator=g) < p_good, 0, torch.randint(1, K, (n,), generator=g))
    a = amp[0] + (amp[1] - amp[0]) * torch.rand(n, generator=g)
    cy = torch.randint(5, SIZE - 5, (n,), generator=g).float()[:, None, None]
    cx = torch.randint(5, SIZE - 5, (n,), generator=g).float()[:, None, None]
    blob = torch.exp(-((YY[None] - cy) ** 2 + (XX[None] - cx) ** 2) / (2 * 1.5 ** 2))
    half = 3 + torch.randint(0, 3, (n,), generator=g).float()[:, None, None]
    hline = torch.exp(-((YY[None] - cy) ** 2) / (2 * 0.7 ** 2)) * ((XX[None] - cx).abs() <= half)
    vline = torch.exp(-((XX[None] - cx) ** 2) / (2 * 0.7 ** 2)) * ((YY[None] - cy).abs() <= half)
    sign = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)[:, None, None]
    shape = torch.zeros(n, SIZE, SIZE)
    for k, s in ((1, blob), (2, -blob), (3, hline * sign), (4, vline * sign)):
        shape = torch.where((y == k)[:, None, None], s, shape)
    img = bg * bright + a[:, None, None] * shape + NOISE * torch.randn(n, SIZE, SIZE, generator=g)
    img = torch.round(img.clamp(0, 1) * 255) / 255
    return img[:, None], y, torch.where(y == 0, torch.zeros_like(a), a)


# ---------------------------------------------------------------- 模型和训练
def cbr(i, o, stride=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU())


class Net(nn.Module):
    def __init__(self, w=WIDTHS, k=K):
        super().__init__()
        self.features = nn.Sequential(cbr(1, w[0]), cbr(w[0], w[1], 2), cbr(w[1], w[2], 2), cbr(w[2], w[3]))
        self.fc = nn.Linear(w[3], k)
        self.widths = tuple(w)

    def forward(self, x):
        return self.fc(self.features((x - 0.5) / 0.25).mean((2, 3)))


def train(net, seed, steps=STEPS, lr=4e-3, pool=None, soft=None, temp=1.0, after_step=None):
    """pool 为 None 时每一步都是新生成的图；给了 pool=(图, 标签) 就只在这批图里抽样。
    soft 不为 None 时是蒸馏：soft 是教师在 pool 的每张图上的输出（事先算好），损失只有
    “学生和教师在温度 temp 下的分布的 KL 散度乘 temp 的平方”这一项，不用标签。"""
    torch.manual_seed(seed)
    g = gen(seed, 1)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.15)
    net.train()
    for _ in range(steps):
        if pool is None:
            x, y, _ = make_data(BATCH, g)
        else:
            idx = torch.randint(0, len(pool[0]), (BATCH,), generator=g)
            x, y = pool[0][idx], pool[1][idx]
        out = net(x)
        if soft is None:
            loss = F.cross_entropy(out, y)
        else:
            loss = F.kl_div(F.log_softmax(out / temp, -1), F.softmax(soft[idx] / temp, -1), reduction="batchmean") * temp * temp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if after_step is not None:
            after_step(net)
    return net.eval()


@torch.no_grad()
def predict(net, x, half=False):
    if half:
        return torch.cat([net(x[i:i + 1000].half()).float() for i in range(0, len(x), 1000)])
    return torch.cat([net(x[i:i + 1000]) for i in range(0, len(x), 1000)])


def count_cost(net, size=SIZE):
    """参数个数，以及一次前向里卷积和全连接的乘加次数。"""
    macs = 0
    hooks = []

    def hook(m, inp, out):
        nonlocal macs
        if isinstance(m, nn.Conv2d):
            macs += out.shape[2] * out.shape[3] * m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        else:
            macs += m.in_features * m.out_features

    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        net(torch.zeros(1, 1, size, size))
    for h in hooks:
        h.remove()
    return sum(p.numel() for p in net.parameters()), int(macs)


# ---------------------------------------------------------------- BN 折叠后的网络和量化的模拟
class Folded(nn.Module):
    """BN 折叠进卷积之后的同一个网络：4 个带偏置的卷积，每个后面一个 ReLU，再接全局平均池化和全连接。
    act_q[i] 不是 None 时，第 i 个 ReLU 之后做一次量化再反量化。"""

    def __init__(self, net):
        super().__init__()
        convs = []
        for blk in net.features:
            conv, bn = blk[0], blk[1]
            w, b = hw.fold_bn_into_conv(conv.weight, conv.bias, bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps)
            c = nn.Conv2d(conv.in_channels, conv.out_channels, 3, conv.stride, 1, bias=True).to(w.dtype)
            c.weight.data.copy_(w)
            c.bias.data.copy_(b)
            convs.append(c)
        self.convs = nn.ModuleList(convs)
        self.fc = copy.deepcopy(net.fc)
        self.act_q = [None] * len(convs)

    def forward(self, x, taps=None):
        x = (x - 0.5) / 0.25
        for i, c in enumerate(self.convs):
            x = F.relu(c(x))
            if taps is not None:
                taps.append(x)
            if self.act_q[i] is not None:
                scale, zp, qmin, qmax = self.act_q[i]
                x = (torch.clamp(torch.round(x / scale) + zp, qmin, qmax) - zp) * scale
        return self.fc(x.mean((2, 3)))


def quant_weight(w, bits, per_channel):
    """对称量化一个权重张量，返回反量化之后的浮点权重。两种粒度都调用作业的函数。"""
    wn = w.detach().numpy()
    if per_channel:
        q, s = hw.quantize_weight_per_channel(wn, bits)
        return torch.from_numpy(q * s.reshape((-1,) + (1,) * (wn.ndim - 1))).to(w.dtype)
    scale, zp, qmin, qmax = hw.quant_params(float(wn.min()), float(wn.max()), bits, symmetric=True)
    return torch.from_numpy(hw.quantize_dequantize(wn, scale, zp, qmin, qmax)[1]).to(w.dtype)


def quant_err(v, hi, bits):
    """非负数据 v 按 [0, hi] 做无符号量化后的均方误差。"""
    s, zp, qmin, qmax = hw.quant_params(0.0, hi, bits)
    return float(((hw.quantize_dequantize(v, s, zp, qmin, qmax)[1] - v) ** 2).mean())


def calib_range(v, method, bits):
    """v 是一维的非负激活样本（ReLU 之后）。返回量化范围的上界。
    minmax：最大值；p99.9、p99.99：百分位；mse：在最大值的 5% 到 100% 之间找量化均方误差最小的上界。"""
    v = np.asarray(v, dtype=np.float64)
    if method == "minmax":
        return float(v.max())
    if method.startswith("p"):
        return float(np.percentile(v, float(method[1:])))
    cands = float(v.max()) * np.linspace(0.05, 1.0, 96)
    return float(cands[int(np.argmin([quant_err(v, c, bits) for c in cands]))])


def act_samples(fnet, calib_x, sub=40000):
    """每个 ReLU 之后的激活值，各随机抽 sub 个。"""
    taps = []
    with torch.no_grad():
        fnet(calib_x, taps)
    rng = np.random.default_rng(0)
    return [t.reshape(-1).numpy()[rng.choice(t.numel(), min(sub, t.numel()), replace=False)] for t in taps]


def ptq(fnet, calib_x, w_bits=8, a_bits=8, per_channel=True, method="mse", layers_w=None, layers_a=None, act_hi=None):
    """训练后量化，返回一个新的 Folded。w_bits 或 a_bits 为 None 表示那一部分保持浮点；
    layers_w、layers_a 给出只量化哪几层（敏感度分析），None 表示全部；act_hi 可以直接给出各层激活值的上界。"""
    q = copy.deepcopy(fnet)
    q.act_q = [None] * len(q.convs)
    n = len(q.convs)
    if w_bits is not None:
        for i in (range(n + 1) if layers_w is None else layers_w):
            m = q.convs[i] if i < n else q.fc
            m.weight.data.copy_(quant_weight(m.weight, w_bits, per_channel))
    if a_bits is not None:
        samples = None if act_hi is not None else act_samples(fnet, calib_x)
        for i in (range(n) if layers_a is None else layers_a):
            hi = act_hi[i] if act_hi is not None else calib_range(samples[i], method, a_bits)
            q.act_q[i] = hw.quant_params(0.0, hi, a_bits)
    return q


# ---------------------------------------------------------------- 评价
def score_of(logits):
    p = F.softmax(logits.double(), -1).numpy()
    return 1.0 - p[:, 0], p.argmax(1)


def margin_of(logits):
    """画分布用：缺陷四类的 logsumexp 减去良品的 logit，等于缺陷分数的对数几率。"""
    lg = logits.double()
    return (torch.logsumexp(lg[:, 1:], 1) - lg[:, 0]).numpy()


_CTX: dict = {}


def ctx():
    """数据、两个种子的浮点模型、它们折叠后的版本、浮点模型在验证集上定的阈值。只算一次。"""
    if _CTX:
        return _CTX
    xva, yva, ava = make_data(N_VAL, gen(6, 6))
    xte, yte, ate = make_data(N_TEST, gen(7, 7))
    xcal, ycal, _ = make_data(N_CAL, gen(8, 8))
    nets = [train(Net(), s) for s in SEEDS]
    folded = [Folded(n).eval() for n in nets]
    _CTX.update(xva=xva, yva=yva.numpy(), ava=ava.numpy(), xte=xte, yte=yte.numpy(), ate=ate.numpy(), xcal=xcal, ycal=ycal.numpy(), nets=nets, folded=folded)
    _CTX["ref"] = [predict(f, xte) for f in folded]
    _CTX["thr"] = []
    for f in folded:
        sv, _ = score_of(predict(f, xva))
        _CTX["thr"].append(hw19.threshold_at_escape(sv[_CTX["yva"] == 0], sv[_CTX["yva"] > 0], MAX_ESCAPE)[0])
    return _CTX


def evaluate(model, si, full=True, half=False):
    """si 是种子的下标（决定沿用哪个浮点模型的阈值）。full=False 时只算精度和 AUROC，不动验证集。"""
    c = ctx()
    lt = predict(model, c["xte"], half)
    st, pred = score_of(lt)
    y, a = c["yte"], c["ate"]
    bad, weak, strong = y > 0, (y > 0) & (a < WEAK), (y > 0) & (a >= WEAK)
    out = {"分类精度": float((pred == y).mean()), "AUROC": hw19.auroc(st, bad.astype(int)),
           "logits 和浮点模型的最大差": float((lt - c["ref"][si]).abs().max()), "logits 和浮点模型的差的中位数": float((lt - c["ref"][si]).abs().median()),
           "预测类别和浮点模型一致的比例": float((pred == score_of(c["ref"][si])[1]).mean()), "_logits": lt}
    if not full:
        return out
    thr = c["thr"][si]
    out.update({"沿用原阈值：过检率": float((st[~bad] >= thr).mean()), "沿用原阈值：漏检率": float((st[bad] < thr).mean()),
                "沿用原阈值：弱缺陷漏检率": float((st[weak] < thr).mean()), "沿用原阈值：强缺陷漏检率": float((st[strong] < thr).mean())})
    sv, _ = score_of(predict(model, c["xva"], half))
    new = hw19.threshold_at_escape(sv[c["yva"] == 0], sv[c["yva"] > 0], MAX_ESCAPE)[0]
    out.update({"重新定阈值：过检率": float((st[~bad] >= new).mean()), "重新定阈值：漏检率": float((st[bad] < new).mean()),
                "重新定阈值：弱缺陷漏检率": float((st[weak] < new).mean())})
    return out


_EVAL: dict = {}


def eval_cached(key, make, full=True):
    """同一个配置只评价一次。make(si) 返回要评价的模型。返回 [种子 0 的结果, 种子 1 的结果]。"""
    if (key, full) not in _EVAL:
        _EVAL[(key, full)] = [evaluate(make(si), si, full) for si in range(len(SEEDS))]
    return _EVAL[(key, full)]


def avg(rows, name):
    return float(np.mean([r[name] for r in rows]))


def pack(rows, names=None, digits=4):
    """把两个种子的结果整理成 {指标: 平均值, 各种子: {...}}。"""
    names = names or [k for k in rows[0] if not k.startswith("_")]
    out = {n: round(avg(rows, n), digits) for n in names}
    out["各种子"] = {n: [round(float(r[n]), digits) for r in rows] for n in names}
    return out


def ptq_rows(w_bits, a_bits, per_channel, method, full=True):
    c = ctx()
    return eval_cached(("ptq", w_bits, a_bits, per_channel, method), lambda si: ptq(c["folded"][si], c["xcal"], w_bits, a_bits, per_channel, method), full)


def float_rows():
    c = ctx()
    return eval_cached(("float",), lambda si: c["folded"][si])


MAIN_COLS = ["分类精度", "AUROC", "沿用原阈值：过检率", "沿用原阈值：漏检率", "沿用原阈值：弱缺陷漏检率", "沿用原阈值：强缺陷漏检率",
             "重新定阈值：过检率", "重新定阈值：漏检率", "重新定阈值：弱缺陷漏检率", "logits 和浮点模型的最大差", "logits 和浮点模型的差的中位数",
             "预测类别和浮点模型一致的比例"]


# ---------------------------------------------------------------- 图 2：量化的基本量
def rms(v):
    return float(np.sqrt(np.mean(np.square(v))))


def fig_basics():
    rng = np.random.default_rng(0)
    lo, hi = -1.0, 3.0
    ex = {}
    for name, bits, sym in (("3 位，非对称", 3, False), ("8 位，非对称", 8, False), ("8 位，对称", 8, True)):
        s, zp, qmin, qmax = hw.quant_params(lo, hi, bits, sym)
        ex[name] = {"缩放系数": r4(s), "零点": zp, "整数范围": [qmin, qmax], "能表示的最小实数": r4((qmin - zp) * s), "能表示的最大实数": r4((qmax - zp) * s),
                    "没被截断时的最大误差（半格）": r4(s / 2), "均匀量化噪声的标准差（一格除以根号 12）": r4(s / math.sqrt(12))}
    NUM["量化参数的例子"] = {
        "说明": f"实数范围 [{lo}, {hi}]，用作业的 quant_params 算。非对称用无符号整数，对称用有符号整数的窄范围（8 位是 -127 到 127）。"
              "零点取整之后，能表示的实数范围会相对 [-1, 3] 整体挪动不到半格",
        **ex, "3 位时零点取整前的值": r4(0 - lo / ((hi - lo) / 7)), "8 位时零点取整前的值": r4(0 - lo / ((hi - lo) / 255)),
    }

    n = 200000
    gauss = rng.normal(size=n)
    relu = np.abs(rng.normal(size=n))
    bits_list = [2, 3, 4, 5, 6, 7, 8]
    rows = {"高斯数据，对称": [], "高斯数据，理论值": [], "非负数据，非对称": [], "非负数据，对称": []}
    for b in bits_list:
        p = hw.quant_params(gauss.min(), gauss.max(), b, True)
        rows["高斯数据，对称"].append(rms(hw.quantize_dequantize(gauss, *p)[1] - gauss) / rms(gauss))
        rows["高斯数据，理论值"].append(p[0] / math.sqrt(12) / rms(gauss))
        for name, sym in (("非负数据，非对称", False), ("非负数据，对称", True)):
            p = hw.quant_params(relu.min(), relu.max(), b, sym)
            rows[name].append(rms(hw.quantize_dequantize(relu, *p)[1] - relu) / rms(relu))
    ratio = [a / b for a, b in zip(rows["非负数据，对称"], rows["非负数据，非对称"])]
    per_bit = [rows["高斯数据，对称"][i] / rows["高斯数据，对称"][i + 1] for i in range(len(bits_list) - 1)]
    NUM["位数与误差"] = {
        "说明": f"{n} 个标准正态的样本（像权重），和它们的绝对值（像 ReLU 之后的激活值）。范围取样本的最小最大值，用作业的 quant_params 和 quantize_dequantize。"
              "相对误差 = 量化误差的均方根 / 数据本身的均方根。理论值是“一格除以根号 12”再除以数据的均方根，假设误差在一格内均匀分布、没有截断",
        "位数": bits_list, **{k: [float(f"{v:.3g}") for v in vals] for k, vals in rows.items()},
        "非负数据上对称是非对称的几倍": [r3(v) for v in ratio],
        "高斯数据上位数每减 1，误差变成几倍（8 到 7、7 到 6……3 到 2）": [r3(v) for v in per_bit[::-1]],
        "每多 1 位信噪比提高多少分贝（20 lg 2）": r3(20 * math.log10(2)),
        "高斯数据上 4 位的误差是 8 位的几倍": sig(rows["高斯数据，对称"][2] / rows["高斯数据，对称"][-1]),
        "8 位、高斯数据的信噪比（分贝）": r3(-20 * math.log10(rows["高斯数据，对称"][-1])),
        "4 位、高斯数据的信噪比（分贝）": r3(-20 * math.log10(rows["高斯数据，对称"][2])),
    }

    out_ch = 16
    base = rng.normal(size=(out_ch, 8, 3, 3))
    spreads = [1, 3, 10, 30, 100]
    table, curves = {}, {}
    for spread in spreads:
        stds = spread ** (-np.arange(out_ch) / (out_ch - 1))
        w = base * stds[:, None, None, None]
        for bits in (8, 4):
            p = hw.quant_params(w.min(), w.max(), bits, True)
            q_t, w_t = hw.quantize_dequantize(w, *p)
            q_c, s_c = hw.quantize_weight_per_channel(w, bits)
            w_c = q_c * s_c[:, None, None, None]
            rel_t = np.array([rms(w_t[o] - w[o]) / rms(w[o]) for o in range(out_ch)])
            rel_c = np.array([rms(w_c[o] - w[o]) / rms(w[o]) for o in range(out_ch)])
            curves[(spread, bits)] = (stds, rel_t, rel_c)
            if bits == 8:
                table[f"范围相差 {spread} 倍"] = {
                    "逐张量：范围最大的通道的相对误差": float(f"{rel_t[0]:.3g}"), "逐张量：范围最小的通道的相对误差": float(f"{rel_t[-1]:.3g}"),
                    "逐张量：范围最小的通道用到了几个不同的整数": int(len(np.unique(q_t[-1]))),
                    "逐通道：范围最大的通道的相对误差": float(f"{rel_c[0]:.3g}"), "逐通道：范围最小的通道的相对误差": float(f"{rel_c[-1]:.3g}"),
                    "逐通道：范围最小的通道用到了几个不同的整数": int(len(np.unique(q_c[-1]))),
                    "逐张量是逐通道的几倍（各通道相对误差的平均）": float(f"{rel_t.mean() / rel_c.mean():.3g}"),
                }
    NUM["逐张量与逐通道（合成权重）"] = {
        "说明": f"一个 {out_ch}x8x3x3 的卷积权重，每个输出通道 72 个高斯随机数，各通道的标准差从 1 按等比数列降到 1/R（R 是“范围相差几倍”）。8 位对称量化。"
              "逐张量：整个张量共用一个缩放系数（quant_params 加 quantize_dequantize）；逐通道：每个输出通道各用各的（quantize_weight_per_channel）。"
              "相对误差 = 这个通道量化误差的均方根 / 这个通道权重的均方根",
        **table,
    }

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3), layout="constrained")
    s3, zp3, qmin3, qmax3 = hw.quant_params(lo, hi, 3)
    xs = np.linspace(-2.0, 4.0, 1201)
    _, xh = hw.quantize_dequantize(xs, s3, zp3, qmin3, qmax3)
    ax[0].plot(xs, xs, color=S.MUTED, lw=1, ls=":", label="不量化")
    ax[0].plot(xs, xh, color=S.BLUE, lw=1.8, label="量化再反量化（3 位，8 个值）")
    ax[0].plot(xs, xh - xs, color=S.RED, lw=1.2, label="误差")
    for v in (lo, hi):
        ax[0].axvline(v, color=S.MUTED, lw=0.9, ls="--")
    ax[0].plot([0], [0], "o", color=S.ORANGE, ms=6, label=f"实数 0 落在整数 {zp3} 上，没有误差")
    ax[0].set_xlabel("量化前的实数")
    ax[0].set_ylabel("量化再反量化之后")
    ax[0].set_title("范围 [-1, 3] 内误差不超过半格，范围外被截断")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    S.clean(ax[0])
    ax[1].plot(bits_list, rows["高斯数据，对称"], "o-", color=S.BLUE, lw=1.8, label="高斯数据，对称量化")
    ax[1].plot(bits_list, rows["高斯数据，理论值"], "x:", color=S.INK, lw=1, ms=5, label="理论值：一格 / 根号 12")
    ax[1].plot(bits_list, rows["非负数据，非对称"], "s-", color=S.GREEN, lw=1.8, label="非负数据，非对称量化")
    ax[1].plot(bits_list, rows["非负数据，对称"], "s--", color=S.ORANGE, lw=1.8, label="非负数据，对称量化")
    ax[1].set_yscale("log")
    ax[1].set_yticks([0.003, 0.01, 0.03, 0.1, 0.3], ["0.003", "0.01", "0.03", "0.1", "0.3"])
    ax[1].minorticks_off()
    ax[1].set_xticks(bits_list)
    ax[1].set_xlabel("位数")
    ax[1].set_ylabel("相对误差（误差均方根 / 数据均方根）")
    ax[1].set_title("每少 1 位误差约翻倍；非负数据用对称量化，误差是非对称的 2 倍", fontsize=10.5)
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    S.clean(ax[1])
    for bits, ls in ((8, "-"), (4, "--")):
        stds, rel_t, rel_c = curves[(100, bits)]
        ax[2].plot(1 / stds, rel_t, "o" + ls, color=S.RED, lw=1.6, ms=4, label=f"逐张量，{bits} 位")
        ax[2].plot(1 / stds, rel_c, "s" + ls, color=S.BLUE, lw=1.6, ms=4, label=f"逐通道，{bits} 位")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    ax[2].set_yticks([0.01, 0.1, 1], ["0.01", "0.1", "1"])
    ax[2].minorticks_off()
    ax[2].set_xlabel("最大的通道范围是这个通道的几倍")
    ax[2].set_ylabel("这个通道的相对误差")
    ax[2].set_title("逐张量：通道的范围小几倍，相对误差就大几倍；逐通道：各通道一样", fontsize=10.5)
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    S.clean(ax[2])
    S.save(fig, CH, "fig02_量化的基本量")


# ---------------------------------------------------------------- 图 3：校准
CAL_METHODS = [("minmax", "最小最大值"), ("p99.99", "99.99% 分位"), ("p99.9", "99.9% 分位"), ("mse", "均方误差最小")]


def fig_calibration():
    c = ctx()
    rng = np.random.default_rng(1)
    n, n_out = 100000, 20
    bulk = np.abs(rng.normal(size=n))
    mags = [5, 10, 20, 50, 100, 200]
    syn = {}
    for bits in (8, 4):
        for key, _ in CAL_METHODS:
            errs, keeps, his = [], [], []
            for mag in mags:
                v = bulk.copy()
                v[:n_out] = mag
                hi = calib_range(v, key, bits)
                s, zp, qmin, qmax = hw.quant_params(0.0, hi, bits)
                vh = hw.quantize_dequantize(v, s, zp, qmin, qmax)[1]
                errs.append(rms((vh - v)[n_out:]) / rms(bulk[n_out:]))
                keeps.append(float(vh[:n_out].mean() / mag))
                his.append(hi)
            syn[(bits, key)] = (errs, keeps, his)
    NUM["校准方法与离群值（合成数据）"] = {
        "说明": f"{n} 个标准正态的绝对值当作主体（像 ReLU 之后的激活值，最大的约 {bulk[n_out:].max():.1f}），把其中 {n_out} 个（万分之二）换成同一个大数当作离群值。"
              "三类定范围的办法：最小最大值；百分位（99.99% 和 99.9%）；在最大值的 5% 到 100% 之间找量化均方误差最小的上界。范围下界都是 0，非对称量化。"
              "主体的相对误差 = 主体样本量化误差的均方根 / 主体的均方根；离群值保留的比例 = 离群值量化再反量化之后的值 / 原值",
        "离群值的大小": mags,
        **{f"{bits} 位，{label}": {"主体的相对误差": [float(f"{v:.3g}") for v in syn[(bits, key)][0]], "离群值保留的比例": [r3(v) for v in syn[(bits, key)][1]],
                                 "定出的上界": [r3(v) for v in syn[(bits, key)][2]]} for bits in (8, 4) for key, label in CAL_METHODS},
    }

    # 真模型的激活值：各层的分布、几种方法定出的上界、最大的那些激活值来自哪种图
    layer_rows = {}
    taps = []
    with torch.no_grad():
        c["folded"][0](c["xcal"], taps)
    ycal = c["ycal"]
    for i, t in enumerate(taps):
        v = t.reshape(-1).numpy().astype(np.float64)
        per_img = t.reshape(len(t), -1).numpy()
        p999 = np.percentile(v, 99.9)
        top = per_img >= p999
        layer_rows[f"第 {i + 1} 个卷积之后"] = {
            "最大值": r3(v.max()), "99.99% 分位": r3(np.percentile(v, 99.99)), "99.9% 分位": r3(p999),
            "均方误差最小的上界（8 位）": r3(calib_range(v, "mse", 8)), "均方误差最小的上界（4 位）": r3(calib_range(v, "mse", 4)),
            "最大值是 99.9% 分位的几倍": r3(v.max() / p999),
            "超过 99.9% 分位的激活值里来自缺陷图的比例": r3(top[ycal > 0].sum() / top.sum()),
            "为 0 的激活值的比例": r3((v == 0).mean()),
        }
    NUM["真模型的激活值分布"] = {
        "说明": f"种子 0 的浮点模型（BN 已折叠）在 {N_CAL} 张校准图上每个 ReLU 之后的全部激活值。校准图里缺陷图占 {float((ycal > 0).mean()):.3f}。" + MODEL_NOTE,
        **layer_rows,
    }

    cal_good = c["xcal"][torch.from_numpy(ycal == 0)]
    res = {}
    for bits in (8, 4):
        for key, label in CAL_METHODS:
            res[(bits, label)] = ptq_rows(None, bits, True, key)
        res[(bits, "最小最大值，校准图只有良品")] = eval_cached(("ptq-good", bits), lambda si, b=bits: ptq(c["folded"][si], cal_good, None, b, True, "minmax"))
    cols = ["分类精度", "AUROC", "沿用原阈值：过检率", "沿用原阈值：弱缺陷漏检率", "沿用原阈值：强缺陷漏检率", "重新定阈值：过检率", "重新定阈值：弱缺陷漏检率", "预测类别和浮点模型一致的比例"]
    NUM["校准方法（真模型，只量化激活值）"] = {
        "说明": "权重保持浮点，只量化 4 个 ReLU 之后的激活值，比较定范围的办法。2 个种子的平均。“校准图只有良品”指从校准图里只取良品（约一半）来统计范围。"
              + QUANT_NOTE + "。" + EVAL_NOTE + "。" + DATA_NOTE,
        "浮点模型": pack(float_rows(), cols),
        **{f"{bits} 位，{label}": pack(rows, cols) for (bits, label), rows in res.items()},
    }

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), layout="constrained")
    last = taps[-1].reshape(len(taps[-1]), -1).numpy()
    edges = np.linspace(0, float(last.max()) * 1.02, 60)
    ax[0].hist(last[ycal == 0].reshape(-1), bins=edges, color=S.MUTED, alpha=0.75, label="良品图的激活值")
    ax[0].hist(last[ycal > 0].reshape(-1), bins=edges, histtype="step", color=S.RED, lw=1.5, label="缺陷图的激活值")
    v = last.reshape(-1).astype(np.float64)
    for (key, label), color in zip(CAL_METHODS, (S.INK, S.PURPLE, S.ORANGE, S.GREEN)):
        ax[0].axvline(calib_range(v, key, 8), color=color, lw=1.5, ls="--", label=f"上界：{label}")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("最后一个卷积之后的激活值")
    ax[0].set_ylabel("个数（对数）")
    ax[0].set_title("真模型的激活值：长尾里主要是缺陷图")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    S.clean(ax[0])
    for (key, label), color, marker in zip(CAL_METHODS, (S.INK, S.PURPLE, S.ORANGE, S.GREEN), "osv^"):
        ax[1].plot(mags, syn[(8, key)][0], marker + "-", color=color, lw=1.7, ms=5, label=f"{label}，8 位")
        ax[1].plot(mags, syn[(4, key)][0], marker + "--", color=color, lw=1.2, ms=4, alpha=0.8)
    ax[1].plot([], [], color=S.MUTED, lw=1.2, ls="--", label="虚线：4 位")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xticks(mags, [str(m) for m in mags])
    ax[1].set_yticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
    ax[1].minorticks_off()
    ax[1].set_xlabel("离群值的大小（主体的标准差为 1）")
    ax[1].set_ylabel("主体的相对误差")
    ax[1].set_title("合成数据：范围跟着离群值走，主体的误差就跟着涨", fontsize=10.5)
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    S.clean(ax[1])
    labels = [label for _, label in CAL_METHODS] + ["最小最大值，校准图只有良品"]
    short = ["最小\n最大值", "99.99%\n分位", "99.9%\n分位", "均方误差\n最小", "最小最大值\n只用良品校准"]
    gx = np.arange(len(labels))
    for j, (bits, color) in enumerate(((8, S.BLUE), (4, S.ORANGE))):
        vals = [avg(res[(bits, lab)], "分类精度") for lab in labels]
        bars = ax[2].bar(gx + (j - 0.5) * 0.38, vals, 0.36, color=color, alpha=0.88, label=f"激活值 {bits} 位")
        for b, val, lab in zip(bars, vals, labels):
            pts = [r["分类精度"] for r in res[(bits, lab)]]
            ax[2].plot([b.get_x() + b.get_width() / 2] * len(pts), pts, ".", color=S.INK, ms=4)
            ax[2].text(b.get_x() + b.get_width() / 2, max(pts) + 0.012, f"{val:.3f}", ha="center", fontsize=8)
    ax[2].axhline(avg(float_rows(), "分类精度"), color=S.GREEN, lw=1.2, ls=":")
    ax[2].text(len(labels) - 0.5, avg(float_rows(), "分类精度") + 0.008, "浮点模型", ha="right", fontsize=8.5, color=S.GREEN)
    ax[2].set_xticks(gx, short)
    ax[2].tick_params(axis="x", labelsize=8.5)
    ax[2].set_ylim(0.3, 1.06)
    ax[2].set_ylabel("5 类分类精度（权重保持浮点）")
    ax[2].set_title("真模型：哪种办法好，8 位和 4 位的答案不同", fontsize=10.5)
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    ax[2].grid(axis="x", visible=False)
    S.save(fig, CH, "fig03_校准方法")


# ---------------------------------------------------------------- 图 4：敏感层、位数、量化感知训练
def qat_finetune(fnet, bits, act_hi, seed, steps, lr=1e-3, quant=True):
    """从折叠后的浮点模型出发微调。quant=True 时前向里权重（逐通道）和激活值都做量化再反量化，反向用直通估计；
    quant=False 是对照：同样的步数和学习率，前向不量化。"""
    m = copy.deepcopy(fnet).train()
    torch.manual_seed(seed)
    g = gen(seed, 4)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    qw, qa = 2 ** (bits - 1) - 1, 2 ** bits - 1

    def fq_w(w):
        s = (w.detach().abs().flatten(1).max(1).values.clamp_min(1e-12) / qw).reshape((-1,) + (1,) * (w.dim() - 1))
        return w + (torch.clamp(torch.round(w / s), -qw, qw) * s - w).detach()

    for _ in range(steps):
        x, y, _ = make_data(BATCH, g)
        h = (x - 0.5) / 0.25
        for i, cv in enumerate(m.convs):
            h = F.relu(F.conv2d(h, fq_w(cv.weight) if quant else cv.weight, cv.bias, cv.stride, cv.padding))
            if quant:
                hc = h.clamp(max=act_hi[i])
                sa = act_hi[i] / qa
                h = hc + (torch.round(hc / sa) * sa - hc).detach()
        loss = F.cross_entropy(F.linear(h.mean((2, 3)), fq_w(m.fc.weight) if quant else m.fc.weight, m.fc.bias), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m.eval()


QAT_STEPS = 150


def fig_network():
    c = ctx()
    n_conv = len(c["folded"][0].convs)
    # BN 折叠前后各通道的权重范围
    spread = {}
    for i in range(n_conv):
        before, after = [], []
        for net, f in zip(c["nets"], c["folded"]):
            r0 = net.features[i][0].weight.detach().abs().flatten(1).max(1).values
            r1 = f.convs[i].weight.detach().abs().flatten(1).max(1).values
            before.append(float(r0.max() / r0.min()))
            after.append(float(r1.max() / r1.min()))
        spread[f"第 {i + 1} 个卷积"] = {"折叠前": r3(np.mean(before)), "折叠后": r3(np.mean(after)), "各种子（折叠前，折叠后）": [[r3(a), r3(b)] for a, b in zip(before, after)]}
    NUM["BN 折叠前后各通道的权重范围"] = {
        "说明": "每个卷积里，各输出通道权重最大绝对值的最大者除以最小者。2 个种子的平均。" + MODEL_NOTE,
        **spread,
    }

    w_only = {}
    for bits in (8, 6, 4):
        for pc, label in ((False, "逐张量"), (True, "逐通道")):
            w_only[f"{bits} 位，{label}"] = pack(ptq_rows(bits, None, pc, "mse", full=False), ["分类精度", "AUROC", "预测类别和浮点模型一致的比例"])
    NUM["只量化权重：逐张量与逐通道（真模型）"] = {
        "说明": "激活值保持浮点，只量化 4 个卷积和全连接的权重（BN 折叠之后）。2 个种子的平均。" + EVAL_NOTE.split("。")[0] + "。" + MODEL_NOTE,
        "浮点模型": pack(float_rows(), ["分类精度", "AUROC"]), **w_only,
    }

    names_w = [f"第 {i + 1} 个卷积的权重" for i in range(n_conv)] + ["全连接的权重"]
    names_a = [f"第 {i + 1} 个卷积之后的激活值" for i in range(n_conv)]
    sens = {}
    for i, name in enumerate(names_w):
        sens[name] = eval_cached(("sens-w", i), lambda si, i=i: ptq(c["folded"][si], c["xcal"], 4, None, True, layers_w=[i]), full=False)
    for i, name in enumerate(names_a):
        sens[name] = eval_cached(("sens-a", i), lambda si, i=i: ptq(c["folded"][si], c["xcal"], None, 4, True, "mse", layers_a=[i]), full=False)
    fl = float_rows()
    NUM["逐层敏感度（4 位）"] = {
        "说明": "每次只把一处量化到 4 位，其余全部保持浮点：某一层的权重（逐通道），或者某一个 ReLU 之后的激活值（按均方误差最小定范围）。2 个种子的平均。" + MODEL_NOTE,
        "浮点模型": pack(fl, ["分类精度", "AUROC"]),
        **{name: dict(pack(rows, ["分类精度", "AUROC", "预测类别和浮点模型一致的比例"]),
                      **{"分类精度下降": r4(avg(fl, "分类精度") - avg(rows, "分类精度")), "AUROC 下降": r4(avg(fl, "AUROC") - avg(rows, "AUROC"))}) for name, rows in sens.items()},
        "各卷积层的权重个数": [int(cv.weight.numel()) for cv in c["folded"][0].convs] + [int(c["folded"][0].fc.weight.numel())],
        "各 ReLU 之后的激活值个数（一张图）": [int(WIDTHS[i] * (SIZE // (1, 2, 4, 4)[i]) ** 2) for i in range(n_conv)],
    }

    bits_list = [8, 6, 5, 4]
    grid = {(b, kind): ptq_rows(b, b, pc, method) for b in bits_list for kind, pc, method in (("直接", False, "minmax"), ("仔细", True, "mse"))}
    NUM["位数从 8 降到 4（权重和激活值都量化）"] = {
        "说明": "2 个种子的平均。" + QUANT_NOTE + "。" + EVAL_NOTE + "。" + DATA_NOTE + "。" + MODEL_NOTE,
        "浮点模型": pack(fl, MAIN_COLS),
        **{f"{b} 位，{kind}": pack(rows, MAIN_COLS) for (b, kind), rows in grid.items()},
    }

    def mixed(si):
        f = c["folded"][si]
        q = ptq(f, c["xcal"], 4, 4, True, "mse")
        q.convs[0].weight.data.copy_(quant_weight(f.convs[0].weight, 8, True))
        return q

    his, tuned, ctrl = [], [], []
    for si, s in enumerate(SEEDS):
        f = c["folded"][si]
        hi = [calib_range(v, "mse", 4) for v in act_samples(f, c["xcal"])]
        his.append(hi)
        tuned.append(qat_finetune(f, 4, hi, s, QAT_STEPS))
        ctrl.append(qat_finetune(f, 4, hi, s, QAT_STEPS, quant=False))
    qat = {
        "训练后量化": ptq_rows(4, 4, True, "mse"),
        "训练后量化，第 1 个卷积的权重留 8 位": eval_cached(("mixed",), mixed),
        "先照常微调再做训练后量化（对照）": eval_cached(("ctrl", 4), lambda si: ptq(ctrl[si], c["xcal"], 4, 4, True, "mse")),
        "量化感知训练": eval_cached(("qat", 4), lambda si: ptq(tuned[si], c["xcal"], 4, 4, True, act_hi=his[si])),
    }
    w = c["folded"][0].convs[1].weight.detach().clone().requires_grad_(True)
    s = w.detach().abs().max() / 7
    (torch.round(w / s) * s).sum().backward()
    g_plain = float(w.grad.abs().sum())
    w.grad = None
    (w + (torch.round(w / s) * s - w).detach()).sum().backward()
    qat_cols = [n for n in MAIN_COLS if not n.startswith("logits")]
    NUM["量化感知训练（4 位）"] = {
        "说明": f"权重和激活值都是 4 位。量化感知训练：从 BN 已折叠的浮点模型出发再训练 {QAT_STEPS} 步（AdamW，学习率 0.001，每步新图），前向里权重逐通道量化、激活值按固定范围量化"
              "（范围是训练前按均方误差最小定的），反向时把取整当成恒等（直通估计），被截断的激活值梯度为 0；训练完按同样的量化方式评价。"
              "对照：同样的步数和学习率，前向不量化，训练完再做训练后量化。“第 1 个卷积的权重留 8 位”：其余和训练后量化相同，不训练。2 个种子的平均。" + EVAL_NOTE,
        "浮点模型": pack(fl, qat_cols),
        **{name: pack(rows, qat_cols) for name, rows in qat.items()},
        "取整的梯度": {"说明": "对“量化再反量化后的权重之和”求导，一个卷积层权重梯度的绝对值之和", "不用直通估计": g_plain, "用直通估计": float(w.grad.abs().sum()), "这一层的权重个数": int(w.numel())},
    }

    # 同一个函数换一种参数化：第 2 个卷积的各输出通道乘上不同的正数，第 3 个卷积对应的输入通道除回去
    def rescaled(si, spread):
        g = copy.deepcopy(c["folded"][si])
        rng = np.random.default_rng(100 + si)
        alpha = torch.tensor(float(spread) ** (rng.random(g.convs[1].out_channels) - 0.5), dtype=torch.float32)
        g.convs[1].weight.data.mul_(alpha[:, None, None, None])
        g.convs[1].bias.data.mul_(alpha)
        g.convs[2].weight.data.div_(alpha[None, :, None, None])
        return g

    spreads = [1, 10, 100, 1000]
    rep = {}
    for sp in spreads:
        gs = [rescaled(si, sp) for si in range(len(SEEDS))]
        same = max(float((predict(g, c["xte"][:500]) - ref[:500]).abs().max()) for g, ref in zip(gs, c["ref"]))
        ratio = np.mean([float(r.max() / r.min()) for r in (g.convs[1].weight.detach().abs().flatten(1).max(1).values for g in gs)])
        row = {"第 2 个卷积各通道权重范围的最大比值": float(f"{ratio:.3g}"), "换参数化之后浮点输出的最大变化": float(f"{same:.2g}")}
        for pc, label in ((False, "逐张量"), (True, "逐通道")):
            rows = eval_cached(("rescale", sp, pc), lambda si, gs=gs, pc=pc: ptq(gs[si], c["xcal"], 8, None, pc, layers_w=[1]), full=False)
            row[label] = pack(rows, ["分类精度", "AUROC", "预测类别和浮点模型一致的比例", "logits 和浮点模型的最大差"])
        rep[f"缩放相差至多 {sp} 倍"] = row
    NUM["逐张量与逐通道（真模型，换一种等价的参数化）"] = {
        "说明": "把 BN 已折叠的模型里第 2 个卷积的每个输出通道（权重和偏置）乘上一个正数，第 3 个卷积对应的输入通道除以同一个数。ReLU 对正数缩放是齐次的，网络的输出不变。"
              "各通道的乘数在对数坐标上均匀随机取，最大和最小相差至多 R 倍。然后只把第 2 个卷积的权重量化到 8 位（其余全部浮点），比较逐张量和逐通道。2 个种子的平均，只在测试集上看",
        **rep,
    }

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.5), gridspec_kw={"width_ratios": [1.3, 1.0, 1.05]}, layout="constrained")
    names = names_w + names_a
    short = [f"卷积 {i + 1}\n{n} 个" for i, n in enumerate(NUM["逐层敏感度（4 位）"]["各卷积层的权重个数"][:n_conv])] + ["全连接"] + [f"ReLU {i + 1}" for i in range(n_conv)]
    gx = np.arange(len(names))
    drops = [avg(fl, "分类精度") - avg(sens[n], "分类精度") for n in names]
    bars = ax[0].bar(gx, drops, 0.62, color=[S.BLUE] * len(names_w) + [S.ORANGE] * len(names_a), alpha=0.9)
    for b, name, d in zip(bars, names, drops):
        pts = [f["分类精度"] - r["分类精度"] for f, r in zip(fl, sens[name])]
        ax[0].plot([b.get_x() + b.get_width() / 2] * len(pts), pts, ".", color=S.INK, ms=5)
        ax[0].text(b.get_x() + b.get_width() / 2, max(max(pts), 0) + 0.008, f"{d:.3f}", ha="center", fontsize=8.5)
    ax[0].axhline(0, color=S.MUTED, lw=0.8)
    ax[0].set_xticks(gx, short)
    ax[0].tick_params(axis="x", labelsize=8.5)
    ax[0].set_ylabel("分类精度下降（只把这一处量化到 4 位）")
    ax[0].set_title("最敏感的是权重最少的第 1 个卷积（蓝：权重，橙：激活值，黑点：各种子）", fontsize=10.5)
    S.clean(ax[0])
    ax[0].grid(axis="x", visible=False)
    for kind, color, marker in (("直接", S.RED, "o"), ("仔细", S.BLUE, "s")):
        ax[1].plot(range(len(bits_list)), [avg(grid[(b, kind)], "分类精度") for b in bits_list], marker + "-", color=color, lw=1.8, label=f"{kind}：分类精度")
        ax[1].plot(range(len(bits_list)), [avg(grid[(b, kind)], "重新定阈值：过检率") for b in bits_list], marker + "--", color=color, lw=1.3, alpha=0.85, label=f"{kind}：过检率（重新定阈值）")
        for j, b in enumerate(bits_list):
            ax[1].plot([j] * len(SEEDS), [r["分类精度"] for r in grid[(b, kind)]], ".", color=color, ms=4, alpha=0.5)
    ax[1].axhline(avg(fl, "分类精度"), color=S.GREEN, lw=1.1, ls=":")
    ax[1].text(0, avg(fl, "分类精度") + 0.012, "浮点模型的精度", fontsize=8.5, color=S.GREEN)
    ax[1].set_xticks(range(len(bits_list)), [f"{b} 位" for b in bits_list])
    ax[1].set_ylim(-0.03, 1.08)
    ax[1].set_xlabel("权重和激活值的位数（训练后量化）")
    ax[1].set_title("8 位和 6 位贴着浮点，4 位时两种做法都垮了", fontsize=10.5)
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    S.clean(ax[1])
    kinds = list(qat)
    short_k = ["训练后\n量化", "第 1 个卷积\n的权重留 8 位", "照常微调再\n训练后量化", "量化感知\n训练"]
    for j, (kind, color) in enumerate(zip(kinds, (S.MUTED, S.PURPLE, S.ORANGE, S.BLUE))):
        pts = [r["分类精度"] for r in qat[kind]]
        ax[2].bar(j, np.mean(pts), 0.62, color=color, alpha=0.9)
        ax[2].plot([j] * len(pts), pts, ".", color=S.INK, ms=5)
        ax[2].text(j, max(pts) + 0.015, f"{np.mean(pts):.3f}", ha="center", fontsize=8.5)
    ax[2].axhline(avg(fl, "分类精度"), color=S.GREEN, lw=1.1, ls=":")
    ax[2].text(-0.45, avg(fl, "分类精度") + 0.012, "浮点模型", ha="left", fontsize=8.5, color=S.GREEN)
    ax[2].set_xticks(range(len(kinds)), short_k)
    ax[2].tick_params(axis="x", labelsize=8.5)
    ax[2].set_ylim(0.0, 1.1)
    ax[2].set_ylabel("5 类分类精度（权重和激活值都是 4 位）")
    ax[2].set_title("4 位怎样救回来：两条路都回到 0.9 以上", fontsize=10.5)
    S.clean(ax[2])
    ax[2].grid(axis="x", visible=False)
    S.save(fig, CH, "fig04_敏感层_位数_量化感知训练")


# ---------------------------------------------------------------- 剪枝和蒸馏用到的模型（图 1、图 5 和计时共用）
STUDENT_W = tuple(w // 2 for w in WIDTHS)
FT_STEPS = 120
N_UNLAB = 12000
N_POOL = 200
KD_TEMPS = (1.0, 4.0)
_MODELS: dict = {}


def prune_structured(net, keep):
    """去掉整个通道：每个卷积按“折叠后滤波器的 L1 范数”留下最大的一部分输出通道，后一层对应的输入通道一起去掉。返回一个更窄的 Net。"""
    new_w = tuple(max(2, int(round(w * keep))) for w in net.widths)
    small = Net(new_w)
    prev = torch.arange(1)
    for blk, new_blk, k in zip(net.features, small.features, new_w):
        conv, bn = blk[0], blk[1]
        w, _ = hw.fold_bn_into_conv(conv.weight, None, bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps)
        idx = torch.sort(torch.topk(w.abs().flatten(1).sum(1), k).indices).values
        new_blk[0].weight.data.copy_(conv.weight.data[idx][:, prev])
        for name in ("weight", "bias", "running_mean", "running_var"):
            getattr(new_blk[1], name).data.copy_(getattr(bn, name).data[idx])
        prev = idx
    small.fc.weight.data.copy_(net.fc.weight.data[:, prev])
    small.fc.bias.data.copy_(net.fc.bias.data)
    return small.eval()


def prune_unstructured(net, sparsity):
    """非结构化：全部卷积和全连接的权重按绝对值全局排序，最小的一部分置零。返回 (模型, 各层的掩码)。"""
    m = copy.deepcopy(net)
    layers = [mod for mod in m.modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]
    allw = torch.cat([mod.weight.detach().abs().flatten() for mod in layers])
    thr = torch.kthvalue(allw, int(round(sparsity * allw.numel()))).values
    masks = [(mod.weight.detach().abs() > thr).float() for mod in layers]
    for mod, mask in zip(layers, masks):
        mod.weight.data.mul_(mask)
    return m.eval(), masks


def remask(masks):
    def fn(net):
        layers = [mod for mod in net.modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]
        with torch.no_grad():
            for mod, mask in zip(layers, masks):
                mod.weight.mul_(mask)
    return fn


def models():
    """把剪枝、蒸馏要用的模型都训练出来，放进 _MODELS[名字] = [种子 0 的模型, 种子 1 的模型]。"""
    if _MODELS:
        return _MODELS
    c = ctx()
    for keep in (0.75, 0.5, 0.25):
        pruned = [prune_structured(n, keep) for n in c["nets"]]
        _MODELS[("结构化", keep, "不微调")] = pruned
        _MODELS[("结构化", keep, "微调")] = [train(copy.deepcopy(p), s, steps=FT_STEPS, lr=2e-3) for p, s in zip(pruned, SEEDS)]
    for sp in (0.5, 0.75, 0.9):
        pairs = [prune_unstructured(n, sp) for n in c["nets"]]
        _MODELS[("非结构化", sp, "不微调")] = [p[0] for p in pairs]
        _MODELS[("非结构化", sp, "微调")] = [train(copy.deepcopy(p[0]), s, steps=FT_STEPS, lr=2e-3, after_step=remask(p[1])) for p, s in zip(pairs, SEEDS)]
    # 学生（通道减半）：都在同一批 N_UNLAB 张图里抽样。直接训练用这些图的标签；蒸馏只用教师在这些图上的输出
    xu, yu, _ = make_data(N_UNLAB, gen(9, 9))
    soft = [predict(n, xu) for n in c["nets"]]
    _MODELS[("学生", "直接训练")] = [train(Net(STUDENT_W), 10 + s, pool=(xu, yu)) for s in SEEDS]
    for temp in KD_TEMPS:
        _MODELS[("学生", "蒸馏", temp)] = [train(Net(STUDENT_W), 10 + s, pool=(xu, yu), soft=t, temp=temp) for t, s in zip(soft, SEEDS)]
    few = (xu[:N_POOL], yu[:N_POOL])
    _MODELS[("学生", "少标签直接训练")] = [train(Net(STUDENT_W), 10 + s, pool=few) for s in SEEDS]
    _MODELS[("学生", "少图蒸馏")] = [train(Net(STUDENT_W), 10 + s, pool=few, soft=t[:N_POOL], temp=4.0) for t, s in zip(soft, SEEDS)]
    return _MODELS


def model_rows(key, full=False):
    ms = models()[key]
    return eval_cached(("model",) + key, lambda si: ms[si], full)


# ---------------------------------------------------------------- 图 1：压缩之后分数会移动
def fig_shift():
    c = ctx()
    cal_good = c["xcal"][torch.from_numpy(c["ycal"] == 0)]
    half_rows = _EVAL.setdefault((("half",), True), [evaluate(copy.deepcopy(c["folded"][si]).half(), si, True, half=True) for si in range(len(SEEDS))])
    variants = {
        "浮点（BN 已折叠）": float_rows(),
        "半精度（FP16）": half_rows,
        "8 位，仔细": ptq_rows(8, 8, True, "mse"),
        "8 位，直接": ptq_rows(8, 8, False, "minmax"),
        "8 位，校准图只有良品": eval_cached(("ptq-good-full", 8), lambda si: ptq(c["folded"][si], cal_good, 8, 8, True, "minmax")),
        "5 位，仔细": ptq_rows(5, 5, True, "mse"),
        "4 位，仔细": ptq_rows(4, 4, True, "mse"),
        "非结构化剪掉 50% 的权重，不微调": model_rows(("非结构化", 0.5, "不微调"), full=True),
    }
    NUM["压缩之后分数会移动"] = {
        "说明": "同一个训练好的模型（2 个种子各一个）用几种办法压缩，不重新训练。2 个种子的平均。“校准图只有良品”：权重逐通道 8 位，激活值 8 位、按最小最大值定范围，但校准图里只有良品。"
              "半精度：权重和激活值都用 float16，在 CPU 上直接算。" + QUANT_NOTE + "。" + EVAL_NOTE + "。" + DATA_NOTE + "。" + MODEL_NOTE,
        **{name: pack(rows, MAIN_COLS) for name, rows in variants.items()},
        "测试集里的张数": {"良品": int((c["yte"] == 0).sum()), "弱缺陷": int(((c["yte"] > 0) & (c["ate"] < WEAK)).sum()), "强缺陷": int(((c["yte"] > 0) & (c["ate"] >= WEAK)).sum())},
        "浮点模型的阈值（各种子）": [r4(t) for t in c["thr"]],
        "测试集里 1 张图对应多少个百分点": {"良品（过检率）": round(100 / int((c["yte"] == 0).sum()), 2), "弱缺陷（弱缺陷漏检率）": round(100 / int(((c["yte"] > 0) & (c["ate"] < WEAK)).sum()), 2),
                              "强缺陷（强缺陷漏检率）": round(100 / int(((c["yte"] > 0) & (c["ate"] >= WEAK)).sum()), 2)},
        "校准图只有良品时漏掉的强缺陷张数（各种子）": [int(round(r["沿用原阈值：强缺陷漏检率"] * int(((c["yte"] > 0) & (c["ate"] >= WEAK)).sum()))) for r in variants["8 位，校准图只有良品"]],
        "5 位仔细：重新定阈值后的过检率是浮点的几倍": sig(avg(variants["5 位，仔细"], "重新定阈值：过检率") / avg(variants["浮点（BN 已折叠）"], "重新定阈值：过检率")),
        "5 位仔细：分类精度比浮点低多少（百分点）": round(100 * (avg(variants["浮点（BN 已折叠）"], "分类精度") - avg(variants["5 位，仔细"], "分类精度")), 1),
        "5 位仔细：沿用原阈值的过检率是浮点的几倍": sig(avg(variants["5 位，仔细"], "沿用原阈值：过检率") / avg(variants["浮点（BN 已折叠）"], "沿用原阈值：过检率")),
        "剪掉 50% 不微调：沿用原阈值的过检率是浮点的几倍": sig(avg(variants["非结构化剪掉 50% 的权重，不微调"], "沿用原阈值：过检率") / avg(variants["浮点（BN 已折叠）"], "沿用原阈值：过检率")),
        "浮点 logits 绝对值的平均": r3(float(torch.cat(c["ref"]).abs().mean())), "浮点 logits 绝对值的最大": r3(float(torch.cat(c["ref"]).abs().max())),
    }

    y, a = c["yte"], c["ate"]
    groups = [("良品", y == 0, S.MUTED), ("弱缺陷", (y > 0) & (a < WEAK), S.ORANGE), ("强缺陷", (y > 0) & (a >= WEAK), S.RED)]
    thr_margin = math.log(c["thr"][0] / (1 - c["thr"][0]))
    shown = ["浮点（BN 已折叠）", "8 位，校准图只有良品", "5 位，仔细"]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.3), gridspec_kw={"width_ratios": [1, 1, 1, 1.55]}, layout="constrained")
    edges = np.linspace(-16, 16, 65)
    for axi, name in zip(ax[:3], shown):
        mg = np.clip(margin_of(variants[name][0]["_logits"]), -15.9, 15.9)
        for label, sel, color in groups:
            axi.hist(mg[sel], bins=edges, histtype="stepfilled" if label == "良品" else "step", color=color, alpha=0.55 if label == "良品" else 1.0, lw=1.5, label=label)
        axi.axvline(thr_margin, color=S.INK, lw=1.3, ls="--", label="浮点模型定的阈值")
        r = variants[name][0]
        axi.set_yscale("log")
        axi.set_xlabel("缺陷分数的对数几率（越大越像缺陷）")
        axi.set_title(f"{name}（种子 0）\n过检 {r['沿用原阈值：过检率']:.1%}，弱缺陷漏检 {r['沿用原阈值：弱缺陷漏检率']:.1%}，强缺陷漏检 {r['沿用原阈值：强缺陷漏检率']:.1%}", fontsize=10)
        S.clean(axi)
    ax[0].set_ylabel("测试图张数（对数）")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    names = list(variants)
    short = ["浮点", "半精度", "8 位\n仔细", "8 位\n直接", "8 位\n只用良品\n校准", "5 位\n仔细", "4 位\n仔细", "剪掉 50%\n不微调"]
    gx = np.arange(len(names))
    for j, (col, color, label) in enumerate((("沿用原阈值：过检率", S.MUTED, "过检率"), ("沿用原阈值：弱缺陷漏检率", S.ORANGE, "弱缺陷漏检率"), ("沿用原阈值：强缺陷漏检率", S.RED, "强缺陷漏检率"))):
        vals = [avg(variants[n], col) for n in names]
        bars = ax[3].bar(gx + (j - 1) * 0.27, vals, 0.25, color=color, alpha=0.9, label=label)
        for b, v in zip(bars, vals):
            ax[3].text(b.get_x() + b.get_width() / 2, min(v, 0.62) + 0.008, f"{100 * v:.1f}", ha="center", fontsize=7.5)
    ax[3].set_xticks(gx, short)
    ax[3].tick_params(axis="x", labelsize=8.5)
    ax[3].set_ylim(0, 0.68)
    ax[3].set_ylabel("沿用浮点模型的阈值时的比例（柱上的数是百分数）")
    ax[3].set_title("同一个阈值，压缩之后的过检和漏检（2 个种子平均）", fontsize=10.5)
    ax[3].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    S.clean(ax[3])
    ax[3].grid(axis="x", visible=False)
    S.save(fig, CH, "fig01_压缩之后分数会移动")


# ---------------------------------------------------------------- 表：BN 折叠、算子融合、导出对拍
def table_fold_export():
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    c = ctx()
    net, f, x = c["nets"][0], c["folded"][0], c["xte"]
    ref, out = predict(net, x), predict(f, x)
    net64 = copy.deepcopy(net).double()
    out64 = predict(Folded(net64).eval(), x.double())
    ref64 = predict(net64, x.double())
    lib_w, lib_b = 0.0, 0.0
    for blk, cv in zip(net.features, f.convs):
        fused = fuse_conv_bn_eval(blk[0], blk[1])
        lib_w = max(lib_w, float((fused.weight - cv.weight).abs().max()))
        lib_b = max(lib_b, float((fused.bias - cv.bias).abs().max()))

    def wrong(kind):
        g = copy.deepcopy(f)
        for blk, cv in zip(net.features, g.convs):
            conv, bn = blk[0], blk[1]
            if kind == "方差当成了标准差":
                k = bn.weight / (bn.running_var + bn.eps)
            elif kind == "eps 加在根号外面":
                k = bn.weight / (bn.running_var.sqrt() + bn.eps)
            else:
                k = bn.weight / (bn.running_var + bn.eps).sqrt()
            cv.weight.data.copy_(conv.weight * k[:, None, None, None])
            cv.bias.data.copy_(bn.bias if kind == "偏置里漏了减均值" else bn.bias - bn.running_mean * k)
        o = predict(g, x)
        return {"和原模型输出的最大差": float(f"{float((o - ref).abs().max()):.3g}"), "分类精度": r4((score_of(o)[1] == c["yte"]).mean())}

    n_bn_stats = sum(blk[1].running_mean.numel() + blk[1].running_var.numel() for blk in net.features)
    n_bn_params = sum(blk[1].weight.numel() + blk[1].bias.numel() for blk in net.features)
    params, macs = count_cost(net)
    bn_elems = sum(WIDTHS[i] * (SIZE // (1, 2, 4, 4)[i]) ** 2 for i in range(4))
    var_all = torch.cat([blk[1].running_var for blk in net.features])
    NUM["BN 折叠"] = {
        "说明": f"种子 0 的模型，{N_TEST} 张测试图。折叠用作业的 fold_bn_into_conv。float64 一行是把模型和输入都转成 float64 之后再比。" + MODEL_NOTE,
        "折叠前后输出的最大差（float32）": float(f"{float((out - ref).abs().max()):.2g}"), "折叠前后输出的最大差（float64）": float(f"{float((out64 - ref64).abs().max()):.2g}"),
        "折叠前后预测类别一致的比例": float((score_of(out)[1] == score_of(ref)[1]).mean()),
        "和 torch.nn.utils.fusion.fuse_conv_bn_eval 比：权重的最大差": float(f"{lib_w:.2g}"), "和它比：偏置的最大差": float(f"{lib_b:.2g}"),
        "原模型的参数个数（卷积、BN 的缩放和偏移、全连接）": params, "其中 BN 的缩放和偏移": n_bn_params, "BN 另有的运行均值和运行方差": n_bn_stats,
        "折叠后的参数个数（卷积带偏置、全连接）": sum(p.numel() for p in f.parameters()),
        "卷积和全连接的乘加次数": macs, "BN 逐元素的乘加次数（每个元素一次）": bn_elems, "BN 的乘加次数占卷积和全连接的比例": r4(bn_elems / macs),
        "运行方差的最小值和最大值": [float(f"{float(var_all.min()):.3g}"), float(f"{float(var_all.max()):.3g}")],
        "三种写错的折叠": {k: wrong(k) for k in ("方差当成了标准差", "eps 加在根号外面", "偏置里漏了减均值")},
        "eps 位置的影响（方差为 1e-6、eps 为 1e-5 时两种写法的缩放系数之比）": r3((math.sqrt(1e-6) + 1e-5) ** -1 / (1e-6 + 1e-5) ** -0.5),
    }

    fx_note: dict = {}
    try:
        import torch.fx.experimental.optimization as fxo

        fused = fxo.fuse(copy.deepcopy(net))

        def n_mod(m):
            return sum(1 for nd in torch.fx.symbolic_trace(m).graph.nodes if nd.op == "call_module")

        fx_note = {"融合前图里的模块调用个数": n_mod(net), "融合后": n_mod(fused), "融合后和原模型输出的最大差": float(f"{float((predict(fused, x) - ref).abs().max()):.2g}")}
    except Exception as exc:  # noqa: BLE001
        fx_note = {"没有跑通": f"{type(exc).__name__}"}
    NUM["用 torch.fx 做融合"] = {"说明": "torch.fx.experimental.optimization.fuse 把图里相邻的卷积和 BN 合成一个卷积，种子 0 的模型。ReLU 仍然是单独的节点", **fx_note}

    rows = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        batch = torch.export.Dim("batch", min=1, max=64)
        ep = torch.export.export(f, (x[:4],), dynamic_shapes=({0: batch},))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.pt2"
            torch.export.save(ep, str(path))
            loaded = torch.export.load(str(path)).module()
        with torch.no_grad():
            rows["导出时的样例（4 张）"] = float((loaded(x[:4]) - f(x[:4])).abs().max())
            rows["另外 16 张图（批大小 16）"] = float((loaded(x[4:20]) - f(x[4:20])).abs().max())
            rows["单张（批大小 1）"] = float((loaded(x[20:21]) - f(x[20:21])).abs().max())
            rows[f"全部 {N_TEST} 张（每批 50 张）"] = float(max((loaded(x[i:i + 50]) - f(x[i:i + 50])).abs().max() for i in range(0, len(x), 50)))
        static = torch.export.export(f, (x[:4],)).module()
        try:
            with torch.no_grad():
                static(x[:8])
            fixed = "没有报错"
        except Exception as exc:  # noqa: BLE001
            fixed = f"报错 {type(exc).__name__}"
    NUM["torch.export 导出对拍"] = {
        "说明": f"torch {torch.__version__.split('+')[0]}。把 BN 已折叠的模型（种子 0）用 torch.export.export 导出，批的维度声明成动态（1 到 64），torch.export.save 存盘、torch.export.load 读回，"
              "再和原模型比输出的最大绝对差。最后一行是不声明动态维度导出之后，换一个批大小去调用",
        "和原模型输出的最大差": rows, "不声明动态维度，导出样例是 4 张，用 8 张去调用": fixed,
    }


# ---------------------------------------------------------------- 表：半精度
def table_half():
    info = np.finfo(np.float16)
    f16 = np.float16
    ones = np.ones(4096, dtype=np.float16)
    seq = f16(0)
    for v in ones:
        seq = f16(seq + v)
    hundred = np.full(4096, 100, dtype=np.float16)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        cum = np.cumsum(hundred)
        naive = np.exp(np.array([12.0, 0.0], dtype=np.float16))
        naive_sm = naive / naive.sum()
        sq = f16(256) * f16(256)
        l2_naive = np.sqrt((np.array([300.0, 1.0], dtype=np.float16) ** 2).sum())
        div = f16(1) / (f16(0) + f16(1e-8))
    t16 = torch.tensor([12.0, 0.0], dtype=torch.float16)
    big = torch.full((1, 1, 64, 64), 100.0, dtype=torch.float16)
    NUM["半精度的数值"] = {
        "说明": "float16 的表示范围和几个数值现象，用 NumPy 和 PyTorch 在 CPU 上算。“间隔”指这个数和下一个能表示的数之间的距离",
        "最大值": float(info.max), "最小的正规数": float(f"{float(info.tiny):.4g}"), "最小的非正规数": float(f"{float(info.smallest_subnormal):.3g}"),
        "1 附近的间隔（机器精度）": float(f"{float(info.eps):.3g}"), "有效十进制位数": int(info.precision),
        "各量级附近的间隔": {str(v): float(np.spacing(f16(v))) for v in (1, 16, 256, 1024, 2048, 4096, 32768)},
        "2048 加 1 的结果": float(f16(2048) + f16(1)), "能连续精确表示的最大整数": 2048,
        "4096 个 1 相加": {"逐个累加（np.cumsum 的最后一个值）": float(np.cumsum(ones)[-1]), "逐个累加（Python 循环）": float(seq), "np.sum": float(ones.sum()),
                         "torch.sum": float(torch.ones(4096, dtype=torch.float16).sum())},
        "4096 个 100 相加再求平均": {"逐个累加到第几个变成 inf": int(np.argmax(np.isinf(cum)) + 1), "np.mean": float(hundred.mean()), "torch.mean（64x64 的全局平均）": float(big.mean()),
                              "torch.sum": float(big.sum())},
        "4096 个 100 的和（精确值）": 4096 * 100, "float32 下 exp 溢出的门槛（ln 3.4e38）": r3(math.log(float(np.finfo(np.float32).max))),
        "exp 溢出的门槛（ln 65504）": r3(math.log(float(info.max))), "平方溢出的门槛（根号 65504）": r3(math.sqrt(float(info.max))),
        "logits 为 [12, 0] 时": {"exp 的结果": [float(v) for v in naive], "先 exp 再除以和": [float(v) for v in naive_sm],
                             "torch.softmax": [float(f"{float(v):.3g}") for v in torch.softmax(t16, 0)], "torch.log_softmax": [float(v) for v in torch.log_softmax(t16, 0)]},
        "256 的平方": float(sq), "向量 [300, 1] 的长度：先平方求和再开方": float(l2_naive), "向量 [300, 1] 的长度：torch.linalg.norm": float(torch.linalg.norm(torch.tensor([300.0, 1.0], dtype=torch.float16))),
        "常用的小常数存成 float16": {"1e-5": float(f"{float(f16(1e-5)):.4g}"), "1e-6": float(f"{float(f16(1e-6)):.4g}"), "1e-7": float(f"{float(f16(1e-7)):.4g}"), "1e-8": float(f16(1e-8))},
        "1 除以 (0 + 1e-8)": float(div),
    }

    c = ctx()
    rows = _EVAL.setdefault((("half",), True), [evaluate(copy.deepcopy(c["folded"][si]).half(), si, True, half=True) for si in range(len(SEEDS))])
    NUM["半精度跑真模型"] = {
        "说明": "BN 已折叠的模型整个转成 float16（权重和激活值），输入也转成 float16，在 CPU 上算，和 float32 比。2 个种子的平均。" + EVAL_NOTE,
        **pack(rows, MAIN_COLS),
        "预测类别不一致的图有几张（各种子）": [int(round((1 - r["预测类别和浮点模型一致的比例"]) * N_TEST)) for r in rows],
    }


# ---------------------------------------------------------------- 图 5：剪枝与蒸馏
def fig_prune_kd():
    c = ctx()
    ms = models()
    cols = ["分类精度", "AUROC"]
    fl = eval_cached(("orig",), lambda si: c["nets"][si], False)
    p0, m0 = count_cost(c["nets"][0])
    keeps = [0.75, 0.5, 0.25]
    st = {}
    for keep in keeps:
        p, m = count_cost(ms[("结构化", keep, "微调")][0])
        st[keep] = {"通道数": list(ms[("结构化", keep, "微调")][0].widths), "参数个数": p, "乘加次数": m, "参数是原来的比例": r3(p / p0), "乘加次数是原来的比例": r3(m / m0),
                    "剪完不微调": pack(model_rows(("结构化", keep, "不微调")), cols), f"剪完微调 {FT_STEPS} 步": pack(model_rows(("结构化", keep, "微调")), cols)}
        if keep == 0.5:
            st[keep][f"同样的窄结构从头训练 {STEPS} 步"] = pack(model_rows(("学生", "直接训练")), cols)
    NUM["结构化剪枝"] = {
        "说明": f"每个卷积按“BN 折叠后滤波器的 L1 范数”留下最大的一部分输出通道，后一层对应的输入通道一起去掉，得到一个更窄的稠密网络。微调：{FT_STEPS} 步，学习率 0.002，每步新图。"
              "乘加次数只数卷积和全连接，32x32 的输入。2 个种子的平均，只在测试集上看分类精度和 AUROC。" + MODEL_NOTE,
        "原模型": {"通道数": list(WIDTHS), "参数个数": p0, "乘加次数": m0, **pack(fl, cols)},
        **{f"通道留 {keep}": v for keep, v in st.items()},
    }
    sps = [0.5, 0.75, 0.9]
    un = {}
    for sp in sps:
        layers = [mod for mod in ms[("非结构化", sp, "微调")][0].modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]
        nz = sum(int((mod.weight != 0).sum()) for mod in layers)
        tot = sum(mod.weight.numel() for mod in layers)
        un[sp] = {"非零权重的比例": r3(nz / tot), "按稠密方式计算的乘加次数": count_cost(ms[("非结构化", sp, "微调")][0])[1],
                  "第 1 个卷积里被置零的比例": r3(float((layers[0].weight == 0).float().mean())), "最后一个卷积里被置零的比例": r3(float((layers[3].weight == 0).float().mean())),
                  "剪完不微调": pack(model_rows(("非结构化", sp, "不微调")), cols), f"剪完微调 {FT_STEPS} 步（置零的权重保持为零）": pack(model_rows(("非结构化", sp, "微调")), cols)}
    NUM["非结构化剪枝"] = {
        "说明": "全部卷积和全连接的权重按绝对值全局排序，把最小的一部分置零。张量的形状不变，按稠密方式计算时乘加次数不变。其余同上",
        **{f"置零 {sp}": v for sp, v in un.items()},
    }
    ps, mstu = count_cost(Net(STUDENT_W))
    kd = {"直接用标签训练": pack(model_rows(("学生", "直接训练")), cols), **{f"蒸馏，温度 {t:g}": pack(model_rows(("学生", "蒸馏", t)), cols) for t in KD_TEMPS}}
    few = {f"只有 {N_POOL} 张带标签的图，直接训练": pack(model_rows(("学生", "少标签直接训练")), cols),
           f"只在这 {N_POOL} 张图上蒸馏（温度 4）": pack(model_rows(("学生", "少图蒸馏")), cols),
           f"在 {N_UNLAB} 张无标签图上蒸馏（温度 4，同上）": pack(model_rows(("学生", "蒸馏", 4.0)), cols)}
    with torch.no_grad():
        t_logits = c["nets"][0](c["xte"][:1000])
    soft = {f"温度 {t:g}": r3(float(F.softmax(t_logits / t, -1).max(1).values.mean())) for t in KD_TEMPS}
    NUM["知识蒸馏"] = {
        "说明": f"教师是上面的浮点模型（通道数 {list(WIDTHS)}），学生的通道数减半（{list(STUDENT_W)}，和“通道留 0.5”的结构相同）。学生都训练 {STEPS} 步，每一步从同一批 {N_UNLAB} 张图里抽 {BATCH} 张。"
              "直接训练用这些图的标签；蒸馏不用标签，损失只有一项：学生和教师在同一个温度下的分布的 KL 散度，乘温度的平方。"
              f"“标签很少时”：带标签的图只有其中 {N_POOL} 张。2 个种子的平均，教师和学生一一对应，只在测试集上看分类精度和 AUROC",
        "教师": {"参数个数": p0, "乘加次数": m0, **pack(fl, cols)}, "学生的参数个数": ps, "学生的乘加次数": mstu, "学生的乘加次数是教师的比例": r3(mstu / m0),
        **kd, "标签很少时": few, "教师输出的最大类概率的平均（1000 张测试图）": soft,
    }

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.4), layout="constrained")

    def line(axi, xs, keys, style, color, label):
        rows = [model_rows(k) for k in keys]
        axi.plot(xs, [avg(r, "分类精度") for r in rows], style, color=color, lw=1.8, ms=5.5, label=label)
        for xv, r in zip(xs, rows):
            axi.plot([xv] * len(r), [q["分类精度"] for q in r], ".", color=color, ms=3.5, alpha=0.5)

    rel = [st[k]["乘加次数是原来的比例"] for k in keeps]
    line(ax[0], rel, [("结构化", k, "不微调") for k in keeps], "x:", S.RED, "剪完不微调")
    line(ax[0], rel, [("结构化", k, "微调") for k in keeps], "o-", S.BLUE, f"剪完微调 {FT_STEPS} 步")
    line(ax[0], [st[0.5]["乘加次数是原来的比例"]], [("学生", "直接训练")], "s", S.GREEN, f"同样的窄结构从头训练 {STEPS} 步")
    ax[0].plot([1.0], [avg(fl, "分类精度")], "D", color=S.INK, ms=6, label="原模型")
    ax[0].set_xlabel("乘加次数是原模型的比例（结构化剪枝）")
    ax[0].set_ylabel("5 类分类精度")
    ax[0].set_ylim(0.1, 1.03)
    ax[0].set_title("结构化剪枝：计算量真的少了，精度要靠微调拿回来", fontsize=10.5)
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    S.clean(ax[0])
    line(ax[1], sps, [("非结构化", s, "不微调") for s in sps], "x:", S.RED, "置零后不微调")
    line(ax[1], sps, [("非结构化", s, "微调") for s in sps], "o-", S.BLUE, f"置零后微调 {FT_STEPS} 步")
    ax[1].plot([0.0], [avg(fl, "分类精度")], "D", color=S.INK, ms=6, label="原模型")
    ax[1].set_xticks([0.0] + sps, ["0"] + [f"{int(100 * s)}%" for s in sps])
    ax[1].set_xlabel("置零的权重比例（非结构化剪枝，乘加次数不变）")
    ax[1].set_ylim(0.1, 1.03)
    ax[1].set_title("非结构化剪枝：精度同样能微调回来，但稠密计算量没变", fontsize=10.5)
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    S.clean(ax[1])
    bars_def = [("直接训练\n标签不缺", ("学生", "直接训练"), S.MUTED)] + [(f"蒸馏\n温度 {t:g}", ("学生", "蒸馏", t), S.BLUE) for t in KD_TEMPS] + \
               [(f"{N_POOL} 张标签\n直接训练", ("学生", "少标签直接训练"), S.ORANGE), (f"只在这 {N_POOL} 张\n图上蒸馏", ("学生", "少图蒸馏"), S.PURPLE)]
    for j, (label, key, color) in enumerate(bars_def):
        r = model_rows(key)
        v = avg(r, "分类精度")
        ax[2].bar(j, v, 0.66, color=color, alpha=0.9)
        ax[2].plot([j] * len(r), [q["分类精度"] for q in r], ".", color=S.INK, ms=4)
        ax[2].text(j, max(q["分类精度"] for q in r) + 0.008, f"{v:.3f}", ha="center", fontsize=8.5)
    ax[2].axhline(avg(fl, "分类精度"), color=S.GREEN, lw=1.1, ls=":")
    ax[2].text(len(bars_def) - 0.6, avg(fl, "分类精度") + 0.006, "教师", ha="right", fontsize=8.5, color=S.GREEN)
    ax[2].set_xticks(range(len(bars_def)), [b[0] for b in bars_def])
    ax[2].tick_params(axis="x", labelsize=8.5)
    ax[2].set_ylim(0.3, 1.04)
    ax[2].set_ylabel("学生（通道减半）的分类精度")
    ax[2].set_title(f"蒸馏：这组设置下温度 4 高于温度 1；只有 {N_POOL} 张图可用时差一大截", fontsize=10.5)
    S.clean(ax[2])
    ax[2].grid(axis="x", visible=False)
    S.save(fig, CH, "fig05_剪枝与蒸馏")


# ---------------------------------------------------------------- 图 6：延迟的统计和按节拍选批大小
TIME_SIZE = 256
BATCHES = (1, 2, 4, 8, 16)


def machine_note():
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        cpu = ""
    return (f"{cpu or platform.processor()}，{platform.system()} {platform.machine()}，只用 CPU，torch.set_num_threads(2)，torch {torch.__version__.split('+')[0]}。"
            "测量时机器上同时有别的任务在跑，绝对值偏大，只看量级和倍数")


def clock(fn):
    t = time.perf_counter()
    fn()
    return (time.perf_counter() - t) * 1e3


def measure_latency():
    c = ctx()
    ms = models()
    net, f = c["nets"][0], c["folded"][0]
    torch.manual_seed(0)
    xs = {b: torch.rand(b, 1, TIME_SIZE, TIME_SIZE) for b in BATCHES}
    x1 = xs[1]
    fh, xh = copy.deepcopy(f).half(), x1.half()
    variants = {"原模型（卷积、BN、ReLU 分开算）": lambda: net(x1), "BN 折叠进卷积之后": lambda: f(x1),
                "结构化剪枝，通道留 0.5": lambda m=ms[("结构化", 0.5, "微调")][0]: m(x1), "结构化剪枝，通道留 0.25": lambda m=ms[("结构化", 0.25, "微调")][0]: m(x1),
                "非结构化剪枝，置零 50%": lambda m=ms[("非结构化", 0.5, "微调")][0]: m(x1), "非结构化剪枝，置零 90%": lambda m=ms[("非结构化", 0.9, "微调")][0]: m(x1)}

    def block(fn, n, skip=5):
        """连续调用 skip + n 次，前 skip 次不计。换模型、换批大小之后的头几次调用偏慢，要空跑掉。"""
        return [clock(fn) for _ in range(n + skip)][skip:]

    rng = np.random.default_rng(0)
    with torch.inference_mode():
        fresh = copy.deepcopy(f)
        first = [clock(lambda: fresh(x1)) for _ in range(3)]
        for _ in range(10):
            f(x1)
        seq = [clock(lambda: f(x1)) for _ in range(300)]
        per_b = {b: [] for b in BATCHES}
        for _ in range(4):
            for b in BATCHES:
                per_b[b] += block(lambda b=b: f(xs[b]), 25)
        torch.backends.nnpack.set_flags(False)
        off = {b: block(lambda b=b: f(xs[b]), 25) for b in (8, 16)}
        torch.backends.nnpack.set_flags(True)
        var_t = {k: [] for k in variants}
        names = list(variants)
        for _ in range(3):
            for i in rng.permutation(len(names)):
                var_t[names[i]] += block(variants[names[i]], 30)
        half = block(lambda: fh(xh), 8, skip=2)
    seq_a = np.array(seq)
    base = np.median(var_t["原模型（卷积、BN、ReLU 分开算）"])
    return {
        "说明": f"种子 0 的模型，输入换成 {TIME_SIZE}x{TIME_SIZE}（网络是全卷积加全局平均池化，可以直接换尺寸；乘加次数是 32x32 时的 64 倍）。" + machine_note() + "。"
              "连续调用：预热 10 次后对同一个输入调用 300 次。批大小：每种批大小连续测一段（先空跑 5 次，再记 25 次），5 种批大小轮 4 遍，各 100 个样本。"
              "各版本：同样分段测（先空跑 5 次，再记 30 次），各版本按随机顺序轮 3 遍，取 90 次的中位数；半精度单独测 8 次。关掉 NNPACK：torch.backends.nnpack.set_flags(False) 之后同样测一段",
        "新建模型后前 3 次调用（ms）": [sig(v) for v in first],
        "连续 300 次调用（ms）": {"平均": sig(seq_a.mean()), "中位数": sig(np.median(seq_a)), "P95": sig(np.percentile(seq_a, 95)), "P99": sig(np.percentile(seq_a, 99)),
                             "最小": sig(seq_a.min()), "最大": sig(seq_a.max()), "超过平均值的次数比例": sig((seq_a > seq_a.mean()).mean()),
                             "超过中位数 1.5 倍的次数比例": sig((seq_a > 1.5 * np.median(seq_a)).mean()),
                             "P99 是中位数的几倍": sig(np.percentile(seq_a, 99) / np.median(seq_a)), "最大是中位数的几倍": sig(seq_a.max() / np.median(seq_a))},
        "关掉 NNPACK 后整批耗时的中位数（ms）": {str(b): sig(np.median(v)) for b, v in off.items()},
        "各版本批大小 1 的耗时中位数（ms）": dict({k: sig(np.median(v)) for k, v in var_t.items()}, **{"半精度（float16，在 CPU 上）": sig(np.median(half))}),
        "各版本是原模型的几倍": dict({k: sig(np.median(v) / base) for k, v in var_t.items()}, **{"半精度（float16，在 CPU 上）": sig(np.median(half) / base)}),
        "连续 300 次的样本（ms，按测量顺序）": [sig(v, 3) for v in seq],
        "各批大小整批耗时的样本（ms）": {str(b): [sig(v, 3) for v in per_b[b]] for b in BATCHES},
    }


def measure_threads():
    """同一个模型、同样的量法，只改 PyTorch 的线程数：看凑批有没有用是不是取决于有没有闲着的核。量完把线程数改回 2。"""
    f = ctx()["folded"][0]
    torch.manual_seed(0)
    xs = {b: torch.rand(b, 1, TIME_SIZE, TIME_SIZE) for b in BATCHES}
    try:
        default = int(subprocess.run([sys.executable, "-c", "import torch; print(torch.get_num_threads())"], capture_output=True, text=True, timeout=120).stdout.strip())
    except Exception:  # noqa: BLE001
        default = 0
    out = {}
    try:
        for th in (2, 6):
            torch.set_num_threads(th)
            samples = {b: [] for b in BATCHES}
            with torch.inference_mode():
                for _ in range(4):
                    for b in BATCHES:
                        samples[b] += [clock(lambda b=b: f(xs[b])) for _ in range(30)][5:]
            per = {b: float(np.median(v)) / b for b, v in samples.items()}
            out[f"{th} 个线程"] = {"折合每张（ms，按中位数）": {str(b): sig(v) for b, v in per.items()},
                                 "吞吐量最高的批大小": int(min(per, key=per.get)), "最高的吞吐量是批大小 1 的几倍": sig(per[1] / min(per.values())),
                                 "批大小 8 折合每张是批大小 1 的比例": sig(per[8] / per[1])}
    finally:
        torch.set_num_threads(2)
    return {"说明": f"种子 0 的模型（BN 已折叠），输入 {TIME_SIZE}x{TIME_SIZE}，只改 torch.set_num_threads。每种批大小连续测一段（先空跑 5 次，再记 25 次），5 种批大小轮 4 遍，各 100 个样本。"
                  "本章其余的耗时都是 2 个线程下测的。" + machine_note(),
            "PyTorch 在这台机器上默认的线程数（新开一个进程查到的）": default, **out}


def fig_latency():
    tim = NUM["耗时"] = S.pinned(CH, "耗时", measure_latency, RETIME)
    NUM["线程数与凑批"] = S.pinned(CH, "线程数与凑批", measure_threads, RETIME)
    seq = np.array(tim["连续 300 次的样本（ms，按测量顺序）"])
    samples = {int(b): np.array(v) for b, v in tim["各批大小整批耗时的样本（ms）"].items()}
    bs = sorted(samples)
    p50 = {b: float(np.percentile(v, 50)) for b, v in samples.items()}
    p99 = {b: float(np.percentile(v, 99)) for b, v in samples.items()}
    # 节拍的两个例子都由上面钉住的样本换算出来，样本不变它们就不变
    takt = sig(1.5 * p99[1])            # 宽松：相邻两张图的间隔是单张 P99 的 1.5 倍
    tight = sig(0.8 * p50[1])           # 太紧：比单张的中位数耗时还短
    scen = {}
    for dl in (4, 8, 16):
        best, stats = hw.plan_batch(samples, takt, dl * takt, 99.0)
        best50, _ = hw.plan_batch(samples, takt, dl * takt, 50.0)
        scen[f"最多等 {dl} 个节拍"] = {"允许的最大延迟（ms）": sig(dl * takt, 3), "按 P99 选出的最大批大小": best, "按中位数选出的最大批大小": best50}
    _, stats = hw.plan_batch(samples, takt, 16 * takt, 99.0)
    best_tight, _ = hw.plan_batch(samples, tight, 1000 * tight, 99.0)
    NUM["延迟统计与节拍（由钉住的样本换算）"] = {
        "说明": "由“耗时”一组里钉住的样本算出来的量，样本不重测这些数就不变。用作业的 plan_batch 选批大小：整批耗时的分位数不超过 批大小 x 节拍（跟得上），"
              "并且 (批大小 - 1) x 节拍 + 整批耗时的分位数 不超过允许的最大延迟。宽松的例子：节拍取批大小 1 的 P99 的 1.5 倍。太紧的例子：节拍取批大小 1 的中位数耗时的 0.8 倍，允许的延迟放到 1000 个节拍",
        "各批大小": {str(b): {"整批耗时的中位数（ms）": sig(p50[b]), "整批耗时的 P99（ms）": sig(p99[b]), "P99 是中位数的几倍": sig(p99[b] / p50[b]), "折合每张（ms，按中位数）": sig(p50[b] / b),
                          "吞吐量（张每秒，按中位数）": sig(1000 * b / p50[b]), "批大小 x 节拍（ms）": sig(b * takt, 3), "跟得上吗（P99 不超过 批大小 x 节拍）": bool(stats[b] <= b * takt),
                          "第一张图等多久（ms）：(批大小 - 1) x 节拍 + P99": sig((b - 1) * takt + stats[b], 3)} for b in bs},
        "吞吐量最高的批大小": int(max(bs, key=lambda b: b / p50[b])),
        "最高的吞吐量是批大小 1 的几倍": sig(max(b / p50[b] for b in bs) / (1 / p50[1])),
        "批大小 8 的整批耗时是批大小 1 的几倍": sig(p50[8] / p50[1]),
        "批大小 16 折合每张是批大小 8 的几倍": sig((p50[16] / 16) / (p50[8] / 8)),
        "关掉 NNPACK 后批大小 16 折合每张是批大小 8 的几倍": sig((tim["关掉 NNPACK 后整批耗时的中位数（ms）"]["16"] / 16) / (tim["关掉 NNPACK 后整批耗时的中位数（ms）"]["8"] / 8)),
        "宽松的例子：节拍（ms）": takt, "宽松的例子": scen,
        "批大小 8 时第一张图等的时间是批大小 1 的 P99 的几倍": sig(((8 - 1) * takt + stats[8]) / stats[1]),
        "太紧的例子：节拍（ms）": tight, "太紧的例子：选出的批大小": best_tight,
        "100 个样本时 P99 由最大的几个样本决定": 2,
    }

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3), layout="constrained")
    ax[0].plot(np.arange(1, len(seq) + 1), seq, ".", color=S.BLUE, ms=3.5)
    for v, color, label in ((seq.mean(), S.ORANGE, "平均"), (np.percentile(seq, 50), S.GREEN, "中位数"), (np.percentile(seq, 99), S.RED, "P99")):
        ax[0].axhline(v, color=color, lw=1.3, ls="--", label=f"{label} {sig(v):g} ms")
    ax[0].set_xlabel("第几次调用（同一个输入，批大小 1）")
    ax[0].set_ylabel("耗时（ms，一台 Mac 的 CPU）")
    ax[0].set_title(f"同样的输入连续调用 300 次：最慢的一次是中位数的 {seq.max() / np.median(seq):.2g} 倍", fontsize=10.5)
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    S.clean(ax[0])
    xs_ = np.arange(len(bs))
    ax[1].plot(xs_, [p50[b] for b in bs], "o-", color=S.BLUE, lw=1.8, label="整批耗时，中位数")
    ax[1].plot(xs_, [p99[b] for b in bs], "o--", color=S.RED, lw=1.4, label="整批耗时，P99")
    ax[1].plot(xs_, [p50[b] / b for b in bs], "s-", color=S.GREEN, lw=1.8, label="折合每张（中位数）")
    ax[1].set_yscale("log")
    ticks = [v for v in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500) if min(p50[b] / b for b in bs) / 2 < v < max(p99.values()) * 2]
    ax[1].set_yticks(ticks, [f"{v:g}" for v in ticks])
    ax[1].minorticks_off()
    ax[1].set_xticks(xs_, [str(b) for b in bs])
    ax[1].set_xlabel("批大小")
    ax[1].set_ylabel("耗时（ms，对数）")
    ax[1].set_title(f"整批耗时随批大小涨；折合每张在 16 时是 8 的 {(p50[16] / 16) / (p50[8] / 8):.2g} 倍", fontsize=10.5)
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    S.clean(ax[1])
    dl = 8
    w = 0.36
    ax[2].bar(xs_ - w / 2, [p99[b] for b in bs], w, color=S.BLUE, alpha=0.9, label="整批耗时的 P99")
    ax[2].bar(xs_ + w / 2, [(b - 1) * takt + p99[b] for b in bs], w, color=S.ORANGE, alpha=0.9, label="第一张图等多久：(批大小 - 1) x 节拍 + P99")
    ax[2].plot(xs_ - w / 2, [b * takt for b in bs], "_", color=S.INK, ms=20, mew=2.2, label="跟得上的上限：批大小 x 节拍")
    ax[2].axhline(dl * takt, color=S.RED, lw=1.3, ls="--", label=f"允许的最大延迟（{dl} 个节拍）")
    ax[2].set_xticks(xs_, [str(b) for b in bs])
    ax[2].set_xlabel(f"批大小（节拍 {takt:g} ms）")
    ax[2].set_ylabel("ms")
    ax[2].set_title(f"蓝柱不超过黑线才跟得上，橙柱不超过红线才来得及：选出 {scen[f'最多等 {dl} 个节拍']['按 P99 选出的最大批大小']}", fontsize=10.5)
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    S.clean(ax[2])
    ax[2].grid(axis="x", visible=False)
    S.save(fig, CH, "fig06_延迟与批大小")


# ---------------------------------------------------------------- 其余数字
def numbers_misc():
    c = ctx()
    net, f = c["nets"][0], c["folded"][0]
    params, macs = count_cost(net)
    n_w = sum(m.weight.numel() for m in list(f.convs) + [f.fc])
    n_b = sum(m.bias.numel() for m in list(f.convs) + [f.fc])
    NUM["模型规模和存储"] = {
        "说明": "本章的小模型。存储按“每个权重占几个字节”直接算，不含文件格式本身的开销；真模型的大小以模型卡和实际文件为准",
        "参数个数（训练时，含 BN 的缩放和偏移）": params, "乘加次数（32x32 输入）": macs, "乘加次数（256x256 输入）": count_cost(net, TIME_SIZE)[1],
        "折叠后的权重个数": n_w, "折叠后的偏置个数": n_b,
        "权重按 float32 存（kB）": r3(4 * n_w / 1000), "按 float16 存（kB）": r3(2 * n_w / 1000), "按 8 位整数存（kB）": r3(n_w / 1000), "按 4 位整数存（kB）": r3(n_w / 2000),
        "8 位能表示的整数个数": 2 ** 8, "8 位对称量化用到的整数个数（-127 到 127）": 2 ** 8 - 1, "4 位能表示的整数个数": 2 ** 4, "4 位对称量化用到的整数个数（-7 到 7）": 2 ** 4 - 1,
        "学生的通道数": list(STUDENT_W),
    }
    # 节拍换算的演示：数字是假设的，只为了走一遍算法
    wafers_per_hour, images_per_wafer, overhead = 20, 10000, 0.3
    img_side, tile = 4096, 256
    per_wafer = 3600 / wafers_per_hour
    budget = per_wafer * (1 - overhead)
    takt_img = 1000 * budget / images_per_wafer
    tiles = (img_side // tile) ** 2
    NUM["节拍换算的例子（数字是假设的）"] = {
        "说明": "为了演示换算步骤随手取的数，不代表任何设备、任何产线。假设：每小时要检 20 片，每片拍 10000 张图，每片的时间里有 30% 花在上下片、对准这些和计算无关的事情上；"
              "每张图 4096x4096，切成 256x256 的块分别送进模型，不重叠",
        "每小时片数": wafers_per_hour, "每片的图像数": images_per_wafer, "和计算无关的时间占比": overhead,
        "每片的总时间（秒）": per_wafer, "每片留给成像和计算的时间（秒）": budget, "每张图的节拍（ms）": r3(takt_img),
        "要求的吞吐量（张每秒）": r3(images_per_wafer / budget), "图像边长（px）": img_side, "块的边长（px）": tile, "每张图切成几块": tiles,
        "每块的节拍（ms）": r4(takt_img / tiles), "要求的吞吐量（块每秒）": int(round(tiles * images_per_wafer / budget)),
        "一片的像素数（亿）": r3(images_per_wafer * img_side ** 2 / 1e8),
        "块之间重叠 32 px 时每张图切成几块": int(math.ceil((img_side - 32) / (tile - 32)) ** 2),
        "重叠 32 px 时块数是不重叠的几倍": round(math.ceil((img_side - 32) / (tile - 32)) ** 2 / tiles, 2),
    }
    # 作业文档字符串里的例子，用参考实现核对，讲义第 12 节引用
    s, zp, qmin, qmax = hw.quant_params(-1.0, 3.0, 8)
    q, xh = hw.quantize_dequantize(np.array([-5.0, -1.0, 0.0, 0.01, 3.0, 9.0]), s, zp, qmin, qmax)
    best, stats = hw.plan_batch({1: [10.0, 10.0], 2: [12.0, 14.0], 4: [30.0, 34.0]}, 8.0, 40.0, 50.0)
    NUM["作业自检"] = {
        "说明": "用参考实现核对的几个小例子",
        "[-1, 3]、8 位非对称：实数 0 量化再反量化的误差": float(abs(xh[2])), "超出范围的 -5 和 9 被截到的整数": [int(q[0]), int(q[-1])],
        "plan_batch 小例子（节拍 8 ms，最多等 40 ms，按中位数）：各批大小的延迟": {str(k): v for k, v in stats.items()}, "选出的批大小": best,
    }
    NUM["引用的常数（不是实测）"] = {
        "说明": "讲义里提到的几个常数，列在这里只是为了让每个数都有出处。读者的显卡型号和显存来自课程设定；opset 的取值只是示例，以你安装的版本支持的范围为准",
        "读者的显卡": "RTX 4060", "显存（GB）": 8, "示例里的 opset": 18, "示例里的输入尺寸": [1, 1, 32, 32], "示例里动态批的最小、常用、最大": [1, 8, 32],
        "写课程的机器上的 PyTorch 版本": torch.__version__.split("+")[0], "ONNX 导出默认改用新导出器的 PyTorch 版本（官方文档的说法）": "2.9",
        "8 位有符号整数的最小值（对称量化不用它）": -128, "4 除以 255": round(4 / 255, 6), "4 除以 256": 4 / 256, "一小时的秒数": 3600,
        "Notebook 里小模型的训练步数": 400,
        "float32 的有效十进制位数（约）": 7, "float32 能连续精确表示的最大整数": 2 ** 24,
    }
    NUM["演示页"] = {
        "说明": "demos/量化误差.html：一组 4000 个数（主体加可调的离群值），在浏览器里按作业的同一套公式算量化参数、量化再反量化，画直方图和误差。页面里的数值实时计算，讲义不引用",
        "样本个数": 4000, "位数范围": [2, 8], "百分位": 99.9,
    }


def main() -> None:
    S.setup()
    warnings.simplefilter("ignore")
    t_all = time.perf_counter()
    for fn in (ctx, models, fig_shift, table_fold_export, fig_basics, fig_calibration, fig_network, table_half, fig_prune_kd, fig_latency, numbers_misc):
        if ONLY and fn not in (ctx, models) and ONLY not in fn.__name__:
            continue
        t = time.perf_counter()
        fn()
        TIMES[fn.__name__] = time.perf_counter() - t
        print(f"完成 {fn.__name__}（{TIMES[fn.__name__]:.0f} 秒）", flush=True)
    total = time.perf_counter() - t_all
    if ONLY:
        print(f"只跑了一部分（CH31_ONLY={ONLY}），不写 实测数字.json。用时 {total:.0f} 秒")
        return

    NUM["训练规模"] = {
        "说明": f"整个脚本一共训练了多少个模型（都是上面那种小卷积网络）。从头训练都是 {STEPS} 步，微调 {FT_STEPS} 步，量化感知训练和它的对照各 {QAT_STEPS} 步",
        "浮点模型（教师）": len(SEEDS), "剪枝后微调": sum(1 for k in _MODELS if k[-1] == "微调") * len(SEEDS), "学生（直接训练和蒸馏）": sum(1 for k in _MODELS if k[0] == "学生") * len(SEEDS),
        "量化感知训练和对照": 2 * len(SEEDS),
    }
    NUM["训练规模"]["合计"] = sum(v for k, v in NUM["训练规模"].items() if k != "说明")
    NUM["脚本耗时"] = S.pinned(CH, "脚本耗时", lambda: {
        "说明": "在一台 Mac 的 CPU 上（2 个线程，同时还有别的程序在跑）跑完整个配图脚本的一次结果，只看量级。设置 CH31_RETIME=1 重测",
        "整个配图脚本（秒）": sig(total), "其中训练浮点模型、剪枝微调和学生（秒）": sig(TIMES["ctx"] + TIMES["models"]),
        "训练一个浮点模型（秒，600 步）": sig(TIMES["ctx"] / len(SEEDS)),
    }, RETIME)
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=2))
    print(f"总耗时 {total:.0f} 秒")


if __name__ == "__main__":
    main()
