"""第 8 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch08.py

输出：<主分支>/08_神经网络基础/figures/*.svg 和 实测数字.json。
两层网络的前向和交叉熵全部调用本章作业的参考实现 two_layer_forward、cross_entropy_from_logits，
反向传播和三种优化器在本脚本里用 numpy 手写（并和 torch 的 autograd、torch.optim 对过）。
数据用代码生成（本机没有 MNIST）：XOR 点集和 12x12 的合成数字。只和 PyTorch 接口有关的实验用 torch 在 CPU 上做。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。整个脚本在一台 Mac 的 CPU 上一两分钟跑完，重复运行结果相同。
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

torch.set_num_threads(2)
warnings.filterwarnings("ignore")

CH = "08_神经网络基础"
mod = S.load_solution(CH, "作业_手写两层网络.py")
fwd, ce = mod.two_layer_forward, mod.cross_entropy_from_logits
NUM: dict = {}

SIZE, D, C = 12, 144, 10
N_TRAIN, N_VAL = 4000, 2000
DATA_NOTE = ("合成数字：用 cv2.putText 在 48x48 的画布上画 0 到 9（字号、笔画粗细随机，位置偏移最多 6 px，旋转最多 12 度），"
             "缩小到 12x12，再加标准差 0.15 的高斯噪声，展平成 144 维，像素值大致在 0 到 1；训练集 4000 张，验证集 2000 张，10 类")


def r(x, k=3):
    return round(float(x), k)


def sci(x, digits=2):
    """保留 digits 位有效数字。"""
    x = float(x)
    return x if not math.isfinite(x) or x == 0 else float(f"{x:.{digits - 1}e}")


# ---------------------------------------------------------------- 数据
def digits(n, seed, label_noise=0.0):
    rng = np.random.default_rng(seed)
    big = 4 * SIZE
    x = np.zeros((n, SIZE * SIZE))
    y = rng.integers(0, 10, n)
    for i in range(n):
        img = np.zeros((big, big), np.uint8)
        scale = rng.uniform(1.1, 1.5)
        thick = int(rng.integers(2, 5))
        (tw, th), _ = cv2.getTextSize(str(y[i]), cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ox = (big - tw) // 2 + int(rng.integers(-6, 7))
        oy = (big + th) // 2 + int(rng.integers(-6, 7))
        cv2.putText(img, str(y[i]), (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thick, cv2.LINE_AA)
        M = cv2.getRotationMatrix2D((big / 2, big / 2), rng.uniform(-12, 12), 1.0)
        img = cv2.warpAffine(img, M, (big, big), flags=cv2.INTER_LINEAR)
        small = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(float) / 255.0
        x[i] = (small + rng.normal(0, 0.15, (SIZE, SIZE))).ravel()
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        y = y.copy()
        y[flip] = rng.integers(0, 10, int(flip.sum()))
    return x, y


def xor_data(seed=8, rep=32, noise=0.08):
    """和本章 Notebook 相同的点集：四个角各 32 个点，加标准差 0.08 的抖动。"""
    rng = np.random.default_rng(seed)
    base = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    x = np.tile(base, (rep, 1)) + noise * rng.normal(size=(4 * rep, 2))
    return x, np.tile(np.array([0, 1, 1, 0]), rep)


XTR, YTR = digits(N_TRAIN, 0)
XVA, YVA = digits(N_VAL, 1)


# ---------------------------------------------------------------- 两层网络（numpy）
def init(d, h, c, rng, scale=1.0, kind="he"):
    """he：权重 N(0, 2/fan_in)、偏置 0；torch：和 nn.Linear 的默认初始化同分布，权重和偏置都是 U(-1/sqrt(fan_in), 1/sqrt(fan_in))。"""
    if kind == "he":
        return [rng.normal(0, scale * np.sqrt(2 / d), (d, h)), np.zeros(h), rng.normal(0, scale * np.sqrt(2 / h), (h, c)), np.zeros(c)]
    b1, b2 = 1 / np.sqrt(d), 1 / np.sqrt(h)
    return [rng.uniform(-b1, b1, (d, h)) * scale, rng.uniform(-b1, b1, h) * scale, rng.uniform(-b2, b2, (h, c)) * scale, rng.uniform(-b2, b2, c) * scale]


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def forward(p, x, act="relu"):
    w1, b1, w2, b2 = p
    if act == "relu":
        return fwd(x, w1, b1, w2, b2)
    h = np.tanh(x @ w1 + b1) if act == "tanh" else x @ w1 + b1
    return h, h @ w2 + b2


def loss_grad(p, x, y, act="relu"):
    """平均交叉熵和四个参数的梯度。前向和损失调用参考实现。"""
    h, logits = forward(p, x, act)
    loss = ce(logits, y)
    d = softmax(logits)
    d[np.arange(len(y)), y] -= 1
    d /= len(y)
    dh = d @ p[2].T
    if act == "relu":
        dh = dh * (h > 0)
    elif act == "tanh":
        dh = dh * (1 - h * h)
    return loss, [x.T @ dh, dh.sum(0), h.T @ d, d.sum(0)]


def accuracy(p, x, y, act="relu"):
    return float((forward(p, x, act)[1].argmax(1) == y).mean())


def dead_frac(p, x):
    return float((forward(p, x)[0].max(axis=0) <= 0).mean())


def flat(gs):
    return np.concatenate([a.ravel() for a in gs])


class Opt:
    """SGD、动量、Adam，更新公式和 torch.optim 的 SGD、SGD(momentum)、Adam 相同。wd 是加在梯度上的 L2（只作用于权重矩阵）。"""

    def __init__(self, kind, lr, beta=0.9, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.kind, self.lr, self.beta, self.b1, self.b2, self.eps, self.wd = kind, lr, beta, b1, b2, eps, wd
        self.t, self.m, self.v = 0, None, None

    def step(self, p, g):
        if self.wd:
            g = [gi + self.wd * pi if pi.ndim == 2 else gi for gi, pi in zip(g, p)]
        self.t += 1
        if self.m is None:
            self.m = [np.zeros_like(pi) for pi in p]
            self.v = [np.zeros_like(pi) for pi in p]
        for pi, gi, mi, vi in zip(p, g, self.m, self.v):
            if self.kind == "sgd":
                pi -= self.lr * gi
            elif self.kind == "mom":
                mi *= self.beta
                mi += gi
                pi -= self.lr * mi
            else:
                mi *= self.b1
                mi += (1 - self.b1) * gi
                vi *= self.b2
                vi += (1 - self.b2) * gi * gi
                pi -= self.lr * (mi / (1 - self.b1 ** self.t)) / (np.sqrt(vi / (1 - self.b2 ** self.t)) + self.eps)


def train(p, x, y, opt, steps, batch=None, rng=None, act="relu"):
    """返回每一步更新前在当前 batch 上的损失。损失变成 nan 或 inf 就停。"""
    hist = []
    for _ in range(steps):
        if batch is None or batch >= len(x):
            xb, yb = x, y
        else:
            idx = rng.integers(0, len(x), batch)
            xb, yb = x[idx], y[idx]
        with np.errstate(all="ignore"):
            loss, g = loss_grad(p, xb, yb, act)
            hist.append(loss)
            if not np.isfinite(loss):
                break
            opt.step(p, g)
    return hist


def full_loss(p, x, y):
    with np.errstate(all="ignore"):
        v = loss_grad(p, x, y)[0]
    return v if np.isfinite(v) else float("inf")


# ---------------------------------------------------------------- torch 小工具
def t_mlp(seed, h=128, p_drop=0.0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(D, h), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(h, C))


TX, TY = torch.tensor(XTR, dtype=torch.float32), torch.tensor(YTR)
TXV, TYV = torch.tensor(XVA, dtype=torch.float32), torch.tensor(YVA)


def t_fit(model, steps=600, lr=0.1, opt="sgd", bug=None, batch=64, seed=0, xs=None, ys=None, wd=0.0):
    xs, ys = (TX if xs is None else xs), (TY if ys is None else ys)
    g = torch.Generator().manual_seed(seed)
    make = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[opt]
    o = make(model.parameters(), lr=lr, weight_decay=wd)
    lf = nn.CrossEntropyLoss()
    model.train()
    losses = []
    for _ in range(steps):
        idx = torch.randint(0, len(xs), (batch,), generator=g)
        xb, yb = xs[idx], ys[idx]
        if bug == "标签和图片错位":
            yb = yb[torch.randperm(batch, generator=g)]
        if bug != "忘了 zero_grad":
            o.zero_grad()
        out = model(xb)
        if bug == "先 softmax 再交叉熵":
            out = torch.softmax(out, dim=1)
        loss = lf(out, yb)
        loss.backward()
        if bug != "忘了 step":
            o.step()
        losses.append(loss.item())
    return losses


def t_acc(model, xv=None, train_mode=False):
    model.train(train_mode)
    with torch.no_grad():
        return (model(TXV if xv is None else xv).argmax(1) == TYV).float().mean().item()


# ---------------------------------------------------------------- 图 1：XOR 与隐藏单元数
def fig_xor():
    x, y = xor_data()
    seeds, steps, lr = 200, 250, 0.05
    hs = [1, 2, 3, 4, 8, 16, 32]
    table, keep = {}, {}
    for h in hs:
        accs, dead = [], []
        for s in range(seeds):
            rng = np.random.default_rng(1000 + s)
            p = init(2, h, 2, rng, kind="torch")
            train(p, x, y, Opt("adam", lr), steps)
            a = accuracy(p, x, y)
            accs.append(a)
            dead.append(int((forward(p, x)[0].max(axis=0) <= 0).sum()))
            if (h == 8 and a == 1.0 and "ok" not in keep) or (h == 2 and abs(a - 0.75) < 0.02 and "bad" not in keep):
                keep["ok" if h == 8 else "bad"] = [v.copy() for v in p]
        accs = np.array(accs)
        table[str(h)] = {"学会的次数（训练准确率 100%）": int((accs == 1.0).sum()), "成功率（%）": r(100 * (accs == 1.0).mean(), 1),
                         "停在 75% 附近的次数": int((np.abs(accs - 0.75) < 0.02).sum()), "停在 50% 附近的次数": int((np.abs(accs - 0.5) < 0.02).sum()),
                         "训练结束时对全部样本都输出 0 的隐藏单元数（平均）": r(np.mean(dead), 2)}
    lin_acc, lin_loss, lin_p = [], [], None
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        p = init(2, 8, 2, rng, kind="torch")
        train(p, x, y, Opt("adam", lr), steps, act="none")
        lin_acc.append(accuracy(p, x, y, "none"))
        lin_loss.append(loss_grad(p, x, y, "none")[0])
        lin_p = lin_p or [v.copy() for v in p]
    long_ok = 0
    for s in range(seeds):
        p = init(2, 2, 2, np.random.default_rng(1000 + s), kind="torch")
        train(p, x, y, Opt("adam", lr), 1000)
        long_ok += accuracy(p, x, y) == 1.0
    rng = np.random.default_rng(5)
    w1, b1, w2, b2 = init(3, 4, 2, rng, kind="torch")
    xx = rng.normal(size=(5, 3))
    two = (xx @ w1 + b1) @ w2 + b2
    one = xx @ (w1 @ w2) + (b1 @ w2 + b2)
    NUM["XOR"] = {
        "说明": "和 Notebook 相同的点集（四个角各 32 个点共 128 个，抖动标准差 0.08）和训练设置（Adam，学习率 0.05，全批量 250 步），"
              "权重按 nn.Linear 的默认方式初始化，用 numpy 重做，每种隐藏单元数换 200 个随机种子；学会 = 训练准确率 100%",
        "按隐藏单元数": table,
        "2 个隐藏单元，步数从 250 加到 1000（同样 200 个种子）": {"学会的次数": int(long_ok), "成功率（%）": r(100 * long_ok / seeds, 1)},
        "去掉激活函数的两层网络（8 个隐藏单元，200 个种子）": {"训练准确率最小（%）": r(100 * min(lin_acc), 1), "训练准确率中位数（%）": r(100 * np.median(lin_acc), 1),
                                              "训练准确率最大（%）": r(100 * max(lin_acc), 1), "最终损失中位数": r(np.median(lin_loss), 3), "ln2": r(math.log(2), 3)},
        "两层线性叠加与合并成一层的输出最大差": sci(np.abs(two - one).max()),
    }

    gx, gy = np.meshgrid(np.linspace(-0.4, 1.4, 181), np.linspace(-0.4, 1.4, 181))
    grid = np.c_[gx.ravel(), gy.ravel()]
    fig, ax = plt.subplots(1, 4, figsize=(14.5, 3.7), gridspec_kw={"width_ratios": [1, 1, 1, 1.35]}, layout="constrained")
    panels = [("去掉激活：边界是直线，准确率 " + f"{100 * accuracy(lin_p, x, y, 'none'):.0f}%", lin_p, "none"),
              ("8 个隐藏单元，学会的一次", keep["ok"], "relu"),
              ("2 个隐藏单元，停在 " + f"{100 * accuracy(keep['bad'], x, y):.0f}% 的一次", keep["bad"], "relu")]
    for a_, (title, p, act) in zip(ax[:3], panels):
        z = forward(p, grid, act)[1]
        a_.contourf(gx, gy, (z[:, 1] - z[:, 0]).reshape(gx.shape), levels=[-1e9, 0, 1e9], colors=["#dbe7ff", "#ffe3cc"])
        a_.scatter(x[y == 0, 0], x[y == 0, 1], s=9, color=S.BLUE, label="类别 0")
        a_.scatter(x[y == 1, 0], x[y == 1, 1], s=9, color=S.ORANGE, marker="s", label="类别 1")
        a_.set_title(title, fontsize=10.5)
        a_.set_aspect("equal")
        S.image_axes(a_)
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    rate = [table[str(h)]["成功率（%）"] for h in hs]
    bars = ax[3].bar(range(len(hs)), rate, 0.6, color=S.BLUE, alpha=0.85)
    for b, v in zip(bars, rate):
        ax[3].text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}", ha="center", fontsize=9.5)
    ax[3].set_xticks(range(len(hs)), [str(h) for h in hs])
    ax[3].set_ylim(0, 112)
    ax[3].set_xlabel("隐藏单元数")
    ax[3].set_ylabel("200 个种子里学会的比例（%）")
    ax[3].set_title(f"2 个单元理论上够用，实际只成 {rate[1]:.0f}%")
    S.clean(ax[3])
    ax[3].grid(axis="x", visible=False)
    S.save(fig, CH, "fig01_XOR与隐藏单元数")


# ---------------------------------------------------------------- 表：线性、无激活、ReLU 三种模型
def table_baseline():
    steps, lr, h = 2000, 0.3, 64
    rows = {}
    rng = np.random.default_rng(0)
    w, b = rng.normal(0, np.sqrt(1 / D), (D, C)), np.zeros(C)
    for _ in range(steps):
        idx = rng.integers(0, N_TRAIN, 64)
        d = softmax(XTR[idx] @ w + b)
        d[np.arange(64), YTR[idx]] -= 1
        w -= lr * XTR[idx].T @ d / 64
        b -= lr * d.sum(0) / 64
    rows["一层线性（softmax 回归）"] = {"参数量": D * C + C, "训练准确率（%）": r(100 * ((XTR @ w + b).argmax(1) == YTR).mean(), 1),
                               "验证准确率（%）": r(100 * ((XVA @ w + b).argmax(1) == YVA).mean(), 1)}
    for name, act in (("两层，去掉激活", "none"), ("两层，ReLU", "relu")):
        rng = np.random.default_rng(0)
        p = init(D, h, C, rng)
        train(p, XTR, YTR, Opt("sgd", lr), steps, batch=64, rng=rng, act=act)
        rows[name] = {"参数量": D * h + h + h * C + C, "训练准确率（%）": r(100 * accuracy(p, XTR, YTR, act), 1), "验证准确率（%）": r(100 * accuracy(p, XVA, YVA, act), 1)}
    NUM["数据"] = DATA_NOTE
    NUM["线性与非线性"] = {"说明": "合成数字，64 个隐藏单元，SGD，学习率 0.3，batch 64，训练 2000 步，种子 0", "结果": rows,
                      "讲义例子 784 -> 128 -> 10 的参数量": 784 * 128 + 128 + 128 * 10 + 10, "其中第一层": 784 * 128 + 128, "其中第二层": 128 * 10 + 10,
                      "第一层占全部参数的比例（%）": r(100 * (784 * 128 + 128) / (784 * 128 + 128 + 128 * 10 + 10), 1),
                      "合成数字上两层网络的参数量是一层线性的多少倍": r((D * h + h + h * C + C) / (D * C + C), 1)}


# ---------------------------------------------------------------- 图 2：交叉熵的数值稳定性
def naive_ce(logits, labels):
    """直接照公式算：先 exp，再归一化，再取对数。"""
    e = np.exp(logits)
    pr = e / e.sum(axis=1, keepdims=True)
    return float(-np.log(pr[np.arange(len(labels)), labels]).mean())


def fig_ce():
    lab = np.array([2])
    scales = np.unique(np.round(np.logspace(0, 3.2, 120)))
    curves, first_bad = {}, {}
    with np.errstate(all="ignore"):
        for name, dt in (("float32", np.float32), ("float64", np.float64)):
            curves[name] = {"直接算": [naive_ce(np.array([[s, 0, -s]], dt), lab) for s in scales],
                            "参考实现": [ce(np.array([[s, 0, -s]], dt), lab) for s in scales]}
            inf_at = nan_at = None
            for s in range(1, 2000):
                if inf_at is None and np.isinf(naive_ce(np.array([[s, 0, -s]], dt), lab)):
                    inf_at = s
                if nan_at is None and np.isnan(naive_ce(np.array([[s, 0, -s]], dt), np.array([0]))):
                    nan_at = s
            first_bad[name] = {"正确类别是最后一个（预测错）时，直接算第一次得到 inf 的 s": inf_at, "正确类别是第一个（预测对）时，直接算第一次得到 nan 的 s": nan_at,
                               "exp 的上溢界 ln(最大浮点数)": r(np.log(np.finfo(dt).max), 2),
                               "exp 结果变成 0 的下界 ln(最小正数)": r(np.log(np.finfo(dt).smallest_subnormal), 2),
                               "s=50（真值约 100）时直接算的结果": r(naive_ce(np.array([[50, 0, -50]], dt), lab), 3),
                               "s=50 时参考实现的结果": r(ce(np.array([[50, 0, -50]], dt), lab), 3)}
        case = np.array([[1000.0, 1001.0], [-1000.0, -999.0]])
        case_naive = naive_ce(case, np.array([1, 0]))
    raw = forward(init(D, 64, C, np.random.default_rng(0)), XTR * 255.0)[1]
    norm = forward(init(D, 64, C, np.random.default_rng(0)), XTR)[1]
    NUM["交叉熵的数值稳定性"] = {
        "说明": "logits 取 [s, 0, -s]，正确类别是最后一个，真值约为 2s；直接算 = exp 后归一化再取对数；参考实现 = 先减去每行最大值的 log-sum-exp 写法",
        "按浮点类型": first_bad,
        "验收脚本里的用例 [[1000,1001],[-1000,-999]]": {"直接算": None if np.isnan(case_naive) else r(case_naive, 4), "直接算得到的是 nan": bool(np.isnan(case_naive)),
                                               "参考实现": r(ce(case, np.array([1, 0])), 4), "理论值 (ln(1+e^-1)+ln(1+e))/2": r((math.log1p(math.exp(-1)) + math.log1p(math.e)) / 2, 4)},
        "讲义里的两个数": {"-ln(0.9)": r(-math.log(0.9), 3), "-ln(0.01)": r(-math.log(0.01), 3)},
        "初始化后 logits 的最大绝对值（He 初始化，64 个隐藏单元）": {"输入在 0 到 1": r(np.abs(norm).max(), 1), "输入乘 255（不归一化）": r(np.abs(raw).max(), 0)},
    }

    ps = [0.5, 0.1, 0.01, 0.001]
    NUM["MSE 与交叉熵的梯度"] = {
        "说明": "二分类，输出 p = sigmoid(z)，真实标签是 1；p 是模型给正确类别的概率；交叉熵对 z 的导数是 p-1，平方误差 (p-1)^2 对 z 的导数是 2(p-1)p(1-p)",
        "p": ps, "交叉熵的梯度绝对值": [r(1 - p, 4) for p in ps], "平方误差的梯度绝对值": [sci(2 * (1 - p) * p * (1 - p), 3) for p in ps],
        "p=0.01 时交叉熵梯度是平方误差的多少倍": r((1 - 0.01) / (2 * 0.99 * 0.01 * 0.99), 0)}

    runs = {}
    for opt, lr in (("sgd", 0.1), ("adam", 1e-3)):
        for bug in (None, "先 softmax 再交叉熵"):
            m = t_mlp(0)
            ls = t_fit(m, opt=opt, lr=lr, bug=bug)
            runs[(opt, bug)] = (ls, t_acc(m))
    floor = -math.log(math.e / (math.e + C - 1))
    NUM["先 softmax 再交叉熵"] = {
        "说明": "合成数字，nn.Linear(144,128)-ReLU-nn.Linear(128,10)，batch 64，训练 600 步，种子 0；正确写法把 logits 直接交给 CrossEntropyLoss，错误写法先做一次 softmax",
        "10 类时错误写法的损失下界 -ln(e/(e+9))": r(floor, 3), "10 类时均匀输出的损失 ln10": r(math.log(10), 3),
        "结果": {f"{'SGD，学习率 0.1' if o == 'sgd' else 'Adam，学习率 0.001'}，{'正确写法' if b is None else '先 softmax'}":
                   {"最后 50 步的平均损失": r(np.mean(ls[-50:]), 3), "验证准确率（%）": r(100 * a, 1)} for (o, b), (ls, a) in runs.items()}}

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 3.9), layout="constrained")
    with np.errstate(all="ignore"):
        for name, color in (("float32", S.ORANGE), ("float64", S.BLUE)):
            v = np.array(curves[name]["直接算"], float)
            ok = np.isfinite(v)
            ax[0].plot(scales[ok], v[ok], "-", color=color, lw=2, label=f"直接算，{name}")
            ax[0].plot(scales[ok][-1], v[ok][-1], "x", color=color, ms=9, mew=2)
        ax[0].plot(scales, curves["float32"]["参考实现"], "--", color=S.GREEN, lw=1.4, label="减最大值的写法，float32")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    ax[0].set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    ax[0].minorticks_off()
    ax[0].set_xlabel("logits 的大小 s（logits = [s, 0, -s]）")
    ax[0].set_ylabel("算出的损失（真值约 2s）")
    ax[0].set_title(f"直接算：float32 到 s={first_bad['float32']['正确类别是最后一个（预测错）时，直接算第一次得到 inf 的 s']} 就断了（叉号）")
    ax[0].legend(fontsize=8.5, loc="upper left")
    S.clean(ax[0])
    pp = np.logspace(-4, -0.01, 200)
    ax[1].plot(pp, 1 - pp, "-", color=S.BLUE, lw=2, label="交叉熵：|p - 1|")
    ax[1].plot(pp, 2 * (1 - pp) * pp * (1 - pp), "-", color=S.RED, lw=2, label="平方误差：2(1-p)p(1-p)")
    ax[1].set_xscale("log")
    ax[1].set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ["0.0001", "0.001", "0.01", "0.1", "1"])
    ax[1].minorticks_off()
    ax[1].set_xlabel("模型给正确类别的概率 p")
    ax[1].set_ylabel("损失对 z 的梯度绝对值")
    ax[1].set_title("错得越自信，平方误差的梯度越小")
    ax[1].legend(fontsize=8.5, loc="center left")
    S.clean(ax[1])

    def smooth(v, k=20):
        return np.convolve(v, np.ones(k) / k, mode="valid")
    for (o, b), (ls, _) in runs.items():
        ax[2].plot(np.arange(len(smooth(ls))) + 20, smooth(ls), "-" if b is None else "--", color=S.BLUE if o == "sgd" else S.ORANGE, lw=1.8,
                   label=f"{'SGD' if o == 'sgd' else 'Adam'}，{'正确' if b is None else '先 softmax'}")
    ax[2].axhline(floor, color=S.MUTED, lw=1, ls=":")
    ax[2].text(590, floor - 0.02, f"先 softmax 时的下界 {floor:.2f}", color=S.MUTED, fontsize=9, ha="right", va="top")
    ax[2].set_xlabel("训练步数")
    ax[2].set_ylabel("训练损失（20 步滑动平均）")
    ax[2].set_ylim(0.4, 2.4)
    ax[2].set_title("多做一次 softmax：不报错，只是学得慢")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    S.save(fig, CH, "fig02_交叉熵的数值稳定性")


# ---------------------------------------------------------------- 图 3：梯度检查
def rel_err(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b)))


def fig_gradcheck():
    x, y, hu = XTR[:32], YTR[:32], 16
    hs = [10.0 ** -k for k in range(1, 14)]

    def sweep(dtype, act):
        p64 = init(D, hu, C, np.random.default_rng(1))
        _, g = loss_grad(p64, x, y, act)
        p, xd = [a.astype(dtype) for a in p64], x.astype(dtype)
        rng = np.random.default_rng(3)
        picks = [(k, int(i)) for k in range(4) for i in rng.integers(0, p[k].size, 40)]
        ana = np.array([g[k].flat[i] for k, i in picks])
        out = {"前向差分": [], "中心差分": [], "扰动后有 ReLU 换了开关的参数占比（%）": []}
        for h in hs:
            l0 = float(ce(forward(p, xd, act)[1], y))
            fd, cd, flips = [], [], 0
            for k, i in picks:
                old = p[k].flat[i]
                p[k].flat[i] = old + dtype(h)
                zp = xd @ p[0] + p[1]
                lp = float(ce(forward(p, xd, act)[1], y))
                p[k].flat[i] = old - dtype(h)
                zm = xd @ p[0] + p[1]
                lm = float(ce(forward(p, xd, act)[1], y))
                p[k].flat[i] = old
                fd.append((lp - l0) / h)
                cd.append((lp - lm) / (2 * h))
                flips += bool(((zp > 0) != (zm > 0)).any())
            out["前向差分"].append(rel_err(fd, ana))
            out["中心差分"].append(rel_err(cd, ana))
            out["扰动后有 ReLU 换了开关的参数占比（%）"].append(100 * flips / len(picks))
        return out

    res = {("float64", "relu"): sweep(np.float64, "relu"), ("float64", "tanh"): sweep(np.float64, "tanh"), ("float32", "relu"): sweep(np.float32, "relu")}

    def best(v):
        i = int(np.argmin(v))
        return {"最小相对误差": sci(v[i]), "对应的步长": hs[i]}
    names = {("float64", "relu"): "float64，ReLU 网络", ("float64", "tanh"): "float64，把激活换成 tanh", ("float32", "relu"): "float32，ReLU 网络"}
    NUM["梯度检查"] = {
        "说明": "两层网络 144 -> 16 -> 10，32 个合成数字样本，He 初始化；随机抽 160 个参数（四组参数各 40 个），数值梯度和手写反向传播的结果比；"
              "相对误差 = |数值 - 解析| / (|数值| + |解析|)，按 160 个参数的向量范数算；解析梯度一律用 float64 计算",
        "步长": hs,
        "相对误差": {names[k]: {m: [sci(e) for e in v[m]] for m in ("前向差分", "中心差分")} for k, v in res.items()},
        "最准的步长": {names[k]: {m: best(v[m]) for m in ("前向差分", "中心差分")} for k, v in res.items()},
        "float64 的 ReLU 网络里，扰动后有 ReLU 换了开关的参数占比（%）": {f"{h:.0e}": r(v, 1) for h, v in zip(hs[:6], res[("float64", "relu")]["扰动后有 ReLU 换了开关的参数占比（%）"][:6])},
    }

    # 常见的反向传播错误，梯度检查各给出多大的误差
    p = init(D, hu, C, np.random.default_rng(1))
    _, g = loss_grad(p, x, y)
    num = []
    for k in range(4):
        gk = np.zeros_like(p[k])
        for i in range(p[k].size):
            old = p[k].flat[i]
            p[k].flat[i] = old + 1e-5
            lp = ce(fwd(x, *p)[1], y)
            p[k].flat[i] = old - 1e-5
            lm = ce(fwd(x, *p)[1], y)
            p[k].flat[i] = old
            gk.flat[i] = (lp - lm) / 2e-5
        num.append(gk)
    w1, b1, w2, b2 = p
    h, logits = fwd(x, *p)
    B = len(y)
    d = softmax(logits)
    d[np.arange(B), y] -= 1
    dn = d / B
    dz = (dn @ w2.T) * (h > 0)
    dz_raw = (d @ w2.T) * (h > 0)
    d_plain = softmax(logits) / B
    dz_plain = (d_plain @ w2.T) * (h > 0)
    variants = {
        "正确的实现": g,
        "忘了除以 batch 大小": [x.T @ dz_raw, dz_raw.sum(0), h.T @ d, d.sum(0)],
        "ReLU 的掩码写成 h >= 0（等于没有掩码）": [x.T @ ((dn @ w2.T) * (h >= 0)), ((dn @ w2.T) * (h >= 0)).sum(0), h.T @ dn, dn.sum(0)],
        "偏置的梯度对 batch 又取了一次平均": [x.T @ dz, dz.mean(0), h.T @ dn, dn.mean(0)],
        "softmax 之后忘了减 one-hot": [x.T @ dz_plain, dz_plain.sum(0), h.T @ d_plain, d_plain.sum(0)],
    }
    groups = ("w1", "b1", "w2", "b2")
    bug_tab = {k: {"全部参数": sci(rel_err(flat(v), flat(num))), **{gname: sci(rel_err(a, b)) for gname, a, b in zip(groups, v, num)}} for k, v in variants.items()}
    tp = [torch.tensor(a, requires_grad=True) for a in p]
    tx, ty = torch.tensor(x), torch.tensor(y)

    def t_loss():
        return nn.functional.cross_entropy(torch.relu(tx @ tp[0] + tp[1]) @ tp[2] + tp[3], ty)
    t_loss().backward()
    g_auto = [a.grad.numpy().copy() for a in tp]
    t_loss().backward()
    NUM["反向传播的错误"] = {
        "说明": "同一个网络和 32 个样本，float64，中心差分步长 1e-5，对全部 2490 个参数算数值梯度，再和几种写法的解析梯度比",
        "参数个数": int(sum(a.size for a in p)), "batch 大小": B, "相对误差": bug_tab,
        "隐藏层输出里正好等于 0 的比例（%）": r(100 * (h == 0).mean(), 1), "忘了除以 B 的理论误差 (B-1)/(B+1)": r((B - 1) / (B + 1), 4),
        "和 torch 自动求导对照": {"损失的差": sci(abs(float(t_loss()) - loss_grad(p, x, y)[0])), "梯度的最大差": sci(max(np.abs(a - b).max() for a, b in zip(g_auto, g))),
                           "不清梯度连续 backward 两次，.grad 是一次的多少倍": r(float(np.nanmedian(tp[2].grad.numpy() / g_auto[2])), 3)},
    }

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained")
    style = [(("float64", "relu"), "前向差分", S.MUTED, "s--", "float64，前向差分"), (("float64", "relu"), "中心差分", S.BLUE, "o-", "float64，中心差分"),
             (("float64", "tanh"), "中心差分", S.GREEN, "^-", "float64，中心差分，激活换成 tanh"), (("float32", "relu"), "中心差分", S.ORANGE, "D-", "float32，中心差分")]
    for key, m, color, fmt, label in style:
        ax[0].plot(hs, res[key][m], fmt, color=color, lw=1.6, ms=4.5, label=label)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xticks([1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13], ["1e-1", "1e-3", "1e-5", "1e-7", "1e-9", "1e-11", "1e-13"])
    ax[0].set_yticks([1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10], ["1", "1e-2", "1e-4", "1e-6", "1e-8", "1e-10"])
    ax[0].minorticks_off()
    ax[0].invert_xaxis()
    ax[0].set_xlabel("差分步长 h（越往右越小）")
    ax[0].set_ylabel("数值梯度和解析梯度的相对误差")
    b64 = best(res[("float64", "relu")]["中心差分"])
    b32 = best(res[("float32", "relu")]["中心差分"])
    ax[0].set_title(f"float64 最准到 {b64['最小相对误差']:.0e}（h={b64['对应的步长']:.0e}），float32 只到 {b32['最小相对误差']:.0e}")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    S.clean(ax[0])
    labels = ["正确的实现", "忘了除以\nbatch 大小", "掩码写成\nh >= 0", "偏置梯度\n多取一次平均", "忘了减\none-hot"]
    vals = [bug_tab[k]["全部参数"] for k in variants]
    ax[1].bar(range(5), vals, 0.6, color=[S.GREEN] + [S.RED] * 4, alpha=0.85)
    for i, v in enumerate(vals):
        ax[1].text(i, v * 1.6, f"{v:.0e}" if v < 1e-3 else f"{v:.2f}", ha="center", fontsize=9.5)
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-11, 30)
    ax[1].set_yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1], ["1e-10", "1e-8", "1e-6", "1e-4", "1e-2", "1"])
    ax[1].minorticks_off()
    ax[1].set_xticks(range(5), labels, fontsize=9)
    ax[1].set_ylabel("相对误差（h = 1e-5，float64）")
    ax[1].set_title("写对了是 1e-10 量级，写错了都在 0.1 以上")
    S.clean(ax[1])
    ax[1].grid(axis="x", visible=False)
    S.save(fig, CH, "fig03_梯度检查的步长")


# ---------------------------------------------------------------- 图 4：初始化尺度与各层信号
def deep_stats(act, std_of, depth=10, width=128, batch=256, seed=0):
    """depth 层全连接（不带偏置）加一个 10 类输出层，输入是标准正态；返回每层输出的均方根和每层权重梯度的均方根。"""
    rng = np.random.default_rng(seed)
    x, y = rng.normal(size=(batch, width)), rng.integers(0, 10, batch)
    ws, hs = [], [x]
    for _ in range(depth):
        w = rng.normal(0, std_of(width), (width, width))
        z = hs[-1] @ w
        hs.append(np.maximum(z, 0) if act == "relu" else (np.tanh(z) if act == "tanh" else 1 / (1 + np.exp(-z))))
        ws.append(w)
    wo = rng.normal(0, np.sqrt(1 / width), (width, 10))
    d = softmax(hs[-1] @ wo)
    d[np.arange(batch), y] -= 1
    g = (d / batch) @ wo.T
    grads = []
    for l in reversed(range(depth)):
        h = hs[l + 1]
        dz = g * ((h > 0) if act == "relu" else ((1 - h * h) if act == "tanh" else h * (1 - h)))
        grads.append(float(np.sqrt(((hs[l].T @ dz) ** 2).mean())))
        g = dz @ ws[l].T
    return [float(np.sqrt((h ** 2).mean())) for h in hs[1:]], grads[::-1]


def fig_init():
    depth, width = 10, 128
    cases = {
        "ReLU，He x 0.5": ("relu", lambda n: 0.5 * np.sqrt(2 / n)), "ReLU，He": ("relu", lambda n: np.sqrt(2 / n)), "ReLU，He x 2": ("relu", lambda n: 2 * np.sqrt(2 / n)),
        "tanh，Xavier": ("tanh", lambda n: np.sqrt(1 / n)), "sigmoid，Xavier": ("sigmoid", lambda n: np.sqrt(1 / n)),
    }
    stats = {k: deep_stats(a, f, depth, width) for k, (a, f) in cases.items()}
    torch.manual_seed(0)
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(width, width), nn.ReLU()]
    net = nn.Sequential(*layers, nn.Linear(width, 10))
    h = torch.randn(256, width)
    acts = []
    for m in net:
        h = m(h)
        if isinstance(m, nn.ReLU):
            acts.append(h.pow(2).mean().sqrt().item())
    nn.functional.cross_entropy(h, torch.randint(0, 10, (256,))).backward()
    gr = [m.weight.grad.pow(2).mean().sqrt().item() for m in net if isinstance(m, nn.Linear)][:depth]
    stats["ReLU，nn.Linear 默认初始化"] = (acts, gr)
    w_std = net[0].weight.std().item()

    def row(a, g):
        return {"各层输出的均方根": [sci(v, 3) for v in a], "各层权重梯度的均方根": [sci(v, 3) for v in g],
                "第 10 层输出 / 第 1 层输出": sci(a[-1] / a[0]), "第 1 层梯度 / 第 10 层梯度": sci(g[0] / g[-1])}
    # 10 层网络的训练效果
    def deep_fit(kind, opt, lr, steps=500, w=64):
        fin, va = [], []
        for seed in range(3):
            torch.manual_seed(seed)
            ls = [nn.Linear(D, w), nn.ReLU()]
            for _ in range(depth - 1):
                ls += [nn.Linear(w, w), nn.ReLU()]
            m = nn.Sequential(*ls, nn.Linear(w, C))
            if kind == "He":
                for mod_ in m:
                    if isinstance(mod_, nn.Linear):
                        nn.init.kaiming_normal_(mod_.weight, nonlinearity="relu")
                        nn.init.zeros_(mod_.bias)
            loss = t_fit(m, steps=steps, lr=lr, opt=opt, seed=seed)
            fin.append(np.mean(loss[-50:]))
            va.append(t_acc(m))
        return {"最后 50 步的平均损失": r(np.mean(fin), 3), "验证准确率（%）": r(100 * np.mean(va), 1), "三个种子各自的验证准确率（%）": [r(100 * v, 1) for v in va]}
    deep_train = {f"{k}，{'SGD 学习率 0.1' if o == 'sgd' else 'Adam 学习率 0.001'}": deep_fit(k, o, lr) for k in ("nn.Linear 默认初始化", "He") for o, lr in (("sgd", 0.1), ("adam", 1e-3))}
    NUM["深层网络的初始化"] = {
        "说明": "10 层全连接，每层 128 个单元，再接 10 类输出层；输入是 256 个标准正态样本，交叉熵损失，只做一次前向和反向，不训练；"
              "He 指权重标准差 sqrt(2/fan_in)，Xavier 指 sqrt(1/fan_in)，前五行不带偏置；最后一行直接用 torch 的 nn.Linear（带偏置）",
        "结果": {k: row(a, g) for k, (a, g) in stats.items()},
        "理论值": {"0.5^10": sci(0.5 ** 10), "2^10": 2 ** 10, "sigmoid 导数的最大值": 0.25},
        "nn.Linear 默认初始化的权重标准差 / He 的标准差": r(w_std / np.sqrt(2 / width), 3), "1/sqrt(6)": r(1 / math.sqrt(6), 3),
        "10 层 ReLU 网络（每层 64 个单元）在合成数字上训练 500 步": {"说明": "batch 64，3 个种子取平均，输入是 144 维的合成数字", "结果": deep_train},
    }

    scales = [0, 1e-4, 1e-2, 0.1, 0.3, 1, 3, 10, 30, 100]
    two = {}
    for sc in scales:
        rows = []
        for seed in range(3):
            rng = np.random.default_rng(seed)
            p = init(D, 64, C, rng, scale=sc)
            l0 = full_loss(p, XTR, YTR)
            train(p, XTR, YTR, Opt("sgd", 0.1), 500, batch=64, rng=rng)
            rows.append((l0, full_loss(p, XTR, YTR), accuracy(p, XVA, YVA)))
        o = np.array(rows).mean(axis=0)
        two[f"{sc:g}"] = {"初始损失": r(o[0], 3), "训练 500 步后的损失": r(o[1], 3), "验证准确率（%）": r(100 * o[2], 1)}
    rng = np.random.default_rng(0)
    pc = [np.full((D, 64), 0.01), np.zeros(64), np.full((64, C), 0.01), np.zeros(C)]
    train(pc, XTR, YTR, Opt("sgd", 0.1), 500, batch=64, rng=rng)
    NUM["两层网络的初始化尺度"] = {
        "说明": "合成数字，144 -> 64 -> 10，权重标准差取 He 初始化的若干倍（0 表示全零），SGD 学习率 0.1，batch 64，训练 500 步，3 个种子取平均",
        "按倍数": two, "ln10": r(math.log(10), 3),
        "所有权重都设成同一个常数 0.01": {"验证准确率（%）": r(100 * accuracy(pc, XVA, YVA), 1), "训练后互不相同的隐藏单元个数": int(len(np.unique(np.round(pc[0], 12), axis=1).T))},
    }

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0), layout="constrained")
    look = {"ReLU，He x 0.5": (S.BLUE, "v-"), "ReLU，He": (S.GREEN, "o-"), "ReLU，He x 2": (S.RED, "^-"), "ReLU，nn.Linear 默认初始化": (S.ORANGE, "s--"), "sigmoid，Xavier": (S.PURPLE, "D:")}
    xs = np.arange(1, depth + 1)
    for k, (color, fmt) in look.items():
        ax[0].plot(xs, stats[k][0], fmt, color=color, lw=1.6, ms=4.5, label=k)
        ax[1].plot(xs, stats[k][1], fmt, color=color, lw=1.6, ms=4.5, label=k)
    for a_, yl in ((ax[0], "这一层输出的均方根"), (ax[1], "这一层权重梯度的均方根")):
        a_.set_yscale("log")
        a_.set_xticks(xs)
        a_.set_xlabel("第几层（1 最靠近输入）")
        a_.set_ylabel(yl)
        a_.minorticks_off()
        S.clean(a_)
    ax[0].set_yticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], ["0.001", "0.01", "0.1", "1", "10", "100", "1000"])
    ax[1].set_yticks([1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 10], ["1e-11", "1e-9", "1e-7", "1e-5", "1e-3", "0.1", "10"])
    ax[0].set_title("前向：标准差差 2 倍，10 层后差 1000 倍")
    ax[1].set_title("反向：sigmoid 的梯度越靠前越小")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    xv = [s for s in scales if s > 0]
    ax[2].plot(xv, [two[f"{s:g}"]["验证准确率（%）"] for s in xv], "o-", color=S.BLUE, lw=1.8, ms=5)
    ax[2].axhline(two["0"]["验证准确率（%）"], color=S.MUTED, lw=1, ls=":")
    ax[2].text(3e-4, two["0"]["验证准确率（%）"] + 1.5, "全零初始化", color=S.MUTED, fontsize=9)
    ax[2].set_xscale("log")
    ax[2].set_xticks([1e-4, 1e-2, 1, 100], ["0.0001", "0.01", "1", "100"])
    ax[2].minorticks_off()
    ax[2].set_ylim(0, 70)
    ax[2].set_xlabel("权重标准差是 He 初始化的多少倍")
    ax[2].set_ylabel("训练 500 步后的验证准确率（%）")
    ax[2].set_title("两层网络：太小学不动，太大学不好")
    S.clean(ax[2])
    S.save(fig, CH, "fig04_初始化尺度与各层信号")


# ---------------------------------------------------------------- 图 5：学习率的边界
def hand_example(lr, steps=30):
    """讲义 8.4 的手算例子：x=2, y=3, w=0.5, b=0, L=(wx+b-y)^2。返回每一步更新前的损失。"""
    w, b, out = 0.5, 0.0, []
    for _ in range(steps):
        z = w * 2 + b
        out.append((z - 3) ** 2)
        w, b = w - lr * 2 * (z - 3) * 2, b - lr * 2 * (z - 3)
    return out


def top_eig(p, x, y, iters=60, eps=1e-4):
    """幂迭代求损失对全部参数的 Hessian 的最大特征值，Hessian 乘向量用梯度的中心差分。"""
    rng = np.random.default_rng(0)
    v = rng.normal(size=sum(a.size for a in p))
    v /= np.linalg.norm(v)
    lam = 0.0

    def shifted(sign):
        out, i = [], 0
        for a in p:
            out.append(a + sign * eps * v[i:i + a.size].reshape(a.shape))
            i += a.size
        return out
    for _ in range(iters):
        hv = (flat(loss_grad(shifted(1), x, y)[1]) - flat(loss_grad(shifted(-1), x, y)[1])) / (2 * eps)
        lam = float(v @ hv)
        v = hv / np.linalg.norm(hv)
    return lam


def fig_lr():
    z0 = 0.5 * 2 + 0
    g_w, g_b = 2 * (z0 - 3) * 2, 2 * (z0 - 3)
    hand_lrs = [0.01, 0.05, 0.1, 0.15, 0.19, 0.2, 0.21, 0.25]
    hand = {lr: hand_example(lr) for lr in hand_lrs}
    w, b, n_inf = np.float64(0.5), np.float64(0.0), None
    with np.errstate(all="ignore"):
        for step in range(1, 5000):
            z = w * 2 + b
            if not np.isfinite((z - 3) * (z - 3)):
                n_inf = step - 1
                break
            w, b = w - 0.25 * 2 * (z - 3) * 2, b - 0.25 * 2 * (z - 3)
    NUM["手算例子的学习率"] = {
        "说明": "讲义 8.4 的例子：x=2，y=3，w=0.5，b=0，损失 (wx+b-y)^2；损失对 (w, b) 的 Hessian 只有一个非零特征值 2(x^2+1)=10，稳定的条件是学习率小于 2/10",
        "前向": {"z": z0, "损失": (z0 - 3) ** 2}, "梯度": {"dL/dw": g_w, "dL/db": g_b},
        "学习率 0.1 更新一步": {"w": r(0.5 - 0.1 * g_w, 3), "b": r(0 - 0.1 * g_b, 3), "更新后的损失": r(hand[0.1][1], 6)},
        "Hessian 的非零特征值": 2 * (2 * 2 + 1), "稳定边界 2/特征值": 2 / (2 * (2 * 2 + 1)),
        "每一步损失乘的倍数 (1-10*lr)^2": {str(lr): r((1 - 10 * lr) ** 2, 4) for lr in hand_lrs},
        "第 1、2、5、10 步之后的损失": {str(lr): [sci(hand[lr][k], 3) for k in (1, 2, 5, 10)] for lr in hand_lrs},
        "学习率 0.25 时损失变成 inf 之前走了多少步": n_inf,
    }

    xs, ys = XTR[:1000], YTR[:1000]
    p0 = init(D, 64, C, np.random.default_rng(0))
    lam0 = top_eig(p0, xs, ys)
    edge = 2 / lam0
    lrs = [0.05, 0.1, 0.2, 0.3, round(edge, 2), 0.7, 1.0, 1.4, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 100.0, 1e4, 1e6]
    gd, curves = {}, {}
    for lr in lrs:
        p = [a.copy() for a in p0]
        ls = np.array(train(p, xs, ys, Opt("sgd", lr), 300))
        fin = full_loss(p, xs, ys)
        curves[lr] = ls
        gd[f"{lr:g}"] = {"第 300 步后的损失": r(fin, 4) if np.isfinite(fin) else None, "过程中出现过 nan 或 inf": bool(not np.all(np.isfinite(ls)) or not np.isfinite(fin)),
                         "过程中的最大损失": sci(np.nanmax(np.where(np.isfinite(ls), ls, np.nan)), 3), "损失比上一步变大的步数占比（%）": r(100 * (np.diff(ls) > 0).mean(), 1),
                         "结束时对全部样本都输出 0 的隐藏单元占比（%）": r(100 * dead_frac(p, xs), 1) if np.isfinite(fin) else None}
    cls_prior = np.bincount(ys, minlength=C) / len(ys)
    NUM["两层网络的学习率边界"] = {
        "说明": "合成数字的前 1000 张训练图，144 -> 64 -> 10，He 初始化（种子 0），全批量梯度下降 300 步；Hessian 的最大特征值用幂迭代估计",
        "初始化处 Hessian 的最大特征值": r(lam0, 2), "按二次函数的规则算出的边界 2/特征值": r(edge, 2), "按学习率": gd,
        "只按类别比例猜的损失（类别先验的熵）": r(float(-(cls_prior * np.log(cls_prior)).sum()), 3),
        "学习率 2 是二次规则边界的多少倍": r(2.0 / edge, 1),
    }

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0), layout="constrained")
    for lr, color, fmt in ((0.05, S.BLUE, "o-"), (0.1, S.GREEN, "s-"), (0.19, S.PURPLE, "^-"), (0.2, S.ORANGE, "D-"), (0.21, S.RED, "v-")):
        v = np.array(hand[lr][:16])
        ax[0].plot(np.arange(16), np.maximum(v, 1e-12), fmt, color=color, lw=1.5, ms=4, label=f"学习率 {lr:g}")
    ax[0].set_yscale("log")
    ax[0].set_ylim(5e-13, 1e3)
    ax[0].set_yticks([1e-12, 1e-9, 1e-6, 1e-3, 1, 1e3], ["0", "1e-9", "1e-6", "0.001", "1", "1000"])
    ax[0].minorticks_off()
    ax[0].set_xlabel("更新步数")
    ax[0].set_ylabel("损失")
    ax[0].set_title("手算例子：0.2 原地来回，0.21 发散")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    S.clean(ax[0])
    for lr, color in ((0.1, S.BLUE), (round(edge, 2), S.GREEN), (2.0, S.ORANGE), (3.0, S.PURPLE), (5.0, S.RED)):
        ax[1].plot(np.arange(len(curves[lr])), curves[lr], "-", color=color, lw=1.4, label=f"学习率 {lr:g}")
    ax[1].set_yscale("log")
    ax[1].set_yticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
    ax[1].minorticks_off()
    ax[1].set_xlabel("更新步数（全批量）")
    ax[1].set_ylabel("训练损失")
    ax[1].set_title("两层网络：学习率 2 抖着降，5 掉进平台")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    S.clean(ax[1])
    show = [lr for lr in lrs if lr <= 10]
    ax[2].plot(show, [gd[f"{lr:g}"]["第 300 步后的损失"] for lr in show], "o-", color=S.BLUE, lw=1.8, ms=5)
    ax[2].axvline(edge, color=S.MUTED, lw=1.2, ls="--")
    ax[2].text(edge * 1.12, 6.5, f"2 / 最大特征值\n= {edge:.2f}", color=S.MUTED, fontsize=9, va="top", ha="left")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax[2].set_yticks([0.01, 0.1, 1], ["0.01", "0.1", "1"])
    ax[2].set_ylim(0.008, 8)
    ax[2].minorticks_off()
    ax[2].set_xlabel("学习率")
    ax[2].set_ylabel("300 步后的训练损失")
    ax[2].set_title("超过虚线几倍仍在下降，再大就卡在 2.3")
    S.clean(ax[2])
    S.save(fig, CH, "fig05_学习率的边界")


# ---------------------------------------------------------------- 图 6：三种优化器与批大小
OPT_NAMES = {"sgd": "SGD", "mom": "SGD + 动量 0.9", "adam": "Adam"}


def fig_optim():
    grid = [float(f"{10 ** e:.3g}") for e in np.arange(-6, 1.01, 0.25)]
    sweeps = {}
    for tag, sc in (("输入在 0 到 1", 1.0), ("输入乘 255（不归一化）", 255.0)):
        xt, xv = XTR * sc, XVA * sc
        sweeps[tag] = {}
        for kind in OPT_NAMES:
            loss_row, acc_row = [], []
            for lr in grid:
                ls, ac = [], []
                for seed in range(3):
                    rng = np.random.default_rng(seed)
                    p = init(D, 64, C, rng)
                    train(p, xt, YTR, Opt(kind, lr), 400, batch=64, rng=rng)
                    fin = full_loss(p, xt, YTR)
                    ls.append(fin)
                    ac.append(accuracy(p, xv, YVA) if np.isfinite(fin) else 0.1)
                loss_row.append(float(np.median(ls)))
                acc_row.append(100 * float(np.mean(ac)))
            best = int(np.argmax(acc_row))
            good = [lr for lr, a in zip(grid, acc_row) if a >= acc_row[best] - 5]
            sweeps[tag][kind] = {"训练损失（3 个种子的中位数）": [None if not np.isfinite(v) else r(v, 3) for v in loss_row], "验证准确率（%，3 个种子平均）": [r(a, 1) for a in acc_row],
                                 "最好的学习率": grid[best], "最好的验证准确率（%）": r(acc_row[best], 1),
                                 "离最好不超过 5 个百分点的学习率范围": [min(good), max(good)], "这个范围的宽度（最大 / 最小）": r(max(good) / min(good), 1)}
    # 和 torch.optim 对一遍
    xb, yb = XTR[:64], YTR[:64]
    worst = {}
    for kind, make in (("sgd", lambda ps: torch.optim.SGD(ps, lr=0.05)), ("mom", lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9)), ("adam", lambda ps: torch.optim.Adam(ps, lr=0.01))):
        p = init(D, 16, C, np.random.default_rng(2))
        tp = [torch.tensor(a.copy(), requires_grad=True) for a in p]
        o, mine = make(tp), Opt(kind, 0.05 if kind != "adam" else 0.01)
        tx, ty = torch.tensor(xb), torch.tensor(yb)
        for _ in range(20):
            mine.step(p, loss_grad(p, xb, yb)[1])
            o.zero_grad()
            nn.functional.cross_entropy(torch.relu(tx @ tp[0] + tp[1]) @ tp[2] + tp[3], ty).backward()
            o.step()
        worst[OPT_NAMES[kind]] = sci(max(np.abs(a - b.detach().numpy()).max() for a, b in zip(p, tp)))
    p = init(D, 64, C, np.random.default_rng(0))
    before = [a.copy() for a in p]
    g = loss_grad(p, XTR[:64], YTR[:64])[1]
    Opt("adam", 1e-3).step(p, g)
    delta, gabs = np.abs(flat(p) - flat(before)), np.abs(flat(g))
    NUM["三种优化器"] = {
        "说明": "合成数字，144 -> 64 -> 10，He 初始化，batch 64，训练 400 步，每个学习率 3 个种子；学习率从 0.000001 到 10，每个数量级取 4 个点；"
              "动量系数 0.9，Adam 用默认的 beta1=0.9、beta2=0.999、eps=1e-8；发散或损失为 nan 的按准确率 10% 计",
        "学习率": grid, "结果": {tag: {OPT_NAMES[k]: v for k, v in d.items()} for tag, d in sweeps.items()},
        "动量 0.9 的等效步长倍数 1/(1-0.9)": r(1 / (1 - 0.9), 1),
        "本脚本的优化器和 torch.optim 连走 20 步后参数的最大差": worst,
        "Adam 第一步（学习率 0.001）": {"梯度不为零的参数里更新量的最小值": sci(delta[gabs > 1e-6].min(), 3), "更新量的最大值": sci(delta.max(), 3),
                                "这些参数的梯度绝对值范围": [sci(gabs[gabs > 1e-6].min()), sci(gabs.max())], "梯度正好为 0 的参数占比（%）": r(100 * (gabs == 0).mean(), 1)},
    }

    noise = {}
    for stage, steps in (("刚初始化", 0), ("训练 300 步后", 300), ("训练 2000 步后", 2000)):
        rng = np.random.default_rng(0)
        p = init(D, 64, C, rng)
        if steps:
            train(p, XTR, YTR, Opt("sgd", 0.3), steps, batch=64, rng=rng)
        full = flat(loss_grad(p, XTR, YTR)[1])
        row = {}
        for bs in (1, 4, 16, 64, 256, 1024):
            rel, cos = [], []
            for _ in range(200):
                idx = rng.integers(0, N_TRAIN, bs)
                gb = flat(loss_grad(p, XTR[idx], YTR[idx])[1])
                rel.append(np.linalg.norm(gb - full) / np.linalg.norm(full))
                cos.append(gb @ full / (np.linalg.norm(gb) * np.linalg.norm(full) + 1e-300))
            rms = float(np.sqrt(np.mean(np.square(rel))))
            row[str(bs)] = {"相对误差（均方根）": r(rms, 3), "相对误差 x sqrt(B)": r(rms * math.sqrt(bs), 2), "和全量梯度夹角余弦的中位数": r(np.median(cos), 3),
                            "方向和全量梯度相反的比例（%）": r(100 * float((np.array(cos) < 0).mean()), 1)}
        noise[stage] = {"全量训练损失": r(full_loss(p, XTR, YTR), 3), "按 batch 大小": row}
    NUM["批大小与梯度噪声"] = {
        "说明": "合成数字 4000 张训练图，144 -> 64 -> 10；在同一组参数上，随机抽 batch 算梯度，和 4000 张全量梯度比，每种 batch 大小抽 200 次；"
              "相对误差 = |batch 梯度 - 全量梯度| / |全量梯度|；训练用 SGD 学习率 0.3、batch 64",
        "结果": noise}
    epochs = {}
    for bs in (8, 32, 128, 512, 2000):
        for mode in ("学习率固定 0.1", "学习率按 batch/32 缩放"):
            lr = 0.1 if mode == "学习率固定 0.1" else 0.1 * bs / 32
            steps = int(5 * N_TRAIN / bs)
            va = []
            for seed in range(3):
                rng = np.random.default_rng(seed)
                p = init(D, 64, C, rng)
                train(p, XTR, YTR, Opt("sgd", lr), steps, batch=bs, rng=rng)
                va.append(accuracy(p, XVA, YVA) if np.isfinite(full_loss(p, XTR, YTR)) else 0.1)
            epochs.setdefault(str(bs), {"更新步数": steps})[mode] = {"学习率": r(lr, 4), "验证准确率（%）": r(100 * np.mean(va), 1)}
    NUM["同样 5 个 epoch 换批大小"] = {"说明": "同上的网络，SGD，每种设置都只看 5 遍训练集（4000 张），3 个种子取平均", "按 batch 大小": epochs}

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0), layout="constrained")
    for a_, tag in zip(ax[:2], sweeps):
        for kind, color, fmt in (("sgd", S.BLUE, "o-"), ("mom", S.GREEN, "s-"), ("adam", S.ORANGE, "^-")):
            a_.plot(grid, sweeps[tag][kind]["验证准确率（%，3 个种子平均）"], fmt, color=color, lw=1.6, ms=4, label=OPT_NAMES[kind])
        a_.set_xscale("log")
        a_.set_xticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10], ["1e-6", "1e-5", "1e-4", "0.001", "0.01", "0.1", "1", "10"])
        a_.minorticks_off()
        a_.set_ylim(0, 100)
        a_.set_xlabel("学习率")
        a_.set_ylabel("训练 400 步后的验证准确率（%）")
        S.clean(a_)
    ax[0].set_title("输入在 0 到 1：三条曲线形状相近，左右各错开约 10 倍")
    ax[1].set_title("输入乘 255：三种优化器的最高点都明显变低")
    ax[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    bss = [1, 4, 16, 64, 256, 1024]
    for (stage, d), color, fmt in zip(noise.items(), (S.BLUE, S.ORANGE, S.RED), ("o-", "s-", "^-")):
        ax[2].plot(bss, [d["按 batch 大小"][str(b)]["相对误差（均方根）"] for b in bss], fmt, color=color, lw=1.6, ms=4.5, label=stage)
    ref = noise["刚初始化"]["按 batch 大小"]["1"]["相对误差（均方根）"]
    ax[2].plot(bss, [ref / math.sqrt(b) for b in bss], "-", color=S.BLUE, lw=6, alpha=0.2, label="按 1/sqrt(B) 下降的参考线", zorder=1)
    ax[2].axhline(1, color=S.INK, lw=0.6)
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xticks(bss, [str(b) for b in bss])
    ax[2].set_yticks([0.2, 0.5, 1, 2, 5, 10], ["0.2", "0.5", "1", "2", "5", "10"])
    ax[2].minorticks_off()
    ax[2].set_xlabel("batch 大小 B")
    ax[2].set_ylabel("batch 梯度的相对误差")
    ax[2].set_title("batch 变 4 倍，梯度噪声减半")
    ax[2].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    S.clean(ax[2])
    S.save(fig, CH, "fig06_优化器与批大小")


# ---------------------------------------------------------------- 图 7：小数据上的过拟合与权重衰减
def overfit_run(n, wd=0.0, seed=0, steps=3000, lr=0.2, h=128, l1=0.0, prox=False, log=50):
    x, y = digits(n, 100 + seed)
    rng = np.random.default_rng(seed)
    p = init(D, h, C, rng)
    opt = Opt("sgd", lr, wd=wd)
    curve = []
    for s in range(steps + 1):
        if s % log == 0:
            curve.append((s, full_loss(p, x, y), full_loss(p, XVA, YVA), accuracy(p, x, y), accuracy(p, XVA, YVA)))
        if s == steps:
            break
        idx = rng.integers(0, n, min(64, n))
        g = loss_grad(p, x[idx], y[idx])[1]
        if l1 and not prox:
            g = [gi + l1 * np.sign(pi) if pi.ndim == 2 else gi for gi, pi in zip(g, p)]
        opt.step(p, g)
        if l1 and prox:
            for pi in (p[0], p[2]):
                pi[:] = np.sign(pi) * np.maximum(np.abs(pi) - lr * l1, 0)
    return np.array(curve), p


def fig_overfit():
    by_n, curve300 = {}, None
    for n in (100, 300, 1000, 4000):
        c, _ = overfit_run(n)
        if n == 300:
            curve300 = c
        i = int(c[:, 2].argmin())
        by_n[str(n)] = {"最终训练损失": r(c[-1, 1], 4), "最终训练准确率（%）": r(100 * c[-1, 3], 1), "最终验证损失": r(c[-1, 2], 3), "最终验证准确率（%）": r(100 * c[-1, 4], 1),
                        "验证损失的最小值": r(c[i, 2], 3), "验证损失最小时是第几步": int(c[i, 0]), "验证损失最小时的验证准确率（%）": r(100 * c[i, 4], 1)}
    wds = [0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    by_wd, early = {}, []
    for wd in wds:
        rows = []
        for seed in range(3):
            c, p = overfit_run(300, wd=wd, seed=seed, log=100)
            rows.append((c[-1, 1], c[-1, 2], c[-1, 3], c[-1, 4], np.sqrt((p[0] ** 2).sum() + (p[2] ** 2).sum())))
            if wd == 0:
                i = int(c[:, 2].argmin())
                early.append((c[i, 0], c[i, 4], c[-1, 4], c[:, 4].max(), c[i, 2], c[-1, 2]))
        o = np.array(rows).mean(axis=0)
        by_wd[f"{wd:g}"] = {"训练损失": r(o[0], 3), "验证损失": r(o[1], 3), "训练准确率（%）": r(100 * o[2], 1), "验证准确率（%）": r(100 * o[3], 1), "权重的范数": r(o[4], 1)}
    e = np.array(early).mean(axis=0)
    sparse = {}
    for name, kw in (("不加正则", {}), ("L2，weight_decay=0.01", dict(wd=1e-2)), ("L1，系数 0.001，直接加到梯度上", dict(l1=1e-3)), ("L1，系数 0.001，梯度步之后做软阈值", dict(l1=1e-3, prox=True))):
        c, p = overfit_run(300, log=3000, **kw)
        w = np.concatenate([p[0].ravel(), p[2].ravel()])
        sparse[name] = {"正好等于 0 的权重占比（%）": r(100 * np.mean(w == 0), 1), "绝对值小于 0.001 的权重占比（%）": r(100 * np.mean(np.abs(w) < 1e-3), 1), "验证准确率（%）": r(100 * c[-1, 4], 1)}
    NUM["小数据上的过拟合"] = {
        "说明": "合成数字，144 -> 128 -> 10，He 初始化，SGD 学习率 0.2，batch 64，训练 3000 步；训练集另外生成，验证集是同一份 2000 张；"
              "weight_decay 按 torch.optim.SGD 的做法把 wd*w 加到权重矩阵的梯度上，相当于损失里加 (wd/2)*sum(w^2)",
        "按训练集大小（不加正则，种子 0）": by_n,
        "300 张训练图，按 weight_decay（3 个种子平均）": by_wd,
        "300 张训练图不加正则，按验证损失早停（3 个种子平均）": {"验证损失最小时是第几步": r(e[0], 0), "这时的验证损失": r(e[4], 3), "这时的验证准确率（%）": r(100 * e[1], 1),
                                                "训练到 3000 步的验证损失": r(e[5], 3), "训练到 3000 步的验证准确率（%）": r(100 * e[2], 1), "全程最高的验证准确率（%）": r(100 * e[3], 1)},
        "L1 与 L2 的稀疏性（300 张训练图，种子 0）": sparse,
    }

    drop = {}
    for pd in (0.2, 0.5):
        m = t_mlp(0, p_drop=pd)
        t_fit(m, steps=1500)
        a_eval = t_acc(m)
        torch.manual_seed(1)
        a_tr = [t_acc(m, train_mode=True) for _ in range(30)]
        drop[f"p={pd}"] = {"eval 模式的验证准确率（%）": r(100 * a_eval, 1), "忘了切 eval，30 次评估的平均（%）": r(100 * np.mean(a_tr), 1), "30 次评估的标准差（百分点）": r(100 * np.std(a_tr), 2),
                           "30 次里的最小值（%）": r(100 * min(a_tr), 1), "30 次里的最大值（%）": r(100 * max(a_tr), 1), "平均比 eval 模式低多少（百分点）": r(100 * (a_eval - np.mean(a_tr)), 1)}
    NUM["Dropout 忘了切 eval"] = {"说明": "合成数字，nn.Linear(144,128)-ReLU-Dropout(p)-nn.Linear(128,10)，SGD 学习率 0.1，batch 64，训练 1500 步；同一个模型在 2000 张验证图上评估", "结果": drop}
    x3, y3 = digits(300, 100)
    x3, y3 = torch.tensor(x3, dtype=torch.float32), torch.tensor(y3)
    adamw = {}
    for opt in ("adam", "adamw"):
        for wd in (0.0, 0.01, 0.1, 1.0):
            m = t_mlp(0)
            t_fit(m, steps=3000, opt=opt, lr=1e-3, wd=wd, xs=x3, ys=y3)
            with torch.no_grad():
                wn = float(sum((q ** 2).sum() for q in m.parameters() if q.ndim == 2).sqrt())
                tr = (m.eval()(x3).argmax(1) == y3).float().mean().item()
            adamw.setdefault("Adam（weight_decay 当 L2 加进梯度）" if opt == "adam" else "AdamW（解耦的权重衰减）", {})[f"weight_decay={wd:g}"] = {
                "权重的范数": r(wn, 1), "训练准确率（%）": r(100 * tr, 1), "验证准确率（%）": r(100 * t_acc(m), 1)}
    NUM["Adam 加 L2 与 AdamW"] = {"说明": "300 张训练图，nn.Linear(144,128)-ReLU-nn.Linear(128,10)，学习率 0.001，batch 64，训练 3000 步，种子 0", "结果": adamw}

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0), layout="constrained")
    s = curve300[:, 0]
    ax[0].plot(s, curve300[:, 1], "-", color=S.BLUE, lw=1.8, label="训练损失")
    ax[0].plot(s, curve300[:, 2], "-", color=S.RED, lw=1.8, label="验证损失")
    i = int(curve300[:, 2].argmin())
    ax[0].axvline(s[i], color=S.MUTED, lw=1, ls=":")
    ax[0].text(s[i] + 60, curve300[:, 2].max() * 1.02, f"验证损失最小：第 {int(s[i])} 步", color=S.MUTED, fontsize=9, va="top")
    ax[0].set_xlabel("训练步数")
    ax[0].set_ylabel("损失")
    ax[0].set_ylim(bottom=0)
    ax[0].set_title("300 张训练图：验证损失先降后升")
    ax[0].legend(fontsize=8.5, loc="center right")
    S.clean(ax[0])
    ax[1].plot(s, 100 * curve300[:, 3], "-", color=S.BLUE, lw=1.8, label="训练准确率")
    ax[1].plot(s, 100 * curve300[:, 4], "-", color=S.RED, lw=1.8, label="验证准确率")
    ax[1].axvline(s[i], color=S.MUTED, lw=1, ls=":")
    ax[1].set_xlabel("训练步数")
    ax[1].set_ylabel("准确率（%）")
    ax[1].set_ylim(0, 105)
    ax[1].set_title("同一次训练：验证准确率没有跟着掉")
    ax[1].legend(fontsize=8.5, loc="lower right")
    S.clean(ax[1])
    xi = np.arange(len(wds))
    ax[2].plot(xi, [by_wd[f"{w:g}"]["训练准确率（%）"] for w in wds], "o-", color=S.BLUE, lw=1.8, ms=5, label="训练准确率")
    ax[2].plot(xi, [by_wd[f"{w:g}"]["验证准确率（%）"] for w in wds], "s-", color=S.RED, lw=1.8, ms=5, label="验证准确率")
    ax[2].set_xticks(xi, [f"{w:g}" for w in wds])
    ax[2].set_ylim(0, 105)
    ax[2].set_xlabel("weight_decay")
    ax[2].set_ylabel("训练 3000 步后的准确率（%）")
    ax[2].set_title("权重衰减：0.01 附近略有帮助，0.1 连训练集也学不会")
    ax[2].legend(fontsize=8.5, loc="lower left")
    S.clean(ax[2])
    S.save(fig, CH, "fig07_过拟合与权重衰减")


# ---------------------------------------------------------------- 表：常见坑的实际表现
def table_pitfalls():
    m = t_mlp(0)
    with torch.no_grad():
        init_loss = nn.functional.cross_entropy(m.eval()(TX), TY).item()
    bugs = {}
    for bug in (None, "忘了 zero_grad", "忘了 step", "先 softmax 再交叉熵", "标签和图片错位"):
        for opt, lr in (("sgd", 0.1), ("adam", 1e-3)):
            m = t_mlp(0)
            ls = t_fit(m, opt=opt, lr=lr, bug=bug)
            bugs.setdefault("正确写法" if bug is None else bug, {})["SGD，学习率 0.1" if opt == "sgd" else "Adam，学习率 0.001"] = {
                "最后 50 步的平均损失": r(np.mean(ls[-50:]), 3), "验证准确率（%）": r(100 * t_acc(m), 1), "出现过 nan": bool(np.isnan(ls).any())}
    m = t_mlp(0)
    xs, ys = TX[:32], TY[:32]
    o = torch.optim.Adam(m.parameters(), lr=1e-3)
    trace = []
    for _ in range(500):
        o.zero_grad()
        out = m(xs)
        loss = nn.functional.cross_entropy(out, ys)
        loss.backward()
        o.step()
        trace.append((loss.item(), (out.argmax(1) == ys).float().mean().item()))
    first = next(i for i, v in enumerate(trace) if v[1] == 1.0)
    raw = {}
    for opt, lr in (("sgd", 0.1), ("adam", 1e-3)):
        m = t_mlp(0)
        ls = t_fit(m, opt=opt, lr=lr, xs=TX * 255.0)
        raw["SGD，学习率 0.1" if opt == "sgd" else "Adam，学习率 0.001"] = {"第一步的损失": r(ls[0], 1), "最后 50 步的平均损失": r(np.mean(ls[-50:]), 3),
                                                                 "验证准确率（%）": r(100 * t_acc(m, TXV * 255.0), 1), "出现过 nan": bool(np.isnan(ls).any())}
    big = {}
    for lr in (10.0, 1e3, 1e6):
        m = t_mlp(0)
        ls = t_fit(m, lr=lr, steps=200)
        big[f"{lr:g}"] = {"出现过 nan": bool(np.isnan(ls).any()), "第 200 步的损失": None if not np.isfinite(ls[-1]) else r(ls[-1], 3)}
    NUM["常见坑的实际表现"] = {
        "说明": "合成数字，nn.Linear(144,128)-ReLU-nn.Linear(128,10)，nn.Linear 默认初始化，float32，batch 64，训练 600 步，种子 0；每次只引入一个错误",
        "默认初始化的初始损失": r(init_loss, 3), "ln10": r(math.log(10), 3), "按错误": bugs,
        "过拟合 32 张图（Adam，学习率 0.001，全批量）": {"训练准确率第一次到 100% 是第几步": first, "这时的损失": r(trace[first][0], 3), "第 500 步的损失": r(trace[-1][0], 4)},
        "输入乘 255 不归一化": raw,
        "SGD 把学习率调到很大（训练 200 步）": big,
    }

    case, lab2 = np.array([[1000.0, 1001.0], [-1000.0, -999.0]]), np.array([1, 0])

    def describe(fn):
        with np.errstate(all="ignore"):
            try:
                v = float(fn())
            except Exception as exc:  # noqa: BLE001
                return f"报错：{type(exc).__name__}"
        return "nan" if np.isnan(v) else ("inf" if np.isinf(v) else r(v, 4))

    def lse(z, m):
        zz = z - m
        return zz - np.log(np.exp(zz).sum(axis=1, keepdims=True))
    right = lse(case, case.max(axis=1, keepdims=True))
    NUM["作业里常见的写法错误"] = {
        "说明": "交叉熵的几种错误写法，在验收脚本的用例 logits=[[1000,1001],[-1000,-999]]、labels=[1,0] 上各算出什么；正确答案是 0.8133",
        "正确写法（参考实现）": describe(lambda: ce(case, lab2)),
        "直接 exp 再归一化": describe(lambda: naive_ce(case, lab2)),
        "减的是整个 batch 的最大值，不是每行的最大值": describe(lambda: -lse(case, case.max())[np.arange(2), lab2].mean()),
        "取每行最大值时漏了 keepdims": describe(lambda: -lse(case, case.max(axis=1))[np.arange(2), lab2].mean()),
        "对 batch 求和而不是求平均": describe(lambda: -right[np.arange(2), lab2].sum()),
        "取正确类别时写成 log_probs[:, labels]": describe(lambda: -right[:, lab2].mean()),
        "把 logits 和 labels 都换成 float32 再用参考实现": describe(lambda: ce(case.astype(np.float32), lab2)),
    }

    lf = nn.CrossEntropyLoss()
    logits = torch.randn(4, 3, generator=torch.Generator().manual_seed(0))
    lab = torch.tensor([0, 2, 1, 1])
    kinds = {"(B,) 的 long": lab, "(B,) 的 float": lab.float(), "(B,) 的 int32": lab.int(), "(B,1) 的 long": lab[:, None],
             "(B,C) 的 one-hot，float": nn.functional.one_hot(lab, 3).float(), "(B,C) 的 one-hot，long": nn.functional.one_hot(lab, 3)}
    accept = {}
    for name, t in kinds.items():
        try:
            accept[name] = {"结果": "接受", "损失": r(lf(logits, t).item(), 4)}
        except Exception as exc:  # noqa: BLE001
            accept[name] = {"结果": "报错", "报错信息": str(exc).splitlines()[0][:90]}
    xb = torch.tensor([-800.0, -200.0, -100.0, -50.0])
    one = torch.ones_like(xb)
    NUM["PyTorch 接口的实际行为"] = {
        "PyTorch 版本": torch.__version__, "torch.cuda.is_available()": bool(torch.cuda.is_available()),
        "CrossEntropyLoss 接受什么样的标签（logits 是 (4,3) 的 float32）": accept,
        "二分类，标签为 1，logit 取 -800、-200、-100、-50（float32）": {
            "-log(sigmoid(x)) 直接算": [None if not math.isfinite(v) else r(v, 1) for v in (-torch.log(torch.sigmoid(xb))).tolist()],
            "直接算得到 inf 的个数": int(torch.isinf(torch.log(torch.sigmoid(xb))).sum()),
            "BCEWithLogitsLoss": [r(v, 1) for v in nn.functional.binary_cross_entropy_with_logits(xb, one, reduction="none").tolist()],
            "BCELoss(sigmoid(x))": [r(v, 1) for v in nn.functional.binary_cross_entropy(torch.sigmoid(xb), one, reduction="none").tolist()]},
    }


def main() -> None:
    S.setup()
    for fn in (fig_xor, table_baseline, fig_ce, fig_gradcheck, fig_init, fig_lr, fig_optim, fig_overfit, table_pitfalls):
        fn()
        print("完成", fn.__name__, flush=True)
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False))


if __name__ == "__main__":
    main()
