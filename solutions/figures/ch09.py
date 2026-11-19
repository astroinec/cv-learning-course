"""第 9 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch09.py

输出：<主分支>/09_CNN核心/figures/*.svg 和 实测数字.json。
输出尺寸调用本章作业的参考实现 conv_output_size，残差网络用参考实现 BasicResidualBlock。
数据全部由代码生成（本机没有 CIFAR-10），网络很小，全部在 CPU 上跑，固定随机种子。
只有“第一层卷积核”那一组读 torchvision 的 ResNet-18 ImageNet 权重（题 DL-3 用的就是它）。
耗时会随机器负载波动，默认沿用 实测数字.json 里上次的值，设置 CH09_RETIME=1 才重测。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。
"""

from __future__ import annotations

import itertools
import json
import math
import os
import re
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
warnings.filterwarnings("ignore", category=UserWarning)

CH = "09_CNN核心"
HOMEWORK = "作业_卷积shape与残差块.py"
mod = S.load_solution(CH, HOMEWORK)
REFERENCE_SOURCE = (S.SOLUTIONS / CH / HOMEWORK).read_text(encoding="utf-8")
conv_out = mod.conv_output_size
ResBlock = mod.BasicResidualBlock
NUM: dict = {}
CLASS_NAMES = ["横线", "竖线", "方块", "斜线"]


def r(x, k=3):
    return round(float(x), k)


def acceptance(source: str) -> dict:
    """把一份作业实现写进临时目录，用 COURSE_IMPL_DIR 指过去运行本章的验收脚本，返回通过几项、哪几项没通过。"""
    script = S.MAIN / CH / "tests" / "check_progress.py"
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / CH
        target.mkdir()
        (target / HOMEWORK).write_text(source, encoding="utf-8")
        env = dict(os.environ, COURSE_IMPL_DIR=tmp, PYTHONDONTWRITEBYTECODE="1")
        res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, encoding="utf-8", env=env, timeout=600)
    done, total = re.findall(r"通过 (\d+)/(\d+)", res.stdout)[-1]
    failed = [line[len("[FAIL] "):].split(" — ")[0] for line in res.stdout.splitlines() if line.startswith("[FAIL] ")]
    out = {"通过项数": int(done), "总项数": int(total), "没通过的项数": int(total) - int(done), "没通过的检查": failed}
    grid = re.search(r"(\d+) 种组合里有 (\d+) 种不一致", res.stdout)
    if grid:
        out["和 PyTorch 逐个比的组合数"] = int(grid.group(1))
        out["其中对不上的组合数"] = int(grid.group(2))
    return out


def variant(*pairs: tuple[str, str]) -> str:
    """在参考实现的源码上做几处替换，得到一份故意写错的实现。要替换的片段必须正好出现一次。"""
    src = REFERENCE_SOURCE
    for old, new in pairs:
        assert src.count(old) == 1, f"参考实现里找不到唯一的片段：{old}"
        src = src.replace(old, new)
    return src


def sig2(x):
    """保留两位有效数字（耗时只报到这个精度）。"""
    return float(f"{float(x):.2g}")


def gen(seed):
    return torch.Generator().manual_seed(int(seed))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class single_thread:
    """本章训练的网络只有 8 个通道、十几像素见方，多线程只增加调度开销；训练时临时改成单线程，结果和 2 线程逐位相同。"""

    def __enter__(self):
        torch.set_num_threads(1)

    def __exit__(self, *exc):
        torch.set_num_threads(2)


# ---------------------------------------------------------------- 数据与通用工具
def shapes(g, batch, n=16, cols=None, noise=0.2):
    """四类 5 px 大小的图案（横线、竖线、3x3 方块、斜线）放在 n x n 的噪声图上。

    cols 是图案左上角所在列的取值范围 [lo, hi)，默认整幅图。行的范围固定为 1 到 n-6。
    """
    lo, hi = cols if cols is not None else (0, n - 4)
    y = torch.randint(0, 4, (batch,), generator=g)
    x = torch.randn(batch, 1, n, n, generator=g) * noise
    rows = torch.randint(1, n - 5, (batch,), generator=g).tolist()
    cs = torch.randint(lo, hi, (batch,), generator=g).tolist()
    for i, (row, c) in enumerate(zip(rows, cs)):
        k = int(y[i])
        if k == 0:
            x[i, 0, row + 2, c:c + 5] += 1
        elif k == 1:
            x[i, 0, row:row + 5, c + 2] += 1
        elif k == 2:
            x[i, 0, row + 1:row + 4, c + 1:c + 4] += 1
        else:
            for t in range(5):
                x[i, 0, row + t, c + t] += 1
    return x, y


def count_macs(model, x):
    """一次前向里卷积和全连接的乘加次数（不计偏置、BN、池化）。"""
    total = 0

    def hook(m, _inp, out):
        nonlocal total
        if isinstance(m, nn.Conv2d):
            total += out.shape[1] * out.shape[2] * out.shape[3] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            total += m.in_features * m.out_features

    hooks = [m.register_forward_hook(hook) for m in model.modules()]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return int(total)


def read_counts(size, k, s, p, d=1, pool=False):
    """每个输入像素被多少个窗口读到：对全 1 的核求输出总和的梯度。"""
    x = torch.ones(1, 1, size, size, requires_grad=True)
    y = F.avg_pool2d(x, k, s, p, divisor_override=1) if pool else F.conv2d(x, torch.ones(1, 1, k, k), stride=s, padding=p, dilation=d)
    y.sum().backward()
    return x.grad[0, 0].round().to(torch.int64).numpy()


def rf_theory(layers):
    """layers 是 (核, 步长, 膨胀) 的列表，返回理论感受野 r 和间隔 j。"""
    rr, j = 1, 1
    for k, s, d in layers:
        rr += (k - 1) * d * j
        j *= s
    return rr, j


# ---------------------------------------------------------------- 图 1：全连接、卷积加展平、卷积加全局池化
def mlp_net(n=16, hidden=64):
    return nn.Sequential(nn.Flatten(), nn.Linear(n * n, hidden), nn.ReLU(), nn.Linear(hidden, 4))


class SmallCNN(nn.Module):
    """两层 3x3 卷积（8 通道，补零），后面接全局平均池化或者展平，再接一层全连接。"""

    def __init__(self, head="gap", c=8, n=16):
        super().__init__()
        self.c1 = nn.Conv2d(1, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.head = head
        self.fc = nn.Linear(c if head == "gap" else c * n * n, 4)

    def forward(self, x):
        x = torch.relu(self.c2(torch.relu(self.c1(x))))
        return self.fc(x.mean((2, 3)) if self.head == "gap" else x.flatten(1))


def fig_position():
    steps, batch, lr, seeds = 400, 32, 1e-2, (0, 1, 2)
    train_cols, test_cols = (0, 3), (8, 11)      # 左上角列 0..2 时图案占第 0 到 7 列；8..10 时占第 8 到 15 列
    makers = {"全连接": mlp_net, "卷积 + 展平": lambda: SmallCNN("flat"), "卷积 + 全局平均池化": lambda: SmallCNN("gap")}
    res, params, resize_ok = {}, {}, {}
    xs, ys = shapes(gen(4242), 1000, cols=train_cols)
    xu, yu = shapes(gen(4243), 1000, cols=test_cols)
    for name, make in makers.items():
        seen, unseen = [], []
        for seed in seeds:
            torch.manual_seed(seed)
            net = make()
            g = gen(seed + 50)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            with single_thread():
                for _ in range(steps):
                    x, y = shapes(g, batch, cols=train_cols)
                    loss = F.cross_entropy(net(x), y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                seen.append(float((net(xs).argmax(1) == ys).float().mean()))
                unseen.append(float((net(xu).argmax(1) == yu).float().mean()))
        res[name] = (seen, unseen)
        params[name] = n_params(net)
        try:
            with torch.no_grad():
                net(torch.zeros(1, 1, 24, 24))
            resize_ok[name] = "能跑，输出 4 个数"
        except RuntimeError:
            resize_ok[name] = "报错（矩阵形状对不上）"
    NUM["位置泛化"] = {
        "说明": "16x16 的图，噪声标准差 0.2，四类 5 px 大小的图案（横线、竖线、3x3 方块、斜线）；训练时图案只出现在左半边（第 0 到 7 列），"
              "测试分两组各 1000 张：图案在左半边（见过的位置）和右半边（第 8 到 15 列，没见过的位置）；Adam 学习率 0.01，批 32，400 步，3 个种子；"
              "全连接是 256-64-4，卷积网络是两层 3x3（8 通道，补零）；随机猜是 25%",
        "准确率（%）": {name: {"见过的位置：3 个种子的平均": r(100 * np.mean(a), 1), "见过的位置：各种子": [r(100 * v, 1) for v in a],
                           "没见过的位置：3 个种子的平均": r(100 * np.mean(b), 1), "没见过的位置：各种子": [r(100 * v, 1) for v in b]} for name, (a, b) in res.items()},
        "参数量": params,
        "把输入换成 24x24": resize_ok,
        "全连接的参数量是卷积加全局池化的多少倍": r(params["全连接"] / params["卷积 + 全局平均池化"], 1),
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.7), layout="constrained", gridspec_kw={"width_ratios": [1.25, 1.5, 0.95]})
    ax = axes[0]
    tiles = []
    for cols, seed in ((train_cols, 7), (test_cols, 8)):
        x, y = shapes(gen(seed), 64, cols=cols)
        row = [x[int((y == k).nonzero()[0])][0].numpy() for k in range(4)]
        tiles.append(np.concatenate([np.pad(t, 1, constant_values=np.nan) for t in row], axis=1))
    ax.imshow(np.concatenate(tiles, axis=0), cmap="gray", vmin=-0.5, vmax=1.3)
    for k, name in enumerate(CLASS_NAMES):
        ax.text(9 + 18 * k, -1.5, name, ha="center", va="bottom", fontsize=9.5)
    ax.text(-1.5, 9, "训练\n左半边", ha="right", va="center", fontsize=9.5)
    ax.text(-1.5, 27, "测试\n右半边", ha="right", va="center", fontsize=9.5)
    ax.set_title("四类图案，训练和测试的位置不同", pad=20)
    ax.axis("off")

    ax = axes[1]
    names = list(makers)
    xpos = np.arange(len(names))
    for off, idx, color, label in ((-0.2, 0, S.MUTED, "图案在左半边（见过）"), (0.2, 1, S.BLUE, "图案在右半边（没见过）")):
        vals = [100 * np.mean(res[n][idx]) for n in names]
        ax.bar(xpos + off, vals, 0.36, color=color, alpha=0.85, label=label)
        for i, n in enumerate(names):
            ax.plot([xpos[i] + off] * 3, [100 * v for v in res[n][idx]], "o", color=S.INK, ms=3, zorder=3)
            ax.text(xpos[i] + off, vals[i] + 2, f"{vals[i]:.0f}", ha="center", fontsize=9.5)
    ax.axhline(25, color=S.RED, lw=1, ls="--", label="随机猜 25%")
    ax.set_xticks(xpos, ["全连接", "卷积 + 展平", "卷积 +\n全局平均池化"])
    ax.set_ylim(0, 112)
    ax.set_ylabel("准确率（%）")
    ax.set_title("换到没见过的位置，只有卷积加全局池化还对")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9, columnspacing=1.0, handlelength=1.4)
    S.clean(ax)

    ax = axes[2]
    vals = [params[n] for n in names]
    ax.bar(xpos, vals, 0.55, color=[S.MUTED, S.ORANGE, S.GREEN])
    for i, v in enumerate(vals):
        ax.text(i, v * 1.12, f"{v}", ha="center", fontsize=9.5)
    ax.set_yscale("log")
    ax.set_ylim(100, 60000)
    ax.set_xticks(xpos, ["全连接", "卷积 +\n展平", "卷积 +\n全局池化"])
    ax.set_ylabel("参数量")
    ax.set_title("参数量")
    ax.minorticks_off()
    S.clean(ax)
    S.save(fig, CH, "fig01_位置泛化")


# ---------------------------------------------------------------- 图 2：输出尺寸、被丢掉的像素、参数量和计算量
def wrong_trunc(size, kernel, stride=1, padding=0, dilation=1):
    """一种常见的错误写法：用浮点除法再 int()，负数会向零取整。"""
    return int((size + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1


def bench(model, x, reps=30):
    with torch.no_grad():
        for _ in range(5):
            model(x)
        best = math.inf
        for _ in range(3):
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                model(x)
                ts.append(time.perf_counter() - t0)
            best = min(best, float(np.median(ts)))
    return best * 1e3


def conv_variants(c=64):
    torch.manual_seed(0)
    return {
        "普通 3x3": nn.Conv2d(c, c, 3, padding=1, bias=False),
        "分组 3x3，4 组": nn.Conv2d(c, c, 3, padding=1, groups=4, bias=False),
        "逐通道 3x3（组数等于通道数）": nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
        "深度可分离（逐通道 3x3 加 1x1）": nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False), nn.Conv2d(c, c, 1, bias=False)),
        "1x1": nn.Conv2d(c, c, 1, bias=False),
        "两层 3x3": nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.Conv2d(c, c, 3, padding=1, bias=False)),
        "一层 5x5": nn.Conv2d(c, c, 5, padding=2, bias=False),
    }


def fig_cost():
    # 输出尺寸公式和 PyTorch 的实际输出逐个比
    grid = list(itertools.product(range(1, 41), (1, 2, 3, 4, 5, 7), (1, 2, 3, 4), (0, 1, 2, 3), (1, 2, 3)))
    agree = both_reject = trunc_wrong = 0
    trunc_example = None
    for size, k, s, p, d in grid:
        try:
            actual = F.conv2d(torch.zeros(1, 1, size, size), torch.zeros(1, 1, k, k), stride=s, padding=p, dilation=d).shape[-1]
        except RuntimeError:
            actual = None
        try:
            pred = conv_out(size, k, s, p, d)
        except ValueError:
            pred = None
        agree += actual == pred
        both_reject += actual is None and pred is None
        if pred is None and wrong_trunc(size, k, s, p, d) >= 1:
            trunc_wrong += 1
            if trunc_example is None and (size, k, s, p, d) == (2, 3, 2, 0, 1):
                trunc_example = wrong_trunc(size, k, s, p, d)
    NUM["输出尺寸公式"] = {
        "说明": "参考实现 conv_output_size 和 torch.nn.functional.conv2d 的实际输出逐个比较；输入边长 1 到 40，核 1、2、3、4、5、7，步长 1 到 4，填充 0 到 3，膨胀 1 到 3",
        "组合数": len(grid), "两者一致的组合数": int(agree), "其中 PyTorch 报错、参考实现也拒绝的组合数": int(both_reject),
        "错误写法 int(x / stride) + 1 在本该拒绝时返回正数的组合数": int(trunc_wrong),
        "例：输入 2、核 3、步长 2、无填充，错误写法返回": trunc_example,
        "原讲义的两个算例": {"32，核 3，步长 1，填充 1": conv_out(32, 3, 1, 1), "32，核 3，步长 2，填充 1": conv_out(32, 3, 2, 1)},
        "验收脚本最早的三个算例（都能整除）": {"32，核 3，步长 1，填充 1": conv_out(32, 3, 1, 1), "31，核 3，步长 2，填充 1": conv_out(31, 3, 2, 1), "15，核 3，步长 1，填充 2，膨胀 2": conv_out(15, 3, 1, 2, 2)},
    }

    # 向下取整丢掉了什么
    cases = [("2x2 池化，步长 2，输入 7x7", 7, 2, 2, 0, True), ("2x2 池化，步长 2，输入 33x33", 33, 2, 2, 0, True),
             ("3x3 卷积，步长 2，无填充，输入 8x8", 8, 3, 2, 0, False), ("3x3 卷积，步长 2，无填充，输入 32x32", 32, 3, 2, 0, False),
             ("3x3 卷积，步长 2，填充 1，输入 32x32", 32, 3, 2, 1, False), ("3x3 卷积，步长 2，填充 1，输入 31x31", 31, 3, 2, 1, False),
             ("3x3 卷积，步长 3，无填充，输入 32x32", 32, 3, 3, 0, False), ("1x1 卷积，步长 2，输入 8x8", 8, 1, 2, 0, False),
             ("1x1 卷积，步长 2，输入 32x32", 32, 1, 2, 0, False)]
    dropped = {}
    for name, size, k, s, p, pool in cases:
        cnt = read_counts(size, k, s, p, pool=pool)
        zero = int((cnt == 0).sum())
        dropped[name] = {"输出边长": conv_out(size, k, s, p), "公式里除以步长的余数": (size + 2 * p - (k - 1) - 1) % s,
                         "没有被任何窗口读到的像素数": zero, "占全部像素的比例（%）": r(100 * zero / size / size, 1)}
    NUM["向下取整丢掉的像素"] = {"说明": "把核换成全 1，对输出总和求梯度，梯度为 0 的输入像素就是没有被任何窗口读到的像素", "各种情况": dropped}
    c0, c1 = read_counts(8, 3, 1, 0), read_counts(8, 3, 1, 1)
    NUM["边缘像素被读到的次数"] = {"说明": "8x8 输入，3x3 卷积，步长 1；每个输入像素被多少个窗口读到",
                          "无填充": {"角": int(c0[0, 0]), "边": int(c0[0, 3]), "内部": int(c0[3, 3])},
                          "填充 1": {"角": int(c1[0, 0]), "边": int(c1[0, 3]), "内部": int(c1[3, 3])}}

    # 参数量、乘加次数、耗时
    c = 64
    variants = conv_variants(c)
    x1 = torch.randn(1, c, 56, 56, generator=gen(0))
    stat = {name: {"参数量": n_params(m), "乘加次数（百万）": r(count_macs(m, x1) / 1e6, 2)} for name, m in variants.items()}
    first, late = nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.Conv2d(256, 256, 3, padding=1, bias=False)
    xf, xl = torch.randn(1, 3, 224, 224, generator=gen(1)), torch.randn(1, 256, 14, 14, generator=gen(2))

    def measure():
        out = {"说明": "在一台 Apple 芯片的 Mac 上用 CPU（2 线程）测的一次结果，每个数是 3 轮、每轮 30 次前向的中位数里最小的一轮，单位毫秒，只保留两位有效数字；"
                     "测的时候机器上还有别的任务在跑，只能看量级和明显的倍数关系"}
        for batch in (1, 16):
            xb = torch.randn(batch, c, 56, 56, generator=gen(3))
            out[f"批 {batch}"] = {name: sig2(bench(m, xb)) for name, m in variants.items()}
        out["两层的对比（批 1）"] = {"3 到 64 通道，224x224": sig2(bench(first, xf)), "256 到 256 通道，14x14": sig2(bench(late, xl))}
        return out

    timing = S.pinned(CH, "耗时", measure, "CH09_RETIME")
    NUM["耗时"] = timing
    base = stat["普通 3x3"]
    for name, v in stat.items():
        v["参数量是普通 3x3 的几分之一"] = r(base["参数量"] / v["参数量"], 1)
        v["乘加次数是普通 3x3 的几分之一"] = r(base["乘加次数（百万）"] / v["乘加次数（百万）"], 1)
        for batch in (1, 16):
            v[f"批 {batch} 的耗时是普通 3x3 的几分之一"] = sig2(timing[f"批 {batch}"]["普通 3x3"] / timing[f"批 {batch}"][name])
    NUM["参数量与计算量"] = {
        "说明": "输入 64 通道、56x56，输出 64 通道，都不带偏置；乘加次数按 输出高 x 输出宽 x 输出通道 x 每组输入通道 x 核高 x 核宽 计，由前向钩子统计",
        "各种卷积": stat,
        "两层的对比": {
            "3 到 64 通道的 3x3，输入 224x224": {"参数量": n_params(first), "乘加次数（百万）": r(count_macs(first, xf) / 1e6, 1), "输出元素个数": 64 * 224 * 224},
            "256 到 256 通道的 3x3，输入 14x14": {"参数量": n_params(late), "乘加次数（百万）": r(count_macs(late, xl) / 1e6, 1), "输出元素个数": 256 * 14 * 14},
            "后者的参数量是前者的多少倍": r(n_params(late) / n_params(first), 0),
            "前者的输出元素个数是后者的多少倍": r(64 * 224 * 224 / (256 * 14 * 14), 0),
        },
    }
    NUM["原讲义里的算例"] = {
        "224x224x3 的元素个数": 224 * 224 * 3, "全连接到 4096 个神经元的权重数": 224 * 224 * 3 * 4096, "约合多少亿": r(224 * 224 * 3 * 4096 / 1e8, 2),
        "3 到 64 通道的 3x3 卷积的权重数": 64 * 3 * 3 * 3, "3 到 16 通道的 3x3 卷积带偏置的参数量": 16 * (3 * 3 * 3 + 1),
        "两层 3x3 的权重数是 C 平方的多少倍": 18, "一层 5x5 的权重数是 C 平方的多少倍": 25,
        "C=64 时两层 3x3 的权重数": 2 * 9 * 64 * 64, "C=64 时一层 5x5 的权重数": 25 * 64 * 64,
        "二维卷积输入的例子": [32, 3, 32, 32], "第一层输出": [32, 16, 32, 32], "第二层输出": [32, 32, 16, 16],
    }

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1, 1, 1.75]})
    shows = [("2x2 池化，步长 2\n输入 7x7", 7, 2, 2, 0, True), ("3x3 卷积，步长 2，无填充\n输入 8x8", 8, 3, 2, 0, False), ("1x1 卷积，步长 2\n输入 8x8", 8, 1, 2, 0, False)]
    for ax, (title, size, k, s, p, pool) in zip(axes[:3], shows):
        cnt = read_counts(size, k, s, p, pool=pool)
        rgb = np.ones((size, size, 3))
        rgb[cnt == 0] = (0.96, 0.78, 0.76)
        rgb[cnt == 1] = (0.84, 0.90, 0.99)
        rgb[cnt >= 2] = (0.62, 0.76, 0.97)
        ax.imshow(rgb)
        for i in range(size):
            for j in range(size):
                ax.text(j, i, str(int(cnt[i, j])), ha="center", va="center", fontsize=8, color=S.RED if cnt[i, j] == 0 else S.INK)
        ax.set_xticks(np.arange(-0.5, size), minor=True)
        ax.set_yticks(np.arange(-0.5, size), minor=True)
        ax.grid(which="minor", color="white", lw=1)
        ax.tick_params(which="minor", length=0)
        S.image_axes(ax)
        zero = int((cnt == 0).sum())
        ax.set_title(f"{title}", fontsize=10.5)
        ax.set_xlabel(f"{zero} 个像素没被读到（{100 * zero / size / size:.0f}%）", color=S.RED)

    ax = axes[3]
    names = ["普通 3x3", "分组 3x3，4 组", "深度可分离（逐通道 3x3 加 1x1）"]
    xpos = np.arange(len(names))
    series = [("参数量", [stat[n]["参数量"] / base["参数量"] for n in names], S.MUTED),
              ("乘加次数", [stat[n]["乘加次数（百万）"] / base["乘加次数（百万）"] for n in names], S.BLUE),
              ("CPU 耗时，批 1", [timing["批 1"][n] / timing["批 1"]["普通 3x3"] for n in names], S.ORANGE),
              ("CPU 耗时，批 16", [timing["批 16"][n] / timing["批 16"]["普通 3x3"] for n in names], S.RED)]
    for i, (label, vals, color) in enumerate(series):
        ax.bar(xpos + (i - 1.5) * 0.2, vals, 0.19, color=color, label=label)
        for xx, v in zip(xpos + (i - 1.5) * 0.2, vals):
            ax.text(xx, v * (1.08 if i % 2 == 0 else 1.34), f"{v:.2g}", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(0.06, 3.0)
    ax.set_yticks([0.1, 0.25, 0.5, 1], ["0.1", "0.25", "0.5", "1"])
    ax.minorticks_off()
    ax.set_xticks(xpos, ["普通 3x3", "分组，4 组", "深度可分离"])
    ax.set_ylabel("相对普通 3x3（对数轴）")
    ax.set_title("参数和乘加次数少了，耗时没有按比例少")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=9, columnspacing=1.0, handlelength=1.2)
    S.clean(ax)
    S.save(fig, CH, "fig02_取整与计算量")


# ---------------------------------------------------------------- 图 3：步长和填充怎样破坏平移等变
def seamless_texture(seed, batch=8, n=64):
    """首尾相接的平滑随机纹理：白噪声做循环的 5x5 均值滤波。"""
    x = torch.randn(batch, 1, n, n, generator=gen(seed))
    return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="circular"), torch.ones(1, 1, 5, 5) / 25) * 5


class DownNet(nn.Module):
    """三层 3x3 卷积加 ReLU（8 通道，循环填充）。kind 决定怎么下采样：不下采样、步长 2 的卷积、卷积后 2x2 最大池化或平均池化。"""

    def __init__(self, kind, depth=3, c=8, pad_mode="circular"):
        super().__init__()
        self.kind = kind
        stride = 2 if kind == "步长 2 的卷积" else 1
        self.convs = nn.ModuleList([nn.Conv2d(1 if i == 0 else c, c, 3, stride=stride, padding=1, padding_mode=pad_mode) for i in range(depth)])

    def forward(self, x):
        for conv in self.convs:
            x = torch.relu(conv(x))
            if self.kind == "2x2 最大池化":
                x = F.max_pool2d(x, 2)
            elif self.kind == "2x2 平均池化":
                x = F.avg_pool2d(x, 2)
        return x


def fig_equivariance():
    kinds = ["不下采样（步长 1）", "步长 2 的卷积", "2x2 最大池化", "2x2 平均池化"]
    shifts = list(range(0, 17))
    seeds = range(5)
    x = seamless_texture(11)
    eq = {k: np.zeros(len(shifts)) for k in kinds}
    inv = {k: np.zeros(len(shifts)) for k in kinds}
    for kind in kinds:
        for seed in seeds:
            torch.manual_seed(seed)
            net = DownNet(kind)
            with torch.no_grad():
                y0 = net(x)
                g0 = y0.mean((2, 3))
                for i, t in enumerate(shifts):
                    y1 = net(torch.roll(x, t, dims=-1))
                    best = min(float((torch.roll(y0, s, dims=-1) - y1).norm() / y1.norm()) for s in range(0, t + 1))
                    eq[kind][i] += best / len(seeds)
                    inv[kind][i] += float((y1.mean((2, 3)) - g0).norm() / g0.norm()) / len(seeds)
    pick = [1, 2, 4, 7, 8, 16]
    NUM["平移等变"] = {
        "说明": "64x64 的首尾相接随机纹理 8 张，沿 x 方向循环平移 t 个像素；网络是三层 3x3 卷积加 ReLU（8 通道，随机初始化，循环填充，没有训练），5 组随机权重取平均；"
              "等变误差：把平移前的输出循环平移整数格去对平移后的输出，取最好的那个整数格，相对误差（差的范数除以输出的范数）；"
              "全局平均后的变化：每个通道做全局平均得到 8 维向量，平移前后的相对变化。三种下采样的总步长都是 8",
        "平移量（px）": pick,
        "等变误差": {k: [float(f"{eq[k][t]:.2g}") if eq[k][t] < 1e-4 else r(eq[k][t], 3) for t in pick] for k in kinds},
        "全局平均后的相对变化（%）": {k: [float(f"{100 * inv[k][t]:.2g}") if inv[k][t] < 1e-6 else r(100 * inv[k][t], 2) for t in pick] for k in kinds},
        "平移 1 到 7 px 的全局平均变化的平均值（%）": {k: r(100 * inv[k][1:8].mean(), 2) for k in kinds},
        "平移 1 到 7 px 的等变误差的平均值": {k: r(eq[k][1:8].mean(), 3) for k in kinds},
    }

    # 真实平移（不是循环平移）加补边：误差离边界多远
    depth, n, t = 4, 48, 3
    modes = {"补零": "zeros", "镜像": "reflect", "复制边缘": "replicate"}
    prof = {m: np.zeros(n - t) for m in modes}
    for name, mode in modes.items():
        for seed in seeds:
            big = seamless_texture(100 + seed, batch=1, n=96)
            x0, x1 = big[..., 24:24 + n, 24:24 + n], big[..., 24:24 + n, 24 + t:24 + t + n]
            torch.manual_seed(seed)
            net = DownNet("不下采样（步长 1）", depth=depth, pad_mode=mode)
            with torch.no_grad():
                y0, y1 = net(x0), net(x1)
            d = (y1[..., : n - t] - y0[..., t:]).pow(2).mean((0, 1, 2)).sqrt() / y0.pow(2).mean().sqrt()
            prof[name] += d.numpy() / len(seeds)
    band = {name: int((p[: (n - t) // 2] > 1e-5).sum()) for name, p in prof.items()}
    NUM["补边与平移"] = {
        "说明": f"从一张 96x96 的纹理上裁两块 {n}x{n}，第二块比第一块向右错开 {t} px（真实平移，边上进来的是新内容）；网络是 {depth} 层 3x3 卷积加 ReLU，步长 1，随机初始化，5 组取平均；"
              "把两个输出按 3 列对齐后逐列比较，误差是这一列的均方根差除以整个输出的均方根",
        "左边受影响的列数": band,
        "按离左边界的列数（0 是最外一列）：相对误差": {name: [r(v, 3) if v > 1e-5 else float(f"{v:.1g}") for v in p[:6]] for name, p in prof.items()},
        "中间一列的相对误差": {name: float(f"{p[(n - t) // 2]:.1g}") for name, p in prof.items()},
        "理论感受野": rf_theory([(3, 1, 1)] * depth)[0],
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), layout="constrained")
    colors = dict(zip(kinds, [S.GREEN, S.RED, S.ORANGE, S.BLUE]))
    for ax, data, ylabel, title in ((axes[0], eq, "等变误差（相对）", "有下采样：只有平移 8 的倍数才等变"),
                                    (axes[1], inv, "全局平均后的相对变化", "全局平均后：步长 2 的卷积变化最大")):
        for k in kinds:
            ax.plot(shifts, data[k], "o-", color=colors[k], ms=3.5, lw=1.5, label=k)
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
        ax.set_xlabel("输入平移（px）")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        S.clean(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower left", ncol=4)
    ax = axes[2]
    for (name, p), color in zip(prof.items(), (S.RED, S.BLUE, S.MUTED)):
        ax.plot(np.arange(12), p[:12], "o-", color=color, ms=3.5, lw=1.5, label=name)
    ax.axvspan(-0.5, depth - 0.5, color=S.ORANGE, alpha=0.1)
    ax.text(depth + 0.3, ax.get_ylim()[1] * 0.5, f"阴影：最外 {depth} 列\n（{depth} 层 3x3）", ha="left", va="center", fontsize=9.5, color=S.ORANGE)
    ax.set_xticks(range(0, 12, 2))
    ax.set_xlabel("离左边界的列数")
    ax.set_ylabel("这一列的相对误差")
    ax.set_title("步长 1 加补边：只有边界附近不等变")
    ax.legend(loc="upper right")
    S.clean(ax)
    S.save(fig, CH, "fig03_平移等变")


# ---------------------------------------------------------------- 图 4：理论感受野和有效感受野
def grad_support(layers):
    """把核换成全 1，对中心输出求梯度，返回梯度非零区域的边长。"""
    rr, j = rf_theory(layers)
    n = rr + 2 * j + 8
    x = torch.ones(1, 1, n, n, requires_grad=True)
    y = x
    for k, s, d in layers:
        y = F.conv2d(y, torch.ones(1, 1, k, k), stride=s, dilation=d)
    c = y.shape[-1] // 2
    y[0, 0, c, c].backward()
    rows = (x.grad[0, 0].sum(1) != 0).nonzero().flatten()
    return int(rows[-1] - rows[0] + 1)


def erf_map(layers, trials=40, c=8, seed=0):
    """有效感受野：输入边长正好等于理论感受野、不补边，输出只有一个位置；对它求梯度，梯度平方按随机权重和随机输入取平均。"""
    rr, _ = rf_theory(layers)
    acc = torch.zeros(rr, rr)
    for t in range(trials):
        torch.manual_seed(seed * 1000 + t)
        mods = []
        for i, (k, s, d) in enumerate(layers):
            mods.append(nn.Conv2d(1 if i == 0 else c, c, k, stride=s, dilation=d, bias=False))
            if i < len(layers) - 1:
                mods.append(nn.ReLU())
        net = nn.Sequential(*mods)
        x = torch.randn(1, 1, rr, rr, requires_grad=True)
        net(x).sum().backward()
        g = x.grad[0, 0] ** 2
        acc += g / g.sum()
    return (acc / trials).numpy()


def box_energy(depth):
    """各条路径的权重独立同分布时，梯度平方的期望正比于“路径条数”：3x3 的全 1 核自卷积 depth 次。"""
    k = np.ones(3)
    p = np.ones(1)
    for _ in range(depth):
        p = np.convolve(p, k)
    e = np.outer(p, p)
    return e / e.sum()


def erf_width(e):
    n = e.shape[0]
    c = n // 2
    xs = np.arange(n) - c
    std = float(np.sqrt((e.sum(0) * xs ** 2).sum()))
    for h in range(c + 1):
        if e[c - h:c + h + 1, c - h:c + h + 1].sum() >= 0.95:
            break
    return std, 2 * h + 1


def fig_rf():
    configs = {
        "原讲义的例子：3x3，2x2 步长 2，3x3": [(3, 1, 1), (2, 2, 1), (3, 1, 1)],
        "两层 3x3": [(3, 1, 1)] * 2,
        "一层 5x5": [(5, 1, 1)],
        "三层 3x3 步长 2": [(3, 2, 1)] * 3,
        "3x3 膨胀 2 再 3x3 膨胀 4": [(3, 1, 2), (3, 1, 4)],
        "7x7 步长 2，3x3 步长 2，两层 3x3": [(7, 2, 1), (3, 2, 1), (3, 1, 1), (3, 1, 1)],
        "十层 3x3": [(3, 1, 1)] * 10,
    }
    check = {name: {"公式：感受野": rf_theory(l)[0], "公式：间隔": rf_theory(l)[1], "梯度非零区域的边长": grad_support(l)} for name, l in configs.items()}
    steps = []
    rr, j = 1, 1
    for k, s, d in configs["原讲义的例子：3x3，2x2 步长 2，3x3"]:
        rr += (k - 1) * d * j
        j *= s
        steps.append([rr, j])
    NUM["感受野公式"] = {"说明": "把每层的核换成全 1、不加非线性，对中心输出求梯度，梯度非零的区域就是它实际依赖的输入范围；和公式 r += (k-1)*d*j、j *= s 的结果比较",
                    "各种结构": check, "一致的个数": sum(v["公式：感受野"] == v["梯度非零区域的边长"] for v in check.values()), "结构个数": len(check),
                    "原讲义例子逐层的 r 和 j": steps}

    depths = [2, 3, 5, 10, 15, 20, 30]
    rows, maps = {}, {}
    for depth in depths:
        e = erf_map([(3, 1, 1)] * depth)
        maps[depth] = e
        std, w95 = erf_width(e)
        std_t, w95_t = erf_width(box_energy(depth))
        rf = 2 * depth + 1
        rows[depth] = {"理论感受野边长": rf, "实测：包住 95% 梯度能量的中心方块边长": w95, "这个方块占理论感受野面积的比例（%）": r(100 * w95 * w95 / rf / rf, 0),
                       "实测：梯度能量沿一个方向的标准差（px）": r(std, 2), "按路径条数算的标准差（px）": r(std_t, 2), "按路径条数算的 95% 方块边长": w95_t,
                       "理论感受野最外一圈占的梯度能量（%）": float(f"{100 * (1 - e[1:-1, 1:-1].sum()):.2g}")}
    strided = [(3, 1, 1)] * 3 + [(3, 2, 1)] + [(3, 1, 1)] * 2 + [(3, 2, 1)] + [(3, 1, 1)] * 3
    es = erf_map(strided)
    std_s, w95_s = erf_width(es)
    NUM["有效感受野"] = {
        "说明": "n 层 3x3 卷积（8 通道，步长 1，PyTorch 默认初始化，层间加 ReLU，没有训练）；对最后一层中心位置的输出求输入的梯度，梯度平方按 40 组随机权重和随机输入取平均后归一化，当作“每个输入像素的影响有多大”；"
              "“按路径条数算”指把 3x3 的全 1 核自卷积 n 次，对应各条路径权重独立时梯度平方的期望",
        "按层数": rows,
        "十层里第 4、7 层改成步长 2": {"理论感受野边长": rf_theory(strided)[0], "实测：包住 95% 梯度能量的中心方块边长": w95_s, "实测：标准差（px）": r(std_s, 2),
                               "对照：十层都是步长 1 的理论感受野": 21, "对照：十层都是步长 1 的 95% 方块边长": rows[10]["实测：包住 95% 梯度能量的中心方块边长"]},
        "标准差之比：20 层比 5 层": r(rows[20]["实测：梯度能量沿一个方向的标准差（px）"] / rows[5]["实测：梯度能量沿一个方向的标准差（px）"], 2),
        "理论感受野之比：20 层比 5 层": r(41 / 11, 2),
    }

    # ResNet-18 主干的理论感受野
    main = [(7, 2, 1), (3, 2, 1)] + [(3, 1, 1)] * 4 + [(3, 2, 1)] + [(3, 1, 1)] * 3 + [(3, 2, 1)] + [(3, 1, 1)] * 3 + [(3, 2, 1)] + [(3, 1, 1)] * 3
    stages = {"第一层卷积加池化": 2, "layer1": 6, "layer2": 10, "layer3": 14, "layer4": 18}
    NUM["ResNet-18 的理论感受野"] = {
        "说明": "按 torchvision 的 ResNet-18 主干逐层套公式：7x7 步长 2，3x3 池化步长 2，然后四个阶段各四层 3x3，后三个阶段的第一层步长 2；捷径上的 1x1 卷积感受野更小，不影响最大范围",
        "各阶段输出的（感受野，间隔）": {name: list(rf_theory(main[:cut])) for name, cut in stages.items()},
        "梯度法核对": {name: grad_support(main[:cut]) for name, cut in stages.items()},
        "224 输入时各阶段的输出边长": {"第一层卷积": conv_out(224, 7, 2, 3), "池化": conv_out(conv_out(224, 7, 2, 3), 3, 2, 1), "layer1": 56,
                             "layer2": conv_out(56, 3, 2, 1), "layer3": conv_out(conv_out(56, 3, 2, 1), 3, 2, 1), "layer4": conv_out(conv_out(conv_out(56, 3, 2, 1), 3, 2, 1), 3, 2, 1)},
        "输入边长": 224,
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), layout="constrained", gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, depth in zip(axes[:2], (5, 20)):
        e = maps[depth]
        rf = 2 * depth + 1
        w95 = rows[depth]["实测：包住 95% 梯度能量的中心方块边长"]
        ax.imshow(e / e.max(), cmap="magma", vmin=0, vmax=1, extent=(-rf / 2, rf / 2, rf / 2, -rf / 2))
        ax.add_patch(plt.Rectangle((-w95 / 2, -w95 / 2), w95, w95, fill=False, ec="#34d4ff", lw=1.6, ls="--"))
        ax.set_title(f"{depth} 层 3x3：理论感受野 {rf}x{rf}")
        ax.set_xlabel(f"虚线方块 {w95}x{w95} 包住 95% 的梯度能量")
        ax.set_xticks([-(rf // 2), 0, rf // 2])
        ax.set_yticks([-(rf // 2), 0, rf // 2])
    ax = axes[2]
    ax.plot(depths, [2 * d + 1 for d in depths], "o-", color=S.MUTED, lw=1.6, ms=4, label="理论感受野边长 2n+1")
    ax.plot(depths, [rows[d]["实测：包住 95% 梯度能量的中心方块边长"] for d in depths], "o-", color=S.BLUE, lw=1.8, ms=4, label="实测：95% 能量的方块边长")
    ax.plot(depths, [rows[d]["按路径条数算的 95% 方块边长"] for d in depths], "x--", color=S.ORANGE, lw=1.2, ms=5, label="按路径条数算")
    ax.set_xlabel("3x3 卷积的层数 n")
    ax.set_ylabel("边长（px）")
    ax.set_title("层数加倍，理论感受野加倍，有效的那块没有")
    ax.legend(loc="upper left")
    S.clean(ax)
    S.save(fig, CH, "fig04_感受野")


# ---------------------------------------------------------------- 图 5：BatchNorm 的训练模式和推理模式
def bn_images(g, batch, n=16):
    """每张图有自己的亮度和对比度：亮度偏置的标准差 0.5，对比度在 0.5 到 1.5 之间，内容是白噪声。"""
    base = torch.randn(batch, 1, 1, 1, generator=g) * 0.5
    gain = torch.rand(batch, 1, 1, 1, generator=g) + 0.5
    return base + gain * torch.randn(batch, 1, n, n, generator=g)


class TinyResNet(nn.Module):
    """3x3 卷积加 BN，接若干个作业里的残差块，全局平均池化后接全连接。"""

    def __init__(self, blocks, c=8, block=None):
        super().__init__()
        self.stem = nn.Conv2d(1, c, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.blocks = nn.ModuleList([(block or ResBlock)(c) for _ in range(blocks)])
        self.fc = nn.Linear(c, 4)

    def forward(self, x):
        x = torch.relu(self.bn(self.stem(x)))
        for b in self.blocks:
            x = b(x)
        return self.fc(x.mean((2, 3)))


def set_momentum(net, value):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = value


def fig_bn():
    torch.manual_seed(0)
    conv = nn.Conv2d(1, 8, 3, padding=1, bias=False)
    g = gen(1)
    probe, pool = bn_images(g, 1), bn_images(g, 4096)
    bn = nn.BatchNorm2d(8).train()
    with torch.no_grad():
        for _ in range(300):
            bn(conv(pool[torch.randint(0, 4096, (256,), generator=g)]))
        bn.eval()
        ref = bn(conv(probe))[0]
        true_var = bn.running_var.clone()
    ref_rms = ref.pow(2).mean().sqrt()

    # (a) 同一张图放进不同的批
    batches = [1, 2, 4, 8, 16, 32, 64, 128]
    dev, spread = [], []
    live = nn.BatchNorm2d(8).train()
    live.load_state_dict(bn.state_dict())
    live.momentum = 0.0
    for b in batches:
        outs = []
        with torch.no_grad():
            for _ in range(100):
                x = torch.cat([probe, pool[torch.randint(0, 4096, (b - 1,), generator=g)]]) if b > 1 else probe
                outs.append(live(conv(x))[0])
        outs = torch.stack(outs)
        dev.append(float((outs - ref).pow(2).mean().sqrt() / ref_rms))
        spread.append(float(outs.std(0).pow(2).mean().sqrt() / ref_rms))
    with torch.no_grad():
        gn = nn.GroupNorm(2, 8)
        outs_gn = torch.stack([gn(conv(torch.cat([probe, pool[torch.randint(0, 4096, (7,), generator=g)]])))[0] for _ in range(20)])
        outs_ev = torch.stack([bn(conv(torch.cat([probe, pool[torch.randint(0, 4096, (7,), generator=g)]])))[0] for _ in range(20)])
        x1 = torch.randn(1, 1, 16, 16, generator=g)
        d_train = float((live(conv(x1)) - live(conv(3 * x1))).abs().max())
        d_eval = float((bn(conv(3 * x1)).pow(2).mean().sqrt() / bn(conv(x1)).pow(2).mean().sqrt()))
        bn1 = nn.BatchNorm1d(4, momentum=1.0).train()
        v = torch.randn(2, 4, generator=g)
        out1 = bn1(v)
        var_ratio = float((bn1.running_var / v.var(0, unbiased=False)).mean())
    try:
        nn.BatchNorm1d(4).train()(torch.randn(1, 4))
        one = "没有报错"
    except ValueError as exc:
        one = "ValueError: " + str(exc).split(",")[0]
    NUM["BN 与批大小"] = {
        "说明": "一层 3x3 卷积（1 到 8 通道）后面接 BatchNorm2d；数据是 16x16 的噪声图，每张图有自己的亮度（偏置标准差 0.5）和对比度（0.5 到 1.5）；"
              "先用批 256 跑 300 步让 running 统计收敛，推理模式的输出当基准；再把同一张图和随机抽的 B-1 张图拼成一批，用训练模式前向 100 次（不更新 running 统计）",
        "批大小": batches,
        "训练模式输出和推理模式输出的相对差": [r(v, 3) for v in dev],
        "同一张图换同伴后输出的相对波动": [float(f"{v:.2g}") if v < 1e-6 else r(v, 3) for v in spread],
        "相对差乘以根号批大小": [r(v * math.sqrt(b), 2) for v, b in zip(dev, batches)],
        "推理模式下换同伴后输出的最大波动": float(outs_ev.std(0).max()),
        "GroupNorm（2 组）换同伴后输出的最大波动": float(outs_gn.std(0).max()),
        "批 1、训练模式：输入乘以 3 后输出的最大变化": float(f"{d_train:.1g}"),
        "推理模式：输入乘以 3 后输出的均方根是原来的几倍": r(d_eval, 2),
        "BatchNorm1d 批 2 的训练模式输出（2 个样本、4 个特征）": [[r(t, 3) for t in row] for row in out1.tolist()],
        "批里只有 2 个值时，running_var 记下的方差是训练时所用方差的几倍": r(var_ratio, 2),
        "BatchNorm1d 批 1 的训练模式": one,
    }

    # (b) running 统计要多少步才跟上
    ks = [1, 2, 5, 10, 20, 30, 50, 100, 200]
    ratio, var_rel, pred = [], [], []
    for k in ks:
        b = nn.BatchNorm2d(8).train()
        with torch.no_grad():
            for _ in range(k):
                b(conv(pool[torch.randint(0, 4096, (64,), generator=g)]))
            x = pool[torch.randint(0, 4096, (256,), generator=g)]
            b.momentum = 0.0
            yt = b(conv(x))
            ye = b.eval()(conv(x))
        ratio.append(float(ye.pow(2).mean().sqrt() / yt.pow(2).mean().sqrt()))
        var_rel.append(float((b.running_var / true_var).mean()))
        pred.append(float(((0.9 ** k + (1 - 0.9 ** k) * true_var) / true_var).mean()))
    NUM["BN 的 running 统计"] = {
        "说明": "同一层卷积和数据，新建一个 BatchNorm2d（running_var 初值 1，momentum 默认 0.1），用批 64 在训练模式下前向 k 次，再比较同一批数据在推理模式和训练模式下的输出",
        "前向次数 k": ks,
        "推理模式输出的均方根除以训练模式的": [r(v, 2) for v in ratio],
        "running_var 除以收敛后的值": [r(v, 2) for v in var_rel],
        "按 0.9 的 k 次方推算的 running_var 比值": [r(v, 2) for v in pred],
        "初值占的权重 0.9^k（%）": [r(100 * 0.9 ** k, 1) for k in ks],
        "收敛后各通道 running_var 的平均": r(true_var.mean(), 3),
    }

    # (c) 小样本过拟合：训练模式和推理模式的准确率
    marks = [10, 20, 30, 40, 60, 80, 120, 200]
    acc_t, acc_e = np.zeros((5, len(marks))), np.zeros((5, len(marks)))
    for seed in range(5):
        torch.manual_seed(seed)
        net = TinyResNet(2)
        x, y = shapes(gen(seed + 5), 16, n=12, noise=0.3)
        opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
        for step in range(1, marks[-1] + 1):
            loss = F.cross_entropy(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step in marks:
                i = marks.index(step)
                set_momentum(net, 0.0)
                with torch.no_grad():
                    acc_t[seed, i] = float((net(x).argmax(1) == y).float().mean())
                    net.eval()
                    acc_e[seed, i] = float((net(x).argmax(1) == y).float().mean())
                net.train()
                set_momentum(net, 0.1)
    NUM["小样本过拟合时的 BN"] = {
        "说明": "固定 16 张 12x12 的四类图案图（噪声 0.3），网络是 3x3 卷积加 BN、2 个作业里的残差块、全局平均池化、全连接；SGD 学习率 0.05、动量 0.9，每步都用这 16 张；"
              "在同样这 16 张图上分别用训练模式和推理模式算准确率，5 个种子取平均",
        "步数": marks,
        "训练模式的准确率（%）": [r(100 * v, 1) for v in acc_t.mean(0)],
        "推理模式的准确率（%）": [r(100 * v, 1) for v in acc_e.mean(0)],
        "5 个种子里推理模式低于训练模式的个数": [int((acc_e[:, i] < acc_t[:, i] - 1e-9).sum()) for i in range(len(marks))],
    }

    # 卷积的偏置接 BN
    torch.manual_seed(3)
    c2, b2 = nn.Conv2d(1, 8, 3, padding=1, bias=True), nn.BatchNorm2d(8).train()
    xb = pool[:32]
    out = b2(c2(xb))
    (out * torch.randn(out.shape, generator=g)).sum().backward()
    with torch.no_grad():
        y1 = b2(c2(xb))
        c2.bias += 5
        y2 = b2(c2(xb))
    NUM["卷积偏置接 BN"] = {"说明": "带偏置的 3x3 卷积后面接训练模式的 BatchNorm2d，批 32；损失是输出和一组随机数的内积",
                       "偏置的梯度范数除以权重的梯度范数": float(f"{float(c2.bias.grad.norm() / c2.weight.grad.norm()):.1g}"),
                       "把偏置整体加 5 后输出的最大变化": float(f"{float((y1 - y2).abs().max()):.1g}"),
                       "BatchNorm2d(8) 的可训练参数个数": n_params(b2), "BatchNorm2d(8) 的状态量（running_mean、running_var、计数）个数": sum(v.numel() for v in b2.buffers())}

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), layout="constrained")
    ax = axes[0]
    ax.plot(batches, dev, "o-", color=S.BLUE, lw=1.8, ms=4.5, label="训练模式和推理模式输出的相对差")
    ax.plot(batches, [dev[3] * math.sqrt(8 / b) for b in batches], "--", color=S.MUTED, lw=1.2, label="参考线：和根号批大小成反比")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(batches, [str(b) for b in batches])
    ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1], ["0.05", "0.1", "0.2", "0.5", "1"])
    ax.minorticks_off()
    ax.set_xlabel("批大小")
    ax.set_ylabel("相对差")
    ax.set_title("批越小，训练模式的输出离推理模式越远")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax)
    ax = axes[1]
    ax.plot(ks, ratio, "o-", color=S.BLUE, lw=1.8, ms=4.5, label="推理模式输出的均方根 / 训练模式的")
    ax.plot(ks, [1 / math.sqrt(p) for p in pred], "--", color=S.ORANGE, lw=1.3, label="按 running_var 的指数平均推算")
    ax.axhline(1, color=S.MUTED, lw=1)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200], ["1", "2", "5", "10", "20", "50", "100", "200"])
    ax.minorticks_off()
    ax.set_xlabel("训练模式下前向的次数")
    ax.set_ylabel("比值")
    ax.set_title("running 统计要几十步才跟上")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax)
    ax = axes[2]
    ax.plot(marks, 100 * acc_t.mean(0), "o-", color=S.ORANGE, lw=1.8, ms=4.5, label="训练模式")
    ax.plot(marks, 100 * acc_e.mean(0), "o-", color=S.BLUE, lw=1.8, ms=4.5, label="推理模式")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("这 16 张图上的准确率（%）")
    ax.set_title("过拟合 16 张图：推理模式的准确率来得晚")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    S.clean(ax)
    S.save(fig, CH, "fig05_BN两种模式")


# ---------------------------------------------------------------- 图 6：普通网络和残差网络
class PlainBlock(ResBlock):
    """和作业里的残差块层数、参数完全相同，只是不加捷径。"""

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(y)))


class NoBNBlock(nn.Module):
    def __init__(self, c, skip):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.skip = skip

    def forward(self, x):
        y = self.conv2(torch.relu(self.conv1(x)))
        return torch.relu(y + x) if self.skip else torch.relu(y)


KINDS = ["普通，无 BN，默认初始化", "普通，无 BN，He 初始化", "残差，无 BN，默认初始化", "普通，带 BN", "残差，带 BN"]


def deep_net(kind, blocks, c=8):
    if kind == "普通，带 BN":
        return TinyResNet(blocks, c, PlainBlock)
    if kind == "残差，带 BN":
        return TinyResNet(blocks, c, ResBlock)
    net = TinyResNet(blocks, c, lambda ch: NoBNBlock(ch, kind.startswith("残差")))
    net.bn = nn.Identity()
    if "He" in kind:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    return net


def train_deep(kind, blocks, seed, steps=200):
    torch.manual_seed(seed)
    net = deep_net(kind, blocks)
    g = gen(seed + 7)
    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    losses = []
    for _ in range(steps):
        x, y = shapes(g, 32, n=12, noise=0.3)
        loss = F.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    xt, yt = shapes(gen(999), 512, n=12, noise=0.3)
    set_momentum(net, 0.0)
    with torch.no_grad():
        acc_train_mode = float((net(xt).argmax(1) == yt).float().mean())
        acc_eval = float((net.eval()(xt).argmax(1) == yt).float().mean())
    return np.array(losses), acc_eval, acc_train_mode


def fig_residual():
    depths = [1, 2, 4, 8, 16, 32]
    seeds = (0, 1, 2)
    grad_first = {k: [] for k in KINDS}
    grad_ratio = {k: [] for k in KINDS}
    for kind in KINDS:
        for blocks in depths:
            first, last = [], []
            for seed in seeds:
                torch.manual_seed(seed)
                net = deep_net(kind, blocks)
                x, y = shapes(gen(100 + seed), 64, n=12, noise=0.3)
                F.cross_entropy(net(x), y).backward()
                first.append(float(net.blocks[0].conv1.weight.grad.norm()))
                last.append(float(net.blocks[-1].conv2.weight.grad.norm()))
            gm = lambda v: float(np.exp(np.mean(np.log(v)))) if min(v) > 0 else 0.0  # noqa: E731
            grad_first[kind].append(gm(first))
            grad_ratio[kind].append(gm(first) / gm(last) if gm(last) > 0 else None)
    fmt = lambda v: None if v is None else float(f"{v:.2g}")  # noqa: E731
    NUM["初始化时的梯度"] = {
        "说明": "网络是一层 3x3 卷积、若干个块（每块两层 3x3 卷积，8 通道）、全局平均池化、全连接；输入是 64 张 12x12 的四类图案图，损失是交叉熵；只做一次反向传播，不训练；"
              "记第一个块第一层卷积的权重梯度范数，以及它和最后一个块第二层卷积的权重梯度范数之比；3 组随机初始化取几何平均。默认初始化指 PyTorch 的 Conv2d 默认值；"
              "带 BN 的两种网络层和参数完全相同，区别只有块的输出加不加输入",
        "块数": depths,
        "总卷积层数": [2 * d + 1 for d in depths],
        "第一个块的梯度范数": {k: [fmt(v) for v in grad_first[k]] for k in KINDS},
        "第一个块的梯度范数除以最后一个块的": {k: [fmt(v) for v in grad_ratio[k]] for k in KINDS},
    }

    train_depths = [1, 4, 8, 16]
    with single_thread():
        runs = {kind: {d: [train_deep(kind, d, s) for s in seeds] for d in train_depths} for kind in ("普通，带 BN", "残差，带 BN")}
        nobn = {kind: [train_deep(kind, 1, s) for s in seeds] for kind in ("普通，无 BN，默认初始化", "普通，无 BN，He 初始化")}

    def summary(rs):
        final = [float(l[-20:].mean()) for l, _, _ in rs]
        return {"最后 20 步的训练损失：各种子": [r(v, 3) for v in final], "最后 20 步的训练损失：平均": r(np.mean(final), 3),
                "推理模式的测试准确率（%）：各种子": [r(100 * a, 1) for _, a, _ in rs], "推理模式的测试准确率（%）：平均": r(100 * np.mean([a for _, a, _ in rs]), 1),
                "训练模式的测试准确率（%）：各种子": [r(100 * a, 1) for _, _, a in rs], "训练模式的测试准确率（%）：平均": r(100 * np.mean([a for _, _, a in rs]), 1)}

    NUM["小任务上的训练"] = {
        "说明": "12x12 的四类图案图（噪声 0.3，位置随机），每步新生成 32 张；SGD 学习率 0.05、动量 0.9，训练 200 步，3 个种子；测试用另外 512 张；"
              "训练损失取最后 20 步的平均，随机猜的损失是 ln4",
        "随机猜的损失 ln4": r(math.log(4), 3),
        "块数": train_depths,
        "普通，带 BN": {f"{d} 个块": summary(runs["普通，带 BN"][d]) for d in train_depths},
        "残差，带 BN": {f"{d} 个块": summary(runs["残差，带 BN"][d]) for d in train_depths},
        "1 个块、去掉 BN": {k: summary(v) for k, v in nobn.items()},
    }
    p16, r16 = NUM["小任务上的训练"]["普通，带 BN"]["16 个块"], NUM["小任务上的训练"]["残差，带 BN"]["16 个块"]
    gaps = [100 * (a_train - a_eval) for kind in runs for d in train_depths for _, a_eval, a_train in runs[kind][d]]
    NUM["小任务上的训练"]["带 BN 的训练次数"] = len(gaps)
    NUM["小任务上的训练"]["其中推理模式的测试准确率比训练模式低 10 个百分点以上的次数"] = int(sum(g > 10 for g in gaps))
    NUM["小任务上的训练"]["推理模式比训练模式低得最多的一次（百分点）"] = r(max(gaps), 1)
    NUM["小任务上的训练"]["推理模式减训练模式的差的中位数（百分点）"] = r(-float(np.median(gaps)), 1)
    NUM["小任务上的训练"]["16 个块时普通网络的训练损失是残差网络的多少倍"] = r(p16["最后 20 步的训练损失：平均"] / r16["最后 20 步的训练损失：平均"], 1)

    # 残差块最外层的 ReLU 关掉了多少位置
    off = []
    for seed in range(5):
        torch.manual_seed(seed)
        blk = ResBlock(8).train()
        with torch.no_grad():
            off.append(float((blk(torch.relu(torch.randn(64, 8, 12, 12))) == 0).float().mean()))
    NUM["残差块外层的 ReLU"] = {"说明": "作业里的残差块（随机初始化，训练模式），输入是 64x8x12x12 的非负随机数（标准正态过一次 ReLU）；统计输出里等于 0 的位置，这些位置上相加之后的 ReLU 没有打开，捷径上的梯度也过不去；5 组取平均",
                         "输出为 0 的位置占比（%）": r(100 * np.mean(off), 1)}

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), layout="constrained")
    ax = axes[0]
    style = {"普通，无 BN，默认初始化": (S.MUTED, "s--", "普通，无 BN"), "残差，无 BN，默认初始化": (S.GREEN, "s--", "残差，无 BN"),
             "普通，带 BN": (S.RED, "o-", "普通，带 BN"), "残差，带 BN": (S.BLUE, "o-", "残差，带 BN")}
    for kind, (color, mk, label) in style.items():
        xs = [d for d, v in zip(depths, grad_first[kind]) if v > 0]
        ys = [v for v in grad_first[kind] if v > 0]
        ax.plot(xs, ys, mk, color=color, lw=1.6, ms=4.5, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(depths, [str(d) for d in depths])
    ax.set_yticks([1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1, 1e3], ["1e-15", "1e-12", "1e-9", "1e-6", "0.001", "1", "1000"])
    ax.minorticks_off()
    ax.set_xlabel("块数（每块两层卷积）")
    ax.set_ylabel("第一个块的梯度范数")
    ax.set_title("普通网络：无 BN 梯度消失，带 BN 梯度变大")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9, columnspacing=1.0)
    S.clean(ax)

    ax = axes[1]
    smooth = lambda v: np.convolve(v, np.ones(10) / 10, mode="valid")  # noqa: E731
    for kind, color in (("普通，带 BN", S.RED), ("残差，带 BN", S.BLUE)):
        for i, (losses, _, _) in enumerate(runs[kind][16]):
            ax.plot(np.arange(10, 201), smooth(losses), color=color, lw=1.3, alpha=0.85, label=kind if i == 0 else None)
    ax.axhline(math.log(4), color=S.MUTED, lw=1, ls="--")
    ax.text(198, math.log(4) * 1.07, "随机猜 ln4", ha="right", fontsize=9, color=S.MUTED)
    ax.set_yscale("log")
    ax.set_yticks([0.03, 0.1, 0.3, 1], ["0.03", "0.1", "0.3", "1"])
    ax.minorticks_off()
    ax.set_xlabel("训练步数")
    ax.set_ylabel("训练损失（10 步滑动平均）")
    ax.set_title("16 个块：残差网络的损失降得更低")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9, columnspacing=1.0)
    S.clean(ax)

    ax = axes[2]
    for kind, color, off_x in (("普通，带 BN", S.RED, -0.03), ("残差，带 BN", S.BLUE, 0.03)):
        means = []
        for d in train_depths:
            final = [float(l[-20:].mean()) for l, _, _ in runs[kind][d]]
            ax.plot([d * (1 + off_x)] * len(final), final, "o", color=color, ms=3.5, alpha=0.55)
            means.append(np.mean(final))
        ax.plot(train_depths, means, "-", color=color, lw=1.8, label=kind)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(train_depths, [str(d) for d in train_depths])
    ax.set_yticks([0.03, 0.1, 0.3, 1], ["0.03", "0.1", "0.3", "1"])
    ax.minorticks_off()
    ax.set_ylim(0.01, 1.5)
    ax.set_xlabel("块数")
    ax.set_ylabel("最后 20 步的训练损失")
    ax.set_title("普通网络越深损失越高，残差网络没有")
    ax.legend(loc="upper left")
    S.clean(ax)
    S.save(fig, CH, "fig06_普通与残差")


# ---------------------------------------------------------------- 表：第一层卷积核和 Sobel 的相似度
def table_first_layer():
    import cv2
    import torchvision

    sob3 = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kx, ky = cv2.getDerivKernels(1, 0, 7)
    sob7 = torch.tensor(np.outer(ky, kx), dtype=torch.float32)

    def cos_sobel(gray, sob):
        g0 = gray - gray.mean((1, 2), keepdim=True)
        return torch.stack([(g0 * k).sum((1, 2)).abs() / (g0.flatten(1).norm(dim=1) * k.norm() + 1e-12) for k in (sob, sob.T)]).amax(0)

    rnd = torch.randn(64 * 2000, 3, 3, generator=gen(0))
    c3 = cos_sobel(rnd, sob3)

    def describe(w):
        gray = w.mean(1)
        cos = cos_sobel(gray, sob7)
        lum = 3 * gray.pow(2).sum((1, 2)) / w.pow(2).sum((1, 2, 3)).clamp_min(1e-30)
        norm = w.flatten(1).norm(dim=1)
        return {"和 7x7 Sobel 的余弦绝对值：中位数": r(cos.median(), 2), "最大": r(cos.max(), 2), "超过 0.5 的核数": int((cos > 0.5).sum()), "超过 0.3 的核数": int((cos > 0.3).sum()),
                "亮度分量占能量的比例：中位数": r(lum.median(), 2), "亮度分量占 90% 以上的核数（接近灰度核）": int((lum > 0.9).sum()), "亮度分量不到 10% 的核数（接近纯颜色对比）": int((lum < 0.1).sum()),
                "范数不到最大核 5% 的核数": int((norm < 0.05 * norm.max()).sum()), "最小范数除以最大范数": float(f"{float(norm.min() / norm.max()):.1g}")}

    pre = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).conv1.weight.detach()
    rand = {}
    for seed in range(3):
        torch.manual_seed(seed)
        rand[f"种子 {seed}"] = describe(nn.Conv2d(3, 64, 7, bias=False).weight.detach())
    NUM["第一层卷积核"] = {
        "说明": "余弦是把核减去均值后和 Sobel x、Sobel y 分别算余弦绝对值，取较大的一个。3x3 的基线用 128000 个标准正态随机核；"
              "7x7 的核取 torchvision ResNet-18 第一层的 64 个 7x7x3 核（ImageNet 预训练权重 IMAGENET1K_V1），先对 RGB 三个通道取平均再和 7x7 的 Sobel 比；"
              "亮度分量指三个通道的平均，它的能量占比在三个通道完全相同时是 1，随机核的期望是三分之一",
        "3x3 随机核和 Sobel 的余弦绝对值": {"中位数": r(c3.median(), 2), "95% 分位": r(c3.quantile(0.95), 2), "超过 0.7 的比例（%）": r(100 * (c3 > 0.7).float().mean(), 1),
                                 "64 个随机核里最大的那个：中位数": r(c3.view(-1, 64).amax(1).median(), 2)},
        "ResNet-18 预训练的第一层（64 个核）": describe(pre),
        "同样形状的随机初始化（64 个核）": rand,
    }


# ---------------------------------------------------------------- 表：参数量、激活元素个数和批大小；二次 softmax
def table_memory():
    import torchvision

    torch.manual_seed(0)
    net = torchvision.models.resnet18(weights=None).eval()

    def act_elems(batch, size):
        total = 0

        def hook(_m, _i, out):
            nonlocal total
            total += out.numel()

        kinds = (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Linear)
        hooks = [m.register_forward_hook(hook) for m in net.modules() if isinstance(m, kinds)]
        with torch.no_grad():
            net(torch.zeros(batch, 3, size, size))
        for h in hooks:
            h.remove()
        return total

    mib = lambda n: r(n * 4 / 2 ** 20, 1)  # noqa: E731
    params = n_params(net)
    a1 = act_elems(1, 224)
    NUM["参数量与激活"] = {
        "说明": "torchvision 的 ResNet-18 结构（随机初始化，只数个数，不涉及精度）；激活元素个数是一次前向里所有卷积、BN、池化、全连接输出的元素个数之和（ReLU 是原地操作，不另算）；"
              "MiB 按每个元素 4 字节（float32）折算。这是数出来的张量大小，不是在任何显卡上量到的显存：训练时还要加上梯度、优化器状态和框架自己的开销",
        "参数量": params, "参数占用（MiB）": mib(params),
        "224x224、批 1 的乘加次数（十亿）": r(count_macs(net, torch.zeros(1, 3, 224, 224)) / 1e9, 2),
        "224x224 时按批大小": {f"批 {b}": {"激活元素个数（百万）": r(act_elems(b, 224) / 1e6, 1) if b <= 2 else r(b * a1 / 1e6, 1), "激活占用（MiB）": mib(b * a1)} for b in (1, 2, 8, 32, 64)},
        "批 2 的激活元素个数除以批 1 的": r(act_elems(2, 224) / a1, 3),
        "批 1 时按输入边长": {f"{s}x{s}": {"激活元素个数（百万）": r(act_elems(1, s) / 1e6, 1), "激活占用（MiB）": mib(act_elems(1, s))} for s in (112, 224, 448)},
        "448 的激活元素个数除以 224 的": r(act_elems(1, 448) / a1, 2),
        "激活占用超过参数占用的最小批大小": next(b for b in range(1, 100) if b * a1 > params),
    }

    # 学习计划里题 DL-4 推荐的起点结构：参数在哪，计算在哪
    feats = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
    head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 10)).eval()
    x = torch.zeros(1, 3, 32, 32)
    conv_macs = count_macs(feats.eval(), x)
    fc_macs = count_macs(head, torch.zeros(1, 128, 4, 4))
    total = n_params(feats) + n_params(head)
    NUM["DL-4 起点结构"] = {
        "说明": "视觉算法学习计划.md 里题 DL-4 推荐的 SimpleCNN：三个 Conv-BN-ReLU-MaxPool 块（32、64、128 通道），展平后接 2048-256-10 的全连接；输入 3x32x32；只数个数，没有训练",
        "总参数量": total, "第一个全连接层的参数量": n_params(head[1]), "第一个全连接层占总参数的比例（%）": r(100 * n_params(head[1]) / total, 1),
        "卷积的乘加次数（百万）": r(conv_macs / 1e6, 2), "全连接的乘加次数（百万）": r(fc_macs / 1e6, 2), "卷积占乘加次数的比例（%）": r(100 * conv_macs / (conv_macs + fc_macs), 1),
        "把展平加两层全连接换成全局平均池化加一层 128 到 10 的全连接后的总参数量": n_params(feats) + 128 * 10 + 10,
    }

    k = 10
    one_hot = torch.zeros(1, k)
    one_hot[0, 0] = 1.0
    logits = torch.tensor([[8.0] + [0.0] * (k - 1)])
    NUM["交叉熵前多做一次 softmax"] = {
        "说明": "10 类。CrossEntropyLoss 内部已经做了 log-softmax；如果先手动做一次 softmax 再送进去，网络最理想的输出是一个 one-hot 概率向量，这时损失仍然不是 0",
        "损失的下限 ln(1 + 9/e)": r(F.cross_entropy(one_hot, torch.tensor([0])), 3),
        "均匀输出时的损失 ln10": r(math.log(k), 3),
        "例：logits 是 [8, 0, ..., 0]、标签是第 0 类，正确用法的损失": r(F.cross_entropy(logits, torch.tensor([0])), 4),
        "同一个例子多做一次 softmax 的损失": r(F.cross_entropy(F.softmax(logits, 1), torch.tensor([0])), 3),
    }


# ---------------------------------------------------------------- 表：几种经典结构的参数量和乘加次数
def table_architectures():
    import torchvision.models as M

    x = torch.zeros(1, 3, 224, 224)
    rows = {}
    for name, make in (("AlexNet", M.alexnet), ("VGG-16", M.vgg16), ("ResNet-18", M.resnet18), ("ResNet-50", M.resnet50), ("MobileNetV2", M.mobilenet_v2), ("EfficientNet-B0", M.efficientnet_b0)):
        torch.manual_seed(0)
        net = make(weights=None).eval()
        total = n_params(net)
        fc = sum(n_params(m) for m in net.modules() if isinstance(m, nn.Linear))
        rows[name] = {"参数量（百万）": r(total / 1e6, 1), "全连接层占参数的比例（%）": r(100 * fc / total, 1), "乘加次数（十亿）": r(count_macs(net, x) / 1e9, 2)}
        del net
    NUM["经典结构"] = {
        "说明": "torchvision 里的模型定义（weights=None，随机初始化，只数个数，不涉及精度），输入 3x224x224、批 1；乘加次数只算卷积和全连接。LeNet-5 不在 torchvision 里，没有统计",
        "各结构": rows,
        "AlexNet 的参数量是 ResNet-50 的多少倍": r(rows["AlexNet"]["参数量（百万）"] / rows["ResNet-50"]["参数量（百万）"], 1),
        "ResNet-50 的乘加次数是 AlexNet 的多少倍": r(rows["ResNet-50"]["乘加次数（十亿）"] / rows["AlexNet"]["乘加次数（十亿）"], 1),
        "VGG-16 的乘加次数是 ResNet-18 的多少倍": r(rows["VGG-16"]["乘加次数（十亿）"] / rows["ResNet-18"]["乘加次数（十亿）"], 1),
    }


# ---------------------------------------------------------------- 表：几种错误写法、验收脚本能不能查出来、自己怎么查
def table_selfcheck():
    cases = [(32, 3, 1, 1, 1), (31, 3, 2, 1, 1), (15, 3, 1, 2, 2)]
    num = lambda size, k, s, p, d: size + 2 * p - d * (k - 1) - 1  # noqa: E731
    variants = {"向上取整": lambda *a: math.ceil(num(*a) / a[2]) + 1, "四舍五入": lambda *a: round(num(*a) / a[2]) + 1, "向零取整": lambda *a: int(num(*a) / a[2]) + 1}
    same = {name: all(f(size, k, s, p, d) == conv_out(size, k, s, p, d) for size, k, s, p, d in cases) for name, f in variants.items()}
    extra = {"输入 32、核 3、步长 2、无填充": {"参考实现": conv_out(32, 3, 2, 0), "向上取整的写法": variants["向上取整"](32, 3, 2, 0, 1)},
             "输入 32、核 3、步长 3、无填充": {"参考实现": conv_out(32, 3, 3, 0), "四舍五入的写法": variants["四舍五入"](32, 3, 3, 0, 1)}}
    try:
        conv_out(2, 3, 2, 0)
        reject = "没有拒绝"
    except ValueError:
        reject = "抛出 ValueError"

    class NoSkip(ResBlock):
        def forward(self, x):
            return self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x)))))

    def passes(block_cls):
        torch.manual_seed(9)
        block = block_cls(4).eval()
        x = torch.randn(2, 4, 11, 13, requires_grad=True)
        y = block(x)
        y.mean().backward()
        return bool(y.shape == x.shape and torch.isfinite(y).all() and x.grad is not None and torch.isfinite(x.grad).all())

    def zero_gamma(block_cls):
        torch.manual_seed(9)
        block = block_cls(4).eval()
        nn.init.zeros_(block.bn2.weight)
        x = torch.randn(2, 4, 11, 13)
        with torch.no_grad():
            y = block(x)
        return float((y - torch.relu(x)).abs().max()), int((y < 0).sum())

    ref_diff, ref_neg = zero_gamma(ResBlock)
    bad_diff, _ = zero_gamma(NoSkip)
    torch.manual_seed(1)
    y = NoSkip(4).eval()(torch.randn(2, 4, 11, 13))
    torch.manual_seed(0)
    w = nn.Conv2d(64, 64, 3, bias=False).weight
    numer = "(size + 2 * padding - dilation * (kernel - 1) - 1)"
    wrong = {
        "参考实现": variant(),
        "尺寸：向上取整": variant((f"{numer} // stride + 1", f"-(-{numer} // stride) + 1")),
        "尺寸：四舍五入": variant((f"{numer} // stride + 1", f"round({numer} / stride) + 1")),
        "尺寸：向零取整，写成 int(x / stride) + 1": variant((f"{numer} // stride + 1", f"int({numer} / stride) + 1")),
        "尺寸：不拒绝非正的输出": variant(('    if out <= 0:\n        raise ValueError("输出尺寸必须为正")\n', "")),
        "尺寸：公式里漏了膨胀": variant(("dilation * (kernel - 1)", "(kernel - 1)")),
        "残差块：去掉捷径": variant(("return torch.relu(y + x)", "return torch.relu(y)")),
        "残差块：去掉最后的 ReLU": variant(("return torch.relu(y + x)", "return y + x")),
        "残差块：去掉捷径和最后的 ReLU": variant(("return torch.relu(y + x)", "return y")),
        "残差块：bn1 之后没有 ReLU": variant(("y = torch.relu(self.bn1(self.conv1(x)))", "y = self.bn1(self.conv1(x))")),
        "残差块：相加之前多了一次 ReLU": variant(("y = self.bn2(self.conv2(y))", "y = torch.relu(self.bn2(self.conv2(y)))")),
    }
    NUM["验收与自检"] = {
        "说明": "验收脚本 tests/check_progress.py 最早只查三个都能整除的尺寸算例，以及残差块输出的 shape、数值有限、梯度能传回输入，后来补了检查。这里记三件事：几种错误写法在最早那几项检查下的表现；把参考实现改成错误写法之后实际运行现在的验收脚本的结果；一个不依赖验收脚本的自检",
        "最早的三个尺寸算例下和参考实现结果相同": same,
        "能把取整方式区分开的算例": extra,
        "参考实现对输入 2、核 3、步长 2、无填充": reject,
        "去掉捷径和最后的 ReLU 的块能否通过最早的 shape 与梯度检查": passes(NoSkip),
        "现在的验收脚本对各种写法的结果（每种只改一处，用 COURSE_IMPL_DIR 实际运行）": {name: acceptance(src) for name, src in wrong.items()},
        "去掉捷径和最后的 ReLU 的块：输出里负数的比例（%）": r(100 * float((y < 0).float().mean()), 1),
        "自检：把第二个 BN 的 γ 置 0（推理模式）后，输出和 ReLU(x) 的最大差": {"参考实现": ref_diff, "去掉捷径的块": r(bad_diff, 2)},
        "自检：参考实现的输出里负数的个数": ref_neg,
        "Conv2d 默认初始化的权重标准差是 He 初始化的多少倍": r(float(w.std()) / math.sqrt(2 / (64 * 9)), 3),
        "理论值 1/sqrt(6)": r(1 / math.sqrt(6), 3),
    }


# ---------------------------------------------------------------- 表：演示页“按顺序试”各步应显示的数
def calc_layers(size, c_in, layers):
    """和 demos/卷积尺寸与感受野.html 里的 compute 同样的算法。layers: (类型, 核, 步长, 填充, 膨胀, 输出通道, 分组)。"""
    rows, rr, j = [], 1, 1
    for kind, k, s, p, d, c_out, groups in layers:
        out = conv_out(size, k, s, p, d)
        rem = (size + 2 * p - d * (k - 1) - 1) % s
        rr += (k - 1) * d * j
        j *= s
        if kind == "conv":
            params = c_out * (c_in // groups) * k * k
            macs = out * out * params
        else:
            c_out, params, macs = c_in, 0, 0
        rows.append({"输出边长": out, "输出通道": c_out, "感受野": rr, "间隔": j, "参数量": params, "乘加次数": macs,
                     "没用到的行数（含填充）": rem, "其中真实输入的行数": max(0, rem - p)})
        size, c_in = out, c_out
    return rows


def table_demo():
    conv = lambda k, s, p, c, g=1, d=1: ("conv", k, s, p, d, c, g)  # noqa: E731
    presets = {
        "第 1 步：原讲义的例子（输入 32，3 通道）": (32, 3, [conv(3, 1, 0, 16), ("max", 2, 2, 0, 1, 0, 1), conv(3, 1, 0, 32)]),
        "第 2 步：两层 3x3（输入 56，64 通道）": (56, 64, [conv(3, 1, 1, 64), conv(3, 1, 1, 64)]),
        "第 2 步：一层 5x5（输入 56，64 通道）": (56, 64, [conv(5, 1, 2, 64)]),
        "第 3 步：普通 3x3（输入 56，64 通道）": (56, 64, [conv(3, 1, 1, 64)]),
        "第 3 步：深度可分离（输入 56，64 通道）": (56, 64, [conv(3, 1, 1, 64, 64), conv(1, 1, 0, 64)]),
        "第 4 步：3x3 步长 2 无填充（输入 32）": (32, 3, [conv(3, 2, 0, 16)]),
        "第 4 步：3x3 步长 2 填充 1（输入 32）": (32, 3, [conv(3, 2, 1, 16)]),
        "第 5 步：ResNet-18 主干（输入 224）": (224, 3, [conv(7, 2, 3, 64), ("max", 3, 2, 1, 1, 0, 1)] + [conv(3, 1, 1, 64)] * 4 + [conv(3, 2, 1, 128)] + [conv(3, 1, 1, 128)] * 3
                                      + [conv(3, 2, 1, 256)] + [conv(3, 1, 1, 256)] * 3 + [conv(3, 2, 1, 512)] + [conv(3, 1, 1, 512)] * 3),
    }
    out = {"说明": "演示页 demos/卷积尺寸与感受野.html 载入各步预设后应显示的数，用同样的公式在这里算一遍，供 node 核对；卷积都不带偏置，池化不计参数和乘加次数"}
    for name, (size, c_in, layers) in presets.items():
        rows = calc_layers(size, c_in, layers)
        out[name] = {"逐层输出边长": [x["输出边长"] for x in rows], "逐层感受野": [x["感受野"] for x in rows], "逐层间隔": [x["间隔"] for x in rows],
                     "总参数量": sum(x["参数量"] for x in rows), "总乘加次数（百万）": r(sum(x["乘加次数"] for x in rows) / 1e6, 2),
                     "最后一层没用到的行数（含填充）": rows[-1]["没用到的行数（含填充）"], "最后一层没用到的真实输入行数": rows[-1]["其中真实输入的行数"]}
    r18 = out["第 5 步：ResNet-18 主干（输入 224）"]
    r18["总乘加次数（十亿）"] = r(r18["总乘加次数（百万）"] / 1e3, 2)
    r18["总参数量（百万）"] = r(r18["总参数量"] / 1e6, 2)
    size, c_in, layers = presets["第 5 步：ResNet-18 主干（输入 224）"]
    rows = calc_layers(size, c_in, layers)
    tp, tm = sum(x["参数量"] for x in rows), sum(x["乘加次数"] for x in rows)
    r18["最后 4 层（layer4）占参数量的比例（%）"] = r(100 * sum(x["参数量"] for x in rows[-4:]) / tp, 1)
    r18["最后 4 层（layer4）占乘加次数的比例（%）"] = r(100 * sum(x["乘加次数"] for x in rows[-4:]) / tm, 1)
    r18["前 6 层（到 layer1 为止）占参数量的比例（%）"] = r(100 * sum(x["参数量"] for x in rows[:6]) / tp, 1)
    r18["前 6 层（到 layer1 为止）占乘加次数的比例（%）"] = r(100 * sum(x["乘加次数"] for x in rows[:6]) / tm, 1)
    r18["输入边长改成 64 和 32 时最后的输出边长"] = [calc_layers(s, c_in, layers)[-1]["输出边长"] for s in (64, 32)]
    size, c_in, layers = presets["第 1 步：原讲义的例子（输入 32，3 通道）"]
    out["第 1 步：把池化的步长改成 1 后的逐层感受野"] = [x["感受野"] for x in calc_layers(size, c_in, [layers[0], ("max", 2, 1, 0, 1, 0, 1), layers[2]])]
    g4 = calc_layers(56, 64, [conv(3, 1, 1, 64, 4), conv(1, 1, 0, 64)])
    out["第 3 步：深度可分离的第一层分组改成 4"] = {"总参数量": sum(x["参数量"] for x in g4), "总乘加次数（百万）": r(sum(x["乘加次数"] for x in g4) / 1e6, 2)}
    out["第 4 步：输入边长改成 31 时的输出边长（无填充，填充 1）"] = [conv_out(31, 3, 2, 0), conv_out(31, 3, 2, 1)]
    NUM["演示页预设"] = out


def main() -> None:
    S.setup()
    t0 = time.time()
    NUM["环境"] = {"说明": "全部实验在一台 Apple 芯片的 Mac 上用 CPU 跑，torch.set_num_threads(2)，没有用 GPU", "PyTorch 版本": torch.__version__.split("+")[0]}
    for fn in (fig_position, fig_cost, fig_equivariance, fig_rf, fig_bn, fig_residual, table_first_layer, table_memory, table_architectures, table_selfcheck, table_demo):
        t1 = time.time()
        fn()
        print(f"完成 {fn.__name__}（{time.time() - t1:.1f} 秒）", flush=True)
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False))
    print(f"总用时 {time.time() - t0:.1f} 秒", flush=True)


if __name__ == "__main__":
    main()
