"""生成第 31 章的互动实验 Notebook。

用法（在仓库根目录运行）：
    python scripts/part3_notebooks/ch31.py
修改实验内容时改本文件再重新生成，不要直接手改 .ipynb，避免两边不一致。
Notebook 只调用学员自己在作业文件里的实现，不包含参考答案；
设置环境变量 COURSE_IMPL_DIR 可以让 Notebook 改用另一份实现（维护者验证用）。

本文件自成一体：HELPER 和 notebook() 抄自 scripts/build_track2_notebooks.py，不 import 那个文件。
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = "31_部署与加速"
ASSIGNMENT = "31-1_量化与部署零件.py"
OUTPUT = "实验_部署与加速.ipynb"

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


SETUP = '''import copy, math, tempfile, time
import torch
import torch.nn.functional as F
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
print("设备：", device, "｜", torch.cuda.get_device_name(0) if device.type == "cuda" else "没有检测到 CUDA，用 CPU")
print("说明：量化的几个实验要在 PyTorch 张量和你写的 NumPy 函数之间来回转，小模型一律放在 CPU 上；只有实验 6 量延迟时把模型放到上面这个设备上。")

def implemented(fn, *args, **kwargs):
    """试着调用一次作业函数，返回它是否已经实现。"""
    try:
        fn(*args, **kwargs)
        return True
    except NotImplementedError as exc:
        print(f"先完成作业里的 {fn.__name__}：{exc}")
        return False

SIZE, K = 32, 5
CLASSES = ["良品", "亮点", "暗点", "横划痕", "竖划痕"]
AMP, WEAK, NOISE = (0.06, 0.30), 0.10, 0.02          # 缺陷幅度的范围；幅度小于 WEAK 的叫弱缺陷；像素噪声
YY, XX = torch.meshgrid(torch.arange(SIZE, dtype=torch.float32), torch.arange(SIZE, dtype=torch.float32), indexing="ij")

def make_data(n, g):
    """n 张 32x32 的 8 位灰度图：竖直线条的背景，一半是良品，另一半带一个缺陷。返回 (图像, 类别, 缺陷幅度)。"""
    ph = torch.rand(n, generator=g) * 8
    bg = 0.5 + 0.06 * torch.sign(torch.sin(2 * math.pi * (XX[None] + ph[:, None, None]) / 8))
    bright = 0.9 + 0.2 * torch.rand(n, 1, 1, generator=g)
    y = torch.where(torch.rand(n, generator=g) < 0.5, 0, torch.randint(1, K, (n,), generator=g))
    a = AMP[0] + (AMP[1] - AMP[0]) * torch.rand(n, generator=g)
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
    img = torch.round(img.clamp(0, 1) * 255) / 255                           # 存成 8 位
    return img[:, None], y, torch.where(y == 0, torch.zeros_like(a), a)

def cbr(i, o, stride=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU())

class Net(nn.Module):
    """4 个 Conv-BN-ReLU（第 2、3 个步长 2）、全局平均池化、全连接。"""
    def __init__(self, w=(8, 16, 32, 32)):
        super().__init__()
        self.features = nn.Sequential(cbr(1, w[0]), cbr(w[0], w[1], 2), cbr(w[1], w[2], 2), cbr(w[2], w[3]))
        self.fc = nn.Linear(w[3], K)
    def forward(self, x):
        return self.fc(self.features((x - 0.5) / 0.25).mean((2, 3)))

def train(net, steps=400, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(1000 + seed)
    opt = torch.optim.AdamW(net.parameters(), lr=4e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=steps, pct_start=0.15)
    net.train()
    for step in range(steps):
        x, y, _ = make_data(64, g)
        loss = F.cross_entropy(net(x), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"  第 {step:4d} 步，损失 {loss.item():.3f}")
    return net.eval()                                                        # 记得切到推理模式（第 14 章 14.8 节）

@torch.no_grad()
def predict(model, x):
    return torch.cat([model(x[i:i + 500]) for i in range(0, len(x), 500)])

g = torch.Generator().manual_seed(31)
demo, yd, _ = make_data(400, g)
fig, ax = plt.subplots(1, K, figsize=(10, 2.4))
for k in range(K):
    ax[k].imshow(demo[yd == k][0, 0], cmap="gray", vmin=0.2, vmax=0.8); ax[k].set_title(CLASSES[k]); ax[k].set_xticks([]); ax[k].set_yticks([])
plt.tight_layout(); plt.show()'''

BASICS = '''lo, hi = -1.0, 3.0
p3 = attempt(mod.quant_params, lo, hi, 3)
if p3 is not None:
    for bits, sym in ((3, False), (8, False), (8, True)):
        s, zp, qmin, qmax = mod.quant_params(lo, hi, bits, sym)
        print(f"{bits} 位，{'对称' if sym else '非对称'}：scale = {s:.5f}，zero_point = {zp}，整数范围 {qmin} 到 {qmax}，能表示的实数范围 {(qmin - zp) * s:.3f} 到 {(qmax - zp) * s:.3f}")
xs = np.linspace(-2.0, 4.0, 1201)
out = attempt(mod.quantize_dequantize, xs, *p3) if p3 is not None else None
if out is not None:
    q, xh = out
    rng = np.random.default_rng(0)
    data = np.abs(rng.normal(size=100000))                    # 非负的数据，像 ReLU 之后的激活值
    bits_list = [2, 3, 4, 5, 6, 7, 8]
    err = {"非对称": [], "对称": []}
    for bits in bits_list:
        for name, sym in (("非对称", False), ("对称", True)):
            p = mod.quant_params(float(data.min()), float(data.max()), bits, sym)
            e = mod.quantize_dequantize(data, *p)[1] - data
            err[name].append(np.sqrt((e ** 2).mean()) / np.sqrt((data ** 2).mean()))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(xs, xs, ":", color="gray", label="不量化"); ax[0].plot(xs, xh, label="量化再反量化（3 位）"); ax[0].plot(xs, xh - xs, label="误差")
    ax[0].axvline(lo, color="gray", ls="--", lw=0.8); ax[0].axvline(hi, color="gray", ls="--", lw=0.8); ax[0].set_xlabel("量化前的实数"); ax[0].legend()
    for name in err:
        ax[1].semilogy(bits_list, err[name], "o-", label=f"非负数据，{name}量化")
    ax[1].set_xlabel("位数"); ax[1].set_ylabel("相对误差（误差均方根 / 数据均方根）"); ax[1].legend()
    plt.tight_layout(); plt.show()
    print("对称是非对称的几倍：", [round(float(a / b), 2) for a, b in zip(err["对称"], err["非对称"])])'''

PER_CHANNEL = '''rng = np.random.default_rng(1)
O = 16
stds = 100.0 ** (-np.arange(O) / (O - 1))                       # 各输出通道的标准差从 1 等比降到 0.01
w = rng.normal(size=(O, 8, 3, 3)) * stds[:, None, None, None]
out = attempt(mod.quantize_weight_per_channel, w, 8)
p = attempt(mod.quant_params, float(w.min()), float(w.max()), 8, True)
wt = attempt(mod.quantize_dequantize, w, *p) if p is not None else None
if out is not None and wt is not None:
    q_c, s_c = out
    w_c = q_c * s_c[:, None, None, None]                        # 逐通道：每个输出通道各用各的缩放系数
    q_t, w_t = wt                                               # 逐张量：整个张量共用一个
    rel = lambda a: np.sqrt(((a - w) ** 2).reshape(O, -1).mean(1)) / np.sqrt((w ** 2).reshape(O, -1).mean(1))
    print("逐张量时，范围最小的那个通道用到了几个不同的整数：", len(np.unique(q_t[-1])), "｜逐通道时：", len(np.unique(q_c[-1])))
    plt.figure(figsize=(6, 3.6))
    plt.loglog(1 / stds, rel(w_t), "o-", label="逐张量"); plt.loglog(1 / stds, rel(w_c), "s-", label="逐通道")
    plt.xlabel("最大的通道范围是这个通道的几倍"); plt.ylabel("这个通道的相对误差"); plt.legend(); plt.tight_layout(); plt.show()'''

FOLD = '''t0 = time.perf_counter()
net = train(Net())
print(f"训练用时 {time.perf_counter() - t0:.1f} 秒")
x_te, y_te, a_te = make_data(2000, torch.Generator().manual_seed(7))
x_va, y_va, a_va = make_data(1000, torch.Generator().manual_seed(6))
x_cal, y_cal, _ = make_data(256, torch.Generator().manual_seed(8))
ref = predict(net, x_te)
print("浮点模型的 5 类分类精度：", round(float((ref.argmax(1) == y_te).float().mean()), 3))

class Folded(nn.Module):
    """BN 折叠进卷积之后的同一个网络。act_q[i] 不是 None 时，第 i 个 ReLU 之后用你写的 quantize_dequantize 量化一次。"""
    def __init__(self, net):
        super().__init__()
        convs = []
        for blk in net.features:
            conv, bn = blk[0], blk[1]
            w, b = mod.fold_bn_into_conv(conv.weight, conv.bias, bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps)
            c = nn.Conv2d(conv.in_channels, conv.out_channels, 3, conv.stride, 1, bias=True)
            c.weight.data.copy_(w); c.bias.data.copy_(b)
            convs.append(c)
        self.convs, self.fc = nn.ModuleList(convs), copy.deepcopy(net.fc)
        self.act_q = [None] * len(convs)
    def forward(self, x, taps=None):
        x = (x - 0.5) / 0.25
        for i, c in enumerate(self.convs):
            x = F.relu(c(x))
            if taps is not None:
                taps.append(x)
            if self.act_q[i] is not None:
                x = torch.from_numpy(mod.quantize_dequantize(x.numpy(), *self.act_q[i])[1]).float()
        return self.fc(x.mean((2, 3)))

folded = None
c0, b0 = net.features[0][0], net.features[0][1]
if implemented(mod.fold_bn_into_conv, c0.weight, c0.bias, b0.weight, b0.bias, b0.running_mean, b0.running_var, b0.eps):
    folded = Folded(net).eval()
    out = predict(folded, x_te)
    print("折叠前后输出的最大差：", f"{float((out - ref).abs().max()):.2e}", "｜预测类别一致的比例：", float((out.argmax(1) == ref.argmax(1)).float().mean()))
    print("参数张量个数：折叠前", len(list(net.parameters())), "，折叠后", len(list(folded.parameters())))
    r0 = c0.weight.detach().abs().flatten(1).max(1).values
    r1 = folded.convs[0].weight.detach().abs().flatten(1).max(1).values
    print(f"第 1 个卷积各通道权重范围的最大比值：折叠前 {float(r0.max() / r0.min()):.1f}，折叠后 {float(r1.max() / r1.min()):.1f}")'''

PTQ = '''def ptq(fnet, calib_x, bits, per_channel=True, method="minmax"):
    """训练后量化：权重量化再反量化后写回，激活值在每个 ReLU 之后量化。全部调用你写的函数。"""
    q = copy.deepcopy(fnet)
    for m in list(q.convs) + [q.fc]:
        w = m.weight.detach().numpy()
        if per_channel:
            qi, s = mod.quantize_weight_per_channel(w, bits)
            wq = qi * s.reshape((-1,) + (1,) * (w.ndim - 1))
        else:
            wq = mod.quantize_dequantize(w, *mod.quant_params(float(w.min()), float(w.max()), bits, symmetric=True))[1]
        m.weight.data.copy_(torch.from_numpy(wq).float())
    taps = []
    with torch.no_grad():
        fnet(calib_x, taps)
    for i, t in enumerate(taps):
        v = t.numpy().ravel()
        hi = float(v.max()) if method == "minmax" else float(np.percentile(v, 99.9))
        q.act_q[i] = mod.quant_params(0.0, hi, bits)                # ReLU 之后没有负数，用非对称量化
    return q

mod19 = None
try:
    mod19 = load_module(COURSE_ROOT / "19_异常检测与深度缺陷检测", "19-1_PatchCore核心.py")
except Exception as exc:
    print("没有找到第 19 章的作业文件：", type(exc).__name__, exc)
use19 = mod19 is not None and implemented(mod19.threshold_at_escape, np.array([0.1, 0.2]), np.array([0.8, 0.9]), 0.5)

def defect_score(logits):
    return 1.0 - F.softmax(logits.double(), -1)[:, 0].numpy()        # 缺陷分数 = 1 - “良品”这一类的概率

ready = (folded is not None and implemented(mod.quant_params, 0.0, 1.0, 8) and implemented(mod.quantize_dequantize, np.zeros(3), 0.1, 0, 0, 255)
         and implemented(mod.quantize_weight_per_channel, np.ones((2, 2)), 8))
if ready:
    thr = None
    if use19:
        sv = defect_score(predict(folded, x_va))
        thr = mod19.threshold_at_escape(sv[y_va.numpy() == 0], sv[y_va.numpy() > 0], 0.02)[0]    # 浮点模型在验证集上定的阈值：漏检不超过 2%
        print(f"用你在第 19 章写的 threshold_at_escape 定阈值：{thr:.3f}。下面各行都沿用这个阈值。")
    else:
        print("第 19 章的 threshold_at_escape 还没完成，下面只显示分类精度。")
    bad, weak = y_te.numpy() > 0, (y_te.numpy() > 0) & (a_te.numpy() < WEAK)
    cal_good = x_cal[y_cal == 0]
    rows = [("浮点", folded), ("8 位，逐通道，最小最大值", ptq(folded, x_cal, 8)), ("8 位，校准图只有良品", ptq(folded, cal_good, 8)),
            ("5 位，逐通道，99.9% 分位", ptq(folded, x_cal, 5, method="percentile")), ("4 位，逐通道，99.9% 分位", ptq(folded, x_cal, 4, method="percentile")),
            ("4 位，逐张量，最小最大值", ptq(folded, x_cal, 4, per_channel=False))]
    print(f"{'做法':26s}  分类精度   过检率   弱缺陷漏检率   强缺陷漏检率")
    for name, m in rows:
        lg = predict(m, x_te)
        acc = float((lg.argmax(1) == y_te).float().mean())
        if thr is None:
            print(f"{name:26s}  {acc:.3f}")
            continue
        s = defect_score(lg)
        print(f"{name:26s}  {acc:.3f}     {float((s[~bad] >= thr).mean()):.3f}    {float((s[weak] < thr).mean()):.3f}          {float((s[bad & ~weak] < thr).mean()):.3f}")'''

HALF = '''f16 = np.float16
ones = np.ones(4096, dtype=np.float16)
print("float16 的最大值：", np.finfo(np.float16).max, "｜1 附近相邻两个数的间隔：", np.finfo(np.float16).eps)
print("2048 + 1 =", f16(2048) + f16(1), "（2048 附近的间隔是", np.spacing(f16(2048)), "）")
print("4096 个 1 逐个累加：", np.cumsum(ones)[-1], "｜np.sum（成对相加）：", ones.sum())
with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    z = np.array([12.0, 0.0], dtype=np.float16)
    e = np.exp(z)
    print("logits = [12, 0]：exp =", e, "｜先 exp 再除以和 =", e / e.sum())
    print("256 的平方：", f16(256) * f16(256), "｜1 / (0 + 1e-8) =", f16(1) / (f16(0) + f16(1e-8)), "（1e-8 存成 float16 是", f16(1e-8), "）")
t = torch.tensor([12.0, 0.0], dtype=torch.float16)
print("torch.softmax：", torch.softmax(t, 0).tolist(), "｜torch.log_softmax：", torch.log_softmax(t, 0).tolist())
half = copy.deepcopy(net).half()
with torch.no_grad():
    d = (half(x_te[:500].half()).float() - net(x_te[:500])).abs()
print(f"整个小模型换成 float16（在 CPU 上算）：logits 最大差 {float(d.max()):.3f}，中位数 {float(d.median()):.4f}")'''

LATENCY = '''TIME_SIZE, BATCHES = 256, (1, 2, 4, 8, 16)
model_t = copy.deepcopy(net).to(device).eval()

def bench(model, x, n=40, warm=5):
    """连续调用 warm + n 次，前 warm 次不计。显卡上每次调用之后要同步，否则量到的只是“把任务交出去”的时间。"""
    ts = []
    with torch.inference_mode():
        for i in range(warm + n):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= warm:
                ts.append((time.perf_counter() - t0) * 1e3)
    return np.array(ts)

lat = {b: bench(model_t, torch.rand(b, 1, TIME_SIZE, TIME_SIZE, device=device)) for b in BATCHES}
print(f"{'批大小':6s} 中位数(ms)  P99(ms)   折合每张(ms)  吞吐量(张/秒)")
for b, ts in lat.items():
    print(f"{b:6d}  {np.median(ts):8.2f}  {np.percentile(ts, 99):8.2f}  {np.median(ts) / b:10.2f}  {1000 * b / np.median(ts):12.0f}")
takt = 1.5 * float(np.percentile(lat[1], 99))                    # 假设的节拍：相邻两张图的间隔是单张 P99 的 1.5 倍
deadline = 8 * takt                                              # 假设每张图最多等 8 个节拍
res = attempt(mod.plan_batch, {b: ts.tolist() for b, ts in lat.items()}, takt, deadline)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(lat[1], ".", ms=4); ax[0].axhline(np.median(lat[1]), color="g", ls="--", label="中位数"); ax[0].axhline(np.percentile(lat[1], 99), color="r", ls="--", label="P99")
ax[0].set_xlabel("第几次调用（批大小 1）"); ax[0].set_ylabel("耗时（ms）"); ax[0].legend()
xs = np.arange(len(BATCHES))
ax[1].bar(xs - 0.2, [np.percentile(lat[b], 99) for b in BATCHES], 0.4, label="整批耗时的 P99")
ax[1].bar(xs + 0.2, [(b - 1) * takt + np.percentile(lat[b], 99) for b in BATCHES], 0.4, label="第一张图等多久")
ax[1].plot(xs - 0.2, [b * takt for b in BATCHES], "_k", ms=18, mew=2, label="批大小 x 节拍")
ax[1].axhline(deadline, color="r", ls="--", label="允许的最大延迟")
ax[1].set_xticks(xs, [str(b) for b in BATCHES]); ax[1].set_xlabel("批大小"); ax[1].set_ylabel("ms"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
if res is not None:
    print(f"节拍 {takt:.2f} ms，最多等 {deadline:.1f} ms：plan_batch 选出的最大批大小是 {res[0]}（0 表示一个都不可行）")'''

REAL = '''RUN_ONNX = False          # 装好 onnx 和 onnxruntime 之后改成 True 再运行这个单元
if not RUN_ONNX:
    print("跳过。把 RUN_ONNX 改成 True 再运行；要装的包、每个参数的含义、应该看到什么，见教程第 8 节。")
else:
    try:
        import onnxruntime as ort
        path = Path(tempfile.gettempdir()) / "ch31_small.onnx"
        cpu_net = copy.deepcopy(net).cpu().eval()
        x = torch.rand(4, 1, SIZE, SIZE)
        batch = torch.export.Dim("batch", min=1, max=64)
        torch.onnx.export(cpu_net, (x,), str(path), input_names=["input"], output_names=["logits"], opset_version=18,
                          dynamic_shapes=({0: batch},))      # 旧版 PyTorch 的写法不同，见教程 8.2 节
        print("可用的执行提供者：", ort.get_available_providers())
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for xb in (x, torch.rand(1, 1, SIZE, SIZE), torch.rand(9, 1, SIZE, SIZE)):       # 导出时的样例、另一个批大小、再一个批大小
            got = sess.run(None, {"input": xb.numpy()})[0]
            with torch.no_grad():
                want = cpu_net(xb).numpy()
            print("输入", tuple(xb.shape), "：和 PyTorch 输出的最大差", float(np.abs(got - want).max()))
    except Exception as exc:
        print("没有跑通：", type(exc).__name__, exc)'''

CELLS = [
    ("md", "## 准备：数据和小模型\n\n数据是代码生成的 32×32 灰度小图：竖直线条的背景，一半是良品，另一半带一个缺陷（亮点、暗点、横划痕、竖划痕）。缺陷幅度从很弱到很强都有，幅度小于 0.10 的叫弱缺陷。这个单元不涉及作业，直接运行。"),
    ("code", SETUP),
    ("md", "## 实验 1：缩放系数、零点、量化再反量化\n\n用你写的 `quant_params` 和 `quantize_dequantize`。左图是范围 [-1, 3]、3 位时的阶梯：范围内的误差不超过半格，范围外被截断。右图是非负数据在不同位数下的相对误差。\n\n先预测：非负的数据用对称量化，误差是非对称的几倍？对照教程图 2。"),
    ("code", BASICS),
    ("md", "## 实验 2：逐张量和逐通道\n\n一个 16×8×3×3 的权重，各输出通道的范围从 1 等比降到 0.01。分别用整个张量共用一个缩放系数（`quant_params` 加 `quantize_dequantize`）和每个输出通道各用各的（`quantize_weight_per_channel`）做 8 位量化。\n\n先预测：逐张量时，范围最小的那个通道还能用上几个不同的整数？"),
    ("code", PER_CHANNEL),
    ("md", "## 实验 3：训练小模型，把 BatchNorm 折叠进卷积\n\n先训练一个小卷积网络（400 步，CPU 上十几秒；不涉及作业）。然后用你写的 `fold_bn_into_conv` 把每个 BatchNorm 折叠进它前面的卷积，比较折叠前后的输出。\n\n先预测：折叠前后的输出最大差是 0，还是 1e-5 量级，还是 1e-2 量级？对照教程 2.2 节。"),
    ("code", FOLD),
    ("md", "## 实验 4：训练后量化\n\n在折叠后的模型上做训练后量化：权重量化再反量化后写回，激活值在每个 ReLU 之后量化。校准用 256 张图。比较几种设置，除了分类精度，还看“沿用浮点模型的阈值”时过检率和弱缺陷、强缺陷的漏检率怎么变（阈值用你在第 19 章写的 `threshold_at_escape` 定；第 19 章还没做的话只显示精度）。\n\n先预测：校准图里只放良品，分类精度和强缺陷漏检率会怎样变，弱缺陷呢？对照教程图 1 和第 1 节的表。训练步数比配图脚本少，数值不会和教程完全一样：分类精度那一列的变化应该同样明显，强缺陷那一列可能比教程小得多。"),
    ("code", PTQ),
    ("md", "## 实验 5：半精度的几个数值现象\n\n不涉及作业。float16 的最大值是 65504，1 附近相邻两个数的间隔大约是千分之一。看大数相加、指数溢出、很小的常数各会出什么事，对照教程第 4 节。"),
    ("code", HALF),
    ("md", "## 实验 6：延迟的统计和按节拍选批大小\n\n把小模型放到你的设备上（有 CUDA 就用显卡），输入换成 256×256，每个批大小连续调用几十次，记下每一次的耗时。然后假设一个节拍和一个允许的最大延迟，用你写的 `plan_batch` 选批大小。\n\n这里量到的是你自己机器上的数，而且没有限制线程数；教程里的是一台 Mac 的 CPU 上限制 2 个线程的数，两者不用对得上。要看的是：P99 比中位数高多少；批大小加大，折合每张是不是真的变快了；批越大，第一张图等得越久。"),
    ("code", LATENCY),
    ("md", "## 选做：导出 ONNX，用 ONNX Runtime 推理并和 PyTorch 对拍\n\n需要先装 `onnx` 和 `onnxruntime`（新版 PyTorch 的导出器还需要 `onnxscript`）。这个单元默认跳过，写课程的机器上没有运行过，接口以官方文档为准，说明见教程第 8 节。"),
    ("code", REAL),
]

OUTRO = ("## 破坏实验与记录\n\n"
         "- 实验 3 里把 `train(Net())` 之后的模型改回 `net.train()` 再折叠，看折叠前后的输出还对不对得上，想一想为什么。\n"
         "- 实验 4 里把校准图从 256 张减到 8 张，再把百分位从 99.9 改成 99，各看一次 4 位的结果。\n"
         "- 实验 6 里把预热去掉（`warm=0`），把每一次的耗时都打印出来看前几次；有显卡的话把同步那一行注释掉再量一次。\n"
         "- 把实验 4 的表、实验 6 的中位数和 P99 填进 `实验记录.md`。")


def main():
    nb = notebook(
        "第 31 章互动实验：量化、BN 折叠、半精度和延迟",
        "用你在作业里写的五个零件，在一个小卷积网络上把训练后量化做一遍，再量一次延迟、按节拍选批大小。全部数据由代码生成，不需要下载任何东西，也不需要 onnx 和 TensorRT；作业还没写完时，对应的单元只打印提示。",
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
