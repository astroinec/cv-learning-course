"""生成第 16–24 章的互动实验 Notebook。

用法：python3 scripts/build_track2_notebooks.py
修改实验内容时改本文件再重新生成，不要直接手改 .ipynb，避免两边不一致。
Notebook 只调用学员自己在作业文件里的实现，不包含参考答案；
设置环境变量 COURSE_IMPL_DIR 可以让 Notebook 改用另一份实现（维护者验证用）。
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

HELPER = '''%matplotlib inline
import importlib.util, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
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


BOOKS = {}

BOOKS["16_图像对齐与亚像素/实验_图像对齐.ipynb"] = notebook(
    "第 16 章互动实验：对齐、亚像素与周期图案",
    "所有估计都调用你在 `16-1_相位相关与亚像素.py` 里的实现。先预测结果，再运行。",
    "16_图像对齐与亚像素", "16-1_相位相关与亚像素.py",
    [
        ("code", '''def texture(shape=(96, 96), seed=0, width=0.08):
    rng = np.random.default_rng(seed)
    f = np.fft.fft2(rng.normal(size=shape))
    fy = np.fft.fftfreq(shape[0])[:, None]; fx = np.fft.fftfreq(shape[1])[None, :]
    return np.real(np.fft.ifft2(f * np.exp(-(fy**2 + fx**2) / (2 * width**2))))

def fourier_shift(img, dy, dx):
    fy = np.fft.fftfreq(img.shape[0])[:, None]; fx = np.fft.fftfreq(img.shape[1])[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j * np.pi * (fy * dy + fx * dx))))

ref = texture()
y, x = np.mgrid[0:96, 0:96]
periodic = ((x % 8) < 4).astype(float) + 0.5 * (((y % 8) < 4))
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(ref, cmap="gray"); ax[0].set_title("随机纹理")
ax[1].imshow(periodic, cmap="gray"); ax[1].set_title("周期 8 px 的图案")
plt.show()'''),
        ("md", "## 实验 1：亚像素误差随小数位移的变化\n\n预测：误差会不会随小数部分呈周期性变化？最大误差在 0.25 还是 0.5 附近？"),
        ("code", '''fracs = np.linspace(0, 1, 21)
errs = []
for f in fracs:
    est = attempt(mod.phase_correlation, ref, fourier_shift(ref, 3 + f, -2 - f), use_window=False)
    if est is None:
        break
    errs.append((est[0] - (3 + f), est[1] - (-2 - f)))
if errs:
    errs = np.array(errs)
    plt.plot(fracs[:len(errs)], errs[:, 0], "o-", label="dy 误差")
    plt.plot(fracs[:len(errs)], errs[:, 1], "s-", label="dx 误差")
    plt.axhline(0, color="k", lw=0.5); plt.xlabel("真值的小数部分"); plt.ylabel("误差 (px)"); plt.legend(); plt.show()
    print("最大绝对误差", np.abs(errs).max().round(3), "px")'''),
        ("md", "## 实验 2：周期图案的假峰\n\n周期 8 px。平移 3 px 和平移 5 px 时，估计结果分别是多少？为什么 5 px 会出问题？"),
        ("code", '''for true in [(0, 3), (0, 5), (0, 9)]:
    est = attempt(mod.phase_correlation, periodic, np.roll(periodic, true, axis=(0, 1)), use_window=False)
    if est is not None:
        print(f"真值 {true} → 估计 ({est[0]:.2f}, {est[1]:.2f})")
print("对照第 16 章教程第 7 节：怎样用对位标记或限制搜索范围解决？")'''),
        ("md", "## 实验 3：ZNCC 对亮度变化的不变性"),
        ("code", '''img = (ref - ref.min()) / np.ptp(ref) * 200
tpl = img[30:50, 40:64]
dim = img * 0.6 + 30
for name, target in [("原图", img), ("变暗加偏置", dim)]:
    m = attempt(mod.ncc_map, target, tpl)
    if m is None:
        break
    py, px = np.unravel_index(np.argmax(m), m.shape)
    print(f"{name}：峰值位置 ({py}, {px})，峰值 {m.max():.4f}")
if m is not None:
    plt.imshow(m, cmap="magma"); plt.colorbar(); plt.title("ZNCC 响应图"); plt.show()
if cv2 is not None:
    r = cv2.matchTemplate(img.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    print("OpenCV TM_CCOEFF_NORMED 峰值位置", np.unravel_index(np.argmax(r), r.shape))'''),
        ("md", "## 实验 4：噪声与加窗\n\n真实图像边界不连续。下面用“裁剪”而不是循环平移制造位移，比较加窗和不加窗。"),
        ("code", '''big = texture((160, 160), seed=3)
rng = np.random.default_rng(1)
rows = []
for use_window in (False, True):
    e = []
    for _ in range(20):
        dy, dx = rng.integers(-6, 7, size=2)
        a = big[30:126, 30:126]
        b = big[30 - dy:126 - dy, 30 - dx:126 - dx] + rng.normal(scale=0.05 * big.std(), size=(96, 96))
        est = attempt(mod.phase_correlation, a, b, use_window=use_window)
        if est is None:
            break
        e.append(max(abs(est[0] - dy), abs(est[1] - dx)))
    if e:
        rows.append((use_window, np.mean(e), np.max(e)))
for w, mean_e, max_e in rows:
    print(f"加窗={w}：平均误差 {mean_e:.3f} px，最大 {max_e:.3f} px")'''),
    ],
    "## 破坏实验与记录\n\n- 把噪声加到纹理标准差的 50%，误差怎么变？\n- 把纹理换成更平滑的（width=0.02），相位相关还可靠吗？\n- 把实验 1 的曲线、实验 2 的失败案例写进 `实验记录.md`。",
)

BOOKS["17_相机标定/实验_相机标定.ipynb"] = notebook(
    "第 17 章互动实验：合成相机标定",
    "用已知真值的合成相机检验你的标定实现，观察噪声、视图数和覆盖范围对精度的影响。",
    "17_相机标定", "17-1_张正友标定.py",
    [
        ("code", '''def rodrigues(r):
    t = np.linalg.norm(r)
    if t < 1e-12: return np.eye(3)
    k = r / t; K_ = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * K_ + (1 - np.cos(t)) * K_ @ K_

K_TRUE = np.array([[820.0, 0, 318.0], [0, 790.0, 247.0], [0, 0, 1]])
xs, ys = np.meshgrid(np.arange(9) * 15.0, np.arange(6) * 15.0)
BOARD = np.stack([xs.ravel(), ys.ravel()], 1)
BOARD3 = np.hstack([BOARD, np.zeros((len(BOARD), 1))])

def project(K, R, t, P):
    pc = P @ R.T + t; uv = pc @ K.T; return uv[:, :2] / uv[:, 2:3]

def make_views(n, noise, seed, tilt=0.35, spread=60):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        R = rodrigues(rng.uniform(-tilt, tilt, 3))
        t = np.array([rng.uniform(-spread, 0) - 30, rng.uniform(-spread, 0) - 20, rng.uniform(380, 500)])
        out.append((project(K_TRUE, R, t, BOARD3) + rng.normal(scale=noise, size=(len(BOARD), 2)), R, t))
    return out

views = make_views(6, 0.3, 0)
for uv, _, _ in views:
    plt.plot(uv[:, 0], uv[:, 1], ".", ms=3)
plt.xlim(0, 640); plt.ylim(480, 0); plt.gca().set_aspect("equal"); plt.title("6 个视图的角点"); plt.show()'''),
        ("code", '''def calibrate(views):
    Hs = [mod.homography_dlt(BOARD, uv) for uv, _, _ in views]
    return mod.intrinsics_from_homographies(Hs)

K = attempt(calibrate, views)
if K is not None:
    np.set_printoptions(precision=2, suppress=True)
    print("估计 K：\\n", K)
    print("真值 K：\\n", K_TRUE)'''),
        ("md", "## 实验 1：噪声与视图数\n\n预测：视图从 3 张增加到 15 张，fx 误差大致怎么变？"),
        ("code", '''res = {}
for noise in (0.1, 0.3, 1.0):
    for n in (3, 5, 10, 15):
        errs = []
        for s in range(20):
            K = attempt(calibrate, make_views(n, noise, s))
            if K is None: break
            errs.append(abs(K[0, 0] - 820) / 820)
        if not errs: break
        res[(noise, n)] = np.median(errs)
if res:
    for noise in (0.1, 0.3, 1.0):
        ns = [n for n in (3, 5, 10, 15) if (noise, n) in res]
        plt.plot(ns, [100 * res[(noise, n)] for n in ns], "o-", label=f"噪声 {noise} px")
    plt.xlabel("视图数"); plt.ylabel("fx 相对误差中位数 (%)"); plt.legend(); plt.show()'''),
        ("md", "## 实验 2：姿态太单一会怎样\n\n把倾斜角限制在 ±0.05 rad（几乎正对相机），观察主点和焦距。"),
        ("code", '''for tilt in (0.35, 0.05):
    K = attempt(calibrate, make_views(10, 0.3, 1, tilt=tilt))
    if K is None: break
    print(f"倾斜 ±{tilt} rad：fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")'''),
        ("md", "## 实验 3：重投影误差小不等于标定准"),
        ("code", '''views = make_views(10, 0.3, 2)
K = attempt(calibrate, views)
if K is not None:
    rmse = []
    for uv, _, _ in views:
        H = mod.homography_dlt(BOARD, uv)
        R, t = mod.extrinsics_from_homography(K, H)
        rmse.append(mod.reprojection_rmse(K, R, t, BOARD3, uv))
    print("各视图重投影 RMSE：", np.round(rmse, 3))
    print("想一想：这些误差是用估计的 K 自洽地算出来的，它能发现哪些错误，发现不了哪些？")
if cv2 is not None:
    obj = [BOARD3.astype(np.float32)] * len(views)
    img = [uv.astype(np.float32) for uv, _, _ in views]
    rms, Kcv, dist, _, _ = cv2.calibrateCamera(obj, img, (640, 480), None, None)
    print("OpenCV RMS", round(rms, 3), "\\nOpenCV K\\n", Kcv.round(2))'''),
    ],
    "## 破坏实验与记录\n\n- 把某张视图的角点顺序打乱，结果怎样？\n- 把棋盘格边长写错（15 写成 10），哪些参数受影响，哪些不受影响？\n- 把实验 1 的曲线和实验 2 的结论写进 `实验记录.md`。",
)

BOOKS["18_传统缺陷检测/实验_缺陷检测.ipynb"] = notebook(
    "第 18 章互动实验：金模板、阈值与过检漏检",
    "用合成的周期版图和注入缺陷，观察阈值 k、统计方法对检出和误报的影响。",
    "18_传统缺陷检测", "18-1_金模板比对.py",
    [
        ("code", '''def pattern(h=96, w=96):
    y, x = np.mgrid[0:h, 0:w]
    return 40.0 + ((x // 4) % 2) * 80.0 + (((y // 16) % 2 == 0) & ((x // 16) % 2 == 0)) * 60.0

def scene(seed, n_goods=12, noise=3.0):
    rng = np.random.default_rng(seed)
    base = pattern()
    goods = base + rng.normal(scale=noise, size=(n_goods, 96, 96))
    test = base + rng.normal(scale=noise, size=(96, 96))
    truth = []
    for _ in range(5):
        cy, cx = rng.integers(8, 88, 2)
        amp = rng.choice([-1, 1]) * rng.uniform(12, 40)
        test[cy - 1:cy + 2, cx - 1:cx + 2] += amp
        truth.append((float(cy), float(cx)))
    return goods, test, truth

goods, test, truth = scene(0)
res = attempt(mod.robust_golden, goods)
if res is not None:
    median, sigma = res
    z = np.abs(test - median) / sigma
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(test, cmap="gray"); ax[0].set_title("待检图")
    ax[1].imshow(sigma, cmap="viridis"); ax[1].set_title("逐像素 sigma")
    ax[2].imshow(z, cmap="magma", vmax=10); ax[2].set_title("z 图")
    for cy, cx in truth: ax[2].plot(cx, cy, "c+", ms=12)
    plt.show()'''),
        ("md", "## 实验 1：k 与检出、误报\n\n预测：k 从 3 增加到 9，检出率和误报数各怎么变？曲线的拐点在哪？"),
        ("code", '''ks = [3, 4, 5, 6, 7, 8, 9]
cap, fp = [], []
for k in ks:
    tp_all = n_all = fp_all = 0
    for s in range(10):
        g, t, tr = scene(100 + s)
        r = attempt(mod.robust_golden, g)
        if r is None: break
        found = attempt(mod.detect_defects, t, *r, k=k, min_area=3)
        if found is None: break
        m = attempt(mod.match_defects, [d["centroid"] for d in found], tr, 3.0)
        if m is None: break
        tp_all += m["tp"]; n_all += len(tr); fp_all += m["fp"]
    else:
        cap.append(tp_all / n_all); fp.append(fp_all / 10); continue
    break
if len(cap) == len(ks):
    fig, ax1 = plt.subplots(); ax2 = ax1.twinx()
    ax1.plot(ks, cap, "o-b"); ax1.set_ylabel("检出率", color="b")
    ax2.plot(ks, fp, "s-r"); ax2.set_ylabel("每张图误报", color="r")
    ax1.set_xlabel("k"); plt.title("阈值与检出、误报"); plt.show()'''),
        ("md", "## 实验 2：良品里混进坏图\n\n把 3 张良品替换成带大缺陷的图，比较中位数加 MAD 和均值加标准差。"),
        ("code", '''g, t, tr = scene(7)
g = g.copy()
g[:3, 40:56, 40:56] += 80
r = attempt(mod.robust_golden, g)
if r is not None:
    med, sig = r
    mean, std = g.mean(0), np.maximum(g.std(0), 1.0)
    for name, (c, s) in {"中位数+MAD": (med, sig), "均值+标准差": (mean, std)}.items():
        found = mod.detect_defects(t, c, s, k=6, min_area=3)
        m = mod.match_defects([d["centroid"] for d in found], tr, 3.0)
        print(f"{name}：检出 {m['tp']}/{len(tr)}，误报 {m['fp']}，污染区 sigma 均值 {s[40:56, 40:56].mean():.1f}")'''),
    ],
    "## 破坏实验与记录\n\n- 让待检图整体亮 10%，误报会发生什么？怎样修？\n- 把 `min_area` 设为 1 和 9，分别会丢掉什么？\n- 把实验 1 的曲线和你选的工作点写进 `实验记录.md`。",
)

BOOKS["19_异常检测与深度缺陷检测/实验_PatchCore.ipynb"] = notebook(
    "第 19 章互动实验：迷你 PatchCore",
    "先用简单的局部纹理特征把你实现的零件串成完整流程，不需要下载任何权重；最后一节可选地换成预训练 ResNet 特征。",
    "19_异常检测与深度缺陷检测", "19-1_PatchCore核心.py",
    [
        ("code", '''def make_image(seed, defect=False):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:64, 0:64]
    img = 0.5 + 0.3 * np.sin(2 * np.pi * x / 8 + rng.uniform(0, 0.3)) + rng.normal(scale=0.06, size=(64, 64))
    mask = np.zeros((64, 64), bool)
    if defect:
        cy, cx = rng.integers(10, 54, 2); r = rng.integers(2, 4)
        mask = (y - cy) ** 2 + (x - cx) ** 2 <= r * r
        img[mask] += rng.choice([-0.2, 0.2])
    return img, mask

def patch_features(img, p=4):
    """把每个 4x4 patch 展平成 16 维向量，再拼上邻域平均（模仿 PatchCore 的局部聚合）。"""
    h, w = img.shape
    pats = np.lib.stride_tricks.sliding_window_view(img, (p, p))[::2, ::2]
    feats = pats.reshape(pats.shape[0], pats.shape[1], -1)
    pad = np.pad(feats, ((1, 1), (1, 1), (0, 0)), mode="edge")
    agg = sum(pad[dy:dy + feats.shape[0], dx:dx + feats.shape[1]] for dy in range(3) for dx in range(3)) / 9
    return agg

train = [make_image(s)[0] for s in range(20)]
bank = np.concatenate([patch_features(im).reshape(-1, 16) for im in train])
print("记忆库大小：", bank.shape)'''),
        ("code", '''idx = attempt(mod.greedy_coreset, bank, max(1, len(bank) // 20))
if idx is not None:
    memory = bank[idx]
    print("核心集大小：", memory.shape)
    test = [make_image(1000 + s, defect=(s % 2 == 0)) for s in range(40)]
    labels = np.array([int(m.any()) for _, m in test])
    img_scores, maps = [], []
    for im, _ in test:
        f = patch_features(im); s = mod.knn_scores(memory, f.reshape(-1, 16), k=1).reshape(f.shape[:2])
        maps.append(s); img_scores.append(s.max())
    img_scores = np.array(img_scores)
    print("Image AUROC：", round(mod.auroc(img_scores, labels), 4))
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(4):
        ax[0, i].imshow(test[i][0], cmap="gray"); ax[0, i].contour(test[i][1], colors="c")
        ax[1, i].imshow(maps[i], cmap="magma"); ax[1, i].set_title(f"分数 {img_scores[i]:.2f}")
    plt.show()'''),
        ("md", "## 实验：从 AUROC 到产线工作点\n\n要求漏检率不超过 0%、5%、10% 时，过检率分别是多少？再看核心集比例从 100% 压到 1%，AUROC 掉了多少，想想为什么掉得这么少。"),
        ("code", '''if idx is not None:
    for esc in (0.0, 0.05, 0.10):
        r = attempt(mod.threshold_at_escape, img_scores[labels == 0], img_scores[labels == 1], esc)
        if r: print(f"漏检上限 {esc:.0%}：阈值 {r[0]:.3f}，实际漏检 {r[1]:.0%}，过检 {r[2]:.0%}")
    for frac in (1.0, 0.2, 0.05, 0.01):
        m = max(1, int(len(bank) * frac))
        mem = bank[mod.greedy_coreset(bank, m)] if frac < 1 else bank
        sc = [mod.knn_scores(mem, patch_features(im).reshape(-1, 16)).max() for im, _ in test]
        print(f"核心集比例 {frac:>5.0%}：AUROC {mod.auroc(np.array(sc), labels):.4f}")'''),
        ("md", "## 选做：换成预训练 ResNet 特征\n\n需要 torch、torchvision 和网络（首次下载权重）。在 MVTec AD 上跑时，建议直接用 Anomalib，把它的结果和这里的迷你版对照。"),
        ("code", '''try:
    import torch, torchvision
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    net = torchvision.models.resnet18(weights=weights).eval()
    print("已加载 ResNet18。取 layer2 和 layer3 的输出、上采样拼接，就得到 PatchCore 的 patch 特征。")
except Exception as exc:
    print("跳过：", type(exc).__name__, exc)'''),
    ],
    "## 破坏实验与记录\n\n- 训练集里混入 2 张带缺陷的图，AUROC 怎么变？为什么？\n- 去掉局部聚合（直接用 16 维 patch），结果怎么变？\n- 把漏检约束下的过检率写进 `实验记录.md`。",
)

BOOKS["20_合成缺陷与生成模型/实验_缺陷合成.ipynb"] = notebook(
    "第 20 章互动实验：合成缺陷与扩散前向过程",
    "比较“先插缺陷再成像”和“成像后直接贴缺陷”的差别，并可视化 DDPM 的前向加噪。",
    "20_合成缺陷与生成模型", "20-1_缺陷合成与扩散前向.py",
    [
        ("code", '''def blur(img, sigma):
    fy = np.fft.fftfreq(img.shape[0])[:, None]; fx = np.fft.fftfreq(img.shape[1])[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2 * (np.pi * sigma) ** 2 * (fy**2 + fx**2))))

rng = np.random.default_rng(0)
y, x = np.mgrid[0:96, 0:96]
clean = 0.3 + 0.4 * ((x % 10) < 5)
m = attempt(mod.scratch_mask, (96, 96), (20.0, 15.0), (70.0, 80.0), 2.0, 1.0)
if m is not None:
    right = blur(mod.blend(clean, 0.95, m), 1.5) + rng.normal(scale=0.03, size=clean.shape)
    wrong = mod.blend(blur(clean, 1.5) + rng.normal(scale=0.03, size=clean.shape), 0.95, m)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(m, cmap="gray"); ax[0].set_title("软掩码（标签）")
    ax[1].imshow(right, cmap="gray"); ax[1].set_title("先插缺陷再模糊加噪（正确）")
    ax[2].imshow(wrong, cmap="gray"); ax[2].set_title("成像后直接贴（贴纸感）")
    plt.show()
    print("观察：右图划痕边缘比背景锐利、没有噪声。检测器很容易学到这种捷径。")'''),
        ("md", "## DDPM 前向过程\n\n预测：t=300 时图像还能认出条纹吗？"),
        ("code", '''betas = attempt(mod.linear_beta_schedule, 1000)
if betas is not None:
    abar = mod.alpha_bars(betas)
    x0 = (clean - 0.5) * 2
    noise = np.random.default_rng(1).normal(size=x0.shape)
    ts = [0, 50, 150, 300, 600, 999]
    fig, ax = plt.subplots(1, len(ts), figsize=(15, 3))
    for a, t in zip(ax, ts):
        a.imshow(mod.q_sample(x0, t, abar, noise), cmap="gray"); a.set_title(f"t={t}, ᾱ={abar[t]:.3f}"); a.axis("off")
    plt.show()
    plt.plot(np.sqrt(abar), label="信号系数 sqrt(ᾱ)"); plt.plot(np.sqrt(1 - abar), label="噪声系数 sqrt(1-ᾱ)")
    plt.xlabel("t"); plt.legend(); plt.show()'''),
        ("md", "## VAE 重参数化：采样的梯度从哪里来"),
        ("code", '''z = attempt(mod.reparameterize, np.zeros(5), np.log(np.full(5, 4.0)), np.ones(5))
if z is not None:
    print("mu=0、方差 4、eps=1 时 z =", z, "（dz/dmu = 1，dz/dlogvar = 0.5·sigma·eps，都和 eps 的采样无关）")'''),
    ],
    "## 破坏实验与记录\n\n- 把软掩码的 softness 设得很小（接近硬边），再看“贴纸感”。\n- 用第 24 章的 `数据生成.py`，只用划痕缺陷调阈值，再测颗粒和缺失，检出率差多少？\n- 记录到 `实验记录.md`。",
)

BOOKS["21_3D数学与渲染管线/实验_软光栅.ipynb"] = notebook(
    "第 21 章互动实验：用自己的光栅器渲染立方体",
    "调用你实现的 look_at、perspective、to_screen、render_depth，渲染深度图，再加上逐面 Lambert 光照。",
    "21_3D数学与渲染管线", "21-1_软光栅.py",
    [
        ("code", '''V = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
              [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
FACES = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1),
         (3, 2, 6), (3, 6, 7), (0, 3, 7), (0, 7, 4), (1, 5, 6), (1, 6, 2)]
W, H = 240, 180

def render(eye, fov=50.0):
    view = mod.look_at(np.array(eye, float), np.zeros(3), np.array([0, 1, 0.0]))
    proj = mod.perspective(fov, W / H, 0.5, 20.0)
    s = mod.to_screen(V, proj @ view, W, H)
    tris = np.stack([s[list(f)] for f in FACES])
    return tris, mod.render_depth(tris, W, H)

out = attempt(render, [3.0, 2.5, 5.0])
if out is not None:
    tris, depth = out
    plt.imshow(np.where(np.isfinite(depth), depth, np.nan), cmap="viridis"); plt.colorbar(label="深度 [0,1]")
    plt.title("z-buffer 深度图"); plt.show()'''),
        ("md", "## Lambert 光照\n\n逐个三角形单独光栅化，再用深度最小者决定每个像素属于哪个面，乘上 max(0, n·l)。"),
        ("code", '''if out is not None:
    light = np.array([0.4, 0.8, 0.6]); light /= np.linalg.norm(light)
    per_face = np.stack([mod.render_depth(tris[i:i + 1], W, H) for i in range(len(FACES))])
    face_id = np.argmin(per_face, axis=0); covered = np.isfinite(per_face.min(axis=0))
    shade = np.zeros((H, W))
    for i, (a, b, c) in enumerate(FACES):
        n = np.cross(V[b] - V[a], V[c] - V[a]); n /= np.linalg.norm(n)
        shade[(face_id == i) & covered] = 0.15 + 0.85 * max(0.0, float(n @ light))
    plt.imshow(shade, cmap="gray", vmin=0, vmax=1); plt.title("逐面 Lambert"); plt.show()'''),
        ("md", "## 实验：near 平面与深度精度"),
        ("code", '''if out is not None:
    z = np.linspace(-0.5, -20, 400)
    for near in (0.01, 0.5, 2.0):
        P = mod.perspective(50, 1.0, near, 20.0)
        d = mod.to_screen(np.stack([np.zeros_like(z), np.zeros_like(z), z], 1), P, 2, 2)[:, 2]
        plt.plot(-z, d, label=f"near={near}")
    plt.xlabel("到相机的距离"); plt.ylabel("屏幕深度"); plt.legend(); plt.title("深度是非线性的"); plt.show()'''),
    ],
    "## 破坏实验与记录\n\n- 把 up 改成 (0, -1, 0)，图像怎样？\n- 把相机放到立方体内部，会发生什么？`to_screen` 的 NaN 处理在哪里起作用？\n- 把 OpenGL 和 OpenCV 相机约定的对照写进 `实验记录.md`。",
)

BOOKS["22_成像与设备仿真/实验_成像仿真.ipynb"] = notebook(
    "第 22 章互动实验：能不能看见这个缺陷",
    "用你的成像仿真器回答一个设计问题：在给定光学和曝光下，多大的缺陷能被检出。",
    "22_成像与设备仿真", "22-1_成像仿真.py",
    [
        ("code", '''for lam, na in [(532, 0.5), (532, 0.9), (266, 0.9), (193, 0.95)]:
    r = attempt(mod.rayleigh_resolution_um, lam, na)
    if r is None: break
    print(f"λ={lam} nm, NA={na}：分辨率 {r:.3f} µm，建议像素（物面）≤ {r / 2:.3f} µm")'''),
        ("md", "## 实验 1：缺陷信噪比 vs 缺陷尺寸\n\n场景：灰度 0.5 的平坦背景上一个亮颗粒。PSF sigma 固定，改变颗粒直径和曝光量。"),
        ("code", '''def defect_snr(diam_px, full_scale_e, psf_sigma=1.5, contrast=0.3, trials=30):
    y, x = np.mgrid[0:41, 0:41]
    scene = 0.5 + contrast * (((y - 20) ** 2 + (x - 20) ** 2) <= (diam_px / 2) ** 2)
    k = mod.gaussian_psf(psf_sigma)
    sharp = mod.convolve_same(scene, k)
    bg = mod.convolve_same(np.full_like(scene, 0.5), k)
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(trials):
        a = mod.simulate_capture(sharp * full_scale_e, 5.0, 2.0, 12, rng).astype(float)
        b = mod.simulate_capture(bg * full_scale_e, 5.0, 2.0, 12, rng).astype(float)
        diffs.append((a - b)[20, 20])
    noise = np.std([mod.simulate_capture(bg * full_scale_e, 5.0, 2.0, 12, rng)[20, 20] for _ in range(200)]) * np.sqrt(2)
    return np.mean(diffs) / noise

diams = [0.5, 1, 1.5, 2, 3, 4, 6]
for fs in (500, 2000, 6000):
    snr = attempt(lambda: [defect_snr(d, fs) for d in diams])
    if snr is None: break
    plt.plot(diams, snr, "o-", label=f"满量程 {fs} e-")
else:
    plt.axhline(5, color="k", ls="--", label="SNR=5 参考线")
    plt.xlabel("缺陷直径 (px)"); plt.ylabel("中心像素信噪比"); plt.legend(); plt.show()'''),
        ("md", "## 实验 2：光子转移曲线"),
        ("code", '''rng = np.random.default_rng(3)
levels = np.array([20, 50, 100, 300, 800, 1500, 3000, 6000, 12000, 20000], float)
means, vars_ = [], []
for lv in levels:
    f = attempt(mod.simulate_capture, np.full((100, 100), lv), 8.0, 1.5, 12, rng)
    if f is None: break
    means.append(f.mean()); vars_.append(f.var())
if len(means) == len(levels):
    plt.loglog(means, vars_, "o-"); plt.xlabel("均值 (ADU)"); plt.ylabel("方差 (ADU²)"); plt.title("PTC：注意高端的饱和弯折"); plt.show()
    ok = np.array(means) < 0.7 * 4095
    print("只用未饱和点拟合：", mod.estimate_gain_from_ptc(np.array(means)[ok], np.array(vars_)[ok]), "真值 (1.5, 8.0)")'''),
    ],
    "## 破坏实验与记录\n\n- 用全部点（含饱和点）拟合 PTC，增益估计偏向哪边？\n- 把 PSF sigma 从 1.5 改成 3（离焦），实验 1 的曲线怎样？\n- 写下你的结论：在你设定的参数下最小可检出缺陷多大，依据是什么。",
)

BOOKS["23_半导体检测领域知识/实验_晶圆图.ipynb"] = notebook(
    "第 23 章互动实验：晶圆图与空间模式",
    "画出 300 mm 晶圆的 die 网格，合成几种典型的缺陷空间分布，用径向密度和边缘比例区分它们。",
    "23_半导体检测领域知识", "23-1_晶圆图.py",
    [
        ("code", '''grid = attempt(mod.die_grid, 300.0, 12.0, 14.0, 3.0, (-6.0, -7.0))
if grid is not None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for col, row, cx, cy in grid:
        ax.add_patch(plt.Rectangle((cx - 6, cy - 7), 12, 14, fill=False, lw=0.5))
    ax.add_patch(plt.Circle((0, 0), 150, fill=False, color="k")); ax.add_patch(plt.Circle((0, 0), 147, fill=False, ls="--", color="gray"))
    ax.set_xlim(-155, 155); ax.set_ylim(-155, 155); ax.set_aspect("equal"); ax.set_title(f"完整 die 数：{len(grid)}"); plt.show()'''),
        ("code", '''rng = np.random.default_rng(0)
def sample_disk(n, rmax=147):
    r = rmax * np.sqrt(rng.random(n)); t = rng.random(n) * 2 * np.pi
    return np.stack([r * np.cos(t), r * np.sin(t)], 1)
patterns = {
    "随机": sample_disk(400),
    "中心": rng.normal(scale=35, size=(400, 2)),
    "边缘环": (lambda r, t: np.stack([r * np.cos(t), r * np.sin(t)], 1))(rng.uniform(132, 147, 400), rng.random(400) * 2 * np.pi),
    "划痕": np.stack([np.linspace(-90, 60, 400) + rng.normal(scale=1.5, size=400), 0.6 * np.linspace(-90, 60, 400) + 20 + rng.normal(scale=1.5, size=400)], 1),
}
fig, ax = plt.subplots(2, 4, figsize=(16, 7))
for i, (name, p) in enumerate(patterns.items()):
    ax[0, i].scatter(p[:, 0], p[:, 1], s=2); ax[0, i].add_patch(plt.Circle((0, 0), 150, fill=False))
    ax[0, i].set_xlim(-155, 155); ax[0, i].set_ylim(-155, 155); ax[0, i].set_aspect("equal"); ax[0, i].set_title(name)
    dens = attempt(mod.radial_density, p, 150.0, 6)
    er = attempt(mod.edge_ratio, p, 150.0, 15.0)
    if dens is not None:
        ax[1, i].bar(np.arange(6), dens); ax[1, i].set_title(f"径向密度｜边缘比例 {er:.2f}")
plt.show()'''),
        ("md", "## die 级缺陷计数热图\n\n把缺陷分配到 die，画出每个 die 的缺陷数，这就是良率工程师每天看的图。"),
        ("code", '''if grid is not None:
    pts = np.concatenate([patterns["随机"][:150], patterns["边缘环"][:200]])
    ids = attempt(mod.assign_to_die, pts, 12.0, 14.0, (-6.0, -7.0))
    if ids is not None:
        counts = {}
        for c, r in map(tuple, ids): counts[(c, r)] = counts.get((c, r), 0) + 1
        fig, ax = plt.subplots(figsize=(6, 6))
        for col, row, cx, cy in grid:
            n = counts.get((int(col), int(row)), 0)
            ax.add_patch(plt.Rectangle((cx - 6, cy - 7), 12, 14, color=plt.cm.Reds(min(n, 5) / 5), ec="k", lw=0.3))
        ax.set_xlim(-155, 155); ax.set_ylim(-155, 155); ax.set_aspect("equal"); ax.set_title("每个 die 的缺陷数"); plt.show()
        bad = sum(1 for col, row, _, _ in grid if counts.get((int(col), int(row)), 0) > 0)
        print(f"有缺陷的完整 die：{bad}/{len(grid)}，若每个缺陷都致命，良率上限约 {1 - bad / len(grid):.1%}")'''),
    ],
    "## 记录\n\n- 哪两种模式只靠径向密度分不开？还需要什么特征？\n- 把入职前术语清单整理进 `实验记录.md`。",
)

BOOKS["24_综合项目_晶圆检测流水线/实验_流水线调试.ipynb"] = notebook(
    "第 24 章：流水线调试",
    "逐步可视化数据和你的流水线输出。验收指标以 `tests/check_progress.py` 为准，这里用于找问题。",
    "24_综合项目_晶圆检测流水线", "24-1_检测流水线.py",
    [
        ("code", '''_spec = importlib.util.spec_from_file_location("gen24", CHAPTER / "数据生成.py")
gen = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(gen)
data = gen.make_dataset(seed=0)
g, t, truth = data["golden"], data["test"], data["truth"]
print("良品", g.shape, "待检", t.shape, "缺陷数", sum(map(len, truth)))
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].imshow(data["layout"], cmap="gray"); ax[0].set_title("版图（含对位十字）")
ax[1].imshow(g[0], cmap="gray"); ax[1].set_title("良品 0")
ax[2].imshow(g[1] - g[0], cmap="coolwarm"); ax[2].set_title("良品 1 - 良品 0（未对齐）")
i = next(k for k, tr in enumerate(truth) if tr)
ax[3].imshow(t[i], cmap="gray"); ax[3].set_title(f"待检 die {i}")
for cy, cx in truth[i]: ax[3].plot(cx, cy, "r+", ms=14)
plt.show()'''),
        ("code", '''preds = attempt(mod.run_pipeline, g.copy(), t.copy())
if preds is not None:
    def match(p, tr, r=4.0):
        pairs = sorted((np.hypot(a[0] - b[0], a[1] - b[1]), i, j) for i, a in enumerate(p) for j, b in enumerate(tr))
        up, ut = set(), set()
        for d, i, j in pairs:
            if d <= r and i not in up and j not in ut: up.add(i); ut.add(j)
        return up, ut
    tp = fp = fn = 0; misses = []
    for k, (p, tr) in enumerate(zip(preds, truth)):
        up, ut = match(p, tr); tp += len(up); fp += len(p) - len(up); fn += len(tr) - len(ut)
        if len(ut) < len(tr) or len(up) < len(p): misses.append(k)
    print(f"检出率 {tp / (tp + fn):.1%}，每 die 误报 {fp / len(preds):.2f}，有问题的 die：{misses}")'''),
        ("code", '''if preds is not None and misses:
    show = misses[:4]
    fig, ax = plt.subplots(1, len(show), figsize=(4 * len(show), 4))
    ax = np.atleast_1d(ax)
    for a, k in zip(ax, show):
        a.imshow(t[k], cmap="gray")
        for cy, cx in truth[k]: a.plot(cx, cy, "g+", ms=14, mew=2)
        for cy, cx in preds[k]: a.plot(cx, cy, "rx", ms=10, mew=2)
        a.set_title(f"die {k}：绿=真值 红=预测")
    plt.show()'''),
    ],
    "## 调试顺序建议\n\n1. 先只看对齐：把对齐后的良品两两相减，差分图应该只剩噪声。\n2. 再看亮度归一化：对齐后的图和金模板做散点图，应该落在 y=x 附近。\n3. 再看 z 图：缺陷处应该明显高于周围，图形边缘不应该一片亮。\n4. 最后调 k 和 min_area，并记录到 `项目报告.md`。",
)


def main():
    for rel, nb in BOOKS.items():
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(nb, path)
        print("写入", rel)


if __name__ == "__main__":
    main()
