"""生成第 16–24 章的互动实验 Notebook。

用法：python3 scripts/build_track2_notebooks.py
修改实验内容时改本文件再重新生成，不要直接手改 .ipynb，避免两边不一致。
Notebook 只调用学员自己在作业文件里的实现，不包含参考答案；
设置环境变量 COURSE_IMPL_DIR 可以让 Notebook 改用另一份实现（维护者验证用）。
"""

from __future__ import annotations

from pathlib import Path

import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

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

def run_or_hint(experiment):
    """运行一个实验函数；它用到的作业函数还没完成时，打印提示并返回 None。"""
    try:
        return experiment()
    except NotImplementedError as exc:
        print(f"先完成作业：{exc}")
        return None

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
print("对照第 16 章教程第 8 节：怎样用对位标记或限制搜索范围解决？实验 7 会试其中一种。")'''),
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
        ("md", "## 实验 5：同一个开关，换一种数据\n\n实验 4 的两幅图是从大图里裁出来的，边界不连续。这次只改一处：第二幅图用 `np.roll` 循环平移得到，其余（纹理、噪声、平移范围）都不变。验收里的大部分用例、第 24 章的合成数据都属于循环平移这一类。\n\n预测：在循环平移的数据上，加窗让误差变小还是变大？和实验 4 的结论一样吗？写下理由再运行。"),
        ("code", '''def window_experiment(trials=40):
    big = texture((160, 160), seed=3)
    a = big[30:126, 30:126]
    rows = []
    for cyclic in (False, True):
        for use_window in (False, True):
            rng = np.random.default_rng(1)
            e = []
            for _ in range(trials):
                dy, dx = rng.integers(-6, 7, size=2)
                b = np.roll(a, (dy, dx), axis=(0, 1)) if cyclic else big[30 - dy:126 - dy, 30 - dx:126 - dx]
                b = b + rng.normal(scale=0.05 * big.std(), size=(96, 96))
                est = mod.phase_correlation(a, b, use_window=use_window)
                e.append(max(abs(est[0] - dy), abs(est[1] - dx)))
            e = np.array(e)
            rows.append(("循环平移" if cyclic else "裁剪视场", use_window, np.median(e), e.max(), int((e > 1).sum())))
    return rows

rows = run_or_hint(window_experiment)
if rows:
    for kind, w, med_e, max_e, n_bad in rows:
        print(f"{kind}，加窗={w}：误差中位数 {med_e:.3f} px，最大 {max_e:.2f} px，超过 1 px 的有 {n_bad}/40 次")
    print("想一想：真实相机拍到的图更像哪一种？只用循环平移的数据验证算法，会漏掉什么问题？（教程第 7 节）")'''),
        ("md", "## 实验 6：把估计出的平移移回去，再估一次\n\n实验 1 的曲线说明三点抛物线有系统偏差，偏差的大小随真值的小数部分变化。教程第 6.3 节的第一种改进是迭代：按当前估计把第二幅图移回去，对剩下的残差再做一次相位相关，把两次结果相加。下面的 `refine` 只调用你的 `phase_correlation`，没有新的算法。\n\n预测：移回去之后，残差的小数部分更接近 0 还是更接近 0.5？对照实验 1 的曲线，这时抛物线的偏差是大还是小？迭代两轮后最大误差大约降到原来的几分之一？"),
        ("code", '''def refine(ref_img, mov_img, rounds):
    """rounds=0 是只做一次相位相关；每多一轮，就把 mov_img 按当前估计移回去，再估一次残差。"""
    total = np.array(mod.phase_correlation(ref_img, mov_img, use_window=False))
    for _ in range(rounds):
        back = fourier_shift(mov_img, -total[0], -total[1])
        total = total + np.array(mod.phase_correlation(ref_img, back, use_window=False))
    return total

def refine_experiment():
    fracs = np.linspace(0, 1, 21)
    curves = {}
    for rounds in (0, 1, 2):
        errs = []
        for f in fracs:
            est = refine(ref, fourier_shift(ref, 3 + f, -2 - f), rounds)
            errs.append(max(abs(est[0] - (3 + f)), abs(est[1] - (-2 - f))))
        curves[rounds] = np.array(errs)
    return fracs, curves

result = run_or_hint(refine_experiment)
if result:
    fracs, curves = result
    for rounds, errs in curves.items():
        plt.plot(fracs, errs, "o-", ms=3, label=f"迭代 {rounds} 轮：最大 {errs.max():.3f} px")
    plt.xlabel("真值的小数部分"); plt.ylabel("误差（dy、dx 中较大者，px）"); plt.legend(); plt.show()
    print("最大误差：", "，".join(f"迭代 {rounds} 轮 {errs.max():.3f} px" for rounds, errs in curves.items()))
    print("想一想：这里的第二幅图是无噪声的循环平移。换成实验 4 的裁剪视场，“移回去”这一步还能这样做吗？")'''),
        ("md", "## 实验 7：给周期图案加一个对位标记\n\n实验 2 里，周期图案上的平移一旦超过半个周期就估错。在图案左上角叠一个不重复的十字，其余不变，两幅图各加一点噪声，随机平移 40 次。\n\n预测：十字占的面积不到图像的百分之一，它能把结果救回来吗？没有标记时，40 次里大约错多少次，错的时候是不是正好错了整数个周期？"),
        ("code", '''marked = periodic.copy()
marked[11:13, 5:20] += 1.5
marked[5:20, 11:13] += 1.5

def mark_experiment(trials=40):
    out = {}
    for name, img in (("没有标记", periodic), ("有十字标记", marked)):
        rng = np.random.default_rng(4)
        errs = []
        for _ in range(trials):
            true = rng.integers(-7, 8, size=2)
            a = img + rng.normal(scale=0.02, size=img.shape)
            b = np.roll(img, true, axis=(0, 1)) + rng.normal(scale=0.02, size=img.shape)
            est = mod.phase_correlation(a, b, use_window=False)
            errs.append((est[0] - true[0], est[1] - true[1]))
        out[name] = np.abs(np.array(errs))
    return out

result = run_or_hint(mark_experiment)
if result:
    for name, errs in result.items():
        wrong = errs[errs.max(axis=1) > 1]
        whole = int((np.abs(wrong - 8 * np.round(wrong / 8)) < 0.5).all(axis=1).sum())
        print(f"{name}：40 次里误差超过 1 px 的有 {len(wrong)} 次，其中两个方向都正好错了整数个周期（8 px 的整数倍）的有 {whole} 次")
    plt.imshow(marked, cmap="gray"); plt.title("带十字对位标记的周期图案"); plt.show()
    print("想一想：教程第 8 节讲了周期图案上的两种失败。没有标记时这里是哪一种，还是两种都有？十字的频谱和周期图案的频谱有什么不同？")'''),
    ],
    "## 破坏实验与记录\n\n- 把噪声加到纹理标准差的 50%，误差怎么变？\n- 把纹理换成更平滑的（width=0.02），相位相关还可靠吗？\n- 把实验 7 的十字改小（例如只留 3×3 的一个点），或者把噪声加大 10 倍，标记还管用吗？\n- 把实验 1 的曲线、实验 2 的失败案例、实验 5 的四行结果写进 `实验记录.md`。",
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

def run_or_hint(experiment):
    """运行一个实验函数；它用到的作业函数还没完成时，打印提示并返回 None。"""
    try:
        return experiment()
    except NotImplementedError as exc:
        print(f"先完成作业：{exc}")
        return None

views = make_views(6, 0.3, 0)
for uv, _, _ in views:
    plt.plot(uv[:, 0], uv[:, 1], ".", ms=3)
plt.xlim(0, 640); plt.ylim(480, 0); plt.gca().set_aspect("equal"); plt.title("6 个视图的角点"); plt.show()'''),
        ("code", '''def calibrate(views):
    Hs = [mod.homography_dlt(BOARD, uv) for uv, _, _ in views]
    return mod.intrinsics_from_homographies(Hs)

def try_calibrate(views):
    """和 calibrate 一样，但闭式解解不出有效内参时返回 None。
    视图少、噪声大或者姿态太单一时会这样：公式解法在开方时出现 nan，Cholesky 解法报“不是正定矩阵”。"""
    try:
        with np.errstate(all="ignore"):
            K = calibrate(views)
    except np.linalg.LinAlgError:
        return None
    return K if np.all(np.isfinite(K)) else None

K = attempt(calibrate, views)
if K is not None:
    np.set_printoptions(precision=2, suppress=True)
    print("估计 K：\\n", K)
    print("真值 K：\\n", K_TRUE)'''),
        ("md", "## 实验 1：噪声与视图数\n\n预测：视图从 3 张增加到 15 张，fx 误差大致怎么变？"),
        ("code", '''def views_and_noise(noises=(0.1, 0.3, 1.0), counts=(3, 5, 10, 15), repeats=20):
    res, failed = {}, []
    for noise in noises:
        for n in counts:
            errs = []
            for s in range(repeats):
                K = try_calibrate(make_views(n, noise, s))
                if K is None:
                    failed.append((noise, n))   # 解不出有效的内参，不计入中位数
                else:
                    errs.append(abs(K[0, 0] - 820) / 820)
            res[(noise, n)] = np.median(errs) if errs else np.nan
    return res, failed

result = run_or_hint(views_and_noise)
if result:
    res, failed = result
    for noise in (0.1, 0.3, 1.0):
        plt.plot((3, 5, 10, 15), [100 * res[(noise, n)] for n in (3, 5, 10, 15)], "o-", label=f"噪声 {noise} px")
    plt.xlabel("视图数"); plt.ylabel("fx 相对误差中位数 (%)"); plt.legend(); plt.show()
    if failed:
        print(f"有 {len(failed)} 次标定解不出有效的内参，没有计入中位数，出现在（噪声, 视图数）= {sorted(set(failed))}。想一想：为什么偏偏是这种条件？")'''),
        ("md", "## 实验 2：姿态太单一会怎样\n\n把倾斜角限制在 ±0.05 rad（几乎正对相机），观察主点和焦距。"),
        ("code", '''rows = run_or_hint(lambda: [(tilt, try_calibrate(make_views(10, 0.3, 1, tilt=tilt))) for tilt in (0.35, 0.05)])
for tilt, K in rows or []:
    if K is None:
        print(f"倾斜 ±{tilt} rad：解不出有效的内参")
    else:
        print(f"倾斜 ±{tilt} rad：fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")'''),
        ("md", "## 实验 3：重投影误差小不等于标定准"),
        ("code", '''views = make_views(10, 0.3, 2)

def per_view_rmse():
    K = calibrate(views)
    rmse = []
    for uv, _, _ in views:
        H = mod.homography_dlt(BOARD, uv)
        R, t = mod.extrinsics_from_homography(K, H)
        rmse.append(mod.reprojection_rmse(K, R, t, BOARD3, uv))
    return rmse

rmse = run_or_hint(per_view_rmse)
if rmse is not None:
    print("各视图重投影 RMSE：", np.round(rmse, 3))
    print("想一想：这些误差是用估计的 K 自洽地算出来的，它能发现哪些错误，发现不了哪些？")
if cv2 is not None:
    obj = [BOARD3.astype(np.float32)] * len(views)
    img = [uv.astype(np.float32) for uv, _, _ in views]
    rms, Kcv, dist, _, _ = cv2.calibrateCamera(obj, img, (640, 480), None, None)
    print("OpenCV RMS", round(rms, 3), "\\nOpenCV K\\n", Kcv.round(2))'''),
        ("md", "## 实验 4：倾斜角变小时，哪个数字会报警\n\n实验 2 只看了一次标定。这次把倾斜角的上限从 0.5 rad 一路降到 0.05 rad，每个角度重复 20 次，同时记录 fx 的相对误差和重投影 RMS（10 张视图，角点噪声 0.3 px，没有畸变）。\n\n预测：倾斜角变小时，fx 误差和重投影 RMS 各怎么变？如果手里只有标定程序报告的 RMS，你能发现姿态太单一吗？"),
        ("code", '''def linear_calibration(views):
    """用作业里的线性解标定。返回 (K, 所有视图合起来的重投影 RMS)；闭式解解不出有效内参时返回 None。"""
    K = try_calibrate(views)
    if K is None:
        return None
    sq = []
    with np.errstate(all="ignore"):
        for uv, _, _ in views:
            R, t = mod.extrinsics_from_homography(K, mod.homography_dlt(BOARD, uv))
            sq.append(mod.reprojection_rmse(K, R, t, BOARD3, uv) ** 2)
    rms = float(np.sqrt(np.mean(sq)))
    return (K, rms) if np.isfinite(rms) else None

def tilt_experiment(tilts=(0.5, 0.35, 0.2, 0.1, 0.05), repeats=20):
    rows = []
    for tilt in tilts:
        results = [linear_calibration(make_views(10, 0.3, 300 + s, tilt=tilt)) for s in range(repeats)]
        ok = [r for r in results if r is not None]
        fx_err = [abs(K[0, 0] - 820) / 820 * 100 for K, _ in ok]
        rows.append((tilt, len(results) - len(ok), np.median(fx_err) if ok else np.nan, np.median([r for _, r in ok]) if ok else np.nan))
    return rows

rows = run_or_hint(tilt_experiment)
if rows:
    for tilt, failed, fx_err, rms in rows:
        print(f"倾斜上限 ±{tilt:.2f} rad（约 {np.degrees(tilt):.0f}°）：fx 误差中位数 {fx_err:.2f}%，重投影 RMS 中位数 {rms:.3f} px，解不出有效内参 {failed}/20 次")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    deg = [np.degrees(r[0]) for r in rows]
    ax[0].semilogy(deg, [r[2] for r in rows], "o-"); ax[0].set_xlabel("倾斜上限（度）"); ax[0].set_ylabel("fx 相对误差中位数 (%)")
    ax[1].plot(deg, [r[3] for r in rows], "s-", color="C3"); ax[1].set_ylim(0, 0.6); ax[1].set_xlabel("倾斜上限（度）"); ax[1].set_ylabel("重投影 RMS 中位数 (px)")
    plt.tight_layout(); plt.show()
    print("想一想：RMS 几乎不随倾斜角变，为什么？对照教程第 3.4 节“为什么板子要倾斜”和第 6.1 节。")'''),
        ("md", "## 实验 5：镜头有畸变，模型里没有\n\n给合成相机加上 $k_1 = -0.1$ 的桶形畸变，它让画面四角的点向内移动将近 10 px。拍 40 张覆盖全画面的视图（角点噪声 0.2 px），用三种办法标定：你的线性解（模型里没有畸变）；OpenCV 的非线性优化，但不建畸变模型；OpenCV 建 $k_1$、$k_2$。没装 OpenCV 时只做第一种。\n\n每种办法都把角点的残差沿半径方向的分量，按“到主点的距离”分组取平均。随机噪声在平均后互相抵消，剩下的就是模型没解释掉的部分。\n\n预测：三种办法的重投影 RMS 各是好几个像素、1 px 上下，还是零点几？不建畸变模型时 fx 偏多少？哪一条残差曲线是平的？"),
        ("code", '''def project_k1(K, R, t, P, k1):
    pc = P @ R.T + t
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    f = 1 + k1 * (x * x + y * y)
    return np.stack([K[0, 0] * x * f + K[0, 2], K[1, 1] * y * f + K[1, 2]], axis=1)

def views_k1(n, noise, seed, k1=-0.1, spread=0.85, tilt=0.5):
    """带径向畸变的视图。spread 是板中心偏离画面中心的范围（0 到 1）；只保留整块板都在画面内的姿态。"""
    rng = np.random.default_rng(seed); out = []
    center = np.append(BOARD.mean(axis=0), 0.0)
    while len(out) < n:
        R = rodrigues(rng.uniform(-tilt, tilt, 3))
        aim = np.array([318 + spread * rng.uniform(-1, 1) * 320, 247 + spread * rng.uniform(-1, 1) * 240, 1.0])
        t = rng.uniform(380, 520) * np.linalg.inv(K_TRUE) @ aim - R @ center
        uv = project_k1(K_TRUE, R, t, BOARD3, k1)
        if uv[:, 0].min() > 2 and uv[:, 0].max() < 638 and uv[:, 1].min() > 2 and uv[:, 1].max() < 478:
            out.append((uv + rng.normal(scale=noise, size=uv.shape), R, t))
    return out

def residuals(K, views):
    """K 固定，逐视图用作业函数求外参。返回所有角点的位置和残差向量（观测位置减重投影位置）。"""
    pos, res = [], []
    for uv, _, _ in views:
        R, t = mod.extrinsics_from_homography(K, mod.homography_dlt(BOARD, uv))
        pos.append(uv); res.append(uv - project(K, R, t, BOARD3))
    return np.vstack(pos), np.vstack(res)

def rms_of(res):
    return float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))

EDGES = np.arange(0, 351, 50)

def radial_means(pos, res):
    """残差沿半径方向（从主点指向角点）的分量，按角点到主点的距离每 50 px 一组取平均。"""
    offset = pos - K_TRUE[:2, 2]
    radius = np.hypot(offset[:, 0], offset[:, 1])
    radial = (offset * res).sum(axis=1) / np.maximum(radius, 1e-9)
    return [radial[(radius >= a) & (radius < b)].mean() for a, b in zip(EDGES[:-1], EDGES[1:])]

def cv_calibrate(views, with_distortion):
    """OpenCV 的 calibrateCamera。返回 K、畸变系数、RMS，以及所有角点的位置和残差。"""
    obj = BOARD3.astype(np.float32)
    img = [uv.astype(np.float32).reshape(-1, 1, 2) for uv, _, _ in views]
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    if not with_distortion:
        flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj] * len(img), img, (640, 480), None, None, flags=flags)
    res = [uv - cv2.projectPoints(obj, rv, tv, K, dist)[0].reshape(-1, 2) for (uv, _, _), rv, tv in zip(views, rvecs, tvecs)]
    return K, dist, float(rms), np.vstack([uv for uv, _, _ in views]), np.vstack(res)

distorted = views_k1(40, 0.2, 3)
curves = {}

def linear_on_distorted():
    K = try_calibrate(distorted)
    if K is None:
        return "解不出有效的内参"
    pos, res = residuals(K, distorted)
    return K, rms_of(res), radial_means(pos, res)

result = run_or_hint(linear_on_distorted)
if isinstance(result, str):
    print("你的线性解：" + result)
elif result:
    K, rms, curves["线性解，没有畸变模型"] = result
    print(f"你的线性解：重投影 RMS {rms:.2f} px，fx={K[0, 0]:.1f}（真值 820，偏 {100 * (K[0, 0] - 820) / 820:+.1f}%）")
if cv2 is not None:
    for name, with_distortion in (("OpenCV 优化，没有畸变模型", False), ("OpenCV 优化，建 k1、k2", True)):
        K, dist, rms, pos, res = cv_calibrate(distorted, with_distortion)
        curves[name] = radial_means(pos, res)
        print(f"{name}：重投影 RMS {rms:.2f} px，fx={K[0, 0]:.1f}（偏 {100 * (K[0, 0] - 820) / 820:+.1f}%），k1={dist.ravel()[0]:+.3f}")
if curves:
    for name, means in curves.items():
        plt.plot(EDGES[:-1] + 25, means, "o-", label=name)
    plt.axhline(0, color="k", lw=0.5); plt.xlabel("角点到主点的距离 (px)"); plt.ylabel("径向残差的平均 (px)"); plt.legend(); plt.show()
    if cv2 is not None:
        print("想一想：优化之后、没有畸变模型的那一次，RMS 看上去很正常，只看 RMS 你会放过它吗？它的 fx 和残差曲线说明了什么？（教程第 6.2 节）")
    else:
        print("想一想：畸变让四角的点移动了将近 10 px，线性解的 RMS 却小得多，差的那部分被谁吸收了？装上 OpenCV 后重跑，可以看到另外两条曲线。（教程第 6.2 节）")'''),
        ("md", "## 实验 6：留出验证，按位置看残差\n\n相机仍带 $k_1 = -0.1$ 的畸变。两种拍法各标定一次：标定板只出现在画面中央，或者覆盖全画面，各 12 张。然后另拍 30 张覆盖全画面的图，**不参与标定**，用标定出的内参去算它们的重投影误差，并且把画面中央和四角分开统计。\n\n先用你的线性解做（模型里没有畸变），有 OpenCV 时再用带 $k_1$、$k_2$ 的模型做一遍。\n\n预测：线性解的两种拍法里，哪一种自己报告的 RMS 更小？它在留出视图上还是这么小吗？换成带畸变的模型之后，两种拍法自己报告的 RMS 差多少，留出视图四角的残差又差多少？"),
        ("code", '''held = views_k1(30, 0.2, 100, spread=0.85)
TRAIN = {"只拍画面中央": views_k1(12, 0.2, 7, spread=0.15), "覆盖全画面": views_k1(12, 0.2, 8, spread=0.85)}

def in_middle(pos):
    return (np.abs(pos[:, 0] - 320) < 110) & (np.abs(pos[:, 1] - 240) < 80)

def in_corners(pos):
    return (np.abs(pos[:, 0] - 320) > 210) & (np.abs(pos[:, 1] - 240) > 150)

def report(name, self_rms, pos, res):
    e = np.hypot(res[:, 0], res[:, 1])
    print(f"  {name}：标定自己报告的 RMS {self_rms:.2f} px；留出视图 RMS {rms_of(res):.2f} px，"
          f"其中画面中央平均 {e[in_middle(pos)].mean():.2f} px，四角平均 {e[in_corners(pos)].mean():.2f} px")
    return e

def holdout_linear():
    out = {}
    for name, train in TRAIN.items():
        K = try_calibrate(train)
        if K is None:
            print(f"  {name}：线性解解不出有效的内参")
            continue
        out[name] = (rms_of(residuals(K, train)[1]), *residuals(K, held))
    return out

result = run_or_hint(holdout_linear)
if result:
    print("线性解（模型里没有畸变）：")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, (name, (self_rms, pos, held_res)) in zip(ax, result.items()):
        e = report(name, self_rms, pos, held_res)
        sc = a.scatter(pos[:, 0], pos[:, 1], c=e, s=5, cmap="magma", vmin=0, vmax=3)
        a.set_xlim(0, 640); a.set_ylim(480, 0); a.set_aspect("equal"); a.set_title(f"留出残差（标定时{name}）")
    fig.colorbar(sc, ax=ax, label="px"); plt.show()

if cv2 is not None:
    print("OpenCV，模型里有 k1、k2：")
    obj = BOARD3.astype(np.float32)
    for name, train in TRAIN.items():
        Kcv, dist, rms, _, _ = cv_calibrate(train, True)
        pos, res = [], []
        for uv, _, _ in held:
            _, rv, tv = cv2.solvePnP(obj, uv.astype(np.float32).reshape(-1, 1, 2), Kcv, dist)
            pos.append(uv); res.append(uv - cv2.projectPoints(obj, rv, tv, Kcv, dist)[0].reshape(-1, 2))
        report(name, rms, np.vstack(pos), np.vstack(res))
        print(f"    估出的 k1 = {dist.ravel()[0]:+.3f}（真值 -0.100），fx = {Kcv[0, 0]:.1f}（真值 820）")
if result or cv2 is not None:
    print("想一想：哪些数字是标定程序自己会报告的，哪些只有留出验证并且按位置看才看得到？这里每种拍法只标定了一次，教程图 5 是重复 30 次的统计。（教程第 6.3 节）")'''),
    ],
    "## 破坏实验与记录\n\n- 把某张视图的角点顺序打乱，结果怎样？\n- 把棋盘格边长写错（15 写成 10），哪些参数受影响，哪些不受影响？\n- 实验 5 里把 `views_k1(40, 0.2, 3)` 的 `k1` 换成 +0.1（枕形畸变），残差曲线怎么变？再换几个随机种子，fx 的偏差方向固定吗？\n- 把实验 1 的曲线、实验 2 和实验 4 的结论、实验 6 的那张表写进 `实验记录.md`。",
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

def run_or_hint(experiment):
    """运行一个实验函数；它用到的作业函数还没完成时，打印提示并返回 None。"""
    try:
        return experiment()
    except NotImplementedError as exc:
        print(f"先完成作业：{exc}")
        return None

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
        ("md", "## 实验 2：良品里混进坏图\n\n12 张良品里有 3 张在同一块 16×16 的区域整体偏亮 80。待检图除了原来的 5 个缺陷，在这块区域里还有一个 3×3、幅度 40 的缺陷。比较中位数加 MAD 和均值加标准差。\n\n预测：两种方法各自能不能检出污染区域里的那个缺陷？检不出的那一种，主要是模板被拉偏了，还是 sigma 被撑大了？"),
        ("code", '''def contaminated(n_bad=3):
    g, t, tr = scene(7)
    g, t = g.copy(), t.copy()
    g[:n_bad, 40:56, 40:56] += 80
    t[47:50, 47:50] += 40
    inside = (48.0, 48.0)
    med, sig = mod.robust_golden(g)
    mean, std = g.mean(0), np.maximum(g.std(0), 1.0)
    rows = []
    for name, (c, s) in {"中位数+MAD": (med, sig), "均值+标准差": (mean, std)}.items():
        found = mod.detect_defects(t, c, s, k=6, min_area=3)
        centroids = [d["centroid"] for d in found]
        m = mod.match_defects(centroids, tr + [inside], 3.0)
        hit = mod.match_defects(centroids, [inside], 3.0)["tp"] == 1
        rows.append((name, m["tp"], len(tr) + 1, m["fp"], hit, s[40:56, 40:56].mean(), np.abs(c - pattern())[40:56, 40:56].mean()))
    return rows

rows = run_or_hint(contaminated)
if rows:
    for name, tp, n, fp_count, hit, sigma_in, bias_in in rows:
        print(f"{name}：检出 {tp}/{n}，误报 {fp_count}；污染区里的缺陷{'检出' if hit else '漏掉'}，污染区 sigma 均值 {sigma_in:.1f}（真实噪声 3），模板偏差均值 {bias_in:.1f}")
    print("想一想：把 contaminated(3) 换成 contaminated(5) 再跑，中位数加 MAD 还撑得住吗？（教程第 3.2 节）")'''),
        ("md", "## 实验 3：良品张数与误报\n\n这次图里没有任何缺陷，只有标准差为 3 的高斯噪声。用 N 张良品建模板，再拿一张新的良品算 z，统计 z > 5 的像素占多少。sigma 准确已知时，这个比例的理论值是 5.7e-7，一张 96×96 的图上平均不到 0.01 个像素。\n\n预测：N 取 5、9、15、30、100 时，一张良品图上大约各有多少个像素超过 5？和理论值差几个数量级？均值加标准差的做法会更多还是更少？"),
        ("code", '''import math

def goods_count_experiment(k=5.0, counts=(5, 9, 15, 30, 100), repeats=10):
    rows = []
    for n in counts:
        over_mad = over_std = total = 0
        for s in range(repeats):
            rng = np.random.default_rng(1000 * n + s)
            goods = pattern() + rng.normal(scale=3.0, size=(n, 96, 96))
            fresh = pattern() + rng.normal(scale=3.0, size=(96, 96))
            median, sigma = mod.robust_golden(goods)
            over_mad += int((np.abs(fresh - median) / sigma > k).sum())
            over_std += int((np.abs(fresh - goods.mean(0)) / np.maximum(goods.std(0, ddof=1), 1.0) > k).sum())
            total += fresh.size
        rows.append((n, over_mad / total, over_std / total, over_mad / repeats))
    return rows

rows = run_or_hint(goods_count_experiment)
if rows:
    theory = math.erfc(5.0 / math.sqrt(2))
    for n, p_mad, p_std, per_image in rows:
        print(f"N={n:3d}：中位数+MAD 的单像素误报概率 {p_mad:.1e}（每张图约 {per_image:.1f} 个像素），均值+标准差 {p_std:.1e}，理论值 {theory:.1e}")
    ns = [r[0] for r in rows]
    plt.loglog(ns, [max(r[1], 1e-7) for r in rows], "o-", label="中位数 + MAD")
    plt.loglog(ns, [max(r[2], 1e-7) for r in rows], "s-", label="均值 + 标准差")
    plt.axhline(theory, color="g", ls="--", label="sigma 已知时的理论值")
    plt.xlabel("良品张数 N"); plt.ylabel("良品像素 z > 5 的概率"); plt.legend(); plt.show()
    print("想一想：N 很小时误报为什么多？把“k=5 对应百万分之一”写进报告错在哪？（教程第 3.3 节）")'''),
        ("md", "## 实验 4：待检图整体变亮\n\n照明或曝光的漂移会让待检图相对模板整体变亮。把一张良品图乘上 1.00 到 1.30 的增益，直接比对，统计亮区（灰度 180 的方块）里 z > 5 的像素比例。再做线性亮度归一化（把待检图拟合成 a × 模板 + b，换算回去再比），重做一遍。\n\n预测：教程第 4 节给的门槛估算是 k × sigma / 亮度。代入 k = 5、sigma = 3、亮度 180，增益到多少时亮区大约一半像素报警？暗区（灰度 40）的门槛是多少？归一化之后还剩多少？"),
        ("code", '''def brightness_experiment(gains=np.arange(1.0, 1.301, 0.02)):
    rng = np.random.default_rng(11)
    goods = pattern() + rng.normal(scale=3.0, size=(30, 96, 96))
    fresh = pattern() + rng.normal(scale=3.0, size=(96, 96))
    median, sigma = mod.robust_golden(goods)
    bright, dark = pattern() > 150, pattern() < 60
    rows = []
    for g in gains:
        shifted = fresh * g
        a, b = np.polyfit(median.ravel(), shifted.ravel(), 1)
        z_raw = np.abs(shifted - median) / sigma
        z_fix = np.abs((shifted - b) / a - median) / sigma
        rows.append((g, (z_raw > 5)[bright].mean(), (z_raw > 5)[dark].mean(), (z_fix > 5)[bright].mean()))
    return np.array(rows)

rows = run_or_hint(brightness_experiment)
if rows is not None:
    pct = 100 * (rows[:, 0] - 1)
    plt.plot(pct, 100 * rows[:, 1], "o-", label="亮区（灰度 180），不归一化")
    plt.plot(pct, 100 * rows[:, 2], "s-", label="暗区（灰度 40），不归一化")
    plt.plot(pct, 100 * rows[:, 3], "^-", label="亮区，归一化之后")
    plt.axvline(100 * 5 * 3 / 180, color="gray", ls="--", label="k × sigma / 亮度（亮区）")
    plt.xlabel("待检图比模板整体亮 (%)"); plt.ylabel("z > 5 的像素比例 (%)"); plt.legend(); plt.show()
    half = pct[np.argmax(rows[:, 1] >= 0.5)] if (rows[:, 1] >= 0.5).any() else None
    print("亮区报警比例第一次超过一半时的增益：", f"亮 {half:.0f}%" if half is not None else "这个范围内没有超过一半")
    print("想一想：换一台噪声只有三分之一的相机，这条曲线往哪边移？噪声低反而更怕亮度漂移，为什么？（教程第 4 节）")'''),
        ("md", "## 实验 5：匹配半径\n\n检测结果固定不变，只改 `match_defects` 的 `radius`。真值坐标加了标准差 1.5 px 的随机偏移，模拟人工标注或复查设备给的坐标本身有误差。这次用弱一些的缺陷（幅度 6 到 15）、k = 4.5、min_area = 2，让漏检和误报都出现。\n\n预测：半径从 1 加到 20，检出率怎么变？是一直涨，还是到某个值就平了？半径很大时多出来的“检出”是真的检出吗？"),
        ("code", '''def radius_experiment(radii=(1, 2, 3, 5, 8, 20), images=30):
    rng = np.random.default_rng(5)
    cases = []
    for s in range(images):
        base = pattern()
        goods = base + rng.normal(scale=3.0, size=(12, 96, 96))
        test_img = base + rng.normal(scale=3.0, size=(96, 96))
        truth = []
        for _ in range(5):
            cy, cx = rng.integers(8, 88, 2)
            test_img[cy - 1:cy + 2, cx - 1:cx + 2] += rng.choice([-1, 1]) * rng.uniform(6, 15)
            truth.append((float(cy), float(cx)))
        found = mod.detect_defects(test_img, *mod.robust_golden(goods), k=4.5, min_area=2)
        jittered = [(cy + rng.normal(scale=1.5), cx + rng.normal(scale=1.5)) for cy, cx in truth]
        cases.append(([d["centroid"] for d in found], truth, jittered))
    rows = []
    for radius in radii:
        tp = fp = n = 0
        for pred, _, jittered in cases:
            m = mod.match_defects(pred, jittered, radius)
            tp += m["tp"]; fp += m["fp"]; n += len(jittered)
        rows.append((radius, tp / n, fp / images))
    exact = sum(mod.match_defects(pred, truth, 3.0)["tp"] for pred, truth, _ in cases) / sum(len(t) for _, t, _ in cases)
    return rows, exact

result = run_or_hint(radius_experiment)
if result:
    rows, exact = result
    for radius, capture, fp_per_image in rows:
        print(f"半径 {radius:2d} px：检出率 {capture:.0%}，每张图误报 {fp_per_image:.1f}")
    print(f"真值坐标没有误差、半径 3 px 时的检出率：{exact:.0%}")
    fig, ax1 = plt.subplots(); ax2 = ax1.twinx()
    ax1.semilogx([r[0] for r in rows], [r[1] for r in rows], "o-b"); ax1.axhline(exact, color="b", ls=":")
    ax2.semilogx([r[0] for r in rows], [r[2] for r in rows], "s-r")
    ax1.set_xlabel("匹配半径 (px)"); ax1.set_ylabel("检出率", color="b"); ax2.set_ylabel("每张图误报", color="r")
    plt.title("检测结果不变，只改匹配半径"); plt.show()
    print("想一想：同一份检测结果能报出差这么多的检出率。半径太小和太大各错在哪？合理的半径由什么决定？（教程第 6.1 节）")'''),
    ],
    "## 破坏实验与记录\n\n- 实验 4 里让待检图整体亮 10%，误报出现在哪些区域？把 z 图画出来看。\n- 把 `min_area` 设为 1 和 9，分别会丢掉什么？\n- 实验 3 里把 `min_sigma` 改成 0.1 再跑一遍，N=5 时的误报怎么变？为什么？\n- 把实验 1 的曲线和你选的工作点、实验 3 的那张表、实验 5 里你认为合理的半径写进 `实验记录.md`。",
)

BOOKS["19_异常检测与深度缺陷检测/实验_PatchCore.ipynb"] = notebook(
    "第 19 章互动实验：迷你 PatchCore",
    "先用简单的局部纹理特征把你实现的零件串成完整流程，不需要下载任何权重；中间两个“对照”单元把数据换难一点，让局部聚合的边界问题和核心集的代价显出来；最后一节可选地换成预训练 ResNet 特征，拼成完整的 PatchCore。",
    "19_异常检测与深度缺陷检测", "19-1_PatchCore核心.py",
    [
        ("code", '''def make_image(seed, defect=False, aligned=True, amp=0.2):
    """一张 64x64 的条纹图。aligned=True 时条纹相位只在很小的范围内变化（相当于已经对齐好），False 时相位完全随机；amp 是缺陷的灰度幅度。"""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:64, 0:64]
    phase = rng.uniform(0, 0.3) if aligned else rng.uniform(0, 2 * np.pi)
    img = 0.5 + 0.3 * np.sin(2 * np.pi * x / 8 + phase) + rng.normal(scale=0.06, size=(64, 64))
    mask = np.zeros((64, 64), bool)
    if defect:
        cy, cx = rng.integers(10, 54, 2); r = rng.integers(2, 4)
        mask = (y - cy) ** 2 + (x - cx) ** 2 <= r * r
        img[mask] += rng.choice([-1.0, 1.0]) * amp
    return img, mask

def patch_features(img, p=4, agg="edge"):
    """把每个 4x4 patch（步长 2）展平成 16 维向量，再做 3x3 邻域平均（模仿 PatchCore 的局部聚合）。

    agg="edge"：图像边界上的 patch 邻域不完整，用复制边缘的办法补齐（默认）；
    agg="valid"：只对真实存在的邻居平均；agg=None：不做聚合。后两种只在“对照 1”里用到。
    """
    pats = np.lib.stride_tricks.sliding_window_view(img, (p, p))[::2, ::2]
    feats = pats.reshape(pats.shape[0], pats.shape[1], -1).astype(np.float64)
    if agg is None:
        return feats
    gh, gw = feats.shape[:2]
    if agg == "valid":
        pad, ones = np.pad(feats, ((1, 1), (1, 1), (0, 0))), np.pad(np.ones((gh, gw)), 1)
        count = sum(ones[dy:dy + gh, dx:dx + gw] for dy in range(3) for dx in range(3))
        return sum(pad[dy:dy + gh, dx:dx + gw] for dy in range(3) for dx in range(3)) / count[..., None]
    pad = np.pad(feats, ((1, 1), (1, 1), (0, 0)), mode="edge")
    return sum(pad[dy:dy + gh, dx:dx + gw] for dy in range(3) for dx in range(3)) / 9

def score_maps(memory, images, agg="edge"):
    """对每张图算 31x31 的 patch 分数图，返回 (n, 31, 31)。"""
    out = []
    for im in images:
        f = patch_features(im, agg=agg)
        out.append(mod.knn_scores(memory, f.reshape(-1, 16), k=1).reshape(f.shape[:2]))
    return np.stack(out)

train = [make_image(s)[0] for s in range(20)]
bank = np.concatenate([patch_features(im).reshape(-1, 16) for im in train])
print("记忆库大小：", bank.shape)'''),
        ("code", '''idx = attempt(mod.greedy_coreset, bank, max(1, len(bank) // 20))
img_scores = None
if idx is not None:
    try:
        memory = bank[idx]
        print("核心集大小：", memory.shape)
        test = [make_image(1000 + s, defect=(s % 2 == 0)) for s in range(40)]
        labels = np.array([int(m.any()) for _, m in test])
        maps = score_maps(memory, [im for im, _ in test])
        img_scores = maps.reshape(len(maps), -1).max(axis=1)
        print("Image AUROC：", round(mod.auroc(img_scores, labels), 4))
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        for i in range(4):
            ax[0, i].imshow(test[i][0], cmap="gray"); ax[0, i].contour(test[i][1], colors="c")
            ax[1, i].imshow(maps[i], cmap="magma"); ax[1, i].set_title(f"分数 {img_scores[i]:.2f}")
        plt.show()
    except NotImplementedError as exc:
        img_scores = None
        print("先完成作业里剩下的函数：", exc)'''),
        ("md", "## 对照 1：良品的最高分落在哪（局部聚合的边界问题）\n\n"
               "上面第二行的分数图里，没有缺陷的图最亮的地方总在最外一圈。原因在 `patch_features`：图像边界上的 patch 没有完整的 3×3 邻域，"
               "默认用复制边缘的办法补齐，补出来的邻居就是它自己，平均时噪声压得没有内部那么低，分数就偏高。"
               "这是讲义第 3.2 节的内容，所以 Notebook 保留了这种补法，没有把它“修掉”。\n\n"
               "上面那组数据是对齐的，缺陷也明显，这个毛病不影响 AUROC。下面的单元先数一下“良品图的最高分有多少落在最外一圈”，"
               "再换成难一点的数据（条纹相位完全随机、缺陷幅度 0.12），比较三种聚合方式和两种取整图分数的办法。\n\n"
               "先预测：换成“只对真实存在的邻居平均”，最外一圈的问题会消失吗？去掉最外一圈之后，聚合和不聚合哪个 AUROC 高？"),
        ("code", '''RING = np.ones((31, 31), bool); RING[1:-1, 1:-1] = False      # 31x31 分数图的最外一圈

def ring_report(score_stack, y):
    """返回：良品图的最高分落在最外一圈的比例、全图取最大的 image AUROC、去掉最外一圈再取最大的 image AUROC。"""
    flat = score_stack.reshape(len(score_stack), -1)
    on_ring = RING.ravel()[flat[y == 0].argmax(axis=1)].mean()
    return on_ring, mod.auroc(flat.max(axis=1), y), mod.auroc(flat[:, ~RING.ravel()].max(axis=1), y)

hard = None
if img_scores is not None:
    print(f"最外一圈占 patch 总数的 {RING.mean():.1%}")
    r = ring_report(maps, labels)
    print(f"上面那组数据（对齐、缺陷幅度 0.2）：良品最高分落在最外一圈的比例 {r[0]:.0%}，AUROC 全图取最大 {r[1]:.4f}，去掉最外一圈 {r[2]:.4f}")
    hard_train = [make_image(s, aligned=False, amp=0.12)[0] for s in range(20)]
    hard_test = [make_image(2000 + s, defect=(s % 2 == 0), aligned=False, amp=0.12) for s in range(100)]
    hard_labels = np.array([int(m.any()) for _, m in hard_test])
    hard = {}
    print("换成不对齐、缺陷幅度 0.12 的数据（20 张良品，100 张待检图，贪心核心集 10%）：")
    for agg, name in (("edge", "聚合，复制填充（默认）"), ("valid", "聚合，只对真实存在的邻居平均"), (None, "不聚合")):
        hb = np.concatenate([patch_features(im, agg=agg).reshape(-1, 16) for im in hard_train])
        order = mod.greedy_coreset(hb, len(hb) // 10)
        hm = score_maps(hb[order], [im for im, _ in hard_test], agg=agg)
        hard[agg] = (hb, order, hm)
        r = ring_report(hm, hard_labels)
        print(f"  {name}：良品最高分落在最外一圈的比例 {r[0]:.0%}，AUROC 全图取最大 {r[1]:.3f}，去掉最外一圈 {r[2]:.3f}")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, (agg, name) in zip(ax, (("edge", "复制填充"), ("valid", "只对真实邻居平均"), (None, "不聚合"))):
        im = a.imshow(hard[agg][2][hard_labels == 0].mean(axis=0), cmap="viridis"); a.set_title(name); a.axis("off")
        plt.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("良品的平均分数图（31x31 个 patch，三栏的色标范围不同）"); plt.show()
    print("对照讲义第 3.2 节的表读这几行。100 张待检图时 AUROC 的最小变化是 0.0004，一两个百分点以内的差别不要当真。")'''),
        ("md", "## 实验：从 AUROC 到产线工作点\n\n要求漏检率不超过 0%、5%、10% 时，过检率分别是多少？再看核心集比例从 100% 压到 1%，AUROC 掉了多少，想想为什么掉得这么少。"),
        ("code", '''if img_scores is not None:
    for esc in (0.0, 0.05, 0.10):
        r = attempt(mod.threshold_at_escape, img_scores[labels == 0], img_scores[labels == 1], esc)
        if r: print(f"漏检上限 {esc:.0%}：阈值 {r[0]:.3f}，实际漏检 {r[1]:.0%}，过检 {r[2]:.0%}")
    for frac in (1.0, 0.2, 0.05, 0.01):
        m = max(1, int(len(bank) * frac))
        mem = bank[mod.greedy_coreset(bank, m)] if frac < 1 else bank
        sc = score_maps(mem, [im for im, _ in test]).reshape(len(test), -1).max(axis=1)
        print(f"核心集比例 {frac:>5.0%}：AUROC {mod.auroc(sc, labels):.4f}")
    print("这组数据是对齐的，缺陷也明显，压到 1% 几乎看不出差别。40 张待检图时 AUROC 的最小变化是 0.0025，上面几行的差别只是一两对样本。")
    print("不要据此认为“核心集可以随便压”，下一个单元换一组数据再压一次。")'''),
        ("md", "## 对照 2：换一组难一点的数据再压核心集\n\n"
               "还是“对照 1”里那组不对齐、缺陷幅度 0.12 的数据，整图分数去掉最外一圈。把核心集从 10% 压到 0.3%，"
               "同时和“随机抽同样多的点”比较。先预测：贪心选出来的核心集覆盖半径更小，它的 AUROC 一定比随机抽的高吗？（讲义第 3.3 节）"),
        ("code", '''if hard is not None:
    hb, order, _ = hard["edge"]
    rng = np.random.default_rng(0)
    inner = lambda s: s[:, 1:-1, 1:-1].reshape(len(s), -1).max(axis=1)
    hard_imgs = [im for im, _ in hard_test]
    print(f"不对齐、缺陷幅度 0.12，记忆库 {len(hb)} 个 patch，100 张待检图，整图分数去掉最外一圈")
    for frac in (0.10, 0.03, 0.01, 0.003):
        m = max(1, int(len(hb) * frac))
        greedy = mod.auroc(inner(score_maps(hb[order[:m]], hard_imgs)), hard_labels)      # 贪心的前 m 个点就是大小为 m 的核心集
        randoms = [mod.auroc(inner(score_maps(hb[rng.choice(len(hb), m, replace=False)], hard_imgs)), hard_labels) for _ in range(3)]
        print(f"  保留 {frac:>5.1%}（{m:>4} 个点）：贪心 AUROC {greedy:.3f}，随机抽 3 次 " + "、".join(f"{v:.3f}" for v in randoms))
    print("和上一个单元对比：同样压到 1%，这里的 AUROC 比保留 10% 时低了多少？贪心每一行都比随机抽的高吗？")
    print("随机抽的 3 次之间也有波动，只看方向，不看第三位小数。讲义第 3.3 节的表是同一种设置下 3 个种子、每个 200 张待检图的平均。")'''),
        ("md", "## 选做：换成预训练 ResNet 特征，拼成完整的 PatchCore\n\n"
               "前面的“特征”是 4×4 的像素块。真正的 PatchCore 用预训练 CNN 的中间层：下面的单元取 ResNet-18 的 layer2 和 layer3，"
               "各做 3×3 平均（局部聚合），把 layer3 上采样到 layer2 的尺寸后拼成 384 维，其余步骤全部调用你在作业里写的函数："
               "记忆库 → `greedy_coreset` → `knn_scores` → 上采样成热图 → `auroc` 和 `threshold_at_escape`。\n\n"
               "需要 torch、torchvision 和 ResNet-18 的 ImageNet 权重（约 45 MB）。本机没有权重时，这个单元打印说明后跳过，不会报错，也不会自己去联网；"
               "想下载就把单元第一行的 `ALLOW_DOWNLOAD` 改成 `True`。有显卡时自动用显卡，没有就用 CPU，这个规模在 CPU 上几秒钟。\n\n"
               "在 MVTec AD 上跑时，建议直接用 Anomalib，把它的结果和这里的迷你版对照。"),
        ("code", '''ALLOW_DOWNLOAD = False      # 本机没有 ResNet-18 权重时，改成 True 允许 torchvision 联网下载一次

def patchcore_resnet(train_imgs, test_pairs, keep=0.10):
    try:
        import torch
        import torch.nn.functional as F
        import torchvision
    except ImportError as exc:
        print("跳过：没有装 torch 或 torchvision（" + str(exc) + "）。装好后重新运行这个单元。")
        return None
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    ckpt = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    if not ckpt.exists() and not ALLOW_DOWNLOAD:
        print("跳过：本机没有 ResNet-18 的预训练权重，这一节没有运行。")
        print("  找的位置：", ckpt)
        print("  想运行：把本单元第一行的 ALLOW_DOWNLOAD 改成 True 再运行（需要联网，约 45 MB，只下载一次）。")
        return None
    try:
        net = torchvision.models.resnet18(weights=weights).eval()
    except Exception as exc:
        print("跳过：权重加载失败（" + type(exc).__name__ + "：" + str(exc) + "）。检查网络，或者把权重文件放到", ckpt)
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def extract(images):
        """(n, 64, 64) 的灰度图 → (n, 16, 16, 384) 的 patch 特征。"""
        x = torch.tensor(np.stack(images), dtype=torch.float32, device=device)[:, None]
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False).clamp(0, 1).repeat(1, 3, 1, 1)   # 放大到 128x128，复制成 3 通道
        h = net.maxpool(net.relu(net.bn1(net.conv1((x - mean) / std))))
        f2 = net.layer2(net.layer1(h))                                   # 16x16，128 维
        f3 = net.layer3(f2)                                              # 8x8，256 维
        f2, f3 = F.avg_pool2d(f2, 3, 1, 1), F.avg_pool2d(f3, 3, 1, 1)    # 局部聚合：3x3 邻域平均
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1).permute(0, 2, 3, 1).cpu().numpy().astype(np.float64)

    y = np.array([int(m.any()) for _, m in test_pairs])
    f_train = extract(train_imgs)                                        # 1. 良品特征
    n, gh, gw, d = f_train.shape
    bank_r = f_train.reshape(-1, d)                                      # 2. 记忆库
    core = mod.greedy_coreset(bank_r, max(1, int(len(bank_r) * keep)))   # 3. 核心集
    f_test = extract([im for im, _ in test_pairs])
    smap = mod.knn_scores(bank_r[core], f_test.reshape(-1, d), k=1).reshape(len(test_pairs), gh, gw)   # 4. 最近邻打分
    score = smap[:, 1:-1, 1:-1].reshape(len(smap), -1).max(axis=1)       # 5. 整图分数：去掉最外一圈再取最大（对照 1）
    heat = F.interpolate(torch.tensor(smap)[:, None], size=(64, 64), mode="bilinear", align_corners=False)[:, 0].numpy()   # 6. 上采样成热图
    print(f"设备 {device}｜patch 特征 {gh}x{gw}，{d} 维｜记忆库 {len(bank_r)} → 核心集 {len(core)}")
    print("Image AUROC（ResNet-18 layer2 + layer3）：", round(mod.auroc(score, y), 4))
    thr, esc, ovk = mod.threshold_at_escape(score[y == 0], score[y == 1], 0.05)
    print(f"漏检上限 5%：阈值 {thr:.3f}，实际漏检 {esc:.0%}，过检 {ovk:.0%}")
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(4):
        ax[0, i].imshow(test_pairs[i][0], cmap="gray"); ax[0, i].contour(test_pairs[i][1], colors="c"); ax[0, i].axis("off")
        ax[1, i].imshow(heat[i], cmap="magma", vmin=heat[:4].min(), vmax=heat[:4].max()); ax[1, i].set_title(f"分数 {score[i]:.2f}"); ax[1, i].axis("off")
    plt.show()
    return score

if img_scores is not None:
    try:
        resnet_scores = patchcore_resnet(train, test)
        if resnet_scores is not None:
            print("和前面 16 维像素块的结果比一比。缺陷只有两三个像素时，layer2 的一个位置对应 8x8 的输入像素，缺陷被稀释，")
            print("预训练特征在这种合成条纹上不一定更好（讲义第 3.7 节）；它的价值要在纹理复杂的真实图像上才显出来。")
    except NotImplementedError as exc:
        print("先完成作业里剩下的函数：", exc)'''),
    ],
    "## 破坏实验与记录\n\n- 训练集里混入 2 张带缺陷的图，AUROC 怎么变？为什么？\n- 去掉局部聚合（`patch_features(im, agg=None)`，直接用 16 维 patch），结果怎么变？\n- 把漏检约束下的过检率写进 `实验记录.md`。",
)

BOOKS["20_合成缺陷与生成模型/实验_缺陷合成.ipynb"] = notebook(
    "第 20 章互动实验：合成缺陷与扩散前向过程",
    "比较“先插缺陷再成像”和几种“成像后再贴缺陷”的做法，把锐度、噪声、峰值三件事拆开看；再可视化 DDPM 的前向加噪，并量出每一步还剩多少信号。",
    "20_合成缺陷与生成模型", "20-1_缺陷合成与扩散前向.py",
    [
        ("md", "## 实验 1：缺陷在成像链的哪一步插入\n\n"
               "同一条划痕画五遍。第一种是先画到版图上，再一起经过光学模糊和噪声（成像前插入）；后四种都是在已经成像的良品图上再贴，"
               "它们和第一种的差别可以拆成三件事：\n\n"
               "| 做法 | 边缘锐度 | 划痕上的噪声 | 峰值 |\n|---|---|---|---|\n"
               "| A 成像前插入 | 被光学模糊过 | 和背景一样 | 被模糊压低 |\n"
               "| B 成像后替换成固定灰度 | 锐 | 没有 | 高 |\n"
               "| C 同 B，但把噪声补回去 | 锐 | 有 | 高 |\n"
               "| D 同 B，但把灰度降到 A 的峰值 | 锐 | 没有 | 和 A 一样 |\n"
               "| E 掩膜先按光学模糊的 sigma 模糊再贴 | 和 A 接近 | 少一些 | 和 A 接近 |\n\n"
               "B 是最常见的“贴纸”，它同时改了三件事，只看 A 和 B 分不出是哪一件让它显得假。C、D、E 各改回一件。"
               "先预测：C、D、E 里哪一张看起来最接近 A？再运行，对照打印出来的三列数字。"),
        ("code", '''def blur(img, sigma):
    fy = np.fft.fftfreq(img.shape[0])[:, None]; fx = np.fft.fftfreq(img.shape[1])[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2 * (np.pi * sigma) ** 2 * (fy**2 + fx**2))))

SIGMA, NOISE, GRAY = 1.5, 0.03, 0.95        # 光学模糊、噪声标准差、划痕的灰度
rng = np.random.default_rng(0)
y, x = np.mgrid[0:96, 0:96]
clean = 0.3 + 0.4 * ((x % 10) < 5)
m = attempt(mod.scratch_mask, (96, 96), (20.0, 15.0), (70.0, 80.0), 2.0, 1.0)
variants = None
if m is not None:
    try:
        noise = rng.normal(scale=NOISE, size=clean.shape)
        good0 = blur(clean, SIGMA)                           # 没有噪声的良品图
        a0 = blur(mod.blend(clean, GRAY, m), SIGMA)          # A：先画到版图上，再模糊
        core = m >= 1.0                                      # 划痕的中心线附近
        peak_a = float(np.mean(a0[core]))                    # A 在划痕中心的平均灰度
        b0 = mod.blend(good0, GRAY, m)
        d0 = mod.blend(good0, peak_a, m)
        e0 = mod.blend(good0, GRAY, blur(m, SIGMA).clip(0, 1))
        # 每项：(没有噪声的图, 带噪声的图)。B、D、E 是贴在带噪声的良品图上，所以掩膜盖住的地方噪声被盖掉了
        variants = {
            "A 成像前插入": (a0, a0 + noise),
            "B 成像后替换成固定灰度": (b0, mod.blend(good0 + noise, GRAY, m)),
            "C 同 B，补回噪声": (b0, b0 + noise),
            "D 同 B，灰度降到 A 的峰值": (d0, mod.blend(good0 + noise, peak_a, m)),
            "E 掩膜先模糊再贴": (e0, mod.blend(good0 + noise, GRAY, blur(m, SIGMA).clip(0, 1))),
        }
    except NotImplementedError as exc:
        print("先完成作业里剩下的函数：", exc)
if variants is not None:
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.4))
    ax[0, 0].imshow(m, cmap="gray"); ax[0, 0].set_title("软掩码（标签）")
    for a, (name, (_, img)) in zip(ax.ravel()[1:], variants.items()):
        a.imshow(img, cmap="gray", vmin=0.1, vmax=1.05); a.set_title(name)
    for a in ax.ravel(): a.axis("off")
    plt.tight_layout(); plt.show()
    near = (blur(m, 3.0) > 0.02)                             # 划痕附近的一条带子，用来量边缘
    print("三件事各自的数字（越接近 A 那一行，这件事就越像真的成像）：")
    print("  边缘最大梯度  划痕上的噪声标准差  划痕中心的平均灰度   做法")
    for name, (img0, img) in variants.items():
        gy, gx = np.gradient(img0)
        print(f"  {np.hypot(gy, gx)[near].max():>10.3f}{np.std((img - img0)[core]):>16.3f}{np.mean(img0[core]):>18.3f}       {name}")
    print("读法：C 只把噪声改回来了，D 只把峰值改回来了，E 把边缘（连带峰值）改回来了。哪一张图看起来最不像贴纸？")
    print("这个单元只能告诉你每种做法改了图像的哪些统计量。哪一项真的影响检测器，要训练分类器来量：")
    print("讲义第 3.2 节的结果是，起作用的主要是锐度和随之而来的峰值，不是噪声。")'''),
        ("md", "## DDPM 前向过程\n\n预测：t=300 时图像还能认出条纹吗？\n\n"
               "“认不认得出”要有个判断办法。下面的单元除了画图，还对每个 t 打印两个相关系数：\n\n"
               "- **逐像素**：x_t 和 x_0 逐个像素比。它回答“只看一个像素，能不能分出这里原来是亮还是暗”。\n"
               "- **按条纹平均**：先把每条条纹（5 列宽、96 行长）里的像素平均成一个数，再和 x_0 比。它回答“看一整条，能不能分出亮暗”，眼睛做的更接近这一种。\n\n"
               "先写下你对 t=300 时这两个数的预测，再运行。两个数差多少，说明“某一步还剩多少信息”取决于你问的是多大的结构（讲义第 5.3.1 节和第 5.3.3 节）。"),
        ("code", '''betas = attempt(mod.linear_beta_schedule, 1000)
if betas is not None:
    try:
        abar = mod.alpha_bars(betas)
        x0 = (clean - 0.5) * 2
        noise = np.random.default_rng(1).normal(size=x0.shape)
        ts = [0, 50, 150, 300, 600, 999]
        stripe_mean = lambda img: img[:, :95].reshape(96, 19, 5).mean(axis=(0, 2))    # 前 19 条完整的条纹，每条 5 列
        fig, ax = plt.subplots(1, len(ts), figsize=(15, 3))
        rows = []
        for a, t in zip(ax, ts):
            xt = mod.q_sample(x0, t, abar, noise)
            a.imshow(xt, cmap="gray"); a.set_title(f"t={t}, alpha_bar={abar[t]:.3f}"); a.axis("off")
            rows.append((t, abar[t], np.corrcoef(xt.ravel(), x0.ravel())[0, 1], np.corrcoef(stripe_mean(xt), stripe_mean(x0))[0, 1]))
        plt.show()
        print("     t   alpha_bar   逐像素的相关系数   按条纹平均后的相关系数")
        for t, ab, c_pix, c_stripe in rows:
            print(f"  {t:>4}{ab:>12.4f}{c_pix:>14.3f}{c_stripe:>18.3f}")
        print("条纹取值是 -0.4 和 0.4。逐像素的相关系数按公式是 sqrt(ab * 0.16 / (ab * 0.16 + 1 - ab))，ab 是 alpha_bar，可以自己代一个 t 核对；")
        print("按条纹平均后的那一列只有 19 条条纹，换一份噪声会有波动。")
        plt.plot(np.sqrt(abar), label="信号系数 sqrt(alpha_bar)"); plt.plot(np.sqrt(1 - abar), label="噪声系数 sqrt(1 - alpha_bar)")
        plt.xlabel("t"); plt.legend(); plt.show()
    except NotImplementedError as exc:
        print("先完成作业里剩下的函数：", exc)'''),
        ("md", "## VAE 重参数化：采样的梯度从哪里来"),
        ("code", '''z = attempt(mod.reparameterize, np.zeros(5), np.log(np.full(5, 4.0)), np.ones(5))
if z is not None:
    print("mu=0、方差 4、eps=1 时 z =", z, "（dz/dmu = 1，dz/dlogvar = 0.5·sigma·eps，都和 eps 的采样无关）")'''),
        ("md", "## 破坏实验 2 的起步代码：只用一种缺陷调阈值\n\n"
               "讲义第 11 节的破坏实验 2：用第 24 章的 `数据生成.py`，只在带划痕的 die 上调一个差值阈值，再用这个阈值去测颗粒和图形缺失，检出率差多少？\n\n"
               "下面的单元把数据准备好：借用第 24 章生成器里的版图和三种缺陷（划痕、颗粒、图形缺失），每张图放一个缺陷，经过模糊和噪声，"
               "再给每张图算一个分数（和无噪声参考图的差，先做 3×3 平均，再取绝对值的最大值）。为了让这个实验只看“缺陷种类”这一件事，"
               "这里没有加第 24 章的随机错位和亮度漂移，那两件事留给第 24 章。这个单元不依赖本章作业。\n\n"
               "剩下的三步由你来写，单元末尾有提示。"),
        ("code", '''def load_ch24_generator():
    path = COURSE_ROOT / "24_综合项目_晶圆检测流水线" / "数据生成.py"
    if not path.exists():
        print("跳过：没有找到", path)
        return None
    spec = importlib.util.spec_from_file_location("ch24_datagen", path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    if not all(hasattr(gen, name) for name in ("make_layout", "_add_defect", "_gaussian_blur")):
        print("跳过：第 24 章的 数据生成.py 里没有找到 make_layout、_add_defect、_gaussian_blur，可能已经改过名字。")
        return None
    return gen

def box3(img):
    """3x3 平均，边界用复制填充。"""
    p = np.pad(img, 1, mode="edge")
    return sum(p[dy:dy + img.shape[0], dx:dx + img.shape[1]] for dy in range(3) for dx in range(3)) / 9

gen = load_ch24_generator()
kind_scores = None
if gen is not None:
    layout = gen.make_layout()
    reference = gen._gaussian_blur(layout, 1.0)               # 无噪声的参考图
    rng24 = np.random.default_rng(24)

    def make_die(kind=None):
        """一张 die：kind 为 None 是良品，否则放一个指定种类的缺陷（先加缺陷，再模糊，再加噪声）。"""
        img = layout.copy()
        if kind is not None:
            cy, cx = rng24.uniform(12, layout.shape[0] - 12, size=2)
            gen._add_defect(img, kind, cy, cx, rng24)
        return gen._gaussian_blur(img, 1.0) + rng24.normal(scale=0.02, size=img.shape)

    score = lambda img: float(np.abs(box3(img - reference)).max())
    kind_scores = {name: np.array([score(make_die(kind)) for _ in range(200)])
                   for name, kind in (("良品", None), ("划痕", "scratch"), ("颗粒", "particle"), ("图形缺失", "missing"))}
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bins = np.linspace(0, max(v.max() for v in kind_scores.values()), 60)
    for name, v in kind_scores.items():
        ax.hist(v, bins=bins, alpha=0.55, label=name)
        print(f"{name}：200 张，分数的最小值 {v.min():.3f}，中位数 {np.median(v):.3f}，最大值 {v.max():.3f}")
    ax.set_xlabel("分数（和参考图的差，3x3 平均后取绝对值的最大值）"); ax.set_ylabel("张数"); ax.legend(); plt.show()
    print("接下来由你来写：")
    print("  1. 只看 kind_scores[\\"划痕\\"]，选一个阈值，让划痕的漏检率不超过 5%（阈值越大过检越少）。")
    print("  2. 用这个阈值统计良品的过检率，以及颗粒、图形缺失的检出率。")
    print("  3. 再只用“图形缺失”调一次阈值，良品的过检率变成多少？哪一种缺陷最容易被“只用一种调参”坑到？")
    print("  提示：作业第 19 章的 threshold_at_escape 做的就是第 1 步；没做第 19 章就用 np.quantile(划痕分数, 0.05)。")'''),
    ],
    "## 破坏实验与记录\n\n- 把软掩码的 softness 设得很小（接近硬边），再看实验 1 的 A 和 B 两张图：哪一张变了，哪一张几乎没变？\n- 完成上面“破坏实验 2 的起步代码”后面的三步：只用划痕调阈值，再测颗粒和图形缺失，检出率差多少？\n- 把 `linear_beta_schedule` 的步数改成 100、β 范围不变，重跑 DDPM 那个单元（`ts` 也要改），看最后一步的两个相关系数。\n- 记录到 `实验记录.md`。",
)

BOOKS["21_3D数学与渲染管线/实验_软光栅.ipynb"] = notebook(
    "第 21 章互动实验：用自己的光栅器渲染立方体",
    "调用你实现的 look_at、perspective、to_screen、render_depth，渲染深度图，再加上逐面 Lambert 光照。",
    "21_3D数学与渲染管线", "21-1_软光栅.py",
    [
        ("code", '''V = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
              [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
# 面表和 demos/渲染管线.html 相同：12 个三角形，每个从立方体外面看都是逆时针。
# 这样 cross(V[b] - V[a], V[c] - V[a]) 算出的法线朝外，后面的光照才是对的。
FACES = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
         (3, 6, 2), (3, 7, 6), (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5)]
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
        ("md", "## Lambert 光照\n\n逐个三角形单独光栅化，再用深度最小者决定每个像素属于哪个面，乘上 max(0, n·l)。\n\n"
               "法线由面表的绕序决定：`n = cross(V[b] - V[a], V[c] - V[a])`。三角形从外面看是逆时针时，这样算出的法线朝外。"
               "下面的单元先数一下 12 条法线里有几条朝外（应该是 12 条），再画图，最后打印可见面上出现的亮度值。"
               "从 (3, 2.5, 5) 能看到三个面，法线分别朝 +x、+y、+z，三个面的亮度应该各不相同，朝上的面最亮。"
               "如果三个面一样暗（亮度都是 0.15），说明法线朝里了，先查面表的绕序（讲义第 4 节“绕序和背面剔除”）。"),
        ("code", '''if out is not None:
    light = np.array([0.4, 0.8, 0.6]); light /= np.linalg.norm(light)
    normals = np.array([np.cross(V[b] - V[a], V[c] - V[a]) for a, b, c in FACES])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    centers = np.array([V[list(f)].mean(axis=0) for f in FACES])        # 立方体的中心在原点，所以“面的中心”本身就是朝外的方向
    print(f"法线朝外的三角形：{int(np.sum(np.sum(normals * centers, axis=1) > 0))}/{len(FACES)}")
    per_face = np.stack([mod.render_depth(tris[i:i + 1], W, H) for i in range(len(FACES))])
    face_id = np.argmin(per_face, axis=0); covered = np.isfinite(per_face.min(axis=0))
    face_shade = 0.15 + 0.85 * np.maximum(0.0, normals @ light)
    shade = np.zeros((H, W))
    shade[covered] = face_shade[face_id[covered]]
    plt.imshow(shade, cmap="gray", vmin=0, vmax=1); plt.title("逐面 Lambert"); plt.show()
    print("可见的像素数：", int(covered.sum()), "｜可见面上出现的亮度值：", sorted({round(float(v), 3) for v in shade[covered]}))'''),
        ("md", "## 实验：near 平面与深度精度"),
        ("code", '''if out is not None:
    z = np.linspace(-0.5, -20, 400)
    for near in (0.01, 0.5, 2.0):
        P = mod.perspective(50, 1.0, near, 20.0)
        d = mod.to_screen(np.stack([np.zeros_like(z), np.zeros_like(z), z], 1), P, 2, 2)[:, 2]
        plt.plot(-z, d, label=f"near={near}")
    plt.xlabel("到相机的距离"); plt.ylabel("屏幕深度"); plt.legend(); plt.title("深度是非线性的"); plt.show()'''),
    ],
    "## 破坏实验与记录\n\n- 把 up 改成 (0, -1, 0)，图像怎样？\n- 把相机放到立方体内部，会发生什么？`to_screen` 的 NaN 处理在哪里起作用？\n- 把 `FACES` 里每个三角形的后两个下标对调（绕序反过来），深度图和“逐面 Lambert”各有什么变化？先预测再运行。\n- 把 OpenGL 和 OpenCV 相机约定的对照写进 `实验记录.md`。",
)

BOOKS["22_成像与设备仿真/实验_成像仿真.ipynb"] = notebook(
    "第 22 章互动实验：能不能看见这个缺陷",
    "用你的成像仿真器回答一个设计问题：在给定光学和曝光下，多大的缺陷能被检出。",
    "22_成像与设备仿真", "22-1_成像仿真.py",
    [
        ("code", '''for lam, na in [(532, 0.5), (532, 0.9), (266, 0.9), (193, 0.95)]:
    r = attempt(mod.rayleigh_resolution_um, lam, na)
    if r is None: break
    print(f"波长 {lam} nm, NA {na}：分辨率 {r:.3f} um，建议像素（物面）不大于 {r / 2:.3f} um")'''),
        ("md", "## 实验 1：缺陷信噪比 vs 缺陷尺寸\n\n场景：反射率 0.5 的平坦背景上一个亮颗粒（比背景高 0.3）。PSF sigma 固定，改变颗粒直径和曝光量。\n\n- 颗粒按**覆盖比例**画：把每个像素切成 16×16 个小格，像素的值取落在圆里的小格所占的比例（`disk_coverage`）。如果只看像素中心在不在圆里，直径 0.5、1、1.5 像素都只点亮同一个像素，曲线的前三个点会完全相同（教程第 12.1 节）。\n- 曝光量用“反射率 1 对应的平均电子数”表示（`exposure_e`）。作业的模型里没有满阱这个参数，饱和只发生在 ADU 超过 `2**bit_depth - 1` 的时候。这里是 12 位、增益 2 e-/ADU，上限 4095 ADU 对应 8190 e-，下面三档曝光都没有到。"),
        ("code", '''def disk_coverage(diam_px, size=41, sub=16):
    """圆心在正中那个像素的中心、直径 diam_px（像素）的圆：返回每个像素被圆盖住的比例，0 到 1。"""
    c = (size - 1) / 2
    t = (np.arange(size * sub) + 0.5) / sub - 0.5          # 每个小格中心的坐标，单位是像素
    inside = (t[:, None] - c) ** 2 + (t[None, :] - c) ** 2 <= (diam_px / 2) ** 2
    return inside.reshape(size, sub, size, sub).mean(axis=(1, 3))

def defect_snr(diam_px, exposure_e, psf_sigma=1.5, contrast=0.3, trials=30):
    scene = 0.5 + contrast * disk_coverage(diam_px)
    k = mod.gaussian_psf(psf_sigma)
    sharp = mod.convolve_same(scene, k)
    bg = mod.convolve_same(np.full_like(scene, 0.5), k)
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(trials):
        a = mod.simulate_capture(sharp * exposure_e, 5.0, 2.0, 12, rng).astype(float)
        b = mod.simulate_capture(bg * exposure_e, 5.0, 2.0, 12, rng).astype(float)
        diffs.append((a - b)[20, 20])
    noise = np.std([mod.simulate_capture(bg * exposure_e, 5.0, 2.0, 12, rng)[20, 20] for _ in range(200)]) * np.sqrt(2)
    return np.mean(diffs) / noise

diams = [0.5, 1, 1.5, 2, 3, 4, 6]
print("各直径的覆盖面积（像素）：", [round(float(disk_coverage(d).sum()), 2) for d in diams])
for exposure_e in (500, 2000, 6000):
    snr = attempt(lambda: [defect_snr(d, exposure_e) for d in diams])
    if snr is None: break
    plt.plot(diams, snr, "o-", label=f"反射率 1 对应 {exposure_e} e-")
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
    est = attempt(mod.estimate_gain_from_ptc, np.array(means)[ok], np.array(vars_)[ok])
    if est is not None:
        print("只用未饱和点拟合：", est, "真值 (1.5, 8.0)")'''),
        ("md", "## 实验 3：暗处的统计量和黑电平\n\n读出噪声有正有负。信号很暗时，电子数里负的那一部分换成 ADU 之后被裁成 0，均值被抬高，标准差被压小。`simulate_capture` 的 `black_level_adu` 在量化之前加一个固定偏置，把整个分布抬离 0。\n\n下面拍全暗的一帧，读出噪声 8 e-、增益 1.5 e-/ADU，比较不加黑电平和加 64 ADU 两种情况。先预测：标准差应该是多少 ADU？不加黑电平时会偏大还是偏小？"),
        ("code", '''rng = np.random.default_rng(5)
read_e, gain = 8.0, 1.5
want = np.sqrt((read_e / gain) ** 2 + 1 / 12)          # 读出噪声换算成 ADU，再加四舍五入带来的量化噪声
frames = {}
for black in (0.0, 64.0):
    f = attempt(mod.simulate_capture, np.zeros((300, 300)), read_e, gain, 12, rng, black_level_adu=black)
    if f is None: break
    frames[black] = f
    print(f"黑电平 {black:2.0f} ADU：均值减去黑电平 {f.mean() - black:5.2f} ADU，标准差 {f.std():.2f} ADU（没有裁剪时应约 {want:.2f}），等于 0 的像素占 {np.mean(f == 0):.0%}")
if len(frames) == 2:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for a, (black, f) in zip(ax, frames.items()):
        a.hist(f.ravel() - black, bins=np.arange(-25.5, 26.5)); a.set_xlabel("像素值减去黑电平 (ADU)"); a.set_title(f"黑电平 {black:.0f} ADU")
    plt.show()'''),
    ],
    "## 破坏实验与记录\n\n- 用全部点（含饱和点）拟合 PTC，增益估计偏向哪边？\n- 把 PSF sigma 从 1.5 改成 3（离焦），实验 1 的曲线怎样？\n- 在实验 2 的亮度档里加一档全暗（0 e-），分别不加黑电平和加 64 ADU 做拟合（加了黑电平要先从均值里减掉它），读出噪声的估计差多少？对照教程第 6.2 节。\n- 写下你的结论：在你设定的参数下最小可检出缺陷多大，依据是什么。",
)

BOOKS["23_半导体检测领域知识/实验_晶圆图.ipynb"] = notebook(
    "第 23 章互动实验：晶圆图与空间模式",
    "画出 300 mm 晶圆的 die 网格，合成几种典型的缺陷空间分布，看径向密度和边缘比例这两个特征能把其中哪些分开、哪些分不开。",
    "23_半导体检测领域知识", "23-1_晶圆图.py",
    [
        ("code", '''grid = attempt(mod.die_grid, 300.0, 12.0, 14.0, 3.0, (-6.0, -7.0))
if grid is not None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for col, row, cx, cy in grid:
        ax.add_patch(plt.Rectangle((cx - 6, cy - 7), 12, 14, fill=False, lw=0.5))
    ax.add_patch(plt.Circle((0, 0), 150, fill=False, color="k")); ax.add_patch(plt.Circle((0, 0), 147, fill=False, ls="--", color="gray"))
    ax.set_xlim(-155, 155); ax.set_ylim(-155, 155); ax.set_aspect("equal"); ax.set_title(f"完整 die 数：{len(grid)}"); plt.show()'''),
        ("md", "## 四种分布和两个径向特征\n\n下面合成随机、中心、边缘环、划痕四种分布，各算一遍径向密度和边缘比例。运行之前先猜：哪几种的径向密度形状明显不同，哪两种会很像？\n\n**两个半径不一样，是故意留的。** 随机缺陷只撒到半径 147 mm（去边 3 mm 以内，`sample_disk` 的 `rmax`），统计用的却是整片晶圆的半径 150 mm。最外面 3 mm 没有缺陷，所以“随机”这一栏的最外环密度会比里面几环低一成左右，边缘比例约 0.16，而不是缺陷铺满 150 mm 时的 0.19。这不是工艺现象，是撒点范围和统计范围不一致造成的。教程第 8.1 节用它说明“两个范围不一致会凭空造出边缘偏少的假象”，第 12 节的破坏实验 1 让你把两个半径统一后再比。"),
        ("code", '''rng = np.random.default_rng(0)
def sample_disk(n, rmax=147):   # 撒点半径 147 mm：去边以内。下面统计用的半径是 150 mm，见上面的说明
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
    dens = attempt(mod.radial_density, p, 150.0, 6)      # 统计半径 150 mm
    er = attempt(mod.edge_ratio, p, 150.0, 15.0)
    if dens is not None:
        ax[1, i].bar(np.arange(6), dens)
        ax[1, i].set_title("径向密度｜边缘比例 " + (f"{er:.2f}" if er is not None else "还没实现"))
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
    "## 记录\n\n- 四种模式里，径向密度和边缘比例把哪些分开了？\n- 哪两种模式只靠径向密度分不开？还需要什么特征？对照教程第 8.1 节。\n- 把入职前术语清单整理进 `实验记录.md`。",
)

BOOKS["24_综合项目_晶圆检测流水线/实验_流水线调试.ipynb"] = notebook(
    "第 24 章：流水线调试",
    "逐步可视化数据和你的流水线输出。验收指标以 `tests/check_progress.py` 为准，这里用于找问题。这里用到的种子都不是验收种子。",
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
        ("code", '''def match(p, tr, r=4.0):
    pairs = sorted((np.hypot(a[0] - b[0], a[1] - b[1]), i, j) for i, a in enumerate(p) for j, b in enumerate(tr))
    up, ut = set(), set()
    for d, i, j in pairs:
        if d <= r and i not in up and j not in ut: up.add(i); ut.add(j)
    return up, ut

preds = attempt(mod.run_pipeline, g.copy(), t.copy())
if preds is not None:
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
        ("md", "## 换几个种子看波动\n\n一个种子只有三十几个缺陷，多漏一两个，检出率就差好几个百分点。验收用 8 个固定种子，要求至少 6 个达标（检出率不低于 85%，每 die 误报不超过 0.3），并且 8 个种子合计也达标。\n\n调参时不要盯着验收种子，用下面这样的其他种子看平均水平。参考实现在很多种子上测，也有少数几个种子的检出率低于 85%（教程第 4 节），所以个别种子不达标不一定是流水线有错，先看合计和多数种子。"),
        ("code", '''rows = []
for seed in range(100, 108):
    d = gen.make_dataset(seed=seed)
    p = attempt(mod.run_pipeline, d["golden"].copy(), d["test"].copy())
    if p is None: break
    tp = fp = fn = 0
    for a, tr in zip(p, d["truth"]):
        up, ut = match(a, tr); tp += len(up); fp += len(a) - len(up); fn += len(tr) - len(ut)
    rows.append((tp, tp + fn, fp / len(p)))
    print(f"种子 {seed}：检出率 {tp / (tp + fn):.1%}（{tp}/{tp + fn}），每 die 误报 {fp / len(p):.2f}")
if rows:
    r = np.array(rows, dtype=float)
    ok = (r[:, 0] / r[:, 1] >= 0.85) & (r[:, 2] <= 0.3)
    print(f"{len(rows)} 个种子合计：检出率 {r[:, 0].sum() / r[:, 1].sum():.1%}（{int(r[:, 0].sum())}/{int(r[:, 1].sum())}），每 die 误报 {r[:, 2].mean():.2f}，达到验收线的种子 {int(ok.sum())}/{len(rows)}")'''),
    ],
    "## 调试顺序建议\n\n1. 先只看对齐：把对齐后的良品两两相减，差分图应该只剩噪声。\n2. 再看亮度归一化：对齐后的图和金模板做散点图，应该落在 y=x 附近。\n3. 再看 z 图：缺陷处应该明显高于周围，图形边缘不应该一片亮。\n4. 最后调 k 和 min_area，并记录到 `项目报告.md`。",
)


def main():
    only = set(sys.argv[1:])  # 例如 python scripts/build_track2_notebooks.py 19 21 只生成这两章
    for rel, nb in BOOKS.items():
        if only and rel[:2] not in only:
            continue
        for i, cell in enumerate(nb.cells):
            cell["id"] = f"c{i:03d}"  # 固定单元 id，重复生成不产生差异
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(nb, path)
        print("写入", rel)


if __name__ == "__main__":
    main()
