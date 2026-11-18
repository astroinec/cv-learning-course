"""第 1 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch01.py

输出：<主分支>/01_像素与色彩/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。
"""

from __future__ import annotations

import json
import math
import platform
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

CH = "01_像素与色彩"
m_eq = S.load_solution(CH, "1-1_直方图均衡化.py")
m_col = S.load_solution(CH, "1-2_通道分离与伪彩色.py")
m_nz = S.load_solution(CH, "1-3_椒盐噪声与中值滤波.py")
NUM: dict = {}


# ---------------------------------------------------------------- 工具
def to_u8(x):
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def r3(v, n=3):
    return round(float(v), n)


def sig(v, n=3):
    return float(f"{float(v):.{n}g}")


def salt_pepper(img, amount, seed):
    """调用作业 1-3 的参考实现。它用 np.random 的全局状态，所以每次调用前固定种子。"""
    np.random.seed(seed)
    return m_nz.add_salt_pepper_noise(img, amount)


def auc(pos, neg):
    """正样本的值大于负样本的概率（并列算一半）。"""
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ends = np.cumsum(cnt)
    rank = ((ends - cnt + ends + 1) / 2.0)[inv]
    return float((rank[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def check_median_reference():
    """扫参数的实验用 cv2.medianBlur 提速，这里先确认它和参考实现逐像素相同（包括边界）。"""
    rng = np.random.default_rng(11)
    img = rng.integers(0, 256, (48, 64), dtype=np.uint8)
    noisy = salt_pepper(to_u8(100 + rng.normal(0, 4, (48, 64))), 0.1, 11)
    same = all(np.array_equal(m_nz.median_filter_manual(x, k), cv2.medianBlur(x, k)) for x in (img, noisy) for k in (3, 5, 7))
    assert same, "参考实现的中值滤波和 cv2.medianBlur 不一致"
    return same


# ---------------------------------------------------------------- 图 1：uint8 相减
def pattern(n=128):
    """合成检测图：暗背景上一组亮线、一块亮焊盘和一块中灰区。"""
    img = np.full((n, n), 60.0)
    y, x = np.mgrid[0:n, 0:n]
    img[(y >= 16) & (y < 56) & (x >= 16) & (x < 112) & ((x % 12) < 6)] = 150.0
    img[72:112, 20:60] = 190.0
    img[72:112, 72:112] = 110.0
    return img


def fig_uint8():
    base = pattern()
    rng = np.random.default_rng(0)
    T, trials, amp, sigma = 20, 200, 30.0, 3.0
    names = ["uint8 直接相减再取绝对值", "cv2.subtract(test, ref)", "cv2.absdiff(test, ref)", "先转 int16 再相减取绝对值"]
    st = {k: {"误报像素": 0, "亮缺陷": 0, "暗缺陷": 0} for k in names}
    bright, dark = np.s_[84:88, 84:88], np.s_[92:96, 36:40]
    dm = np.zeros(base.shape, bool)
    dm[bright] = dm[dark] = True
    example = None
    for t in range(trials):
        ref = to_u8(base + rng.normal(0, sigma, base.shape))
        f = base + rng.normal(0, sigma, base.shape)
        f[bright] += amp
        f[dark] -= amp
        test = to_u8(f)
        diffs = {names[0]: np.abs(test - ref), names[1]: cv2.subtract(test, ref), names[2]: cv2.absdiff(test, ref),
                 names[3]: np.abs(test.astype(np.int16) - ref.astype(np.int16))}
        for k, d in diffs.items():
            hit = d > T
            st[k]["误报像素"] += int((hit & ~dm).sum())
            st[k]["亮缺陷"] += int(hit[bright].mean() >= 0.5)
            st[k]["暗缺陷"] += int(hit[dark].mean() >= 0.5)
        if t == 0:
            example = (ref, test, diffs)
            first = {"待检图比参考图暗的像素比例": r3((test < ref).mean()), "直接相减后数值大于 200 的像素比例": r3((diffs[names[0]] > 200).mean()),
                     "这些回绕像素的数值中位数": float(np.median(diffs[names[0]][test < ref])),
                     "cv2.absdiff 与 int16 做法逐像素相同": bool(np.array_equal(diffs[names[2]], diffs[names[3]]))}
    NUM["uint8 相减"] = {
        "说明": f"128x128 合成检测图，参考图和待检图各带标准差 {sigma:g} 的高斯噪声并量化成 uint8；待检图上有一个 4x4 的亮缺陷（+{amp:g}）和一个 4x4 的暗缺陷（-{amp:g}）；"
              f"差值大于 {T} 算超限，缺陷内至少一半像素超限算检出；共 {trials} 次",
        "每张图的像素数": int(base.size),
        "每张图的误报像素数": {k: r3(v["误报像素"] / trials, 1) for k, v in st.items()},
        "误报像素占全图的比例": {k: sig(v["误报像素"] / trials / base.size) for k, v in st.items()},
        f"亮缺陷检出次数（共 {trials}）": {k: v["亮缺陷"] for k, v in st.items()},
        f"暗缺陷检出次数（共 {trials}）": {k: v["暗缺陷"] for k, v in st.items()},
        "第一次试验": first,
    }

    ref, test, diffs = example
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.9), layout="constrained")
    ax[0].imshow(test, cmap="gray", vmin=0, vmax=255)
    ax[0].plot(85.5, 85.5, "o", mfc="none", mec=S.ORANGE, ms=17, mew=1.6)
    ax[0].plot(37.5, 93.5, "o", mfc="none", mec=S.BLUE, ms=17, mew=1.6)
    ax[0].text(64, 124, "橙圈：亮缺陷  蓝圈：暗缺陷", fontsize=8.5, color="white", ha="center")
    ax[0].set_title("待检图（uint8）")
    im = ax[1].imshow(diffs[names[0]], cmap="magma", vmin=0, vmax=255)
    ax[1].set_title(f"np.abs(test - ref)：{100 * first['直接相减后数值大于 200 的像素比例']:.0f}% 的像素回绕成 200 以上")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03).ax.tick_params(labelsize=8)
    ax[2].imshow(diffs[names[1]], cmap="magma", vmin=0, vmax=40)
    ax[2].plot(37.5, 93.5, "o", mfc="none", mec=S.BLUE, ms=17, mew=1.6)
    ax[2].set_title("cv2.subtract：负数截成 0，暗缺陷不见了")
    im3 = ax[3].imshow(diffs[names[2]], cmap="magma", vmin=0, vmax=40)
    ax[3].set_title("cv2.absdiff：两个缺陷都在")
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.03).ax.tick_params(labelsize=8)
    for a in ax:
        S.image_axes(a)
    S.save(fig, CH, "fig01_uint8相减")


def numbers_dtype():
    a = np.array([[200]], np.uint8)
    b = np.array([[100]], np.uint8)
    lo, hi = np.array([[100]], np.uint8), np.array([[105]], np.uint8)
    try:
        _ = a + 300
        py_int = "没有报错"
    except OverflowError as exc:
        py_int = f"OverflowError：{exc}"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = (a + b, a - b, lo - hi, a * 2)
        array_warn = [str(x.message) for x in w]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scalar = int(np.uint8(200) + np.uint8(100))
        scalar_warn = [str(x.message) for x in w]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f = np.array([300.0, -5.0, 255.6, 256.0])
        cast = f.astype(np.uint8).tolist()
        cast_warn = [str(x.message) for x in w]
    v = np.arange(256, dtype=np.uint8)
    g = (v.astype(np.float32) / 255.0) ** 0.5
    back_trunc = ((g ** 2.0) * 255).clip(0, 255).astype(np.uint8)
    back_round = np.round((g ** 2.0) * 255).clip(0, 255).astype(np.uint8)
    ident = (v.astype(np.float32) / 255.0 * 255).clip(0, 255).astype(np.uint8)
    x = np.random.default_rng(0).uniform(0, 255, 100000)
    big = np.zeros((1080, 1920, 3), np.uint8)
    NUM["uint8 运算规则"] = {
        "说明": f"NumPy {np.__version__}、OpenCV {cv2.__version__}，在 {platform.system()} {platform.machine()} 上实测",
        "数组 200 + 100": int((a + b)[0, 0]), "cv2.add(200, 100)": int(cv2.add(a, b)[0, 0]),
        "数组 100 - 105": int((lo - hi)[0, 0]), "cv2.subtract(100, 105)": int(cv2.subtract(lo, hi)[0, 0]), "cv2.absdiff(100, 105)": int(cv2.absdiff(lo, hi)[0, 0]),
        "数组 200 * 2": int((a * 2)[0, 0]), "数组 200 * 1.5 的结果和类型": [float((a * 1.5)[0, 0]), str((a * 1.5).dtype)],
        "(200 + 200) // 2（两帧取平均的错误写法）": int(((a + a) // 2)[0, 0]), "np.mean([200, 200])": float(np.mean(np.array([200, 200], np.uint8))),
        "uint8 数组之间加、减、乘发生回绕时的警告（空表示没有）": array_warn,
        "uint8 数组 + Python 整数 300": py_int,
        "uint8 标量 200 + 100": [scalar, scalar_warn],
        "不 clip 直接 astype(uint8)：输入": f.tolist(), "不 clip 直接 astype(uint8)：本机输出": cast, "不 clip 直接 astype(uint8)：警告": cast_warn,
        "先 round 再 clip 再转": to_u8(f).tolist(),
        "0 到 255 均匀分布的浮点数：astype 截断的平均偏差": r3((x.astype(np.uint8) - x).mean()), "先 round 的平均偏差": r3((np.round(x) - x).mean()),
        "除以 255 再乘 255 后截断，256 个灰度里变了的个数": int((ident != v).sum()),
        "开方再平方（一次伽马往返）后截断，256 个灰度里变了的个数": int((back_trunc != v).sum()), "同上但先 round": int((back_round != v).sum()),
        "1920x1080 图像的 shape": list(big.shape), "cv2.resize(img, (640, 360)) 的 shape": list(cv2.resize(big, (640, 360)).shape),
        "讲义里的均值例子 (8*100+255)/9": r3((8 * 100 + 255) / 9, 1),
    }


# ---------------------------------------------------------------- 图 2：通道与灰度
def fig_color():
    w = np.array([0.114, 0.587, 0.299])
    bg, stain, dark = np.array([60.0, 140.0, 60.0]), np.array([107.0, 80.0, 160.0]), 12.0
    rng = np.random.default_rng(1)
    n, sigma = 160, 4.0
    img = np.tile(bg, (n, n, 1))
    dA = np.zeros((n, n), bool)
    dB = np.zeros((n, n), bool)
    dA[40:56, 40:56] = True
    dB[100:116, 100:116] = True
    img[dA] = stain
    img[dB] = bg - dark
    noisy = to_u8(img + rng.normal(0, sigma, img.shape))
    gray = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(noisy, cv2.COLOR_BGR2HSV)
    ref = ~(dA | dB)
    chans = {"灰度": gray, "B": noisy[..., 0], "G": noisy[..., 1], "R": noisy[..., 2], "H": hsv[..., 0], "S": hsv[..., 1], "V": hsv[..., 2]}
    cnr = {}
    for k, c in chans.items():
        c = c.astype(float)
        s = c[ref].std()
        cnr[k] = {"背景标准差": r3(s, 2), "色斑": r3((c[dA].mean() - c[ref].mean()) / s, 1), "变暗": r3((c[dB].mean() - c[ref].mean()) / s, 1)}
    NUM["通道与灰度"] = {
        "说明": f"160x160 合成彩色图，背景 BGR=({bg[0]:g},{bg[1]:g},{bg[2]:g})；色斑 BGR=({stain[0]:g},{stain[1]:g},{stain[2]:g})，是特意构造的和背景灰度相同的颜色；"
              f"变暗缺陷是三个通道各减 {dark:g}；每个通道独立加标准差 {sigma:g} 的高斯噪声；对比噪声比 =（缺陷均值 - 背景均值）/ 背景标准差，各 16x16 像素",
        "背景和色斑按 0.299R+0.587G+0.114B 算出的灰度": [r3(bg @ w, 1), r3(stain @ w, 1)],
        "灰度图噪声相对单通道噪声的理论倍数 sqrt(0.299^2+0.587^2+0.114^2)": r3(math.sqrt((w ** 2).sum())),
        "各通道的对比噪声比": cnr,
    }

    # 灰度公式、通道顺序、HSV 的几组数字
    rc = np.random.default_rng(0).integers(0, 256, (400, 400, 3), dtype=np.uint8)
    g = cv2.cvtColor(rc, cv2.COLOR_BGR2GRAY).astype(float)
    f601 = np.round(0.299 * rc[..., 2] + 0.587 * rc[..., 1] + 0.114 * rc[..., 0])
    g_wrong = cv2.cvtColor(rc, cv2.COLOR_RGB2GRAY).astype(float)
    pure = {name: int(cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_BGR2GRAY)[0, 0]) for name, c in (("纯红", (0, 0, 255)), ("纯绿", (0, 255, 0)), ("纯蓝", (255, 0, 0)))}
    t = m_col.create_test_color_image()
    NUM["灰度公式与通道顺序"] = {
        "说明": "400x400 的均匀随机颜色（种子 0）；测试图指作业 1-2 的 create_test_color_image()",
        "cv2 的灰度与四舍五入的 0.299R+0.587G+0.114B 的最大差": int(np.abs(g - f601).max()), "两者不相等的像素比例": sig((g != f601).mean()),
        "纯色的灰度": pure,
        "灰度与 (R+G+B)/3 的最大差": r3(np.abs(g - rc.mean(-1)).max(), 1),
        "把 BGR 图当 RGB 转灰度：平均差和最大差": [r3(np.abs(g - g_wrong).mean(), 1), int(np.abs(g - g_wrong).max())],
        "测试图 B、G、R 三通道的均值": [r3(t[..., i].mean(), 1) for i in range(3)],
        "测试图红色块里一个像素的 BGR 值": t[10, 10].tolist(),
        "不转换就交给按 RGB 解释的显示函数时，这个像素被读成的 (R, G, B)": t[10, 10].tolist(),
        "测试图蓝色块里一个像素的 BGR 值": t[10, 400].tolist(),
        "灰度系数（R、G、B）": [0.299, 0.587, 0.114],
    }
    hsv_r = cv2.cvtColor(rc, cv2.COLOR_BGR2HSV)
    back = cv2.cvtColor(hsv_r, cv2.COLOR_HSV2BGR)
    d = np.abs(back.astype(int) - rc.astype(int))
    hue = {name: int(cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_BGR2HSV)[0, 0, 0]) for name, c in
           (("纯红 (0,0,255)", (0, 0, 255)), ("略偏品红的红 (10,0,255)", (10, 0, 255)), ("略偏橙的红 (0,10,255)", (0, 10, 255)), ("纯绿", (0, 255, 0)), ("纯蓝", (255, 0, 0)))}
    scale = 1.5
    out = m_col.enhance_brightness_hsv(rc, scale)
    direct = np.clip(rc.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    noclip = rc.max(-1) * scale <= 255
    dd = np.abs(out.astype(int) - direct.astype(int)).max(-1)
    full0 = cv2.cvtColor(rc, cv2.COLOR_BGR2HSV_FULL).astype(float)
    vivid = (full0[..., 1] > 60) & (full0[..., 2] > 60)

    def hue_err(x):
        h = cv2.cvtColor(x, cv2.COLOR_BGR2HSV_FULL).astype(float)[..., 0]
        e = np.abs(h - full0[..., 0])
        return np.minimum(e, 256 - e) * 360 / 256

    e_hsv, e_dir = hue_err(out), hue_err(direct)
    px = np.uint8([[[40, 120, 200]]])
    px_hsv = m_col.enhance_brightness_hsv(px, scale)
    px_dir = np.clip(px.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    hv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).astype(np.float32)
    hv[..., 2] *= scale
    NUM["HSV"] = {
        "说明": "同一批 400x400 的随机颜色；亮度增强调用作业 1-2 的参考实现 enhance_brightness_hsv(img, 1.5)，对照是直接把 BGR 乘 1.5 再 clip；"
              "色调误差只统计饱和度和亮度都大于 60（满刻度 255）的像素，单位是度",
        "uint8 下 H、S、V 的取值范围": [[int(hsv_r[..., i].min()), int(hsv_r[..., i].max())] for i in range(3)],
        "V 等于 max(B, G, R)": bool(np.array_equal(hsv_r[..., 2], rc.max(-1))),
        "几种颜色的 H": hue,
        "BGR→HSV→BGR 往返：最大误差、平均误差、完全还原的像素比例": [int(d.max()), r3(d.mean(), 2), r3((d.max(-1) == 0).mean())],
        "乘 1.5 后不会超过 255 的像素比例": r3(noclip.mean()),
        "不截断的像素上，两种做法的最大差和平均差": [int(dd[noclip].max()), r3(dd[noclip].mean(), 2)],
        "会截断的像素上的平均色调误差": {"HSV 里乘 V": r3(e_hsv[~noclip & vivid].mean(), 1), "BGR 直接乘": r3(e_dir[~noclip & vivid].mean(), 1)},
        "会截断的像素上色调误差的 95% 分位": {"HSV 里乘 V": r3(np.percentile(e_hsv[~noclip & vivid], 95), 1), "BGR 直接乘": r3(np.percentile(e_dir[~noclip & vivid], 95), 1)},
        "橙色 BGR=(40,120,200) 的例子": {
            "原来的 H（度）": int(cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0, 0]) * 2,
            "HSV 里乘 V 得到的 BGR": px_hsv[0, 0].tolist(), "它的 H（度）": int(cv2.cvtColor(px_hsv, cv2.COLOR_BGR2HSV)[0, 0, 0]) * 2,
            "BGR 直接乘得到的 BGR": px_dir[0, 0].tolist(), "直接乘之后的 H（度）": int(cv2.cvtColor(px_dir, cv2.COLOR_BGR2HSV)[0, 0, 0]) * 2,
            "V 乘 1.5 后是 300，不 clip 直接转 uint8 的本机结果": int(hv.astype(np.uint8)[0, 0, 2]),
        },
    }

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.9), gridspec_kw={"width_ratios": [1, 1, 1, 1.5]}, layout="constrained")
    ax[0].imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
    ax[0].set_title("彩色图：左上色斑，右下变暗")
    ax[1].imshow(gray, cmap="gray", vmin=60, vmax=150)
    ax[1].set_title("转成灰度：色斑看不见了")
    ax[2].imshow(noisy[..., 2], cmap="gray", vmin=20, vmax=200)
    ax[2].set_title("只看 R 通道：色斑又出现了")
    for a in ax[:3]:
        S.image_axes(a)
    keys = list(chans)
    xs = np.arange(len(keys))
    ax[3].bar(xs - 0.2, [abs(cnr[k]["色斑"]) for k in keys], 0.4, color=S.ORANGE, label="色斑（和背景灰度相同）")
    ax[3].bar(xs + 0.2, [abs(cnr[k]["变暗"]) for k in keys], 0.4, color=S.BLUE, label="变暗（三个通道各减 12）")
    for i, k in enumerate(keys):
        ax[3].text(i - 0.2, abs(cnr[k]["色斑"]) + 0.8, f"{abs(cnr[k]['色斑']):.1f}", ha="center", fontsize=8)
        ax[3].text(i + 0.2, abs(cnr[k]["变暗"]) + 0.8, f"{abs(cnr[k]['变暗']):.1f}", ha="center", fontsize=8)
    ax[3].set_xticks(xs, keys)
    ax[3].set_ylabel("对比噪声比的绝对值")
    ax[3].set_title("同一张图，不同通道里缺陷的对比噪声比")
    ax[3].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    S.clean(ax[3])
    ax[3].grid(axis="x", visible=False)
    S.save(fig, CH, "fig02_通道与灰度")


# ---------------------------------------------------------------- 图 3：均衡化
DEFECTS = [(170, 170), (200, 200), (160, 220), (215, 150)]
BG_BOX = np.s_[140:240, 130:250]


def low_contrast(rng, defect=True, n=256, sigma=2.0, amp=5.0):
    """对比度不足的检测图：灰度只占 90 到 125 这一小段，里面有四个 6x6、比背景亮 5 的弱缺陷。"""
    img = np.full((n, n), 100.0)
    y, x = np.mgrid[0:n, 0:n]
    img[(y >= 24) & (y < 120) & (x >= 24) & (x < 232) & ((x % 16) < 8)] = 112.0
    img[150:230, 30:110] = 118.0
    dm = np.zeros((n, n), bool)
    for cy, cx in DEFECTS:
        dm[cy:cy + 6, cx:cx + 6] = True
    if defect:
        img[dm] += amp
    bg = np.zeros((n, n), bool)
    bg[BG_BOX] = True
    return to_u8(img + rng.normal(0, sigma, img.shape)), dm, bg & ~dm


def clahe(img, clip):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(img)


def stretch(img):
    lo, hi = float(img.min()), float(img.max())
    return to_u8((img.astype(float) - lo) / (hi - lo) * 255)


def fig_equalize():
    methods = {"原图": lambda x: x, "线性拉伸到 0-255": stretch, "全局均衡化": m_eq.histogram_equalization,
               "CLAHE clipLimit=2": lambda x: clahe(x, 2.0), "CLAHE clipLimit=4": lambda x: clahe(x, 4.0), "CLAHE clipLimit=40": lambda x: clahe(x, 40.0)}
    reps = 50
    acc = {k: [] for k in methods}
    zs = {"直接相减": [], "两张图各自均衡化后相减": []}
    off = {"直接相减": [], "两张图各自均衡化后相减": []}
    levels = []
    example = None
    for s in range(reps):
        rng = np.random.default_rng(100 + s)
        img, dm, bg = low_contrast(rng)
        for k, fn in methods.items():
            f = fn(img).astype(float)
            acc[k].append((f[bg].std(), f[dm].mean() - f[bg].mean(), (f[dm].mean() - f[bg].mean()) / f[bg].std(), auc(f[dm], f[bg]), len(np.unique(f))))
        ref, _, _ = low_contrast(rng, defect=False)
        e_img, e_ref = m_eq.histogram_equalization(img).astype(float), m_eq.histogram_equalization(ref).astype(float)
        for name, d in (("直接相减", img.astype(float) - ref), ("两张图各自均衡化后相减", e_img - e_ref)):
            zs[name].append((d[dm].mean() - d[bg].mean()) / d[bg].std())
        brighter = to_u8(img.astype(float) * 1.08)
        for name, d in (("直接相减", brighter.astype(float) - ref), ("两张图各自均衡化后相减", m_eq.histogram_equalization(brighter).astype(float) - e_ref)):
            off[name].append(d[bg].mean() / d[bg].std())
        levels.append((int(img.min()), int(img.max())))
        if s == 0:
            example = (img, m_eq.histogram_equalization(img), dm)
    mean = {k: np.mean(v, axis=0) for k, v in acc.items()}
    img, eq_img, dm = example
    h0 = np.bincount(img.ravel(), minlength=256)
    h1 = np.bincount(eq_img.ravel(), minlength=256)
    ideal = (np.arange(256) + 1) / 256
    lut = {int(k): int(eq_img[img == k][0]) for k in np.unique(img)}
    NUM["均衡化与弱缺陷"] = {
        "说明": f"256x256 的低对比度合成检测图：背景 100，线条 112，方块 118，高斯噪声标准差 2；四个 6x6 的弱缺陷比背景亮 5；"
              f"背景噪声和对比噪声比在缺陷所在的平坦背景区里统计；{reps} 张不同噪声的图取平均；全局均衡化调用作业 1-1 的参考实现，CLAHE 用 OpenCV（8x8 分块）",
        "原图灰度范围（50 张图里的最小和最大）": [min(a for a, _ in levels), max(b for _, b in levels)],
        "背景噪声标准差": {k: r3(v[0], 2) for k, v in mean.items()},
        "背景噪声相对原图的放大倍数": {k: r3(v[0] / mean["原图"][0], 1) for k, v in mean.items()},
        "缺陷与背景的灰度差": {k: r3(v[1], 1) for k, v in mean.items()},
        "对比噪声比": {k: r3(v[2], 2) for k, v in mean.items()},
        "对比噪声比相对原图的变化": {k: f"{100 * (v[2] / mean['原图'][2] - 1):+.0f}%" for k, v in mean.items()},
        "缺陷像素与背景像素的 AUC": {k: r3(v[3], 4) for k, v in mean.items()},
        "用到的灰度级个数": {k: r3(v[4], 1) for k, v in mean.items()},
        "AUC 与原图之差：50 张图的均值和标准误": {k: [r3(np.mean(d_), 4), r3(np.std(d_, ddof=1) / math.sqrt(reps), 4)] for k, d_ in
                                     ((k, np.array(acc[k])[:, 3] - np.array(acc["原图"])[:, 3]) for k in methods if k != "原图")},
        "缺陷灰度差的放大倍数": {k: r3(v[1] / mean["原图"][1], 1) for k, v in mean.items()},
        "第一张图": {
            "灰度范围": [int(img.min()), int(img.max())],
            "原图和均衡化后最高的直方图柱": [int(h0.max()), int(h1.max())],
            "累积分布与均匀分布的最大偏差（原图，均衡化后）": [r3(np.abs(h0.cumsum() / h0.sum() - ideal).max()), r3(np.abs(h1.cumsum() / h1.sum() - ideal).max())],
            "查找表在 96 到 106 的取值": {str(k): lut[k] for k in range(96, 107) if k in lut},
        },
        "模板比对里缺陷的 z 值（差值图上缺陷均值除以背景标准差）": {k: r3(np.mean(v), 2) for k, v in zs.items()},
        "待检图整体亮 8% 时，背景差值的均值相当于几倍标准差": {k: r3(np.mean(v), 2) for k, v in off.items()},
    }

    np.random.seed(0)
    t = m_eq.create_test_image()
    te = m_eq.histogram_equalization(t)
    worst, ndiff, total = 0, 0, 0
    for s in range(300):
        r = np.random.default_rng(s)
        im = [r.integers(0, 256, (64, 64), dtype=np.uint8), to_u8(r.normal(100, 8, (200, 300))), to_u8(r.normal(60, 25, (240, 320)))][s % 3]
        dd = np.abs(m_eq.histogram_equalization(im).astype(int) - cv2.equalizeHist(im).astype(int))
        worst, ndiff, total = max(worst, int(dd.max())), ndiff + int((dd > 0).sum()), total + dd.size
    tiny = np.array([[50, 50, 200, 200]], np.uint8)
    ht = np.bincount(t.ravel(), minlength=256)
    NUM["均衡化的实现核对"] = {
        "说明": "参考实现与 cv2.equalizeHist 在 300 张图上逐像素比较（均匀随机、窄高斯、宽高斯三类，种子 0 到 299）；测试图指作业 1-1 的 create_test_image()（np.random.seed(0)）",
        "与 cv2.equalizeHist 的最大差": worst, "不相同的像素个数": ndiff, "比较的像素总数": total,
        "四个像素 [50,50,200,200]：减 cdf_min 的结果": m_eq.histogram_equalization(tiny)[0].tolist(),
        "四个像素 [50,50,200,200]：不减 cdf_min 的结果": np.round(np.array([2, 2, 4, 4]) / 4 * 255).astype(int).tolist(),
        "测试图里灰度为 0 的像素比例（也就是 CDF 在 0 处的值）": r3(ht[0] / ht.sum()),
        "测试图用到的灰度级个数（均衡化前，均衡化后）": [int((ht > 0).sum()), int((np.bincount(te.ravel(), minlength=256) > 0).sum())],
        "np.histogram(img.flatten(), 256, [0, 256]) 返回的两个数组的长度": [len(x) for x in np.histogram(t.flatten(), 256, [0, 256])],
    }

    fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.35, 1.35]}, layout="constrained")
    for a, im, title in ((ax[0], img, f"原图：灰度只占 {img.min()} 到 {img.max()}"), (ax[1], eq_img, "全局均衡化：背景变成一片噪声")):
        a.imshow(im, cmap="gray", vmin=0, vmax=255)
        for cy, cx in DEFECTS:
            a.add_patch(plt.Rectangle((cx - 3.5, cy - 3.5), 12, 12, fill=False, ec=S.ORANGE, lw=1.1))
        a.set_title(title)
        S.image_axes(a)
    ks = np.arange(88, 130)
    ax[2].bar(ks, h0[ks] / h0.sum(), color="#c9ced6", width=0.9, label="原图直方图（左轴）")
    ax[2].set_xlabel("原图灰度")
    ax[2].set_ylabel("像素占比")
    a2 = ax[2].twinx()
    lk = sorted(lut)
    a2.plot(lk, [lut[k] for k in lk], "o-", color=S.BLUE, ms=3, lw=1.6, label="均衡化的查找表（右轴）")
    a2.plot([lk[0], lk[-1]], [0, 255], "--", color=S.GREEN, lw=1.3, label="线性拉伸（右轴）")
    a2.axvspan(96, 104, color=S.ORANGE, alpha=0.10)
    a2.text(95.6, 244, "背景 100±2", fontsize=8.5, color=S.ORANGE, ha="right")
    a2.axvline(105, color=S.RED, lw=1, ls=":")
    a2.text(105.6, 244, "缺陷 105", fontsize=8.5, color=S.RED)
    a2.set_ylabel("输出灰度")
    a2.set_ylim(0, 260)
    ax[2].set_title("像素最多的灰度被拉得最开")
    h_a, l_a = ax[2].get_legend_handles_labels()
    h_b, l_b = a2.get_legend_handles_labels()
    ax[2].legend(h_a + h_b, l_a + l_b, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[2])
    a2.spines["top"].set_visible(False)
    show = ["原图", "线性拉伸到 0-255", "全局均衡化", "CLAHE clipLimit=2", "CLAHE clipLimit=40"]
    labels = ["原图", "线性拉伸", "全局均衡化", "CLAHE\nclip=2", "CLAHE\nclip=40"]
    vals = [mean[k][2] for k in show]
    bars = ax[3].bar(labels, vals, color=[S.MUTED, S.GREEN, S.BLUE, S.PURPLE, S.PURPLE], alpha=0.85, width=0.6)
    for b, k, v in zip(bars, show, vals):
        ax[3].text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=9.5)
        ax[3].text(b.get_x() + b.get_width() / 2, 0.12, f"噪声\n{mean[k][0]:.1f}", ha="center", fontsize=8.5, color="white")
    ax[3].set_ylabel("弱缺陷的对比噪声比")
    ax[3].set_ylim(0, 3.1)
    ax[3].set_title("画面更“清楚”，缺陷的对比噪声比没有变好")
    S.clean(ax[3])
    ax[3].grid(axis="x", visible=False)
    S.save(fig, CH, "fig03_均衡化与弱缺陷")


# ---------------------------------------------------------------- 图 4：椒盐残留
def sp_theory(p, k):
    """平坦图上 k×k 中值滤波后仍是坏点的概率：窗口里同一种极性的坏点达到一半以上。"""
    n, q = k * k, p / 2
    need = (n + 1) // 2
    return 2 * sum(math.comb(n, j) * q ** j * (1 - q) ** (n - j) for j in range(need, n + 1))


def sp_theory_border(p, n):
    """3×3 中值、边界复制填充时，n×n 图最外一圈的残留概率：边上的像素窗口里只有 6 个不同的像素（3 个算两次），四角只有 4 个。"""
    q = p / 2

    def one(weights):
        total = 0.0
        for mask in range(1 << len(weights)):
            bits = [(mask >> i) & 1 for i in range(len(weights))]
            if sum(w for w, b in zip(weights, bits) if b) >= 5:
                total += q ** sum(bits) * (1 - q) ** (len(weights) - sum(bits))
        return 2 * total

    edge, corner = one([2, 2, 2, 1, 1, 1]), one([4, 2, 2, 1])
    return (4 * (n - 2) * edge + 4 * corner) / (4 * (n - 1))


def fig_salt_pepper():
    ps = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    ks = [3, 5, 7]
    size, reps, margin = 512, 20, 3
    flat = np.full((size, size), 120, np.uint8)
    inner = np.zeros(flat.shape, bool)
    inner[margin:-margin, margin:-margin] = True
    ring = np.ones(flat.shape, bool)
    ring[1:-1, 1:-1] = False
    meas = {k: [] for k in ks}
    ring3, mean_spread, mean_mae, noisy_mae = [], [], [], []
    for p in ps:
        bad = dict.fromkeys(ks, 0)
        rb = sp_ = mm = nm = 0.0
        for s in range(reps):
            noisy = salt_pepper(flat, p, 1000 + s)
            for k in ks:
                out = cv2.medianBlur(noisy, k)
                bad[k] += int((out[inner] != 120).sum())
                if k == 3:
                    rb += float((out[ring] != 120).mean())
            mean3 = cv2.blur(noisy.astype(np.float32), (3, 3))
            sp_ += float((np.abs(mean3 - 120) > 5)[inner].mean())
            mm += float(np.abs(mean3 - 120)[inner].mean())
            nm += float(np.abs(noisy.astype(np.float32) - 120)[inner].mean())
        for k in ks:
            meas[k].append(bad[k] / (reps * int(inner.sum())))
        ring3.append(rb / reps)
        mean_spread.append(sp_ / reps)
        mean_mae.append(mm / reps)
        noisy_mae.append(nm / reps)
    floor = 1 / (reps * int(inner.sum()))
    NUM["椒盐残留"] = {
        "说明": f"{size}x{size} 的平坦图（灰度 120），椒盐噪声调用作业 1-3 的参考实现（椒和盐各占一半），每个噪声比例 {reps} 张图；"
              f"残留指滤波后仍不等于 120 的像素比例，内部指去掉最外 {margin} 圈之后的区域；理论值是窗口内同一极性的坏点不少于一半的二项概率",
        "噪声比例": ps,
        "中值滤波后内部的残留比例（实测）": {f"{k}x{k}": [sig(v) for v in meas[k]] for k in ks},
        "中值滤波后的残留比例（理论）": {f"{k}x{k}": [sig(sp_theory(p, k)) for p in ps] for k in ks},
        "本实验能分辨的最小比例": sig(floor, 2),
        "3x3 中值滤波后最外一圈像素的残留比例（实测）": [sig(v) for v in ring3],
        "3x3 中值滤波后最外一圈像素的残留比例（理论，边界复制填充）": [sig(sp_theory_border(p, size)) for p in ps],
        "最外一圈的残留是内部的几倍（实测，内部为 0 的记为 null）": [r3(a / b, 0) if b > 0 else None for a, b in zip(ring3, meas[3])],
        "实测到的残留像素个数（20 张图合计）": {f"{k}x{k}": [int(round(v * reps * int(inner.sum()))) for v in meas[k]] for k in ks},
        "3x3 窗口在边上和角上最少几个同极性坏点就会出错（内部是 5 个）": [3, 2],
        "3x3 中值在 512x512 图上平均残留的坏点个数（按内部比例折算）": [r3(v * size * size, 1) for v in meas[3]],
        "5x5 中值在 512x512 图上平均残留的坏点个数（按内部比例折算）": [r3(v * size * size, 1) for v in meas[5]],
        "3x3 均值滤波后误差超过 5 个灰度的像素比例": [r3(v) for v in mean_spread],
        "窗口内至少有一个坏点的概率 1-(1-p)^9（理论）": [r3(1 - (1 - p) ** 9) for p in ps],
        "加噪图的平均绝对误差": [r3(v, 2) for v in noisy_mae],
        "3x3 均值滤波后的平均绝对误差": [r3(v, 2) for v in mean_mae],
        "3x3 窗口里同一极性最多能容忍的坏点个数": 4,
        "参考实现的中值滤波与 cv2.medianBlur 逐像素相同（3、5、7 三种窗口，含边界）": check_median_reference(),
    }

    # 示例图：带结构的 96x96 小图，10% 椒盐，中值用参考实现
    rng = np.random.default_rng(6)
    clean = to_u8(pattern(128)[8:104, 8:104] + rng.normal(0, 2, (96, 96)))
    noisy = salt_pepper(clean, 0.10, 6)
    mean3 = cv2.blur(noisy, (3, 3))
    med3 = m_nz.median_filter_manual(noisy, 3)
    err = {name: r3(np.abs(x.astype(float) - clean).mean(), 2) for name, x in (("加噪图", noisy), ("3x3 均值", mean3), ("3x3 中值（参考实现）", med3))}
    NUM["椒盐示例图"] = {"说明": "96x96 带线条和方块的合成图（背景噪声标准差 2），加 10% 椒盐；数值是和加噪前相比的平均绝对误差", "平均绝对误差": err}

    # 带图形的图上，中值滤波的残留误差集中在哪里：和“无噪声的图做同样滤波”的结果比
    scene = pattern().astype(np.uint8)
    band = cv2.dilate(scene, np.ones((5, 5), np.uint8)) != cv2.erode(scene, np.ones((5, 5), np.uint8))
    keep = np.zeros(scene.shape, bool)
    keep[4:-4, 4:-4] = True
    rng2 = np.random.default_rng(7)
    jitter = {}
    for k in (3, 5):
        ref_k = cv2.medianBlur(scene, k).astype(int)
        nb = nf = 0
        for i in range(20):
            out_k = cv2.medianBlur(salt_pepper(to_u8(pattern() + rng2.normal(0, 2, scene.shape)), 0.10, 7000 + i), k).astype(int)
            bad_k = np.abs(out_k - ref_k) > 20
            nb += int((bad_k & band & keep).sum())
            nf += int((bad_k & ~band & keep).sum())
        jitter[f"{k}x{k}"] = {"边缘附近出错的像素比例": sig(nb / (20 * int((band & keep).sum()))), "平坦区出错的像素比例": sig(nf / (20 * int((~band & keep).sum()))),
                              "出错像素里落在边缘附近的比例": r3(nb / (nb + nf))}
    NUM["中值滤波在边缘的残留"] = {
        "说明": "128x128 的合成检测图（同图 1 的图形，背景噪声标准差 2）加 10% 椒盐，20 张；和无噪声的图做同样的中值滤波后相比，差超过 20 个灰度算出错；"
              "边缘附近指 5x5 邻域内有不止一种灰度的像素（离图形边缘 2 个像素以内）；去掉图像最外 4 圈",
        "边缘附近的像素占统计区域的比例": r3((band & keep).sum() / keep.sum()),
        "按窗口": jitter,
    }

    fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.9), gridspec_kw={"width_ratios": [1, 1, 1, 1.55]}, layout="constrained")
    for a, im, title in ((ax[0], noisy, "加 10% 椒盐"), (ax[1], mean3, "3×3 均值：坏点被抹成一片灰斑"), (ax[2], med3, "3×3 中值：坏点基本消失，角被削圆")):
        a.imshow(im, cmap="gray", vmin=0, vmax=255)
        a.set_title(title)
        S.image_axes(a)
    xs = [100 * p for p in ps]
    for k, color, marker in ((3, S.RED, "o"), (5, S.ORANGE, "s"), (7, S.BLUE, "^")):
        ax[3].plot(xs, [max(sp_theory(p, k), floor / 3) for p in ps], "-", color=color, lw=1.4, alpha=0.75)
        ax[3].plot(xs, [max(v, floor) for v in meas[k]], marker, color=color, ms=6, label=f"{k}×{k} 中值，内部实测")
    ax[3].plot(xs, [sp_theory_border(p, size) for p in ps], "-", color=S.MUTED, lw=1.4, alpha=0.75)
    ax[3].plot(xs, [max(v, floor) for v in ring3], "x", color=S.MUTED, ms=7, mew=1.6, label="3×3 中值，最外一圈实测")
    ax[3].plot([], [], "-", color=S.MUTED, lw=1.4, label="细线：二项分布算出的理论值")
    ax[3].axhline(floor, color=S.MUTED, lw=1, ls=":")
    ax[3].text(xs[-1], floor * 1.35, "本实验能分辨的下限", fontsize=8, color=S.MUTED, ha="right")
    ax[3].set_yscale("log")
    ax[3].set_ylim(floor / 2, 1)
    ax[3].set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], ["1e-6", "1e-5", "1e-4", "0.001", "0.01", "0.1"])
    ax[3].minorticks_off()
    ax[3].set_xticks(xs, [f"{x:g}" for x in xs])
    ax[3].set_xlabel("椒盐噪声比例（%）")
    ax[3].set_ylabel("滤波后仍是坏点的像素比例")
    ax[3].set_title("噪声越密残留越多，最外一圈比内部差得多")
    ax[3].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[3])
    S.save(fig, CH, "fig04_椒盐残留")


# ---------------------------------------------------------------- 图 5：滤波器与弱缺陷
FILTERS = {
    "不滤波": lambda x: x.astype(np.float32),
    "均值 3×3": lambda x: cv2.blur(x.astype(np.float32), (3, 3)),
    "高斯 sigma=1": lambda x: cv2.GaussianBlur(x.astype(np.float32), (0, 0), 1.0),
    "中值 3×3": lambda x: cv2.medianBlur(x, 3).astype(np.float32),
    "中值 5×5": lambda x: cv2.medianBlur(x, 5).astype(np.float32),
    "双边滤波": lambda x: cv2.bilateralFilter(x, 5, 25, 1.5).astype(np.float32),
}


def defect_trial(with_sp, size, seed, amp=10.0, sigma=4.0, p=0.05, reps=40, n=256, fa=1e-4):
    rng = np.random.default_rng(seed)
    centers = [(cy, cx) for cy in range(16, n - 16, 24) for cx in range(16, n - 16, 24)]
    bgs = {k: [] for k in FILTERS}
    peaks = {k: [] for k in FILTERS}
    for i in range(reps):
        noise = rng.normal(0, sigma, (n, n))
        good = to_u8(100.0 + rng.normal(0, sigma, (n, n)))
        f = 100.0 + noise
        for cy, cx in centers:
            f[cy:cy + size, cx:cx + size] += amp
        test = to_u8(f)
        if with_sp:
            good = salt_pepper(good, p, seed * 1000 + 2 * i)
            test = salt_pepper(test, p, seed * 1000 + 2 * i + 1)
        for k, fn in FILTERS.items():
            bgs[k].append(np.abs(fn(good)[8:-8, 8:-8] - 100).ravel())
            ft = np.abs(fn(test) - 100)
            peaks[k].append([ft[cy - 1:cy + size + 1, cx - 1:cx + size + 1].max() for cy, cx in centers])
    out = {}
    for k in FILTERS:
        thr = float(np.quantile(np.concatenate(bgs[k]), 1 - fa))
        out[k] = (thr, float((np.concatenate(peaks[k]) > thr).mean()))
    return out, len(centers) * reps


def fig_filters_defect():
    sizes = [1, 2, 3, 4, 6]
    conds = {"只有高斯噪声": False, "高斯噪声再加 5% 椒盐": True}
    table, thr = {}, {}
    count = 0
    for ci, (cname, with_sp) in enumerate(conds.items()):
        table[cname] = {k: [] for k in FILTERS}
        thr[cname] = {}
        for si, s in enumerate(sizes):
            res, count = defect_trial(with_sp, s, seed=10 * ci + si + 1)
            for k, (t, d) in res.items():
                table[cname][k].append(d)
                thr[cname].setdefault(k, []).append(t)
    NUM["滤波器与弱缺陷"] = {
        "说明": "256x256 的平坦图（灰度 100），高斯噪声标准差 4；缺陷是边长 1 到 6 的方块，比背景亮 10（噪声标准差的 2.5 倍）；"
              "椒盐一项是在同样的高斯噪声上再加 5%（调用作业 1-3 的参考实现）；滤波后看 |输出 - 100|，阈值取无缺陷图上该量的 99.99% 分位"
              f"（每个像素的误报概率 1e-4），缺陷范围内的最大值超过阈值算检出；每格 {count} 个缺陷；"
              "双边滤波参数 d=5、sigmaColor=25、sigmaSpace=1.5",
        "缺陷边长": sizes,
        "检出率": {c: {k: [r3(v, 3) for v in vals] for k, vals in t.items()} for c, t in table.items()},
        "阈值（各边长取平均）": {c: {k: r3(np.mean(v), 1) for k, v in t.items()} for c, t in thr.items()},
    }

    # 没有噪声时，中值滤波对细线和小方块的作用（确定性的）
    lines, blobs = {}, {}
    for wd in (1, 2, 3, 4):
        img = np.full((41, 41), 100, np.uint8)
        img[:, 20:20 + wd] = 160
        lines[str(wd)] = {f"{k}x{k}": {"中值后线的对比度": int(cv2.medianBlur(img, k)[20].max()) - 100, "均值后线的峰值对比度": r3(cv2.blur(img.astype(np.float32), (k, k))[20].max() - 100, 1)} for k in (3, 5, 7)}
    for s in (1, 2, 3, 4, 5):
        img = np.full((41, 41), 100, np.uint8)
        img[18:18 + s, 18:18 + s] = 160
        blobs[str(s)] = {f"{k}x{k}": {"中值后剩下的像素数": int((cv2.medianBlur(img, k) > 100).sum()), "均值后的峰值对比度": r3(cv2.blur(img.astype(np.float32), (k, k)).max() - 100, 1)} for k in (3, 5, 7)}
    NUM["中值滤波与细节"] = {"说明": "没有噪声的 41x41 图，背景 100，细线或方块的灰度 160（对比度 60）；这一组是确定性的结果",
                      "竖线，按线宽（像素）": lines, "方块，按边长（像素）": blobs}

    rng = np.random.default_rng(0)
    g = (120 + rng.normal(0, 8, (512, 512))).astype(np.float32)
    inner = np.s_[8:-8, 8:-8]
    resid = {"均值 3×3": cv2.blur(g, (3, 3)), "中值 3×3": cv2.medianBlur(g, 3), "均值 5×5": cv2.blur(g, (5, 5)), "中值 5×5": cv2.medianBlur(g, 5),
             "高斯 sigma=1": cv2.GaussianBlur(g, (0, 0), 1.0)}
    ratio = {k: r3(v[inner].std() / 8, 3) for k, v in resid.items()}
    a = rng.integers(0, 120, (64, 64)).astype(np.float32)
    b = rng.integers(0, 120, (64, 64)).astype(np.float32)
    lin = {name: sig(np.abs(fn(a + b) - (fn(a) + fn(b)))[4:-4, 4:-4].max()) for name, fn in
           (("均值 3×3", lambda x: cv2.blur(x, (3, 3))), ("高斯 sigma=1", lambda x: cv2.GaussianBlur(x, (0, 0), 1.0)),
            ("中值 3×3", lambda x: cv2.medianBlur(x, 3)), ("双边滤波", lambda x: cv2.bilateralFilter(x, 5, 25, 1.5)))}
    pois = {}
    for mu in (20, 80, 180):
        x = rng.poisson(mu, 200000)
        pois[str(mu)] = {"实测方差": r3(x.var(), 1), "实测标准差": r3(x.std(), 2), "sqrt(均值)": r3(math.sqrt(mu), 2), "Anscombe 变换后的方差": r3((2 * np.sqrt(x + 3 / 8)).var(), 3)}
    NUM["高斯噪声下的残留"] = {
        "说明": "512x512 的平坦图加标准差 8 的高斯噪声（float32，不量化），数值是滤波后的噪声标准差与滤波前之比",
        "实测": ratio, "均值 3×3 的理论值 1/3": r3(1 / 3), "均值 5×5 的理论值 1/5": 0.2,
        "中值 3×3 的大样本近似 sqrt(pi/2)/3": r3(math.sqrt(math.pi / 2) / 3), "中值 5×5 的大样本近似 sqrt(pi/2)/5": r3(math.sqrt(math.pi / 2) / 5),
        "同样窗口下中值的噪声是均值的几倍": {"3×3": r3(ratio["中值 3×3"] / ratio["均值 3×3"], 2), "5×5": r3(ratio["中值 5×5"] / ratio["均值 5×5"], 2)},
    }
    NUM["线性性检验"] = {"说明": "两张 64x64 的随机图 a、b（float32，取值 0 到 119），max |f(a+b) - f(a) - f(b)|，去掉边界 4 个像素", "最大偏差": lin}
    NUM["泊松噪声"] = {"说明": "每个均值各 20 万个泊松随机数；Anscombe 变换是 2*sqrt(x + 3/8)", "按均值": pois}

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2), layout="constrained")
    names = list(FILTERS)
    titles = {"只有高斯噪声": "2×2 的缺陷线性滤波检出最多，中值吃掉小缺陷", "高斯噪声再加 5% 椒盐": "只有中值滤波还能检出"}
    for a_, (cname, t) in zip(ax, table.items()):
        mat = np.array([t[k] for k in names])
        im = a_.imshow(mat, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                a_.text(j, i, f"{100 * mat[i, j]:.0f}%", ha="center", va="center", fontsize=10, color="white" if mat[i, j] > 0.55 else S.INK)
        a_.set_xticks(range(len(sizes)), [f"{s}×{s}" for s in sizes])
        a_.set_yticks(range(len(names)), names)
        a_.set_xlabel("缺陷大小（像素）")
        a_.set_title(f"{cname}：{titles[cname]}")
        a_.tick_params(length=0)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02).ax.tick_params(labelsize=8)
    S.save(fig, CH, "fig05_滤波器与弱缺陷")


# ---------------------------------------------------------------- 图 6：保边的门槛
EDGE_FILTERS = {
    "均值 5×5": (lambda x: cv2.blur(x.astype(np.float32), (5, 5)), S.ORANGE, "s"),
    "高斯 sigma=1.5": (lambda x: cv2.GaussianBlur(x.astype(np.float32), (0, 0), 1.5), S.GREEN, "^"),
    "中值 5×5": (lambda x: cv2.medianBlur(x, 5).astype(np.float32), S.BLUE, "o"),
    "双边滤波": (lambda x: cv2.bilateralFilter(x, 9, 30, 2.0).astype(np.float32), S.PURPLE, "D"),
}


def edge_width(profile):
    lo, hi = profile[:8].mean(), profile[-8:].mean()
    p = (profile - lo) / (hi - lo)

    def cross(level):
        i = int(np.argmax(p >= level))
        return i - 1 + (level - p[i - 1]) / (p[i] - p[i - 1])

    return cross(0.9) - cross(0.1)


def fig_edges():
    steps = [10, 20, 40, 80, 160]
    sigma, reps = 6.0, 20
    rng = np.random.default_rng(2)
    width = {k: [] for k in EDGE_FILTERS}
    flat = {k: [] for k in EDGE_FILTERS}
    raw_width = []
    profiles = {}
    for h in steps:
        img = np.full((128, 128), 60.0)
        img[:, 64:] = 60.0 + h
        noisy = [to_u8(img + rng.normal(0, sigma, img.shape)) for _ in range(reps)]
        raw_width.append(edge_width(np.mean([x[16:-16, 40:88].mean(axis=0) for x in noisy], axis=0)))
        for k, (fn, _, _) in EDGE_FILTERS.items():
            outs = [fn(x) for x in noisy]
            prof = np.mean([o[16:-16, 40:88].mean(axis=0) for o in outs], axis=0)
            width[k].append(edge_width(prof))
            flat[k].append(float(np.mean([o[16:-16, 8:48].std() for o in outs])))
            profiles[(h, k)] = (prof - 60.0) / h
    NUM["保边的门槛"] = {
        "说明": f"128x128 的竖直台阶（左边 60，右边 60 加台阶高度），高斯噪声标准差 {sigma:g}，每个高度 {reps} 张图；"
              "边缘宽度是按行平均后的剖面从 10% 升到 90% 走过的像素数；双边滤波参数 d=9、sigmaColor=30、sigmaSpace=2",
        "台阶高度": steps,
        "台阶高度与噪声标准差之比": [r3(h / sigma, 1) for h in steps],
        "边缘宽度（px）": {k: [r3(v, 2) for v in vals] for k, vals in width.items()},
        "不滤波时的边缘宽度（px）": [r3(v, 2) for v in raw_width],
        "平坦区的噪声标准差（各台阶高度取平均，滤波前是 6）": {k: r3(np.mean(v), 2) for k, v in flat.items()},
        "台阶高度与 sigmaColor 之比": [r3(h / 30, 1) for h in steps],
        "双边滤波的值域权重 exp(-d^2/(2*sigmaColor^2))（理论）": {
            "sigmaColor=30，d 取各台阶高度": {str(h): sig(math.exp(-h * h / (2 * 30.0 ** 2))) for h in steps},
            "sigmaColor=25（图 5 的参数），d=155（盐点 255 与背景 100 的差）": sig(math.exp(-155.0 ** 2 / (2 * 25.0 ** 2)), 2),
        },
    }

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9), layout="constrained")
    xs = np.arange(40, 88) - 63.5
    for a, h, title in ((ax[0], 160, f"台阶 160（噪声的 {160 / sigma:.0f} 倍）：中值和双边保住了边"), (ax[1], 10, f"台阶 10（噪声的 {10 / sigma:.1f} 倍）：四种滤波都把边抹宽")):
        for k, (_, color, _) in EDGE_FILTERS.items():
            a.plot(xs, profiles[(h, k)], color=color, lw=1.8, label=k)
        a.set_xlim(-8, 8)
        a.set_ylim(-0.1, 1.1)
        a.set_xlabel("到边缘的距离（px）")
        a.set_ylabel("归一化的灰度（0 是暗侧，1 是亮侧）")
        a.set_title(title)
        S.clean(a)
    ax[0].legend(fontsize=9, loc="upper left")
    for k, (_, color, marker) in EDGE_FILTERS.items():
        ax[2].plot(steps, width[k], marker + "-", color=color, lw=1.8, ms=5, label=k)
    ax[2].set_xscale("log")
    ax[2].set_xticks(steps, [str(s) for s in steps])
    ax[2].minorticks_off()
    ax[2].set_ylim(0, 5)
    ax[2].set_xlabel("台阶高度（灰度，噪声标准差是 6）")
    ax[2].set_ylabel("滤波后的边缘宽度（px）")
    ax[2].set_title("保边有门槛：台阶要比噪声高得多")
    ax[2].legend(fontsize=9, loc="lower left")
    S.clean(ax[2])
    S.save(fig, CH, "fig06_保边的门槛")


def main() -> None:
    S.setup()
    plt.rcParams["svg.hashsalt"] = "ch01"  # 固定 SVG 里的内部编号，重复运行时文件逐字节相同
    numbers_dtype()
    for fn in (fig_uint8, fig_color, fig_equalize, fig_salt_pepper, fig_filters_defect, fig_edges):
        fn()
        print("完成", fn.__name__, flush=True)
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
