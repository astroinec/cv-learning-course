"""第 24 章：合成 die 图像数据生成器（已写好，学员不需要修改）。

模拟的是 die-to-die 比对场景：
- 同一种周期版图的 die，被相机拍到时带有随机亚像素错位（机台定位误差）
- 光学模糊（高斯 PSF）、整体亮度漂移（照明波动）、散粒噪声和读出噪声
- 部分 die 上有缺陷：颗粒（亮或暗斑点）、短划痕、图形缺失
真值坐标是缺陷在“该 die 自身图像坐标系”下的中心 (y, x)。

用法：
    from 数据生成 import make_dataset
    data = make_dataset(seed=0)
    data["golden"]  # (G, H, W) float64，已知良品 die
    data["test"]    # (N, H, W) float64，待检 die
    data["truth"]   # 长度 N 的列表，每项是该 die 的缺陷中心列表 [(y, x), ...]
    data["shifts"]  # (N, 2) 真实错位，只用于分析，流水线不能读取
"""

from __future__ import annotations

import numpy as np

SIZE = 96


def _fourier_shift(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.fftfreq(img.shape[1])[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j * np.pi * (fy * dy + fx * dx))))


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.fftfreq(img.shape[1])[None, :]
    otf = np.exp(-2 * (np.pi * sigma) ** 2 * (fy ** 2 + fx ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * otf))


def make_layout(size: int = SIZE) -> np.ndarray:
    """周期版图：竖直金属线 + 接触孔阵列 + 一条横向总线 + 左上角对位十字，取值约 [0, 1]。

    金属线和接触孔是周期的，只靠它们做相位相关会在周期整数倍处出现几乎等高的假峰；
    真实 die 上也有同样问题，所以版图上会放对位标记（alignment mark）。
    """
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    lines = ((x % 8) < 4).astype(np.float64)
    contacts = (((y % 12) - 6) ** 2 + ((x % 12) - 6) ** 2 <= 5).astype(np.float64)
    bus = ((y > 40) & (y < 48)).astype(np.float64)
    layout = 0.25 + 0.45 * lines + 0.25 * contacts * (1 - lines) + 0.2 * bus
    mark = (((np.abs(y - 18) <= 1) & (np.abs(x - 18) <= 8)) | ((np.abs(x - 18) <= 1) & (np.abs(y - 18) <= 8)))
    pad = (np.abs(y - 18) <= 10) & (np.abs(x - 18) <= 10)
    layout = np.where(pad, 0.15, layout)
    layout = np.where(mark, 0.95, layout)
    return _gaussian_blur(layout, 0.7)


def _add_defect(img: np.ndarray, kind: str, cy: float, cx: float, rng: np.random.Generator) -> None:
    y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float64)
    if kind == "particle":
        r = rng.uniform(1.2, 2.2)
        amp = rng.choice([-1, 1]) * rng.uniform(0.35, 0.6)
        img += amp * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * r * r))
    elif kind == "scratch":
        ang = rng.uniform(0, np.pi)
        length = rng.uniform(6, 10)
        vy, vx = np.sin(ang), np.cos(ang)
        t = np.clip((y - cy) * vy + (x - cx) * vx, -length / 2, length / 2)
        d = np.hypot(y - (cy + t * vy), x - (cx + t * vx))
        img += rng.uniform(0.3, 0.5) * np.clip(1.5 - d, 0, 1)
    elif kind == "missing":
        h, w = rng.integers(3, 6), rng.integers(3, 6)
        y0, x0 = int(round(cy - h / 2)), int(round(cx - w / 2))
        patch = img[y0:y0 + h, x0:x0 + w]
        img[y0:y0 + h, x0:x0 + w] = patch * 0.2 + 0.1
    else:
        raise ValueError(kind)


def _capture(clean: np.ndarray, rng: np.random.Generator, full_scale_e: float = 2000.0,
             read_noise_e: float = 6.0) -> np.ndarray:
    gain = rng.uniform(0.85, 1.15)
    offset = rng.uniform(-0.05, 0.05)
    signal = np.clip((clean * gain + offset) * full_scale_e, 0, None)
    electrons = rng.poisson(signal) + rng.normal(0, read_noise_e, size=signal.shape)
    return electrons / full_scale_e


def make_dataset(seed: int = 0, n_golden: int = 8, n_test: int = 30, defect_ratio: float = 0.6) -> dict:
    rng = np.random.default_rng(seed)
    layout = make_layout()
    margin = 10

    def render(defects):
        dy, dx = rng.uniform(-3.5, 3.5, size=2)
        clean = _fourier_shift(layout, dy, dx)
        truth = []
        for kind in defects:
            cy, cx = rng.uniform(margin, SIZE - margin, size=2)
            _add_defect(clean, kind, cy, cx, rng)
            truth.append((float(cy), float(cx)))
        blurred = _gaussian_blur(clean, 1.0)
        return _capture(blurred, rng), truth, (float(dy), float(dx))

    golden = np.stack([render([])[0] for _ in range(n_golden)])
    test, truth, shifts = [], [], []
    kinds = ["particle", "scratch", "missing"]
    for _ in range(n_test):
        k = int(rng.integers(1, 4)) if rng.random() < defect_ratio else 0
        img, t, s = render(list(rng.choice(kinds, size=k)))
        test.append(img)
        truth.append(t)
        shifts.append(s)
    return {
        "golden": golden,
        "test": np.stack(test),
        "truth": truth,
        "shifts": np.asarray(shifts),
        "layout": layout,
    }


if __name__ == "__main__":
    d = make_dataset(0)
    n_def = sum(len(t) for t in d["truth"])
    print(f"golden {d['golden'].shape}，test {d['test'].shape}，缺陷 {n_def} 个，"
          f"有缺陷 die {sum(1 for t in d['truth'] if t)} 个")
