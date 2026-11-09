"""参考实现 22-1。学员完成本章前不要查看。"""

from __future__ import annotations

import numpy as np


def rayleigh_resolution_um(wavelength_nm, na):
    if not (0 < na <= 1.5):
        raise ValueError("NA 应在 (0, 1.5]")
    return float(0.61 * wavelength_nm / 1000.0 / na)


def gaussian_psf(sigma, radius=None):
    if sigma <= 0:
        raise ValueError("sigma 必须大于 0")
    r = int(np.ceil(3 * sigma)) if radius is None else int(radius)
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    k = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return k / k.sum()


def convolve_same(image, kernel):
    img = np.asarray(image, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)[::-1, ::-1]
    ry, rx = k.shape[0] // 2, k.shape[1] // 2
    pad = np.pad(img, ((ry, ry), (rx, rx)), mode="reflect")
    out = np.zeros_like(img)
    for dy in range(k.shape[0]):
        for dx in range(k.shape[1]):
            out += k[dy, dx] * pad[dy:dy + img.shape[0], dx:dx + img.shape[1]]
    return out


def simulate_capture(mean_electrons, read_noise_e, gain_e_per_adu, bit_depth, rng):
    lam = np.maximum(np.asarray(mean_electrons, dtype=np.float64), 0.0)
    e = rng.poisson(lam).astype(np.float64) + rng.normal(0.0, read_noise_e, size=lam.shape)
    adu = np.round(e / gain_e_per_adu)
    return np.clip(adu, 0, 2 ** bit_depth - 1).astype(np.int64)


def estimate_gain_from_ptc(means_adu, variances_adu):
    a, b = np.polyfit(np.asarray(means_adu, float), np.asarray(variances_adu, float), 1)
    return float(1.0 / a), float(np.sqrt(max(b, 0.0)) / a)
