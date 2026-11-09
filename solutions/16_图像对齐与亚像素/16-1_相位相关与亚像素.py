"""参考实现 16-1。学员完成本章前不要查看。"""

from __future__ import annotations

import numpy as np


def parabola_peak_offset(left: float, center: float, right: float) -> float:
    denom = left - 2.0 * center + right
    if abs(denom) < 1e-12:
        return 0.0
    delta = (left - right) / (2.0 * denom)
    return float(np.clip(delta, -0.5, 0.5))


def ncc_map(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    tpl = np.asarray(template, dtype=np.float64)
    h, w = tpl.shape
    out_h, out_w = img.shape[0] - h + 1, img.shape[1] - w + 1
    t = tpl - tpl.mean()
    t_norm = np.sqrt((t * t).sum())
    out = np.zeros((out_h, out_w), dtype=np.float64)
    if t_norm < 1e-12:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(img, (h, w))
    for y in range(out_h):
        for x in range(out_w):
            win = windows[y, x]
            wz = win - win.mean()
            w_norm = np.sqrt((wz * wz).sum())
            if w_norm < 1e-12:
                continue
            out[y, x] = float((wz * t).sum() / (w_norm * t_norm))
    return np.clip(out, -1.0, 1.0)


def phase_correlation(reference, moving, use_window=True):
    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    if use_window:
        win = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1]))
        ref = ref * win
        mov = mov * win
    cross = np.fft.fft2(mov) * np.conj(np.fft.fft2(ref))
    cross /= np.abs(cross) + 1e-12
    r = np.real(np.fft.ifft2(cross))
    py, px = np.unravel_index(np.argmax(r), r.shape)
    h, w = r.shape
    oy = parabola_peak_offset(r[(py - 1) % h, px], r[py, px], r[(py + 1) % h, px])
    ox = parabola_peak_offset(r[py, (px - 1) % w], r[py, px], r[py, (px + 1) % w])
    dy = py + oy
    dx = px + ox
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return float(dy), float(dx)
