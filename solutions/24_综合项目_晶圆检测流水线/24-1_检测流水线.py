"""参考实现 24-1（自包含，不依赖其他章节）。学员完成本章前不要查看。"""

from __future__ import annotations

from collections import deque

import numpy as np


def _parabola(l, c, r):
    d = l - 2 * c + r
    return 0.0 if abs(d) < 1e-12 else float(np.clip((l - r) / (2 * d), -0.5, 0.5))


def _phase_corr(ref, mov, max_shift=8, window=False):
    a, b = ref - ref.mean(), mov - mov.mean()
    if window:
        win = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1]))
        a, b = a * win, b * win
    cross = np.fft.fft2(b) * np.conj(np.fft.fft2(a))
    r = np.real(np.fft.ifft2(cross / (np.abs(cross) + 1e-12)))
    h, w = r.shape
    yy = np.fft.fftfreq(h, 1.0 / h)[:, None]
    xx = np.fft.fftfreq(w, 1.0 / w)[None, :]
    searchable = np.where((np.abs(yy) <= max_shift) & (np.abs(xx) <= max_shift), r, -np.inf)
    py, px = np.unravel_index(np.argmax(searchable), r.shape)
    dy = py + _parabola(r[(py - 1) % h, px], r[py, px], r[(py + 1) % h, px])
    dx = px + _parabola(r[py, (px - 1) % w], r[py, px], r[py, (px + 1) % w])
    return (dy - h if dy > h / 2 else dy), (dx - w if dx > w / 2 else dx)


def _shift(img, dy, dx):
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.fftfreq(img.shape[1])[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j * np.pi * (fy * dy + fx * dx))))


def _align(ref, img):
    dy, dx = _phase_corr(ref, img)
    ry, rx = _phase_corr(ref, _shift(img, -dy, -dx), max_shift=1)
    return dy + ry, dx + rx


def _labels(mask):
    h, w = mask.shape
    lab = np.zeros((h, w), dtype=np.int32)
    n = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not lab[y, x]:
                n += 1
                lab[y, x] = n
                q = deque([(y, x)])
                while q:
                    cy, cx = q.popleft()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not lab[ny, nx]:
                                lab[ny, nx] = n
                                q.append((ny, nx))
    return lab, n


def _normalize(img, ref):
    a, b = np.polyfit(img.ravel(), ref.ravel(), 1)
    return a * img + b


def run_pipeline(golden, test, k=6.0, min_area=4, border=5):
    ref = golden[0]
    aligned_golden = []
    for g in golden:
        dy, dx = _align(ref, g)
        aligned_golden.append(_shift(g, -dy, -dx))
    stack = np.stack(aligned_golden)
    template = np.median(stack, axis=0)
    stack = np.stack([_normalize(g, template) for g in stack])
    median = np.median(stack, axis=0)
    sigma = np.maximum(1.4826 * np.median(np.abs(stack - median), axis=0), 0.02)

    out = []
    for img in test:
        dy, dx = _align(ref, img)
        a = _normalize(_shift(img, -dy, -dx), median)
        z = np.abs(a - median) / sigma
        mask = z > k
        mask[:border, :] = mask[-border:, :] = False
        mask[:, :border] = mask[:, -border:] = False
        lab, n = _labels(mask)
        found = []
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            if len(ys) < min_area:
                continue
            wgt = z[ys, xs]
            cy, cx = float((ys * wgt).sum() / wgt.sum()), float((xs * wgt).sum() / wgt.sum())
            found.append(((cy + dy) % img.shape[0], (cx + dx) % img.shape[1]))
        out.append(found)
    return out
