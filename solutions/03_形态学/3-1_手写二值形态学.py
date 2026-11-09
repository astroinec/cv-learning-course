"""参考实现（学员完成本章前不要查看）。"""

from __future__ import annotations

import numpy as np


def _windows(image, kernel, fill):
    k = np.asarray(kernel).astype(bool)
    ry, rx = k.shape[0] // 2, k.shape[1] // 2
    fg = np.asarray(image) > 0
    pad = np.pad(fg, ((ry, ry), (rx, rx)), constant_values=fill)
    h, w = fg.shape
    return [pad[dy:dy + h, dx:dx + w] for dy, dx in zip(*np.nonzero(k))]


def erode_binary(image, kernel):
    out = np.logical_and.reduce(_windows(image, kernel, False))
    return (out * 255).astype(np.uint8)


def dilate_binary(image, kernel):
    out = np.logical_or.reduce(_windows(image, kernel, False))
    return (out * 255).astype(np.uint8)
