"""参考实现（学员完成本章前不要查看）。"""

from __future__ import annotations

import numpy as np


def correlate2d(image, kernel):
    img = np.asarray(image, dtype=np.float32)
    k = np.asarray(kernel, dtype=np.float32)
    if img.ndim != 2 or k.ndim != 2 or k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
        raise ValueError("需要二维图像和奇数尺寸核")
    ry, rx = k.shape[0] // 2, k.shape[1] // 2
    pad = np.pad(img, ((ry, ry), (rx, rx)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    for dy in range(k.shape[0]):
        for dx in range(k.shape[1]):
            out += k[dy, dx] * pad[dy:dy + img.shape[0], dx:dx + img.shape[1]]
    return out.astype(np.float32)
