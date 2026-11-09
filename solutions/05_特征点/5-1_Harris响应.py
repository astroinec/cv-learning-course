"""参考实现（学员完成本章前不要查看）。"""

from __future__ import annotations

import cv2
import numpy as np


def harris_response(gray, k=0.04, window_size=3):
    g = np.asarray(gray, dtype=np.float32)
    ix = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    box = lambda a: cv2.boxFilter(a, cv2.CV_32F, (window_size, window_size), normalize=True)
    sxx, syy, sxy = box(ix * ix), box(iy * iy), box(ix * iy)
    r = sxx * syy - sxy * sxy - k * (sxx + syy) ** 2
    return np.nan_to_num(r).astype(np.float32)
