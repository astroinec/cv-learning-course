"""作业 2-1：手写二维相关运算（不要调用 cv2.filter2D）。"""

from __future__ import annotations

import numpy as np


def correlate2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """对单通道图像执行二维相关，并使用 reflect padding 保持尺寸。

    输入契约：
      image: (H, W) 数值数组，H/W >= 2。
      kernel: (Kh, Kw) 奇数尺寸数值数组，Kh/Kw >= 1。
    输出契约：
      (H, W) 的 np.float32 数组；核不翻转；边界使用 NumPy reflect 语义。

    TODO：验证输入后填充图像，逐位置计算窗口与 kernel 的乘积和。
    禁止在核心实现中调用 cv2.filter2D、scipy.signal 或现成卷积函数。
    """
    raise NotImplementedError("TODO: 实现二维相关、reflect padding 与 float32 输出")


def _demo() -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[:, 32:] = 1.0
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    try:
        response = correlate2d(image, sobel_x)
    except NotImplementedError as exc:
        print(f"TODO: {exc}")
        return
    print("输出:", response.shape, response.dtype, float(response.min()), float(response.max()))


if __name__ == "__main__":
    _demo()
