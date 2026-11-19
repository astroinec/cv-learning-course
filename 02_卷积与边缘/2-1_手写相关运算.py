"""作业 2-1：手写二维相关运算（不要调用 cv2.filter2D）。"""

from __future__ import annotations

import numpy as np


def correlate2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """对单通道图像执行二维相关，并使用 reflect padding 保持尺寸。

    输入契约：
      image: (H, W) 数值数组，H/W >= 2；可能是 uint8、别的整数类型或浮点类型。
      kernel: (Kh, Kw) 奇数尺寸数值数组，Kh/Kw >= 1；Kh 和 Kw 可以不相等（例如 1×5、5×1），
        类型同样可能是整数或浮点。
    输出契约：
      (H, W) 的 np.float32 数组；核不翻转；边界使用 NumPy reflect 语义。展开来说：
      - out[y, x] 等于把 kernel 的中心对准 (y, x) 后，各个 kernel[i, j] 与它盖住的像素
        (y + i - Kh//2, x + j - Kw//2) 相乘再求和。kernel 不做翻转、转置或归一化。
      - 盖到图像外面的位置，按 np.pad 的 mode="reflect" 取值：以边界像素为轴做镜像，边界像素
        自己不重复（OpenCV 里叫 BORDER_REFLECT_101，是 cv2.filter2D 的默认值）。
      - 先把 image 和 kernel 转成浮点再算。整数类型直接相乘会回绕；输出保留正负号。
      - 不修改传进来的 image 和 kernel。

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
