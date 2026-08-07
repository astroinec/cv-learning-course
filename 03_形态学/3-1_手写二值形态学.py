"""作业 3-1：手写二值腐蚀与膨胀。"""

from __future__ import annotations

import numpy as np


def erode_binary(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """执行一次二值腐蚀。

    输入契约：image 为 (H,W) uint8，取值仅 0/255；kernel 为奇数尺寸
    (Kh,Kw) uint8/bool，至少含一个非零元素。
    输出契约：同 shape 的 uint8 二值图。图像外视为背景 0；只考察 kernel
    非零位置，所有对应像素均为前景时输出 255。
    """
    raise NotImplementedError("TODO: 实现一次二值腐蚀")


def dilate_binary(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """执行一次二值膨胀。

    输入契约与 erode_binary 相同。
    输出契约：同 shape 的 uint8 二值图。图像外视为背景 0；kernel 非零
    位置只要有一个对应前景像素，输出即为 255。
    """
    raise NotImplementedError("TODO: 实现一次二值膨胀")


def _demo() -> None:
    image = np.zeros((9, 9), dtype=np.uint8)
    image[3:6, 3:6] = 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    try:
        print("腐蚀前景数:", np.count_nonzero(erode_binary(image, kernel)))
        print("膨胀前景数:", np.count_nonzero(dilate_binary(image, kernel)))
    except NotImplementedError as exc:
        print(f"TODO: {exc}")


if __name__ == "__main__":
    _demo()
