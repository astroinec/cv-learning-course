"""作业 3-1：手写二值腐蚀与膨胀。"""

from __future__ import annotations

import numpy as np


def erode_binary(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """执行一次二值腐蚀。

    输入契约：image 为 (H,W) uint8，取值仅 0/255；kernel 为奇数尺寸
    (Kh,Kw) uint8/bool，至少含一个非零元素。Kh 和 Kw 可以不相等（例如 1×5）；
    kernel 里凡是非零的位置都算（1、255、True 都一样），为 0 的位置不参与判断。
    输出契约：同 shape 的 uint8 二值图（前景 255，背景 0）。图像外视为背景 0；只考察 kernel
    非零位置，所有对应像素均为前景时输出 255。
    “对应像素”的定义：把 kernel 的中心对准输出像素 (y, x)，kernel[i, j] 对应输入像素
    (y + i - Kh//2, x + j - Kw//2)。kernel 不翻转、不转置，和 cv2.erode 的约定相同。
    注意图像外是背景：贴着图像边界的前景会从边界往里缩。cv2.erode 的默认参数不是这样，
    和它对照时要写 borderType=cv2.BORDER_CONSTANT, borderValue=0。
    不修改传进来的 image 和 kernel，结果写到新数组里。
    """
    raise NotImplementedError("TODO: 实现一次二值腐蚀")


def dilate_binary(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """执行一次二值膨胀。

    输入契约与 erode_binary 相同，“对应像素”的定义也相同：kernel 的中心对准输出像素 (y, x)，
    kernel[i, j] 对应输入像素 (y + i - Kh//2, x + j - Kw//2)，kernel 不翻转（和 cv2.dilate 相同）。
    输出契约：同 shape 的 uint8 二值图（前景 255，背景 0）。图像外视为背景 0；kernel 非零
    位置只要有一个对应前景像素，输出即为 255。
    注意这是“命中”的定义，不是“把 kernel 盖到每个前景点上”：两者在 kernel 关于中心对称时相同，
    不对称时差一个 180 度的翻转（讲义 3.3 节）。
    不修改传进来的 image 和 kernel，结果写到新数组里。
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
