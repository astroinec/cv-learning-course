"""作业 5-1：计算 Harris 角点响应图。"""

from __future__ import annotations

import numpy as np


def harris_response(gray: np.ndarray, k: float = 0.04, window_size: int = 3) -> np.ndarray:
    """计算每个像素的 Harris 响应 R = det(M) - k*trace(M)^2。

    输入契约：gray 为 (H,W) 灰度数组，可能是 uint8，也可能是浮点（0 到 1 或 0 到 255）；
    window_size 为正奇数；0 < k < 0.25。
    输出契约：与输入同 shape 的 np.float32 数组，所有值有限。
      - M 的三个元素是 Ix^2、Iy^2、IxIy 在以该像素为中心的 window_size x window_size
        窗口里的汇总。方框窗或同样大小的高斯窗都可以，归一化与否都可以；
        window_size=1 时窗口里只有像素自己。
      - 梯度在浮点图上计算，x、y 两个方向用同一种对称的差分算子（3x3 Sobel 即可）。
      - 输入和输出都不做依赖图像内容的缩放或截断（不除以最大值、不拉伸对比度、
        不把负值截成 0）：图像整体乘 a，响应乘 a^4；直边上的响应是负数。
      - 求梯度时图像边界用反射或复制的方式外推（cv2.Sobel 的默认边界即可），不补零：
        常数图上每个像素的响应都是 0。窗口汇总时边界怎么处理不作要求。
      - 参数 k 和 window_size 都要真正用上。

    TODO 建议步骤：转 float32；用 Sobel 得 Ix/Iy；计算 Ix^2、Iy^2、IxIy；
    用方框或高斯窗口汇总；构造 det/trace。可用 cv2.Sobel 和 cv2.boxFilter，
    但不得调用 cv2.cornerHarris 或 cv2.goodFeaturesToTrack 完成核心计算。
    """
    raise NotImplementedError("TODO: 实现结构张量与 Harris 响应")


def _demo() -> None:
    image = np.zeros((64, 64), dtype=np.uint8)
    image[16:48, 16:48] = 255
    try:
        response = harris_response(image)
    except NotImplementedError as exc:
        print(f"TODO: {exc}")
        return
    y, x = np.unravel_index(np.argmax(response), response.shape)
    print("最大响应位置:", (int(x), int(y)), "值:", float(response[y, x]))


if __name__ == "__main__":
    _demo()
