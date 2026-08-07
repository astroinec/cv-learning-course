"""作业 5-1：计算 Harris 角点响应图。"""

from __future__ import annotations

import numpy as np


def harris_response(gray: np.ndarray, k: float = 0.04, window_size: int = 3) -> np.ndarray:
    """计算每个像素的 Harris 响应 R = det(M) - k*trace(M)^2。

    输入契约：gray 为 (H,W) 灰度数组；window_size 为正奇数；0 < k < 0.25。
    输出契约：与输入同 shape 的 np.float32 数组，所有值有限。

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
