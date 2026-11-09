"""作业 16-1：模板匹配、相位相关与亚像素峰值。

只允许使用 NumPy。可以在 Notebook 里用 cv2.matchTemplate 或
cv2.phaseCorrelate 对照结果，但这里的核心实现不能调用它们。
"""

from __future__ import annotations

import numpy as np


def parabola_peak_offset(left: float, center: float, right: float) -> float:
    """用三点抛物线估计峰值相对中心采样点的亚像素偏移。

    三个输入是相邻三个采样点的响应值，center 是离散峰值。
    返回 delta，真实峰值位置约为 center_index + delta。
    契约：
    - 公式 delta = (left - right) / (2 * (left - 2 * center + right))
    - 分母绝对值小于 1e-12 时返回 0.0（三点共线，没有曲率信息）
    - 结果裁剪到 [-0.5, 0.5]
    - 返回 Python float
    """
    raise NotImplementedError("TODO: 实现三点抛物线亚像素插值")


def ncc_map(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """零均值归一化互相关（ZNCC）响应图，valid 模式。

    输入：image 为 (H, W)，template 为 (h, w)，h <= H 且 w <= W，任意数值类型。
    输出：(H - h + 1, W - w + 1) 的 float64 数组，取值在 [-1, 1]。
    对每个窗口：先减去窗口均值和模板均值，再算点积 / (两者 L2 范数之积)。
    窗口或模板方差为 0（完全平坦）时，该位置输出 0.0。

    先写最直白的双重循环版本保证正确，再思考如何用积分图加速分母。
    """
    raise NotImplementedError("TODO: 实现零均值归一化互相关")


def phase_correlation(
    reference: np.ndarray,
    moving: np.ndarray,
    use_window: bool = True,
) -> tuple[float, float]:
    """用相位相关估计 moving 相对 reference 的平移 (dy, dx)，带亚像素。

    约定：如果 moving[y, x] == reference[y - dy, x - dx]（循环意义下），
    函数返回 (dy, dx)。也就是说 np.roll(reference, (dy, dx), axis=(0, 1))
    得到的 moving 应返回 (dy, dx)。

    步骤：
    1. 转为 float64；use_window=True 时两幅图都乘以二维 Hanning 窗。
    2. F_ref = fft2(reference)，F_mov = fft2(moving)。
    3. 互功率谱 R = F_mov * conj(F_ref) / (|F_mov * conj(F_ref)| + 1e-12)。
    4. r = real(ifft2(R))，找最大值的整数位置 (py, px)。
    5. 分别沿 y 和 x 方向取峰值两侧邻点（循环取下标），用 parabola_peak_offset 求亚像素偏移。
    6. 大于 N/2 的位移换算成负数（循环卷绕）。
    返回两个 Python float。
    """
    raise NotImplementedError("TODO: 实现带亚像素的相位相关")


def _demo() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(64, 64))
    mov = np.roll(ref, (3, -5), axis=(0, 1))
    try:
        print("估计平移：", phase_correlation(ref, mov, use_window=False), "期望 (3, -5)")
    except NotImplementedError as exc:
        print(f"TODO: {exc}")


if __name__ == "__main__":
    _demo()
