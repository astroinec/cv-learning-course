"""作业 6-1：用状态和前向—反向误差筛选光流轨迹。"""

from __future__ import annotations

import numpy as np


def filter_tracks(
    previous_points: np.ndarray,
    forward_points: np.ndarray,
    backward_points: np.ndarray,
    status: np.ndarray,
    max_fb_error: float = 1.0,
) -> np.ndarray:
    """返回可信轨迹的布尔掩码。

    输入契约：前三个点集均可为 (N,2) 或 (N,1,2)，且 shape 一致；status
    含 N 个元素；max_fb_error >= 0。forward_points 用于检查有限值。
    输出契约：(N,) bool。仅当 status 非零、前向/反向坐标均有限，且
    ||previous_points - backward_points||_2 <= max_fb_error 时为 True。

    TODO：统一点集为 (N,2)，逐点计算往返误差并组合条件。
    """
    raise NotImplementedError("TODO: 实现状态、有限值与往返误差联合筛选")


def _demo() -> None:
    prev = np.array([[5, 5], [10, 10]], dtype=np.float32)
    forward = prev + np.array([2, 1], dtype=np.float32)
    backward = np.array([[5.1, 5], [13, 10]], dtype=np.float32)
    try:
        print(filter_tracks(prev, forward, backward, np.array([1, 1]), 0.5))
    except NotImplementedError as exc:
        print(f"TODO: {exc}")


if __name__ == "__main__":
    _demo()
