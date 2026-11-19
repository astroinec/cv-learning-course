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

    输入契约：前三个点集均可为 (N,2) 或 (N,1,2)，且 shape 一致，N >= 1；status
    含 N 个元素，形状可能是 (N,) 或 (N,1)，类型可能是 uint8 或 bool；
    max_fb_error >= 0。status 应传“前向和反向两次跟踪都成功”的结果，也就是两次
    calcOpticalFlowPyrLK 返回的 status 逐点相与；只做了一次跟踪时传那一次的。
    forward_points 只用于检查有限值，不参与误差计算。
    输出契约：(N,) bool。仅当 status 非零（不只是等于 1）、前向和反向坐标均有限
    （没有 NaN，也没有 inf），且起点到反向点的直线距离
    ||previous_points - backward_points||_2 <= max_fb_error 时为 True。
    一条轨迹无效不影响同一批里的其他轨迹。不修改传进来的数组。

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
