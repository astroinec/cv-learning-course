"""第 6 章自动验收：TODO / FAIL / PASS。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "6-1_前向反向筛选.py"


def load_assignment():
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    spec = importlib.util.spec_from_file_location("chapter6_assignment", base / ASSIGNMENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(status: str, name: str, detail: str) -> None:
    print(f"[{status:4}] {name} — {detail}")


def run(name: str, check, func) -> bool:
    try:
        with np.errstate(all="ignore"):
            ok, detail = check(func)
    except NotImplementedError as exc:
        report("TODO", name, str(exc))
        return False
    except Exception as exc:  # noqa: BLE001
        report("FAIL", name, f"运行异常：{type(exc).__name__}: {exc}")
        return False
    report("PASS" if ok else "FAIL", name, detail)
    return ok


def is_mask(mask, n: int) -> bool:
    return isinstance(mask, np.ndarray) and mask.shape == (n,) and mask.dtype == np.bool_


def tracks(n: int = 6):
    """n 条起点各不相同、前向位移 (2, 1)、反向正好回到起点的轨迹，形状 (n,2)，float32。"""
    prev = np.array([[10.0 * i, 5.0 + 3.0 * i] for i in range(n)], dtype=np.float32)
    return prev, prev + np.array([2.0, 1.0], dtype=np.float32), prev.copy(), np.ones(n, dtype=np.uint8)


def check_contract(func):
    prev, forward, backward, status = tracks(4)
    backward[2] += [2.0, 0.0]
    try:
        flat = func(prev, forward, backward, status, 1.0)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        return False, f"点集是 (N,2)、status 是 (N,) 时运行出错（{type(exc).__name__}）：契约允许点集是 (N,2) 或 (N,1,2)，status 只保证含 N 个元素"
    if not is_mask(flat, 4):
        return False, "点集是 (N,2)、status 是 (N,) 时，输出必须是 (N,) bool：打印返回值的 shape 和 dtype"
    nested_points = [a.reshape(-1, 1, 2) for a in (prev, forward, backward)]
    try:
        nested = func(*nested_points, status.reshape(-1, 1), 1.0)
        mixed = func(*nested_points, status, 1.0)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        return False, f"点集是 (N,1,2) 时运行出错（{type(exc).__name__}）：契约允许点集是 (N,2) 或 (N,1,2)，status 只保证含 N 个元素"
    if not is_mask(nested, 4):
        return False, "点集是 (N,1,2)、status 是 (N,1) 时，输出也必须是 (N,) bool：检查有没有广播成 (N,N) 或者留下了多余的维度"
    if not is_mask(mixed, 4):
        return False, "点集是 (N,1,2)、status 是 (N,) 时，输出也必须是 (N,) bool：status 先展平再用"
    if not (np.array_equal(nested, flat) and np.array_equal(mixed, flat)):
        return False, "同一批数据只是换了点集和 status 的形状，结果应完全相同"
    try:
        single = func(prev[:1].reshape(1, 1, 2), forward[:1].reshape(1, 1, 2), backward[:1].reshape(1, 1, 2), status[:1].reshape(1, 1), 1.0)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        return False, f"只有一条轨迹时运行出错（{type(exc).__name__}）：统一形状时不要用会把长度为 1 的那一维也去掉的办法"
    if not is_mask(single, 1):
        return False, "只有一条轨迹时应返回形状 (1,) 的 bool 数组"
    return True, "两种点集形状、两种 status 形状和单条轨迹都返回 (N,) bool，结果一致"


def check_error(func):
    prev, forward, backward, status = tracks(6)
    backward[1] += [0.0, 1.0]     # 误差正好 1
    backward[2] += [2.0, 0.0]     # 误差 2
    backward[3] += [3.0, 4.0]     # 误差正好 5
    backward[4] += [4.0, 4.0]     # 误差约 5.66
    backward[5] += [-1.5, 0.0]    # 误差 1.5
    got = func(prev, forward, backward, status, 1.0)
    if not is_mask(got, 6):
        return False, "输出必须是 (N,) bool"
    if not bool(got[0]):
        return False, "反向点正好回到起点的轨迹被剔除了：往返误差是起点和反向点之间的距离，和前向点无关"
    if not bool(got[1]):
        return False, "往返误差正好等于阈值的轨迹应保留：契约是不超过阈值"
    if got[2:].any():
        return False, "阈值是 1 时，往返误差为 2、5、5.66、1.5 的轨迹都应剔除"
    got = func(prev, forward, backward, status, 5.0)
    if not np.array_equal(got[:3], [True, True, True]):
        return False, "阈值改成 5 之后，误差为 0、1、2 的轨迹都应保留：检查参数 max_fb_error 有没有真正用上"
    if not bool(got[3]):
        return False, "起点和反向点相差 (3, 4) 时往返误差是 5，阈值为 5 应保留：误差是两点之间的直线距离"
    if bool(got[4]):
        return False, "起点和反向点相差 (4, 4) 时往返误差约为 5.66，阈值为 5 应剔除：误差是两点之间的直线距离，不是单个坐标的差"
    if not bool(got[5]):
        return False, "阈值为 5 时误差 1.5 的轨迹应保留"
    got = func(prev, forward, backward, status, 2.0)
    if not np.array_equal(got, [True, True, True, False, False, True]):
        return False, "阈值为 2 时应保留误差为 0、1、2、1.5 的轨迹，剔除误差为 5 和 5.66 的：检查距离有没有开方"
    got = func(prev, forward, backward, status)
    if not np.array_equal(got, [True, True, False, False, False, False]):
        return False, "不传 max_fb_error 时应按默认值 1.0 筛选"
    got = func(prev, forward, backward, status, 0.0)
    if not np.array_equal(got, [True, False, False, False, False, False]):
        return False, "阈值为 0 时只保留反向点和起点完全重合的轨迹"
    return True, "往返误差按直线距离计算，阈值边界和默认值正确"


def check_status(func):
    prev, forward, backward, _ = tracks(5)
    status = np.array([1, 0, 2, 255, 1], dtype=np.uint8)
    got = func(prev, forward, backward, status, 1.0)
    if not is_mask(got, 5):
        return False, "输出必须是 (N,) bool"
    if bool(got[1]):
        return False, "status 为 0 的轨迹必须剔除，即使往返误差是 0"
    if not (got[0] and got[4]):
        return False, "status 为 1、其余条件都满足的轨迹应保留"
    if not (got[2] and got[3]):
        return False, "status 为 2 或 255 的轨迹也应保留：契约是 status 非零"
    as_bool = func(prev, forward, backward, np.array([True, False, True, True, False]), 1.0)
    if not np.array_equal(as_bool, [True, False, True, True, False]):
        return False, "status 是 bool 数组时结果不对"
    return True, "status 为 0 的轨迹被剔除，非零值都按有效处理"


def check_finite(func):
    prev, forward, backward, status = tracks(7)
    forward[1, 0] = np.nan
    forward[2, 1] = np.nan
    forward[3, 0] = np.inf
    backward[4, 1] = np.nan
    backward[5, 0] = -np.inf
    got = func(prev, forward, backward, status, 1.0)
    if not is_mask(got, 7):
        return False, "输出必须是 (N,) bool"
    if not (got[0] and got[6]):
        return False, "同一批里别的轨迹含 NaN 或 inf，不应影响坐标正常的轨迹"
    if got[1] or got[2]:
        return False, "前向点有一个坐标是 NaN 的轨迹应剔除：x 和 y 都要查，而且前向点不参与往返误差，不会自动被比较筛掉"
    if got[3]:
        return False, "前向点坐标是 inf 的轨迹应剔除：契约是坐标有限，不只是不为 NaN"
    if got[4] or got[5]:
        return False, "反向点坐标是 NaN 或 inf 的轨迹应剔除"
    return True, "前向点、反向点里的 NaN 和 inf 都被剔除，不影响同批其余轨迹"


def check_inputs_untouched(func):
    prev, forward, backward, status = tracks(4)
    forward[1, 0] = np.nan
    backward[2, 1] = np.inf
    status[3] = 0
    copies = [a.copy() for a in (prev, forward, backward, status)]
    func(prev, forward, backward, status, 1.0)
    for name, now, before in zip(("previous_points", "forward_points", "backward_points", "status"), (prev, forward, backward, status), copies):
        if not np.array_equal(now, before, equal_nan=True):
            return False, f"调用之后传进来的 {name} 被改了：不要在输入数组上原地运算"
    return True, "没有修改传进来的数组"


CHECKS = [
    ("形状与类型", check_contract),
    ("往返误差", check_error),
    ("status", check_status),
    ("有限值", check_finite),
    ("不改输入", check_inputs_untouched),
]


def run_all(func) -> list[bool]:
    return [run(name, check, func) for name, check in CHECKS]


def main() -> int:
    print("第 6 章进度检查\n" + "=" * 52)
    try:
        func = load_assignment().filter_tracks
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "加载作业", f"{type(exc).__name__}: {exc}")
        return 1
    results = run_all(func)
    print("=" * 52)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("全部通过；阈值取多大要在真实跟踪里自己调，见教程 6.3 节。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
