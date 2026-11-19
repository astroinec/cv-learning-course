"""第 7 章自动验收：TODO / FAIL / PASS。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "7-1_针孔投影.py"
CAMERA = np.array([[500, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
# 绕 Z 轴转 90 度：(x, y, z) 变成 (-y, x, z)
ROT_Z90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
# 绕 X 轴转 90 度：(x, y, z) 变成 (x, -z, y)
ROT_X90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def load_assignment():
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    spec = importlib.util.spec_from_file_location("chapter7_assignment", base / ASSIGNMENT)
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


def is_contract(out, n: int) -> bool:
    return isinstance(out, np.ndarray) and out.shape == (n, 2) and out.dtype == np.float64


def check_identity_pose(func):
    points = np.array([[0, 0, 2], [1, 0, 2], [0, 1, 3], [0, 0, -1]], dtype=np.float64)
    pixels = func(points, CAMERA, np.eye(3), np.zeros(3))
    if not is_contract(pixels, 4):
        return False, "输出必须是 (N,2) float64：打印返回值的 shape 和 dtype"
    if np.isnan(pixels[:3]).any():
        return False, "一批点里只有一个在相机后方时，其余的点应照常计算"
    if not np.allclose(pixels[:3], [[320, 240], [570, 240], [320, 440]], atol=1e-9):
        return False, "单位旋转、零平移下的结果不对：检查主点、fx 和 fy 各自乘在哪个坐标上，以及除的是不是深度 Z"
    if not np.isnan(pixels[3]).all():
        return False, "相机后方的点两个坐标都应是 NaN"
    ray = func(np.array([[1, 2, 4], [2, 4, 8]], dtype=np.float64), CAMERA, np.eye(3), np.zeros(3))
    if not is_contract(ray, 2) or not np.allclose(ray[0], ray[1]):
        return False, "同一条相机射线上深度不同的两个点应投到同一个像素"
    return True, "单位位姿下主点、焦距、透视除法和相机后方的点都正确"


def check_pose(func):
    # 手算的期望值：R 把 (x, y, z) 变成 (-y, x, z)，再加 t = (1, -2, 4)
    points = np.array([[2, 1, 0], [1, 3, 4], [0, 0, 0], [5, 2, 2]], dtype=np.int64)
    translation = np.array([1, -2, 4], dtype=np.int64)
    expected = np.array([[320.0, 240.0], [195.0, 165.0], [445.0, -60.0], [320.0 - 500.0 / 6.0, 540.0]])
    pixels = func(points, CAMERA, ROT_Z90, translation)
    if not is_contract(pixels, 4):
        return False, "输入是整数数组时输出也必须是 (N,2) float64"
    if not np.isfinite(pixels).all():
        return False, "绕 Z 轴转 90 度再平移 (1, -2, 4) 之后，这组点在相机坐标系里的深度是 4、8、4、6，都在相机前方，不应出现 NaN：检查判断深度用的是不是经过 R 和 t 之后的 Z"
    if np.allclose(pixels, np.trunc(expected), atol=1e-9) and not np.allclose(pixels, expected, atol=1e-9):
        return False, "输入是整数数组时结果的小数部分丢了：检查中间结果存进了什么类型的数组"
    if not np.allclose(pixels, expected, atol=1e-3):
        return False, ("绕 Z 轴转 90 度再平移 (1, -2, 4) 的位姿下结果不对：世界点 (2, 1, 0) 的相机坐标应是 (0, 0, 4)，投到主点。"
                       "拿这个点手算一遍，检查 R 和 t 有没有用上、R 有没有写成转置、旋转和平移的先后")
    if not np.allclose(pixels, expected, atol=1e-9, rtol=0):
        return False, "结果只有单精度的准确度：输入要先转成 float64 再计算，而不是最后才转"
    # 恰好 3 个点、translation 是 (3,1)：形状不统一时不会报错，但结果是错的
    try:
        column = func(points[:3], CAMERA, ROT_Z90, translation.reshape(3, 1).astype(np.float64))
        two = func(points[:2], CAMERA, ROT_Z90, translation.reshape(3, 1).astype(np.float64))
    except ValueError as exc:
        return False, f"translation 是 (3,1) 时运行出错（{exc}）：契约允许 (3,) 和 (3,1) 两种形状，相加之前打印两边的 shape"
    if not is_contract(column, 3) or not np.allclose(column, expected[:3], atol=1e-9):
        return False, "translation 是 (3,1)、点数恰好是 3 时结果不对：打印相机坐标的 shape，应是 (N,3)，每个点加的是同一个 t"
    if not is_contract(two, 2) or not np.allclose(two, expected[:2], atol=1e-9):
        return False, "translation 是 (3,1) 时结果应和 (3,) 时相同"
    return True, "旋转、平移和两种 translation 形状下的结果都和手算一致，整数输入不丢小数"


def check_depth(func):
    # R 把 (x, y, z) 变成 (x, -z, y)，t = (0, 0, 1)：相机坐标的深度是世界坐标的 y 加 1
    translation = np.array([0.0, 0.0, 1.0])
    points = np.array([[0.0, 3.0, -5.0], [1.0, -3.0, 2.0], [2.0, 1.0, 0.0]])
    pixels = func(points, CAMERA, ROT_X90, translation)
    if not is_contract(pixels, 3):
        return False, "输出必须是 (N,2) float64"
    if np.isnan(pixels[0]).any():
        return False, "世界坐标 Z 为负、但转到相机坐标后深度为 4 的点应正常投影：有效与否看的是相机坐标的 Z"
    if not np.isnan(pixels[1]).all():
        return False, "世界坐标 Z 为正、但转到相机坐标后深度为 -2 的点应返回 NaN：有效与否看的是相机坐标的 Z"
    if not np.allclose(pixels[[0, 2]], [[320.0, 990.0], [820.0, 240.0]], atol=1e-9):
        return False, "旋转改变深度的位姿下，有效点的结果不对"
    edge = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 1e-13], [0.0, 0.0, -1e-13], [0.0, 0.0, 1e-11]])
    pixels = func(edge, CAMERA, np.eye(3), np.zeros(3))
    if not np.isnan(pixels[0]).all():
        return False, "深度正好为 0 的点应返回 NaN（不是 inf，也不是报错）"
    if not np.isnan(pixels[1]).all() or not np.isnan(pixels[2]).all():
        return False, "深度是 1e-13 或 -1e-13 的点也应返回 NaN：契约是 Z 不超过 1e-12"
    if not np.allclose(pixels[3], [320.0, 240.0], atol=1e-6):
        return False, "深度是 1e-11 的点仍应正常计算：判无效的门限是 1e-12，不要自己另取一个更大的"
    return True, "按相机坐标的深度判断有效，门限两侧都正确"


def check_against_opencv(func):
    try:
        import cv2
    except ImportError:
        return True, "没有安装 OpenCV，跳过和 cv2.projectPoints 的对照"
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(5):
        rvec = rng.uniform(-1.2, 1.2, size=3)
        rotation, _ = cv2.Rodrigues(rvec)
        translation = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(9, 12)])
        points = rng.uniform(-2.0, 2.0, size=(40, 3))     # 离原点不超过 3.5，深度一定为正
        camera = np.array([[rng.uniform(400, 900), 0, rng.uniform(250, 400)], [0, rng.uniform(400, 900), rng.uniform(180, 300)], [0, 0, 1]])
        want, _ = cv2.projectPoints(points, rvec, translation, camera, None)
        got = func(points, camera, rotation, translation)
        if not is_contract(got, 40):
            return False, "输出必须是 (N,2) float64"
        if not np.isfinite(got).all():
            return False, "随机位姿下这些点都在相机前方 6 个单位以外，不应出现 NaN"
        worst = max(worst, float(np.abs(got - want.reshape(-1, 2)).max()))
    if worst > 1e-7:
        return False, f"随机位姿下和 cv2.projectPoints（畸变传零）最大相差 {worst:.3g} px：按世界到相机、透视除法、乘内参三步逐步打印中间结果对照"
    return True, "5 组随机位姿下和 cv2.projectPoints 一致"


def check_inputs_untouched(func):
    points = np.array([[0.5, -0.25, 1.0], [1.0, 2.0, -3.0], [0.0, 0.0, 4.0]])
    camera, rotation, translation = CAMERA.copy(), ROT_Z90.copy(), np.array([[0.5], [1.5], [2.0]])
    copies = [a.copy() for a in (points, camera, rotation, translation)]
    func(points, camera, rotation, translation)
    for name, now, before in zip(("points_3d", "camera_matrix", "rotation", "translation"), (points, camera, rotation, translation), copies):
        if not np.array_equal(now, before):
            return False, f"调用之后传进来的 {name} 被改了：不要在输入数组上原地运算"
    return True, "没有修改传进来的数组"


CHECKS = [
    ("单位位姿", check_identity_pose),
    ("旋转与平移", check_pose),
    ("深度判断", check_depth),
    ("对照 OpenCV", check_against_opencv),
    ("不改输入", check_inputs_untouched),
]


def run_all(func) -> list[bool]:
    return [run(name, check, func) for name, check in CHECKS]


def main() -> int:
    print("第 7 章进度检查\n" + "=" * 52)
    try:
        func = load_assignment().project_points
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "加载作业", f"{type(exc).__name__}: {exc}")
        return 1
    results = run_all(func)
    print("=" * 52)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("全部通过；本题不含畸变，畸变和位姿估计见教程 7.1、7.2 节和第 17 章。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
