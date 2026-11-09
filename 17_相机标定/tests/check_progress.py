"""第 17 章自动验收：TODO / FAIL / PASS。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "17-1_张正友标定.py"


def load_assignment():
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    spec = importlib.util.spec_from_file_location("chapter17_assignment", base / ASSIGNMENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(status, name, detail):
    print(f"[{status:4}] {name} — {detail}")


def run(name, func):
    try:
        ok, detail = func()
    except NotImplementedError as exc:
        report("TODO", name, str(exc))
        return False
    except Exception as exc:  # noqa: BLE001
        report("FAIL", name, f"运行异常：{type(exc).__name__}: {exc}")
        return False
    report("PASS" if ok else "FAIL", name, detail)
    return ok


def rodrigues(rvec):
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * kx @ kx


K_TRUE = np.array([[820.0, 0.0, 318.0], [0.0, 790.0, 247.0], [0.0, 0.0, 1.0]])
POSES = [
    (np.array([0.25, -0.15, 0.05]), np.array([-60.0, -45.0, 420.0])),
    (np.array([-0.30, 0.20, -0.10]), np.array([-40.0, -50.0, 380.0])),
    (np.array([0.10, 0.35, 0.20]), np.array([-70.0, -30.0, 450.0])),
    (np.array([-0.20, -0.30, 0.15]), np.array([-50.0, -60.0, 400.0])),
    (np.array([0.40, 0.05, -0.20]), np.array([-55.0, -40.0, 500.0])),
]


def board():
    xs, ys = np.meshgrid(np.arange(9) * 15.0, np.arange(6) * 15.0)
    return np.stack([xs.ravel(), ys.ravel()], axis=1)


def project(K, R, t, pts3):
    pc = pts3 @ R.T + t
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:3]


def views(noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    xy = board()
    pts3 = np.hstack([xy, np.zeros((len(xy), 1))])
    out = []
    for rvec, t in POSES:
        R = rodrigues(rvec)
        uv = project(K_TRUE, R, t, pts3) + rng.normal(scale=noise, size=(len(xy), 2))
        out.append((xy, pts3, uv, R, t))
    return out


def check_normalize(mod):
    rng = np.random.default_rng(0)
    pts = rng.uniform(100, 500, size=(30, 2))
    pn, T = mod.normalize_points(pts)
    if pn.shape != (30, 2) or T.shape != (3, 3):
        return False, "输出形状应为 (N,2) 与 (3,3)"
    if not np.allclose(pn.mean(axis=0), 0, atol=1e-9):
        return False, "归一化后质心应在原点"
    if abs(np.linalg.norm(pn, axis=1).mean() - np.sqrt(2)) > 1e-9:
        return False, "归一化后平均距离应为 sqrt(2)"
    homo = np.hstack([pts, np.ones((30, 1))]) @ T.T
    if not np.allclose(homo[:, :2] / homo[:, 2:3], pn):
        return False, "points_n 必须等于 T 作用在原始点上的结果"
    try:
        mod.normalize_points(np.ones((5, 2)))
    except ValueError:
        return True, "质心、尺度和 T 一致，退化输入报错"
    return False, "所有点重合时应抛出 ValueError"


def check_homography(mod):
    xy, _, uv, R, t = views()[0]
    H = mod.homography_dlt(xy, uv)
    if H.shape != (3, 3) or abs(H[2, 2] - 1) > 1e-9:
        return False, "H 必须是 (3,3) 且 H[2,2] == 1"
    mapped = np.hstack([xy, np.ones((len(xy), 1))]) @ H.T
    err = np.abs(mapped[:, :2] / mapped[:, 2:3] - uv).max()
    if err > 1e-6:
        return False, f"无噪声时映射误差 {err:.2e} px，应小于 1e-6"
    scaled = mod.homography_dlt(xy * 1000.0, uv)
    mapped2 = np.hstack([xy * 1000.0, np.ones((len(xy), 1))]) @ scaled.T
    if np.abs(mapped2[:, :2] / mapped2[:, 2:3] - uv).max() > 1e-4:
        return False, "世界坐标放大 1000 倍后映射误差变大，检查数值稳定性"
    try:
        mod.homography_dlt(xy[:3], uv[:3])
    except ValueError:
        return True, f"无噪声误差 {err:.1e} px，大坐标下仍准确，点数不足报错"
    return False, "点数少于 4 时应抛出 ValueError"


def check_intrinsics(mod):
    vs = views()
    Hs = [mod.homography_dlt(xy, uv) for xy, _, uv, _, _ in vs]
    K = mod.intrinsics_from_homographies(Hs)
    if K.shape != (3, 3) or abs(K[2, 2] - 1) > 1e-9 or abs(K[1, 0]) > 1e-9:
        return False, "K 必须是上三角且 K[2,2] == 1"
    rel = np.abs(K - K_TRUE)[[0, 0, 1, 1], [0, 2, 1, 2]] / np.array([820, 318, 790, 247])
    if rel.max() > 1e-5:
        return False, f"无噪声时 fx, cx, fy, cy 最大相对误差 {rel.max():.2e}"
    noisy = views(noise=0.3, seed=1)
    Kn = mod.intrinsics_from_homographies([mod.homography_dlt(xy, uv) for xy, _, uv, _, _ in noisy])
    rel_n = abs(Kn[0, 0] - 820) / 820
    if rel_n > 0.03:
        return False, f"0.3 px 噪声下 fx 相对误差 {rel_n:.1%}，超过 3%"
    try:
        mod.intrinsics_from_homographies(Hs[:2])
    except ValueError:
        return True, f"无噪声精确，0.3 px 噪声下 fx 误差 {rel_n:.2%}"
    return False, "少于 3 个视图时应抛出 ValueError"


def check_extrinsics_and_rmse(mod):
    xy, pts3, uv, R_true, t_true = views()[2]
    H = mod.homography_dlt(xy, uv)
    R, t = mod.extrinsics_from_homography(K_TRUE, H)
    if R.shape != (3, 3) or np.asarray(t).shape != (3,):
        return False, "R 应为 (3,3)，t 应为 (3,)"
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-9) or abs(np.linalg.det(R) - 1) > 1e-9:
        return False, "R 必须是正交矩阵且 det = +1"
    if np.abs(R - R_true).max() > 1e-6 or np.abs(t - t_true).max() > 1e-4:
        return False, "无噪声时外参应与真值一致"
    rmse = mod.reprojection_rmse(K_TRUE, R, t, pts3, uv)
    if not isinstance(rmse, float) or rmse > 1e-6:
        return False, f"无噪声重投影误差应接近 0 且为 float，得到 {rmse!r}"
    shifted = uv + np.array([3.0, 4.0])
    if abs(mod.reprojection_rmse(K_TRUE, R, t, pts3, shifted) - 5.0) > 1e-6:
        return False, "所有点偏移 (3, 4) 时 RMSE 应为 5"
    return True, "外参恢复正确，RMSE 定义正确"


def main() -> int:
    print("第 17 章进度检查\n" + "=" * 52)
    try:
        mod = load_assignment()
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "加载作业", f"{type(exc).__name__}: {exc}")
        return 1
    results = [
        run("Hartley 归一化", lambda: check_normalize(mod)),
        run("归一化 DLT 单应", lambda: check_homography(mod)),
        run("Zhang 闭式内参", lambda: check_intrinsics(mod)),
        run("外参与重投影误差", lambda: check_extrinsics_and_rmse(mod)),
    ]
    print("=" * 52)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("线性部分通过；下一步在 Notebook 里加畸变和非线性优化，并和 OpenCV 对照。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
