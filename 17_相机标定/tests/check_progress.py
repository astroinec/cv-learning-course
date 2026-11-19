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


def close(got, want, tol) -> bool:
    """逐元素比较；形状对不上、出现 nan 或 inf 一律算不相等。"""
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        return False
    return bool(np.all(np.abs(got - want) <= tol))


def is_f64(arr, shape) -> bool:
    return isinstance(arr, np.ndarray) and arr.shape == shape and arr.dtype == np.float64


def rodrigues(rvec):
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * kx @ kx


K_TRUE = np.array([[820.0, 0.0, 318.0], [0.0, 790.0, 247.0], [0.0, 0.0, 1.0]])
# 带一点倾斜（skew）的相机，用来检查没有把 skew 当成 0
K_SKEW = np.array([[905.0, 3.5, 301.0], [0.0, 870.0, 262.0], [0.0, 0.0, 1.0]])
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


def plane_homography(K, rvec, t, scale=1.0):
    """讲义 3.1 节的模型：Z = 0 的平面到图像的单应是 K [r1 r2 t]，再乘一个任意倍数。"""
    R = rodrigues(rvec)
    H = K @ np.column_stack([R[:, 0], R[:, 1], t])
    return scale * H / H[2, 2]


def apply_h(H, pts):
    mapped = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(H, dtype=np.float64).T
    return mapped[:, :2] / mapped[:, 2:3]


def check_normalize(mod):
    rng = np.random.default_rng(0)
    # x 方向铺得开、y 方向很窄：两个方向的离散程度差很多
    pts = np.column_stack([rng.uniform(0, 640, size=30), rng.uniform(200, 260, size=30)])
    keep = pts.copy()
    pn, T = mod.normalize_points(pts)
    if not np.array_equal(pts, keep):
        return False, "函数修改了传入的点，需要改动时先复制一份"
    if not is_f64(pn, (30, 2)) or not is_f64(T, (3, 3)):
        return False, "输出应为 (N,2) 与 (3,3) 的 float64 数组"
    if not close(pn.mean(axis=0), [0, 0], 1e-9):
        return False, "归一化后质心应在原点"
    mean_dist = float(np.linalg.norm(pn, axis=1).mean())
    if not close(mean_dist, np.sqrt(2), 1e-9):
        return False, f"归一化后各点到原点距离的算术平均应为 sqrt(2)，得到 {mean_dist:.6f}"
    if not close(apply_h(T, pts), pn, 1e-9):
        return False, "points_n 必须等于 T 作用在原始点上的结果"
    if not close(T[2], [0, 0, 1], 1e-12) or not close([T[0, 1], T[1, 0]], [0, 0], 1e-12):
        return False, "T 应是 [[s, 0, tx], [0, s, ty], [0, 0, 1]] 的形式"
    if not (T[0, 0] > 0 and close(T[0, 0], T[1, 1], 1e-9 * abs(T[0, 0]))):
        return False, f"T 是相似变换，x 和 y 要用同一个正的缩放系数，得到 {T[0, 0]:.6g} 和 {T[1, 1]:.6g}"
    # 手算：正方形四个顶点，质心 (10, 20)，各点到质心的距离都是 sqrt(2)，所以 s = 1
    square = np.array([[9.0, 19.0], [11.0, 19.0], [11.0, 21.0], [9.0, 21.0]])
    sq_n, sq_T = mod.normalize_points(square)
    if not close(sq_n, square - [10.0, 20.0], 1e-9) or not close(sq_T, [[1, 0, -10], [0, 1, -20], [0, 0, 1]], 1e-9):
        return False, "中心在 (10, 20)、边长 2 的正方形四个顶点，归一化后应只是平移到原点（s = 1）"
    try:
        mod.normalize_points(np.ones((5, 2)))
    except ValueError:
        return True, "质心、尺度、T 的形式和 T 与点的一致性都正确，退化输入报错"
    return False, "所有点重合时应抛出 ValueError"


def check_homography(mod):
    xy, _, uv, _, _ = views()[0]
    xy_keep, uv_keep = xy.copy(), uv.copy()
    H = mod.homography_dlt(xy, uv)
    if not (np.array_equal(xy, xy_keep) and np.array_equal(uv, uv_keep)):
        return False, "函数修改了传入的点，需要改动时先复制一份"
    if not is_f64(H, (3, 3)) or not close(H[2, 2], 1.0, 1e-9):
        return False, "H 必须是 (3,3) float64 且 H[2,2] == 1"
    if not np.all(np.isfinite(H)):
        return False, "H 里有 nan 或 inf"
    err = np.abs(apply_h(H, xy) - uv).max()
    if not err <= 1e-6:
        return False, f"无噪声时把 src 映射到 dst 的误差是 {err:.2e} px，应小于 1e-6。检查方程的列顺序、去归一化的顺序和方向"
    scaled = mod.homography_dlt(xy * 1000.0, uv)
    if not np.abs(apply_h(scaled, xy * 1000.0) - uv).max() <= 1e-4:
        return False, "世界坐标放大 1000 倍后映射误差变大，检查数值稳定性"
    # 两组坐标都带很大的整体偏移（例如平台坐标）
    far_src, far_dst = xy + 1.0e6, uv + 1.0e6
    far = mod.homography_dlt(far_src, far_dst)
    far_err = np.abs(apply_h(far, far_src) - far_dst).max()
    if not far_err <= 1e-3:
        return False, f"两组坐标都加上 1e6 的整体偏移后，映射误差是 {far_err:.2e} px。检查两组点是不是都先做了归一化再列方程"
    # 带噪声：54 个点都参与时，估计出的单应应比单个角点的噪声准得多
    rng = np.random.default_rng(0)
    rms = []
    for _, pts3, clean, _, _ in views():
        noisy = clean + rng.normal(scale=0.5, size=clean.shape)
        est = mod.homography_dlt(xy, noisy)
        rms.append(np.sqrt(np.mean(np.sum((apply_h(est, xy) - clean) ** 2, axis=1))))
    mean_rms = float(np.mean(rms))
    if not mean_rms <= 0.3:
        return False, (f"角点加 0.5 px 噪声后，估计的单应离无噪声位置的均方根距离是 {mean_rms:.2f} px（5 个视图平均），应小于 0.3。"
                       "54 个点都参与最小二乘才能把噪声平均掉，检查是不是只用了一部分点")
    try:
        mod.homography_dlt(xy[:3], uv[:3])
    except ValueError:
        return True, f"无噪声误差 {err:.1e} px，大坐标和大偏移下仍准确，带噪声时 {mean_rms:.2f} px，点数不足报错"
    return False, "点数少于 4 时应抛出 ValueError"


def intrinsics_problem(K, K_want, where):
    """返回 None 表示 K 合格，否则返回一句说明。"""
    if not is_f64(K, (3, 3)):
        return f"{where}：K 必须是 (3,3) float64"
    if not np.all(np.isfinite(K)):
        return f"{where}：K 里有 nan 或 inf。检查开方之前的量是不是负的，以及两个约束各自的符号"
    if not close(K[2], [0, 0, 1], 1e-9) or not close(K[1, 0], 0, 1e-9):
        return f"{where}：K 必须是上三角且 K[2,2] == 1"
    if not (K[0, 0] > 0 and K[1, 1] > 0):
        return f"{where}：fx 和 fy 应为正"
    names = ["fx", "fy", "cx", "cy"]
    got = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
    want = np.array([K_want[0, 0], K_want[1, 1], K_want[0, 2], K_want[1, 2]])
    rel = np.abs(got - want) / want
    if not rel.max() <= 1e-5:
        i = int(np.argmax(rel))
        return f"{where}：{names[i]} 应为 {want[i]:g}，得到 {got[i]:.4f}"
    if not close(K[0, 1], K_want[0, 1], 1e-3):
        return f"{where}：skew（K[0,1]）应为 {K_want[0, 1]:g}，得到 {K[0, 1]:.4f}"
    return None


def check_intrinsics(mod):
    f = mod.intrinsics_from_homographies
    exact = [plane_homography(K_TRUE, r, t) for r, t in POSES]
    keep = [h.copy() for h in exact]
    problem = intrinsics_problem(f(exact), K_TRUE, "5 个无噪声的单应")
    if problem:
        return False, problem
    if not all(np.array_equal(a, b) for a, b in zip(exact, keep)):
        return False, "函数修改了传入的单应，需要改动时先复制一份"
    problem = intrinsics_problem(f(exact[:3]), K_TRUE, "恰好 3 个无噪声的单应")
    if problem:
        return False, problem
    for label, scales in [("带 skew 的相机", (1, 1, 1, 1, 1)), ("带 skew 的相机，各单应乘了不同的倍数", (-1, 2, -3, 0.5, 4))]:
        Hs = [plane_homography(K_SKEW, r, t, s) for (r, t), s in zip(POSES, scales)]
        problem = intrinsics_problem(f(Hs), K_SKEW, label)
        if problem:
            return False, problem + "。检查 v_ij 的每一项、从 b 解出各参数的公式，以及 b 整体符号为负时的处理"
    for label, order, scales in [("打乱顺序并乘上负的倍数", (4, 3, 2, 1, 0), (-2.5, 1, -1, 3, -0.5)), ("后 3 个单应", (2, 3, 4), (1, -1, 1))]:
        Hs = [plane_homography(K_TRUE, *POSES[i], s) for i, s in zip(order, scales)]
        problem = intrinsics_problem(f(Hs), K_TRUE, label)
        if problem:
            return False, problem + "。单应只确定到一个非零倍数，结果不应依赖它"
    # 下面用你自己的 homography_dlt 从角点估计单应
    try:
        Hs = [mod.homography_dlt(xy, uv) for xy, _, uv, _, _ in views()]
    except NotImplementedError as exc:
        raise NotImplementedError(f"内参函数在给定的无噪声单应上已经正确；带噪声的整条流程要等 homography_dlt 完成后才能检查（{exc}）") from None
    problem = intrinsics_problem(f(Hs), K_TRUE, "用 homography_dlt 估出的无噪声单应")
    if problem:
        return False, problem
    noisy = views(noise=0.3, seed=1)
    Kn = f([mod.homography_dlt(xy, uv) for xy, _, uv, _, _ in noisy])
    rel_n = abs(Kn[0, 0] - 820) / 820
    if not rel_n <= 0.03:
        return False, f"0.3 px 噪声下 fx 相对误差 {rel_n:.1%}，超过 3%"
    try:
        f(exact[:2])
    except ValueError:
        return True, f"无噪声精确（含 skew 和任意倍数的单应），0.3 px 噪声下 fx 误差 {rel_n:.2%}"
    return False, "少于 3 个视图时应抛出 ValueError"


def check_extrinsics(mod):
    f = mod.extrinsics_from_homography
    for K, (rvec, t_true), scale in [(K_TRUE, POSES[2], 1.0), (K_SKEW, POSES[0], 3.0), (K_SKEW, POSES[2], -2.5), (K_TRUE, POSES[4], -1.0)]:
        R_true = rodrigues(rvec)
        H = plane_homography(K, rvec, t_true, scale)
        K_in, H_in = K.copy(), H.copy()
        R, t = f(K_in, H_in)
        if not (np.array_equal(K_in, K) and np.array_equal(H_in, H)):
            return False, "函数修改了传入的 K 或 H，需要改动时先复制一份"
        if not is_f64(R, (3, 3)) or not is_f64(t, (3,)):
            return False, "R 应为 (3,3)，t 应为 (3,)，都是 float64"
        if not close(R @ R.T, np.eye(3), 1e-9) or not close(np.linalg.det(R), 1.0, 1e-9):
            return False, f"R 必须是正交矩阵且 det = +1，得到 det = {np.linalg.det(R):.4f}"
        if not close(t, t_true, 1e-6) or not close(R, R_true, 1e-8):
            hint = "单应的整体符号是任意的，检查 t[2] < 0 时哪些量要一起变号" if scale < 0 else "检查 K 的逆、尺度 s 和 r3 的叉乘顺序"
            return False, f"无噪声的单应乘以 {scale:g} 后，外参应与真值一致（t 真值 {t_true.tolist()}，得到 {np.round(t, 3).tolist()}）。{hint}"
    # 带噪声的单应：[r1 r2 r3] 不再正交，输出仍然必须是旋转矩阵
    rng = np.random.default_rng(6)
    rvec, t_true = POSES[1]
    H = plane_homography(K_TRUE, rvec, t_true) + rng.normal(scale=[[0.01], [0.01], [1e-5]], size=(3, 3))
    R, t = f(K_TRUE, H)
    if not close(R @ R.T, np.eye(3), 1e-9) or not close(np.linalg.det(R), 1.0, 1e-9):
        return False, "单应带噪声时 [r1 r2 r3] 不严格正交，输出的 R 仍应是正交矩阵且 det = +1。检查最后有没有投影到旋转矩阵"
    if not close(R, rodrigues(rvec), 0.05):
        return False, "单应带一点噪声时，R 应该仍然接近真值"
    # 手算：K = I，h1 = (1, 0, 0)，h2 = (sin a, cos a, 0)，h3 = (0, 0, 5)，a = 0.2。
    # r1 和 r2 夹角比 90 度小 0.2 rad；离它最近的旋转矩阵把这 0.2 rad 平分给两列，是绕 z 轴转 -0.1 rad。
    a = 0.2
    H = np.array([[1.0, np.sin(a), 0.0], [0.0, np.cos(a), 0.0], [0.0, 0.0, 5.0]])
    want = np.array([[np.cos(a / 2), np.sin(a / 2), 0.0], [-np.sin(a / 2), np.cos(a / 2), 0.0], [0.0, 0.0, 1.0]])
    R, t = f(np.eye(3), H)
    if not close(R, want, 1e-9) or not close(t, [0, 0, 5], 1e-9):
        return False, ("r1 和 r2 不垂直（夹角差 0.2 rad）时，最近的旋转矩阵应把偏差平分给两列（绕 z 轴转 -0.1 rad）。"
                       f"得到的 R 第一列是 {np.round(R[:, 0], 4).tolist()}。对照文档字符串最后一步，逐列正交化得到的不是最近的旋转矩阵")
    return True, "无噪声外参恢复正确（含负倍数的单应），带噪声时仍输出最近的旋转矩阵"


def check_rmse(mod):
    f = mod.reprojection_rmse
    # 手算：K = I、R = I、t = 0 时投影就是 (X/Z, Y/Z)
    world = np.array([[2.0, 4.0, 2.0], [0.0, 0.0, 5.0], [3.0, -6.0, 3.0], [-8.0, 4.0, 4.0]])
    exact_uv = np.array([[1.0, 2.0], [0.0, 0.0], [1.0, -2.0], [-2.0, 1.0]])
    eye, zero = np.eye(3), np.zeros(3)
    got = f(eye, eye, zero, world, exact_uv)
    if not isinstance(got, float):
        return False, "返回值必须是 Python float"
    if not close(got, 0.0, 1e-12):
        return False, f"K = I、R = I、t = 0 时点 (X, Y, Z) 应投影到 (X/Z, Y/Z)，观测正好在这些位置时误差应为 0，得到 {got}"
    one_off = exact_uv.copy()
    one_off[0] += [3.0, 4.0]
    got = f(eye, eye, zero, world, one_off)
    if not close(got, 2.5, 1e-12):
        return False, f"4 个点里只有 1 个偏了 (3, 4)（距离 5），均方根误差应为 sqrt(25 / 4) = 2.5，得到 {got}。检查是先平方、对点取平均、再开方"
    xy, pts3, uv, R_true, t_true = views()[2]
    got = f(K_TRUE, R_true, t_true, pts3, uv)
    if not isinstance(got, float) or not close(got, 0.0, 1e-9):
        return False, f"用真实的 K、R、t 投影无噪声的角点，误差应为 0，得到 {got!r}。检查 R、t 的用法和除以深度"
    got = f(K_TRUE, R_true, t_true, pts3, uv + np.array([3.0, 4.0]))
    if not close(got, 5.0, 1e-9):
        return False, f"所有点偏移 (3, 4) 时 RMSE 应为 5，得到 {got}"
    mixed = uv.copy()
    mixed[::2] += [0.6, 0.8]
    got = f(K_TRUE, R_true, t_true, pts3, mixed)
    want = float(np.sqrt(0.5))
    if not close(got, want, 1e-9):
        return False, f"一半的点偏移 (0.6, 0.8)（距离 1）、另一半不偏时，RMSE 应为 sqrt(1/2) = {want:.4f}，得到 {got}"
    return True, "投影和均方根的定义正确"


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
        run("外参恢复", lambda: check_extrinsics(mod)),
        run("重投影误差", lambda: check_rmse(mod)),
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
