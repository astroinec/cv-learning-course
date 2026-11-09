"""第 21 章自动验收：TODO / FAIL / PASS。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "21-1_软光栅.py"


def load_assignment():
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    spec = importlib.util.spec_from_file_location("chapter21_assignment", base / ASSIGNMENT)
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


def check_look_at(mod):
    V = mod.look_at(np.array([0.0, 0.0, 5.0]), np.zeros(3), np.array([0.0, 1.0, 0.0]))
    if V.shape != (4, 4) or V.dtype != np.float64:
        return False, "应返回 4x4 float64"
    if not np.allclose(V, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -5], [0, 0, 0, 1]]):
        return False, "从 (0,0,5) 看原点时 V 应为单位旋转加平移 -5"
    V2 = mod.look_at(np.array([3.0, 2.0, 1.0]), np.array([0.0, 1.0, -2.0]), np.array([0.0, 1.0, 0.0]))
    R = V2[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3)) or abs(np.linalg.det(R) - 1) > 1e-9:
        return False, "旋转部分必须正交且 det = +1"
    eye_c = V2 @ np.array([3.0, 2.0, 1.0, 1.0])
    tgt_c = V2 @ np.array([0.0, 1.0, -2.0, 1.0])
    if not np.allclose(eye_c[:3], 0) or not (tgt_c[2] < 0 and np.allclose(tgt_c[:2], 0)):
        return False, "相机应在原点，目标点应在 -Z 轴上"
    try:
        mod.look_at(np.zeros(3), np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    except ValueError:
        return True, "视图矩阵正交、相机位置和朝向正确"
    return False, "up 与视线平行时应抛出 ValueError"


def check_perspective_and_screen(mod):
    P = mod.perspective(90.0, 2.0, 1.0, 10.0)
    if P.shape != (4, 4) or abs(P[0, 0] - 0.5) > 1e-12 or abs(P[1, 1] - 1.0) > 1e-12 or P[3, 2] != -1:
        return False, "透视矩阵元素不正确（fovy=90 时 f=1，aspect=2）"
    for bad in [(90.0, 1.0, 0.0, 10.0), (90.0, 1.0, 5.0, 5.0)]:
        try:
            mod.perspective(*bad)
            return False, f"非法参数 {bad} 应抛出 ValueError"
        except ValueError:
            pass
    pts = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -10.0], [10.0, 5.0, -5.0], [0.0, 0.0, 2.0]])
    s = mod.to_screen(pts, P, 200, 100)
    if s.shape != (4, 3) or s.dtype != np.float64:
        return False, "to_screen 应返回 (N, 3) float64"
    if not np.allclose(s[0], [100, 50, 0]) or not np.allclose(s[1], [100, 50, 1]):
        return False, "近平面中心应映射到 (W/2, H/2, 0)，远平面中心深度为 1"
    if not np.allclose(s[2, :2], [200, 0]):
        return False, "右上角视锥边缘应映射到屏幕 (W, 0)，检查 y 方向是否翻转"
    if not np.isnan(s[3]).all():
        return False, "相机后方的点应输出 NaN"
    return True, "透视矩阵、NDC、视口和 y 翻转都正确"


def check_raster(mod):
    tri_ccw = np.array([[[0.0, 0.0, 0.5], [8.0, 0.0, 0.5], [0.0, 8.0, 0.5]]])
    d = mod.render_depth(tri_ccw, 10, 10)
    if d.shape != (10, 10) or d.dtype != np.float64:
        return False, "深度图应为 (H, W) float64"
    covered = np.isfinite(d)
    if covered.sum() != 36:
        return False, f"直角边为 8 的三角形应覆盖 36 个像素中心，得到 {covered.sum()}"
    if not covered[0, 0] or covered[9, 9] or not np.allclose(d[covered], 0.5):
        return False, "覆盖区域或深度不对"
    tri_cw = tri_ccw[:, [0, 2, 1], :]
    if not np.array_equal(np.isfinite(mod.render_depth(tri_cw, 10, 10)), covered):
        return False, "顺时针和逆时针绕序应得到同样的覆盖"
    near = np.array([[0.0, 0.0, 0.2], [10.0, 0.0, 0.2], [0.0, 10.0, 0.2]])
    far = np.array([[0.0, 0.0, 0.8], [10.0, 0.0, 0.8], [10.0, 10.0, 0.8]])
    for order in ([far, near], [near, far]):
        d2 = mod.render_depth(np.stack(order), 10, 10)
        if abs(d2[1, 1] - 0.2) > 1e-12 or abs(d2[8, 8] - 0.8) > 1e-12:
            return False, "z-buffer 应保留更近的深度，且与绘制顺序无关"
    ramp = np.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 1.0], [0.0, 10.0, 0.0]]])
    d3 = mod.render_depth(ramp, 10, 10)
    if abs(d3[0, 4] - 0.45) > 1e-9:
        return False, f"深度应按重心坐标插值，(x=4.5) 处应为 0.45，得到 {d3[0, 4]}"
    flat = np.array([[[0.0, 0.0, 0.1], [5.0, 5.0, 0.1], [9.0, 9.0, 0.1]]])
    if np.isfinite(mod.render_depth(flat, 10, 10)).any():
        return False, "退化三角形应跳过"
    return True, "覆盖测试、绕序、深度比较、插值和退化情况都正确"


def main() -> int:
    print("第 21 章进度检查\n" + "=" * 52)
    try:
        mod = load_assignment()
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "加载作业", f"{type(exc).__name__}: {exc}")
        return 1
    results = [
        run("视图矩阵 look_at", lambda: check_look_at(mod)),
        run("透视投影与屏幕映射", lambda: check_perspective_and_screen(mod)),
        run("z-buffer 光栅化", lambda: check_raster(mod)),
    ]
    print("=" * 52)
    print(f"通过 {sum(results)}/{len(results)}")
    if not all(results):
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("软光栅通过；下一步在 Notebook 里加 Lambert 光照，并读 LearnOpenGL 的着色器章节。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
