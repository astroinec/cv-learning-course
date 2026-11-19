"""作业 21-1：3D 变换与软件光栅化（z-buffer）。

只允许使用 NumPy。约定与 OpenGL 一致：右手坐标系，相机看向 -Z，
NDC 范围 [-1, 1]，深度映射到 [0, 1]（近平面 0，远平面 1）。

有一处故意和 GPU 不一样：render_depth 里“像素中心正好落在三角形的边上”算在三角形内，
不是 GPU 的左上规则。验收按这个约定数像素，详见 render_depth 的文档字符串。

对所有函数都成立的约定（验收会检查）：不改动传入的数组。
"""

from __future__ import annotations

import numpy as np


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """构造视图矩阵 V（4x4 float64），把世界坐标变换到相机坐标。

    f = normalize(target - eye)，s = normalize(f x up)，u = s x f。
    V 的旋转部分三行依次为 s、u、-f，平移部分为 -R @ eye。
    up 只给方向，不保证是单位向量。
    up 与视线平行（同向或反向，f x up 的长度为 0）时抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现 look_at")


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """OpenGL 风格透视投影矩阵（4x4 float64）。

    f = 1 / tan(fovy / 2)
    P = [[f/aspect, 0, 0, 0],
         [0, f, 0, 0],
         [0, 0, (far+near)/(near-far), 2*far*near/(near-far)],
         [0, 0, -1, 0]]
    fovy_deg 是角度，是垂直方向的整个视场角。
    需要 0 < near < far：near <= 0 或 far <= near 时抛出 ValueError。
    """
    raise NotImplementedError("TODO: 实现透视投影矩阵")


def to_screen(points: np.ndarray, mvp: np.ndarray, width: int, height: int) -> np.ndarray:
    """把世界点 (N, 3) 经 MVP 变换到屏幕坐标 (N, 3) float64。

    步骤：齐次坐标 → clip = mvp @ p → NDC = clip[:3] / clip[3]
    x_pix = (ndc_x + 1) / 2 * width，y_pix = (1 - ndc_y) / 2 * height（y 向下），
    depth = (ndc_z + 1) / 2。
    clip w <= 0 的点（在相机后方，或者 w 正好为 0、和相机在同一个平面上）
    三个分量都输出 NaN，不做除法。
    """
    raise NotImplementedError("TODO: 实现 MVP 到屏幕坐标")


def render_depth(triangles: np.ndarray, width: int, height: int) -> np.ndarray:
    """z-buffer 光栅化，只输出深度图。

    triangles: (T, 3, 3)，每个三角形三个顶点的 (x_pix, y_pix, depth)。T 可以是 0。
    在像素中心 (x + 0.5, y + 0.5) 采样；用重心坐标判断是否在三角形内，
    在内则按重心坐标插值三个顶点的深度，同一个像素保留更小的深度。

    ====================================================================
    本作业的约定：边上的像素算在内
    --------------------------------------------------------------------
    三个重心坐标都 >= 0 就算在三角形内，像素中心正好落在边上或顶点上也算。
    例：顶点 (0,0)、(8,0)、(0,8) 的三角形覆盖 36 个像素中心，其中 8 个在斜边上。
    这不是 GPU 的做法。GPU 用左上规则（边上的像素只归其中一个三角形），
    “边上不算”则是另一种常见写法，这两种在上面的例子里都得到 28 个，验收会判为不通过。
    先按本约定写，通过验收后再把左上规则当扩展做（讲义第 4 节和第 12 节）。
    ====================================================================

    其余要求：
    - 两种顶点绕序（顺时针、逆时针）都要支持，结果相同。
    - 面积为 0 的三角形（三点共线、顶点重合）跳过，不能让 NaN 进入深度图。
    - 含 NaN 顶点的三角形跳过（to_screen 对相机后方的点输出 NaN），不抛异常，
      也不影响别的三角形。
    - 三角形可以部分或全部落在画面之外，只画画面内的部分。
    - 可以只遍历三角形包围盒里的像素来加速（不要求），结果必须和遍历整幅图相同。
    返回 (height, width) float64，没有被覆盖的像素为 np.inf。
    """
    raise NotImplementedError("TODO: 实现 z-buffer 光栅化")
