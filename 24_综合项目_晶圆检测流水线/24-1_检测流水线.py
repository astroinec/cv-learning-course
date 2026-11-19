"""作业 24-1：die-to-die 缺陷检测流水线。

你要实现 run_pipeline：输入已知良品 die 和待检 die，输出每个待检 die 上的缺陷中心。
建议流程（每一步都可以复用你前面章节的实现）：
1. 选一张参考图（例如第一张良品），用第 16 章的相位相关估计每张图相对参考的亚像素错位
2. 把所有图对齐到参考坐标系（傅里叶平移或双线性插值）
3. 亮度归一化：照明会让整体亮度和偏置漂移，用良品中位数做线性拟合校正
4. 用第 18 章的鲁棒金模板（中位数 + MAD）得到 median 和 sigma
5. z-score 阈值 + 连通域 + 面积过滤，得到对齐坐标系下的缺陷中心
6. 把缺陷中心变换回该 die 自身的图像坐标（加回错位），这是验收用的坐标系

约束：
- 只能用 golden 和 test 两个输入，不能读取 data["shifts"] 或 data["truth"]
- 只允许 NumPy（可以 import 你自己前面章节的作业文件，见 load_chapter）
- 单次运行 30 张 96x96 的图应在 30 秒内完成

验收：python tests/check_progress.py（macOS 和 Linux 上是 python3）。它用 8 个固定种子各生成一份数据，按真值独立计算指标。
一个种子达标指检出率不低于 85%、每 die 误报不超过 0.3、单次运行不超过 30 秒；
8 个种子里至少 6 个达标，并且 8 个种子合计的检出率和每 die 误报也满足这两条线，才算通过。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

COURSE_ROOT = Path(__file__).resolve().parents[1]


def load_chapter(folder: str, filename: str):
    """导入前面章节的作业文件，例如 load_chapter("16_图像对齐与亚像素", "16-1_相位相关与亚像素.py")。"""
    path = COURSE_ROOT / folder / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_pipeline(golden: np.ndarray, test: np.ndarray) -> list[list[tuple[float, float]]]:
    """返回长度为 N 的列表（N 是待检 die 的张数，即 test.shape[0]），第 i 项是第 i 张待检 die 上检出的缺陷中心 [(y, x), ...]，
    坐标在该 die 自身的图像坐标系下，行在前、列在后。每个中心是两个有限的数，不能是 nan 或 inf。
    没有缺陷的 die 返回空列表，不能是 None。
    """
    raise NotImplementedError("TODO: 实现 die-to-die 检测流水线")
