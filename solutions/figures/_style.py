"""配图公共设置。配图脚本放在 solutions 分支，因为画图需要完整的参考实现。

输出目录由环境变量 COURSE_MAIN_DIR 指定（主分支的工作目录），
默认写到本文件所在仓库的章节目录下。
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

SOLUTIONS = Path(__file__).resolve().parents[1]
REPO = SOLUTIONS.parent
MAIN = Path(os.environ.get("COURSE_MAIN_DIR", REPO)).resolve()
PREVIEW = os.environ.get("FIG_PREVIEW_DIR")

INK = "#1f2329"
MUTED = "#646a73"
BLUE = "#2f6fed"
ORANGE = "#e8710a"
GREEN = "#1a9c4b"
RED = "#d6362f"
PURPLE = "#7a4cd6"
GRID = "#e3e5e8"


def setup() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    wanted = ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti", "Arial Unicode MS",
              "Microsoft YaHei", "Noto Sans CJK SC", "SimHei"]
    found = [n for n in wanted if n in names]
    if not found:
        raise RuntimeError("没有找到中文字体，配图里的中文会变成方块")
    plt.rcParams.update({
        "font.sans-serif": found + ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "path",
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#b9bec7",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": False,
    })


def load_solution(chapter: str, filename: str):
    path = SOLUTIONS / chapter / filename
    spec = importlib.util.spec_from_file_location(Path(filename).stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def save(fig, chapter: str, name: str) -> Path:
    out_dir = MAIN / chapter / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.svg"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.12, metadata={"Date": None})
    if PREVIEW:
        Path(PREVIEW).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(PREVIEW) / f"{name}.png", dpi=110, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return out


def clean(ax, grid: bool = True) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def image_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color("#b9bec7")
