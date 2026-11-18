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
        "svg.hashsalt": "cv-learning-course",
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
    # constrained 布局解出的坐标轴位置在不同次运行之间会差最后几位浮点数，SVG 里裁剪路径的编号按坐标算哈希，会跟着变。
    # 先让布局算完，把位置取整后定住，重复运行才能得到逐字节相同的文件。
    try:
        fig.canvas.draw()
        fig.set_layout_engine("none")
        for ax in fig.axes:
            ax.set_position([round(v, 4) for v in ax.get_position().bounds])
    except Exception:  # noqa: BLE001  定不住就按原样保存，只影响可复现性
        pass
    fig.savefig(out, bbox_inches="tight", pad_inches=0.12, metadata={"Date": None})
    if PREVIEW:
        Path(PREVIEW).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(PREVIEW) / f"{name}.png", dpi=110, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return out


def pinned(chapter: str, key: str, measure, env: str):
    """每次运行都会波动的测量（耗时之类）：默认沿用 实测数字.json 里上次记下的值，设置环境变量 env=1 才重测。

    measure 是一个不带参数的函数，返回要写进 NUM[key] 的内容。这样重复运行脚本，JSON 和讲义里的数字不会变；
    换机器、换库版本之后设置环境变量重测，再按新数字改讲义。
    """
    import json

    old = MAIN / chapter / "figures" / "实测数字.json"
    if os.environ.get(env) != "1" and old.exists():
        prev = json.loads(old.read_text(encoding="utf-8")).get(key)
        if prev is not None:
            print(f"{key}：沿用上次的测量（设置 {env}=1 重测）", flush=True)
            return prev
    return measure()


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
