"""维护者工具：执行第二部分 Notebook 的全部代码单元，检查能否跑通。

用法：
    COURSE_IMPL_DIR=<参考实现目录> python scripts/run_track2_notebooks.py
不设置 COURSE_IMPL_DIR 时，作业还是 TODO，Notebook 应该打印“先完成作业里的 …”而不是报错。
需要 nbformat 和 matplotlib；图会用 Agg 后端渲染但不显示。
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import nbformat  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = [f"{n}_" for n in range(16, 25)]


def run(path: Path) -> bool:
    nb = nbformat.read(path, as_version=4)
    env: dict = {"__name__": "__main__"}
    cwd = os.getcwd()
    os.chdir(path.parent)
    try:
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            src = "\n".join(line for line in cell.source.splitlines() if not line.startswith("%"))
            try:
                exec(compile(src, f"{path.name}#cell{i}", "exec"), env)
            except Exception:  # noqa: BLE001
                print(f"[FAIL] {path.relative_to(ROOT)} 第 {i} 个单元")
                traceback.print_exc()
                return False
            finally:
                import matplotlib.pyplot as plt
                plt.close("all")
    finally:
        os.chdir(cwd)
    print(f"[ OK ] {path.relative_to(ROOT)}")
    return True


def main() -> int:
    books = sorted(p for p in ROOT.glob("*/实验_*.ipynb") if p.parent.name.startswith(tuple(CHAPTERS)))
    results = [run(p) for p in books]
    print(f"{sum(results)}/{len(results)} 个 Notebook 跑通")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
