"""维护者工具：执行 Notebook 的全部代码单元，检查能否跑通。默认第 1–31 章，NB_FROM、NB_TO 环境变量可改范围。

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
CHAPTERS = [f"{n:02d}_" for n in range(int(os.environ.get("NB_FROM", 1)), int(os.environ.get("NB_TO", 31)) + 1)]
if os.environ.get("COURSE_IMPL_DIR"):  # 执行时会切到章节目录，相对路径要先转成绝对路径
    os.environ["COURSE_IMPL_DIR"] = str(Path(os.environ["COURSE_IMPL_DIR"]).resolve())


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
    if len(sys.argv) == 3 and sys.argv[1] == "--one":  # 子进程：只执行一个 Notebook
        return 0 if run(Path(sys.argv[2])) else 1
    books = sorted(p for p in ROOT.glob("*/实验_*.ipynb") if p.parent.name.startswith(tuple(CHAPTERS)))
    # 每个 Notebook 在独立的进程里执行，和 Jupyter 里每个 Notebook 一个内核的情况一致，互相不影响
    import subprocess
    results = []
    for p in books:
        res = subprocess.run([sys.executable, __file__, "--one", str(p)], capture_output=True, text=True, encoding="utf-8", errors="replace")
        print((res.stdout + res.stderr).strip()[-3000:] if res.returncode else res.stdout.strip().splitlines()[-1])
        results.append(res.returncode == 0)
    print(f"{sum(results)}/{len(results)} 个 Notebook 跑通")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
