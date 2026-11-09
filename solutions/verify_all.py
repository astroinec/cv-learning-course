"""对第 1–24 章逐章运行验收：参考实现应全部通过，主目录作业（初始 TODO）应不报错。

用法（在仓库根目录）：.venv/bin/python solutions/verify_all.py（第 1、5 章要 OpenCV，第 9–13 章要 PyTorch）
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOL = Path(__file__).resolve().parent


def run(test: Path, impl_dir: Path | None) -> tuple[int, str]:
    env = dict(os.environ)
    env.pop("COURSE_IMPL_DIR", None)
    if impl_dir is not None:
        env["COURSE_IMPL_DIR"] = str(impl_dir)
    p = subprocess.run([sys.executable, str(test)], capture_output=True, text=True, env=env)
    lines = [l for l in p.stdout.splitlines() if l.strip()]
    return p.returncode, lines[-1] if lines else p.stderr.strip()[-200:]


def main() -> int:
    chapters = sorted(d for d in ROOT.iterdir() if d.is_dir() and d.name[:2].isdigit() and 1 <= int(d.name[:2]) <= 24)
    ok = True
    for ch in chapters:
        test = ch / "tests" / "check_progress.py"
        rc_sol, last_sol = run(test, SOL)
        rc_todo, out_todo = run(test, None)
        sol_ok = rc_sol == 0
        todo_ok = "Traceback" not in out_todo
        ok &= sol_ok and todo_ok
        print(f"{'OK  ' if sol_ok and todo_ok else 'FAIL'} {ch.name}｜参考实现：{last_sol}｜作业初始状态：{out_todo}")
    print("全部符合预期" if ok else "有章节不符合预期，见上")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
