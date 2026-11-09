"""第 15 章自动验收：编译 C++ 作业并运行测试。

需要一个支持 C++17 的编译器（macOS 自带 clang++ 即可）。
环境变量 COURSE_IMPL_DIR 可指向另一份实现目录，用于维护者验证测试。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "15-1_图像基础.cpp"


def find_compiler() -> str | None:
    for name in (os.environ.get("CXX"), "c++", "clang++", "g++"):
        if name and shutil.which(name):
            return name
    return None


def main() -> int:
    print("第 15 章进度检查\n" + "=" * 52)
    override = os.environ.get("COURSE_IMPL_DIR")
    source = (Path(override) / CHAPTER.name if override else CHAPTER) / ASSIGNMENT
    cxx = find_compiler()
    if cxx is None:
        print("[TODO] 编译器 — 没找到 c++/clang++/g++。macOS 运行 xcode-select --install 后重试。")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "test_ch15"
        cmd = [
            cxx, "-std=c++17", "-O2", "-Wall", "-Wextra",
            "-I", str(CHAPTER / "cpp"),
            str(CHAPTER / "tests" / "test_main.cpp"), str(source),
            "-o", str(exe),
        ]
        build = subprocess.run(cmd, capture_output=True, text=True)
        if build.returncode != 0:
            print("[FAIL] 编译 — 编译失败，前 30 行错误：")
            print("\n".join(build.stderr.splitlines()[:30]))
            return 1
        warnings = [line for line in build.stderr.splitlines() if "warning" in line]
        try:
            proc = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            print("[FAIL] 运行 — 超过 120 秒，检查是否有死循环或 O(r^2) 的实现")
            return 1
    if proc.returncode != 0:
        print(f"[FAIL] 运行 — 程序异常退出（返回码 {proc.returncode}），常见原因是越界访问或递归栈溢出")
        print(proc.stdout)
        return 1
    passed = total = 0
    for line in proc.stdout.splitlines():
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        status, name, detail = parts
        total += 1
        passed += status == "PASS"
        print(f"[{status:4}] {name} — {detail}")
    if warnings:
        print(f"注意：编译有 {len(warnings)} 条警告，建议清零（-Wall -Wextra）。")
    print("=" * 52)
    print(f"通过 {passed}/{total}")
    if passed != total or total == 0:
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("基本功能通过；下一步装 OpenCV C++ 版，用 cv::integral 和 cv::matchTemplate 对照性能。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
