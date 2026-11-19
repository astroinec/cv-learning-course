"""环境检查：装好环境后运行一次，逐项告诉你哪里不对。

用法（在仓库根目录、激活虚拟环境之后）：
    python scripts/check_env.py

只检查，不安装、不修改任何东西。
"""

from __future__ import annotations

import importlib
import importlib.metadata as meta
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS: list[tuple[str, str, str]] = []

try:  # Windows 控制台不支持某些字符时不要崩
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass


def report(level: str, name: str, detail: str) -> None:
    RESULTS.append((level, name, detail))
    print(f"[{level}] {name}：{detail}")


def check_python() -> None:
    v = sys.version_info
    detail = f"{platform.python_version()}，{platform.architecture()[0]}，{sys.executable}"
    if v < (3, 10):
        report("错误", "Python", detail + "。需要 3.10 或更新，推荐 3.12")
    elif platform.architecture()[0] != "64bit":
        report("错误", "Python", detail + "。需要 64 位 Python")
    else:
        report("正常", "Python", detail)
    if sys.prefix == sys.base_prefix:
        report("提醒", "虚拟环境", "当前不在虚拟环境里。先运行 .venv\\Scripts\\activate（Windows）或 source .venv/bin/activate")
    else:
        report("正常", "虚拟环境", sys.prefix)


def check_packages() -> None:
    required = {"numpy": "numpy", "cv2": "opencv-python", "matplotlib": "matplotlib", "PIL": "pillow",
                "torch": "torch", "torchvision": "torchvision", "nbformat": "nbformat", "ipywidgets": "ipywidgets"}
    for module, dist in required.items():
        try:
            importlib.import_module(module)
            try:
                ver = meta.version(dist)
            except meta.PackageNotFoundError:
                ver = getattr(sys.modules[module], "__version__", "?")
            report("正常", dist, ver)
        except Exception as exc:  # noqa: BLE001
            report("错误", dist, f"导入失败（{type(exc).__name__}）。运行 pip install -r requirements.txt")
    for module, dist in {"jupyterlab": "jupyterlab", "ipykernel": "ipykernel"}.items():
        try:
            importlib.import_module(module)
            report("正常", dist, meta.version(dist))
        except Exception:  # noqa: BLE001
            report("提醒", dist, "没有装，打不开 Notebook。运行 pip install -r requirements.txt")


def check_torch() -> None:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return
    cuda_build = torch.version.cuda
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        try:
            x = torch.randn(256, 256, device="cuda")
            ok = float((x @ x).sum().item()) == float((x @ x).sum().item())
        except Exception as exc:  # noqa: BLE001
            report("错误", "显卡", f"{name} 能看到，但计算失败：{exc}")
            return
        report("正常", "显卡", f"{name}，显存 {total:.1f} GB，PyTorch 的 CUDA 版本 {cuda_build}" + ("" if ok else "（结果不稳定）"))
    elif cuda_build is None:
        report("提醒", "显卡", "装的是只能用 CPU 的 PyTorch。想用显卡见 docs/环境安装.md 第 3.3 节；不用显卡也能学完全部课程")
    else:
        report("提醒", "显卡", f"PyTorch 带 CUDA {cuda_build}，但找不到可用的显卡。运行 nvidia-smi 检查驱动")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        report("正常", "Apple 显卡", "MPS 可用")


def check_fonts() -> None:
    try:
        from matplotlib import font_manager
    except Exception:  # noqa: BLE001
        return
    names = {f.name for f in font_manager.fontManager.ttflist}
    wanted = ["Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC", "Noto Sans CJK SC"]
    found = [n for n in wanted if n in names]
    if found:
        report("正常", "中文字体", "、".join(found))
    else:
        report("提醒", "中文字体", "matplotlib 找不到常见中文字体，图里的中文会显示成方框")


def check_compiler() -> None:
    for name in (os.environ.get("CXX"), "g++", "clang++", "c++", "cl"):
        if name and shutil.which(name):
            report("正常", "C++ 编译器", f"{name}：{shutil.which(name)}")
            return
    report("提醒", "C++ 编译器", "没找到。只有第 15 章需要，安装见 docs/环境安装.md 第 6 节“C++ 编译器”")


def check_files() -> None:
    try:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "编码测试.txt"
            p.write_text("中文和 UTF-8", encoding="utf-8")
            assert p.read_text(encoding="utf-8") == "中文和 UTF-8"
        report("正常", "文件读写", f"系统默认编码 {__import__('locale').getpreferredencoding(False)}，课程代码都显式使用 UTF-8")
    except Exception as exc:  # noqa: BLE001
        report("错误", "文件读写", f"临时目录里读写中文文件名失败：{exc}")
    missing = [d for d in ("01_像素与色彩", "site", "demos") if not (ROOT / d).exists()]
    if missing:
        report("错误", "仓库", f"找不到 {missing}，请在完整克隆的仓库里运行")
    else:
        report("正常", "仓库", str(ROOT))


def main() -> int:
    print(f"系统：{platform.system()} {platform.release()}\n")
    for fn in (check_python, check_packages, check_torch, check_fonts, check_compiler, check_files):
        fn()
    errors = sum(1 for level, _, _ in RESULTS if level == "错误")
    notes = sum(1 for level, _, _ in RESULTS if level == "提醒")
    print(f"\n{errors} 项错误，{notes} 项提醒。" + ("环境可以用了。" if not errors else "先处理“错误”的几项，说明见 docs/环境安装.md。"))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
