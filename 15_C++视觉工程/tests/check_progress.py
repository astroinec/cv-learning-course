"""第 15 章自动验收：编译 C++ 作业并运行测试。

需要一个支持 C++17 的编译器。查找顺序：环境变量 CXX，然后 g++、clang++、c++，最后是 MSVC 的 cl。
cl 只有在 Visual Studio 的“x64 Native Tools Command Prompt”里才找得到。
Windows 上怎样安装编译器见 docs/环境安装.md 的“C++ 编译器”一节。

做法：把作业文件、tests/test_main.cpp 和 cpp/ 下的头文件复制到一个临时目录，换成纯英文的文件名后在那里编译，
编译器看到的路径里没有中文。测试程序只输出英文的状态和代码，中文说明由本脚本给出。

用法：
    python tests/check_progress.py            编译并运行全部 4 项测试
    python tests/check_progress.py --timing   只做盒式均值的计时，打印 r=1 和 r=10、30、200 的耗时，填实验记录时用

环境变量：
    CXX              指定编译器（命令名或完整路径）
    COURSE_IMPL_DIR  指向另一份实现所在的目录，用于维护者验证测试
    CH15_KEEP_BUILD  设为 1 时保留临时构建目录并打印它的位置，排查编译问题时用
"""

from __future__ import annotations

import locale
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "15-1_图像基础.cpp"
BUILD_ASSIGNMENT = "hw15_1.cpp"  # 作业文件在临时目录里的名字
BUILD_TEST = "test_main.cpp"
EXE = "test_ch15.exe" if os.name == "nt" else "test_ch15"
RUN_TIMEOUT = 120
RULE = "=" * 52

TESTS = [("integral", "积分图"), ("box_mean", "盒式均值"), ("ncc", "NCC 模板匹配"), ("blobs", "连通域计数")]
NAMES = dict(TESTS)

TODO_TEXT = {
    "integral": "TODO: 实现积分图",
    "box_mean": "TODO: 用积分图实现盒式均值",
    "ncc": "TODO: 实现 NCC 模板匹配",
    "blobs": "TODO: 用显式队列实现连通域计数",
}

PASS_TEXT = {
    "integral": "数值与暴力求和一致，空图和细长图正确，4200x4200 的大图不溢出",
    "box_mean": "与暴力均值一致，边界和大半径正确；{n}x{n} 的图上 r=1 最快 {small_ms} ms，r={r} 最快 {large_ms} ms，没有随半径明显增长",
    "ncc": "位置、分数、亮度不变性、同分取第一个和异常处理正确",
    "blobs": "4/8 连通正确，图像边界不相连，大区域不栈溢出",
}

FAIL_TEXT = {
    "integral_size": "输入 {in_rows}x{in_cols} 的图，积分图尺寸应为 (rows+1, cols+1)，实际是 {rows}x{cols}",
    "integral_value": "输入 {in_rows}x{in_cols} 的随机图，积分图位置 ({y}, {x}) 应为 {expect}，实际是 {got}。"
                      "检查第 0 行和第 0 列是不是全为 0，递推时行和列有没有用反",
    "integral_empty": "空图（{in_rows} 行 {in_cols} 列）应返回 (rows+1) x (cols+1) 的全 0 图，实际得到 {rows}x{cols}，或者里面有非 0 的值",
    "integral_big": "{n}x{n} 的全白图，积分图位置 ({y}, {x}) 应为 {expect}，实际是 {got}。{diagnosis}"
                    "小图上是对的、数大了才错，检查累加和中间结果用的类型能装下多大的数",
    "box_size": "输入 {in_rows}x{in_cols}、r={r} 时输出尺寸应与输入相同，实际是 {rows}x{cols}",
    "box_value": "输入 {in_rows}x{in_cols}、r={r} 时，位置 ({y}, {x}) 的均值应为 {expect}，实际是 {got}。{where}",
    "box_large_r": "r={r} 时窗口比图像的边还长（图是 {in_rows}x{in_cols}），位置 ({y}, {x}) 的均值应为 {expect}，实际是 {got}。"
                   "这时窗口会从两侧同时伸出图像，检查上下左右是不是各自裁剪到了图像内，以及 r 有没有被改动过",
    "box_value_big": "{in_rows}x{in_cols} 的大图、r={r} 时，位置 ({y}, {x}) 的均值应为 {expect}，实际是 {got}。"
                     "小图上是对的，检查大图大半径下求和用的类型和下标",
    "box_negative_r": "r < 0 应抛 std::invalid_argument，实际没有抛异常",
    "box_negative_r_type": "r < 0 应抛 std::invalid_argument，实际抛的是 {kind}。检查参数检查是不是排在了别的操作后面",
    "box_slow": "耗时随半径增长，没有做到每像素 O(1)：{n}x{n} 的图上预热后交替计时 {reps} 轮，r=1 最快 {small_ms} ms，"
                "r={r} 最快 {large_ms} ms，是 {ratio} 倍（允许的上限是 {limit_ms} ms）。检查是不是每个像素都把窗口里的像素重新加了一遍",
    "ncc_position": "模板取自图中 ({expect_y}, {expect_x})，应在原位置找到，实际返回 ({got_y}, {got_x})，分数 {score}",
    "ncc_score_exact": "模板和窗口完全相同时分数应为 1（误差 1e-9 以内），实际是 {score}。检查归一化的分母，以及累加用的是不是 double",
    "ncc_last_position": "模板取自最靠右下的合法位置 ({expect_y}, {expect_x})，应在原位置找到且分数为 1，实际返回 ({got_y}, {got_x})，分数 {score}。"
                         "检查搜索范围的两端是不是都包含了",
    "ncc_same_size": "模板和图一样大时只有 (0, 0) 一个位置，分数应为 1，实际返回 ({got_y}, {got_x})，分数 {score}。检查搜索范围的两端是不是都包含了",
    "ncc_brightness_position": "模板做了线性亮度变化后，应仍在原位置 ({expect_y}, {expect_x}) 找到，实际返回 ({got_y}, {got_x})，分数 {score}。"
                               "检查窗口和模板是不是都减了各自的均值",
    "ncc_brightness_score": "模板等于窗口乘 2 加 30 时，原位置的分数应仍为 1（误差 1e-9 以内），实际是 {score}。"
                            "检查窗口和模板是不是都减了各自的均值，分母是不是两边方差之积的平方根",
    "ncc_flat_score": "图和模板都是平坦的（方差为 0），分数应为 0，实际是 {score}。方差为 0 时要单独处理，否则是 0 除以 0",
    "ncc_flat_position": "所有位置同分（都是 0 分）时应返回光栅扫描顺序的第一个位置 (0, 0)，实际返回 ({got_y}, {got_x})。"
                         "检查最好分数的初值，以及更新时用的比较符号",
    "ncc_flat_template": "模板是平坦的（方差为 0）、图不平坦时，每个位置都是 0 分，应返回 (0, 0) 和 0 分，实际返回 ({got_y}, {got_x})，分数 {score}。"
                         "检查模板方差为 0 的情况有没有处理",
    "ncc_tie_first": "图里 ({expect_y}, {expect_x}) 和 ({other_y}, {other_x}) 两处的窗口都和模板逐像素相同，应返回光栅扫描顺序的第一个 ({expect_y}, {expect_x})，"
                     "实际返回 ({got_y}, {got_x})，分数 {score}。检查更新最好位置时用的比较符号，以及内外层循环哪个是 y",
    "ncc_too_big": "模板 {tpl_rows}x{tpl_cols} 比图 {img_rows}x{img_cols} 大（行数或列数有一个超出就算），应抛 std::invalid_argument，实际没有抛异常",
    "ncc_too_big_type": "模板 {tpl_rows}x{tpl_cols} 比图 {img_rows}x{img_cols} 大，应抛 std::invalid_argument，实际抛的是 {kind}",
    "blobs_small": "{connectivity} 连通时这张小图应有 {expect} 个连通域，实际数出 {got} 个。图里有主对角线和反对角线上各一对像素、一条横线、"
                   "一个方块和一条竖线，灰度各不相同（非 0 都算前景）。检查邻居的偏移量是不是齐全，以及前景的判断条件",
    "blobs_row_wrap": "{connectivity} 连通时应有 {expect} 个连通域，实际数出 {got} 个。这张图的前景在一行的最后一列和下一行的第一列，"
                      "它们在内存里相邻，但不是邻居。检查邻居坐标的越界判断",
    "blobs_corners": "3x3 的图四个角各有一个前景像素，{connectivity} 连通时应有 {expect} 个连通域，实际数出 {got} 个。检查邻居坐标的越界判断",
    "blobs_background": "全是背景的图和空图应返回 0，实际返回 {got}（{connectivity} 连通）",
    "blobs_full": "全是前景的图应返回 1，实际返回 {got}（{connectivity} 连通）。检查像素有没有被重复计数",
    "blobs_checker": "60x60 的棋盘格，{connectivity} 连通时应有 {expect} 个连通域，实际数出 {got} 个。检查访问标记是在什么时候打上的",
    "blobs_bad_connectivity": "connectivity={connectivity} 应抛 std::invalid_argument，实际没有抛异常",
    "blobs_bad_connectivity_type": "connectivity={connectivity} 应抛 std::invalid_argument，实际抛的是 {kind}",
    "blobs_snake": "3000x3000 的蛇形大区域应为 1 个连通域，实际数出 {got} 个",
}

STAGE_TEXT = {
    "start": "刚开始",
    "small": "小图上的数值检查",
    "empty": "空图",
    "big": "4200x4200 的全白大图",
    "large_r": "窗口比图像的边还长",
    "thin": "只有一行或一列的图",
    "negative_r": "r 为负数",
    "timing": "1200x1200 大图上的数值和计时检查",
    "exact": "模板取自图中某处",
    "last_position": "模板取自最靠右下的位置",
    "brightness": "模板做了亮度变化",
    "flat": "平坦的图或模板",
    "tie": "多个位置同分",
    "too_big": "模板比图大",
    "row_wrap": "前景在行尾和下一行行首",
    "corners": "前景在四个角",
    "trivial": "全背景、空图、全前景",
    "checker": "棋盘格",
    "bad_connectivity": "connectivity 不合法",
    "snake": "3000x3000 的蛇形大区域",
}

KIND_TEXT = {
    "out_of_range": "std::out_of_range",
    "invalid_argument": "std::invalid_argument",
    "length_error": "std::length_error",
    "bad_alloc": "std::bad_alloc",
    "logic_error": "std::logic_error",
    "runtime_error": "std::runtime_error",
    "exception": "一个别的类型的 std::exception",
    "unknown": "一个不是 std::exception 的异常",
}

EXCEPTION_HINT = {
    "out_of_range": "多半是 at() 的下标越界，检查循环范围和邻居坐标",
    "bad_alloc": "申请内存失败。先检查有没有算错要申请的大小；这一项测试本身要用一百多 MB 内存，电脑内存很紧张时关掉别的程序再试",
    "length_error": "申请的数组长度不合理（常见于尺寸算成了负数），检查输出尺寸是怎么算的",
    "invalid_argument": "这一步的输入是合法的，不应该抛这个异常，检查参数检查的条件",
}

# Windows 上进程被系统终止时的返回码（微软文档里的 NTSTATUS 值）
WINDOWS_DLL_NOT_FOUND = 0xC0000135
WINDOWS_EXIT = {
    0xC00000FD: "栈溢出",
    0xC0000005: "访问了不属于自己的内存（越界读写）",
    WINDOWS_DLL_NOT_FOUND: "找不到运行需要的 DLL",
}


class Fields(dict):
    """format_map 用：模板里用到、输出里没有的字段显示成问号，不抛异常。"""

    def __missing__(self, key: str) -> str:
        return "?"


def decode_program(data: bytes | None) -> str:
    """测试程序的输出：按 UTF-8 解码，解不了的字节换成占位符。"""
    return (data or b"").decode("utf-8", errors="replace")


def decode_tool(data: bytes | None) -> str:
    """编译器的输出：先按 UTF-8，不行再按系统的本地编码（中文 Windows 上 cl 的提示是 GBK）。"""
    data = data or b""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    # mbcs 是 Windows 的系统代码页（中文系统是 GBK），只在 Windows 上有
    fallback = "mbcs" if os.name == "nt" else (locale.getpreferredencoding(False) or "utf-8")
    try:
        return data.decode(fallback, errors="replace")
    except LookupError:
        return data.decode("utf-8", errors="replace")


def find_compiler() -> tuple[str, str] | None:
    """返回 (编译器路径, 命令行风格)。风格是 "gnu"（g++、clang++、c++）或 "msvc"（cl）。"""
    wanted = os.environ.get("CXX", "").strip().strip('"')
    if wanted:
        path = shutil.which(wanted)
        if path:
            return path, compiler_style(path)
        print(f"注意：环境变量 CXX={wanted} 指向的编译器找不到，改为自动查找。")
    for name in ("g++", "clang++", "c++", "cl"):
        path = shutil.which(name)
        if path:
            return path, compiler_style(path)
    return None


def compiler_style(path: str) -> str:
    return "msvc" if Path(path).stem.lower() in ("cl", "clang-cl") else "gnu"


def no_compiler_hint() -> str:
    looked = "没找到 C++ 编译器（依次找了环境变量 CXX、g++、clang++、c++、cl）。"
    if sys.platform.startswith("win"):
        return (looked + "安装办法见 docs/环境安装.md 的“C++ 编译器”一节。装的是 Visual Studio 生成工具时，"
                "要在开始菜单里的“x64 Native Tools Command Prompt”里运行本脚本才找得到 cl；"
                "装的是 MinGW-w64 或 LLVM 时，确认它的 bin 目录已经加进 PATH，并且重新打开了终端。")
    if sys.platform == "darwin":
        return looked + "macOS 运行 xcode-select --install 后重试。"
    return looked + "用发行版的包管理器安装 g++（例如 Debian、Ubuntu 上是 sudo apt install g++）后重试。"


def compile_command(cxx: str, style: str) -> list[str]:
    if style == "msvc":
        return [cxx, "/nologo", "/std:c++17", "/EHsc", "/O2", "/utf-8", "/W4", BUILD_TEST, BUILD_ASSIGNMENT, "/Fe:" + EXE]
    return [cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", BUILD_TEST, BUILD_ASSIGNMENT, "-o", EXE]


def compile_failure_hints(style: str) -> list[str]:
    hints = []
    if style == "msvc":
        hints.append("用的是 cl：如果报找不到标准头文件或 LIB，说明不是在“x64 Native Tools Command Prompt”里运行的。")
    elif os.name == "nt":
        hints.append("如果报不认识 -std=c++17，说明这个编译器太旧；如果报找不到 <vector> 这类标准头文件，"
                     "说明装的是需要配合 Visual Studio 生成工具使用的 clang++。两种情况都见 docs/环境安装.md 的“C++ 编译器”一节。")
    hints.append("想自己重现：设置环境变量 CH15_KEEP_BUILD=1 再运行本脚本，然后到它打印的临时目录里执行上面的编译命令。")
    return hints


def parse_fields(detail: str) -> Fields:
    """把 "k=v k=v what=任意文字" 拆成字典。what 总在最后，里面可以有空格。"""
    fields = Fields()
    head, sep, what = detail.partition(" what=")
    if detail.startswith("what="):
        head, sep, what = "", "what=", detail[5:]
    if sep:
        fields["what"] = what.strip()
    for item in head.split():
        key, eq, value = item.partition("=")
        if eq:
            fields[key] = value
    return fields


def to_int(text: str) -> int | None:
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def describe_failure(code: str, detail: str) -> str:
    f = parse_fields(detail)
    if "kind" in f:
        f["kind"] = KIND_TEXT.get(f["kind"], f["kind"])
    if code == "exception":
        kind = parse_fields(detail).get("kind", "unknown")
        kind_text = KIND_TEXT.get(kind, kind)
        text = f"运行到“{STAGE_TEXT.get(f['stage'], f['stage'])}”这一步时抛出了" + (" " if kind_text.isascii() else "") + kind_text
        if f.get("what"):
            text += f"（{f['what']}）"
        return text + "。" + EXCEPTION_HINT.get(kind, "检查这一步用到的输入是不是没有处理")
    if code == "integral_big":
        got, expect = to_int(f["got"]), to_int(f["expect"])
        f["diagnosis"] = ""
        if got is not None and expect is not None:
            if got % 2 ** 32 == expect % 2 ** 32:
                f["diagnosis"] = "实际值和应得值的低 32 位相同，高位丢了。"
            elif abs(got - expect) < expect * 1e-5:
                f["diagnosis"] = f"实际值和应得值只差 {abs(got - expect)}，像是累加用的类型有效位数不够。"
    if code == "box_value":
        r, y, x, rows, cols = (to_int(f[k]) for k in ("r", "y", "x", "in_rows", "in_cols"))
        if None in (r, y, x, rows, cols):
            f["where"] = ""
        elif r == 0:
            f["where"] = "r=0 时输出应等于输入本身，检查窗口两端的下标。"
        elif y < r or x < r or y + r >= rows or x + r >= cols:
            f["where"] = "这个像素离边不到 r，检查窗口的裁剪和除数（只对图像内的像素求平均）。"
        else:
            f["where"] = "这个像素的窗口完全在图像内，检查矩形求和时四个角的下标。"
    template = FAIL_TEXT.get(code)
    if template is None:
        return f"没通过（代码 {code}，{detail.strip()}）"
    return template.format_map(f)


def describe_exit(code: int) -> str:
    if code < 0:
        try:
            return f"被信号 {signal.Signals(-code).name} 终止，返回码 {code}"
        except ValueError:
            return f"被信号 {-code} 终止，返回码 {code}"
    unsigned = code & 0xFFFFFFFF
    if unsigned in WINDOWS_EXIT:
        return f"{WINDOWS_EXIT[unsigned]}，返回码 0x{unsigned:08X}"
    return f"返回码 {code}" if code < 0x10000 else f"返回码 0x{unsigned:08X}"


class Run:
    """一次或几次运行测试程序的累计结果。"""

    def __init__(self) -> None:
        self.results: dict[str, tuple[str, str]] = {}  # key -> (状态, 中文说明)
        self.stage: dict[str, str] = {}
        self.current: str | None = None
        self.other: list[str] = []
        self.stderr: list[str] = []

    def feed(self, text: str) -> None:
        for line in text.splitlines():
            parts = line.strip().split("|", 3)
            tag = parts[0]
            if tag == "RUN" and len(parts) >= 2:
                self.current = parts[1]
            elif tag == "STAGE" and len(parts) >= 3:
                self.stage[parts[1]] = parts[2]
            elif tag in ("PASS", "FAIL", "TODO") and len(parts) == 4 and parts[1] in NAMES:
                key, code, detail = parts[1], parts[2], parts[3]
                if tag == "PASS":
                    text_cn = PASS_TEXT[key].format_map(parse_fields(detail))
                elif tag == "TODO":
                    text_cn = TODO_TEXT[key]
                else:
                    text_cn = describe_failure(code, detail)
                self.results[key] = (tag, text_cn)
            elif line.strip():
                self.other.append(line.rstrip())

    def stage_text(self, key: str) -> str:
        name = self.stage.get(key, "start")
        return STAGE_TEXT.get(name, name)


def quiet_crash_dialogs() -> None:
    """Windows：程序崩溃或缺 DLL 时不弹出对话框，让它直接带着错误码退出。子进程会继承这个设置。"""
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)  # SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX
    except Exception:  # noqa: BLE001  设不上只影响体验，不影响结果
        pass


def launch(exe: Path, args: list[str], env: dict, run: Run) -> tuple[str, int | None]:
    """运行测试程序。返回 ("ok" / "crash" / "timeout" / "cannot_start", 返回码)。"""
    try:
        proc = subprocess.run([str(exe), *args], capture_output=True, timeout=RUN_TIMEOUT, env=env, cwd=str(exe.parent))
    except subprocess.TimeoutExpired as exc:
        run.feed(decode_program(exc.stdout))
        return "timeout", None
    except OSError as exc:
        run.stderr.append(f"{type(exc).__name__}: {exc}")
        return "cannot_start", None
    run.feed(decode_program(proc.stdout))
    run.stderr.extend(decode_program(proc.stderr).splitlines())
    return ("ok" if proc.returncode == 0 else "crash"), proc.returncode


def abnormal_text(run: Run, key: str, status: str, code: int | None) -> str:
    where = run.stage_text(key)
    if status == "timeout":
        return f"运行超过 {RUN_TIMEOUT} 秒，停在“{where}”这一步。检查有没有死循环，或者每个像素都把整个窗口重新遍历一遍的写法"
    if status == "cannot_start":
        return "测试程序启动不了，见下面的系统报错。Windows 上常见原因是杀毒软件拦截了刚编译出来的程序"
    if code is not None and code & 0xFFFFFFFF == WINDOWS_DLL_NOT_FOUND:
        return ("测试程序启动不了：找不到运行需要的 DLL（返回码 0xC0000135）。多半是编译器自带的运行库不在 PATH 里，"
                "把编译器的 bin 目录加进 PATH 后重新打开终端再试")
    text = f"程序在“{where}”这一步异常退出（{describe_exit(code if code is not None else 1)}）。"
    if key == "blobs" and run.stage.get(key) == "snake":
        return text + "前面的小图都已通过，到大区域才崩溃，多半是递归太深把栈用完了，按作业要求改用显式的栈或队列"
    return text + "常见原因是越界读写或递归栈溢出；把 operator() 临时换成带检查的 at()，越界时会得到明确的报错"


def run_tests(exe: Path, env: dict) -> Run:
    run = Run()
    status, code = launch(exe, [], env, run)
    pending = [key for key, _ in TESTS if key not in run.results]
    if status == "ok" and not pending:
        return run
    if status == "cannot_start":
        for key in pending:
            run.results[key] = ("FAIL", abnormal_text(run, key, status, code))
        return run
    # 没跑完：正在运行的那一项记为失败，剩下的每项单独再运行一次，互不影响
    broken = run.current if run.current in pending else (pending[0] if pending else None)
    if broken is not None and status != "ok":
        run.results[broken] = ("FAIL", abnormal_text(run, broken, status, code))
    for key in [k for k in pending if k not in run.results]:
        run.current = None
        status, code = launch(exe, [key], env, run)
        if key in run.results:
            continue
        if status == "ok":
            run.results[key] = ("FAIL", "测试程序正常结束了，但没有输出这一项的结果。检查作业代码里有没有调用 exit 之类提前结束程序的函数")
        else:
            run.results[key] = ("FAIL", abnormal_text(run, key, status, code))
    return run


def report_timing(exe: Path, env: dict) -> int:
    """--timing：只做盒式均值的计时，逐个半径打印耗时。"""
    try:
        proc = subprocess.run([str(exe), "timing"], capture_output=True, timeout=RUN_TIMEOUT, env=env, cwd=str(exe.parent))
    except subprocess.TimeoutExpired:
        print(f"[FAIL] 计时 — 运行超过 {RUN_TIMEOUT} 秒")
        return 1
    except OSError as exc:
        print(f"[FAIL] 计时 — 测试程序启动不了：{type(exc).__name__}: {exc}")
        return 1
    rows = 0
    stopped = False
    tick = "?"
    for line in decode_program(proc.stdout).splitlines():
        parts = line.strip().split("|", 3)
        if parts[0] != "TIMING" or len(parts) != 4:
            continue
        f = parse_fields(parts[3])
        if parts[2] == "exception":
            print("[TODO] 计时 — box_mean 还没有实现，或者运行时抛出了异常，先让验收的数值检查通过")
            return 1
        rows += 1
        tick = f["tick_ms"]
        stopped = parts[2] == "slow"
        verdict = "超过上限，判为耗时随半径增长" if stopped else "在上限以内"
        print("{n}x{n} 的图，交替计时 {reps} 轮取最快：r=1 {small_ms} ms，r={r} {large_ms} ms，是 {ratio} 倍；上限 {limit_ms} ms，".format_map(f) + verdict)
    if proc.returncode != 0 or rows == 0:
        print(f"[FAIL] 计时 — 测试程序没有正常结束（{describe_exit(proc.returncode)}）")
        return 1
    if stopped:
        print("已经超过上限，更大的半径不再测（和验收的做法一样）。")
    print("判据：r 大的那个最快耗时超过 r=1 最快耗时的 4 倍再加余量（2 毫秒，计时器很粗时取它分辨率的 4 倍）才算超过。")
    print(f"这次运行量到的计时器分辨率：{tick} ms（显示 0.000 表示小于 1 微秒）。")
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # 控制台编码里没有的字符换成问号，不让 print 报错
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    timing_only = "--timing" in sys.argv[1:]
    print(("第 15 章盒式均值计时" if timing_only else "第 15 章进度检查") + "\n" + RULE)
    override = os.environ.get("COURSE_IMPL_DIR")
    source = (Path(override) / CHAPTER.name if override else CHAPTER) / ASSIGNMENT
    if not source.is_file():
        print(f"[FAIL] 作业文件 — 找不到 {source}")
        return 1
    found = find_compiler()
    if found is None:
        print("[TODO] 编译器 — " + no_compiler_hint())
        return 1
    cxx, style = found
    print(f"编译器：{cxx}")

    keep = os.environ.get("CH15_KEEP_BUILD") == "1"
    build = Path(tempfile.mkdtemp(prefix="ch15_"))
    try:
        shutil.copyfile(source, build / BUILD_ASSIGNMENT)
        shutil.copyfile(CHAPTER / "tests" / "test_main.cpp", build / BUILD_TEST)
        for header in sorted((CHAPTER / "cpp").iterdir()):
            if header.suffix.lower() in (".hpp", ".h"):
                shutil.copyfile(header, build / header.name)
        cmd = compile_command(cxx, style)
        env = dict(os.environ)
        if os.name == "nt":  # 编译器自己和编译出来的测试程序都要能找到它 bin 目录里的 DLL（MinGW-w64 需要）
            env["PATH"] = str(Path(cxx).parent) + os.pathsep + env.get("PATH", "")
        try:
            proc = subprocess.run(cmd, capture_output=True, cwd=str(build), env=env)
        except OSError as exc:
            print(f"[FAIL] 编译 — 编译器启动不了：{type(exc).__name__}: {exc}")
            return 1
        # cl 把诊断信息写到标准输出，g++ 和 clang++ 写到标准错误，两边都要看
        log = (decode_tool(proc.stdout) + "\n" + decode_tool(proc.stderr)).replace(BUILD_ASSIGNMENT, ASSIGNMENT)
        log_lines = [line for line in log.splitlines() if line.strip()]
        if proc.returncode != 0 or not (build / EXE).is_file():
            print("[FAIL] 编译 — 编译失败，前 30 行输出：")
            print("\n".join(log_lines[:30]))
            print("编译命令：" + " ".join(cmd))
            for hint in compile_failure_hints(style):
                print("提示：" + hint)
            return 1
        warnings = [line for line in log_lines if "warning" in line.lower() and ASSIGNMENT in line]
        quiet_crash_dialogs()
        if timing_only:
            return report_timing(build / EXE, env)
        run = run_tests(build / EXE, env)
    finally:
        if keep:
            print(f"临时构建目录保留在：{build}")
        else:
            shutil.rmtree(build, ignore_errors=True)

    passed = 0
    for key, name in TESTS:
        status, text = run.results.get(key, ("FAIL", "没有运行"))
        passed += status == "PASS"
        print(f"[{status:4}] {name} — {text}")
    if run.other:
        print("测试程序的其他输出（最多 10 行，多半是作业代码自己打印的）：")
        print("\n".join("    " + line for line in run.other[:10]))
    if run.stderr and any(status == "FAIL" for status, _ in run.results.values()):
        print("测试程序写到标准错误的内容（最多 10 行）：")
        print("\n".join("    " + line for line in run.stderr[:10]))
    if warnings:
        flags = "/W4" if style == "msvc" else "-Wall -Wextra"
        print(f"注意：作业文件编译有 {len(warnings)} 条警告，建议清零（{flags}）。前 5 条：")
        print("\n".join("    " + line for line in warnings[:5]))
    total = len(TESTS)
    print(RULE)
    print(f"通过 {passed}/{total}")
    if passed != total:
        print("提示：TODO 是正常学习状态；实现后重新运行。")
        return 1
    print("基本功能通过；下一步装 OpenCV C++ 版，用 cv::integral 和 cv::matchTemplate 对照性能。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
