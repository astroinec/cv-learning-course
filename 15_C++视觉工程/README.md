# 第 15 章：C++ 视觉工程

本章用时：速通 **4 小时**，标准 **15 小时**，深挖 **22 小时**，各档做什么见教程第 0 节（C++ 基础薄弱时再加 1 周）。目标：能读懂并修改工业视觉团队的 C++ 工程，写出不越界、不溢出、性能可预期的图像处理代码，会用 CMake 构建、会用 Sanitizer 定位内存问题、会把 OpenCV 接进来。

## 为什么先学这一章

工业视觉的算法验证常在 Python 里做，但上机台的版本几乎都是 C++：检测速度要跟上机台的产能，内存和延迟要可控，还要和运动控制、相机 SDK、上位机软件集成。新人最常见的卡点不是算法，而是看不懂工程、编不过、跑起来崩溃或慢。

## 开始之前：装一个 C++ 编译器

本章是全书唯一需要 C++ 编译器的一章，装下面任意一种就够。安装步骤见 `docs/环境安装.md` 的“C++ 编译器”一节，三条路各自的前提和注意事项见教程第 7.2 节。

| 系统 | 编译器 | 前提 |
|---|---|---|
| Windows | MinGW-w64 的 `g++`（MSYS2 或其他发行版） | 编译器的 `bin` 目录在 `PATH` 里 |
| Windows | LLVM 的 `clang++` | LLVM 的 `bin` 目录在 `PATH` 里；官方安装包里的 `clang++` 还需要装好 Visual Studio 生成工具 |
| Windows | Visual Studio 生成工具的 `cl` | 要在开始菜单里的“x64 Native Tools Command Prompt”里运行验收脚本，普通终端里找不到 `cl` |
| macOS | 系统自带的 `clang++` | 运行过 `xcode-select --install` |
| Linux | `g++` | 用发行版的包管理器安装 |

确认装好了：新开一个终端，运行 `g++ --version`、`clang++ --version` 或 `cl`，能打印出版本信息就可以。

> Windows 相关的步骤没有在 Windows 上实测过（写课程的机器是 Mac）。第一次运行时照教程第 7.2 节“第一次在 Windows 上运行先确认什么”逐条核对，遇到问题以你机器上的报错为准。

## 学习顺序

1. 阅读 `教程_C++视觉工程.md`，重点是“内存布局”“整数溢出”“避免递归”三节。
2. 打开 `cpp/image.hpp`，看懂 `Image<T>` 的设计：它和 `cv::Mat` 的概念一一对应。四个作业函数要满足的行为都写在这个文件的注释里。
3. 先不写代码，运行一次验收（见下面“验收”一节），确认编译器能用：应该看到四行 `[TODO]`。
4. 在 `15-1_图像基础.cpp` 里依次实现 4 个函数，每实现一个就运行一次验收。
5. （选做）装好 CMake，用 `CMakeLists.txt` 构建，打开 `-DWITH_SANITIZERS=ON` 故意制造一次越界，看报错长什么样。这个选项只支持 GCC 和 Clang，Windows 上建议在 WSL 里做（教程第 7.2 节）。
6. （选做）装好 C++ 版 OpenCV，编译 `cpp/opencv_compare.cpp`，对照自己的实现和 OpenCV 的结果与耗时。
7. 在 `实验记录.md` 写下耗时对比、一次崩溃的定位过程。

## 文件

| 文件 | 用途 |
|---|---|
| `教程_C++视觉工程.md` | 面向视觉的 C++ 要点、CMake、调试和性能 |
| `figures/` | 教程配图和 `实测数字.json`（教程里所有数字的来源） |
| `../demos/连通域遍历.html` | 交互演示：递归、显式栈、队列三种遍历，浏览器直接打开 |
| `cpp/image.hpp` | 已写好的图像容器、函数声明（作业的契约）和 PGM 读写 |
| `15-1_图像基础.cpp` | 作业：积分图、盒式均值、NCC 匹配、连通域计数 |
| `tests/check_progress.py` | 验收脚本：找编译器、编译、运行测试，用中文输出 TODO/FAIL/PASS |
| `tests/test_main.cpp` | C++ 测试程序，只输出英文的状态和代码，由验收脚本翻译 |
| `CMakeLists.txt` | CMake 工程，含 MSVC 分支、Sanitizer 和 OpenCV 选项（选做才用到） |
| `cpp/opencv_compare.cpp` | 选做：和 OpenCV 对照 |
| `实验记录.md` | 实验记录模板 |

## 验收

在课程的根目录下运行。Windows（PowerShell 或 cmd；用 `cl` 时在“x64 Native Tools Command Prompt”里）：

```text
python "15_C++视觉工程\tests\check_progress.py"
```

macOS 和 Linux：

```bash
python3 15_C++视觉工程/tests/check_progress.py
```

脚本不需要 CMake 和 OpenCV。它做的事：

1. 找编译器，顺序是环境变量 `CXX`、`g++`、`clang++`、`c++`、`cl`，开头会打印一行“编译器：”，说明用的是哪一个。想指定时设置 `CXX`：PowerShell 里 `$env:CXX = "clang++"`，cmd 里 `set CXX=clang++`，macOS 和 Linux 在命令前面加 `CXX=clang++`。
2. 把作业和测试复制到系统的临时目录里编译（`g++`、`clang++` 用 `-std=c++17 -O2 -Wall -Wextra`，`cl` 用 `/std:c++17 /EHsc /O2 /utf-8 /W4`），用完就删。
3. 运行测试，逐项显示 `[TODO]`、`[FAIL]` 或 `[PASS]`。`[FAIL]` 后面的说明会指出是哪个输入、哪个位置不对，该查什么。某一项让程序崩溃时，其余几项照常显示。

另外两个用法：

- 加 `--timing`：只做盒式均值的计时，打印 r=1 和 r=10、30、200 的耗时与倍数，填实验记录时用。
- 设置环境变量 `CH15_KEEP_BUILD=1`：保留临时构建目录并打印位置，编译失败想自己重现时用。

验收过程中内存占用的峰值约 200 MB（主要是一张 4200×4200 图的 64 位积分图），整个验收在写课程的机器上一两秒跑完。

## 通过标准

- 4 项测试全部 PASS，作业文件编译零警告（`g++` 和 `clang++` 是 `-Wall -Wextra`，`cl` 是 `/W4`）。
- 能解释为什么积分图要用 64 位累加、为什么连通域不能写递归。
- 能说清 `cv::Mat` 的浅拷贝语义：`Mat b = a;` 之后改 `b` 会不会影响 `a`，怎样才是深拷贝。
- 能独立写出一个 CMake 工程并链接 OpenCV。

## 推荐资源

- [LearnCpp](https://www.learncpp.com/)：系统的 C++ 教程，挑第 1–12 章、第 16–17 章（vector、array）和第 22 章（智能指针）补基础。
- [OpenCV 官方教程：Mat 基础](https://docs.opencv.org/4.x/d6/d6d/tutorial_mat_the_basic_image_container.html)：引用计数和浅拷贝。
- [OpenCV 官方教程：高效遍历图像](https://docs.opencv.org/4.x/db/da5/tutorial_how_to_scan_images.html)：`ptr` 逐行访问、LUT 和性能对比。
- [CMake 官方教程](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)：前 4 步足够应付日常工程。
