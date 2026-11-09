# 第 15 章：C++ 视觉工程

本章用时建议 **12–18 小时**（C++ 基础薄弱时再加 1 周）。目标：能读懂并修改工业视觉团队的 C++ 工程，写出不越界、不溢出、性能可预期的图像处理代码，会用 CMake 构建、会用 Sanitizer 定位内存问题、会把 OpenCV 接进来。

## 为什么先学这一章

工业视觉的算法验证常在 Python 里做，但上机台的版本几乎都是 C++：检测速度要跟上机台的产能，内存和延迟要可控，还要和运动控制、相机 SDK、上位机软件集成。新人最常见的卡点不是算法，而是看不懂工程、编不过、跑起来崩溃或慢。

## 学习顺序

1. 阅读 `教程_C++视觉工程.md`，重点是“内存布局”“整数溢出”“避免递归”三节。
2. 打开 `cpp/image.hpp`，看懂 `Image<T>` 的设计：它和 `cv::Mat` 的概念一一对应。
3. 在 `15-1_图像基础.cpp` 里依次实现 4 个函数，每实现一个就运行一次验收。
4. （选做）`brew install cmake opencv`，用 `CMakeLists.txt` 构建，打开 `-DWITH_SANITIZERS=ON` 故意制造一次越界，看报错长什么样。
5. （选做）编译 `cpp/opencv_compare.cpp`，对照自己的实现和 OpenCV 的结果与耗时。
6. 在 `实验记录.md` 写下耗时对比、一次崩溃的定位过程。

## 文件

| 文件 | 用途 |
|---|---|
| `教程_C++视觉工程.md` | 面向视觉的 C++ 要点、CMake、调试和性能 |
| `cpp/image.hpp` | 已写好的图像容器、函数声明和 PGM 读写 |
| `15-1_图像基础.cpp` | 作业：积分图、盒式均值、NCC 匹配、连通域计数 |
| `tests/test_main.cpp` | C++ 测试程序 |
| `tests/check_progress.py` | 编译并运行测试，输出 TODO/FAIL/PASS |
| `CMakeLists.txt` | CMake 工程，含 Sanitizer 和 OpenCV 选项 |
| `cpp/opencv_compare.cpp` | 选做：和 OpenCV 对照 |
| `实验记录.md` | 实验记录模板 |

## 验收

```bash
python3 15_C++视觉工程/tests/check_progress.py
```

脚本会用系统自带的 `c++` 编译器编译，不需要 CMake 和 OpenCV。

## 通过标准

- 4 项测试全部 PASS，编译零警告（`-Wall -Wextra`）。
- 能解释为什么积分图要用 64 位累加、为什么连通域不能写递归。
- 能说清 `cv::Mat` 的浅拷贝语义：`Mat b = a;` 之后改 `b` 会不会影响 `a`，怎样才是深拷贝。
- 能独立写出一个 CMake 工程并链接 OpenCV。

## 推荐资源

- [LearnCpp](https://www.learncpp.com/)：系统的 C++ 教程，挑第 1–12 章、第 16–17 章（vector、array）和第 22 章（智能指针）补基础。
- [OpenCV 官方教程：Mat 基础](https://docs.opencv.org/4.x/d6/d6d/tutorial_mat_the_basic_image_container.html)：引用计数和浅拷贝。
- [OpenCV 官方教程：高效遍历图像](https://docs.opencv.org/4.x/db/da5/tutorial_how_to_scan_images.html)：`ptr` 逐行访问、LUT 和性能对比。
- [CMake 官方教程](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)：前 4 步足够应付日常工程。
