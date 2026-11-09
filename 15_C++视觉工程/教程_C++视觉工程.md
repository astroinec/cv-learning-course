# 教程：面向视觉的 C++ 工程

这一章不是完整的 C++ 教程，而是一份“工业视觉团队里最常用到、最容易出错”的清单。语法细节请配合 LearnCpp 查。

## 1. 图像在内存里长什么样

一张 `H x W` 的单通道 8 位图，在内存里是 `H * W` 个连续字节，按行排列：

```text
地址：  0        1        ...  W-1      W        W+1 ...
像素：  (0,0)    (0,1)    ...  (0,W-1)  (1,0)    (1,1) ...
```

下标公式 `index = y * step + x`。`step` 是“一行占多少个元素”，通常等于 `W`，但不总是：

- 从大图里取 ROI 时，子图的 `step` 还是大图的宽度，数据不连续。
- 有些相机 SDK 为了对齐会在每行末尾补字节（padding），`step > W`。

所以写遍历代码时，用 `step` 而不是 `W` 算行首地址，这是新人改别人代码时最常见的越界来源。OpenCV 里对应 `mat.step`（字节数）和 `mat.ptr<T>(y)`（第 y 行行首指针）。

**多通道**：BGR 三通道是交错存储 `B G R B G R ...`，下标为 `(y * step + x) * 3 + c`。

**遍历顺序**：外层循环 y、内层循环 x，按地址顺序访问，缓存命中最好。反过来写，大图上可能慢好几倍。

## 2. 类型与溢出

| 场景 | 坑 | 做法 |
|---|---|---|
| 两个 `uint8_t` 相加 | 255 + 1 在存回 `uint8_t` 时变成 0 | 先提升到 `int` 计算，再饱和截断（OpenCV 的 `saturate_cast`） |
| 积分图 | 4000x4000 的全白图求和约 4e9，超过 `uint32_t` 上限 | 用 `uint64_t` 或 `double` |
| 平方和（方差、NCC） | `int` 很容易溢出 | 用 `double` 或 `int64_t` |
| 有符号和无符号混用 | `size_t i; i - 1` 在 i=0 时变成巨大正数 | 下标用 `int`，或者先判断再减 |
| 浮点比较 | `a == b` 不可靠 | 用 `std::fabs(a - b) < eps` |

`-Wall -Wextra` 会提示大部分有符号和无符号比较的问题，警告要清零，不要忽略。

## 3. 值语义、引用和拷贝

```cpp
void process(Image<uint8_t> img);         // 按值传：整张图拷贝一次，慢
void process(const Image<uint8_t>& img);  // const 引用：不拷贝，也不能改，读图首选
void process(Image<uint8_t>& img);        // 非 const 引用：原地修改
```

`std::vector` 是值语义，赋值就是深拷贝。**`cv::Mat` 不一样**：它是引用计数的句柄，`Mat b = a;` 只拷贝句柄，`b` 和 `a` 指向同一块像素。要深拷贝用 `a.clone()` 或 `a.copyTo(b)`。这个差异导致过无数“我明明改的是副本，原图怎么也变了”的 bug。

**RAII**：资源跟对象生命周期绑定。用 `std::vector`、`std::unique_ptr` 管理内存，不要手写 `new`/`delete`。相机缓冲区、文件句柄也用类包起来，在析构函数里释放。

## 4. 不要用递归做连通域和泛洪填充

一个 3000x3000 的蛇形前景区域，递归深度可达几百万层，默认栈（macOS 主线程 8 MB，子线程通常更小）一定会溢出，表现是程序直接崩溃（Segmentation fault），而且只在大图上出现。正确做法是用 `std::queue`（BFS）或 `std::vector` 当显式栈（DFS）。本章测试专门放了一个蛇形大区域来检查这一点。

## 5. 积分图：让任意矩形求和变成 O(1)

定义 `II(y, x) = sum(img[0..y-1, 0..x-1])`，尺寸为 `(H+1) x (W+1)`，多出的第 0 行和第 0 列全为 0，这样就不用处理边界特判。矩形 `[y0, y1) x [x0, x1)` 的和是：

```text
S = II(y1, x1) - II(y0, x1) - II(y1, x0) + II(y0, x0)
```

递推：`II(y+1, x+1) = II(y, x+1) + rowsum(y, 0..x)`。

用途：盒式滤波、局部均值和方差（同时维护平方的积分图）、NCC 分母加速、Haar 特征。本章的盒式均值要求“耗时与半径无关”，测试会对比 r=1 和 r=30 的耗时。

## 6. NCC 模板匹配的工程细节

公式和第 16 章相同：

```text
NCC = Σ (I - Ī)(T - T̄) / sqrt( Σ (I - Ī)^2 · Σ (T - T̄)^2 )
```

- 模板的均值和方差只算一次。
- 窗口方差为 0（平坦区域）时分母为 0，必须特判，否则产生 NaN，NaN 和任何数比较都是 false，会导致“最大值”找错。
- 直接实现是 O(H·W·h·w)，大图大模板很慢。工程上的做法：积分图算窗口均值和方差，FFT 算分子；或者图像金字塔先在缩小图上粗搜，再在原图局部精搜。

## 7. CMake 最小知识

```cmake
cmake_minimum_required(VERSION 3.16)
project(demo CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)                 # 找到 brew 安装的 OpenCV
add_executable(demo main.cpp)
target_link_libraries(demo PRIVATE ${OpenCV_LIBS})
```

构建三步：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # 配置
cmake --build build -j                           # 编译
./build/demo                                     # 运行
```

几个要点：

- **Debug 和 Release 性能差距可以是 10 倍以上**，测速一定用 Release。
- 用 `target_*` 系列命令（`target_include_directories`、`target_link_libraries`），不要用全局的 `include_directories`。
- 构建目录和源码分开（`-B build`），不要在源码目录里直接 `cmake .`。

## 8. 调试与定位

| 问题 | 工具 |
|---|---|
| 越界、use-after-free | 编译加 `-fsanitize=address`，出错时直接打印出错行和调用栈 |
| 整数溢出、未定义行为 | `-fsanitize=undefined` |
| 崩溃后看现场 | `lldb ./build/demo`，`run` 后 `bt` 看调用栈 |
| 慢 | macOS 上 Instruments 的 Time Profiler；或者先用 `std::chrono` 给各阶段计时 |

本章 `CMakeLists.txt` 已经提供 `-DWITH_SANITIZERS=ON` 选项。建议故意在 `box_mean` 里把 `y1` 写成 `y + r + 2`，体验一次 ASan 报错，记住它长什么样。

## 9. 性能的几条朴素规则

1. 先测量再优化，只优化最热的那一段。
2. 热循环里不要用带边界检查的 `at()`，用行指针。
3. 不要在循环里分配内存（`std::vector` 的 `push_back` 提前 `reserve`）。
4. 能用积分图、可分离滤波（先行后列）降低复杂度的，先降复杂度，再考虑 SIMD 和多线程。
5. OpenCV 的大部分函数已经做了 SIMD 和多线程优化，自己重写前先对比一下。

## 10. Python 和 C++ 怎么协作

常见模式是 Python 里做算法原型和评估，验证完再移植到 C++。移植时：

- 用同一组输入图和同一份期望输出做对拍，逐像素比较差异。
- 注意边界策略（反射、复制、补零）、取整方式（四舍五入还是截断）、颜色通道顺序（OpenCV 是 BGR）这三类最容易不一致的地方。
- 需要在 Python 里调用 C++ 时，了解 pybind11 即可，入门阶段不必深挖。

## 自测问题

1. 一个 1920x1080 的 BGR 图，像素 (y=100, x=200) 的 G 通道在第几个字节？
2. 从大图取 ROI 后，为什么不能用 `memcpy(dst, roi.data, W*H)` 拷贝？
3. `Mat a = imread(...); Mat b = a; b.setTo(0);` 之后 `a` 是什么？
4. 用积分图求局部方差需要几张积分图？
5. 为什么 NCC 对亮度的线性变化不敏感？
