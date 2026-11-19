// 选做：装好 OpenCV 后，对照自己的实现和 OpenCV 的结果与耗时。
// 编译（在本章目录下，先装好 C++ 版 OpenCV 和 CMake）：
//   Windows（Visual Studio 生成工具）：
//     cmake -S . -B build -DWITH_OPENCV=ON
//     cmake --build build --config Release
//     build\Release\ch15_opencv_compare.exe
//   macOS 和 Linux：
//     cmake -S . -B build -DWITH_OPENCV=ON && cmake --build build && ./build/ch15_opencv_compare
// 程序的输出用英文：中文 Windows 的控制台默认不是 UTF-8 编码，直接输出中文会乱码。
#include "image.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

using namespace course;

int main() {
    cv::Mat src(1024, 1024, CV_8UC1);
    cv::randu(src, 0, 256);

    // 关键点 1：cv::Mat 和 course::Image 的数据布局都是行优先，所以可以一行一行地整段拷贝。
    // 这里做的是拷贝，不是零拷贝：course::Image 自己拥有一块 std::vector，没有“借用别人内存”的视图功能。
    // 逐行拷而不是整块 memcpy，是因为 cv::Mat 可能有行尾填充或者是 ROI（第 1 节），行与行之间不一定连续。
    ImageU8 mine(src.rows, src.cols);
    for (int y = 0; y < src.rows; ++y) {
        const std::uint8_t* row = src.ptr<std::uint8_t>(y);  // 用 ptr 逐行访问，比 at 快
        std::copy(row, row + src.cols, &mine(y, 0));
    }

    auto t0 = std::chrono::steady_clock::now();
    ImageF m = box_mean(mine, 7);
    auto t1 = std::chrono::steady_clock::now();
    cv::Mat ref;
    cv::blur(src, ref, cv::Size(15, 15), cv::Point(-1, -1), cv::BORDER_REFLECT_101);
    auto t2 = std::chrono::steady_clock::now();

    // 关键点 2：边界策略不同（我们裁剪窗口，OpenCV 默认反射），只比较内部区域。
    double max_diff = 0;
    for (int y = 7; y < src.rows - 7; ++y)
        for (int x = 7; x < src.cols - 7; ++x)
            max_diff = std::max(max_diff, std::abs(m(y, x) - ref.at<std::uint8_t>(y, x)));

    // OpenCV 的输出取整到 8 位，内部区域的最大差在 1 以内是正常的。
    std::cout << "max abs diff (interior): " << max_diff << "  (OpenCV rounds to 8 bit, below 1 is expected)\n";
    std::cout << "mine: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms, "
              << "OpenCV: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";
    return 0;
}
