// 选做：装好 OpenCV 后，对照自己的实现和 OpenCV 的结果与耗时。
// 编译：cmake -S . -B build -DWITH_OPENCV=ON && cmake --build build && ./build/ch15_opencv_compare
#include "image.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>

using namespace course;

int main() {
    cv::Mat src(1024, 1024, CV_8UC1);
    cv::randu(src, 0, 256);

    // 关键点 1：cv::Mat 和 course::Image 的数据布局都是行优先，可以零拷贝转换。
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

    std::cout << "内部区域最大差 " << max_diff << "（OpenCV 输出取整到 8 位，差 1 以内正常）\n";
    std::cout << "自己实现 " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms，"
              << "OpenCV " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";
    return 0;
}
