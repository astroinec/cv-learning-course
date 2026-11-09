// 作业 15-1：用 C++17 实现积分图、盒式均值、NCC 模板匹配和连通域计数。
// 只能用标准库，不依赖 OpenCV。声明和契约见 cpp/image.hpp。
// 编译检查：python3 tests/check_progress.py（会自动调用 c++ 编译并运行测试）。

#include "image.hpp"

#include <stdexcept>

namespace course {

ImageU64 integral_image(const ImageU8& img) {
    (void)img;
    throw std::logic_error("TODO: 实现积分图");
}

ImageF box_mean(const ImageU8& img, int r) {
    (void)img;
    (void)r;
    throw std::logic_error("TODO: 用积分图实现盒式均值");
}

MatchResult ncc_best_match(const ImageU8& img, const ImageU8& tpl) {
    (void)img;
    (void)tpl;
    throw std::logic_error("TODO: 实现 NCC 模板匹配");
}

int count_blobs(const ImageU8& binary, int connectivity) {
    (void)binary;
    (void)connectivity;
    throw std::logic_error("TODO: 用显式队列实现连通域计数");
}

}  // namespace course
