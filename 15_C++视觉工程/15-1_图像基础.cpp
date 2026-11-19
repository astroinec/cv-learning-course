// 作业 15-1：用 C++17 实现积分图、盒式均值、NCC 模板匹配和连通域计数。
// 只能用标准库，不依赖 OpenCV。声明和契约见 cpp/image.hpp，每个函数要满足的行为都写在那里的注释里。
// 编译检查：python tests/check_progress.py（macOS 和 Linux 上是 python3），会自动找编译器（g++、clang++ 或 cl）编译并运行测试。
// 用到标准库的东西时自己加头文件，例如 std::max 和 std::min 在 <algorithm> 里；有的编译器不加也能过，换一个就报错。

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
