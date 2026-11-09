// 第 15 章 C++ 测试：每个用例输出一行 "STATUS|名称|说明"，由 check_progress.py 解析。
#include "image.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>

using namespace course;

namespace {

struct Outcome {
    bool ok;
    std::string detail;
};

void run(const char* name, const std::function<Outcome()>& fn) {
    try {
        Outcome o = fn();
        std::printf("%s|%s|%s\n", o.ok ? "PASS" : "FAIL", name, o.detail.c_str());
    } catch (const std::logic_error& e) {
        std::string msg = e.what();
        bool todo = msg.rfind("TODO", 0) == 0;
        std::printf("%s|%s|%s\n", todo ? "TODO" : "FAIL", name, msg.c_str());
    } catch (const std::exception& e) {
        std::printf("FAIL|%s|运行异常：%s\n", name, e.what());
    }
}

ImageU8 random_image(int rows, int cols, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(0, 255);
    ImageU8 img(rows, cols);
    for (auto& v : img.data) v = static_cast<std::uint8_t>(d(rng));
    return img;
}

Outcome test_integral() {
    ImageU8 img = random_image(13, 17, 1);
    ImageU64 ii = integral_image(img);
    if (ii.rows != 14 || ii.cols != 18) return {false, "积分图尺寸应为 (rows+1, cols+1)"};
    for (int x = 0; x < ii.cols; ++x)
        if (ii.at(0, x) != 0) return {false, "第 0 行应全为 0"};
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) {
            std::uint64_t s = 0;
            for (int yy = 0; yy <= y; ++yy)
                for (int xx = 0; xx <= x; ++xx) s += img.at(yy, xx);
            if (ii.at(y + 1, x + 1) != s) return {false, "积分图数值与暴力求和不一致"};
        }
    ImageU8 big(4000, 4000, 255);
    if (integral_image(big).at(4000, 4000) != 4000ull * 4000ull * 255ull)
        return {false, "大图累加溢出，检查累加类型"};
    return {true, "数值与暴力求和一致，大图不溢出"};
}

Outcome test_box_mean() {
    ImageU8 img = random_image(20, 25, 2);
    for (int r : {0, 1, 3}) {
        ImageF m = box_mean(img, r);
        if (m.rows != img.rows || m.cols != img.cols) return {false, "输出尺寸应与输入相同"};
        for (int y = 0; y < img.rows; ++y)
            for (int x = 0; x < img.cols; ++x) {
                double s = 0;
                int n = 0;
                for (int yy = y - r; yy <= y + r; ++yy)
                    for (int xx = x - r; xx <= x + r; ++xx)
                        if (yy >= 0 && yy < img.rows && xx >= 0 && xx < img.cols) {
                            s += img.at(yy, xx);
                            ++n;
                        }
                if (std::fabs(m.at(y, x) - s / n) > 1e-9) return {false, "均值或边界裁剪不对（r=" + std::to_string(r) + "）"};
            }
    }
    try {
        box_mean(img, -1);
        return {false, "r < 0 应抛 std::invalid_argument"};
    } catch (const std::invalid_argument&) {
    }
    ImageU8 big = random_image(2000, 2000, 3);
    auto t0 = std::chrono::steady_clock::now();
    box_mean(big, 1);
    auto t1 = std::chrono::steady_clock::now();
    box_mean(big, 30);
    auto t2 = std::chrono::steady_clock::now();
    double small = std::chrono::duration<double>(t1 - t0).count();
    double large = std::chrono::duration<double>(t2 - t1).count();
    if (large > small * 4 + 0.05) return {false, "r 增大时耗时明显增长，没有做到 O(1) 每像素"};
    return {true, "与暴力均值一致，边界正确，耗时与半径无关"};
}

Outcome test_ncc() {
    ImageU8 img = random_image(40, 50, 4);
    ImageU8 tpl(9, 11);
    for (int y = 0; y < 9; ++y)
        for (int x = 0; x < 11; ++x) tpl.at(y, x) = img.at(17 + y, 23 + x);
    MatchResult m = ncc_best_match(img, tpl);
    if (m.y != 17 || m.x != 23) return {false, "最佳位置应为模板来源 (17, 23)"};
    if (std::fabs(m.score - 1.0) > 1e-9) return {false, "完全匹配的 NCC 应为 1"};
    ImageU8 brighter = img;
    for (auto& v : brighter.data) v = static_cast<std::uint8_t>(v / 2 + 40);
    ImageU8 tpl2(9, 11);
    for (int y = 0; y < 9; ++y)
        for (int x = 0; x < 11; ++x) tpl2.at(y, x) = brighter.at(17 + y, 23 + x);
    MatchResult m2 = ncc_best_match(img, tpl2);
    if (m2.y != 17 || m2.x != 23) return {false, "亮度线性变化后应仍能找到原位置"};
    ImageU8 flat(10, 10, 5);
    if (ncc_best_match(flat, ImageU8(3, 3, 5)).score != 0.0) return {false, "平坦窗口分数应为 0"};
    try {
        ncc_best_match(tpl, img);
        return {false, "模板比图大时应抛 std::invalid_argument"};
    } catch (const std::invalid_argument&) {
    }
    return {true, "位置、分数、亮度不变性和异常处理正确"};
}

Outcome test_blobs() {
    ImageU8 b(8, 10, 0);
    b.at(1, 1) = b.at(2, 2) = 255;
    for (int x = 6; x < 9; ++x) b.at(1, x) = 1;
    b.at(5, 3) = b.at(5, 4) = b.at(6, 3) = b.at(6, 4) = 7;
    if (count_blobs(b, 8) != 3) return {false, "8 连通应有 3 个连通域"};
    if (count_blobs(b, 4) != 4) return {false, "4 连通应有 4 个连通域"};
    ImageU8 snake(3000, 3000, 0);
    for (int y = 0; y < 3000; y += 2)
        for (int x = 0; x < 3000; ++x) snake.at(y, x) = 1;
    for (int y = 1; y < 3000; y += 4) snake.at(y, 2999) = 1;
    for (int y = 3; y < 3000; y += 4) snake.at(y, 0) = 1;
    if (count_blobs(snake, 4) != 1) return {false, "蛇形大区域应为 1 个（如果崩溃，多半是递归栈溢出）"};
    try {
        count_blobs(b, 6);
        return {false, "connectivity 非法时应抛 std::invalid_argument"};
    } catch (const std::invalid_argument&) {
    }
    return {true, "4/8 连通正确，大区域不栈溢出"};
}

}  // namespace

int main() {
    run("积分图", test_integral);
    run("盒式均值", test_box_mean);
    run("NCC 模板匹配", test_ncc);
    run("连通域计数", test_blobs);
    return 0;
}
