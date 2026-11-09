// image.hpp：第 15 章的最小图像容器与函数声明。
// 这个文件已经写好，学员不需要修改；作业在 15-1_图像基础.cpp 里实现。
// 设计故意和 cv::Mat 的核心概念对齐：行优先存储、step（每行元素数）、只读访问与边界检查。
#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace course {

template <typename T>
struct Image {
    int rows = 0;
    int cols = 0;
    std::vector<T> data;  // 行优先，下标 y * cols + x

    Image() = default;
    Image(int r, int c, T fill = T{}) : rows(r), cols(c), data(static_cast<size_t>(r) * c, fill) {
        if (r < 0 || c < 0) throw std::invalid_argument("rows/cols 不能为负");
    }

    T& at(int y, int x) {
        check(y, x);
        return data[static_cast<size_t>(y) * cols + x];
    }
    const T& at(int y, int x) const {
        check(y, x);
        return data[static_cast<size_t>(y) * cols + x];
    }
    // 不做边界检查的快速访问，热循环里用；调用方保证下标合法。
    T& operator()(int y, int x) { return data[static_cast<size_t>(y) * cols + x]; }
    const T& operator()(int y, int x) const { return data[static_cast<size_t>(y) * cols + x]; }

    bool empty() const { return rows == 0 || cols == 0; }

private:
    void check(int y, int x) const {
        if (y < 0 || y >= rows || x < 0 || x >= cols) throw std::out_of_range("像素下标越界");
    }
};

using ImageU8 = Image<std::uint8_t>;
using ImageF = Image<double>;
using ImageU64 = Image<std::uint64_t>;

struct MatchResult {
    int y = -1;
    int x = -1;
    double score = 0.0;
};

// ---------- 作业函数（在 15-1_图像基础.cpp 里实现） ----------

// 积分图：输出 (rows+1) x (cols+1)，第 0 行和第 0 列为 0，
// out(y+1, x+1) = sum(img[0..y, 0..x])。
ImageU64 integral_image(const ImageU8& img);

// 盒式均值滤波，窗口 (2r+1)x(2r+1)，必须用积分图实现 O(1) 每像素。
// 边界处窗口裁剪到图像内，只对图像内像素求平均（不补零）。r < 0 抛 std::invalid_argument。
ImageF box_mean(const ImageU8& img, int r);

// 零均值归一化互相关模板匹配（valid 模式），返回最高分的位置和分数。
// 多个位置同分时取光栅扫描顺序第一个。模板比图大时抛 std::invalid_argument。
// 窗口或模板方差为 0 时该位置分数为 0。
MatchResult ncc_best_match(const ImageU8& img, const ImageU8& tpl);

// 统计二值图（非 0 为前景）的连通域个数，connectivity 为 4 或 8，否则抛 std::invalid_argument。
// 要求不能递归（大图递归会栈溢出），用显式栈或队列。
int count_blobs(const ImageU8& binary, int connectivity);

// ---------- 已实现的工具：PGM 读写，方便在实验里看结果 ----------

inline void write_pgm(const std::string& path, const ImageU8& img) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("无法写入 " + path);
    f << "P5\n" << img.cols << " " << img.rows << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.data.data()), static_cast<std::streamsize>(img.data.size()));
}

inline ImageU8 read_pgm(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("无法读取 " + path);
    std::string magic;
    int w = 0, h = 0, maxv = 0;
    f >> magic >> w >> h >> maxv;
    if (magic != "P5" || maxv != 255) throw std::runtime_error("只支持 8 位 P5 格式");
    f.get();
    ImageU8 img(h, w);
    f.read(reinterpret_cast<char*>(img.data.data()), static_cast<std::streamsize>(img.data.size()));
    return img;
}

}  // namespace course
