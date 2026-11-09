// 参考实现 15-1。学员完成本章前不要查看。
#include "image.hpp"

#include <cmath>
#include <queue>
#include <stdexcept>
#include <utility>

namespace course {

ImageU64 integral_image(const ImageU8& img) {
    ImageU64 ii(img.rows + 1, img.cols + 1, 0);
    for (int y = 0; y < img.rows; ++y) {
        std::uint64_t row = 0;
        for (int x = 0; x < img.cols; ++x) {
            row += img(y, x);
            ii(y + 1, x + 1) = ii(y, x + 1) + row;
        }
    }
    return ii;
}

namespace {
std::uint64_t rect_sum(const ImageU64& ii, int y0, int x0, int y1, int x1) {
    // 半开区间 [y0, y1) x [x0, x1)
    return ii(y1, x1) - ii(y0, x1) - ii(y1, x0) + ii(y0, x0);
}
}  // namespace

ImageF box_mean(const ImageU8& img, int r) {
    if (r < 0) throw std::invalid_argument("r 不能为负");
    ImageU64 ii = integral_image(img);
    ImageF out(img.rows, img.cols, 0.0);
    for (int y = 0; y < img.rows; ++y) {
        int y0 = std::max(0, y - r), y1 = std::min(img.rows, y + r + 1);
        for (int x = 0; x < img.cols; ++x) {
            int x0 = std::max(0, x - r), x1 = std::min(img.cols, x + r + 1);
            double n = static_cast<double>((y1 - y0) * (x1 - x0));
            out(y, x) = static_cast<double>(rect_sum(ii, y0, x0, y1, x1)) / n;
        }
    }
    return out;
}

MatchResult ncc_best_match(const ImageU8& img, const ImageU8& tpl) {
    if (tpl.rows > img.rows || tpl.cols > img.cols || tpl.empty())
        throw std::invalid_argument("模板必须非空且不大于图像");
    const int h = tpl.rows, w = tpl.cols;
    const double n = static_cast<double>(h) * w;
    double tmean = 0;
    for (auto v : tpl.data) tmean += v;
    tmean /= n;
    double tvar = 0;
    for (auto v : tpl.data) tvar += (v - tmean) * (v - tmean);
    MatchResult best;
    best.score = -2.0;
    for (int y = 0; y + h <= img.rows; ++y) {
        for (int x = 0; x + w <= img.cols; ++x) {
            double wmean = 0;
            for (int dy = 0; dy < h; ++dy)
                for (int dx = 0; dx < w; ++dx) wmean += img(y + dy, x + dx);
            wmean /= n;
            double cov = 0, wvar = 0;
            for (int dy = 0; dy < h; ++dy)
                for (int dx = 0; dx < w; ++dx) {
                    double a = img(y + dy, x + dx) - wmean;
                    double b = tpl(dy, dx) - tmean;
                    cov += a * b;
                    wvar += a * a;
                }
            double s = (wvar < 1e-12 || tvar < 1e-12) ? 0.0 : cov / std::sqrt(wvar * tvar);
            if (s > best.score) best = {y, x, s};
        }
    }
    return best;
}

int count_blobs(const ImageU8& binary, int connectivity) {
    if (connectivity != 4 && connectivity != 8) throw std::invalid_argument("connectivity 只能是 4 或 8");
    std::vector<std::pair<int, int>> offs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    if (connectivity == 8) {
        offs.insert(offs.end(), {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}});
    }
    Image<std::uint8_t> seen(binary.rows, binary.cols, 0);
    int count = 0;
    std::queue<std::pair<int, int>> q;
    for (int y = 0; y < binary.rows; ++y) {
        for (int x = 0; x < binary.cols; ++x) {
            if (!binary(y, x) || seen(y, x)) continue;
            ++count;
            seen(y, x) = 1;
            q.push({y, x});
            while (!q.empty()) {
                auto [cy, cx] = q.front();
                q.pop();
                for (auto [dy, dx] : offs) {
                    int ny = cy + dy, nx = cx + dx;
                    if (ny < 0 || ny >= binary.rows || nx < 0 || nx >= binary.cols) continue;
                    if (binary(ny, nx) && !seen(ny, nx)) {
                        seen(ny, nx) = 1;
                        q.push({ny, nx});
                    }
                }
            }
        }
    }
    return count;
}

}  // namespace course
