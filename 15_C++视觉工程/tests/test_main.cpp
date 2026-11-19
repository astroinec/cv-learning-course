// 第 15 章 C++ 测试程序，由 tests/check_progress.py 编译、运行并解析输出。
//
// 输出约定（这个程序只输出 ASCII，中文说明由 check_progress.py 按 code 翻译，
// 这样在中文 Windows 的控制台和管道里都不会乱码）：
//   RUN|key                      开始运行一项测试
//   STAGE|key|stage              这一项测试进行到哪一步，程序崩溃时用来定位
//   PASS|key|code|k=v k=v ...    通过
//   FAIL|key|code|k=v k=v ...    没通过，code 说明是哪一条检查
//   TODO|key|todo|               作业函数还是初始的 TODO 状态
// 每行输出后立刻刷新：后面的测试即使让程序崩溃，前面几项的结果也已经送出去了。
//
// 用法：
//   test_ch15                 运行全部 4 项
//   test_ch15 box_mean        只运行一项（integral、box_mean、ncc、blobs）
//   test_ch15 timing          只做盒式均值的计时，输出每个半径的耗时，填实验记录时可以用
//
// 随机图直接取 std::mt19937 的输出（标准规定了它的数列），不用 std::uniform_int_distribution
// （各家标准库的实现不同），所以三种编译器下测试用的图完全相同。
#include "image.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <initializer_list>
#include <new>
#include <random>
#include <stdexcept>
#include <string>

using namespace course;

namespace {

struct Outcome {
    bool ok;
    std::string code;
    std::string detail;
};

const char* g_key = "";
const char* g_stage = "";
volatile double g_sink = 0.0;  // 把计时用的结果读出来存到这里，防止整段计算被优化掉

void put_line(const std::string& line) {
    std::fputs(line.c_str(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

void stage(const char* name) {
    g_stage = name;
    put_line(std::string("STAGE|") + g_key + "|" + name);
}

std::string one_line(std::string s) {
    for (char& ch : s)
        if (ch == '\n' || ch == '\r') ch = ' ';
    return s;
}

std::string num(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.10g", v);
    return buf;
}

std::string ms(double seconds) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.3f", seconds * 1e3);
    return buf;
}

template <typename T>
std::string kv(const char* key, T value) {
    return std::string(" ") + key + "=" + std::to_string(value);
}

std::string kvs(const char* key, const std::string& value) { return std::string(" ") + key + "=" + value; }

const char* exception_kind(const std::exception& e) {
    if (dynamic_cast<const std::out_of_range*>(&e)) return "out_of_range";
    if (dynamic_cast<const std::invalid_argument*>(&e)) return "invalid_argument";
    if (dynamic_cast<const std::length_error*>(&e)) return "length_error";
    if (dynamic_cast<const std::bad_alloc*>(&e)) return "bad_alloc";
    if (dynamic_cast<const std::logic_error*>(&e)) return "logic_error";
    if (dynamic_cast<const std::runtime_error*>(&e)) return "runtime_error";
    return "exception";
}

void run(const char* key, const std::function<Outcome()>& fn) {
    g_key = key;
    g_stage = "start";
    put_line(std::string("RUN|") + key);
    try {
        Outcome o = fn();
        put_line(std::string(o.ok ? "PASS|" : "FAIL|") + key + "|" + o.code + "|" + one_line(o.detail));
    } catch (const std::exception& e) {
        const std::string what = e.what();
        if (dynamic_cast<const std::logic_error*>(&e) && what.rfind("TODO", 0) == 0) {
            put_line(std::string("TODO|") + key + "|todo|");
        } else {
            // what 放在最后：里面可能有空格，也可能不是 ASCII
            put_line(std::string("FAIL|") + key + "|exception|" + one_line(kvs("kind", exception_kind(e)) + kvs("stage", g_stage) + kvs("what", what)));
        }
    } catch (...) {
        put_line(std::string("FAIL|") + key + "|exception|" + kvs("kind", "unknown") + kvs("stage", g_stage) + kvs("what", ""));
    }
}

// 返回 0：抛了 std::invalid_argument；1：没有抛异常；2：抛了别的异常（种类写进 kind）
template <typename F>
int throws_invalid_argument(F&& fn, std::string& kind) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return 0;
    } catch (const std::exception& e) {
        kind = exception_kind(e);
        return 2;
    } catch (...) {
        kind = "unknown";
        return 2;
    }
    return 1;
}

ImageU8 random_image(int rows, int cols, unsigned seed, unsigned max_value = 255u) {
    std::mt19937 rng(seed);
    ImageU8 img(rows, cols);
    for (auto& v : img.data) v = static_cast<std::uint8_t>(rng() % (max_value + 1u));
    return img;
}

ImageU8 crop(const ImageU8& img, int y0, int x0, int h, int w) {
    ImageU8 out(h, w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) out.at(y, x) = img.at(y0 + y, x0 + x);
    return out;
}

// ------------------------------------------------------------------ 积分图

// 逐个位置和暴力求和比较，第一处不一致写进 detail。
bool integral_matches(const ImageU8& img, const ImageU64& ii, std::string& detail) {
    for (int y = 0; y <= img.rows; ++y)
        for (int x = 0; x <= img.cols; ++x) {
            std::uint64_t s = 0;
            for (int yy = 0; yy < y; ++yy)
                for (int xx = 0; xx < x; ++xx) s += img.at(yy, xx);
            if (ii.at(y, x) != s) {
                detail = kv("in_rows", img.rows) + kv("in_cols", img.cols) + kv("y", y) + kv("x", x) + kv("got", ii.at(y, x)) + kv("expect", s);
                return false;
            }
        }
    return true;
}

Outcome test_integral() {
    stage("small");
    const int shapes[][2] = {{13, 17}, {1, 9}, {7, 1}};
    unsigned seed = 1;
    for (const auto& shape : shapes) {
        ImageU8 img = random_image(shape[0], shape[1], seed++);
        ImageU64 ii = integral_image(img);
        if (ii.rows != img.rows + 1 || ii.cols != img.cols + 1)
            return {false, "integral_size", kv("in_rows", img.rows) + kv("in_cols", img.cols) + kv("rows", ii.rows) + kv("cols", ii.cols)};
        std::string detail;
        if (!integral_matches(img, ii, detail)) return {false, "integral_value", detail};
    }

    stage("empty");
    const int empties[][2] = {{0, 0}, {0, 5}, {4, 0}};
    for (const auto& shape : empties) {
        ImageU8 img(shape[0], shape[1]);
        ImageU64 ii = integral_image(img);
        bool ok = ii.rows == img.rows + 1 && ii.cols == img.cols + 1;
        for (auto v : ii.data) ok = ok && v == 0;
        if (!ok) return {false, "integral_empty", kv("in_rows", img.rows) + kv("in_cols", img.cols) + kv("rows", ii.rows) + kv("cols", ii.cols)};
    }

    // 全白大图：位置 (y, x) 的值有闭式 255*y*x，不需要另一份实现就能核对。
    // 边长取到右下角的值超过 32 位无符号整数的上限（4294967295），32 位的累加类型在这里会露出来。
    stage("big");
    const int n = 4200;
    ImageU64 big = integral_image(ImageU8(n, n, 255));
    if (big.rows != n + 1 || big.cols != n + 1) return {false, "integral_size", kv("in_rows", n) + kv("in_cols", n) + kv("rows", big.rows) + kv("cols", big.cols)};
    const int probes[][2] = {{1, 1}, {1, n}, {n, 1}, {257, 257}, {2100, 2100}, {2902, 2902}, {3000, 4000}, {4105, 4105}, {n - 1, n}, {n, n - 1}, {n, n}};
    for (const auto& p : probes) {
        const std::uint64_t expect = 255ull * static_cast<std::uint64_t>(p[0]) * static_cast<std::uint64_t>(p[1]);
        const std::uint64_t got = big.at(p[0], p[1]);
        if (got != expect) return {false, "integral_big", kv("n", n) + kv("y", p[0]) + kv("x", p[1]) + kv("got", got) + kv("expect", expect)};
    }
    return {true, "ok", ""};
}

// ------------------------------------------------------------------ 盒式均值

double brute_mean(const ImageU8& img, int y, int x, int r) {
    double s = 0;
    int n = 0;
    for (int yy = std::max(0, y - r); yy <= std::min(img.rows - 1, y + r); ++yy)
        for (int xx = std::max(0, x - r); xx <= std::min(img.cols - 1, x + r); ++xx) {
            s += img.at(yy, xx);
            ++n;
        }
    return s / n;
}

bool box_matches(const ImageU8& img, int r, std::string& code, std::string& detail) {
    ImageF m = box_mean(img, r);
    if (m.rows != img.rows || m.cols != img.cols) {
        code = "box_size";
        detail = kv("in_rows", img.rows) + kv("in_cols", img.cols) + kv("r", r) + kv("rows", m.rows) + kv("cols", m.cols);
        return false;
    }
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) {
            const double expect = brute_mean(img, y, x, r);
            // 判断写成“不是在容差以内”，这样 NaN 也算不一致
            if (!(std::fabs(m.at(y, x) - expect) <= 1e-9)) {
                code = "box_value";
                detail = kv("in_rows", img.rows) + kv("in_cols", img.cols) + kv("r", r) + kv("y", y) + kv("x", x) + kvs("got", num(m.at(y, x))) + kvs("expect", num(expect));
                return false;
            }
        }
    return true;
}

double seconds_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// 计时器两次读数之间能分辨的最小间隔。大多数环境是 1 微秒以下；个别工具链的 steady_clock 很粗，要把余量放大。
double clock_tick() {
    double best = 1.0;
    auto prev = std::chrono::steady_clock::now();
    int changes = 0;
    for (long i = 0; i < 20000000L && changes < 8; ++i) {
        const auto cur = std::chrono::steady_clock::now();
        if (cur != prev) {
            best = std::min(best, seconds_between(prev, cur));
            prev = cur;
            ++changes;
        }
    }
    return best;
}

// 计时判据（宁可放过，不误判）：
//   同一张图，先各调用一次预热，再让 r=1 和 r=R 交替运行，各取最多 7 次里最快的一次；
//   只有 最快(R) > 4 * 最快(1) + 余量 才算“耗时随半径增长”，余量取 2 毫秒和 4 倍计时器分辨率里较大的一个。
//   R 依次取 10、30、200，哪一级超了就停（很慢的实现在前两级就被拦下，不会拖到 r=200）。O(1) 的实现这个比值在 1 左右。
const int kTimingSize = 1200;
const int kTimingRadii[] = {10, 30, 200};
const double kTimingRatio = 4.0;
const double kTimingSlack = 0.002;
const int kTimingMaxReps = 7;
const int kTimingMinReps = 3;
const double kTimingBudget = 1.0;  // 一级里计时累计超过这么多秒、并且已经测满最少次数，就不再重复

struct Rung {
    int r = 0;
    int reps = 0;
    double small_s = 0.0;
    double large_s = 0.0;
    double limit_s = 0.0;
    bool too_slow = false;
};

Rung time_rung(const ImageU8& img, int r_large, double slack) {
    using clock = std::chrono::steady_clock;
    Rung out;
    out.r = r_large;
    g_sink = g_sink + box_mean(img, 1).at(0, 0);  // 预热：第一次调用要向系统要内存，不计入
    g_sink = g_sink + box_mean(img, r_large).at(0, 0);
    double small = 1e30, large = 1e30;
    const auto begin = clock::now();
    for (int i = 0; i < kTimingMaxReps; ++i) {
        const auto t0 = clock::now();
        ImageF a = box_mean(img, 1);
        const auto t1 = clock::now();
        ImageF b = box_mean(img, r_large);
        const auto t2 = clock::now();
        g_sink = g_sink + a.at(img.rows / 2, img.cols / 2) + b.at(img.rows / 2, img.cols / 2);
        small = std::min(small, seconds_between(t0, t1));
        large = std::min(large, seconds_between(t1, t2));
        out.reps = i + 1;
        if (out.reps >= kTimingMinReps && seconds_between(begin, t2) > kTimingBudget) break;
    }
    out.small_s = small;
    out.large_s = large;
    out.limit_s = kTimingRatio * small + slack;
    out.too_slow = large > out.limit_s;
    return out;
}

std::string rung_detail(const Rung& g, double tick) {
    return kv("n", kTimingSize) + kv("r", g.r) + kvs("small_ms", ms(g.small_s)) + kvs("large_ms", ms(g.large_s)) + kvs("limit_ms", ms(g.limit_s)) +
           kvs("ratio", num(std::round(g.large_s / std::max(g.small_s, 1e-9) * 10.0) / 10.0)) + kv("reps", g.reps) + kvs("tick_ms", ms(tick));
}

Outcome test_box_mean() {
    std::string code, detail, kind;

    stage("small");
    ImageU8 img = random_image(20, 25, 2);
    for (int r : {0, 1, 3})
        if (!box_matches(img, r, code, detail)) return {false, code, detail};

    // 窗口 (2r+1) 比图像的一条边或两条边还长：窗口要在两侧同时裁剪
    stage("large_r");
    for (int r : {12, 22, 60})
        if (!box_matches(img, r, code, detail)) return {false, code == "box_value" ? "box_large_r" : code, detail};

    stage("thin");
    if (!box_matches(random_image(1, 11, 5), 2, code, detail)) return {false, code, detail};
    if (!box_matches(random_image(9, 1, 6), 3, code, detail)) return {false, code, detail};

    stage("negative_r");
    const int thrown = throws_invalid_argument([&] { box_mean(img, -1); }, kind);
    if (thrown == 1) return {false, "box_negative_r", ""};
    if (thrown == 2) return {false, "box_negative_r_type", kvs("kind", kind)};

    stage("timing");
    ImageU8 big = random_image(kTimingSize, kTimingSize, 3);
    const double tick = clock_tick();
    const double slack = std::max(kTimingSlack, 4.0 * tick);
    Rung last;
    for (int r : kTimingRadii) {
        // 大图大半径下抽几个点核对数值，防止只在小图上对
        ImageF m = box_mean(big, r);
        const int pts[][2] = {{0, 0}, {kTimingSize / 2, kTimingSize / 2}, {kTimingSize - 1, kTimingSize - 1}, {r, r - 1}, {17, kTimingSize - 3}};
        for (const auto& p : pts) {
            const double expect = brute_mean(big, p[0], p[1], r);
            if (!(std::fabs(m.at(p[0], p[1]) - expect) <= 1e-9))
                return {false, "box_value_big", kv("in_rows", kTimingSize) + kv("in_cols", kTimingSize) + kv("r", r) + kv("y", p[0]) + kv("x", p[1]) +
                                                    kvs("got", num(m.at(p[0], p[1]))) + kvs("expect", num(expect))};
        }
        last = time_rung(big, r, slack);
        if (last.too_slow) return {false, "box_slow", rung_detail(last, tick)};
    }
    return {true, "ok", rung_detail(last, tick)};
}

// ------------------------------------------------------------------ NCC 模板匹配

std::string match_detail(const MatchResult& m, int expect_y, int expect_x) {
    return kv("got_y", m.y) + kv("got_x", m.x) + kvs("score", num(m.score)) + kv("expect_y", expect_y) + kv("expect_x", expect_x);
}

Outcome test_ncc() {
    std::string kind;

    stage("exact");
    ImageU8 img = random_image(40, 50, 4);
    MatchResult m = ncc_best_match(img, crop(img, 17, 23, 9, 11));
    if (m.y != 17 || m.x != 23) return {false, "ncc_position", match_detail(m, 17, 23)};
    if (!(std::fabs(m.score - 1.0) <= 1e-9)) return {false, "ncc_score_exact", match_detail(m, 17, 23)};

    // 模板来源贴着右下角：这是最后一个合法位置
    stage("last_position");
    m = ncc_best_match(img, crop(img, 31, 39, 9, 11));
    if (m.y != 31 || m.x != 39 || !(std::fabs(m.score - 1.0) <= 1e-9)) return {false, "ncc_last_position", match_detail(m, 31, 39)};
    m = ncc_best_match(img, img);
    if (m.y != 0 || m.x != 0 || !(std::fabs(m.score - 1.0) <= 1e-9)) return {false, "ncc_same_size", match_detail(m, 0, 0)};

    // 亮度线性变化：模板 = 2 * 窗口 + 30（图的灰度不超过 100，不会饱和），原位置的分数仍应是 1
    stage("brightness");
    ImageU8 dim = random_image(36, 44, 7, 100u);
    ImageU8 tpl = crop(dim, 12, 30, 8, 10);
    for (auto& v : tpl.data) v = static_cast<std::uint8_t>(2 * v + 30);
    m = ncc_best_match(dim, tpl);
    if (m.y != 12 || m.x != 30) return {false, "ncc_brightness_position", match_detail(m, 12, 30)};
    if (!(std::fabs(m.score - 1.0) <= 1e-9)) return {false, "ncc_brightness_score", match_detail(m, 12, 30)};
    // 带取整的亮度变化（v / 2 + 40）：分数不再正好是 1，但位置不应该变
    ImageU8 darker = img;
    for (auto& v : darker.data) v = static_cast<std::uint8_t>(v / 2 + 40);
    m = ncc_best_match(img, crop(darker, 17, 23, 9, 11));
    if (m.y != 17 || m.x != 23) return {false, "ncc_brightness_position", match_detail(m, 17, 23)};

    // 平坦的窗口或模板：分数是 0，不能是 NaN；全部位置同分时取第一个
    stage("flat");
    m = ncc_best_match(ImageU8(10, 10, 5), ImageU8(3, 3, 5));
    if (!(m.score == 0.0)) return {false, "ncc_flat_score", match_detail(m, 0, 0)};
    if (m.y != 0 || m.x != 0) return {false, "ncc_flat_position", match_detail(m, 0, 0)};
    m = ncc_best_match(img, ImageU8(5, 6, 90));
    if (!(m.score == 0.0) || m.y != 0 || m.x != 0) return {false, "ncc_flat_template", match_detail(m, 0, 0)};

    // 同一个小块出现在两处：A = (3, 25) 和 B = (15, 4)，两处的窗口和模板逐像素相同，分数必须完全相等。
    // 光栅扫描顺序（先比 y，再比 x）的第一个是 A。
    stage("tie");
    ImageU8 twice = random_image(30, 40, 8);
    ImageU8 patch = crop(twice, 3, 25, 6, 7);
    for (int y = 0; y < patch.rows; ++y)
        for (int x = 0; x < patch.cols; ++x) twice.at(15 + y, 4 + x) = patch.at(y, x);
    m = ncc_best_match(twice, patch);
    if (m.y != 3 || m.x != 25 || !(std::fabs(m.score - 1.0) <= 1e-9)) return {false, "ncc_tie_first", match_detail(m, 3, 25) + kv("other_y", 15) + kv("other_x", 4)};

    // 模板的行数或列数任一个比图大都要抛 std::invalid_argument
    stage("too_big");
    const int shapes[][2] = {{img.rows + 1, img.cols + 1}, {img.rows + 1, 3}, {3, img.cols + 1}};
    for (const auto& shape : shapes) {
        const int thrown = throws_invalid_argument([&] { ncc_best_match(img, ImageU8(shape[0], shape[1], 1)); }, kind);
        const std::string where = kv("tpl_rows", shape[0]) + kv("tpl_cols", shape[1]) + kv("img_rows", img.rows) + kv("img_cols", img.cols);
        if (thrown == 1) return {false, "ncc_too_big", where};
        if (thrown == 2) return {false, "ncc_too_big_type", where + kvs("kind", kind)};
    }
    return {true, "ok", ""};
}

// ------------------------------------------------------------------ 连通域计数

Outcome blobs_mismatch(const char* code, int connectivity, int got, int expect) {
    return {false, code, kv("connectivity", connectivity) + kv("got", got) + kv("expect", expect)};
}

Outcome test_blobs() {
    std::string kind;

    // 5 组前景，灰度各不相同（非 0 都算前景）：主对角线上的一对、反对角线上的一对、一条横线、一个 2x2 方块、一条竖线
    stage("small");
    ImageU8 b(9, 12, 0);
    b.at(1, 1) = b.at(2, 2) = 255;
    b.at(1, 5) = b.at(2, 4) = 128;
    for (int x = 8; x <= 10; ++x) b.at(1, x) = 1;
    b.at(5, 3) = b.at(5, 4) = b.at(6, 3) = b.at(6, 4) = 7;
    for (int y = 5; y <= 7; ++y) b.at(y, 8) = 200;
    int got = count_blobs(b, 8);
    if (got != 5) return blobs_mismatch("blobs_small", 8, got, 5);
    got = count_blobs(b, 4);
    if (got != 7) return blobs_mismatch("blobs_small", 4, got, 7);

    // 一行的最后一列和下一行的第一列在内存里相邻，但不是邻居
    stage("row_wrap");
    ImageU8 w(6, 7, 0);
    w.at(0, 6) = w.at(1, 0) = w.at(4, 6) = w.at(5, 0) = 1;
    for (int c : {4, 8}) {
        got = count_blobs(w, c);
        if (got != 4) return blobs_mismatch("blobs_row_wrap", c, got, 4);
    }

    stage("corners");
    ImageU8 corners(3, 3, 0);
    corners.at(0, 0) = corners.at(0, 2) = corners.at(2, 0) = corners.at(2, 2) = 9;
    for (int c : {4, 8}) {
        got = count_blobs(corners, c);
        if (got != 4) return blobs_mismatch("blobs_corners", c, got, 4);
    }

    stage("trivial");
    for (int c : {4, 8}) {
        got = count_blobs(ImageU8(5, 6, 0), c);
        if (got != 0) return blobs_mismatch("blobs_background", c, got, 0);
        got = count_blobs(ImageU8(), c);
        if (got != 0) return blobs_mismatch("blobs_background", c, got, 0);
        got = count_blobs(ImageU8(5, 6, 3), c);
        if (got != 1) return blobs_mismatch("blobs_full", c, got, 1);
    }

    // 棋盘格：4 连通时每个前景像素自成一个，8 连通时全部连成一个
    stage("checker");
    ImageU8 checker(60, 60, 0);
    for (int y = 0; y < 60; ++y)
        for (int x = 0; x < 60; ++x)
            if ((y + x) % 2 == 0) checker.at(y, x) = 1;
    got = count_blobs(checker, 4);
    if (got != 1800) return blobs_mismatch("blobs_checker", 4, got, 1800);
    got = count_blobs(checker, 8);
    if (got != 1) return blobs_mismatch("blobs_checker", 8, got, 1);

    stage("bad_connectivity");
    for (int c : {6, 0}) {
        const int thrown = throws_invalid_argument([&] { count_blobs(b, c); }, kind);
        if (thrown == 1) return {false, "blobs_bad_connectivity", kv("connectivity", c)};
        if (thrown == 2) return {false, "blobs_bad_connectivity_type", kv("connectivity", c) + kvs("kind", kind)};
    }

    // 3000x3000 的蛇形大区域放在最后：递归实现会在这里栈溢出，整个程序被系统终止
    stage("snake");
    const int n = 3000;
    ImageU8 snake(n, n, 0);
    for (int y = 0; y < n; y += 2)
        for (int x = 0; x < n; ++x) snake.at(y, x) = 1;
    for (int y = 1; y < n; y += 4) snake.at(y, n - 1) = 1;
    for (int y = 3; y < n; y += 4) snake.at(y, 0) = 1;
    got = count_blobs(snake, 4);
    if (got != 1) return blobs_mismatch("blobs_snake", 4, got, 1);
    return {true, "ok", ""};
}

int print_timing() {
    g_key = "box_mean";
    ImageU8 big = random_image(kTimingSize, kTimingSize, 3);
    const double tick = clock_tick();
    const double slack = std::max(kTimingSlack, 4.0 * tick);
    try {
        for (int r : kTimingRadii) {
            const Rung g = time_rung(big, r, slack);
            put_line(std::string("TIMING|box_mean|") + (g.too_slow ? "slow" : "ok") + "|" + rung_detail(g, tick));
            if (g.too_slow) break;  // 和验收一样，超了就停：很慢的实现在更大的半径上要跑很久
        }
    } catch (const std::exception& e) {
        put_line(std::string("TIMING|box_mean|exception|") + kvs("kind", exception_kind(e)));
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string only = argc > 1 ? argv[1] : "";
    if (only == "timing") return print_timing();
    struct Case {
        const char* key;
        Outcome (*fn)();
    };
    const Case cases[] = {{"integral", test_integral}, {"box_mean", test_box_mean}, {"ncc", test_ncc}, {"blobs", test_blobs}};
    bool any = false;
    for (const auto& c : cases)
        if (only.empty() || only == c.key) {
            run(c.key, c.fn);
            any = true;
        }
    if (!any) {
        put_line("ERROR|unknown test name, use one of: integral box_mean ncc blobs timing");
        return 2;
    }
    return 0;
}
