"""第 15 章配图与实测数字。

用法（在 solutions 分支的工作目录运行）：
    COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch15.py

输出：<主分支>/15_C++视觉工程/figures/*.svg 和 实测数字.json。
讲义里引用的数字全部来自这个脚本，改讲义数字之前先重跑。

这一章的作业是 C++，所以脚本分两部分：
1. 用 g++ 把参考实现和下面内嵌的基准程序编译到 /tmp/ch15_build 下运行，把耗时、递归深度、
   溢出阈值读回来。
2. 用 numpy 和 Python 版 OpenCV 复现内存布局、整数溢出、边界和取整这些和语言无关的概念。

耗时每次运行都会波动，所以含耗时的分组（PINNED_KEYS 里列出的那些）都用 S.pinned 钉住：
默认沿用 实测数字.json 里上次记下的值，不重新计时，重复运行时 figures/ 下所有文件逐字节不变。
要重测时在机器空闲的时候运行：
    CH15_RETIME=1 COURSE_MAIN_DIR=../cv-learning-course python solutions/figures/ch15.py
重测之后讲义里的耗时、倍数和 SECOND_RUN 里抄录的对照都要按新数字核对一遍。
图里和耗时有关的内容只用写进 JSON 的值来画，图的标题里依赖耗时的结论由 claim() 对着这些值检查。
深度、阈值、误差、测试结果这些不是耗时的数字每次运行都重新计算。

故意触发的栈溢出都由程序自己的信号处理函数接住后正常退出，不会在系统里留下崩溃报告。
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _style import plt  # noqa: E402

CH = "15_C++视觉工程"
REF = S.SOLUTIONS / CH / "15-1_图像基础.cpp"
INC = S.MAIN / CH / "cpp"
TEST_MAIN = S.MAIN / CH / "tests" / "test_main.cpp"
BUILD = Path("/tmp/ch15_build")
CXX = "g++"
NUM: dict = {}
RETIME = "CH15_RETIME"
PINNED_KEYS: list = []  # 含耗时、被钉住的分组，按写入顺序


def sig(x: float, n: int = 3) -> float:
    return float(f"{float(x):.{n}g}")


def two(x: float) -> str:
    """图上标注耗时和倍数用的写法：两位有效数字，和讲义一致。"""
    s = sig(x, 2)
    return f"{s:.0f}" if s >= 10 else f"{s:.1f}" if s >= 1 else f"{s:.2f}"


def pin(key: str, measure) -> dict:
    """含耗时的一组：默认沿用 实测数字.json 里上次的值，设置 CH15_RETIME=1 才调用 measure 重新计时。

    measure 不带参数，返回这一组要写进 JSON 的完整内容。之后画图、算倍数都只用返回的这份内容，
    这样图、JSON 和讲义里的数是同一批。
    """
    PINNED_KEYS.append(key)
    NUM[key] = S.pinned(CH, key, measure, RETIME)
    return NUM[key]


def claim(ok: bool, text: str) -> None:
    """图的标题里依赖耗时的结论，对着写进 JSON 的值检查一遍。重测之后不成立就停下来，提醒改标题和讲义。"""
    if not ok:
        raise RuntimeError(f"图标题里的结论“{text}”和 实测数字.json 里的耗时对不上，要改标题和讲义里对应的说法")


# 同一个脚本（钉住之前的版本）在这台机器很忙时跑出的另一次结果，用来判断讲义里依赖耗时的结论稳不稳。
# 这些数是从那次运行的输出里抄录的，不是本脚本算出来的，所以不随重跑变化。
SECOND_RUN = {
    "说明": "同一个脚本在同一台机器很忙时（同时有十几个别的任务在跑）跑出的另一次结果，2026-11-18。"
          "只摘录讲义里用来判断结论稳不稳的数，分组和条目的名字与上下各组相同；这些是抄录的记录，不是这次运行算出来的。和钉住的那次相比，多数耗时大一到两成，"
          "同一次运行里算出的倍数大多只差百分之几，个别差到一倍以上。设置 CH15_RETIME=1 重测之后，这一组和讲义里引用它的地方要一起更新",
    "同一项耗时是钉住那次的几倍": {"比较的耗时项数": 173, "中位数": 1.14, "最小": 0.94, "最大": 2.56, "在 1.05 到 1.25 倍之间的项数": 157, "超过 1.3 倍的项数": 11},
    "计时条件": {"同一段代码连续计时 30 次的耗时（ms，盒式均值 2000x2000、r=7、-O2）": {"最快": 3.85, "中位数": 5.02, "最慢": 10.1}},
    "遍历顺序": {
        "统计直方图，-O2": {"1000": {"列优先慢的倍数": 1.2}, "1024": {"列优先慢的倍数": 2.3}, "2000": {"列优先慢的倍数": 1.9}, "2048": {"列优先慢的倍数": 3.5},
                       "4000": {"列优先慢的倍数": 2.6}, "4096": {"列优先慢的倍数": 10.2}, "8000": {"列优先慢的倍数": 4.8}, "8192": {"列优先慢的倍数": 10.3}},
        "统计直方图，-O0": {"1000": {"列优先慢的倍数": 1.1}, "1024": {"列优先慢的倍数": 1.1}, "4000": {"列优先慢的倍数": 1.5}, "4096": {"列优先慢的倍数": 3.2}},
        "求和，-O2": {"1000": {"列优先慢的倍数": 5.5}, "1024": {"列优先慢的倍数": 12.2}, "2000": {"列优先慢的倍数": 5.6}, "2048": {"列优先慢的倍数": 20.7},
                   "4000": {"列优先慢的倍数": 11.7}, "4096": {"列优先慢的倍数": 66.8}, "8000": {"列优先慢的倍数": 20.3}, "8192": {"列优先慢的倍数": 66.2}},
    },
    "盒式均值的复杂度": {"积分图在各半径下的最慢与最快之比": 1.1, "r=30 时朴素写法比积分图慢的倍数": 730, "r=30 时可分离写法比积分图慢的倍数": 39},
    "NCC 耗时": {"分母用积分图相对直接计算快的倍数": [3.9, 6.0, 7.5, 7.7, 8.0], "OpenCV 相对直接计算快的倍数": [4, 20, 50, 81, 166], "t 从最小到最大，直接计算的耗时增长倍数": 39},
    "优化级别": {"-O0 比 -O2 慢的倍数的范围": [4.4, 35.4], "开 Sanitizer 比 -O2 慢的倍数": [19.2, 18.4, 3.4, 2.7], "-O3 与 -O2 的耗时之比": [0.96, 1.01, 0.99, 1.07]},
    "访问方式": {"-O2 下 at() 比 operator() 慢": "12%", "-O2 下 operator() 比行指针慢的倍数": 5.5, "关掉向量化后 operator() 比行指针慢的倍数": 1.5},
    "按值传参": {"2000x2000（4 MB）": {"按值，-O2": 0.0666, "按值，-O0": 8.65}, "4000x4000（16 MB）": {"按值，-O2": 0.287, "按值，-O0": 35.1}},
    "循环里分配内存": {"中值：不 reserve 比提到循环外慢的倍数": 1.63, "中值：循环里 reserve 比提到循环外慢": "11%", "push_back：不 reserve 慢的倍数": 1.6},
    "和 OpenCV 对照": {"积分图 2000x2000（cv2.integral，输出 64 位浮点）": {"参考实现慢的倍数": 1.2}, "盒式均值 2000x2000，r=7（cv2.boxFilter，输出 64 位浮点）": {"参考实现慢的倍数": 1.7},
                   "NCC 320x240，模板 24x24（cv2.matchTemplate）": {"参考实现慢的倍数": 50.0}, "连通域 2000x2000（cv2.connectedComponents，同时输出标签图）": {"参考实现慢的倍数": 4.8}},
    "Python 和 C++ 的积分图": {"纯 Python 双重循环": 39, "numpy 两次 cumsum（uint64）": 5.7, "C++ 参考实现": 0.36, "纯 Python 比 C++ 慢的倍数": 107},
    "测试的计时检查": {
        "积分图 O(1)（参考实现）": {"r=1 的耗时": 8.36, "r=30 的耗时": 4.29, "r=30 是 r=1 的几倍": 0.5, "判据允许的 r=30 耗时上限": 83.4, "5 次里按判据算通过的次数": 5},
        "可分离 O(r)": {"r=1 的耗时": 14.7, "r=30 的耗时": 93.1, "r=30 是 r=1 的几倍": 6.3, "判据允许的 r=30 耗时上限": 109.0, "5 次里按判据算通过的次数": 5},
        "朴素 O(r^2)": {"r=1 的耗时": 19.9, "r=30 的耗时": 2840.0, "r=30 是 r=1 的几倍": 142.4, "判据允许的 r=30 耗时上限": 130.0, "3 次里按判据算通过的次数": 0},
    },
    "测试能抓住什么": {"盒式均值写成可分离的 O(r)": "四项全部 PASS", "盒式均值写成 O(r^2)": "盒式均值 FAIL：r 增大时耗时明显增长，没有做到 O(1) 每像素",
                  "可分离的 O(r) 实现再连跑 5 次，“盒式均值”一项通过的次数": 5},
}


# ---------------------------------------------------------------- 内嵌的 C++ 基准程序
BENCH_CPP = r'''
// 第 15 章基准程序：由 ch15.py 写到 /tmp/ch15_build 后编译。每个模式输出一行 JSON。
#include "image.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace course;
using Clock = std::chrono::steady_clock;
#define NOINLINE __attribute__((noinline))

static volatile std::uint64_t sink;

struct Stat {
    double best = 1e30;
    std::vector<double> all;
};

template <typename F>
Stat measure(F&& f, double budget_ms = 100.0, int min_reps = 3, int max_reps = 25) {
    Stat s;
    double total = 0;
    for (int i = 0; i < max_reps && (i < min_reps || total < budget_ms); ++i) {
        auto t0 = Clock::now();
        f();
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        s.all.push_back(ms);
        s.best = std::min(s.best, ms);
        total += ms;
    }
    return s;
}

template <typename F>
double best_ms(F&& f, double budget_ms = 100.0, int min_reps = 3) {
    return measure(f, budget_ms, min_reps).best;
}

ImageU8 random_image(int rows, int cols, unsigned seed) {
    std::mt19937 rng(seed);
    ImageU8 img(rows, cols);
    for (auto& v : img.data) v = static_cast<std::uint8_t>(rng() & 255u);
    return img;
}

// ---- 遍历顺序
NOINLINE std::uint64_t sum_row(const ImageU8& img) {
    std::uint64_t s = 0;
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) s += img(y, x);
    return s;
}
NOINLINE std::uint64_t sum_col(const ImageU8& img) {
    std::uint64_t s = 0;
    for (int x = 0; x < img.cols; ++x)
        for (int y = 0; y < img.rows; ++y) s += img(y, x);
    return s;
}
NOINLINE void hist_row(const ImageU8& img, std::uint32_t* hist) {
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) ++hist[img(y, x)];
}
NOINLINE void hist_col(const ImageU8& img, std::uint32_t* hist) {
    for (int x = 0; x < img.cols; ++x)
        for (int y = 0; y < img.rows; ++y) ++hist[img(y, x)];
}

// ---- 3x3 均值的四种访问方式
NOINLINE void mean3_at(const ImageU8& img, ImageU8& out) {
    for (int y = 1; y < img.rows - 1; ++y)
        for (int x = 1; x < img.cols - 1; ++x) {
            int s = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) s += img.at(y + dy, x + dx);
            out.at(y, x) = static_cast<std::uint8_t>(s / 9);
        }
}
NOINLINE void mean3_op(const ImageU8& img, ImageU8& out) {
    for (int y = 1; y < img.rows - 1; ++y)
        for (int x = 1; x < img.cols - 1; ++x) {
            int s = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) s += img(y + dy, x + dx);
            out(y, x) = static_cast<std::uint8_t>(s / 9);
        }
}
NOINLINE void mean3_local(const ImageU8& img, ImageU8& out) {
    const int H = img.rows, W = img.cols;
    const std::uint8_t* d = img.data.data();
    std::uint8_t* o = out.data.data();
    for (int y = 1; y < H - 1; ++y)
        for (int x = 1; x < W - 1; ++x) {
            int s = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) s += d[static_cast<size_t>(y + dy) * W + (x + dx)];
            o[static_cast<size_t>(y) * W + x] = static_cast<std::uint8_t>(s / 9);
        }
}
NOINLINE void mean3_ptr(const ImageU8& img, ImageU8& out) {
    const int H = img.rows, W = img.cols;
    for (int y = 1; y < H - 1; ++y) {
        const std::uint8_t* a = &img.data[static_cast<size_t>(y - 1) * W];
        const std::uint8_t* b = a + W;
        const std::uint8_t* c = b + W;
        std::uint8_t* o = &out.data[static_cast<size_t>(y) * W];
        for (int x = 1; x < W - 1; ++x) {
            int s = a[x - 1] + a[x] + a[x + 1] + b[x - 1] + b[x] + b[x + 1] + c[x - 1] + c[x] + c[x + 1];
            o[x] = static_cast<std::uint8_t>(s / 9);
        }
    }
}

// ---- 按值传参和按引用传参
NOINLINE void pass_value(ImageU8 img) { sink = img.data[5]; }
NOINLINE void pass_ref(const ImageU8& img) { sink = img.data[5]; }

// ---- 盒式均值的另外两种复杂度（和参考实现的边界约定相同：窗口裁剪到图像内）
ImageF box_naive(const ImageU8& img, int r) {
    ImageF out(img.rows, img.cols, 0.0);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) {
            std::uint64_t s = 0;
            int n = 0;
            for (int yy = std::max(0, y - r); yy <= std::min(img.rows - 1, y + r); ++yy)
                for (int xx = std::max(0, x - r); xx <= std::min(img.cols - 1, x + r); ++xx) {
                    s += img(yy, xx);
                    ++n;
                }
            out(y, x) = static_cast<double>(s) / n;
        }
    return out;
}
ImageF box_separable(const ImageU8& img, int r) {
    Image<std::uint32_t> tmp(img.rows, img.cols, 0);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) {
            std::uint32_t s = 0;
            for (int xx = std::max(0, x - r); xx <= std::min(img.cols - 1, x + r); ++xx) s += img(y, xx);
            tmp(y, x) = s;
        }
    ImageF out(img.rows, img.cols, 0.0);
    for (int y = 0; y < img.rows; ++y) {
        int y0 = std::max(0, y - r), y1 = std::min(img.rows - 1, y + r);
        for (int x = 0; x < img.cols; ++x) {
            std::uint64_t s = 0;
            for (int yy = y0; yy <= y1; ++yy) s += tmp(yy, x);
            int n = (y1 - y0 + 1) * (std::min(img.cols - 1, x + r) - std::max(0, x - r) + 1);
            out(y, x) = static_cast<double>(s) / n;
        }
    }
    return out;
}

// ---- NCC：窗口的和与平方和用积分图取，分子仍然逐像素算
MatchResult ncc_integral(const ImageU8& img, const ImageU8& tpl) {
    const int h = tpl.rows, w = tpl.cols;
    const double n = static_cast<double>(h) * w;
    ImageU64 s1(img.rows + 1, img.cols + 1, 0), s2(img.rows + 1, img.cols + 1, 0);
    for (int y = 0; y < img.rows; ++y) {
        std::uint64_t r1 = 0, r2 = 0;
        for (int x = 0; x < img.cols; ++x) {
            std::uint64_t v = img(y, x);
            r1 += v;
            r2 += v * v;
            s1(y + 1, x + 1) = s1(y, x + 1) + r1;
            s2(y + 1, x + 1) = s2(y, x + 1) + r2;
        }
    }
    double tsum = 0;
    for (auto v : tpl.data) tsum += v;
    const double tmean = tsum / n;
    double tvar = 0;
    for (auto v : tpl.data) tvar += (v - tmean) * (v - tmean);
    MatchResult best;
    best.score = -2.0;
    for (int y = 0; y + h <= img.rows; ++y)
        for (int x = 0; x + w <= img.cols; ++x) {
            const double ws = static_cast<double>(s1(y + h, x + w) - s1(y, x + w) - s1(y + h, x) + s1(y, x));
            const double wq = static_cast<double>(s2(y + h, x + w) - s2(y, x + w) - s2(y + h, x) + s2(y, x));
            std::uint64_t dot = 0;
            for (int dy = 0; dy < h; ++dy)
                for (int dx = 0; dx < w; ++dx) dot += static_cast<std::uint32_t>(img(y + dy, x + dx)) * tpl(dy, dx);
            const double wvar = wq - ws * ws / n;
            const double cov = static_cast<double>(dot) - ws * tmean;
            const double s = (wvar < 1e-9 || tvar < 1e-12) ? 0.0 : cov / std::sqrt(wvar * tvar);
            if (s > best.score) best = {y, x, s};
        }
    return best;
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "traverse") {
        std::printf("{");
        for (int i = 2; i < argc; ++i) {
            const int n = std::atoi(argv[i]);
            ImageU8 img = random_image(n, n, 1);
            std::vector<std::uint32_t> hist(256);
            const double px = static_cast<double>(n) * n;
            double a = best_ms([&] { sink = sum_row(img); });
            double b = best_ms([&] { sink = sum_col(img); });
            double c = best_ms([&] { std::fill(hist.begin(), hist.end(), 0u); hist_row(img, hist.data()); sink = hist[7]; });
            double d = best_ms([&] { std::fill(hist.begin(), hist.end(), 0u); hist_col(img, hist.data()); sink = hist[7]; });
            std::printf("%s\"%d\": {\"sum_row\": %.4f, \"sum_col\": %.4f, \"hist_row\": %.4f, \"hist_col\": %.4f}", i > 2 ? ", " : "", n,
                        a * 1e6 / px, b * 1e6 / px, c * 1e6 / px, d * 1e6 / px);
        }
        std::printf("}\n");
    } else if (mode == "access") {
        const int n = 2000;
        ImageU8 img = random_image(n, n, 2), out(n, n);
        double a = best_ms([&] { mean3_at(img, out); sink = out(5, 5); });
        double b = best_ms([&] { mean3_op(img, out); sink = out(5, 5); });
        double c = best_ms([&] { mean3_local(img, out); sink = out(5, 5); });
        double d = best_ms([&] { mean3_ptr(img, out); sink = out(5, 5); });
        std::printf("{\"at\": %.4f, \"op\": %.4f, \"local\": %.4f, \"ptr\": %.4f}\n", a, b, c, d);
    } else if (mode == "alloc") {
        const int n = 600;
        ImageU8 img = random_image(n, n, 3), out(n, n);
        double a = best_ms([&] {
            for (int y = 2; y < n - 2; ++y)
                for (int x = 2; x < n - 2; ++x) {
                    std::vector<std::uint8_t> w;
                    for (int dy = -2; dy <= 2; ++dy)
                        for (int dx = -2; dx <= 2; ++dx) w.push_back(img(y + dy, x + dx));
                    std::nth_element(w.begin(), w.begin() + 12, w.end());
                    out(y, x) = w[12];
                }
            sink = out(5, 5);
        });
        double b = best_ms([&] {
            for (int y = 2; y < n - 2; ++y)
                for (int x = 2; x < n - 2; ++x) {
                    std::vector<std::uint8_t> w;
                    w.reserve(25);
                    for (int dy = -2; dy <= 2; ++dy)
                        for (int dx = -2; dx <= 2; ++dx) w.push_back(img(y + dy, x + dx));
                    std::nth_element(w.begin(), w.begin() + 12, w.end());
                    out(y, x) = w[12];
                }
            sink = out(5, 5);
        });
        double c = best_ms([&] {
            std::vector<std::uint8_t> w;
            w.reserve(25);
            for (int y = 2; y < n - 2; ++y)
                for (int x = 2; x < n - 2; ++x) {
                    w.clear();
                    for (int dy = -2; dy <= 2; ++dy)
                        for (int dx = -2; dx <= 2; ++dx) w.push_back(img(y + dy, x + dx));
                    std::nth_element(w.begin(), w.begin() + 12, w.end());
                    out(y, x) = w[12];
                }
            sink = out(5, 5);
        });
        const int m = 4000000;
        double d = best_ms([&] { std::vector<std::pair<int, int>> v; for (int i = 0; i < m; ++i) v.push_back({i, i}); sink = v.size(); });
        double e = best_ms([&] { std::vector<std::pair<int, int>> v; v.reserve(m); for (int i = 0; i < m; ++i) v.push_back({i, i}); sink = v.size(); });
        std::printf("{\"median_in_loop\": %.3f, \"median_in_loop_reserve\": %.3f, \"median_hoisted\": %.3f, \"push_no_reserve\": %.3f, \"push_reserve\": %.3f}\n", a, b, c, d, e);
    } else if (mode == "copy") {
        std::printf("{");
        bool first = true;
        for (int n : {2000, 4000}) {
            ImageU8 img = random_image(n, n, 4);
            double a = best_ms([&] { pass_value(img); }, 60.0, 10);
            double b = best_ms([&] { pass_ref(img); }, 1.0, 10);
            std::printf("%s\"%d\": {\"value\": %.5f, \"ref\": %.7f}", first ? "" : ", ", n, a, b);
            first = false;
        }
        std::printf("}\n");
    } else if (mode == "funcs") {
        const int reps = argc > 2 ? std::atoi(argv[2]) : 3;
        ImageU8 a = random_image(2000, 2000, 3);
        double t_int = best_ms([&] { sink = integral_image(a)(2000, 2000); }, 60.0, reps);
        double t_box = best_ms([&] { sink = static_cast<std::uint64_t>(box_mean(a, 7)(1000, 1000)); }, 60.0, reps);
        ImageU8 b = random_image(240, 320, 4);
        ImageU8 tpl(24, 24);
        for (int y = 0; y < 24; ++y)
            for (int x = 0; x < 24; ++x) tpl(y, x) = b(100 + y, 150 + x);
        double t_ncc = best_ms([&] { sink = static_cast<std::uint64_t>(ncc_best_match(b, tpl).x); }, 60.0, reps);
        ImageU8 blobs = random_image(2000, 2000, 5);
        for (auto& v : blobs.data) v = v < 100 ? 1 : 0;
        int count = 0;
        double t_cc = best_ms([&] { count = count_blobs(blobs, 8); sink = count; }, 60.0, reps);
        std::printf("{\"integral\": %.4f, \"box_mean\": %.4f, \"ncc\": %.4f, \"blobs\": %.4f, \"blob_count\": %d}\n", t_int, t_box, t_ncc, t_cc, count);
    } else if (mode == "box") {
        ImageU8 a = random_image(1000, 1000, 3);
        std::printf("{\"r\": {");
        bool first = true;
        for (int r : {1, 2, 3, 5, 10, 20, 30, 50, 100}) {
            double ti = best_ms([&] { sink = static_cast<std::uint64_t>(box_mean(a, r)(500, 500)); });
            double ts = best_ms([&] { sink = static_cast<std::uint64_t>(box_separable(a, r)(500, 500)); }, 60.0, 2);
            double tn = r <= 30 ? best_ms([&] { sink = static_cast<std::uint64_t>(box_naive(a, r)(500, 500)); }, 60.0, 2) : -1.0;
            std::printf("%s\"%d\": {\"integral\": %.4f, \"separable\": %.4f, \"naive\": %.4f}", first ? "" : ", ", r, ti, ts, tn);
            first = false;
        }
        std::printf("}}\n");
    } else if (mode == "ncc") {
        ImageU8 b = random_image(240, 320, 4);
        std::printf("{");
        bool first = true;
        for (int t : {8, 16, 24, 32, 48}) {
            ImageU8 tpl(t, t);
            for (int y = 0; y < t; ++y)
                for (int x = 0; x < t; ++x) tpl(y, x) = b(100 + y, 150 + x);
            MatchResult m1, m2;
            double t1 = best_ms([&] { m1 = ncc_best_match(b, tpl); sink = m1.x; }, 60.0, 2);
            double t2 = best_ms([&] { m2 = ncc_integral(b, tpl); sink = m2.x; }, 60.0, 2);
            std::printf("%s\"%d\": {\"direct\": %.4f, \"integral\": %.4f, \"same_pos\": %s, \"y\": %d, \"x\": %d, \"score_diff\": %.3g}", first ? "" : ", ", t, t1, t2,
                        (m1.y == m2.y && m1.x == m2.x) ? "true" : "false", m1.y, m1.x, std::fabs(m1.score - m2.score));
            first = false;
        }
        std::printf("}\n");
    } else if (mode == "ncc_check") {  // 不计时：两种实现各算一次，只比较结果
        ImageU8 b = random_image(240, 320, 4);
        std::printf("{");
        bool first = true;
        for (int t : {8, 16, 24, 32, 48}) {
            ImageU8 tpl(t, t);
            for (int y = 0; y < t; ++y)
                for (int x = 0; x < t; ++x) tpl(y, x) = b(100 + y, 150 + x);
            const MatchResult m1 = ncc_best_match(b, tpl), m2 = ncc_integral(b, tpl);
            std::printf("%s\"%d\": {\"same_pos\": %s, \"y\": %d, \"x\": %d, \"score_diff\": %.3g}", first ? "" : ", ", t,
                        (m1.y == m2.y && m1.x == m2.x) ? "true" : "false", m1.y, m1.x, std::fabs(m1.score - m2.score));
            first = false;
        }
        std::printf("}\n");
    } else if (mode == "blob_count") {  // 不计时：只数连通域个数
        ImageU8 blobs = random_image(2000, 2000, 5);
        for (auto& v : blobs.data) v = v < 100 ? 1 : 0;
        std::printf("{\"blob_count\": %d}\n", count_blobs(blobs, 8));
    } else if (mode == "noise") {
        ImageU8 a = random_image(2000, 2000, 3);
        Stat s = measure([&] { sink = static_cast<std::uint64_t>(box_mean(a, 7)(1000, 1000)); }, 0.0, 30, 30);
        std::sort(s.all.begin(), s.all.end());
        std::printf("{\"min\": %.4f, \"median\": %.4f, \"max\": %.4f, \"reps\": %zu}\n", s.all.front(), s.all[s.all.size() / 2], s.all.back(), s.all.size());
    }
    return 0;
}
'''

REC_CPP = r'''
// 递归泛洪填充的深度、每层栈帧大小和溢出阈值；队列和显式栈的峰值长度。
// 用法：rec <mode> <shape> <n> [conn]
//   mode：rec_main 主线程默认栈；rec_thread 默认栈大小的子线程；rec_big 4 GB 栈的子线程（只为量深度）；bfs；dfs；info
//   shape：snake 和 tests/test_main.cpp 里的蛇形相同；solid 全部是前景
// 栈溢出时由备用栈上的信号处理函数输出一行 JSON 后正常退出，不产生崩溃报告。
#include "image.hpp"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <queue>
#include <string>
#include <unistd.h>
#include <vector>

using namespace course;

static long g_depth = 0, g_max_depth = 0;
static const char* g_first = nullptr;
static const char* g_deep = nullptr;
static ImageU8* g_img;
static std::vector<std::uint8_t>* g_seen;
static int g_rows, g_cols, g_conn;
static char g_msg_segv[128], g_msg_bus[128];

static void on_fault(int signo) {
    const char* msg = signo == SIGSEGV ? g_msg_segv : g_msg_bus;
    ssize_t ignored = write(1, msg, std::strlen(msg));
    (void)ignored;
    _exit(0);
}

static void install_handler() {
    std::snprintf(g_msg_segv, sizeof g_msg_segv, "{\"overflow\": true, \"signal\": \"SIGSEGV\", \"signo\": %d}\n", SIGSEGV);
    std::snprintf(g_msg_bus, sizeof g_msg_bus, "{\"overflow\": true, \"signal\": \"SIGBUS\", \"signo\": %d}\n", SIGBUS);
    stack_t ss;
    ss.ss_size = 1 << 16;
    ss.ss_sp = std::malloc(ss.ss_size);
    ss.ss_flags = 0;
    sigaltstack(&ss, nullptr);
    struct sigaction sa;
    std::memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_fault;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_ONSTACK;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
}

__attribute__((noinline)) void fill_rec(int y, int x) {
    if (y < 0 || y >= g_rows || x < 0 || x >= g_cols) return;
    size_t idx = static_cast<size_t>(y) * g_cols + x;
    if (!g_img->data[idx] || (*g_seen)[idx]) return;
    (*g_seen)[idx] = 1;
    char marker;
    ++g_depth;
    if (g_depth == 1) g_first = &marker;
    if (g_depth > g_max_depth) {
        g_max_depth = g_depth;
        g_deep = &marker;
    }
    fill_rec(y - 1, x);
    fill_rec(y + 1, x);
    fill_rec(y, x - 1);
    fill_rec(y, x + 1);
    if (g_conn == 8) {
        fill_rec(y - 1, x - 1);
        fill_rec(y - 1, x + 1);
        fill_rec(y + 1, x - 1);
        fill_rec(y + 1, x + 1);
    }
    --g_depth;
}

void* thread_main(void*) {
    install_handler();  // 备用栈是每个线程各自的
    int count = 0;
    for (int y = 0; y < g_rows; ++y)
        for (int x = 0; x < g_cols; ++x) {
            size_t idx = static_cast<size_t>(y) * g_cols + x;
            if (g_img->data[idx] && !(*g_seen)[idx]) {
                ++count;
                fill_rec(y, x);
            }
        }
    std::printf("{\"overflow\": false, \"count\": %d, \"max_depth\": %ld, \"frame_bytes\": %.1f}\n", count, g_max_depth,
                g_max_depth > 1 ? static_cast<double>(g_first - g_deep) / static_cast<double>(g_max_depth - 1) : 0.0);
    std::fflush(stdout);
    return nullptr;
}

ImageU8 make(const std::string& shape, int n) {
    ImageU8 img(n, n, 0);
    if (shape == "solid") {
        for (auto& v : img.data) v = 1;
    } else {
        for (int y = 0; y < n; y += 2)
            for (int x = 0; x < n; ++x) img.at(y, x) = 1;
        for (int y = 1; y < n; y += 4) img.at(y, n - 1) = 1;
        for (int y = 3; y < n; y += 4) img.at(y, 0) = 1;
    }
    return img;
}

int main(int argc, char** argv) {
    const std::string mode = argv[1];
    if (mode == "info") {
        pthread_attr_t a;
        pthread_attr_init(&a);
        size_t def = 0;
        pthread_attr_getstacksize(&a, &def);
        std::printf("{\"main_stack_bytes\": %zu, \"thread_stack_bytes\": %zu, \"pair_bytes\": %zu}\n",
                    pthread_get_stacksize_np(pthread_self()), def, sizeof(std::pair<int, int>));
        return 0;
    }
    const std::string shape = argv[2];
    const int n = std::atoi(argv[3]);
    g_conn = argc > 4 ? std::atoi(argv[4]) : 4;
    ImageU8 img = make(shape, n);
    std::vector<std::uint8_t> seen(img.data.size(), 0);
    g_img = &img;
    g_seen = &seen;
    g_rows = g_cols = n;
    long fg = 0;
    for (auto v : img.data) fg += v != 0;
    if (mode == "rec_main") {
        thread_main(nullptr);
    } else if (mode == "rec_big" || mode == "rec_thread") {
        pthread_attr_t a;
        pthread_attr_init(&a);
        if (mode == "rec_big") pthread_attr_setstacksize(&a, static_cast<size_t>(4) << 30);
        pthread_t t;
        if (pthread_create(&t, &a, thread_main, nullptr) != 0) return 3;
        pthread_join(t, nullptr);
    } else {
        const int dy[8] = {-1, 1, 0, 0, -1, -1, 1, 1}, dx[8] = {0, 0, -1, 1, -1, 1, -1, 1};
        size_t peak = 0;
        int count = 0;
        std::queue<std::pair<int, int>> q;
        std::vector<std::pair<int, int>> st;
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                if (!img(y, x) || seen[static_cast<size_t>(y) * n + x]) continue;
                ++count;
                seen[static_cast<size_t>(y) * n + x] = 1;
                if (mode == "bfs") q.push({y, x}); else st.push_back({y, x});
                while (mode == "bfs" ? !q.empty() : !st.empty()) {
                    peak = std::max(peak, mode == "bfs" ? q.size() : st.size());
                    std::pair<int, int> cur;
                    if (mode == "bfs") { cur = q.front(); q.pop(); } else { cur = st.back(); st.pop_back(); }
                    for (int k = 0; k < g_conn; ++k) {
                        int ny = cur.first + dy[k], nx = cur.second + dx[k];
                        if (ny < 0 || ny >= n || nx < 0 || nx >= n) continue;
                        size_t idx = static_cast<size_t>(ny) * n + nx;
                        if (img(ny, nx) && !seen[idx]) {
                            seen[idx] = 1;  // 入队或入栈时就标记，同一个像素不会进两次
                            if (mode == "bfs") q.push({ny, nx}); else st.push_back({ny, nx});
                        }
                    }
                }
            }
        std::printf("{\"count\": %d, \"peak\": %zu, \"foreground\": %ld}\n", count, peak, fg);
    }
    return 0;
}
'''

CHECKS_CPP = r'''
// 小型对照：整数提升与回绕、取整、NaN 比较、负尺寸构造。带 TAG 的行用来对应编译警告和 UBSan 报告。
#include "image.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

template <typename It>
int argmax_init_low(It b, It e) {  // 写法一：最好分数初始化成一个很小的数，用 > 更新
    int best = -1, i = 0;
    double score = -2.0;
    for (It p = b; p != e; ++p, ++i)
        if (*p > score) { score = *p; best = i; }
    return best;
}
template <typename It>
int argmax_init_first(It b, It e) {  // 写法二：用第一个元素初始化
    int best = 0, i = 1;
    double score = *b;
    for (It p = b + 1; p != e; ++p, ++i)
        if (*p > score) { score = *p; best = i; }
    return best;
}

int main(int argc, char**) {
    std::vector<int> v(10, 1);
    int total = 0;
    for (int i = 0; i < v.size(); ++i) total += v[i];  // TAG:sign_compare
    std::uint8_t a = static_cast<std::uint8_t>(199 + argc), b = static_cast<std::uint8_t>(99 + argc);
    std::uint8_t wrap = a + b;  // TAG:u8_wrap
    int promoted = a + b;
    std::uint8_t diff = b - a;
    std::uint8_t top = static_cast<std::uint8_t>(254 + argc), one = static_cast<std::uint8_t>(argc);
    std::uint8_t wrap255 = top + one;
    std::size_t k = static_cast<std::size_t>(argc - 1);
    std::size_t prev = k - 1;  // TAG:size_t_underflow
    int pixels = 33100 + argc;
    int sq = 65025 * pixels;  // TAG:int_overflow
    std::uint32_t acc = 4000000000u;
    std::uint32_t step = 300000000u * static_cast<std::uint32_t>(argc);
    acc += step;  // TAG:u32_wrap
    double x = 0.1 * argc + 0.2 * argc;
    bool eq = (x == 0.3);  // TAG:float_eq
    float f24 = 16777216.0f * static_cast<float>(argc);
    float f24p1 = f24 + 1.0f;
    std::fprintf(stderr, "total=%d sq=%d\n", total, sq);

    const double z = argc - 1;
    const double nan = z / z;
    std::vector<double> mid = {0.2, nan, 0.9, 0.5}, head = {nan, 0.2, 0.9, 0.5}, all = {nan, nan, nan};
    const double half[] = {0.5, 1.5, 2.5, 254.5};

    const char* neg_one = "none";
    try { course::ImageU8 img(-1, 5); } catch (const std::invalid_argument&) { neg_one = "invalid_argument"; } catch (const std::length_error&) { neg_one = "length_error"; } catch (const std::exception&) { neg_one = "other"; }
    const char* neg_two = "none";
    try { course::ImageU8 img(-2, -3); } catch (const std::invalid_argument&) { neg_two = "invalid_argument"; } catch (const std::length_error&) { neg_two = "length_error"; } catch (const std::exception&) { neg_two = "other"; }

    std::printf("{\"u8_255_plus_1\": %d, ", wrap255);
    std::printf("\"u8_200_plus_100\": %d, \"int_200_plus_100\": %d, \"u8_100_minus_200\": %d, \"size_t_0_minus_1\": \"%zu\", \"u32_4e9_plus_3e8\": %u, "
                "\"double_0.1_plus_0.2_eq_0.3\": %s, \"float_2p24_plus_1_eq_2p24\": %s, ",
                wrap, promoted, diff, prev, acc, eq ? "true" : "false", f24p1 == f24 ? "true" : "false");
    std::printf("\"round\": {");
    for (int i = 0; i < 4; ++i) {
        const double h = half[i] + (argc - 1);
        std::printf("%s\"%.1f\": {\"cast\": %d, \"round\": %.0f, \"nearbyint\": %.0f}", i ? ", " : "", h, static_cast<int>(static_cast<std::uint8_t>(h)), std::round(h), std::nearbyint(h));
    }
    std::printf("}, \"nan_mid\": {\"init_low\": %d, \"init_first\": %d, \"max_element\": %d}, \"nan_head\": {\"init_low\": %d, \"init_first\": %d, \"max_element\": %d}, \"nan_all_init_low\": %d, ",
                argmax_init_low(mid.begin(), mid.end()), argmax_init_first(mid.begin(), mid.end()), static_cast<int>(std::max_element(mid.begin(), mid.end()) - mid.begin()),
                argmax_init_low(head.begin(), head.end()), argmax_init_first(head.begin(), head.end()), static_cast<int>(std::max_element(head.begin(), head.end()) - head.begin()),
                argmax_init_low(all.begin(), all.end()));
    std::printf("\"image_neg1_5\": \"%s\", \"image_neg2_neg3\": \"%s\"}\n", neg_one, neg_two);
    return 0;
}
'''

TESTTIME_CPP = r'''
// 和 tests/test_main.cpp 里盒式均值的计时检查完全相同的步骤：同一张图，r=1 和 r=30 各调用一次，不重复、不预热。
#include "image.hpp"

#include <chrono>
#include <cstdio>
#include <random>

using namespace course;

int main() {
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> d(0, 255);
    ImageU8 big(2000, 2000);
    for (auto& v : big.data) v = static_cast<std::uint8_t>(d(rng));
    auto t0 = std::chrono::steady_clock::now();
    box_mean(big, 1);
    auto t1 = std::chrono::steady_clock::now();
    box_mean(big, 30);
    auto t2 = std::chrono::steady_clock::now();
    const double small = std::chrono::duration<double>(t1 - t0).count();
    const double large = std::chrono::duration<double>(t2 - t1).count();
    std::printf("{\"small_ms\": %.3f, \"large_ms\": %.3f, \"pass\": %s}\n", small * 1e3, large * 1e3, large > small * 4 + 0.05 ? "false" : "true");
    return 0;
}
'''

ASAN_CPP = r'''
// 越界读的三种情况：0 = 列越界但还在缓冲区内（落到下一行行首）；1 = 最后一行列越界（出了缓冲区）；2 = 用 at() 做同样的访问
#include "image.hpp"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    course::ImageU8 img(4, 6, 7);
    for (int x = 0; x < 6; ++x) img(1, x) = static_cast<std::uint8_t>(100 + x);
    const int which = argc > 1 ? std::atoi(argv[1]) : 0;
    volatile int y = which == 1 ? 3 : 0, x = 6;
    if (which == 2) {
        try {
            std::printf("value %d\n", img.at(y, x));
        } catch (const std::out_of_range&) {
            std::printf("out_of_range\n");
        }
        return 0;
    }
    std::printf("value %d\n", img(y, x));
    return 0;
}
'''


# ---------------------------------------------------------------- 编译和运行
def compile_cpp(out: str, sources: list, flags: list, check: bool = True):
    cmd = [CXX, "-std=c++17", *flags, "-I", str(INC), *[str(s) for s in sources], "-o", str(BUILD / out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"编译失败：{' '.join(cmd)}\n{res.stderr[:2000]}")
    return res


def run(exe: str, *args, env: dict | None = None, timeout: int = 120):
    full_env = dict(os.environ, **(env or {}))
    return subprocess.run([str(BUILD / exe), *[str(a) for a in args]], capture_output=True, text=True, timeout=timeout, env=full_env)


def run_json(exe: str, *args):
    res = run(exe, *args)
    lines = [ln for ln in res.stdout.strip().splitlines() if ln.startswith("{")]
    if not lines:
        raise RuntimeError(f"{exe} {args} 没有输出 JSON（返回码 {res.returncode}）：{res.stderr[:500]}")
    return json.loads(lines[-1])


def merge_min(a, b):
    if isinstance(a, dict):
        return {k: merge_min(a[k], b[k]) for k in a}
    if isinstance(a, bool) or not isinstance(a, (int, float)):
        return a
    return min(a, b)


_TIMED: dict = {}


def timed(jobs: list, rounds: int = 3) -> list:
    """计时用：把几条命令轮流各跑 rounds 遍，每个数取各遍里的最小值。

    轮流跑是为了让机器负载的变化平摊到每个配置上；相同的命令全脚本只测一组，保证讲义里同一个量只有一个数。
    """
    todo = [j for j in jobs if j not in _TIMED]
    for _ in range(rounds):
        for j in todo:
            res = run_json(*j)
            _TIMED[j] = merge_min(_TIMED[j], res) if j in _TIMED else res
    return [_TIMED[j] for j in jobs]


def run_tests(exe: str) -> dict:
    res = run(exe, env={"ASAN_OPTIONS": "abort_on_error=0"})
    out = {}
    for line in res.stdout.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            out[parts[1]] = f"{parts[0]}：{parts[2]}" if parts[0] != "PASS" else "PASS"
    return out


SEPARABLE_BOX = '''ImageF box_mean(const ImageU8& img, int r) {
    if (r < 0) throw std::invalid_argument("r 不能为负");
    Image<std::uint32_t> tmp(img.rows, img.cols, 0);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) {
            std::uint32_t s = 0;
            for (int xx = std::max(0, x - r); xx <= std::min(img.cols - 1, x + r); ++xx) s += img(y, xx);
            tmp(y, x) = s;
        }
    ImageF out(img.rows, img.cols, 0.0);
    for (int y = 0; y < img.rows; ++y) {
        int y0 = std::max(0, y - r), y1 = std::min(img.rows - 1, y + r);
        for (int x = 0; x < img.cols; ++x) {
            std::uint64_t s = 0;
            for (int yy = y0; yy <= y1; ++yy) s += tmp(yy, x);
            int n = (y1 - y0 + 1) * (std::min(img.cols - 1, x + r) - std::max(0, x - r) + 1);
            out(y, x) = static_cast<double>(s) / n;
        }
    }
    return out;
}
'''
NAIVE_BOX = '''ImageF box_mean(const ImageU8& img, int r) {
    if (r < 0) throw std::invalid_argument("r 不能为负");
    ImageF out(img.rows, img.cols, 0.0);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) {
            std::uint64_t s = 0;
            int n = 0;
            for (int yy = std::max(0, y - r); yy <= std::min(img.rows - 1, y + r); ++yy)
                for (int xx = std::max(0, x - r); xx <= std::min(img.cols - 1, x + r); ++xx) {
                    s += img(yy, xx);
                    ++n;
                }
            out(y, x) = static_cast<double>(s) / n;
        }
    return out;
}
'''
SAN = ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
CONFIGS = {"O0": ["-O0"], "O1": ["-O1"], "O2": ["-O2"], "O3": ["-O3"], "san": SAN, "novec": ["-O2", "-fno-vectorize", "-fno-slp-vectorize"]}
VARIANTS = {
    "累加用 uint32": ("v_u32", [("std::uint64_t row = 0;", "std::uint32_t row = 0;"),
                               ("ii(y + 1, x + 1) = ii(y, x + 1) + row;", "ii(y + 1, x + 1) = static_cast<std::uint32_t>(static_cast<std::uint32_t>(ii(y, x + 1)) + row);")]),
    "累加用 int": ("v_i32", [("std::uint64_t row = 0;", "std::uint32_t row = 0;"),
                            ("ii(y + 1, x + 1) = ii(y, x + 1) + row;",
                             "ii(y + 1, x + 1) = static_cast<std::uint64_t>(static_cast<std::int32_t>(static_cast<std::uint32_t>(ii(y, x + 1)) + row));")]),
    "盒式均值写成可分离的 O(r)": ("v_sep", [(r"re:ImageF box_mean\(.*?\n}\n", SEPARABLE_BOX)]),
    "盒式均值写成 O(r^2)": ("v_naive", [(r"re:ImageF box_mean\(.*?\n}\n", NAIVE_BOX)]),
    "盒式均值边界按补零算": ("v_zero", [("double n = static_cast<double>((y1 - y0) * (x1 - x0));", "double n = static_cast<double>((2 * r + 1) * (2 * r + 1));")]),
    "NCC 不特判平坦窗口": ("v_nan", [("double s = (wvar < 1e-12 || tvar < 1e-12) ? 0.0 : cov / std::sqrt(wvar * tvar);", "double s = cov / std::sqrt(wvar * tvar);")]),
    "盒式均值的 y1 不裁剪（越界读）": ("v_oob", [("y1 = std::min(img.rows, y + r + 1);", "y1 = y + r + 2;")]),
}


def build_all() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    for name, text in (("bench.cpp", BENCH_CPP), ("rec.cpp", REC_CPP), ("checks.cpp", CHECKS_CPP), ("asan_demo.cpp", ASAN_CPP), ("testtime.cpp", TESTTIME_CPP)):
        (BUILD / name).write_text(text.lstrip("\n"), encoding="utf-8")
    jobs = [(f"bench_{k}", [BUILD / "bench.cpp", REF], f) for k, f in CONFIGS.items()]
    jobs += [("rec_O0", [BUILD / "rec.cpp"], ["-O0"]), ("rec_O2", [BUILD / "rec.cpp"], ["-O2"]),
             ("checks", [BUILD / "checks.cpp"], ["-O2"]),
             ("checks_ubsan", [BUILD / "checks.cpp"], ["-O1", "-g", "-fsanitize=undefined"]),
             ("checks_ubsan_ext", [BUILD / "checks.cpp"], ["-O1", "-g", "-fsanitize=undefined,unsigned-integer-overflow,implicit-conversion"]),
             ("asan_demo", [BUILD / "asan_demo.cpp"], SAN)]
    for exe, edits in VARIANTS.values():
        src = REF.read_text(encoding="utf-8")
        for old, new in edits:
            if old.startswith("re:"):
                src, n = re.subn(old[3:], lambda _m, new=new: new, src, count=1, flags=re.S)
                assert n == 1, exe
            else:
                assert src.count(old) == 1, (exe, old)
                src = src.replace(old, new)
        (BUILD / f"{exe}.cpp").write_text(src, encoding="utf-8")
        jobs.append((exe, [TEST_MAIN, BUILD / f"{exe}.cpp"], ["-O2"]))
    jobs.append(("v_oob_asan", [TEST_MAIN, BUILD / "v_oob.cpp"], SAN))
    jobs.append(("testtime_ref", [BUILD / "testtime.cpp", REF], ["-O2"]))
    jobs.append(("testtime_sep", [BUILD / "testtime.cpp", BUILD / "v_sep.cpp"], ["-O2"]))
    jobs.append(("testtime_naive", [BUILD / "testtime.cpp", BUILD / "v_naive.cpp"], ["-O2"]))
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda j: compile_cpp(*j), jobs))


# ---------------------------------------------------------------- 计时条件
CONDITION_NOTE = (
    "所有耗时都在下面这一台机器上测得，单线程，是机器空闲时某一次运行的结果。每项在程序里重复至少 2 次，整组命令再轮流跑 2 到 3 遍，取所有结果里的最小值。"
    "耗时已经钉住：默认运行脚本时沿用这个文件里上次的值，设置环境变量 CH15_RETIME=1 才重新计时，这一组里的处理器、系统、编译器也跟着那次测量走。"
    "同一台机器很忙时另测过一次，见“另一次运行的对照”一组。耗时只适合比较量级和明显的倍数，不是普遍结论；讲义里只写两位有效数字。"
    "深度、阈值、溢出边界、误差这些不是耗时的数字，每次运行都重新计算，结果不变")


def exp_conditions():
    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return "未知"

    def measure():
        noise = run_json("bench_O2", "noise")
        return {
            "说明": CONDITION_NOTE,
            "处理器": sh("sysctl", "-n", "machdep.cpu.brand_string"),
            "系统": "macOS " + sh("sw_vers", "-productVersion"),
            "编译器（命令 g++ 在这台机器上就是它）": sh(CXX, "--version").splitlines()[0],
            "公共编译选项": "-std=c++17，再加表里写的优化级别；Sanitizer 一档是 " + " ".join(SAN),
            "同一段代码连续计时 30 次的耗时（ms，盒式均值 2000x2000、r=7、-O2）": {"最快": sig(noise["min"]), "中位数": sig(noise["median"]), "最慢": sig(noise["max"])},
            "最慢比最快多出的比例": f"{100 * (noise['max'] / noise['min'] - 1):.0f}%",
        }

    cond = pin("计时条件", measure)
    NUM["计时条件"] = {**cond, "说明": CONDITION_NOTE}  # 说明是脚本里的文字，不属于测量，每次按现在的写法写
    NUM["另一次运行的对照"] = SECOND_RUN


# ---------------------------------------------------------------- 图 1：内存布局读错
def make_die(H=120, W=250):
    rng = np.random.default_rng(0)
    img = np.full((H, W), 60.0)
    y, x = np.mgrid[0:H, 0:W]
    img[(x >= 20) & (x < 110) & (y >= 15) & (y < 105) & ((x % 10) < 5)] = 150.0
    img[20:50, 140:170] = 210.0
    img[70:100, 140:170] = 210.0
    img[(y - 60) ** 2 + (x - 210) ** 2 < 26 ** 2] = 180.0
    img[:, 124:127] = 240.0
    img += rng.normal(scale=4.0, size=img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def fig_layout():
    die = make_die()
    H, W = die.shape
    pad = 2
    step = W + pad
    buf = np.zeros((H, step), np.uint8)
    buf[:, :W] = die
    wrong_step = buf.ravel()[:H * W].reshape(H, W)

    BH, BW, oy, ox = 200, 400, 40, 80
    rng = np.random.default_rng(1)
    big = np.clip(np.round(25 + rng.normal(scale=3.0, size=(BH, BW))), 0, 255).astype(np.uint8)
    big[oy:oy + H, ox:ox + W] = die
    big[oy:oy + H, ox + W + 12:] = die[:, :BW - (ox + W + 12)]
    big[oy:oy + H, :ox - 12] = die[:, W - (ox - 12):]
    roi = big[oy:oy + H, ox:ox + W]
    start = oy * BW + ox
    wrong_roi = big.ravel()[start:start + H * W].reshape(H, W)

    def summary(got):
        return {"整行完全正确的行数": int((got == die).all(axis=1).sum()), "总行数": H,
                "平均绝对误差（灰度）": round(float(np.abs(got.astype(int) - die).mean()), 1)}

    # 三通道：每行 W*3 个字节，行尾补到 4 的倍数
    bgr = np.stack([die, die[::-1], die[:, ::-1]], axis=2)
    step3 = (W * 3 + 3) // 4 * 4
    buf3 = np.zeros((H, step3), np.uint8)
    buf3[:, :W * 3] = bgr.reshape(H, W * 3)
    flat3 = buf3.ravel()
    yy, xx = np.mgrid[0:H, 0:W]
    g_right = flat3[yy * step3 + xx * 3 + 1]
    g_old = flat3[(yy * W + xx) * 3 + 1]

    view = big[oy:oy + H, ox:ox + W]
    before = big.copy()
    view[:10, :10] = 0
    changed_by_view = int((big != before).sum())
    big[:] = before
    deep = big[oy:oy + H, ox:ox + W].copy()
    deep[:10, :10] = 0
    changed_by_copy = int((big != before).sum())

    NUM["内存布局读错"] = {
        "说明": f"合成 die 图 {H} 行 x {W} 列，8 位灰度，带标准差 4 的噪声；带填充的缓冲区每行 {W} 个像素后面有 {pad} 个填充字节（step={step}）；"
              f"ROI 取自一张 {BH}x{BW} 大图的 ({oy}, {ox}) 处。用 numpy 模拟读内存，不涉及 C++",
        "正确读法（行首 = y * step）": summary(buf[:, :W]),
        "把 W 当成 step 读带填充的缓冲区": summary(wrong_step),
        "把 ROI 当成连续内存拷 W*H 个字节": summary(wrong_roi),
        "ROI 视图的行步长（字节）": int(roi.strides[0]),
        "ROI 视图在内存里是否连续": bool(roi.flags["C_CONTIGUOUS"]),
        "三通道": {
            "说明": f"同样大小的 BGR 图，每行 {W * 3} 个字节，补齐到 4 的倍数后 step={step3} 字节，读 G 通道",
            "y*step + x*3 + c（step 按字节）整行正确的行数": int((g_right == bgr[:, :, 1]).all(axis=1).sum()),
            "(y*W + x)*3 + c（原讲义的写法，step 取每行像素数）整行正确的行数": int((g_old == bgr[:, :, 1]).all(axis=1).sum()),
        },
        "自测题 1 的答案：1920x1080 连续 BGR 图里 (y=100, x=200) 的 G 通道字节下标": (100 * 1920 + 200) * 3 + 1,
        "视图和深拷贝（用 numpy 的视图类比 cv::Mat 的浅拷贝）": {
            "改视图里 10x10 个像素后，大图里变化的像素数": changed_by_view,
            "改 .copy() 得到的副本后，大图里变化的像素数": changed_by_copy,
        },
    }

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.5), gridspec_kw={"width_ratios": [1.25, 1, 1, 1]}, layout="constrained")
    a = ax[0]
    for r in range(4):
        for c in range(8):
            is_pad = c >= 6
            a.add_patch(plt.Rectangle((c, 3 - r), 1, 1, fc="#eceef1" if is_pad else "#dbe7ff", ec="#8f959e", lw=0.8, hatch="///" if is_pad else None))
            a.text(c + 0.5, 3.5 - r, str(r * 8 + c), ha="center", va="center", fontsize=8.5, color=S.MUTED if is_pad else S.INK)
    a.plot([0, 6], [-0.25, -0.25], color=S.BLUE, lw=2)
    a.text(3, -0.7, "W = 6 个像素", ha="center", color=S.BLUE, fontsize=9.5)
    a.plot([0, 8], [4.25, 4.25], color=S.RED, lw=2)
    a.text(4, 4.5, "step = 8 个字节（含 2 个填充）", ha="center", color=S.RED, fontsize=9.5)
    for r in range(4):
        a.text(-0.25, 3.5 - r, f"行首 {r * 8}", ha="right", va="center", fontsize=8.5, color=S.MUTED)
    a.set_xlim(-1.9, 8.2)
    a.set_ylim(-1.1, 5.0)
    a.set_aspect("equal")
    a.axis("off")
    a.set_title("格子里是字节地址：行首 = y × step")
    for a, img, title in ((ax[1], die, "按 step 读：正确"),
                          (ax[2], wrong_step, f"把 W 当 step：每行多错开 {pad} px"),
                          (ax[3], wrong_roi, "ROI 当连续内存拷：串到邻居")):
        a.imshow(img, cmap="gray", vmin=0, vmax=255)
        a.set_title(title)
        S.image_axes(a)
    S.save(fig, CH, "fig01_内存布局读错")


# ---------------------------------------------------------------- 图 2：遍历顺序
def fig_traverse():
    sizes = [250, 256, 500, 512, 1000, 1024, 2000, 2048, 3000, 4000, 4096, 6000, 8000, 8192]
    pick = [1000, 1024, 4000, 4096]

    def measure():
        o2, o0, novec = timed([("bench_O2", "traverse", *sizes), ("bench_O0", "traverse", *pick), ("bench_novec", "traverse", *pick)], rounds=2)

        def table(res, keys):
            return {str(n): {"行优先": sig(res[str(n)][keys[0]]), "列优先": sig(res[str(n)][keys[1]]),
                             "列优先慢的倍数": round(res[str(n)][keys[1]] / res[str(n)][keys[0]], 1)} for n in (int(k) for k in res)}

        return {
            "说明": "N x N 的 8 位随机图，同一个函数只交换内外层循环；数值是每像素纳秒数，取多次里最快的一次。"
                  "按编译器的向量化报告（-Rpass=loop-vectorize），统计直方图的两种顺序都没有被自动向量化，差距只来自访存；"
                  "求和的两种顺序编译器都报告做了向量化，但列优先时每个向量要从相隔一整行的地址拼起来",
            "统计直方图，-O2": table(o2, ("hist_row", "hist_col")),
            "求和，-O2": table(o2, ("sum_row", "sum_col")),
            "统计直方图，-O0": table(o0, ("hist_row", "hist_col")),
            "求和，-O2 但关掉自动向量化": table(novec, ("sum_row", "sum_col")),
        }

    num = pin("遍历顺序", measure)
    h2, h0 = num["统计直方图，-O2"], num["统计直方图，-O0"]
    slow = {n: h2[str(n)]["列优先慢的倍数"] for n in sizes}
    pow2 = [n for n in sizes if n & (n - 1) == 0]
    other = [n for n in sizes if n not in pow2]
    claim(all(slow[p] > slow[max(n for n in other if n < p)] for p in pow2) and slow[8000] > slow[1000], "列优先在大图和 2 的幂边长上变慢")
    r2 = [slow[n] for n in pick]
    r0 = [h0[str(n)]["列优先慢的倍数"] for n in pick]
    claim(all(a <= b for a, b in zip(r0, r2)), "同样的循环，不开优化时倍数更小")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained")
    ax[0].plot(sizes, [h2[str(n)]["行优先"] for n in sizes], "o-", color=S.BLUE, lw=1.8, ms=4, label="行优先（所有尺寸）")
    ax[0].plot(other, [h2[str(n)]["列优先"] for n in other], "s-", color=S.ORANGE, lw=1.8, ms=5, label="列优先，边长不是 2 的幂")
    ax[0].plot(pow2, [h2[str(n)]["列优先"] for n in pow2], "^--", color=S.RED, lw=1.8, ms=6, label="列优先，边长是 2 的幂")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xticks([250, 500, 1000, 2000, 4000, 8000], ["250", "500", "1000", "2000", "4000", "8000"])
    ax[0].set_yticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
    ax[0].minorticks_off()
    ax[0].set_xlabel("图像边长 N（像素）")
    ax[0].set_ylabel("每像素耗时（ns，-O2）")
    ax[0].set_title("统计直方图：列优先在大图和 2 的幂边长上变慢")
    ax[0].legend(fontsize=9, loc="upper left")
    S.clean(ax[0])

    xs = np.arange(len(pick))
    b1 = ax[1].bar(xs - 0.2, r2, 0.4, color=S.ORANGE, alpha=0.9, label="-O2")
    b2 = ax[1].bar(xs + 0.2, r0, 0.4, color=S.MUTED, alpha=0.75, label="-O0")
    for bars, vals in ((b1, r2), (b2, r0)):
        for b, v in zip(bars, vals):
            ax[1].text(b.get_x() + b.get_width() / 2, v + max(r2) * 0.015, two(v), ha="center", fontsize=9)
    ax[1].axhline(1, color=S.MUTED, lw=1, ls="--")
    ax[1].set_xticks(xs, [str(n) for n in pick])
    ax[1].set_xlabel("图像边长 N")
    ax[1].set_ylabel("列优先耗时 / 行优先耗时")
    ax[1].set_title("同样的循环，不开优化时倍数更小")
    ax[1].legend(fontsize=9, loc="upper left")
    S.clean(ax[1])
    ax[1].grid(axis="x", visible=False)
    S.save(fig, CH, "fig02_遍历顺序")


# ---------------------------------------------------------------- 图 3：类型与溢出
def box_inner(ii, r):
    """窗口完整的内部像素上的盒内求和，按 C++ 里四个角从左到右加减的顺序。"""
    k = 2 * r + 1
    return ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]


def fig_overflow():
    chk = run_json("checks")
    limits = {"float32 能精确表示的连续整数上限 2^24": 2 ** 24, "int32 上限": 2 ** 31 - 1, "uint32 上限": 2 ** 32 - 1}

    def first_side(per_pixel, limit):
        n = math.isqrt(limit // per_pixel)
        while per_pixel * n * n <= limit:
            n += 1
        return n

    sides = {name: first_side(255, lim) for name, lim in limits.items()}
    sides_sq = {name: first_side(255 * 255, lim) for name, lim in limits.items()}
    cv_default = {}
    for n in (sides["int32 上限"] - 1, sides["int32 上限"]):
        s = cv2.integral(np.full((n, n), 255, np.uint8))
        cv_default[f"{n}x{n} 全白图"] = {"右下角的值": int(s[-1, -1]), "真值": 255 * n * n, "输出类型": str(s.dtype)}

    # float32 积分图：递推和参考实现相同（先累加行内和，再加上一行），只把类型换成 float32
    f32 = {}
    sizes = [500, 1000, 2000, 4000]
    for n in sizes:
        img = np.random.default_rng(n).integers(0, 256, (n, n), dtype=np.uint8)
        exact = np.zeros((n + 1, n + 1), np.int64)
        exact[1:, 1:] = img.astype(np.int64).cumsum(axis=1).cumsum(axis=0)
        approx = np.zeros((n + 1, n + 1), np.float32)
        approx[1:, 1:] = np.cumsum(np.cumsum(img, axis=1, dtype=np.float32), axis=0, dtype=np.float32)
        f32[n] = {}
        for r in (1, 7):
            err = np.abs(box_inner(approx, r).astype(np.float64) - box_inner(exact, r)) / (2 * r + 1) ** 2
            f32[n][r] = (float(err.max()), float((err > 0.5).mean()))
        if n == 2000:
            cv32 = cv2.integral(img, sdepth=cv2.CV_32F)
            err = np.abs(box_inner(cv32, 1).astype(np.float64) - box_inner(exact, 1)) / 9
            cv_f32 = (float(err.max()), float((err > 0.5).mean()))
            total_2000 = int(exact[-1, -1])

    # uint32 回绕后的盒内求和（无符号数的加减是模 2^32 的运算）
    n = 5000
    img = np.random.default_rng(7).integers(240, 256, (n, n), dtype=np.uint8)
    exact = np.zeros((n + 1, n + 1), np.uint64)
    exact[1:, 1:] = img.astype(np.uint64).cumsum(axis=1).cumsum(axis=0)
    over = exact > 2 ** 32 - 1
    assert not over[:n // 2].any() and not over[:, :n // 2].any()
    q_exact = exact[n // 2:, n // 2:]
    q_wrapped = q_exact.astype(np.uint32)
    wrap_info = {"说明": f"{n}x{n}、灰度 240 到 255 的随机图，积分图用 uint32 存（等价于累加时回绕），盒内求和也用 uint32 做加减。"
                       "回绕的单元全部落在右下四分之一，只在这一块上逐个比较",
                 "右下角的真值": int(exact[-1, -1]), "回绕后右下角的值": int(q_wrapped[-1, -1]),
                 "已经回绕的积分图单元占全图的比例": f"{100 * float(over.mean()):.1f}%",
                 "比较的 3x3 窗口个数": int(box_inner(q_exact, 1).size),
                 "3x3 盒内求和算错的个数": int((box_inner(q_wrapped, 1) != box_inner(q_exact, 1)).sum()),
                 "15x15 盒内求和算错的个数": int((box_inner(q_wrapped, 7) != box_inner(q_exact, 7)).sum())}
    del exact, over, q_exact, q_wrapped, img

    ramp = np.arange(256, dtype=np.uint8)
    wrap60 = ramp + np.uint8(60)
    sat60 = cv2.add(ramp.reshape(1, -1), np.full((1, 256), 60, np.uint8)).ravel()
    NUM["类型与溢出"] = {
        "说明": "溢出边界由脚本按类型上限算出；C++ 一栏的数值来自 -O2 编译的小程序，numpy 和 OpenCV 一栏来自 Python",
        "C++ 实测": {"uint8 的 255 + 1 存回 uint8": chk["u8_255_plus_1"], "uint8 的 200 + 100 存回 uint8": chk["u8_200_plus_100"], "同样的加法存到 int": chk["int_200_plus_100"],
                   "uint8 的 100 - 200 存回 uint8": chk["u8_100_minus_200"], "size_t 的 0 - 1": chk["size_t_0_minus_1"],
                   "uint32 的 4000000000 + 300000000": chk["u32_4e9_plus_3e8"], "double 的 0.1 + 0.2 == 0.3": chk["double_0.1_plus_0.2_eq_0.3"],
                   "float 的 16777216 + 1 仍等于 16777216": chk["float_2p24_plus_1_eq_2p24"]},
        "numpy 和 OpenCV 实测": {"numpy uint8 的 200 + 100": int((np.uint8(200) + np.uint8(100))), "cv2.add 的 200 + 100（饱和）": int(cv2.add(np.full((1, 1), 200, np.uint8), np.full((1, 1), 100, np.uint8))[0, 0]),
                             "numpy uint8 的 100 - 200": int(np.uint8(100) - np.uint8(200)), "cv2.subtract 的 100 - 200（饱和）": int(cv2.subtract(np.full((1, 1), 100, np.uint8), np.full((1, 1), 200, np.uint8))[0, 0]),
                             "uint8 整体加 60 时回绕成暗值的灰度级个数": int((wrap60 < ramp).sum())},
        "类型上限": {k: int(v) for k, v in limits.items()},
        "原讲义的例子：4000x4000 全白图的像素和": {"数值": 255 * 4000 * 4000, "占 uint32 上限的比例": f"{100 * 255 * 4000 * 4000 / (2 ** 32 - 1):.1f}%",
                                      "是否超过 uint32 上限": bool(255 * 4000 * 4000 > 2 ** 32 - 1), "是否超过 int32 上限": bool(255 * 4000 * 4000 > 2 ** 31 - 1)},
        "全白正方形图的像素和第一次超过上限时的边长": sides,
        "全白正方形窗口的平方和第一次超过上限时的边长": sides_sq,
        "double 能精确累加的全白平方和对应的最大边长": math.isqrt(2 ** 53 // (255 * 255)),
        "cv2.integral 对 8 位图的默认输出": cv_default,
        "float32 积分图": {
            "说明": "0 到 255 均匀随机图，积分图用 float32 递推，只统计窗口完整的内部像素；误差是盒式均值相对精确值的绝对差（灰度）",
            "边长": sizes,
            "3x3 均值的最大误差": [round(f32[n][1][0], 2) for n in sizes],
            "3x3 均值误差超过 0.5 的像素比例": [f"{100 * f32[n][1][1]:.1f}%" for n in sizes],
            "15x15 均值的最大误差": [round(f32[n][7][0], 2) for n in sizes],
            "15x15 均值误差超过 0.5 的像素比例": [f"{100 * f32[n][7][1]:.1f}%" for n in sizes],
            "2000x2000 图的像素总和": total_2000,
            "对照：cv2.integral(sdepth=CV_32F) 在 2000x2000 上的 3x3 均值（最大误差，超过 0.5 的比例）": [round(cv_f32[0], 2), f"{100 * cv_f32[1]:.1f}%"],
        },
        "uint32 回绕": wrap_info,
    }

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.0), layout="constrained")
    ax[0].plot(ramp, np.minimum(ramp.astype(int) + 60, 255), color=S.BLUE, lw=2, label="先提升到 int 再饱和截断")
    ax[0].plot(ramp, wrap60, color=S.RED, lw=1.8, ls="--", label="直接存回 uint8（回绕）")
    ax[0].axvline(196, color=S.MUTED, lw=1, ls=":")
    ax[0].text(193, 120, "灰度 196 起回绕", fontsize=9, color=S.MUTED, ha="right")
    ax[0].set_xlabel("原来的灰度")
    ax[0].set_ylabel("加 60 之后存进 uint8 的值")
    ax[0].set_title("uint8 加 60：最亮的 60 级回绕成最暗")
    ax[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    S.clean(ax[0])
    assert np.array_equal(sat60, np.minimum(ramp.astype(int) + 60, 255))

    ns = np.unique(np.round(np.logspace(2, 4, 200)).astype(int))
    ax[1].plot(ns, 255.0 * ns * ns, color=S.INK, lw=2, label="全白 N x N 图的像素和")
    for (name, lim), color in zip(limits.items(), (S.GREEN, S.ORANGE, S.RED)):
        ax[1].axhline(lim, color=color, lw=1.3, ls="--")
        ax[1].text(105, lim * 1.25, name.split(" 2^")[0], color=color, fontsize=9)
        ax[1].plot([sides[name]], [lim], "o", color=color, ms=6)
        ax[1].annotate(f"N = {sides[name]}", (sides[name], lim), textcoords="offset points",
                       xytext=(7, 5) if name.startswith("uint32") else (7, -13), fontsize=9, color=color)
    ax[1].plot([4000], [255 * 4000 * 4000], "*", color=S.BLUE, ms=13)
    ax[1].annotate("测试用的 4000 x 4000\n像素和 4.08e9", (4000, 255 * 4000 * 4000), xytext=(700, 1.5e8), fontsize=9, color=S.BLUE,
                   arrowprops={"arrowstyle": "-", "color": S.BLUE, "lw": 1})
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xticks([100, 300, 1000, 3000, 10000], ["100", "300", "1000", "3000", "10000"])
    ax[1].minorticks_off()
    ax[1].set_ylim(1e6, 1e11)
    ax[1].set_xlabel("图像边长 N")
    ax[1].set_ylabel("积分图右下角的值")
    ax[1].set_title("4000 x 4000 全白图超过 int32，没超过 uint32")
    S.clean(ax[1])

    for r, color, marker in ((1, S.RED, "o"), (7, S.BLUE, "s")):
        ax[2].plot(sizes, [100 * f32[n][r][1] for n in sizes], marker + "-", color=color, lw=1.8, ms=5, label=f"{2 * r + 1} x {2 * r + 1} 窗口")
    ax[2].set_xscale("log")
    ax[2].set_xticks(sizes, [str(n) for n in sizes])
    ax[2].minorticks_off()
    ax[2].set_ylim(-3, 100)
    ax[2].set_xlabel("图像边长 N（随机图）")
    ax[2].set_ylabel("均值误差超过 0.5 灰度的像素比例（%）")
    ax[2].set_title("float32 积分图：图越大、窗口越小，错得越多")
    ax[2].legend(fontsize=9, loc="upper left")
    S.clean(ax[2])
    S.save(fig, CH, "fig03_类型与溢出")


# ---------------------------------------------------------------- 图 4：递归深度和栈
def fig_recursion():
    info = run_json("rec_O2", "info")
    sizes = [50, 100, 200, 400, 800, 1600, 3000]
    depth = {shape: [run_json("rec_O2", "rec_big", shape, n)["max_depth"] for n in sizes] for shape in ("solid", "snake")}
    frame = {opt: run_json(f"rec_{opt}", "rec_big", "solid", 200)["frame_bytes"] for opt in ("O0", "O2")}

    def threshold(opt, mode, shape):
        lo, hi = 8, 4096
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if run_json(f"rec_{opt}", mode, shape, mid)["overflow"]:
                hi = mid
            else:
                lo = mid
        ok = run_json(f"rec_{opt}", mode, shape, lo)
        bad = run_json(f"rec_{opt}", mode, shape, hi)
        return {"最大能跑完的边长": lo, "此时的递归深度": ok["max_depth"], "第一个溢出的边长": hi, "溢出时收到的信号": f"{bad['signal']}（编号 {bad['signo']}）"}

    names = {"solid": "实心块", "snake": "蛇形"}
    thr = {}
    for opt in ("O2", "O0"):
        for mode, mname in (("rec_main", "主线程"), ("rec_thread", "默认栈大小的子线程")):
            for shape in ("solid", "snake"):
                thr[f"-{opt}，{mname}，{names[shape]}"] = threshold(opt, mode, shape)
    a, b = thr["-O2，主线程，实心块"]["第一个溢出的边长"], thr["-O0，主线程，实心块"]["最大能跑完的边长"]
    mid = (a + b) // 2
    between = {"边长": mid, "-O0 编译是否溢出": run_json("rec_O0", "rec_main", "solid", mid)["overflow"], "-O2 编译是否溢出": run_json("rec_O2", "rec_main", "solid", mid)["overflow"]}
    snake_test = {f"-{opt}": run_json(f"rec_{opt}", "rec_main", "snake", 3000)["overflow"] for opt in ("O2", "O0")}
    peaks = {}
    for shape in ("solid", "snake"):
        for conn in (4, 8):
            q = run_json("rec_O2", "bfs", shape, 3000, conn)
            st = run_json("rec_O2", "dfs", shape, 3000, conn)
            assert q["count"] == st["count"] == 1
            peaks[f"{names[shape]}，{conn} 连通"] = {"前景像素数": q["foreground"], "队列（BFS）峰值长度": q["peak"], "显式栈（DFS）峰值长度": st["peak"],
                                                 "显式栈峰值占前景像素的比例": f"{100 * st['peak'] / q['foreground']:.1f}%"}
    d3000 = {names[s]: depth[s][-1] for s in depth}
    NUM["递归深度和栈"] = {
        "说明": "本脚本内嵌的递归泛洪填充（4 连通，按上、下、左、右的顺序递归，带深度计数），在 N x N 的图上运行；"
              "蛇形和 tests/test_main.cpp 里的相同，实心块是全前景。每层栈帧大小由首尾两层局部变量的地址差除以层数得到。"
              "阈值和帧大小只对这份代码、这个编译器成立，换一种写法会变",
        "主线程栈大小（字节）": info["main_stack_bytes"], "主线程栈大小（MB）": round(info["main_stack_bytes"] / 2 ** 20, 2),
        "子线程默认栈大小（字节）": info["thread_stack_bytes"], "子线程默认栈大小（KB）": info["thread_stack_bytes"] // 1024,
        "每层递归占用的栈（字节）": {"-O2": frame["O2"], "-O0": frame["O0"]},
        "边长": sizes,
        "递归深度": {names[s]: depth[s] for s in depth},
        "3000x3000 时的递归深度": d3000,
        "3000x3000 时按 -O2 的帧大小需要的栈（MB）": {k: round(v * frame["O2"] / 2 ** 20) for k, v in d3000.items()},
        "栈溢出阈值": thr,
        "介于两个阈值之间的实心块": between,
        "测试里的 3000x3000 蛇形用递归在主线程上是否溢出": snake_test,
        "不用递归时容器的峰值长度（3000x3000）": peaks,
        "队列或显式栈每个元素的大小（字节，std::pair<int, int>）": info["pair_bytes"],
        "实心块 4 连通时显式栈峰值占用的堆内存（MB）": round(peaks["实心块，4 连通"]["显式栈（DFS）峰值长度"] * info["pair_bytes"] / 2 ** 20, 1),
    }

    small = {}
    for shape, n in (("solid", 32), ("solid", 31), ("snake", 32)):
        q4, q8, st = run_json("rec_O2", "bfs", shape, n, 4), run_json("rec_O2", "bfs", shape, n, 8), run_json("rec_O2", "dfs", shape, n, 4)
        small[f"{names[shape]} {n}x{n}"] = {"前景像素数": q4["foreground"], "递归深度": run_json("rec_O2", "rec_big", shape, n)["max_depth"],
                                          "显式栈峰值（4 连通）": st["peak"], "队列峰值（4 连通）": q4["peak"], "队列峰值（8 连通）": q8["peak"]}
    NUM["交互演示用的小图"] = {"说明": "demos/连通域遍历.html 里几步引导用到的尺寸，用同一个 C++ 程序算出来，和页面上显示的数应当一致", **small}

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.1), gridspec_kw={"width_ratios": [1.25, 1]}, layout="constrained")
    ax[0].plot(sizes, depth["solid"], "o-", color=S.BLUE, lw=1.8, ms=5, label="实心块（全前景）")
    ax[0].plot(sizes, depth["snake"], "s-", color=S.ORANGE, lw=1.8, ms=5, label="蛇形（测试里的形状）")
    for key, color, text in (("-O2，主线程，实心块", S.RED, "主线程"), ("-O2，默认栈大小的子线程，实心块", S.PURPLE, "子线程")):
        lim = thr[key]["此时的递归深度"]
        ax[0].axhline(lim, color=color, lw=1.3, ls="--")
        ax[0].text(3300, lim * 1.25, f"{text}的栈在 -O2 下最多到 {lim} 层", color=color, fontsize=9, ha="right")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xticks(sizes, [str(n) for n in sizes])
    ax[0].minorticks_off()
    ax[0].set_xlabel("图像边长 N")
    ax[0].set_ylabel("递归的最大深度（层）")
    ax[0].set_title("递归深度随面积增长，几百像素见方就超过栈")
    ax[0].legend(fontsize=9, loc="lower right")
    S.clean(ax[0])

    groups = ["实心块", "蛇形"]
    series = [("递归深度", [d3000[g] for g in groups], S.RED),
              ("显式栈峰值", [peaks[f"{g}，4 连通"]["显式栈（DFS）峰值长度"] for g in groups], S.ORANGE),
              ("队列峰值", [peaks[f"{g}，4 连通"]["队列（BFS）峰值长度"] for g in groups], S.BLUE)]
    xs = np.arange(len(groups))
    for i, (name, vals, color) in enumerate(series):
        bars = ax[1].bar(xs + (i - 1) * 0.26, vals, 0.26, color=color, alpha=0.88, label=name, bottom=0.5)
        for bar, v in zip(bars, vals):
            ax[1].text(bar.get_x() + bar.get_width() / 2, (v + 0.5) * 1.25, f"{v / 10000:.0f} 万" if v >= 10000 else str(v), ha="center", fontsize=9)
    ax[1].set_yscale("log")
    ax[1].set_ylim(0.5, 2e8)
    ax[1].set_yticks([1, 100, 10000, 1000000], ["1", "100", "1 万", "100 万"])
    ax[1].minorticks_off()
    ax[1].set_xticks(xs, [f"{g} 3000 x 3000" for g in groups])
    ax[1].set_ylabel("层数或容器长度（对数轴）")
    ax[1].set_title("同一张图，三种写法要记住多少个像素")
    ax[1].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    S.clean(ax[1])
    ax[1].grid(axis="x", visible=False)
    S.save(fig, CH, "fig04_递归深度和栈")


# ---------------------------------------------------------------- 图 5：积分图与边界
def clipped_box_mean(img, r):
    """和作业契约相同的盒式均值：窗口裁剪到图像内，只对图像内的像素求平均。"""
    H, W = img.shape
    ii = np.zeros((H + 1, W + 1), np.float64)
    ii[1:, 1:] = img.astype(np.float64).cumsum(axis=0).cumsum(axis=1)
    y0, y1 = np.maximum(0, np.arange(H) - r), np.minimum(H, np.arange(H) + r + 1)
    x0, x1 = np.maximum(0, np.arange(W) - r), np.minimum(W, np.arange(W) + r + 1)
    s = ii[np.ix_(y1, x1)] - ii[np.ix_(y0, x1)] - ii[np.ix_(y1, x0)] + ii[np.ix_(y0, x0)]
    return s / ((y1 - y0)[:, None] * (x1 - x0)[None, :])


def fig_box():
    def measure():
        res, = timed([("bench_O2", "box")], rounds=2)
        rs = [int(k) for k in res["r"]]
        t = {name: [res["r"][str(r)][name] for r in rs] for name in ("integral", "separable", "naive")}
        return {
            "说明": "1000x1000 的 8 位随机图，-O2，耗时 ms；积分图是参考实现，另外两种是本脚本内嵌的对照实现，三者边界约定相同。朴素写法只测到 r=30",
            "半径 r": rs,
            "朴素 O(r^2)": [sig(v) if v > 0 else None for v in t["naive"]],
            "可分离 O(r)": [sig(v) for v in t["separable"]],
            "积分图 O(1)": [sig(v) for v in t["integral"]],
            "积分图在各半径下的最慢与最快之比": round(max(t["integral"]) / min(t["integral"]), 2),
            "r=30 时朴素写法比积分图慢的倍数": round(res["r"]["30"]["naive"] / res["r"]["30"]["integral"]),
            "r=30 时可分离写法比积分图慢的倍数": round(res["r"]["30"]["separable"] / res["r"]["30"]["integral"]),
        }

    num = pin("盒式均值的复杂度", measure)
    rs = num["半径 r"]
    t = {"integral": num["积分图 O(1)"], "separable": num["可分离 O(r)"], "naive": num["朴素 O(r^2)"]}
    claim(num["积分图在各半径下的最慢与最快之比"] < 1.3 and t["separable"][-1] > 5 * t["separable"][0] and t["naive"][6] > 5 * t["naive"][0],
          "只有积分图的耗时不随半径变化")

    img = np.random.default_rng(0).integers(0, 256, (2000, 2000), dtype=np.uint8)
    border = {}
    for r in (1, 7, 30):
        k = 2 * r + 1
        clip = clipped_box_mean(img, r)
        refl = cv2.boxFilter(img, cv2.CV_64F, (k, k), borderType=cv2.BORDER_REFLECT_101)
        zero = cv2.boxFilter(img, cv2.CV_64F, (k, k), borderType=cv2.BORDER_CONSTANT)
        border[f"r={r}"] = {"离边不到 r 的像素占比": f"{100 * (1 - (2000 - 2 * r) ** 2 / 2000 ** 2):.1f}%",
                            "裁剪窗口与反射 101 的最大差": round(float(np.abs(clip - refl).max()), 1),
                            "裁剪窗口与补零的最大差": round(float(np.abs(clip - zero).max()), 1),
                            "内部像素上三者的最大差": float(f"{max(np.abs(clip - refl)[r:-r, r:-r].max(), np.abs(clip - zero)[r:-r, r:-r].max()):.1g}")}
    u8 = cv2.blur(img, (15, 15), borderType=cv2.BORDER_REFLECT_101)
    ref7 = clipped_box_mean(img, 7)
    NUM["边界策略"] = {
        "说明": "2000x2000 的 8 位随机图，同一个盒式均值用三种边界约定各算一遍：裁剪窗口（本章作业的契约）、反射 101（OpenCV 的默认）、补零",
        **border,
        "cv2.blur 的默认边界是否等于 BORDER_REFLECT_101": bool(np.array_equal(cv2.blur(img, (15, 15)), u8)),
        "cv2.blur 输出 8 位时，内部像素与浮点均值的最大差": round(float(np.abs(ref7 - u8)[7:-7, 7:-7].max()), 3),
    }

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.0), layout="constrained")
    rn = [r for r, v in zip(rs, t["naive"]) if v is not None]
    ax[0].plot(rn, [v for v in t["naive"] if v is not None], "^-", color=S.RED, lw=1.8, ms=5, label="朴素：每个像素把窗口加一遍")
    ax[0].plot(rs, t["separable"], "s-", color=S.ORANGE, lw=1.8, ms=5, label="可分离：先行后列")
    ax[0].plot(rs, t["integral"], "o-", color=S.BLUE, lw=1.8, ms=5, label="积分图：四个角一加一减")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xticks(rs, [str(r) for r in rs])
    ax[0].set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    ax[0].minorticks_off()
    ax[0].set_xlabel("半径 r（窗口边长 2r+1）")
    ax[0].set_ylabel("耗时（ms，1000 x 1000，-O2）")
    ax[0].set_title("只有积分图的耗时不随半径变化")
    ax[0].legend(fontsize=9, loc="upper left")
    S.clean(ax[0])

    r = 7
    row = np.concatenate([40 + 6.0 * np.arange(25), np.full(35, 184.0)])
    strip = np.tile(row, (41, 1)).astype(np.uint8)
    k = 2 * r + 1
    curves = [("裁剪窗口（本章作业的契约）", clipped_box_mean(strip, r)[20], S.BLUE, "o-"),
              ("反射 101（OpenCV 默认）", cv2.boxFilter(strip, cv2.CV_64F, (k, k), borderType=cv2.BORDER_REFLECT_101)[20], S.GREEN, "s--"),
              ("补零", cv2.boxFilter(strip, cv2.CV_64F, (k, k), borderType=cv2.BORDER_CONSTANT)[20], S.RED, "^:")]
    xs = np.arange(22)
    ax[1].plot(xs, row[:22], color=S.MUTED, lw=1.2, label="原始的一行（左边界处是一段斜坡）")
    for name, vals, color, style in curves:
        ax[1].plot(xs, vals[:22], style, color=color, lw=1.6, ms=4, label=name)
    ax[1].axvline(r, color=S.MUTED, lw=1, ls="--")
    ax[1].text(r + 0.3, 12, f"x = r = {r}", fontsize=9, color=S.MUTED)
    ax[1].set_ylim(0, 190)
    ax[1].set_xlabel("离左边界的距离 x（px）")
    ax[1].set_ylabel(f"{k} x {k} 盒式均值")
    ax[1].set_title("三种边界约定只在离边 r 以内不一样")
    ax[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    S.clean(ax[1])
    NUM["边界策略"]["图 5 右栏的例子（15x15 窗口，x=0 处的均值）"] = {name.split("（")[0]: round(float(vals[0]), 1) for name, vals, _, _ in curves}
    S.save(fig, CH, "fig05_积分图与边界")


# ---------------------------------------------------------------- NCC 的耗时和 NaN
def py_best(fn, reps=10):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def exp_ncc():
    sides = [8, 16, 24, 32, 48]  # 和 bench.cpp 里 ncc、ncc_check 两个模式用的边长相同
    rng = np.random.default_rng(4)
    img = rng.integers(0, 256, (240, 320), dtype=np.uint8)
    tpls = {t: img[100:100 + t, 150:150 + t].copy() for t in sides}
    for t in sides:
        out = cv2.matchTemplate(img, tpls[t], cv2.TM_CCOEFF_NORMED)
        assert np.unravel_index(int(out.argmax()), out.shape) == (100, 150)

    def measure():
        res, = timed([("bench_O2", "ncc")], rounds=2)
        assert [int(k) for k in res] == sides
        cv_ms = [py_best(lambda tpl=tpls[t]: cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)) for t in sides]
        direct = [res[str(t)]["direct"] for t in sides]
        integ = [res[str(t)]["integral"] for t in sides]
        return {
            "说明": "320x240 的 8 位随机图，模板是从图里 (100, 150) 处截下的 t x t 小块，-O2，耗时 ms。直接计算是参考实现；"
                  "“分母用积分图”是本脚本内嵌的对照实现，窗口的和与平方和查积分图，分子仍逐像素相乘；"
                  "OpenCV 一栏是从 Python 调用 cv2.matchTemplate(TM_CCOEFF_NORMED)，限制为单线程，图是另一张同尺寸随机图",
            "模板边长 t": sides,
            "直接计算": [sig(v) for v in direct],
            "分母用积分图": [sig(v) for v in integ],
            "OpenCV matchTemplate": [sig(v) for v in cv_ms],
            "分母用积分图相对直接计算快的倍数": [round(a / b, 1) for a, b in zip(direct, integ)],
            "OpenCV 相对直接计算快的倍数": [round(a / b) for a, b in zip(direct, cv_ms)],
            "t 从最小到最大，直接计算的耗时增长倍数": round(direct[-1] / direct[0]),
            "t 从最小到最大，模板面积增长倍数": round((sides[-1] / sides[0]) ** 2),
            "两种 C++ 实现找到的位置是否都相同": None,
            "两种 C++ 实现的分数最大相差": None,
        }

    num = pin("NCC 耗时", measure)
    same = run_json("bench_O2", "ncc_check")  # 不是耗时，每次运行都重新核对
    assert [int(k) for k in same] == sides and all(same[str(t)]["y"] == 100 and same[str(t)]["x"] == 150 for t in sides)
    NUM["NCC 耗时"] = {**num, "两种 C++ 实现找到的位置是否都相同": all(same[str(t)]["same_pos"] for t in sides),
                     "两种 C++ 实现的分数最大相差": float(f"{max(same[str(t)]['score_diff'] for t in sides):.1g}")}
    flat = cv2.matchTemplate(np.full((10, 10), 5, np.uint8), np.full((3, 3), 5, np.uint8), cv2.TM_CCOEFF_NORMED)
    chk = run_json("checks")
    NUM["NaN 与最大值"] = {
        "说明": "四个分数里有一个是 NaN，真正的最大值 0.9 在下标 2。C++ 用 -O2 编译的小程序实测，numpy 用 np.argmax",
        "NaN 在中间（下标 1）时各写法返回的下标": {"初始化成 -2 再用 > 更新": chk["nan_mid"]["init_low"], "用第一个元素初始化再用 > 更新": chk["nan_mid"]["init_first"],
                                     "std::max_element": chk["nan_mid"]["max_element"], "np.argmax": int(np.argmax([0.2, np.nan, 0.9, 0.5]))},
        "NaN 在开头（下标 0）时各写法返回的下标": {"初始化成 -2 再用 > 更新": chk["nan_head"]["init_low"], "用第一个元素初始化再用 > 更新": chk["nan_head"]["init_first"],
                                     "std::max_element": chk["nan_head"]["max_element"], "np.argmax": int(np.argmax([np.nan, 0.2, 0.9, 0.5]))},
        "全部是 NaN 时“初始化成 -2”写法返回的下标": chk["nan_all_init_low"],
        "OpenCV 5.0 的 matchTemplate 在全平坦图上给出的分数": round(float(flat.max()), 3),
    }


# ---------------------------------------------------------------- 图 6：构建配置与访问方式
def fig_build():
    labels = {"O0": "-O0", "O1": "-O1", "O2": "-O2", "O3": "-O3", "san": "-O1 加 ASan 和 UBSan"}
    funcs = {"integral": "积分图 2000x2000", "box_mean": "盒式均值 2000x2000，r=7", "ncc": "NCC 320x240，模板 24x24", "blobs": "连通域计数 2000x2000"}
    names = {"at": "at()，带边界检查", "op": "operator()，不检查", "local": "把 data 指针和宽度取到局部变量再算下标", "ptr": "三个行指针"}
    cfg = {"O2": "-O2", "novec": "-O2 但关掉自动向量化", "O0": "-O0"}

    def measure_opt():
        res = dict(zip(labels, timed([(f"bench_{k}", "funcs", 2) for k in labels])))
        ratio = {f: res["O0"][f] / res["O2"][f] for f in funcs}
        san_ratio = {f: res["san"][f] / res["O2"][f] for f in funcs}
        return {
            "说明": "参考实现的四个函数，同一份源码只换编译选项，耗时 ms。CMake 的 Debug 在 GCC 和 Clang 上默认不带优化（相当于 -O0 加 -g），Release 默认是 -O3",
            "函数": list(funcs.values()),
            **{labels[k]: [sig(res[k][f]) for f in funcs] for k in labels},
            "-O0 比 -O2 慢的倍数": [round(ratio[f], 1) for f in funcs],
            "-O0 比 -O2 慢的倍数的范围": [round(min(ratio.values()), 1), round(max(ratio.values()), 1)],
            "开 Sanitizer 比 -O2 慢的倍数": [round(san_ratio[f], 1) for f in funcs],
            "-O3 与 -O2 的耗时之比": [round(res["O3"][f] / res["O2"][f], 2) for f in funcs],
            "连通域个数（各配置应相同）": None,
        }

    def measure_access():
        acc = dict(zip(cfg, timed([(f"bench_{k}", "access") for k in cfg])))
        return {
            "说明": "2000x2000 的 8 位随机图做 3x3 均值，输出 8 位，四种取像素的写法，耗时 ms",
            **{cfg[k]: {names[m]: sig(acc[k][m]) for m in names} for k in cfg},
            "-O2 下 at() 比 operator() 慢": f"{100 * (acc['O2']['at'] / acc['O2']['op'] - 1):.0f}%",
            "-O2 下 operator() 比行指针慢的倍数": round(acc["O2"]["op"] / acc["O2"]["ptr"], 1),
            "关掉向量化后 operator() 比行指针慢的倍数": round(acc["novec"]["op"] / acc["novec"]["ptr"], 1),
            "-O0 下 at() 比行指针慢的倍数": round(acc["O0"]["at"] / acc["O0"]["ptr"], 1),
        }

    def measure_copy():
        copy = dict(zip(("O2", "O0"), timed([(f"bench_{k}", "copy") for k in ("O2", "O0")])))
        assert max(copy["O2"][k]["ref"] for k in ("2000", "4000")) < 0.001
        return {
            "说明": "把一张 8 位图传给一个只读一个像素的函数，每次调用的耗时 ms；按值传会拷贝整张图",
            "2000x2000（4 MB）": {"按值，-O2": sig(copy["O2"]["2000"]["value"]), "按 const 引用，-O2": "低于 0.001", "按值，-O0": sig(copy["O0"]["2000"]["value"])},
            "4000x4000（16 MB）": {"按值，-O2": sig(copy["O2"]["4000"]["value"]), "按 const 引用，-O2": "低于 0.001", "按值，-O0": sig(copy["O0"]["4000"]["value"])},
        }

    def measure_alloc():
        alloc, = timed([("bench_O2", "alloc")])
        return {
            "说明": "-O2，耗时 ms。600x600 图做 5x5 中值，窗口缓冲区是一个 std::vector；另一项是往 vector 里 push_back 400 万个 std::pair<int, int>",
            "中值：vector 在循环里新建，不 reserve": sig(alloc["median_in_loop"]),
            "中值：vector 在循环里新建，先 reserve(25)": sig(alloc["median_in_loop_reserve"]),
            "中值：vector 提到循环外复用": sig(alloc["median_hoisted"]),
            "中值：不 reserve 比提到循环外慢的倍数": round(alloc["median_in_loop"] / alloc["median_hoisted"], 2),
            "中值：循环里 reserve 比提到循环外慢": f"{100 * (alloc['median_in_loop_reserve'] / alloc['median_hoisted'] - 1):.0f}%",
            "push_back 400 万次：不 reserve": sig(alloc["push_no_reserve"]),
            "push_back 400 万次：先 reserve": sig(alloc["push_reserve"]),
            "push_back：不 reserve 慢的倍数": round(alloc["push_no_reserve"] / alloc["push_reserve"], 2),
        }

    opt = pin("优化级别", measure_opt)
    counts = sorted({run_json(f"bench_{k}", "blob_count")["blob_count"] for k in labels})  # 不是耗时，每次运行都重新数
    NUM["优化级别"] = opt = {**opt, "连通域个数（各配置应相同）": counts}
    access = pin("访问方式", measure_access)
    pin("按值传参", measure_copy)
    pin("循环里分配内存", measure_alloc)

    # 图 6 只用写进 JSON 的值。相对 -O2 的倍数优先用 JSON 里已经算好的，-O1 没有单独记，用两个耗时相除
    rel = {"O0": opt["-O0 比 -O2 慢的倍数"], "san": opt["开 Sanitizer 比 -O2 慢的倍数"],
           "O1": [a / b for a, b in zip(opt["-O1"], opt["-O2"])], "O3": opt["-O3 与 -O2 的耗时之比"]}
    lo, hi = opt["-O0 比 -O2 慢的倍数的范围"]
    claim(2 <= lo < 10 and 30 <= hi < 40, "不开优化慢几倍到三十几倍，各函数差得很远")
    acc = {k: {m: access[cfg[k]][names[m]] for m in names} for k in cfg}
    claim(acc["O2"]["at"] < 1.3 * acc["O2"]["op"] and acc["O2"]["op"] > 3 * acc["O2"]["ptr"], "-O2 下 at() 和 operator() 接近，行指针快得多")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1]}, layout="constrained")
    order = ["O0", "san", "O1", "O3"]
    colors = {"O0": S.RED, "san": S.PURPLE, "O1": S.ORANGE, "O3": S.GREEN}
    xs = np.arange(len(funcs))
    for i, k in enumerate(order):
        vals = rel[k]
        bars = ax[0].bar(xs + (i - 1.5) * 0.2, vals, 0.2, color=colors[k], alpha=0.88, label=labels[k])
        for b, v in zip(bars, vals):
            ax[0].text(b.get_x() + b.get_width() / 2, v * 1.08, two(v), ha="center", fontsize=8,
                       bbox={"fc": "white", "ec": "none", "pad": 0.6})
    ax[0].set_yscale("log")
    ax[0].set_ylim(0.6, 80)
    ax[0].set_yticks([1, 3, 10, 30], ["1", "3", "10", "30"])
    ax[0].minorticks_off()
    ax[0].set_xticks(xs, ["积分图", "盒式均值", "NCC", "连通域计数"])
    ax[0].set_ylabel("耗时相对 -O2 的倍数（对数轴）")
    ax[0].set_title("不开优化慢几倍到三十几倍，各函数差得很远")
    ax[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    S.clean(ax[0])
    ax[0].grid(axis="x", visible=False)

    methods = [("at", "at()", S.RED), ("op", "operator()", S.ORANGE), ("ptr", "行指针", S.BLUE)]
    groups = ["O2", "novec", "O0"]
    xs = np.arange(len(groups))
    for i, (m, name, color) in enumerate(methods):
        vals = [acc[g][m] for g in groups]
        bars = ax[1].bar(xs + (i - 1) * 0.26, vals, 0.26, color=color, alpha=0.88, label=name)
        for b, v in zip(bars, vals):
            ax[1].text(b.get_x() + b.get_width() / 2, v * 1.1, two(v), ha="center", fontsize=8.5)
    ax[1].set_yscale("log")
    ax[1].set_ylim(0.4, 500)
    ax[1].set_yticks([1, 10, 100], ["1", "10", "100"])
    ax[1].minorticks_off()
    ax[1].set_xticks(xs, ["-O2", "-O2\n关掉向量化", "-O0"])
    ax[1].set_ylabel("3 x 3 均值耗时（ms，对数轴）")
    ax[1].set_title("-O2 下 at() 和 operator() 接近，行指针快得多")
    ax[1].legend(fontsize=9, loc="upper left", ncol=3)
    S.clean(ax[1])
    ax[1].grid(axis="x", visible=False)
    S.save(fig, CH, "fig06_优化级别与访问方式")


# ---------------------------------------------------------------- Python 对拍、OpenCV 对照、工具能抓住什么
def exp_python():
    rng = np.random.default_rng(5)
    small = rng.integers(0, 256, (500, 500), dtype=np.uint8)
    px = small.tolist()

    def pure_python():
        H, W = len(px), len(px[0])
        ii = [[0] * (W + 1) for _ in range(H + 1)]
        for y in range(H):
            row = 0
            above, cur, src = ii[y], ii[y + 1], px[y]
            for x in range(W):
                row += src[x]
                cur[x + 1] = above[x + 1] + row
        return ii

    assert pure_python()[-1][-1] == int(small.sum())
    big = rng.integers(0, 256, (2000, 2000), dtype=np.uint8)
    binary = (rng.integers(0, 256, (2000, 2000)) < 100).astype(np.uint8)
    im = rng.integers(0, 256, (240, 320), dtype=np.uint8)
    tpl = im[100:124, 150:174].copy()
    names = {"integral": "积分图 2000x2000（cv2.integral，输出 64 位浮点）", "box_mean": "盒式均值 2000x2000，r=7（cv2.boxFilter，输出 64 位浮点）",
             "ncc": "NCC 320x240，模板 24x24（cv2.matchTemplate）", "blobs": "连通域 2000x2000（cv2.connectedComponents，同时输出标签图）"}

    def measure_opencv():
        funcs, = timed([("bench_O2", "funcs", 2)])  # 和“优化级别”的 -O2 一行是同一组测量
        cv_ms = {"integral": py_best(lambda: cv2.integral(big, sdepth=cv2.CV_64F)),
                 "box_mean": py_best(lambda: cv2.boxFilter(big, cv2.CV_64F, (15, 15))),
                 "ncc": py_best(lambda: cv2.matchTemplate(im, tpl, cv2.TM_CCOEFF_NORMED)),
                 "blobs": py_best(lambda: cv2.connectedComponents(binary, connectivity=8), 5)}
        return {
            "说明": "同样尺寸的随机图。参考实现用 -O2 编译后在 C++ 里计时；OpenCV 是 Python 包 opencv-python " + cv2.__version__ + "，用 cv2.setNumThreads(1) 限制为单线程后从 Python 计时。"
                  "两边的输出类型和边界约定不完全相同，只比较量级。耗时 ms",
            **{names[k]: {"参考实现": sig(funcs[k]), "OpenCV": sig(cv_ms[k]), "参考实现慢的倍数": round(funcs[k] / cv_ms[k], 1)} for k in names},
        }

    def measure_python():
        funcs, = timed([("bench_O2", "funcs", 2)])
        t_py = py_best(pure_python, 3) / small.size * 1e6
        return {
            "说明": "同一个递推，每像素耗时 ns。纯 Python 双重循环在 500x500 上测；numpy 是两次 cumsum，C++ 是参考实现（-O2），后两者在 2000x2000 上测",
            "纯 Python 双重循环": round(t_py),
            "numpy 两次 cumsum（uint64）": round(py_best(lambda: big.astype(np.uint64).cumsum(0).cumsum(1)) / big.size * 1e6, 1),
            "C++ 参考实现": round(funcs["integral"] / big.size * 1e6, 2),
            "纯 Python 比 C++ 慢的倍数": round(t_py / (funcs["integral"] / big.size * 1e6)),
        }

    pin("和 OpenCV 对照", measure_opencv)
    pin("Python 和 C++ 的积分图", measure_python)

    chk = run_json("checks")
    vals = np.array([[0.5, 1.5, 2.5, 254.5]])
    cv_round = cv2.add(vals, np.zeros_like(vals), dtype=cv2.CV_8U).ravel()
    img = rng.integers(0, 256, (2000, 2000), dtype=np.uint8).astype(np.float64)
    m4 = (img[:-1, :-1] + img[1:, :-1] + img[:-1, 1:] + img[1:, 1:]) / 4
    m9 = clipped_box_mean(img[:600, :600], 1)[1:-1, 1:-1]
    color = rng.integers(0, 256, (500, 500, 3), dtype=np.uint8)
    g_bgr = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(int)
    g_rgb = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY).astype(int)
    NUM["对拍时最容易不一致的三类"] = {
        "说明": "均匀随机的 8 位图；取整一项比较的是同一个浮点均值转成整数的不同做法",
        "取整：0.5、1.5、2.5、254.5 的结果": {
            "C++ static_cast（截断）": [chk["round"][k]["cast"] for k in ("0.5", "1.5", "2.5", "254.5")],
            "C++ std::round（四舍五入，.5 远离零）": [int(chk["round"][k]["round"]) for k in ("0.5", "1.5", "2.5", "254.5")],
            "C++ std::nearbyint（默认舍入模式，.5 取偶数）": [int(chk["round"][k]["nearbyint"]) for k in ("0.5", "1.5", "2.5", "254.5")],
            "numpy np.round": [int(v) for v in np.round(vals).ravel()],
            "OpenCV 饱和转换到 8 位（cv2.add 指定 dtype）": [int(v) for v in cv_round],
        },
        "2x2 均值：截断与四舍五入结果不同的像素比例": f"{100 * float((np.floor(m4) != np.floor(m4 + 0.5)).mean()):.1f}%",
        "2x2 均值：四舍五入与取偶结果不同的像素比例": f"{100 * float((np.floor(m4 + 0.5) != np.round(m4)).mean()):.1f}%",
        "3x3 均值：截断与四舍五入结果不同的像素比例": f"{100 * float((np.floor(m9) != np.floor(m9 + 0.5)).mean()):.1f}%",
        "3x3 均值：四舍五入与取偶结果不同的像素比例": f"{100 * float((np.floor(m9 + 0.5) != np.round(m9)).mean()):.1f}%",
        "通道顺序：同一张彩色图按 BGR 和按 RGB 转灰度的平均绝对差": round(float(np.abs(g_bgr - g_rgb).mean()), 1),
        "通道顺序：最大差": int(np.abs(g_bgr - g_rgb).max()),
        "边界": "见“边界策略”一组",
    }


def exp_tools():
    src = (BUILD / "checks.cpp").read_text(encoding="utf-8").splitlines()
    tags = {i + 1: m.group(1) for i, line in enumerate(src) if (m := re.search(r"// TAG:(\w+)", line))}
    cn = {"sign_compare": "int 和 size() 比较", "u8_wrap": "uint8 相加后存回 uint8", "size_t_underflow": "size_t 的 0 - 1",
          "int_overflow": "int 乘法溢出", "u32_wrap": "uint32 累加回绕", "float_eq": "浮点数用 == 比较"}

    def hit_lines(text, pattern):
        return {tags[int(m.group(1))] for m in re.finditer(pattern, text) if int(m.group(1)) in tags}

    def warn(flags):
        res = subprocess.run([CXX, "-std=c++17", "-fsyntax-only", *flags, "-I", str(INC), str(BUILD / "checks.cpp")], capture_output=True, text=True)
        return hit_lines(res.stderr, r"checks\.cpp:(\d+):\d+: warning")

    w_basic = warn(["-Wall", "-Wextra"])
    w_more = warn(["-Wall", "-Wextra", "-Wconversion", "-Wfloat-equal"])
    ub = hit_lines(run("checks_ubsan").stderr, r"checks\.cpp:(\d+):\d+: runtime error")
    ub_ext = hit_lines(run("checks_ubsan_ext").stderr, r"checks\.cpp:(\d+):\d+: runtime error")
    yes = lambda s, k: "报" if k in s else "不报"  # noqa: E731
    env = {"ASAN_OPTIONS": "abort_on_error=0"}
    asan = [run("asan_demo", i, env=env) for i in (0, 1, 2)]
    oob_plain = run_tests("v_oob")
    oob_asan = run("v_oob_asan", env=env)
    where = re.search(r"#0 .*? in (\S+)\(.*?(\w+\.cpp):(\d+)", oob_asan.stderr)
    NUM["工具能抓住什么"] = {
        "说明": "同一个小程序里放六种典型的整数和浮点错误，看编译警告和运行时检查各报不报；编译器见“计时条件”。"
              "扩展检查指 -fsanitize=undefined 之外再加 unsigned-integer-overflow 和 implicit-conversion（Clang 才有）",
        "错误": [cn[k] for k in cn],
        "-Wall -Wextra": [yes(w_basic, k) for k in cn],
        "再加 -Wconversion -Wfloat-equal": [yes(w_more, k) for k in cn],
        "运行时 -fsanitize=undefined": [yes(ub, k) for k in cn],
        "运行时再加两项扩展检查": [yes(ub_ext, k) for k in cn],
        "六种里 -Wall -Wextra 报出的个数": len(w_basic),
        "越界读（4 行 6 列的图，用不检查的 operator() 读第 6 列）": {
            "第 0 行：ASan 是否报错": "heap-buffer-overflow" in asan[0].stderr,
            "第 0 行：读到的值（第 1 行第 0 列存的是 100）": int(asan[0].stdout.split()[-1]),
            "最后一行：ASan 是否报错": "heap-buffer-overflow" in asan[1].stderr,
            "同样的下标用 at() 访问": asan[2].stdout.strip(),
        },
        "把参考实现的 y1 改成 y + r + 2 之后": {
            "不开 Sanitizer 时程序是否崩溃": False if oob_plain else True,
            "不开 Sanitizer 时盒式均值一项的测试结果": oob_plain.get("盒式均值", "没有输出"),
            "开 ASan 后的报告类型": "heap-buffer-overflow" if "heap-buffer-overflow" in oob_asan.stderr else "没有报告",
            "报告是否指到出错的函数和行号": bool(where and "box_mean" in where.group(1)),
        },
    }
    chk = run_json("checks")
    NUM["负尺寸构造"] = {"说明": "cpp/image.hpp 的构造函数先用 rows*cols 分配内存，再检查负数",
                     "Image(-1, 5) 抛出的异常": "std::" + chk["image_neg1_5"], "Image(-2, -3) 抛出的异常": "std::" + chk["image_neg2_neg3"]}

    timing_msg = "r 增大时耗时明显增长"
    by_timing = ("盒式均值写成可分离的 O(r)", "盒式均值写成 O(r^2)")  # 这两种改法过不过只取决于测试里的计时检查
    count_key = "可分离的 O(r) 实现再连跑 5 次，“盒式均值”一项通过的次数"

    def summarize(res):
        return ("四项全部 PASS" if all(v == "PASS" for v in res.values()) and len(res) == 4 else
                "；".join(f"{k} {v}" for k, v in res.items() if v != "PASS"))

    def stable_tests(exe):
        # 这几种改法没有动盒式均值的复杂度，它们的结果不取决于耗时。测试里的计时检查万一因为机器一时卡顿没过，重跑
        for _ in range(3):
            res = run_tests(exe)
            if not any(timing_msg in v for v in res.values()):
                break
        return res

    def test_timing(exe, n=5):
        runs = [run_json(exe) for _ in range(n)]
        small, large = sorted(r["small_ms"] for r in runs)[n // 2], sorted(r["large_ms"] for r in runs)[n // 2]
        return {"r=1 的耗时": sig(small), "r=30 的耗时": sig(large), "r=30 是 r=1 的几倍": round(large / small, 1),
                "判据允许的 r=30 耗时上限": sig(small * 4 + 50.0), f"{n} 次里按判据算通过的次数": int(sum(r["pass"] for r in runs))}

    def measure_timing_check():
        return {
            "说明": "按 tests/test_main.cpp 里盒式均值计时检查的写法单独计时：2000x2000 的随机图，r=1 和 r=30 各调用一次，不重复、不预热；"
                  "判据是 r=30 的耗时大于 r=1 耗时的 4 倍再加 0.05 秒才判失败。每种实现跑 5 次，耗时取中位数（ms），-O2",
            "积分图 O(1)（参考实现）": test_timing("testtime_ref"),
            "可分离 O(r)": test_timing("testtime_sep"),
            "朴素 O(r^2)": test_timing("testtime_naive", 3),
        }

    def measure_gaps():
        out = {name: summarize(run_tests(VARIANTS[name][0])) for name in by_timing}
        out[count_key] = int(sum(run_tests("v_sep").get("盒式均值") == "PASS" for _ in range(5)))
        return out

    pin("测试的计时检查", measure_timing_check)
    timed_gaps = pin("测试能抓住什么", measure_gaps)
    NUM["测试能抓住什么"] = {
        "说明": "把参考实现改一处得到几种常见写法，用 -O2 编译后跑 tests/test_main.cpp。改动只存在于 /tmp，仓库里的作业、测试和参考实现都没有动。"
              "两种改了盒式均值复杂度的写法过不过只取决于测试里的计时检查，结果会随机器变化，和耗时一样钉住；其余几种每次运行都重新跑",
        **{name: timed_gaps[name] if name in by_timing else summarize(stable_tests(exe)) for name, (exe, _) in VARIANTS.items()},
        count_key: timed_gaps[count_key],
    }
    NUM["文档值（没有实测）"] = {
        "说明": "讲义里提到、但没有条件在这台 Mac 上验证的数字，来自厂商文档，用之前以自己机器上的实测为准",
        "MSVC 链接器默认给主线程保留的栈（MB）": 1,
        "64 位 Windows 上 long 的位数": 32,
        "64 位 macOS 和 Linux 上 long 的位数": 64,
        "CMake 在 GCC 和 Clang 下的默认编译选项": {"Debug": "-g", "Release": "-O3 -DNDEBUG"},
    }


def two_sig_view() -> dict:
    """钉住的分组里不小于 100 的耗时和倍数，取两位有效数字后的写法。讲义里只写两位有效数字，这样每个数都能在 JSON 里找到。"""
    out: dict = {}

    def walk(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k != "说明" and "连通域个数" not in k:
                    walk(v, path + [k])
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, path + [f"第 {i + 1} 个"])
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool) and abs(obj) >= 100:
            v2 = sig(obj, 2)
            if v2 != obj:
                out[" / ".join(path)] = int(v2)

    for key in PINNED_KEYS:
        walk(NUM[key], [key])
    return out


def main() -> None:
    t0 = time.perf_counter()
    S.setup()
    cv2.setNumThreads(1)
    build_all()
    print(f"编译完成，用时 {time.perf_counter() - t0:.0f} 秒", flush=True)
    exp_conditions()
    for fn in (fig_layout, fig_traverse, fig_overflow, fig_recursion, fig_box, exp_ncc, fig_build, exp_python, exp_tools):
        fn()
        print("完成", fn.__name__, f"{time.perf_counter() - t0:.0f} 秒", flush=True)
    NUM["两位有效数字的写法"] = {
        "说明": "讲义里的耗时和由耗时算出的倍数只写两位有效数字。上面各组里不小于一百的耗时和倍数，取两位有效数字后和记下的值不一样，"
              "这里按“分组 / 条目”列出讲义里的写法，由脚本从钉住的值换算，不是另一次测量",
        **two_sig_view(),
    }
    out = S.MAIN / CH / "figures" / "实测数字.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NUM, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(NUM, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
