# 第 16 章：图像对齐与亚像素配准

本章用时建议 **8–12 小时**。目标：能根据场景选对齐方法，实现模板匹配和相位相关，把精度做到亚像素，并知道周期图案下会怎样失败。

## 为什么重要

半导体检测的核心比对方式（die-to-die、die-to-database）都建立在“先把两张图精确对齐”之上。对齐差 0.3 个像素，在图形边缘上就会产生明显的差值，被当成缺陷报出来。对齐是检测算法里最基础、也最容易被低估的一环。

## 学习顺序

1. 复习第 4 章（几何变换、RANSAC）和第 5 章（特征点）。
2. 阅读 `教程_图像对齐与亚像素.md`。
3. 运行 `实验_图像对齐.ipynb`：对比 SSD、NCC、相位相关在亮度变化和噪声下的表现，复现周期图案的假峰。
4. 实现 `16-1_相位相关与亚像素.py` 的三个函数。
5. 运行 `python3 16_图像对齐与亚像素/tests/check_progress.py`。
6. 在 `实验记录.md` 记录亚像素误差曲线和一个失败案例。

## 文件

| 文件 | 用途 |
|---|---|
| `教程_图像对齐与亚像素.md` | 对齐方法分类、相位相关推导、亚像素与精度评估 |
| `实验_图像对齐.ipynb` | 互动实验 |
| `16-1_相位相关与亚像素.py` | 作业：抛物线亚像素、ZNCC、相位相关 |
| `tests/check_progress.py` | 自动验收 |
| `实验记录.md` | 实验记录模板 |

## 通过标准

- 三项测试 PASS，亚像素误差小于 0.25 px。
- 能说出至少 3 种对齐方法的适用场景和失败条件。
- 能解释周期图案为什么会让相位相关出现多个等高峰，以及工程上怎么规避。
- 知道怎样评估对齐精度（重复性、3σ），而不是只看一两张图。

## 推荐资源

- First Principles of Computer Vision（Shree Nayar）：[Template Matching](https://fpcv.cs.columbia.edu/)、Fourier Transform 和 Convolution Theorem 几节，网站上按目录点进去即可。
- [OpenCV 文档：phaseCorrelate](https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html)，看函数说明里的公式和返回值约定。
- [OpenCV 文档：findTransformECC](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)，了解基于强度的迭代对齐。
- 论文：Guizar-Sicairos 等，Efficient subpixel image registration algorithms（Optics Letters, 2008），工业界常用的上采样细化方法，`skimage.registration.phase_cross_correlation` 的实现依据。
