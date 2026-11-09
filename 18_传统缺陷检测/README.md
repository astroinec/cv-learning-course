# 第 18 章：传统缺陷检测

本章用时建议 **10–14 小时**。目标：搭出一条完整的“比对 → 阈值 → 连通域 → 特征 → 统计”传统检测流程，理解工业指标（检出率、误报率、过检、漏检），会用数据选阈值而不是凭感觉。

## 为什么重要

半导体光学检测的主力方法至今仍是比对类方法：拿待检图和参考图（相邻 die、多 die 统计的金模板、或设计数据渲染图）做差，差异超过噪声水平的地方就是候选缺陷。深度学习更多用在后面的缺陷分类和误报过滤上。新人进组，大概率先接触的就是这条流水线。

## 学习顺序

1. 复习第 1 章（直方图、噪声）、第 3 章（形态学、阈值、轮廓）。
2. 阅读 `教程_传统缺陷检测.md`。
3. 运行 `实验_缺陷检测.ipynb`：改变阈值 k，画检出率和误报率的关系曲线；比较“均值加标准差”和“中位数加 MAD”在良品里混入坏图时的表现。
4. 实现 `18-1_金模板比对.py`，运行验收。
5. 在 `实验记录.md` 画出 k 与过检、漏检的关系，写一个误报案例的根因。

## 文件

| 文件 | 用途 |
|---|---|
| `教程_传统缺陷检测.md` | 检测流程、比对策略、阈值与噪声建模、Blob 特征、工业指标 |
| `实验_缺陷检测.ipynb` | 互动实验 |
| `18-1_金模板比对.py` | 作业：鲁棒金模板、连通域、缺陷检测、匹配统计 |
| `tests/check_progress.py` | 自动验收 |
| `实验记录.md` | 实验记录模板 |

## 通过标准

- 4 项测试 PASS，良品零误报。
- 能区分 die-to-die、多 die 金模板、die-to-database 三种比对的优缺点。
- 能解释为什么阈值要按局部噪声自适应，而不是全图一个固定灰度差。
- 能说清过检（overkill）和漏检（escape）的代价为什么不对称。
- 能用曲线而不是单个数字汇报检测性能。

## 推荐资源

- First Principles of Computer Vision：Binary Images 模块（几何属性、二值分割）和 Template Matching，[课程网站](https://fpcv.cs.columbia.edu/)。
- [MIT 6.801 Machine Vision](https://ocw.mit.edu/courses/6-801-machine-vision-fall-2020/)：Lecture 14 讲工业检测中的模板定位与检查（PatQuick 等），能看到传统机器视觉公司的思路。
- [OpenCV 文档：connectedComponentsWithStats](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html)：连通域加统计量的标准接口。
- 数据集：[DAGM 2007](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)（纹理表面缺陷，带弱标注），适合练传统方法。
