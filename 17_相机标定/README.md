# 第 17 章：相机标定

本章用时建议 **10–14 小时**。目标：从零实现张正友标定的线性部分，理解畸变模型和非线性优化，知道工业检测里像素到物理尺寸的标定怎么做、精度怎么评估。

## 为什么重要

JD 里单独列了“相机标定算法的设计、开发、测试”。检测设备上要标定的不只是相机内参，还有像素尺寸（每像素多少微米）、相机和运动平台的关系、多相机之间的关系。标定不准，缺陷尺寸、位置和套刻测量全部跟着错。

## 学习顺序

1. 复习第 7 章（针孔模型、投影链路）。
2. 阅读 `教程_相机标定.md`，跟着推导一遍 Zhang 的两个约束。
3. 实现 `17-1_张正友标定.py`，运行验收。
4. 运行 `实验_相机标定.ipynb`：加噪声、减少视图数、让棋盘只占画面一角，观察内参误差怎么变；有 OpenCV 时和 `cv2.calibrateCamera` 对照。
5. （选做）用手机拍 15 张棋盘图，跑一遍真实标定。
6. 在 `实验记录.md` 写下视图数、覆盖范围和误差的关系。

## 文件

| 文件 | 用途 |
|---|---|
| `教程_相机标定.md` | 相机模型、畸变、Zhang 推导、非线性优化、工业标定 |
| `实验_相机标定.ipynb` | 合成相机标定实验 |
| `17-1_张正友标定.py` | 作业：归一化、DLT、闭式内参、外参、重投影误差 |
| `tests/check_progress.py` | 自动验收 |
| `实验记录.md` | 实验记录模板 |

## 通过标准

- 4 项测试 PASS，0.3 px 噪声下 fx 误差小于 3%。
- 能手推 `h1^T B h2 = 0` 和 `h1^T B h1 = h2^T B h2` 两个约束的来源。
- 能解释为什么至少要 3 个不同姿态的平面，以及哪些姿态组合会退化。
- 能说清重投影误差小不等于标定准的原因。
- 知道远心镜头（telecentric）为什么常用于测量，以及它的标定和针孔模型有什么不同。

## 推荐资源

- First Principles of Computer Vision：Camera Calibration 模块（Linear Camera Model、Camera Calibration、Intrinsic and Extrinsic Matrices），[课程网站](https://fpcv.cs.columbia.edu/)；同一内容的[讲义 PDF](https://cave.cs.columbia.edu/Statics/monographs/Camera%20Calibration%20FPCV-4-1.pdf)。
- 原论文：Zhengyou Zhang, A Flexible New Technique for Camera Calibration（IEEE TPAMI, 2000），附录有闭式解公式。
- [OpenCV 相机标定教程](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)：真实图像标定的完整流程。
- [MIT 6.801 Machine Vision（Berthold Horn）](https://ocw.mit.edu/courses/6-801-machine-vision-fall-2020/)：外方位、内方位、光束法平差几讲，偏工业测量视角。
