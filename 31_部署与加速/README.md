# 第 31 章：部署与加速：ONNX、TensorRT、量化

本章用时：速通 **3 小时**，标准 **10 小时**，深挖 **16 小时**（深挖档含在自己的显卡上导出 ONNX、用 ONNX Runtime 对拍、用 TensorRT 构建引擎），各档做什么见教程第 0 节。目标：弄清模型从 PyTorch 到产线推理程序要过的三关（导出、加速、验收）各自的原理和容易出错的地方，亲手实现量化和部署里的五个核心零件，会把设备的节拍换算成对延迟和批大小的要求，并且知道加速之后要按漏检和过检重新验收，而不是只看精度。

## 为什么重要

检测设备的产能是硬指标：扫描速度定了，每张图留给算法的时间就定了。模型在 PyTorch 里验证通过只是第一步，接下来要交给 C++ 的推理程序，在带显卡的工控机上按节拍出结果。半精度、8 位量化、剪枝、蒸馏都能让模型变快，但每一种都会让输出的分数挪动一点；缺陷检测的阈值卡在分布的尾巴上，分数挪一点，过检或漏检就可能变成原来的几倍，而分类精度和 AUROC 几乎看不出来。会开一个加速开关不难，难的是知道它改变了什么、怎样验收、出了问题往哪一步查。

## 学习顺序

1. 复习第 14 章（预处理契约、`eval()`、导出的一致性、推理耗时怎样量）、第 9 章 9.5 节（BatchNorm）、第 19 章第 5 节（漏检约束下的阈值）。
2. 阅读 `教程_部署与加速.md`。
3. 实现 `31-1_量化与部署零件.py`，运行验收。
4. 运行 `实验_部署与加速.ipynb`：用你写的零件在一个小卷积网络上折叠 BatchNorm、做训练后量化、看过检和漏检怎么变，再在自己的设备上量延迟、按节拍选批大小。
5. 做教程第 14 节的破坏实验，前三个必做。
6. 深挖档：按教程第 8 节在自己的机器上导出 ONNX，用 ONNX Runtime 推理并和 PyTorch 对拍，用 TensorRT 构建引擎，写一个最小的 C++ 推理程序。
7. 在 `实验记录.md` 写下：折叠和量化前后的输出差、沿用原阈值和重新定阈值时的过检率和弱缺陷漏检率、自己机器上各批大小的中位数和 P99。

## 文件

| 文件 | 用途 |
|---|---|
| `教程_部署与加速.md` | BatchNorm 折叠与算子融合、量化、半精度、剪枝与蒸馏、延迟与节拍、导出对拍、ONNX 和 TensorRT 怎么用、工业检测里怎样验收 |
| `figures/` | 教程配图和 `实测数字.json`（教程里所有数字的来源） |
| `../demos/量化误差.html` | 交互演示：位数、对称与否、校准方式怎样决定量化误差，浏览器直接打开 |
| `实验_部署与加速.ipynb` | 折叠、训练后量化、半精度的数值现象、延迟统计和按节拍选批大小 |
| `31-1_量化与部署零件.py` | 作业：缩放系数和零点、量化与反量化、逐通道量化权重、BatchNorm 折叠、按节拍选批大小 |
| `tests/check_progress.py` | 自动验收 |
| `实验记录.md` | 实验记录模板 |
| `data/README.md` | 数据说明 |

## 通过标准

- 5 项测试 PASS。
- 能写出均匀量化的缩放系数和零点，说清对称和非对称、逐张量和逐通道各用在哪里、为什么。
- 能写出把 BatchNorm 折叠进卷积的公式，并说明折叠前后的输出为什么不是逐位相同、对拍的容差该怎么定。
- 能解释校准集为什么必须包含缺陷图，以及校准图只有良品时哪些指标看得出问题、哪些看不出。
- 能把节拍换算成“跟得上”和“来得及”两个条件，并说明为什么按 P99 而不是按平均值。
- 能说出加速之后的验收顺序：FP32 对拍、沿用原阈值、重新定阈值、弱缺陷和强缺陷分开看、看翻转的样本。

## 本章额外需要的包

作业、验收、Notebook 的必做部分只用 numpy、torch、matplotlib，不需要额外安装，也不需要显卡。教程第 8 节和 Notebook 的选做单元需要：

- `onnx`、`onnxscript`：导出和检查 ONNX 模型。
- `onnxruntime-gpu`（只用 CPU 时是 `onnxruntime`，两个不要同时装）：用 ONNX Runtime 推理。
- TensorRT：按 NVIDIA 官方安装指南安装，对显卡驱动和 CUDA 的版本有要求。

这些包写课程的机器上都没有装过，第 8 节的内容没有运行过。环境安装见 `docs/环境安装.md`。

## 推荐资源

- PyTorch 官方文档的 `torch.onnx` 和 `torch.export` 两页：导出接口的参数、动态形状的写法、各版本的变化。
- ONNX Runtime 官方文档：Execution Providers 一节（CUDA 和 TensorRT 两个执行提供者的版本要求和选项），以及 C/C++ API 的说明。
- NVIDIA TensorRT 官方文档：Quick Start Guide、C++ API 和 Python API 的入门两页、Work With Dynamic Shapes、Work With Quantized Types、Engine Compatibility。官方仓库 `NVIDIA/TensorRT` 里有示例和 `trtexec` 的源码。
- 论文：Jacob 等，Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference（CVPR 2018）。整数推理和量化感知训练的经典做法。
- 论文：Krishnamoorthi，Quantizing deep convolutional networks for efficient inference: A whitepaper（2018）。逐通道量化、校准和量化感知训练的综述。
- 论文：Nagel 等，A White Paper on Neural Network Quantization（2021）。训练后量化和量化感知训练的流程和排查顺序。
- 论文：Hinton、Vinyals、Dean，Distilling the Knowledge in a Neural Network（2015）。知识蒸馏和温度。
- 论文：Han、Mao、Dally，Deep Compression（ICLR 2016）；Li 等，Pruning Filters for Efficient ConvNets（ICLR 2017）。非结构化剪枝和按滤波器的结构化剪枝。
- 第 14 章和第 15 章的教程：预处理契约、导出一致性、计时方法、Python 和 C++ 的对拍。
