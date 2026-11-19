# 第 28 章：视觉语言模型与开放词汇检测

本章用时：速通 **3 小时**，标准 **10 小时**，深挖 **16 小时**（深挖档含在自己的显卡上跑 CLIP 的预训练权重），各档做什么见教程第 0 节。目标：弄清图文双塔模型怎样把一句话变成一个分类器，提示词的写法影响多少，模型在组合泛化和属性绑定上弱在哪里，把文本向量用到“按文字找目标”上还缺什么；亲手实现零样本分类、检索指标、模态间隙和区域打分五个零件；知道这类模型在缺陷复查里能帮什么忙、不能指望它做什么。

## 为什么重要

检测现场的类别经常变：新产品、新工艺带来新的缺陷类型，一开始一张标注都没有。视觉语言模型把“类别”从训练时定死的编号换成了推理时现写的一句话，零样本分类、按文字检索历史图像、开放词汇检测都建立在这一点上，能看图写描述的多模态大模型也用它当眼睛。会调用不难，难的是知道它什么时候靠不住：提示词换一种写法结果差多少，它有没有读懂句子里谁配谁，没在训练文字里出现过的属性它认不认得，几个像素的缺陷在它的输入分辨率下还剩什么。这一章用一个从零训练的小模型把这些问题逐个量出来。

## 学习顺序

1. 复习第 25 章（Transformer 与 ViT）、第 13 章 5.1 和 5.3 节（对比损失、温度、CLIP 的损失函数）、第 26 章 4.1 节（线性探测、k 近邻、检索）。
2. 阅读 `教程_视觉语言模型.md`，把第 10 节的演示按五步试一遍。
3. 实现 `28-1_零样本分类与图文检索.py`，运行验收。
4. 运行 `实验_视觉语言模型.ipynb`：在 CPU 上从零训练一个图文双塔小模型，用你写的零件做零样本分类、提示词对比、组合泛化、检索、模态间隙、文字查询找目标、少样本适配和“训练句子里没有的属性”八个实验。
5. 做教程第 14 节的破坏实验，前三个必做。
6. 深挖档：按教程第 9 节在自己的机器上加载 CLIP 的预训练权重，把真模型的向量交给你写的函数做零样本分类，再放一个少样本线性探测作参照。
7. 在 `实验记录.md` 写下：几种提示词模板各自的精度，没见过的组合上的精度，属性绑定的二选一结果，三种区域特征的定位精度。

## 文件

| 文件 | 用途 |
|---|---|
| `教程_视觉语言模型.md` | 双塔模型、零样本分类与提示词、组合泛化与属性绑定、模态间隙、开放词汇检测的原理、少样本适配、工业检测里的用法和限制、怎样用真模型 |
| `figures/` | 教程配图和 `实测数字.json`（教程里所有数字的来源） |
| `../demos/零样本分类与提示词.html` | 交互演示：二维平面上拖动文本向量、换模板、调放大倍数、加背景提示，看决策区域怎么变，浏览器直接打开 |
| `实验_视觉语言模型.ipynb` | 从零训练图文双塔小模型，做八个实验 |
| `28-1_零样本分类与图文检索.py` | 作业：多模板平均的零样本分类器、零样本预测、Recall@K、模态间隙的度量、区域打分并取最大 |
| `tests/check_progress.py` | 自动验收 |
| `实验记录.md` | 实验记录模板 |
| `data/README.md` | 数据说明 |

## 通过标准

- 5 项测试 PASS。
- 能画出双塔模型的数据流：图像和句子各过一个编码器，投影到同一个空间，归一化后算余弦，一批里配对的在对角线上。
- 能说清零样本分类器是怎样由句子得到的，多模板平均为什么要归一化两次，放大倍数改变什么、不改变什么。
- 能解释为什么一个分类很准的图文模型可能分不开“红方块在蓝圆左边”和“蓝方块在红圆左边”，以及怎样测出来。
- 能说出按文字找目标时区域特征的几种来源和各自的问题，背景提示解决什么。
- 能说出把通用图文模型用到工业图像上的四个限制（训练数据里没有这类图文对、分辨率、属性绑定、分数的刻度），以及三种可行的辅助用法。

## 本章额外需要的包

作业、验收、Notebook 的必做部分只用 numpy、torch、matplotlib，不需要额外安装。教程第 9 节和 Notebook 的选做单元要用预训练权重，有两条路，任选其一，第一次运行都要联网下载权重：

- OpenCLIP：`pip install open_clip_torch`。
- Hugging Face `transformers`：`pip install transformers`。开放词汇检测模型（OWL-ViT、Grounding DINO）也可以通过它使用。

环境安装见 `docs/环境安装.md`。

## 推荐资源

- 论文：Radford 等，Learning Transferable Visual Models From Natural Language Supervision（ICML 2021），即 CLIP。官方仓库 `openai/CLIP`。
- OpenCLIP：官方仓库 `mlfoundations/open_clip`，仓库的说明里有各个结构和权重的名称与用法。
- Hugging Face `transformers` 文档里的 CLIP、OWL-ViT、OWLv2、Grounding DINO 各页：加载、预处理和后处理的写法以它为准。
- 论文：Liang 等，Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning（NeurIPS 2022）。模态间隙。
- 论文：Yuksekgonul 等，When and why vision-language models behave like bags-of-words, and what to do about it?（ICLR 2023）。属性绑定和词序，以及用难负样本改进。
- 论文：Minderer 等，Simple Open-Vocabulary Object Detection with Vision Transformers（ECCV 2022），即 OWL-ViT。
- 论文：Liu 等，Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection。官方仓库 `IDEA-Research/GroundingDINO`。
- 论文：Zhou 等，Learning to Prompt for Vision-Language Models（IJCV 2022），即 CoOp。提示学习。
- 论文：Jeong 等，WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation（CVPR 2023）。用文字描述正常和异常做异常检测。
- 论文：Liu 等，Visual Instruction Tuning（NeurIPS 2023），即 LLaVA。多模态大模型的一种代表性结构。
