# 第 25 章：Transformer 与 ViT

本章用时：速通 **3 小时**，标准 **10 小时**，深挖 **16 小时**（深挖档含在自己的显卡上加载预训练的 ViT），各档做什么见教程第 0 节。目标：亲手实现切块、缩放点积注意力、多头注意力和二维正余弦位置编码，用它们拼出一个很小的 ViT；量出位置编码、LayerNorm 的位置、训练样本数、平移和分辨率各自带来的影响；知道高分辨率、小缺陷的检测图直接用 ViT 难在哪、常见的折中是什么。

## 为什么重要

2020 年以后的大多数视觉模型都建在 Transformer 上：第 26 章的 MAE 和 DINOv2、第 27 章的 SAM、第 28 章的视觉语言模型，图像这一侧常用的骨干都是 ViT 或它的变体。读这些模型的论文和代码时，“标记、注意力、位置编码、类别标记、前置归一化”是默认你已经会的词。对检测方向来说还有一层：ViT 把图像切成固定大小的块，计算量随块数的平方增长，这两点和“图很大、缺陷很小”正好冲突，用之前要知道代价在哪。

## 学习顺序

1. 复习第 9 章（卷积、感受野、平移等变、残差连接、显存怎么量）。
2. 阅读 `教程_Transformer与ViT.md`，读到第 3 节时把 `../demos/自注意力.html` 的五步试一遍。
3. 实现 `25-1_ViT核心零件.py`，运行验收：`python tests/check_progress.py`。
4. 运行 `实验_小ViT.ipynb`：用你写的零件拼一个小 ViT，在代码生成的小图上和小 CNN 对比，做位置编码、注意力图和耗时的实验。全部在 CPU 上几分钟内跑完。
5. 做教程第 14 节的破坏实验，前三个必做。
6. 深挖档：按教程第 10 节在自己的机器上加载 torchvision 的 `vit_b_16` 预训练权重，取出中间特征和注意力，量一次显存峰值。
7. 在 `实验记录.md` 写下：小 ViT 和小 CNN 在不同训练样本数下的准确率，打乱块之后哪个任务掉了，以及一个失败案例。

## 文件

| 文件 | 用途 |
|---|---|
| `教程_Transformer与ViT.md` | 切块、自注意力、位置编码、编码块、和 CNN 的对比、注意力图、分辨率的代价、工业检测里的用法、怎样用真模型 |
| `figures/` | 教程配图和 `实测数字.json`（教程里所有数字的来源） |
| `../demos/自注意力.html` | 交互演示：6 个二维的标记，拖动查询和键看注意力权重，调温度、头数和掩码，浏览器直接打开 |
| `实验_小ViT.ipynb` | 用作业里的零件拼小 ViT，做对比实验 |
| `25-1_ViT核心零件.py` | 作业：切块、还原、缩放点积注意力（带掩码）、多头注意力的前向、二维正余弦位置编码 |
| `tests/check_progress.py` | 自动验收 |
| `实验记录.md` | 实验记录模板 |
| `data/README.md` | 数据说明 |

## 通过标准

- 5 项测试 PASS。
- 能画出 ViT 的数据流：切块 → 线性嵌入 → 加位置编码（可选类别标记）→ 若干个“注意力加 MLP”的编码块 → 汇聚 → 分类头，并标出每一步张量的形状。
- 能说清自注意力本身不知道标记的顺序，位置信息从哪里来，去掉位置编码之后哪类任务不受影响、哪类任务退回随机猜。
- 能解释为什么训练样本少的时候小 CNN 比同样大小的小 ViT 好，以及输入边长加倍时标记数和注意力权重各变成几倍。
- 能说出把 ViT 用在高分辨率、小缺陷的检测图上的三个困难和两种常见的折中。

## 本章额外需要的包

作业、验收和 Notebook 只需要课程环境里已有的 numpy、torch、matplotlib、nbformat。教程第 10 节用到 torchvision（课程环境里已有，预训练权重要联网下载一次）；timm 是可选的，只在第 10 节末尾和扩展里提到，需要自己安装。安装方法见 `docs/环境安装.md`。

## 推荐资源

只列论文题目和官方来源的名字，网址请自己搜索确认。

- 论文：Vaswani 等，Attention Is All You Need（NeurIPS 2017）。Transformer 的出处，缩放点积注意力和多头注意力的定义在这里。
- 论文：Dosovitskiy 等，An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale（ICLR 2021），即 ViT。
- 论文：Touvron 等，Training data-efficient image transformers & distillation through attention（ICML 2021），即 DeiT，讲只用 ImageNet 怎样把 ViT 训练好。
- 论文：Liu 等，Swin Transformer: Hierarchical Vision Transformer using Shifted Windows（ICCV 2021）。分层结构和窗口注意力。
- 论文：Abnar 和 Zuidema，Quantifying Attention Flow in Transformers（ACL 2020）。注意力展开的出处。
- PyTorch 官方文档里 `torch.nn.MultiheadAttention` 和 `torch.nn.functional.scaled_dot_product_attention` 两页：验收脚本拿前者核对你的多头注意力。
- torchvision 官方文档的 Models 部分：`vit_b_16`、`ViT_B_16_Weights` 和 `torchvision.models.feature_extraction`。
- timm（官方仓库名 `huggingface/pytorch-image-models`）：ViT 各种变体和预训练权重最全的库，可选。
