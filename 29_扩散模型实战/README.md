# 第 29 章：扩散模型实战：SD、ControlNet、LoRA

本章用时：速通 **3 小时**，标准 **12 小时**，深挖 **20 小时**（深挖档含在自己的显卡上跑 Stable Diffusion），各档做什么见教程第 0 节。目标：接着第 20 章的前向过程，弄清扩散模型怎样采样、怎样加条件、怎样用少量样本微调，亲手实现六个核心零件，会在良品图上指定位置生成缺陷，并且能用实验回答“少量真实缺陷微调出来的生成器，样本够不够多样、是不是在复制训练图、拿去训练检测器有没有用”。

## 为什么重要

岗位要求里有“检测过程中用到的图像生成”和“熟悉 GAN/VAE/SD”。第 20 章讲了合成缺陷什么时候帮倒忙，但扩散模型只讲到前向加噪和训练目标，没有生成过一张图。真正用扩散模型造缺陷时，要解决的是另外几件事：采样要多少步、缺陷长在哪里由谁决定、手上只有十几张新缺陷时模型怎么学、生成出来的图标签对不对。Stable Diffusion、ControlNet、LoRA、inpainting 这几个名字各对应其中一件。会调库不难，难的是知道每个开关在改什么、生成的数据什么时候不能用。

## 学习顺序

1. 复习第 20 章 5.3 节（前向过程、噪声表、信噪比、训练目标），本章的符号和它一致。第 13 章第 4 节是更早的入门。
2. 阅读 `教程_扩散模型实战.md`。
3. 实现 `29-1_采样引导与低秩微调.py`，运行验收。
4. 运行 `实验_扩散模型实战.ipynb`：用你写的零件在二维点分布上做采样、参数化和引导，再在 16×16 的小图上做掩膜条件、低秩微调和修复。全部在 CPU 上一两分钟跑完。
5. 做教程第 12 节的破坏实验，前三个必做。
6. 深挖档：按教程第 8 节在自己的机器上跑一次 Stable Diffusion 的文生图和 inpainting，再试 ControlNet 或加载一个 LoRA。
7. 在 `实验记录.md` 写下：几种采样器在少步时各是什么样，引导强度的权衡曲线，掩膜条件对得上的程度，微调后的复制比例，以及闭环实验里生成样本有没有帮上忙。

## 文件

| 文件 | 用途 |
|---|---|
| `教程_扩散模型实战.md` | 采样、参数化、无分类器引导、空间条件、低秩微调、修复式生成、可控缺陷合成的流程和风险、怎样用真模型 |
| `figures/` | 教程配图和 `实测数字.json`（教程里所有数字的来源） |
| `../demos/采样与引导.html` | 交互演示：二维的几团点，调采样器、步数和引导强度，看样本怎么从噪声走到数据，浏览器直接打开 |
| `实验_扩散模型实战.ipynb` | 二维点分布上的采样与引导，小图上的掩膜条件、低秩微调和修复 |
| `29-1_采样引导与低秩微调.py` | 作业：三种参数化的换算、DDIM 的一步、引导的组合、LoRA 的前向与合并、修复时已知区域的替换 |
| `tests/check_progress.py` | 自动验收 |
| `实验记录.md` | 实验记录模板 |
| `data/README.md` | 数据说明 |

## 通过标准

- 5 项测试 PASS。
- 能写出 DDIM 的一步更新，说清它和 DDPM 的一步差在哪一项，以及步数减少时两者的样本各往哪个方向偏。
- 能说清预测噪声、预测原图、预测 v 三种参数化在噪声最多和最少的两端各有什么数值问题。
- 能解释无分类器引导在训练时和采样时各做了什么，引导强度增大时符合程度、多样性和失真各怎么变。
- 能说清 ControlNet 的旁路为什么要零初始化，LoRA 的增量为什么可以并回原权重，以及只有几张训练图时怎样量出生成器是不是在复制。
- 能说出修复式生成里“已知区域每一步都要替换成加噪后的原图”的理由，以及生成的缺陷图进训练集之前要做的三项检查。

## 本章额外需要的包

作业、验收、Notebook 的必做部分只用 numpy、torch、matplotlib，不需要额外安装。教程第 8 节和 Notebook 的选做单元要用预训练权重：

- `diffusers`、`transformers`、`accelerate`：`pip install diffusers transformers accelerate`。第一次运行要联网下载权重，文件有几个 GB，放在默认缓存目录，不要放进仓库。

环境安装见 `docs/环境安装.md`。模型的许可证各不相同，使用前先读模型卡。

## 推荐资源

- 论文：Ho、Jain、Abbeel，Denoising Diffusion Probabilistic Models（NeurIPS 2020），即 DDPM。
- 论文：Song、Meng、Ermon，Denoising Diffusion Implicit Models（ICLR 2021），即 DDIM。
- 论文：Ho、Salimans，Classifier-Free Diffusion Guidance（2022）。
- 论文：Salimans、Ho，Progressive Distillation for Fast Sampling of Diffusion Models（ICLR 2022）。预测 v 的参数化出自这里。
- 论文：Rombach 等，High-Resolution Image Synthesis with Latent Diffusion Models（CVPR 2022）。Stable Diffusion 的基础。
- 论文：Zhang、Rao、Agrawala，Adding Conditional Control to Text-to-Image Diffusion Models（ICCV 2023），即 ControlNet。
- 论文：Hu 等，LoRA: Low-Rank Adaptation of Large Language Models（ICLR 2022）。
- 论文：Lugmayr 等，RePaint: Inpainting using Denoising Diffusion Probabilistic Models（CVPR 2022）。本章修复式采样里“替换已知区域”的做法出自这一类方法。
- Hugging Face `diffusers` 的官方文档和仓库 `huggingface/diffusers`：各条管线的用法、省显存的开关，以及 `examples/` 目录下的 LoRA 训练脚本。
- 第 20 章推荐的 Hugging Face Diffusers 课程：从零实现 DDPM 到使用 Stable Diffusion，配 Notebook。
