# 第 20 章：合成缺陷与生成模型

本章用时建议 **10–16 小时**。目标：会用程序化方法合成带准确标签的缺陷，理解 VAE、GAN、扩散模型的原理和它们在缺陷数据生成里的用法，并且会设计实验证明合成数据确实有用。

## 为什么重要

JD 里有两条和图像生成相关：“检测过程中用到的图像生成”和“熟悉 GAN/VAE/SD”。原因很现实：真实缺陷样本少、类型不全、标注贵，新工艺刚上线时甚至一个缺陷都没有。合成数据是补齐训练数据和测试算法边界的主要手段。

## 学习顺序

1. 复习第 13 章（自编码器、生成与自监督）。
2. 阅读 `教程_合成缺陷与生成模型.md`。
3. 实现 `20-1_缺陷合成与扩散前向.py`，运行验收。
4. 运行 `实验_缺陷合成.ipynb`：在良品纹理上合成划痕和颗粒；看 DDPM 前向加噪的每一步。
5. 做一次对照实验：只用合成缺陷训练一个分类器（或调一个阈值），在第 24 章生成器的“真实”缺陷上测试，记录合成和真实之间的差距。
6. （选做，需要 GPU 或耐心）用 diffusers 跑一次 Stable Diffusion inpainting，在良品图上指定区域生成缺陷。

## 文件

| 文件 | 用途 |
|---|---|
| `教程_合成缺陷与生成模型.md` | 合成方法分类、VAE/GAN/扩散模型、可控生成、效果验证 |
| `实验_缺陷合成.ipynb` | 互动实验 |
| `20-1_缺陷合成与扩散前向.py` | 作业：划痕掩码、混合、噪声调度、前向加噪、重参数化 |
| `tests/check_progress.py` | 自动验收 |
| `实验记录.md` | 实验记录模板 |

## 通过标准

- 4 项测试 PASS。
- 能写出 DDPM 前向公式 `x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε`，并解释训练目标为什么是预测噪声。
- 能说清 VAE、GAN、扩散模型各自的优缺点和在缺陷生成里的适用场景。
- 能设计“合成数据是否有用”的对照实验，并说明 FID 为什么不够。

## 推荐资源

- [Hugging Face Diffusers 课程](https://huggingface.co/learn/diffusion-course)：从零实现 DDPM 到使用 Stable Diffusion，配 Notebook。
- [Diffusers 文档：Inpainting](https://huggingface.co/docs/diffusers/using-diffusers/inpaint)：在指定掩码区域生成内容。
- 论文：Ho 等，Denoising Diffusion Probabilistic Models（2020）；Kingma 和 Welling，Auto-Encoding Variational Bayes（2013）；Zhang 等，ControlNet（2023）。
- 论文：Zavrtanik 等，DRAEM（ICCV 2021）；Li 等，CutPaste（CVPR 2021），两种用合成异常训练检测器的经典方法。
- [OpenCV 文档：seamlessClone](https://docs.opencv.org/4.x/df/da0/group__photo__clone.html)：泊松融合。
