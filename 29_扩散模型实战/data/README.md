# 第 29 章数据说明

作业、验收、Notebook 的必做部分和教程里的全部数字，都只用代码生成的数据，不需要下载任何东西：

- 二维的 6 团点分布（团心在半径 2 的圆上，分成 3 类），用于采样器、参数化和引导的实验。
- 16×16 的竖条纹小图，加斑点或横向划痕两种缺陷，用于掩膜条件、低秩微调、修复和闭环实验。

教程第 8 节和 Notebook 的选做单元要用 Stable Diffusion、ControlNet 或 LoRA 的预训练权重，第一次运行时由 `diffusers` 自动下载到 Hugging Face 的缓存目录（可以用环境变量 `HF_HOME` 改位置），不要放进仓库。权重的大小和许可以各自的模型卡为准；不同权重的许可证不一样，有的限制商用或者限制用途，使用前先读模型卡。

深挖档可以用的公开数据（和第 19 章相同）：

- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)：15 类，像素级标注，许可 CC BY-NC-SA 4.0，仅限非商业用途。
- [VisA](https://github.com/amazon-science/spot-diff)：12 类，含 PCB，许可 CC BY 4.0。

也可以用自己的图像：准备一批良品图和十几张带掩膜的缺陷图，就够做一次“微调、生成、筛选、训练检测器、在留出的真实缺陷上测”的闭环。下载或自备的数据放 `data/external/`，生成的图放 `data/generated/`，都不要提交到仓库。
