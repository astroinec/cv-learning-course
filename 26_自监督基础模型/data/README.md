# 第 26 章数据说明

作业、验收、Notebook 的必做部分和教程里的全部数字，都只用代码生成的条纹小图（32×32，方向 4 种、周期 3 种，相位、幅度、亮度随机），不需要下载任何数据。

教程第 6 节和 Notebook 的选做单元要用 DINOv2 的预训练权重，第一次运行时由 `torch.hub` 或 Hugging Face `transformers` 自动下载到各自的缓存目录，不要放进仓库。权重的大小和许可以官方仓库 `facebookresearch/dinov2` 和模型卡为准。

深挖档可以用的公开数据（和第 19 章相同）：

- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)：15 类，像素级标注，许可 CC BY-NC-SA 4.0，仅限非商业用途。
- [VisA](https://github.com/amazon-science/spot-diff)：12 类，含 PCB，许可 CC BY 4.0。

也可以用自己的图像：每类准备几张到几十张就够做 k 近邻和线性探测。下载或自备的数据放 `data/external/`，不要提交到仓库。
