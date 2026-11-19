# 第 28 章数据说明

作业、验收、Notebook 的必做部分和教程里的全部数字，都只用代码生成的“图文对”：32×32 的彩色小图（2×2 个格子，每格至多一个目标，目标有形状、颜色、大小、纹理四种属性），和由这些属性拼成的短句（词表 34 个词）。不需要下载任何数据。

教程第 9 节和 Notebook 的选做单元要用 CLIP 的预训练权重，第一次运行时由 OpenCLIP 或 Hugging Face `transformers` 自动下载到各自的缓存目录，不要放进仓库。权重的大小和许可以官方仓库（`openai/CLIP`、`mlfoundations/open_clip`）和模型卡为准。

深挖档可以用自己的图像：

- 零样本和少样本分类：每类准备 10 张以上带标签的图，另外留出一部分只用来测试，不参与挑选提示词。
- 属性绑定：自己拍或者画二三十张“两个物体、两种属性”的图，每张配一句正确的话和一句词相同、搭配不同的话。
- 异常检测：可以用第 19 章用过的公开数据。[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)：15 类，像素级标注，许可 CC BY-NC-SA 4.0，仅限非商业用途。[VisA](https://github.com/amazon-science/spot-diff)：12 类，含 PCB，许可 CC BY 4.0。

下载或自备的数据放 `data/external/`，不要提交到仓库。
