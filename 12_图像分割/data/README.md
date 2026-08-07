# 第 12 章数据说明

Notebook 在内存中生成圆形前景和像素掩码；合成 mask 的类别值仅为 0/1，不写入磁盘。

## 可选官方来源

- Oxford-IIIT Pet 官方页面：<https://www.robots.ox.ac.uk/~vgg/data/pets/>
- torchvision OxfordIIITPet 文档：<https://pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html>
- ISBI Challenge 官方门户：<https://www.biomedicalimaging.org/challenges/>

不同分割数据集的图像、标注和医学数据使用条件不同。下载前必须核对当前许可、引用、研究用途限制和隐私要求；医学数据还可能要求注册或数据使用协议。本课程不下载任何数据。

自行准备数据时建议按原始样本分组后再切 patch，并记录 mask 类别表、`ignore_index`、插值规则与标注版本。
