# 第 8 章数据说明

Notebook 默认在内存中合成带轻微抖动的 XOR 二分类点，不创建数据文件、不联网。合成数据可自由用于本课程实验，但不代表真实任务分布。

## 可选官方来源

- MNIST 官方页面：<http://yann.lecun.com/exdb/mnist/>
- torchvision MNIST 文档：<https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html>

MNIST 常用于教学与研究，但下载前仍应查看来源页当前许可和引用要求。不要在 Notebook 中设置自动下载；若自行准备，请放入本目录下的 `external/`，并记录下载日期、版本、校验值和许可。

## 建议目录

```text
data/
├── README.md
└── external/    # 可选，本课程不自带
```

真实数据还应记录隐私、类别分布、划分方式和是否允许再分发。
