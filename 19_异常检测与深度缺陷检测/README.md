# 第 19 章：异常检测与深度缺陷检测

本章用时建议 **12–18 小时**（含在 MVTec AD 上跑实验）。目标：理解工业场景为什么偏爱无监督异常检测，亲手实现 PatchCore 的核心零件，会用 Anomalib 在公开数据集上跑通基线，并用工业指标汇报结果。

## 为什么重要

产线上良品多、缺陷少，而且新缺陷类型会不断出现。只用良品训练、把“和良品不一样”的东西都找出来的异常检测，非常适合这种场景。有足够标注时，再用监督的检测和分割模型做分类。JD 里写的“基于深度学习的检测、分割、分类”，两条路都要会。

## 学习顺序

1. 复习第 10 章（迁移学习、特征提取）、第 11 章（检测）、第 12 章（分割）。
2. 阅读 `教程_异常检测.md`。
3. 实现 `19-1_PatchCore核心.py`，运行验收。
4. 运行 `实验_PatchCore.ipynb`：用合成纹理跑一个迷你 PatchCore；有网络和 torchvision 时，换成预训练 ResNet 特征。
5. 下载 MVTec AD 的一个类别（比如 bottle 或 grid），用 Anomalib 跑 PatchCore 和 PaDiM，对比 image AUROC 和 pixel AUROC。
6. 在 `实验记录.md` 写下：漏检率不超过 1% 时的过检率，以及最难的 3 个样本为什么难。

## 文件

| 文件 | 用途 |
|---|---|
| `教程_异常检测.md` | 方法谱系、PatchCore 详解、监督方法、指标、落地问题 |
| `实验_PatchCore.ipynb` | 迷你 PatchCore 实验 |
| `19-1_PatchCore核心.py` | 作业：距离矩阵、核心集、kNN 分数、AUROC、漏检约束阈值 |
| `tests/check_progress.py` | 自动验收 |
| `实验记录.md` | 实验记录模板 |

## 通过标准

- 5 项测试 PASS。
- 能画出 PatchCore 的完整流程：特征提取 → 局部聚合 → 记忆库 → 核心集 → 最近邻打分 → 上采样得到热图。
- 能解释 image AUROC 和 pixel AUROC 各在衡量什么，为什么 AUROC 高不代表产线可用。
- 在 MVTec AD 一个类别上跑通 Anomalib，结果和论文报告的量级一致。

## 推荐资源

- 论文：Roth 等，Towards Total Recall in Industrial Anomaly Detection（CVPR 2022），即 PatchCore。
- 论文：Defard 等，PaDiM（ICPR 2021）；Batzner 等，EfficientAD（WACV 2024）。
- [Anomalib](https://github.com/open-edge-platform/anomalib)：Intel 维护的异常检测库，内置 PatchCore、PaDiM、EfficientAD 等，支持 MVTec AD 数据接口和 OpenVINO 导出。
- [MVTec AD 数据集](https://www.mvtec.com/company/research/datasets/mvtec-ad)：15 类工业物体和纹理，像素级标注。许可为 CC BY-NC-SA 4.0，只能用于非商业用途。
- [VisA 数据集](https://github.com/amazon-science/spot-diff)：12 类，含 PCB 等电子元件，更接近电子制造场景。
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)：有标注时的监督检测基线。
