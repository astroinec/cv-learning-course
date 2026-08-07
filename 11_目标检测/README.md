# 第 11 章：目标检测与 NMS

本章用少量合成框观察 IoU、置信度排序和 NMS 阈值，不训练或下载检测模型。

## 学习顺序

1. 阅读 `教程_目标检测.md`。
2. 运行 `实验_IoU与NMS阈值.ipynb`，修改框位置和阈值。
3. 完成 `作业_批量IoU与类别NMS.py`。
4. 运行 `python tests/check_progress.py`。
5. 在 `实验记录.md` 记录密集目标、边界接触和阈值失败案例。

## 文件说明

| 文件 | 作用 |
|---|---|
| `实验_IoU与NMS阈值.ipynb` | 标量 IoU 与贪心抑制的可视化直觉 |
| `作业_批量IoU与类别NMS.py` | 向量化两两 IoU 与类别感知 NMS |
| `tests/check_progress.py` | 检查空输入、重合、分离及跨类别情况 |
| `data/README.md` | COCO、VOC 等官方来源与许可提示 |

Notebook 只给小规模标量演示；作业要求独立完成批量、空输入和类别隔离逻辑。
