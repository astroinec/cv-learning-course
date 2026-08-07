# 第 12 章：图像分割与 U-Net

本章在内存中生成圆形前景与掩码，用极小 U-Net 观察编码、跳跃连接和解码输出。

## 学习顺序

1. 阅读 `教程_图像分割.md`。
2. 运行 `实验_合成圆形U-Net.ipynb`，修改前景大小和训练步数。
3. 完成 `作业_Dice与跳跃连接.py`。
4. 运行 `python tests/check_progress.py`。
5. 在 `实验记录.md` 记录 Dice、阈值、边界和空掩码案例。

## 文件说明

| 文件 | 作用 |
|---|---|
| `实验_合成圆形U-Net.ipynb` | 合成 mask、极小 U-Net 与预测可视化 |
| `作业_Dice与跳跃连接.py` | Dice loss 和 skip 对齐拼接 |
| `tests/check_progress.py` | 检查 shape、范围、空掩码与梯度 |
| `data/README.md` | Oxford-IIIT Pet、ISBI 等来源与许可提示 |

## 通过标准

- 区分二分类 logits `[N,1,H,W]` 与 mask 契约。
- 能解释最近邻 mask resize、skip connection 和 Dice 的空类策略。
- Notebook 不依赖外部文件且可快速从头执行。
