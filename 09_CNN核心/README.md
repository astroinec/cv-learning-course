# 第 9 章：CNN 核心

本章用合成几何图观察卷积核、padding、stride 和特征图，不下载 CIFAR-10。

## 学习顺序

1. 阅读 `教程_CNN核心.md`。
2. 运行 `实验_卷积与感受野.ipynb`，修改核、stride 与 padding。
3. 完成 `作业_卷积shape与残差块.py`。
4. 运行 `python tests/check_progress.py`。
5. 在 `实验记录.md` 记录 shape 推导与至少一个边界失败案例。

## 文件说明

| 文件 | 作用 |
|---|---|
| `实验_卷积与感受野.ipynb` | 手工边缘核与特征图可视化 |
| `作业_卷积shape与残差块.py` | 输出尺寸计算和残差块 shape 契约 |
| `tests/check_progress.py` | 边界条件、输出 shape 与梯度检查 |
| `data/README.md` | CIFAR-10 等官方来源与许可提示 |

## 通过标准

- 能从 `NCHW` 输入手算卷积输出 shape。
- 能解释 padding、stride 与边界响应的影响。
- 能说明残差相加前为何必须 shape 一致。
- Notebook 可从头执行，检查器初始安全显示 `TODO`。
