# 视觉算法动手课程

一套面向“完成过本科基础学习、希望在实践中重新巩固数学与视觉算法”的中文课程。

课程覆盖传统计算机视觉、深度学习视觉和端到端项目实践。每章采用统一学习闭环：

> 教程 → 互动 Notebook → 独立 Python 作业 → 自动验收 → 破坏实验 → 实验复盘

## 课程结构

| 阶段 | 章节 | 主题 |
|---|---:|---|
| 传统 CV | 01-03 | 像素与色彩、卷积与边缘、形态学 |
| 几何视觉 | 04-07 | 几何变换、特征点、光流、相机与三维 |
| 深度学习基础 | 08-10 | 神经网络、CNN、迁移学习与模型分析 |
| 视觉任务 | 11-14 | 目标检测、图像分割、生成与自监督、综合项目与部署 |

每章目录包含：

```text
章节/
├── README.md              # 学习顺序和通过标准
├── 教程_主题.md            # 直觉、数学、API、限制与常见错误
├── 实验_主题.ipynb         # 可交互的最小实验
├── 作业.py                 # 独立实现，不复制 Notebook 基线
├── tests/check_progress.py # TODO / FAIL / PASS 自动验收
├── data/README.md          # 数据来源、许可和目录约定
└── 实验记录.md              # 参数、结果、失败案例和复盘
```

## 建议学习时间

- 完整学习：约 **90-140 小时**
- 稳健节奏：10-12 周，每周 8-12 小时
- 集中节奏：5-7 周，每周 18-25 小时

详细能力目标和周计划见 [`开始这里.md`](开始这里.md)。

## 安装

推荐 Python 3.10-3.13，使用独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name cv-learning --display-name "Python 3 (cv-learning)"
```

Apple Silicon 可通过 PyTorch MPS 后端加速：

```python
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

## 启动

macOS/Linux：

```bash
./start_course.sh
```

或手动运行：

```bash
python -m jupyter lab .
```

macOS 也可以双击 `启动课程.command`。

进入 `01_像素与色彩/`，依次阅读教程和 Notebook。完成作业后运行：

```bash
python 01_像素与色彩/tests/check_progress.py
```

## 数据集

课程 Notebook 默认使用合成小数据，保证无需联网即可运行。较大的公开数据集不提交到仓库；各章 `data/README.md` 提供官方来源、用途、目录约定和许可提醒。

根目录测试图片可运行以下命令生成：

```bash
python scripts/generate_sample_images.py
```

## 学习原则

- 先预测结果，再运行代码。
- 不把库函数基线复制成“从零实现”的答案。
- 每章至少制造一次失败案例，理解算法依赖的假设。
- 自动测试通过只是底线，还要能解释结果和错误。
- 训练任务必须保存验证指标、错误样本和实验配置。

## 仓库状态

- 14 章完整中文教程
- 14 个可独立运行的互动 Notebook
- 章节作业与自动检查器
- 数据来源说明与实验记录模板

欢迎通过 Issue 提交勘误或学习体验建议。
