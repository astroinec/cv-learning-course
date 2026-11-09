# 视觉算法动手课程

一套中文的视觉算法动手课程，分两部分：

- **第一部分（01–14 章）**：通用计算机视觉，面向“完成过本科基础学习、希望在实践中重新巩固数学与视觉算法”的学习者。
- **第二部分（15–24 章）**：工业视觉专项，面向半导体检测方向的视觉算法岗，覆盖 C++ 工程、亚像素对齐、相机标定、缺陷检测、异常检测、合成数据、渲染管线、成像仿真和领域知识，最后用一个晶圆 die-to-die 检测项目串起来。

每章采用统一学习闭环：

> 教程 → 互动 Notebook → 独立作业 → 自动验收 → 破坏实验 → 实验复盘

## 从哪里开始

| 你想要 | 打开 |
|---|---|
| 一页看懂学习路线和优先级 | [`docs/工业视觉路线_概览.md`](docs/工业视觉路线_概览.md) |
| 逐日的 6 周计划和入职后 90 天 | [`docs/工业视觉路线_详细.md`](docs/工业视觉路线_详细.md) |
| 可视化路线图、勾选进度 | 浏览器打开 [`路线图.html`](路线图.html) |
| 岗位要求和章节的对应 | [`docs/JD对照.md`](docs/JD对照.md) |
| 视频、教程、论文、数据集 | [`docs/资源清单.md`](docs/资源清单.md) |
| 交互演示 | [`demos/`](demos/)：过检与漏检权衡、相机投影与畸变、渲染管线 |
| 记录进度 | [`学习进度.md`](学习进度.md) |
| 遇到环境或作业问题 | [`docs/常见问题.md`](docs/常见问题.md) |
| 让 AI 助手接手教学或维护 | [`AGENTS.md`](AGENTS.md) |

## 课程结构

| 阶段 | 章节 | 主题 |
|---|---:|---|
| 传统 CV | 01-03 | 像素与色彩、卷积与边缘、形态学 |
| 几何视觉 | 04-07 | 几何变换、特征点、光流、相机与三维 |
| 深度学习基础 | 08-10 | 神经网络、CNN、迁移学习与模型分析 |
| 视觉任务 | 11-14 | 目标检测、图像分割、生成与自监督、综合项目与部署 |
| 工业工程基础 | 15 | C++ 视觉工程（积分图、NCC、连通域、CMake、Sanitizer） |
| 对齐与标定 | 16-17 | 相位相关与亚像素配准、张正友标定 |
| 缺陷检测 | 18-19 | 金模板比对与工业指标、PatchCore 与异常检测 |
| 生成与仿真 | 20-22 | 合成缺陷与扩散模型、3D 数学与软光栅、成像与传感器仿真 |
| 领域与项目 | 23-24 | 半导体检测领域知识、合成晶圆 die-to-die 检测流水线 |

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

- 第一部分完整学习：约 **90-140 小时**，详细能力目标和周计划见 [`开始这里.md`](开始这里.md)。
- 第二部分：P0 加 P1 约 **130 小时**，按 6 周、每周 20–25 小时编排，见 [`docs/工业视觉路线_详细.md`](docs/工业视觉路线_详细.md)。
- 目标是入职工业视觉岗的话，不需要先学完第一部分，按概览版的优先级挑章节。

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

第二部分（15–24 章）的作业和验收只依赖 NumPy，第 15 章另需系统自带的 C++ 编译器。一次检查第二部分全部章节：

```bash
for d in 1[5-9]_*/tests/check_progress.py 2[0-4]_*/tests/check_progress.py; do echo "$d"; python3 "$d" | tail -2; done
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

- 24 章中文教程（第一部分 14 章，第二部分 10 章）
- 23 个互动 Notebook（第 15 章为 C++，用 CMake 和测试程序代替 Notebook）
- 每章作业与自动检查器；第二部分的检查器都用参考实现验证过“正确实现能通过、初始状态显示 TODO”
- 参考实现在 `solutions` 分支，学员完成该章后再看
- 交互式路线图和 3 个演示网页，无需安装，浏览器直接打开

欢迎通过 Issue 提交勘误或学习体验建议。
