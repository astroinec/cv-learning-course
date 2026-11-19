# 视觉算法动手课程

一套中文的视觉算法动手课程，分两部分：

- **第一部分（01–14 章）**：通用计算机视觉，从像素到检测、分割、生成与部署。
- **第二部分（15–24 章）**：工业视觉专项，面向半导体检测方向的视觉算法岗，覆盖 C++ 工程、亚像素对齐、相机标定、缺陷检测、异常检测、合成数据、渲染管线、成像仿真和领域知识，用一个晶圆 die-to-die 检测项目串起来。
- **第三部分（25–31 章）**：前沿方法，ViT、自监督基础模型（MAE/DINOv2）、SAM、视觉语言模型、扩散模型、神经渲染、部署与加速；每章既有从零实现的小模型，也有在自己机器上用真模型的一节。

课程最终在一台 Windows 笔记本（RTX 4060，8 GB 显存）上使用。环境安装（Windows 为主）见 [`docs/环境安装.md`](docs/环境安装.md)；写课程和全部实测在 macOS 上用 CPU 完成，讲义里没有显卡上的数字。

每章采用统一学习闭环：

> 教程 → 互动 Notebook → 独立作业 → 自动验收 → 破坏实验 → 实验复盘

## 从哪里开始

**先装好环境**（见 [`docs/环境安装.md`](docs/环境安装.md)，Windows 双击 `安装环境.bat`），**再打开课程网页**：双击 `打开课程网页.bat`，或运行 `python -m http.server 8000` 后访问 `http://localhost:8000/site/index.html`；也可直接看在线版 [`astroinec.github.io/cv-learning-course`](https://astroinec.github.io/cv-learning-course/)。首页按学习路线列出 31 章的应用场景、优先级和三档用时，不需要从第 1 章按顺序学。每章的用时和优先级以 `course.json` 为准，下面各文档里的旧用时表仅供参考。

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

完整步骤（Windows 为主，含带 CUDA 的 PyTorch、第 15 章编译器、第三部分可选包）见 [`docs/环境安装.md`](docs/环境安装.md)。

Windows：双击 `安装环境.bat`（它会提示先按文档装好带 CUDA 的 PyTorch）。手动或在 macOS/Linux 上：

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
# 想用 NVIDIA 显卡：先按 docs/环境安装.md 第 3.3 节装带 CUDA 的 PyTorch，再装下面这行
python -m pip install -r requirements.txt
python -m ipykernel install --user --name cv-learning --display-name "Python 3 (cv-learning)"
python scripts/check_env.py        # 逐项检查环境
```

## 启动

Windows 双击 `启动课程.bat`，macOS/Linux 运行 `./start_course.sh`，或手动 `python -m jupyter lab .`。内核选 “Python 3 (cv-learning)”。

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
- 每章作业与自动检查器；全部 24 章的检查器都用参考实现验证过“正确实现能通过、初始状态显示 TODO”
- 参考实现在 `solutions` 分支，学员完成该章后再看
- 交互式路线图和 3 个演示网页，无需安装，浏览器直接打开

欢迎通过 Issue 提交勘误或学习体验建议。
