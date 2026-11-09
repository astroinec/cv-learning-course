# AGENTS.md：给 AI 助手和协作者的说明

> 任何 AI 助手接手这个仓库（无论是继续教学、更新内容还是检查作业），先读完本文件，再读 `docs/学员画像.md` 和 `学习进度.md`。

## 1. 仓库是什么

一套中文的视觉算法动手课程，服务于一位即将入职**半导体检测设备方向视觉算法岗**的学员。

- **第一部分（01–14 章）**：通用计算机视觉，传统方法加深度学习。2026 年 8 月建立。
- **第二部分（15–24 章）**：工业视觉专项，围绕目标岗位 JD 设计。2026 年 11 月建立。

每章统一结构：

```text
NN_主题/
├── README.md               学习顺序、文件表、通过标准、推荐资源
├── 教程_*.md                讲义，末尾有自测问题
├── 实验_*.ipynb             互动 Notebook（第 16–24 章由脚本生成，见第 4 节）
├── NN-1_*.py 或 .cpp        作业，函数初始抛 NotImplementedError（C++ 抛 logic_error("TODO")）
├── tests/check_progress.py  自动验收，输出 TODO / FAIL / PASS
├── data/README.md           数据来源和许可
└── 实验记录.md              学员填写的实验记录
```

## 2. 关键文件

| 文件 | 用途 | 谁来更新 |
|---|---|---|
| `学习进度.md` | 每章状态、完成日期、卡点、自测没答上来的问题 | 学员或 AI，每次学习后更新 |
| `docs/学员画像.md` | 背景、能力评估、学习偏好、带教注意点 | 能力有明显变化时更新 |
| `docs/工业视觉路线_概览.md` | 一页路线和优先级 | 路线调整时 |
| `docs/工业视觉路线_详细.md` | 6 周逐日计划和入职后 90 天 | 路线调整时 |
| `docs/JD对照.md` | JD 条目到章节的映射 | 岗位方向明确后可细化 |
| `docs/资源清单.md` | 外部资源 | 链接失效或发现更好资源时 |
| `docs/常见问题.md` | 环境和学习问题 | 遇到新问题时追加 |
| `路线图.html` | 可视化路线图，学员在浏览器里勾进度（存在 localStorage） | 新增章节时同步改里面的 `CHAPTERS` 数组 |
| `demos/*.html` | 交互演示，无依赖的单文件网页 | 按需 |
| `scripts/build_track2_notebooks.py` | 生成第 16–24 章 Notebook | 改 Notebook 内容时改这里再运行 |
| `scripts/run_track2_notebooks.py` | 执行 Notebook 全部代码单元，检查能否跑通 | 维护用 |

## 3. 参考答案

参考实现放在 **`solutions` 分支**的 `solutions/` 目录，主分支没有。

- 学员没完成某章之前，**不要主动展示该章答案**。学员卡住时给提示，顺序是：指出 FAIL 说明里的线索 → 指向教程对应小节 → 给伪代码或关键一步 → 学员明确要求时才给完整代码。
- 维护者验证测试本身时：

  ```bash
  git worktree add ../cv-solutions solutions
  COURSE_IMPL_DIR=../cv-solutions/solutions python3 16_图像对齐与亚像素/tests/check_progress.py
  ```

  所有第二部分的验收脚本和 Notebook 都支持 `COURSE_IMPL_DIR` 环境变量。

## 4. 修改内容的规则

1. **改测试前先想清楚**：测试要同时满足“正确实现能过”和“典型错误会被拦下”。改完必须用参考实现跑一遍确认全过，再用初始 TODO 状态跑一遍确认显示 TODO 而不是报错。
2. **测试说明要诚实**：只写测试真正检查了的东西。例如第 17 章曾经声称能发现“没做归一化的 DLT”，实测发现在 float64 下检测不出来，已改掉。
3. **Notebook 不手改**：第 16–24 章的 `.ipynb` 由 `scripts/build_track2_notebooks.py` 生成。改内容就改脚本，然后：

   ```bash
   python scripts/build_track2_notebooks.py
   COURSE_IMPL_DIR=../cv-solutions/solutions python scripts/run_track2_notebooks.py   # 应全部 OK
   python scripts/run_track2_notebooks.py                                              # TODO 状态也应全部 OK
   ```

   需要 `nbformat` 和 `matplotlib`。
4. **Notebook 不放答案**：只调用学员的作业函数，用 `attempt()` 包装，没实现时打印提示。
5. **第 24 章的数据生成器**改动后，要重新确认：参考流水线在验收种子 11 和 23 上达标、在 10 个以上其他种子上稳定；去掉对齐、去掉亮度归一化时指标明显变差（说明这些步骤确实被考到）。README 里的参考数字要同步。
6. **新增章节**：照现有章节结构，同时更新根目录 `README.md` 的课程表、`路线图.html` 的 `CHAPTERS`、`学习进度.md`、`docs/JD对照.md`。

## 5. 写作风格

- 中文，简洁，结论先行。术语第一次出现给英文原名。
- 不用破折号，不用表情符号，列表不嵌套。
- 讲义按“为什么需要 → 原理 → 工程细节 → 常见错误 → 自测问题”组织。
- 数字和链接要可核实。不确定的写“以具体设备和公司资料为准”，不要编造规格。
- 仓库是公开的：不写学员姓名、学校、公司名、联系方式、薪资等隐私信息。

## 6. 教学时的工作方式

每次学习会话：

1. 读 `学习进度.md`，确认当前章节和上次的卡点。
2. 问学员今天的可用时间，按 `docs/工业视觉路线_详细.md` 给出今天的具体任务。
3. 学员做作业时，回答问题优先给提示而不是答案（见第 3 节）。
4. 学员完成一章后：运行验收确认 PASS；用教程末尾的自测问题做检查；检查 `实验记录.md` 有没有失败案例分析。
5. 会话结束前更新 `学习进度.md`：状态、日期、卡点、没答上来的自测问题、下一步。

评估学员时参考 `docs/学员画像.md` 的能力表：C++ 是最大短板，需要更多耐心和可编译的小例子；评测和信号处理是强项，可以讲得快、讲得深。

## 7. 验收命令速查

```bash
# 单章
python3 18_传统缺陷检测/tests/check_progress.py

# 第二部分全部（看汇总行）
for d in 1[5-9]_*/tests/check_progress.py 2[0-4]_*/tests/check_progress.py; do echo "$d"; python3 "$d" | tail -2; done
```
