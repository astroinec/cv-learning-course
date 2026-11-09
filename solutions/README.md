# 参考实现（solutions 分支）

这里是第二部分（15–24 章）作业的参考实现，供两类人使用：

- **学员**：某章自己做完、验收通过之后，对照看看有没有更简洁或更稳健的写法。卡住时先看验收输出的 FAIL 说明和教程，不要第一时间来这里。
- **维护者和 AI 助手**：修改测试或 Notebook 后，用这里的实现确认“正确实现能通过”。

## 一键验证

```bash
python3 solutions/verify_all.py
```

它会对每章分别运行两次验收：一次用 `solutions/` 里的参考实现（应全部 PASS），一次用主目录里的作业文件（初始状态应显示 TODO 而不是报错）。

## 用 worktree 在主分支旁边打开本分支

```bash
git worktree add ../cv-solutions solutions
COURSE_IMPL_DIR=../cv-solutions/solutions python3 18_传统缺陷检测/tests/check_progress.py
```

## 说明

- 参考实现追求正确和清晰，不追求最快。例如第 16 章的 ZNCC 用的是直白的双重循环。
- 第 24 章的参考流水线是自包含的，不依赖其他章节的作业文件。它用了“不加窗的相位相关 + ±8 px 搜索范围 + 一轮残差细化”的对齐，原因见第 16 章教程第 6、7 节和第 24 章 README。
