"""第 24 章自动验收：独立生成数据、调用流水线、按真值计算指标。

指标（匹配半径 4 px）：
- capture rate（检出率）= 匹配上的真缺陷 / 真缺陷总数，要求 >= 0.85
- nuisance per die（每 die 误报数）= 未匹配的预测 / 待检 die 数，要求 <= 0.3
- 良品 die 误判率（有缺陷报警但实际无缺陷的 die 比例），只报告不设门槛
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "24-1_检测流水线.py"
SEEDS = (11, 23)
RADIUS = 4.0


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def match(pred, truth, radius):
    pairs = sorted(
        (float(np.hypot(p[0] - t[0], p[1] - t[1])), i, j)
        for i, p in enumerate(pred) for j, t in enumerate(truth)
    )
    used_p, used_t = set(), set()
    for d, i, j in pairs:
        if d <= radius and i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
    return len(used_p), len(pred) - len(used_p), len(truth) - len(used_t)


def main() -> int:
    print("第 24 章综合项目验收\n" + "=" * 52)
    gen = load(CHAPTER / "数据生成.py", "chapter24_data")
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    try:
        mod = load(base / ASSIGNMENT, "chapter24_assignment")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 加载作业 — {type(exc).__name__}: {exc}")
        return 1
    all_ok = True
    for seed in SEEDS:
        data = gen.make_dataset(seed=seed)
        t0 = time.perf_counter()
        try:
            preds = mod.run_pipeline(data["golden"].copy(), data["test"].copy())
        except NotImplementedError as exc:
            print(f"[TODO] 流水线 — {exc}")
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] 种子 {seed} — 运行异常：{type(exc).__name__}: {exc}")
            return 1
        elapsed = time.perf_counter() - t0
        if not isinstance(preds, list) or len(preds) != len(data["truth"]):
            print(f"[FAIL] 种子 {seed} — 返回值应是长度 {len(data['truth'])} 的列表")
            return 1
        tp = fp = fn = 0
        false_alarm_dies = 0
        clean_dies = 0
        for p, t in zip(preds, data["truth"]):
            p = [tuple(map(float, c)) for c in p]
            a, b, c = match(p, t, RADIUS)
            tp, fp, fn = tp + a, fp + b, fn + c
            if not t:
                clean_dies += 1
                false_alarm_dies += bool(p)
        capture = tp / (tp + fn)
        nuisance = fp / len(preds)
        ok = capture >= 0.85 and nuisance <= 0.3 and elapsed <= 30
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] 种子 {seed} — 检出率 {capture:.1%}（{tp}/{tp + fn}），"
              f"每 die 误报 {nuisance:.2f}，良品 die 误判 {false_alarm_dies}/{clean_dies}，耗时 {elapsed:.1f}s")
    print("=" * 52)
    if not all_ok:
        print("要求：检出率 >= 85%，每 die 误报 <= 0.3，单次 <= 30 秒。先检查对齐，再检查亮度归一化和阈值。")
        return 1
    print("通过。请在 项目报告.md 里写清每一步的作用、消融结果（去掉对齐/归一化后指标怎么变）和漏检样本分析。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
