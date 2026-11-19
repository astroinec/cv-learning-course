"""第 24 章自动验收：独立生成数据、调用流水线、按真值计算指标。

指标（匹配半径 4 px）：
- capture rate（检出率）= 匹配上的真缺陷 / 真缺陷总数
- nuisance per die（每 die 误报数）= 未匹配的预测 / 待检 die 数
- 良品 die 误判率（有缺陷报警但实际无缺陷的 die 比例），只报告不设门槛

判据：用 8 个固定种子各生成一份数据。
- 一个种子“达标”：检出率 >= 85%，每 die 误报 <= 0.3，单次运行 <= 30 秒
- 通过：8 个种子里至少 6 个达标，并且 8 个种子合在一起算的检出率 >= 85%、每 die 误报 <= 0.3

为什么不要求每个种子都达标：一个种子只有三十几个缺陷，多漏一两个，检出率就差好几个百分点。
一条正确的流水线在很多种子上测，也会有少数几个种子低于 85%（教程第 4 节有实测）。
只看两个种子、要求全部达标时，通不通过有一部分取决于这两个种子碰巧难不难。
合计的那一条用来挡住另一种情况：多数种子正常，个别种子上整片误报。
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
ASSIGNMENT = "24-1_检测流水线.py"
SEEDS = (11, 23, 35, 47, 59, 71, 83, 95)
NEED = 6            # 至少这么多个种子达标
RADIUS = 4.0
MIN_CAPTURE = 0.85
MAX_NUISANCE = 0.3
MAX_SECONDS = 30.0
LINE = "=" * 52


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


def tidy(preds, n):
    """把返回值整理成 [[(y, x), ...], ...]。格式不符合作业说明时返回 (None, 原因)。"""
    if not isinstance(preds, list):
        return None, f"返回值应是一个列表（每张待检 die 一项），得到的是 {type(preds).__name__}"
    if len(preds) != n:
        return None, f"返回的列表长度应等于待检 die 的张数 {n}，得到 {len(preds)}"
    out = []
    for i, item in enumerate(preds):
        if item is None:
            return None, f"第 {i} 张 die 的结果是 None；没有缺陷的 die 要返回空列表"
        try:
            points = [tuple(float(v) for v in c) for c in item]
        except (TypeError, ValueError):
            return None, f"第 {i} 张 die 的结果应是若干个 (y, x) 组成的列表，现在不是这种形式"
        if any(len(c) != 2 for c in points):
            return None, f"第 {i} 张 die 的结果里，有的缺陷中心不是 (y, x) 两个数"
        if not all(math.isfinite(v) for c in points for v in c):
            return None, f"第 {i} 张 die 的结果里有 nan 或 inf；缺陷中心应是图像坐标系里的有限数值"
        out.append(points)
    return out, ""


def finish(ok: bool, status: str, detail: str) -> int:
    print(f"[{status:4}] 流水线 — {detail}")
    print(LINE)
    print(f"通过 {int(ok)}/1")
    return 0 if ok else 1


def main() -> int:
    print("第 24 章综合项目验收\n" + LINE)
    gen = load(CHAPTER / "数据生成.py", "chapter24_data")
    override = os.environ.get("COURSE_IMPL_DIR")
    base = Path(override) / CHAPTER.name if override else CHAPTER
    try:
        mod = load(base / ASSIGNMENT, "chapter24_assignment")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 加载作业 — {type(exc).__name__}: {exc}")
        return 1
    good = 0
    tp_all = fp_all = fn_all = die_all = 0
    for seed in SEEDS:
        data = gen.make_dataset(seed=seed)
        t0 = time.perf_counter()
        try:
            preds = mod.run_pipeline(data["golden"].copy(), data["test"].copy())
        except NotImplementedError as exc:
            code = finish(False, "TODO", str(exc))
            print("提示：TODO 是正常学习状态；实现后重新运行。")
            return code
        except Exception as exc:  # noqa: BLE001
            return finish(False, "FAIL", f"种子 {seed} 运行异常：{type(exc).__name__}: {exc}")
        elapsed = time.perf_counter() - t0
        preds, why = tidy(preds, len(data["truth"]))
        if preds is None:
            return finish(False, "FAIL", f"种子 {seed}：{why}")
        tp = fp = fn = 0
        false_alarm_dies = 0
        clean_dies = 0
        for p, t in zip(preds, data["truth"]):
            a, b, c = match(p, t, RADIUS)
            tp, fp, fn = tp + a, fp + b, fn + c
            if not t:
                clean_dies += 1
                false_alarm_dies += bool(p)
        capture = tp / (tp + fn)
        nuisance = fp / len(preds)
        ok = capture >= MIN_CAPTURE and nuisance <= MAX_NUISANCE and elapsed <= MAX_SECONDS
        good += ok
        tp_all, fp_all, fn_all, die_all = tp_all + tp, fp_all + fp, fn_all + fn, die_all + len(preds)
        print(f"  种子 {seed}：{'达标' if ok else '未达标'}，检出率 {capture:.1%}（{tp}/{tp + fn}），"
              f"每 die 误报 {nuisance:.2f}，良品 die 误判 {false_alarm_dies}/{clean_dies}，耗时 {elapsed:.1f}s", flush=True)
    capture_all = tp_all / (tp_all + fn_all)
    nuisance_all = fp_all / die_all
    pooled_ok = capture_all >= MIN_CAPTURE and nuisance_all <= MAX_NUISANCE
    passed = good >= NEED and pooled_ok
    detail = (f"{len(SEEDS)} 个种子里 {good} 个达标（至少要 {NEED} 个）；"
              f"合计检出率 {capture_all:.1%}（{tp_all}/{tp_all + fn_all}），合计每 die 误报 {nuisance_all:.2f}")
    code = finish(passed, "PASS" if passed else "FAIL", detail)
    if not passed:
        print(f"要求：至少 {NEED} 个种子达标（检出率 >= {MIN_CAPTURE:.0%}，每 die 误报 <= {MAX_NUISANCE}，单次 <= {MAX_SECONDS:.0f} 秒），"
              "并且合计的检出率和每 die 误报也满足这两条线。")
        print("先检查对齐，再检查亮度归一化和阈值。只差一两个种子时，先在验收种子以外的数据上多跑一些种子，看平均水平（教程第 4 节）。")
        return code
    print("通过。请在 项目报告.md 里写清每一步的作用、消融结果（去掉对齐/归一化后指标怎么变）和漏检样本分析。")
    return code


if __name__ == "__main__":
    sys.exit(main())
