"""讲义数字核对：把讲义里出现的数字和本章的 实测数字.json 对一遍，列出对不上的。

用法（在仓库根目录运行）：
    python scripts/audit_numbers.py 17            # 只看汇总和对不上的数字
    python scripts/audit_numbers.py 17 18 --all   # 多章；--all 连对上的也列出来

它回答的问题是“讲义里这个数在 JSON 里有没有出处”，规则故意从宽：
讲义里的数 x（保留 d 位小数）只要等于 JSON 里某个数 j 按 d 位四舍五入的结果，
或者等于 j×100、j÷100、1/j、1-j、100-j 这些一步换算的结果，就算对上。
JSON 里的数包括所有数值，以及键名和说明文字里出现的数字。

对不上不一定是错：两个 JSON 数相除、相减得到的倍数和差值，这个脚本认不出来，要人看。
但重跑配图脚本之后数字变了、讲义忘了改，会在这里直接冒出来。
只核对带小数点的数、百分数和不小于 20 的整数；代码块、公式、链接、图号、章节号不看。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUM_RE = re.compile(r"(?<![\w.\-/])(-?\d+(?:\.\d+)?)(?:[eE](-?\d+))?(\s?%)?(?![\w]*\.(?:py|md|json|svg|html|cpp|hpp|ipynb))")


def json_numbers(obj, out: set) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, str):
        for m in NUM_RE.finditer(obj):
            try:
                out.add(float(m.group(1)) * (10 ** int(m.group(2)) if m.group(2) else 1))
            except (ValueError, OverflowError):
                pass
    elif isinstance(obj, dict):
        for k, v in obj.items():
            json_numbers(str(k), out)
            json_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            json_numbers(v, out)


def strip_noise(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)               # 代码块和 math 围栏
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)             # 独立公式
    text = re.sub(r"\$[^$\n]*\$", " ", text)                          # 行内公式
    text = re.sub(r"`[^`\n]*`", " ", text)                            # 行内代码
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"\]\([^)]*\)", "]", text)                          # 链接目标
    text = re.sub(r"^#{1,6} .*$", " ", text, flags=re.M)              # 标题（带小节编号）
    text = re.sub(r"第 ?\d+(?:\.\d+)* ?(?:章|节|步|条|题|天|行|列|栏|个|次|档|类|种|组|项|张|帧|周)", " ", text)
    text = re.sub(r"(?<![\d.])\d+(?:\.\d+)+ ?节", " ", text)
    text = re.sub(r"图 ?\d+", " ", text)
    text = re.sub(r"(?:Lecture|版本|v|OpenCV|NumPy|Python|C\+\+|ISO|EMVA|BT\.|Rec\.) ?\d+(?:\.\d+)*", " ", text)
    text = re.sub(r"(?:19|20)\d\d ?年?", " ", text)                   # 年份
    return text


def matches(x: float, decimals: int, pool: list) -> bool:
    tol = 0.5 * 10 ** (-decimals) * 1.0001 + 1e-12
    for j in pool:
        cands = [j, j * 100, j / 100, 1 - j, 100 - j, 100 * (1 - j), abs(j), abs(j) * 100]
        if j:
            cands += [1 / j, 100 / j]
        if any(abs(c - x) <= tol for c in cands):
            return True
    return False


def audit(n: str, show_all: bool) -> int:
    course = json.loads((ROOT / "course.json").read_text(encoding="utf-8"))
    ch = next((c for c in course["chapters"] if c["n"] == n and not c.get("planned")), None)
    if ch is None:
        print(f"course.json 里没有第 {n} 章")
        return 0
    folder = ROOT / ch["dir"]
    tutorial = folder / ch["tutorial"] if ch.get("tutorial") else next(iter(sorted(folder.glob("教程_*.md"))), None)
    numbers = folder / "figures" / "实测数字.json"
    if tutorial is None or not tutorial.exists() or not numbers.exists():
        print(f"第 {n} 章：缺讲义或 实测数字.json，跳过")
        return 0
    pool_set: set = set()
    json_numbers(json.loads(numbers.read_text(encoding="utf-8")), pool_set)
    pool = sorted(pool_set)
    text = strip_noise(tutorial.read_text(encoding="utf-8"))
    checked, missing = 0, []
    seen = set()
    for m in NUM_RE.finditer(text):
        raw, exp, pct = m.group(1), m.group(2), m.group(3)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        value = float(raw) * (10 ** int(exp) if exp else 1)
        if exp:
            decimals = max(0, decimals - int(exp))
        if decimals == 0 and not pct and abs(value) < 20:
            continue
        key = (raw, exp, bool(pct))
        if key in seen:
            continue
        seen.add(key)
        checked += 1
        ok = matches(value, decimals, pool) or matches(abs(value), decimals, pool)
        ctx = text[max(0, m.start() - 26):m.end() + 14].replace("\n", " ")
        if not ok:
            missing.append((m.group(0).strip(), ctx))
        elif show_all:
            print("   对上：", m.group(0).strip(), "｜", ctx)
    print(f"第 {n} 章 {ch['title']}：核对 {checked} 个不同的数，JSON 里找不到出处的 {len(missing)} 个")
    for tok, ctx in missing:
        print(f"   {tok:>10} ｜ …{ctx}…")
    return len(missing)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    total = sum(audit(f"{int(a):02d}", "--all" in sys.argv) for a in args)
    print(f"合计 {total} 个数需要人工看一眼")
    return 0


if __name__ == "__main__":
    sys.exit(main())
