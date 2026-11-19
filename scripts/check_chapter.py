"""单章自检：升级完一章后运行，不写 site/，可以和别的章同时跑。

用法（在仓库根目录运行）：
    python scripts/check_chapter.py 17
    python scripts/check_chapter.py 17 --solutions ../cv-solutions-wt/solutions
    python scripts/check_chapter.py 25 --solutions ../cv-solutions-wt/solutions --allow-changes   # 新写的章，或者正在改作业和测试

检查内容：
1. 讲义能否转成网页（小测格式、提示框、链接和配图文件是否存在）。
2. 全部公式能否被站点自带的 KaTeX 渲染（需要 node）。
3. 结构：有“本章怎么学”一节，配图按出现顺序编号，小测 3 到 5 道，有嵌入演示，有 实测数字.json。
4. 演示页不引用外部资源。
5. 本章目录里除讲义、README、figures 以外的文件没有被改动（作业、测试、数据不能动）。
6. 验收脚本：初始作业应显示未完成；给了 --solutions 时参考实现应全部通过。

退出码：有“错误”为 1，只有“提醒”为 0。
"""

from __future__ import annotations

import html
import json
import os
import posixpath
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_site as B  # noqa: E402

ROOT = B.ROOT
ERRORS: list[str] = []
NOTES: list[str] = []


def check_math(body: str) -> None:
    items = []
    for tag, tex in re.findall(r'<(div|span) class="arithmatex">(.*?)</\1>', body, flags=re.S):
        tex = html.unescape(tex).strip()
        tex = re.sub(r"^\\[(\[]", "", tex)
        tex = re.sub(r"\\[)\]]$", "", tex)
        items.append({"display": tag == "div", "tex": tex})
    katex = ROOT / "site" / "assets" / "vendor" / "katex" / "katex.min.js"
    if not items:
        NOTES.append("讲义里没有公式")
        return
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
        path = f.name
    js = ("const katex=require(process.argv[1]);const items=JSON.parse(require('fs').readFileSync(process.argv[2],'utf8'));"
          "const bad=[];for(const it of items){try{katex.renderToString(it.tex,{displayMode:it.display,throwOnError:true,strict:'error'});}"
          "catch(e){bad.push(it.tex.slice(0,80).replace(/\\n/g,' ')+'  =>  '+e.message.slice(0,100));}}"
          "console.log(JSON.stringify({n:items.length,bad}));")
    try:
        out = subprocess.run(["node", "-e", js, str(katex), path], capture_output=True, text=True, timeout=120)
        res = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        NOTES.append(f"公式没有检查（node 不可用或出错：{exc}）")
        return
    finally:
        os.unlink(path)
    print(f"公式 {res['n']} 个，渲染失败 {len(res['bad'])} 个")
    ERRORS.extend(f"公式渲染失败：{b}" for b in res["bad"])


def check_structure(text: str, body: str, ch: dict) -> None:
    if not re.search(r"^## 0\. 本章怎么学", text, flags=re.M):
        ERRORS.append("缺少“## 0. 本章怎么学”一节")
    m0 = re.search(r"^## 0\. 本章怎么学.*?(?=^## |\Z)", text, flags=re.M | re.S)
    head = m0.group(0) if m0 else ""
    for word in ("应用场景", "优先级", "速通", "标准", "深挖"):
        if word not in head:
            ERRORS.append(f"第 0 节里没有出现“{word}”")
    hours = ch.get("hours") or []
    for h, name in zip(hours, ("速通", "标准", "深挖")):
        if not re.search(rf"\|\s*{name}\s*\|\s*{h} 小时", head):
            ERRORS.append(f"第 0 节的{name}档用时应为 {h} 小时（以 course.json 为准）")
    figs = re.findall(r"!\[图 (\d+)：[^\]]*\]\(([^)]+)\)", text)
    other = len(re.findall(r"!\[", text)) - len(figs)
    if other:
        ERRORS.append(f"有 {other} 张图的说明不是“图 N：……”的格式")
    nums = [int(n) for n, _ in figs]
    if nums != list(range(1, len(nums) + 1)):
        ERRORS.append(f"配图没有按出现顺序从 1 连续编号：{nums}")
    for n, path in figs:
        name = posixpath.basename(path)
        if not re.match(rf"fig{int(n):02d}_", name):
            ERRORS.append(f"图 {n} 的文件名应以 fig{int(n):02d}_ 开头：{name}")
    if len(figs) < 4:
        NOTES.append(f"配图只有 {len(figs)} 张，样板章是 5 到 6 张")
    quizzes = body.count('class="quiz"')
    if not 3 <= quizzes <= 5:
        ERRORS.append(f"小测应为 3 到 5 道，现在是 {quizzes} 道")
    embeds = re.findall(r"<!--\s*embed:\s*(\S+)\s+(\d+)\s*-->", text)
    if not embeds:
        NOTES.append("没有嵌入交互演示")
    for path, _ in embeds:
        demo = ROOT / posixpath.normpath(posixpath.join(ch["dir"], path))
        if demo.exists():
            src = demo.read_text(encoding="utf-8")
            ext = re.findall(r'(?:src|href)\s*=\s*["\'](?:https?:)?//[^"\']+', src) + re.findall(r"@import|url\(\s*['\"]?https?:", src)
            if ext:
                ERRORS.append(f"演示页 {demo.name} 引用了外部资源：{ext[:2]}")
    if any("\\\\" in block for block in re.findall(r"^\$\$\n(.*?)\n\$\$$", text, flags=re.M | re.S)):
        ERRORS.append("含换行符 \\\\ 的多行公式要用 ```math 围栏，不要用 $$")
    if "−" in text:
        NOTES.append("讲义里有全角减号“−”，表格和正文里建议统一用“-”")
    numbers = ROOT / ch["dir"] / "figures" / "实测数字.json"
    if not numbers.exists():
        ERRORS.append("缺少 figures/实测数字.json")
    else:
        try:
            json.loads(numbers.read_text(encoding="utf-8"))
        except ValueError as exc:
            ERRORS.append(f"实测数字.json 不是合法 JSON：{exc}")
    if "实测数字.json" not in text:
        ERRORS.append("讲义开头要说明数字来自 figures/实测数字.json 和哪个脚本")
    for sec in ("常见错误", "破坏实验", "自测问题"):
        if not re.search(rf"^## .*{sec}", text, flags=re.M):
            ERRORS.append(f"缺少“{sec}”一节")
    print(f"配图 {len(figs)} 张，小测 {quizzes} 道，嵌入演示 {len(embeds)} 个，讲义 {len(text.splitlines())} 行")


def check_untouched(ch: dict) -> None:
    out = subprocess.run(["git", "status", "--porcelain", "--", ch["dir"]], cwd=ROOT, capture_output=True, text=True).stdout
    allowed = (ch.get("tutorial") or "", "README.md", "figures/")
    for line in out.splitlines():
        path = line[3:].strip().strip('"')
        rel = path[len(ch["dir"]) + 1:] if path.startswith(ch["dir"] + "/") else path
        if "__pycache__" in rel or rel.endswith(".pyc"):
            continue
        if not any(rel == a or (a.endswith("/") and rel.startswith(a)) for a in allowed if a):
            ERRORS.append(f"不该改动的文件有变化：{path}（作业、测试、数据、Notebook、实验记录都不能动）")


def run_tests(ch: dict, solutions: str | None) -> None:
    script = ROOT / ch["dir"] / "tests" / "check_progress.py"
    if not script.exists():
        NOTES.append("本章没有 tests/check_progress.py")
        return

    def run(env_extra):
        env = dict(os.environ, **env_extra)
        env.pop("COURSE_IMPL_DIR", None) if not env_extra else None
        try:
            res = subprocess.run([sys.executable, str(script)], cwd=ROOT, capture_output=True, text=True, timeout=900, env=env)
        except subprocess.TimeoutExpired:
            return None
        m = re.findall(r"通过 (\d+)/(\d+)", res.stdout)
        return (int(m[-1][0]), int(m[-1][1])) if m else res.stdout.strip().splitlines()[-1:] or ["（无输出）"]

    first = run({})
    print("初始作业：", first)
    try:
        raw = subprocess.run([sys.executable, str(script)], cwd=ROOT, capture_output=True, text=True, timeout=900,
                             env={k: v for k, v in os.environ.items() if k != "COURSE_IMPL_DIR"}).stdout
        bad = [l for l in raw.splitlines() if l.lstrip().startswith(("[PASS", "[FAIL", "[ OK"))]
        if bad:
            ERRORS.append(f"初始作业的输出里出现了 PASS 或 FAIL（应全部是 TODO）：{bad[:2]}")
    except subprocess.TimeoutExpired:
        pass
    if isinstance(first, tuple) and first[0] != 0:
        ERRORS.append(f"初始作业不应有测试通过（{first[0]}/{first[1]}），检查是否把答案写进了作业文件")
    if solutions:
        ref = run({"COURSE_IMPL_DIR": str(Path(solutions).resolve())})
        print("参考实现：", ref)
        if isinstance(ref, tuple) and ref[0] != ref[1]:
            ERRORS.append(f"参考实现没有全部通过（{ref[0]}/{ref[1]}）")
        if ref is None:
            ERRORS.append("参考实现的验收超时")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    n = f"{int(args[0]):02d}"
    solutions = sys.argv[sys.argv.index("--solutions") + 1] if "--solutions" in sys.argv else None
    chapters = [c for c in B.COURSE["chapters"] if not c.get("planned")]
    ch = next((c for c in chapters if c["n"] == n), None)
    if ch is None:
        print(f"course.json 里没有第 {n} 章")
        return 2
    name = ch.get("tutorial")
    if not name:
        found = sorted((ROOT / ch["dir"]).glob("教程_*.md"))
        if not found:
            print(f"第 {n} 章没有 教程_*.md")
            return 2
        name = found[0].name
        ch = dict(ch, tutorial=name)
        NOTES.append(f"course.json 里这一章还没有 tutorial 字段，按 {name} 检查；合并时要把它加进 course.json")
    lesson_of = {posixpath.join(c["dir"], c.get("tutorial") or c.get("readme")): f"{c['n']}.html" for c in chapters}
    lesson_of[posixpath.join(ch["dir"], name)] = f"{n}.html"
    text = (ROOT / ch["dir"] / name).read_text(encoding="utf-8")
    print(f"== 第 {n} 章 {ch['title']}（{ch['dir']}/{name}）")
    _, body, _ = B.render_tutorial(text, ch["dir"], lesson_of)
    ERRORS.extend(B.WARNINGS)
    check_math(body)
    check_structure(text, body, ch)
    if "--allow-changes" not in sys.argv:
        check_untouched(ch)
    run_tests(ch, solutions)
    for msg in NOTES:
        print("提醒：", msg)
    for msg in ERRORS:
        print("错误：", msg)
    print("结果：", "通过" if not ERRORS else f"{len(ERRORS)} 个错误")
    return 1 if ERRORS else 0


if __name__ == "__main__":
    sys.exit(main())
