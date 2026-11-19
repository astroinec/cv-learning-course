"""把各章讲义生成为 site/ 下的离线网页。

用法（在仓库根目录运行）：
    python scripts/build_site.py            # 生成
    python scripts/build_site.py --strict   # 有警告就以非零状态退出

依赖：markdown、pymdown-extensions、pygments（维护者才需要，学员不用装）。

讲义只有一份，就是各章目录里的 教程_*.md。它在 GitHub 上能直接阅读，
本脚本再把同一份文件转成带小测、配图和交互演示的网页。约定见 docs/写作规范.md。
生成物：site/index.html、site/lessons/*.html、site/assets/pygments.css。
site/assets 下的 style.css、app.js 和 vendor/ 是手写或第三方文件，本脚本不会改。
"""

from __future__ import annotations

import html
import json
import posixpath
import re
import sys
from pathlib import Path

import markdown
from markdown.extensions.toc import slugify_unicode
from pymdownx.arithmatex import arithmatex_fenced_format
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
TEMPLATE = (ROOT / "scripts" / "site_template.html").read_text(encoding="utf-8")
COURSE = json.loads((ROOT / "course.json").read_text(encoding="utf-8"))
WARNINGS: list[str] = []

CALLOUTS = {"NOTE": ("note", "说明"), "TIP": ("tip", "提示"), "IMPORTANT": ("key", "要点"),
            "WARNING": ("warn", "注意"), "CAUTION": ("danger", "容易踩的坑")}
TIER_INDEX = {"fast": 0, "std": 1, "deep": 2}


def warn(msg: str) -> None:
    WARNINGS.append(msg)


def md_convert(text: str, toc: bool = False):
    exts = ["tables", "sane_lists", "md_in_html", "attr_list", "pymdownx.superfences",
            "pymdownx.highlight", "pymdownx.arithmatex", "pymdownx.tilde"]
    cfg = {"pymdownx.arithmatex": {"generic": True}, "pymdownx.highlight": {"use_pygments": True, "guess_lang": False},
           "pymdownx.superfences": {"custom_fences": [{"name": "math", "class": "arithmatex", "format": arithmatex_fenced_format(which="generic")}]}}
    if toc:
        exts.append("toc")
        cfg["toc"] = {"slugify": slugify_unicode, "toc_depth": "2-3"}
    md = markdown.Markdown(extensions=exts, extension_configs=cfg)
    out = md.convert(text)
    return (out, md.toc_tokens) if toc else out


def md_inline(text: str) -> str:
    out = md_convert(text.strip()).strip()
    m = re.fullmatch(r"<p>(.*)</p>", out, flags=re.S)
    return m.group(1) if m else out


# ---------------------------------------------------------------- 讲义里的扩展块
QUIZ_RE = re.compile(r"<!--\s*quiz\s*-->(.*?)<!--\s*/quiz\s*-->", re.S)
EMBED_RE = re.compile(r"<!--\s*embed:\s*(\S+)\s+(\d+)\s*-->")


def render_quiz(body: str, where: str) -> str:
    opts = re.findall(r"^- ([A-D])\. (.+)$", body, flags=re.M)
    first = re.search(r"^- A\. ", body, flags=re.M)
    detail = re.search(r"</summary>(.*?)</details>", body, flags=re.S)
    answer = re.search(r"\*\*([A-D])\*\*", detail.group(1)) if detail else None
    if not (opts and first and detail and answer):
        warn(f"{where}：小测格式不对，需要题干、A/B/C 选项和带 **答案字母** 的折叠块")
        return ""
    if answer.group(1) not in {k for k, _ in opts}:
        warn(f"{where}：小测答案 {answer.group(1)} 不在选项里")
    question = md_inline(body[:first.start()])
    buttons = "".join(f'<button type="button" data-k="{k}"><b>{k}</b><span>{md_inline(v)}</span></button>' for k, v in opts)
    return (f'<div class="quiz" data-answer="{answer.group(1)}"><div class="quiz-tag">小测</div>'
            f'<div class="quiz-q">{question}</div><div class="quiz-opts">{buttons}</div>'
            f'<div class="quiz-exp" hidden>{md_convert(detail.group(1).strip())}</div></div>')


def fix_url(url: str, chapter_dir: str, lesson_of: dict[str, str]) -> str:
    """讲义里的相对路径是相对章节目录写的，网页在 site/lessons/ 下，要换算。"""
    if re.match(r"^([a-z]+:|#|//)", url):
        return url
    path, _, frag = url.partition("#")
    target = posixpath.normpath(posixpath.join(chapter_dir, path))
    if target in lesson_of:
        return lesson_of[target] + ("#" + frag if frag else "")
    if not (ROOT / target).exists():
        warn(f"{chapter_dir}：链接目标不存在 {url}")
    return "../../" + target + ("#" + frag if frag else "")


def render_tutorial(text: str, chapter_dir: str, lesson_of: dict[str, str]):
    where = chapter_dir
    title = None
    m = re.match(r"\s*#\s+(.+)\n", text)
    if m:
        title, text = m.group(1).strip(), text[m.end():]
    stash: list[str] = []

    def keep(block: str) -> str:
        stash.append(block)
        return f"\n\nSTASH{len(stash) - 1}BLOCK\n\n"

    text = re.sub(r"(^>.*\n)\n(?=>)", r"\1\n<!-- -->\n\n", text, flags=re.M)  # 相邻的两个引用块之间加分隔，否则会被并成一个
    text = QUIZ_RE.sub(lambda mm: keep(render_quiz(mm.group(1), where)), text)
    text = EMBED_RE.sub(lambda mm: keep(
        f'<div class="embed"><iframe src="{html.escape(fix_url(mm.group(1), chapter_dir, lesson_of))}" '
        f'height="{mm.group(2)}" loading="lazy" title="交互演示"></iframe></div>'), text)
    body, toc = md_convert(text, toc=True)
    body = re.sub(r"<p>STASH(\d+)BLOCK</p>", lambda mm: stash[int(mm.group(1))], body)

    def callout(mm):
        kind, label = CALLOUTS[mm.group(1)]
        return f'<div class="callout callout-{kind}"><div class="callout-title">{label}</div><p>{mm.group(2)}</div>'

    body = re.sub(r"<blockquote>\s*<p>\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*(.*?)</blockquote>", callout, body, flags=re.S)

    def figure(mm):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', mm.group(1)))
        src = html.escape(fix_url(html.unescape(attrs.get("src", "")), chapter_dir, lesson_of))
        alt = attrs.get("alt", "")
        return f'<figure><img src="{src}" alt="{alt}" loading="lazy"><figcaption>{alt}</figcaption></figure>'

    body = re.sub(r"<p><img ([^>]*?)\s*/?></p>", figure, body)
    body = re.sub(r'(<a [^>]*?href=")([^"]+)(")',
                  lambda mm: mm.group(1) + html.escape(fix_url(html.unescape(mm.group(2)), chapter_dir, lesson_of)) + mm.group(3), body)
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    if "[!" in body:
        warn(f"{where}：有没转换成功的提示框标记")
    return title, body, toc


# ---------------------------------------------------------------- 页面拼装
def tier_hours(ch: dict, tier: str):
    return ch["hours"][TIER_INDEX[tier]] if ch.get("hours") else None


def lesson_href(ch: dict, root: str) -> str:
    return f"{root}lessons/{ch['n']}.html"


def sidebar(current: str | None, root: str) -> str:
    out = []
    for part in COURSE["parts"]:
        out.append(f'<div class="side-part">{html.escape(part["title"])}</div><ul>')
        for ch in (c for c in COURSE["chapters"] if c["part"] == part["id"]):
            label = f'<span class="dot {ch["priority"]}"></span><span class="side-n">{ch["n"]}</span>{html.escape(ch["title"])}'
            if ch.get("planned"):
                out.append(f'<li class="planned"><span>{label}</span></li>')
            else:
                cls = ' class="current"' if ch["n"] == current else ""
                out.append(f'<li{cls} data-ch="{ch["n"]}"><a href="{lesson_href(ch, root)}">{label}</a></li>')
        out.append("</ul>")
    return "\n".join(out)


def toc_html(tokens: list) -> str:
    if not tokens:
        return ""
    items = []
    for t in tokens:
        items.append(f'<li><a href="#{t["id"]}">{t["name"]}</a></li>')
        items += [f'<li class="sub"><a href="#{c["id"]}">{c["name"]}</a></li>' for c in t.get("children", [])]
    return '<div class="toc-title">本页目录</div><ul>' + "".join(items) + "</ul>"


def page(**kw) -> str:
    out = TEMPLATE
    for key, value in kw.items():
        out = out.replace("{{" + key + "}}", value)
    left = re.findall(r"\{\{(\w+)\}\}", out)
    if left:
        warn(f"模板里有没填的占位符 {sorted(set(left))}")
    return out


def lesson_header(ch: dict, by_n: dict, root: str) -> str:
    tiers = COURSE["tiers"]
    hours = "".join(f'<span class="chip"><b>{tiers[k]["name"]}</b> {tier_hours(ch, k)} 小时</span>' for k in ("fast", "std", "deep"))
    needs = "、".join(f'<a href="{lesson_href(by_n[n], root)}">第 {int(n)} 章 {html.escape(by_n[n]["title"])}</a>' for n in ch["needs"]) or "无"
    state = ('<span class="badge ok">已升级</span>' if ch.get("upgraded") else '<span class="badge">原版讲义</span>')
    banner = "" if ch.get("upgraded") else ('<div class="callout callout-note"><div class="callout-title">说明</div><p>这一章还是原版讲义，内容完整，'
                                            '配图、推导、小测和交互演示会按第 16 章的样式补上。</p></div>')
    return (f'<div class="lesson-head"><div class="crumbs">第 {int(ch["n"])} 章 <span class="tag {ch["priority"]}">{ch["priority"]}</span> {state}</div>'
            f'<h1>{html.escape(ch["title"])}</h1><p class="scene"><b>用在哪：</b>{html.escape(ch["scene"])}</p>'
            f'<div class="meta-row">{hours}<span class="chip">前置：{needs}</span></div>'
            f'<div class="meta-row"><span class="chip mono">验收：python {html.escape(ch["dir"])}/tests/check_progress.py</span>'
            f'<label class="chip done-pick">我的进度 <select data-done="{ch["n"]}"><option value="">未开始</option>'
            f'<option value="fast">速通完成</option><option value="std">标准完成</option><option value="deep">深挖完成</option></select></label></div></div>{banner}')


def build_lessons() -> None:
    chapters = [c for c in COURSE["chapters"] if not c.get("planned")]
    by_n = {c["n"]: c for c in COURSE["chapters"]}
    lesson_of = {}
    for c in chapters:
        src = c.get("tutorial") or c.get("readme")
        lesson_of[posixpath.join(c["dir"], src)] = f"{c['n']}.html"
    (SITE / "lessons").mkdir(parents=True, exist_ok=True)
    for i, ch in enumerate(chapters):
        src = ROOT / ch["dir"] / (ch.get("tutorial") or ch["readme"])
        if not src.exists():
            warn(f"第 {ch['n']} 章：找不到讲义 {src.relative_to(ROOT)}")
            continue
        _, body, toc = render_tutorial(src.read_text(encoding="utf-8"), ch["dir"], lesson_of)
        nav = '<nav class="prev-next">'
        nav += f'<a href="{chapters[i - 1]["n"]}.html">← 第 {int(chapters[i - 1]["n"])} 章 {html.escape(chapters[i - 1]["title"])}</a>' if i else "<span></span>"
        nav += f'<a href="{chapters[i + 1]["n"]}.html">第 {int(chapters[i + 1]["n"])} 章 {html.escape(chapters[i + 1]["title"])} →</a>' if i + 1 < len(chapters) else "<span></span>"
        nav += "</nav>"
        main = lesson_header(ch, by_n, "../") + f'<article class="prose">{body}</article>' + nav
        out = page(page_title=f"第 {int(ch['n'])} 章 {ch['title']}", site_title=COURSE["title"], root="../", body_class="lesson",
                   sidebar=sidebar(ch["n"], "../"), main=main, toc=toc_html(toc), data="null")
        (SITE / "lessons" / f"{ch['n']}.html").write_text(out, encoding="utf-8")


def build_home() -> None:
    data = json.dumps(COURSE, ensure_ascii=False)
    built = [c for c in COURSE["chapters"] if not c.get("planned")]
    std_total = sum(c["hours"][1] for c in built)
    fast_total = sum(c["hours"][0] for c in built)
    tiers = "".join(
        f'<div class="tier-card"><h3>{t["name"]}</h3><p>{html.escape(t["what"])}。</p><p class="muted">算完成：{html.escape(t["done"])}。</p></div>'
        for t in COURSE["tiers"].values())
    demos = "".join(f'<a class="demo-card" href="../{html.escape(d["file"])}"><b>{html.escape(d["title"])}</b><span>{html.escape(d["chapters"])}</span></a>'
                    for d in COURSE["demos"])
    for d in COURSE["demos"]:
        if not (ROOT / d["file"]).exists():
            warn(f"首页：演示文件不存在 {d['file']}")
    main = f'''
<section class="hero">
  <h1>{html.escape(COURSE["title"])}</h1>
  <p>{html.escape(COURSE["subtitle"])}</p>
  <div class="hero-stats">
    <div><b>{len(built)}</b><span>章已写完</span></div>
    <div><b>{len(COURSE["chapters"]) - len(built)}</b><span>章规划中</span></div>
    <div><b>{fast_total}</b><span>小时，全书速通</span></div>
    <div><b>{std_total}</b><span>小时，全书标准档</span></div>
  </div>
</section>
<section>
  <h2>先选路线，再开始学</h2>
  <p class="muted">不需要从第 1 章按顺序学到最后。每条路线只列它需要的章节和档位，总时长是有限的。时长都是估计值，供排计划用。</p>
  <div class="route-pills" id="routes"></div>
  <div class="route-info" id="routeInfo"></div>
  <div class="route-list" id="routeList"></div>
</section>
<section>
  <h2>每章三个档位</h2>
  <div class="tier-cards">{tiers}</div>
  <div class="callout callout-tip"><div class="callout-title">超时规则</div><p>一章用时超过所选档位的 1.5 倍还没完成，就把卡点记到“学习进度.md”，先往下走。速通档做完就算见过这一章，以后用到再升档。</p></div>
</section>
<section>
  <h2>优先级的含义</h2>
  <div class="prio-list">{"".join(f'<div><span class="tag {k}">{k}</span>{html.escape(v)}</div>' for k, v in COURSE["priorities"].items())}</div>
</section>
<section>
  <h2>交互演示</h2>
  <div class="demo-cards">{demos}</div>
</section>'''
    out = page(page_title="首页", site_title=COURSE["title"], root="", body_class="home", sidebar=sidebar(None, ""),
               main=main, toc="", data=data.replace("</", "<\\/"))
    (SITE / "index.html").write_text(out, encoding="utf-8")


def check_course() -> None:
    known = {c["n"] for c in COURSE["chapters"]}
    for c in COURSE["chapters"]:
        for n in c["needs"]:
            if n not in known:
                warn(f"course.json：第 {c['n']} 章的前置 {n} 不存在")
        if not c.get("planned") and not (ROOT / c["dir"]).is_dir():
            warn(f"course.json：目录不存在 {c['dir']}")
    for r in COURSE["routes"]:
        if isinstance(r["items"], list):
            for n, tier in r["items"]:
                if n not in known or tier not in TIER_INDEX:
                    warn(f"course.json：路线 {r['name']} 里的 {n}/{tier} 无效")


def main() -> int:
    check_course()
    (SITE / "assets").mkdir(parents=True, exist_ok=True)
    (SITE / "assets" / "pygments.css").write_text(HtmlFormatter(style="monokai").get_style_defs(".highlight") + "\n", encoding="utf-8")
    build_lessons()
    build_home()
    (ROOT / "index.html").write_text(
        '<!DOCTYPE html>\n<html lang="zh-CN"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=site/index.html">'
        '<title>视觉算法动手课程</title></head><body><a href="site/index.html">打开课程网页</a></body></html>\n', encoding="utf-8")
    (ROOT / ".nojekyll").write_text("", encoding="utf-8")
    for w in WARNINGS:
        print("警告：", w)
    print(f"生成 {len(list((SITE / 'lessons').glob('*.html')))} 个章节页和首页，警告 {len(WARNINGS)} 条")
    return 1 if (WARNINGS and "--strict" in sys.argv) else 0


if __name__ == "__main__":
    sys.exit(main())
