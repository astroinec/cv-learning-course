"""用无界面 Chrome 检查 demos/ 下的演示页，并核对各章讲义里写的嵌入高度。

用法（在仓库根目录运行，需要本机装有 Google Chrome）：
    python scripts/check_demos.py            # 只报告
    python scripts/check_demos.py --fix      # 把讲义里的嵌入高度改成 实测高度 + 24

检查内容：在 860 px 宽（讲义正文里嵌入框的宽度）下的实际高度、有没有横向溢出、加载和逐个点击按钮时有没有脚本报错、
有没有引用外部资源、有没有整张空白的画布、“载入”按钮有几个。
"""

from __future__ import annotations

import functools
import http.server
import json
import re
import subprocess
import sys
import threading
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MARGIN = 24


def measure(files: list) -> list:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):  # 不打印访问日志
            pass

    handler = functools.partial(Quiet, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    out = []
    try:
        for i in range(0, len(files), 6):
            query = urllib.parse.quote("|".join(files[i:i + 6]))
            url = f"http://127.0.0.1:{server.server_address[1]}/scripts/demo_check.html?files={query}"
            res = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-first-run", "--hide-scrollbars", "--window-size=1200,1400",
                                  "--virtual-time-budget=60000", "--dump-dom", url], capture_output=True, text=True, timeout=180)
            m = re.search(r"RESULT(.*?)ENDRESULT", res.stdout, flags=re.S)
            if not m:
                out += [{"file": f, "fatal": "没有拿到结果"} for f in files[i:i + 6]]
                continue
            import html
            out += json.loads(html.unescape(m.group(1)))
    finally:
        server.shutdown()
    return out


def main() -> int:
    files = sorted(p.name for p in (ROOT / "demos").glob("*.html"))
    results = {r["file"]: r for r in measure(files)}
    embeds = {}
    for md in sorted(ROOT.glob("[0-9][0-9]_*/教程_*.md")):
        text = md.read_text(encoding="utf-8")
        for m in re.finditer(r"<!--\s*embed:\s*\.\./demos/(\S+)\s+(\d+)\s*-->", text):
            embeds.setdefault(m.group(1), []).append((md, int(m.group(2))))
    problems = 0
    for f in files:
        r = results.get(f, {})
        if r.get("fatal"):
            print(f"错误  {f}: {r['fatal']}")
            problems += 1
            continue
        notes = []
        if r["errors"]:
            notes.append(f"脚本报错 {r['errors']}")
        if r["overflowX"] > 2:
            notes.append(f"横向溢出 {r['overflowX']} px")
        if r["external"]:
            notes.append(f"引用外部资源 {r['external']} 处")
        if r["blankCanvas"]:
            notes.append(f"空白画布 {r['blankCanvas']}/{r['canvases']}")
        if not r["loaders"]:
            notes.append("没有“载入”按钮")
        used = embeds.get(f, [])
        if not used:
            notes.append("没有讲义嵌入它")
        for md, h in used:
            want = r["height"] + MARGIN
            if abs(h - want) > 30:
                notes.append(f"{md.parent.name[:2]} 章写的嵌入高度 {h}，实测需要 {want}")
                if "--fix" in sys.argv:
                    text = md.read_text(encoding="utf-8")
                    text = re.sub(rf"(<!--\s*embed:\s*\.\./demos/{re.escape(f)}\s+)\d+(\s*-->)", rf"\g<1>{want}\g<2>", text)
                    md.write_text(text, encoding="utf-8")
        hard = [n for n in notes if not n.startswith(("没有“载入”", "没有讲义")) and "嵌入高度" not in n]
        problems += len(hard)
        print(f"{'问题' if hard else '正常'}  {f}: 高 {r['height']} px，按钮 {r['buttons']} 个（载入 {r['loaders']}）" + ("；" + "；".join(notes) if notes else ""))
    print(f"共 {len(files)} 个演示页，{problems} 个需要处理的问题" + ("；已按实测改写嵌入高度" if "--fix" in sys.argv else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
