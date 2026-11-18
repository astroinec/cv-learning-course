/* 课程站点交互。手写文件，scripts/build_site.py 不会覆盖它。没有第三方依赖（公式渲染用同目录 vendor 里的 KaTeX）。 */
(function () {
  "use strict";
  const KEY = "cv-course-progress-v1";
  const TIERS = ["fast", "std", "deep"];
  const TIER_NAME = { fast: "速通", std: "标准", deep: "深挖" };
  const $ = (s, el) => (el || document).querySelector(s);
  const $$ = (s, el) => [...(el || document).querySelectorAll(s)];
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* 无痕模式下不保存 */ } },
  };
  let progress = store.get(KEY, {});

  // 主题与移动端目录
  $(".theme-btn").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("cv-theme", next); } catch (e) { /* 忽略 */ }
  });
  $(".menu-btn").addEventListener("click", () => document.body.classList.toggle("menu-open"));
  $(".mask").addEventListener("click", () => document.body.classList.remove("menu-open"));

  // 进度
  function paintProgress() {
    const items = $$(".sidebar li[data-ch]");
    let done = 0;
    for (const li of items) { const ok = !!progress[li.dataset.ch]; li.classList.toggle("done", ok); done += ok; }
    $(".overall .bar i").style.width = (100 * done / Math.max(1, items.length)) + "%";
    $(".overall-text").textContent = `${done}/${items.length} 章`;
    for (const sel of $$("select[data-done]")) sel.value = progress[sel.dataset.done] || "";
  }
  for (const sel of $$("select[data-done]")) {
    sel.addEventListener("change", () => {
      if (sel.value) progress[sel.dataset.done] = sel.value; else delete progress[sel.dataset.done];
      store.set(KEY, progress); paintProgress();
    });
  }
  const cur = $(".sidebar li.current");
  if (cur) cur.scrollIntoView({ block: "center" });

  // 公式
  if (window.katex) {
    for (const el of $$(".arithmatex")) {
      const display = el.tagName === "DIV";
      const tex = el.textContent.trim().replace(/^\\[(\[]/, "").replace(/\\[)\]]$/, "");
      try { katex.render(tex, el, { displayMode: display, throwOnError: false }); } catch (e) { /* 保留原文 */ }
    }
  }

  // 小测
  for (const quiz of $$(".quiz")) {
    const answer = quiz.dataset.answer;
    for (const btn of $$(".quiz-opts button", quiz)) {
      btn.addEventListener("click", () => {
        for (const b of $$(".quiz-opts button", quiz)) b.classList.remove("right", "wrong");
        btn.classList.add(btn.dataset.k === answer ? "right" : "wrong");
        if (btn.dataset.k !== answer) $(`.quiz-opts button[data-k="${answer}"]`, quiz).classList.add("right");
        $(".quiz-exp", quiz).hidden = false;
      });
    }
  }

  // 图片放大
  const box = $(".lightbox");
  for (const img of $$("figure img")) img.addEventListener("click", () => { $("img", box).src = img.src; box.hidden = false; });
  box.addEventListener("click", () => { box.hidden = true; });
  document.addEventListener("keydown", e => { if (e.key === "Escape") box.hidden = true; });

  // 右侧目录跟随滚动
  const tocLinks = $$(".toc a");
  if (tocLinks.length) {
    const heads = tocLinks.map(a => document.getElementById(decodeURIComponent(a.getAttribute("href").slice(1)))).filter(Boolean);
    const spy = () => {
      let idx = 0;
      heads.forEach((h, i) => { if (h.getBoundingClientRect().top < 90) idx = i; });
      tocLinks.forEach((a, i) => a.classList.toggle("active", i === idx));
    };
    document.addEventListener("scroll", spy, { passive: true }); spy();
  }

  // 首页：路线
  const course = window.COURSE;
  if (course && $("#routes")) {
    const byN = Object.fromEntries(course.chapters.map(c => [c.n, c]));
    let routeId = store.get("cv-route", "main");
    if (!course.routes.some(r => r.id === routeId)) routeId = "main";
    let weekly = store.get("cv-weekly", 20);
    const itemsOf = r => Array.isArray(r.items) ? r.items : course.chapters.map(c => [c.n, r.items === "all-fast" ? "fast" : "std"]);
    const isDone = (n, tier) => progress[n] && TIERS.indexOf(progress[n]) >= TIERS.indexOf(tier);
    const esc = s => String(s).replace(/[&<>"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));

    function renderHome() {
      const route = course.routes.find(r => r.id === routeId);
      $("#routes").innerHTML = course.routes.map(r => `<button type="button" data-r="${r.id}" class="${r.id === routeId ? "on" : ""}">${esc(r.name)}</button>`).join("");
      const items = itemsOf(route);
      let total = 0, left = 0, planned = 0, done = 0;
      const rows = items.map(([n, tier]) => {
        const c = byN[n], h = c.hours ? c.hours[TIERS.indexOf(tier)] : null, ok = isDone(n, tier);
        if (c.planned) planned++; else { total += h; if (ok) done++; else left += h; }
        const state = c.planned ? `<span class="badge">${c.proposed ? "建议新增，规划中" : "规划中"}</span>` : (c.upgraded ? '<span class="badge ok">已升级</span>' : "");
        const title = c.planned ? esc(c.title) : `<a href="lessons/${n}.html">${esc(c.title)}</a>`;
        return `<div class="row${ok ? " done" : ""}${c.planned ? " planned" : ""}"><span class="mark">${ok ? "✓" : ""}</span><span class="n">${n}</span>
          <span class="title">${title}</span>
          <span class="tags"><span class="tag ${c.priority}">${c.priority}</span><span class="badge">${TIER_NAME[tier]}${h === null ? "" : " " + h + " 小时"}</span>${state}</span>
          <p class="scene">${esc(c.scene)}</p></div>`;
      });
      const weeks = left ? Math.ceil(left / Math.max(1, weekly)) : 0;
      $("#routeInfo").innerHTML = `<span><b>${items.length - planned}</b> 章${planned ? `，另有 ${planned} 章规划中未计入` : ""}</span>
        <span>共 <b>${total}</b> 小时</span><span>已完成 <b>${done}</b> 章，剩余 <b>${left}</b> 小时</span>
        <span>每周 <input type="number" id="weekly" min="1" max="80" value="${weekly}"> 小时，还需约 <b>${weeks}</b> 周</span>
        <span class="when">适用：${esc(route.when)}</span>`;
      $("#routeList").innerHTML = rows.join("");
      $("#weekly").addEventListener("change", e => { weekly = Math.max(1, +e.target.value || 20); store.set("cv-weekly", weekly); renderHome(); });
    }
    $("#routes").addEventListener("click", e => { const id = e.target.dataset.r; if (id) { routeId = id; store.set("cv-route", id); renderHome(); } });
    renderHome();
  }

  paintProgress();
})();
