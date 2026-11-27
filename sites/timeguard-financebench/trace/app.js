/* TimeGuard-RAG · 逐题回放
   数据来自 window.TRACE_DATA(开发时由 data.js 提供)或页面内嵌的 JSON。
   页面内容一律用 textContent 渲染;回答正文先经 esc() 转义,再套极简的 Markdown 规则,不执行任何来自数据的内容。 */
(function () {
  "use strict";
  var DATA = window.TRACE_DATA || JSON.parse(document.getElementById("trace-data").textContent);
  var QS = DATA.questions, CH = DATA.chunks, DS = DATA.dataset, META = DATA.meta;
  var A0 = "A0", A1 = "A1";
  var ARM_NAME = { A0: "原版", A1: "增强版" };
  var ARM_SUB = { A0: "RAGFlow 原生流程,按原生规则选证据", A1: "同一套原生流程 + 你做的增强步骤" };
  var SPLIT = { tune: "调优组", dev: "开发组", test: "保留组" };
  var TYPE = { "domain-relevant": "常规分析题", "novel-generated": "分析师新编题", "metrics-generated": "财务指标生成题" };
  var DECIDE = ["复查已检索原文", "补搜后按事实筛选"];
  var TOOL = { rag: "检索研究", formalize: "规范化问题", retrieve: "补搜" };
  var CHECK = { intact: "与索引原文一致", verbatim_excerpt: "原文里的连续摘录", native_table_view: "原生表格视图", edited: "与索引原文对不上" };
  var CALL = {
    formalize_question: { title: "RAGFlow 原有 · 把问题规范化", note: "模型把题干整理成规范的问题和关键词" },
    rag_agent: { title: "研究思考" },
    sca: { title: "RAGFlow 原有 · 检查材料够不够", note: "专门的检查模型看现有材料能不能回答这道题" },
    query_rewrite: { title: "RAGFlow 原有 · 把缺口改写成新检索词", note: "检查认为材料不够,把缺的部分改写成检索词,回到研究循环再搜" },
    formalize_answer: { title: "写最终回答" }
  };
  var IDS = Object.keys(QS).sort();
  var STEPS = [["scope", "范围"], ["search", "检索"], ["enh", "增强"], ["blocks", "证据"], ["answer", "回答"], ["judge", "评分"]];
  var CMP = [["", "全部题目"], ["a1", "增强版达标、原版没达标"], ["a0", "原版达标、增强版没达标"], ["both", "两组都达标"], ["none", "两组都没达标"], ["unk", "其余(含未知)"]];
  var state = { filter: { text: "", split: "", type: "", company: "", cmp: "" }, qid: null, visible: [] };

  /* ------------------------------------------------------------ 小工具 */
  function append(el, k) {
    if (k == null || k === false) return;
    if (Array.isArray(k)) { k.forEach(function (x) { append(el, x); }); return; }
    el.append(k.nodeType ? k : document.createTextNode(String(k)));
  }
  function h(tag, attrs) {
    var el = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (v == null || v === false) return;
      if (k === "class") el.className = v;
      else if (k === "text") el.textContent = v;
      else if (k === "html") el.innerHTML = v;
      else if (k.indexOf("on") === 0) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    });
    for (var i = 2; i < arguments.length; i++) append(el, arguments[i]);
    return el;
  }
  function num(n) { return n == null ? "—" : Number(n).toLocaleString("en-US"); }
  function sec(s) { if (s == null) return "—"; return s < 10 ? Math.round(s * 10) / 10 : Math.round(s); }
  function shortId(id) { return id.slice(-5); }
  function stripPdf(name) { return (name || "").replace(/\.pdf$/i, ""); }
  function pagesText(p) {
    if (!p || !p.length) return "";
    var u = p.filter(function (x, i) { return p.indexOf(x) === i; }).sort(function (a, b) { return a - b; });
    if (u.length === 1) return "第 " + u[0] + " 页";
    var run = u.every(function (x, i) { return i === 0 || x === u[i - 1] + 1; });
    if (run) return "第 " + u[0] + "–" + u[u.length - 1] + " 页";
    return "第 " + u.slice(0, 6).join("、") + (u.length > 6 ? " …" : "") + " 页";
  }
  function joint(a) {
    var d = (a && a.judge) || {}, x = d.answer_correct, y = d.citation_support;
    if (x === false || y === false) return false;
    return (x === true && y === true) ? true : null;
  }
  function vc(v) { return v === true ? "ok" : v === false ? "bad" : "unk"; }
  function mark(v) { return v === true ? "✓" : v === false ? "✗" : "?"; }
  function vt(v, okText, badText) { return v === true ? okText : v === false ? badText : "无法判断"; }
  function badge(v, okText, badText) { return h("span", { class: "badge " + vc(v), text: vt(v, okText, badText) }); }
  function chip(label, v) {
    return h("span", { class: "chip " + vc(v), text: label + mark(v), title: label + ":" + vt(v, "是", "否") });
  }
  function delivered(arm) { return arm.blocks.filter(function (b) { return b.delivered; }).length; }

  /* ------------------------------------------------------------ 极简 Markdown(先转义) */
  var CITE = /\[(?:ID:)?([0-9]+)\]/g;
  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
  function inline(s) {
    var t = esc(s);
    t = t.replace(/`([^`]+)`/g, function (m, c) { return "<code>" + c + "</code>"; });
    t = t.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    t = t.replace(CITE, function (m, n) { return '<span class="cite" data-n="' + n + '" role="button" tabindex="0">[' + n + "]</span>"; });
    return t;
  }
  function cellsOf(r) {
    return r.trim().replace(/^\|/, "").replace(/\|$/, "").split(/(?<!\\)\|/).map(function (c) { return c.trim().replace(/\\\|/g, "|"); });
  }
  function isSep(c) { return /^:?-{2,}:?$/.test(c); }
  function tableHtml(rows) {
    var body = rows.map(cellsOf).filter(function (cs) { return !cs.every(isSep); });
    if (!body.length) return "";
    var w = body[0].length, out = '<div class="tw"><table><thead><tr>';
    body[0].forEach(function (c) { out += "<th>" + inline(c) + "</th>"; });
    out += "</tr></thead><tbody>";
    body.slice(1).forEach(function (r) {
      out += "<tr>";
      for (var i = 0; i < w; i++) out += "<td>" + inline(r[i] || "") + "</td>";
      out += "</tr>";
    });
    return out + "</tbody></table></div>";
  }
  function md(src) {
    var lines = String(src || "").replace(/\r\n?/g, "\n").split("\n"), out = [], para = [], i = 0, m;
    function flush() { if (para.length) { out.push("<p>" + para.map(inline).join("<br>") + "</p>"); para = []; } }
    // 表格行:一行里至少两个竖线,首尾的竖线可以省略(模型的输出两种写法都有)。只有一个竖线的多半是正文里的"a | b",不当表格。
    function isRow(l) {
      var cells = l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(function (c) { return c.trim(); });
      return cells.length >= 3 && cells.filter(Boolean).length >= 2;
    }
    while (i < lines.length) {
      var line = lines[i];
      if (!line.trim()) { flush(); i++; continue; }
      if (isRow(line)) { flush(); var rows = []; while (i < lines.length && isRow(lines[i])) rows.push(lines[i++]); out.push(tableHtml(rows)); continue; }
      if ((m = line.match(/^\s{0,3}(#{1,6})\s+(.*)$/))) { flush(); out.push("<h4>" + inline(m[2]) + "</h4>"); i++; continue; }
      if (/^\s*([-*•]|\d+[.)])\s+/.test(line)) {
        flush();
        var ordered = /^\s*\d+[.)]\s+/.test(line), items = [];
        while (i < lines.length && (m = lines[i].match(/^\s*(?:([-*•])|(\d+[.)]))\s+(.*)$/))) { items.push("<li>" + inline(m[3]) + "</li>"); i++; }
        out.push(ordered ? "<ol>" + items.join("") + "</ol>" : "<ul>" + items.join("") + "</ul>");
        continue;
      }
      para.push(line.trim()); i++;
    }
    flush();
    return out.join("");
  }

  /* ------------------------------------------------------------ 题库 */
  function cmpOf(q) {
    var a = joint(q.arms[A0]), b = joint(q.arms[A1]);
    if (a === true && b === true) return "both";
    if (a === false && b === false) return "none";
    if (b === true) return "a1";
    if (a === true) return "a0";
    return "unk";
  }
  function fillFilters() {
    function opts(el, pairs) { pairs.forEach(function (p) { el.append(h("option", { value: p[0], text: p[1] })); }); }
    opts(document.getElementById("f-split"), [["", "全部分组"]].concat(Object.keys(SPLIT).map(function (k) { return [k, SPLIT[k]]; })));
    opts(document.getElementById("f-type"), [["", "全部题型"]].concat(Object.keys(TYPE).map(function (k) { return [k, TYPE[k]]; })));
    var companies = {};
    IDS.forEach(function (id) { var c = DS[id] && DS[id].company; if (c) companies[c] = (companies[c] || 0) + 1; });
    opts(document.getElementById("f-company"), [["", "全部公司"]].concat(Object.keys(companies).sort(function (a, b) { return a.localeCompare(b); }).map(function (c) { return [c, c + "(" + companies[c] + " 题)"]; })));
    opts(document.getElementById("f-cmp"), CMP);
  }
  function applyFilter() {
    var f = state.filter, t = f.text.trim().toLowerCase();
    state.visible = IDS.filter(function (id) {
      var q = QS[id], d = DS[id] || {};
      if (f.split && q.split !== f.split) return false;
      if (f.type && d.type !== f.type) return false;
      if (f.company && d.company !== f.company) return false;
      if (f.cmp && cmpOf(q) !== f.cmp) return false;
      if (t && (id + " " + (q.q || "") + " " + (d.company || "") + " " + (d.doc || "")).toLowerCase().indexOf(t) < 0) return false;
      return true;
    });
  }
  function rowFor(id) {
    var q = QS[id], d = DS[id] || {};
    function vline(key, label) { var j = q.arms[key].judge || {}; return h("span", { class: "vline" }, h("i", null, label), chip("答", j.answer_correct), chip("引", j.citation_support)); }
    var row = h("button", { type: "button", class: "qrow" + (id === state.qid ? " sel" : ""), "aria-pressed": id === state.qid ? "true" : "false",
      onclick: function () { select(id, true); } },
      h("div", { class: "qrow-top" }, h("span", { class: "qid" }, "#" + shortId(id)), h("span", { class: "tag" }, SPLIT[q.split] || q.split), d.type ? h("span", { class: "tag" }, TYPE[d.type] || d.type) : null),
      h("div", { class: "qrow-v" }, vline(A0, "原"), vline(A1, "增")),
      h("div", { class: "qrow-q" }, q.q || ""),
      h("div", { class: "qrow-meta" }, [d.company, d.doc].filter(Boolean).join(" · ")));
    row.dataset.id = id;
    return row;
  }
  function renderList() {
    var list = document.getElementById("list");
    list.replaceChildren.apply(list, state.visible.length ? state.visible.map(rowFor) : [h("div", { class: "list-empty" }, "没有符合当前条件的题目。")]);
    document.getElementById("list-count").textContent = "当前显示 " + state.visible.length + " 题(共 " + IDS.length + " 题)";
    document.getElementById("picker-count").textContent = "(" + state.visible.length + "/" + IDS.length + ")";
  }

  /* ------------------------------------------------------------ 本题:骨架 */
  function armsBox(l, r) { return h("div", { class: "arms" }, armBox(A0, l), armBox(A1, r)); }
  function armBox(key, node) {
    return h("section", { class: "arm arm-" + key, "data-arm": key },
      h("header", { class: "arm-h" }, h("b", null, ARM_NAME[key]), h("span", null, ARM_SUB[key])), h("div", { class: "arm-b" }, node));
  }
  function step(no, id, title, help, body) {
    return h("section", { class: "step", id: "step-" + id },
      h("div", { class: "step-h" }, h("span", { class: "step-n", text: no }), h("h2", null, title), h("p", null, help)), body);
  }

  function questionCard(q, d) {
    return h("section", { class: "card" },
      h("div", { class: "qcard-top" },
        h("span", { class: "qid" }, "#" + shortId(q.id)),
        h("span", { class: "tag" }, SPLIT[q.split] || q.split),
        d.type ? h("span", { class: "tag" }, TYPE[d.type] || d.type) : null,
        d.reasoning ? h("span", { class: "tag" }, d.reasoning) : null),
      h("strong", null, "用户输入 · 作答系统只收到下面这道题"),
      h("p", { class: "qtext" }, q.q || ""),
      h("p", { class: "muted" }, "题干在实验前已经固定。两个版本各自从这句话出发查财报、写回答。展开下方材料，可以用裁判看到的答案和出处做对照。"),
      h("details", { class: "refbox" },
        h("summary", { class: "ref-label" }, "查看评分用的固定材料 · 参考答案和人工标注证据"),
        h("p", null, [d.company, d.doc].filter(Boolean).join(" · ")),
        h("dl", { class: "jrow" }, h("dt", null, "参考答案"), h("dd", { class: "plain", text: d.answer || "(无)" })),
        (d.evidence || []).map(function (ev) {
          return h("details", { class: "evd" },
            h("summary", null, "标注证据 · " + (ev.doc || d.doc || "") + (ev.page != null ? " · PDF 第 " + (ev.page + 1) + " 页" : "")),
            h("div", { class: "source", text: ev.text || "" }));
        })));
  }

  function compareTable(a0, a1) {
    function delta(x, y, unit) {
      if (typeof x !== "number" || typeof y !== "number") return null;
      var d = Math.round((y - x) * 10) / 10;
      return d ? h("span", { class: "delta " + (d > 0 ? "up" : "down"), text: (d > 0 ? "+" : "") + num(d) + (unit || "") }) : null;
    }
    function row(name, x, y, dx, dy, unit) { return h("tr", null, h("td", null, name), h("td", null, x), h("td", null, y), h("td", null, delta(dx, dy, unit))); }
    var j0 = a0.judge || {}, j1 = a1.judge || {}, e1 = a1.enh || {};
    return h("section", { class: "card" },
      h("div", { class: "card-title" }, "两个版本在这道题上的对比"),
      h("div", { class: "cmp-wrap" }, h("table", { class: "cmp" },
        h("thead", null, h("tr", null, h("th", null, "项目"), h("th", null, "原版"), h("th", null, "增强版"), h("th", null, "差值"))),
        h("tbody", null,
          row("判定(答案 / 引用)", h("span", { class: "vline" }, chip("答", j0.answer_correct), chip("引", j0.citation_support)), h("span", { class: "vline" }, chip("答", j1.answer_correct), chip("引", j1.citation_support))),
          row("检索次数", a0.searches, a1.searches, a0.searches, a1.searches),
          row("其中补搜", 0, e1.supplement_calls == null ? "—" : e1.supplement_calls),
          row("模型调用", a0.calls, a1.calls, a0.calls, a1.calls),
          row("输入 token", num(a0.tok_in), num(a1.tok_in), a0.tok_in, a1.tok_in),
          row("证据块(选中 / 完整送达)", a0.blocks.length + " / " + delivered(a0), a1.blocks.length + " / " + delivered(a1)),
          row("用时", sec(a0.elapsed) + " 秒", sec(a1.elapsed) + " 秒", a0.elapsed, a1.elapsed, " 秒")))));
  }

  /* ------------------------------------------------------------ 1 确定范围 */
  function planCard(a) {
    var p = a.plan || {}, docs = p.docs || [], none = !(p.companies || []).length;
    var period = [(p.years || []).join("、"), (p.quarters || []).length ? "第 " + p.quarters.join("、") + " 季度" : ""].filter(Boolean).join(" / ");
    return h("dl", { class: "plan" },
      h("dt", null, "认出的公司"), h("dd", null, none ? h("span", { class: "muted" }, "没有认出公司") : p.companies.join("、")),
      h("dt", null, "财年 / 季度"), h("dd", null, period || "题目没有指明"),
      h("dt", null, "截止日期"), h("dd", null, p.as_of || "题目没有给"),
      a.scope ? [h("dt", null, "范围怎么定的"), h("dd", null, a.scope.method === "semi_auto" ? "程序按题干里的公司名限定报告范围,不用模型判断" : a.scope.method)] : null,
      h("dt", null, "允许查的报告(" + docs.length + " 份)"), h("dd", { class: "docs" }, docs.length ? docs.map(function (n) { return h("span", { class: "docchip", text: stripPdf(n) }); }) : h("span", { class: "muted" }, "无")),
      none ? [h("dt", null, "结果"), h("dd", null, "题干里没有写公司。程序没有检索,也没有调用模型,直接回答“" + (a.answer || "") + "”。题库里这道题问的是哪家公司,只有评分时才看得到。")] : null);
  }
  function scopeBody(a0, a1) {
    if (JSON.stringify(a0.plan) === JSON.stringify(a1.plan)) return h("div", null, planCard(a0), h("p", { class: "note" }, "两个版本在这一步完全相同。"));
    return armsBox(planCard(a0), planCard(a1));
  }

  /* ------------------------------------------------------------ 2 检索与研究 */
  function callLabels(arm, key) {
    var tl = arm.timeline, enh = arm.enh || {}, labels = {}, fin = [];
    tl.forEach(function (e, i) { if (e.t === "call" && e.node === "formalize_answer") fin.push(i); });
    var nDecide = (enh.stages || []).filter(function (s) { return DECIDE.indexOf(s) >= 0; }).length;
    if (key === A1 && enh.path && fin.length === nDecide + 2) {
      labels[fin[0]] = { title: "TimeGuard 新增 · 缺项检查", note: "把原生选出的证据和问题交给模型,问还缺哪些事实", enh: true };
      var k = 1;
      (enh.stages || []).forEach(function (s) {
        if (DECIDE.indexOf(s) < 0) return;
        labels[fin[k++]] = { title: "TimeGuard 新增 · " + (s === DECIDE[0] ? "按事实筛选" : "补搜后按事实筛选"), note: "让模型从候选原文里挑出能补齐缺失事实的块", enh: true };
      });
      labels[fin[fin.length - 1]] = { title: "写最终回答" };
    } else {
      fin.forEach(function (i, n) {
        labels[i] = key === A0 ? { title: "写最终回答" } : { title: "写回答阶段的模型调用(" + (n + 1) + "/" + fin.length + ")", note: "记录没有标明这次调用的具体用途" };
      });
    }
    return labels;
  }
  function docChips(byDoc) {
    return h("div", { class: "bydoc" }, (byDoc || []).map(function (d) {
      var name = stripPdf(d.name);
      return h("span", { class: "docchip", title: name + ":" + d.n + " 块" + (d.pages && d.pages.length ? ",第 " + d.pages.join("、") + " 页" : "") }, name, h("b", null, d.n), "块", d.pages && d.pages.length ? "第 " + d.pages.slice(0, 4).join("、") + (d.pages.length > 4 ? " …" : "") + " 页" : "");
    }));
  }
  function searchCard(e, no, finalIds) {
    var sup = e.node === "formalize_answer", text = e.q || "", half = text.length / 2;
    var doubled = half >= 8 && text.slice(0, half) === text.slice(half);
    var hits = (e.ids || []).filter(function (id) { return finalIds[id]; }).length;
    return h("div", { class: "ev ev-search" + (sup ? " enh" : "") },
      h("span", { class: "tag" }, "检索"),
      h("div", { class: "ev-b" },
        h("div", { class: "ev-t" }, sup ? "TimeGuard 新增 · 补搜(第 " + no + " 次检索)" : "RAGFlow 原有 · 第 " + no + " 次检索",
          e.node === "query_rewrite" ? h("span", { class: "hint" }, "改写检索词之后") : null),
        h("div", { class: "query", title: "模型写的查询词,后面可能追加了系统用来加强关键词匹配的强调词;这里照记录原样显示" },
          "“" + (doubled ? text.slice(0, half) : text) + "”",
          doubled ? h("span", { class: "muted" }, "(系统在后面把关键词原样又追加了一遍,用来加强关键词匹配)") : null),
        h("div", { class: "ev-m" }, ["查了 " + e.docs + " 份报告", "返回 " + e.n + " 块", sec(e.s) + " 秒", hits ? "其中 " + hits + " 块进入最终证据" : null].filter(Boolean).join(" · ")),
        docChips(e.by_doc)));
  }
  function timelineBox(arm, key) {
    if (!arm.timeline.length) return h("p", { class: "empty" }, "这一版没有检索,也没有调用模型。");
    var labels = callLabels(arm, key), finalIds = {}, no = 0;
    arm.blocks.forEach(function (b) { finalIds[b.id] = true; });
    var box = h("div", null,
      h("p", { class: "blk-sum muted" }, "模型调用 " + arm.calls + " 次 · 检索 " + arm.searches + " 次 · 输入 " + num(arm.tok_in) + " / 输出 " + num(arm.tok_out) + " token"),
      h("div", { class: "tl" }));
    var tl = box.lastChild;
    arm.timeline.forEach(function (e, i) {
      if (e.t === "tool") { tl.append(h("div", { class: "ev ev-tool" }, h("span", { class: "tag" }, "工具"), h("span", null, "调用工具:" + (TOOL[e.name] || e.name) + "(" + e.name + ")"))); return; }
      if (e.t === "search") { tl.append(searchCard(e, ++no, finalIds)); return; }
      if (e.t !== "call") return;
      var L = labels[i] || CALL[e.node] || { title: "决定先调用哪个工具", note: "主控模型读完问题,决定先用检索研究工具" };
      var title = e.node === "rag_agent" ? "RAGFlow 原有 · 研究思考 · 第 " + ((e.round || 0) + 1) + " 轮" : L.title;
      var hint = e.ok ? (e.tools === 1 ? "→ 决定发起一次检索" : null) : null;
      tl.append(h("div", { class: "ev ev-call" + (e.ok ? "" : " fail") + (L.enh ? " enh" : "") },
        h("span", { class: "tag" }, "模型"),
        h("div", { class: "ev-b" },
          h("div", { class: "ev-t" }, title, hint ? h("span", { class: "hint" }, hint) : null),
          L.note ? h("div", { class: "ev-note" }, L.note) : null,
          h("div", { class: "ev-m" }, (e.ok ? "" : "这次调用失败" + (e.err ? "(" + e.err + ")" : "") + ",不计入用量 · ") + "输入 " + num(e.in) + " · 输出 " + num(e.out) + " token · " + sec(e.s) + " 秒"))));
    });
    return box;
  }

  /* ------------------------------------------------------------ 3 增强步骤 */
  function stageCard(on, n, title, why, body) {
    return h("div", { class: "stage" + (on ? "" : " off") }, h("span", { class: "dot", text: n }), h("div", null, h("b", null, title), h("div", { class: "why" }, why), body));
  }
  function factCards(facts) {
    var NAMES = [["metric", "指标"], ["period", "期间"], ["scope", "范围"], ["unit", "单位"]];
    return h("div", { class: "factlist" }, facts.map(function (f, i) {
      var dims = f.dimensions || {};
      return h("div", { class: "fact" },
        h("div", { class: "fact-h" }, "事实 " + (i + 1) + " · ",
          f.n ? "来自最终证据第 " + f.n + " 块" : "对应候选第 " + (f.block == null ? "?" : f.block) + " 块(没有进入最终证据)",
          (f.lines || []).length ? " · 原文第 " + f.lines.join("、") + " 行" : "",
          f.missing_index != null ? " · 用来补第 " + f.missing_index + " 条缺项" : " · 原有证据已经覆盖的内容"),
        h("div", { class: "fact-t", text: f.fact || "" }),
        h("div", { class: "fact-dim" }, NAMES.filter(function (n) { return dims[n[0]] && dims[n[0]].text; }).map(function (n) {
          return h("span", null, h("b", null, n[1]), dims[n[0]].text);
        })),
        f.reason ? h("div", { class: "fact-why" }, "模型的理由:" + f.reason) : null);
    }));
  }
  function calcCard(c) {
    if (!c) return h("div", { class: "enh-calc" }, h("b", null, "计算核验:"), "这一题没有进入计算核验。");
    if (!c.active) return h("div", { class: "enh-calc" }, h("b", null, "计算核验:"), "程序没有把这题识别为需要验算的计算题。");
    return h("div", { class: "enh-calc" }, h("b", null, "计算核验:"),
      "程序把这题识别为计算题(指标:" + c.metric + ")。需要核验 " + c.required + " 个操作数,核验通过 " + c.verified + " 个。",
      c.missing && c.missing.length ? h("div", { class: "muted" }, "没核验通过的:" + c.missing.join(";")) : null);
  }
  function enhanceBox(arm, key) {
    if (key === A0) {
      return h("div", { class: "enh enh-none" },
        h("p", null, "原版没有这一步:检索结束后直接按 RAGFlow 的原生规则选证据。"),
        h("p", { class: "muted" }, "原生规则:最多 12 块 —— 先放研究阶段已经用过的片段,再放相似度最高的 4 块表格,其余按相似度补齐;然后按 " + num(META.budget) + " token 的篇幅预算裁剪,装不下的块会被截掉。"));
    }
    var e = arm.enh || {};
    if (!e.path) return h("p", { class: "empty" }, "这一题没有进入增强步骤。");
    var stages = e.stages || [], has = function (s) { return stages.indexOf(s) >= 0; }, missing = e.missing || [], fell = !!e.fallback;
    var selected = e.path === "事实筛选";
    var box = h("div", { class: "enh" },
      h("div", { class: "enh-result " + (fell ? "warn" : "ok") },
        h("b", null, fell ? "结果:回退到原生选块" : selected ? "结果:采用了按事实筛出的证据" : "结果：沿用原生选块"),
        fell ? "原因:" + (e.reason || "(记录里没有写)") + "。最终交给模型的还是原生规则选出的证据。"
             : selected ? "最终选中 " + (e.selected_count == null ? "—" : e.selected_count) + " 块证据。" : "本题最终使用 RAGFlow 原先选出的材料。"));
    box.append(stageCard(true, 1, "缺项检查",
      "原生规则选出了 " + (e.original_count == null ? "—" : e.original_count) + " 块证据。把它们和问题一起交给模型,问:要答这道题还缺哪些事实。",
      missing.length ? h("ol", { class: "facts" }, missing.map(function (m) { return h("li", null, m); })) : h("p", { class: "muted" }, "模型认为没有缺项,不需要再找材料。")));
    if (missing.length) box.append(stageCard(has(DECIDE[0]), 2, "回到已搜过的原文里按事实筛选",
      "不重新搜索。先把已经检索过的候选原文重新读一遍(" + (e.rehydrated_candidates == null ? "—" : e.rehydrated_candidates) + " 块通过核验),让模型挑出能补齐缺失事实的块,每块要说出对应哪条事实、在原文第几行。"));
    var sup = null, no = 0;
    arm.timeline.forEach(function (x) { if (x.t === "search") { no++; if (x.node === "formalize_answer") sup = searchCard(x, no, {}); } });
    box.append(stageCard(has("同范围补搜"), missing.length ? 3 : 2, "在同一范围补搜一次",
      has("同范围补搜") ? "还有事实没找齐。把它们拼成一条检索词,在允许范围的报告里再搜一次(最多补搜这一次)。"
                        : "本次记录中没有执行补搜。",
      sup));
    if (has(DECIDE[1])) box.append(stageCard(true, 4, "补搜之后再次按事实筛选", "把补搜得到的新原文和先前的候选合在一起,再筛一次。"));
    if ((e.facts || []).length) box.append(h("div", null, h("div", { class: "muted" }, "模型挑出的事实(共 " + e.facts.length + " 条,每条都要能在原文里定位):"), factCards(e.facts)));
    if ((e.unresolved || []).length) box.append(h("div", null, h("div", { class: "muted" }, "到最后仍没找到的事实:"), h("ol", { class: "facts" }, e.unresolved.map(function (m) { return h("li", null, m); }))));
    box.append(calcCard(e.calculation));
    return box;
  }

  /* ------------------------------------------------------------ 4 证据 */
  function textBlock(text, limit) {
    if (!text) return h("div", { class: "blk-text muted" }, "记录里没有这一块的正文。");
    if (text.length <= limit) return h("div", { class: "blk-text", text: text });
    var box = h("div", { class: "blk-text", text: text.slice(0, limit) }), open = false;
    var more = h("button", { class: "more", type: "button" });
    function paint() { box.textContent = open ? text : text.slice(0, limit); more.textContent = open ? "收起" : "展开全文(共 " + num(text.length) + " 字)"; box.scrollTop = 0; }
    more.addEventListener("click", function () { open = !open; paint(); });
    paint();
    return h("div", null, box, more);
  }
  function blocksBox(arm, key) {
    var bl = arm.blocks, cited = {}, from = {};
    (((arm.judge || {}).mapping || {}).marker_indexes || []).forEach(function (n) { cited[n] = true; });
    var no = 0;
    arm.timeline.forEach(function (e) { if (e.t === "search") { no++; (e.ids || []).forEach(function (id) { (from[id] = from[id] || []).push(no); }); } });
    var box = h("div", { class: "blocks" },
      h("p", { class: "blk-sum" }, h("b", null, "候选池 " + arm.pool.length + " 块"), " → ", h("b", null, "选中 " + bl.length + " 块"), " → ",
        h("b", null, "完整送达 " + delivered(arm) + " 块"), h("span", { class: "muted" }, "(篇幅上限 " + num(META.budget) + " token)")));
    var chosen = {};
    bl.forEach(function (b) {
      chosen[b.id] = true;
      var cls = b.delivered ? (b.check === "edited" ? "edited" : "ok") : "missing";
      var st = b.delivered ? h("span", { class: "badge " + (cls === "ok" ? "ok" : "unk"), title: cls === "ok" ? "裁判认可这一块的文字能对回索引原文" : "送到了模型手里,但文字和索引原文对不上,裁判不把它算作可核验的引用" },
          b.dup ? "内容与另一块相同" : CHECK[b.check] || b.check)
        : h("span", { class: "badge unk", title: "被篇幅预算截掉或只送了一部分,回答如果引用它,裁判无法核验" }, "未完整送达");
      box.append(h("article", { class: "blk blk-" + cls + (cited[b.n] ? " cited" : ""), id: "blk-" + key + "-" + b.n },
        h("header", { class: "blk-h" }, h("span", { class: "blk-n" }, "#" + b.n), h("span", { class: "blk-doc", text: stripPdf(b.doc) || "(未知报告)" }),
          h("span", { class: "blk-pages", text: pagesText(b.pages) }), st, cited[b.n] ? h("span", { class: "badge cite-badge" }, "回答引用了它") : null),
        b.delivered ? null : h("div", { class: "blk-f" }, "这一块没有完整送到模型手里。下面是它在索引里的原文,供对照。"),
        textBlock(CH[b.id] && CH[b.id].text, 520),
        from[b.id] ? h("footer", { class: "blk-f" }, "来自第 " + from[b.id].join("、") + " 次检索的结果") : null));
    });
    var rest = arm.pool.filter(function (c) { return !chosen[c.id]; });
    if (rest.length) box.append(h("details", { class: "rest" }, h("summary", null, "候选池里没被选中的 " + rest.length + " 块"),
      h("ul", null, rest.map(function (c) {
        return h("li", null, h("span", { class: "blk-doc", text: stripPdf(c.doc) }), " ", h("span", { class: "muted", text: pagesText(c.pages) }),
          CH[c.id] && CH[c.id].text ? h("div", { class: "prev", text: CH[c.id].text.slice(0, 160) + (CH[c.id].text.length > 160 ? "…" : "") }) : null);
      }))));
    return box;
  }

  /* ------------------------------------------------------------ 5 回答 */
  function isUnmapped(m, n) { return (m.unmapped_indexes || []).indexOf(n) >= 0 || (m.missing_text_indexes || []).indexOf(n) >= 0; }
  function citeReason(arm, n) {
    var b = arm.blocks[n - 1];
    if (!b) return "证据一共只有 " + arm.blocks.length + " 块,没有第 " + n + " 块";
    if (b.dup) return "有两块内容完全相同,分不清指的是哪一块";
    if (!b.delivered) return "这一块没有完整送到模型手里";
    return "这一块送到了,但文字与索引原文对不上";
  }
  function answerBox(arm, key) {
    if (!arm.answer) return h("p", { class: "empty" }, "没有回答。");
    var m = (arm.judge || {}).mapping || {}, body = h("div", { class: "ans-md", html: md(arm.answer) }), bad = 0;
    body.querySelectorAll(".cite[data-n]").forEach(function (el) {
      var n = +el.getAttribute("data-n"), b = arm.blocks[n - 1];
      if (m.mapping_status && isUnmapped(m, n)) { el.classList.add("bad"); el.title = "裁判没能把它对回原文:" + citeReason(arm, n); bad++; }
      else el.title = "第 " + n + " 块:" + (b ? stripPdf(b.doc) + " " + pagesText(b.pages) : "不存在");
    });
    return h("div", null, body,
      h("p", { class: "cite-legend" }, (m.marker_indexes || []).length ? "回答引用了 " + m.marker_indexes.length + " 个编号" + (bad ? "," + bad + " 个对不回原文(黄色)" : ",全部对回了原文") + "。点编号跳到那一块。" : "回答里没有标引用编号。"));
  }

  /* ------------------------------------------------------------ 6 评分 */
  function judgeName() {
    var ms = META.judge_models || [];
    return ms.length ? ms.map(function (x) { return x.model + "(思考强度 " + x.effort + ")"; }).join("、") : (META.judge ? META.judge.model + "(思考强度 " + META.judge.effort + ")" : "—");
  }
  function judgeBox(arm) {
    var j = arm.judge || {}, m = j.mapping || {}, bl = arm.blocks;
    var box = h("div", { class: "judge" });
    if (j.judge_type === "rule" || j.judge_status === "clarification") box.append(h("p", { class: "warn-text" }, "这题没有调用裁判,按规则记录(" + (j.judge_status || "") + ")。它照样计入分母。"));
    else if (j.judge_status !== "ok") box.append(h("p", { class: "warn-text" }, "评分没有完成(" + j.judge_status + "):" + (j.error || "") + " 这道题按“无法判断”计入分母。"));
    box.append(h("div", { class: "verdicts" }, badge(j.answer_correct, "答案正确", "答案错误"),
      j.numeric_and_scope_correct == null ? null : badge(j.numeric_and_scope_correct, "数值与口径正确", "数值或口径有误"),
      badge(j.citation_support, "引用支持结论", "引用不支持")));
    box.append(h("p", { class: "score-readout" }, "本题怎样计分？答案正确数 " + (j.answer_correct === true ? "+1" : "+0") + "，答对且引用支持数 " + (j.answer_correct === true && j.citation_support === true ? "+1" : "+0") + "。本题在两项指标的分母中各占一题。"));
    var dl = h("dl", { class: "jrow" });
    dl.append(h("dt", null, "引用怎么判"), h("dd", null, j.citation_support_rule || "—"));
    if (j.citation_support_model != null && j.citation_support_model !== j.citation_support) {
      dl.append(h("dt", null, "按规则改记"), h("dd", null, "裁判模型原本判引用“" + (j.citation_support_model === true ? "支持" : "不支持") + "”,按上面的规则改记为“无法判断”。"));
    }
    var marks = m.marker_indexes || [];
    if (m.mapping_status) {
      var line = m.mapping_status === "mapped" ? "回答里的引用编号全部对回了原文。"
        : m.mapping_status === "incomplete" ? "有引用编号对不回原文,所以引用支持只能记为“无法判断”。" : "回答里没有标任何引用编号。";
      dl.append(h("dt", null, "引用映射"), h("dd", null, line,
        marks.length ? h("ul", { class: "facts" }, marks.map(function (n) {
          var bad = isUnmapped(m, n), b = bl[n - 1];
          return h("li", null, h("span", { class: "cite static" + (bad ? " bad" : ""), text: "[" + n + "]" }), " ",
            bad ? "无法核验:" + citeReason(arm, n) : "对回第 " + n + " 块:" + stripPdf(b.doc) + " " + pagesText(b.pages) + "(" + (CHECK[b.check] || "") + ")");
        })) : null));
    }
    if ((m.malformed_markers || []).length) dl.append(h("dt", null, "格式异常的引用"), h("dd", null, m.malformed_markers.join("   ")));
    dl.append(h("dt", null, "裁判的理由"), h("dd", { class: "jreason", text: j.judgement_reason || "—" }));
    box.append(dl);
    if (j.judge_status === "ok") box.append(h("p", { class: "jmeta" }, "裁判 " + judgeName() + " · 用时 " + sec(j.elapsed) + " 秒 · 输入 " + num(j.tok_in) + " / 输出 " + num(j.tok_out) + " token · " + (j.human_spot_check ? "有人工复核记录" : "模型判断，尚无人工复核")));
    return box;
  }

  /* ------------------------------------------------------------ 概况、页脚 */
  function conditions() {
    var am = (META.answer_models || []).map(function (x) { return x.model + "(思考强度 " + x.effort + ")"; }).join("、");
    return "作答与研究:" + (am || "—") + " · 裁判:" + judgeName() + " · 证据篇幅上限 " + num(META.budget) + " token · "
      + (META.transport === "openai-responses-v1" ? "模型入口为官方 OpenAI API" : "模型入口:" + META.transport) + "。";
  }
  function overview() {
    var s = META.summary, tb = h("tbody");
    [["all", "全部"], ["tune", "调优组"], ["dev", "开发组"], ["test", "保留组"]].forEach(function (o) {
      var g = s[o[0]], a = g.arms[A0], b = g.arms[A1];
      tb.append(h("tr", null, h("td", null, o[1]), h("td", null, a.denominator), h("td", null, a.joint_pass + " → " + b.joint_pass),
        h("td", null, a.joint_unknown + " / " + b.joint_unknown), h("td", null, a.answer_pass + " → " + b.answer_pass), h("td", null, a.citation_pass + " → " + b.citation_pass),
        h("td", null, g.comparable_pairs + "(" + g.gain + " / " + g.loss + ")")));
    });
    var a = s.all.arms[A0], b = s.all.arms[A1];
    var unsupported = IDS.filter(function (id) {
      var j = QS[id].arms[A1].judge || {};
      return j.answer_correct === true && j.citation_support === false;
    }).length;
    return h("details", { class: "overview" },
      h("summary", null, "实验概况:答对且引用支持 " + a.joint_pass + "/150 → " + b.joint_pass + "/150;答案正确 " + a.answer_pass + "/150 → " + b.answer_pass + "/150。点开看分组和条件"),
      h("div", { class: "ov-body" },
        h("p", null, "增强版答对 " + b.answer_pass + " 题（" + (100*b.answer_pass/b.denominator).toFixed(1) + "%），其中实际引用也支持答案的有 " + b.joint_pass + " 题（占全部题目的 " + (100*b.joint_pass/b.denominator).toFixed(1) + "%）。两次都用全部 " + b.denominator + " 题作分母。"),
        h("p", null, "增强版答对的题里，还有 " + unsupported + " 题引用不足，" + (b.answer_pass - b.joint_pass - unsupported) + " 题的引用无法判断。"),
        h("table", { class: "ov" }, h("thead", null, h("tr", null, ["分组", "题数", "答对且引用支持(原版 → 增强版)", "其中无法判断(原版 / 增强版)", "答案正确(原版 → 增强版)", "引用支持(原版 → 增强版)", "两组都有明确判定的题对(升 / 降)"].map(function (t) { return h("th", null, t); }))), tb),
        h("p", { class: "muted" }, conditions() + "无法判断的题按未达标计入分母。调优、开发、保留三组按公司划分,彼此没有相同的公司。")));
  }
  function footNote() {
    return h("p", { class: "foot" }, "数据来源:服务器目录 TimeGuard-RAG-FinanceBench-API/runs/" + META.run + "(代码提交 " + (META.commit || "—") + "),导出于 " + META.exported_at + "。"
      + "页面离线可用,重新导出和构建的方法见同目录 trace/README.md。");
  }

  function readingGuide(a0, a1) {
    function link(label, target) {
      return h("button", { type: "button", class: "btn", onclick: function () { jump(target); } }, label);
    }
    var e = a1.enh || {};
    function score(arm, name) {
      var j = arm.judge || {};
      return h("div", { class: "guide-score" }, h("b", null, name),
        h("span", { class: "badge " + vc(j.answer_correct) }, "答案" + vt(j.answer_correct, "正确", "错误")),
        h("span", { class: "badge " + vc(j.citation_support) }, "引用" + vt(j.citation_support, "支持", "不足")),
        h("span", null, "答案分子 +" + Number(j.answer_correct === true) + "，联合分子 +" + Number(joint(arm) === true)));
    }
    return h("section", { class: "card guide" }, h("h2", null, "跟着这道题看懂评测"),
      h("div", { class: "guide-path" },
        h("div", null, h("strong", null, "RAGFlow 原有 · 检索研究"),
          h("p", null, "原版检索 " + a0.searches + " 次，增强版记录 " + a1.searches + " 次检索（含补搜）。每次查询的词和返回页码都可以展开查看。"), link("看查询词和结果", "search")),
        h("div", null, h("strong", null, "TimeGuard 新增 · 作答前核对材料"),
          h("p", null, e.path ? "本题记录 " + (e.missing || []).length + " 项缺失事实，限域补搜 " + (e.supplement_calls || 0) + " 次。" +
            (e.fallback ? "最终回退到原生选块。" : "最终采用" + e.path + "。") : "本题没有进入增强步骤。"), link("看本题的处理记录", "enh"))),
      h("h3", null, "这道题如何计入两项指标"),
      score(a0, "原版"), score(a1, "增强版"),
      h("p", { class: "muted" }, "每版各有 150 题。答对计入答案正确数；答对且实际引用能支持结论，才同时计入联合达标数。无法判断的题也占分母一题。"),
      link("看回答与引用编号", "answer"), " ", link("看评分依据", "judge"));
  }

  function renderDetail() {
    var q = QS[state.qid], d = DS[state.qid] || {}, a0 = q.arms[A0], a1 = q.arms[A1];
    var pos = state.visible.indexOf(state.qid), root = document.getElementById("detail-body");
    document.getElementById("pos").textContent = (pos >= 0 ? "当前筛选第 " + (pos + 1) + " / " + state.visible.length + " 题" : "这道题不在当前筛选结果里") + " · " + shortId(state.qid);
    document.getElementById("prev").disabled = pos <= 0;
    document.getElementById("next").disabled = pos < 0 || pos >= state.visible.length - 1;
    root.replaceChildren(
      overview(),
      questionCard(q, d),
      readingGuide(a0, a1),
      compareTable(a0, a1),
      step(1, "scope", "确定范围", "程序先从题干里认出公司、财年或季度、截止日期,只允许在这个范围的财报里找。", scopeBody(a0, a1)),
      step(2, "search", "检索与研究", "模型自己写检索词、看返回的片段、决定下一步搜什么。材料不够时,RAGFlow 原生流程会把缺口改写成新检索词,再搜一轮。", armsBox(timelineBox(a0, A0), timelineBox(a1, A1))),
      step(3, "enh", "增强步骤", "你加的部分,只有增强版有。原生选完证据后先检查还缺什么事实;缺了就回到已搜过的原文里按事实重新筛,还不够再在同一范围补搜一次;核验不通过就回退到原生选块。", armsBox(enhanceBox(a0, A0), enhanceBox(a1, A1))),
      step(4, "blocks", "送给模型的证据", "最终交给作答模型的材料。回答里的 [ID:n] 指的就是第 n 块。", armsBox(blocksBox(a0, A0), blocksBox(a1, A1))),
      step(5, "answer", "回答", "模型的原样输出。点编号可以跳到它引用的那一块证据;黄色编号表示裁判没能把它对回原文。", armsBox(answerBox(a0, A0), answerBox(a1, A1))),
      step(6, "judge", "评分", "独立裁判只看这一份回答、它实际引用到的原文、参考答案和标注证据。参考答案到这一步才出现。", armsBox(judgeBox(a0), judgeBox(a1))),
      footNote());
  }

  /* ------------------------------------------------------------ 交互 */
  function select(id, moveHash) {
    if (!QS[id]) return;
    state.qid = id;
    document.querySelectorAll("#list .qrow").forEach(function (el) {
      var on = el.dataset.id === id;
      el.classList.toggle("sel", on);
      el.setAttribute("aria-pressed", on ? "true" : "false");
    });
    renderDetail();
    if (moveHash) { try { history.replaceState(null, "", "#q=" + id); } catch (e) { /* 本地文件上个别浏览器不允许改地址,不影响使用 */ } }
    var picker = document.getElementById("picker");
    picker.classList.remove("open");
    document.getElementById("picker-toggle").setAttribute("aria-expanded", "false");
    scrollToY(0);
  }
  /* 右栏在宽屏下自己滚动,窄屏下整页滚动。不依赖 CSS 的平滑滚动,自己算目标位置。 */
  function scroller() {
    var d = document.getElementById("detail");
    return (window.innerWidth >= 1180 && d.scrollHeight > d.clientHeight) ? { el: d, box: d.getBoundingClientRect().top } : { el: null, box: 0 };
  }
  function scrollToY(y) {
    var s = scroller();
    if (s.el) s.el.scrollTop = Math.max(0, y);
    else window.scrollTo(0, Math.max(0, y + (window.pageYOffset || 0)));
  }
  function offsetOf(el) {
    var s = scroller();
    return el.getBoundingClientRect().top - s.box - 52;
  }
  function jump(id) {
    var t = document.getElementById("step-" + id);
    if (t) scrollToY((scroller().el ? scroller().el.scrollTop : 0) + offsetOf(t));
  }
  function init() {
    fillFilters();
    ["f-split", "f-type", "f-company", "f-cmp"].forEach(function (id) {
      document.getElementById(id).addEventListener("change", function (e) { state.filter[id.slice(2)] = e.target.value; applyFilter(); renderList(); });
    });
    document.getElementById("f-text").addEventListener("input", function (e) { state.filter.text = e.target.value; applyFilter(); renderList(); });
    document.getElementById("picker-toggle").addEventListener("click", function () {
      var open = document.getElementById("picker").classList.toggle("open");
      this.setAttribute("aria-expanded", open ? "true" : "false");
    });
    document.getElementById("prev").addEventListener("click", function () { var i = state.visible.indexOf(state.qid); if (i > 0) select(state.visible[i - 1], true); });
    document.getElementById("next").addEventListener("click", function () { var i = state.visible.indexOf(state.qid); if (i >= 0 && i < state.visible.length - 1) select(state.visible[i + 1], true); });
    var jumps = document.getElementById("jumps");
    STEPS.forEach(function (s) { jumps.append(h("button", { class: "btn", type: "button", onclick: function () { jump(s[0]); } }, s[1])); });
    var body = document.getElementById("detail-body");
    body.addEventListener("click", function (e) {
      var el = e.target.closest ? e.target.closest(".cite[data-n]") : null;
      if (!el) return;
      var host = el.closest("[data-arm]"), key = host ? host.getAttribute("data-arm") : "";
      var target = document.getElementById("blk-" + key + "-" + el.getAttribute("data-n"));
      if (!target) return;
      var s = scroller();
      scrollToY((s.el ? s.el.scrollTop : 0) + offsetOf(target) - Math.max(0, (s.el ? s.el.clientHeight : window.innerHeight) * 0.38));
      target.classList.remove("flash");
      void target.offsetWidth;
      target.classList.add("flash");
    });
    body.addEventListener("keydown", function (e) {
      var el = e.target.closest ? e.target.closest(".cite[data-n]") : null;
      if (el && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); el.click(); }
    });
    window.addEventListener("hashchange", function () {
      var m = location.hash.match(/q=([A-Za-z0-9_]+)/);
      if (m && QS[m[1]] && m[1] !== state.qid) select(m[1], false);
    });
    applyFilter();
    renderList();
    var m = location.hash.match(/q=([A-Za-z0-9_]+)/);
    select(m && QS[m[1]] ? m[1] : (QS.financebench_id_01079 ? "financebench_id_01079" : IDS[0]), false);
  }
  init();
})();
