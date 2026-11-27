/* 逐题回放页面的结构测试。不依赖浏览器和第三方库。
   用法:node test.js   (先运行 python build.py 生成 dist/)
   做法:提供一个只实现了页面脚本所用接口的最小 DOM,把全部题目逐题渲染出来,检查每一步的内容和交互。 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

/* ------------------------------------------------------------ 最小 DOM */
class ClassList {
  constructor(set) { this.set = set; }
  add(c) { this.set.add(c); } remove(c) { this.set.delete(c); }
  contains(c) { return this.set.has(c); }
  toggle(c, on) { if (on === undefined) on = !this.set.has(c); if (on) this.set.add(c); else this.set.delete(c); return on; }
  toString() { return [...this.set].join(" "); }
}
class Node {
  constructor(tag) {
    this.nodeType = 1; this.tagName = tag.toUpperCase(); this.children = []; this.attrs = {}; this.classes = new Set(); this._text = "";
    this.listeners = {}; this.parentNode = null; this.scrollTop = 0; this.scrollHeight = 0; this.clientHeight = 0; this.offsetWidth = 0; this.disabled = false;
  }
  set className(v) { this.classes = new Set(String(v).split(/\s+/).filter(Boolean)); }
  get className() { return [...this.classes].join(" "); }
  // 不缓存:className 被整体重新赋值后,classList 必须指向新的集合
  get classList() { return new ClassList(this.classes); }
  get id() { return this.attrs.id || ""; }
  set id(v) { this.attrs.id = v; if (v) byId[v] = this; }
  get dataset() { return this._ds || (this._ds = {}); }
  set textContent(v) { this.children = []; this._text = String(v); }
  get textContent() { return this.children.length ? this.children.map(c => c.textContent).join("") : this._text; }
  set innerHTML(v) {
    // 只解析页面脚本会生成的那几种标签;文字里的实体还原成字符
    this.children = []; this._text = ""; this.html = String(v);
    const stack = [this], re = /<([a-z0-9]+)([^>]*)>|<\/([a-z0-9]+)>|([^<]+)/gi;
    let m;
    const unescape = s => s.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&amp;/g, "&");
    while ((m = re.exec(v))) {
      if (m[4]) { stack[stack.length - 1].appendChild(textNode(unescape(m[4]))); continue; }
      if (m[3]) { if (stack.length > 1) stack.pop(); continue; }
      const n = new Node(m[1]);
      for (const a of m[2].matchAll(/([\w-]+)(?:="([^"]*)")?/g)) n.setAttribute(a[1], a[2] === undefined ? true : unescape(a[2]));
      stack[stack.length - 1].appendChild(n);
      if (!/^(br|img|input)$/i.test(m[1])) stack.push(n);
    }
  }
  get innerHTML() { return this.html || ""; }
  append(...items) { for (const it of items) this.appendChild(coerce(it)); }
  appendChild(n) { this.children.push(n); n.parentNode = this; this._text = ""; return n; }
  prepend(n) { this.children.unshift(coerce(n)); }
  replaceChildren(...items) { this.children = []; this._text = ""; this.append(...items); }
  setAttribute(k, v) { this.attrs[k] = v === true ? "" : String(v); if (k === "id" && v) byId[v] = this; if (k === "class") this.className = this.attrs[k]; }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  removeAttribute(k) { delete this.attrs[k]; }
  addEventListener(k, fn) { (this.listeners[k] = this.listeners[k] || []).push(fn); }
  click() { (this.listeners.click || []).forEach(fn => fn({ target: this, preventDefault() {} })); }
  get lastChild() { return this.children[this.children.length - 1]; }
  getBoundingClientRect() { return { top: 0, left: 0, width: 0, height: 0 }; }
  scrollIntoView() {}
  *walk() { for (const c of this.children) { if (c.nodeType === 1) { yield c; yield* c.walk(); } } }
  querySelectorAll(sel) { return [...this.walk()].filter(n => matches(n, sel)); }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  closest(sel) { for (let n = this; n; n = n.parentNode) if (n.nodeType === 1 && matches(n, sel)) return n; return null; }
}
const textNode = t => ({ nodeType: 3, textContent: String(t), children: [], parentNode: null });
function coerce(x) {
  if (x == null || x === false) return textNode("");
  if (x.nodeType) return x;
  if (Array.isArray(x)) { const holder = textNode(""); holder.splice = true; return holder; }
  return textNode(x);
}
Node.prototype.append = function (...items) {
  const flat = [];
  (function push(list) { for (const it of list) { if (it == null || it === false) continue; if (Array.isArray(it)) push(it); else flat.push(it); } })(items);
  for (const it of flat) this.appendChild(coerce(it));
};
function simple(sel) {
  // 把一个简单选择器拆成若干条件:tag、#id、.class、[attr]、[attr="值"],顺序和个数不限
  const conds = [];
  const re = /^([a-z0-9]+)|#([\w-]+)|\.([\w-]+)|\[([\w-]+)(?:="([^"]*)")?\]/gi;
  let pos = 0, m;
  while (pos < sel.length) {
    re.lastIndex = pos;
    if (!(m = re.exec(sel)) || m.index !== pos) throw new Error("测试用的选择器不支持: " + sel);
    conds.push(m);
    pos = re.lastIndex;
  }
  return n => conds.every(c => c[1] ? n.tagName === c[1].toUpperCase() : c[2] ? n.id === c[2] : c[3] ? n.classes.has(c[3])
    : (c[5] === undefined ? n.attrs[c[4]] !== undefined : n.attrs[c[4]] === c[5]));
}
function matches(n, sel) {
  if (sel.indexOf(", ") >= 0) return sel.split(", ").some(s => simple(s)(n));
  const parts = sel.split(/\s+(?![^\[]*\])/).filter(Boolean);
  let ok = simple(parts.pop())(n);
  if (!ok) return false;
  let cur = n.parentNode;
  for (const p of parts.reverse()) { while (cur && !simple(p)(cur)) cur = cur.parentNode; if (!cur) return false; cur = cur.parentNode; }
  return true;
}

/* ------------------------------------------------------------ 搭建页面 */
const appSrc = fs.readFileSync(path.join(__dirname, "dist", "app.js"), "utf8");
const dataJs = fs.readFileSync(path.join(__dirname, "dist", "data.js"), "utf8");
const byId = {};
const el = id => byId[id] || (byId[id] = Object.assign(new Node("div"), { id }));
const html = new Node("html");
["picker", "picker-toggle", "picker-count", "f-text", "f-split", "f-type", "f-company", "f-cmp", "list", "list-count", "detail", "detail-body", "pos", "jumps", "prev", "next"].forEach(id => html.appendChild(el(id)));
const traceData = new Node("script"); traceData.id = "trace-data";
const win = { innerWidth: 1600, addEventListener() {}, scrollTo() {}, pageYOffset: 0 };
const sandbox = {
  window: win, history: { replaceState() {} }, location: { hash: "" }, console, setTimeout, clearTimeout, Math, Date, JSON, Number, String, Array, Object, RegExp, Error,
  document: { createElement: t => new Node(t), createTextNode: textNode, getElementById: id => byId[id] || null,
    querySelector: sel => html.querySelector(sel), querySelectorAll: sel => html.querySelectorAll(sel), title: "", documentElement: {} },
};
traceData.textContent = dataJs.replace(/^window\.TRACE_DATA=/, "").replace(/;$/, "");
sandbox.document.getElementById("trace-data") || (sandbox.document.getElementById = id => id === "trace-data" ? traceData : (byId[id] || null));
vm.createContext(sandbox);
vm.runInContext(dataJs, sandbox);
vm.runInContext(appSrc, sandbox);
const D = win.TRACE_DATA;
const IDS = Object.keys(D.questions).sort();

/* ------------------------------------------------------------ 检查 */
const out = [];
const ok = (name, cond, detail = "") => out.push((cond ? "PASS  " : "FAIL  ") + name + (detail !== "" ? "  | " + detail : ""));
const rows = () => byId.list.querySelectorAll(".qrow");
const rowOf = id => rows().find(r => r.dataset.id === id);
const open = id => { const r = rowOf(id); if (!r) throw new Error("题库里找不到 " + id); r.click(); };
const text = n => (n ? n.textContent : "");
const stepText = (step, arm) => { const s = byId["step-" + step]; return text(arm ? s.querySelector(".arm-" + arm) : s); };
const joint = a => { const j = a.judge; return (j.answer_correct === false || j.citation_support === false) ? "fail" : (j.answer_correct === true && j.citation_support === true) ? "pass" : "unknown"; };
const setVal = (id, v, evt) => { byId[id].value = v; (byId[id].listeners[evt] || []).forEach(f => f({ target: byId[id] })); };

try {
  ok("题库列出全部 150 题", rows().length === 150, rows().length);
  ok("默认选中 01079", rowOf("financebench_id_01079").classes.has("sel"));
  ok("每道题的列表行都带两组的判定标记", rows().every(r => r.querySelectorAll(".vline").length === 2 && r.querySelectorAll(".chip").length === 4));

  // 筛选
  setVal("f-split", "dev", "change");
  ok("分组 = 开发组 → 39 题", rows().length === 39, rows().length);
  setVal("f-split", "", "change");
  const want = { a1: 0, a0: 0, both: 0, none: 0, unk: 0 };
  for (const id of IDS) { const a = joint(D.questions[id].arms.A0), b = joint(D.questions[id].arms.A1); want[a === "pass" && b === "pass" ? "both" : a === "fail" && b === "fail" ? "none" : b === "pass" ? "a1" : a === "pass" ? "a0" : "unk"]++; }
  for (const k of Object.keys(want)) { setVal("f-cmp", k, "change"); ok("两组对比 = " + k, rows().length === want[k], rows().length); }
  ok("对比筛选的各类相加等于 150", Object.values(want).reduce((s, x) => s + x, 0) === 150, JSON.stringify(want));
  setVal("f-cmp", "", "change");
  setVal("f-text", "Amcor", "input");
  ok("搜索 Amcor → 9 题", rows().length === 9, rows().length);
  setVal("f-text", "", "input");
  ok("清空筛选 → 150 题", rows().length === 150, rows().length);

  // 逐题渲染
  const STEPS = ["scope", "search", "enh", "blocks", "answer", "judge"];
  const problems = [];
  let stats = { fell: 0, sup: 0, factSelect: 0, clarify: 0, judgeFailed: 0, calcActive: 0, calcVerified: 0, unmapped: 0, edited: 0, notDelivered: 0, blocks: 0, delivered: 0, calls: [0, 0], searches: [0, 0], enhCalls: [0, 0], multiRequest: 0, retryFail: 0, tables: 0 };
  for (const id of IDS) {
    const q = D.questions[id];
    open(id);
    if (!rowOf(id).classes.has("sel")) problems.push(id.slice(-5) + ":列表行没有标成选中");
    if (!byId["detail-body"].querySelector(".qtext") || text(byId["detail-body"].querySelector(".qtext")) !== q.q) problems.push(id.slice(-5) + ":题目卡片不对");
    if (byId.pos.textContent.indexOf(id.slice(-5)) < 0) problems.push(id.slice(-5) + ":位置提示不对");
    if (byId["detail-body"].querySelectorAll(".step").length !== 6) problems.push(id.slice(-5) + ":不是六个步骤");
    for (const s of STEPS) {
      const arms = byId["step-" + s].querySelectorAll(".arm-b");
      if (s !== "scope" && arms.length !== 2) problems.push(id.slice(-5) + ":" + s + " 不是两栏");
      if (s !== "scope" && arms.some(a => !text(a).trim())) problems.push(id.slice(-5) + ":" + s + " 有空白栏");
    }
    // 题目卡片里的参考答案与标注证据
    const evd = byId["detail-body"].querySelectorAll(".evd");
    if (evd.length !== (D.dataset[id].evidence || []).length) problems.push(id.slice(-5) + ":标注证据条数不对");
    for (const [i, k] of ["A0", "A1"].entries()) {
      const a = q.arms[k], e = a.enh || {};
      stats.calls[i] += a.calls; stats.searches[i] += a.searches;
      stats.blocks += a.blocks.length; stats.delivered += a.blocks.filter(b => b.delivered).length;
      stats.notDelivered += a.blocks.filter(b => !b.delivered).length;
      stats.edited += a.blocks.filter(b => b.delivered && b.check === "edited").length;
      // 证据块:编号连续、每块都有报告名和页码
      a.blocks.forEach((b, n) => {
        if (b.n !== n + 1) problems.push(id.slice(-5) + ":" + k + " 证据块编号不连续");
        const card = byId["blk-" + k + "-" + b.n];
        if (!card) { problems.push(id.slice(-5) + ":" + k + " 缺第 " + b.n + " 块的卡片"); return; }
        if (b.doc && text(card.querySelector(".blk-doc")) !== b.doc.replace(/\.pdf$/i, "")) problems.push(id.slice(-5) + ":" + k + " 第 " + b.n + " 块报告名不对");
        if (b.delivered !== card.classes.has("blk-ok") && !(b.delivered && card.classes.has("blk-edited"))) problems.push(id.slice(-5) + ":" + k + " 第 " + b.n + " 块送达状态显示不对");
      });
      const boxes = byId["step-blocks"].querySelectorAll(".arm-" + k + " .blk");
      if (boxes.length !== a.blocks.length) problems.push(id.slice(-5) + ":" + k + " 证据卡片数不对");
      // 回答里的引用编号:数量与数据一致
      const wantCites = (a.answer.match(/\[(?:ID:)?[0-9]+\]/g) || []).length;
      const gotCites = byId["step-answer"].querySelectorAll(".arm-" + k + " .cite").length;
      if (gotCites !== wantCites) problems.push(id.slice(-5) + ":" + k + " 引用编号个数 " + gotCites + " ≠ 回答里的 " + wantCites);
      // 时间线:模型调用和检索条数
      const calls = byId["step-search"].querySelectorAll(".arm-" + k + " .ev-call").length;
      const searches = byId["step-search"].querySelectorAll(".arm-" + k + " .ev-search").length;
      if (calls !== a.calls || searches !== a.searches) problems.push(id.slice(-5) + ":" + k + " 时间线调用/检索条数不对");
      // 评分栏:判定标记与数据一致
      const badges = byId["step-judge"].querySelectorAll(".arm-" + k + " .badge");
      if (badges.length < 2) problems.push(id.slice(-5) + ":" + k + " 评分栏判定标记不全");
      const j = a.judge;
      const vtext = text(byId["step-judge"].querySelector(".arm-" + k));
      if (j.judgement_reason && vtext.indexOf(j.judgement_reason.trim().slice(0, 40)) < 0) problems.push(id.slice(-5) + ":" + k + " 裁判理由没有显示");
      if (j.judge_status === "failed" && !/评分没有完成/.test(vtext)) problems.push(id.slice(-5) + ":" + k + " 评分失败没有提示");
      if (j.judge_status === "clarification" && !/没有调用裁判/.test(vtext)) problems.push(id.slice(-5) + ":" + k + " 澄清题没有提示");
      if (j.judge_status === "failed") stats.judgeFailed++;
      // 澄清题两组各有一条记录,按题目数只记一次
      if (j.judge_status === "clarification" && k === "A1") stats.clarify++;
      // 裁判记录的 marker_indexes 是去重后的编号;回答里同一个编号可能出现多次,每一处都应标出,所以按出现次数算期望值
      const m = j.mapping || {};
      const badSet = new Set([...(m.unmapped_indexes || []), ...(m.missing_text_indexes || [])]);
      const badWant = (a.answer.match(/\[(?:ID:)?[0-9]+\]/g) || []).map(s => +s.match(/[0-9]+/)[0]).filter(n => badSet.has(n));
      const badGot = byId["step-answer"].querySelectorAll(".arm-" + k + " .cite.bad").length;
      if (badGot !== badWant.length) problems.push(id.slice(-5) + ":" + k + " 标黄的引用编号数不对(期望 " + badWant.length + ",实际 " + badGot + ")");
      stats.unmapped += badWant.length;
      if (m.mapping_status === "mapped" && !/全部对回了原文/.test(vtext)) problems.push(id.slice(-5) + ":" + k + " 映射完整的说明缺失");
      if (m.mapping_status === "incomplete" && !/对不回原文/.test(vtext)) problems.push(id.slice(-5) + ":" + k + " 映射不完整的说明缺失");
    }
    // 窄屏下页面不该被撑宽:所有表格都必须在自己的可横向滚动的容器里(回答里的 .tw、对比表的 .cmp-wrap、概况表的 .ov-body)
    const scroll = n => { for (let p = n.parentNode; p && p.nodeType === 1; p = p.parentNode) if (/\b(tw|cmp-wrap|ov-body)\b/.test(p.className)) return true; return false; };
    for (const t of byId["detail-body"].querySelectorAll("table")) {
      stats.tables++;
      if (!scroll(t)) problems.push(id.slice(-5) + ":有表格没放进可滚动的容器(" + t.className + ")");
    }
    // 增强步骤
    const e = q.arms.A1.enh || {}, t = text(byId["step-enh"].querySelector(".arm-A1"));
    if (e.fallback) { stats.fell++; if (!/回退到原生选块/.test(t) || t.indexOf(e.reason.slice(0, 12)) < 0) problems.push(id.slice(-5) + ":回退原因没有显示"); }
    else if (e.path === "事实筛选") { stats.factSelect++; if (!/采用了按事实筛出的证据/.test(t)) problems.push(id.slice(-5) + ":采用事实筛选的说明缺失"); }
    else if (e.path && !/没有缺项|不需要/.test(t)) problems.push(id.slice(-5) + ":走原生选块的题,增强说明缺失");
    if (e.path === "原生选块" && !e.fallback && !t.includes("沿用原生选块")) problems.push(id + "：原生选块的归属有误");
    const guides = byId["detail-body"].querySelectorAll(".guide-score");
    for (const [i, k] of ["A0", "A1"].entries()) {
      const j = q.arms[k].judge;
      const expected = "答案分子 +" + Number(j.answer_correct === true) + "，联合分子 +" + Number(j.answer_correct === true && j.citation_support === true);
      if (!guides[i] || !text(guides[i]).includes(expected)) problems.push(id + "：" + k + " 计分解释不符");
    }
    if ((e.supplement_calls || 0) > 0) { stats.sup++; if (!/TimeGuard 新增 · 补搜/.test(text(byId["step-search"].querySelector(".arm-A1")))) problems.push(id.slice(-5) + ":补搜没有在时间线里标出"); }
    if ((e.facts || []).length !== byId["step-enh"].querySelectorAll(".arm-A1 .fact").length) problems.push(id.slice(-5) + ":事实卡片数不对");
    if (e.calculation && e.calculation.active) { stats.calcActive++; stats.calcVerified += e.calculation.verified; if (t.indexOf("核验通过 " + e.calculation.verified + " 个") < 0) problems.push(id.slice(-5) + ":计算核验结果显示不对"); }
    if (q.arms.A0.status === "clarification" && !/请明确要查询的公司/.test(stepText("scope"))) problems.push(id.slice(-5) + ":澄清题的范围说明缺失");
  }
  ok("150 题逐题渲染:没有结构性问题", problems.length === 0, problems.slice(0, 8).join("; ") + (problems.length > 8 ? " …共 " + problems.length + " 处" : ""));

  // 交互
  open(IDS[0]);
  byId.next.click();
  ok("“下一题”切到第二题", rowOf(IDS[1]).classes.has("sel"));
  byId.prev.click();
  ok("“上一题”回到第一题", rowOf(IDS[0]).classes.has("sel"));
  ok("跳转按钮共 6 个", byId.jumps.children.length === 6);
  // 滚动位置和几何布局在这个替身里量不了,“跳转后滚动”“点编号后滚动”两项以浏览器里的实测为准。
  // 这里只确认点引用编号能找到对应的证据块并给它加上高亮标记。
  // 页面把点击事件委托给 #detail-body 处理,所以从那里触发,和浏览器里冒泡上来的点击是同一段处理逻辑。
  open("financebench_id_01079");
  const cite = byId["step-answer"].querySelector(".arm-A1 .cite");
  const n = cite.attrs["data-n"];
  const onClick = byId["detail-body"].listeners.click[0];
  const host = byId["step-answer"].querySelector(".arm-A1");
  onClick({ target: { closest: sel => sel === ".cite[data-n]" ? cite : "[data-arm]" ? host : null } });
  const blk = byId["blk-A1-" + n];
  ok("点引用编号 [" + n + "] → 对应的证据块被高亮", !!(blk && blk.classes.has("flash")));

  ok("统计:增强回退 " + stats.fell + " 题、补搜 " + stats.sup + " 题、采用事实筛选 " + stats.factSelect + " 题、澄清 " + stats.clarify + " 题、评分失败 " + stats.judgeFailed + " 题", stats.fell === 15 && stats.sup === 34 && stats.factSelect === 45 && stats.clarify === 3 && stats.judgeFailed === 10);
  ok("统计:证据块共 " + stats.blocks + " 块,完整送达 " + stats.delivered + ",未完整送达 " + stats.notDelivered + ",送达但与原文对不上 " + stats.edited, stats.blocks === stats.delivered + stats.notDelivered);
  ok("统计:计算核验启用 " + stats.calcActive + " 题,操作数核验通过 " + stats.calcVerified + " 个", stats.calcActive === 12 && stats.calcVerified === 0);
  console.log("INFO  回答里对不回原文的引用标记共 " + stats.unmapped + " 处(同一个编号出现几次算几处)");
  console.log("INFO  逐题打开时页面里的表格共 " + stats.tables + " 个,全部在可横向滚动的容器里");
} catch (e) {
  out.push("ERROR  " + (e.stack || e.message).toString().split("\n").slice(0, 3).join(" | "));
}
console.log(out.join("\n"));
console.log(out.some(l => l.startsWith("FAIL") || l.startsWith("ERROR")) ? "RESULT: 有未通过项" : "RESULT: 全部通过");
if (out.some(l => l.startsWith("FAIL") || l.startsWith("ERROR"))) process.exitCode = 1;
