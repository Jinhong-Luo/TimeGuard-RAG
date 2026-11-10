"""财报计算流程：检索、取证、校验、确定性计算及缺参追问。"""
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
import re
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .scope import scope
from .common import digest

Metric = Literal["growth", "margin", "ratio", "percentage_point_change", "period_sum"]


class Operand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str = Field(min_length=1, max_length=128)
    line_number: int | None = Field(default=None, ge=1)
    quote: str = Field(default="", max_length=2400)
    number: str = Field(max_length=64, pattern=r"^(?:-?\d[\d,]*(?:\.\d+)?|\(\d[\d,]*(?:\.\d+)?\))$")
    year: int = Field(ge=1900, le=2100)
    period: str = Field(pattern=r"^(FY|Q[1-4])$")
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    scale: Literal["ones", "thousands", "millions", "billions"]
    kind: Literal["money", "percentage"]
    row_label: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def require_evidence(self):
        if self.line_number is None and not self.quote.strip():
            raise ValueError("操作数须提供展示行号或逐字原句")
        return self


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: Metric | None
    left: Operand | None
    right: Operand | None
    additional: list[Operand] = Field(default_factory=list, max_length=10)
    missing: str = Field(max_length=800)


class CalculationArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: Metric
    left: Operand
    right: Operand
    additional: list[Operand] = Field(default_factory=list, max_length=10)


class ClarifyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=800)


def tool_selection(message):
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        raise ValueError("模型没有返回单一的财报工具调用")
    if calls[0].get("type") != "function" or not isinstance(calls[0].get("id"), str) or not calls[0]["id"]:
        raise ValueError("每步必须恰好返回一个带标识的工具调用")
    call = calls[0]
    fn = call.get("function")
    if not isinstance(fn, dict) or not isinstance(fn.get("arguments"), str):
        raise ValueError("工具参数须为 JSON 对象文本")
    if fn["name"] == "calculate_financial_metric":
        values = CalculationArgs.model_validate_json(fn["arguments"])
        selection = Selection(**values.model_dump(), missing="")
    elif fn["name"] == "request_financial_inputs":
        values = ClarifyArgs.model_validate_json(fn["arguments"])
        selection = Selection(metric=None, left=None, right=None, missing=values.question)
    else:
        raise ValueError("收到未声明的财报工具")
    return selection, {"id": call["id"], "name": fn["name"]}


class AgentState(TypedDict, total=False):
    question: str
    as_of: str | None
    candidates: list
    plan: dict
    selected: dict
    facts: list
    answer: str
    status: str
    usage: dict | None
    tool_call: dict


def amount(operand):
    number = operand.number
    if not re.fullmatch(r"(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", number.strip("()-")):
        raise ValueError("金额分隔符无效")
    value = Decimal(number.strip("()-").replace(",", ""))
    return -value if number.startswith(("-", "(")) else value


def calculate_financial_metric(metric, left, right, additional=()):
    operands = [left, right, *additional]
    if additional and metric != "period_sum":
        raise ValueError("此公式只允许两个操作数")
    if metric in ("percentage_point_change", "period_sum"):
        if not left.row_label or any(o.row_label.casefold() != left.row_label.casefold() or o.period != left.period for o in operands):
            raise ValueError("多期比较需要同一指标和同类期间")
        if len({o.year for o in operands}) != len(operands):
            raise ValueError("多期操作数出现重复年份")
        if metric == "percentage_point_change":
            if left.year <= right.year or any(o.kind != "percentage" for o in operands):
                raise ValueError("百分点差需要本期和较早期间的两个百分数")
            return f"{amount(left) - amount(right):f} 个百分点"
        if any((o.kind, o.currency, o.scale) != ("money", left.currency, left.scale) for o in operands):
            raise ValueError("汇总需要统一币种与单位的金额")
        if not re.search(r"sales|revenue|income|profit|expense|cash flow|销售|收入|利润|费用|现金流", left.row_label, re.I):
            raise ValueError("汇总仅支持已识别的期间流量指标")
        if left.currency == "XXX" and any(o.chunk_id != left.chunk_id for o in operands):
            raise ValueError("币种未知时仅允许在同一表内汇总")
        return f"{sum(map(amount, operands)):f} {left.currency} {left.scale}"
    if left.kind != "money" or right.kind != "money":
        raise ValueError("该工具需要两个金额，不能混入百分比")
    if (left.currency, left.scale) != (right.currency, right.scale):
        raise ValueError("须先统一币种和金额单位")
    if left.currency == "XXX" and left.chunk_id != right.chunk_id:
        raise ValueError("币种未明时，两个数值必须来自同一张财务表")
    if left.period != right.period or left.year != right.year + int(metric == "growth"):
        raise ValueError("增长率需要相邻年度的同一期间，其他比率需要同一期间")
    if metric == "growth" and left.row_label.casefold() != right.row_label.casefold():
        raise ValueError("增长率的两个操作数需要相同指标")
    if metric == "margin" and right.row_label and not re.search(r"revenue|sales|营收|收入|销售", right.row_label, re.I):
        raise ValueError("利润率分母必须为对应期间的收入")
    a, b = amount(left), amount(right)
    if b == 0:
        raise ValueError("分母为零，无法计算")
    value = (a-b)/b if metric == "growth" else a/b
    if metric != "ratio":
        value *= 100
    return f"{value:.4f}".rstrip("0").rstrip(".") + ("%" if metric != "ratio" else "")


class TableText(HTMLParser):
    """按原顺序保留表题、行和单元格，包括空列。"""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("tr", "caption", "br"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.parts.append(" | ")
        elif tag in ("tr", "caption"):
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def evidence_lines(content):
    if "<table" in content.lower():
        parser = TableText()
        parser.feed(content)
        parser.close()
        content = "".join(parser.parts)
    return [" ".join(line.split()) for line in content.splitlines() if line.strip()]


def present_evidence(chunks):
    budget = 18000
    result = []
    for chunk in chunks:
        available = min(4000, budget)
        shown = []
        for i, line in enumerate(evidence_lines(chunk["content"]), 1):
            if available <= 0:
                break
            fragment = line[:available]
            shown.append({"number": i, "text": fragment})
            available -= len(fragment) + 1
            budget -= len(fragment) + 1
        if shown:
            result.append({**chunk, "visible_lines": shown})
    return result


def verify_operand(operand, evidence, visible_lines=None):
    content = evidence["content"]
    quote = operand.quote
    if operand.line_number is not None:
        lines = evidence_lines(content)
        shown = next((line["text"] for line in visible_lines or () if line["number"] == operand.line_number), None)
        if not shown or operand.line_number > len(lines) or not lines[operand.line_number-1].startswith(shown):
            raise ValueError("所选行号无法在实收资料和回读原文中核对")
        quote = shown
    elif not quote or quote not in content or (visible_lines is not None and
            not any(quote in line["text"] for line in visible_lines)):
        raise ValueError("所引原句未出现在证据中")
    number = re.escape(operand.number)
    ending = r"\s*%" if operand.kind == "percentage" else r"(?![\d.,)%])"
    match = re.compile(r"(?<![\d.,(−\-])" + number + ending)
    if not match.search(quote):
        raise ValueError("数值及正负号无法与引用原句核对")
    if "<table" in content.lower():
        if (not operand.line_number or content.lower().count("<table") != 1
                or re.search(r"(?:rowspan|colspan)\s*=\s*['\"]?(?:[2-9]|\d{2,})", content, re.I)):
            raise ValueError("表格列对应关系无法核验")
        cells = [c.strip() for c in quote.removesuffix("|").split("|")]
        if operand.row_label and "".join(cells[0].casefold().split()) != "".join(operand.row_label.casefold().split()):
            raise ValueError("指标与所选表格行不一致")
        headers = [line["text"] for line in visible_lines or () if line["number"] < operand.line_number
                   and "|" in line["text"] and re.search(r"\b(?:19|20)\d{2}\b", line["text"])]
        header = [c.strip() for c in headers[-1].removesuffix("|").split("|")] if headers else []
        year_columns = [(i, m[1]) for i, cell in enumerate(header)
                        if (m := re.fullmatch(r"(?:FY\s*)?((?:19|20)\d{2})\s*\**", cell, re.I))]
        spans = [(i, next((j for j,_ in year_columns if j > i), len(header)))
                 for i,y in year_columns if y == str(operand.year)]
        if len(spans) != 1 or len(header) != len(cells) or not quote.endswith("|"):
            raise ValueError("所选数值与可见表头的年份列不一致")
        start, end = spans[0]
        # 财报解析可能把货币符号单独排成一列。仅合并无标题的相邻列且要求唯一数值。
        values = [v for v in cells[start:end] if v not in ("", "$", "€", "£")]
        if any(header[start+1:end]) or len(values) != 1 or not match.search(values[0]):
            raise ValueError("所选数值与可见表头的年份列不一致")
    elif str(operand.year) not in quote or (operand.row_label and operand.row_label.casefold() not in quote.casefold()):
        raise ValueError("原句缺少对应年份或指标")
    if operand.period.startswith("Q") and operand.period.lower() not in content.lower():
        raise ValueError("证据缺少所选季度")
    if operand.kind == "percentage":
        return quote
    currencies = {"USD": r"\bUSD\b|U\.S\. dollars?|US dollars?|US\$",
                  "EUR": r"\bEUR\b|euros?|€", "GBP": r"\bGBP\b|sterling|£"}
    currency_pattern = currencies.get(operand.currency, r"\b" + operand.currency + r"\b")
    if operand.currency == "XXX":
        if content.lower().count("<table") != 1 or re.search(r"\b(EUR|GBP|CAD|AUD|USD|CNY)\b|€|£", content, re.I):
            raise ValueError("未明确币种的计算需要来自同一张单币种金额表")
        currency_pattern = r"\$"
    if not re.search(currency_pattern, content, re.I):
        raise ValueError("证据缺少明确币种")
    if operand.scale != "ones" and not re.search(operand.scale.rstrip("s"), content, re.I):
        raise ValueError("证据缺少所选金额单位")
    if operand.scale == "ones" and re.search(r"\b(?:million|thousand|billion)s?\b", content, re.I):
        raise ValueError("原文标有缩放单位，无法按基本单位计算")
    return quote


def build_graph(search_reports, get_evidence, select_operands, documents):
    async def search_node(s):
        try:
            plan = scope(s["question"], documents, s.get("as_of"))
        except ValueError:
            return {"status": "clarify", "answer": "请用 YYYY-MM-DD 补充有效的历史截止日期。"}
        if len(plan["companies"]) != 1:
            return {"status": "clarify", "answer": "请明确要计算的公司。", "plan": plan}
        if not plan["years"]:
            return {"status": "clarify", "answer": "请补充要计算的财年或期间。", "plan": plan}
        chunks = present_evidence(await search_reports(s["question"], plan))
        if not chunks:
            return {"status": "clarify", "answer": "本次检索范围内缺少可用证据。", "plan": plan}
        return {"status": "search_ok", "candidates": chunks, "plan": plan}

    async def select_node(s):
        selection, usage, call = await select_operands(s["question"], s["candidates"])
        if not selection.left or not selection.right or selection.missing:
            return {"status": "clarify", "answer": selection.missing or "需要两个可核验的金额。", "usage": usage, "tool_call": call}
        return {"status": "selected", "selected": selection.model_dump(), "usage": usage, "tool_call": call}

    async def calculation(s):
        sel = Selection.model_validate(s["selected"])
        available = {c["id"]: c for c in s["candidates"]}
        facts = []
        try:
            for operand in (sel.left, sel.right, *sel.additional):
                candidate = available.get(operand.chunk_id)
                if not candidate or candidate["document_id"] not in s["plan"]["document_ids"]:
                    raise ValueError("引用不在本次检索范围内")
                evidence = await get_evidence(candidate)
                if (evidence["id"], evidence["document_id"]) != (candidate["id"], candidate["document_id"]):
                    raise ValueError("返回的证据标识不匹配")
                if evidence["content"] != candidate["content"]:
                    raise ValueError("检索后证据正文已变化，请重新检索")
                if (re.search(r"\bconsolidated\b|合并报表", s["question"], re.I) and
                        not re.search(r"consolidated|合并", evidence["content"], re.I)):
                    raise ValueError("当前证据没有明确标注问题要求的合并口径")
                quote = verify_operand(operand, evidence, candidate["visible_lines"])
                title = re.search(r"<caption\b[^>]*>(.*?)</caption>", evidence["content"], re.I | re.S)
                facts.append({**operand.model_dump(), "company": s["plan"]["companies"][0],
                              "table_caption": re.sub(r"<[^>]+>", "", title[1]).strip() if title else None,
                              "document_name": candidate.get("document_name") or candidate.get("document_keyword"),
                              "quote": quote, "content_sha256": digest(evidence["content"]),
                              "document_id": evidence["document_id"],
                              "positions": evidence.get("positions", [])})
            if sel.left.year not in s["plan"]["years"] or (sel.metric != "growth" and sel.right.year not in s["plan"]["years"]):
                raise ValueError("计算期间与问题指定年份不一致")
            requested = s["plan"]["quarters"]
            if sel.metric in ("margin", "ratio") and len(s["plan"]["years"]) > 1:
                raise ValueError("需要逐年计算并比较；单个比率不足以回答多期问题")
            if requested and (len(requested) != 1 or sel.left.period != f"Q{requested[0]}"):
                raise ValueError("计算期间与问题指定季度不一致")
            if sel.metric == "period_sum" and set(s["plan"]["years"]) != {f["year"] for f in facts}:
                raise ValueError("汇总操作数未完全覆盖问题指定的年份")
            result = calculate_financial_metric(sel.metric, sel.left, sel.right, sel.additional)
        except (ValueError, InvalidOperation) as exc:
            return {"status": "clarify", "answer": str(exc), "facts": facts}
        a, b = sel.left.number, sel.right.number
        formula = f"({a} − {b}) ÷ {b}" if sel.metric == "growth" else f"{a} ÷ {b}"
        if sel.metric == "percentage_point_change":
            formula = f"{a}% − {b}%"
        elif sel.metric == "period_sum":
            formula = " + ".join(f["number"] for f in facts)
        return {"status": "answered", "answer": f"{formula} = {result}", "facts": facts}

    graph = StateGraph(AgentState)
    graph.add_node("search_reports", search_node)
    graph.add_node("select_operands", select_node)
    graph.add_node("calculate", calculation)
    graph.add_edge(START, "search_reports")
    graph.add_conditional_edges("search_reports", lambda s: "select_operands" if s["status"] == "search_ok" else END)
    graph.add_conditional_edges("select_operands", lambda s: "calculate" if s["status"] == "selected" else END)
    graph.add_edge("calculate", END)
    return graph.compile()

