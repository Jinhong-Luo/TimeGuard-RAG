"""财报计算流程：检索、取证、校验、确定性计算及缺参追问。"""
from decimal import Decimal, InvalidOperation
import re
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from .scope import scope

Metric = Literal["growth", "margin", "ratio"]


class Operand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str = Field(min_length=1, max_length=128)
    quote: str = Field(min_length=1, max_length=2400)
    number: str = Field(max_length=64, pattern=r"^(?:-?\d[\d,]*(?:\.\d+)?|\(\d[\d,]*(?:\.\d+)?\))$")
    year: int = Field(ge=1900, le=2100)
    period: str = Field(pattern=r"^(FY|Q[1-4])$")
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    scale: Literal["ones", "thousands", "millions", "billions"]
    kind: Literal["money", "percentage"]


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: Metric | None
    left: Operand | None
    right: Operand | None
    missing: str = Field(max_length=800)


class CalculationArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: Metric
    left: Operand
    right: Operand


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


def calculate_financial_metric(metric, left, right):
    if left.kind != "money" or right.kind != "money":
        raise ValueError("该工具需要两个金额，不能混入百分比")
    if (left.currency, left.scale) != (right.currency, right.scale):
        raise ValueError("须先统一币种和金额单位")
    if left.currency == "XXX" and left.chunk_id != right.chunk_id:
        raise ValueError("币种未明时，两个数值必须来自同一张财务表")
    if left.period != right.period or left.year != right.year + int(metric == "growth"):
        raise ValueError("增长率需要相邻年度的同一期间，其他比率需要同一期间")
    a, b = amount(left), amount(right)
    if b == 0:
        raise ValueError("分母为零，无法计算")
    value = (a-b)/b if metric == "growth" else a/b
    if metric != "ratio":
        value *= 100
    return f"{value:.4f}".rstrip("0").rstrip(".") + ("%" if metric != "ratio" else "")


def verify_operand(operand, evidence):
    content = evidence["content"]
    if operand.quote not in content:
        raise ValueError("所引原句未出现在证据中")
    number = re.escape(operand.number)
    if not re.search(r"(?<![\d.,(\-])" + number + r"(?![\d.,)%])", operand.quote):
        raise ValueError("数值及正负号无法与引用原句核对")
    if str(operand.year) not in content:
        raise ValueError("证据缺少所选年份")
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
    if operand.period.startswith("Q") and operand.period.lower() not in content.lower():
        raise ValueError("证据缺少所选季度")


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
        chunks = await search_reports(s["question"], plan)
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
            for operand in (sel.left, sel.right):
                candidate = available.get(operand.chunk_id)
                if not candidate or candidate["document_id"] not in s["plan"]["document_ids"]:
                    raise ValueError("引用不在本次检索范围内")
                evidence = await get_evidence(candidate)
                if (evidence["id"], evidence["document_id"]) != (candidate["id"], candidate["document_id"]):
                    raise ValueError("返回的证据标识不匹配")
                verify_operand(operand, evidence)
                facts.append({**operand.model_dump(), "document_id": evidence["document_id"],
                              "positions": evidence.get("positions", [])})
            if sel.left.year not in s["plan"]["years"] or (sel.metric != "growth" and sel.right.year not in s["plan"]["years"]):
                raise ValueError("计算期间与问题指定年份不一致")
            requested = s["plan"]["quarters"]
            if requested and (len(requested) != 1 or sel.left.period != f"Q{requested[0]}"):
                raise ValueError("计算期间与问题指定季度不一致")
            result = calculate_financial_metric(sel.metric, sel.left, sel.right)
        except (ValueError, InvalidOperation) as exc:
            return {"status": "clarify", "answer": str(exc), "facts": facts}
        a, b = sel.left.number, sel.right.number
        formula = f"({a} − {b}) ÷ {b}" if sel.metric == "growth" else f"{a} ÷ {b}"
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

