"""按缺少的指标和期间回读本题此前检索到、但未进入最终池的原块。"""
import re
import time
from collections import defaultdict
from urllib.parse import quote

from .common import digest
from .financial_agent import amount, evidence_lines
from .formula_guard import ROWS, candidates, requirements, row_without_acronym
from .scope import quarters, scope


def verified_values(shown, raw, needs, period, usd, allow_acronym=False):
    """复用原文行列和单位检查，保留冲突值供调用方判断。"""
    values = defaultdict(set)
    if re.search(r"segment|division|geographic", raw["content"], re.I) and not re.search(
            r"consolidated", raw["content"], re.I):
        return values
    for metric, year in needs:
        pattern = ROWS[metric][1:-1].replace(" ", r"\s*")
        if not re.search(pattern, shown["content"], re.I):
            continue
        for operand in candidates(shown, raw, metric, year, period, allow_acronym=allow_acronym):
            if usd and metric != "毛利率" and operand.currency != "USD":
                continue
            values[metric, year].add((amount(operand), operand.currency, operand.scale))
    return values


def related_table(text, metric, year):
    """识别有目标行和期间的单表原文，不把相关性当作金额或币种核验。"""
    if text.casefold().count("<table") != 1 or not re.search(r"consolidated", text, re.I):
        return False
    table = re.search(r"<table\b[^>]*>.*?</table>", text, re.I | re.S)
    if not table:
        return False
    header_sizes = set()
    for line in evidence_lines(table[0]):
        cells = line.rstrip("|").split("|")
        if len(cells) < 2:
            continue
        label = row_without_acronym(cells[0].strip().casefold())
        if len(cells) in header_sizes and re.fullmatch(ROWS[metric].replace(" ", r"\s*"), label):
            if any(re.search(r"\d", cell) for cell in cells[1:]):
                return True
        if any(re.search(r"\b(?:19|20)\d{2}\b", cell) for cell in cells[1:]):
            header_sizes = {len(cells)} if any(re.search(rf"\b{year}\b", c) for c in cells[1:]) else set()
    return False


def recover(question, visible, final_pool, calls, allowed, read, allow_tables=False):
    started = time.monotonic()
    audit = {"读取": 0, "读取失败": 0, "指纹不符": 0, "历史池外块": 0, "核对原块": 0,
             "所需事实": [], "缺项": [], "补入": [], "待核原表": [], "冲突": []}
    result = {"chunks": [], "audit": audit}

    def finish():
        audit["秒"] = round(time.monotonic() - started, 4)
        return result

    plan = requirements(question)
    if visible is None or not allowed or not plan or re.search(
            r"\b(segment|division|department|domestic|international|geographic|business unit)\b", question, re.I):
        return finish()
    needs = plan[1]
    period_list = quarters(question)
    if len(period_list) > 1:
        return finish()
    period = f"Q{period_list[0]}" if period_list else "FY"
    usd = bool(re.search(r"\bUSD\b|U\.S\. dollars", question, re.I))
    audit["所需事实"] = needs

    def fetch(doc, ident):
        if doc not in allowed:
            raise ValueError("历史回读超出允许范围")
        audit["读取"] += 1
        try:
            raw = read(doc, ident)
        except (OSError, RuntimeError):
            audit["读取失败"] += 1
            return None
        if raw is None:
            audit["读取失败"] += 1
            return None
        if (raw["document_id"], raw["id"]) != (doc, ident):
            raise ValueError("原文回读来源不匹配")
        return raw

    present = defaultdict(set)
    for c in visible:
        raw = fetch(c["document_id"], c["id"])
        if raw is None:
            return finish()
        for fact, values in verified_values(c, raw, needs, period, usd, allow_tables).items():
            present[fact].update(values)
    missing = [key for key in needs if not present[key]]
    audit["缺项"] = missing
    audit["冲突"] = [key for key in needs if len(present[key]) > 1]
    if not missing:
        return finish()
    excluded = {(c["document_id"], c["id"]) for c in final_pool}
    history = defaultdict(set)
    for call in calls:
        for c in call.get("chunks") or []:
            key = c["document_id"], c["id"]
            if key[0] in allowed and key[1] and key not in excluded:
                history[key].add(c["content_sha256"])
    audit["历史池外块"] = len(history)
    matches = defaultdict(dict)
    tables = defaultdict(list)
    for (doc, ident), hashes in history.items():
        raw = fetch(doc, ident)
        if raw is None:
            continue
        if digest(raw["content"]) not in hashes:
            audit["指纹不符"] += 1
            continue
        audit["核对原块"] += 1
        for fact, values in verified_values(raw, raw, missing, period, usd, allow_tables).items():
            for value in values:
                matches[fact].setdefault(value, raw)
        if allow_tables and period == "FY":
            for fact in missing:
                if related_table(raw["content"], *fact):
                    tables[fact].append(raw)
    if audit["读取失败"]:
        return finish()
    used = set()
    for fact in missing:
        options = matches[fact]
        if len(options) > 1:
            audit["冲突"].append(fact)
        if len(options) == 1:
            c = next(iter(options.values()))
            audit["补入"].append(fact)
        elif not options and len(tables[fact]) == 1:
            c = tables[fact][0]
            audit["待核原表"].append(fact)
        else:
            continue
        key = c["document_id"], c["id"]
        if key not in used:
            result["chunks"].append(c)
            used.add(key)
    return finish()


def recover_row(row, client, documents, dataset, allow_tables=False):
    trace = row["native_trace"]["evidence_layers"]
    answer, = trace["answers"]
    request, = [r for r in answer["requests"] if r.get("status") == 200]
    snapshot = answer["snapshot"]["input"]
    if dataset != snapshot["identity"]["dataset_id"]:
        raise ValueError("快照与当前资料库不一致")
    plan = scope(row["question"], documents, row.get("as_of"))
    allowed = set(snapshot["tools"]["doc_scope"] or ())
    if len(plan["companies"]) != 1:
        allowed.clear()
    elif not allowed.issubset(plan["document_ids"] or []):
        raise ValueError("研究材料范围超过本题允许的公司或截止日")

    names = {d["id"]: d["doc_name"] for d in documents}

    def read(doc, ident):
        data = client.request("GET", f"/datasets/{dataset}/documents/{quote(doc,safe='')}/chunks?id={quote(ident,safe='')}")
        chunks = data["chunks"]
        return {**chunks[0], "document_name": names[doc]} if len(chunks) == 1 and chunks[0].get("available") else None

    return recover(row["question"], request["complete_chunks"], answer["pool_locations"],
                   trace["retrievals"], allowed, read, allow_tables)


def history_reader(source):
    from pathlib import Path
    from .common import Client, read_json, read_jsonl
    from .matrix import frozen_index
    from .retrieval_experiment import catalog
    freeze = frozen_index(Path("data/ingestion.json"), Path("runs/r0-gpu-retrieval-all"),
                          Path("data/financebench/queries.jsonl"))
    if freeze != read_json(source/"manifest.json")["identity"]["freeze"]:
        raise ValueError("历史回读的索引与快照不符")
    state = read_json("data/ingestion.json")
    docs = catalog(read_jsonl("data/financebench/documents.jsonl"), state)
    client = Client(read_json("configs/r0-low.json")["base_url"])
    return lambda row, allow_tables: recover_row(row, client, docs, state["dataset_id"], allow_tables)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from .common import read_jsonl, write_json
    from .paired_trial import render
    parser = argparse.ArgumentParser(description="回放全部历史候选，只报告脱敏计数，不回答或评分")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--history-tables", action="store_true")
    parser.add_argument("--history-swap", action="store_true")
    args = parser.parse_args()
    reader = history_reader(args.source)
    summary = {}
    for row in read_jsonl(args.source/"results.jsonl"):
        answers = row["native_trace"].get("evidence_layers", {}).get("answers", [])
        if len(answers) != 1 or not answers[0].get("snapshot") or sum(
                r.get("status") == 200 for r in answers[0]["requests"]) != 1:
            summary[row["id"]] = {"状态": "缺少可核对的回答快照"}
            continue
        found = reader(row, args.history_tables)
        same = True
        fallback = None
        displaced = 0
        if found["chunks"]:
            pair, = render([row], ["original","history_read"], history={row["id"]:found},
                           history_swap=args.history_swap)
            same = pair["original"]["body"] == pair["history_read"]["body"]
            fallback = pair["history_read"].get("history_fallback")
            displaced = pair["history_read"].get("history_displaced",0)
        summary[row["id"]] = {**found["audit"], "状态":"已回放", "补入块数":len(found["chunks"]),
                               "最终消息相同":same,"预算回退":fallback,"原有实收块减少数":displaced}
    write_json(args.output, {"类型":"固定快照离线回放，没有调用回答模型和裁判", "逐题":summary})
