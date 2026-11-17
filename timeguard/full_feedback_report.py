"""仅在配对问答与模型裁判完成后，汇总联合达标率和每道获救题的增量成本。"""
import argparse
from collections import Counter
from pathlib import Path

from .common import read_json, read_jsonl, write_json
from .formula_guard import check
from .four_blocks import identity
from .full_feedback import ARMS
from .layer_report import page_set


STAGES = ("主模型反馈","旧池复查","定向补搜","补搜后筛选")
EARLIER = {"financebench_id_00382","financebench_id_00464","financebench_id_00685"}


def counted(seconds=0,usage=None):
    usage = usage or {}
    return Counter({"次数":1,"累计秒":seconds,"输入token":usage.get("prompt_tokens",0),
                    "输出token":usage.get("completion_tokens",0)})


def transitions(rows,left,right,stages):
    gains,losses = [],[]
    correct_gains,correct_losses = [],[]
    model = searches = 0
    for qid,v in rows.items():
        before,after = v[left]["达标"],v[right]["达标"]
        if after and not before:
            gains.append(qid)
        if before and not after:
            losses.append(qid)
        if v[left].get("判对") is False and v[right].get("判对") is True:
            correct_gains.append(qid)
        if v[left].get("判对") is True and v[right].get("判对") is False:
            correct_losses.append(qid)
        for stage in stages:
            n = v["阶段"].get(stage,{}).get("次数",0)
            if stage=="定向补搜":
                searches += n
            else:
                model += n
    net = len(gains)-len(losses)
    def price(denominator):
        if denominator<=0:
            return None
        return {"模型":round(model/denominator,3),"索引":round(searches/denominator,3),
                "合计":round((model+searches)/denominator,3)}
    return {"救回":gains,"损失":losses,"净增":net,"增量模型调用":model,"增量索引调用":searches,
            "每救回一道":price(len(gains)),"每净增一道":price(net),
            "仅按正确性错转对":correct_gains,"仅按正确性对转错":correct_losses}


def make(root,labels):
    questions = {r["id"] for r in read_jsonl("data/financebench/queries.jsonl") if r["split"]=="dev"}
    if len(questions)!=39:
        raise ValueError("开发集分母发生变化")
    rows, costs = {},{}
    unique_usage = {"成功回答":Counter(),"成功裁判":Counter(),"失败裁判":Counter(),"失败回答":Counter()}
    resources = Counter()
    for qid in sorted(questions):
        folder = root/qid
        keys = read_json(folder/"assignment.json")
        item = {"阶段":{}}
        rows[qid] = item
        first = read_json(folder/"pool.json") if (folder/"pool.json").exists() else {"chunks":[],"allowed":[]}
        targets = {(e["doc_name"].casefold(),e["evidence_page_num"]+1) for e in labels[qid]["evidence"]}
        item["标准题页数"] = len(targets)
        old_pages,old_valid = page_set(first["chunks"])
        item["初搜块数"] = len(first["chunks"])
        item["初搜命中标准题页数"] = len(targets & old_pages) if old_valid else None
        available = {identity(c):c for c in first["chunks"]}
        if (folder/"feedback.json").exists():
            dec = read_json(folder/"feedback.json")["audit"]
            select = read_json(folder/"selection.json")
            events = [{**e,"阶段":"首次筛选四块"} for e in select["log"]] + dec["log"]
            events += [{"阶段":"初搜","seconds":first["seconds"]}]
            for e in events:
                name = e["阶段"]
                c = counted(e["seconds"],e.get("usage"))
                if name=="定向补搜" and "response" in e:
                    new = e["response"]
                    pages,valid = page_set(new)
                    item["补搜首次找回标准题页数"] = len((pages-old_pages)&targets) if valid and old_valid else None
                    available.update({identity(v):v for v in new})
                if e.get("fallback") or dec.get("回退阶段")==name:
                    c["回退"] += 1
                item["阶段"][name] = dict(c)
                costs.setdefault(name,Counter()).update(c)
            item.update({k:v for k,v in dec.items() if k not in ("log","最终排名","起始排名","目录补搜")})
            if "目录补搜" in dec:
                nav = dec["目录补搜"]
                item["目录补搜"] = {k:v for k,v in nav.items() if k!="章节范围"}
                item["目录补搜"]["选用章节数"] = len(nav.get("章节范围", []))
        item["回看改变实发输入"] = keys[ARMS[0]]!=keys[ARMS[1]]
        item["补搜改变实发输入"] = keys[ARMS[1]]!=keys[ARMS[2]]
        used = set()
        for arm,key in keys.items():
            gen = read_json(folder/"answers"/(key+".json"))
            score = read_json(folder/"judges"/(key+".json"))
            if score["judge_status"]!="ok":
                raise ValueError("评分未完成")
            correct,support = score["answer_correct"],score["citation_support"]
            if key not in used:
                used.add(key)
                if key!="scope":
                    unique_usage["成功回答"].update(counted(gen["elapsed_seconds"],gen.get("usage")))
                unique_usage["成功裁判"].update(counted(score["elapsed_seconds"],score.get("usage")))
            v = {"判对":correct,"引用支持":support,"达标":correct is True and support is True,
                 "回答用量":gen.get("usage"),"回答秒":gen.get("elapsed_seconds",0),
                 "裁判用量":score.get("usage"),"裁判秒":score["elapsed_seconds"]}
            if key=="scope":
                v.update({"澄清公司":True,"实收块数":0,"实收标准题页数":0})
            else:
                a, = gen["native_trace"]["evidence_layers"]["answers"]
                request, = a["requests"]
                visible = request["complete_chunks"]
                pages,valid = page_set(visible)
                selected,known = page_set(a["selected"])
                v.update({"实收块数":len(visible),"选中块数":len(a["selected"]),
                          "选中标准题页数":len(selected&targets) if known else None,
                          "实收标准题页数":len(pages&targets) if valid else None})
                guard = check(gen["question"],visible,first["allowed"],lambda c:available[identity(c)])
                v["操作数"] = {k:guard[k] for k in ("required","verified") if k in guard}
            item[arm] = v
        resources["实际成功回答模型调用"] += len(set(keys.values())-{"scope"})
        resources["实际成功裁判调用"] += len(set(keys.values()))
        resources["失败回答调用"] += len(list((folder/"answers").glob("*.failure-*.json")))
        resources["失败裁判调用"] += len(list((folder/"judges").glob("*.failure-*.json")))
        for sub,stage in (("answers","失败回答"),("judges","失败裁判")):
            for path in (folder/sub).glob("*.failure-*.json"):
                failed = read_json(path)
                unique_usage[stage].update(counted(failed.get("elapsed_seconds",failed.get("seconds",0)),failed.get("usage")))
    result = {"类型":"完整开发集同题配对；同题实发请求与引用映射相同则共享本次真实生成及评分",
              "题数":len(rows),"实际执行":dict(resources),"实际问答及裁判用量":{k:dict(v) for k,v in unique_usage.items()},
              "步骤成本":{k:dict(v) for k,v in costs.items()},"逐题":rows}
    for arm in ARMS:
        values = [r[arm] for r in rows.values()]
        result[arm] = {"正确":sum(v["判对"] is True for v in values),"联合达标":sum(v["达标"] for v in values),
                       "判错":sum(v["判对"] is False for v in values),"正确性未知":sum(v["判对"] is None for v in values),
                       "引用支持未知":sum(v["引用支持"] is None for v in values)}
        result[arm]["逻辑回答用量"] = dict(sum((counted(v["回答秒"],v["回答用量"]) for v in values if not v.get("澄清公司")),Counter()))
    for title,subset in (("全量",rows),("除去先前保护题",{k:v for k,v in rows.items() if k not in EARLIER})):
        result[title] = {"题数":len(subset),
                         "完整流程对无反馈":transitions(subset,ARMS[0],ARMS[2],STAGES),
                         "仅补搜对只回看":transitions(subset,ARMS[1],ARMS[2],STAGES[2:]),
                         "只回看对无反馈":transitions(subset,ARMS[0],ARMS[1],STAGES[:2])}
    return result


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("out",type=Path)
    args = p.parse_args()
    gold = {r["id"]:r for r in read_jsonl("data/financebench/gold.jsonl")}
    write_json(args.out,make(args.root,gold))
