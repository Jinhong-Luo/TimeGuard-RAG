"""仅导出计数、题号和成本；原文、问题和模型请求留在私有运行目录。"""
import argparse
from collections import Counter
from pathlib import Path
from timeguard.common import read_json, write_json
from timeguard.evals.trials.fact_final_trial import ARMS


def usage_sum(values):
    vals=list(values)
    out={"请求":len(vals),"用量缺失":sum(v is None for v in vals)}
    for key in ("prompt_tokens","completion_tokens","total_tokens"):
        out[key]=sum(v.get(key,0) for v in vals if v)
    return out


def make(root):
    raw=read_json(root/"summary.json")
    rows=raw["逐题"]
    groups={}
    for arm in ARMS:
        group=[r[arm] for r in rows.values()]
        counts=Counter("对" if r["判对"] is True else "错" if r["判对"] is False else "未知" for r in group)
        groups[arm]={"题数":len(group),"答对":counts["对"],"答错":counts["错"],"未知":counts["未知"],
                     "正确且引用支持":sum(r["达标"] for r in group)}
        for key in ("选中块","实收块","选中题页","实收题页","事实应收","事实实收","操作数应有","操作数核验"):
            groups[arm][key]=sum(r.get(key) or 0 for r in group)
        groups[arm]["无新筛选沿用原版"] = sum(r["装入回退"]=="无新事实筛选" for r in group)
        groups[arm]["装入失败回退"] = sum(bool(r["装入回退"]) and r["装入回退"]!="无新事实筛选" for r in group)
        groups[arm]["回答逻辑用量"]=usage_sum(r["回答用量"] for r in group if r["回答用量"] is not None)
        groups[arm]["回答累计秒"]=sum(r["回答秒"] or 0 for r in group)
    reviews=[e for r in rows.values() for e in r.get("筛选用量",[])]
    selector=usage_sum(reviews)
    selector.update(累计秒=sum(r.get("筛选秒",0) for r in rows.values()),
                    回退题数=sum(bool(r.get("回退")) for r in rows.values()))
    actual={}
    for name,pattern in (("初次回答","*/answers/*.json"),("初次裁判","*/judges/*.json"),
                         ("复验回答","confirmation/*/answers/*.json"),("复验裁判","confirmation/*/judges/*.json")):
        paths=[p for p in root.glob(pattern) if '.failure-' not in p.name and ("裁判" in name or p.name!='scope.json')]
        data=[read_json(p) for p in paths]
        actual[name]=usage_sum(d.get("usage") for d in data)
        actual[name]["累计秒"]=sum(d.get("elapsed_seconds",0) for d in data)
    public={"类型":"冻结搜索与反馈的三组真实回答配对，不是重新搜索的端到端运行", "题数":len(rows),"判分来源":"模型裁判，未进行独立人工逐题裁定",
            "候选命中题页":sum(r.get("候选命中题页",0) for r in rows.values()),"标准题页分母":sum(r.get("标准题页",0) for r in rows.values()),
            "比较组":groups,"相对原版的联合指标变化":raw["变化"],"翻转题独立复验":raw.get("独立复验",{}),
            "本轮新筛选":selector,"本轮原文回读":{"请求":sum(r.get("回读次数",0) for r in rows.values()),
            "失败":sum(r.get("回读失败",0) for r in rows.values()),"累计秒":sum(r.get("回读秒",0) for r in rows.values())},
            "本轮新增搜索":0,"本轮实际调用":actual,"回答裁判失败记录":raw["失败次数"],
            "成本口径":"本轮新增执行筛选，对照沿用冻结选择。正常路径可替换原筛选，但异常回退也复用冻结原版结果；未验证线上回退成本，不可据此声称端到端额外调用为0。原文回读、回答和裁判分别计数。",
            "事实口径":"由筛选模型从问题与允许候选提出，程序核对原文与实收。事实实收不是人工标注完整率，也不是独立的语义正确性证明。",
            "逐题":rows}
    public["分步配对"]={}
    for left,right in (("fixed","fact_select"),("fact_select","fact_deliver"),("fixed","fact_deliver")):
        public["分步配对"][left+"→"+right]={
            "联合救回":[q for q,r in rows.items() if r[right]["达标"] and not r[left]["达标"]],
            "联合损失":[q for q,r in rows.items() if r[left]["达标"] and not r[right]["达标"]],
            "答案新增正确":[q for q,r in rows.items() if r[right]["判对"] is True and r[left]["判对"] is not True],
            "答案失去正确":[q for q,r in rows.items() if r[left]["判对"] is True and r[right]["判对"] is not True]}
    if (root/"pair.json").exists():
        base=Path(read_json(root/"pair.json")["base"])
        events=[e for q in rows if (base/q/"feedback.json").exists()
                for e in read_json(base/q/"feedback.json")["audit"]["log"] if e.get("阶段")=="补搜后筛选"]
        prior=usage_sum(e.get("usage") for e in events)
        prior["累计秒"]=sum(e.get("seconds",0) for e in events)
        public["历史同一步筛选成本"]={**prior,"口径":"原冻结流程实测；本轮未重复旧筛选，不是同期延迟对照"}
        public["替换筛选的用量差"]={k:selector[k]-prior[k] for k in ("请求","prompt_tokens","completion_tokens","total_tokens")}
    return public


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    write_json(a.output,make(a.root))
