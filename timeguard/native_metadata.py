"""原生基线的文档元数据准备与恢复；只写入本次专用字段，冲突时停止。"""
import asyncio
from collections import defaultdict
import json
import sys

KEYS = ("tg_native_company", "tg_native_report_year")


def expected_metadata(old, document):
    if any(k in old for k in KEYS):
        raise ValueError("原元数据已有本次专用字段，拒绝覆盖")
    return {**old, KEYS[0]:document["company"], KEYS[1]:str(document["doc_period"])}


def restore_allowed(current, old, document):
    return current == expected_metadata(old, document)


async def execute(spec):
    from common import settings
    settings.init_settings()
    from api.db.services.doc_metadata_service import DocMetadataService as D
    from common.metadata_utils import apply_meta_data_filter
    action=spec["action"];docs=spec["documents"];kb=spec["dataset_id"]
    health=settings.docStoreConn.health()
    if health.get("status") not in ("green","yellow") or health.get("timed_out"):
        raise RuntimeError("元数据索引后端不健康，禁止将空回读当作恢复成功")
    if action=="snapshot":
        return {d["id"]:D.get_document_metadata(d["id"]) for d in docs}
    if action in ("populate","restore"):
        results=[]
        for doc in docs:
            ident=doc["id"];old=spec["backup"][ident]
            current=D.get_document_metadata(ident)
            wanted=expected_metadata(old,doc)
            if action=="populate":
                if current != old:
                    raise ValueError("元数据在快照后发生变化，停止："+ident)
                ok=D.update_document_metadata(ident,wanted)
            else:
                if current==old:
                    results.append({"id":ident,"restored":True});continue
                if not restore_allowed(current,old,doc):
                    raise ValueError("元数据出现其他写入，禁止恢复覆盖："+ident)
                ok=D.update_document_metadata(ident,old) if old else D.delete_document_metadata(ident,kb)
            if not ok:
                raise ValueError("原生元数据写入或恢复失败："+ident)
            if D.get_document_metadata(ident)!=(wanted if action=="populate" else old):
                raise ValueError("原生元数据回读不一致："+ident)
            results.append({"id":ident,"restored":action=="restore"})
        if action=="restore":
            remaining=D.get_flatted_meta_by_kbs([kb])
            if any(k in remaining for k in KEYS):raise ValueError("专用元数据字段仍有残留")
        return {"action":action,"documents":len(results),"verified":True,"index_health":health["status"]}
    if action=="checks":
        groups=defaultdict(set)
        for d in docs:groups[d["company"]].add(d["id"])
        checks=[]
        for company,ids in sorted(groups.items()):
            cfg={"method":"manual","manual":[{"key":KEYS[0],"op":"=","value":company}]}
            found=await apply_meta_data_filter(cfg,kb_ids=[kb])
            if set(found)!=ids:
                raise ValueError("原生公司过滤不符合台账："+company)
            checks.append({"case":"公司相等","company":company,"matched":len(found),"passed":True})
        cfg={"method":"manual","manual":[{"key":KEYS[0],"op":"=","value":"不存在的测试公司"}]}
        found=await apply_meta_data_filter(cfg,kb_ids=[kb])
        checks.append({"case":"空匹配保留无结果语义","result":found,"passed":found==["-999"]})
        company=sorted(groups)[0];cfg["manual"][0]["value"]=company
        other=next(d["id"] for d in docs if d["company"]!=company)
        found=await apply_meta_data_filter(cfg,base_doc_ids=[other],kb_ids=[kb])
        checks.append({"case":"与已有范围取交集","result":found,"passed":found==["-999"]})
        years=sorted({int(d["doc_period"]) for d in docs})
        year=years[len(years)//2]
        conditions=[{"key":KEYS[1],"op":"≥","value":str(year)}]
        expected={d["id"] for d in docs if int(d["doc_period"])>=year}
        found=await apply_meta_data_filter({"method":"manual","manual":conditions},kb_ids=[kb])
        checks.append({"case":"报告年份范围","year":year,"matched":len(found),"passed":set(found)==expected})
        company_condition={"key":KEYS[0],"op":"=","value":company}
        for logic in ("and","or"):
            expected={d["id"] for d in docs if ((d["company"]==company and int(d["doc_period"])>=year)
                      if logic=="and" else (d["company"]==company or int(d["doc_period"])>=year))}
            found=await apply_meta_data_filter({"method":"manual","logic":logic,"manual":[company_condition,*conditions]},kb_ids=[kb])
            checks.append({"case":"多条件"+logic,"matched":len(found),"passed":set(found)==expected if expected else found==["-999"]})
        found=await apply_meta_data_filter({"method":"manual","manual":[{"key":"不存在的元数据字段","op":"=","value":"值"}]},kb_ids=[kb])
        checks.append({"case":"缺失字段不误召回","result":found,"passed":found==["-999"]})
        return {"action":action,"checks":checks,"passed":all(c["passed"] for c in checks)}
    raise ValueError("未知元数据操作")


if __name__=="__main__":
    result=asyncio.run(execute(json.load(sys.stdin)))
    print("TIMEGUARD_METADATA "+json.dumps(result,ensure_ascii=False),flush=True)
