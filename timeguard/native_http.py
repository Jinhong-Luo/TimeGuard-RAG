"""验证固定底座原生会话与SSE协议；不把完整回答包装流称为实时模型流。"""
import json
import os
from pathlib import Path
import time
import urllib.request

from .common import Client,digest,read_json,write_json
from .native_baseline import private_runtime
from .run import chat_body


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();root=args.output;out=root/"http-session-stream";out.mkdir(exist_ok=False)
    setup=read_json(root/"setup.json");config=setup["config"]
    client=Client(config["base_url"],timeout=1200)
    payload=chat_body(config,"default","TimeGuard-原生会话与SSE-20261109")
    payload["prompt_config"]["refine_multiturn"]=True
    chat=client.request("POST","/chats",payload)
    write_json(out/"manifest.json",{"chat_id":chat["id"],"config":payload,"source_sha256":digest(Path(__file__).read_bytes()),
        "questions":["What were 3M's net sales in FY2018?","And what were its net sales in FY2017?"],
        "scope":"会话历史、指代消解与原生SSE协议；不用于开发集选优。"})
    private,_=private_runtime();audit=private/"bridge-audit.jsonl";begin=audit.stat().st_size
    records=[];session=None
    for question in ("What were 3M's net sales in FY2018?","And what were its net sales in FY2017?"):
        request={"chat_id":chat["id"],"question":question,"stream":False}
        if session:request["session_id"]=session
        started=time.monotonic()
        try:
            value=client.request("POST","/chat/completions",request)
            status="ok" if value.get("answer") and "**ERROR**" not in value["answer"] else "generation_error"
            if session and value.get("session_id")!=session:raise ValueError("续问未保持原会话ID")
            session=value.get("session_id")
            row={"question":question,"status":status,"response":value,"elapsed_seconds":time.monotonic()-started}
        except Exception as exc:
            row={"question":question,"status":"request_error","error":str(exc)[:500],"elapsed_seconds":time.monotonic()-started}
        records.append(row);write_json(out/"turns.json",records)
        if row["status"]!="ok":break
    stream={"status":"pending","event_count":0,"first_answer_seconds":None,"real_token_stream_verified":False}
    raw=json.dumps({"chat_id":chat["id"],"question":"What were 3M's net sales in FY2018?","stream":True}).encode()
    req=urllib.request.Request(client.base_url+"/chat/completions",raw,headers={"Authorization":"Bearer "+client.key,"Content-Type":"application/json"})
    started=time.monotonic()
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req,timeout=1200) as response,(out/"stream-events.jsonl").open("x") as save:
            stream["content_type"]=response.headers.get("Content-Type","")
            for line in response:
                text=line.decode().strip()
                if not text.startswith("data:"):continue
                data=text[5:].strip()
                if data=="[DONE]":break
                value=json.loads(data);stream["event_count"]+=1
                save.write(json.dumps({"seconds":time.monotonic()-started,"event":value},ensure_ascii=False)+"\n");save.flush()
                item=value.get("data")
                if isinstance(item,dict) and item.get("answer") and stream["first_answer_seconds"] is None:
                    stream["first_answer_seconds"]=time.monotonic()-started
        stream["status"]="ok" if stream["event_count"] and "text/event-stream" in stream.get("content_type","") else "protocol_error"
    except Exception as exc:stream.update(status="request_error",error=str(exc)[:500])
    stream["elapsed_seconds"]=time.monotonic()-started
    end=audit.stat().st_size;window=audit.read_bytes()[begin:end]
    calls=[json.loads(l) for l in window.splitlines() if l.strip()]
    write_json(out/"summary.json",{"turn_count":len(records),"turn_statuses":[r["status"] for r in records],
        "same_session":len(records)==2 and all(r["status"]=="ok" for r in records) and records[0]["response"].get("session_id")==records[1]["response"].get("session_id"),
        "stream":stream,"audit_byte_start":begin,"audit_byte_end":end,"audit_sha256":digest(window),
        "actual_calls":len(calls),"failed_calls":sum(r.get("success") is not True for r in calls),
        "tokens":sum((r.get("usage") or {}).get("totalTokens",0) for r in calls),"cost_usd":None,
        "说明":"SSE只验证原生HTTP协议和事件时间；当前Codex桥完整返回后才包装增量，未验证模型逐token实时流。会话ID一致也不单独证明续问答案正确。"})
    print("原生会话与SSE检查已保存",flush=True)


if __name__=="__main__":main()
