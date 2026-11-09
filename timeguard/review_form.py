"""把已冻结的抽查包渲染为可在本机打开的人工复核表。"""
import argparse
import html
import json
from pathlib import Path
from .common import digest, read_json, read_jsonl


def render(root):
    packet = read_jsonl(root / "review.jsonl")
    summary = read_json(root / "summary.json")
    if summary["sample_ids"] != [r["id"] for r in packet]:
        raise ValueError("复核包与抽样清单不一致")
    def esc(value):
        return html.escape(str(value))
    questions = []
    for row in packet:
        refs = "".join(f"<details><summary>候选原位置 {i} · {esc(c['document_name'])} · {esc(c['id'])}</summary><pre>{esc(c['content'])}</pre></details>"
                       for i, c in enumerate(row["reference_candidates"]))
        gold = "".join(f"<details><summary>{esc(e['doc_name'])} · PDF 第 {int(e['evidence_page_num'])+1} 页</summary><pre>{esc(e['evidence_text'])}</pre></details>"
                       for e in row["reference_evidence"])
        choices = "".join(f'<label>{label} <select name="{field}"><option value="">待核</option><option value="true">是</option><option value="false">否</option><option value="unknown">无法判断</option></select></label>'
                          for field, label in (("gold_sufficient", "标准证据足够"), ("answer_correct", "回答正确"),
                                               ("citation_support", "引用支持回答"), ("judge_reasonable", "原裁判合理")))
        hypotheses = esc(json.dumps(row["mapping_hypotheses"], ensure_ascii=False, indent=2))
        questions.append(f"""<article data-id="{esc(row['id'])}"><h2>{esc(row['id'])}</h2><p>{esc(row['question'])}</p>
<h3>模型原答</h3><pre>{esc(row['answer'])}</pre><h3>标准答案及证据</h3><pre>{esc(row['reference_answer'])}</pre>{gold}
<h3>问答返回的候选块</h3>{refs}<details><summary>两种旧映射的排查线索</summary><pre>{hypotheses}</pre></details>
<details><summary>原模型裁判意见</summary><pre>{esc(row['historical_judge']['judgement_reason'])}</pre></details>
<fieldset>{choices}<label>依据或疑点<textarea name="notes"></textarea></label></fieldset></article>""")
    header = """<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>TimeGuard 引用复核</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{font:16px/1.55 system-ui;color:#1a2633;max-width:920px;margin:2rem auto;padding:0 1rem}
article{border-top:2px solid #47706c;padding:1rem 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f5;padding:.75rem}fieldset{display:grid;gap:.6rem;margin:1rem 0}
textarea{display:block;width:96%;min-height:4rem}button,input,select{padding:.4rem}summary{cursor:pointer}header{position:sticky;top:0;background:white;border-bottom:1px solid #bcc;padding:.5rem 0}
</style><h1>历史引用人工复核</h1><p>先对照答案和原文，再展开原裁判意见。原始记录未保存最终提示词和自动补引用来源。导出文件保存你的判断，刷新页面会清除尚未导出的输入。</p>
<header><label>复核人 <input id="reviewer" required></label> <button id="save">导出复核记录</button> <span id="message" role="status"></span></header>"""
    script = """<script>document.querySelector('#save').onclick=()=>{
const reviewer=document.querySelector('#reviewer').value.trim(), message=document.querySelector('#message');
if(!reviewer){message.textContent='请填写复核人';return}
const rows=[...document.querySelectorAll('article')].map(a=>{
const r={id:a.dataset.id,reviewer,judge_type:'human',review_packet_sha256:HASH,
unknown_fields:[...a.querySelectorAll('select')].filter(el=>el.value==='unknown').map(el=>el.name)};
for(const el of a.querySelectorAll('[name]'))r[el.name]=el.tagName==='TEXTAREA'?el.value:
el.value==='true'?true:el.value==='false'?false:null;
return r}).filter(r=>r.notes||r.unknown_fields.length||[r.gold_sufficient,r.answer_correct,r.citation_support,r.judge_reasonable].some(v=>v!==null));
if(!rows.length){message.textContent='请先填写至少一项判断或理由';return}
const url=URL.createObjectURL(new Blob([rows.map(r=>JSON.stringify(r)).join('\\n')+'\\n'],{type:'application/x-ndjson'}));
const link=document.createElement('a');link.href=url;link.download='人工复核.jsonl';link.click();
setTimeout(()=>URL.revokeObjectURL(url),2000);message.textContent='已导出 '+rows.length+' 道题';};</script>"""
    return header + "".join(questions) + script.replace("HASH", json.dumps(digest((root / "review.jsonl").read_bytes()))) + "</html>"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("review", type=Path)
    args = p.parse_args()
    path = args.review / "人工复核.html"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(render(args.review))


if __name__ == "__main__":
    main()

