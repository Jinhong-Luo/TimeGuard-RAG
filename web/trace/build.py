"""构建「逐题回放」页面:单文件,离线可用。

用法:python build.py
输入:同目录的 data.json(由 export_trace.py 从服务器导出),以及项目里的 data/financebench/gold.jsonl。
输出:上一级目录的 TimeGuard-RAG-逐题回放.html(单文件,双击即可打开);dist/ 是分文件的开发版。

页面只读这两个文件,不连服务器、不联网。
"""
import datetime
import hashlib
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
GOLD = os.environ.get("FINANCEBENCH_GOLD") or os.path.join(ROOT, "data", "financebench", "gold.jsonl")
OUT_SINGLE = os.path.join(HERE, "..", "TimeGuard-RAG-逐题回放.html")
DIST = os.path.join(HERE, "dist")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load():
    data = json.load(open(os.path.join(HERE, "data.json"), encoding="utf-8"))
    dataset = {}
    for line in open(GOLD, encoding="utf-8"):
        if not line.strip():
            continue
        g = json.loads(line)
        # FinanceBench 的 evidence_page_num 从 0 开始(已用内容重叠度对照索引片段核实),页面按 PDF 页号显示,所以加 1。
        dataset[g["id"]] = {
            "company": g.get("company"), "doc": g.get("doc_name"), "answer": g.get("answer"),
            "type": g.get("question_type"), "reasoning": g.get("question_reasoning"),
            "evidence": [{"doc": e.get("doc_name") or g.get("doc_name"), "page": e.get("evidence_page_num"), "text": e.get("evidence_text")}
                         for e in g.get("evidence") or []],
        }
    missing = [q for q in data["questions"] if q not in dataset]
    extra = [q for q in dataset if q not in data["questions"]]
    if missing or extra:
        raise SystemExit(f"题库和运行记录对不上:运行里有、题库里没有 {missing[:3]};题库里有、运行里没有 {extra[:3]}")
    data["dataset"] = dataset
    # 构建信息只写进页面,不回写 data.json:数据文件保持导出时的原样,它的指纹才稳定、可用来核对。
    data["meta"]["build"] = {
        "gold_sha256": sha256(GOLD), "data_sha256": sha256(os.path.join(HERE, "data.json")),
        "gold_file": "data/financebench/gold.jsonl", "built_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    return data


def main():
    data = load()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    tpl = open(os.path.join(HERE, "index.html"), encoding="utf-8").read()
    css = open(os.path.join(HERE, "style.css"), encoding="utf-8").read()
    app = open(os.path.join(HERE, "app.js"), encoding="utf-8").read()
    for marker in ("<!--STYLE-->", "<!--DATA-->", "<!--APP-->"):
        if tpl.count(marker) != 1:
            raise SystemExit(f"模板里应当恰好有一个 {marker}")

    single = (tpl.replace("<!--STYLE-->", "<style>" + css + "</style>")
                 .replace("<!--DATA-->", '<script type="application/json" id="trace-data">' + payload + "</script>")
                 .replace("<!--APP-->", "<script>" + app + "</script>"))
    with open(OUT_SINGLE, "w", encoding="utf-8", newline="\n") as f:
        f.write(single)

    os.makedirs(DIST, exist_ok=True)
    dev = (tpl.replace("<!--STYLE-->", '<link rel="stylesheet" href="style.css">')
              .replace("<!--DATA-->", '<script src="data.js"></script>')
              .replace("<!--APP-->", '<script src="app.js"></script>'))
    with open(os.path.join(DIST, "index.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(dev)
    with open(os.path.join(DIST, "style.css"), "w", encoding="utf-8", newline="\n") as f:
        f.write(css)
    with open(os.path.join(DIST, "app.js"), "w", encoding="utf-8", newline="\n") as f:
        f.write(app)
    with open(os.path.join(DIST, "data.js"), "w", encoding="utf-8", newline="\n") as f:
        f.write("window.TRACE_DATA=" + payload + ";")

    def joint(arm):
        x, y = arm["judge"].get("answer_correct"), arm["judge"].get("citation_support")
        if x is False or y is False:
            return "fail"
        return "pass" if (x is True and y is True) else "unknown"

    qs = data["questions"]
    print(f"题目 {len(qs)} 道 | 有正文的片段 {len(data['chunks'])} 块 | 运行 {data['meta']['run']} | 导出于 {data['meta']['exported_at']}")
    for arm in ("A0", "A1"):
        c = {"pass": 0, "fail": 0, "unknown": 0}
        for q in qs.values():
            c[joint(q["arms"][arm])] += 1
        print(f"  {arm}: 答对且引用支持 {c['pass']} | 未达标 {c['fail']} | 无法判断 {c['unknown']}")
    print(f"单文件 {os.path.getsize(OUT_SINGLE) / 1e6:.2f} MB -> {os.path.normpath(OUT_SINGLE)}")
    print(f"开发版 -> {DIST}")
    print(f"数据指纹 {data['meta']['build']['data_sha256'][:16]}… | 题库指纹 {data['meta']['build']['gold_sha256'][:16]}… | 构建于 {data['meta']['build']['built_at']}")


main()
