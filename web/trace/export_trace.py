"""读取已经完成的 FinanceBench 配对运行，将逐题回放 JSON 输出到标准输出。

可用 FINANCEBENCH_RUN 指向其他冻结运行目录。诊断信息写入标准错误。
"""
import collections
import datetime
import glob
import io
import json
import os
import re
import sys

OUT = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ERR = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
RUN = os.path.abspath(os.environ.get("FINANCEBENCH_RUN") or "runs/financebench-openai-20261125-v2")
BASE = os.path.dirname(os.path.dirname(RUN))
PREVIEW = 260          # characters kept for chunks that never entered the final evidence
VERIFIED = ("intact", "verbatim_excerpt", "native_table_view")
DECIDE = ("复查已检索原文", "补搜后按事实筛选")


def say(*parts):
    ERR.write(" ".join(str(p) for p in parts) + "\n")


def pages(positions):
    return sorted({p[0] for p in (positions or []) if isinstance(p, list) and p and isinstance(p[0], int)})


def load():
    recs = []
    for fp in sorted(glob.glob(os.path.join(RUN, "generation", "*", "results.jsonl"))):
        for line in open(fp, encoding="utf-8", errors="replace"):
            recs.append(json.loads(line))
    jud = {}
    for fp in glob.glob(os.path.join(RUN, "judgements", "*.json")):
        d = json.load(open(fp, encoding="utf-8"))
        jud[d.get("position")] = d
    summary = json.load(open(os.path.join(RUN, "summary.json"), encoding="utf-8"))
    manifest = json.load(open(os.path.join(RUN, "manifest.json"), encoding="utf-8"))
    return recs, jud, summary, manifest


def answers_of(rec):
    return ((rec.get("native_trace") or {}).get("evidence_layers") or {}).get("answers") or []


def gold_path():
    path = os.environ.get("FINANCEBENCH_GOLD") or os.path.join(BASE, "data", "financebench", "gold.jsonl")
    return path if os.path.isfile(path) else None


def main():
    recs, jud, summary, manifest = load()
    say(f"records {len(recs)} | judgements {len(jud)}")
    # model / reasoning effort actually reported by the answering endpoint (not read from any config file)
    answer_models = collections.Counter()
    for rec in recs:
        for e in (rec.get("native_trace") or {}).get("events") or []:
            if e.get("kind") == "openai_request" and e.get("success"):
                answer_models[(e.get("actual_model"), e.get("reasoning_effort"))] += 1
    say("answering endpoint reported:", dict(answer_models))

    # ---------------------------------------------------------------- document names and chunk texts
    docs, texts = {}, {}
    full_ids = set()

    def learn(c):
        if isinstance(c, dict) and c.get("document_id") and c.get("document_name"):
            docs[c["document_id"]] = c["document_name"]

    def keep_text(c, priority, shown=False):
        """priority 3: text as shown to the model; 2: index original from the candidate pool; 1: native reference list."""
        if not isinstance(c, dict) or not c.get("id") or not c.get("content"):
            return
        cid = c["id"]
        cur = texts.get(cid)
        if cur is None or priority > cur["p"]:
            texts[cid] = {"p": priority, "doc": c.get("document_name"), "pages": pages(c.get("positions")), "text": c["content"]}

    for rec in recs:
        t = rec.get("native_trace") or {}
        for a in answers_of(rec):
            full_ids.update(loc.get("id") for loc in (a.get("packet") or []) if loc.get("id"))
    for rec in recs:
        t = rec.get("native_trace") or {}
        for a in answers_of(rec):
            for req in a.get("requests") or []:
                for c in req.get("complete_chunks") or []:
                    learn(c)
                    keep_text(c, 3)
        for c in (rec.get("reference") or {}).get("chunks") or []:
            learn(c)
            keep_text(c, 2)
        for c in rec.get("retrieved_chunks") or []:
            learn(c)
            keep_text(c, 1)
        for c in t.get("retrieved") or []:
            learn(c)
            keep_text(c, 1)
        for x in (t.get("evidence_layers") or {}).get("retrievals") or []:
            learn(x)
            for c in x.get("chunks") or []:
                learn(c)
    chunks = {}
    sizes = collections.Counter()
    for cid, v in texts.items():
        if cid in full_ids:
            chunks[cid] = {"doc": v["doc"], "pages": v["pages"], "full": True, "text": v["text"]}
            sizes["full"] += len(v["text"])
        else:
            chunks[cid] = {"doc": v["doc"], "pages": v["pages"], "full": False, "text": v["text"][:PREVIEW]}
            sizes["preview"] += min(len(v["text"]), PREVIEW)
    say(f"documents {len(docs)} | chunks with text {len(chunks)} (full {len(full_ids & set(texts))}) | text chars {dict(sizes)}")
    missing_text = [i for i in full_ids if i not in texts]
    say(f"evidence chunks without any text: {len(missing_text)}")

    # ---------------------------------------------------------------- one arm of one question
    def arm(rec):
        t = rec.get("native_trace") or {}
        plan = dict(t.get("plan") or {})
        plan["docs"] = [docs.get(d, d) for d in (plan.get("document_ids") or [])]
        plan.pop("document_ids", None)
        events = t.get("events") or []
        layer = [x for x in ((t.get("evidence_layers") or {}).get("retrievals") or []) if x.get("channel") == "retrieval"]
        timeline, pending, li, scope = [], None, 0, None
        calls = searches = 0
        tok_in = tok_out = 0
        for e in events:
            kind = e.get("kind")
            if kind == "bridge_request":
                pending = e
            elif kind == "metadata_filter":
                scope = {"method": (e.get("config") or {}).get("method"), "docs": [docs.get(d, d) for d in (e.get("selected_doc_ids") or [])]}
            elif kind == "agent_tool":
                timeline.append({"t": "tool", "name": e.get("name")})
            elif kind == "openai_request":
                b = pending or {}
                u = e.get("raw_usage") or e.get("usage") or {}
                pin = u.get("input_tokens", u.get("prompt_tokens"))
                pout = u.get("output_tokens", u.get("completion_tokens"))
                calls += 1
                if e.get("success"):
                    tok_in += pin or 0
                    tok_out += pout or 0
                timeline.append({"t": "call", "node": b.get("node"), "phase": b.get("phase"), "round": b.get("search_round"),
                                 "in": pin, "out": pout, "s": round(e.get("elapsed_seconds") or 0, 1),
                                 "tools": e.get("tool_calls"), "ok": bool(e.get("success")), "err": e.get("error_type")})
            elif kind == "retrieval":
                x = layer[li] if li < len(layer) else {}
                li += 1
                searches += 1
                by_doc, ids = {}, []
                for c in x.get("chunks") or []:
                    name = c.get("document_name") or docs.get(c.get("document_id"))
                    ids.append(c.get("id"))
                    d = by_doc.setdefault(name, {"name": name, "n": 0, "pages": set()})
                    d["n"] += 1
                    for p in pages(c.get("positions")):
                        d["pages"].add(p)
                timeline.append({"t": "search", "node": x.get("node"), "round": x.get("search_round"), "q": e.get("question"),
                                 "docs": len(e.get("doc_ids") or []), "n": e.get("chunks"), "s": round(e.get("seconds") or 0, 2),
                                 "by_doc": [{"name": v["name"], "n": v["n"], "pages": sorted(v["pages"])[:16]} for v in by_doc.values()],
                                 "ids": ids})
        ans = answers_of(rec)
        a0 = ans[0] if ans else {}
        hashes, packet = a0.get("block_hashes") or [], a0.get("packet") or []
        ok_reqs = [q for q in (a0.get("requests") or []) if q.get("status") == 200]
        shown = {}
        if len(ok_reqs) == 1:
            for c in ok_reqs[0].get("complete_chunks") or []:
                shown[c.get("block_sha256")] = c
        dup = {h for h, n in collections.Counter(hashes).items() if n > 1}
        blocks, by_identity = [], {}
        for i, h in enumerate(hashes):
            loc = packet[i] if i < len(packet) else {}
            c = shown.get(h)
            item = {"n": i + 1, "id": loc.get("id"), "doc": loc.get("document_name"), "pages": pages(loc.get("positions")),
                    "delivered": c is not None and h not in dup, "check": (c.get("source_check") if c else None),
                    "shown": bool(c)}
            if h in dup:
                item["dup"] = True
            blocks.append(item)
            by_identity[(loc.get("document_id"), loc.get("id"))] = i + 1
        enh = dict(a0.get("enhancement") or {})
        facts = []
        for f in enh.pop("selected_facts", []) or []:
            g = {k: v for k, v in f.items() if k not in ("source", "source_sha256")}
            g["n"] = by_identity.get(tuple(f.get("source") or ()))
            facts.append(g)
        enh["facts"] = facts
        enh["decide_calls"] = sum(1 for s in (enh.get("stages") or []) if s in DECIDE)
        raw = t.get("raw_model_answer")
        return {
            "status": rec.get("answer_status"), "elapsed": round(rec.get("elapsed_seconds") or 0, 1),
            "answer": rec.get("answer") or "", "postprocessed": bool(raw is not None and raw != (rec.get("answer") or "")),
            "plan": plan, "scope": scope,
            "calls": calls, "searches": searches, "tok_in": tok_in, "tok_out": tok_out,
            "timeline": timeline, "enh": enh,
            "pool": [{"id": c.get("id"), "doc": c.get("document_name"), "pages": pages(c.get("positions"))} for c in (a0.get("pool_locations") or [])],
            "blocks": blocks,
        }

    def judge_of(rec):
        d = jud.get(rec["position"]) or {}
        m = d.get("citation_mapping") or {}
        keep = ("answer_correct", "numeric_and_scope_correct", "citation_support", "citation_support_model", "has_factual_claims",
                "judgement_reason", "failure_category", "citation_support_rule", "judge_status", "judge_type", "error", "human_spot_check")
        out = {k: d.get(k) for k in keep}
        out["elapsed"] = round(d.get("elapsed_seconds") or 0, 1)
        out["mapping"] = {k: m.get(k) for k in ("mapping_status", "marker_indexes", "unmapped_indexes", "missing_text_indexes", "malformed_markers", "reference_candidate_count")}
        u = d.get("usage") or {}
        out["tok_in"], out["tok_out"] = u.get("prompt_tokens"), u.get("completion_tokens")
        return out

    questions = {}
    for rec in recs:
        q = questions.setdefault(rec["id"], {"id": rec["id"], "q": rec.get("question"), "split": rec.get("split"), "arms": {}})
        q["arms"][rec["pair_arm"]] = {**arm(rec), "judge": judge_of(rec)}

    # ---------------------------------------------------------------- cross-check against the stored summary
    def joint(d):
        f = (d.get("answer_correct"), d.get("citation_support"))
        if False in f:
            return False
        return True if f == (True, True) else None
    check = {}
    for arm_name in ("A0", "A1"):
        c = collections.Counter(joint(q["arms"].get(arm_name, {}).get("judge", {})) for q in questions.values())
        check[arm_name] = (c[True], c[False], c[None])
    say("recomputed from export (pass, fail, unknown):", check, "| stored:", {a: (summary["all"]["arms"][a]["joint_pass"], 150 - summary["all"]["arms"][a]["joint_pass"] - summary["all"]["arms"][a]["joint_unknown"], summary["all"]["arms"][a]["joint_unknown"]) for a in ("A0", "A1")})

    # ---------------------------------------------------------------- annotated-evidence page base (diagnostic only)
    gp = gold_path()
    if gp:
        def words(s):
            return set(re.findall(r"[a-z]{4,}", (s or "").lower()))
        bydoc = collections.defaultdict(list)
        for v in chunks.values():
            if v["pages"]:
                bydoc[(v["doc"] or "").replace(".pdf", "").casefold()].append((v["pages"], words(v["text"])))
        diff = collections.Counter()
        for line in open(gp, encoding="utf-8"):
            if not line.strip():
                continue
            g = json.loads(line)
            for ev in g.get("evidence") or []:
                tw = words(ev.get("evidence_text"))
                doc = (ev.get("doc_name") or g.get("doc_name") or "").casefold()
                if len(tw) < 8 or not isinstance(ev.get("evidence_page_num"), int):
                    continue
                scored = sorted(((len(tw & cw) / len(tw), pg) for pg, cw in bydoc.get(doc, [])), reverse=True)
                if scored and scored[0][0] >= 0.6:
                    diff[scored[0][1][0] - ev["evidence_page_num"]] += 1
        say("annotated page base: (best-overlap chunk min page - gold page) ->", dict(sorted(diff.items())))

    identity = manifest.get("identity") or {}
    meta = {
        "run": os.path.basename(RUN), "exported_at": datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "commit": None, "budget": identity.get("evidence_budget"), "transport": identity.get("transport"),
        "judge": {k: (identity.get("judge") or {}).get(k) for k in ("model", "effort", "concurrency", "timeout_seconds")},
        "answer_models": [{"model": k[0], "effort": k[1], "calls": v} for k, v in sorted(answer_models.items(), key=lambda kv: -kv[1])],
        "judge_models": [{"model": k[0], "effort": k[1], "judgements": v} for k, v in collections.Counter(
            (d.get("judge_model"), d.get("judge_effort")) for d in jud.values() if d.get("judge_status") == "ok").items()],
        "signature": manifest.get("signature"), "summary": summary,
    }
    try:
        import subprocess
        meta["commit"] = subprocess.run(["git", "-C", BASE, "log", "-1", "--format=%h %ad", "--date=format:%Y-%m-%d %H:%M"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    data = {"meta": meta, "chunks": chunks, "questions": questions}
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    say(f"export {len(text) / 1e6:.2f} MB | questions {len(questions)} | postprocessed answers {sum(1 for q in questions.values() for a in q['arms'].values() if a['postprocessed'])}")
    OUT.write(text)
    OUT.flush()
    ERR.flush()


main()
