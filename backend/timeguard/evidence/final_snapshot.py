"""在最终选块前封存输入，并调用固定版本的原生函数回放。"""
import asyncio
import hashlib
import json
from types import SimpleNamespace


BRIEF_RULE = "按问题的语言简洁回答。只列出问题要求的结论、必要操作数及公式，每项关键事实紧跟其实际证据编号。区分期间、部门与合并口径、百分比与百分点，不加入无来源的数字。"


def encode(value):
    # 优先引用按字典插入顺序遍历，不能对键排序。
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def freeze(state, tools, answer_conf, identity):
    fields = ("question", "kbinfos", "partial_answer", "abstain", "empty_result", "slot_evidence")
    config = {name: getattr(tools, name, None) for name in
              ("doc_scope", "kb_ids", "empty_response", "user_defined_prompts", "system_prompt", "_bound_dataset_names")}
    value = {"version": 1, "state": {k: state[k] for k in fields if k in state},
             "tools": config, "max_length": tools.chat_mdl.max_length,
             "identity": identity, "answer_conf": answer_conf}
    # 立即断开与原图可变状态的引用。
    return {"input": json.loads(encode(value)), "sha256": fingerprint(value)}


async def replay(snapshot, compose, model):
    value = snapshot["input"]
    if fingerprint(value) != snapshot["sha256"] or value["version"] != 1:
        raise ValueError("最终节点快照版本或指纹不匹配")
    value = json.loads(encode(value))
    if model.max_length != value["max_length"]:
        raise ValueError("回放模型长度不同")
    tools = SimpleNamespace(**value["tools"], chat_mdl=model)
    return await compose(value["state"], tools, asyncio.Queue(), value["answer_conf"])


class RequestCaptured(BaseException):
    """在专用桥发包前退出回放，避免上游捕获普通异常后重试。"""
