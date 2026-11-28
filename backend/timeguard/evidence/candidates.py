"""证据块身份、展示格式和模型选择结果的校验。"""


def identity(chunk):
    return chunk["document_id"], chunk["id"]


def display(chunks):
    return [{"编号":i,"报告":c["document_name"],"位置":c["positions"],"正文":c["content"]}
            for i,c in enumerate(chunks,1)]


def pick(reply, count, adaptive=False):
    ids = reply["selected"]
    minimum, maximum = (1, min(count, 12)) if adaptive else (4, 4)
    if (not isinstance(ids,list) or not minimum <= len(ids) <= maximum
            or any(type(i)!=int or not 1<=i<=count for i in ids) or len(set(ids))!=len(ids)):
        raise ValueError("选择数量、编号范围或去重检查失败")
    return ids


def gaps(reply):
    missing = reply["missing"]
    if (not isinstance(missing,list) or len(missing)>3 or
            any(not isinstance(m,str) or not 4<=len(m.strip())<=350 for m in missing)
            or len(set(missing))!=len(missing)):
        raise ValueError("缺项必须为至多三个不同的具体事实")
    return missing
