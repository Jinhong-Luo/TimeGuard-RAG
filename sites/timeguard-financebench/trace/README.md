# FinanceBench 逐题回放

页面使用同一批冻结作答和评分记录，按题号对照原生参照与增强版。本题导读展示答题输入、模型实际走过的步骤、答案与引用各自的判定。回答中的引用可以跳到证据块。

在本地准备只读导出脚本生成的 `data.json`，放到本目录。通过环境变量 `FINANCEBENCH_GOLD` 指定对应版本的 `gold.jsonl`。然后依次运行 `python build.py` 和 `node test.js`，双击上一级目录的 `TimeGuard-RAG-逐题回放.html` 即可使用。题库与运行记录仅在本地构建时读取。
