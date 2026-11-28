# FinanceBench 逐题回放

页面按题号并排展示原生参照和增强版，沿检索、证据送达、回答到评分查看记录。点击回答中的引用编号可以跳到相应的证据块。

在服务器项目根目录执行，把示例路径换成要回放的已完成配对目录。导出器读取两组的作答和评分，输出供回放使用的 JSON。

~~~bash
FINANCEBENCH_RUN=runs/financebench-openai-20261125-v2 python3 web/trace/export_trace.py > web/trace/data.json
python3 web/trace/build.py
node web/trace/test.js
~~~

构建器默认读仓库根目录的 `data/financebench/gold.jsonl`，也可用 `FINANCEBENCH_GOLD` 指定。使用与该次运行一致的评分文件。打开 `web/TimeGuard-RAG-逐题回放.html` 即可逐题查看。导出的 JSON、分文件构建目录和离线 HTML 都保存在本地。
