# FinanceBench 官方 API 配对实验

通过 `--model-provider openai` 选择官方 Responses API。原有命令及新配对入口默认用 `codex`，对应专用 Codex 桥。官方路径仅替换独立评测容器进程内发往已登记模型入口的请求。RAGFlow v0.27.3 仍执行检索、研究工具和最终作答，保留原生的 DeepDOC、BGE 与 Elasticsearch 索引。

两组作答和研究均使用 `gpt-6.1-sol/low`，盲判使用 `gpt-6.1-sol/medium`。官方传输保留同题内的工具调用结果和加密续轮状态，关闭响应存储。独立裁判一次只看一个答案，标准答案、标准证据和该答案已匹配的实际引用仅进入评分。

从 `OPENAI_API_KEY` 或 `~/.config/timeguard/openai-api-key` 读取官方密钥。也可以用 `TIMEGUARD_OPENAI_KEY_FILE` 指定私有文件。在 Linux 上限制文件权限为 `0600`。

## 执行

在服务器已有 FinanceBench 数据与索引的环境中，加载项目的 `RAGFLOW_API_KEY`，准备对应原生对话的 `setup.json` 和逐题排期。运行二十个程序位置时，使用已核验的 3M 年报表格和上一轮真实模型连通验证的单题记录。程序位置通过离线回放、PDF 和原生检索验证，无新增生成或裁判请求。

```bash
python -m timeguard.financebench_checks \
  --parent-table runs/r1-table-3m/parents.jsonl \
  --trace runs/ragflow-official-smoke/combined-final-openai-navfixed-dev/results.jsonl \
  --output runs/financebench-checks.json

python -m timeguard.financebench_paired \
  --schedule "$TIMEGUARD_SCHEDULE" --setup "$TIMEGUARD_SETUP" \
  --output runs/financebench-openai --program-report runs/financebench-checks.json \
  --model-provider openai --stage all
```

`--stage plan` 只验证 150 对排期。默认 `--stage pilot` 只跑开发组 6 对。正式档 `--stage all` 在首批通过后继续，保持逐题配置一致，并计入已通过的首批位置。每题从独立空会话开始，按排期交替两版先后，并发默认两对。A0 保留原生证据选择。A1 从当题实收证据识别缺项，复查已检索候选，必要时在同范围补搜一次。候选原文核验通过后最多选 4 块，无法完整送达时回退到本轮原生选块。Decimal 计算要求操作数、单位与期间均可回读核验。

同一输出目录按冻结签名续跑。程序会保留已启动位置、所有生成尝试、裁判记录和运行摘要。成功的生成位置不会重跑，未成功的生成位置最多追加一次尝试。裁判中断或未知单列。汇总按原始分母和历史分组报告答案正确、引用支持、联合达标和配对翻转。独立模型裁判与人工盲核分别保留。逐题输出、财报和凭据属于服务器私有运行材料。

## 2026 年 11 月 25 日接线复验

固定源码提交为 `78fdd2c`。219 项离线测试及服务器 F01–F20 全部通过。首批 6 对全部作答并评分，其中一对两组都请求澄清，因此产生 10 次独立模型裁判。作答与研究共记录 153 次官方请求尝试，其中两次传输异常已保留，同题仍取得有效结果。

| 指标 | A0 原生参照 | A1 财报增强 |
| --- | ---: | ---: |
| 答案正确 | 1/6 | 3/6 |
| 实际引用支持 | 2/6 | 4/6 |
| 答对且实际引用支持 | 1/6 | 3/6 |

有 2 对联合指标从未达标变为达标，0 对反向变化。01079 的增强版从空会话跑完缺项识别、旧池回读、一次同范围补搜、事实筛选和最终送达。第一轮验收曾遇到原生研究摘要与索引正文不一致。修复后先按同一报告和原块编号回读正文，把原生表格视图与同版渲染结果核对，同时保留模型实收片段交给裁判。

这 6 对计入全量。服务器已按同一冻结配置启动其余 288 个答题位置。可在私有运行目录 `runs/financebench-openai-20261125-v2` 查看 `manifest.json`、`pilot.json` 和后续 `summary.json`。首批摘要的 SHA-256 为 `209b5a966e7e4493f825ecae0d46de195246a9edd4f77f566647ec90d262c834`。
