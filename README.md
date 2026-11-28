# TimeGuard-RAG

TimeGuard 用财报回答带有公司、期间和披露日约束的问题，并记录回答所引用的原文。RAGFlow v0.27.3 提供 PDF 解析、索引、混合检索与原生问答。TimeGuard 实现财报范围限定、证据回读、缺项补搜、财务计算核验，以及用于比较两条问答流程的 FinanceBench 配对评测。

## 从一个问题读起

| 想看什么 | 从哪里开始 |
| --- | --- |
| 一个请求怎样变成答案 | [会话 API](backend/timeguard/server/app.py) 接收请求，[财报流程](backend/timeguard/research/financial_agent.py) 组织取证、核验、计算与缺参追问。 |
| 哪些财报能参与检索 | [披露日期目录](backend/timeguard/retrieval/catalog.py) 校验原件身份，[范围判定](backend/timeguard/retrieval/scope.py) 从题干和截止日限定公司与文档。 |
| 原生研究结束后怎样补证据 | [增强入口](backend/timeguard/research/live_finance.py) 检查缺项、回读已检索原块，必要时在同一范围补搜一次。[事实筛选](backend/timeguard/evidence/fact_selection.py) 按行验证期间、单位和口径。 |
| 回答里的引用指向哪段原文 | [证据记录](backend/timeguard/evidence/evidence_layers.py) 跟踪索引候选、送入模型的文本与回答引用。 |
| 一份财报如何进入检索库 | [数据准备](backend/timeguard/indexing/data.py)、[入库与接续](backend/timeguard/indexing/ingest.py)、[索引核验](backend/timeguard/indexing/audit_index.py)。 |
| 如何对比原生参照和增强版 | [配对入口](backend/timeguard/evals/financebench_paired.py) 按冻结顺序逐题运行，[独立裁判](backend/timeguard/evals/judge.py) 在作答结束后读取参考答案和实际引用。 |

会话 API 和 FinanceBench 配对评测是两个入口。配对评测启动独立工作进程，直接调用固定版本 RAGFlow 的原生研究图。

## 目录

| 路径 | 内容 |
| --- | --- |
| [backend/timeguard/](backend/timeguard/) | 在线服务、入库、检索、研究流程和证据模块。模型协议在 `llm/`，环境准备在 `operations/`。 |
| [backend/timeguard/evals/](backend/timeguard/evals/) | 配对调度、评分与人工复核。原生基线在 `baselines/`，组件对照在 `trials/`，汇总和诊断代码在 `analysis/`。 |
| [web/trace/](web/trace/) | 离线逐题回放的前端、导出和构建脚本。 |
| [tests/](tests/) | 后端回归测试。模型桥的协议测试随 [model_bridge/](model_bridge/) 维护。 |
| [deploy/](deploy/) 和 [configs/](configs/) | 容器、GPU 构建依赖锁、解析配置、披露日期和 FinanceBench 冻结排期。上游提交见 [版本锁](upstream.lock.json)。 |
| [scripts/](scripts/) | 解析运维、访问转发和独立重放脚本。 |
| [timeguard/](timeguard/) | 从仓库根目录运行 Python 的入口，兼容旧版 `timeguard.<模块>` 导入和命令。 |

## 开发与验证

在仓库根目录执行。安装应用依赖可以覆盖 FastAPI 与 LangGraph 相关测试。依赖和应用镜像按 Python 3.12 配置。

~~~bash
python -m pip install -r requirements-app.txt
python -m unittest discover -s tests -v
python -m unittest discover -s model_bridge -v
python -m timeguard.evals.financebench_paired --help
~~~

有工程数据的环境可以继续按[部署与入库](docs/部署与入库.md)、[配对评测](docs/配对评测.md)或[逐题回放说明](web/trace/README.md)操作。从之前的版本升级时，为新代码创建新的运行目录。评测会校验源码、索引和数据的指纹。

实验汇总和逐轮分析存入本地 `reports/`，逐题轨迹存入 `runs/`。这两个目录都由 Git 忽略。仓库提供评分实现、排期和部署输入。此前公开的冻结版本仍可在 Git 历史中查阅。

目录划分参考了 Onyx 按能力组织后端、将前端和部署单独放置的做法。[Onyx 后端源码](https://github.com/onyx-dot-app/onyx/tree/main/backend/onyx) · [固定版本的 RAGFlow](https://github.com/infiniflow/ragflow/tree/v0.27.3)
