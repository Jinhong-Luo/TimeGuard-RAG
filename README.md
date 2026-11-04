# TimeGuard-RAG

企业披露资料检索、证据核验与持续评测项目。底座固定 **RAGFlow v0.27.3**；当前交付为可复现的 R0 数据、部署和独立评测工具，尚不宣称已经取得端到端基线成绩。

## 当前范围

- 复用上游 Python/Quart 服务、原生 PDF 解析、分块、Embedding、Elasticsearch、混合检索、问答和引用。
- 新增 FinanceBench 固定版本下载与哈希核验、按公司分组、原始文档台账、独立运行记录、零召回保留分母和人工复核流程。
- 基线运行器始终只把问题传给 RAGFlow。标准答案、证据页和金标文档名称不会用于逐题检索限制，也不会作为模型上下文。
- 财报分块二开、实体和时间约束、中文年报、GEPA、自优化尚未实现，不计作本阶段成果。

## 环境和固定版本

评测工具只需要 Python 3.10+ 标准库。部署使用 Linux、Git、Docker Engine 和支持 `include` 的 Docker Compose v2。实际服务以 `upstream.lock.json` 中的提交与镜像为准。

| 组件 | 固定值 |
|---|---|
| RAGFlow | `v0.27.3` / `8090ffd8ba6098813d7f7ec2c4b8774f36c8423a` |
| FinanceBench | `cc39aeb4afdf33909ee1412188bf89035950c2eb` |
| 本地英文 Embedding | `BAAI/bge-small-en-v1.5` / `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| TEI CPU 镜像 | 上游指定的 `infiniflow/text-embeddings-inference:cpu-1.8` |
| 基础生成模型 | `gpt-6.1-sol`，以实际可调用入口返回的标识为准 |

镜像 tag 下载后还应记录 `RepoDigests`；源码固定不能替代镜像内容校验。使用 Codex 登录桥接时应在运行配置和报告中注明 `model_transport`，与原生商用 API 的结果分开，不声称两者完全等价。

## 部署与数据

在 `/home/luojinghong/LLM_Proj/TimeGuard-RAG` 执行：

```bash
python3 -m timeguard.bootstrap --prepare-compose --local-embedding
docker compose -p timeguard_ljh -f deploy/compose.generated.json config --quiet
docker compose -p timeguard_ljh -f deploy/compose.generated.json up -d
python3 -m timeguard.data --pdfs
```

生成配置复用上游原生 CPU 服务，项目名为 `timeguard_ljh`。只在回环地址开放网页 `18080` 和 API `19380`，数据库没有宿主机端口。生成前请确认这两个端口空闲；需要换端口时修改生成配置并记录。所有持久数据放在本项目 `data/services`，模型缓存放在 `data/model-cache`。Elasticsearch 使用当前用户 UID；其他原生镜像按其入口初始化数据目录，不设置全局 `777`。每个服务有内存、CPU和日志轮换限制。只对该项目执行启停，不操作其他用户容器。

配置生成在 `deploy/runtime.env` 内创建随机服务密码，并限制文件权限；这两个生成文件都被 Git 忽略。可用磁盘不足时先停止拉取，不清理别人的镜像或卷。官方镜像与原始财报下载需要网络，启动成功不等于模型权重和解析器资源已准备完毕。

通过 SSH 转发后访问本机网页，完成用户注册、获取 RAGFlow API key，并配置生成模型提供商。`--local-embedding` 启用 RAGFlow 原生 TEI 服务，固定英文轻量模型，不依赖付费 Embedding API。中文扩展需另立多语 Embedding 实验，不能沿用本次英文结果代表中文能力。

```bash
export RAGFLOW_API_KEY='在本机环境中设置实际值'
python3 -m timeguard.ingest --embedding-model 'BAAI/bge-small-en-v1.5@Builtin'
python3 -m timeguard.ingest --embedding-model 'BAAI/bge-small-en-v1.5@Builtin' --status
```

`--embedding-model` 必须与服务实际列出的完整标识一致。全部文档解析成功且分块数大于零才运行基线。导入中断后可续跑已保存步骤；极端情况下上传完成、导入记录尚未落盘之间中断，可能残留重复文档，需要先核对文档列表，不能称为严格事务恢复。

真实数据准备结果：150 题、84 份关联文档、32 家公司；按公司固定分成调优 88 题、开发 39 题、保留测试 23 题。公开完整 150 题可以单独报告，不能称为独立泛化成绩。`queries.jsonl` 只有题号、问题和集合；`gold.jsonl` 在离线评分阶段使用。PDF 来源、报告期和哈希保存在 `documents.jsonl`。

## 基线与强度比较

先修改 `configs/r0.json` 的实际模型名和服务地址。`dataset_ids` 为空时从导入记录读取；每次运行保存完整固定配置、数据哈希、模型、Embedding、上游版本和题目清单。

```bash
python3 -m timeguard.run --split dev --limit 3 --retrieval-only --output runs/r0-retrieval-smoke
python3 -m timeguard.score runs/r0-retrieval-smoke
python3 -m timeguard.run --split dev --limit 3 --output runs/r0-default-smoke
python3 -m timeguard.run --split dev --output runs/r0-default-dev
python3 -m timeguard.score runs/r0-default-dev
```

检索测试调用原生 `/api/v1/retrieval`；问答调用原生 `/api/v1/chat/completions`，不传 `session_id`，使每题建立独立会话。单独检索和原生问答各发生一次召回，报告分别保存诊断召回与真实回答引用，两者不能混称为同一次模型输入。请求失败、无答案和零召回均记录；续跑跳过已经记录的失败样本，避免悄悄重试直到成功。中断实验的未运行样本也保留在计划分母中。

思考强度候选为 `low medium high xhigh max ultra`。具体入口可能只支持其中一部分，先探测，再完整运行支持的档位；不能把不支持的档位自动降级成另一档。

```bash
python3 -m timeguard.probe --base-url http://127.0.0.1:18861/v1 --model gpt-6.1-sol
```

该探测读取 `OPENAI_API_KEY`。若桥接入口使用本地占位凭证，应按桥接服务说明配置，不把 ChatGPT 登录凭证复制进仓库。`accepted` 只说明接口接受请求，不能证明 RAGFlow 实际传递或模型实际采用该强度。

**上游兼容性已核查：** RAGFlow 的 `Base._clean_conf` 会过滤 `reasoning_effort`，其 LiteLLM 分支有不同的允许列表。`prompt_config.reasoning` 也是另一个功能，不是模型的思考强度。必须选择实际支持的提供商路径、透明记录最小传输补丁，或使用经验证的模型别名 `gpt-6.1-sol@high`，再从模型入口日志核对最终请求。不能仅以保存聊天配置成功为依据。

传递确认文件结构如下，只有真实验证后才能填写：

```json
{
  "model": "gpt-6.1-sol",
  "verified_efforts": ["low", "medium"],
  "verification": "填写实际日志路径、请求编号、验证时间和适配方式"
}
```

确认之后运行各档（路径中的档位同配置一致）：

```bash
python3 -m timeguard.run --split dev --effort low --transport-proof runs/transport-proof.json --output runs/r0-low-dev
python3 -m timeguard.score runs/r0-low-dev
```

使用别名时保存每档真实别名和共同基础模型，报告其传输方式；当前 `select` 严格要求配置除 effort 外相同，因此别名比较应先做一致性审计，不能强行跳过检查。

## 评分与选优

- `page_hit_recall_at_k_proxy`：标准证据页在前 k 个块中的命中比例，按题平均。FinanceBench 从 0 计页，RAGFlow 从 1 计页；同时核对文档名。
- `strict_text_recall_at_k_proxy`：标准证据全文归一化后在同文档召回内容中完整出现的比例。分块、OCR和表格格式可能导致假阴性。
- 上述两项是可复算代理指标，**不能称为已验证的语义证据 Recall@k 或引用支持率**。精确语义支持需要人工核对，或冻结模型裁判并抽查。
- `review_template.jsonl` 提供独立人工复核入口：答案正确性、数值与口径、引用支持及失败分类。空值保留；部分复核不会产生完整准确率。
- 原生接口不返回可靠费用时，`cost_usd` 保留为 `null`。检索诊断额外调用也计入实际总耗时；最终成本应补充网关账单与 Embedding/解析费用。

完成独立人工复核后：

```bash
python3 -m timeguard.score runs/r0-low-dev --reviews runs/r0-low-dev/reviews.jsonl
python3 -m timeguard.select runs/r0-low-dev runs/r0-medium-dev
```

选优规则预先固定：在同一开发集上，答案正确率距最好档不超过 2 个百分点时选平均耗时最短的一档；费用未知时不宣称最省钱。完成档位选择后冻结配置，再在保留测试集运行一次。开发集调参、最终保留评估和公开全量成绩分别报告。

## 校验与问题记录

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q timeguard
```

目前 9 项离线测试通过，覆盖公司隔离、问题与金标分离、零召回/失败/未完成样本留分母、页码基准、重复样本拒绝及人工复核完整性。实际模型调用与服务运行状态以主机报告为准。问题与处理记录见 [实现记录](docs/实现记录.md)。

## 核查依据

- [RAGFlow 固定版本](https://github.com/infiniflow/ragflow/tree/v0.27.3)、[原生 HTTP API](https://github.com/infiniflow/ragflow/blob/v0.27.3/docs/references/http_api_reference.md)。
- [原 benchmark 删除零召回查询的行为](https://github.com/infiniflow/ragflow/blob/v0.27.3/rag/benchmark.py)、[页码写入函数](https://github.com/infiniflow/ragflow/blob/v0.27.3/rag/nlp/__init__.py)。
- [模型参数过滤](https://github.com/infiniflow/ragflow/blob/v0.27.3/rag/llm/chat_model.py)、[内置 Embedding](https://github.com/infiniflow/ragflow/blob/v0.27.3/rag/llm/embedding_model.py)、[官方部署环境变量](https://github.com/infiniflow/ragflow/blob/v0.27.3/docker/.env)。
- [FinanceBench 数据与许可证](https://github.com/patronus-ai/financebench/tree/cc39aeb4afdf33909ee1412188bf89035950c2eb)、[BGE 模型卡](https://huggingface.co/BAAI/bge-small-en-v1.5)、[TEI 官方说明](https://github.com/huggingface/text-embeddings-inference)。

上游源码与原始数据使用各自许可证。本仓库保留其固定版本和来源，不把复用能力包装成个人从零实现。
