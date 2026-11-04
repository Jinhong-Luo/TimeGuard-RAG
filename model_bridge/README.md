# Codex 登录桥接

这个标准库 Python 桥接使用当前机器的 Codex ChatGPT 登录，提供本机 `/v1/chat/completions` 接口，供两个项目进行受控实验。它不读取或复制 `auth.json`，也不导出 OAuth 令牌。

实验标识固定为 `codex-app-server-structured-output-v1`。**这是 Codex 结构化输出传输变体，不能写成原生模型 API 基线。** τ-bench 的 LLMAgent、工具执行、用户模拟和评分器仍可复用，但消息历史通过 JSON 包装传入 Codex，工具调用通过结构化文本返回，提示格式和运行时与 API 不同。RAGFlow 通过同一接口生成答案，也必须记录这个变体。

## 启动与预检

需要 Python 3.11+，PATH 中有已登录的 `codex`。服务器可先执行：

```bash
export PATH="$HOME/.nvm/versions/node/v24.21.0/bin:$PATH"
python3 codex_bridge.py --check --timeout 45
python3 codex_bridge.py --port 18089 --runtime-dir /home/luojinghong/LLM_Proj/.bridge-runtime
```

`--check` 只初始化 app-server、读取模型目录、创建无推理的临时线程，检查返回的模型、思考强度和指令来源。它不证明模型推理可用。必须再进行实际的一条普通回答和一条虚拟业务工具请求，检查返回值，才能开始真实题库。

接口地址为 `http://127.0.0.1:18089/v1`。桥接默认仅监听回环地址；如设置 `CODEX_BRIDGE_TOKEN`，调用者必须提供相同的 Bearer 值。未设置时，客户端可使用任意本地占位 API Key，该值不是 OpenAI 密钥。

RAGFlow 在容器内运行时，容器的 `127.0.0.1` 指向容器自身。可新启一个只绑定实际 Docker bridge 网关的桥接进程，例如 `--host 172.18.0.1 --port 18863`，再让容器访问 `http://172.18.0.1:18863/v1`；地址须以本机实际网络为准。非回环启动必须设置 `CODEX_BRIDGE_TOKEN`，脚本会通过 `docker network inspect` 核验指定地址确为当前 bridge 网关。禁止 `0.0.0.0`、公网地址和无法核验的内网地址。不要将访问令牌写入 Git。

默认只并发一个请求。每次请求启动独立 app-server 和临时线程，防止题目、用户模拟器和裁判共享历史。启动成本计入端到端耗时。

## 模型与思考强度

只允许 `gpt-6.1-sol`。六个待测档位为 `low`、`medium`、`high`、`xhigh`、`max`、`ultra`，默认 `low`。支持请求字段 `reasoning_effort` 或模型别名，例如 `gpt-6.1-sol@high`；两者冲突时报错。每次线程启动都核对实际模型名和实际思考强度，不允许静默切换。

本机 Codex 原模型缓存记录六个档位；独立 app-server 的目录可能暂不列出该模型。目录未列出时保留请求的模型名进行实际调用，由服务端明确接受或拒绝，绝不回退到 GPT-6-Sol。`/v1/models` 列的是桥接配置的候选别名，不是远端授权证明。

`ultra` 的 Codex 产品说明包含自动任务委派。本实验关闭多代理工具，因此必须写明“ultra，委派关闭”，不能声称测到了 Codex 完整自动委派模式。

## 隔离与评测约束

- `thread/start` 和 `turn/start` 均传入 `environments: []`，协议规定这会关闭环境访问；线程没有动态执行工具。
- 禁用 shell、图像查看、浏览器、计算机控制、Web 搜索、应用、插件、MCP、技能、记忆、钩子和多代理，使用只读权限与 `approvalPolicy=never`。
- 原用户配置只读取 MCP/插件名称以生成关闭参数；任何实际服务端工具请求或非文本动作都会使该调用失败。桥接不执行任何业务工具。
- 每次模型只能返回一个业务动作；工具名必须在调用者提供的函数集合中，参数必须能解析为 JSON 对象。业务参数的完整语义验证仍由原工具负责。
- 运行目录只包含临时空目录，不放题目答案。允许已知的用户全局 `~/.codex/AGENTS.md`，预检记录其 SHA256；其他指令文件会触发拒绝。评测前须确认该全局文件不含题库、答案或调优策略。
- 模型消息只包含调用者显式提供的 `messages`、`tools`、`tool_choice` 和 `response_format`。调用者必须隔离标准答案；裁判输入包含答案是评分用途，不可复用到解题模型。
- 这个 HTTP 服务不提供 embeddings。RAGFlow 仍须另外配置实际可用的 Embedding 模型。

`temperature`、`seed`、`max_tokens` 等采样参数没有对应的 app-server 等价控制，不能宣称已应用。响应元数据写明 `sampling_parameters_applied=false`。输出采用固定一步、超时限制；支持完整返回后转换的 SSE，不是上游逐 token 实时流。

如果 Codex 报告 token 使用量，按原值返回；若没有则不伪造。订阅额度没有逐请求账单价格，`billed_cost=null`，不可把零元或 LiteLLM 推算价当成实际费用。

## 检查

```bash
python3 -m unittest discover -s . -p 'test_codex_bridge.py' -v
```

测试覆盖思考强度冲突、未知模型、工具白名单、参数格式、禁止并行动作、JSON 裁判输出、禁止文件/图片输入、拒绝运行时工具请求、环境隔离参数和指令来源。这些测试不进行任何模型调用。

真实烟测及逐档记录：

```bash
python3 probe.py --base-url http://127.0.0.1:18861/v1 --effort low --output ../reports/bridge-smoke-low.json
python3 probe.py --base-url http://127.0.0.1:18861/v1 --all-efforts --output ../reports/bridge-smoke-all.json
```

每档默认包含普通回答、虚拟订单工具和 JSON 裁判三个请求；使用 `--cases tool` 可只测工具格式。探测脚本不执行工具，默认仅测 low。报告保存每条请求的输出、模型名、实际强度元数据、token、耗时和错误分类，每次完成请求即落盘。烟测通过只能证明接入链路和输出格式可用，不能替代业务基线或选优。

2026-11-04 初次服务器调用曾遇到 `workspace routing discovery failed`；经出口代理后错误明确为 `workspace routing discovery unauthorized (401)`，`account/read` 返回 `requiresOpenaiAuth=true`。这属于登录认证阻塞，应在服务器重新完成 Codex 登录后重试；不能记为模型解题失败，也不能填入成功率或最佳档位。模型目录与线程预检成功不代表登录凭据仍可用于推理。

协议依据：[Codex App Server 官方文档](https://developers.openai.com/codex/app-server/)、[配置参考](https://developers.openai.com/codex/config-reference/)，并以安装的 `codex app-server generate-json-schema --experimental` 输出核对字段。协议属于实验接口，升级 Codex 后应重新预检。
