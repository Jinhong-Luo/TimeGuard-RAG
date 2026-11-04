# GPU 设备访问异常与精确任务恢复

2026-11-04 23:23 起，为增加原生执行器进程数，对本项目运行中的 GPU 容器执行了 `docker update --memory 64g`。随后新增的四个执行器均进入 CPU 分支；原有八个执行器持有已有 GPU 句柄，仍在处理任务。这次在线扩容方法不适用于当前宿主机的 NVIDIA 容器运行路径，已停用 `scripts/add_parse_worker.py`。

只读对照实验中，无论使用默认环境，还是补齐原生入口的库路径、jemalloc 和 GPU 可见设备环境，新进程打开三个 NVIDIA 设备都得到 `EPERM`，PyTorch 枚举到零张卡，NVML 返回 `Unknown Error`。这与 [NVIDIA 官方记录的容器更新后 GPU cgroup 访问丢失问题](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html#containers-losing-access-to-gpus-with-error-failed-to-initialize-nvml-unknown-error)一致。项目没有修改系统驱动、Docker 配置、cgroup 或其他用户容器。

恢复采用以下边界：

1. 先暂停四个错误设备进程，保留进程现场和原日志；再暂停本项目全部执行器，保存 Task、Redis 消费者、消息和原生累计块数快照。
2. 唯一已经进入后期的页段单独继续 5.7 秒，正常完成并 ACK 后再暂停该执行器。最终保留 134 个成功页段，剩余 11 条待确认消息。
3. 对 11 条消息逐项检查：任务进度低于 0.7、`chunk_ids` 为空、没有完成记账键，当前任务日志没有构块后 Embedding、索引或进度写入失败标记。这里的结论只适用于待恢复任务，不能外推为执行器整个历史都没有提交；后续完整日志审计发现一个更早完成的 CPU 页段，见下文更正。
4. 仅重建本项目主服务，启动时固定 16 核、64GiB、单张 GPU、12 个执行器，每进程解析和分块各 1 个任务。持久数据库、对象存储、索引和队列不重建。
5. 新执行器逐个完成真实 CUDA 分配、运算与原生四模型会话检查，然后等待恢复门闩。原 0—7 消费者名称保留，只对四条额外消费者消息执行精确 `XCLAIM`，消息 ID 和内容不变。其余七条所有者不变，不新增消息、不重置重试次数、不重跑整份文档。

`reports/gpu_access_after_memory_update.json` 保存设备探针，`gpu_recovery_boundaries.json` 保存静止任务边界，`gpu_recovery_logs_archive.json` 保存原日志摘要，`gpu_recovery_claims.json` 保存恢复时全部执行器的真实预检及消息所有者转移。只有相应报告实际生成成功，才能视为该步骤已经完成；本文件不是完整 84 份索引完成证明。

独立完整日志复核发现，`extra_08` 曾在 UTC 15:27:43 用 CPU 完成 `ADOBE_2022_10K` 的第13–24页（内部页段 `[12,24)`）。对应 Task 有38个唯一文字块，REST逐项确认38/38在库且有效。此前根据当前待恢复任务片段写出的“没有CPU提交”结论错误，现已明确更正并保留原报告来源链。该12页仍使用相同原生算法、权重、分块和默认旋转；为避免破坏共享块ID与累计计数，保留它并在 `execution_hardware_provenance` 中注明混合硬件来源，不冒称全GPU处理。GPU吞吐比较须排除或单列这12页。详见 `reports/cpu_completed_segment_audit.json`。

UTC 16:22:49 的独立检查进一步确认：134个原完成Task的全部冻结字段仍相同，11条恢复消息已全部完成并ACK、真实领取次数均为2，消息内容没有改写。后续六档实验共用同一个完成后冻结的索引，历史目录名中的`gpu`不构成逐Task硬件验证。

## 启动守卫

`scripts/bin/python3` 只在原生 `rag/svr/task_executor.py` 入口前插入 `gpu_worker_guard.py`，其他调用使用原解释器。守卫与原生任务在同一进程中运行，使用相同原生模型缓存键，不改 RAGFlow 源码、模型、分块或表格自动旋转。预检加载用项目内文件锁串行进行；实际任务执行不持有该锁。

每个执行器必须看到且只看到一张 GPU，并完成实际 CUDA 运算。原生 det、rec、layout、tsr 会话的首选 provider 必须是 CUDA；det 额外运行一次 64×64 合成输入。该输入只验证设备链路，不作为解析质量或速度成绩。默认带旋转的语义按固定源码核验；关闭旋转会拒绝启动。

恢复时使用 `--hold-workers` 生成门闩配置。受限 `data/runtime-control/workers.release` 的内容必须精确匹配本次 `--worker-generation`，否则执行器不进入原生任务入口。恢复脚本还必须核对 12 个证据文件的代次、PID、消费者参数和等待阶段，不能只看旧的 ready 文件。不要在任务还活跃时随意删除门闩或改变消费者名称；不要对运行中的 GPU 容器再次执行资源更新。

## 线程参数的实际边界

固定上游创建了 `SessionOptions` 并请求 intra/inter 线程各 2，但传给 ORT 的参数名是 `options`，已安装 ORT 1.23.2 的对应参数名为 `sess_options`。原生会话实际值为 0，即自动线程设置。守卫同时记录请求值和每个实际会话值；本项目没有修补这个上游行为。CUDA provider 的显存上限、arena 策略与执行后的收缩选项另行记录，不能用线程请求值替代实际会话证据。
