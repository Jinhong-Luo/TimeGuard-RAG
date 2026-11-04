# GPU 原生路径核验

核验日期：2026-11-04。最初只读核验确认了固定底座的原生 GPU 路径；随后部署已验证真实 ONNX CUDA 推理，并完成9页PDF解析。派生镜像摘要和实际输出见 `deploy/gpu-image-lock.json`，下载与构建没有修改上游源码或系统驱动。当前正在进一步验证160页长财报；全84份文档索引尚未完成。下文保留最初依赖核查与空间预算，运行中资源变化以 `reports` 的真实记录为准。

部署代理报告的实际环境：RAGFlow v0.27.3，镜像 config `ccc3109ccfd456550a73d8015abc02b265843033267843b422be681143ee21cc`；Python 3.13；`onnxruntime-gpu==1.23.2`，列出 CUDA/TensorRT/CPU provider；没有 torch、triton、任何 nvidia-* Python 发行包，系统库缓存也未见 CUDA/cuDNN/cuBLAS。服务器有 4 张 RTX 3090 24GB，驱动 `595.58.03`。这些服务器状态来自部署代理的只读实查，本报告未另开远程部署。

## 最小可行路径

固定版本的 [pip_install_torch](https://github.com/infiniflow/ragflow/blob/8090ffd8ba6098813d7f7ec2c4b8774f36c8423a/common/misc_utils.py#L249) 在 `DEVICE != cpu` 时安装 `torch>=2.5,<3`，每进程只执行一次；默认 cu121 索引可以用 `TORCH_CUDA_INDEX_URL` 覆盖。原生 [OCR load_model](https://github.com/infiniflow/ragflow/blob/8090ffd8ba6098813d7f7ec2c4b8774f36c8423a/deepdoc/vision/ocr.py#L74) 依赖 `torch.cuda.is_available()` 和设备数，随后才创建 ONNX CUDA 会话。因此仅看到 `get_available_providers()` 列出 CUDA 不足以证明 GPU 可运行。

官方 [PyTorch cu128 索引](https://download.pytorch.org/whl/cu128/torch/) 确有 `torch-2.8.0+cu128-cp313-cp313-manylinux_2_28_x86_64.whl`。固定 `torch==2.8.0+cu128`，只安装 torch 及其依赖，不需要 torchvision/torchaudio。当前 [ORT CUDA 兼容说明](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) 将 1.21–1.26 GPU 包列为 CUDA 12.8、cuDNN 9，选择 cu128 可避免 CUDA 次版本过旧。运行库由 Python 依赖提供，不需要在宿主机另装完整 CUDA Toolkit。

部署时建议使用原容器 `/ragflow/.venv/bin/python`，将 `DEVICE=gpu`、`TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu128` 固定下来；先单独完成固定包安装，再启动任务执行器，避免多个进程同时首次写入虚拟环境。可通过 `PIP_CONSTRAINT` 指向仅含 `torch==2.8.0+cu128` 的受控约束文件，让原生运行时安装分支也保持版本。安装示意如下，**本核验没有执行**：

```bash
/ragflow/.venv/bin/python -m pip install 'torch==2.8.0+cu128' \
  --index-url https://download.pytorch.org/whl/cu128
```

不使用 `--no-deps` 跳过缺失的 CUDA 库。保留固定基础镜像摘要，额外运行时包版本和 wheel SHA 单独登记；不要把安装后的可写容器层误称为未变的运行环境。下载缓存与重建后的恢复方式由部署配置明确管理。

精确 [安装约束](../reports/gpu-install-constraints.txt) 已列出16个新包及镜像已有的七个小依赖和 ORT。下载先按锁清单验证 SHA256，再在项目专用 builder 容器内用只读 wheelhouse、`--no-index --find-links=/wheelhouse --no-cache-dir -c /path/to/gpu-install-constraints.txt` 安装上述固定 torch；安装时保留依赖解析，不用 `--no-deps`。派生镜像运行时也设置 `PIP_CONSTRAINT` 指向同一文件，原生 `torch>=2.5,<3` 分支将保持已安装的版本。不要把只读 wheelhouse 整体 COPY 入最终镜像层。

## 下载体积

通过官方 torch wheel 的 HEAD、其 PEP 658 metadata，以及各依赖的 [PyPI 版本元数据](https://pypi.org/pypi/nvidia-cudnn-cu12/9.10.2.21/json)核对，Linux x86_64 / CPython 3.13 的主要下载量如下：

| 包 | 固定版本 | 字节 |
|---|---|---:|
| torch | 2.8.0+cu128 | 889,052,836 |
| nvidia-cuda-nvrtc-cu12 | 12.8.93 | 88,040,029 |
| nvidia-cuda-runtime-cu12 | 12.8.90 | 954,765 |
| nvidia-cuda-cupti-cu12 | 12.8.90 | 10,248,621 |
| nvidia-cudnn-cu12 | 9.10.2.21 | 706,758,467 |
| nvidia-cublas-cu12 | 12.8.4.1 | 594,346,921 |
| nvidia-cufft-cu12 | 11.3.3.83 | 193,118,695 |
| nvidia-curand-cu12 | 10.3.9.90 | 63,619,976 |
| nvidia-cusolver-cu12 | 11.7.3.90 | 267,506,905 |
| nvidia-cusparse-cu12 | 12.5.8.93 | 288,216,466 |
| nvidia-cusparselt-cu12 | 0.7.1 | 287,193,691 |
| nvidia-nccl-cu12 | 2.27.3 | 322,364,134 |
| nvidia-nvtx-cu12 | 12.8.90 | 89,954 |
| nvidia-nvjitlink-cu12 | 12.8.93 | 39,254,836 |
| nvidia-cufile-cu12 | 1.13.1.3 | 1,197,834 |
| triton | 3.4.0 | 155,569,223 |
| 合计 | torch + 14个NVIDIA包 + triton | **3,907,533,353** |

总计约 **3.91 GB / 3.64 GiB 压缩下载**。部署代理随后确认小依赖已满足：filelock 3.20.3、typing-extensions 4.15.0、setuptools 80.9.0、sympy 1.14.0、networkx 3.6.1、jinja2 3.1.6、fsspec 2026.1.0，无需另补这七个包。解压安装还需要额外空间，不能把压缩下载量当成最终磁盘增量。2.7.1+cu128 的 torch 主包反而约 1.039 GB；2.9.1+cu128 主包约 0.901 GB 且新增 nvshmem 依赖，因此未选择它们作为本次最小候选。可下载并验哈希的16包清单见 [官方wheel锁清单](../reports/gpu-wheel-lock.json)。

进一步只用 HTTP Range 读取全部16个 wheel 的 ZIP 尾部和中央目录，共读取约2.54 MB元数据，得到解压文件合计 **6,646,215,976 B（6.65 GB）**，详见 [解压体积核验](../reports/gpu-wheel-size-check.json)。wheelhouse 与安装文件合计至少10.56 GB；builder提交时可能同时保留两份约6.65 GB安装层，临时tar、字节码和存储实现还会增加用量，保守按新增峰值18–25 GB检查当前余量。服务器当前约38GB空闲，应持续保留10–15GB；禁用额外pip缓存，提交成功后只回收本项目临时builder，不做系统全局清理。此峰值是预算，解压文件总量才是ZIP目录实测值。

## 部署后验收与并发

现驱动版本高于 [NVIDIA 对 CUDA 12.8 Update 1 列出的 Linux 驱动版本 570.124.06](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)，无需为此更换系统驱动；仍须验证 NVIDIA 容器设备映射及实际动态库加载。原生 Compose 使用 NVIDIA GPU reservation。检查 `torch.cuda.is_available()`、设备数量、实际 ONNX session 的 `get_providers()`，并用原烟测 PDF 验证 GPU 利用率、解析完成状态和输出，不能仅凭“uses GPU”日志下结论。

[settings.check_and_install_torch](https://github.com/infiniflow/ragflow/blob/8090ffd8ba6098813d7f7ec2c4b8774f36c8423a/common/settings.py#L522) 自动将可见设备数赋给 `PARALLEL_DEVICES`；[PDF 原生 OCR](https://github.com/infiniflow/ragflow/blob/8090ffd8ba6098813d7f7ec2c4b8774f36c8423a/deepdoc/parser/pdf_parser.py#L1721) 支持按页轮转可见 GPU，不需要臆造 `PARALLEL_DEVICES` 环境变量。当前明确执行方案为**只按UUID绑定一张空闲3090、一个受限任务执行器**；单卡不足再评估扩展，不默认占用全部四卡。原生 `OCR_GPU_MEM_LIMIT_MB` 默认每会话 2048 MB，多模型、多进程总量仍会累加。

若转为 GPU 正式索引，应统一重新解析本次正式语料，CPU 烟测另留报告，不把两种运行条件混成同一个冻结索引。维持原生 DeepDOC 模型、渲染/分块设置与完整 84 份文档，不降级解析来缩短等待。已知 CPU 烟测为 9 页 245.58 秒，整套语料 12,013 页；简单线性外推约 91 小时串行，但样本太小、已调整 CPU/并发，**不能当作正式耗时预测或 GPU 提速倍数**。

## 已实现的项目构建入口

在项目根目录操作；确认目标GPU空闲，并将命令中的UUID换成已核验设备。下载器只接受锁清单中的官方PyTorch/PyPI来源，单段有范围核验、超时和有限重试，最终每个wheel均验证完整SHA256。当前锁清单面向镜像内CPython3.13，不依赖宿主机Python版本。

```bash
python3 scripts/download_gpu_wheels.py
python3 -m timeguard.gpu_image --gpu-uuid GPU-71384a25-5e87-bd05-8f42-8117a841e9fc
python3 -m timeguard.bootstrap --prepare-compose --local-embedding \
  --gpu-uuid GPU-71384a25-5e87-bd05-8f42-8117a841e9fc \
  --task-workers 4 --tasks-per-worker 2
```

生成配置不自动启停服务；已有任务运行时不得直接重建。构建器要求指定GPU空闲、基础镜像摘要一致、轮子全部校验通过、空间充足，并拒绝覆盖已存在的本项目builder。仅在独立构建容器中安装固定依赖，原生CUDA会话真实推理通过后提交派生镜像并清理该builder。

长文首次触发原生每会话2048MiB限制后，配置调整为8192MiB、`OCR_ARENA_EXTEND_STRATEGY=kSameAsRequested`、`OCR_GPUMEM_ARENA_SHRINKAGE=1`。这些是原生资源选项；默认表格四角度OCR保持开启。执行器数量通过原生入口 `--workers=N` 控制，每进程任务上限单独冻结；单纯设置 `NUM_WORKERS` 环境变量对该入口无效。上面的4×2是待完整长文验证后部署的配置，不代表全库已经成功。
