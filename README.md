# DeepSeek V4.1 Flash 1017 on 8× RTX 4090 48 GB

双 PEX88096 单机部署 | PCIe P2P | SM89 | 1M 上下文

本项目提供能独立安装的 vLLM Vision14 部署档、双交换芯片的通信配置，以及预填充和解码的八卡 indexer 行分片。安装器校验运行时 wheel 和补丁，模型由固定快照与校验清单锁定。服务默认监听本机。 [完整部署手册](docs/pex88096.md) · [六条单流原始记录](results/pex88096-cold.jsonl)

## 基础 PEX88096 配方

在八张 48 GB RTX 4090 上，**512K 单流冷 prefill 实测 10,712 tok/s**，同一批请求的输出中位数为 **211.6 tok/s**。默认配置窗口为 **1,048,576 token**。历史近满窗测试在 **1,047,532 个输入 token** 下生成 384 个 token，并正确找回三处检索码。

### 单流实测

| 输入约 | 冷 prefill（tok/s） | 首 token 延迟（s） | Decode（ms/验证轮） | 输出（tok/s） |
| --- | ---: | ---: | ---: | ---: |
| 8K | 10,305 | 0.853 | 14.85 | 191.5 |
| 128K | 11,924 | 11.488 | 14.80 | 173.3 |
| 512K | 10,712 | 50.980 | 15.74 | 211.6 |

每个长度使用两个匹配输入 seed，单并发，每份输出 384 token。所有样本的前缀缓存命中为零，且正确找回三处检索码。Prefill 与输出吞吐按引擎各阶段时间计算，首 token 延迟从客户端计时。DSpark 每轮接受的草稿数随内容变化，比较 decode 时可同时看验证轮耗时和输出吞吐。

近满窗历史样本采用同一 prefill 分片方案，引擎冷 prefill 为 112.44 秒，约 9,317 tok/s。该样本用的是当时的 decode 路径。上述速度都来自原版 FlashInfer Vision4。公开安装器默认使用可下载的 Vision4 post1，已验证依赖安装、补丁摘要、导入和启动 dry-run。 [测试方法与版本说明](docs/pex88096.md#单流实测)

### 配置

| 项目 | 默认值 |
| --- | --- |
| 权重 | DeepSeek-V4.1-Flash-1017，固定 revision，48 个分片，约 475 GiB |
| 并行 | 模型 TP8、draft TP8、专家分布在八卡、PP1 |
| 上下文 | 1,048,576 个 token，含输入、模板与输出 |
| 调度 | 8192 batched tokens，最多 4 个活动序列 |
| KV cache | FP8，显存使用率 0.96，参考机器实测 KV 池 3,522,266 token |
| Prefill | CED，八卡 prefill indexer 行分片，Ada 8192 行投影 tile |
| Decode | 八卡 decode indexer 行分片，DSpark K7 自适应验证 |
| CUDA Graph | `FULL_AND_PIECEWISE`，最大捕获尺寸 48 |
| 通信 | `NCCL_P2P_LEVEL=SYS`，启用 PCIe IPC 小消息归约 |
| API | 默认 `127.0.0.1:8011`，模型名 `deepseek-v4.1-flash-1017` |

参考拓扑的两个交换芯片分别连接 GPU `{2,3,4,5}` 和 `{0,1,6,7}`。示例按这两个组排列 CUDA rank，模型仍使用一个 TP8 组。实际部署按本机 PCIe 树重新核对编号与通信网卡。参考机器上，P2P 读写矩阵各有 56 个有向跨卡设备对通过，启动日志可核对 `P2P/CUMEM` 和 PCIe IPC。 [配置示例](profiles/pex88096.env.example) · [拓扑核对方法](docs/pex88096.md#识别双交换芯片与验证-p2p)

### 三项 SM89 修改

1. **Prefill indexer 行分片**。把八卡原先重复执行的查询行分给各 rank，每行保留原来的因果搜索范围和本地打分工作区上限，完成后汇总整数 Top-K。
2. **Decode indexer 行分片**。符合条件的原生全历史打分分给八卡，汇总整数 Top-K 和下游候选块。合成算子测试中 logits 逐元素相等。并列打分时，现有 Top-K 仍可能选取不同位置，整模型没有建立逐 token 等价。
3. **Ada prefill 投影 tile**。在已测的 8192 行 MXFP8 投影形状上选择 128×64 tile，保留已有的 K32 缩放流程。

安装器在独立 vLLM overlay 上应用三份补丁并校验源码摘要。可以从同一源码和依赖复建，也便于追踪 wheel 变更。 [生产补丁](patches/pex88096/) · [运行时依赖](profiles/pex88096-constraints.txt)

### 安装和手动启动

参考环境为 Linux x86-64、CUDA Toolkit 13.0、Python 3.12、uv 和可用的 4090 PCIe P2P 驱动。参考主机使用 NVIDIA 595.71.05，配有 2 TiB RAM。Engram 的两张表在 TP8 下合计约用 189 GiB 主机 pinned memory。模型、Python 依赖和编译缓存都需要本地存储空间。 [硬件与驱动版本](docs/pex88096.md#硬件与版本)

```bash
git clone https://github.com/bluewowo/deepseek-v4.1-flash-8x4090-pex88096-p2p.git
cd deepseek-v4.1-flash-8x4090-pex88096-p2p
./setup_pex88096.sh
cp profiles/pex88096.env.example .env.pex88096
uvx --from huggingface-hub hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --local-dir models/DeepSeek-V4.1-Flash-1017
./tools/check-host.sh
```

按本机修改 `.env.pex88096` 里的设备顺序、通信网卡和路径，然后验证启动参数。独占八张卡后启动前台服务，按 Ctrl+C 停止。

```bash
DRY_RUN=1 ./launch_pex88096.sh
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH ./launch_pex88096.sh
```

另开终端检查健康状态，再发起冷请求。服务空闲时运行，使用新的 cache salt。已有自定义模型目录时，给基准脚本传 `--model-dir`。

```bash
curl --fail --noproxy '*' http://127.0.0.1:8011/health
ENV_FILE="$PWD/.env.pex88096" ./tools/smoke-test.sh
.venv-pex88096/bin/python tools/benchmark-pex88096.py \
  --output 384 --seed 1017 --repeats 2 --require-cold \
  --cache-salt "cold-$(date +%s%N)" 8192 131072 524288
```

## 档 15 · 序列并行与通信减量

档 15 在基础配方上优化完整 8192 行的单请求 encoder prefill。各 rank 保留本地序列行，部分层交换较窄的 WO 激活。紧凑 CED 先做本地边界投影再汇合，两层 Engram 使用 rank-major 门控。FFN 合并本地专家结果后用 BF16 ReduceScatter，直接得到各 rank 所需的输出行。 [实现机制与 Spark 来源](docs/mode16.md#档-15-底座) · [档 15 底座补丁](patches/pex88096/mode16/mode15-foundation.patch)

| 输入约 | 档 9 冷 prefill | 档 15 冷 prefill | 配对中位提速 |
| --- | ---: | ---: | ---: |
| 8.4K | 11,795 tok/s | 12,399 tok/s | 7.58% |
| 32K | 13,366 tok/s | 14,385 tok/s | 7.56% |

这组实测为同实例单流、零前缀缓存配对，增幅取各对的中位数。档 15 支持 8 个活动序列，参考机共享 KV 池为 2,753,458 token。档 15 是档 16 的直接底座。用档 16 的 overlay，将 `mode` 设为 `15`，并将 `ffn-mode` 与 `scheduler-mode` 设为 `base`，即可复现档 15 的对照配置。 [开关与复跑](docs/mode16.md#复跑单变量对照)

## 档 16 · FFN 窄包与 C8 调度

档 16 在档 15 基础上增加 MXFP8 FFN 输入窄包和 `burst8late`。512K 单流冷 prefill 实测 **14,183 tok/s**，相对同实例档 15 提升 **4.23%**。活动序列上限仍为 8，单请求配置窗口为 1,048,576 token。 [档 16 部署与完整实测](docs/mode16.md)

| 改动 | 实测 |
| --- | --- |
| MXFP8 FFN 输入窄包 | 每 rank 的 AllGather 从 10 MiB 降至 5.203 MiB，32K / 128K / 512K 冷 prefill 分别提升 3.71% / 4.43% / 4.23% |
| `burst8late` 长短请求调度 | 独立 C8 对照中，短流输出片间隔 p95 从 978–985 ms 降至 556–562 ms，512K 长流 TTFT 从 59.82–60.42 s 降至 47.68–48.03 s |
| 联合验收 | 一个约 256K 长流和七个短流均完成，零前缀缓存命中 |

窄包和调度的收益分别来自各自的单变量配对。窄包路由和激活在逐层样本上逐位一致，Marlin 最终输出仍有少量 BF16 位差。 [单流测试数据](results/mode16-prefill.jsonl) · [调度测试数据](results/mode16-scheduler.jsonl)

完成基础配置和模型下载后，手动启动档 16。服务默认监听本机。

```bash
./setup_pex88096.sh --mode16
DRY_RUN=1 ./launch_pex88096_mode16.sh
./launch_pex88096_mode16.sh
```

## 项目来源

本项目 fork 自 ykk648 的八卡部署。1017 运行时来自 vLLM SM89 fork。解码索引行分片、序列并行和窄包设计参考 Spark 的相应实验，`burst8late` 按本机 8192 行快路径设计。 [上游部署](https://github.com/ykk648/deepseek-v4.1-flash-8x4090) · [vLLM SM89](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89) · [Spark 索引分片](https://github.com/christopherowen/spark-ds41f/blob/8e5ea208f3d22bc69df91ace94d6975932c858a8/patches/vllm/0025-deepseek-v41-indexer-sp-split.patch) · [档 16 机制及引用](docs/mode16.md) · [MIT](LICENSE) · [vLLM Apache-2.0](patches/pex88096/licenses/vllm-LICENSE)

## English summary

Single-node DeepSeek V4.1 Flash 1017 on eight 48 GB RTX 4090s connected through two PEX88096 switches. The base profile configures PCIe P2P, shards prefill and decode indexer rows, and tunes an Ada prefill GEMM tile. It measured 10,712 cold prefill tok/s and 211.6 decode tok/s at 512K. Mode 15 adds sequence-parallel prefill, compact CED, and reduce-scatter. Mode 16 builds on Mode 15 with MXFP8 FFN input packets and mixed-length scheduling. Its paired 512K result was 14,183 cold prefill tok/s, up 4.23% from Mode 15. [Base deployment](docs/pex88096.md) · [Modes 15 and 16](docs/mode16.md)
