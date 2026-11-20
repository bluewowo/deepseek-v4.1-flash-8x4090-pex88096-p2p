# Mode 16 on eight RTX 4090s

档 16 是独立可选的八并发配方。它从本仓的三份 SM89 基础补丁构建，依次加入档 15 的序列并行底座、MXFP8 FFN 输入窄包和 `burst8late` 调度。公开安装器从固定 Vision14 wheel 重建 overlay。现有 PEX88096 profile 仍可作为独立基线。 [部署与硬件](pex88096.md) · [档 15 底座补丁](../patches/pex88096/mode16/mode15-foundation.patch) · [档 16 补丁](../patches/pex88096/mode16/mode16.patch)

## 启动

在 [基础部署手册](pex88096.md#从空环境安装) 中完成模型下载并配置 `.env.pex88096`。在八张卡空闲时，运行以下命令。服务在前台运行，按 Ctrl+C 停止。

```bash
./setup_pex88096.sh --mode16
DRY_RUN=1 ./launch_pex88096_mode16.sh
./launch_pex88096_mode16.sh
```

默认监听 `127.0.0.1:8011`，使用 TP8、EP8、PP1、NCCL P2P SYS、FlashInfer PCIe IPC、DSpark K7、8192 行调度块和 8 个活动序列。1,048,576 token 的单请求窗口包含输入、模板和生成预算。参考机的 2,753,458 token 共享 KV 池折合约 2.63 路满窗。混合长度可以达到更高并发。权重快照、专家 scale 原格式、FP8 KV 和两个 indexer 分片沿用基础配方。

安装器验证 wheel 摘要和 16 个受影响文件的 SHA256，把档 16 写入 `overlays/pex88096-vision14-mode16/`。基础档的 overlay 有独立目录。公开安装默认采用可下载的 FlashInfer Vision4 post1。下文八卡数据来自相同 vLLM Vision14 与原版 Vision4。持有校验和匹配的原版 wheel 时，可在安装时设置 `FLASHINFER_VARIANT=original`。

另开终端，检查健康状态并用零缓存请求确认窄包。该请求的前几块会触发 20 个 encoder 层的 `Ada FFN narrow gather executed mode=narrow-padded`。

```bash
curl --fail --noproxy '*' http://127.0.0.1:8011/health
.venv-pex88096/bin/python tools/benchmark-pex88096.py \
  --retrieve --require-cold --cache-salt "mode16-$(date +%s%N)" 32768
```

## 如何减少 prefill 通信

### 档 15 底座

对满足条件的 8192 行单请求 encoder prefill，将 residual 与 mHC 的一部分按序列行保留在各 rank。WO 路径在部分层交换较窄的 WO-A 激活，再投影本地 WO-B。18 层保有相应副本。两层 Engram 采用投影后交换以及 rank-major 门控。紧凑 CED 在本地先投影边界 KV，再汇合较窄的张量。FFN 将共享与 routed expert 的本地结果相加，然后用 BF16 ReduceScatter 直接得到各自需要的输出行。 [Spark 序列并行与 MoE](https://github.com/christopherowen/spark-ds41f/blob/8e5ea208f3d22bc69df91ace94d6975932c858a8/patches/vllm/0015-deepseek-v41-prefill-sp-dataflow.patch) · [Spark 1107 的 WO、Engram、CED 和 rank-major 补丁索引](https://github.com/christopherowen/spark-ds41f/blob/8e5ea208f3d22bc69df91ace94d6975932c858a8/experiments/2026-10-26-spark-ds41f-candidate/1107/README.md)

### FFN 输入窄包

档 16 将每 rank 负责的 1024 行路由和官方 K32 MXFP8 activation cast 提到 AllGather 前。每行发送 5120 B E4M3、160 B UE8M0 scale、24 B FP32 top-6 权重及 24 B int32 专家 ID，共 5328 B。BF16 原路径每行 10,240 B，每卡每次 AllGather 从 10 MiB 降到 5.203 MiB，减少约 48%。输出阶段仍用档 15 的 BF16 ReduceScatter。 [Spark 的 sequence-parallel MoE 输入实验](https://github.com/christopherowen/spark-ds41f/blob/fbff20b837f4b62dee7fb9935eea9329bbdc1492/experiments/2026-10-14-sp-moe-gather/README.md) · [Spark C58 接入](https://github.com/christopherowen/spark-ds41f/blob/fbff20b837f4b62dee7fb9935eea9329bbdc1492/experiments/2026-11-14-c58-prefill-10k/README.md)

SM89 实现将连续量化激活和 scale 传给共享专家，再反量化给现有 Marlin routed expert。每个 rank 在 GateLinear 中保留原 8192 行形状，将别的 rank 的行补零来维持 GEMM 的数值路径。在 8409 token 审计样本的 20 层和 8 rank 上，专家 ID、路由权重、FP8 与 scale 字节、Marlin 首层输入以及 shared expert 输出逐位一致。直接缩为 1024 行的探针出现了 8 个路由 ID 变化，因此发布 `narrow-padded`。

同输入的原 Marlin 重复运行也有 BF16 末位差异。窄包对原路径在最终分片中有 801 个变化元素，原路径重复对照有 1361 个变化元素，分母均为 838,860,800。最大绝对差分别为 0.0078125 和 0.00390625。六组性能配对的首 token 一致，两份 128K 和 512K 三处检索的全文哈希一致。质量证据覆盖上述样本和算子。整模型逐 token 等价、通用任务能力与档 16 的独立 decode 吞吐仍待验证。

### 长短请求混合调度

Spark 的 1107 实验关注长 prefill 对短流的阻塞，使用分级块宽与插入 decode-only 步骤。这里的 `burst8late` 按 Ada 实测形状保留 8192 行块。至少一个约 250K 以上的文本长请求与可继续 decode 的短请求重叠时，优先分开纯 prefill 和纯 decode，最多安排八轮 decode 后推进长流。只有一个这类长流运行且其余待 prefill 的短文本不超过 16K 时，可以先给短流最多两步。视觉请求等不符合覆盖条件的工作负载走原调度。 [Spark 1107 的混合调度补丁索引](https://github.com/christopherowen/spark-ds41f/blob/8e5ea208f3d22bc69df91ace94d6975932c858a8/experiments/2026-10-26-spark-ds41f-candidate/1107/README.md) · [档 16 调度源码](../patches/pex88096/mode16/mode16.patch)

## 八卡实测

| 输入约 | 档 15 单流冷 prefill | 窄包版单流冷 prefill | 配对增幅 |
| --- | ---: | ---: | ---: |
| 32K | 14,277 tok/s | 14,806 tok/s | 3.71% |
| 128K | 15,279 tok/s | 15,956 tok/s | 4.43% |
| 512K | 13,608 tok/s | 14,183 tok/s | 4.23% |

同实例按 A/B/B/A 顺序跑匹配的两份 seed。各对输入 token 相同且缓存命中为零，性能卷只生成一个 token。表内 tok/s 是 `prompt_tokens / engine_prefill_s` 的均值，提升为两组配对增幅的均值。512K 客户端 TTFT 从平均 40.57 秒降至 38.87 秒。 [脱敏单流数据](../results/mode16-prefill.jsonl)

| 独立调度 A/B | 原调度 | `burst8late` |
| --- | ---: | ---: |
| C4 晚到短流首 token | 26.93–27.12 s | 1.34–1.50 s |
| C8 短流输出片间隔 p95 | 978–985 ms | 556–562 ms |
| C8 512K 长流首 token | 59.82–60.42 s | 47.68–48.03 s |

C4 含一条约 256K 长流与三条约 8K 短流，其中一条晚到。C8 含一条约 512K 长流与七条已出字的短流。独立调度对照的短流检索原档和候选各 20/20。联合档另用约 256K 长流和七条短流做一次验收，8/8 完成且缓存命中为零，晚到短流 TTFT 为 2.007 秒。窄包与调度的增幅来自各自的独立对照。 [脱敏混合调度数据](../results/mode16-scheduler.jsonl)

## 复跑单变量对照

默认 launcher 读 [版本化的三个设置文件](../profiles/mode16/)。如需切换模式，在启动前复制这些文件到临时目录，用 `MODE16_STATE_DIR` 指定副本。现场 cold-start 先关掉窄包和新调度，依次用短检索预热模式 `0 4 7 10 9 13 11 15`，再用模式 15 跑一次 32K 检索，随后开启两个新增开关。

```bash
state=$(mktemp -d)
cp profiles/mode16/* "$state/"
MODE16_STATE_DIR="$state" ./launch_pex88096_mode16.sh
```

记录 `state` 路径，在另一终端针对空闲实例测试。下面的 `set_switch` 原子替换开关。比单流 prefill 时把 `scheduler-mode` 固定为 `base`，按 A/B/B/A 切换 `ffn-mode` 的 `base` 和 `narrow-padded`。比混合调度时把 `ffn-mode` 固定为 `base`，切换 `scheduler-mode` 的 `base` 和 `burst8late`。使用相同 seed 和输入长度，逐对比较。

```bash
set_switch() {
  printf '%s\n' "$2" > "$state/$1.next"
  mv "$state/$1.next" "$state/$1"
}

.venv-pex88096/bin/python tools/benchmark-pex88096.py \
  --seed 6124 --output 1 --require-cold --cache-salt "ab-$(date +%s%N)" \
  32768 131072 524288

.venv-pex88096/bin/python tools/benchmark-mode16-mixed.py \
  http://127.0.0.1:8011 models/DeepSeek-V4.1-Flash-1017/tokenizer.json \
  4:262144:5361:1 8:524288:5301
```

英文概览见项目首页。部署封装遵循项目的 MIT 许可，vLLM overlay 衍生补丁沿用 Apache-2.0；本仓发布的是应用到现有 wheel 的源码补丁和合成跑分指标。 [项目许可证](../LICENSE) · [vLLM 补丁许可证](../patches/pex88096/licenses/vllm-LICENSE)
