# 档 17 · Indexer K 暂存

档 17 以档 16 的 MXFP8 FFN 窄包和 `burst8late` 为底座，加入 SM89 单请求冷 prefill 的 Indexer K 暂存。八个活动序列、TP8/EP8/PP1、DSpark K7、自适应校验和 1,048,576 token 单请求配置窗口沿用档 16。请求分片采样属于 16→17 的待选实验。 [档 16](mode16.md) · [16→17 实验记录](mode17-experiments.md)

## 实现

原生 V4.1 indexer 在各个 query 行打分时，重复按页表读取和解包 E2M1 MXFP4 K 与 UE8M0 scale。新路径先把当块可见的 K 用同一解包公式展开到逻辑顺序的 BF16 暂存区，记录每个位置的页有效性，供该块各行和打分子批复用。两条路径都使用 BF16 dot、FP32 分数和现有 Top-K。持久缓存继续存原生 packed K。单卡暂存上界约 64.25 MiB，包含 262,144×128 个 BF16 值和各位置有效标志。 [实现](../patches/pex88096/mode17/indexer-staged-k.patch)

开关为 `DS1017_STAGED_K=1`。启用还要求 SM89、单请求纯 prefill、连续的单行页表、至少 1024 本地 query 行、可见压缩 K 为 16,384–262,144，以及 dense 打分。可见压缩 K 的行数以模型各层压缩后的索引为准。候选集读取、多请求、decode、Graph capture、上下文并行重排和超出窗口的 chunk 沿用原生 scorer。连续的页表行可以指向乱序或不连续的物理页。C8 仍由档 16 的调度处理。

暂存机制受 HyperFlare 中重复 K 物化的设计启发。本实现针对 vLLM SM89 的 1017 paged MXFP4 indexer，直接修改原生打分核和调度入口。HyperFlare 的相关 PR 面向 Kimi DCP 的 KV gather，提供的是复用思路。 [HyperFlare #374](https://github.com/NVIDIA/hyperflare-inference/pull/374) · [vLLM SM89](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89)

## 八卡单流冷 prefill

| 标称输入 | 档 16（tok/s） | Indexer K 暂存（tok/s） | 配对相对变化 |
| --- | ---: | ---: | ---: |
| 32K | 14,788 | 14,735 | −0.36% |
| 128K | 15,948 | 16,434 | +3.05% |
| 512K | 14,178 | 14,551 | +2.63% |

在 8× RTX 4090 48 GB、双 PEX88096、P2P SYS 上，同一引擎实例做两组种子配对，依次用 A→B、B→A。请求固定输出 1 token，前缀缓存命中为 0。吞吐用实际输入 token 除以引擎 prefill 时间。128K 两组分别提升 2.86%、3.24%，512K 分别提升 2.61%、2.66%，后者每次约节省 0.95 秒。试验保留原 K7 采样和原 `burst8late`，共享 KV 池为 2,753,458 token。 [两组配对数据](../results/mode17-indexer-prefill.jsonl)

## 档 17 单流 decode

| 输入约 | 档 17 Decode 中位数（tok/s） | 两条样本范围（tok/s） |
| --- | ---: | ---: |
| 32K | 171.9 | 163.5–180.2 |
| 128K | 219.6 | 179.2–260.0 |
| 512K | 200.6 | 180.9–220.3 |

单流冷输入，每个长度两个种子，固定生成 384 token，温度 0。首 token 之外的 383 个输出除以引擎 decode 时间。六条都正确找回三处植入号码，前缀缓存命中为零。输出速度会随每轮接受的 DSpark 草稿数量变化。两条 512K 中，生成较快的一条有 12.0% 的重复四元组，以列表中的相似句式为主。 [单流 decode 样本](../results/mode17-decode.jsonl)

## 数值范围

- 算子探针四组覆盖 16K、64K、256K 可见压缩 K，包括乱序物理页和无效页。FP32 logits、Top-K 集合与对应分值均逐位一致。
- 八卡真实 128K 激活审计抽查 840 个打分块、6720 个 query 行、308,274,432 个 FP32 分数，全部逐位一致。
- 六对单 token 冷 prefill 输出哈希相同。512K 检索对照两边均找回 3/3 个号码，分别生成 12 和 14 token，输出哈希有差异。整模型生成的逐 token 等价及更广任务质量仍未验证。 [512K 检索记录](../results/mode17-indexer-retrieval.jsonl)

## 安装与启动

按基础配方准备依赖、固定权重、`.env.pex88096` 和本机 GPU 编号。使用独立档 17 overlay，安装器按档 16 补丁序列继续应用 Indexer K 补丁并校验 17 个改动文件。 [基础部署与 GPU 拓扑](pex88096.md)

```bash
./setup_pex88096.sh --mode17
DRY_RUN=1 ./launch_pex88096_mode17.sh
./launch_pex88096_mode17.sh
```

默认服务监听 127.0.0.1:8011。日志 `Ada staged-K dense prefill executed` 说明合格请求已进入暂存路径。用不同 cache salt 比较同种子、同长度的单流请求。分别以默认启动和 `DS1017_STAGED_K=0 ./launch_pex88096_mode17.sh` 启动两个配置，用同一块显卡配置依次测试。用服务端指标计时，先预热各自路径，清空请求队列再采样。冷请求校验前缀缓存命中为零。

```bash
.venv-pex88096/bin/python tools/benchmark-pex88096.py \
  --output 1 --seed 22501 --require-cold \
  --cache-salt "indexer-cold-$(date +%s%N)" 32768 131072 524288
.venv-pex88096/bin/python tools/benchmark-pex88096.py \
  --retrieve --seed 22571 --require-cold \
  --cache-salt "indexer-quality-$(date +%s%N)" 524288
```

回到档 16 可停稳当前服务，再运行 `./launch_pex88096_mode16.sh`。如尚未安装它的 overlay，先运行 `./setup_pex88096.sh --mode16`。两个发布档的文件互相独立。
