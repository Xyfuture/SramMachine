# SramMachine 与 SA 脚本使用指南

## 项目简介

SramMachine 是面向 SRAM-centric AI 加速器的离散事件模拟器。目前主要用于研究 decoding 阶段中 Attention、MoE、存储和片上/片间通信的延迟，以及 SplitTree microbatch 对流水并行效果的影响。

当前主要流程为：

```text
Model + InferenceConfig
        ↓
HardwareMapper
        ↓
RootNode / SplitTree
        ↓
TreeParser → CommandGraph
        ↓
GraphExecutor + Desim
        ↓
延迟、单用户吞吐、总吞吐和逐 command trace
```

目前支持的模型：

- `deepseek-v3`
- `deepseek-v3.2`
- `kimi-k2.5`
- `glm-5.1`

模拟器支持 MoE TP/EP、MTP1、FP8/FP16 KV cache，以及由模拟退火搜索 SplitTree。TP 与 EP 可以在同一次脚本运行中搜索；模拟退火同时考察单用户吞吐和总吞吐，并保留搜索得到的 Pareto 最优 SplitTree。

当前 MoE 映射不按 BS 分档：TP 的16个chip仅沿 intermediate N
切分；每chip视为统一的8×8 PU mesh，直接做 hidden K8 × intermediate N8。
TP 的 MoE 权重读取、SiLU 以整个chip为资源粒度；Up输入和Down输出
使用显式总字节数和横向NoC带宽计时。当前仍省略Up K partial归约，属于实验假设。
EP仍由16个chip分别持有所属experts，chip内同样使用统一的8×8 PU mesh
对每个本地expert做K8×N8张量并行；dispatch/combine仍是EP16 all-to-all。
两种模式中，每个chip的64个PU对**每个本地expert**做张量并行，
不按expert给PU分工。activation按FP8（1 byte/元素）计费，
Fabric为每chip每方向1.6TB/s。切分依据和通信计费见
[MoE TP/EP映射说明](MOE_PURE_TP_EXPERIMENT.md)。旧策略代码仍在，但
默认16-chip四模型不再触发。
- 小 batch 允许只激活部分experts；inactive expert不产生weight或计算命令。

## 并行 SA 脚本

通信算子分为 `NoCOp`（片内）和 `FabricOp`（片间）；原 `CommOp` 已移除。
NoC 使用两个独立带宽资源：NocHorizontal (`chip0.noc_horizontal`) 和
NocVertical (`chip0.noc_vertical`)，横向资源 ID 的旧拼写 `noc_horizotal` 不再使用。

| NoCOp.operation | 占用资源 | size_bytes 的含义 |
| --- | --- | --- |
| `ToIOdie` | 横向 | IO 相连 PU 向 IO die 发送的总字节数 |
| `FromIOdie` | 横向 | IO die 向相连 PU 发送的总字节数 |
| `PUVertical` | 纵向 | 发送量最大的 PU 的发送字节数 |
| `PUHorizontal` | 横向 | 发送量最大的 PU 的发送字节数 |
| `PUMesh` | 横向、纵向 | 发送量最大的 PU 的发送字节数 |

`NoCOp(op_id, operation, size_bytes, kind=None, group=())` 中，PU 通信必须提供
集合通信标记 `kind`；标记和参与者数量均不参与仿真延迟计算。IO 通信可省略标记。
`NoCCmd` 根据 `operation` 自动补齐资源；错误方向和跨 chip 的资源组合会被拒绝。
`PUMesh` 等待两个资源均空闲后原子占用，一起释放；两方向上的单独指令可以重叠。
IO 输入、输出竞争同一个横向资源。

延迟为 `ceil(size_bytes / bandwidth_bytes_per_second * 1e9)` 纳秒，不再乘集合
通信系数或并行链路数，也不加 `link_latency_ns`。
`NoCConfig.horizontal_bandwidth_bytes_per_second` 和
`vertical_bandwidth_bytes_per_second` 可分别设置；省略时均沿用
`link_bandwidth_bytes_per_second`。Mesh 使用两方向带宽的较小值，且只计一次延迟。

现有模型 Mapper 的 BMM/FlashAttention 输入输出分别生成 `FromIOdie`/`ToIOdie`，
保留原映射的输入输出字节总量，移除原 8/32 路带宽放大假设。
PU 集合通信的旧输入 payload 在映射阶段按现有集合通信算法转换为发送量；die
级 payload 按 `sender_partition_count` 均分到发送 PU。非均匀 alltoall 取发送矩阵
非对角行和的最大值，不取接收量最大值。转换后的 `NoCOp.size_bytes` 是最终发送量，
TreeParser 只按 microbatch 缩放一次。Fabric 仍使用原集合通信计时逻辑。

Trace 记录 `operation`、`size_bytes`、`volume_semantics`、带宽和完整 `resource_ids`。
Perfetto 在 Mesh 的两个方向轨道上显示相同通信。

入口为：

```text
srammachine/script/run_sa.py
```

建议在 SramMachine 项目根目录、已激活项目 conda 环境后，通过模块方式运行：

```powershell
conda activate SramMachine
python -m srammachine.script.run_sa --help
```

脚本将每个 `(model, MoE strategy, batch size, MTP)` 组合视为一个独立 SA case。由于 Desim 使用进程内全局仿真状态，脚本采用多进程而不是多线程。每个 worker 只运行一个 case，完成后立即退出，由操作系统完整回收 Desim、greenlet 和 Python/native allocator 占用，再创建新 worker 处理后续 case。默认并发数为：

```text
min(系统逻辑核心数, case 数量)
```

Batch size会按降序进入任务队列，使耗时通常更长的大batch优先运行，减少并行末期只剩少数重型case的长尾。不同batch之间没有同步屏障；任一worker空闲后都会立即领取队列中的下一项。完整CSV也按batch降序整理，运行中checkpoint仍按实际完成顺序追加。

可通过 `--workers` 手动限制进程数。如果内存不足，建议降低该值。

每个 case 默认执行4次完整且确定性的 restart。各 restart 的温度、预热边界和接受链相互独立，但共享该 case 已完成的仿真评分。邻居生成会优先探索尚未评估的合法 SplitTree，只有当前树的全部直接邻居都已评估后才回退到缓存候选。

## 基本示例

### 全量 400-round 测试规则

当前全量测试固定覆盖：

- 模型：`deepseek-v3`、`deepseek-v3.2`、`kimi-k2.5`、`glm-5.1`
- MTP：`off`、`on`
- MoE：`tp`、`ep`
- Batch size：`32 64 128 256 512 1024 2048 4096 8192`
- 每次 restart：400 个正式 SA rounds
- Restart：4 次

总 case 数为 `4 × 2 × 2 × 9 = 144`。默认4次restart下，每个case实际执行
`4 × 400 = 1600`个正式proposal，全量共`144 × 4 × 400 = 230400`个正式
proposal，另有每次restart的warmup。推荐显式写出`--restarts 4`，避免对轮数口径
产生歧义：

```powershell
python -m srammachine.script.run_sa --models deepseek-v3 deepseek-v3.2 kimi-k2.5 glm-5.1 --mtp off on --batch-sizes 32 64 128 256 512 1024 2048 4096 8192 --rounds 400 --restarts 4 --input-sequence-length 36000 --output-sequence-length 600 --moe-strategy tp ep --output-dir "..\csv result"
```

如果“每个case总共400轮”指跨所有restart合计400轮，则应使用
`--rounds 400 --restarts 1`。`--rounds`始终表示每次restart的正式轮数，
不会自动除以restart数。

同时运行四个模型、四种 batch size，并分别搜索 MTP 关闭和开启状态：

```powershell
python -m srammachine.script.run_sa --models deepseek-v3 deepseek-v3.2 kimi-k2.5 glm-5.1 --mtp off on --batch-sizes 256 512 1024 2048 --rounds 50 --input-sequence-length 20000 --output-sequence-length 600 --moe-strategy tp ep
```

同一命令也可运行多个 ISL，例如：

```powershell
python -m srammachine.script.run_sa --models deepseek-v3 deepseek-v3.2 --mtp off on --batch-sizes 512 1024 --rounds 50 --input-sequence-length 20000 36000 --output-sequence-length 600 --moe-strategy tp ep
```

这条命令包含 `2 × 2 × 2 × 2 × 2 = 32` 个独立 case（模型 × MTP ×
BS × ISL × MoE 策略）。提交队列按 BS 从大到小、同一 BS 内按 ISL 从大到小；
并行 worker 的完成顺序及中途 CSV 写入顺序可能不同。最终 CSV 按 BS、ISL
降序整理。

只运行 DeepSeek V3 的 EP、MTP 关闭场景，并限制为 8 个 worker：

```powershell
python -m srammachine.script.run_sa --models deepseek-v3 --mtp off --batch-sizes 256 512 1024 --rounds 100 --input-sequence-length 36000 --output-sequence-length 600 --moe-strategy ep --workers 8
```

使用 FP16 KV cache：

```powershell
python -m srammachine.script.run_sa --models deepseek-v3.2 --mtp off on --batch-sizes 256 512 --rounds 50 --input-sequence-length 20000 --output-sequence-length 600 --moe-strategy tp --kv-cache-dtype fp16
```

同时运行 DeepSeek V3 的 TP/EP、MTP on/off，并搜索 BS=128 到8192：

```powershell
python -m srammachine.script.run_sa --models deepseek-v3 --mtp off on --batch-sizes 128 256 512 1024 2048 4096 8192 --rounds 200 --input-sequence-length 36000 --output-sequence-length 600 --moe-strategy tp ep --output-dir "..\csv result"
```

## 参数说明

必须提供：

| 参数 | 含义 |
|---|---|
| `--models` | 一个或多个模型名称 |
| `--mtp` | `off`、`on`，也可以同时指定 |
| `--batch-sizes` | 一个或多个全局 batch size |
| `--rounds` | 每次 restart 的正式 SA 轮数 |
| `--input-sequence-length` | 一个或多个 decoding 历史上下文长度；单值用法保持兼容 |
| `--output-sequence-length` | 输出长度元数据 |
| `--moe-strategy` | 一个或多个MoE并行策略：`tp`、`ep`，也可以同时指定 |

可选参数及默认值：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--kv-cache-dtype` | `fp8` | KV cache 数据类型，可选 `fp8`/`fp16` |
| `--warmup-rounds` | `16` | 每个 case 的 SA 预热轮数 |
| `--initial-temperature` | `1.0` | 初始温度 |
| `--final-temperature` | `0.01` | 最终温度 |
| `--layer-count` | `4` | 用于估算稳态层间隔的代表层数量 |
| `--seed` | `20260912` | SA 基础随机种子 |
| `--restarts` | `4` | 每个 case 的确定性完整重启次数 |
| `--workers` | 自动 | 最大并行 worker 进程数 |
| `--output-dir` | `best split tree result` | CSV 和 JSON 默认输出目录 |
| `--output-csv` | 自动命名 | 指定汇总 CSV 的完整路径 |

基础 seed 会结合 model、MoE 策略、batch 和 MTP，确定性地产生 workload seed。
对应的不同 ISL case 共用这个 seed，以保持原有单 ISL 运行的搜索轨迹；
改变 worker 数量或任务完成顺序不会改变搜索结果。

## 输出结果

一次完整运行会产生：

1. 一份跨模型、MoE策略、batch和MTP的汇总CSV。
2. 每个 `(model, MoE strategy, MTP, ISL)` 一份包含最优SplitTree的JSON；上述单ISL全量配置共16份。
3. 每个case选择一棵最优SplitTree重放并导出一份Perfetto trace；上述全量配置共144份。
4. `case checkpoints/<CSV文件名>/`目录中每个已完成case一份可独立重放的SplitTree JSON；上述全量配置共144份。

CSV在worker启动前就会创建。每个worker完成SA后，会先原子写入该case的SplitTree checkpoint，再生成并原子写入Perfetto trace；两者均成功后才向主进程报告case完成。主进程随后将完整指标、路径和PU利用率写入CSV并强制刷新。因此程序被终止或某个case失败时，已经报告完成的case仍保留CSV、SplitTree和trace。若恰好在某个case的trace重放期间退出，该case尚未记入CSV，但已经写好的SplitTree checkpoint仍会保留。部分CSV的`is_model_global_pareto`字段为空；只有全部case成功后，该文件才会写入最终Pareto标记。

自动文件名以本地时间精确到分钟，并在重名时追加序号，不会覆盖已有结果。
每个 case 的 trace、SplitTree checkpoint 和最终 SplitTree JSON 文件名都包含
`isl<长度>`；最终 JSON 的workloads与Pareto front只包含对应ISL。

CSV 的关键字段包括：

- `baseline_latency_ns`
- `pareto_best_latency_ns`
- `baseline_single_user_throughput_per_second`
- `pareto_best_single_user_throughput_per_second`
- `baseline_total_throughput_tokens_per_second`
- `pareto_best_total_throughput_tokens_per_second`
- `latency_reduction_fraction`
- `throughput_improvement_fraction`
- `is_model_global_pareto`
- `restart_count`
- proposal、仿真次数、cache hit 和接受次数等搜索统计

吞吐定义为：

```text
单用户吞吐 = accepted_tokens_per_step / 总延迟
总吞吐 = 单用户吞吐 × global_batch_size
```

MTP 关闭时 `accepted_tokens_per_step=1`；MTP1 开启时假设第二个 token 总能被接受，因此该值为 2。

每个模型、MoE策略和MTP组合的最终JSON保存全部batch workload指标、Pareto front、每次restart的统计，以及所有并列最优SplitTree。`best_split_trees_by_workload`明确列出每个batch的最佳树；其中`split_tree_records`的每一棵树都附带`global_batch_size`和`mtp_enabled`。旧的`split_trees`裸结构字段继续保留以兼容已有工具。最终JSON和case checkpoint中的树都可以通过`load_pareto_split_tree()`读取和复现。

Pareto front 只在固定配置内部计算。分组字段包括模型、MoE策略、ISL、OSL、KV dtype、layer count和chip count；TP与EP或不同推理配置不会互相支配。CSV中的`pareto_best_*`表示固定workload内搜索到的最好结果，`is_model_global_pareto`表示该结果是否位于对应固定配置的跨batch/MTP Pareto front。

## 最低 Batch 要求

- TP只要求batch为正整数，因此最低为`BS=1`。
- EP要求`BS × accepted_tokens_per_step × top_k ≥ chip_count`，确保每个chip至少有一个活跃expert assignment。
- 默认16 chips、top-k=8时，EP的MTP off最低`BS=2`，MTP on最低`BS=1`。
- 32 chips时对应最低值为`BS=4`和`BS=2`。
- 当同时指定`--moe-strategy tp ep`时，所有batch会与两种策略做笛卡尔积，因此batch列表必须同时满足EP约束；否则整个任务会报告具体失败case并停止。

## 使用注意事项

- `--rounds` 是每次 restart 的正式轮数；`--warmup-rounds` 也会在每次 restart 中完整执行。
- 默认每个 `(model, MoE strategy, batch, MTP)` case 运行4次 restart，因此正式 proposal 总数为 `case数 × 4 × --rounds`。
- 各 restart 使用稳定且不同的随机种子，搜索状态相互独立，但会共享同一 case 已完成的仿真评分以避免重复运行 Desim。
- SA候选仍使用原TreeParser、GraphExecutor和Desim调度语义；worker只跳过Desim优先队列每次操作附带的重复全表调试扫描，并通过轻量结果路径直接读取候选latency。baseline、最佳树和trace仍生成完整SimulationResult，且最佳树重新物化时会严格核对latency。
- case 数量等于 `模型数 × MoE策略数 × batch数 × MTP状态数`。
- 默认会使用尽可能多的逻辑核心；大型搜索也会占用较多内存，必要时使用 `--workers` 限制并发。
- 不同 case 位于独立进程中，不共享 Desim 状态或 SA 搜索轨迹。
- 所有case在进入进程池前按BS全局降序排列：全部最大BS先入队，再依次进入较小BS；不设置批次间屏障，空闲worker会立即领取队列中的下一个case。
- worker在每个case结束后都会回收，因此已完成case不会持续累积内存。默认仍使用全部逻辑线程；在16核32线程Windows机器上峰值内存通常约为`32 × 单个case内存`，若出现分页或内存不足，应显式使用`--workers 16`。
- 一个命令可以同时指定`--moe-strategy tp ep`；汇总CSV合并两种策略，最终JSON按模型、策略和MTP分别导出。
- 显式指定的 `--output-csv` 如果已经存在，脚本会拒绝覆盖。
- 每个case搜索完成后，脚本立即保存独立SplitTree checkpoint、重放该树并生成Perfetto trace和稳态PU利用率；trace阶段不增加SA搜索轮数。
- 中断后，已完成case的CSV、独立SplitTree checkpoint和trace均保留；正在运行的case仍需重新搜索，脚本暂不自动续跑。

### PartitionChain 驱动的 BMM lowering

`BMMOp` 的 `B/M/K/N` 可以描述切分前的完整逻辑计算，`partition_chain`
记录该计算到参与 PU 网格的有序映射。空链表示单 PU。示例：

```python
from srammachine.frontend.modules import BMMOp, PartitionChain

op = BMMOp(
    "projection", B=8, M=16, K=1024, N=4096,
    partition_chain=PartitionChain((
        ("B", "p", 4), ("N", "p", 2),
        ("K", "p", 4), ("N", "q", 64),
    )),
    input_source="io_die",
    output_destination="io_die",
)
assert op.partition_chain.grid_shape == (32, 64)
print(op.partition_chain.tile_ranges(
    {"B": op.B, "M": op.M, "K": op.K, "N": op.N}, p=0, q=0,
))
print(op.communication_volumes())
```

切分按顺序逐级向上取整，坐标按各物理维度的混合进制解码；有效范围同时裁剪到
父 tile 和原始 tensor 范围。零元素 PU 不贡献通信量。完整 shape 先按 PipeTree
microbatch 缩放，再进行 PU 切分。计算命令取最大有效 PU tile，沿用
`systolic_gemm_duration_ns` 的阵列数量因子搜索。

- Input：`io_die`（默认）或 `none`。逐行对有效 B/M/K 范围取并集；N 切分的
  broadcast 不重复计数，B/M/K 切分的 scatter 累加不重叠部分。
- Output：`io_die`（默认）、`none` 或 `pu_vertical`。IO 路径逐行对有效
  B/M/N 范围取并集；K 切分在行内 reduce，不增加输出量，其余切分进行 gather。
  不同行的 K 部分和送到 IO die，PU 横向通信不承担跨行归约。
- `pu_vertical`：同列中相同 B/M/N 输出分片的 K 部分和向最上方参与 PU 做
  reduce，沿途合并部分和。无 K 纵向切分时无需通信。不同归约组交错映射时，
  共享纵向链路的字节数相加，最终取最大发送链路负载；不包含横向通信或 all-reduce。

IO 命令的 `io_bytes_per_row` 保留各行有效字节数，`size_bytes` 为最大行字节数，
按单条横向链路带宽计时（各行并行）。非整除时各行可能不同。显式 `NoCOp`
没有该字段时保留原来的 `size_bytes` 语义，trace 用 `volume_semantics` 区分。

lower 生成 `.input -> .core -> .output`，`none` 或零通信量时省略对应通信节点。
相邻算子和跨层依赖均从输出完成连接到下一个输入开始。`GemmCmd.weight_load_bytes`
自动取本 PU 的 B×K×N×weight_dtype_bytes，`shared_weight=True` 时不乘 B。
默认权重 1 byte、输入输出各 2 bytes，可分别配置。权重端口传输与计算重叠，
耗时取较大值；已有 mapping 内存流量时沿用显式流量，避免重复计费。

兼容边界：现有模型 `HardwareMapper` 仍提供预切分的单 PU shape 和独立通信
节点，因此它创建的 BMM 显式选择 `none/none`、空链，并关闭自动权重加载。
这些历史模型策略的通信公式尚未迁移为 PartitionChain；新完整逻辑 BMM 使用上面
的接口直接交给 `TreeParser`。从旧策略迁移时应同时提供完整 shape、真实切分链，
并删除对应的显式输入输出通信节点，避免双重切分或重复通信。

### PartitionChain 驱动的 FlashAttention lowering

`FlashAttentionOp` 接收完整逻辑 shape，只允许对 batch `B` 和 KV sequence
length `S` 进行 PU 切分。`S = qk_N = sv_K` 必须相等；`M`、`qk_K` 和
`sv_N` 保持完整。空链表示单 PU，microbatch 缩放先于 PU 切分。

```python
from srammachine.frontend.modules import FlashAttentionOp, PartitionChain

op = FlashAttentionOp(
    "attention", B=5, M=2, qk_K=7, qk_N=9, sv_K=9, sv_N=3,
    partition_chain=PartitionChain((("B", "p", 2), ("S", "q", 4))),
)
assert op.communication_volumes()["input_bytes_per_row"] == (84, 56)
assert op.communication_volumes()["output_bytes_per_row"] == (36, 24)
```

该例两行的 B 范围为 `[0, 3)` 和 `[3, 5)`；每行四列的 S 范围为
`[0, 3)`、`[3, 6)`、`[6, 9)`、`[9, 9)`，最后一列为空。
代表 PU 的两个 GEMM 使用 `B=3, M=2, qk_K=7, qk_N=sv_K=3, sv_N=3`。

- 输入是 Q `[B, M, qk_K]`：同一行对有效 B 范围取并集，B 切分 scatter，
  S 切分 broadcast。空 PU 不贡献输入量，不同行不去重。
- 输出是 O `[B, M, sv_N]`：同一行的不同 B 分片 gather，相同 B 的 S 分片
  先做 online-softmax 状态合并，再从该行向 IO die 发送一次。
- 对每个有效 B 区间，若该行覆盖全部 S，则发送最终 O；若只覆盖部分 S，
  则发送 `(m, l, u)`，由 IO die 继续合并并归一化。这里 m 是 score 最大值，
  l 是 `sum(exp(score-m))`，u 是 `sum(exp(score-m)*V)`，最终 O 为 `u/l`。
  不能对局部归一化 O 直接求和。合并状态 1/2 时取 `m=max(m1,m2)`，
  再分别以 `exp(m1-m)`、`exp(m2-m)` 加权相加 l 和 u。

Q 和最终 O 默认各 2 bytes；状态 u 向量及 m/l 标量默认各 4 bytes，分别通过
`input_dtype_bytes`、`output_dtype_bytes`、`state_dtype_bytes` 配置。
每行的完整 batch 数 c 和部分 batch 数 t 对应：

```text
input_bytes  = (c+t) * M * qk_K * input_dtype_bytes
output_bytes = M * (c * sv_N * output_dtype_bytes
                   + t * (sv_N+2) * state_dtype_bytes)
```

例如改为 `(("S", "p", 2), ("B", "q", 2))`，两行分别覆盖 S `[0,5)`、
`[5,9)`，每行 gather 两个 batch 分片并发送 5 个 batch 的状态。
输入为 `(140, 140)` bytes，输出为 `(200, 200)` bytes。
同一行增加更多 S 分片不会重复发送同一 batch 的状态。

默认生成 `.input -> .core -> .output`；`input_source` 和
`output_destination` 均支持 `io_die`（默认）或 `none`，本次不提供纵向归约。
`none` 仅省略通信，部分状态仍需由调用者负责合并。
IO 命令保留每行字节数，以最大行字节数按单条横向链路带宽计时。
输出 trace 记录完整/部分 batch 数和 Attention 状态合并语义。

本模型计算归约后的 IO 载荷，不单独模拟行内合并和 IO die 合并的算术延迟。
KV 加载仍由显式 OperatorMapping 提供，FlashAttention 计算计时保持现有模型。
HardwareMapper 尚未迁移：其预切分 shape、独立通信节点不能直接与新默认自动
通信叠加使用，后续重构需同时迁移完整 shape、切分链和显式通信。

### PartitionChain 驱动的 FusedIndexerScore top-k 输出

`FusedIndexerScoreOp` 保留 `B/M/K/N` 构造参数，其中 B 是 batch，M 是
indexer head 数，K 是 head dimension，N 是 sequence length。
切分链只接受 `B/S`，其中 `S` 映射到 N；不允许切分 M/K，也不用 N 作为
切分链维度名。每个 PU 执行完整的 head reduction 后，独立对每个 batch
取 top-k。`top_k` 默认 2048，可显式配置。

```python
from srammachine.frontend.modules import FusedIndexerScoreOp, PartitionChain

op = FusedIndexerScoreOp(
    "indexer", B=5, M=3, K=7, N=9, top_k=2,
    partition_chain=PartitionChain((("B", "p", 2), ("S", "q", 4))),
)
assert op.communication_volumes()["input_bytes_per_row"] == (126, 84)
assert op.communication_volumes()["topk_candidates_per_row"] == (6, 4)
assert op.communication_volumes()["output_bytes_per_row"] == (36, 24)
```

输入按行对 Q 的 B 范围去重：S 切分 broadcast，B 切分 scatter，输入大小为
有效 batch 数乘 `M*K*input_dtype_bytes`。每个 PU 输出的候选数量为
`local_B * min(top_k, local_S)`，而不是对该 PU 的所有 batch 一起取 k 个。

同一行相同 batch 的候选先合并并再次截断，再向 IO die 发送。对该行每个
有效 B 区间，设 batch 数为 b、S 范围并集的元素数为 s，则：

```text
row_candidates += b * min(top_k, s)
row_output_bytes = row_candidates * (output_dtype_bytes + index_dtype_bytes)
```

这里不能把每个 PU 的候选数量直接相加，也不能把不同 batch 一起截断到 k。
当每个 PU 的局部 S 少于 k 时，合并后的数量可以增长，但每个 batch 最多 k 个。
输出已完成 head reduction，因此不乘 M。空 tile 不贡献候选或输入通信。

候选携带 score 和全局 sequence index，默认分别为 2 bytes 和 4 bytes；
可通过 `output_dtype_bytes` 和 `index_dtype_bytes` 配置，索引宽度必须足够
表示 `[0, N)`。即使某行覆盖完整 S，当前接口也统一发送 score/index 对。
跨行序列分片由 IO die 继续做相同的 top-k merge，每个 batch 最终仍最多 k 个。
逐级合并需使用一致的排序规则，例如 score 降序、全局 index 升序处理同分。

默认生成 `.input -> .core -> .output`，支持 `io_die` 和 `none` 路由。
沿用每行并行、以最大行字节数计时的 IO 模型，trace 记录 `top_k`、每行
候选数及 `per_batch_topk_merge` 语义。`none` 将后续候选合并交由调用者负责。

本次建模 top-k 的候选量和截断效应，不单独计入选择或合并的计算延迟。
原有 GEMM/ReLU/head reduction 计时与加载策略保持不变。
`main_gemm_flops_override` 仅保留用于空切分链；非空切分链必须由局部 shape
推导 FLOPs，避免把旧的单 PU 覆盖值应用到新的完整逻辑 shape。
HardwareMapper 的预切分及独立通信路径暂未迁移。
