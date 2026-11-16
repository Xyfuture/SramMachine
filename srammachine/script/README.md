# SramMachine 实验脚本使用指南

**当前禁用 SplitTree 搜索。** `run_sa` 保留历史入口名称，默认只执行
PartitionChain 枚举、顺序仿真和 trace 导出，无需提供任何 SA 轮数。
Python 设置 `SimulatedAnnealingConfig(enabled=True)`、CLI 指定
`--split-tree-search`，或提供非零 `--rounds` / `--warmup-rounds` / `--restarts`，
都会立即抛出 `NotImplementedError`。PartitionChain 枚举仍默认开启。

当前批量运行示例：

```bash
.venv/bin/python -m srammachine.script.run_sa --models deepseek-v3.2 --mtp on --batch-sizes 64 128 192 256 320 384 448 512 --input-sequence-length 32768 --output-sequence-length 600
```

下方保留的 SA 轮数、温度、restart 和 SplitTree 搜索说明属于历史接口文档，
不能用于当前运行。

## MiniMax M3 快速运行

M3 使用独立的 MSA/Indexer 实现，投影与专家计算复用 BMM。默认结果按
57 层 MSA/MoE 与 3 层 dense GQA/FFN 分别按固定顺序仿真后加权，
dense 前缀使用固定顺序 baseline。Vector 操作暂不计时。详细语义及近似见
[Mapper 文档](../mapping/README.md#minimax-m3)。

默认枚举 MiniMax 代表层的 PartitionChain：BMM 使用 `B/M/K/N`，MSA 使用
`B/H/S`（H 为完整 KV head group），Indexer 使用 `B/S` 并按 block top-k
通信量评分。支持普通版、XFold、TP/EP、MTP 和 FP8/FP16，添加
`--no-partition-optimization` 可关闭。完整 decoder 估算附带的 dense 前缀仍用
原始 baseline；单独指定 `--minimax-layer-type dense` 导出时，会枚举该 dense 层。

```bash
.venv/bin/python -m srammachine.script.export_trace --model minimax-m3 --batch-size 256 --input-sequence-length 20000
.venv/bin/python -m srammachine.script.export_trace --model minimax-m3 --minimax-layer-type dense --batch-size 256 --input-sequence-length 20000
.venv/bin/python -m srammachine.script.run_sa --models minimax-m3 --mtp off --batch-sizes 256 --input-sequence-length 20000 --output-sequence-length 600 --moe-strategy tp
```

第一条导出 MSA 代表层 trace 并报告完整 decoder 的加权估算；第二条仅检查
dense 前缀。支持 `fp8/fp16` KV、TP/EP 和现有理想 MTP1。添加 `--xfold` 可启用
MiniMax 的 512 维 index-query folding：包含 residual/shared/routed contribution，
在 routed FC1 后提前执行分组 QK 和 Block Top-16，再重叠 DRAM FC2 与 Flash KV
预取。vector 操作不计时，dense 前缀仍走普通路径。

```bash
.venv/bin/python -m srammachine.script.export_trace --model minimax-m3 --xfold --batch-size 256 --input-sequence-length 20000
.venv/bin/python -m srammachine.script.run_sa --models minimax-m3 --xfold --mtp off --batch-sizes 256 --input-sequence-length 20000 --output-sequence-length 600 --moe-strategy ep
```

稳态估算及矩阵形状见 [MiniMax XFold](../mapping/README.md#minimax-xfold)。
`--minimax-layer-type dense` 不接受 `--xfold`。

## 项目简介

SramMachine 是面向 DRAM/Flash 直连 PU 的 AI 加速器的离散事件模拟器。目前主要用于研究 decoding 阶段中 Attention、MoE、存储和片上/片间通信的延迟，以及 SplitTree microbatch 对流水并行效果的影响。

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
- `minimax-m3`
- `deepseek-v3.2`
- `kimi-k2.5`
- `glm-5.1`

模拟器支持 MoE TP/EP、MTP1、FP8/FP16 KV cache，以及 PartitionChain 枚举优化。TP 与 EP 可以在同一次脚本运行中比较，并导出各 workload 的指标、布局和 Perfetto trace。SplitTree 搜索当前禁用。

## 默认 PartitionChain 枚举优化

`SplitTreeOptimizer`、`run_sa` 和 `export_trace` 默认对 DeepSeek V3 / V3.2 / GLM 5.1 / MiniMax M3
启用 PartitionChain 枚举，支持普通版、Fold、TP/EP 和 FP8/FP16。
DeepSeek V3 不含 Indexer，完整读取 KV cache；即使传入 `--xfold` 也执行普通版，
结果记录 `xfold=false`。Flash 带宽使用默认 CSV 的 V3 `dense_kv` 数据。

```bash
.venv/bin/python -m srammachine.script.run_sa --models deepseek-v3 --xfold --mtp off on --batch-sizes 256 --input-sequence-length 32768 --output-sequence-length 600 --moe-strategy ep tp
```

每个 `(batch, MTP)` workload 在完成 chip 级映射及 HBF 带宽查表后，先枚举
各 Op 的片内布局，再固定这些布局顺序执行一次。每个 workload 只枚举一次，
不启动 SplitTree 搜索、不生成邻居，也不执行 restart。
其他模型和直接调用 Mapper 的行为保持原样。

BMM 枚举 B/M/K/N 在 p、q 两方向的 2 的幂次因子分配，MLA/DSA Attention 与
DSA/MiniMax Indexer 枚举 B/S，MiniMax MSA 枚举 B/H/S。
候选先切 p 再切 q，覆盖完整 32×64 PU 网格，支持不能整除时的裁剪。
按 `输入 NoC 时间 + max(计算时间, DRAM/Flash 并行加载时间) + 输出 NoC 时间`
选择最低成本布局，保留原布局参与比较；BMM 的最佳三个候选通过 TreeParser
精确通信量独立核对。Flash 使用每个算子的实际带宽。XFold 的 KV 预取作为
独立 Flash 指令参与资源竞争，不计入 FC2 的 PU 时间或布局评分；其完成约束
下一层 Core Attention 的输出。枚举优化各算子的局部执行成本，
也不代表候选范围以外的全局最优。

关闭枚举以使用 Mapper 原始布局：

```bash
.venv/bin/python -m srammachine.script.run_sa --models deepseek-v3.2 --mtp on --batch-sizes 256 --input-sequence-length 32768 --output-sequence-length 600 --no-partition-optimization
.venv/bin/python -m srammachine.script.export_trace --model deepseek-v3.2 --batch-size 256 --input-sequence-length 32768 --mtp --no-partition-optimization
```

Python API 使用 `SimulatedAnnealingConfig(optimize_partition_chains=False)` 关闭。
`enabled=False` 是默认值；设置 `enabled=True` 会立即抛出 `NotImplementedError`。
批量入口的 `baseline_*` 指标来自 Mapper 原始布局，`pareto_best_*` 来自枚举后的
顺序执行结果；关闭枚举时两者相同。所有 SA proposal/restart/接受计数均为 0。

结果 JSON 和 case checkpoint 保存 `partition_chains`，每棵树的 record 也携带该字段；
CSV 记录 `optimize_partition_chains`，完整布局在它引用的 JSON 内。
Trace 重放使用保存的布局，不重新枚举，并核对记录的仿真延迟。
旧文件没有布局字段时保留原 Mapper 重放方式。Python 重放示例：

```python
from srammachine.search import (
    apply_partition_chains, load_pareto_partition_chains, load_pareto_split_tree,
)

# mapped 必须使用与结果相同的模型、推理配置、硬件及 HBF 带宽表重建。
mapped = apply_partition_chains(
    mapped, load_pareto_partition_chains(result_path), mapper.hardware_config,
)
tree = load_pareto_split_tree(result_path)
graph = TreeParser(mapper.hardware_config).parse(
    tree, mapped.operators, mapped.operator_mappings, layer_count=4,
)
```

## Nexus 默认硬件

| 项目 | 默认值 |
| --- | --- |
| 系统 | 8 Chip；Attention 使用 DP，FFN/MoE 使用 EP |
| 每 Chip PU 网格 | 32×64，共 2048 PU |
| 每 PU 脉动阵列 | 4 个 16×16 阵列，1 GHz |
| 每 PU DRAM / Flash 带宽 | 16 GB/s / 4 GB/s |
| 片内互联 | 2D mesh，全双工；每条链路每方向 128 GB/s |

带宽采用十进制 GB（10⁹ bytes）。双源权重加载默认按 DRAM:Flash = 4:1
分配，理想并行带宽为 20 GB/s；算子可覆盖 Flash 带宽。
当前配置层次保留每 Chip 2×2 logic die，各 die 为 16×32 PU，拼成 32×64
mesh。未另行指定的容量、向量单元和 Chip 间 Fabric 参数沿用现有设置。

DeepSeek V3 和 DSA 模型的片内切分见 [Mapper 说明](../mapping/README.md)。Kimi K2.5
在 Chip 间使用 Attention DP / MoE EP，片内投影按 K32×N64 切分；显式 TP
模式额外按 Chip 数切分 intermediate 维度。原 16-Chip、8×8 mesh 实验记录
见 [历史 MoE TP/EP 映射说明](MOE_PURE_TP_EXPERIMENT.md)，不代表当前默认值。

## HBF 实测带宽表

`run_sa` 和 `export_trace` 默认读取仓库内的
[`hbf_read_bandwidth.csv`](../model_cards/hbf_read_bandwidth.csv)。每次运行加载一次，
根据模型、**global batch size**、MTP 和 `input_sequence_length`，为 Core Attention
的 KV 读取和 indexer 设置 Flash 带宽。SA 拆分 microbatch 后继续使用原 global batch
对应的带宽。普通权重加载仍使用其原有内存策略。

原始 `mean_GBps` 是单 chip 的总带宽，采用十进制 GB；导出的
`flash_bandwidth_bytes_per_second = mean_GBps * 1e9 / 2048` 是每 PU 带宽。
表中保留测量元数据，新增 `op_id`、`bandwidth_scope=chip`、`pu_count=2048`
和原始名称。`Kimi-K2` 对齐为 `kimi-k2.5`；`GLM-5.2` 保留为 `glm-5.2`，
仅作为数据保留，尚未添加模型仿真支持。

查表规则：

- 精确命中时直接取值；否则沿 batch、sequence length 两轴做线性/双线性插值，
  使用原数值坐标，不使用对数坐标。任何一轴越界或缺少插值角点都报错。
- 当前表覆盖 global batch **16–4096**，sequence length **16384–65536**。
- `dense_kv`、`dense_indexer` 缺少 MTP=1 系列时沿用 MTP=0；稀疏 KV 分别查 MTP=0/1。
- MiniMax 的 dense attention 层使用硬件默认 Flash 带宽并发出提示，不套用 sparse 测量。
- Fold 模式下，保留的 indexer score 使用 indexer 数据（DeepSeek V3.2/MiniMax
  仅 `moe.indexer_score`，GLM 5.1 还包括 `attn.indexer_score`），FC2 的 Flash KV 预取使用 sparse KV
  数据；Core Attention 仍读 DRAM。DeepSeek-V3/Kimi 的旧 KV 路径转换为每 PU Flash
  读取，按每请求唯一 KV 数据量均分到 2048 个 PU，不按 head 或 MTP query 重复读取。
- 表格没有 dtype 维度，FP8/FP16 使用相同测量带宽；实际读取字节数仍由 KV dtype 决定。

指定另一份规范化表，或关闭实测带宽：

```bash
.venv/bin/python -m srammachine.script.run_sa --models deepseek-v3.2 --mtp off on --batch-sizes 256 --rounds 10 --input-sequence-length 20000 --output-sequence-length 600 --hbf-bandwidth-csv /home/xyfuture/nex/SramMachine/HBF-read-bandwidth-results-normalized.csv
.venv/bin/python -m srammachine.script.export_trace --model deepseek-v3 --no-hbf-bandwidth
```

SA 的 CSV/JSON 记录带宽表路径，搜索结果的 trace 重建时也使用该路径。旧结果未记录
此字段时按旧配置重建。Python API 显式启用：

```python
from srammachine.mapping import DEFAULT_HBF_BANDWIDTH_CSV, HBFBandwidthTable, mapper_for_model
from srammachine.inference import InferenceConfig

table = HBFBandwidthTable.from_csv(DEFAULT_HBF_BANDWIDTH_CSV)
mapped = mapper_for_model("deepseek-v3.2").map(
    InferenceConfig(256, 20000, 600), hbf_bandwidth_table=table,
)
# SplitTreeOptimizer(hbf_bandwidth_csv=DEFAULT_HBF_BANDWIDTH_CSV) 同样支持。
```

重新规范化原始测量文件（默认输出到源文件旁的 `*-normalized.csv`）：

```bash
.venv/bin/python -m srammachine.script.normalize_hbf_csv /home/xyfuture/nex/SramMachine/HBF-read-bandwidth-results.csv
```

## 并行运行与历史 SA 说明

当前 worker 仅运行顺序仿真。下文涉及 SA 预热、restart、搜索轮数和温度的说明
均为历史记录；非零轮数会触发 `NotImplementedError`。进程隔离、checkpoint、CSV
和 trace 的保存方式继续适用，文件名中的 `sa` / `splittree` 前后缀保留兼容。

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

下面历史实验示例中如包含 batch 8192，已超出当前 HBF 表的 4096 上限；
需要补充测量数据，或显式添加 `--no-hbf-bandwidth` 运行原有带宽配置。

### 历史全量 400-round 测试规则（当前禁用）

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

可选参数及默认值：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--moe-strategy` | `ep` | 一个或多个 MoE 并行策略：`tp`、`ep` |
| `--kv-cache-dtype` | `fp8` | KV cache 数据类型，可选 `fp8`/`fp16` |
| `--partition-optimization` | 开启 | V3/DSA/MiniMax workload 仿真前枚举 PartitionChain；用 `--no-partition-optimization` 关闭 |
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
- 默认8 chips、top-k=8时，EP的MTP off/on最低均为`BS=1`。
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
相邻算子和跨层依赖均从输出完成连接到下一个输入开始。权重通过 `PUMemoryReadCmd`
从 DRAM 或 Flash 直接流入 PU，不经过 SRAM，也不受 SRAM 容量限制。
权重字节数取本 PU 的 B×K×N×weight_dtype_bytes，`shared_weight=True` 时不乘 B。
未指定 `weight_memory` 的自动权重加载默认使用 DRAM；显式策略支持 DRAM、Flash
或两者同时使用。双端口按算子的实际带宽比例分配数据，并行耗时为 W/(B_dram+B_flash)。
旧映射使用 `weight_read_fixed_bytes` 指定 PU 权重字节数，不再生成额外的预取或中转读取。
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

`FusedIndexerScoreOp` 保留 `B/M/K/N` 构造参数，其中 B 是请求 batch，M 是
`queries_per_request * indexer_heads`，K 是 head dimension，N 是 sequence length。
`queries_per_request` 默认 1；MTP1 使用 2，同请求的 query 共享缓存，
每个 query 的 head reduction 和 top-k 独立执行，M 必须能被 query 数整除。
切分链只接受 `B/S`，其中 `S` 映射到 N；不允许切分 M/K，也不用 N 作为
切分链维度名。每个 PU 执行各 query 的 head reduction 后，独立对每个请求/query
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
`local_B * queries_per_request * min(top_k, local_S)`，而不是对该 PU 的所有 batch 一起取 k 个。

同一行相同请求/query 的候选先合并并再次截断，再向 IO die 发送。对该行每个
有效 B 区间，设 batch 数为 b、S 范围并集的元素数为 s，则：

```text
row_candidates += b * queries_per_request * min(top_k, s)
row_output_bytes = row_candidates * (output_dtype_bytes + index_dtype_bytes)
```

这里不能把每个 PU 的候选数量直接相加，也不能把不同 batch 一起截断到 k。
当每个 PU 的局部 S 少于 k 时，合并后的数量可以增长，但每个请求/query 最多 k 个。
输出已完成每个 query 的 head reduction，因此乘 query 数、不乘 M。空 tile 不贡献候选或输入通信。

候选携带 score 和全局 sequence index，默认分别为 2 bytes 和 4 bytes；
可通过 `output_dtype_bytes` 和 `index_dtype_bytes` 配置，索引宽度必须足够
表示 `[0, N)`。即使某行覆盖完整 S，当前接口也统一发送 score/index 对。
跨行序列分片由 IO die 继续做相同的 top-k merge，每个请求/query 最终仍最多 k 个。
逐级合并需使用一致的排序规则，例如 score 降序、全局 index 升序处理同分。

默认生成 `.input -> .core -> .output`，支持 `io_die` 和 `none` 路由。
沿用每行并行、以最大行字节数计时的 IO 模型，trace 记录 `top_k`、每行
候选数及 `per_batch_topk_merge` 语义。`none` 将后续候选合并交由调用者负责。

本次建模 top-k 的候选量和截断效应，不单独计入选择或合并的计算延迟。
原有 GEMM/ReLU/head reduction 计时与加载策略保持不变。
`main_gemm_flops_override` 仅保留用于空切分链；非空切分链必须由局部 shape
推导 FLOPs，避免把旧的单 PU 覆盖值应用到新的完整逻辑 shape。
HardwareMapper 的预切分及独立通信路径暂未迁移。

### DSA Mapper / Fold

DeepSeek V3.2 和 GLM 5.1 现在通过各自的 Mapper 生成顺序 Op 列表，默认使用
8 chips，每 Chip 使用 32×64 PU 切分。`--xfold` 启用 `FusedFoldingMoE`，包括下一层
indexer 预测，以及 DRAM FC2 与 Flash KV 预取重叠。MiniMax M3 使用专用
`MiniMaxFoldingMoE`；DeepSeek V3 接受该开关并回退到普通版，Kimi K2.5 不支持。
V3/DSA 路径默认枚举片内 PartitionChain，跨 Chip 默认 Attention DP、MoE EP；
`--moe-strategy tp` 可显式切换为跨 Chip MoE TP。
接口、形状、带宽参数和模拟口径见 [mapping/README.md](../mapping/README.md)。

### 直接导出 Perfetto trace

无需运行 SplitTree SA，可直接导出顺序执行基线的普通版或 Fold 版；
V3/DSA/MiniMax 默认先枚举 PartitionChain，添加 `--no-partition-optimization` 使用原始布局：

```bash
.venv/bin/python -m srammachine.script.export_trace
.venv/bin/python -m srammachine.script.export_trace --xfold
```

默认配置为 DeepSeek V3.2、batch 256、输入长度 20000、输出长度 600、FP8 KV、
MTP 关闭，模拟连续 4 层。使用 `--help` 查看模型、batch、序列长度、层数等参数。
文件写入 `test result/perfetto/`，名称包含模型、版本、配置和 `baseline`，不覆盖已有文件。
将 JSON 加载到 Perfetto 的 Open trace file 中即可分析。

PU 计算、DRAM、Flash、NoC 各有独立轨道。内存事件名称标明端口、读取用途、
数据量和带宽；Fold 的独立 Flash 预取标记为 `next_layer_kv`，名称包含
`kv_prefetch L0->L1` 等来源层和目标层。它与 FC2 core 共享前序依赖，
与下一层 Core Attention core 共享后续依赖；FC2 输出不等待预取完成。
XFold 的 PreAttention、中间 Indexer 和 Core Attention 默认全部只读 DRAM，
为跨层 KV 预取保留 Flash 端口；PostAttention 和 MoE 沿用各自的内存策略。
不同 microbatch 按 token 范围的交集关联。首层保留初始 KV 已位于 DRAM 的
稳态假设；末层的 look-ahead 读取仍显示，但在当前有限图内没有消费者。
Flash 实际读取时间由数据量和带宽决定，端口仍与普通权重读取竞争。
点击事件可查看 `component`、`partition_chain`、`global_dimensions`、
`mapped_pu_dimensions`、`weight_source`、层号和 token 范围。映射维度是完整
batch 的计划形状，计算事件自身的 B/M/K/N 则是该微批次实际执行的形状。
JSON 的 `otherData` 保存实验配置；事件时间戳使用微秒，args 中保留纳秒值。

`Simulator.run_and_trace(..., mapping_result=mapped)` 会自动携带映射信息；
已有 `SimulationResult` 也可以调用
`export_perfetto_trace(result, mapping_result=mapped)`。
`run_sa` 导出的搜索结果 trace 同样包含这些信息。
