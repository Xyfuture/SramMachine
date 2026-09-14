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

MoE 的当前映射规则为：

- TP：小 batch（256-expert 模型 BS≤512、Kimi BS<1536）沿用 TP16。
  大 batch 使用固定 token-group × chip-TP：256-expert 模型依次为
  G2×TP8、G4×TP4、G8×TP2，Kimi 的对应阈值为1536、3072、6144。
  group 间切 token，group 内 chip 沿 intermediate 维做 TP，chip 内4个dies
  按expert划分。TP2 在权重超过每die SRAM时分两个expert权重波次执行。
- EP：expert 在 chips 间保持 EP16。DeepSeek-V3、DeepSeek-V3.2 和 GLM-5.1 在 BS≥2048 时使用 hierarchical EP：每 chip 的16个experts连续分给4个dies，每个expert使用一行4个PU；其他情况沿用原 chip 内 die-TP mapping。
- 小 batch 允许只激活部分experts；inactive expert不产生weight或计算命令。

## 并行 SA 脚本

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
| `--input-sequence-length` | decoding 时使用的历史上下文长度 |
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

基础 seed 会结合 model、MoE 策略、batch 和 MTP，确定性地产生每个 case 的独立 seed。因此改变 worker 数量或任务完成顺序不会改变搜索结果。

## 输出结果

一次完整运行会产生：

1. 一份跨模型、MoE策略、batch和MTP的汇总CSV。
2. 每个 `(model, MoE strategy, MTP)` 一份包含最优SplitTree的JSON；上述全量配置共16份。
3. 每个case选择一棵最优SplitTree重放并导出一份Perfetto trace；上述全量配置共144份。
4. `case checkpoints/<CSV文件名>/`目录中每个已完成case一份可独立重放的SplitTree JSON；上述全量配置共144份。

CSV在worker启动前就会创建。每个worker完成SA后，会先原子写入该case的SplitTree checkpoint，再生成并原子写入Perfetto trace；两者均成功后才向主进程报告case完成。主进程随后将完整指标、路径和PU利用率写入CSV并强制刷新。因此程序被终止或某个case失败时，已经报告完成的case仍保留CSV、SplitTree和trace。若恰好在某个case的trace重放期间退出，该case尚未记入CSV，但已经写好的SplitTree checkpoint仍会保留。部分CSV的`is_model_global_pareto`字段为空；只有全部case成功后，该文件才会写入最终Pareto标记。

自动文件名以本地时间精确到分钟，并在重名时追加序号，不会覆盖已有结果。

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
- case 数量等于 `模型数 × MoE策略数 × batch数 × MTP状态数`。
- 默认会使用尽可能多的逻辑核心；大型搜索也会占用较多内存，必要时使用 `--workers` 限制并发。
- 不同 case 位于独立进程中，不共享 Desim 状态或 SA 搜索轨迹。
- 所有case在进入进程池前按BS全局降序排列：全部最大BS先入队，再依次进入较小BS；不设置批次间屏障，空闲worker会立即领取队列中的下一个case。
- worker在每个case结束后都会回收，因此已完成case不会持续累积内存。峰值内存通常约为`并发worker数 × 单个最大case内存`；若峰值过高，应降低`--workers`。
- 一个命令可以同时指定`--moe-strategy tp ep`；汇总CSV合并两种策略，最终JSON按模型、策略和MTP分别导出。
- 显式指定的 `--output-csv` 如果已经存在，脚本会拒绝覆盖。
- 每个case搜索完成后，脚本立即保存独立SplitTree checkpoint、重放该树并生成Perfetto trace和稳态PU利用率；trace阶段不增加SA搜索轮数。
- 中断后，已完成case的CSV、独立SplitTree checkpoint和trace均保留；正在运行的case仍需重新搜索，脚本暂不自动续跑。
