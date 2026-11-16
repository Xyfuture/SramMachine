# 模型 Mapper 与顺序 Op 列表

每个模型对应一个 Mapper：`DeepSeekV3Mapper`、`DeepSeekV32Mapper`、
`KimiK25Mapper`、`GLM51Mapper`、`MiniMaxM3Mapper`。`HardwareMapper.map(request)` 是兼容入口，
按模型名分发。DeepSeek V3 / V3.2 和 GLM 5.1 使用显式 PartitionChain 组件流程；
Kimi K2.5 继续使用原来的 MLA 实现；M3 使用独立 MSA
实现。旧实验可显式使用 `LegacyHardwareMapper`。

DeepSeek V3 / V3.2、GLM 5.1 和 MiniMax M3 的 BMM 输入为 1 B、输出为
2 B、模型权重为 1 B，覆盖普通版、XFold 以及 M3 dense 前缀。Indexer
输出候选统一使用 2 B score 和 4 B index，每个候选共 6 B；读取的历史
index key 仍跟随 KV cache 的 FP8/FP16 配置。EP combine 使用 BMM 输出
的 2 B 宽度。FlashAttention / FlashMSA 最终输出为 1 B，中间状态为 4 B。

## DeepSeek V3

V3 复用 V3.2 普通版的 `PreAttn → CoreAttn → PostAttn → FusedMoE`，
没有 Indexer 或 Fold 算子。`qk_nope` 与 `qk_absorb` 分开计算，MoE FC1
融合 gate/up。Core Attention 的 `qk_N`、`sv_K` 均为完整输入上下文长度，
不做 top-k 截断；KV 通过 Flash 读取，使用 CSV 中 V3 的 `dense_kv` 带宽。
MTP 将 query heads 打包到 M 维，同一请求的 KV 只读一次。

`map(..., xfold=True)`、`build_op_list(..., xfold=True)` 和 CLI `--xfold`
均自动运行普通版，映射结果、实验记录及 trace 的 `xfold` 为 false。
支持 TP/EP、FP8/FP16，以及 PartitionChain 枚举、保存和重放；BMM 枚举
`B/M/K/N`，完整 Attention 枚举 `B/S`，与 V3.2 使用相同的 32×64 PU 网格。

## XFold 跨层 KV 预取

XFold 默认将 PreAttention 到 Core Attention（含中间 Indexer）的全部算子
设为仅从 DRAM 读取；普通路径保持原策略。PostAttention 和 MoE 的内存策略
独立设置。该规则同时适用于 DeepSeek V3.2、GLM 5.1 和 MiniMax M3；
MiniMax 的普通 dense 前缀不受影响。显式 `memory_overrides` 仍可覆盖单算子策略。

仅 XFold 路径设置 `flash_prefetch_bytes_per_request`。该流量会生成独立的
Flash `PUMemoryReadCmd`，与 FC2 core 共享前序依赖，并与下一层
Core Attention core 共享后续依赖。FC2 的 PU 耗时和输出均不等待 KV 预取。
预取耗时仍按实际 Flash 带宽计算，与普通权重读取竞争同一端口；读取完成后
立即释放端口。Attention 的输出须同时等待其自身计算、DRAM 读取和前层预取。
这是算子级的理想流式重叠，未逐块模拟 KV 到达与 Attention 的计算进度。

不同 microbatch 的生产者和消费者按 token 范围交集关联。首层沿用初始 KV
已在 DRAM 的稳态假设；末层保留下一层的 look-ahead 读取，但有限图内没有
对应消费者。PartitionChain 枚举只优化 FC2 自身成本，不把独立预取计入
FC2 的局部评分；跨层等待和 Flash 端口竞争由完整图仿真计入。

## MiniMax M3

实现文件：`mapping/minimax_config.py`、`mapping/minimax.py`、
`frontend/modules/minimax.py` 和 `commands/minimax.py`。
配置读取仓库模型卡的 `text_config`；依据
[官方配置](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json) 与
[MSA 论文 §3.1](https://arxiv.org/html/2606.13392v2#S3.SS1)。

默认 `mapper_for_model("minimax-m3")` 返回 MSA/MoE 代表层。QKV、输出、
Indexer Q/K、路由、共享专家和 routed experts 的投影全部使用 `BMMOp`。
MSA 分支顺序为 `MiniMaxPreAttn → MiniMaxIndexer → MiniMaxCoreAttn →
MiniMaxPostAttn → MiniMaxSharedFFN → FusedMoE`；TP/EP 沿用现有映射。
`xfold=True` 启用 MiniMax 专用的跨层 index-query folding，见下文。

令 `B` 为请求数、`t` 为每请求 query 数、`T=B*t`，全局形状如下：

| Op | 形状／语义 |
| --- | --- |
| `attn.qkv_projection` | BMM `(1,T,6144,9216)`，64 个 Q heads、4 个 K/V heads，head dim 128 |
| `attn.indexer_q` | BMM `(1,T,6144,512)`，每个 GQA 组一个 128 维 query |
| `attn.indexer_k` | BMM `(1,T,6144,128)`，组间共享一个 index K head |
| `attn.indexer_score` | `MiniMaxFusedIndexerScoreOp(B,4*t,128,history)`，token QK → block max → 每组独立 top-16 blocks |
| `attn.flash_msa` | `FlashMSAOp`：`B=B,H=4,M=16*t`，QK/SV 的 head dim 128，`S=min(history,16*128)` |
| `attn.output_projection` | BMM `(1,T,8192,6144)` |
| MoE | 128 个 routed experts、top-4、每专家宽度 3072，另有一个宽度 3072 的 shared expert |

`FlashMSAOp` 的 `H` 是 **KV head / GQA 组**，`M` 是组内 query heads × query
tokens；`PartitionChain` 支持 `B/H/S`，不会把不同 H 的结果归约到一起。
Mapper 初始切分为 `B-p-2 → B-q-4 → H-p-4 → S-p-4 → S-q-16`。
每 PU 的 KV 读取量为 `B_local * H_local * S_local * (d_k+d_v) * dtype_bytes`。
同组共享 KV，多 query（理想 MTP1）增加计算量，不重复读取同一份 cache。

Indexer Score 初始使用 `B-p-2 → B-q-4 → S-p-16 → S-q-16`，不允许 head/K
切分；没有 DSA 的 ReLU、head 权重投影或跨 head 聚合。候选按 query/组独立，
输出 `(score, global block index)`，跨 shard 的同一 block 先按 max 合并。
local block 占 top-k 的一个名额。共享 index K 的读取量不乘 index head 数。

搜索/实验入口默认在映射后枚举 PartitionChain：BMM 枚举 `B/M/K/N`，MSA
枚举 `B/H/S`，Indexer 枚举 `B/S`。候选覆盖完整 PU 网格，按实际通信、计算
及单算子 DRAM/Flash 带宽成本选择；支持普通版和 XFold，并保存布局供 trace
重放。直接调用 Mapper 仍返回上述初始布局。详见
[枚举说明](../script/README.md#默认-partitionchain-枚举优化)。

普通权重默认 DRAM+Flash；MSA KV 和 Indexer K 默认 Flash。可通过
`memory_overrides={op_id: WeightMemory(...)}` 指定单一/双带宽源和单算子 Flash
速率；双源数据按带宽比例分配。shared QKV/output 的覆盖策略也应用于 dense 前缀。

模型卡中前 3 层为 dense GQA+FFN，后 57 层为 MSA+MoE。默认仿真结果使用
`57 * sparse_layer_time + 3 * dense_layer_time`；dense 层有独立的顺序 baseline，
PartitionChain 枚举优化 sparse 代表层，dense 项保持不变；SplitTree 搜索当前禁用。
`result.model_latency_components`
及 Perfetto `otherData` 保存分项。`MiniMaxM3Mapper(layer_type="dense")` 可单独
检查并枚举 dense 前缀，此时 `latency_ns` 只包含 3 层。Trace 中的 command 只对应所选
代表层，分项估算不是实际执行 60 层的 trace，也不估算异类层交界的流水效应。

范围与近似：仅文本 decoder 的 decode；不包含视觉 encoder、prefill、训练。
Vector 操作（RoPE、norm、激活、softmax、block-max/top-k）不计时，但其 shape/
选块语义用于通信与缓存量。稀疏长度使用 `min(history,k*block_size)` 上界，
没有预测实际选块分布、尾块利用率或跨 query 稀疏重叠。MTP 开关沿用全项目的
理想 MTP1（两个 query 均接受），不模拟 M3 原生七个 MTP 模块。顺序组件执行
沿用现有线性图近似，未额外模拟残差/分支并行及 KV 写回。

### MiniMax XFold

本次 XFold 聚焦 EP 下的矩阵融合与固定顺序执行，验证 folded projection、
expert 汇总和 indexer QK 的依赖关系；TP 和 SplitTree 设计不在本次范围内。

`MiniMaxM3Mapper().map(workload, xfold=True)` 将下一层 index query 提前到
当前层 MoE 内生成。令 attention 后 residual 为 `r`，expert 的 SwiGLU
输出为 `z_e`，下一层 index-query 权重为 `Wq`，
`Da = diag(1 + gamma_a)`（Gemma 风格 RMSNorm），则离线融合：

```text
F_base   = Wq Da                 # 512 x 6144
F_e      = Wq Da Wdown_e         # 512 x 3072，每个 routed expert
F_shared = Wq Da Wdown_shared    # 512 x 3072
q_bar = F_base r + 2 * sum_e(g_e * F_e z_e) + F_shared z_shared
```

`g_e` 为原路由权重；routed scaling factor 2 只作用于 routed contribution。
上述 residual/shared/routed contribution 都不可省略。只对线性权重做离线融合，
FC1 后的 SwiGLU 仍在语义上先于 folded projection；归一化、RoPE、路由加权、
求和和 block-max/top-k 等 vector 操作不计时。

| Op | 全局 BMM shape `(B,M,K,N)` | 顺序 |
| --- | --- | --- |
| `moe.fold_base_indexer_q` | `(1,T,6144,512)` | shared FC1 前，逐 token residual contribution |
| `moe.fold_shared_indexer_q` | `(1,T,3072,512)` | shared FC1 后、shared FC2 前 |
| `moe.fold_indexer_q` | `(128,ceil(4*T/128),3072,512)` | routed FC1 后，按 EP 分配 experts |
| `moe.indexer_score` | `(B,4*t,128,history)` | 全部 contribution 到齐后、routed FC2 前 |

EP 在 routed folded projection 后插入 `moe.ep_fold_combine`，按每个 expert
assignment 的 512 维结果计通信量，完成后才执行 score。base/shared 路径按
Attention DP 分配，所有 expert 汇总后每个 query/group 只计算一次 QK。
MiniMax 没有 DSA 的 head-weight 或第二级 query projection。
query group 的 RMS 正标量不改变该组 block 排序；按正式语义独立选 Top-16
blocks 并强制包含 local block。`attn.indexer_k` 保留正式历史 key 的生成，
`attn.indexer_q` 和 `attn.indexer_score` 由提前的结果替代。

Fold 的 `attn.flash_msa` 从 DRAM 读取已预取 KV；routed FC2 只使用 DRAM
权重带宽，同时用 Flash 预取下一层 KV。每请求预取量为
`min(history,2048) * 4 * (128+128) * kv_dtype_bytes`，包括全部 GQA 组的
独立 K/V；沿用理想 MTP1 的共享 cache 口径，不因 query 数翻倍重复读。
HBF 表的 `dense_indexer` 速率用于 `moe.indexer_score`，`sparse_kv` 速率用于
FC2 的 Flash 预取。普通 folded 权重仍默认按 DRAM/Flash 带宽比例读取。

Fold 仍使用 `57 * sparse_fold_layer_time + 3 * dense_layer_time` 的稳态估算，
dense 前缀强制关闭 Fold；显式 dense mapper 不接受 `xfold=True`。
不单独模拟首个 sparse 层的启动、最后一层无下一层可预取、dense/sparse 交界。
因此该层数加权不等于严格展开 56 个 MoE-to-MSA transitions 的完整网络时延。
这里只模拟矩阵形状、通信和时序，不执行真实权重或数值 Top-k。

## 从 config.json 构建

```python
from srammachine.inference import InferenceConfig
from srammachine.mapping import DeepSeekV32Mapper
from srammachine.pipetree import TreeParser
from srammachine.simulator import Simulator

# 也可以直接 DeepSeekV32Mapper()，读取仓库里的 model card。
mapper = DeepSeekV32Mapper.from_json("srammachine/model_cards/deepseek-v3.2.json")
workload = InferenceConfig(256, 20000, 600)
ops = mapper.build_op_list(workload, xfold=True)  # list[Operator]
mapped = mapper.map(workload, xfold=True)
# ops 是全局形状；mapped.op_list 是代表 Chip 的局部形状。

graph = TreeParser(mapper.hardware_config).parse(
    mapped.root_node, mapped.operators, mapped.operator_mappings,
)
result = Simulator(mapper.hardware_config).run(graph, mapping_result=mapped)
```

`config.json` 使用对应 model card 的字段和校验规则；Mapper 不跨模型复用。
`op_list` 是顺序列表，`operators` 保留按 ID 查询的接口，供 PipeTree 使用。
组件自身生成 list，映射和命令生成不会增加模型级的 NoC/Vector Op。
每个计算 Op 的片内输入和输出通信由 TreeParser 自动生成；EP 的跨 Chip
dispatch/combine 由 Mapper 显式插入 FabricOp，绑定 `system.fabric`。

## DSA 组件

普通版顺序：`PreAttn → FusedIndexer → CoreAttn → PostAttn → FusedMoE`。
Fold 版将最后一个组件换为 `FusedFoldingMoE`。DeepSeek V3.2 还将
`FusedIndexer` 换为 `IndexerKeyCache`，仅生成正式的 indexer K cache，
复用上一层 MoE 提前算出的 top-k，不再执行 `attn.indexer_q`、
`attn.indexer_k_w` 或 `attn.indexer_score`。GLM 5.1 仍保留完整的 `FusedIndexer`。

下表中 `T` 是全局 query token 数（batch × MTP token 数），`R` 是每专家的
routed-token 容量，`E` 是专家数。`G` 是请求 batch，`t` 是每请求 query 数
（MTP off 为 1，on 为 2），`T=G*t`。具体维度从模型配置读取。

| 组件 | Op | DeepSeek V3.2 BMM shape `(B, M, K, N)` |
| --- | --- | --- |
| PreAttn | latent_down | `(1, T, 7168, 1536+512+64)` |
| PreAttn | q_rope_projection | `(1, T, 1536, 128*64)` |
| PreAttn | qk_nope | `(1, T, 1536, 128*128)` |
| PreAttn | qk_absorb | `(128, T, 128, 512)` |
| FusedIndexer | indexer_q | `(1, T, 1536, 64*128)` |
| FusedIndexer | indexer_k_w | `(1, T, 7168, 128+64)` |
| FusedIndexer | indexer_score | `(G, t*64, 128, context_length)` |
| IndexerKeyCache（仅 Fold） | indexer_k | `(1, T, 7168, 128)` |
| CoreAttn | flash_attention | B=G，M=t*heads，S=min(context_length, dsa_len) |
| PostAttn | vo_absorb | `(128, T, 512, 128)` |
| PostAttn | output_projection | `(1, T, 128*128, 7168)` |
| FusedMoE | moe_fc1 | `(E, R, 7168, 2*2048)`，合并 gate/up |
| FusedMoE | moe_fc2 | `(E, R, 2048, 7168)` |

CoreAttn 和 Indexer score 将同一请求的 query/head 行合入 M，B 仅表示请求。
MTP 不改变 B/S 的 PU 切分，也不增加缓存读取量；Q 输入及输出通信随 t 增加。
Indexer score 使用 `queries_per_request=t`，每组 `M/t` 个 heads 独立归约、
独立 top-k，不能将不同 query 的候选合并成一个 top-k。其 head reduction
FLOPs 为 `B*N*(2*M-t)`，每个 PU 输出 `B*t*min(top_k, local_S)` 个候选。
CoreAttn 仍假设同一请求的两个 query 共享所需 KV，不建模不同稀疏候选集合的并集。

Fold 的最后一个组件依次为：

1. `moe_fc1`，与普通版相同。
2. `fold_latent_q_indexer_k_w`：`(E, R, 2048, 1536+128+64)`。
3. `indexer_q`：`(1, T, 1536, 64*128)`，与普通版形状相同。
4. `indexer_score`，与普通版相同。
5. `moe_fc2`，DRAM 权重读取并行 Flash KV 预取。

Attention DP + FFN EP 时，Mapper 在 FC1 前插入 `moe.ep_dispatch`，
FC2 后插入 `moe.ep_combine`；Fold 还在融合投影后、indexer_q 前插入
`moe.ep_fold_combine`。这三处通信使用 all-to-all，后续计算等待通信完成。
TP 保留原来的计算映射，本次不增加 TP 通信。

通信量采用明确的合成路由：请求按连续区间均匀分配到 Chip（余数给前面的
Chip），同一请求的 MTP token 留在同一 Chip；全局 token t 选择
`(t * top_k + k) % E` 专家（`0 <= k < top_k`），专家连续均匀分配到 Chip。
因此每个专家的实际 assignment 数与 `expert_token_loads` 一致；通信不发送
BMM 容量补齐的虚拟 token。没有真实 router 数据时，这是均衡路由假设，
不代表任意真实 workload 的路由或负载偏斜。

设 R[i][j] 是源 Chip i 到专家 Chip j 的 assignment 数：dispatch 字节矩阵为
`R[i][j] * hidden_size * input_dtype_bytes`，FC2 combine 为
`R[j][i] * hidden_size * output_dtype_bytes`；Fold combine 的宽度替换为
`q_lora_rank + indexer_head_dim + indexer_num_heads`。每个 assignment 单独传输，
不假设同目标 Chip 的 dispatch 去重或 combine 预聚合；本地专家加权求和不计入
fabric 时间，也不新增相应计算算子。矩阵对角线保留本地数据量，网络耗时排除它。

拆批按全局请求区间重新计算整数 assignment，所有子批次的矩阵逐元素相加
等于整批矩阵；不能按 BMM 的补齐维度或字节比例近似。网络耗时为最大 Chip
发送/接收字节数除以单向带宽，向上取整到 ns。沿用当前理想 fabric 模型，
不同通信命令并发时不争用端口带宽。

投影切分为 `K-p-32, N-q-64`。Attention/Indexer score 使用
`B-p-2, B-q-4, S-p-16, S-q-16`。FC1 使用
`B-p-4, N-p-2, K-p-4, N-q-64`。FC2、Fold 融合投影与 Fold indexer_q 使用
`B-p-4, K-p-2, N-p-4, K-q-64`，包括 B=1 时产生的空闲 PU。

默认硬件为 Nexus：8 Chip，每 Chip 32×64 PU。上述固定切分作用于一个
Chip 内的 2048 PU；DSA Mapper 会检查每 Chip 的 mesh 为 32×64。
`build_op_list()` 提供全局 shape，`map()` 先完成跨 Chip 分配，再生成
代表 Chip 的 Op 列表。RootNode batch 仍为全局值，`global_dimensions`
保留全局 shape，`chip_dimensions` 和 `pu_dimensions` 分别记录局部形状。
Attention 采用 Chip 间 DP，每 Chip 处理 `ceil(batch/8)` 个请求；默认
FFN 采用 EP，每 Chip 持有 1/8 专家。显式选择 TP 时各 Chip 持有全部专家，
FC1 的 N、FC2 及 Fold 融合投影的 K 按 Chip 数切分。
片内切分链保持上述定义。非 DSA Mapper 同样使用 Nexus 默认硬件和 DP/EP。

## DRAM / Flash

`op.weight_memory` 指定 `dram`、`flash` 或 `dram+flash`。所有投影和普通
MoE BMM 默认双源；Indexer score 仅 Flash；CoreAttn 普通版仅 Flash、
Fold 版仅 DRAM；Fold FC2 仅 DRAM，同时启动单独的 Flash KV 读取。

PU 带宽由 `ProcessingUnitConfig.dram_bandwidth_bytes_per_second` 和
`flash_bandwidth_bytes_per_second` 配置，默认值分别为 16 GB/s、4 GB/s。
这两个值属于新 PU 路径，不改变旧代表 die 的带宽参数。每个被模拟的 PU
分别实例化 `.dram` 和 `.flash` 两个资源，资源独立排队。

双源读取 W 字节时，DRAM 与 Flash 分别承担
`W*B_dram/(B_dram+B_flash)` 和 `W*B_flash/(B_dram+B_flash)`，耗时均为
`W/(B_dram+B_flash)`。允许分数字节来表达理想条带化，模拟时间向上取整到 ns。
算子级 Flash 带宽覆盖同时影响分配比例、流读取耗时和 PU 时间；DRAM 带宽固定。

实测 HBF CSV 可通过 `HBFBandwidthTable.from_csv(path)` 加载，并传入
`mapper.map(..., hbf_bandwidth_table=table)`。它按 global batch、输入序列长度和
MTP 查表，为 KV/indexer（含 Fold 预取）覆盖 Flash 带宽；`memory_overrides`
仍决定内存来源，表格随后覆盖相关算子的 Flash 速率。芯片总带宽除以 2048 后
用于每 PU 读取，表内线性插值、表外报错。CLI 默认启用仓库内表格，详见
[HBF 配置说明](../script/README.md#hbf-实测带宽表)。

```python
from srammachine.frontend.modules import WeightMemory, WeightSource

mapped = mapper.map(workload, memory_overrides={
    "attn.qk_nope": WeightMemory(
        WeightSource.DRAM_FLASH, flash_bandwidth_bytes_per_second=8e12,
    ),
})
```

Fold FC2 的预取总量为
`batch * min(context_length, dsa_len) * (kv_lora_rank + qk_rope_head_dim) * kv_dtype_bytes`。
该全局数据流先按 Chip 的 DP 请求分配，再均匀分配到片内网格，
不按专家路由重复，也不随 MTP 翻倍。
权重流和 KV 流在 FC2 开始时启动，与计算重叠。输出以及下一层对应微批次
必须等待这些流完成。命令图的 start edges 在多层复制时保留。

模拟仍采用代表 PU 和现有 NoC 带宽模型，不逐个执行全系统 16384 PU，也没有
额外建模跨 chip 路由时延。MoE 采用均衡路由，`R=ceil(T*top_k/E)`，BMM
按这个均匀容量计时；`expert_token_loads` 保存未填充的精确路由数。
微批次数不得超过可分配的 expert work units，否则解析会拒绝该候选。
首个模拟 Fold 层假定已有所需 DRAM cache；预测准确率、预取 miss、cache
容量以及 gate 激活的额外向量耗时不在这个组件级模型中。

## CLI

普通版不加 `--xfold`，Fold 版加上该参数：

```bash
.venv/bin/python -m srammachine.script.run_sa \
  --models deepseek-v3.2 --mtp off --batch-sizes 256 \
  --rounds 1 --warmup-rounds 1 --restarts 1 --workers 1 \
  --input-sequence-length 20000 --output-sequence-length 600 \
  --moe-strategy tp --xfold
```

搜索和 trace replay 保留 Fold 标志；CSV/JSON 记录 `xfold` 与实际 chip 数。
新的 shape、分组和硬件规模改变了时延口径，旧 SplitTree 不应直接作为新
Mapper 的初始树。旧映射回归测试仍覆盖 Legacy，新测试见 `test_dsa_mapper.py`。
