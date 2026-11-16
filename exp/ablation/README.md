# 算子级消融实验

在仓库根目录执行，使用 `.venv/bin/python`：

```bash
.venv/bin/python -m exp.ablation.sweep --run-id 2026-11-16-v3 --workers 6
.venv/bin/python -m exp.ablation.verify 'test result/ablation/2026-11-16-v3-ablation'
```

默认 MiniMax M3，沿用 `exp/big`：8 芯片，每方向 900 GB/s，EP，FP8 KV，
理想 MTP1（每步接受 2 token），输出长度 600，连续模拟 4 个代表层，
只搜索 PartitionChain，不搜索 SplitTree。全局 batch 为
16、32、64、128、256、512、1024、2048、4096，序列长度为 16384、32768、65536。
完整 XFold 基线加 6 个消融，共 189 点。也支持
`--models deepseek-v3 deepseek-v3.2 glm-5.1`。DeepSeek V3 当前映射没有 Indexer，
所有消融均沿用普通路径（记录实际 `xfold=false`）；NoKVLayout 对它只增加每层延迟。

| 名称 | CoreAttention / IndexerScore | 其他算子 | XFold |
| --- | --- | --- | --- |
| XFold | 基础配置 | 基础配置 | 开 |
| NoXfold | 基础配置 | 基础配置 | 关，普通路径 |
| NoDRAMShare | 基础配置 | PU 网格行数减半，每 PU DRAM ×2 | 开 |
| NoFlashShare | PU 网格行数减半，每 PU Flash ×2 | 基础配置 | 开 |
| NoDRAMFLashShare | PU 网格行数减半，每 PU Flash ×2 | PU 网格行数减半，每 PU DRAM ×2 | 开 |
| NoKVLayout | 仅 IndexerScore 缓存 Flash 读取 ×0.87 | 每层额外延迟，见下文 | 开 |
| NoMoELayout | 基础配置 | 仅 MoE 普通 BMM 权重 Flash 读取 ×1/6 | 开 |

`NoDRAMFLashShare` 保留需求中的拼写。分类按算子类型，XFold 搬到 `moe.indexer_score`
的 IndexerScore 仍属于特殊算子；Q/K 投影属于其他算子。

基础网格要求 32×64。派生硬件保留芯片数、logic die 数、计算单元参数和互连带宽，
把每 die 的 PU 行数从 16 降到 8，从而得到每 chip 16×64 的网格。
默认 DRAM 为 16 GB/s/PU，Flash 为 4 GB/s/PU；修改某个端口不改变另一个端口。
每个算子的硬件视图同时用于切分搜索、内存分流、命令执行；同名物理资源仍然互斥，
不会因配置不同生成可并发使用的第二套物理端口。

Flash KV / indexer 先按基础 2048 PU 配置从 `hbf_read_bandwidth.csv` 查表，
再执行消融倍数，因此不会被硬件默认带宽覆盖。未被 CSV 覆盖的普通权重沿用
基础硬件 Flash 值。双端口权重按照该算子的实际 DRAM/Flash 带宽比例分配。
NoFlashShare 的实测 Flash 带宽也加倍，维持半网格后的总 Flash 带宽。

XFold 下 CoreAttention 从 DRAM 读取，Flash KV 读取被搬到 `moe.moe_fc2`
的独立预取命令。预取使用 FC2 所属的硬件网格和端口配置。
NoDRAMShare 下 FC2 使用半网格，因此其每 PU 预取数据量
也相应增加。NoFlashShare 不改 FC2 的硬件配置。

修订版 NoKVLayout 从基础配置开始，仅把 IndexerScore 的缓存读取带宽设置为
CSV 查表值的 87%，覆盖 `attn.indexer_score` 和 XFold 的 `moe.indexer_score`。
仅对实际使用 Flash / CSV 的读取生效；GLM XFold 的 `attn.indexer_score` 已使用
DRAM，保持基础 DRAM 带宽，其 `moe.indexer_score` 的 Flash 带宽按 87% 缩放。
原先的 CoreAttention/KV 预取 ×0.8 已移除，二者保持基础带宽。
每个 decoder 层额外增加：DeepSeek V3、V3.2、GLM 5.1 为 **14 μs**，
MiniMax M3 为 **25 μs**。按整个模型的层数累加，不按 trace 的 4 层，也不按 batch
或 MTP 接受 token 数再次乘算。MiniMax 包括 57 sparse + 3 dense，合计增加 **1.5 ms**。
这项固定延迟在仿真后计入，不与其他命令重叠；NoMoELayout 等其他消融不添加此延迟。

NoMoELayout 默认 `--moe-flash-scope weights`：只修改 MoE 阶段的普通 `BMMOp`
权重 Flash 带宽，设置为基础值的 1/6，包括 router、shared FFN、专家 FC1/FC2、
Fold 投影中实际使用 Flash 的权重读取。虽然 IndexerScore 在代码中继承 BMMOp，
它是独立的缓存算子，显式排除。Indexer、CoreAttention 和 KV 预取保持基础带宽；
Attention 阶段 BMM 和 MiniMax dense FFN 也保持原样。
XFold FC2 权重只有 DRAM，其 Flash 带宽字段用于 KV 预取，因此保持该字段的 CSV 值。
双端口权重仍按降低后的 Flash 带宽重新分配 DRAM/Flash 数据量。
`--moe-flash-scope stage` 仅用于显式复现旧版“整个 MoE 阶段 Flash 降速”的实验。
缩放比例可用 `--kv-factor`（当前仅指 Indexer 缓存，默认 0.87）、
`--moe-factor`（默认 1/6）调整。二者分别相对基础带宽独立生效，不叠加旧消融。

MiniMax 延迟为 57 层 sparse 与 3 层 dense 的加权单步延迟，不含 prefill，
也不是生成 600 个 token 的总时长。dense 前缀也应用相同消融并输出 trace，
但按 `exp/big` 保留 baseline 切分（半网格时适配切分因子），不额外优化。
CSV 没有 MiniMax dense KV 测量，该部分使用基础硬件默认 Flash 值再执行消融；
CSV、manifest 和日志均记录这一来源。NoKVLayout 不修改 dense KV 带宽，但仍为
每个 dense 层添加 25 μs。

CSV / result.json 同时保留 `simulated_model_step_latency_ns`（仿真值）、
`additional_latency_per_layer_ns`、`additional_model_latency_ns` 和
`model_step_latency_ns`（两者之和）。`model_latency_components` 保留仿真值，
`adjusted_model_latency_components` 记录加延迟后的逐类层耗时。Pareto 吞吐、
总吞吐与 B1024 条状图统一使用最终延迟。trace 命令时间戳保持仿真原值，
`otherData.latency_adjustment` 明确记录仿真后补加的部分。

## 自定义与续跑

`--hardware-config config.json` 接受嵌套的完整或局部 HardwareConfig JSON，
未提供字段沿用默认值。例如：

```json
{
  "chip_count": 8,
  "chip": {
    "logic_die": {
      "processing_unit": {
        "dram_bandwidth_bytes_per_second": 16000000000,
        "flash_bandwidth_bytes_per_second": 4000000000
      }
    }
  }
}
```

小规模验证：

```bash
.venv/bin/python -m exp.ablation.sweep --batch-sizes 1024 --sequence-lengths 16384 --run-id smoke
```

使用同一配置加 `--resume` 跳过结果和 trace 完整的点，失败点保留 `error.json`。
恢复要求源代码哈希一致，避免混用不同实现。`--reports-only` 只从检查点重建图表，
仍需提供原命令的配置参数。默认并行 6 个隔离进程，每个进程只执行一个点。
`--output-root` 改变输出根目录，`--ablations` 可选运行部分消融。
每点立即保存检查点，汇总图表约每 30 秒刷新并在结束时完整重建。
本次修订记录 `ablation_revision=3`，请使用新的 run-id 保留旧版结果。
当前审计脚本同时支持 v2/v3，并按 manifest 中的 scope 核对实际规则。
只跑 NoMoELayout 可使用 `--ablations NoMoELayout`，例如：

```bash
.venv/bin/python -m exp.ablation.sweep --models minimax-m3 deepseek-v3 deepseek-v3.2 glm-5.1 --ablations NoMoELayout --run-id bmm-only
```

## 结果

```text
test result/ablation/<日期>-ablation/
  minimax-m3/
    XFold/bs1024-s16384/
      *perfetto.json       # sparse 优化 trace 和 dense baseline trace
      result.json          # 每 Op 硬件、端口、切分、逐命令时序、完整指标
      run.log
    NoXfold/...
    NoDRAMShare/...
    NoFlashShare/...
    NoDRAMFLashShare/...
    NoKVLayout/...
    NoMoELayout/...
  minimax-m3.csv           # 所有消融、batch、序列长度
  minimax-m3-pareto-s16384.png
  minimax-m3-pareto-s32768.png
  minimax-m3-pareto-s65536.png
  minimax-m3-latency-bs1024.png  # 三个序列长度的分面条状图
  scripts/                # 当前代码 ZIP、实验脚本、CSV、硬件 JSON、依赖、git diff
  manifest.json
  completion.json
  verification.json       # verify 成功后写入
```

每张 Pareto 图都有所有消融曲线，横轴 tokens/s/user，纵轴 tokens/s/chip，
均为线性坐标。每种消融、每个序列长度单独筛选非支配点（两轴均越大越好）；
淡色点保留全部 batch，曲线上标注 batch。不同序列长度不混合筛 Pareto。
条状图是 global batch=1024 的完整模型单步延迟（ms）。

吞吐为 `tokens/s/user = 2e9 / latency_ns`，
`tokens/s/chip = tokens/s/user * global_batch / chip_count`。
`verify` 核对覆盖范围、源码/CSV 哈希、逐算子硬件/带宽、端口真实持续时间、
资源互斥、加权模型延迟、吞吐公式、Pareto、CSV 与 trace 原始数据的一致性。

回归测试：

```bash
.venv/bin/python -m unittest discover -s srammachine/tests -p 'test_*.py'
```
