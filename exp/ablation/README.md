# 算子级消融实验

Indexer 历史缓存的有效读取带宽现为每卡原 Flash 带宽加
`hardware.chip.noc.link_bandwidth_bytes_per_second * 64 * 2`，再除以该卡 PU 数。
默认增加 16.384 TB/s/卡（2048 PU 时为 8 GB/s/PU）。此为等效带宽放大，
不增加独立 NoC 传输或额外 NoC 争用。仅作用于 Flash-backed IndexerScore，
DRAM-only Indexer、KV/预取和投影权重保持原口径；禁用 CSV 时使用硬件 Flash 基准。
NoC sensitivity 缩放新增项，HBF sensitivity 只缩放原 Flash 项。
NoFlashShare 和 NoDRAMFLashShare 去掉新增项，再按旧口径缩放半网格带宽。
NoKVLayout 读取项目根目录 `32k no kvlayout bandwidth.csv` 的 sparse KV 实测 HBF 带宽，
Indexer 及其 NoC/HBM 带宽增量不变，不再使用 0.87 系数或额外延迟。
旧实验数据不自动更新，新旧结果应分开保存。


使用 Conda `NEXUS` 环境时，下列 `.venv/bin/python` 替换为 `python`。
MTP 可通过 `--mtp 0` 关闭（每步 1 token），或 `--mtp 1` 开启（默认，理想每步 2 token）。
查表、吞吐、CSV、图例和校验均遵循所选模式。

在仓库根目录执行，使用 `.venv/bin/python`：

```bash
.venv/bin/python -m exp.ablation.sweep --run-id 2026-11-16-v3 --workers 6
.venv/bin/python -m exp.ablation.verify 'test result/ablation/2026-11-16-v3-ablation'
```

默认 MiniMax M3，沿用 `exp/big`：8 芯片，每方向 900 GB/s，EP，FP8 KV，
理想 MTP1（每步接受 2 token），输出长度 600，连续模拟 4 个代表层，
只搜索 PartitionChain，不搜索 SplitTree。全局 batch 为
16、32、64、128、256、512、1024、2048、4096，默认序列长度为 32768。
默认包含 10 组，共 90 点。也支持
`--models deepseek-v3 deepseek-v3.2 glm-5.1`。DeepSeek V3 当前映射没有 Indexer，
支持的消融沿用普通路径（记录实际 `xfold=false`）；当前专用表没有 DSv3，
选择其 NoKVLayout 会在预检时报缺少数据，不运行、不回退到原版带宽。

| 名称 | CoreAttention / IndexerScore | 其他算子 | XFold |
| --- | --- | --- | --- |
| XFold | 基础配置 | 基础配置 | 开 |
| NoXfold | 基础配置 | 基础配置 | 关，普通路径 |
| NoDRAMShare | 基础配置 | PU 网格行数减半，每 PU DRAM ×2 | 开 |
| NoFlashShare | PU 网格行数减半，每 PU Flash ×2 | 基础配置 | 开 |
| NoDRAMFLashShare | PU 网格行数减半，每 PU Flash ×2 | PU 网格行数减半，每 PU DRAM ×2 | 开 |
| NoKVLayout | HBF sparse KV 读取使用专用实测表，Indexer 不变 | XFold KV 预取同样使用专用表；无额外延迟 | 开 |
| NoMoELayout | 基础配置 | 仅 MoE 普通 BMM 权重 Flash 读取 ×1/6 | 开 |
| NoHBFLayout | 使用 NoKVLayout 实测 KV HBF 带宽，Indexer 不变 | 叠加 NoMoELayout 权重 Flash ×1/6；无额外延迟 | 开 |
| SeparatePath | 基础配置 | GEMM K 分块双缓冲流水 | 开 |
| NoSeparatePath | 横向 NoC 与计算/读取互斥 | 同样分块，同卡横向 NoC 与计算/读取互斥 | 开 |

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

NoKVLayout 从项目根目录 `32k no kvlayout bandwidth.csv` 读取 sparse KV 的实测
HBF 带宽，按模型、MTP、全局 BS 和 seqlen 查表；`mean_GBps × 1e9 / 2048`
转为每 PU bytes/s。普通映射替换 Attention 的 HBF KV 读取，XFold 映射替换
`moe.moe_fc2` 的独立 KV 预取带宽。Indexer 的 HBF 基础带宽和 NoC 共享 HBM
增量均保持原值，MoE 权重和硬件参数不变。删除旧 0.87 系数以及每层 14/25 μs
附加延迟，结果完全来自模拟调度。

用 `--no-kv-bandwidth-csv` 可指定另一份同格式原始测量表。当前表仅支持 32k、
BS16–4096、MTP0/1 的 DSv3.2、GLM5.1、M3（另含尚未支持的 GLM5.2 数据）。
缺失模型/序列长度会在预检时报错，不启动该轮实验，也不回退到旧系数。
原始表复制到结果的 `scripts/no_kv_layout_bandwidth.csv`，manifest 记录 SHA256。

NoMoELayout 默认 `--moe-flash-scope weights`：只修改 MoE 阶段的普通 `BMMOp`
权重 Flash 带宽，设置为基础值的 1/6，包括 router、shared FFN、专家 FC1/FC2、
Fold 投影中实际使用 Flash 的权重读取。虽然 IndexerScore 在代码中继承 BMMOp，
它是独立的缓存算子，显式排除。Indexer、CoreAttention 和 KV 预取保持基础带宽；
Attention 阶段 BMM 和 MiniMax dense FFN 也保持原样。
XFold FC2 权重只有 DRAM，其 Flash 带宽字段用于 KV 预取，因此保持该字段的 CSV 值。
双端口权重仍按降低后的 Flash 带宽重新分配 DRAM/Flash 数据量。
`--moe-flash-scope stage` 仅用于显式复现旧版“整个 MoE 阶段 Flash 降速”的实验。
MoE 缩放比例可用 `--moe-factor`（默认 1/6）调整；旧 `--kv-factor` 已移除。

`NoHBFLayout` 同时应用上面两种 layout 消融：KV/预取使用专用实测 CSV，
普通 MoE BMM 权重的 Flash 带宽乘 `--moe-factor`（默认 1/6）。它固定使用
`weights` 范围，不受旧 `--moe-flash-scope stage` 影响。FC2 Flash 带宽字段属于
KV 预取，只使用实测值，不再乘 1/6，避免重复惩罚。Indexer 及 NoC/HBM 增量、
HBM 权重带宽保持原样，无固定延迟，也不叠加 NoSeparatePath。
与 NoKVLayout 一样，专用表缺少模型/序列长度时直接报错。

MiniMax 延迟为 57 层 sparse 与 3 层 dense 的加权单步延迟，不含 prefill，
也不是生成 600 个 token 的总时长。dense 前缀也应用相同消融并输出 trace，
但按 `exp/big` 保留 baseline 切分（半网格时适配切分因子），不额外优化。
CSV 没有 MiniMax dense KV 测量，该部分使用基础硬件默认 Flash 值再执行消融；
CSV、manifest 和日志均记录这一来源。NoKVLayout 不修改 dense KV 带宽，也不为 dense 层增加延迟。

CSV / result.json 为兼容历史数据保留 `additional_latency_per_layer_ns` 和
`additional_model_latency_ns`，新实验均为 0。`model_step_latency_ns` 等于
`simulated_model_step_latency_ns`，trace 与报告使用同一模拟延迟。
旧结果验证仍根据 manifest revision 保留旧延迟/系数规则，不改写历史结果。

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

吞吐为 `tokens/s/user = accepted_tokens_per_step * 1e9 / latency_ns`，
`tokens/s/chip = tokens/s/user * global_batch / chip_count`。
`verify` 核对覆盖范围、源码/CSV 哈希、逐算子硬件/带宽、端口真实持续时间、
资源互斥、加权模型延迟、吞吐公式、Pareto、CSV 与 trace 原始数据的一致性。

回归测试：

```bash
.venv/bin/python -m unittest discover -s srammachine/tests -p 'test_*.py'
```


### SeparatePath / NoSeparatePath（分块流水对照）

这两个选项使用相同的 XFold 优化布局和相同的 GEMM K 分块。旧 `XFold`
及其他消融仍使用原有整算子模型；按当前实验要求，`NoSeparatePath` 统一与 `XFold` 比较；`SeparatePath`
为可选诊断组，不要求运行。该差异同时包含分块建模和共享路径约束的变化。

- 普通 GEMM 的 K 维分块，默认 K tile 为阵列列数 × 阵列数（16 × 4 = 64）；
  可用 `--separatepath-k-tile` 指定。它是调度粒度假设，并非实测缓冲容量。
- 输入和权重按 K 切分，保持每行输入、每个内存端口总字节数及 FLOPs 不变。
  每块计算沿用阵列尺寸/数量的 systolic cost；每块等待输入及其权重到齐。
- 双缓冲最多提前一块，允许读取下一块时计算当前块。K 部分和驻留 PU，
  最终输出等所有 K 块完成后传输；不引入跨算子的 tile 流水。
- `SeparatePath`：横向 NoC、PU 计算、HBM/HBF 读取各自独立。
- `NoSeparatePath`：同卡横向 NoC 与全部计算和全部 HBM/HBF 读取互斥，
  包括 Indexer cache 和 KV 预取；HBM/HBF 端口以及计算与读取之间仍可并行。
  纵向 NoC、内存写回和跨卡通信不新增限制。
- Attention/Indexer 不切分，保持原来的 compute 模型；dense 辅助层也应用上述规则。
- 不增加固定延迟或减小带宽；低 BS 的降幅由实际 tile 依赖和资源占用产生，
  不保证所有模型/形状都有降幅。manifest、CSV 与 trace 记录分块和互斥口径。

```bash
conda run --no-capture-output -n NEXUS python -m exp.ablation.sweep \
  --models minimax-m3 --ablations SeparatePath NoSeparatePath \
  --batch-sizes 16 32 64 128 256 512 1024 2048 4096 \
  --sequence-lengths 32768 --mtp 0 --workers 3 --run-id m3-separatepath
```

`python -m exp.ablation.verify <结果目录>` 同时检查互斥时间线，以及配对实验
的布局、命令、字节数和计算形状完全一致。
