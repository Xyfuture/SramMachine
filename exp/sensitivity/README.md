可用 `--attention-efficiency 0.423` 仅降低 core attention 的 roofline 计算峰值。
计算划分优化后对 attention 算子配置生效，内存带宽、Indexer、MoE及其他算子不变；
事件模拟重新求访存/通信重叠和稳态层时间，默认效率1保持原行为。

# NoC 远端供数校准（2026-11-17）

NoC sensitivity 新增 `cap_remote_supply`，只在 NoC 扫描映射后应用；核心默认模型、其他参数扫描不变。
1x 原样返回，不改变映射或带宽。系数 s 下，远端份额乘 min(1,s)，本地份额不变。
现有模拟器合并两侧 PU，普通权重/KV 端口按一半本地、一半远端的均匀池化近似，
因此等效端口带宽乘 (1+min(1,s))/2。独立 KV 预取为全远端，带宽乘 min(1,s)。
Indexer 的原 HBF 部分按池化处理，额外 HBM 共享部分以 1x 的 8 GB/s/PU 为上限。
不新增逐跳事件或动态共享链路争用；NoC 通信事件本身仍按完整 s 缩放。
这是以 1x 恰好支撑既定远端份额为假设的敏感性近似，不是物理 D2D 带宽精确模型。

只跑 NoC 可用 `--parameters noc`。以下旧说明中的线性 Indexer 增量描述仍适用于核心映射，
但 NoC sensitivity 会在映射之后按上述规则校正。结果保留原始模拟口径，
不自动执行论文绘图的 Attention 42.3% 效率后处理。

# 单参数敏感性实验

Indexer 历史缓存的有效读取带宽现为每卡原 Flash 带宽加
`hardware.chip.noc.link_bandwidth_bytes_per_second * 64 * 2`，再除以该卡 PU 数。
默认增加 16.384 TB/s/卡（2048 PU 时为 8 GB/s/PU）。此为等效带宽放大，
不增加独立 NoC 传输或额外 NoC 争用。仅作用于 Flash-backed IndexerScore，
DRAM-only Indexer、KV/预取和投影权重保持原口径；禁用 CSV 时使用硬件 Flash 基准。
NoC sensitivity 缩放新增项，HBF sensitivity 只缩放原 Flash 项。
NoFlashShare 和 NoDRAMFLashShare 去掉新增项，再按旧口径缩放半网格带宽。
NoKVLayout 使用项目根目录 `32k no kvlayout bandwidth.csv` 的 sparse KV 实测 HBF 带宽；
Indexer 及其 NoC/HBM 增量不变，无固定附加延迟。缺失测量配置直接报错。
旧实验数据不自动更新，新旧结果应分开保存。


从仓库根目录运行，使用本地 `.venv`。绘图需要 matplotlib（见 requirements.txt）：

```bash
.venv/bin/python -m exp.sensitivity.sweep --run-id 2026-11-16 --workers 6
```

默认 MiniMax M3，context length = 32768（32k），global batch 为
16、32、64、128、256、512、1024、2048、4096。固定 8 chip、XFold、EP、FP8 KV、
默认 MTP0（每步 1 token，`--mtp 1` 可切换）、output length 600、4 个代表层，与 `exp/big` 一致。
4 个参数 × 5 个倍率 × 9 个 batch = **180 个实验点**。

每次从全新的 `HardwareConfig()` 默认值出发，仅缩放一种资源；倍率为
0.5、0.75、1、1.25、1.5。四组 1x 会独立运行，校验工具检查它们的延迟一致。

| 参数 | 缩放范围 | 1x 基准 |
| --- | --- | --- |
| HBF | CSV 查得的每个算子 Flash 带宽，以及硬件 fallback Flash 带宽 | KV/indexer 使用原始 CSV；fallback 为 4 GB/s/PU |
| HBM | PU DRAM 端口与 logic die 内存带宽 | 16 GB/s/PU；16 TB/s/logic die |
| NoC | 片内链路横向和纵向带宽 | 每方向 128 GB/s |
| Logic | 默认同时缩放 PU 矩阵与 logic die vector 算力 | 每 PU 2.048 TFLOPS；每 logic die vector 128 TFLOPS |

Logic 用 `--logic-scope both`（默认）、`vector` 或 `pu` 选择口径。
矩阵算力通过频率缩放实现，不改变阵列尺寸、PU 数量和拓扑。
NoC 在现有模拟器中也服务 PU 与 IO die 之间的传输，因此这些传输随片内带宽变化；
芯片间 fabric 始终保持每方向 900 GB/s。HBM 不改变容量。普通权重沿用原内存策略；
Flash 算子覆盖、XFold prefetch 和 DRAM+Flash 混合加载均由原映射/仿真流程处理。

原始 CSV 不会改写；实验内 `ScaledHBFBandwidthTable` 仅乘查表结果，保留插值、
禁止外推和 dense MTP0 fallback 规则。实验启动前检查全部 batch 的查表组合。
除 `exp/sensitivity` 新文件外，不修改核心库或 `exp/big`；复用后者的序列化和
Pareto 小工具，备份中包含这些依赖。

每个配置重新优化 PartitionChain，SplitTree 不搜索。这衡量各硬件配置经相同优化
流程后的性能上限；由于最优 PartitionChain 可以变化，曲线不是固定映射下的纯时延缩放。
延迟按稳态相邻代表层 core 开始间隔外推到完整 decoder，统计 decode 单步：

- tokens/s/user = (1 + MTP) × 1e9 / model_step_latency_ns。
- tokens/s/chip = tokens/s/user × global_batch / 8。
- 每个 model、参数、倍率分别筛 Pareto；两轴都越大越好，均为线性坐标。
- 每张图叠加 5 个倍率的 Pareto 曲线，浅色点保留全部 batch，前沿点标注 batch。

默认输出到忽略目录 `test result/sensitivity/<日期时间>-sensitivity/`；
`--run-id 2026-11-16` 可指定为 `2026-11-16-sensitivity`：

```text
2026-11-16-sensitivity/
  deepseek-v3.2/
    hbf/                      # hbm/、noc/、logic/ 同结构
      x0.5-bs16.perfetto.json  # 每个倍率 × batch 一个 trace，共 45 个/参数
      x0.5-bs16.result.json   # 硬件、推理配置、切分、原始 command 时序
      x0.5-bs16.log
  hbf.csv                     # hbm.csv、noc.csv、logic.csv，共四个 CSV
  deepseek-v3.2-hbf-pareto.png  # 其余参数同理，共四张图
  scripts/                    # 实验脚本、原 CSV、实际源码 ZIP、git diff、依赖版本
  manifest.json
  completion.json
  verification.json           # 执行 verify 后生成
```

CSV 每行是一组参数/倍率/batch 的结果，含性能指标、实际配置、算子 Flash 带宽、
Pareto 标记和 trace 路径；嵌套字段以 JSON 保存。结果原子落盘，报表约每分钟刷新。
每点使用独立子进程隔离 Desim 状态。失败点写入 `.error.json`，最终返回非零退出码。

恢复同一次运行（保持配置和源码一致；worker 数可以改变）：

```bash
.venv/bin/python -m exp.sensitivity.sweep --run-id 2026-11-16 --resume
.venv/bin/python -m exp.sensitivity.sweep --run-id 2026-11-16 --reports-only
.venv/bin/python -m exp.sensitivity.verify 'test result/sensitivity/2026-11-16-sensitivity'
```

`--resume` 校验源码/CSV 哈希及 manifest，只跳过有完整 checkpoint 和非空 trace 的点。
`--reports-only` 可在源码变化后重画，但仍要求相同实验配置。
使用自定义参数运行后，续跑/重画也需传入原来的配置参数。

快速端到端验证与单元测试：

```bash
.venv/bin/python -m exp.sensitivity.sweep --run-id smoke --batch-sizes 16 32 --factors 0.5 1 --workers 2
.venv/bin/python -m exp.sensitivity.verify 'test result/sensitivity/smoke-sensitivity'
.venv/bin/python -m unittest exp.sensitivity.test_sweep
.venv/bin/python -m unittest discover -s srammachine/tests -p 'test_*.py'
```

`--output-root` 可更换结果根目录，`--bandwidth-csv` 可指定同格式测量表。
默认日期时间可避免同一天多次实验覆盖；已存在的目录必须显式指定续跑或重画。

支持 `--models minimax-m3 deepseek-v3.2`。M3 按 57 层 sparse + 3 层 dense 加权，
每个实验点同时保存 sparse 与 dense 的完整 trace 和原始命令数据。
