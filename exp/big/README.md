# 长序列 sweep

在仓库根目录使用 `.venv/bin/python`。可选报表依赖见 `requirements.txt`：

```bash
uv pip install --python .venv/bin/python -r exp/big/requirements.txt
.venv/bin/python -m exp.big.sweep --workers 6
```

默认模型为 DeepSeek V3、DeepSeek V3.2、GLM 5.1、MiniMax M3；global batch 为
16、32、64、128、256、512、1024、2048、4096。8 chip，每方向 900 GB/s，
全双工总计 1800 GB/s。启用理想 MTP1（每步接受 2 token），除 DeepSeek V3 外
启用 XFold。只枚举 PartitionChain，不搜索 SplitTree。其余沿用当前默认：
EP、FP8 KV、输出长度 600，trace 模拟连续 4 个代表层。

默认序列长度为 16384、32768、65536，即 16k、32k、64k（1k = 1024），
不再运行 8k。四个模型 × 9 个 batch × 3 个序列长度，共 108 个点。
当前仓库 HBF CSV 覆盖这三个长度；64k 的全部 9 个 batch 均有精确测量点，
无需插值或外推。DeepSeek V3 的 dense KV 和其他三个模型的 dense indexer
按现有规则沿用 MTP0 测量；三个 sparse 模型的 KV 使用 MTP1 测量。
脚本会在启动前检查全部查表组合，不会静默外推或替换序列长度。

续跑已经完成 16k、32k 的此次实验，只补齐 64k：

```bash
.venv/bin/python -m exp.big.sweep --workers 6 --run-id 2026-11-16 --resume
```

吞吐口径：

- `tokens_per_second_per_user = 2e9 / model_step_latency_ns`。
- `tokens_per_second_per_chip = tokens_per_second_per_user * global_batch_size / 8`。
- 延迟为稳态相邻代表层 core 开始时刻间隔，外推到完整 decoder 的单步延迟，
  不是生成全部 600 token 的耗时，也不包括 prefill。
- MiniMax 按现有代码计算 57 层 sparse 和 3 层 dense 的加权延迟。dense 前缀
  保留 baseline 切分，因 CSV 没有该部分测量，Flash 沿用硬件默认值。
  每个 MiniMax 点额外导出 dense baseline trace，表中明确标记该带宽来源。
- Flash KV/indexer 带宽从 CSV 按模型、global batch、MTP 和序列长度查表；
  dense 读取缺少 MTP1 时沿用 MTP0，普通权重加载保留原有内存策略。

每个序列长度独立输出到忽略目录 `test result/big/<run-id>-big-seqlen-<N>k/`：

```text
<model>/bs<batch>/
  *perfetto.json     # 优化后的代表层 trace；MiniMax 另有 dense baseline
  result.json       # 指标、硬件、推理配置、布局、逐 command 原始时序
  run.log
<model>.csv         # 每模型一个 CSV，全部 batch 点
<model>.xlsx        # metrics、methodology 工作表
<model>-pareto.png  # 横轴 tokens/s/user，纵轴 tokens/s/chip，线性坐标
scripts/            # sweep 脚本、带宽表、当前源代码 ZIP、git diff、依赖版本
manifest.json
completion.json
```

所有点均保留；Pareto 标记和连线仅包含两轴均按越大越好筛选的非支配点。
不同模型、不同序列长度分别筛选，点标注 batch。CSV 的嵌套原始字段以 JSON
保存，详细逐 command 数据位于对应 `result.json` 和 trace。

最多 6 个仿真子进程，每个进程只处理一个点，隔离 Desim 全局状态；结果逐点
原子写入，并刷新该模型报表。可使用相同参数加 `--resume` 跳过已有结果和 trace
完整的点，失败点单独记录 `error.json`，不影响其余点。恢复运行应使用同一份
源码；`scripts/source.zip` 保留此次未提交源码的实际内容。

只刷新报表：使用原命令的全部配置参数，加 `--reports-only`。
`--models`、`--batch-sizes` 可用于小规模验证；`--output-root` 可更改输出根目录。
