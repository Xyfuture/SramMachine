# Chip-level 8×8 PU mesh MoE TP/EP experiment

The default 16-chip path for DeepSeek-V3, DeepSeek-V3.2,
Kimi-K2.5 and GLM-5.1 uses chip-intermediate TP16 and a unified K8×N8
PU mesh for **every active expert** on each chip. EP retains disjoint expert
ownership across the 16 chips and the existing dispatch/combine all-to-all;
every locally owned expert likewise uses the full K8×N8 PU mesh. There are no
BS thresholds or token groups. Attention, the existing FP8 activation policy
and the 1.6 TB/s per-direction Fabric configuration are unchanged.

The MoE weight prefetch and weight load commands charge the whole chip's
FP8 Up/Gate or Down weight bytes at four times one die's DRAM or SRAM
bandwidth. The chip SRAM capacity used by the prefetch scheduler is likewise
four times one die's capacity. GEMM commands still show one PU's K×N shard.
SiLU and residual vector commands use the four vector units' combined peak.

Up/Gate input and Down output each charge the complete chip-level FP8 tensor
at an **idealized** 32 × 256 GB/s NoC rate (four perimeter edges, eight lanes
per edge). Up/Gate output charges one PU's fused channel width with the
specified ÷2 byte factor; Down input charges the one-PU
intermediate shard. These two PU-level transfers use 256 GB/s. The Up K8
partial reduction is intentionally **omitted** before SiLU. This makes the
trace an optimistic experiment, not a physically complete GEMM dataflow.
No die-specific MoE collectives remain. TP retains entry 16-chip all-to-all
and exit 16-chip reduce-scatter; EP retains dispatch/combine all-to-all.

DeepSeek-V3, BS1024, MTP off has chip-level Up/Gate
`B256 M32 K7168 N256`, Down `B256 M32 K128 N7168`, and PU-level
`B256 M32 K896 N32` / `B256 M32 K16 N896`. Up input and Down output are
each 58,720,256 bytes / 7.168 µs. Up output and Down input are each
131,072 bytes / 0.512 µs. The chip's weights are 469,762,048 Up/Gate bytes
plus 234,881,024 Down bytes. Perfetto events mark the idealized assumptions.

DeepSeek-V3 EP, BS1024, MTP off has 16 local experts and chip-level Up/Gate
`B16 M32 K7168 N4096`, Down `B16 M32 K2048 N7168`; PU-level dimensions
are `B16 M32 K896 N512` and `B16 M32 K256 N896`. Up input and Down output
are each 3,670,016 bytes / 0.448 µs; Up output and Down input are each
131,072 bytes / 0.512 µs. The resident chip weights have the same byte
counts as the TP example because fewer experts each own wider matrices.
