# Attention chip-level 8×8 mesh

All four models use one Attention mapping regardless of MoE TP/EP. Projection
weights and activations are mapped across the full chip with K8×N8 PU tiles.
Projection NoC commands carry the **full chip input/output byte counts** and
use eight parallel 256 GB/s paths, representing one input shared by a PU row
and one output reduced along a PU column. Bytes are not also divided by eight.

FlashAttention uses eight independent request/head rows, with eight sequence
shards per row. Its row-local NoC commands use one path and receive no
cross-row reduction discount. DSA keeps the existing BS-dependent request,
sequence and local top-k rules. At low BS, score merge uses the PU mesh rather
than a die collective. The 2-byte ID plus 2-byte score candidate format is
unchanged.

Attention weights, KV/indexer cache streams and vector work now use chip
resources. The previous effective cache traffic is multiplied by four when
converted from a representative-die command, and the cache SRAM-to-PU stream
retains four times the previous port-limited bandwidth. This preserves the
old cache-read timing; it does **not** revise head reuse or KV sharing. Weight
loads use the full chip SRAM bandwidth. Existing MoE and Fabric mapping are
unchanged.

The four light baseline Perfetto traces for BS1024, ISL32000, MTP off and TP
are generated under `test result/attention chip 8x8/` (an ignored output
directory). These traces describe the stated ideal 8-path NoC model rather
than an explicit physical routing network.
