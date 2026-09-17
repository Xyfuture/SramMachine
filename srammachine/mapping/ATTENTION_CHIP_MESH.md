# Attention chip-level 8×8 mesh

All four models use one Attention mapping regardless of MoE TP/EP. Projection
weights and activations are mapped across the full chip with K8×N8 PU tiles.
Projection NoC commands carry the **full chip input/output byte counts** and
use eight parallel 256 GB/s paths, representing one input shared by a PU row
and one output reduced along a PU column. Bytes are not also divided by eight.

This branch uses separate QK GEMM, full-sequence softmax, and SV GEMM instead
of FlashAttention. Eight rows run independent request/head work; eight PUs in
each row own distinct sequence shards. QK scores move to the chip vector and
softmax probabilities move back to SV as full-chip FP8 streams over eight NoC
paths. SV outputs reduce only within their own row. DSA keeps its existing
BS-dependent request,
sequence and local top-k rules. At low BS, score merge uses the PU mesh rather
than a die collective. The 2-byte ID plus 2-byte score candidate format is
unchanged.

Attention weights, cache streams and vector work use chip resources. The
QK's KV stream keeps the former FlashAttention effective cache bytes and
port-limited timing; SV separately reads the value cache under the same
legacy formula. This does **not** revise head reuse or KV sharing. DSA
indexer-key traffic instead includes every chip-local request and every
sequence shard. DSA SRAM demand reads and weight loads use the full chip SRAM
bandwidth. DSA query input and top-k candidate output aggregate all active
PUs onto eight chip NoC paths. Existing MoE and Fabric mapping are unchanged.

Unfused traces describe the stated ideal 8-path NoC model rather than an
explicit physical routing network. Score and probability streams are modeled
without a full-batch SRAM residency requirement or a DRAM spill.
