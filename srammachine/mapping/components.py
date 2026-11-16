"""MLA/DSA layer components producing sequential, globally shaped operators."""

from dataclasses import dataclass

from srammachine.frontend.modules import (
    BMMOp, FlashAttentionOp, FusedIndexerScoreOp, Operator,
    PartitionChain, WeightMemory, WeightSource,
)
from srammachine.inference import InferenceConfig
from .model import ModelConfig


PROJECTION_SPLIT = PartitionChain((("K", "p", 32), ("N", "q", 64)))
ATTENTION_SPLIT = PartitionChain((
    ("B", "p", 2), ("B", "q", 4), ("S", "p", 16), ("S", "q", 16),
))
FC1_SPLIT = PartitionChain((
    ("B", "p", 4), ("N", "p", 2), ("K", "p", 4), ("N", "q", 64),
))
FC2_SPLIT = PartitionChain((
    ("B", "p", 4), ("K", "p", 2), ("N", "p", 4), ("K", "q", 64),
))


@dataclass(frozen=True)
class LayerContext:
    """Shape information shared by the components of a representative layer."""

    model: ModelConfig
    inference: InferenceConfig
    xfold: bool = False

    @property
    def num_tokens(self) -> int:
        return self.inference.global_batch_size * self.inference.accepted_tokens_per_step

    @property
    def selected_tokens(self) -> int:
        if not self.model.dsa:
            return self.inference.input_sequence_length
        return min(self.model.dsa_len, self.inference.input_sequence_length)

    @property
    def kv_bytes_per_request(self) -> int:
        return (self.selected_tokens * (self.model.kv_lora_rank + self.model.qk_rope_head_dim)
                * self.inference.kv_cache_bytes_per_element)

    @property
    def routed_tokens(self) -> int:
        assignments = self.num_tokens * self.model.top_k
        return (assignments + self.model.num_experts - 1) // self.model.num_experts

    def bmm(self, op_id: str, B: int, M: int, K: int, N: int,
            split: PartitionChain = PROJECTION_SPLIT,
            source: WeightSource = WeightSource.DRAM_FLASH, **kwargs) -> BMMOp:
        """Construct a projection with explicit partition and memory policies."""
        return BMMOp(op_id, B, M, K, N, partition_chain=split,
                     weight_memory=WeightMemory(source), input_dtype_bytes=1,
                     output_dtype_bytes=2, **kwargs)

    def indexer_q(self, prefix: str, split: PartitionChain = PROJECTION_SPLIT) -> BMMOp:
        """Project the query latent into all indexer heads."""
        m = self.model
        return self.bmm(f"{prefix}.indexer_q", 1, self.num_tokens, m.q_lora_rank,
                        m.indexer_num_heads * m.indexer_head_dim, split)

    def indexer_score(self, prefix: str) -> FusedIndexerScoreOp:
        """Score cached keys with all heads and select candidates per query."""
        m = self.model
        return FusedIndexerScoreOp(
            f"{prefix}.indexer_score", self.inference.global_batch_size,
            m.indexer_num_heads * self.inference.accepted_tokens_per_step,
            m.indexer_head_dim, self.inference.input_sequence_length,
            partition_chain=ATTENTION_SPLIT, top_k=m.dsa_len,
            queries_per_request=self.inference.accepted_tokens_per_step,
            input_dtype_bytes=1, output_dtype_bytes=2,
            weight_dtype_bytes=self.inference.kv_cache_bytes_per_element,
            weight_memory=WeightMemory(WeightSource.FLASH),
        )


class PreAttn:
    """Separate latent, RoPE, NoPE and per-head absorption projections."""

    def build(self, context: LayerContext) -> list[Operator]:
        """Return this component's operators in execution order."""
        m, t = context.model, context.num_tokens
        return [
            context.bmm("attn.latent_down", 1, t, m.hidden_size,
                        m.q_lora_rank + m.kv_lora_rank + m.qk_rope_head_dim),
            context.bmm("attn.q_rope_projection", 1, t, m.q_lora_rank,
                        m.num_attention_heads * m.qk_rope_head_dim),
            context.bmm("attn.qk_nope", 1, t, m.q_lora_rank,
                        m.num_attention_heads * m.qk_nope_head_dim),
            context.bmm("attn.qk_absorb", m.num_attention_heads, t,
                        m.qk_nope_head_dim, m.kv_lora_rank),
        ]


class FusedIndexer:
    """Indexer query and key/weight projections followed by fused scoring."""

    def build(self, context: LayerContext) -> list[Operator]:
        """Return this component's operators in execution order."""
        m = context.model
        return [
            context.indexer_q("attn"),
            context.bmm("attn.indexer_k_w", 1, context.num_tokens, m.hidden_size,
                        m.indexer_head_dim + m.indexer_num_heads),
            context.indexer_score("attn"),
        ]


class IndexerKeyCache:
    """Generate formal indexer keys while reusing the preceding MoE's top-k."""

    def build(self, context: LayerContext) -> list[Operator]:
        """Project hidden states to cached keys without query or head weights."""
        m = context.model
        return [context.bmm("attn.indexer_k", 1, context.num_tokens, m.hidden_size,
                            m.indexer_head_dim)]


class CoreAttn:
    """Read dense or selected KV from Flash, or prefetched KV from DRAM in Fold."""

    def build(self, context: LayerContext) -> list[Operator]:
        """Return this component's operators in execution order."""
        m = context.model
        return [FlashAttentionOp(
            "attn.flash_attention", context.inference.global_batch_size,
            m.num_attention_heads * context.inference.accepted_tokens_per_step,
            m.kv_lora_rank + m.qk_rope_head_dim, context.selected_tokens,
            context.selected_tokens, m.kv_lora_rank,
            partition_chain=ATTENTION_SPLIT, input_dtype_bytes=1, output_dtype_bytes=1,
            kv_cache_dtype_bytes=context.inference.kv_cache_bytes_per_element,
            weight_memory=WeightMemory(WeightSource.DRAM if context.xfold else WeightSource.FLASH),
        )]


class PostAttn:
    """Per-head value absorption followed by the hidden output projection."""

    def build(self, context: LayerContext) -> list[Operator]:
        """Return this component's operators in execution order."""
        m, t = context.model, context.num_tokens
        return [
            context.bmm("attn.vo_absorb", m.num_attention_heads, t,
                        m.kv_lora_rank, m.v_head_dim),
            context.bmm("attn.output_projection", 1, t,
                        m.num_attention_heads * m.v_head_dim, m.hidden_size),
        ]


class FusedMoE:
    """Two expert BMMs; FC1 packs the gate and up projections."""

    def fc1(self, context: LayerContext) -> BMMOp:
        """Build the fused expert gate/up projection."""
        m = context.model
        return context.bmm("moe.moe_fc1", m.num_experts, context.routed_tokens,
                           m.hidden_size, 2 * m.moe_intermediate_size, FC1_SPLIT)

    def fc2(self, context: LayerContext) -> BMMOp:
        """Build the expert projection back to hidden width."""
        m = context.model
        return context.bmm("moe.moe_fc2", m.num_experts, context.routed_tokens,
                           m.moe_intermediate_size, m.hidden_size, FC2_SPLIT)

    def build(self, context: LayerContext) -> list[Operator]:
        """Return this component's operators in execution order."""
        return [self.fc1(context), self.fc2(context)]


class FusedFoldingMoE(FusedMoE):
    """Predict next-layer top-k, then overlap DRAM FC2 with Flash KV reads."""

    def build(self, context: LayerContext) -> list[Operator]:
        """Return this component's operators in execution order."""
        m = context.model
        fc2 = self.fc2(context)
        fc2.weight_memory = WeightMemory(WeightSource.DRAM)
        fc2.flash_prefetch_bytes_per_request = context.kv_bytes_per_request
        return [
            self.fc1(context),
            context.bmm("moe.fold_latent_q_indexer_k_w", m.num_experts,
                        context.routed_tokens, m.moe_intermediate_size,
                        m.q_lora_rank + m.indexer_head_dim + m.indexer_num_heads, FC2_SPLIT),
            context.indexer_q("moe", FC2_SPLIT),
            context.indexer_score("moe"),
            fc2,
        ]
