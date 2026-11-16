"""MiniMax M3 text-decoder components and dedicated layer mapper.

Architecture: https://arxiv.org/html/2606.13392v2 (section 3.1).
Dimensions/layer masks: MiniMaxAI/MiniMax-M3 config.json, text_config.
Projection, dense/shared FFN and routed expert products reuse BMMOp.
"""

from dataclasses import replace
from typing import Mapping

from srammachine.frontend.modules import (
    FlashMSAOp, MiniMaxFusedIndexerScoreOp, Operator, PartitionChain,
    WeightMemory, WeightSource,
)
from srammachine.inference import InferenceConfig
from srammachine.hardware import HardwareConfig
from .components import ATTENTION_SPLIT, FC2_SPLIT, FusedMoE, LayerContext
from .mapper import HardwareMappingRequest, HardwareMappingResult
from .minimax_config import MiniMaxM3Config
from .model_mappers import PartitionedLayerMapper
from .hbf_bandwidth import HBFBandwidthTable, apply_hbf_bandwidths


MSA_SPLIT = PartitionChain((
    ("B", "p", 2), ("B", "q", 4), ("H", "p", 4),
    ("S", "p", 4), ("S", "q", 16),
))


class MiniMaxPreAttn:
    """Standard GQA Q/K/V projections without MLA absorption."""

    def build(self, context: LayerContext) -> list[Operator]:
        """Pack Q/K/V projections in one existing BMM."""
        m = context.model
        return [context.bmm("attn.qkv_projection", 1, context.num_tokens, m.hidden_size,
                            (m.num_attention_heads + 2 * m.num_key_value_heads) * m.head_dim)]


class MiniMaxIndexer:
    """Hidden-to-index Q/K projections and per-group block scoring."""

    def score(self, context: LayerContext, prefix: str) -> MiniMaxFusedIndexerScoreOp:
        """Score shared historical keys, retaining independent group top-k."""
        m, inference = context.model, context.inference
        return MiniMaxFusedIndexerScoreOp(
            f"{prefix}.indexer_score", inference.global_batch_size,
            m.indexer_num_heads * inference.accepted_tokens_per_step,
            m.indexer_head_dim, inference.input_sequence_length,
            partition_chain=ATTENTION_SPLIT, top_k=m.sparse_topk_blocks,
            block_size=m.sparse_block_size,
            queries_per_request=inference.accepted_tokens_per_step,
            input_dtype_bytes=1, output_dtype_bytes=2,
            weight_dtype_bytes=inference.kv_cache_bytes_per_element,
            weight_memory=WeightMemory(WeightSource.FLASH),
        )

    def build(self, context: LayerContext) -> list[Operator]:
        """Keep formal K generation; Fold reuses the preceding MoE's top-k."""
        m = context.model
        key = context.bmm("attn.indexer_k", 1, context.num_tokens, m.hidden_size,
                          m.indexer_head_dim)
        if context.xfold:
            return [key]
        return [
            context.bmm("attn.indexer_q", 1, context.num_tokens, m.hidden_size,
                        m.indexer_num_heads * m.indexer_head_dim),
            key,
            self.score(context, "attn"),
        ]


class MiniMaxCoreAttn:
    """Independent K and V per GQA group, with a bounded sparse token budget."""

    def __init__(self, sparse: bool = True) -> None:
        self.sparse = sparse

    def build(self, context: LayerContext) -> list[Operator]:
        """Build FlashMSA for sparse attention or the dense GQA prefix."""
        m, inference = context.model, context.inference
        history = inference.input_sequence_length
        # Match the simulator's fixed-history decode convention. The sparse
        # budget is an upper bound; selected block popularity is not modeled.
        tokens = min(history, m.sparse_topk_blocks * m.sparse_block_size) if self.sparse else history
        return [FlashMSAOp(
            "attn.flash_msa", inference.global_batch_size,
            (m.num_attention_heads // m.num_key_value_heads) * inference.accepted_tokens_per_step,
            m.head_dim, tokens, tokens, m.head_dim, H=m.num_key_value_heads,
            partition_chain=MSA_SPLIT, input_dtype_bytes=1, output_dtype_bytes=1,
            kv_cache_dtype_bytes=inference.kv_cache_bytes_per_element,
            weight_memory=WeightMemory(WeightSource.DRAM if context.xfold else WeightSource.FLASH),
        )]


class MiniMaxPostAttn:
    """Concatenated query heads project directly to hidden width."""

    def build(self, context: LayerContext) -> list[Operator]:
        """Return the standard attention output BMM."""
        m = context.model
        return [context.bmm("attn.output_projection", 1, context.num_tokens,
                            m.num_attention_heads * m.head_dim, m.hidden_size)]


class MiniMaxSharedFFN:
    """Shared expert and routing matrix; routing vector arithmetic is omitted."""

    def build(self, context: LayerContext) -> list[Operator]:
        """Keep shared-expert work independent of routed-expert token counts."""
        m, t = context.model, context.num_tokens
        width = m.n_shared_experts * m.shared_intermediate_size
        index_width = m.indexer_num_heads * m.indexer_head_dim
        return [
            context.bmm("moe.router", 1, t, m.hidden_size, m.num_experts),
            *([context.bmm("moe.fold_base_indexer_q", 1, t, m.hidden_size, index_width)]
              if context.xfold else []),
            context.bmm("moe.shared_fc1", 1, t, m.hidden_size, 2 * width),
            *([context.bmm("moe.fold_shared_indexer_q", 1, t, width, index_width)]
              if context.xfold else []),
            context.bmm("moe.shared_fc2", 1, t, width, m.hidden_size),
        ]


class MiniMaxFoldingMoE(FusedMoE):
    """Fold next-layer index queries into expert down projections.

    Offline weights are Wq * Da * Wdown (512 x 3072 for M3), with
    Da = diag(1 + gamma_a). The base and shared contributions are prepared
    by MiniMaxSharedFFN. After FC1's implicit SwiGLU, routed contributions
    are weighted, scaled by the model's routed factor 2 and summed with
    base/shared outputs. Those vector operations, group query norm and
    RoPE have no timing cost. There is no head-weight or second Q GEMM.
    """

    def build(self, context: LayerContext) -> list[Operator]:
        """Produce group top-k before DRAM FC2 and parallel Flash KV prefetch."""
        m = context.model
        fc2 = self.fc2(context)
        fc2.weight_memory = WeightMemory(WeightSource.DRAM)
        selected = min(context.inference.input_sequence_length,
                       m.sparse_topk_blocks * m.sparse_block_size)
        fc2.flash_prefetch_bytes_per_request = (
            selected * m.num_key_value_heads * 2 * m.head_dim
            * context.inference.kv_cache_bytes_per_element
        )
        return [
            self.fc1(context),
            context.bmm("moe.fold_indexer_q", m.num_experts, context.routed_tokens,
                        m.moe_intermediate_size,
                        m.indexer_num_heads * m.indexer_head_dim, FC2_SPLIT),
            MiniMaxIndexer().score(context, "moe"),
            fc2,
        ]


class MiniMaxDenseFFN:
    """Dense gate/up and down projections used by M3's initial layers."""

    def build(self, context: LayerContext) -> list[Operator]:
        """Return the dense FFN's two matrix products."""
        m, t = context.model, context.num_tokens
        return [
            context.bmm("ffn.fc1", 1, t, m.hidden_size, 2 * m.intermediate_size),
            context.bmm("ffn.fc2", 1, t, m.intermediate_size, m.hidden_size),
        ]


class MiniMaxM3Mapper(PartitionedLayerMapper):
    """Map M3 MSA/MoE or dense GQA/FFN, retaining the existing search API."""

    model_name = "minimax-m3"
    config_type = MiniMaxM3Config

    def __init__(self, hardware_config: HardwareConfig | None = None, *,
                 model_config: MiniMaxM3Config | None = None, layer_type: str = "sparse") -> None:
        super().__init__(hardware_config, model_config=model_config)
        if layer_type not in ("sparse", "dense"):
            raise ValueError("layer_type must be sparse or dense")
        self.layer_type = layer_type

    def _components(self, xfold: bool) -> list:
        sparse = self.layer_type == "sparse"
        if xfold and not sparse:
            raise ValueError("MiniMax M3 xfold requires a sparse MSA/MoE layer")
        return [
            MiniMaxPreAttn(),
            *([MiniMaxIndexer()] if sparse else []),
            MiniMaxCoreAttn(sparse), MiniMaxPostAttn(),
            *([MiniMaxSharedFFN(), MiniMaxFoldingMoE() if xfold else FusedMoE()]
              if sparse else [MiniMaxDenseFFN()]),
        ]

    def _component_ops(self, context: LayerContext) -> tuple:
        if self.layer_type == "dense":
            return [c.build(context) for c in self._components(context.xfold)], {}
        return super()._component_ops(context)

    def build_op_list(self, inference: InferenceConfig, *, xfold: bool = False,
                      memory_overrides: Mapping[str, WeightMemory] | None = None) -> list[Operator]:
        """Build a decoder layer; reject histories beyond the model context."""
        if inference.input_sequence_length > self.model_config.max_position_embeddings:
            raise ValueError("input_sequence_length exceeds MiniMax M3 context limit")
        return super().build_op_list(inference, xfold=xfold, memory_overrides=memory_overrides)

    def map(self, request: HardwareMappingRequest | InferenceConfig, *, xfold: bool = False,
            memory_overrides: Mapping[str, WeightMemory] | None = None,
            hbf_bandwidth_table: HBFBandwidthTable | None = None) -> HardwareMappingResult:
        """Attach dense-layer baseline for a weighted whole-decoder estimate.

        Search tunes the sparse representative tree; the dense prefix uses
        its own sequential baseline. This estimates steady-state costs, not
        pipeline transitions between the two layer types.
        """
        mapped = super().map(request, xfold=xfold, memory_overrides=memory_overrides)
        sparse_count = sum(self.model_config.sparse_attention_layers)
        dense_count = self.model_config.num_hidden_layers - sparse_count
        additional = ()
        if self.layer_type == "sparse":
            dense_mapper = type(self)(self.hardware_config, model_config=self.model_config, layer_type="dense")
            dense_ids = {op.op_id for op in dense_mapper.build_op_list(mapped.request.inference_config)}
            overrides = {key: policy for key, policy in (memory_overrides or {}).items() if key in dense_ids}
            additional = (dense_mapper.map(replace(mapped.request, xfold=False),
                                           memory_overrides=overrides),)
        result = replace(mapped, layer_type=self.layer_type,
                       represented_layer_count=sparse_count if self.layer_type == "sparse" else dense_count,
                       additional_layer_mappings=additional)
        if hbf_bandwidth_table is not None:
            result = apply_hbf_bandwidths(result, self.hardware_config, hbf_bandwidth_table)
        return result
