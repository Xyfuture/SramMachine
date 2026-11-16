"""Model-specific entry points from validated JSON to ordered operators."""

from dataclasses import replace
import json
from pathlib import Path
from typing import Mapping

from srammachine.frontend.modules import (
    FabricOp, FlashAttentionOp, FusedIndexerScoreOp, Operator, WeightMemory, WeightSource,
    FlashMSAOp, MiniMaxFusedIndexerScoreOp,
)
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG, HardwareConfig
from srammachine.inference import InferenceConfig, MoEParallelStrategy
from srammachine.pipetree import GroupNode, OpNode, OperatorMapping, RootNode
from srammachine.pipetree.ep_routing import EPRouting
from .components import (
    CoreAttn, FusedFoldingMoE, FusedIndexer, FusedMoE, IndexerKeyCache,
    LayerContext, PostAttn, PreAttn,
)
from .mapper import (
    HardwareMappingRequest, HardwareMappingResult, LegacyHardwareMapper,
    OperatorHardwareMapping,
)
from .model import DeepSeekV3Config, DeepSeekV32Config, GLM51Config, ModelConfig, load_model_config
from .hbf_bandwidth import HBFBandwidthTable, apply_hbf_bandwidths


class PartitionedLayerMapper:
    """Adapt explicit partitioned components to the existing search API.

    RootNode batches are global; operators represent one chip. Expert batches use a uniform
    ceiling-sized routed-token capacity; exact routing counts remain metadata.
    The explicit partition chains replace the legacy TP/EP heuristic layouts.
    """

    model_name: str
    config_type: type[ModelConfig]

    def __init__(self, hardware_config: HardwareConfig | None = None,
                 *, model_config: ModelConfig | None = None) -> None:
        if hardware_config is None:
            hardware_config = DEFAULT_HARDWARE_CONFIG
        if not isinstance(hardware_config, HardwareConfig):
            raise TypeError("hardware_config must be a HardwareConfig")
        self.hardware_config = hardware_config
        self.model_config = model_config or load_model_config(self.model_name)
        if not isinstance(self.model_config, self.config_type):
            raise TypeError(f"{type(self).__name__} requires {self.config_type.__name__}")
        if self.model_config.model_name != self.model_name:
            raise ValueError("model config does not belong to this mapper")

    @classmethod
    def from_json(cls, path: str | Path,
                  hardware_config: HardwareConfig | None = None) -> "PartitionedLayerMapper":
        """Load this model's config.json using its own configuration validator."""
        with Path(path).open(encoding="utf-8") as stream:
            model = cls.config_type.from_dict(json.load(stream), model_name=cls.model_name)
        return cls(hardware_config, model_config=model)

    def _components(self, xfold: bool) -> list:
        raise NotImplementedError("model mapper must define its layer components")

    def _normalize_request(self, request: HardwareMappingRequest) -> HardwareMappingRequest:
        return request

    def _component_ops(self, context: LayerContext) -> tuple[list[list[Operator]], dict[str, EPRouting]]:
        groups = [component.build(context) for component in self._components(context.xfold)]
        if context.xfold:
            # Leave Flash available for the preceding layer's asynchronous KV
            # prefetch throughout the path from PreAttention to Core Attention.
            for group in groups:
                group[:] = [replace(op, weight_memory=WeightMemory(WeightSource.DRAM))
                            for op in group]
                if any(isinstance(op, FlashAttentionOp) for op in group):
                    break
        routes = {}
        if context.inference.moe_parallel_strategy is not MoEParallelStrategy.EP:
            return groups, routes
        moe = groups[-1]
        expanded = []

        def communication(op_id: str, width: int, dtype: int, *, combine: bool) -> FabricOp:
            routing = EPRouting(
                self.hardware_config.chip_count, context.model.num_experts,
                context.model.top_k, context.inference.accepted_tokens_per_step,
                width * dtype, combine,
            )
            routes[op_id] = routing
            batch = context.inference.global_batch_size
            return FabricOp(op_id, "alltoall", tuple(range(routing.chip_count)),
                            transfer_bytes=routing.transfer_bytes(0, batch, batch))

        fc1 = moe[0]
        expanded.append(communication("moe.ep_dispatch", fc1.K, fc1.input_dtype_bytes,
                                      combine=False))
        for op in moe:
            expanded.append(op)
            if op.op_id in ("moe.moe_fc2", "moe.fold_latent_q_indexer_k_w", "moe.fold_indexer_q"):
                op_id = ("moe.ep_combine" if op.op_id == "moe.moe_fc2"
                         else "moe.ep_fold_combine")
                expanded.append(communication(op_id, op.N, op.output_dtype_bytes,
                                              combine=True))
        groups[-1] = expanded
        return groups, routes

    def build_op_list(self, inference: InferenceConfig, *, xfold: bool = False,
                      memory_overrides: Mapping[str, WeightMemory] | None = None) -> list[Operator]:
        """Produce sequential Ops; optional policies override individual ports/rates."""
        request = self._normalize_request(HardwareMappingRequest(self.model_name, inference, xfold))
        context = LayerContext(self.model_config, inference, request.xfold)
        groups, _ = self._component_ops(context)
        ops = [op for group in groups for op in group]
        overrides = dict(memory_overrides or {})
        unknown = set(overrides) - {op.op_id for op in ops}
        if unknown:
            raise ValueError(f"unknown operator memory overrides: {sorted(unknown)}")
        if any(isinstance(op, FabricOp) and op.op_id in overrides for op in ops):
            raise ValueError("fabric communication has no weight memory policy")
        return [replace(op, weight_memory=overrides[op.op_id])
                if op.op_id in overrides else op for op in ops]

    def map(self, request: HardwareMappingRequest | InferenceConfig, *,
            xfold: bool = False,
            memory_overrides: Mapping[str, WeightMemory] | None = None,
            hbf_bandwidth_table: HBFBandwidthTable | None = None) -> HardwareMappingResult:
        """Map this model only, preserving the list order in the execution tree."""
        if isinstance(request, InferenceConfig):
            request = HardwareMappingRequest(self.model_name, request, xfold)
        elif not isinstance(request, HardwareMappingRequest):
            raise TypeError("request must be HardwareMappingRequest or InferenceConfig")
        elif xfold:
            request = replace(request, xfold=True)
        if request.model_name != self.model_name:
            raise ValueError(f"{type(self).__name__} only maps {self.model_name}")
        request = self._normalize_request(request)
        inference, model = request.inference_config, self.model_config
        chip = self.hardware_config.chip
        if (chip.noc.mesh_rows, chip.noc.mesh_columns) != (32, 64):
            raise ValueError("layer partition chains require a 32 x 64 PU mesh per chip")
        chips = self.hardware_config.chip_count
        local_batch = (inference.global_batch_size + chips - 1) // chips
        local_tokens = local_batch * inference.accepted_tokens_per_step
        ep = inference.moe_parallel_strategy is MoEParallelStrategy.EP
        if model.num_experts % chips:
            raise ValueError("expert count must be divisible by chip count")
        expert_ids = {"moe.moe_fc1", "moe.moe_fc2", "moe.fold_latent_q_indexer_k_w",
                      "moe.fold_indexer_q"}
        global_ops = self.build_op_list(inference, xfold=request.xfold, memory_overrides=memory_overrides)
        ops = []
        for op in global_ops:
            if isinstance(op, FabricOp):
                ops.append(op)
                continue
            if op.op_id in expert_ids:
                if ep:
                    op = replace(op, B=op.B // chips)
                else:
                    axis = "N" if op.op_id == "moe.moe_fc1" else "K"
                    if getattr(op, axis) % chips:
                        raise ValueError("expert tensor dimension must be divisible by chip count")
                    op = replace(op, **{axis: getattr(op, axis) // chips})
            else:
                axis = "B" if isinstance(op, (FlashAttentionOp, FusedIndexerScoreOp, MiniMaxFusedIndexerScoreOp)) else "M"
                op = replace(op, **{axis: local_batch if axis == "B" else local_tokens})
            ops.append(op)
        operators = {op.op_id: op for op in ops}
        mappings, hardware = {}, {}
        component_groups = []
        offset = 0
        context = LayerContext(model, inference, request.xfold)
        components = self._components(request.xfold)
        groups, routes = self._component_ops(context)
        for group in groups:
            count = len(group)
            component_groups.append(GroupNode(tuple(OpNode(op.op_id) for op in ops[offset:offset + count])))
            offset += count
        root = RootNode(inference.global_batch_size, tuple(operators), GroupNode(tuple(component_groups)))
        for global_op, op in zip(global_ops, ops):
            if isinstance(op, FabricOp):
                mappings[op.op_id] = OperatorMapping(
                    "size_bytes", "system.fabric", ep_routing=routes[op.op_id])
                network_bytes = sum(value for i, row in enumerate(op.transfer_bytes)
                                    for j, value in enumerate(row) if i != j)
                dimensions = {"size_bytes": max(1, network_bytes)}
                hardware[op.op_id] = OperatorHardwareMapping(
                    op.op_id, "FabricOp", "attention_dp_ffn_ep",
                    dimensions, dimensions, {}, participant_group=op.group,
                )
                continue
            is_attention = isinstance(op, (FlashAttentionOp, FusedIndexerScoreOp, MiniMaxFusedIndexerScoreOp))
            is_expert = op.op_id in expert_ids
            mappings[op.op_id] = OperatorMapping(
                "B" if is_attention else "M", "chip0.die0.pu0",
                batch_scaling_unit_count=context.routed_tokens if is_expert else None,
                batch_partition_degree=1 if is_expert else chips,
                demand_batch_partition_degree=chips,
            )
            if isinstance(op, FlashAttentionOp):
                dimensions = {key: getattr(op, key) for key in ("B", "M", "qk_K", "qk_N", "sv_K", "sv_N")}
                shape = {"B": op.B, "S": op.qk_N}
                if isinstance(op, FlashMSAOp):
                    dimensions["H"] = shape["H"] = op.H
                tile = op.partition_chain.compute_shape(shape)
                local = dict(dimensions, B=tile["B"], qk_N=tile["S"], sv_K=tile["S"])
                if isinstance(op, FlashMSAOp):
                    local["H"] = tile["H"]
                weight = 0
            else:
                dimensions = {key: getattr(op, key) for key in ("B", "M", "K", "N")}
                if isinstance(op, (FusedIndexerScoreOp, MiniMaxFusedIndexerScoreOp)):
                    tile = op.partition_chain.compute_shape({"B": op.B, "S": op.N})
                    local = dict(dimensions, B=tile["B"], N=tile["S"])
                    weight = 0
                else:
                    local = op.partition_chain.compute_shape(dimensions)
                    weight = local["B"] * local["K"] * local["N"] * op.weight_dtype_bytes
            hardware[op.op_id] = OperatorHardwareMapping(
                op.op_id, type(op).__name__,
                ",".join(f"{s.dimension}-{s.pu_dimension}-{s.factor}" for s in op.partition_chain.steps),
                {key: getattr(global_op, key) for key in dimensions}, dimensions, {}, local,
                weight_bytes=weight,
            )
        assignments = context.num_tokens * model.top_k
        quotient, remainder = divmod(assignments, model.num_experts)
        loads = tuple(quotient + (i < remainder) for i in range(model.num_experts))
        experts_per_chip = model.num_experts // chips
        ownership = tuple(
            tuple(range(index * experts_per_chip, (index + 1) * experts_per_chip))
            if ep else tuple(range(model.num_experts))
            for index in range(chips)
        )
        result = HardwareMappingResult(
            request, model, chips, self.hardware_config.chip.logic_die_count,
            0, local_batch, ownership, loads,
            root, operators, mappings, hardware, ops,
            tuple(type(component).__name__ for component in components),
        )
        if hbf_bandwidth_table is not None:
            result = apply_hbf_bandwidths(result, self.hardware_config, hbf_bandwidth_table)
        return result


class DSAMapper(PartitionedLayerMapper):
    """Shared DSA component assembly for DeepSeek V3.2 and GLM 5.1."""

    def _components(self, xfold: bool) -> list:
        return [PreAttn(), FusedIndexer(), CoreAttn(), PostAttn(),
                FusedFoldingMoE() if xfold else FusedMoE()]


class DeepSeekV32Mapper(DSAMapper):
    """DeepSeek V3.2 ordinary and Fold layer mapper."""

    model_name = "deepseek-v3.2"
    config_type = DeepSeekV32Config

    def _components(self, xfold: bool) -> list:
        components = super()._components(xfold)
        if xfold:
            components[1] = IndexerKeyCache()
        return components


class GLM51Mapper(DSAMapper):
    """GLM 5.1 ordinary and Fold layer mapper."""

    model_name = "glm-5.1"
    config_type = GLM51Config


class _MLAMapper(LegacyHardwareMapper):
    """Model-bound adapter retaining the existing non-DSA MLA implementation."""

    model_name: str

    def map(self, request: HardwareMappingRequest | InferenceConfig, *,
            hbf_bandwidth_table: HBFBandwidthTable | None = None) -> HardwareMappingResult:
        if isinstance(request, InferenceConfig):
            request = HardwareMappingRequest(self.model_name, request)
        if not isinstance(request, HardwareMappingRequest):
            raise TypeError("request must be HardwareMappingRequest or InferenceConfig")
        if request.model_name != self.model_name:
            raise ValueError(f"{type(self).__name__} only maps {self.model_name}")
        result = super().map(request)
        if hbf_bandwidth_table is not None:
            result = apply_hbf_bandwidths(result, self.hardware_config, hbf_bandwidth_table)
        return result

    def build_op_list(self, inference: InferenceConfig) -> list[Operator]:
        """Return the existing MLA layer in execution order."""
        return self.map(inference).op_list


class DeepSeekV3Mapper(PartitionedLayerMapper):
    """Dense MLA using the V3.2 ordinary components without an indexer.

    XFold requests run the ordinary path and report the effective mode as false.
    """

    model_name = "deepseek-v3"
    config_type = DeepSeekV3Config

    def _normalize_request(self, request: HardwareMappingRequest) -> HardwareMappingRequest:
        return replace(request, xfold=False) if request.xfold else request

    def _components(self, xfold: bool) -> list:
        return [PreAttn(), CoreAttn(), PostAttn(), FusedMoE()]


class KimiK25Mapper(_MLAMapper):
    """Mapper dedicated to the non-DSA Kimi K2.5 model."""

    model_name = "kimi-k2.5"


_MAPPERS = {mapper.model_name: mapper for mapper in (
    DeepSeekV3Mapper, DeepSeekV32Mapper, KimiK25Mapper, GLM51Mapper,
)}


def mapper_for_model(model_name: str,
                     hardware_config: HardwareConfig | None = None) -> PartitionedLayerMapper | _MLAMapper:
    """Return the dedicated mapper for a supported model."""
    if model_name == "minimax-m3":
        from .minimax import MiniMaxM3Mapper
        return MiniMaxM3Mapper(hardware_config)
    try:
        mapper = _MAPPERS[model_name]
    except KeyError as error:
        raise ValueError(f"unsupported model: {model_name}") from error
    if hardware_config is None and not issubclass(mapper, DSAMapper):
        hardware_config = DEFAULT_HARDWARE_CONFIG
    return mapper(hardware_config)
