"""Hardware mapping for one representative DeepSeek decode layer."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

from srammachine.frontend.modules import BMMOp, CommOp, Operator, VectorOp
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG, HardwareConfig
from srammachine.inference import InferenceConfig, MoEParallelStrategy
from srammachine.pipetree import (
    OperatorMapping, RootNode, build_root_node,
)

from .model import (
    DeepSeekV3Config, DeepSeekV32Config, ModelConfig, load_model_config,
)


# Fixed execution precisions for the current hardware path. They are not
# properties of an inference workload, so InferenceConfig does not store them.
_WEIGHT_DTYPE_BYTES = 1
_ACTIVATION_DTYPE_BYTES = 2
_KV_CACHE_DTYPE_BYTES = 2


def _positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _exact_div(value: int, divisor: int, name: str) -> int:
    if value % divisor:
        raise ValueError(f"{name} must be divisible by {divisor}, got {value}")
    return value // divisor


def _freeze_dimensions(value: Mapping[str, int]) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError("dimensions must be a mapping")
    copied = {}
    for name, dimension in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError("dimension names must be nonempty strings")
        _positive_integer(name, dimension)
        copied[name] = dimension
    return MappingProxyType(copied)


@dataclass(frozen=True)
class HardwareMappingRequest:
    """A decode-only mapping request for one representative model layer."""

    model_name: str
    inference_config: InferenceConfig

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a nonempty string")
        if not isinstance(self.inference_config, InferenceConfig):
            raise TypeError("inference_config must be an InferenceConfig")


@dataclass(frozen=True)
class OperatorHardwareMapping:
    """Traceable hierarchy dimensions for one generated model operator."""

    op_id: str
    op_kind: str
    parallel_strategy: str
    global_dimensions: Mapping[str, int]
    chip_dimensions: Mapping[str, int]
    die_dimensions: Mapping[str, int]
    pu_dimensions: Mapping[str, int] = field(default_factory=dict)
    participant_group: Tuple[Any, ...] = ()
    expert_ids: Tuple[int, ...] = ()
    tokens_per_expert: Optional[int] = None
    weight_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("op_id", "op_kind", "parallel_strategy"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in (
            "global_dimensions", "chip_dimensions", "die_dimensions",
            "pu_dimensions",
        ):
            object.__setattr__(self, name, _freeze_dimensions(getattr(self, name)))
        group = tuple(self.participant_group)
        if len(set(group)) != len(group):
            raise ValueError("participant_group must not contain duplicates")
        experts = tuple(self.expert_ids)
        if any(type(expert) is not int or expert < 0 for expert in experts):
            raise ValueError("expert_ids must contain nonnegative integers")
        if len(set(experts)) != len(experts):
            raise ValueError("expert_ids must not contain duplicates")
        if self.tokens_per_expert is not None:
            _positive_integer("tokens_per_expert", self.tokens_per_expert)
        if type(self.weight_bytes) is not int or self.weight_bytes < 0:
            raise ValueError("weight_bytes must be a nonnegative integer")
        object.__setattr__(self, "participant_group", group)
        object.__setattr__(self, "expert_ids", experts)


@dataclass(frozen=True)
class HardwareMappingResult:
    """Inputs accepted directly by TreeParser plus mapping provenance."""

    request: HardwareMappingRequest
    model_config: ModelConfig
    chip_count: int
    logic_die_count: int
    representative_chip_id: int
    local_batch_size: int
    expert_ownership: Tuple[Tuple[int, ...], ...]
    expert_token_loads: Tuple[int, ...]
    root_node: RootNode
    operators: Mapping[str, Operator]
    operator_mappings: Mapping[str, OperatorMapping]
    hardware_mappings: Mapping[str, OperatorHardwareMapping]

    def __post_init__(self) -> None:
        if not isinstance(self.request, HardwareMappingRequest):
            raise TypeError("request must be a HardwareMappingRequest")
        if not isinstance(self.model_config, (DeepSeekV3Config, DeepSeekV32Config)):
            raise TypeError("model_config must be a supported DeepSeek config")
        _positive_integer("chip_count", self.chip_count)
        _positive_integer("logic_die_count", self.logic_die_count)
        _positive_integer("local_batch_size", self.local_batch_size)
        if self.representative_chip_id != 0:
            raise ValueError("representative_chip_id must be zero")
        ownership = tuple(tuple(ids) for ids in self.expert_ownership)
        if len(ownership) != self.chip_count:
            raise ValueError("expert_ownership must contain one entry per chip")
        for ids in ownership:
            if any(type(expert) is not int or expert < 0 for expert in ids):
                raise ValueError("expert ownership IDs must be nonnegative integers")
            if len(ids) != len(set(ids)):
                raise ValueError("one chip cannot own the same expert twice")
        loads = tuple(self.expert_token_loads)
        if len(loads) != self.model_config.num_experts:
            raise ValueError("expert_token_loads must cover every routed expert")
        if any(type(load) is not int or load <= 0 for load in loads):
            raise ValueError("every routed expert must have a positive token load")
        if not isinstance(self.root_node, RootNode):
            raise TypeError("root_node must be a RootNode")
        operators = dict(self.operators)
        mappings = dict(self.operator_mappings)
        hardware = dict(self.hardware_mappings)
        expected = set(self.root_node.operator_order)
        if set(operators) != expected or set(mappings) != expected or set(hardware) != expected:
            raise ValueError("result mappings must exactly match RootNode operators")
        if any(not isinstance(value, Operator) for value in operators.values()):
            raise TypeError("operators must contain Operator values")
        if any(not isinstance(value, OperatorMapping) for value in mappings.values()):
            raise TypeError("operator_mappings must contain OperatorMapping values")
        if any(not isinstance(value, OperatorHardwareMapping) for value in hardware.values()):
            raise TypeError("hardware_mappings must contain OperatorHardwareMapping values")
        object.__setattr__(self, "operators", MappingProxyType(operators))
        object.__setattr__(self, "operator_mappings", MappingProxyType(mappings))
        object.__setattr__(self, "hardware_mappings", MappingProxyType(hardware))
        object.__setattr__(self, "expert_ownership", ownership)
        object.__setattr__(self, "expert_token_loads", loads)


class _LayerBuilder:
    PU = "chip0.die0.pu0"
    VECTOR = "chip0.die0.vector"
    DRAM = "chip0.die0.dram"
    SRAM = "chip0.die0.sram"
    NOC = "chip0.noc"
    FABRIC = "system.fabric"

    def __init__(
        self, request: HardwareMappingRequest, hardware_config: HardwareConfig,
        model_config: ModelConfig,
    ) -> None:
        self.request = request
        self.hardware_config = hardware_config
        self.model_config = model_config
        self.operators = {}
        self.mappings = {}
        self.hardware_mappings = {}
        self.order = []
        self.expert_ownership = ()
        self.expert_token_loads = ()

    @property
    def die_group(self) -> Tuple[str, ...]:
        return tuple(
            f"chip0.die{index}"
            for index in range(self.hardware_config.chip.logic_die_count)
        )

    @property
    def chip_group(self) -> Tuple[int, ...]:
        return tuple(range(self.hardware_config.chip_count))

    @property
    def representative_pu_row(self) -> Tuple[str, ...]:
        columns = self.hardware_config.chip.logic_die.pu_mesh_columns
        return tuple(f"chip0.die0.pu{column}" for column in range(columns))

    @property
    def representative_pu_column(self) -> Tuple[str, ...]:
        rows = self.hardware_config.chip.logic_die.pu_mesh_rows
        columns = self.hardware_config.chip.logic_die.pu_mesh_columns
        return tuple(
            f"chip0.die0.pu{row * columns}" for row in range(rows)
        )

    def _record(
        self, operator: Operator, mapping: OperatorMapping,
        hardware_mapping: OperatorHardwareMapping,
    ) -> None:
        op_id = operator.op_id
        if op_id in self.operators:
            raise ValueError(f"duplicate generated operator ID: {op_id}")
        expected_resource = (
            self.PU if isinstance(operator, BMMOp)
            else self.VECTOR if isinstance(operator, VectorOp)
            else self.NOC if operator.scope == "intra_chip"
            else self.FABRIC
        )
        if mapping.resource_id != expected_resource:
            raise ValueError("generated operator is bound to the wrong resource")
        self.order.append(op_id)
        self.operators[op_id] = operator
        self.mappings[op_id] = mapping
        self.hardware_mappings[op_id] = hardware_mapping

    def _pu_dimensions(self, die_dimensions: Mapping[str, int]) -> Mapping[str, int]:
        dimensions = dict(die_dimensions)
        rows = self.hardware_config.chip.logic_die.pu_mesh_rows
        columns = self.hardware_config.chip.logic_die.pu_mesh_columns
        dimensions["K"] = _ceil_div(dimensions["K"], rows)
        dimensions["N"] = _ceil_div(dimensions["N"], columns)
        return dimensions

    def add_bmm(
        self, op_id: str, op_kind: str, *,
        global_dimensions: Mapping[str, int],
        chip_dimensions: Mapping[str, int],
        die_dimensions: Mapping[str, int],
        parallel_strategy: str,
        weight_batches: int = 0,
        dram_read_bytes_per_token: int = 0,
        dram_write_bytes_per_token: int = 0,
        expert_ids: Sequence[int] = (),
        tokens_per_expert: Optional[int] = None,
    ) -> None:
        for dimensions in (global_dimensions, chip_dimensions, die_dimensions):
            if set(dimensions) != {"B", "M", "K", "N"}:
                raise ValueError("BMM dimensions must contain exactly B/M/K/N")
        pu_dimensions = self._pu_dimensions(die_dimensions)
        weight_bytes = 0
        load_bytes = 0
        if weight_batches:
            _positive_integer("weight_batches", weight_batches)
            weight_bytes = (
                weight_batches * die_dimensions["K"] * die_dimensions["N"]
                * _WEIGHT_DTYPE_BYTES
            )
            load_bytes = (
                weight_batches * pu_dimensions["K"] * pu_dimensions["N"]
                * _WEIGHT_DTYPE_BYTES
            )
        input_bytes = (
            pu_dimensions["B"] * pu_dimensions["M"] * pu_dimensions["K"]
            * _ACTIVATION_DTYPE_BYTES
        )
        output_bytes = (
            pu_dimensions["B"] * pu_dimensions["M"] * pu_dimensions["N"]
            * _ACTIVATION_DTYPE_BYTES
        )
        self.add_comm(
            f"{op_id}.input_broadcast", f"{op_kind}_input_broadcast",
            scope="intra_chip", kind="broadcast",
            group=self.representative_pu_row,
            root=self.PU,
            size_bytes=input_bytes,
            parallel_strategy=parallel_strategy,
        )
        operator = BMMOp(op_id, **pu_dimensions)
        mapping = OperatorMapping(
            "M", self.PU,
            self.DRAM if (
                weight_bytes or dram_read_bytes_per_token
                or dram_write_bytes_per_token
            ) else None,
            self.SRAM if (weight_bytes or load_bytes) else None,
            dram_read_once_bytes=weight_bytes,
            dram_read_bytes_per_token=dram_read_bytes_per_token,
            dram_write_bytes_per_token=dram_write_bytes_per_token,
            weight_load_fixed_bytes=load_bytes,
        )
        self._record(operator, mapping, OperatorHardwareMapping(
            op_id=op_id,
            op_kind=op_kind,
            parallel_strategy=parallel_strategy,
            global_dimensions=global_dimensions,
            chip_dimensions=chip_dimensions,
            die_dimensions=die_dimensions,
            pu_dimensions=pu_dimensions,
            expert_ids=tuple(expert_ids),
            tokens_per_expert=tokens_per_expert,
            weight_bytes=weight_bytes,
        ))
        self.add_comm(
            f"{op_id}.output_reduce", f"{op_kind}_output_reduce",
            scope="intra_chip", kind="reduce",
            group=self.representative_pu_column,
            root=self.PU,
            size_bytes=output_bytes,
            reduce_kind="sum",
            parallel_strategy=parallel_strategy,
        )

    def add_vector(
        self, op_id: str, op_kind: str, *,
        global_dimensions: Mapping[str, int],
        chip_dimensions: Mapping[str, int],
        die_dimensions: Mapping[str, int],
        parallel_strategy: str,
        flops_per_element: Optional[int] = None,
        expert_ids: Sequence[int] = (),
        tokens_per_expert: Optional[int] = None,
    ) -> None:
        if set(die_dimensions) != {"m", "n"}:
            raise ValueError("vector dimensions must contain exactly m/n")
        params = {}
        if flops_per_element is not None:
            _positive_integer("flops_per_element", flops_per_element)
            params["flops_per_element"] = flops_per_element
        operator = VectorOp(
            op_id, op_kind, die_dimensions["m"], die_dimensions["n"], params,
        )
        self._record(
            operator,
            OperatorMapping("m", self.VECTOR),
            OperatorHardwareMapping(
                op_id=op_id,
                op_kind=op_kind,
                parallel_strategy=parallel_strategy,
                global_dimensions=global_dimensions,
                chip_dimensions=chip_dimensions,
                die_dimensions=die_dimensions,
                expert_ids=tuple(expert_ids),
                tokens_per_expert=tokens_per_expert,
            ),
        )

    def add_comm(
        self, op_id: str, op_kind: str, *, scope: str, kind: str,
        group: Sequence[Any], parallel_strategy: str,
        size_bytes: Optional[int] = None,
        transfer_bytes: Optional[Sequence[Sequence[int]]] = None,
        root: Optional[Any] = None,
        reduce_kind: str = "sum",
    ) -> None:
        matrix = None
        if transfer_bytes is not None:
            matrix = tuple(tuple(row) for row in transfer_bytes)
        operator = CommOp(
            op_id, kind, scope, tuple(group), size_bytes=size_bytes,
            root=root, reduce_kind=reduce_kind, transfer_bytes=matrix,
        )
        resource = self.NOC if scope == "intra_chip" else self.FABRIC
        if size_bytes is not None:
            dimensions = {"size_bytes": max(size_bytes, 1)}
        else:
            network_bytes = sum(
                value for row_index, row in enumerate(matrix)
                for column_index, value in enumerate(row)
                if row_index != column_index
            )
            dimensions = {"size_bytes": max(network_bytes, 1)}
        self._record(
            operator,
            OperatorMapping("size_bytes", resource),
            OperatorHardwareMapping(
                op_id=op_id,
                op_kind=op_kind,
                parallel_strategy=parallel_strategy,
                global_dimensions=dimensions,
                chip_dimensions=dimensions,
                die_dimensions=dimensions,
                participant_group=tuple(group),
            ),
        )

    def build(self) -> HardwareMappingResult:
        root_node = build_root_node(
            batch_size=_ceil_div(
                self.request.inference_config.global_batch_size,
                self.hardware_config.chip_count,
            ),
            operator_order=self.order,
            operators=self.operators,
        )
        return HardwareMappingResult(
            request=self.request,
            model_config=self.model_config,
            chip_count=self.hardware_config.chip_count,
            logic_die_count=self.hardware_config.chip.logic_die_count,
            representative_chip_id=0,
            local_batch_size=root_node.batch_size,
            expert_ownership=self.expert_ownership,
            expert_token_loads=self.expert_token_loads,
            root_node=root_node,
            operators=self.operators,
            operator_mappings=self.mappings,
            hardware_mappings=self.hardware_mappings,
        )


class HardwareMapper:
    """Map supported DeepSeek decode layers onto the configured hierarchy."""

    def __init__(self, hardware_config: HardwareConfig = DEFAULT_HARDWARE_CONFIG):
        if not isinstance(hardware_config, HardwareConfig):
            raise TypeError("hardware_config must be a HardwareConfig")
        self.hardware_config = hardware_config

    def map(self, request: HardwareMappingRequest) -> HardwareMappingResult:
        if not isinstance(request, HardwareMappingRequest):
            raise TypeError("request must be a HardwareMappingRequest")
        model = load_model_config(request.model_name)
        self._validate(request, model)
        builder = _LayerBuilder(request, self.hardware_config, model)
        self._build_attention(builder)
        if (
            request.inference_config.moe_parallel_strategy
            is MoEParallelStrategy.TP
        ):
            self._build_moe_tp(builder)
        else:
            self._build_moe_ep(builder)
        return builder.build()

    def _validate(self, request: HardwareMappingRequest, model: ModelConfig) -> None:
        chips = self.hardware_config.chip_count
        dies = self.hardware_config.chip.logic_die_count
        if chips not in (16, 32):
            raise ValueError("decode mapping supports only 16 or 32 chips")
        if dies != 4:
            raise ValueError("decode mapping requires exactly four logic dies per chip")
        inference = request.inference_config
        if inference.input_sequence_length > model.max_position_embeddings:
            raise ValueError(
                "input_sequence_length exceeds the model context limit"
            )
        if inference.global_batch_size * model.top_k < model.num_experts:
            raise ValueError(
                "batch is too small to activate every routed expert under "
                "the balanced-routing assumption"
            )
        _exact_div(model.num_attention_heads, dies, "attention heads")
        _exact_div(
            model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim,
            dies,
            "MLA latent width",
        )
        if isinstance(model, DeepSeekV32Config):
            _exact_div(model.indexer_num_heads, dies, "indexer heads")
        tp_degree = (
            chips * dies
            if inference.moe_parallel_strategy is MoEParallelStrategy.TP
            else dies
        )
        _exact_div(model.moe_intermediate_size, tp_degree, "MoE intermediate size")
        _exact_div(model.num_experts, chips, "routed experts")

    @staticmethod
    def _dims(B: int, M: int, K: int, N: int) -> Mapping[str, int]:
        return {"B": B, "M": M, "K": K, "N": N}

    def _build_attention(self, builder: _LayerBuilder) -> None:
        request = builder.request
        model = builder.model_config
        chips = self.hardware_config.chip_count
        dies = self.hardware_config.chip.logic_die_count
        global_batch = request.inference_config.global_batch_size
        local_batch = _ceil_div(global_batch, chips)
        local_heads = model.num_attention_heads // dies
        latent_width = (
            model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim
        )
        local_latent = latent_width // dies
        dtype = _ACTIVATION_DTYPE_BYTES
        strategy = "attention_dp_die_tp4"

        builder.add_vector(
            "attn.input_norm", "rmsnorm",
            global_dimensions={"m": global_batch, "n": model.hidden_size},
            chip_dimensions={"m": local_batch, "n": model.hidden_size},
            die_dimensions={"m": local_batch, "n": model.hidden_size},
            parallel_strategy=strategy,
        )
        builder.add_bmm(
            "attn.latent_down", "mla_latent_down_projection",
            global_dimensions=self._dims(
                1, global_batch, model.hidden_size, latent_width,
            ),
            chip_dimensions=self._dims(
                1, local_batch, model.hidden_size, latent_width,
            ),
            die_dimensions=self._dims(
                1, local_batch, model.hidden_size, local_latent,
            ),
            parallel_strategy=strategy,
            weight_batches=1,
            dram_write_bytes_per_token=(
                model.kv_lora_rank + model.qk_rope_head_dim
            ) * _KV_CACHE_DTYPE_BYTES,
        )
        builder.add_comm(
            "attn.latent_allgather", "latent_allgather",
            scope="intra_chip", kind="allgather", group=builder.die_group,
            size_bytes=local_batch * local_latent * dtype,
            parallel_strategy=strategy,
        )
        for suffix, width in (
            ("q_norm", model.q_lora_rank),
            ("kv_norm", model.kv_lora_rank),
        ):
            builder.add_vector(
                f"attn.{suffix}", "rmsnorm",
                global_dimensions={"m": global_batch, "n": width},
                chip_dimensions={"m": local_batch, "n": width},
                die_dimensions={"m": local_batch, "n": width},
                parallel_strategy=strategy,
            )
        builder.add_vector(
            "attn.k_rope", "rope",
            global_dimensions={
                "m": global_batch, "n": model.qk_rope_head_dim,
            },
            chip_dimensions={
                "m": local_batch, "n": model.qk_rope_head_dim,
            },
            die_dimensions={
                "m": local_batch, "n": model.qk_rope_head_dim,
            },
            parallel_strategy=strategy,
        )
        builder.add_bmm(
            "attn.q_rope_projection", "q_rope_projection",
            global_dimensions=self._dims(
                1, global_batch, model.q_lora_rank,
                model.num_attention_heads * model.qk_rope_head_dim,
            ),
            chip_dimensions=self._dims(
                1, local_batch, model.q_lora_rank,
                model.num_attention_heads * model.qk_rope_head_dim,
            ),
            die_dimensions=self._dims(
                1, local_batch, model.q_lora_rank,
                local_heads * model.qk_rope_head_dim,
            ),
            parallel_strategy=strategy,
            weight_batches=1,
        )
        builder.add_vector(
            "attn.q_rope", "rope",
            global_dimensions={
                "m": global_batch * model.num_attention_heads,
                "n": model.qk_rope_head_dim,
            },
            chip_dimensions={
                "m": local_batch * model.num_attention_heads,
                "n": model.qk_rope_head_dim,
            },
            die_dimensions={
                "m": local_batch * local_heads,
                "n": model.qk_rope_head_dim,
            },
            parallel_strategy=strategy,
        )
        if isinstance(model, DeepSeekV32Config):
            self._build_dsa(builder, local_batch, strategy)

        attention_history = (
            min(request.inference_config.input_sequence_length, model.dsa_len)
            if isinstance(model, DeepSeekV32Config)
            else request.inference_config.input_sequence_length
        )
        global_head_batch = global_batch * model.num_attention_heads
        chip_head_batch = local_batch * model.num_attention_heads
        die_head_batch = local_batch * local_heads
        builder.add_bmm(
            "attn.qk_nope_absorb", "qk_nope_absorb_q",
            global_dimensions=self._dims(
                global_head_batch, 1, model.q_lora_rank, model.kv_lora_rank,
            ),
            chip_dimensions=self._dims(
                chip_head_batch, 1, model.q_lora_rank, model.kv_lora_rank,
            ),
            die_dimensions=self._dims(
                die_head_batch, 1, model.q_lora_rank, model.kv_lora_rank,
            ),
            parallel_strategy=strategy,
            weight_batches=local_heads,
        )
        builder.add_bmm(
            "attn.qk_nope", "qk_nope",
            global_dimensions=self._dims(
                global_head_batch, 1, model.kv_lora_rank, attention_history,
            ),
            chip_dimensions=self._dims(
                chip_head_batch, 1, model.kv_lora_rank, attention_history,
            ),
            die_dimensions=self._dims(
                die_head_batch, 1, model.kv_lora_rank, attention_history,
            ),
            parallel_strategy=strategy,
            dram_read_bytes_per_token=attention_history * (
                model.kv_lora_rank + model.qk_rope_head_dim
            ) * _KV_CACHE_DTYPE_BYTES,
        )
        builder.add_bmm(
            "attn.qk_rope", "qk_rope",
            global_dimensions=self._dims(
                global_head_batch, 1, model.qk_rope_head_dim,
                attention_history,
            ),
            chip_dimensions=self._dims(
                chip_head_batch, 1, model.qk_rope_head_dim,
                attention_history,
            ),
            die_dimensions=self._dims(
                die_head_batch, 1, model.qk_rope_head_dim,
                attention_history,
            ),
            parallel_strategy=strategy,
        )
        for op_id, op_kind, coefficient in (
            ("attn.score_add", "qk_score_add", 1),
            ("attn.softmax", "softmax", None),
        ):
            builder.add_vector(
                op_id, op_kind,
                global_dimensions={
                    "m": global_head_batch, "n": attention_history,
                },
                chip_dimensions={
                    "m": chip_head_batch, "n": attention_history,
                },
                die_dimensions={
                    "m": die_head_batch, "n": attention_history,
                },
                parallel_strategy=strategy,
                flops_per_element=coefficient,
            )
        builder.add_bmm(
            "attn.sv_latent", "sv_latent",
            global_dimensions=self._dims(
                global_head_batch, 1, attention_history, model.kv_lora_rank,
            ),
            chip_dimensions=self._dims(
                chip_head_batch, 1, attention_history, model.kv_lora_rank,
            ),
            die_dimensions=self._dims(
                die_head_batch, 1, attention_history, model.kv_lora_rank,
            ),
            parallel_strategy=strategy,
        )
        builder.add_bmm(
            "attn.vo_absorb", "vo_absorb",
            global_dimensions=self._dims(
                global_head_batch, 1, model.kv_lora_rank, model.hidden_size,
            ),
            chip_dimensions=self._dims(
                chip_head_batch, 1, model.kv_lora_rank, model.hidden_size,
            ),
            die_dimensions=self._dims(
                die_head_batch, 1, model.kv_lora_rank, model.hidden_size,
            ),
            parallel_strategy=strategy,
            weight_batches=local_heads,
        )
        builder.add_vector(
            "attn.local_head_reduce", "head_reduce",
            global_dimensions={"m": global_batch, "n": model.hidden_size},
            chip_dimensions={"m": local_batch, "n": model.hidden_size},
            die_dimensions={"m": local_batch, "n": model.hidden_size},
            parallel_strategy=strategy,
            flops_per_element=max(local_heads - 1, 1),
        )
        builder.add_comm(
            "attn.die_output_reduce", "die_output_reduce",
            scope="intra_chip", kind="allreduce", group=builder.die_group,
            size_bytes=local_batch * model.hidden_size * dtype,
            parallel_strategy=strategy,
        )
        for op_id, kind, coefficient in (
            ("attn.residual", "residual", 1),
            ("attn.output_norm", "rmsnorm", None),
        ):
            builder.add_vector(
                op_id, kind,
                global_dimensions={
                    "m": global_batch, "n": model.hidden_size,
                },
                chip_dimensions={
                    "m": local_batch, "n": model.hidden_size,
                },
                die_dimensions={
                    "m": local_batch, "n": model.hidden_size,
                },
                parallel_strategy=strategy,
                flops_per_element=coefficient,
            )

    def _build_dsa(
        self, builder: _LayerBuilder, local_batch: int, strategy: str,
    ) -> None:
        model = builder.model_config
        request = builder.request
        dies = self.hardware_config.chip.logic_die_count
        local_heads = model.indexer_num_heads // dies
        global_batch = request.inference_config.global_batch_size
        history = request.inference_config.input_sequence_length
        die_width = (
            local_heads * model.indexer_head_dim
            + model.indexer_head_dim + local_heads
        )
        chip_width = (
            model.indexer_num_heads * model.indexer_head_dim
            + model.indexer_head_dim + model.indexer_num_heads
        )
        builder.add_bmm(
            "attn.dsa_qkw", "indexer_qkw_projection",
            global_dimensions=self._dims(
                1, global_batch, model.hidden_size, chip_width,
            ),
            chip_dimensions=self._dims(
                1, local_batch, model.hidden_size, chip_width,
            ),
            die_dimensions=self._dims(
                1, local_batch, model.hidden_size, die_width,
            ),
            parallel_strategy=strategy,
            weight_batches=1,
            dram_write_bytes_per_token=(
                model.indexer_head_dim * _KV_CACHE_DTYPE_BYTES
            ),
        )
        builder.add_bmm(
            "attn.dsa_qk_score", "indexer_qk_score",
            global_dimensions=self._dims(
                global_batch, model.indexer_num_heads,
                model.indexer_head_dim, history,
            ),
            chip_dimensions=self._dims(
                local_batch, model.indexer_num_heads,
                model.indexer_head_dim, history,
            ),
            die_dimensions=self._dims(
                local_batch, local_heads, model.indexer_head_dim, history,
            ),
            parallel_strategy=strategy,
            dram_read_bytes_per_token=(
                history * model.indexer_head_dim * _KV_CACHE_DTYPE_BYTES
            ),
        )
        builder.add_vector(
            "attn.dsa_relu", "relu",
            global_dimensions={
                "m": global_batch * model.indexer_num_heads, "n": history,
            },
            chip_dimensions={
                "m": local_batch * model.indexer_num_heads, "n": history,
            },
            die_dimensions={
                "m": local_batch * local_heads, "n": history,
            },
            parallel_strategy=strategy,
            flops_per_element=1,
        )
        builder.add_vector(
            "attn.dsa_head_reduce", "indexer_weighted_head_reduce",
            global_dimensions={"m": global_batch, "n": history},
            chip_dimensions={"m": local_batch, "n": history},
            die_dimensions={"m": local_batch, "n": history},
            parallel_strategy=strategy,
            flops_per_element=max(2 * local_heads - 1, 1),
        )
        builder.add_comm(
            "attn.dsa_score_reduce", "indexer_score_reduce",
            scope="intra_chip", kind="allreduce", group=builder.die_group,
            size_bytes=(
                local_batch * history * _ACTIVATION_DTYPE_BYTES
            ),
            parallel_strategy=strategy,
        )

    @staticmethod
    def _balanced_expert_loads(
        global_batch: int, top_k: int, num_experts: int,
    ) -> Tuple[int, ...]:
        base, remainder = divmod(global_batch * top_k, num_experts)
        return tuple(
            base + (expert < remainder) for expert in range(num_experts)
        )

    @staticmethod
    def _groups_for_experts(
        expert_ids: Sequence[int], loads: Sequence[int],
    ) -> Tuple[Tuple[Tuple[int, ...], int], ...]:
        grouped = {}
        for expert_id in expert_ids:
            grouped.setdefault(loads[expert_id], []).append(expert_id)
        return tuple(
            (tuple(grouped[token_count]), token_count)
            for token_count in sorted(grouped, reverse=True)
        )

    def _add_expert_group(
        self, builder: _LayerBuilder, *, prefix: str,
        expert_ids: Sequence[int], tokens_per_expert: int,
        global_expert_count: int, tp_degree: int, parallel_strategy: str,
    ) -> None:
        model = builder.model_config
        chips = self.hardware_config.chip_count
        local_intermediate = model.moe_intermediate_size // tp_degree
        expert_count = len(expert_ids)
        _positive_integer("global_expert_count", global_expert_count)
        suffix = f"tokens{tokens_per_expert}"
        chip_tp_degree = chips if parallel_strategy == "moe_tp" else 1
        chip_expert_count = expert_count
        common = dict(
            expert_ids=expert_ids,
            tokens_per_expert=tokens_per_expert,
            parallel_strategy=parallel_strategy,
        )
        builder.add_bmm(
            f"{prefix}.{suffix}.up_gate", "moe_up_gate",
            global_dimensions=self._dims(
                global_expert_count, tokens_per_expert,
                model.hidden_size, 2 * model.moe_intermediate_size,
            ),
            chip_dimensions=self._dims(
                chip_expert_count, tokens_per_expert, model.hidden_size,
                2 * model.moe_intermediate_size // chip_tp_degree,
            ),
            die_dimensions=self._dims(
                expert_count, tokens_per_expert, model.hidden_size,
                2 * local_intermediate,
            ),
            weight_batches=expert_count,
            **common,
        )
        builder.add_vector(
            f"{prefix}.{suffix}.silu", "silu",
            global_dimensions={
                "m": global_expert_count * tokens_per_expert,
                "n": model.moe_intermediate_size,
            },
            chip_dimensions={
                "m": chip_expert_count * tokens_per_expert,
                "n": model.moe_intermediate_size // chip_tp_degree,
            },
            die_dimensions={
                "m": expert_count * tokens_per_expert,
                "n": local_intermediate,
            },
            **common,
        )
        builder.add_bmm(
            f"{prefix}.{suffix}.down", "moe_down",
            global_dimensions=self._dims(
                global_expert_count, tokens_per_expert,
                model.moe_intermediate_size, model.hidden_size,
            ),
            chip_dimensions=self._dims(
                chip_expert_count, tokens_per_expert,
                model.moe_intermediate_size // chip_tp_degree,
                model.hidden_size,
            ),
            die_dimensions=self._dims(
                expert_count, tokens_per_expert,
                local_intermediate, model.hidden_size,
            ),
            weight_batches=expert_count,
            **common,
        )

    def _build_moe_tp(self, builder: _LayerBuilder) -> None:
        request = builder.request
        model = builder.model_config
        chips = self.hardware_config.chip_count
        dies = self.hardware_config.chip.logic_die_count
        global_batch = request.inference_config.global_batch_size
        local_batch = _ceil_div(global_batch, chips)
        dtype = _ACTIVATION_DTYPE_BYTES
        strategy = "moe_tp"
        builder.add_comm(
            "moe.tp_input_allgather", "dp_to_tp_allgather",
            scope="inter_chip", kind="allgather", group=builder.chip_group,
            size_bytes=local_batch * model.hidden_size * dtype,
            parallel_strategy=strategy,
        )
        loads = self._balanced_expert_loads(
            global_batch, model.top_k, model.num_experts,
        )
        builder.expert_token_loads = loads
        builder.expert_ownership = tuple(
            tuple(range(model.num_experts)) for _ in range(chips)
        )
        for expert_ids, token_count in self._groups_for_experts(
            range(model.num_experts), loads,
        ):
            self._add_expert_group(
                builder, prefix="moe.tp", expert_ids=expert_ids,
                tokens_per_expert=token_count,
                global_expert_count=len(expert_ids), tp_degree=chips * dies,
                parallel_strategy=strategy,
            )
        builder.add_comm(
            "moe.tp_die_reduce", "die_partial_reduce",
            scope="intra_chip", kind="allreduce", group=builder.die_group,
            size_bytes=(
                global_batch * model.hidden_size * dtype
            ),
            parallel_strategy=strategy,
        )
        builder.add_comm(
            "moe.tp_chip_reduce_scatter", "tp_to_dp_reduce_scatter",
            scope="inter_chip", kind="reduce_scatter",
            group=builder.chip_group,
            size_bytes=(
                global_batch * model.hidden_size * dtype
            ),
            parallel_strategy=strategy,
        )
        builder.add_vector(
            "moe.tp_residual", "residual",
            global_dimensions={
                "m": global_batch, "n": model.hidden_size,
            },
            chip_dimensions={"m": local_batch, "n": model.hidden_size},
            die_dimensions={"m": local_batch, "n": model.hidden_size},
            parallel_strategy=strategy,
            flops_per_element=1,
        )

    @staticmethod
    def _transport_matrix(
        row_totals: Sequence[int], column_totals: Sequence[int],
        bytes_per_item: int,
    ) -> Tuple[Tuple[int, ...], ...]:
        """Deterministically realize balanced source and owner totals."""
        if sum(row_totals) != sum(column_totals):
            raise ValueError("all-to-all row and column totals must match")
        columns = list(column_totals)
        remaining_total = sum(columns)
        matrix = []
        for row_index, row_total in enumerate(row_totals):
            if row_index == len(row_totals) - 1:
                allocations = columns[:]
            else:
                allocations = [
                    row_total * value // remaining_total for value in columns
                ]
                missing = row_total - sum(allocations)
                order = sorted(
                    range(len(columns)),
                    key=lambda index: (
                        -(row_total * columns[index] % remaining_total),
                        (index - row_index) % len(columns),
                    ),
                )
                for index in order:
                    if not missing:
                        break
                    if allocations[index] < columns[index]:
                        allocations[index] += 1
                        missing -= 1
                if missing:
                    for index in range(len(columns)):
                        take = min(
                            missing, columns[index] - allocations[index],
                        )
                        allocations[index] += take
                        missing -= take
                        if not missing:
                            break
            columns = [
                value - used for value, used in zip(columns, allocations)
            ]
            remaining_total -= row_total
            matrix.append(tuple(
                value * bytes_per_item for value in allocations
            ))
        if any(columns):
            raise ValueError("failed to construct all-to-all transfer matrix")
        return tuple(matrix)

    def _build_moe_ep(self, builder: _LayerBuilder) -> None:
        request = builder.request
        model = builder.model_config
        chips = self.hardware_config.chip_count
        dies = self.hardware_config.chip.logic_die_count
        global_batch = request.inference_config.global_batch_size
        local_batch = _ceil_div(global_batch, chips)
        dtype = _ACTIVATION_DTYPE_BYTES
        strategy = "moe_ep_die_tp4"
        loads = self._balanced_expert_loads(
            global_batch, model.top_k, model.num_experts,
        )
        batch_base, batch_remainder = divmod(
            global_batch, chips,
        )
        row_totals = tuple(
            (batch_base + (chip < batch_remainder)) * model.top_k
            for chip in range(chips)
        )
        experts_per_chip = _ceil_div(model.num_experts, chips)
        ownership = tuple(
            tuple(range(
                start, min(start + experts_per_chip, model.num_experts),
            ))
            for start in range(0, model.num_experts, experts_per_chip)
        )
        if len(ownership) != chips:
            raise ValueError("expert ownership must produce one shard per chip")
        builder.expert_token_loads = loads
        builder.expert_ownership = ownership
        column_totals = tuple(
            sum(loads[expert] for expert in ids) for ids in ownership
        )
        dispatch = self._transport_matrix(
            row_totals, column_totals, model.hidden_size * dtype,
        )
        builder.add_comm(
            "moe.ep_dispatch", "expert_dispatch", scope="inter_chip",
            kind="alltoall", group=builder.chip_group,
            transfer_bytes=dispatch, parallel_strategy=strategy,
        )
        representative_experts = ownership[0]
        for expert_ids, token_count in self._groups_for_experts(
            representative_experts, loads,
        ):
            self._add_expert_group(
                builder, prefix="moe.ep", expert_ids=expert_ids,
                tokens_per_expert=token_count,
                global_expert_count=sum(
                    load == token_count for load in loads
                ),
                tp_degree=dies,
                parallel_strategy=strategy,
            )
        representative_assignments = sum(
            loads[expert] for expert in representative_experts
        )
        builder.add_comm(
            "moe.ep_die_reduce", "die_partial_reduce",
            scope="intra_chip", kind="allreduce", group=builder.die_group,
            size_bytes=(
                representative_assignments * model.hidden_size * dtype
            ),
            parallel_strategy=strategy,
        )
        combine = tuple(
            tuple(dispatch[column][row] for column in range(chips))
            for row in range(chips)
        )
        builder.add_comm(
            "moe.ep_combine", "expert_combine", scope="inter_chip",
            kind="alltoall", group=builder.chip_group,
            transfer_bytes=combine, parallel_strategy=strategy,
        )
        builder.add_vector(
            "moe.ep_residual", "residual",
            global_dimensions={
                "m": global_batch, "n": model.hidden_size,
            },
            chip_dimensions={"m": local_batch, "n": model.hidden_size},
            die_dimensions={"m": local_batch, "n": model.hidden_size},
            parallel_strategy=strategy,
            flops_per_element=1,
        )


__all__ = [
    "MoEParallelStrategy",
    "HardwareMappingRequest",
    "HardwareMappingResult",
    "OperatorHardwareMapping",
    "HardwareMapper",
]
