"""Hardware mapping for one representative MLA/MoE decode layer."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

from srammachine.frontend.modules import (
    BMMOp, CommOp, FlashAttentionOp, FusedIndexerScoreOp, Operator, VectorOp,
)
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG, HardwareConfig
from srammachine.inference import InferenceConfig, MoEParallelStrategy
from srammachine.pipetree import (
    OperatorMapping, RootNode, build_root_node,
)

from .model import ModelConfig, load_model_config


# Fixed execution precisions for the current hardware path. They are not
# properties of an inference workload, so InferenceConfig does not store them.
_WEIGHT_DTYPE_BYTES = 1
_ACTIVATION_DTYPE_BYTES = 1


def _positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _shared_effective_bytes(logical_bytes: int, logic_die_count: int) -> int:
    """Legacy representative-die share of a chip-shared cache stream.

    Attention converts four such shares into one chip-level command. The
    cache/head reuse estimate itself is intentionally unchanged here.
    """
    _positive_integer("logical_bytes", logical_bytes)
    _positive_integer("logic_die_count", logic_die_count)
    return _ceil_div(logical_bytes, logic_die_count)


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
        if not isinstance(self.model_config, ModelConfig):
            raise TypeError("model_config must be a supported model config")
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
        if any(type(load) is not int or load < 0 for load in loads):
            raise ValueError("routed expert token loads must be nonnegative integers")
        if not any(loads):
            raise ValueError("at least one routed expert must have a positive token load")
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
    CHIP_DRAM = "chip0.dram"
    CHIP_SRAM = "chip0.sram"
    CHIP_VECTOR = "chip0.vector"
    NOC_INPUT = "chip0.noc_input"
    NOC_OUTPUT = "chip0.noc_output"
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

    @staticmethod
    def chip_mesh_row() -> Tuple[str, ...]:
        return (_LayerBuilder.PU,) + tuple(
            f"chip0.pu{column}" for column in range(1, 8)
        )

    @staticmethod
    def chip_mesh_column() -> Tuple[str, ...]:
        return (_LayerBuilder.PU,) + tuple(
            f"chip0.pu{row * 8}" for row in range(1, 8)
        )

    @property
    def die_collective_parallel_link_count(self) -> int:
        """Physical NoC links striping one adjacent-die logical edge."""
        chip = self.hardware_config.chip
        return min(
            chip.horizontal_logic_die_boundary_link_count,
            chip.vertical_logic_die_boundary_link_count,
        )

    def _record(
        self, operator: Operator, mapping: OperatorMapping,
        hardware_mapping: OperatorHardwareMapping,
    ) -> None:
        op_id = operator.op_id
        if op_id in self.operators:
            raise ValueError(f"duplicate generated operator ID: {op_id}")
        expected_resource = (
            self.PU if isinstance(operator, (BMMOp, FlashAttentionOp))
            else self.VECTOR if isinstance(operator, VectorOp)
            else self._noc_resource(operator.kind)
            if operator.scope == "intra_chip"
            else self.FABRIC
        )
        valid_resources = (
            {self.NOC_INPUT, self.NOC_OUTPUT}
            if isinstance(operator, CommOp)
            and operator.scope == "intra_chip"
            and operator.kind == "p2p"
            else {expected_resource}
        )
        if (isinstance(operator, VectorOp) and op_id.startswith(("moe.tp", "moe.ep"))
                and operator.kind in ("silu", "residual")):
            valid_resources = {self.VECTOR, self.CHIP_VECTOR}
        if isinstance(operator, VectorOp) and op_id.startswith("attn."):
            valid_resources = {self.VECTOR, self.CHIP_VECTOR}
        if mapping.resource_id not in valid_resources:
            raise ValueError("generated operator is bound to the wrong resource")
        self.order.append(op_id)
        self.operators[op_id] = operator
        self.mappings[op_id] = mapping
        self.hardware_mappings[op_id] = hardware_mapping

    @classmethod
    def _noc_resource(cls, kind: str) -> str:
        return (
            cls.NOC_INPUT
            if kind in ("broadcast", "allgather")
            else cls.NOC_OUTPUT
        )

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
        weight_bytes_override: Optional[int] = None,
        weight_logical_bytes: Optional[int] = None,
        load_bytes_override: Optional[int] = None,
        load_logical_bytes: Optional[int] = None,
        shared_die_factor: int = 1,
        dram_read_bytes_per_token: int = 0,
        dram_read_logical_bytes_per_token: Optional[int] = None,
        dram_write_bytes_per_token: int = 0,
        dram_write_logical_bytes_per_token: Optional[int] = None,
        expert_ids: Sequence[int] = (),
        tokens_per_expert: Optional[int] = None,
        batch_axis: str = "M",
        batch_partition_degree: int = 1,
        demand_batch_partition_degree: Optional[int] = None,
        traffic_batch_partition_degree: Optional[int] = None,
        sram_read_bytes_per_mapped_token: int = 0,
        sram_read_logical_bytes_per_mapped_token: Optional[int] = None,
        sram_read_data_kind: Optional[str] = None,
        pu_dimensions_override: Optional[Mapping[str, int]] = None,
        pu_weight_batches: Optional[int] = None,
        input_group: Optional[Sequence[Any]] = None,
        input_kind: str = "broadcast",
        input_suffix: str = "input_broadcast",
        input_noc_direction: Optional[str] = None,
        output_kind: str = "reduce",
        output_group: Optional[Sequence[Any]] = None,
        output_suffix: str = "output_reduce",
        input_bytes_override: Optional[int] = None,
        output_bytes_override: Optional[int] = None,
        fused_indexer_score: bool = False,
        main_gemm_flops_override: Optional[int] = None,
        batch_scaling_unit_count: Optional[int] = None,
        dram_resource_id: Optional[str] = None,
        sram_resource_id: Optional[str] = None,
        input_parallel_link_count: int = 1,
        output_parallel_link_count: int = 1,
    ) -> None:
        for dimensions in (global_dimensions, chip_dimensions, die_dimensions):
            if set(dimensions) != {"B", "M", "K", "N"}:
                raise ValueError("BMM dimensions must contain exactly B/M/K/N")
        pu_dimensions = (
            self._pu_dimensions(die_dimensions)
            if pu_dimensions_override is None
            else dict(pu_dimensions_override)
        )
        if set(pu_dimensions) != {"B", "M", "K", "N"}:
            raise ValueError("PU BMM dimensions must contain exactly B/M/K/N")
        weight_bytes = 0
        load_bytes = 0
        if weight_batches:
            _positive_integer("weight_batches", weight_batches)
            if pu_weight_batches is None:
                pu_weight_batches = weight_batches
            _positive_integer("pu_weight_batches", pu_weight_batches)
            weight_bytes = (
                weight_batches * die_dimensions["K"] * die_dimensions["N"]
                * _WEIGHT_DTYPE_BYTES
            )
            load_bytes = (
                pu_weight_batches * pu_dimensions["K"] * pu_dimensions["N"]
                * _WEIGHT_DTYPE_BYTES
            )
        elif pu_weight_batches is not None:
            raise ValueError("pu_weight_batches requires resident weights")
        if weight_bytes_override is not None:
            if not weight_batches:
                raise ValueError("weight_bytes_override requires resident weights")
            _positive_integer("weight_bytes_override", weight_bytes_override)
            weight_bytes = weight_bytes_override
        if load_bytes_override is not None:
            if not weight_batches:
                raise ValueError("load_bytes_override requires resident weights")
            _positive_integer("load_bytes_override", load_bytes_override)
            load_bytes = load_bytes_override
        input_bytes = (
            pu_dimensions["B"] * pu_dimensions["M"] * pu_dimensions["K"]
            * _ACTIVATION_DTYPE_BYTES
        )
        if input_bytes_override is not None:
            _positive_integer("input_bytes_override", input_bytes_override)
            input_bytes = input_bytes_override
        output_bytes = (
            pu_dimensions["B"] * pu_dimensions["M"] * pu_dimensions["N"]
            * _ACTIVATION_DTYPE_BYTES
        )
        if output_bytes_override is not None:
            _positive_integer("output_bytes_override", output_bytes_override)
            output_bytes = output_bytes_override
        if main_gemm_flops_override is not None and not fused_indexer_score:
            raise ValueError(
                "main_gemm_flops_override requires fused_indexer_score"
            )
        self.add_comm(
            f"{op_id}.{input_suffix}", f"{op_kind}_{input_suffix}",
            scope="intra_chip", kind=input_kind,
            group=(
                self.representative_pu_row
                if input_group is None else tuple(input_group)
            ),
            root=self.PU if input_kind == "broadcast" else None,
            size_bytes=input_bytes,
            parallel_strategy=parallel_strategy,
            batch_partition_degree=(
                traffic_batch_partition_degree
                if traffic_batch_partition_degree is not None
                else batch_partition_degree
            ),
            noc_direction=input_noc_direction,
            batch_scaling_unit_count=batch_scaling_unit_count,
            parallel_link_count=input_parallel_link_count,
        )
        operator_type = FusedIndexerScoreOp if fused_indexer_score else BMMOp
        operator_kwargs = dict(pu_dimensions)
        if fused_indexer_score:
            operator_kwargs["main_gemm_flops_override"] = (
                main_gemm_flops_override
            )
        operator = operator_type(op_id, **operator_kwargs)
        mapping = OperatorMapping(
            batch_axis, self.PU,
            (dram_resource_id or self.DRAM) if (
                weight_bytes or dram_read_bytes_per_token
                or dram_write_bytes_per_token
            ) else None,
            (sram_resource_id or self.SRAM) if (
                weight_bytes or load_bytes or sram_read_bytes_per_mapped_token
            ) else None,
            dram_read_once_bytes=weight_bytes,
            dram_read_bytes_per_token=dram_read_bytes_per_token,
            dram_write_bytes_per_token=dram_write_bytes_per_token,
            dram_read_once_logical_bytes=weight_logical_bytes,
            dram_read_logical_bytes_per_token=(
                dram_read_logical_bytes_per_token
            ),
            dram_write_logical_bytes_per_token=(
                dram_write_logical_bytes_per_token
            ),
            weight_load_fixed_bytes=load_bytes,
            weight_load_logical_fixed_bytes=load_logical_bytes,
            weight_shape=(
                {"B": weight_batches, "K": die_dimensions["K"],
                 "N": die_dimensions["N"]}
                if weight_batches else None
            ),
            batch_partition_degree=batch_partition_degree,
            demand_batch_partition_degree=demand_batch_partition_degree,
            sram_read_bytes_per_mapped_token=(
                sram_read_bytes_per_mapped_token
            ),
            sram_read_logical_bytes_per_mapped_token=(
                sram_read_logical_bytes_per_mapped_token
            ),
            sram_read_data_kind=sram_read_data_kind,
            shared_die_factor=shared_die_factor,
            batch_scaling_unit_count=batch_scaling_unit_count,
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
            f"{op_id}.{output_suffix}", f"{op_kind}_{output_suffix}",
            scope="intra_chip", kind=output_kind,
            group=(
                self.representative_pu_column
                if output_group is None else tuple(output_group)
            ),
            root=self.PU if output_kind == "reduce" else None,
            size_bytes=output_bytes,
            reduce_kind="sum",
            parallel_strategy=parallel_strategy,
            batch_partition_degree=(
                traffic_batch_partition_degree
                if traffic_batch_partition_degree is not None
                else batch_partition_degree
            ),
            batch_scaling_unit_count=batch_scaling_unit_count,
            parallel_link_count=output_parallel_link_count,
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
        batch_partition_degree: int = 1,
        batch_scaling_unit_count: Optional[int] = None,
        resource_id: Optional[str] = None,
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
            OperatorMapping(
                "m", resource_id or self.VECTOR,
                batch_partition_degree=batch_partition_degree,
                batch_scaling_unit_count=batch_scaling_unit_count,
            ),
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

    def add_flash_attention(
        self, op_id: str, op_kind: str, *,
        global_dimensions: Mapping[str, int],
        chip_dimensions: Mapping[str, int],
        die_dimensions: Mapping[str, int],
        parallel_strategy: str,
        dram_read_bytes_per_token: int,
        sram_read_bytes_per_mapped_token: int,
        dram_read_logical_bytes_per_token: Optional[int] = None,
        sram_read_logical_bytes_per_mapped_token: Optional[int] = None,
        shared_die_factor: int = 1,
        batch_axis: str = "B",
        pu_dimensions_override: Optional[Mapping[str, int]] = None,
        dram_resource_id: Optional[str] = None,
        sram_resource_id: Optional[str] = None,
        input_group: Optional[Sequence[str]] = None,
        output_group: Optional[Sequence[str]] = None,
    ) -> None:
        """Add one fused FlashAttention critical path on a representative PU."""
        names = {"B", "M", "qk_K", "qk_N", "sv_K", "sv_N"}
        for dimensions in (global_dimensions, chip_dimensions, die_dimensions):
            if set(dimensions) != names:
                raise ValueError(
                    "FlashAttention dimensions must contain exactly "
                    "B/M/qk_K/qk_N/sv_K/sv_N"
                )
        rows = self.hardware_config.chip.logic_die.pu_mesh_rows
        columns = self.hardware_config.chip.logic_die.pu_mesh_columns
        pu_dimensions = dict(die_dimensions)
        pu_dimensions["qk_K"] = _ceil_div(die_dimensions["qk_K"], rows)
        pu_dimensions["qk_N"] = _ceil_div(die_dimensions["qk_N"], columns)
        pu_dimensions["sv_K"] = _ceil_div(die_dimensions["sv_K"], columns)
        pu_dimensions["sv_N"] = _ceil_div(die_dimensions["sv_N"], columns)
        if pu_dimensions_override is not None:
            pu_dimensions = dict(pu_dimensions_override)
            if set(pu_dimensions) != names:
                raise ValueError("invalid FlashAttention PU dimensions")

        input_bytes = (
            pu_dimensions["B"] * pu_dimensions["M"]
            * pu_dimensions["qk_K"] * _ACTIVATION_DTYPE_BYTES
        )
        output_bytes = (
            pu_dimensions["B"] * pu_dimensions["M"]
            * pu_dimensions["sv_N"] * _ACTIVATION_DTYPE_BYTES
        )
        self.add_comm(
            f"{op_id}.input_broadcast", f"{op_kind}_input_broadcast",
            scope="intra_chip", kind="broadcast",
            group=self.representative_pu_row if input_group is None else tuple(input_group), root=self.PU,
            size_bytes=input_bytes, parallel_strategy=parallel_strategy,
        )
        operator = FlashAttentionOp(op_id, **pu_dimensions)
        mapping = OperatorMapping(
            batch_axis, self.PU, dram_resource_id or self.DRAM,
            sram_resource_id or self.SRAM,
            dram_read_bytes_per_token=dram_read_bytes_per_token,
            dram_read_logical_bytes_per_token=(
                dram_read_logical_bytes_per_token
            ),
            sram_read_bytes_per_mapped_token=(
                sram_read_bytes_per_mapped_token
            ),
            sram_read_logical_bytes_per_mapped_token=(
                sram_read_logical_bytes_per_mapped_token
            ),
            sram_read_data_kind="flash_kv",
            shared_die_factor=shared_die_factor,
        )
        self._record(operator, mapping, OperatorHardwareMapping(
            op_id=op_id,
            op_kind=op_kind,
            parallel_strategy=parallel_strategy,
            global_dimensions=global_dimensions,
            chip_dimensions=chip_dimensions,
            die_dimensions=die_dimensions,
            pu_dimensions=pu_dimensions,
        ))
        self.add_comm(
            f"{op_id}.output_reduce", f"{op_kind}_output_reduce",
            scope="intra_chip", kind="reduce",
            group=self.representative_pu_column if output_group is None else tuple(output_group), root=self.PU,
            size_bytes=output_bytes, reduce_kind="sum",
            parallel_strategy=parallel_strategy,
        )

    def add_comm(
        self, op_id: str, op_kind: str, *, scope: str, kind: str,
        group: Sequence[Any], parallel_strategy: str,
        size_bytes: Optional[int] = None,
        transfer_bytes: Optional[Sequence[Sequence[int]]] = None,
        root: Optional[Any] = None,
        reduce_kind: str = "sum",
        parallel_link_count: int = 1,
        batch_partition_degree: int = 1,
        dram_write_bytes_per_token: int = 0,
        dram_write_logical_bytes_per_token: Optional[int] = None,
        shared_die_factor: int = 1,
        noc_direction: Optional[str] = None,
        batch_scaling_unit_count: Optional[int] = None,
    ) -> None:
        matrix = None
        if transfer_bytes is not None:
            matrix = tuple(tuple(row) for row in transfer_bytes)
        operator = CommOp(
            op_id, kind, scope, tuple(group), size_bytes=size_bytes,
            root=root, reduce_kind=reduce_kind, transfer_bytes=matrix,
            parallel_link_count=parallel_link_count,
        )
        if noc_direction not in (None, "input", "output"):
            raise ValueError("noc_direction must be input or output")
        if scope == "inter_chip" and noc_direction is not None:
            raise ValueError("noc_direction applies only to intra-chip NoC")
        resource = self.FABRIC
        if scope == "intra_chip":
            resource = (
                self.NOC_INPUT if noc_direction == "input"
                else self.NOC_OUTPUT if noc_direction == "output"
                else self._noc_resource(kind)
            )
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
            OperatorMapping(
                "size_bytes", resource,
                self.DRAM if dram_write_bytes_per_token else None,
                dram_write_bytes_per_token=dram_write_bytes_per_token,
                dram_write_logical_bytes_per_token=(
                    dram_write_logical_bytes_per_token
                ),
                batch_partition_degree=batch_partition_degree,
                shared_die_factor=shared_die_factor,
                batch_scaling_unit_count=batch_scaling_unit_count,
            ),
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
    """Map supported MLA/MoE decode layers onto the configured hierarchy."""

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
        # MTP1 is an explicit simulator-side what-if mode.  It is deliberately
        # allowed even when the source model card reports no native next-token
        # prediction layer (currently Kimi K2.5): the mapper force-applies the
        # same ideal two-token workload transformation requested by the user.
        # ``max_position_embeddings`` remains descriptive model-card metadata,
        # not a simulator limit.  Hardware studies may intentionally evaluate
        # hypothetical contexts beyond the model's published training window.
        # InferenceConfig already guarantees a positive input sequence length.
        assignments = (
            inference.global_batch_size
            * inference.accepted_tokens_per_step
            * model.top_k
        )
        if (
            inference.moe_parallel_strategy is MoEParallelStrategy.EP
            and assignments < chips
        ):
            raise ValueError(
                "EP batch is too small to activate at least one routed expert "
                "on every chip under the balanced-routing assumption"
            )
        _exact_div(model.num_attention_heads, dies, "attention heads")
        _exact_div(
            model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim,
            dies,
            "MLA latent width",
        )
        if model.dsa:
            if model.indexer_num_heads is None:
                raise ValueError("DSA model requires indexer_num_heads")
            if model.indexer_head_dim is None:
                raise ValueError("DSA model requires indexer_head_dim")
            if model.dsa_len is None:
                raise ValueError("DSA model requires dsa_len")
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

    def _attention_projection(
        self, builder: _LayerBuilder, op_id: str, op_kind: str,
        global_dimensions: Mapping[str, int],
        chip_dimensions: Mapping[str, int], strategy: str, *,
        weight_batches: int = 1, batch_axis: str = "M",
        dram_write_bytes_per_token: int = 0,
    ) -> None:
        """One representative tile of a chip-wide K8×N8 projection."""
        chip = chip_dimensions
        pu = self._dims(chip["B"], chip["M"],
                        _ceil_div(chip["K"], 8), _ceil_div(chip["N"], 8))
        builder.add_bmm(
            op_id, op_kind, global_dimensions=global_dimensions,
            chip_dimensions=chip, die_dimensions=chip,
            pu_dimensions_override=pu, parallel_strategy=strategy,
            weight_batches=weight_batches, batch_axis=batch_axis,
            dram_resource_id=builder.CHIP_DRAM,
            sram_resource_id=builder.CHIP_SRAM,
            load_bytes_override=(
                weight_batches * chip["K"] * chip["N"] * _WEIGHT_DTYPE_BYTES
            ),
            dram_write_bytes_per_token=dram_write_bytes_per_token,
            input_bytes_override=(
                chip["B"] * chip["M"] * chip["K"] * _ACTIVATION_DTYPE_BYTES
            ),
            output_bytes_override=(
                chip["B"] * chip["M"] * chip["N"] * _ACTIVATION_DTYPE_BYTES
            ),
            input_group=builder.chip_mesh_row(),
            output_group=builder.chip_mesh_column(),
            input_parallel_link_count=8,
            output_parallel_link_count=8,
        )

    def _build_attention(self, builder: _LayerBuilder) -> None:
        request = builder.request
        model = builder.model_config
        kv_dtype_bytes = request.inference_config.kv_cache_bytes_per_element
        chips = self.hardware_config.chip_count
        dies = self.hardware_config.chip.logic_die_count
        base_global_batch = request.inference_config.global_batch_size
        token_multiplier = request.inference_config.accepted_tokens_per_step
        # MTP1 evaluates two query tokens with the same cached context.  Scale
        # all Attention arithmetic and activation communication by two, while
        # the per-request DRAM/SRAM cache traffic below remains unchanged.
        global_batch = base_global_batch * token_multiplier
        chip_batch = _ceil_div(base_global_batch, chips) * token_multiplier
        pu_rows = self.hardware_config.chip.logic_die.pu_mesh_rows
        pu_columns = self.hardware_config.chip.logic_die.pu_mesh_columns
        latent_width = (
            model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim
        )
        strategy = "attention_chip_dp_pu_k8_n8"

        builder.add_vector(
            "attn.input_norm", "rmsnorm",
            global_dimensions={"m": global_batch, "n": model.hidden_size},
            chip_dimensions={"m": chip_batch, "n": model.hidden_size},
            die_dimensions={"m": chip_batch, "n": model.hidden_size},
            parallel_strategy=strategy,
            resource_id=builder.CHIP_VECTOR,
        )
        self._attention_projection(
            builder,
            "attn.latent_down", "mla_latent_down_projection",
            global_dimensions=self._dims(
                1, global_batch, model.hidden_size, latent_width,
            ),
            chip_dimensions=self._dims(
                1, chip_batch, model.hidden_size, latent_width,
            ),
            strategy=strategy,
            dram_write_bytes_per_token=(
                model.kv_lora_rank + model.qk_rope_head_dim
            ) * kv_dtype_bytes,
        )
        if model.use_qk_norm:
            for suffix, width in (
                ("q_norm", model.q_lora_rank),
                ("kv_norm", model.kv_lora_rank),
            ):
                builder.add_vector(
                    f"attn.{suffix}", "rmsnorm",
                    global_dimensions={"m": global_batch, "n": width},
                    chip_dimensions={"m": chip_batch, "n": width},
                    die_dimensions={"m": chip_batch, "n": width},
                    parallel_strategy=strategy,
                    resource_id=builder.CHIP_VECTOR,
                )
        builder.add_vector(
            "attn.k_rope", "rope",
            global_dimensions={
                "m": global_batch, "n": model.qk_rope_head_dim,
            },
            chip_dimensions={
                "m": chip_batch, "n": model.qk_rope_head_dim,
            },
            die_dimensions={
                "m": chip_batch, "n": model.qk_rope_head_dim,
            },
            parallel_strategy=strategy,
            resource_id=builder.CHIP_VECTOR,
        )
        self._attention_projection(
            builder,
            "attn.q_rope_projection", "q_rope_projection",
            global_dimensions=self._dims(
                1, global_batch, model.q_lora_rank,
                model.num_attention_heads * model.qk_rope_head_dim,
            ),
            chip_dimensions=self._dims(
                1, chip_batch, model.q_lora_rank,
                model.num_attention_heads * model.qk_rope_head_dim,
            ),
            strategy=strategy,
        )
        builder.add_vector(
            "attn.q_rope", "rope",
            global_dimensions={
                "m": global_batch * model.num_attention_heads,
                "n": model.qk_rope_head_dim,
            },
            chip_dimensions={
                "m": chip_batch * model.num_attention_heads,
                "n": model.qk_rope_head_dim,
            },
            die_dimensions={
                "m": chip_batch * model.num_attention_heads,
                "n": model.qk_rope_head_dim,
            },
            parallel_strategy=strategy,
            resource_id=builder.CHIP_VECTOR,
        )
        if model.dsa:
            self._build_dsa(builder, chip_batch, strategy)

        attention_history = (
            min(request.inference_config.input_sequence_length, model.dsa_len)
            if model.dsa
            else request.inference_config.input_sequence_length
        )
        global_head_batch = global_batch * model.num_attention_heads
        chip_head_batch = chip_batch * model.num_attention_heads
        self._attention_projection(
            builder,
            "attn.qk_nope_absorb", "qk_nope_absorb_q",
            global_dimensions=self._dims(
                global_head_batch, 1, model.q_lora_rank, model.kv_lora_rank,
            ),
            chip_dimensions=self._dims(
                chip_head_batch, 1, model.q_lora_rank, model.kv_lora_rank,
            ),
            strategy=strategy,
            weight_batches=model.num_attention_heads,
            batch_axis="B",
        )
        # The unfused path materializes scores after QK, normalizes across
        # the complete sequence, then distributes probabilities for SV.
        # Eight independent request/head rows run in parallel; each row's
        # eight PUs own distinct sequence shards.
        fused_qk_width = model.kv_lora_rank + model.qk_rope_head_dim
        row_head_batch = _exact_div(
            chip_head_batch, 8, "attention row head batch",
        )
        sequence_shard = _ceil_div(attention_history, 8)
        chip_score_bytes = (
            chip_head_batch * attention_history * _ACTIVATION_DTYPE_BYTES
        )
        builder.add_bmm(
            "attn.qk_fused", "qk_fused",
            global_dimensions=self._dims(
                global_head_batch, 1, fused_qk_width, attention_history,
            ),
            chip_dimensions=self._dims(
                chip_head_batch, 1, fused_qk_width, attention_history,
            ),
            die_dimensions=self._dims(
                chip_head_batch, 1, fused_qk_width, attention_history,
            ),
            pu_dimensions_override=self._dims(
                row_head_batch, 1, fused_qk_width, sequence_shard,
            ),
            parallel_strategy=strategy,
            dram_resource_id=builder.CHIP_DRAM,
            sram_resource_id=builder.CHIP_SRAM,
            input_group=builder.chip_mesh_row(),
            output_kind="p2p",
            output_suffix="output_transfer",
            output_group=(builder.PU, builder.CHIP_VECTOR),
            output_bytes_override=chip_score_bytes,
            output_parallel_link_count=8,
            batch_axis="B",
            dram_read_bytes_per_token=(
                4 * _shared_effective_bytes(
                    attention_history * fused_qk_width
                    * kv_dtype_bytes,
                    dies,
                )
            ),
            sram_read_bytes_per_mapped_token=(
                4 * _shared_effective_bytes(
                    (
                        _ceil_div(model.kv_lora_rank, pu_rows)
                        + _ceil_div(model.qk_rope_head_dim, pu_rows)
                    )
                    * _ceil_div(attention_history, pu_columns)
                    * kv_dtype_bytes,
                    dies,
                )
            ),
            sram_read_data_kind="kv_fused",
        )
        builder.add_vector(
            "attn.softmax", "softmax",
            global_dimensions={"m": global_head_batch,
                               "n": attention_history},
            chip_dimensions={"m": chip_head_batch,
                             "n": attention_history},
            die_dimensions={"m": chip_head_batch,
                            "n": attention_history},
            parallel_strategy=strategy,
            resource_id=builder.CHIP_VECTOR,
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
                chip_head_batch, 1, attention_history, model.kv_lora_rank,
            ),
            pu_dimensions_override=self._dims(
                row_head_batch, 1, sequence_shard, model.kv_lora_rank,
            ),
            parallel_strategy=strategy,
            dram_resource_id=builder.CHIP_DRAM,
            sram_resource_id=builder.CHIP_SRAM,
            input_kind="p2p",
            input_suffix="input_transfer",
            input_group=(builder.CHIP_VECTOR, builder.PU),
            input_noc_direction="input",
            input_bytes_override=chip_score_bytes,
            input_parallel_link_count=8,
            output_group=builder.chip_mesh_row(),
            batch_axis="B",
            dram_read_bytes_per_token=(
                4 * _shared_effective_bytes(
                    attention_history * model.kv_lora_rank
                    * kv_dtype_bytes, dies,
                )
            ),
            sram_read_bytes_per_mapped_token=(
                4 * _shared_effective_bytes(
                    _ceil_div(attention_history, pu_rows)
                    * _ceil_div(model.kv_lora_rank, pu_columns)
                    * kv_dtype_bytes, dies,
                )
            ),
            sram_read_data_kind="kv_value",
        )

        # O projection contracts both the local-head and latent dimensions.
        # Folding heads into K is algebraically identical to the batched form,
        # keeps the PU FLOPs and weight tile unchanged, and reduces the NoC
        # output from per-head partials to one hidden tile per request.
        self._attention_projection(
            builder,
            "attn.vo_absorb", "vo_absorb_head_folded_k",
            global_dimensions=self._dims(
                1, global_batch,
                model.num_attention_heads * model.kv_lora_rank,
                model.hidden_size,
            ),
            chip_dimensions=self._dims(
                1, chip_batch,
                model.num_attention_heads * model.kv_lora_rank,
                model.hidden_size,
            ),
            strategy=strategy,
            batch_axis="M",
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
                    "m": chip_batch, "n": model.hidden_size,
                },
                die_dimensions={
                    "m": chip_batch, "n": model.hidden_size,
                },
                parallel_strategy=strategy,
                flops_per_element=coefficient,
                resource_id=builder.CHIP_VECTOR,
            )

    def _build_dsa(
        self, builder: _LayerBuilder, chip_batch: int, strategy: str,
    ) -> None:
        model = builder.model_config
        if model.indexer_num_heads is None or model.indexer_head_dim is None:
            raise ValueError("DSA model requires complete indexer dimensions")
        request = builder.request
        kv_dtype_bytes = request.inference_config.kv_cache_bytes_per_element
        base_global_batch = request.inference_config.global_batch_size
        token_multiplier = request.inference_config.accepted_tokens_per_step
        dies = self.hardware_config.chip.logic_die_count
        pu_rows = self.hardware_config.chip.logic_die.pu_mesh_rows
        pu_columns = self.hardware_config.chip.logic_die.pu_mesh_columns
        pu_count = dies * pu_rows * pu_columns
        global_batch = base_global_batch * token_multiplier
        history = request.inference_config.input_sequence_length
        local_req = _ceil_div(
            base_global_batch, self.hardware_config.chip_count,
        )
        local_heads = model.indexer_num_heads // dies
        chip_width = (
            model.indexer_num_heads * model.indexer_head_dim
            + model.indexer_head_dim + model.indexer_num_heads
        )
        self._attention_projection(
            builder,
            "attn.dsa_qkw", "indexer_qkw_projection",
            global_dimensions=self._dims(
                1, global_batch, model.hidden_size, chip_width,
            ),
            chip_dimensions=self._dims(
                1, chip_batch, model.hidden_size, chip_width,
            ),
            strategy=strategy,
            dram_write_bytes_per_token=(
                model.indexer_head_dim * kv_dtype_bytes
            ),
        )

        # Preserve the old per-PU Q x K_cache GEMM work as an explicit scalar.
        # The fused post-processing terms intentionally use the new BMKN.
        old_pu_B = chip_batch
        old_pu_M = _ceil_div(local_heads, pu_rows)
        old_pu_N = _ceil_div(history, pu_columns)
        old_main_gemm_flops = (
            2 * old_pu_B * old_pu_M * model.indexer_head_dim * old_pu_N
        )
        top_k = min(2048, history)

        if base_global_batch <= 128:
            pu_dimensions = self._dims(
                chip_batch, _ceil_div(model.indexer_num_heads, 8),
                model.indexer_head_dim, _ceil_div(history, 8),
            )
            batch_partition_degree = 1
            output_kind = "reduce"
            output_group = builder.chip_mesh_column()
            output_suffix = "output_reduce"
            output_bytes = (
                chip_batch * pu_dimensions["N"] * _ACTIVATION_DTYPE_BYTES
            )
            sequence_partitions = 8
            output_links = 1
            main_gemm_override = None
        elif base_global_batch < 1024:
            # Use the largest equal per-request PU allocation and leave any
            # remainder idle. This keeps custom meshes and non-power-of-two
            # local request counts valid without overlapping sequence shards.
            pu_per_req = pu_count // local_req
            _positive_integer("PU count per request", pu_per_req)
            pu_dimensions = self._dims(
                token_multiplier, model.indexer_num_heads,
                model.indexer_head_dim, _ceil_div(history, pu_per_req),
            )
            batch_partition_degree = local_req
            output_kind = "p2p"
            output_group = (builder.PU, "chip0.output")
            output_suffix = "output_transfer"
            local_top_k = min(2048, pu_dimensions["N"])
            output_bytes = (
                local_req * pu_per_req * token_multiplier * local_top_k * 4
            )
            sequence_partitions = pu_per_req
            output_links = 8
            main_gemm_override = old_main_gemm_flops
        else:
            # At BS=1024 on a 32-chip system there are only 32 local
            # requests. Inactive PUs are left idle; active PUs own one
            # complete request. For local_req >= 64 the requested ratio is
            # exact and each active PU owns req_per_pu requests.
            req_per_pu = (
                1 if local_req < pu_count else _exact_div(
                    local_req, pu_count, "requests per PU",
                )
            )
            pu_dimensions = self._dims(
                req_per_pu * token_multiplier, model.indexer_num_heads,
                model.indexer_head_dim, history,
            )
            batch_partition_degree = pu_count
            output_kind = "p2p"
            output_group = (builder.PU, "chip0.output")
            output_suffix = "output_transfer"
            # Each top-k candidate carries a 2-byte shard-local token ID and
            # a 2-byte score. The source PU identifies the shard origin.
            output_bytes = local_req * token_multiplier * top_k * 4
            sequence_partitions = 1
            output_links = 8
            main_gemm_override = old_main_gemm_flops

        builder.add_bmm(
            "attn.dsa_qk_score", "indexer_qk_score",
            global_dimensions=self._dims(
                global_batch, model.indexer_num_heads,
                model.indexer_head_dim, history,
            ),
            chip_dimensions=self._dims(
                chip_batch, model.indexer_num_heads,
                model.indexer_head_dim, history,
            ),
            die_dimensions=self._dims(
                chip_batch, model.indexer_num_heads,
                model.indexer_head_dim, history,
            ),
            parallel_strategy=strategy,
            # Retain four former representative-die cache shares as a single
            # chip command. The existing request/sequence reuse estimate is
            # intentionally unchanged.
            dram_read_bytes_per_token=(
                4 * _shared_effective_bytes(
                    history * model.indexer_head_dim * kv_dtype_bytes, dies,
                )
            ),
            dram_resource_id=builder.CHIP_DRAM,
            sram_resource_id=builder.CHIP_SRAM,
            batch_axis="B",
            pu_dimensions_override=pu_dimensions,
            fused_indexer_score=True,
            main_gemm_flops_override=main_gemm_override,
            sram_read_bytes_per_mapped_token=(
                sequence_partitions * 4 * _shared_effective_bytes(
                    model.indexer_head_dim * pu_dimensions["N"]
                    * kv_dtype_bytes,
                    dies,
                )
            ),
            sram_read_data_kind="dsa_key",
            batch_partition_degree=batch_partition_degree,
            demand_batch_partition_degree=1,
            traffic_batch_partition_degree=1,
            input_bytes_override=(
                chip_batch * model.indexer_num_heads
                * model.indexer_head_dim * _ACTIVATION_DTYPE_BYTES
            ),
            input_parallel_link_count=8,
            output_kind=output_kind,
            output_group=output_group,
            output_suffix=output_suffix,
            output_bytes_override=output_bytes,
            output_parallel_link_count=output_links,
            input_group=builder.chip_mesh_row(),
        )
        if base_global_batch <= 128:
            builder.add_comm(
                "attn.dsa_score_allgather", "indexer_score_allgather",
                scope="intra_chip", kind="allgather",
                group=builder.chip_mesh_row(),
                size_bytes=chip_batch * _ceil_div(history, 8)
                * _ACTIVATION_DTYPE_BYTES,
                parallel_strategy=strategy,
            )

    @staticmethod
    def _balanced_expert_loads(
        global_batch: int, top_k: int, num_experts: int,
    ) -> Tuple[int, ...]:
        """Return deterministic TP loads, allowing a sparse active subset."""
        assignments = global_batch * top_k
        if assignments >= num_experts:
            # Preserve the established large-batch model: every expert gets
            # the same floored load and the remainder is intentionally omitted.
            base = assignments // num_experts
            return (base,) * num_experts
        # Routed experts are shape-equivalent here, so deterministic low IDs
        # represent the sparse active subset without changing critical latency.
        return (1,) * assignments + (0,) * (num_experts - assignments)

    @staticmethod
    def _balanced_ep_expert_loads(
        global_batch: int, top_k: int, num_experts: int,
        ownership: Sequence[Sequence[int]],
    ) -> Tuple[int, ...]:
        """Return equal per-chip EP pressure with deterministic sparse experts."""
        assignments = global_batch * top_k
        if assignments >= num_experts:
            base = assignments // num_experts
            return (base,) * num_experts
        assignments_per_chip = assignments // len(ownership)
        loads = [0] * num_experts
        for expert_ids in ownership:
            if assignments_per_chip > len(expert_ids):
                raise ValueError("EP sparse load exceeds one chip's expert ownership")
            for expert_id in expert_ids[:assignments_per_chip]:
                loads[expert_id] = 1
        return tuple(loads)

    @staticmethod
    def _groups_for_experts(
        expert_ids: Sequence[int], loads: Sequence[int],
    ) -> Tuple[Tuple[Tuple[int, ...], int], ...]:
        grouped = {}
        for expert_id in expert_ids:
            if loads[expert_id] == 0:
                continue
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
        is_ep = parallel_strategy == "moe_ep_die_tp4"
        is_tp = parallel_strategy == "moe_tp"
        pu_rows = self.hardware_config.chip.logic_die.pu_mesh_rows
        pu_columns = self.hardware_config.chip.logic_die.pu_mesh_columns
        pu_count = pu_rows * pu_columns
        common = dict(
            expert_ids=expert_ids,
            tokens_per_expert=tokens_per_expert,
            parallel_strategy=parallel_strategy,
            batch_scaling_unit_count=tokens_per_expert,
        )
        if is_tp:
            # Two-dimensional TP follows the physical hierarchy: chips shard
            # Up/Gate N, dies shard K, and the 16 PUs partition experts.
            chip_intermediate = _exact_div(
                model.moe_intermediate_size, chips,
                "TP intermediate dimension across chips",
            )
            up_die_dimensions = self._dims(
                expert_count,
                tokens_per_expert,
                _exact_div(
                    model.hidden_size, len(builder.die_group),
                    "TP Up/Gate K dimension across dies",
                ),
                2 * chip_intermediate,
            )
        else:
            chip_intermediate = local_intermediate
            up_die_dimensions = self._dims(
                expert_count, tokens_per_expert, model.hidden_size,
                2 * local_intermediate,
            )
        up_overrides = {}
        if is_ep:
            # EP Up/Gate uses all four K rows but only two N shards.  The
            # other factor of two partitions experts, so a representative
            # input is broadcast to two columns rather than all four.  This
            # preserves PU FLOPs and weight bytes while halving the input
            # critical path.
            expert_partitions = 2
            n_partitions = pu_columns // expert_partitions
            up_overrides = {
                "pu_dimensions_override": self._dims(
                    _ceil_div(expert_count, expert_partitions),
                    tokens_per_expert,
                    _exact_div(
                        model.hidden_size, pu_rows,
                        "EP Up/Gate K dimension",
                    ),
                    _exact_div(
                        2 * local_intermediate, n_partitions,
                        "EP Up/Gate N dimension",
                    ),
                ),
                "pu_weight_batches": _ceil_div(
                    expert_count, expert_partitions,
                ),
                "input_group": builder.representative_pu_row[:n_partitions],
            }
        elif is_tp:
            # Keep the die-local K/N shards whole within one PU.  Distinct
            # PUs own distinct experts, preserving PU FLOPs and weight bytes
            # without replicating the long input across a PU row.
            pu_experts = _ceil_div(expert_count, pu_count)
            up_overrides = {
                "pu_dimensions_override": self._dims(
                    pu_experts,
                    tokens_per_expert,
                    up_die_dimensions["K"],
                    up_die_dimensions["N"],
                ),
                "pu_weight_batches": pu_experts,
                "input_kind": "p2p",
                "input_suffix": "input_transfer",
                "input_group": ("chip0.die0.input", builder.PU),
                "input_noc_direction": "input",
                "output_kind": "p2p",
                "output_group": (builder.PU, "chip0.die0.output"),
                "output_suffix": "output_transfer",
            }
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
            die_dimensions=up_die_dimensions,
            weight_batches=expert_count,
            **up_overrides,
            **common,
        )
        if is_tp:
            # The four K-sharded dies produce partial Up/Gate outputs.  Model
            # allreduce explicitly as reduce-scatter on noc_output followed
            # by allgather on noc_input.  The end-to-end bytes/time are the
            # same, while the two full-duplex directions can be independently
            # pipelined across microbatches.
            up_output_bytes = (
                expert_count * tokens_per_expert
                * 2 * chip_intermediate * _ACTIVATION_DTYPE_BYTES
            )
            builder.add_comm(
                f"{prefix}.{suffix}.up_die_reduce_scatter",
                "moe_up_gate_die_reduce_scatter",
                scope="intra_chip", kind="reduce_scatter",
                group=builder.die_group,
                size_bytes=up_output_bytes,
                parallel_link_count=(
                    builder.die_collective_parallel_link_count
                ),
                parallel_strategy=parallel_strategy,
                batch_scaling_unit_count=tokens_per_expert,
            )
            builder.add_comm(
                f"{prefix}.{suffix}.up_die_allgather",
                "moe_up_gate_die_allgather",
                scope="intra_chip", kind="allgather",
                group=builder.die_group,
                size_bytes=_exact_div(
                    up_output_bytes, len(builder.die_group),
                    "TP Up/Gate reduce-scatter output",
                ),
                parallel_link_count=(
                    builder.die_collective_parallel_link_count
                ),
                parallel_strategy=parallel_strategy,
                batch_scaling_unit_count=tokens_per_expert,
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
                "n": chip_intermediate,
            },
            **common,
        )
        if is_ep:
            # Up/Gate is column-parallel across dies.  Gather the much
            # smaller post-SiLU intermediate before making Down output-
            # parallel across dies, avoiding a full-hidden die allreduce.
            builder.add_comm(
                f"{prefix}.{suffix}.intermediate_die_allgather",
                "moe_intermediate_die_allgather",
                scope="intra_chip", kind="allgather",
                group=builder.die_group,
                size_bytes=(
                    expert_count * tokens_per_expert * local_intermediate
                    * _ACTIVATION_DTYPE_BYTES
                ),
                parallel_link_count=(
                    builder.die_collective_parallel_link_count
                ),
                parallel_strategy=parallel_strategy,
                batch_scaling_unit_count=tokens_per_expert,
            )
            die_hidden = _exact_div(
                model.hidden_size, len(builder.die_group),
                "EP Down hidden dimension",
            )
            down_pu_experts = _ceil_div(expert_count, pu_rows)
            down_die_dimensions = self._dims(
                expert_count, tokens_per_expert,
                model.moe_intermediate_size, die_hidden,
            )
            down_overrides = {
                "pu_dimensions_override": self._dims(
                    down_pu_experts,
                    tokens_per_expert,
                    model.moe_intermediate_size,
                    _exact_div(
                        die_hidden, pu_columns,
                        "EP Down N dimension",
                    ),
                ),
                "pu_weight_batches": down_pu_experts,
                "output_kind": "p2p",
                "output_group": (
                    builder.PU, "chip0.die0.output",
                ),
                "output_suffix": "output_transfer",
            }
        elif is_tp:
            # Invert the Up/Gate layout: chips shard Down K and dies shard N.
            # PUs again partition experts only, so both local transfers are
            # p2p rather than a broadcast/reduce across a PU row or column.
            down_pu_experts = _ceil_div(expert_count, pu_count)
            down_die_dimensions = self._dims(
                expert_count, tokens_per_expert,
                chip_intermediate,
                _exact_div(
                    model.hidden_size, len(builder.die_group),
                    "TP Down N dimension across dies",
                ),
            )
            down_overrides = {
                "pu_dimensions_override": self._dims(
                    down_pu_experts,
                    tokens_per_expert,
                    down_die_dimensions["K"],
                    down_die_dimensions["N"],
                ),
                "pu_weight_batches": down_pu_experts,
                "input_kind": "p2p",
                "input_suffix": "input_transfer",
                "input_group": ("chip0.die0.input", builder.PU),
                "input_noc_direction": "input",
                "output_kind": "p2p",
                "output_group": (
                    builder.PU, "chip0.die0.output",
                ),
                "output_suffix": "output_transfer",
            }
        else:
            down_die_dimensions = self._dims(
                expert_count, tokens_per_expert,
                local_intermediate, model.hidden_size,
            )
            down_overrides = {}
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
            die_dimensions=down_die_dimensions,
            weight_batches=expert_count,
            **down_overrides,
            **common,
        )

    @staticmethod
    def _hierarchical_moe_tp_degrees(
        model: ModelConfig, base_global_batch: int,
    ) -> Tuple[int, int]:
        """Return the fixed token-group and chip-TP degrees for one model."""
        if model.num_experts == 384:
            thresholds = ((6144, 8, 2), (3072, 4, 4), (1536, 2, 8))
        else:
            thresholds = ((4096, 8, 2), (2048, 4, 4), (1024, 2, 8))
        # The middle G4 x TP4 mapping remains preferable at the largest batch
        # sizes.  Keep the G8 x TP2 tier defined below for reference, but let
        # this rule cover its activation range for every supported model.
        middle_threshold = thresholds[1][0]
        if base_global_batch >= middle_threshold:
            return 4, 4
        for threshold, token_groups, chip_tp_degree in thresholds:
            if base_global_batch >= threshold:
                return token_groups, chip_tp_degree
        return 1, 16

    def _add_hierarchical_tp_expert_group(
        self, builder: _LayerBuilder, *, expert_ids: Sequence[int],
        tokens_per_expert: int, token_groups: int, chip_tp_degree: int,
        phase: str, prefix: str = "moe.tp",
    ) -> None:
        """Map one representative die with expert partitioning inside a TP chip."""
        model = builder.model_config
        dies = self.hardware_config.chip.logic_die_count
        pu_count = (
            self.hardware_config.chip.logic_die.pu_mesh_rows
            * self.hardware_config.chip.logic_die.pu_mesh_columns
        )
        die_experts = len(expert_ids)
        pu_experts = _exact_div(die_experts, pu_count, "experts per PU")
        local_intermediate = _exact_div(
            model.moe_intermediate_size, chip_tp_degree,
            "hierarchical TP intermediate dimension",
        )
        strategy = (
            f"moe_tp_hierarchical_g{token_groups}_p{chip_tp_degree}"
        )
        suffix = f"tokens{tokens_per_expert}"
        common = dict(
            expert_ids=expert_ids,
            tokens_per_expert=tokens_per_expert,
            parallel_strategy=strategy,
            batch_scaling_unit_count=tokens_per_expert,
        )

        if phase == "up_gate":
            up_die = self._dims(
                die_experts, tokens_per_expert, model.hidden_size,
                2 * local_intermediate,
            )
            builder.add_bmm(
                f"{prefix}.{suffix}.up_gate", "moe_up_gate",
                global_dimensions=self._dims(
                    model.num_experts, tokens_per_expert, model.hidden_size,
                    2 * model.moe_intermediate_size,
                ),
                chip_dimensions=self._dims(
                    model.num_experts, tokens_per_expert, model.hidden_size,
                    2 * local_intermediate,
                ),
                die_dimensions=up_die,
                pu_dimensions_override=self._dims(
                    pu_experts, tokens_per_expert, model.hidden_size,
                    2 * local_intermediate,
                ),
                weight_batches=die_experts, pu_weight_batches=pu_experts,
                input_kind="p2p", input_suffix="input_transfer",
                input_group=("chip0.die0.input", builder.PU),
                input_noc_direction="input", output_kind="p2p",
                output_group=(builder.PU, "chip0.die0.output"),
                output_suffix="output_transfer", **common,
            )
            return
        if phase == "silu":
            builder.add_vector(
                f"{prefix}.{suffix}.silu", "silu",
                global_dimensions={
                    "m": model.num_experts * tokens_per_expert,
                    "n": model.moe_intermediate_size,
                },
                chip_dimensions={
                    "m": model.num_experts * tokens_per_expert,
                    "n": local_intermediate,
                },
                die_dimensions={
                    "m": die_experts * tokens_per_expert,
                    "n": local_intermediate,
                },
                **common,
            )
            return
        if phase != "down":
            raise ValueError(f"unsupported hierarchical MoE phase: {phase}")
        down_die = self._dims(
            die_experts, tokens_per_expert, local_intermediate,
            model.hidden_size,
        )
        builder.add_bmm(
            f"{prefix}.{suffix}.down", "moe_down",
            global_dimensions=self._dims(
                model.num_experts, tokens_per_expert,
                model.moe_intermediate_size, model.hidden_size,
            ),
            chip_dimensions=self._dims(
                model.num_experts, tokens_per_expert,
                local_intermediate, model.hidden_size,
            ),
            die_dimensions=down_die,
            pu_dimensions_override=self._dims(
                pu_experts, tokens_per_expert, local_intermediate,
                model.hidden_size,
            ),
            weight_batches=die_experts, pu_weight_batches=pu_experts,
            input_kind="p2p", input_suffix="input_transfer",
            input_group=("chip0.die0.input", builder.PU),
            input_noc_direction="input", output_kind="p2p",
            output_group=(builder.PU, "chip0.die0.output"),
            output_suffix="output_transfer", **common,
        )

    def _build_hierarchical_moe_tp(
        self, builder: _LayerBuilder, token_groups: int, chip_tp_degree: int,
    ) -> None:
        request = builder.request
        model = builder.model_config
        chips = self.hardware_config.chip_count
        dies = self.hardware_config.chip.logic_die_count
        base_global_batch = request.inference_config.global_batch_size
        token_multiplier = request.inference_config.accepted_tokens_per_step
        global_batch = base_global_batch * token_multiplier
        group_batch = _exact_div(
            global_batch, token_groups, "hierarchical TP token group batch",
        )
        local_batch = _exact_div(
            global_batch, chips, "hierarchical TP chip batch",
        )
        dtype = _ACTIVATION_DTYPE_BYTES
        strategy = (
            f"moe_tp_hierarchical_g{token_groups}_p{chip_tp_degree}"
        )
        chip_group = tuple(range(chip_tp_degree))

        builder.add_comm(
            "moe.tp_input_allgather", "dp_to_tp_allgather",
            scope="inter_chip", kind="allgather", group=chip_group,
            size_bytes=local_batch * model.hidden_size * dtype,
            parallel_strategy=strategy,
        )
        loads = self._balanced_expert_loads(
            group_batch, model.top_k, model.num_experts,
        )
        builder.expert_token_loads = loads
        builder.expert_ownership = tuple(
            tuple(range(model.num_experts)) for _ in range(chips)
        )
        die_expert_count = _exact_div(
            model.num_experts, dies, "hierarchical TP experts across dies",
        )
        representative_die_ids = tuple(range(die_expert_count))
        grouped = self._groups_for_experts(representative_die_ids, loads)
        if len(grouped) != 1:
            raise ValueError("hierarchical TP expects one uniform positive expert load")
        active_ids, token_count = grouped[0]
        up_weight_bytes = (
            len(active_ids) * model.hidden_size
            * (2 * model.moe_intermediate_size // chip_tp_degree)
            * _WEIGHT_DTYPE_BYTES
        )
        sram_capacity = self.hardware_config.chip.logic_die.memory.sram_capacity_bytes
        wave_count = _ceil_div(up_weight_bytes, sram_capacity)
        if len(active_ids) % wave_count:
            raise ValueError("expert waves must evenly partition representative die experts")
        experts_per_wave = len(active_ids) // wave_count
        if experts_per_wave % self.hardware_config.chip.logic_die.processing_unit_count:
            raise ValueError("each expert wave must evenly partition across die PUs")
        waves = tuple(
            active_ids[index * experts_per_wave:(index + 1) * experts_per_wave]
            for index in range(wave_count)
        )
        for phase in ("up_gate", "silu", "down"):
            for wave_index, wave_ids in enumerate(waves):
                prefix = (
                    "moe.tp" if wave_count == 1
                    else f"moe.tp.wave{wave_index}"
                )
                self._add_hierarchical_tp_expert_group(
                    builder, expert_ids=wave_ids,
                    tokens_per_expert=token_count,
                    token_groups=token_groups,
                    chip_tp_degree=chip_tp_degree,
                    phase=phase, prefix=prefix,
                )

        builder.add_comm(
            "moe.tp_down_die_reduce_scatter",
            "moe_down_die_reduce_scatter",
            scope="intra_chip", kind="reduce_scatter",
            group=builder.die_group,
            size_bytes=group_batch * model.hidden_size * dtype,
            parallel_link_count=builder.die_collective_parallel_link_count,
            parallel_strategy=strategy,
        )
        builder.add_comm(
            "moe.tp_chip_reduce_scatter", "tp_to_dp_reduce_scatter",
            scope="inter_chip", kind="reduce_scatter", group=chip_group,
            size_bytes=group_batch * model.hidden_size * dtype,
            parallel_strategy=strategy,
        )
        builder.add_comm(
            "moe.tp_output_die_allgather", "tp_output_die_allgather",
            scope="intra_chip", kind="allgather", group=builder.die_group,
            size_bytes=(
                local_batch
                * _exact_div(model.hidden_size, dies, "TP output hidden shard")
                * dtype
            ),
            parallel_link_count=builder.die_collective_parallel_link_count,
            parallel_strategy=strategy,
        )
        builder.add_vector(
            "moe.tp_residual", "residual",
            global_dimensions={"m": global_batch, "n": model.hidden_size},
            chip_dimensions={"m": local_batch, "n": model.hidden_size},
            die_dimensions={"m": local_batch, "n": model.hidden_size},
            parallel_strategy=strategy,
            flops_per_element=1,
        )

    def _use_pure_moe_mapping(self, builder: _LayerBuilder) -> bool:
        """The new MoE layout is defined for the default 16-chip 4x4 mesh."""
        die = self.hardware_config.chip.logic_die
        return (
            self.hardware_config.chip_count == 16
            and self.hardware_config.chip.logic_die_count == 4
            and die.pu_mesh_rows == 4
            and die.pu_mesh_columns == 4
            and builder.model_config.model_name in {
                "deepseek-v3", "deepseek-v3.2", "kimi-k2.5", "glm-5.1",
            }
        )

    def _add_pure_moe_expert_group(
        self, builder: _LayerBuilder, *, prefix: str,
        expert_ids: Sequence[int], tokens_per_expert: int,
        global_expert_count: int, chip_k_degree: int,
        chip_n_degree: int, die_k_degree: int, pu_k_degree: int,
        strategy: str,
    ) -> None:
        """Each of the chip's 64 PUs holds a shard of every local expert.

        Dies and PUs use fixed K×N factor pairs. Every expert uses all 64
        PUs without partitioning expert IDs.
        """
        model = builder.model_config
        dies = self.hardware_config.chip.logic_die_count
        die_n_degree = _exact_div(dies, die_k_degree, "pure MoE die N degree")
        pu_per_die = self.hardware_config.chip.logic_die.processing_unit_count
        pu_n_degree = _exact_div(
            pu_per_die, pu_k_degree, "pure MoE PU N degree",
        )
        local_experts = len(expert_ids)
        chip_intermediate = _exact_div(
            model.moe_intermediate_size, chip_n_degree,
            "pure MoE chip intermediate shard",
        )
        chip_hidden = _exact_div(
            model.hidden_size, chip_k_degree, "pure MoE chip hidden shard",
        )
        hidden_die = _exact_div(
            chip_hidden, die_k_degree, "pure MoE hidden die shard",
        )
        die_intermediate = _exact_div(
            chip_intermediate, die_n_degree,
            "pure MoE intermediate die shard",
        )
        hidden_pu = _exact_div(
            hidden_die, pu_k_degree, "pure MoE PU hidden shard",
        )
        intermediate_pu = _exact_div(
            die_intermediate, pu_n_degree,
            "pure MoE PU intermediate shard",
        )
        up_output_group = tuple(
            f"chip0.die0.pu{index}" for index in range(pu_k_degree)
        )
        down_output_group = tuple(
            f"chip0.die0.pu{index}" for index in range(pu_n_degree)
        )
        suffix = f"tokens{tokens_per_expert}"
        common = dict(
            expert_ids=expert_ids,
            tokens_per_expert=tokens_per_expert,
            parallel_strategy=strategy,
            batch_scaling_unit_count=tokens_per_expert,
        )
        builder.add_bmm(
            f"{prefix}.{suffix}.up_gate", "moe_up_gate",
            global_dimensions=self._dims(
                global_expert_count, tokens_per_expert,
                model.hidden_size, 2 * model.moe_intermediate_size,
            ),
            chip_dimensions=self._dims(
                local_experts, tokens_per_expert, chip_hidden,
                2 * chip_intermediate,
            ),
            die_dimensions=self._dims(
                local_experts, tokens_per_expert, hidden_die,
                2 * die_intermediate,
            ),
            pu_dimensions_override=self._dims(
                local_experts, tokens_per_expert, hidden_pu,
                2 * intermediate_pu,
            ),
            weight_batches=local_experts, pu_weight_batches=local_experts,
            input_kind="p2p" if pu_k_degree > 1 else "broadcast",
            input_suffix=(
                "input_transfer" if pu_k_degree > 1 else "input_broadcast"
            ),
            input_group=(
                ("chip0.die0.input", builder.PU)
                if pu_k_degree > 1 else down_output_group
            ),
            input_noc_direction="input",
            input_bytes_override=(
                local_experts * tokens_per_expert
                * hidden_die * _ACTIVATION_DTYPE_BYTES
            ),
            output_kind="reduce" if pu_k_degree > 1 else "p2p",
            output_group=(
                up_output_group if pu_k_degree > 1
                else (builder.PU, "chip0.die0.output")
            ),
            output_suffix=(
                "output_reduce" if pu_k_degree > 1 else "output_transfer"
            ),
            # The NoC output resource carries all 16 channel shards, even
            # though the representative GEMM describes one PU shard.
            output_bytes_override=(
                local_experts * tokens_per_expert
                * 2 * die_intermediate * _ACTIVATION_DTYPE_BYTES
            ),
            **common,
        )
        up_bytes = (
            local_experts * tokens_per_expert
            * 2 * die_intermediate * _ACTIVATION_DTYPE_BYTES
        )
        if die_k_degree > 1:
            die_k_group = tuple(
                f"chip0.die{index * die_n_degree}"
                for index in range(die_k_degree)
            )
            builder.add_comm(
                f"{prefix}.{suffix}.up_die_reduce_scatter",
                "moe_up_gate_die_reduce_scatter",
                scope="intra_chip", kind="reduce_scatter",
                group=die_k_group, size_bytes=up_bytes,
                parallel_link_count=builder.die_collective_parallel_link_count,
                parallel_strategy=strategy,
                batch_scaling_unit_count=tokens_per_expert,
            )
            builder.add_comm(
                f"{prefix}.{suffix}.up_die_allgather",
                "moe_up_gate_die_allgather",
                scope="intra_chip", kind="allgather",
                group=die_k_group,
                size_bytes=_exact_div(
                    up_bytes, die_k_degree, "pure MoE Up/Gate scatter",
                ),
                parallel_link_count=builder.die_collective_parallel_link_count,
                parallel_strategy=strategy,
                batch_scaling_unit_count=tokens_per_expert,
            )
        if chip_k_degree > 1:
            builder.add_comm(
                f"{prefix}.{suffix}.up_chip_allreduce",
                "moe_up_gate_chip_allreduce", scope="inter_chip",
                kind="allreduce",
                group=tuple(range(0, self.hardware_config.chip_count,
                                  chip_n_degree)),
                size_bytes=up_bytes,
                parallel_strategy=strategy,
                batch_scaling_unit_count=tokens_per_expert,
            )
        builder.add_vector(
            f"{prefix}.{suffix}.silu", "silu",
            global_dimensions={
                "m": global_expert_count * tokens_per_expert,
                "n": model.moe_intermediate_size,
            },
            chip_dimensions={
                "m": local_experts * tokens_per_expert,
                "n": chip_intermediate,
            },
            die_dimensions={
                "m": local_experts * tokens_per_expert,
                "n": die_intermediate,
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
                local_experts, tokens_per_expert,
                chip_intermediate, chip_hidden,
            ),
            die_dimensions=self._dims(
                local_experts, tokens_per_expert,
                die_intermediate, hidden_die,
            ),
            pu_dimensions_override=self._dims(
                local_experts, tokens_per_expert,
                intermediate_pu, hidden_pu,
            ),
            weight_batches=local_experts, pu_weight_batches=local_experts,
            input_kind="broadcast" if pu_k_degree > 1 else "p2p",
            input_suffix=(
                "input_broadcast" if pu_k_degree > 1 else "input_transfer"
            ),
            input_group=(
                up_output_group if pu_k_degree > 1
                else ("chip0.die0.input", builder.PU)
            ),
            input_noc_direction="input",
            input_bytes_override=(
                local_experts * tokens_per_expert
                * die_intermediate * _ACTIVATION_DTYPE_BYTES
            ),
            output_kind="reduce", output_group=down_output_group,
            output_bytes_override=(
                local_experts * tokens_per_expert
                * hidden_die * _ACTIVATION_DTYPE_BYTES
            ),
            **common,
        )
        if die_n_degree > 1:
            builder.add_comm(
                f"{prefix}.{suffix}.down_die_reduce_scatter",
                "moe_down_die_reduce_scatter",
                scope="intra_chip", kind="reduce_scatter",
                group=tuple(
                    f"chip0.die{index}" for index in range(die_n_degree)
                ),
                size_bytes=(
                    local_experts * tokens_per_expert
                    * hidden_die * _ACTIVATION_DTYPE_BYTES
                ),
                parallel_link_count=builder.die_collective_parallel_link_count,
                parallel_strategy=strategy,
                batch_scaling_unit_count=tokens_per_expert,
            )

    def _add_chip_mesh_moe_expert_group(
        self, builder: _LayerBuilder, *, expert_ids: Sequence[int],
        tokens_per_expert: int, global_expert_count: int,
        chip_tp_degree: int, prefix: str, strategy: str,
    ) -> None:
        """Model a TP or EP expert group on one unified 8×8 chip PU mesh.

        Up's K-partial reduction is intentionally omitted in this idealized
        experiment. The Up output command charges half the fused channel
        width as requested, before the following chip-level SiLU command.
        """
        model = builder.model_config
        chip = self.hardware_config.chip
        experts = len(expert_ids)
        m = tokens_per_expert
        hidden = model.hidden_size
        intermediate = _exact_div(
            model.moe_intermediate_size, chip_tp_degree,
            "MoE chip intermediate",
        )
        rows, columns = chip.noc.mesh_rows, chip.noc.mesh_columns
        hidden_pu = _exact_div(hidden, rows, "MoE PU hidden")
        intermediate_pu = _exact_div(intermediate, columns, "MoE PU intermediate")
        chip_links = 4 * rows  # Four perimeter edges, eight ideal lanes each.
        suffix = f"tokens{m}"
        prefix = f"{prefix}.{suffix}"
        common = dict(
            expert_ids=expert_ids, tokens_per_expert=m,
            parallel_strategy=strategy, batch_scaling_unit_count=m,
        )
        builder.add_bmm(
            f"{prefix}.up_gate", "moe_up_gate",
            global_dimensions=self._dims(
                global_expert_count, m, hidden, 2 * model.moe_intermediate_size,
            ),
            chip_dimensions=self._dims(experts, m, hidden, 2 * intermediate),
            # This MoE path treats the entire chip as the memory/vector domain.
            die_dimensions=self._dims(experts, m, hidden, 2 * intermediate),
            pu_dimensions_override=self._dims(
                experts, m, hidden_pu, 2 * intermediate_pu,
            ),
            weight_batches=experts, pu_weight_batches=experts,
            load_bytes_override=experts * hidden * 2 * intermediate,
            dram_resource_id=builder.CHIP_DRAM,
            sram_resource_id=builder.CHIP_SRAM,
            input_kind="p2p", input_suffix="input_transfer",
            input_group=("chip0.input", builder.PU),
            input_noc_direction="input",
            input_bytes_override=experts * m * hidden * _ACTIVATION_DTYPE_BYTES,
            input_parallel_link_count=chip_links,
            output_kind="p2p", output_suffix="output_transfer",
            output_group=(builder.PU, "chip0.vector"),
            output_bytes_override=(
                experts * m * intermediate_pu * _ACTIVATION_DTYPE_BYTES
            ),
            **common,
        )
        builder.add_vector(
            f"{prefix}.silu", "silu",
            global_dimensions={"m": global_expert_count * m,
                               "n": model.moe_intermediate_size},
            chip_dimensions={"m": experts * m, "n": intermediate},
            die_dimensions={"m": experts * m, "n": intermediate},
            resource_id=builder.CHIP_VECTOR,
            **common,
        )
        builder.add_bmm(
            f"{prefix}.down", "moe_down",
            global_dimensions=self._dims(
                global_expert_count, m, model.moe_intermediate_size, hidden,
            ),
            chip_dimensions=self._dims(experts, m, intermediate, hidden),
            die_dimensions=self._dims(experts, m, intermediate, hidden),
            pu_dimensions_override=self._dims(
                experts, m, intermediate_pu, hidden_pu,
            ),
            weight_batches=experts, pu_weight_batches=experts,
            load_bytes_override=experts * intermediate * hidden,
            dram_resource_id=builder.CHIP_DRAM,
            sram_resource_id=builder.CHIP_SRAM,
            input_kind="p2p", input_suffix="input_transfer",
            input_group=("chip0.vector", builder.PU),
            input_noc_direction="input",
            input_bytes_override=(
                experts * m * intermediate_pu * _ACTIVATION_DTYPE_BYTES
            ),
            output_kind="reduce", output_group=(builder.PU,) + tuple(
                f"chip0.pu{index}" for index in range(1, rows)
            ),
            output_bytes_override=experts * m * hidden * _ACTIVATION_DTYPE_BYTES,
            output_parallel_link_count=chip_links,
            **common,
        )

    def _build_pure_moe_tp(self, builder: _LayerBuilder) -> None:
        model = builder.model_config
        inference = builder.request.inference_config
        chips = self.hardware_config.chip_count
        global_batch = (
            inference.global_batch_size * inference.accepted_tokens_per_step
        )
        local_batch = _ceil_div(inference.global_batch_size, chips)
        local_batch *= inference.accepted_tokens_per_step
        dtype = _ACTIVATION_DTYPE_BYTES
        chip_k_degree = 1
        chip_n_degree = chips // chip_k_degree
        strategy = "moe_tp_chip_mesh_k8_n8_ideal_noc32"
        entry_shard_bytes = (
            local_batch * model.hidden_size // chip_k_degree * dtype
        )
        builder.add_comm(
            "moe.tp_input_alltoall", "dp_to_tp_alltoall",
            scope="inter_chip", kind="alltoall", group=builder.chip_group,
            transfer_bytes=tuple(
                tuple(
                    0 if source == destination else entry_shard_bytes
                    for destination in range(chips)
                )
                for source in range(chips)
            ),
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
            self._add_chip_mesh_moe_expert_group(
                builder, expert_ids=expert_ids,
                tokens_per_expert=token_count,
                global_expert_count=len(expert_ids),
                chip_tp_degree=chips, prefix="moe.tp", strategy=strategy,
            )
        if chip_n_degree > 1:
            builder.add_comm(
                "moe.tp_chip_reduce_scatter", "tp_to_dp_reduce_scatter",
                scope="inter_chip", kind="reduce_scatter",
                group=tuple(range(chip_n_degree)),
                size_bytes=(
                    global_batch * model.hidden_size // chip_k_degree * dtype
                ),
                parallel_strategy=strategy,
            )
        # Each N-group now owns BS/N tokens and one K-shard of hidden. The
        # other K groups exchange their complementary hidden slices while
        # returning to the original BS/16-token DP layout.
        if chip_k_degree > 1:
            exchange_bytes = (
                local_batch * model.hidden_size // chip_k_degree * dtype
            )
            builder.add_comm(
                "moe.tp_chip_hidden_exchange", "tp_hidden_exchange",
                scope="inter_chip", kind="alltoall",
                group=tuple(range(0, chips, chip_n_degree)),
                transfer_bytes=tuple(
                    tuple(
                        0 if source == destination else exchange_bytes
                        for destination in range(chip_k_degree)
                    )
                    for source in range(chip_k_degree)
                ),
                parallel_strategy=strategy,
            )
            builder.add_comm(
                "moe.tp_hidden_exchange_die_distribute",
                "tp_hidden_exchange_die_distribute",
                scope="intra_chip", kind="p2p",
                group=("chip0.input", "chip0.die0.input"),
                size_bytes=(chip_k_degree - 1) * exchange_bytes,
                noc_direction="input", parallel_strategy=strategy,
            )
        builder.add_vector(
            "moe.tp_residual", "residual",
            global_dimensions={"m": global_batch, "n": model.hidden_size},
            chip_dimensions={"m": local_batch, "n": model.hidden_size},
            die_dimensions={"m": local_batch, "n": model.hidden_size},
            parallel_strategy=strategy, flops_per_element=1,
            resource_id=builder.CHIP_VECTOR,
        )

    def _build_moe_tp(self, builder: _LayerBuilder) -> None:
        request = builder.request
        model = builder.model_config
        chips = self.hardware_config.chip_count
        dies = self.hardware_config.chip.logic_die_count
        if self._use_pure_moe_mapping(builder):
            self._build_pure_moe_tp(builder)
            return
        base_global_batch = request.inference_config.global_batch_size
        token_multiplier = request.inference_config.accepted_tokens_per_step
        token_groups, chip_tp_degree = self._hierarchical_moe_tp_degrees(
            model, base_global_batch,
        )
        if token_groups > 1:
            self._build_hierarchical_moe_tp(
                builder, token_groups, chip_tp_degree,
            )
            return
        # MTP1 sends both accepted query tokens through MoE, so expert loads,
        # activation collectives and arithmetic all see twice the base batch.
        global_batch = base_global_batch * token_multiplier
        local_batch = _ceil_div(base_global_batch, chips) * token_multiplier
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
            "moe.tp_chip_reduce_scatter", "tp_to_dp_reduce_scatter",
            scope="inter_chip", kind="reduce_scatter",
            group=builder.chip_group,
            size_bytes=(
                global_batch * model.hidden_size * dtype
            ),
            parallel_strategy=strategy,
        )
        builder.add_comm(
            "moe.tp_output_die_allgather", "tp_output_die_allgather",
            scope="intra_chip", kind="allgather", group=builder.die_group,
            size_bytes=(
                local_batch
                * _exact_div(
                    model.hidden_size, dies,
                    "TP output hidden dimension across dies",
                )
                * dtype
            ),
            parallel_link_count=builder.die_collective_parallel_link_count,
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

    def _build_pure_moe_ep(self, builder: _LayerBuilder) -> None:
        model = builder.model_config
        inference = builder.request.inference_config
        chips = self.hardware_config.chip_count
        global_batch = (
            inference.global_batch_size * inference.accepted_tokens_per_step
        )
        local_batch = _ceil_div(inference.global_batch_size, chips)
        local_batch *= inference.accepted_tokens_per_step
        dtype = _ACTIVATION_DTYPE_BYTES
        strategy = "moe_ep_chip_mesh_k8_n8_ideal_noc32"
        experts_per_chip = _exact_div(
            model.num_experts, chips, "pure EP experts per chip",
        )
        ownership = tuple(
            tuple(range(chip * experts_per_chip, (chip + 1) * experts_per_chip))
            for chip in range(chips)
        )
        loads = self._balanced_ep_expert_loads(
            global_batch, model.top_k, model.num_experts, ownership,
        )
        builder.expert_token_loads = loads
        builder.expert_ownership = ownership
        effective_assignments = sum(loads)
        row_totals = (effective_assignments // chips,) * chips
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
        for expert_ids, token_count in self._groups_for_experts(
            ownership[0], loads,
        ):
            self._add_chip_mesh_moe_expert_group(
                builder, expert_ids=expert_ids,
                tokens_per_expert=token_count,
                global_expert_count=sum(
                    load == token_count for load in loads
                ),
                chip_tp_degree=1, prefix="moe.ep", strategy=strategy,
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
            global_dimensions={"m": global_batch, "n": model.hidden_size},
            chip_dimensions={"m": local_batch, "n": model.hidden_size},
            die_dimensions={"m": local_batch, "n": model.hidden_size},
            parallel_strategy=strategy, flops_per_element=1,
            resource_id=builder.CHIP_VECTOR,
        )

    def _build_moe_ep(self, builder: _LayerBuilder) -> None:
        request = builder.request
        model = builder.model_config
        chips = self.hardware_config.chip_count
        dies = self.hardware_config.chip.logic_die_count
        if self._use_pure_moe_mapping(builder):
            self._build_pure_moe_ep(builder)
            return
        base_global_batch = request.inference_config.global_batch_size
        token_multiplier = request.inference_config.accepted_tokens_per_step
        global_batch = base_global_batch * token_multiplier
        local_batch = _ceil_div(base_global_batch, chips) * token_multiplier
        dtype = _ACTIVATION_DTYPE_BYTES
        use_hierarchical_ep = (
            model.model_name in {
                "deepseek-v3", "deepseek-v3.2", "glm-5.1",
            }
            and chips == 16
            and dies == 4
            and self.hardware_config.chip.logic_die.pu_mesh_rows == 4
            and self.hardware_config.chip.logic_die.pu_mesh_columns == 4
        )
        strategy = (
            "moe_ep_hierarchical_die_expert"
            if use_hierarchical_ep else "moe_ep_die_tp4"
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
        loads = self._balanced_ep_expert_loads(
            global_batch, model.top_k, model.num_experts, ownership,
        )
        # Sparse EP discards assignments that cannot form a complete equal
        # round across all chips. Use the resulting routed total consistently
        # for both dispatch rows and expert-owner columns.
        effective_assignments = sum(loads)
        assignments_per_chip = effective_assignments // chips
        row_totals = (assignments_per_chip,) * chips
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
        if use_hierarchical_ep:
            if len(representative_experts) != 16:
                raise ValueError(
                    "hierarchical EP requires exactly 16 experts per chip"
                )
            # A representative die owns one contiguous quarter of the chip's
            # experts. Its four PU rows each execute one expert, while the
            # four columns shard the intermediate dimension.
            representative_experts = representative_experts[:4]
        for expert_ids, token_count in self._groups_for_experts(
            representative_experts, loads,
        ):
            if use_hierarchical_ep:
                self._add_hierarchical_ep_expert_group(
                    builder, expert_ids=expert_ids,
                    tokens_per_expert=token_count,
                    global_expert_count=sum(
                        load == token_count for load in loads
                    ),
                    strategy=strategy,
                )
            else:
                self._add_expert_group(
                    builder, prefix="moe.ep", expert_ids=expert_ids,
                    tokens_per_expert=token_count,
                    global_expert_count=sum(
                        load == token_count for load in loads
                    ),
                    tp_degree=dies,
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
        if use_hierarchical_ep:
            builder.add_comm(
                "moe.ep_output_die_reduce_scatter",
                "ep_output_die_reduce_scatter",
                scope="intra_chip", kind="reduce_scatter",
                group=builder.die_group,
                size_bytes=local_batch * model.hidden_size * dtype,
                parallel_link_count=(
                    builder.die_collective_parallel_link_count
                ),
                parallel_strategy=strategy,
            )
        builder.add_comm(
            "moe.ep_output_die_allgather", "ep_output_die_allgather",
            scope="intra_chip", kind="allgather", group=builder.die_group,
            size_bytes=(
                local_batch
                * _exact_div(
                    model.hidden_size, dies, "EP output hidden dimension",
                )
                * dtype
            ),
            parallel_link_count=builder.die_collective_parallel_link_count,
            parallel_strategy=strategy,
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

    def _add_hierarchical_ep_expert_group(
        self, builder: _LayerBuilder, *, expert_ids: Sequence[int],
        tokens_per_expert: int, global_expert_count: int, strategy: str,
    ) -> None:
        """Map four die-local experts onto four PU rows for large-BS EP."""
        model = builder.model_config
        rows = self.hardware_config.chip.logic_die.pu_mesh_rows
        columns = self.hardware_config.chip.logic_die.pu_mesh_columns
        if rows != 4 or columns != 4 or len(expert_ids) != rows:
            raise ValueError(
                "hierarchical EP requires a 4x4 PU mesh and four experts/die"
            )
        local_intermediate = _exact_div(
            model.moe_intermediate_size, columns,
            "hierarchical EP intermediate dimension",
        )
        suffix = f"tokens{tokens_per_expert}"
        common = dict(
            expert_ids=expert_ids,
            tokens_per_expert=tokens_per_expert,
            parallel_strategy=strategy,
            batch_scaling_unit_count=tokens_per_expert,
        )

        builder.add_bmm(
            f"moe.ep.{suffix}.up_gate", "moe_up_gate",
            global_dimensions=self._dims(
                global_expert_count, tokens_per_expert, model.hidden_size,
                2 * model.moe_intermediate_size,
            ),
            chip_dimensions=self._dims(
                16, tokens_per_expert, model.hidden_size,
                2 * model.moe_intermediate_size,
            ),
            die_dimensions=self._dims(
                4, tokens_per_expert, model.hidden_size,
                2 * model.moe_intermediate_size,
            ),
            pu_dimensions_override=self._dims(
                1, tokens_per_expert, model.hidden_size,
                2 * local_intermediate,
            ),
            weight_batches=4, pu_weight_batches=1,
            input_group=builder.representative_pu_row,
            output_kind="p2p",
            output_group=(builder.PU, "chip0.die0.output"),
            output_suffix="output_transfer",
            **common,
        )
        builder.add_vector(
            f"moe.ep.{suffix}.silu", "silu",
            global_dimensions={
                "m": global_expert_count * tokens_per_expert,
                "n": model.moe_intermediate_size,
            },
            chip_dimensions={
                "m": 16 * tokens_per_expert,
                "n": model.moe_intermediate_size,
            },
            die_dimensions={
                "m": 4 * tokens_per_expert,
                "n": model.moe_intermediate_size,
            },
            **common,
        )
        builder.add_bmm(
            f"moe.ep.{suffix}.down", "moe_down",
            global_dimensions=self._dims(
                global_expert_count, tokens_per_expert,
                model.moe_intermediate_size, model.hidden_size,
            ),
            chip_dimensions=self._dims(
                16, tokens_per_expert,
                model.moe_intermediate_size, model.hidden_size,
            ),
            die_dimensions=self._dims(
                4, tokens_per_expert,
                model.moe_intermediate_size, model.hidden_size,
            ),
            pu_dimensions_override=self._dims(
                1, tokens_per_expert, local_intermediate,
                model.hidden_size,
            ),
            weight_batches=4, pu_weight_batches=1,
            input_kind="p2p", input_suffix="input_transfer",
            input_group=("chip0.die0.input", builder.PU),
            input_noc_direction="input",
            output_kind="reduce",
            output_group=builder.representative_pu_row,
            output_suffix="output_reduce",
            **common,
        )


__all__ = [
    "MoEParallelStrategy",
    "HardwareMappingRequest",
    "HardwareMappingResult",
    "OperatorHardwareMapping",
    "HardwareMapper",
]
