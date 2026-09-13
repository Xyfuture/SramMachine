"""Hardware mapping for one representative MLA/MoE decode layer."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

from srammachine.frontend.modules import (
    BMMOp, CommOp, FlashAttentionOp, Operator, VectorOp,
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
_ACTIVATION_DTYPE_BYTES = 2


def _positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _shared_effective_bytes(logical_bytes: int, logic_die_count: int) -> int:
    """Charge one representative die its ideal share of chip-shared data.

    The simulator intentionally keeps die-granularity commands.  For data that
    is bit-identical on every die (compressed KV and indexer keys), the four
    physical dies are nevertheless treated as one logical chip with perfect
    striping/multicast.  No extra NoC transfer is charged for this idealization.
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
        batch_scaling_unit_count: Optional[int] = None,
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
        output_bytes = (
            pu_dimensions["B"] * pu_dimensions["M"] * pu_dimensions["N"]
            * _ACTIVATION_DTYPE_BYTES
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
            batch_partition_degree=batch_partition_degree,
            noc_direction=input_noc_direction,
            batch_scaling_unit_count=batch_scaling_unit_count,
        )
        operator = BMMOp(op_id, **pu_dimensions)
        mapping = OperatorMapping(
            batch_axis, self.PU,
            self.DRAM if (
                weight_bytes or dram_read_bytes_per_token
                or dram_write_bytes_per_token
            ) else None,
            self.SRAM if (
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
            batch_partition_degree=batch_partition_degree,
            batch_scaling_unit_count=batch_scaling_unit_count,
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
                "m", self.VECTOR,
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
            group=self.representative_pu_row, root=self.PU,
            size_bytes=input_bytes, parallel_strategy=parallel_strategy,
        )
        operator = FlashAttentionOp(op_id, **pu_dimensions)
        mapping = OperatorMapping(
            batch_axis, self.PU, self.DRAM, self.SRAM,
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
            group=self.representative_pu_column, root=self.PU,
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
        local_heads = model.num_attention_heads // dies
        pu_rows = self.hardware_config.chip.logic_die.pu_mesh_rows
        pu_columns = self.hardware_config.chip.logic_die.pu_mesh_columns
        latent_width = (
            model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim
        )
        local_latent = latent_width // dies
        strategy = "attention_chip_dp_die_head_tp_pu_tp"

        builder.add_vector(
            "attn.input_norm", "rmsnorm",
            global_dimensions={"m": global_batch, "n": model.hidden_size},
            chip_dimensions={"m": chip_batch, "n": model.hidden_size},
            die_dimensions={"m": chip_batch, "n": model.hidden_size},
            parallel_strategy=strategy,
        )
        builder.add_bmm(
            "attn.latent_down", "mla_latent_down_projection",
            global_dimensions=self._dims(
                1, global_batch, model.hidden_size, latent_width,
            ),
            chip_dimensions=self._dims(
                1, chip_batch, model.hidden_size, latent_width,
            ),
            die_dimensions=self._dims(
                1, chip_batch, model.hidden_size, local_latent,
            ),
            parallel_strategy=strategy,
            weight_batches=1,
        )
        builder.add_comm(
            "attn.latent_allgather", "latent_allgather",
            scope="intra_chip", kind="allgather", group=builder.die_group,
            size_bytes=chip_batch * local_latent * _ACTIVATION_DTYPE_BYTES,
            parallel_link_count=builder.die_collective_parallel_link_count,
            parallel_strategy=strategy,
            dram_write_bytes_per_token=(
                _shared_effective_bytes(
                    (model.kv_lora_rank + model.qk_rope_head_dim)
                    * kv_dtype_bytes,
                    dies,
                )
            ),
            dram_write_logical_bytes_per_token=(
                model.kv_lora_rank + model.qk_rope_head_dim
            ) * kv_dtype_bytes,
            shared_die_factor=dies,
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
        )
        builder.add_bmm(
            "attn.q_rope_projection", "q_rope_projection",
            global_dimensions=self._dims(
                1, global_batch, model.q_lora_rank,
                model.num_attention_heads * model.qk_rope_head_dim,
            ),
            chip_dimensions=self._dims(
                1, chip_batch, model.q_lora_rank,
                model.num_attention_heads * model.qk_rope_head_dim,
            ),
            die_dimensions=self._dims(
                1, chip_batch, model.q_lora_rank,
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
                "m": chip_batch * model.num_attention_heads,
                "n": model.qk_rope_head_dim,
            },
            die_dimensions={
                "m": chip_batch * local_heads,
                "n": model.qk_rope_head_dim,
            },
            parallel_strategy=strategy,
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
        die_head_batch = chip_batch * local_heads
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
            batch_axis="B",
        )
        # FlashAttention consumes the concatenated NoPE/RoPE Q and streams
        # the corresponding KV tile once.  Scores are never materialized on
        # the NoC; QK and SV are charged as one PU command.
        fused_qk_width = model.kv_lora_rank + model.qk_rope_head_dim
        flash_dimensions = lambda head_batch: {
            "B": head_batch,
            "M": 1,
            "qk_K": fused_qk_width,
            "qk_N": attention_history,
            "sv_K": attention_history,
            "sv_N": model.kv_lora_rank,
        }
        # The compressed KV cache is logically shared by all head-TP dies.
        # Charge only one ideal striped chip copy: both DRAM->SRAM and
        # SRAM->SA are streaming traffic, not a full resident-SRAM allocation.
        builder.add_flash_attention(
            "attn.flash_attention", "flash_attention",
            global_dimensions=flash_dimensions(global_head_batch),
            chip_dimensions=flash_dimensions(chip_head_batch),
            die_dimensions=flash_dimensions(die_head_batch),
            parallel_strategy=strategy,
            dram_read_bytes_per_token=(
                _shared_effective_bytes(
                    attention_history * fused_qk_width
                    * kv_dtype_bytes,
                    dies,
                )
            ),
            dram_read_logical_bytes_per_token=(
                attention_history * fused_qk_width * kv_dtype_bytes
            ),
            sram_read_bytes_per_mapped_token=(
                _shared_effective_bytes(
                    (
                        _ceil_div(model.kv_lora_rank, pu_rows)
                        + _ceil_div(model.qk_rope_head_dim, pu_rows)
                    )
                    * _ceil_div(attention_history, pu_columns)
                    * kv_dtype_bytes,
                    dies,
                )
            ),
            sram_read_logical_bytes_per_mapped_token=(
                (
                    _ceil_div(model.kv_lora_rank, pu_rows)
                    + _ceil_div(model.qk_rope_head_dim, pu_rows)
                )
                * _ceil_div(attention_history, pu_columns)
                * kv_dtype_bytes
            ),
            shared_die_factor=dies,
        )

        # O projection contracts both the local-head and latent dimensions.
        # Folding heads into K is algebraically identical to the batched form,
        # keeps the PU FLOPs and weight tile unchanged, and reduces the NoC
        # output from per-head partials to one hidden tile per request.
        builder.add_bmm(
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
            die_dimensions=self._dims(
                1, chip_batch, local_heads * model.kv_lora_rank,
                model.hidden_size,
            ),
            parallel_strategy=strategy,
            weight_batches=1,
            batch_axis="M",
        )
        builder.add_comm(
            "attn.die_output_reduce", "die_output_reduce",
            scope="intra_chip", kind="allreduce", group=builder.die_group,
            size_bytes=chip_batch * model.hidden_size * _ACTIVATION_DTYPE_BYTES,
            parallel_link_count=builder.die_collective_parallel_link_count,
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
                    "m": chip_batch, "n": model.hidden_size,
                },
                die_dimensions={
                    "m": chip_batch, "n": model.hidden_size,
                },
                parallel_strategy=strategy,
                flops_per_element=coefficient,
            )

    def _build_dsa(
        self, builder: _LayerBuilder, chip_batch: int, strategy: str,
    ) -> None:
        model = builder.model_config
        if model.indexer_num_heads is None or model.indexer_head_dim is None:
            raise ValueError("DSA model requires complete indexer dimensions")
        request = builder.request
        kv_dtype_bytes = request.inference_config.kv_cache_bytes_per_element
        dies = self.hardware_config.chip.logic_die_count
        pu_rows = self.hardware_config.chip.logic_die.pu_mesh_rows
        pu_columns = self.hardware_config.chip.logic_die.pu_mesh_columns
        global_batch = (
            request.inference_config.global_batch_size
            * request.inference_config.accepted_tokens_per_step
        )
        history = request.inference_config.input_sequence_length
        local_heads = model.indexer_num_heads // dies
        chip_width = (
            model.indexer_num_heads * model.indexer_head_dim
            + model.indexer_head_dim + model.indexer_num_heads
        )
        die_width = (
            local_heads * model.indexer_head_dim
            + model.indexer_head_dim + local_heads
        )
        # Query projections and per-head indexer weights are distinct die
        # shards.  Only the shared 128-wide key projection is normalized.
        effective_shared_key_width = _ceil_div(
            model.indexer_head_dim, dies,
        )
        effective_die_width = (
            local_heads * model.indexer_head_dim
            + effective_shared_key_width + local_heads
        )
        logical_weight_bytes = (
            model.hidden_size * die_width * _WEIGHT_DTYPE_BYTES
        )
        effective_weight_bytes = (
            model.hidden_size * effective_die_width * _WEIGHT_DTYPE_BYTES
        )
        logical_load_bytes = (
            _ceil_div(model.hidden_size, pu_rows)
            * _ceil_div(die_width, pu_columns)
            * _WEIGHT_DTYPE_BYTES
        )
        effective_load_bytes = (
            _ceil_div(model.hidden_size, pu_rows)
            * _ceil_div(effective_die_width, pu_columns)
            * _WEIGHT_DTYPE_BYTES
        )
        builder.add_bmm(
            "attn.dsa_qkw", "indexer_qkw_projection",
            global_dimensions=self._dims(
                1, global_batch, model.hidden_size, chip_width,
            ),
            chip_dimensions=self._dims(
                1, chip_batch, model.hidden_size, chip_width,
            ),
            die_dimensions=self._dims(
                1, chip_batch, model.hidden_size, die_width,
            ),
            parallel_strategy=strategy,
            weight_batches=1,
            weight_bytes_override=effective_weight_bytes,
            weight_logical_bytes=logical_weight_bytes,
            load_bytes_override=effective_load_bytes,
            load_logical_bytes=logical_load_bytes,
            shared_die_factor=dies,
            dram_write_bytes_per_token=(
                _shared_effective_bytes(
                    model.indexer_head_dim * kv_dtype_bytes,
                    dies,
                )
            ),
            dram_write_logical_bytes_per_token=(
                model.indexer_head_dim * kv_dtype_bytes
            ),
        )
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
                chip_batch, local_heads,
                model.indexer_head_dim, history,
            ),
            parallel_strategy=strategy,
            # The indexer key cache is the same on every die.  As for MLA KV,
            # model perfect chip-level sharing and stream one fourth of both
            # DRAM and SRAM traffic without adding a synthetic NoC transfer.
            dram_read_bytes_per_token=(
                _shared_effective_bytes(
                    history * model.indexer_head_dim * kv_dtype_bytes,
                    dies,
                )
            ),
            dram_read_logical_bytes_per_token=(
                history * model.indexer_head_dim * kv_dtype_bytes
            ),
            batch_axis="B",
            sram_read_bytes_per_mapped_token=(
                _shared_effective_bytes(
                    _ceil_div(model.indexer_head_dim, pu_rows)
                    * _ceil_div(history, pu_columns)
                    * kv_dtype_bytes,
                    dies,
                )
            ),
            sram_read_logical_bytes_per_mapped_token=(
                _ceil_div(model.indexer_head_dim, pu_rows)
                * _ceil_div(history, pu_columns)
                * kv_dtype_bytes
            ),
            sram_read_data_kind="dsa_key",
            shared_die_factor=dies,
        )
        builder.add_vector(
            "attn.dsa_relu", "relu",
            global_dimensions={
                "m": global_batch * model.indexer_num_heads, "n": history,
            },
            chip_dimensions={
                "m": chip_batch * model.indexer_num_heads, "n": history,
            },
            die_dimensions={
                "m": chip_batch * local_heads, "n": history,
            },
            parallel_strategy=strategy,
            flops_per_element=1,
        )
        builder.add_vector(
            "attn.dsa_head_reduce", "indexer_weighted_head_reduce",
            global_dimensions={"m": global_batch, "n": history},
            chip_dimensions={"m": chip_batch, "n": history},
            die_dimensions={"m": chip_batch, "n": history},
            parallel_strategy=strategy,
            flops_per_element=max(2 * local_heads - 1, 1),
        )
        builder.add_comm(
            "attn.dsa_score_reduce", "indexer_score_reduce",
            scope="intra_chip", kind="allreduce", group=builder.die_group,
            size_bytes=chip_batch * history * _ACTIVATION_DTYPE_BYTES,
            parallel_link_count=builder.die_collective_parallel_link_count,
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

    def _build_moe_tp(self, builder: _LayerBuilder) -> None:
        request = builder.request
        model = builder.model_config
        chips = self.hardware_config.chip_count
        dies = self.hardware_config.chip.logic_die_count
        base_global_batch = request.inference_config.global_batch_size
        token_multiplier = request.inference_config.accepted_tokens_per_step
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

    def _build_moe_ep(self, builder: _LayerBuilder) -> None:
        request = builder.request
        model = builder.model_config
        chips = self.hardware_config.chip_count
        dies = self.hardware_config.chip.logic_die_count
        base_global_batch = request.inference_config.global_batch_size
        token_multiplier = request.inference_config.accepted_tokens_per_step
        global_batch = base_global_batch * token_multiplier
        local_batch = _ceil_div(base_global_batch, chips) * token_multiplier
        dtype = _ACTIVATION_DTYPE_BYTES
        strategy = "moe_ep_die_tp4"
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
        combine = tuple(
            tuple(dispatch[column][row] for column in range(chips))
            for row in range(chips)
        )
        builder.add_comm(
            "moe.ep_combine", "expert_combine", scope="inter_chip",
            kind="alltoall", group=builder.chip_group,
            transfer_bytes=combine, parallel_strategy=strategy,
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


__all__ = [
    "MoEParallelStrategy",
    "HardwareMappingRequest",
    "HardwareMappingResult",
    "OperatorHardwareMapping",
    "HardwareMapper",
]
