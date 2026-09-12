"""High-level, trace-ready statistics for one completed simulation."""

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from srammachine.commands import (
    Command,
    CommandGraph,
    CommCmd,
    DramCmd,
    FlashAttentionCmd,
    GemmCmd,
    InterChipCmd,
    NoCCmd,
    SramReadCmd,
    VectorCmd,
    WeightLoadCmd,
    WeightPrefetchCmd,
)
from srammachine.commands.base import integer, nonempty
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG, HardwareConfig

from .records import CommandState, ExecutionResult


class CommandCategory(Enum):
    COMPUTE = "compute"
    PREFETCH = "prefetch"
    MEMORY = "memory"
    COMMUNICATION = "communication"


def _freeze_value(value: Any) -> Any:
    """Copy JSON-shaped command metadata into immutable containers."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("command parameter mappings require string keys")
            frozen[key] = _freeze_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_value(item) for item in value)
    raise TypeError(
        "command parameters must contain only JSON-compatible values"
    )


def _freeze_parameters(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = _freeze_value(parameters)
    if not isinstance(frozen, Mapping):
        raise TypeError("parameters must be a mapping")
    return frozen


def _command_category(command: Command) -> CommandCategory:
    if isinstance(command, WeightPrefetchCmd):
        return CommandCategory.PREFETCH
    if isinstance(command, (GemmCmd, FlashAttentionCmd, VectorCmd)):
        return CommandCategory.COMPUTE
    if isinstance(command, (DramCmd, WeightLoadCmd, SramReadCmd)):
        return CommandCategory.MEMORY
    if isinstance(command, (NoCCmd, InterChipCmd)):
        return CommandCategory.COMMUNICATION
    raise TypeError(f"unsupported command type: {type(command).__name__}")


def _command_parameters(
    command: Command, hardware_config: HardwareConfig,
) -> Mapping[str, Any]:
    if isinstance(command, WeightPrefetchCmd):
        parameters = {
            "size_bytes": command.size_bytes,
            "weight_size_bytes": command.size_bytes,
            "weight_shape": command.weight_shape,
            "sram_resource_id": command.sram_resource_id,
        }
    elif isinstance(command, DramCmd):
        parameters = {"size_bytes": command.size_bytes}
    elif isinstance(command, WeightLoadCmd):
        parameters = {
            "size_bytes": command.size_bytes,
            "weight_size_bytes": command.size_bytes,
            "weight_shape": command.weight_shape,
        }
    elif isinstance(command, SramReadCmd):
        parameters = {
            "size_bytes": command.size_bytes,
            "data_kind": command.data_kind,
            "operand_role": "right_operand",
        }
    elif isinstance(command, GemmCmd):
        parameters = {
            "B": command.B,
            "M": command.M,
            "K": command.K,
            "N": command.N,
            "gemm_b": command.B,
            "gemm_m": command.M,
            "gemm_k": command.K,
            "gemm_n": command.N,
        }
    elif isinstance(command, FlashAttentionCmd):
        parameters = {
            "B": command.B,
            "M": command.M,
            "qk_K": command.qk_K,
            "qk_N": command.qk_N,
            "sv_K": command.sv_K,
            "sv_N": command.sv_N,
            "qk_flops": command.qk_flops,
            "sv_flops": command.sv_flops,
            "total_flops": command.total_flops,
            "softmax_flops": 0,
        }
    elif isinstance(command, VectorCmd):
        parameters = {
            "kind": command.kind,
            "m": command.m,
            "n": command.n,
            "vector_kind": command.kind,
            "vector_m": command.m,
            "vector_n": command.n,
            "params": command.params,
        }
    elif isinstance(command, CommCmd):
        critical_num, critical_den = _communication_volume_ratio(command)
        parameters = {
            "kind": command.kind,
            "communication_kind": command.kind,
            "scope": command.scope,
            "communication_scope": command.scope,
            "group": command.group,
            "size_bytes": command.size_bytes,
            "communication_size_bytes": command.size_bytes,
            "root": command.root,
            "reduce_kind": command.reduce_kind,
            "transfer_bytes": command.transfer_bytes,
            "communication_transfer_bytes": command.transfer_bytes,
            "communication_critical_path_bytes_numerator": critical_num,
            "communication_critical_path_bytes_denominator": critical_den,
            "communication_critical_path_bytes": critical_num / critical_den,
        }
        if isinstance(command, NoCCmd):
            link_bandwidth = (
                hardware_config.chip.noc.link_bandwidth_bytes_per_second
            )
            parameters.update({
                "noc_direction": (
                    "input" if command.resource_id.endswith("noc_input")
                    else "output" if command.resource_id.endswith("noc_output")
                    else "unspecified"
                ),
                "noc_parallel_link_count": command.parallel_link_count,
                "noc_link_bandwidth_bytes_per_second": link_bandwidth,
                "noc_effective_bandwidth_bytes_per_second": (
                    link_bandwidth * command.parallel_link_count
                ),
            })
    else:
        raise TypeError(f"unsupported command type: {type(command).__name__}")
    if (
        isinstance(command, (DramCmd, WeightLoadCmd, SramReadCmd))
        and command.logical_size_bytes is not None
    ):
        parameters.update({
            "logical_size_bytes": command.logical_size_bytes,
            "effective_size_bytes": command.size_bytes,
            "shared_die_factor": command.shared_die_factor,
        })
    return _freeze_parameters(parameters)


def _communication_volume_ratio(command: CommCmd) -> Tuple[int, int]:
    participant_count = len(command.group)
    if command.transfer_bytes is not None:
        matrix = command.transfer_bytes
        sent = [
            sum(size for column, size in enumerate(row) if column != row_index)
            for row_index, row in enumerate(matrix)
        ]
        received = [
            sum(matrix[row][column] for row in range(participant_count)
                if row != column)
            for column in range(participant_count)
        ]
        return max(sent + received, default=0), 1

    size_bytes = command.size_bytes or 0
    if command.kind in ("p2p", "broadcast", "reduce"):
        return size_bytes, 1
    if command.kind == "allreduce":
        return 2 * (participant_count - 1) * size_bytes, participant_count
    if command.kind == "allgather":
        return (participant_count - 1) * size_bytes, 1
    if command.kind in ("reduce_scatter", "alltoall"):
        return (participant_count - 1) * size_bytes, participant_count
    raise ValueError(f"unsupported communication kind: {command.kind}")


@dataclass(frozen=True)
class CommandResult:
    """One concrete hardware action, ready for a future trace exporter."""

    cmd_id: str
    op_id: str
    resource_id: str
    command_type: str
    category: CommandCategory
    layer_index: int
    instance_index: Optional[int]
    token_start: int
    token_stop: int
    node_path: Tuple[int, ...]
    start_time_ns: int
    end_time_ns: int
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("cmd_id", "op_id", "resource_id", "command_type"):
            nonempty(name, getattr(self, name))
        if not isinstance(self.category, CommandCategory):
            raise TypeError("category must be a CommandCategory")
        integer("layer_index", self.layer_index)
        if self.instance_index is not None:
            integer("instance_index", self.instance_index)
        integer("token_start", self.token_start)
        integer("token_stop", self.token_stop)
        if self.token_stop <= self.token_start:
            raise ValueError("token range must be nonempty")
        path = tuple(self.node_path)
        for index in path:
            integer("node_path index", index)
        integer("start_time_ns", self.start_time_ns)
        integer("end_time_ns", self.end_time_ns)
        if self.end_time_ns < self.start_time_ns:
            raise ValueError("end_time_ns must not precede start_time_ns")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        object.__setattr__(self, "node_path", path)
        object.__setattr__(
            self, "parameters", _freeze_parameters(self.parameters),
        )

    @property
    def duration_ns(self) -> int:
        return self.end_time_ns - self.start_time_ns


@dataclass(frozen=True)
class LayerResult:
    """Diagnostic active span of every command assigned to one layer."""

    layer_index: int
    command_ids: Tuple[str, ...]
    start_time_ns: int
    end_time_ns: int

    def __post_init__(self) -> None:
        integer("layer_index", self.layer_index)
        command_ids = tuple(self.command_ids)
        if not command_ids:
            raise ValueError("a layer result must contain at least one command")
        for cmd_id in command_ids:
            nonempty("command ID", cmd_id)
        if len(set(command_ids)) != len(command_ids):
            raise ValueError("layer command IDs must be unique")
        integer("start_time_ns", self.start_time_ns)
        integer("end_time_ns", self.end_time_ns)
        if self.end_time_ns < self.start_time_ns:
            raise ValueError("end_time_ns must not precede start_time_ns")
        object.__setattr__(self, "command_ids", command_ids)

    @property
    def active_span_ns(self) -> int:
        return self.end_time_ns - self.start_time_ns


@dataclass(frozen=True)
class SimulationResult:
    """In-memory aggregate and command-level results for one graph run."""

    execution_result: ExecutionResult
    hardware_config: HardwareConfig
    command_results: Tuple[CommandResult, ...]
    layer_results: Tuple[LayerResult, ...]
    model_layer_count: Optional[int] = None
    global_batch_size: Optional[int] = None
    accepted_tokens_per_step: int = 1
    _command_lookup: Mapping[str, CommandResult] = field(
        init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.execution_result, ExecutionResult):
            raise TypeError("execution_result must be an ExecutionResult")
        if not isinstance(self.hardware_config, HardwareConfig):
            raise TypeError("hardware_config must be a HardwareConfig")
        command_results = tuple(self.command_results)
        layer_results = tuple(self.layer_results)
        if any(not isinstance(item, CommandResult) for item in command_results):
            raise TypeError("command_results must contain CommandResult values")
        if any(not isinstance(item, LayerResult) for item in layer_results):
            raise TypeError("layer_results must contain LayerResult values")
        if (self.model_layer_count is None) != (self.global_batch_size is None):
            raise ValueError(
                "model_layer_count and global_batch_size must be provided together"
            )
        if self.model_layer_count is not None:
            integer("model_layer_count", self.model_layer_count, 1)
            integer("global_batch_size", self.global_batch_size, 1)
        integer("accepted_tokens_per_step", self.accepted_tokens_per_step, 1)

        lookup = {item.cmd_id: item for item in command_results}
        if len(lookup) != len(command_results):
            raise ValueError("command result IDs must be unique")
        execution_ids = set(self.execution_result.executions)
        if set(lookup) != execution_ids:
            raise ValueError(
                "command_results and execution_result must contain the same IDs"
            )
        expected_layers = tuple(range(len(layer_results)))
        actual_layers = tuple(item.layer_index for item in layer_results)
        if actual_layers != expected_layers:
            raise ValueError("layer results must be consecutive from layer 0")
        grouped_ids = tuple(
            cmd_id for layer in layer_results for cmd_id in layer.command_ids
        )
        if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != set(lookup):
            raise ValueError("layer results must partition all command results")

        object.__setattr__(self, "command_results", command_results)
        object.__setattr__(self, "layer_results", layer_results)
        object.__setattr__(self, "_command_lookup", MappingProxyType(lookup))

    @property
    def total_time_ns(self) -> int:
        return self.execution_result.total_time_ns

    @property
    def layer_count(self) -> int:
        return len(self.layer_results)

    @property
    def average_layer_time_ns(self) -> float:
        return self.pipeline_layer_time_ns

    @property
    def pipeline_layer_time_ns(self) -> float:
        """Steady-state layer throughput using the third layer milestone."""
        if not self.layer_results:
            return 0.0
        if self.layer_count >= 3:
            return (
                self._layer_core_start_time_ns(2)
                - self._layer_core_start_time_ns(1)
            )
        if self.layer_count == 2:
            return (
                self._layer_core_start_time_ns(1)
                - self._layer_core_start_time_ns(0)
            )
        return self._layer_core_span_ns(0)

    @property
    def latency_ns(self) -> Optional[float]:
        """Estimated full-model latency from steady-state representative layers."""
        if self.model_layer_count is None:
            return None
        return self.pipeline_layer_time_ns * self.model_layer_count

    @property
    def throughput_tokens_per_second(self) -> Optional[float]:
        """Accepted output-token throughput for the configured global batch."""
        latency_ns = self.latency_ns
        if latency_ns is None:
            return None
        if latency_ns == 0:
            return 0.0
        return (
            1_000_000_000
            * self.global_batch_size
            * self.accepted_tokens_per_step
            / latency_ns
        )

    def _layer_core_start_time_ns(self, layer_index: int) -> int:
        commands = self._core_commands_for_layer(layer_index)
        if not commands:
            raise ValueError(f"layer {layer_index} has no core commands")
        return min(item.start_time_ns for item in commands)

    def _layer_core_span_ns(self, layer_index: int) -> int:
        commands = self._core_commands_for_layer(layer_index)
        if not commands:
            return self.layer_results[layer_index].active_span_ns
        return (
            max(item.end_time_ns for item in commands)
            - min(item.start_time_ns for item in commands)
        )

    def _core_commands_for_layer(
        self, layer_index: int,
    ) -> Tuple[CommandResult, ...]:
        return tuple(
            item for item in self.commands_for_layer(layer_index)
            if item.cmd_id.endswith(".core")
        )

    def command_result(self, cmd_id: str) -> CommandResult:
        return self._command_lookup[cmd_id]

    def commands_for_layer(self, layer_index: int) -> Tuple[CommandResult, ...]:
        integer("layer_index", layer_index)
        if layer_index >= self.layer_count:
            return ()
        return tuple(
            self._command_lookup[cmd_id]
            for cmd_id in self.layer_results[layer_index].command_ids
        )

    def commands_for_op(
        self, op_id: str, *, layer_index: Optional[int] = None,
    ) -> Tuple[CommandResult, ...]:
        nonempty("op_id", op_id)
        if layer_index is not None:
            integer("layer_index", layer_index)
        return tuple(
            item for item in self.command_results
            if item.op_id == op_id
            and (layer_index is None or item.layer_index == layer_index)
        )

    def commands_by_category(
        self, category: CommandCategory,
    ) -> Tuple[CommandResult, ...]:
        if not isinstance(category, CommandCategory):
            raise TypeError("category must be a CommandCategory")
        return tuple(
            item for item in self.command_results if item.category is category
        )


def build_simulation_result(
    graph: CommandGraph,
    execution_result: ExecutionResult,
    hardware_config: HardwareConfig = DEFAULT_HARDWARE_CONFIG,
    *,
    model_layer_count: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    accepted_tokens_per_step: int = 1,
) -> SimulationResult:
    """Combine a completed low-level run with graph and trace metadata."""
    if not isinstance(graph, CommandGraph):
        raise TypeError("graph must be a CommandGraph")
    if not isinstance(execution_result, ExecutionResult):
        raise TypeError("execution_result must be an ExecutionResult")
    if not isinstance(hardware_config, HardwareConfig):
        raise TypeError("hardware_config must be a HardwareConfig")

    graph_ids = tuple(command.cmd_id for command in graph.commands)
    if set(graph_ids) != set(execution_result.executions):
        raise ValueError("execution result does not match the command graph")
    if any(
        execution_result.states[cmd_id] is not CommandState.COMPLETED
        for cmd_id in graph_ids
    ):
        raise ValueError("execution result contains incomplete commands")
    if graph_ids and set(graph.traces) != set(graph_ids):
        raise ValueError("every command requires CommandTrace metadata")

    layer_indices = sorted({
        trace.layer_index for trace in graph.traces.values()
    })
    if layer_indices != list(range(len(layer_indices))):
        raise ValueError("CommandTrace layer indices must be consecutive from 0")

    command_results = []
    for command in graph.commands:
        trace = graph.traces[command.cmd_id]
        execution = execution_result.executions[command.cmd_id]
        command_results.append(CommandResult(
            cmd_id=command.cmd_id,
            op_id=command.op_id,
            resource_id=command.resource_id,
            command_type=type(command).__name__,
            category=_command_category(command),
            layer_index=trace.layer_index,
            instance_index=trace.instance_index,
            token_start=trace.token_start,
            token_stop=trace.token_stop,
            node_path=trace.node_path,
            start_time_ns=execution.start_time_ns,
            end_time_ns=execution.end_time_ns,
            parameters=_command_parameters(command, hardware_config),
        ))

    layer_results = []
    for layer_index in layer_indices:
        items = tuple(
            item for item in command_results
            if item.layer_index == layer_index
        )
        layer_results.append(LayerResult(
            layer_index=layer_index,
            command_ids=tuple(item.cmd_id for item in items),
            start_time_ns=min(item.start_time_ns for item in items),
            end_time_ns=max(item.end_time_ns for item in items),
        ))

    return SimulationResult(
        execution_result=execution_result,
        hardware_config=hardware_config,
        command_results=tuple(command_results),
        layer_results=tuple(layer_results),
        model_layer_count=model_layer_count,
        global_batch_size=global_batch_size,
        accepted_tokens_per_step=accepted_tokens_per_step,
    )
