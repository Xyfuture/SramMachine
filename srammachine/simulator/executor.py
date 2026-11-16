"""Incremental command-DAG executor backed by Desim resource stages."""

import heapq
from types import MappingProxyType
from typing import Mapping

from Desim import FIFO, SimModule, SimSession
from greenlet import GreenletExit

from srammachine.commands import CommandGraph, InterChipCmd, PUMemoryReadCmd
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG, HardwareConfig
from .records import CommandExecution, CommandState, ExecutionResult
from .stages import PUMemoryResourceStage, stage_class_for_command


class GraphExecutor(SimModule):
    """Issue ready commands and commit them only after resource completion."""

    def __init__(
        self, graph: CommandGraph,
        hardware_config: HardwareConfig = DEFAULT_HARDWARE_CONFIG,
        *, operator_hardware: Mapping[str, HardwareConfig] | None = None,
    ) -> None:
        if SimSession.scheduler is None:
            raise RuntimeError("initialize SimSession before creating GraphExecutor")
        if SimSession.scheduler.status == "finished":
            raise RuntimeError("cannot attach GraphExecutor to a finished SimSession")
        if not isinstance(graph, CommandGraph):
            raise TypeError("graph must be a CommandGraph")
        if not isinstance(hardware_config, HardwareConfig):
            raise TypeError("hardware_config must be a HardwareConfig")
        super().__init__()
        self.graph = graph
        self.hardware_config = hardware_config
        self.operator_hardware = dict(operator_hardware or {})
        for value in self.operator_hardware.values():
            if not isinstance(value, HardwareConfig):
                raise TypeError("operator_hardware values must be HardwareConfig")
            if value.chip_count != hardware_config.chip_count:
                raise ValueError("operator hardware must preserve chip count")
        self._rank = {
            command.cmd_id: index for index, command in enumerate(graph.commands)
        }
        self._remaining_predecessors = {
            command.cmd_id: len(graph.predecessors(command.cmd_id))
            for command in graph.commands
        }
        self._remaining_start_predecessors = {
            command.cmd_id: len(graph.start_predecessors(command.cmd_id))
            for command in graph.commands
        }
        self._states = {
            command.cmd_id: CommandState.PENDING for command in graph.commands
        }
        self._executions = {}
        self._ready = []
        self._busy_resources = set()
        self._completion_fifo = FIFO(max(1, len(graph.commands)))
        self._stage_keys = {
            command.cmd_id: self._stage_key(command)
            for command in graph.commands
        }
        self._stages = self._build_stages()

        for command in graph.commands:
            if (
                self._remaining_predecessors[command.cmd_id] == 0
                and self._remaining_start_predecessors[command.cmd_id] == 0
            ):
                self._make_ready(command.cmd_id)
        self.register_coroutine(self._process)

    @property
    def stages(self):
        return MappingProxyType(self._stages)

    @property
    def states(self):
        return MappingProxyType(self._states)

    @property
    def is_complete(self) -> bool:
        return len(self._executions) == len(self.graph.commands)

    def _build_stages(self):
        stage_types = {}
        stage_resource_ids = {}
        stage_hardware = {}
        stage_classes = {}
        for command in self.graph.commands:
            stage_key = self._stage_keys[command.cmd_id]
            stage_type = stage_class_for_command(command)
            for resource_key in self._resource_keys(command):
                existing = stage_types.setdefault(resource_key, stage_type)
                if existing is not stage_type:
                    raise ValueError(
                        f"resource_id {resource_key!r} mixes "
                        f"{existing.__name__} and {stage_type.__name__}"
                    )
            stage_resource_ids[stage_key] = command.resource_id
            stage_classes[stage_key] = stage_type
            stage_hardware[stage_key] = self.operator_hardware.get(command.op_id, self.hardware_config)
            if isinstance(command, PUMemoryReadCmd):
                pu_id = command.resource_id.rsplit(".", 1)[0]
                for port in ("dram", "flash"):
                    port_id = f"{pu_id}.{port}"
                    stage_types[port_id] = PUMemoryResourceStage
                    stage_resource_ids[port_id] = port_id
                    stage_classes[port_id] = PUMemoryResourceStage
                    stage_hardware.setdefault(port_id, self.hardware_config)
        return {
            stage_key: stage_classes[stage_key](
                resource_id,
                stage_hardware[stage_key],
                self._completion_fifo,
            )
            for stage_key, resource_id in stage_resource_ids.items()
        }

    def _stage_key(self, command) -> str:
        if isinstance(command, InterChipCmd):
            return f"{command.resource_id}::{command.cmd_id}"
        if command.op_id in self.operator_hardware:
            return f"{command.resource_id}::op::{command.op_id}"
        return command.resource_id

    def _resource_keys(self, command) -> tuple[str, ...]:
        if isinstance(command, InterChipCmd):
            return (self._stage_keys[command.cmd_id],)
        return command.resource_ids

    def _make_ready(self, cmd_id: str) -> None:
        if self._states[cmd_id] is not CommandState.PENDING:
            raise RuntimeError(f"command cannot become ready twice: {cmd_id}")
        self._states[cmd_id] = CommandState.READY
        heapq.heappush(self._ready, (
            self.graph.scheduling_priorities[cmd_id], self._rank[cmd_id],
        ))

    def _try_make_ready(self, cmd_id: str) -> None:
        if self._states[cmd_id] is not CommandState.PENDING:
            return
        if (
            self._remaining_predecessors[cmd_id] == 0
            and self._remaining_start_predecessors[cmd_id] == 0
        ):
            self._make_ready(cmd_id)

    def _release_start_successors(self, cmd_id: str) -> None:
        for successor in self.graph.start_successors(cmd_id):
            self._remaining_start_predecessors[successor] -= 1
            if self._remaining_start_predecessors[successor] < 0:
                raise RuntimeError(f"negative start dependency count: {successor}")
            self._try_make_ready(successor)

    def _dispatch_idle_resources(self) -> None:
        blocked = []
        while self._ready:
            priority, rank = heapq.heappop(self._ready)
            command = self.graph.commands[rank]
            resource_keys = self._resource_keys(command)
            if self._busy_resources.intersection(resource_keys):
                blocked.append((priority, rank))
                continue
            if self._states[command.cmd_id] is not CommandState.READY:
                raise RuntimeError(f"invalid ready state: {command.cmd_id}")
            self._states[command.cmd_id] = CommandState.RUNNING
            self._busy_resources.update(resource_keys)
            self._stages[self._stage_keys[command.cmd_id]].submit(command)
            self._release_start_successors(command.cmd_id)
        for item in blocked:
            heapq.heappush(self._ready, item)

    def _commit(self, execution: CommandExecution) -> None:
        cmd_id = execution.cmd_id
        if cmd_id not in self._states:
            raise ValueError(f"completion references unknown command: {cmd_id}")
        if self._states[cmd_id] is not CommandState.RUNNING:
            raise RuntimeError(f"command completed from invalid state: {cmd_id}")
        command = self.graph.command(cmd_id)
        if execution.resource_id != command.resource_id:
            raise ValueError(f"completion resource mismatch: {cmd_id}")
        self._states[cmd_id] = CommandState.COMPLETED
        self._executions[cmd_id] = execution
        self._busy_resources.difference_update(self._resource_keys(command))

        for successor in self.graph.successors(cmd_id):
            self._remaining_predecessors[successor] -= 1
            if self._remaining_predecessors[successor] < 0:
                raise RuntimeError(f"negative dependency count: {successor}")
            self._try_make_ready(successor)

    def _process(self) -> None:
        self._dispatch_idle_resources()
        while not self.is_complete:
            batch = [self._completion_fifo.read()]
            available = self._completion_fifo.empty_semaphore.get_value()
            for _ in range(available):
                batch.append(self._completion_fifo.read())
            batch.sort(key=lambda execution: self._rank[execution.cmd_id])
            for execution in batch:
                self._commit(execution)
            self._dispatch_idle_resources()
        for stage in self._stages.values():
            stage.close()

    def result(self) -> ExecutionResult:
        """Return an immutable result after all commands have completed."""
        if not self.is_complete:
            incomplete = [
                command.cmd_id for command in self.graph.commands
                if self._states[command.cmd_id] is not CommandState.COMPLETED
            ]
            raise RuntimeError(f"simulation is incomplete: {incomplete}")
        executions = {
            command.cmd_id: self._executions[command.cmd_id]
            for command in self.graph.commands
        }
        states = {
            command.cmd_id: self._states[command.cmd_id]
            for command in self.graph.commands
        }
        total_time_ns = max(
            (execution.end_time_ns for execution in executions.values()),
            default=0,
        )
        return ExecutionResult(total_time_ns, executions, states)

    @classmethod
    def simulate(
        cls, graph: CommandGraph,
        hardware_config: HardwareConfig = DEFAULT_HARDWARE_CONFIG,
        *, operator_hardware: Mapping[str, HardwareConfig] | None = None,
    ) -> ExecutionResult:
        """Run one graph in a fresh Desim global session."""
        SimSession.reset()
        SimSession.init()
        completed = False
        try:
            executor = cls(graph, hardware_config, operator_hardware=operator_hardware)
            SimSession.scheduler.run()
            if not executor.is_complete:
                incomplete = [
                    command.cmd_id for command in graph.commands
                    if executor.states[command.cmd_id] is not CommandState.COMPLETED
                ]
                raise RuntimeError(f"simulation deadlocked: {incomplete}")
            result = executor.result()
            completed = True
            return result
        finally:
            cls._release_session(completed)

    @staticmethod
    def _release_session(preserve_finished_scheduler: bool) -> None:
        """Release Desim greenlet stacks and module references after one run."""
        modules = tuple(SimSession.sim_modules)
        for module in modules:
            for coroutine in tuple(module._coroutines):
                if not coroutine.dead:
                    try:
                        coroutine.throw(GreenletExit)
                    except GreenletExit:
                        pass
                module._coroutines.discard(coroutine)

        # Keeping the successfully finished scheduler preserves the existing
        # public ability to inspect SimSession.sim_time after simulate().  The
        # module registry can still be cleared, which releases the graph and all
        # FIFO/Event ownership chains.  Failed sessions are unusable and reset.
        SimSession.sim_modules = []
        if not preserve_finished_scheduler:
            SimSession.reset()
