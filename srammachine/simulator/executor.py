"""Incremental command-DAG executor backed by Desim resource stages."""

import heapq
from types import MappingProxyType

from Desim import FIFO, SimModule, SimSession
from greenlet import GreenletExit

from srammachine.commands import CommandGraph
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG, HardwareConfig
from .records import CommandExecution, CommandState, ExecutionResult
from .stages import stage_class_for_command


class GraphExecutor(SimModule):
    """Issue ready commands and commit them only after resource completion."""

    def __init__(
        self, graph: CommandGraph,
        hardware_config: HardwareConfig = DEFAULT_HARDWARE_CONFIG,
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
        self._rank = {
            command.cmd_id: index for index, command in enumerate(graph.commands)
        }
        self._remaining_predecessors = {
            command.cmd_id: len(graph.predecessors(command.cmd_id))
            for command in graph.commands
        }
        self._states = {
            command.cmd_id: CommandState.PENDING for command in graph.commands
        }
        self._executions = {}
        self._ready_by_resource = {}
        self._busy_resources = set()
        self._completion_fifo = FIFO(max(1, len(graph.commands)))
        self._stages = self._build_stages()

        for command in graph.commands:
            if self._remaining_predecessors[command.cmd_id] == 0:
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
        for command in self.graph.commands:
            stage_type = stage_class_for_command(command)
            existing = stage_types.setdefault(command.resource_id, stage_type)
            if existing is not stage_type:
                raise ValueError(
                    f"resource_id {command.resource_id!r} mixes "
                    f"{existing.__name__} and {stage_type.__name__}"
                )
        return {
            resource_id: stage_type(
                resource_id, self.hardware_config, self._completion_fifo,
            )
            for resource_id, stage_type in stage_types.items()
        }

    def _make_ready(self, cmd_id: str) -> None:
        if self._states[cmd_id] is not CommandState.PENDING:
            raise RuntimeError(f"command cannot become ready twice: {cmd_id}")
        command = self.graph.command(cmd_id)
        self._states[cmd_id] = CommandState.READY
        heap = self._ready_by_resource.setdefault(command.resource_id, [])
        heapq.heappush(heap, (
            self.graph.scheduling_priorities[cmd_id], self._rank[cmd_id],
        ))

    def _dispatch_idle_resources(self) -> None:
        for resource_id, stage in self._stages.items():
            ready = self._ready_by_resource.get(resource_id)
            if resource_id in self._busy_resources or not ready:
                continue
            _, rank = heapq.heappop(ready)
            command = self.graph.commands[rank]
            if self._states[command.cmd_id] is not CommandState.READY:
                raise RuntimeError(f"invalid ready state: {command.cmd_id}")
            self._states[command.cmd_id] = CommandState.RUNNING
            self._busy_resources.add(resource_id)
            stage.submit(command)

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
        self._busy_resources.remove(command.resource_id)

        for successor in self.graph.successors(cmd_id):
            self._remaining_predecessors[successor] -= 1
            if self._remaining_predecessors[successor] < 0:
                raise RuntimeError(f"negative dependency count: {successor}")
            if self._remaining_predecessors[successor] == 0:
                self._make_ready(successor)

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
    ) -> ExecutionResult:
        """Run one graph in a fresh Desim global session."""
        SimSession.reset()
        SimSession.init()
        completed = False
        try:
            executor = cls(graph, hardware_config)
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
