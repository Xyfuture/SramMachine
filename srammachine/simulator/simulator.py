"""User-facing orchestration for command-graph simulation."""

from srammachine.commands import CommandGraph
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG, HardwareConfig

from .executor import GraphExecutor
from .results import SimulationResult, build_simulation_result


class Simulator:
    """Run one command graph and collect immutable in-memory statistics."""

    def __init__(
        self, hardware_config: HardwareConfig = DEFAULT_HARDWARE_CONFIG,
    ) -> None:
        if not isinstance(hardware_config, HardwareConfig):
            raise TypeError("hardware_config must be a HardwareConfig")
        self.hardware_config = hardware_config

    def run(self, graph: CommandGraph) -> SimulationResult:
        if not isinstance(graph, CommandGraph):
            raise TypeError("graph must be a CommandGraph")
        execution_result = GraphExecutor.simulate(graph, self.hardware_config)
        return build_simulation_result(
            graph, execution_result, self.hardware_config,
        )
