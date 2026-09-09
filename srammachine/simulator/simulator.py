"""User-facing orchestration for command-graph simulation."""

from pathlib import Path
from typing import Union

from srammachine.commands import CommandGraph
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG, HardwareConfig

from .executor import GraphExecutor
from .perfetto import (
    DEFAULT_TRACE_OUTPUT_DIR,
    SimulationArtifacts,
    export_perfetto_trace,
)
from .results import SimulationResult, build_simulation_result


class Simulator:
    """Run command graphs either silently or with explicit trace output.

    ``run`` is the side-effect-free hot path for simulated annealing.
    ``run_and_trace`` is the observable path for baseline and final inspection:
    it prints average layer latency and writes a Perfetto trace, but does not
    launch a browser.
    """

    def __init__(
        self, hardware_config: HardwareConfig = DEFAULT_HARDWARE_CONFIG,
    ) -> None:
        if not isinstance(hardware_config, HardwareConfig):
            raise TypeError("hardware_config must be a HardwareConfig")
        self.hardware_config = hardware_config

    def run(self, graph: CommandGraph) -> SimulationResult:
        """Simulate in memory without printing, writing files or opening UI."""
        if not isinstance(graph, CommandGraph):
            raise TypeError("graph must be a CommandGraph")
        execution_result = GraphExecutor.simulate(graph, self.hardware_config)
        return build_simulation_result(
            graph, execution_result, self.hardware_config,
        )

    def run_and_trace(
        self,
        graph: CommandGraph,
        *,
        trace_label: str = "srammachine",
        output_dir: Union[str, Path] = DEFAULT_TRACE_OUTPUT_DIR,
    ) -> SimulationArtifacts:
        """Run observably: print average latency and save one Perfetto trace.

        Unlike ``run``, this method deliberately performs console and file I/O.
        It is intended for baseline/final inspection, not the simulated-
        annealing inner loop. Browser launch remains an explicit caller action.
        """
        result = self.run(graph)
        layer_time_ns = result.average_layer_time_ns
        print(
            f"Pipeline layer time: {layer_time_ns:.3f} ns "
            f"({layer_time_ns / 1_000_000:.6f} ms)"
        )
        trace_path = export_perfetto_trace(
            result,
            trace_label=trace_label,
            output_dir=output_dir,
        )
        print(f"Perfetto trace: {trace_path}")
        return SimulationArtifacts(result, trace_path)
