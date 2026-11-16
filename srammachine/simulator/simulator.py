"""User-facing orchestration for command-graph simulation."""

from dataclasses import replace
from pathlib import Path
from typing import Optional, Union

from srammachine.commands import CommandGraph
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG, HardwareConfig
from srammachine.mapping import HardwareMappingResult

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
        self._additional_layer_cache = {}

    def _execute(self, graph: CommandGraph, mapping: Optional[HardwareMappingResult]):
        """Use each operator's hardware view while retaining shared resources."""
        overrides = {} if mapping is None else {
            op_id: policy.hardware_config
            for op_id, policy in mapping.operator_mappings.items()
            if policy.hardware_config is not None
        }
        return GraphExecutor.simulate(graph, self.hardware_config, operator_hardware=overrides)

    def _latency_components(self, mapping: HardwareMappingResult, layer_time_ns: float) -> tuple:
        """Weight representative layers; cache fixed auxiliary layer baselines."""
        if mapping.represented_layer_count is None:
            return ()
        components = [(mapping.layer_type, mapping.represented_layer_count, layer_time_ns)]
        for additional in mapping.additional_layer_mappings:
            key = id(additional)
            cached = self._additional_layer_cache.get(key)
            if cached is None:
                from srammachine.pipetree import TreeParser
                graph = TreeParser(self.hardware_config).parse(
                    additional.root_node, additional.operators, additional.operator_mappings,
                )
                result = self.run(graph, mapping_result=additional)
                cached = (additional, result.model_latency_components)
                self._additional_layer_cache[key] = cached
            components.extend(cached[1])
        return tuple(components)

    def run(
        self, graph: CommandGraph, *,
        mapping_result: Optional[HardwareMappingResult] = None,
    ) -> SimulationResult:
        """Simulate in memory without printing, writing files or opening UI.

        Pass the corresponding ``mapping_result`` to also derive full-model
        latency and accepted-token throughput. Generic hand-built graphs may
        omit it and retain only graph/layer timing.
        """
        if not isinstance(graph, CommandGraph):
            raise TypeError("graph must be a CommandGraph")
        if mapping_result is not None:
            if not isinstance(mapping_result, HardwareMappingResult):
                raise TypeError(
                    "mapping_result must be a HardwareMappingResult"
                )
            if mapping_result.chip_count != self.hardware_config.chip_count:
                raise ValueError(
                    "mapping_result and simulator hardware chip counts differ"
                )
        execution_result = self._execute(graph, mapping_result)
        model_layer_count = (
            None if mapping_result is None
            else mapping_result.model_config.num_hidden_layers
        )
        inference = (
            None if mapping_result is None
            else mapping_result.request.inference_config
        )
        result = build_simulation_result(
            graph, execution_result, self.hardware_config,
            model_layer_count=model_layer_count,
            global_batch_size=(
                None if inference is None else inference.global_batch_size
            ),
            accepted_tokens_per_step=(
                1 if inference is None else inference.accepted_tokens_per_step
            ),
        )
        if mapping_result is not None:
            components = self._latency_components(mapping_result, result.pipeline_layer_time_ns)
            if components:
                result = replace(result, model_latency_components=components,
                                 model_layer_count=sum(count for _, count, _ in components))
        return result

    def run_latency(
        self, graph: CommandGraph, *, mapping_result: HardwareMappingResult,
    ) -> float:
        """Run the same scheduler and return model latency without trace objects.

        Simulated annealing only needs latency for most candidates.  Building
        immutable command parameter dictionaries and ``CommandResult`` objects
        for every rejected tree adds work and memory but cannot affect the
        scheduler.  Final trees still use :meth:`run` for full materialization.
        """
        if not isinstance(graph, CommandGraph):
            raise TypeError("graph must be a CommandGraph")
        if not isinstance(mapping_result, HardwareMappingResult):
            raise TypeError("mapping_result must be a HardwareMappingResult")
        if mapping_result.chip_count != self.hardware_config.chip_count:
            raise ValueError(
                "mapping_result and simulator hardware chip counts differ"
            )
        execution = self._execute(graph, mapping_result)
        graph_ids = tuple(command.cmd_id for command in graph.commands)
        if set(graph_ids) != set(execution.executions):
            raise ValueError("execution result does not match the command graph")
        if graph_ids and set(graph.traces) != set(graph_ids):
            raise ValueError("every command requires CommandTrace metadata")

        layer_indices = sorted({
            trace.layer_index for trace in graph.traces.values()
        })
        if layer_indices != list(range(len(layer_indices))):
            raise ValueError("CommandTrace layer indices must be consecutive from 0")
        if not layer_indices:
            pipeline_layer_time_ns = 0
        else:
            def bounds(layer_index: int, *, core_only: bool) -> tuple[int, int] | None:
                items = [
                    execution.executions[command.cmd_id]
                    for command in graph.commands
                    if graph.traces[command.cmd_id].layer_index == layer_index
                    and (not core_only or command.cmd_id.endswith(".core"))
                ]
                if not items:
                    return None
                return (
                    min(item.start_time_ns for item in items),
                    max(item.end_time_ns for item in items),
                )

            if len(layer_indices) >= 3:
                previous = bounds(1, core_only=True)
                current = bounds(2, core_only=True)
                if previous is None or current is None:
                    raise ValueError("steady-state layer has no core commands")
                pipeline_layer_time_ns = current[0] - previous[0]
            elif len(layer_indices) == 2:
                previous = bounds(0, core_only=True)
                current = bounds(1, core_only=True)
                if previous is None or current is None:
                    raise ValueError("pipeline layer has no core commands")
                pipeline_layer_time_ns = current[0] - previous[0]
            else:
                active = bounds(0, core_only=True)
                if active is None:
                    active = bounds(0, core_only=False)
                if active is None:
                    pipeline_layer_time_ns = 0
                else:
                    pipeline_layer_time_ns = active[1] - active[0]
        components = self._latency_components(mapping_result, pipeline_layer_time_ns)
        if components:
            return sum(count * latency for _, count, latency in components)
        return (
            pipeline_layer_time_ns
            * mapping_result.model_config.num_hidden_layers
        )

    def run_and_trace(
        self,
        graph: CommandGraph,
        *,
        mapping_result: Optional[HardwareMappingResult] = None,
        trace_label: str = "srammachine",
        output_dir: Union[str, Path] = DEFAULT_TRACE_OUTPUT_DIR,
    ) -> SimulationArtifacts:
        """Run observably: print average latency and save one Perfetto trace.

        Unlike ``run``, this method deliberately performs console and file I/O.
        It is intended for baseline/final inspection, not the simulated-
        annealing inner loop. Browser launch remains an explicit caller action.
        """
        result = self.run(graph, mapping_result=mapping_result)
        layer_time_ns = result.average_layer_time_ns
        print(
            f"Pipeline layer time: {layer_time_ns:.3f} ns "
            f"({layer_time_ns / 1_000_000:.6f} ms)"
        )
        if result.latency_ns is not None:
            if result.model_latency_components:
                print("Layer estimate: " + " + ".join(
                    f"{count} x {name} ({latency:.3f} ns/layer)"
                    for name, count, latency in result.model_latency_components
                ))
            print(
                f"Inference latency: {result.latency_ns:.3f} ns "
                f"({result.latency_ns / 1_000_000:.6f} ms)"
            )
            print(
                "Throughput: "
                f"{result.throughput_tokens_per_second:.3f} tokens/s"
            )
        trace_path = export_perfetto_trace(
            result,
            trace_label=trace_label,
            output_dir=output_dir,
            mapping_result=mapping_result,
        )
        print(f"Perfetto trace: {trace_path}")
        return SimulationArtifacts(result, trace_path)
