"""Desim-based execution of hardware command graphs."""

from .executor import GraphExecutor
from .perfetto import (
    DEFAULT_TRACE_OUTPUT_DIR,
    SimulationArtifacts,
    export_perfetto_trace,
)
from .records import CommandExecution, CommandState, ExecutionResult
from .results import (
    CommandCategory,
    CommandResult,
    LayerResult,
    SimulationResult,
    build_simulation_result,
)
from .simulator import Simulator
from .stages import (
    DramResourceStage,
    HardwareResourceStage,
    InterChipFabricStage,
    NoCResourceStage,
    ProcessingUnitStage,
    SramResourceStage,
    VectorUnitStage,
)
from .vector_costs import VECTOR_FLOPS_PER_ELEMENT

__all__ = [
    "HardwareResourceStage",
    "DramResourceStage",
    "SramResourceStage",
    "ProcessingUnitStage",
    "VectorUnitStage",
    "NoCResourceStage",
    "InterChipFabricStage",
    "GraphExecutor",
    "CommandExecution",
    "ExecutionResult",
    "CommandState",
    "Simulator",
    "SimulationResult",
    "LayerResult",
    "CommandResult",
    "CommandCategory",
    "build_simulation_result",
    "DEFAULT_TRACE_OUTPUT_DIR",
    "SimulationArtifacts",
    "export_perfetto_trace",
    "VECTOR_FLOPS_PER_ELEMENT",
]
