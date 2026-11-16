"""Operator-specific hardware and memory policies for the six ablations."""

from dataclasses import fields, is_dataclass, replace
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from srammachine.frontend.modules import (
    BMMOp, FlashAttentionOp, FusedIndexerScoreOp, MiniMaxFusedIndexerScoreOp, WeightSource,
)
from srammachine.hardware import HardwareConfig
from srammachine.mapping import HardwareMappingResult
from srammachine.search import apply_partition_chains, partition_chains_to_dict

if TYPE_CHECKING:
    from srammachine.simulator import SimulationResult


ABLATIONS = ("XFold", "NoXfold", "NoDRAMShare", "NoFlashShare",
             "NoDRAMFLashShare", "NoKVLayout", "NoMoELayout")
DEFAULT_INDEXER_FACTOR = 0.87
DEFAULT_MOE_FACTOR = 1 / 6
NO_KV_LAYER_DELAY_NS = {
    "deepseek-v3": 14_000,
    "deepseek-v3.2": 14_000,
    "glm-5.1": 14_000,
    "minimax-m3": 25_000,
}


def latency_metrics(result: "SimulationResult", model: str, ablation: str) -> dict:
    """Add NoKVLayout's per-decoder-layer delay after simulation, exactly once.

    The trace and original layer components remain raw simulated timings.
    Reported model latency and throughput include the additive delay for every
    decoder layer, including MiniMax's dense prefix, independently of batch/MTP.
    """
    raw = result.latency_ns
    count = result.model_layer_count
    if raw is None or raw <= 0 or count is None or count <= 0:
        raise ValueError("Expected positive full-model latency and layer count")
    per_layer = NO_KV_LAYER_DELAY_NS[model] if ablation == "NoKVLayout" else 0
    overhead = per_layer * count
    latency = raw + overhead
    return {
        "simulated_model_step_latency_ns": raw,
        "additional_latency_per_layer_ns": per_layer,
        "additional_model_latency_ns": overhead,
        "model_step_latency_ns": latency,
        "model_step_latency_ms": latency / 1e6,
        "adjusted_model_latency_components": [
            [name, layers, duration + per_layer]
            for name, layers, duration in result.model_latency_components
        ],
    }


def load_hardware(path: Path | None = None) -> HardwareConfig:
    """Overlay a nested JSON object on HardwareConfig defaults without mutation."""
    def overlay(base, values):
        if not isinstance(values, dict):
            raise ValueError("hardware configuration must contain JSON objects")
        known = {field.name for field in fields(base)}
        if set(values) - known:
            raise ValueError(f"unknown hardware fields: {sorted(set(values) - known)}")
        return replace(base, **{
            key: overlay(getattr(base, key), value) if is_dataclass(getattr(base, key)) else value
            for key, value in values.items()
        })

    hardware = HardwareConfig() if path is None else overlay(HardwareConfig(), json.loads(path.read_text()))
    if (hardware.chip.noc.mesh_rows, hardware.chip.noc.mesh_columns) != (32, 64):
        raise ValueError("ablation base hardware requires a 32 x 64 PU grid")
    return hardware


def half_grid_hardware(base: HardwareConfig, port: str) -> HardwareConfig:
    """Halve PU rows per die and double one PU port, preserving the die count."""
    if port not in ("dram", "flash"):
        raise ValueError("port must be dram or flash")
    die = base.chip.logic_die
    if die.pu_mesh_rows % 2:
        raise ValueError("halving the PU grid requires an even number of rows per die")
    field = f"{port}_bandwidth_bytes_per_second"
    pu = replace(die.processing_unit, **{field: 2 * getattr(die.processing_unit, field)})
    chip = replace(base.chip,
                   logic_die=replace(die, pu_mesh_rows=die.pu_mesh_rows // 2, processing_unit=pu),
                   noc=replace(base.chip.noc, mesh_rows=base.chip.noc.mesh_rows // 2))
    return replace(base, chip=chip)


def is_attention_or_score(op) -> bool:
    """Classify by operator type, including the score moved into MoE by XFold."""
    return isinstance(op, (FlashAttentionOp, FusedIndexerScoreOp, MiniMaxFusedIndexerScoreOp))


def apply_ablation(mapping: HardwareMappingResult, base: HardwareConfig, name: str,
                   *, kv_factor: float = DEFAULT_INDEXER_FACTOR, moe_factor: float = DEFAULT_MOE_FACTOR,
                   moe_flash_scope: str = "weights") -> HardwareMappingResult:
    """Apply policies after the base-grid CSV lookup and before layout search.

    XFold KV prefetch remains on its actual FC2 hardware view. NoKVLayout
    scales only IndexerScore cache reads from their CSV rate. Default MoE scope
    'weights' changes ordinary MoE BMM weights only, excluding Indexer subclasses
    and KV prefetch. Scope 'stage' reproduces the previous broader ablation.
    Dense prefix mappings receive the same ablation, retaining adapted baseline
    layouts. Explicit measured Flash rates are scaled alongside hardware defaults.
    """
    if name not in ABLATIONS:
        raise ValueError(f"unknown ablation: {name}")
    if moe_flash_scope not in ("stage", "weights"):
        raise ValueError("moe_flash_scope must be stage or weights")
    for factor in (kv_factor, moe_factor):
        if not math.isfinite(factor) or not 0 < factor <= 1:
            raise ValueError("layout factors must be finite and in (0, 1]")
    operators, policies = dict(mapping.operators), dict(mapping.operator_mappings)
    profiles = partition_chains_to_dict(mapping)
    for op_id, op in operators.items():
        special = is_attention_or_score(op)
        half_dram = not special and name in ("NoDRAMShare", "NoDRAMFLashShare")
        half_flash = special and name in ("NoFlashShare", "NoDRAMFLashShare")
        hardware = half_grid_hardware(base, "dram" if half_dram else "flash") \
            if half_dram or half_flash else base
        policies[op_id] = replace(policies[op_id], hardware_config=hardware)
        if (half_dram or half_flash) and op_id in profiles:
            # Retain the nested order and remove one factor of two from p.
            for step in reversed(profiles[op_id]):
                if step["pu_dimension"] == "p" and step["factor"] > 1:
                    if step["factor"] % 2:
                        raise ValueError(f"cannot halve partition for {op_id}")
                    step["factor"] //= 2
                    break
        factor = 2.0 if half_flash else 1.0
        if (name == "NoKVLayout" and isinstance(op, (FusedIndexerScoreOp, MiniMaxFusedIndexerScoreOp))
                and (op.weight_memory is None or op.weight_memory.source != WeightSource.DRAM)):
            if op.weight_memory is None or op.weight_memory.flash_bandwidth_bytes_per_second is None:
                raise ValueError(f"NoKVLayout requires a CSV Indexer bandwidth for {op_id}")
            factor *= kv_factor
        if name == "NoMoELayout" and op_id.startswith("moe."):
            if moe_flash_scope == "stage" or (
                type(op) is BMMOp and not op.flash_prefetch_bytes_per_request
            ):
                # XFold FC2 weights use DRAM only; its Flash override belongs
                # to the independent KV prefetch and must retain the CSV rate.
                factor *= moe_factor
        if op.weight_memory is not None:
            rate = (op.weight_memory.flash_bandwidth_bytes_per_second
                    or base.chip.logic_die.processing_unit.flash_bandwidth_bytes_per_second)
            operators[op_id] = replace(op, weight_memory=replace(
                op.weight_memory, flash_bandwidth_bytes_per_second=rate * factor))
    additional = tuple(apply_ablation(
        item, base, name, kv_factor=kv_factor, moe_factor=moe_factor,
        moe_flash_scope=moe_flash_scope,
    ) for item in mapping.additional_layer_mappings)
    result = replace(mapping, operators=operators, operator_mappings=policies,
                     op_list=[operators[key] for key in mapping.root_node.operator_order],
                     additional_layer_mappings=additional)
    return apply_partition_chains(result, profiles, base)
