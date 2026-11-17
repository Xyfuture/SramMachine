"""Operator-specific hardware and memory policies for ablation experiments."""

from dataclasses import fields, is_dataclass, replace
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from srammachine.frontend.modules import (
    BMMOp, FlashAttentionOp, FusedIndexerScoreOp, MiniMaxFusedIndexerScoreOp, WeightSource,
)
from srammachine.hardware import HardwareConfig
from srammachine.mapping import HardwareMappingResult
from srammachine.mapping.hbf_bandwidth import (indexer_noc_bandwidth_per_pu,
    HBFBandwidthTable, normalize_hbf_csv)
from srammachine.search import apply_partition_chains, partition_chains_to_dict

if TYPE_CHECKING:
    from srammachine.simulator import SimulationResult


ABLATIONS = ("XFold", "NoXfold", "NoDRAMShare", "NoFlashShare",
             "NoDRAMFLashShare", "NoKVLayout", "NoMoELayout", "SeparatePath", "NoSeparatePath", "NoHBFLayout")
KV_LAYOUT_ABLATIONS = ("NoKVLayout", "NoHBFLayout")
MOE_LAYOUT_ABLATIONS = ("NoMoELayout", "NoHBFLayout")
DEFAULT_MOE_FACTOR = 1 / 6
DEFAULT_NO_KV_BANDWIDTH_CSV = Path(__file__).resolve().parents[2] / "32k no kvlayout bandwidth.csv"


def load_no_kv_bandwidth(path=DEFAULT_NO_KV_BANDWIDTH_CSV):
    """Normalize the raw measured per-chip CSV using the existing HBF units."""
    with TemporaryDirectory() as directory:
        normalized = Path(directory) / "normalized.csv"
        normalize_hbf_csv(path, normalized)
        return HBFBandwidthTable.from_csv(normalized)


def latency_metrics(result: "SimulationResult", model: str, ablation: str) -> dict:
    """Report simulated decoder latency; no ablation adds artificial delay."""
    raw = result.latency_ns
    return {
        "simulated_model_step_latency_ns": raw,
        "additional_latency_per_layer_ns": 0,
        "additional_model_latency_ns": 0,
        "model_step_latency_ns": raw,
        "model_step_latency_ms": raw / 1e6,
        "adjusted_model_latency_components": [list(item) for item in result.model_latency_components],
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
                   *, moe_factor: float = DEFAULT_MOE_FACTOR,
                   moe_flash_scope: str = "weights", no_kv_table=None) -> HardwareMappingResult:
    """Apply policies after the base-grid CSV lookup and before layout search.

    NoKVLayout 从专用实测表读取 sparse KV 的 HBF 带宽（每卡 GB/s × 1e9 ÷ 2048 → 每 PU bytes/s），
    替换直接 KV 读取或 XFold FC2 的独立 KV 预取；不缩放 Indexer，
    不改 NoC 共享 HBM 增量，不再增加固定延迟。缺失配置由表查询直接报错。
    MiniMax dense 前缀没有对应实测值，保持原配置。Default MoE scope
    'weights' changes ordinary MoE BMM weights only, excluding Indexer subclasses
    and KV prefetch. Scope 'stage' reproduces the previous broader ablation.
    Dense prefix mappings receive the same ablation, retaining adapted baseline
    layouts. Explicit measured Flash rates are scaled alongside hardware defaults.
    """
    if name not in ABLATIONS:
        raise ValueError(f"unknown ablation: {name}")
    if moe_flash_scope not in ("stage", "weights"):
        raise ValueError("moe_flash_scope must be stage or weights")
    for factor in (moe_factor,):
        if not math.isfinite(factor) or not 0 < factor <= 1:
            raise ValueError("layout factors must be finite and in (0, 1]")
    no_kv_rate = None
    # NoHBFLayout = 实测 NoKVLayout + 权重范围的 NoMoELayout。
    # KV/预取使用专用表；普通 MoE BMM 权重 Flash 带宽乘 moe_factor（默认 1/6）。
    # FC2 的 Flash 字段用于 KV 预取，因此不再乘权重系数；Indexer、HBM 增量不变。
    # 不叠加固定延迟，也不引入 SeparatePath 的分块或互斥规则。
    if name == "NoHBFLayout":
        moe_flash_scope = "weights"
    if name in KV_LAYOUT_ABLATIONS:
        no_kv_table = no_kv_table or load_no_kv_bandwidth()
        request = mapping.request
        inference = request.inference_config
        no_kv_rate = no_kv_table.lookup(request.model_name, "sparse_kv", inference.mtp_enabled,
                                       inference.global_batch_size, inference.input_sequence_length)
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
        if name in MOE_LAYOUT_ABLATIONS and op_id.startswith("moe."):
            if moe_flash_scope == "stage" or (
                type(op) is BMMOp and not op.flash_prefetch_bytes_per_request
            ):
                # XFold FC2 weights use DRAM only; its Flash override belongs
                # to the independent KV prefetch and must retain the CSV rate.
                factor *= moe_factor
        if op.weight_memory is not None:
            rate = (op.weight_memory.flash_bandwidth_bytes_per_second
                    or base.chip.logic_die.processing_unit.flash_bandwidth_bytes_per_second)
            if (no_kv_rate is not None and mapping.layer_type != "dense"
                    and (op.flash_prefetch_bytes_per_request or
                         (isinstance(op, FlashAttentionOp) and op.weight_memory.source != WeightSource.DRAM))):
                rate = no_kv_rate
            if (half_flash and isinstance(op, (FusedIndexerScoreOp, MiniMaxFusedIndexerScoreOp))
                    and op.weight_memory.source != WeightSource.DRAM):
                # NoFlashShare（含 NoDRAMFLashShare）下，HBM 中没有 Indexer cache，
                # 因而不能通过 NoC 共享 HBM 带宽，需去掉这部分增量。
                # Remove the baseline-grid NoC contribution before doubling the
                # old per-PU bandwidth for the half-sized NoFlashShare grid.
                rate -= indexer_noc_bandwidth_per_pu(base)
            operators[op_id] = replace(op, weight_memory=replace(
                op.weight_memory, flash_bandwidth_bytes_per_second=rate * factor))
    additional = tuple(apply_ablation(
        item, base, name, moe_factor=moe_factor,
        moe_flash_scope=moe_flash_scope, no_kv_table=no_kv_table,
    ) for item in mapping.additional_layer_mappings)
    result = replace(mapping, operators=operators, operator_mappings=policies,
                     op_list=[operators[key] for key in mapping.root_node.operator_order],
                     additional_layer_mappings=additional)
    return apply_partition_chains(result, profiles, base)
