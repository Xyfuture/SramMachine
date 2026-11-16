"""Per-workload enumeration of explicit MLA, DSA and MiniMax PU partition layouts."""

from collections import defaultdict
from dataclasses import replace
from functools import lru_cache
import itertools
import math
from typing import Any, Iterator, Mapping

from srammachine.commands import GemmCmd, NoCCmd, PUMemoryReadCmd
from srammachine.frontend.modules import (
    BMMOp, FlashAttentionOp, FlashMSAOp, FusedIndexerScoreOp,
    MiniMaxFusedIndexerScoreOp, PartitionChain,
)
from srammachine.hardware import HardwareConfig
from srammachine.mapping import HardwareMappingResult
from srammachine.pipetree import GroupNode, OpNode, OperatorMapping, RootNode, TreeParser
from srammachine.systolic_cost import systolic_gemm_duration_ns


SUPPORTED_PARTITION_MODELS = frozenset(("deepseek-v3", "deepseek-v3.2", "glm-5.1", "minimax-m3"))
_NATIVE_OPS = (BMMOp, FlashAttentionOp, FusedIndexerScoreOp,
               FlashMSAOp, MiniMaxFusedIndexerScoreOp)
PartitionProfile = dict[str, list[dict[str, Any]]]


def partition_chains_to_dict(mapping: HardwareMappingResult) -> PartitionProfile:
    """Save explicit layouts independently of SplitTree structure."""
    if mapping.request.model_name not in SUPPORTED_PARTITION_MODELS:
        return {}
    return {
        op_id: [dict(dimension=s.dimension, pu_dimension=s.pu_dimension, factor=s.factor)
                for s in op.partition_chain.steps]
        for op_id, op in mapping.operators.items() if type(op) in _NATIVE_OPS
    }


def apply_partition_chains(
    mapping: HardwareMappingResult, profile: Mapping[str, list[dict[str, Any]]],
    hardware: HardwareConfig,
) -> HardwareMappingResult:
    """Apply saved chains and refresh PU dimensions without mutating the mapper.

    The profile may override a subset of supported operators. Every supplied
    chain must cover the chip's complete PU grid and use the op's legal axes.
    """
    operators, metadata = dict(mapping.operators), dict(mapping.hardware_mappings)
    for op_id, steps in profile.items():
        if op_id not in operators or type(operators[op_id]) not in _NATIVE_OPS:
            raise ValueError(f"unsupported partition override: {op_id}")
        op_hardware = mapping.operator_mappings[op_id].hardware_config or hardware
        grid = (op_hardware.chip.noc.mesh_rows, op_hardware.chip.noc.mesh_columns)
        chain = PartitionChain(tuple(
            (step["dimension"], step["pu_dimension"], step["factor"]) for step in steps
        ))
        chain.validate_grid(*grid)
        op = operators[op_id] = replace(operators[op_id], partition_chain=chain)
        if isinstance(op, FlashAttentionOp):
            shape = {"B": op.B, "S": op.qk_N}
            if isinstance(op, FlashMSAOp):
                shape["H"] = op.H
            tile = chain.compute_shape(shape)
            local = dict(metadata[op_id].chip_dimensions, B=tile["B"],
                         qk_N=tile["S"], sv_K=tile["S"])
            if isinstance(op, FlashMSAOp):
                local["H"] = tile["H"]
            weight = 0
        elif isinstance(op, (FusedIndexerScoreOp, MiniMaxFusedIndexerScoreOp)):
            tile = chain.compute_shape({"B": op.B, "S": op.N})
            local = dict(metadata[op_id].chip_dimensions, B=tile["B"], N=tile["S"])
            weight = 0
        else:
            local = chain.compute_shape({axis: getattr(op, axis) for axis in ("B", "M", "K", "N")})
            weight = (1 if op.shared_weight else local["B"]) * local["K"] * local["N"] * op.weight_dtype_bytes
        strategy = ",".join(f"{s.dimension}-{s.pu_dimension}-{s.factor}" for s in chain.steps)
        metadata[op_id] = replace(metadata[op_id], parallel_strategy=strategy,
                                  pu_dimensions=local, weight_bytes=weight)
    return replace(mapping, operators=operators, hardware_mappings=metadata,
                   op_list=[operators[key] for key in mapping.root_node.operator_order])


def _exponents(total: int, length: int) -> Iterator[tuple[int, ...]]:
    if length == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for rest in _exponents(total - first, length - 1):
                yield (first, *rest)


@lru_cache(maxsize=8)
def _grid_factors(rows: int, columns: int, axis_count: int = 4) -> tuple:
    return tuple(
        (tuple(2 ** n for n in p), tuple(2 ** n for n in q))
        for p in _exponents(rows.bit_length() - 1, axis_count)
        for q in _exponents(columns.bit_length() - 1, axis_count)
    )


def _chain(axes: tuple[str, ...], p: tuple, q: tuple) -> PartitionChain:
    return PartitionChain(tuple(
        (axis, direction, factor)
        for direction, factors in (("p", p), ("q", q))
        for axis, factor in zip(axes, factors) if factor > 1
    ))


def _memory_time(op: BMMOp, size: int, hardware: HardwareConfig) -> int:
    pu = hardware.chip.logic_die.processing_unit
    durations = defaultdict(int)
    for port, volume, rate in op.weight_memory.transfers(
        size, pu.dram_bandwidth_bytes_per_second, pu.flash_bandwidth_bytes_per_second,
    ):
        durations[port] += math.ceil(volume * 1e9 / rate)
    # XFold KV prefetch is an independent command. Its duration constrains
    # next-layer Attention, not FC2's layout cost or PU occupancy.
    return max(durations.values(), default=0)


def _bmm_finalists(op: BMMOp, policy: OperatorMapping, batch: int,
                   hardware: HardwareConfig) -> list[tuple[int, PartitionChain]]:
    """Rank canonical layouts using exact worst-row IO and largest-PU shapes.

    With all p splits before q splits, the union across q exactly covers the
    clipped p-parent input/output projection. This remains true for ragged
    tensors, so ranking does not require enumerating 2048 PU boxes per layout.
    The best three layouts are independently lowered by the regular parser.
    """
    axes = ("B", "M", "K", "N")
    shape = tuple(getattr(op, axis) for axis in axes)
    rows, columns = hardware.chip.noc.mesh_rows, hardware.chip.noc.mesh_columns
    ranked = []
    for p, q in _grid_factors(rows, columns):
        factors = tuple(a * b for a, b in zip(p, q))
        if any(f > 2 ** (dim - 1).bit_length() for dim, f in zip(shape, factors)):
            continue
        tile = tuple((dim + f - 1) // f for dim, f in zip(shape, factors))
        parent = tuple((dim + f - 1) // f for dim, f in zip(shape, p))
        B, M, K, N = tile
        core = GemmCmd("probe", op.op_id, policy.resource_id, B, M, K, N)
        compute = systolic_gemm_duration_ns(core, hardware)
        size = (1 if op.shared_weight else B) * K * N * op.weight_dtype_bytes
        memory = _memory_time(op, size, hardware)
        input_bytes = parent[0] * parent[1] * parent[2] * op.input_dtype_bytes
        output_bytes = parent[0] * parent[1] * parent[3] * op.output_dtype_bytes
        noc = hardware.chip.noc
        cost = (math.ceil(input_bytes * 1e9 / noc.bandwidth_for("FromIOdie"))
                + max(compute, memory)
                + math.ceil(output_bytes * 1e9 / noc.bandwidth_for("ToIOdie")))
        ranked.append((cost, size, p, q))
    ranked.sort()
    return [(cost, _chain(axes, p, q)) for cost, _, p, q in ranked[:3]]


def _attention_candidates(op, hardware: HardwareConfig) -> Iterator[PartitionChain]:
    rows, columns = hardware.chip.noc.mesh_rows, hardware.chip.noc.mesh_columns
    sequence = op.qk_N if isinstance(op, FlashAttentionOp) else op.N
    if isinstance(op, FlashMSAOp):
        # H partitions whole GQA groups; only S shards require state merging.
        limits = (2 ** (op.B - 1).bit_length(), 2 ** (op.H - 1).bit_length(), sequence)
        for p, q in _grid_factors(rows, columns, 3):
            if all(a * b <= limit for a, b, limit in zip(p, q, limits)):
                yield _chain(("B", "H", "S"), p, q)
        return
    for bp, bq in itertools.product(
        (2 ** n for n in range(rows.bit_length())),
        (2 ** n for n in range(columns.bit_length())),
    ):
        sp, sq = rows // bp, columns // bq
        if bp * bq <= 2 ** (op.B - 1).bit_length() and sp * sq <= sequence:
            yield _chain(("B", "S"), (bp, sp), (bq, sq))


def _op_cost(op, policy: OperatorMapping, batch: int, parser: TreeParser) -> int:
    tree = RootNode(batch, (op.op_id,), GroupNode((OpNode(op.op_id),)))
    graph = parser.parse(tree, {op.op_id: op}, {op.op_id: policy}, layer_count=1)
    pu = parser.hardware_config.chip.logic_die.processing_unit
    total = 0
    for command in graph.commands:
        if isinstance(command, PUMemoryReadCmd):
            continue  # Both independent streams are included in memory_duration_ns.
        if isinstance(command, NoCCmd):
            rate = parser.hardware_config.chip.noc.bandwidth_for(command.operation)
            total += math.ceil(command.size_bytes * 1e9 / rate)
        elif type(command) is GemmCmd:
            total += max(systolic_gemm_duration_ns(command, parser.hardware_config),
                         command.memory_duration_ns)
        else:
            total += max(math.ceil(command.total_flops * 1e9 / pu.peak_flops),
                         command.memory_duration_ns)
    return total


def optimize_partition_chains(mapping: HardwareMappingResult,
                              hardware: HardwareConfig) -> HardwareMappingResult:
    """Select per-op layouts once per global batch/MTP workload, before execution.

    Supported models use full, power-of-two PU meshes and IO-die input/output
    routes. Other models retain their mappings. This optimizes the unsplit
    representative layer and does not search SplitTree structures. Additional
    layer mappings (MiniMax's dense prefix) retain their baseline layouts.
    """
    if mapping.request.model_name not in SUPPORTED_PARTITION_MODELS:
        return mapping
    batch = mapping.request.inference_config.global_batch_size
    selected = {}
    for op_id, op in mapping.operators.items():
        if type(op) not in _NATIVE_OPS or op.weight_memory is None:
            continue
        policy = mapping.operator_mappings[op_id]
        op_hardware = policy.hardware_config or hardware
        grid = (op_hardware.chip.noc.mesh_rows, op_hardware.chip.noc.mesh_columns)
        if any(value & (value - 1) for value in grid):
            continue
        parser = TreeParser(op_hardware)
        if (op.input_source != "io_die" or op.output_destination != "io_die"
                or policy.dram_read_bytes_per_token or policy.dram_write_bytes_per_token
                or policy.weight_read_fixed_bytes or policy.cache_read_bytes_per_request is not None
                or policy.cache_query_group_size != 1):
            continue
        best_chain, best_cost = op.partition_chain, _op_cost(op, policy, batch, parser)
        if type(op) is BMMOp:
            candidates = _bmm_finalists(op, policy, batch, op_hardware)
        else:
            candidates = [(None, chain) for chain in _attention_candidates(op, op_hardware)]
        for predicted, chain in candidates:
            cost = _op_cost(replace(op, partition_chain=chain), policy, batch, parser)
            if predicted is not None and cost != predicted:
                raise RuntimeError(f"partition cost disagrees with lowered commands for {op_id}")
            if cost < best_cost:
                best_chain, best_cost = chain, cost
        selected[op_id] = [dict(dimension=s.dimension, pu_dimension=s.pu_dimension, factor=s.factor)
                           for s in best_chain.steps]
    return apply_partition_chains(mapping, selected, hardware)


__all__ = ["optimize_partition_chains", "partition_chains_to_dict", "apply_partition_chains"]
