"""Lower a PipeTree and explicit operator mappings into a command DAG.

V1 assumes a token-preserving linear operator flow. It does not infer residual
branches, expert routing, cross-request dependencies or physical memory layout.
One representative stack models the identical SRAM pressure of every stack.
"""
from copy import deepcopy
from dataclasses import replace
from typing import Iterator, Mapping, Tuple

from srammachine.commands import (
    CommandGraph, CommandTrace, DramReadCmd, DramWriteCmd,
    FlashAttentionCmd,
    FusedIndexerScoreCmd, GemmCmd,
    VectorCmd, NoCCmd, InterChipCmd,
)
from srammachine.commands.base import integer
from srammachine.frontend.modules import (
    BMMOp, FabricOp, FlashAttentionOp, FusedIndexerScoreOp, NoCOp, Operator, VectorOp,
)
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG, HardwareConfig
from .mapping import OperatorMapping
from .tree import GroupNode, Node, OpInstance, OpNode, PipeTree


class TreeParser:
    """Hardware-bound tree expansion and command lowering.

    parse requires full-batch operators and per-operator mapping policies.
    iter_instances remains available for inspecting only the tree expansion.
    """

    def __init__(self, hardware_config: HardwareConfig = DEFAULT_HARDWARE_CONFIG):
        if not isinstance(hardware_config, HardwareConfig):
            raise TypeError("hardware_config must be a HardwareConfig")
        self.hardware_config = hardware_config

    def parse(
        self, tree: PipeTree, operators: Mapping[str, Operator],
        mappings: Mapping[str, OperatorMapping], *, layer_count: int = 4,
    ) -> CommandGraph:
        """Generate four consecutive layers by default, as in the design diagram.

        This repeats one layer template, not four different model layer types.
        Each layer has its own prefetch and command IDs but shares hardware
        resource IDs. Pass layer_count=1 to inspect the original single layer.
        """
        integer("layer_count", layer_count, 1)
        single = self._parse_one_layer(tree, operators, mappings)
        if layer_count == 1:
            return single

        # 原设计流程图要求复制单层 CommandGraph，模拟连续四层。
        # 保留原 op_id 以查询模型算子，使用 layer_index 区分不同层。
        instances = tuple(self.iter_instances(tree))
        first = [item for item in instances if item.op_id == tree.operator_order[0]]
        last = [item for item in instances if item.op_id == tree.operator_order[-1]]
        boundary_edges = []
        i = j = 0
        while i < len(last) and j < len(first):
            source, target = last[i], first[j]
            if max(source.token_start, target.token_start) < min(source.token_stop, target.token_stop):
                boundary_edges.append((f"i{source.index}.core", f"i{target.index}.core"))
            if source.token_stop <= target.token_stop:
                i += 1
            if target.token_stop <= source.token_stop:
                j += 1

        commands, edges, traces = [], [], {}
        for layer in range(layer_count):
            prefix = f"layer{layer}."
            for cmd in single.commands:
                copied_id = prefix + cmd.cmd_id
                # Only VectorCmd carries arbitrary nested mutable parameters.
                # Other commands are immutable or normalize mapping values in
                # __post_init__, so avoid recursively copying all their fields.
                copied = (
                    replace(cmd, cmd_id=copied_id, params=deepcopy(cmd.params))
                    if isinstance(cmd, VectorCmd)
                    else replace(cmd, cmd_id=copied_id)
                )
                commands.append(copied)
                traces[copied_id] = replace(single.traces[cmd.cmd_id], layer_index=layer)
            edges.extend((prefix + source, prefix + target) for source, target in single.edges)
            if layer:
                # 仅连接前层输出到后层对应 token 的计算，不能把所有图根节点
                # 都串起来；后层 DRAM 读取仍允许提前执行。
                # KV 写回也不构成整层屏障，竞争由共享硬件资源处理。
                previous = f"layer{layer - 1}."
                edges.extend((previous + source, prefix + target)
                             for source, target in boundary_edges)
        graph = CommandGraph(commands, edges, traces=traces)
        return graph

    def _parse_one_layer(
        self, tree: PipeTree, operators: Mapping[str, Operator],
        mappings: Mapping[str, OperatorMapping],
    ) -> CommandGraph:
        tree.validate()
        expected = set(tree.operator_order)
        if set(operators) != expected or set(mappings) != expected:
            raise ValueError("operator and mapping IDs must exactly match the tree")
        checked = {}
        for op_id in tree.operator_order:
            op, mapping = operators[op_id], mappings[op_id]
            if not isinstance(op, (BMMOp, FlashAttentionOp, VectorOp, NoCOp, FabricOp)):
                raise TypeError(f"unsupported operator: {op_id}")
            if op.op_id != op_id:
                raise ValueError(f"operator key does not match op_id: {op_id}")
            if not isinstance(mapping, OperatorMapping):
                raise TypeError(f"expected OperatorMapping for {op_id}")
            if mapping.additional_resource_ids and not (
                isinstance(op, NoCOp) and op.scope == "intra_chip"
            ):
                raise ValueError("additional resources require intra-chip NoC")
            # Snapshot and revalidate mutable model operators without changing them.
            op = replace(op)
            axes = (("B", "M") if isinstance(op, (BMMOp, FlashAttentionOp)) else
                    ("m",) if isinstance(op, VectorOp) else ("size_bytes",))
            if mapping.batch_axis not in axes:
                raise ValueError(f"{op_id}: batch_axis must be one of {axes}")
            checked[op_id] = op

        instances = tuple(self.iter_instances(tree))
        by_op = {key: [] for key in tree.operator_order}
        for instance in instances:
            by_op[instance.op_id].append(instance)
        commands, edges, traces = [], [], {}
        def emit(cmd, instance, once=False):
            commands.append(cmd)
            traces[cmd.cmd_id] = CommandTrace(
                None if once else instance.index,
                0 if once else instance.token_start,
                tree.batch_size if once else instance.token_stop,
                instance.node_path,
            )
            return cmd.cmd_id

        core_ids = {}
        dynamic_dram_reads = {}
        for instance in instances:
            op_id, batch = instance.op_id, instance.batch_size
            op, mapping = checked[op_id], mappings[op_id]
            prefix = f"i{instance.index}"
            mapped_batch = self._mapped_batch_size(
                batch, mapping.batch_partition_degree,
            )
            demand_batch = self._mapped_batch_size(
                batch,
                mapping.demand_batch_partition_degree
                or mapping.batch_partition_degree,
            )
            memory_bytes = mapping.dram_read_once_bytes
            if mapping.dram_read_once_bytes:
                emit(DramReadCmd(
                    prefix + ".weight_read", op_id, mapping.dram_resource_id,
                    mapping.dram_read_once_bytes,
                    logical_size_bytes=mapping.dram_read_once_logical_bytes,
                    shared_die_factor=mapping.shared_die_factor,
                ), instance)
            if mapping.dram_read_bytes_per_token:
                read_id = emit(DramReadCmd(
                    prefix + ".read", op_id, mapping.dram_resource_id,
                    demand_batch * mapping.dram_read_bytes_per_token,
                    logical_size_bytes=(
                        demand_batch
                        * mapping.dram_read_logical_bytes_per_token
                        if mapping.dram_read_logical_bytes_per_token is not None
                        else None
                    ),
                    shared_die_factor=mapping.shared_die_factor,
                ), instance)
                dynamic_dram_reads[instance.index] = read_id
                memory_bytes += demand_batch * mapping.dram_read_bytes_per_token
            load_bytes = (mapping.weight_load_fixed_bytes
                          + mapped_batch * mapping.weight_load_bytes_per_token)
            if load_bytes:
                emit(DramReadCmd(
                    prefix + ".load", op_id, mapping.dram_resource_id, load_bytes,
                    logical_size_bytes=(
                        (mapping.weight_load_logical_fixed_bytes or 0)
                        + mapped_batch
                        * (mapping.weight_load_logical_bytes_per_token or 0)
                        if (
                            mapping.weight_load_logical_fixed_bytes is not None
                            or mapping.weight_load_logical_bytes_per_token
                            is not None
                        ) else None
                    ),
                    shared_die_factor=mapping.shared_die_factor,
                ), instance)
                memory_bytes += load_bytes
            if mapping.sram_read_bytes_per_mapped_token:
                sram_read = emit(DramReadCmd(
                    prefix + ".sram_read", op_id, mapping.dram_resource_id,
                    demand_batch * mapping.sram_read_bytes_per_mapped_token,
                    logical_size_bytes=(
                        demand_batch
                        * mapping.sram_read_logical_bytes_per_mapped_token
                        if mapping.sram_read_logical_bytes_per_mapped_token
                        is not None else None
                    ),
                    shared_die_factor=mapping.shared_die_factor,
                ), instance)
                dynamic_dram_reads[instance.index] = sram_read
                memory_bytes += demand_batch * mapping.sram_read_bytes_per_mapped_token

            core = self._core_command(
                prefix + ".core", op, mapping, instance, tree.batch_size,
                memory_bytes=memory_bytes,
            )
            core_ids[instance.index] = emit(core, instance)
            if mapping.dram_write_bytes_per_token:
                write = emit(DramWriteCmd(
                    prefix + ".write", op_id, mapping.dram_resource_id,
                    mapped_batch * mapping.dram_write_bytes_per_token,
                    logical_size_bytes=(
                        mapped_batch
                        * mapping.dram_write_logical_bytes_per_token
                        if mapping.dram_write_logical_bytes_per_token is not None
                        else None
                    ),
                    shared_die_factor=mapping.shared_die_factor,
                ), instance)
                edges.append((core.cmd_id, write))

        # Each operator's intervals are ordered and partition the full batch.
        # Merge adjacent partitions in linear time, connecting every overlap.
        for previous, current in zip(tree.operator_order, tree.operator_order[1:]):
            sources, targets = by_op[previous], by_op[current]
            i = j = 0
            while i < len(sources) and j < len(targets):
                source, target = sources[i], targets[j]
                if max(source.token_start, target.token_start) < min(source.token_stop, target.token_stop):
                    edges.append((core_ids[source.index], core_ids[target.index]))
                if source.token_stop <= target.token_stop:
                    i += 1
                if target.token_stop <= source.token_stop:
                    j += 1

        # A dynamic right operand is a demand read, not a look-ahead weight
        # load.  Open it at the same BMM-block boundary as the explicit input
        # broadcast so the two resources can run in parallel.  The GEMM waits
        # for both through their independent edges.
        order_index = {
            op_id: index for index, op_id in enumerate(tree.operator_order)
        }
        for instance_index, dram_read in dynamic_dram_reads.items():
            instance = instances[instance_index]
            position = order_index[instance.op_id]
            broadcast_id = f"{instance.op_id}.input_broadcast"
            if position == 0 or tree.operator_order[position - 1] != broadcast_id:
                raise ValueError(
                    f"dynamic DRAM BMM is missing its input broadcast: "
                    f"{instance.op_id}"
                )
            if position < 2:
                continue
            sources = by_op[tree.operator_order[position - 2]]
            for source in sources:
                if max(source.token_start, instance.token_start) < min(
                    source.token_stop, instance.token_stop,
                ):
                    edges.append((core_ids[source.index], dram_read))

        return CommandGraph(commands, edges, traces=traces)

    @staticmethod
    def _mapped_batch_size(batch: int, partition_degree: int) -> int:
        return (batch + partition_degree - 1) // partition_degree

    @staticmethod
    def _weight_shape(op, size_bytes):
        if not isinstance(op, BMMOp):
            return {"size_bytes": size_bytes}
        matrix_elements = op.K * op.N
        weight_batches, remainder = divmod(size_bytes, matrix_elements)
        shape = {"K": op.K, "N": op.N, "size_bytes": size_bytes}
        if remainder == 0 and weight_batches:
            shape["B"] = weight_batches
        return shape

    @classmethod
    def _prefetch_weight_shape(cls, op, mapping, size_bytes):
        if mapping.weight_shape is None:
            return cls._weight_shape(op, size_bytes)
        return dict(mapping.weight_shape, size_bytes=size_bytes)

    @staticmethod
    def _core_command(cmd_id, op, mapping, instance, full_batch, *, memory_bytes=0):
        def scale(value):
            unit_count = mapping.batch_scaling_unit_count
            if unit_count is not None:
                factor, remainder = divmod(value, unit_count)
                if remainder:
                    raise ValueError(
                        f"{op.op_id}: {mapping.batch_axis} value {value} is "
                        f"not an integer multiple of batch scaling units "
                        f"{unit_count}"
                    )

                # Prefix-ceiling partitioning puts any remainder in earlier
                # microbatches.  It is additive over nested SplitTree ranges,
                # preserves the exact total, and differs by at most one unit
                # between equally sized sibling microbatches.
                def prefix_units(position):
                    return (unit_count * position + full_batch - 1) // full_batch

                local_units = (
                    prefix_units(instance.token_stop)
                    - prefix_units(instance.token_start)
                )
                if local_units <= 0:
                    raise ValueError(
                        f"{op.op_id}: microbatch contains no expert work unit"
                    )
                return factor * local_units

            mapped_batch = TreeParser._mapped_batch_size(
                instance.batch_size, mapping.batch_partition_degree,
            )
            mapped_full_batch = TreeParser._mapped_batch_size(
                full_batch, mapping.batch_partition_degree,
            )
            quotient, remainder = divmod(
                value * mapped_batch, mapped_full_batch,
            )
            if remainder:
                raise ValueError(
                    f"{op.op_id}: {mapping.batch_axis} value {value} cannot be "
                    f"scaled exactly from mapped batch {mapped_full_batch} "
                    f"to {mapped_batch}"
                )
            return quotient

        common = dict(cmd_id=cmd_id, op_id=op.op_id, resource_id=mapping.resource_id)
        if isinstance(op, BMMOp):
            dimensions = {name: getattr(op, name) for name in ("B", "M", "K", "N")}
            dimensions[mapping.batch_axis] = scale(dimensions[mapping.batch_axis])
            command_type = (
                FusedIndexerScoreCmd
                if isinstance(op, FusedIndexerScoreOp) else GemmCmd
            )
            if isinstance(op, FusedIndexerScoreOp):
                dimensions["main_gemm_flops_override"] = (
                    None
                    if op.main_gemm_flops_override is None
                    else scale(op.main_gemm_flops_override)
                )
            return command_type(**common, memory_bytes=memory_bytes, **dimensions)
        if isinstance(op, FlashAttentionOp):
            dimensions = {
                name: getattr(op, name)
                for name in ("B", "M", "qk_K", "qk_N", "sv_K", "sv_N")
            }
            dimensions[mapping.batch_axis] = scale(
                dimensions[mapping.batch_axis]
            )
            return FlashAttentionCmd(**common, memory_bytes=memory_bytes, **dimensions)
        if isinstance(op, VectorOp):
            return VectorCmd(**common, memory_bytes=memory_bytes, kind=op.kind, m=scale(op.m), n=op.n, params=op.params)
        if isinstance(op, NoCOp):
            return NoCCmd(
                **common, operation=op.operation, size_bytes=scale(op.size_bytes),
                kind=op.kind, group=op.group,
                additional_resource_ids=mapping.additional_resource_ids,
            )
        matrix = (None if op.transfer_bytes is None else
                  tuple(tuple(scale(value) for value in row) for row in op.transfer_bytes))
        return InterChipCmd(
            **common, kind=op.kind, group=op.group,
            size_bytes=None if op.size_bytes is None else scale(op.size_bytes),
            root=op.root, reduce_kind=op.reduce_kind, transfer_bytes=matrix,
        )

    def iter_instances(self, tree: PipeTree) -> Iterator[OpInstance]:
        """Stream instances without materializing an entire large execution plan."""
        tree.validate()
        index = 0

        def walk(
            node: Node, start: int, stop: int, path: Tuple[int, ...]
        ) -> Iterator[OpInstance]:
            nonlocal index
            if isinstance(node, OpNode):
                yield OpInstance(index, node.op_id, start, stop, path)
                index += 1
                return
            if not isinstance(node, GroupNode):
                raise TypeError("expected OpNode or GroupNode")
            width = (stop - start) // node.split
            # Child-major order is intentional: visit every slice for this
            # child before moving to its next sibling.
            for child_index, child in enumerate(node.children):
                for part in range(node.split):
                    part_start = start + part * width
                    yield from walk(
                        child, part_start, part_start + width, path + (child_index,)
                    )

        yield from walk(tree.root, 0, tree.batch_size, ())
