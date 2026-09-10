"""Lower a PipeTree and explicit operator mappings into a command DAG.

V1 assumes a token-preserving linear operator flow. It does not infer residual
branches, expert routing, cross-request dependencies or physical memory layout.
One representative stack models the identical SRAM pressure of every stack.
"""
from collections import deque
from copy import deepcopy
from dataclasses import replace
from typing import Iterator, Mapping, Tuple

from srammachine.commands import (
    CommandGraph, CommandTrace, DramCmd, DramReadCmd, DramWriteCmd,
    SramReadCmd, WeightLoadCmd, WeightPrefetchCmd, GemmCmd, VectorCmd, NoCCmd,
    InterChipCmd,
)
from srammachine.commands.base import integer
from srammachine.frontend.modules import BMMOp, VectorOp, CommOp, Operator
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
            return self._apply_weight_constraints(single)

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
                # Copies may carry nested vector parameters: do not share them.
                commands.append(replace(deepcopy(cmd), cmd_id=copied_id))
                traces[copied_id] = replace(single.traces[cmd.cmd_id], layer_index=layer)
            edges.extend((prefix + source, prefix + target) for source, target in single.edges)
            if layer:
                # 仅连接前层输出到后层对应 token 的计算，不能把所有图根节点
                # 都串起来；后层 DRAM 预取和 WeightLoad 仍允许提前执行。
                # KV 写回也不构成整层屏障，竞争由共享硬件资源处理。
                previous = f"layer{layer - 1}."
                edges.extend((previous + source, prefix + target)
                             for source, target in boundary_edges)
        graph = CommandGraph(commands, edges, traces)
        return self._apply_weight_constraints(graph)

    def _apply_weight_constraints(self, graph: CommandGraph) -> CommandGraph:
        consumer_ranks = self._dram_consumer_ranks(graph)
        graph = self._apply_sram_prefetch_constraints(graph, consumer_ranks)
        graph = self._apply_sa_weight_load_constraints(graph)
        return self._apply_dram_scheduling_priorities(graph, consumer_ranks)

    @staticmethod
    def _dram_consumer_ranks(graph: CommandGraph) -> Mapping[str, int]:
        """Rank DRAM traffic by its earliest core consumer in this PipeTree."""
        topological_order = graph.topological_order()
        topological_rank = {
            cmd_id: rank for rank, cmd_id in enumerate(topological_order)
        }
        background_rank = len(topological_order)
        result = {}
        for command in graph.commands:
            if not isinstance(command, DramCmd):
                continue
            pending = list(graph.successors(command.cmd_id))
            visited = set()
            consumers = []
            while pending:
                cmd_id = pending.pop()
                if cmd_id in visited:
                    continue
                visited.add(cmd_id)
                if cmd_id.endswith(".core"):
                    consumers.append(topological_rank[cmd_id])
                    continue
                pending.extend(graph.successors(cmd_id))
            result[command.cmd_id] = (
                min(consumers) if consumers
                else background_rank + topological_rank[command.cmd_id]
            )
        return result

    def _apply_sram_prefetch_constraints(
        self, graph: CommandGraph, consumer_ranks: Mapping[str, int],
    ) -> CommandGraph:
        """Order prefetches by demand and add SRAM-capacity backpressure."""
        graph_rank = {
            command.cmd_id: rank for rank, command in enumerate(graph.commands)
        }
        prefetches = [
            cmd for cmd in graph.commands if isinstance(cmd, WeightPrefetchCmd)
        ]
        if not prefetches:
            return graph
        prefetches.sort(key=lambda command: (
            consumer_ranks[command.cmd_id], graph_rank[command.cmd_id],
        ))

        sram_ids = {cmd.sram_resource_id for cmd in prefetches}
        if len(sram_ids) != 1:
            raise ValueError(
                "all weight prefetches must use one representative "
                "sram_resource_id"
            )

        capacity = (
            self.hardware_config.chip.logic_die.memory.sram_capacity_bytes
        )
        release_commands = {}
        for prefetch in prefetches:
            prefetch_trace = graph.traces[prefetch.cmd_id]
            same_weight = [
                cmd for cmd in graph.commands
                if cmd.op_id == prefetch.op_id
                and graph.traces[cmd.cmd_id].layer_index
                == prefetch_trace.layer_index
            ]
            loads = [
                cmd.cmd_id for cmd in same_weight
                if isinstance(cmd, WeightLoadCmd)
            ]
            consumers = loads or [
                cmd.cmd_id for cmd in same_weight
                if cmd.cmd_id.endswith(".core")
            ]
            if not consumers:
                raise ValueError(
                    f"weight prefetch has no consumer: {prefetch.cmd_id}"
                )
            release_commands[prefetch.cmd_id] = tuple(consumers)

        resident = deque()
        resident_bytes = 0
        capacity_edges = []
        for prefetch in prefetches:
            if prefetch.size_bytes > capacity:
                raise ValueError(
                    f"weight prefetch {prefetch.cmd_id} requires "
                    f"{prefetch.size_bytes} bytes but representative SRAM "
                    f"capacity is {capacity} bytes"
                )
            while resident_bytes + prefetch.size_bytes > capacity:
                evicted = resident.popleft()
                resident_bytes -= evicted.size_bytes
                capacity_edges.extend(
                    (consumer, prefetch.cmd_id)
                    for consumer in release_commands[evicted.cmd_id]
                )
            resident.append(prefetch)
            resident_bytes += prefetch.size_bytes

        # Preserve the consumer order even when an early prefetch is blocked by
        # SRAM capacity. Demand reads remain free to use the idle DRAM resource.
        prefetch_chains = {}
        for prefetch in prefetches:
            prefetch_chains.setdefault(prefetch.resource_id, []).append(
                prefetch.cmd_id
            )
        order_edges = [
            (previous, current)
            for chain in prefetch_chains.values()
            for previous, current in zip(chain, chain[1:])
        ]
        return CommandGraph(
            graph.commands,
            tuple(graph.edges) + tuple(capacity_edges) + tuple(order_edges),
            graph.traces,
            graph.scheduling_priorities,
        )

    @staticmethod
    def _apply_sa_weight_load_constraints(
        graph: CommandGraph,
    ) -> CommandGraph:
        """Allow a static weight load only one actual GEMM ahead per PU.

        The window is defined by the complete GEMM stream, including dynamic
        right-operand operations such as QK and SV.  Once every direct input
        of GEMM ``i`` has completed, GEMM ``i`` is ready on its PU and the
        static weight for GEMM ``i + 1`` may load concurrently on SRAM.
        """
        load_by_core = {}
        for command in graph.commands:
            if not isinstance(command, WeightLoadCmd):
                continue
            consumers = [
                graph.command(successor)
                for successor in graph.successors(command.cmd_id)
                if isinstance(graph.command(successor), GemmCmd)
            ]
            if len(consumers) != 1:
                raise ValueError(
                    f"weight load {command.cmd_id} must directly feed "
                    "exactly one GemmCmd"
                )
            core = consumers[0]
            if core.cmd_id in load_by_core:
                raise ValueError(
                    f"GEMM {core.cmd_id} has more than one WeightLoadCmd"
                )
            load_by_core[core.cmd_id] = command.cmd_id

        gemms_by_pu = {}
        for command in graph.commands:
            if isinstance(command, GemmCmd):
                gemms_by_pu.setdefault(command.resource_id, []).append(
                    command.cmd_id
                )

        buffer_edges = []
        for cores in gemms_by_pu.values():
            for index in range(1, len(cores)):
                load_id = load_by_core.get(cores[index])
                if load_id is None:
                    continue
                previous_core = cores[index - 1]
                buffer_edges.extend(
                    (source, load_id)
                    for source in graph.predecessors(previous_core)
                    if source != load_id
                )

        if not buffer_edges:
            return graph
        return CommandGraph(
            graph.commands,
            tuple(graph.edges) + tuple(buffer_edges),
            graph.traces,
            graph.scheduling_priorities,
        )

    @staticmethod
    def _apply_dram_scheduling_priorities(
        graph: CommandGraph, consumer_ranks: Mapping[str, int],
    ) -> CommandGraph:
        priorities = dict(graph.scheduling_priorities)
        priorities.update(consumer_ranks)
        return CommandGraph(
            graph.commands,
            graph.edges,
            graph.traces,
            priorities,
        )

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
            if not isinstance(op, (BMMOp, VectorOp, CommOp)):
                raise TypeError(f"unsupported operator: {op_id}")
            if op.op_id != op_id:
                raise ValueError(f"operator key does not match op_id: {op_id}")
            if not isinstance(mapping, OperatorMapping):
                raise TypeError(f"expected OperatorMapping for {op_id}")
            # Snapshot and revalidate mutable model operators without changing them.
            op = replace(op)
            axes = (("B", "M") if isinstance(op, BMMOp) else
                    ("m",) if isinstance(op, VectorOp) else ("size_bytes",))
            if mapping.batch_axis not in axes:
                raise ValueError(f"{op_id}: batch_axis must be one of {axes}")
            checked[op_id] = op

        instances = tuple(self.iter_instances(tree))
        by_op = {key: [] for key in tree.operator_order}
        for instance in instances:
            by_op[instance.op_id].append(instance)
        commands, edges, traces = [], [], {}
        prefetched = {}

        def emit(cmd, instance, once=False):
            commands.append(cmd)
            traces[cmd.cmd_id] = CommandTrace(
                None if once else instance.index,
                0 if once else instance.token_start,
                tree.batch_size if once else instance.token_stop,
                instance.node_path,
            )
            return cmd.cmd_id

        # Independent one-time prefetches are deliberately emitted first.
        for op_index, op_id in enumerate(tree.operator_order):
            op = checked[op_id]
            mapping = mappings[op_id]
            if mapping.dram_read_once_bytes:
                prefetched[op_id] = emit(WeightPrefetchCmd(
                    f"op{op_index}.prefetch", op_id, mapping.dram_resource_id,
                    mapping.dram_read_once_bytes, mapping.sram_resource_id,
                    self._prefetch_weight_shape(
                        op, mapping, mapping.dram_read_once_bytes,
                    ),
                ), by_op[op_id][0], once=True)

        core_ids = {}
        demand_sram_reads = {}
        for instance in instances:
            op_id, batch = instance.op_id, instance.batch_size
            op, mapping = checked[op_id], mappings[op_id]
            prefix = f"i{instance.index}"
            mapped_batch = self._mapped_batch_size(
                batch, mapping.batch_partition_degree,
            )
            readiness = []
            if op_id in prefetched:
                readiness.append(prefetched[op_id])
            if mapping.dram_read_bytes_per_token:
                readiness.append(emit(DramReadCmd(
                    prefix + ".read", op_id, mapping.dram_resource_id,
                    mapped_batch * mapping.dram_read_bytes_per_token,
                ), instance))
            load_bytes = (mapping.weight_load_fixed_bytes
                          + mapped_batch * mapping.weight_load_bytes_per_token)
            if load_bytes:
                load = emit(WeightLoadCmd(
                    prefix + ".load", op_id, mapping.sram_resource_id, load_bytes,
                    self._weight_shape(op, load_bytes),
                ), instance)
                edges.extend((source, load) for source in readiness)
                readiness = [load]
            if mapping.sram_read_bytes_per_mapped_token:
                sram_read = emit(SramReadCmd(
                    prefix + ".sram_read", op_id, mapping.sram_resource_id,
                    mapped_batch * mapping.sram_read_bytes_per_mapped_token,
                    mapping.sram_read_data_kind,
                ), instance)
                edges.extend((source, sram_read) for source in readiness)
                readiness = [sram_read]
                demand_sram_reads[instance.index] = sram_read

            core = self._core_command(
                prefix + ".core", op, mapping, batch, tree.batch_size,
            )
            core_ids[instance.index] = emit(core, instance)
            edges.extend((source, core.cmd_id) for source in readiness)
            if mapping.dram_write_bytes_per_token:
                write = emit(DramWriteCmd(
                    prefix + ".write", op_id, mapping.dram_resource_id,
                    mapped_batch * mapping.dram_write_bytes_per_token,
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
        for instance_index, sram_read in demand_sram_reads.items():
            instance = instances[instance_index]
            position = order_index[instance.op_id]
            broadcast_id = f"{instance.op_id}.input_broadcast"
            if position == 0 or tree.operator_order[position - 1] != broadcast_id:
                raise ValueError(
                    f"dynamic SRAM BMM is missing its input broadcast: "
                    f"{instance.op_id}"
                )
            if position < 2:
                continue
            sources = by_op[tree.operator_order[position - 2]]
            for source in sources:
                if max(source.token_start, instance.token_start) < min(
                    source.token_stop, instance.token_stop,
                ):
                    edges.append((core_ids[source.index], sram_read))

        return CommandGraph(commands, edges, traces)

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
    def _core_command(cmd_id, op, mapping, batch, full_batch):
        def scale(value):
            mapped_batch = TreeParser._mapped_batch_size(
                batch, mapping.batch_partition_degree,
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
            return GemmCmd(**common, **dimensions)
        if isinstance(op, VectorOp):
            return VectorCmd(**common, kind=op.kind, m=scale(op.m), n=op.n, params=op.params)
        cls = NoCCmd if op.scope == "intra_chip" else InterChipCmd
        matrix = (None if op.transfer_bytes is None else
                  tuple(tuple(scale(value) for value in row) for row in op.transfer_bytes))
        return cls(
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
