"""Lower a PipeTree and explicit operator mappings into a command DAG.

V1 assumes a token-preserving linear operator flow. It does not infer residual
branches, expert routing, cross-request dependencies or physical memory layout.
"""
from copy import deepcopy
from dataclasses import replace
import math
from typing import Iterator, Mapping, Tuple

from srammachine.commands import (
    CommandGraph, CommandTrace, DramReadCmd, DramWriteCmd, PUMemoryReadCmd,
    FlashAttentionCmd, FlashMSACmd, MiniMaxFusedIndexerScoreCmd,
    FusedIndexerScoreCmd, GemmCmd,
    VectorCmd, NoCCmd, InterChipCmd,
)
from srammachine.commands.base import integer
from srammachine.frontend.modules import (
    BMMOp, BMMOutputDestination, FabricOp, FlashAttentionOp, FusedIndexerScoreOp, NoCOp, Operator, VectorOp, WeightMemory, WeightSource,
)
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG, HardwareConfig
from srammachine.frontend.modules import FlashMSAOp, MiniMaxFusedIndexerScoreOp
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
        Each layer has its own command IDs but shares hardware
        resource IDs. Pass layer_count=1 to inspect the original single layer.
        """
        integer("layer_count", layer_count, 1)
        single = self._parse_one_layer(tree, operators, mappings)
        if layer_count == 1:
            return self._link_kv_prefetch(single)

        # 原设计流程图要求复制单层 CommandGraph，模拟连续四层。
        # 保留原 op_id 以查询模型算子，使用 layer_index 区分不同层。
        instances = tuple(self.iter_instances(tree))
        first = [item for item in instances if item.op_id == tree.operator_order[0]]
        last = [item for item in instances if item.op_id == tree.operator_order[-1]]
        boundary_edges = []
        command_ids = {cmd.cmd_id for cmd in single.commands}
        i = j = 0
        while i < len(last) and j < len(first):
            source, target = last[i], first[j]
            if max(source.token_start, target.token_start) < min(source.token_stop, target.token_stop):
                source_id = f"i{source.index}.output"
                target_id = f"i{target.index}.input"
                boundary_edges.append((
                    source_id if source_id in command_ids else f"i{source.index}.core",
                    target_id if target_id in command_ids else f"i{target.index}.core",
                ))
            if source.token_stop <= target.token_stop:
                i += 1
            if target.token_stop <= source.token_stop:
                j += 1

        commands, edges, traces = [], [], {}
        start_edges = []
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
            start_edges.extend((prefix + source, prefix + target)
                               for source, target in single.start_edges)
            if layer:
                # 仅连接前层输出到后层对应 token 的计算，不能把所有图根节点
                # 都串起来；后层 DRAM 读取仍允许提前执行。
                # KV 写回也不构成整层屏障，竞争由共享硬件资源处理。
                previous = f"layer{layer - 1}."
                edges.extend((previous + source, prefix + target)
                             for source, target in boundary_edges)
        graph = CommandGraph(commands, edges, start_edges=start_edges, traces=traces)
        return self._link_kv_prefetch(graph)

    @staticmethod
    def _link_kv_prefetch(graph: CommandGraph) -> CommandGraph:
        """Fork XFold KV reads beside FC2 and join next-layer Attention outputs.

        Token-range overlap pairs independently partitioned producer/consumer
        instances. Layer zero uses the mapper's initial DRAM cache assumption;
        the last layer's look-ahead read has no consumer inside this graph.
        """
        reads = [command for command in graph.commands
                 if isinstance(command, PUMemoryReadCmd)
                 and command.data_kind == "next_layer_kv"]
        if not reads:
            return graph
        cores = {}
        attention = {}
        for command in graph.commands:
            trace = graph.traces[command.cmd_id]
            if command.cmd_id.endswith(".core"):
                cores[trace.layer_index, trace.instance_index] = command.cmd_id
            if isinstance(command, FlashAttentionCmd):
                attention.setdefault(trace.layer_index, []).append(command.cmd_id)
        edges, start_edges = list(graph.edges), list(graph.start_edges)
        for read in reads:
            source = graph.traces[read.cmd_id]
            fc2 = cores[source.layer_index, source.instance_index]
            edges.extend((parent, read.cmd_id) for parent in graph.predecessors(fc2))
            start_edges.extend((parent, read.cmd_id)
                               for parent in graph.start_predecessors(fc2))
            for target_id in attention.get(source.layer_index + 1, ()):
                target = graph.traces[target_id]
                if max(source.token_start, target.token_start) < min(
                    source.token_stop, target.token_stop,
                ):
                    edges.extend((read.cmd_id, successor)
                                 for successor in graph.successors(target_id))
        return CommandGraph(
            graph.commands, edges, start_edges=start_edges, traces=graph.traces,
            scheduling_priorities=graph.scheduling_priorities,
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
            if mapping.ep_routing is not None:
                routing = mapping.ep_routing
                if (not isinstance(op, FabricOp) or op.kind != "alltoall"
                        or op.group != tuple(range(routing.chip_count))
                        or op.transfer_bytes != routing.transfer_bytes(0, tree.batch_size, tree.batch_size)):
                    raise ValueError("EP routing must match the full-batch FabricOp")
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
        start_edges = []
        def emit(cmd, instance):
            commands.append(cmd)
            traces[cmd.cmd_id] = CommandTrace(
                instance.index,
                instance.token_start,
                instance.token_stop,
                instance.node_path,
            )
            return cmd.cmd_id

        core_ids = {}
        entry_ids = {}
        exit_ids = {}
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
            memory_bytes = 0
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
            core = self._core_command(
                prefix + ".core", op, mapping, instance, tree.batch_size,
                memory_bytes=memory_bytes,
            )
            streams = []
            if op.weight_memory is not None:
                if memory_bytes or mapping.weight_read_fixed_bytes:
                    raise ValueError("explicit weight_memory cannot mix with legacy memory mappings")
                streams = self._memory_streams(op, core, mapping, instance)
            elif mapping.weight_read_fixed_bytes:
                if not isinstance(op, BMMOp):
                    raise ValueError("weight_read_fixed_bytes requires a BMM operator")
                pu = (mapping.hardware_config or self.hardware_config).chip.logic_die.processing_unit
                streams = [("dram", mapping.weight_read_fixed_bytes,
                            pu.dram_bandwidth_bytes_per_second, "weight")]
            elif isinstance(op, BMMOp) and op.auto_weight_load and not memory_bytes:
                streams = self._memory_streams(op, core, mapping, instance)
            if streams:
                port_durations = {}
                for port, size, rate, kind in streams:
                    if kind == "next_layer_kv":
                        continue  # Independent XFold read, not FC2 execution time.
                    port_durations[port] = (
                        port_durations.get(port, 0) + math.ceil(size * 1e9 / rate)
                    )
                duration = max(port_durations.values(), default=0)
                core = replace(core, memory_duration_ns=duration)
            core_ids[instance.index] = emit(core, instance)
            entry_ids[instance.index] = core.cmd_id
            exit_ids[instance.index] = core.cmd_id
            if isinstance(op, (BMMOp, FlashAttentionOp)):
                # Scale the logical invocation before partitioning onto PUs.
                logical = self._core_command(
                    prefix + ".logical", op, mapping, instance, tree.batch_size,
                    partition=False,
                )
                logical_shape = ({"B": logical.B, "S": logical.qk_N, "M": logical.M}
                                 if isinstance(op, FlashAttentionOp) else {
                    axis: getattr(logical, axis) for axis in ("B", "M", "K", "N")
                })
                if isinstance(op, FlashMSAOp):
                    logical_shape["H"] = logical.H
                volumes = op.communication_volumes(logical_shape)
                chip_id = mapping.resource_id.split(".")[0]
                for suffix, operation, rows, size, kind in (
                    ("input", "FromIOdie", volumes["input_bytes_per_row"],
                     max(volumes["input_bytes_per_row"], default=0), None),
                    ("output", "PUVertical" if op.output_destination ==
                     BMMOutputDestination.PU_VERTICAL else "ToIOdie",
                     volumes["output_bytes_per_row"],
                     volumes["vertical_reduce_bytes"] or
                     max(volumes["output_bytes_per_row"], default=0),
                     "reduce" if op.output_destination ==
                     BMMOutputDestination.PU_VERTICAL else None),
                ):
                    if not size and not ((streams or memory_bytes) and suffix == "output"):
                        continue
                    direction = "vertical" if operation == "PUVertical" else "horizontal"
                    comm_id = emit(NoCCmd(
                        prefix + "." + suffix, op_id, f"{chip_id}.noc_{direction}",
                        operation, size, kind, io_bytes_per_row=rows,
                        attention_complete_batches_per_row=(
                            volumes["complete_batches_per_row"]
                            if isinstance(op, FlashAttentionOp) and suffix == "output" else ()
                        ),
                        attention_partial_batches_per_row=(
                            volumes["partial_batches_per_row"]
                            if isinstance(op, FlashAttentionOp) and suffix == "output" else ()
                        ),
                        top_k=(op.top_k if isinstance(op, (FusedIndexerScoreOp, MiniMaxFusedIndexerScoreOp))
                               and suffix == "output" and rows else None),
                        topk_candidates_per_row=(
                            volumes["topk_candidates_per_row"]
                            if isinstance(op, (FusedIndexerScoreOp, MiniMaxFusedIndexerScoreOp)) and suffix == "output" and rows else ()
                        ),
                        topk_selection_unit=("block" if isinstance(op, MiniMaxFusedIndexerScoreOp) else "token"),
                        attention_count_unit=("request_kv_head" if isinstance(op, FlashMSAOp) else "request"),
                    ), instance)
                    if suffix == "input":
                        entry_ids[instance.index] = comm_id
                        edges.append((comm_id, core.cmd_id))
                    else:
                        exit_ids[instance.index] = comm_id
                        edges.append((core.cmd_id, comm_id))
            if instance.index in dynamic_dram_reads:
                edges.append((dynamic_dram_reads[instance.index], exit_ids[instance.index]))
            for stream_index, (port, size, rate, kind) in enumerate(streams):
                read_id = emit(PUMemoryReadCmd(
                    f"{prefix}.{kind}.{port}.{stream_index}", op_id,
                    f"{mapping.resource_id}.{port}", size, port, rate, kind,
                ), instance)
                if kind == "next_layer_kv":
                    # Wired after layer expansion so this read shares FC2's
                    # predecessors and next-layer Attention's successors.
                    continue
                start_edges.append((core.cmd_id, read_id))
                # Streaming overlaps arithmetic. Completion of the Op (and
                # layer boundary) waits for its own operand streams.
                edges.append((read_id, exit_ids[instance.index]))
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
                edges.append((exit_ids[instance.index], write))

        # Each operator's intervals are ordered and partition the full batch.
        # Merge adjacent partitions in linear time, connecting every overlap.
        for previous, current in zip(tree.operator_order, tree.operator_order[1:]):
            sources, targets = by_op[previous], by_op[current]
            i = j = 0
            while i < len(sources) and j < len(targets):
                source, target = sources[i], targets[j]
                if max(source.token_start, target.token_start) < min(source.token_stop, target.token_stop):
                    edges.append((exit_ids[source.index], entry_ids[target.index]))
                if source.token_stop <= target.token_stop:
                    i += 1
                if target.token_stop <= source.token_stop:
                    j += 1

        # A dynamic right operand is a demand read, not a look-ahead weight
        # load.  Open it at the same BMM-block boundary as the explicit input
        # broadcast so the two resources can run in parallel. Operator output
        # waits for both the memory transfer and the compute command.
        order_index = {
            op_id: index for index, op_id in enumerate(tree.operator_order)
        }
        for instance_index, dram_read in dynamic_dram_reads.items():
            instance = instances[instance_index]
            position = order_index[instance.op_id]
            broadcast_id = f"{instance.op_id}.input_broadcast"
            if isinstance(checked[instance.op_id], FlashAttentionOp) or (
                isinstance(checked[instance.op_id], BMMOp) and (
                    entry_ids[instance_index] != core_ids[instance_index]
                    or checked[instance.op_id].auto_weight_load
                )
            ):
                if position:
                    for source in by_op[tree.operator_order[position - 1]]:
                        if max(source.token_start, instance.token_start) < min(
                            source.token_stop, instance.token_stop,
                        ):
                            edges.append((exit_ids[source.index], dram_read))
                continue
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

        return CommandGraph(commands, edges, start_edges=start_edges, traces=traces)

    def _memory_streams(self, op, core, mapping, instance):
        """Lower weights/cache for the representative active PU tile."""
        pu = (mapping.hardware_config or self.hardware_config).chip.logic_die.processing_unit
        if isinstance(op, FlashAttentionOp):
            requests = self._mapped_batch_size(core.B, mapping.cache_query_group_size)
            size = requests * core.qk_N * core.qk_K * op.kv_cache_dtype_bytes
            if isinstance(op, FlashMSAOp):
                size = requests * core.H * core.qk_N * (core.qk_K + core.sv_N) * op.kv_cache_dtype_bytes
            kind = "kv_cache"
        elif isinstance(op, BMMOp):
            size = (1 if op.shared_weight else core.B) * core.K * core.N * op.weight_dtype_bytes
            if isinstance(op, (FusedIndexerScoreOp, MiniMaxFusedIndexerScoreOp)):
                requests = self._mapped_batch_size(core.B, mapping.cache_query_group_size)
                size = requests * core.K * core.N * op.weight_dtype_bytes
            kind = "indexer_cache" if isinstance(op, (FusedIndexerScoreOp, MiniMaxFusedIndexerScoreOp)) else "weight"
        else:
            raise TypeError("explicit memory policy requires BMM or FlashAttention")
        if mapping.cache_read_bytes_per_request is not None:
            requests = self._mapped_batch_size(
                instance.batch_size,
                mapping.demand_batch_partition_degree or mapping.batch_partition_degree,
            )
            size = math.ceil(requests * mapping.cache_read_bytes_per_request)
        policy = op.weight_memory or WeightMemory(WeightSource.DRAM)
        streams = [(*transfer, kind) for transfer in policy.transfers(
            size, pu.dram_bandwidth_bytes_per_second,
            pu.flash_bandwidth_bytes_per_second,
        )]
        if op.flash_prefetch_bytes_per_request:
            # The prefetch is independent of FC2's expert/K/N partition. All
            # chip-local requests' selected KV entries are striped over its grid.
            count = math.prod(op.partition_chain.grid_shape)
            requests = self._mapped_batch_size(
                instance.batch_size, mapping.demand_batch_partition_degree or 1,
            )
            size = requests * op.flash_prefetch_bytes_per_request / count
            rate = (op.weight_memory.flash_bandwidth_bytes_per_second
                    or pu.flash_bandwidth_bytes_per_second)
            streams.append(("flash", size, rate, "next_layer_kv"))
        return streams

    @staticmethod
    def _mapped_batch_size(batch: int, partition_degree: int) -> int:
        return (batch + partition_degree - 1) // partition_degree

    @staticmethod
    def _core_command(cmd_id, op, mapping, instance, full_batch, *, memory_bytes=0, partition=True):
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
            if partition:
                if isinstance(op, (FusedIndexerScoreOp, MiniMaxFusedIndexerScoreOp)):
                    local = op.partition_chain.compute_shape(
                        {"B": dimensions["B"], "S": dimensions["N"]},
                    )
                    dimensions.update(B=local["B"], N=local["S"])
                else:
                    dimensions = op.partition_chain.compute_shape(dimensions)
            command_type = (
                FusedIndexerScoreCmd
                if isinstance(op, FusedIndexerScoreOp) else GemmCmd
            )
            if isinstance(op, FusedIndexerScoreOp):
                dimensions["top_k"] = op.top_k
                dimensions["queries_per_request"] = op.queries_per_request
                dimensions["main_gemm_flops_override"] = (
                    None
                    if op.main_gemm_flops_override is None
                    else scale(op.main_gemm_flops_override)
                )
            elif isinstance(op, MiniMaxFusedIndexerScoreOp):
                command_type = MiniMaxFusedIndexerScoreCmd
                dimensions.update(top_k=op.top_k, block_size=op.block_size,
                                  queries_per_request=op.queries_per_request)
            return command_type(**common, memory_bytes=memory_bytes, **dimensions)
        if isinstance(op, FlashAttentionOp):
            dimensions = {
                name: getattr(op, name)
                for name in ("B", "M", "qk_K", "qk_N", "sv_K", "sv_N")
            }
            dimensions[mapping.batch_axis] = scale(
                dimensions[mapping.batch_axis]
            )
            if isinstance(op, FlashMSAOp):
                dimensions["H"] = op.H
            if partition:
                shape = {"B": dimensions["B"], "S": dimensions["qk_N"]}
                if isinstance(op, FlashMSAOp):
                    shape["H"] = op.H
                local = op.partition_chain.compute_shape(shape)
                dimensions.update(B=local["B"], qk_N=local["S"], sv_K=local["S"])
                if isinstance(op, FlashMSAOp):
                    dimensions["H"] = local["H"]
            command_type = FlashMSACmd if isinstance(op, FlashMSAOp) else FlashAttentionCmd
            return command_type(**common, memory_bytes=memory_bytes, **dimensions)
        if isinstance(op, VectorOp):
            return VectorCmd(**common, memory_bytes=memory_bytes, kind=op.kind, m=scale(op.m), n=op.n, params=op.params)
        if isinstance(op, NoCOp):
            return NoCCmd(
                **common, operation=op.operation, size_bytes=scale(op.size_bytes),
                kind=op.kind, group=op.group,
                additional_resource_ids=mapping.additional_resource_ids,
            )
        if mapping.ep_routing is not None:
            matrix = mapping.ep_routing.transfer_bytes(
                instance.token_start, instance.token_stop, full_batch)
        else:
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
