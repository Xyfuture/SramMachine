"""Lower a PipeTree and explicit operator mappings into a command DAG.

V1 assumes a token-preserving linear operator flow. It does not infer residual
branches, expert routing, cross-request dependencies or physical memory layout.
SRAM is assumed large enough to retain each operator's one-time prefetch.
"""
from dataclasses import replace
from typing import Iterator, Mapping, Tuple

from srammachine.commands import (
    CommandGraph, CommandTrace, DramReadCmd, DramWriteCmd, WeightLoadCmd,
    GemmCmd, VectorCmd, NoCCmd, InterChipCmd,
)
from srammachine.frontend.modules import BMMOp, VectorOp, CommOp, Operator
from .mapping import OperatorMapping
from .tree import GroupNode, Node, OpInstance, OpNode, PipeTree


class TreeParser:
    """Stateless tree expansion and command lowering.

    parse requires full-batch operators and per-operator mapping policies.
    iter_instances remains available for inspecting only the tree expansion.
    """

    def parse(
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
            mapping = mappings[op_id]
            if mapping.dram_read_once_bytes:
                prefetched[op_id] = emit(DramReadCmd(
                    f"op{op_index}.prefetch", op_id, mapping.dram_resource_id,
                    mapping.dram_read_once_bytes,
                ), by_op[op_id][0], once=True)

        core_ids = {}
        for instance in instances:
            op_id, batch = instance.op_id, instance.batch_size
            op, mapping = checked[op_id], mappings[op_id]
            prefix = f"i{instance.index}"
            readiness = []
            if op_id in prefetched:
                readiness.append(prefetched[op_id])
            if mapping.dram_read_bytes_per_token:
                readiness.append(emit(DramReadCmd(
                    prefix + ".read", op_id, mapping.dram_resource_id,
                    batch * mapping.dram_read_bytes_per_token,
                ), instance))
            load_bytes = (mapping.weight_load_fixed_bytes
                          + batch * mapping.weight_load_bytes_per_token)
            if load_bytes:
                load = emit(WeightLoadCmd(
                    prefix + ".load", op_id, mapping.sram_resource_id, load_bytes,
                ), instance)
                edges.extend((source, load) for source in readiness)
                readiness = [load]

            core = self._core_command(
                prefix + ".core", op, mapping, batch, tree.batch_size,
            )
            core_ids[instance.index] = emit(core, instance)
            edges.extend((source, core.cmd_id) for source in readiness)
            if mapping.dram_write_bytes_per_token:
                write = emit(DramWriteCmd(
                    prefix + ".write", op_id, mapping.dram_resource_id,
                    batch * mapping.dram_write_bytes_per_token,
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

        return CommandGraph(commands, edges, traces)

    @staticmethod
    def _core_command(cmd_id, op, mapping, batch, full_batch):
        def scale(value):
            quotient, remainder = divmod(value * batch, full_batch)
            if remainder:
                raise ValueError(
                    f"{op.op_id}: {mapping.batch_axis} value {value} cannot be "
                    f"scaled exactly from batch {full_batch} to {batch}"
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
