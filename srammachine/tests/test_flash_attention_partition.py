"""Attention B/S partitions, row-local state merging and IO lowering."""

import itertools
import unittest

from srammachine.commands import FlashAttentionCmd, NoCCmd
from srammachine.frontend.modules import BMMOp, FlashAttentionOp, PartitionChain
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG
from srammachine.pipetree.mapping import OperatorMapping
from srammachine.pipetree.parser import TreeParser
from srammachine.pipetree.tree import GroupNode, OpNode, PipeTree
from srammachine.simulator import GraphExecutor


class FlashAttentionPartitionTests(unittest.TestCase):
    def op(self, steps=(), **kwargs):
        return FlashAttentionOp("a", 5, 2, 7, 9, 9, 3,
                                partition_chain=PartitionChain(steps), **kwargs)

    def parse(self, op, split=1, layers=1, mapping=None):
        tree = PipeTree(op.B, (op.op_id,), GroupNode((OpNode(op.op_id),), split))
        return TreeParser().parse(tree, {op.op_id: op}, {
            op.op_id: mapping or OperatorMapping("B", "chip0.pu0"),
        }, layer_count=layers)

    def test_batch_rows_sequence_columns_merge_before_io(self):
        op = self.op((("B", "p", 2), ("S", "q", 4)))
        volumes = op.communication_volumes()
        self.assertEqual(volumes["input_bytes_per_row"], (84, 56))
        self.assertEqual(volumes["output_bytes_per_row"], (36, 24))
        self.assertEqual(volumes["complete_batches_per_row"], (3, 2))
        self.assertEqual(volumes["partial_batches_per_row"], (0, 0))
        graph = self.parse(op)
        core = graph.command("i0.core")
        self.assertIsInstance(core, FlashAttentionCmd)
        self.assertEqual((core.B, core.M, core.qk_K, core.qk_N, core.sv_K, core.sv_N),
                         (3, 2, 7, 3, 3, 3))
        output = graph.command("i0.output")
        self.assertEqual(output.size_bytes, 36)
        self.assertEqual(output.attention_complete_batches_per_row, (3, 2))
        self.assertIn(("i0.input", "i0.core"), graph.edges)
        self.assertIn(("i0.core", "i0.output"), graph.edges)

    def test_sequence_rows_send_states_after_column_merge(self):
        op = self.op((("S", "p", 2), ("S", "q", 3), ("B", "q", 2)))
        volumes = op.communication_volumes()
        self.assertEqual(volumes["input_bytes_per_row"], (140, 140))
        # Each row sends five states, not one state per sequence shard.
        self.assertEqual(volumes["output_bytes_per_row"], (200, 200))
        self.assertEqual(volumes["partial_batches_per_row"], (5, 5))
        output = self.parse(op).command("i0.output")
        self.assertEqual(output.attention_partial_batches_per_row, (5, 5))

    def test_projection_and_coverage_against_element_enumeration(self):
        for axes in itertools.product("BS", repeat=4):
            with self.subTest(axes=axes):
                chain = PartitionChain(tuple(zip(axes, ("p", "q", "p", "q"),
                                                  (2, 3, 2, 2))))
                shape = dict(B=3, S=5)
                covered = set()
                expected = []
                rows, columns = chain.grid_shape
                for p in range(rows):
                    sequences_by_batch = {}
                    for q in range(columns):
                        tile = chain.tile_ranges(shape, p, q)
                        elements = set(itertools.product(range(*tile["B"]),
                                                         range(*tile["S"])))
                        self.assertFalse(covered & elements)
                        covered.update(elements)
                        for b, s in elements:
                            sequences_by_batch.setdefault(b, set()).add(s)
                    complete = sum(len(s) == shape["S"] for s in sequences_by_batch.values())
                    expected.append((complete, len(sequences_by_batch) - complete))
                self.assertEqual(len(covered), shape["B"] * shape["S"])
                self.assertEqual(chain.attention_batches_per_row(shape), tuple(expected))

    def test_empty_sequence_rows_do_not_send_or_receive(self):
        op = FlashAttentionOp("a", 2, 1, 4, 1, 1, 3,
                              partition_chain=PartitionChain((("S", "p", 4),)))
        volumes = op.communication_volumes()
        self.assertEqual(volumes["input_bytes_per_row"], (16, 0, 0, 0))
        self.assertEqual(volumes["output_bytes_per_row"], (12, 0, 0, 0))
        self.assertEqual(volumes["partial_batches_per_row"], (0, 0, 0, 0))

    def test_routes_and_widths(self):
        for source, destination in itertools.product(("none", "io_die"), repeat=2):
            with self.subTest(source=source, destination=destination):
                op = self.op(input_source=source, output_destination=destination,
                             input_dtype_bytes=1, output_dtype_bytes=4)
                graph = self.parse(op)
                self.assertEqual(len(graph.commands),
                                 1 + (source == "io_die") + (destination == "io_die"))
                if source == "io_die":
                    self.assertEqual(graph.command("i0.input").size_bytes, 70)
                if destination == "io_die":
                    self.assertEqual(graph.command("i0.output").size_bytes, 120)
        op = self.op((("S", "p", 2),), state_dtype_bytes=8)
        self.assertEqual(op.communication_volumes()["output_bytes_per_row"], (400, 400))

    def test_microbatch_and_cross_layer_dependencies(self):
        op = FlashAttentionOp("a", 8, 2, 7, 9, 9, 3,
                              partition_chain=PartitionChain((("B", "p", 3),
                                                               ("S", "q", 2))))
        graph = self.parse(op, split=2, layers=2)
        self.assertEqual(graph.command("layer0.i0.core").B, 2)
        self.assertEqual(graph.command("layer0.i0.input").io_bytes_per_row, (56, 56, 0))
        self.assertEqual(graph.command("layer0.i0.output").io_bytes_per_row, (24, 24, 0))
        self.assertIn(("layer0.i0.output", "layer1.i0.input"), graph.edges)

    def test_operator_boundary_and_io_timing(self):
        ops = {name: FlashAttentionOp(name, 2, 1, 4, 8, 8, 3) for name in ("a", "b")}
        tree = PipeTree(2, tuple(ops), GroupNode(tuple(OpNode(name) for name in ops)))
        graph = TreeParser().parse(tree, ops, {
            name: OperatorMapping("B", "chip0.pu0") for name in ops
        }, layer_count=1)
        self.assertIn(("i0.output", "i1.input"), graph.edges)
        result = GraphExecutor.simulate(graph)
        self.assertGreater(result.total_time_ns, 0)
        for command in graph.commands:
            if isinstance(command, NoCCmd):
                rate = DEFAULT_HARDWARE_CONFIG.chip.noc.bandwidth_for(command.operation)
                execution = result.executions[command.cmd_id]
                self.assertEqual(execution.end_time_ns - execution.start_time_ns,
                                 (command.size_bytes * 10**9 + rate - 1) // rate)

    def test_dynamic_kv_read_without_legacy_broadcast_node(self):
        op = self.op()
        graph = self.parse(op, mapping=OperatorMapping(
            "B", "chip0.pu0", "chip0.dram",
            dram_read_bytes_per_token=32,
        ))
        self.assertGreater(GraphExecutor.simulate(graph).total_time_ns, 0)

    def test_validation(self):
        for kwargs in (dict(input_source="bad"), dict(output_destination="pu_vertical"),
                       dict(state_dtype_bytes=0), dict(input_dtype_bytes=True)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.op(**kwargs)
        for axis in ("M", "K", "N"):
            with self.subTest(axis=axis), self.assertRaises(ValueError):
                self.op(((axis, "p", 2),))
        with self.assertRaises(ValueError):
            FlashAttentionOp("a", 1, 1, 4, 7, 8, 3)
        with self.assertRaises(ValueError):
            BMMOp("a", 1, 1, 4, 3, partition_chain=PartitionChain((("S", "p", 2),)))


if __name__ == "__main__":
    unittest.main()
