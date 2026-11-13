"""Regression coverage for nested partitions and BMM communication lowering."""

import itertools
import unittest

from srammachine.commands import GemmCmd, NoCCmd
from srammachine.frontend.modules import BMMOp, PartitionChain, SplitStep
from srammachine.pipetree.mapping import OperatorMapping
from srammachine.pipetree.parser import TreeParser
from srammachine.pipetree.tree import GroupNode, OpNode, PipeTree
from srammachine.simulator import GraphExecutor, Simulator
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG
from srammachine.systolic_cost import systolic_gemm_duration_ns


class PartitionChainTests(unittest.TestCase):
    def test_example_mixed_radix_coordinates(self):
        chain = PartitionChain((("B", "p", 4), ("N", "p", 2),
                                ("K", "p", 4), ("N", "q", 64)))
        self.assertEqual(chain.grid_shape, (32, 64))
        shape = dict(B=8, M=3, K=16, N=256)
        self.assertEqual(chain.tile_ranges(shape, 23, 5),
                         dict(B=(4, 6), M=(0, 3), K=(12, 16), N=(138, 140)))
        chain.validate_grid(32, 64)
        with self.assertRaises(ValueError):
            chain.validate_grid(8, 8)

    def test_nested_ceiling_and_empty_tiles(self):
        chain = PartitionChain((("N", "p", 2), ("N", "q", 3)))
        shape = dict(B=1, M=1, K=1, N=5)
        self.assertEqual([chain.tile_ranges(shape, p, q)["N"]
                          for p in range(2) for q in range(3)],
                         [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 5)])
        self.assertEqual(chain.io_elements_per_row(shape, output=True), (3, 2))

    def test_projection_volumes_against_element_enumeration(self):
        shape = dict(B=3, M=3, K=5, N=7)
        for axes in itertools.product("BMKN", repeat=3):
            with self.subTest(axes=axes):
                chain = PartitionChain(((axes[0], "q", 2), (axes[1], "p", 3),
                                        (axes[2], "q", 2)))
                covered = set()
                expected_input, expected_output = [], []
                for p in range(3):
                    inputs, outputs = set(), set()
                    for q in range(4):
                        tile = chain.tile_ranges(shape, p, q)
                        elements = set(itertools.product(*(range(*tile[a]) for a in "BMKN")))
                        self.assertFalse(covered & elements)
                        covered.update(elements)
                        inputs.update((b, m, k) for b, m, k, n in elements)
                        outputs.update((b, m, n) for b, m, k, n in elements)
                    expected_input.append(len(inputs))
                    expected_output.append(len(outputs))
                self.assertEqual(len(covered), 3 * 3 * 5 * 7)
                self.assertEqual(chain.io_elements_per_row(shape, output=False), tuple(expected_input))
                self.assertEqual(chain.io_elements_per_row(shape, output=True), tuple(expected_output))

    def test_broadcast_scatter_reduce_gather(self):
        shape = dict(B=2, M=3, K=12, N=16)
        for axis, input_size, output_size in (("N", 72, 96), ("K", 72, 96),
                                               ("B", 72, 96), ("M", 72, 96)):
            with self.subTest(axis=axis):
                chain = PartitionChain(((axis, "q", 4),))
                self.assertEqual(chain.io_elements_per_row(shape, output=False), (input_size,))
                self.assertEqual(chain.io_elements_per_row(shape, output=True), (output_size,))

    def test_vertical_reduce_and_interleaved_groups(self):
        shape = dict(B=1, M=2, K=8, N=8)
        contiguous = PartitionChain((("N", "p", 2), ("K", "p", 4)))
        interleaved = PartitionChain((("K", "p", 4), ("N", "p", 2)))
        self.assertEqual(contiguous.vertical_reduce_elements(shape), 8)
        self.assertEqual(interleaved.vertical_reduce_elements(shape), 16)
        self.assertEqual(PartitionChain((("N", "p", 8),)).vertical_reduce_elements(shape), 0)
        self.assertEqual(PartitionChain((("K", "q", 8),)).vertical_reduce_elements(shape), 0)

    def test_invalid_inputs(self):
        for args in (("X", "p", 2), ("K", "x", 2), ("N", "q", True), ("B", "p", 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                SplitStep(*args)
        for kwargs in (dict(input_source="unknown"), dict(output_destination="unknown"),
                       dict(input_dtype_bytes=0), dict(shared_weight=1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BMMOp("a", 1, 1, 1, 1, **kwargs)


class BMMPartitionLoweringTests(unittest.TestCase):
    def parse(self, op, *, split=1, layers=1):
        tree = PipeTree(op.M, (op.op_id,), GroupNode((OpNode(op.op_id),), split))
        return TreeParser().parse(tree, {op.op_id: op},
                                 {op.op_id: OperatorMapping("M", "chip0.pu0")},
                                 layer_count=layers)

    def test_default_io_and_derived_compute_shape(self):
        op = BMMOp("a", 2, 4, 12, 16, partition_chain=PartitionChain(
            (("K", "p", 3), ("N", "q", 4))))
        graph = self.parse(op)
        commands = {cmd.cmd_id: cmd for cmd in graph.commands}
        self.assertEqual(len(commands), 3)
        core = commands["i0.core"]
        self.assertEqual((core.B, core.M, core.K, core.N), (2, 4, 4, 4))
        self.assertEqual(core.weight_load_bytes, 32)
        self.assertEqual(commands["i0.input"].io_bytes_per_row, (64, 64, 64))
        self.assertEqual(commands["i0.output"].io_bytes_per_row, (256, 256, 256))
        self.assertIn(("i0.input", "i0.core"), graph.edges)
        self.assertIn(("i0.core", "i0.output"), graph.edges)
        result = Simulator().run(graph)
        self.assertGreater(result.total_time_ns, 0)

    def test_optional_routes_and_vertical_resource(self):
        for route, count in (("none", 1), ("pu_vertical", 2)):
            with self.subTest(route=route):
                op = BMMOp("a", 1, 4, 8, 16, input_source="none",
                           output_destination=route,
                           partition_chain=PartitionChain((("K", "p", 4),)))
                graph = self.parse(op)
                self.assertEqual(len(graph.commands), count)
                if route == "pu_vertical":
                    output = next(c for c in graph.commands if isinstance(c, NoCCmd))
                    self.assertEqual(output.resource_ids, ("chip0.noc_vertical",))
                    self.assertEqual(output.size_bytes, 128)
                    self.assertEqual(output.kind, "reduce")

    def test_microbatch_and_layer_dependencies(self):
        op = BMMOp("a", 1, 8, 5, 7, partition_chain=PartitionChain(
            (("M", "p", 3), ("N", "q", 2))))
        graph = self.parse(op, split=2, layers=2)
        commands = {cmd.cmd_id: cmd for cmd in graph.commands}
        self.assertEqual(commands["layer0.i0.input"].io_bytes_per_row, (20, 20, 0))
        self.assertEqual(commands["layer0.i0.core"].M, 2)
        self.assertIn(("layer0.i0.output", "layer1.i0.input"), graph.edges)
        self.assertIn(("layer0.i1.output", "layer1.i1.input"), graph.edges)

    def test_operator_boundary_waits_for_output(self):
        ops = {name: BMMOp(name, 1, 2, 3, 4) for name in ("a", "b")}
        tree = PipeTree(2, tuple(ops), GroupNode(tuple(OpNode(name) for name in ops)))
        graph = TreeParser().parse(tree, ops, {
            name: OperatorMapping("M", "chip0.pu0") for name in ops
        }, layer_count=1)
        self.assertIn(("i0.output", "i1.input"), graph.edges)

    def test_compute_and_weight_timing(self):
        op = BMMOp("a", 2, 8, 513, 257, input_source="none",
                   output_destination="none", partition_chain=PartitionChain(
                       (("K", "p", 4), ("N", "q", 4))))
        graph = self.parse(op)
        core = graph.command("i0.core")
        hardware = DEFAULT_HARDWARE_CONFIG
        rate = hardware.chip.logic_die.processing_unit.weight_bandwidth_bytes_per_second
        weight_ns = (core.weight_load_bytes * 1_000_000_000 + rate - 1) // rate
        result = GraphExecutor.simulate(graph)
        self.assertEqual(result.total_time_ns, max(
            weight_ns, systolic_gemm_duration_ns(core, hardware),
        ))

    def test_io_timing_uses_parallel_rows(self):
        op = BMMOp("a", 1, 8, 1024, 64, partition_chain=PartitionChain(
            (("K", "p", 4), ("N", "q", 4))))
        graph = self.parse(op)
        command = graph.command("i0.input")
        result = GraphExecutor.simulate(graph)
        execution = result.executions[command.cmd_id]
        rate = DEFAULT_HARDWARE_CONFIG.chip.noc.bandwidth_for("FromIOdie")
        expected = (command.size_bytes * 1_000_000_000 + rate - 1) // rate
        self.assertEqual(execution.end_time_ns - execution.start_time_ns, expected)

    def test_shared_weights(self):
        graph = self.parse(BMMOp("a", 3, 2, 5, 7, shared_weight=True))
        core = next(c for c in graph.commands if isinstance(c, GemmCmd))
        self.assertEqual(core.weight_load_bytes, 35)


if __name__ == "__main__":
    unittest.main()
