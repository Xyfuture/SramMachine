"""Indexer B/S partitioning and per-batch top-k communication truncation."""

import itertools
import unittest

from srammachine.commands import FusedIndexerScoreCmd
from srammachine.frontend.modules import FusedIndexerScoreOp, PartitionChain
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG
from srammachine.pipetree.mapping import OperatorMapping
from srammachine.pipetree.parser import TreeParser
from srammachine.pipetree.tree import GroupNode, OpNode, PipeTree
from srammachine.simulator import GraphExecutor
from srammachine.simulator.results import _command_parameters


class IndexerPartitionTests(unittest.TestCase):
    def op(self, steps=(), **kwargs):
        return FusedIndexerScoreOp("a", 5, 3, 7, 9,
                                  partition_chain=PartitionChain(steps), **kwargs)

    def parse(self, op, split=1, layers=1):
        tree = PipeTree(op.B, (op.op_id,), GroupNode((OpNode(op.op_id),), split))
        return TreeParser().parse(tree, {op.op_id: op}, {
            op.op_id: OperatorMapping("B", "chip0.pu0"),
        }, layer_count=layers)

    def test_batch_scatter_sequence_broadcast_and_row_topk(self):
        op = self.op((("B", "p", 2), ("S", "q", 4)), top_k=2)
        volume = op.communication_volumes()
        self.assertEqual(volume["input_bytes_per_row"], (126, 84))
        self.assertEqual(volume["topk_candidates_per_row"], (6, 4))
        self.assertEqual(volume["output_bytes_per_row"], (36, 24))
        graph = self.parse(op)
        core = graph.command("i0.core")
        self.assertIsInstance(core, FusedIndexerScoreCmd)
        self.assertEqual((core.B, core.M, core.K, core.N), (3, 3, 7, 3))
        self.assertEqual(core.top_k, 2)
        self.assertEqual(core.output_candidates, 6)
        self.assertEqual(core.weight_load_bytes, 3 * 7 * 3)
        self.assertEqual(core.qk_flops, 2 * 3 * 3 * 7 * 3)
        output = graph.command("i0.output")
        self.assertEqual(output.topk_candidates_per_row, (6, 4))
        self.assertEqual(output.size_bytes, 36)
        self.assertIn(("i0.input", "i0.core"), graph.edges)
        self.assertIn(("i0.core", "i0.output"), graph.edges)
        params = _command_parameters(output, DEFAULT_HARDWARE_CONFIG)
        self.assertEqual(params["output_reduction"], "per_batch_topk_merge")

    def test_merge_can_grow_to_k_but_never_above_k(self):
        # Each PU has only 3 positions; the row has 9 and emits 5, not 3 or 9.
        op = self.op((("S", "q", 3),), top_k=5)
        graph = self.parse(op)
        self.assertEqual(graph.command("i0.core").output_candidates, 5 * 3)
        self.assertEqual(op.communication_volumes()["topk_candidates_per_row"], (5 * 5,))
        self.assertEqual(graph.command("i0.output").size_bytes, 5 * 5 * 6)

    def test_sequence_rows_are_capped_independently(self):
        op = self.op((("S", "p", 2), ("S", "q", 3)), top_k=5)
        volume = op.communication_volumes()
        self.assertEqual(volume["input_bytes_per_row"], (210, 210))
        self.assertEqual(volume["topk_candidates_per_row"], (25, 20))
        self.assertEqual(volume["output_bytes_per_row"], (150, 120))

    def test_short_sequence_and_empty_tiles(self):
        op = FusedIndexerScoreOp("a", 2, 3, 7, 1, top_k=10,
                                 partition_chain=PartitionChain((("S", "p", 4),)))
        volume = op.communication_volumes()
        self.assertEqual(volume["input_bytes_per_row"], (84, 0, 0, 0))
        self.assertEqual(volume["topk_candidates_per_row"], (2, 0, 0, 0))
        self.assertEqual(volume["output_bytes_per_row"], (12, 0, 0, 0))

    def test_nested_partitions_against_candidate_merge_oracle(self):
        # Scores contain ties. Global index breaks ties consistently across PUs.
        def topk(items, k):
            return sorted(items, key=lambda item: (-item[0], item[1]))[:k]

        for axes, k in itertools.product(itertools.product("BS", repeat=4), (1, 3, 9)):
            with self.subTest(axes=axes, k=k):
                chain = PartitionChain(tuple(zip(axes, ("q", "p", "q", "p"),
                                                  (2, 2, 3, 2))))
                op = FusedIndexerScoreOp("a", 3, 2, 4, 7,
                                         partition_chain=chain, top_k=k)
                expected_candidates, expected_input = [], []
                global_candidates = {b: [] for b in range(op.B)}
                rows, columns = chain.grid_shape
                for p in range(rows):
                    candidates = {}
                    for q in range(columns):
                        tile = chain.tile_ranges(dict(B=op.B, S=op.N), p, q)
                        for b in range(*tile["B"]):
                            local = topk([((s * 3 + b) % 5, s)
                                          for s in range(*tile["S"])], k)
                            if local:
                                candidates.setdefault(b, []).extend(local)
                    merged = {b: topk(items, k) for b, items in candidates.items()}
                    expected_candidates.append(sum(map(len, merged.values())))
                    expected_input.append(len(merged) * op.M * op.K * 2)
                    for b, items in merged.items():
                        global_candidates[b].extend(items)
                volumes = op.communication_volumes()
                self.assertEqual(volumes["topk_candidates_per_row"], tuple(expected_candidates))
                self.assertEqual(volumes["output_bytes_per_row"],
                                 tuple(count * 6 for count in expected_candidates))
                self.assertEqual(volumes["input_bytes_per_row"], tuple(expected_input))
                for b, candidates in global_candidates.items():
                    self.assertEqual(topk(candidates, k),
                                     topk([((s * 3 + b) % 5, s) for s in range(op.N)], k))

    def test_none_routes_and_candidate_widths(self):
        for source, destination in itertools.product(("none", "io_die"), repeat=2):
            with self.subTest(source=source, destination=destination):
                op = self.op(top_k=2, input_source=source, output_destination=destination,
                             input_dtype_bytes=1, output_dtype_bytes=4, index_dtype_bytes=2)
                graph = self.parse(op)
                self.assertEqual(len(graph.commands),
                                 1 + (source == "io_die") + (destination == "io_die"))
                if source == "io_die":
                    self.assertEqual(graph.command("i0.input").size_bytes, 105)
                if destination == "io_die":
                    self.assertEqual(graph.command("i0.output").size_bytes, 60)

    def test_microbatch_and_layer_dependencies(self):
        op = FusedIndexerScoreOp("a", 8, 3, 7, 9, top_k=2,
                                 partition_chain=PartitionChain((("B", "p", 3),
                                                                  ("S", "q", 2))))
        graph = self.parse(op, split=2, layers=2)
        self.assertEqual(graph.command("layer0.i0.core").B, 2)
        self.assertEqual(graph.command("layer0.i0.core").top_k, 2)
        self.assertEqual(graph.command("layer0.i0.output").io_bytes_per_row, (24, 24, 0))
        self.assertIn(("layer0.i0.output", "layer1.i0.input"), graph.edges)
        result = GraphExecutor.simulate(graph)
        output = graph.command("layer0.i0.output")
        execution = result.executions[output.cmd_id]
        bandwidth = DEFAULT_HARDWARE_CONFIG.chip.noc.bandwidth_for("ToIOdie")
        self.assertEqual(execution.end_time_ns - execution.start_time_ns,
                         (24 * 10**9 + bandwidth - 1) // bandwidth)

    def test_validation(self):
        for kwargs in (dict(top_k=0), dict(top_k=True), dict(index_dtype_bytes=0),
                       dict(output_destination="pu_vertical")):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.op(**kwargs)
        for axis in ("M", "K", "N"):
            with self.subTest(axis=axis), self.assertRaises(ValueError):
                self.op(((axis, "p", 2),))
        with self.assertRaises(ValueError):
            self.op((("S", "q", 2),), main_gemm_flops_override=100)
        with self.assertRaises(ValueError):
            FusedIndexerScoreOp("a", 1, 1, 1, 257, index_dtype_bytes=1)
        self.assertEqual(self.parse(self.op(main_gemm_flops_override=100))
                         .command("i0.core").qk_flops, 100)


if __name__ == "__main__":
    unittest.main()
