"""Cross-layer XFold prefetch dependencies, stalls, and port contention."""

import math
import unittest

from srammachine.commands import FlashAttentionCmd, PUMemoryReadCmd
from srammachine.frontend.modules import BMMOp, FlashAttentionOp, WeightMemory, WeightSource
from srammachine.pipetree import GroupNode, OpNode, OperatorMapping, PipeTree, TreeParser
from srammachine.simulator import Simulator


class XFoldPrefetchTests(unittest.TestCase):
    def make_graph(self, *, batch=10, attention_split=1, fc2_split=1,
                   prefetch=True, flash_weights=False, prefetch_bandwidth=1e9):
        """Use short DRAM kernels and a deliberately long XFold KV read."""
        dram = WeightMemory(WeightSource.DRAM)
        operators = {
            "attention": FlashAttentionOp(
                "attention", batch, 1, 1, 1, 1, 1, weight_memory=dram,
            ),
            "between": BMMOp("between", 1, batch, 1, 1, weight_memory=dram),
            "fc2": BMMOp(
                "fc2", 1, batch, 1, 1,
                weight_memory=WeightMemory(WeightSource.DRAM, prefetch_bandwidth),
                flash_prefetch_bytes_per_request=1024 if prefetch else 0,
            ),
            "after": BMMOp("after", 1, batch, 1, 1, weight_memory=(
                WeightMemory(WeightSource.FLASH) if flash_weights else dram
            )),
        }
        tree = PipeTree(batch, tuple(operators), GroupNode((
            GroupNode((OpNode("attention"),), split=attention_split),
            OpNode("between"),
            GroupNode((OpNode("fc2"),), split=fc2_split),
            OpNode("after"),
        )))
        mappings = {name: OperatorMapping(
            "B" if name == "attention" else "M", "chip0.die0.pu0",
        ) for name in operators}
        return TreeParser().parse(tree, operators, mappings, layer_count=3)

    def test_slow_read_allows_fc2_and_next_attention_to_run_then_gates_output(self):
        graph = self.make_graph()
        result = Simulator().run(graph)
        times = {item.cmd_id: item for item in result.command_results}
        read = next(c for c in graph.commands if isinstance(c, PUMemoryReadCmd)
                    and c.data_kind == "next_layer_kv" and graph.traces[c.cmd_id].layer_index == 0)
        source = graph.traces[read.cmd_id]
        fc2 = f"layer0.i{source.instance_index}.core"
        attention = next(c for c in graph.commands if isinstance(c, FlashAttentionCmd)
                         and graph.traces[c.cmd_id].layer_index == 1)
        self.assertEqual(set(graph.predecessors(read.cmd_id)), set(graph.predecessors(fc2)))
        self.assertEqual(set(graph.successors(read.cmd_id)), set(graph.successors(attention.cmd_id)))
        self.assertEqual(graph.command(fc2).memory_duration_ns, 1)
        self.assertGreater(times[read.cmd_id].duration_ns, times[fc2].duration_ns)
        self.assertLess(times[fc2].end_time_ns, times[read.cmd_id].end_time_ns)
        self.assertLess(times[attention.cmd_id].start_time_ns, times[read.cmd_id].end_time_ns)
        self.assertLess(times[attention.cmd_id].end_time_ns, times[read.cmd_id].end_time_ns)
        for successor in graph.successors(attention.cmd_id):
            self.assertEqual(times[successor].start_time_ns, times[read.cmd_id].end_time_ns)
        self.assertEqual(times[read.cmd_id].duration_ns,
                         math.ceil(read.size_bytes * 1e9 / read.bandwidth_bytes_per_second))

    def test_different_microbatch_partitions_link_all_and_only_overlapping_ranges(self):
        graph = self.make_graph(batch=12, attention_split=2, fc2_split=3)
        for read in graph.commands:
            if not isinstance(read, PUMemoryReadCmd) or read.data_kind != "next_layer_kv":
                continue
            source = graph.traces[read.cmd_id]
            targets = [c for c in graph.commands if isinstance(c, FlashAttentionCmd)
                       and graph.traces[c.cmd_id].layer_index == source.layer_index + 1
                       and max(source.token_start, graph.traces[c.cmd_id].token_start)
                       < min(source.token_stop, graph.traces[c.cmd_id].token_stop)]
            self.assertEqual(set(graph.successors(read.cmd_id)),
                             {s for c in targets for s in graph.successors(c.cmd_id)})
            if source.layer_index == 0 and source.token_start == 4:
                self.assertEqual(len(targets), 2)
        # Also exercise scheduling, including the terminal look-ahead reads.
        self.assertGreater(Simulator().run(graph).total_time_ns, 0)

    def test_fast_prefetch_releases_flash_before_next_attention_without_extra_wait(self):
        graph = self.make_graph(prefetch_bandwidth=1e15)
        result = Simulator().run(graph)
        times = {item.cmd_id: item for item in result.command_results}
        read = next(c for c in graph.commands if isinstance(c, PUMemoryReadCmd)
                    and c.data_kind == "next_layer_kv" and graph.traces[c.cmd_id].layer_index == 0)
        attention = next(c for c in graph.commands if isinstance(c, FlashAttentionCmd)
                         and graph.traces[c.cmd_id].layer_index == 1)
        self.assertLess(times[read.cmd_id].end_time_ns, times[attention.cmd_id].start_time_ns)
        for successor in graph.successors(attention.cmd_id):
            self.assertEqual(times[successor].start_time_ns, times[attention.cmd_id].end_time_ns)

    def test_prefetch_competes_with_weight_reads_on_the_existing_flash_port(self):
        graph = self.make_graph(flash_weights=True)
        result = Simulator().run(graph)
        flash = sorted((c for c in result.command_results
                        if c.resource_id.endswith(".flash")), key=lambda c: c.start_time_ns)
        for first, second in zip(flash, flash[1:]):
            self.assertLessEqual(first.end_time_ns, second.start_time_ns)
        read = next(c for c in flash if c.parameters["data_kind"] == "next_layer_kv"
                    and c.layer_index == 0)
        weight = next(c for c in flash if c.op_id == "after" and c.layer_index == 0)
        self.assertGreaterEqual(weight.start_time_ns, read.end_time_ns)

    def test_non_xfold_graph_has_no_prefetch_and_retains_local_memory_timing(self):
        graph = self.make_graph(prefetch=False)
        self.assertFalse(any(isinstance(c, PUMemoryReadCmd) and c.data_kind == "next_layer_kv"
                             for c in graph.commands))
        for command in graph.commands:
            if command.op_id == "fc2" and command.cmd_id.endswith(".core"):
                self.assertEqual(command.memory_duration_ns, 1)
        self.assertGreater(Simulator().run(graph).total_time_ns, 0)


if __name__ == "__main__":
    unittest.main()
