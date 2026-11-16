"""Direct weight transfers must not require an intermediate memory tier."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from srammachine.commands import CommandGraph, DramReadCmd, PUMemoryReadCmd
from srammachine.frontend.modules import BMMOp, WeightMemory, WeightSource
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG
from srammachine.inference import InferenceConfig, MoEParallelStrategy
from srammachine.mapping import HardwareMappingRequest, LegacyHardwareMapper, load_model_config
from srammachine.pipetree import GroupNode, OpNode, OperatorMapping, PipeTree, TreeParser
from srammachine.simulator import GraphExecutor
from srammachine.tests.hardware_fixtures import LEGACY_HARDWARE_CONFIG


class DirectMemoryTests(unittest.TestCase):
    def test_large_weights_read_once_per_microbatch_and_layer(self):
        """A 1.6 GB operand streams without fitting a resident-weight buffer."""
        op = BMMOp("op", 1, 2, 40000, 40000,
                   input_source="none", output_destination="none")
        tree = PipeTree(2, (op.op_id,), GroupNode((OpNode(op.op_id),), split=2))
        graph = TreeParser().parse(tree, {op.op_id: op}, {
            op.op_id: OperatorMapping("M", "chip0.pu0"),
        }, layer_count=2)
        reads = [cmd for cmd in graph.commands if isinstance(cmd, PUMemoryReadCmd)]
        self.assertEqual(len(reads), 4)
        self.assertFalse(any(isinstance(cmd, DramReadCmd) for cmd in graph.commands))
        result = GraphExecutor.simulate(graph)
        for read in reads:
            self.assertEqual(read.size_bytes, 1_600_000_000)
            self.assertEqual(read.resource_id, "chip0.pu0.dram")
            prefix = read.cmd_id.split(".weight.")[0]
            execution = result.executions[read.cmd_id]
            self.assertEqual(execution.duration_ns, 100_000_000)
            self.assertEqual(execution.start_time_ns,
                             result.executions[prefix + ".core"].start_time_ns)
            self.assertGreaterEqual(result.executions[prefix + ".output"].start_time_ns,
                                    execution.end_time_ns)
        for index in (0, 1):
            self.assertGreaterEqual(result.executions[f"layer1.i{index}.core"].start_time_ns,
                                    result.executions[f"layer0.i{index}.output"].end_time_ns)

    def test_fixed_weight_bytes_use_one_pu_port(self):
        op = BMMOp("op", 4, 1, 100, 200, auto_weight_load=False,
                   input_source="none", output_destination="none")
        tree = PipeTree(1, (op.op_id,), GroupNode((OpNode(op.op_id),)))
        graph = TreeParser().parse(tree, {op.op_id: op}, {
            op.op_id: OperatorMapping("M", "chip0.pu0", weight_read_fixed_bytes=20000),
        }, layer_count=1)
        reads = [cmd for cmd in graph.commands if isinstance(cmd, PUMemoryReadCmd)]
        self.assertEqual(len(reads), 1)
        self.assertEqual(reads[0].size_bytes, 20000)
        self.assertEqual(reads[0].port, "dram")
        self.assertEqual(graph.command("i0.core").memory_bytes, 0)
        self.assertEqual(graph.command("i0.core").memory_duration_ns, 1250)
        GraphExecutor.simulate(graph)

    def test_legacy_cache_has_one_direct_read(self):
        mapped = LegacyHardwareMapper().map(HardwareMappingRequest(
            "deepseek-v3.2", InferenceConfig(256, 128, 1),
        ))
        graph = TreeParser().parse(mapped.root_node, mapped.operators,
                                   mapped.operator_mappings, layer_count=1)
        for op_id in ("attn.dsa_qk_score", "attn.flash_attention"):
            with self.subTest(op_id=op_id):
                reads = [cmd for cmd in graph.commands
                         if cmd.op_id == op_id and isinstance(cmd, DramReadCmd)]
                self.assertEqual(len(reads), 1)
                mapping = mapped.operator_mappings[op_id]
                self.assertEqual(reads[0].size_bytes,
                                 mapped.root_node.batch_size * mapping.dram_read_bytes_per_token)
                index = graph.traces[reads[0].cmd_id].instance_index
                self.assertIn((reads[0].cmd_id, f"i{index}.output"), graph.edges)
        GraphExecutor.simulate(graph)

    def test_hierarchical_experts_are_not_split_into_capacity_waves(self):
        mapper = LegacyHardwareMapper(LEGACY_HARDWARE_CONFIG)
        model = load_model_config("deepseek-v3")
        model = replace(model, moe_intermediate_size=2 * model.moe_intermediate_size)
        with patch.object(mapper, "_use_pure_moe_mapping", return_value=False), \
                patch("srammachine.mapping.mapper.load_model_config", return_value=model):
            mapped = mapper.map(HardwareMappingRequest(
                "deepseek-v3", InferenceConfig(8192, 128, 1, MoEParallelStrategy.TP),
            ))
        order = mapped.root_node.operator_order
        self.assertFalse(any(".wave" in op_id for op_id in order))
        up_ids = [op_id for op_id in order if op_id.endswith(".up_gate")]
        self.assertEqual(len(up_ids), 1)
        self.assertGreater(mapped.hardware_mappings[up_ids[0]].weight_bytes, 768_000_000)
        graph = TreeParser(LEGACY_HARDWARE_CONFIG).parse(
            mapped.root_node, mapped.operators, mapped.operator_mappings, layer_count=1,
        )
        GraphExecutor.simulate(graph, LEGACY_HARDWARE_CONFIG)

    def test_dual_port_split_and_contended_completion(self):
        pu = replace(DEFAULT_HARDWARE_CONFIG.chip.logic_die.processing_unit,
                     dram_bandwidth_bytes_per_second=1e9,
                     flash_bandwidth_bytes_per_second=3e9)
        die = replace(DEFAULT_HARDWARE_CONFIG.chip.logic_die, processing_unit=pu)
        hardware = replace(DEFAULT_HARDWARE_CONFIG,
                           chip=replace(DEFAULT_HARDWARE_CONFIG.chip, logic_die=die))
        op = BMMOp("op", 1, 1, 100, 100, input_source="none", output_destination="none",
                   weight_memory=WeightMemory(WeightSource.DRAM_FLASH,
                                              flash_bandwidth_bytes_per_second=9e9))
        tree = PipeTree(1, (op.op_id,), GroupNode((OpNode(op.op_id),)))
        graph = TreeParser(hardware).parse(tree, {op.op_id: op}, {
            op.op_id: OperatorMapping("M", "chip0.pu0"),
        }, layer_count=1)
        reads = {cmd.port: cmd for cmd in graph.commands if isinstance(cmd, PUMemoryReadCmd)}
        self.assertEqual(reads["dram"].size_bytes, 1000)
        self.assertEqual(reads["flash"].size_bytes, 9000)
        # Occupy Flash first, so ideal compute duration alone cannot gate output.
        blocker = PUMemoryReadCmd("blocker", "other", "chip0.pu0.flash", 90000, "flash", 9e9)
        graph = CommandGraph((blocker,) + graph.commands, graph.edges,
                             start_edges=graph.start_edges)
        result = GraphExecutor.simulate(graph, hardware)
        self.assertEqual(result.executions[reads["dram"].cmd_id].duration_ns, 1000)
        self.assertEqual(result.executions[reads["flash"].cmd_id].duration_ns, 1000)
        self.assertEqual(result.executions["i0.output"].start_time_ns, 11000)


if __name__ == "__main__":
    unittest.main()
