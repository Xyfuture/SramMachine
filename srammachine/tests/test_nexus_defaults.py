"""Nexus defaults must reach mapping, memory timing, and experiment entry points."""

from dataclasses import replace
import unittest

from srammachine.commands import PUMemoryReadCmd
from srammachine.frontend.modules import BMMOp, WeightMemory, WeightSource
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG, HardwareConfig
from srammachine.inference import InferenceConfig, MoEParallelStrategy
from srammachine.mapping import HardwareMappingRequest, mapper_for_model
from srammachine.pipetree import GroupNode, OpNode, OperatorMapping, PipeTree, TreeParser
from srammachine.script.run_sa import build_parser
from srammachine.simulator import Simulator


class NexusDefaultsTests(unittest.TestCase):
    def test_system_topology_and_directional_bandwidth(self):
        hardware = HardwareConfig()
        chip = hardware.chip
        pu = chip.logic_die.processing_unit
        self.assertEqual(hardware, DEFAULT_HARDWARE_CONFIG)
        self.assertEqual(hardware.chip_count, 8)
        self.assertEqual((chip.noc.mesh_rows, chip.noc.mesh_columns), (32, 64))
        self.assertEqual(chip.processing_unit_count, 2048)
        self.assertEqual(hardware.processing_unit_count, 16384)
        self.assertEqual(pu.systolic_array_count, 4)
        self.assertEqual((pu.systolic_array.rows, pu.systolic_array.columns), (16, 16))
        self.assertEqual(pu.systolic_array.frequency_hz, 10**9)
        self.assertEqual(pu.peak_flops, 2048 * 10**9)
        self.assertEqual(chip.systolic_peak_flops, 4194304 * 10**9)
        self.assertTrue(chip.noc.full_duplex)
        self.assertEqual(chip.noc.link_bandwidth_bytes_per_second, 128 * 10**9)
        self.assertEqual(chip.noc.bidirectional_link_bandwidth_bytes_per_second, 256 * 10**9)

    def test_default_memory_ports_split_four_to_one_at_equal_duration(self):
        for source, expected_bytes, duration in (
            (WeightSource.DRAM, {"dram": 20000}, 1250),
            (WeightSource.FLASH, {"flash": 20000}, 5000),
            (WeightSource.DRAM_FLASH, {"dram": 16000, "flash": 4000}, 1000),
        ):
            with self.subTest(source=source):
                op = BMMOp("op", 1, 1, 100, 200, input_source="none",
                           output_destination="none", weight_memory=WeightMemory(source))
                tree = PipeTree(1, (op.op_id,), OpNode(op.op_id))
                graph = TreeParser().parse(tree, {op.op_id: op}, {
                    op.op_id: OperatorMapping("M", "chip0.die0.pu0"),
                }, layer_count=1)
                reads = [cmd for cmd in graph.commands if isinstance(cmd, PUMemoryReadCmd)]
                self.assertEqual({cmd.port: cmd.size_bytes for cmd in reads}, expected_bytes)
                result = Simulator().run(graph)
                times = {cmd.cmd_id: cmd.duration_ns for cmd in result.command_results}
                self.assertEqual({times[cmd.cmd_id] for cmd in reads}, {duration})

    def test_all_models_default_to_attention_dp_and_expert_ownership(self):
        for name in ("deepseek-v3", "deepseek-v3.2", "kimi-k2.5", "glm-5.1"):
            with self.subTest(model=name):
                inference = InferenceConfig(256, 128, 1)
                self.assertIs(inference.moe_parallel_strategy, MoEParallelStrategy.EP)
                mapper = mapper_for_model(name)
                self.assertEqual(mapper.hardware_config, DEFAULT_HARDWARE_CONFIG)
                mapped = mapper.map(HardwareMappingRequest(name, inference))
                self.assertEqual(mapped.local_batch_size, 32)
                self.assertEqual(mapped.chip_count, 8)
                experts = mapped.model_config.num_experts
                self.assertTrue(all(len(ids) == experts // 8 for ids in mapped.expert_ownership))
                self.assertEqual(sorted(e for ids in mapped.expert_ownership for e in ids),
                                 list(range(experts)))
                projection = mapped.hardware_mappings["attn.latent_down"]
                self.assertEqual(projection.global_dimensions["M"], 256)
                self.assertEqual(projection.chip_dimensions["M"], 32)
                self.assertEqual(projection.pu_dimensions["K"], mapped.model_config.hidden_size // 32)
                graph = TreeParser().parse(mapped.root_node, mapped.operators,
                                           mapped.operator_mappings, layer_count=1)
                self.assertGreater(Simulator().run(graph, mapping_result=mapped).total_time_ns, 0)

    def test_dsa_dp_microbatches_and_explicit_tp(self):
        mapper = mapper_for_model("deepseek-v3.2")
        for strategy in (MoEParallelStrategy.EP, MoEParallelStrategy.TP):
            for batch in (1, 9, 256):
                with self.subTest(strategy=strategy, batch=batch):
                    mapped = mapper.map(InferenceConfig(batch, 128, 1, strategy, mtp_enabled=True))
                    local_batch = (batch + 7) // 8
                    self.assertEqual(mapped.operators["attn.latent_down"].M, local_batch * 2)
                    fc1, fc2 = (mapped.operators[key] for key in ("moe.moe_fc1", "moe.moe_fc2"))
                    if strategy is MoEParallelStrategy.EP:
                        self.assertEqual((fc1.B, fc2.B), (32, 32))
                    else:
                        self.assertEqual((fc1.B, fc1.N, fc2.K), (256, 512, 256))
                    root = mapped.root_node
                    if batch == 256:
                        root = replace(root, root=GroupNode(root.root.children, split=2))
                    graph = TreeParser().parse(root, mapped.operators,
                                               mapped.operator_mappings, layer_count=1)
                    self.assertGreater(Simulator().run(graph).total_time_ns, 0)

    def test_sa_cli_defaults_to_ep_and_accepts_explicit_tp(self):
        parser = build_parser()
        arguments = ["--models", "deepseek-v3", "--mtp", "off", "--batch-sizes", "256",
                     "--rounds", "1", "--input-sequence-length", "128",
                     "--output-sequence-length", "1"]
        self.assertEqual(parser.parse_args(arguments).moe_strategy, ["ep"])
        self.assertEqual(parser.parse_args(arguments + ["--moe-strategy", "tp"]).moe_strategy, ["tp"])


if __name__ == "__main__":
    unittest.main()
