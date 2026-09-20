"""Atlas hardware preset integration checks."""

from dataclasses import replace
import unittest

from srammachine.hardware import ATLAS_HARDWARE_CONFIG, HardwareConfig
from srammachine.inference import InferenceConfig, MoEParallelStrategy
from srammachine.mapping import HardwareMapper, HardwareMappingRequest
from srammachine.pipetree import TreeParser
from srammachine.simulator import Simulator


class AtlasHardwareConfigTests(unittest.TestCase):
    def test_uses_canonical_type_and_atlas_parameters(self):
        config = ATLAS_HARDWARE_CONFIG
        self.assertIsInstance(config, HardwareConfig)
        self.assertEqual(
            config.chip.logic_die.processing_unit.systolic_array_count, 8,
        )
        self.assertEqual(
            config.chip.logic_die.memory.dram_capacity_bytes, 80 * 10**9,
        )
        self.assertEqual(
            config.chip.logic_die.memory.sram_bandwidth_bytes_per_second,
            16 * 10**12,
        )
        self.assertEqual(config.moe_time_multiplier, 1.0)

    def test_mapper_and_parser_accept_atlas_preset(self):
        config = ATLAS_HARDWARE_CONFIG
        request = HardwareMappingRequest(
            "deepseek-v3",
            InferenceConfig(
                32, 32000, 600, MoEParallelStrategy.TP,
                mtp_enabled=False,
            ),
        )
        mapping = HardwareMapper(config).map(request)
        graph = TreeParser(config).parse(
            mapping.root_node,
            mapping.operators,
            mapping.operator_mappings,
            layer_count=1,
        )
        self.assertTrue(graph.commands)

    def test_moe_multiplier_scales_only_moe_command_durations(self):
        request = HardwareMappingRequest(
            "deepseek-v3",
            InferenceConfig(
                32, 32000, 600, MoEParallelStrategy.EP,
                mtp_enabled=False,
            ),
        )

        def simulate(config):
            mapping = HardwareMapper(config).map(request)
            graph = TreeParser(config).parse(
                mapping.root_node,
                mapping.operators,
                mapping.operator_mappings,
                layer_count=1,
            )
            return Simulator(config).run(
                graph, mapping_result=mapping,
            ).command_results

        baseline = simulate(ATLAS_HARDWARE_CONFIG)
        doubled = simulate(replace(
            ATLAS_HARDWARE_CONFIG, moe_time_multiplier=2.0,
        ))
        doubled_by_id = {item.cmd_id: item for item in doubled}
        for item in baseline:
            scaled = doubled_by_id[item.cmd_id]
            expected = (
                item.duration_ns * 2
                if item.op_id.startswith("moe.") else item.duration_ns
            )
            self.assertEqual(scaled.duration_ns, expected, item.cmd_id)


if __name__ == "__main__":
    unittest.main()
