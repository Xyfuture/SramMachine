"""Attention chip-mesh invariants and cache-accounting regressions."""

# These regression fixtures intentionally cover the retained legacy mapper.
# DSA list-based mapping is covered by test_dsa_mapper.py.
import unittest

from srammachine.inference import InferenceConfig, MoEParallelStrategy
from srammachine.tests.hardware_fixtures import LEGACY_HARDWARE_CONFIG
from srammachine.mapping import LegacyHardwareMapper as HardwareMapper, HardwareMappingRequest
from srammachine.pipetree import GroupNode, PipeTree, TreeParser
from srammachine.simulator import Simulator


MODELS = ("deepseek-v3", "deepseek-v3.2", "kimi-k2.5", "glm-5.1")
PROJECTIONS = (
    "attn.latent_down", "attn.q_rope_projection",
    "attn.qk_nope_absorb", "attn.vo_absorb",
)


def mapped(name, batch, mode, mtp=False):
    return HardwareMapper(LEGACY_HARDWARE_CONFIG).map(HardwareMappingRequest(
        name, InferenceConfig(batch, 32000, 600, mode, mtp_enabled=mtp),
    ))


class AttentionChipMeshTests(unittest.TestCase):
    def test_all_model_modes_batch_and_mtp(self):
        for name in MODELS:
            for batch in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
                for mtp in (False, True):
                    tp = mapped(name, batch, MoEParallelStrategy.TP, mtp)
                    ep = mapped(name, batch, MoEParallelStrategy.EP, mtp)
                    for current in (tp, ep):
                        with self.subTest(name=name, batch=batch, mtp=mtp,
                                          mode=current.request.inference_config.moe_parallel_strategy):
                            TreeParser(LEGACY_HARDWARE_CONFIG).parse(
                                current.root_node, current.operators,
                                current.operator_mappings, layer_count=1,
                            )
                            self.assertFalse({
                                "attn.latent_allgather", "attn.die_output_reduce",
                            } & set(current.root_node.operator_order))
                            for op_id in PROJECTIONS + (("attn.dsa_qkw",) if current.model_config.dsa else ()):
                                hw = current.hardware_mappings[op_id]
                                self.assertEqual(hw.pu_dimensions["K"] * 8,
                                                 hw.chip_dimensions["K"])
                                self.assertEqual(hw.pu_dimensions["N"] * 8,
                                                 hw.chip_dimensions["N"])
                                op_map = current.operator_mappings[op_id]
                                self.assertEqual(op_map.dram_resource_id, "chip0.dram")
                                self.assertEqual(
                                    op_map.weight_read_fixed_bytes,
                                    hw.weight_bytes // 64,
                                )
                                for suffix in ("input_broadcast", "output_reduce"):
                                    comm = current.operators[op_id + "." + suffix]
                                    self.assertEqual(len(comm.group), 8)
                                    self.assertEqual(comm.operation, "FromIOdie" if suffix == "input_broadcast" else "ToIOdie")
                            for op_id, operator in current.operators.items():
                                if op_id.startswith("attn.") and operator.__class__.__name__ == "VectorOp":
                                    self.assertEqual(
                                        current.operator_mappings[op_id].resource_id,
                                        "chip0.vector",
                                    )
                    tp_attention = {
                        op_id: (tp.hardware_mappings[op_id].pu_dimensions,
                                tp.operator_mappings[op_id])
                        for op_id in tp.root_node.operator_order if op_id.startswith("attn.")
                    }
                    ep_attention = {
                        op_id: (ep.hardware_mappings[op_id].pu_dimensions,
                                ep.operator_mappings[op_id])
                        for op_id in ep.root_node.operator_order if op_id.startswith("attn.")
                    }
                    self.assertEqual(tp_attention, ep_attention)

    def test_projection_atomic_group_microbatch_scales_bytes_once(self):
        current = mapped("deepseek-v3", 1024, MoEParallelStrategy.TP)
        attention, moe = current.root_node.root.children
        children = list(attention.children)
        latent_index = next(index for index, child in enumerate(children)
                            if isinstance(child, GroupNode)
                            and any(getattr(leaf, "op_id", None) == "attn.latent_down"
                                    for leaf in child.children))
        children[latent_index] = GroupNode(children[latent_index].children, split=2)
        tree = PipeTree(current.root_node.batch_size,
                        current.root_node.operator_order,
                        GroupNode((GroupNode(tuple(children)), moe)))
        graph = TreeParser(LEGACY_HARDWARE_CONFIG).parse(
            tree, current.operators, current.operator_mappings, layer_count=1,
        )
        result = Simulator(LEGACY_HARDWARE_CONFIG).run(graph)
        input_commands = [c for c in result.command_results
                          if c.op_id == "attn.latent_down.input_broadcast"]
        output_commands = [c for c in result.command_results
                           if c.op_id == "attn.latent_down.output_reduce"]
        self.assertEqual(len(input_commands), 2)
        self.assertEqual(len(output_commands), 2)
        self.assertEqual([c.parameters["size_bytes"] for c in input_commands],
                         [229376, 229376])
        self.assertEqual([c.parameters["size_bytes"] for c in output_commands],
                         [67584, 67584])


if __name__ == "__main__":
    unittest.main()
