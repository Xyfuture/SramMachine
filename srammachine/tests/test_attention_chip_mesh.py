"""Attention chip-mesh invariants and cache-accounting regressions."""

import unittest

from srammachine.inference import InferenceConfig, MoEParallelStrategy
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG
from srammachine.mapping import HardwareMapper, HardwareMappingRequest
from srammachine.pipetree import GroupNode, PipeTree, TreeParser
from srammachine.simulator import Simulator


MODELS = ("deepseek-v3", "deepseek-v3.2", "kimi-k2.5", "glm-5.1")
PROJECTIONS = (
    "attn.latent_down", "attn.q_rope_projection",
    "attn.qk_nope_absorb", "attn.vo_absorb",
)


def mapped(name, batch, mode, mtp=False):
    return HardwareMapper().map(HardwareMappingRequest(
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
                            TreeParser().parse(
                                current.root_node, current.operators,
                                current.operator_mappings, layer_count=1,
                            )
                            self.assertFalse({
                                "attn.latent_allgather", "attn.die_output_reduce",
                                "attn.flash_attention",
                            } & set(current.root_node.operator_order))
                            attention_order = tuple(op_id for op_id in
                                                    current.root_node.operator_order
                                                    if op_id.startswith("attn."))
                            self.assertLess(attention_order.index("attn.qk_fused"),
                                            attention_order.index("attn.softmax"))
                            self.assertLess(attention_order.index("attn.softmax"),
                                            attention_order.index("attn.sv_latent"))
                            history = min(32000, current.model_config.dsa_len) if current.model_config.dsa else 32000
                            head_batch = (batch // 16) * (2 if mtp else 1) * current.model_config.num_attention_heads
                            qk = current.hardware_mappings["attn.qk_fused"].pu_dimensions
                            sv = current.hardware_mappings["attn.sv_latent"].pu_dimensions
                            self.assertEqual(qk["B"], head_batch // 8)
                            self.assertEqual((qk["K"], qk["N"]),
                                             (current.model_config.kv_lora_rank + current.model_config.qk_rope_head_dim,
                                              (history + 7) // 8))
                            self.assertEqual((sv["B"], sv["K"], sv["N"]),
                                             (head_batch // 8, (history + 7) // 8,
                                              current.model_config.kv_lora_rank))
                            score_bytes = head_batch * history
                            for score_op in (
                                "attn.qk_fused.output_transfer",
                                "attn.sv_latent.input_transfer",
                            ):
                                comm = current.operators[score_op]
                                self.assertEqual(comm.size_bytes, score_bytes)
                                self.assertEqual(comm.parallel_link_count, 8)
                            self.assertEqual(
                                current.operator_mappings["attn.qk_fused"].sram_read_data_kind,
                                "kv_fused",
                            )
                            self.assertEqual(
                                current.operator_mappings["attn.sv_latent"].sram_read_data_kind,
                                "kv_value",
                            )
                            for op_id in PROJECTIONS + (("attn.dsa_qkw",) if current.model_config.dsa else ()):
                                hw = current.hardware_mappings[op_id]
                                self.assertEqual(hw.pu_dimensions["K"] * 8,
                                                 hw.chip_dimensions["K"])
                                self.assertEqual(hw.pu_dimensions["N"] * 8,
                                                 hw.chip_dimensions["N"])
                                op_map = current.operator_mappings[op_id]
                                self.assertEqual(op_map.dram_resource_id, "chip0.dram")
                                self.assertEqual(op_map.sram_resource_id, "chip0.sram")
                                self.assertEqual(
                                    op_map.dram_read_once_bytes,
                                    hw.weight_bytes,
                                )
                                self.assertLessEqual(
                                    hw.weight_bytes,
                                    4 * DEFAULT_HARDWARE_CONFIG.chip.logic_die.memory.sram_capacity_bytes,
                                )
                                for suffix in ("input_broadcast", "output_reduce"):
                                    comm = current.operators[op_id + "." + suffix]
                                    self.assertEqual(len(comm.group), 8)
                                    self.assertEqual(comm.parallel_link_count, 8)
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

    def test_reference_noc_and_cache_times(self):
        current = mapped("deepseek-v3.2", 1024, MoEParallelStrategy.TP)
        result = Simulator().run(TreeParser().parse(
            current.root_node, current.operators,
            current.operator_mappings, layer_count=1,
        ))
        by_command = {(c.op_id, c.command_type): c
                      for c in result.command_results}
        latent = current.hardware_mappings["attn.latent_down"]
        self.assertEqual(dict(latent.pu_dimensions),
                         {"B": 1, "M": 64, "K": 896, "N": 264})
        for suffix, size, duration in (
            ("input_broadcast", 458752, 224),
            ("output_reduce", 135168, 66),
        ):
            command = by_command[("attn.latent_down." + suffix, "NoCCmd")]
            self.assertEqual(command.parameters["size_bytes"], size)
            self.assertEqual(command.parameters["noc_parallel_link_count"], 8)
            self.assertEqual(command.duration_ns, duration)
        for op_id, kind, size, duration in (
            ("attn.qk_fused", "DramReadCmd", 75497472, 1180),
            ("attn.qk_fused", "SramReadCmd", 4718592, 288),
            ("attn.sv_latent", "DramReadCmd", 67108864, 1049),
            ("attn.sv_latent", "SramReadCmd", 4194304, 256),
            ("attn.dsa_qk_score", "DramReadCmd", 262144000, 4096),
            ("attn.dsa_qk_score", "SramReadCmd", 262144000, 1024),
        ):
            command = by_command[(op_id, kind)]
            self.assertEqual(command.parameters["size_bytes"], size)
            self.assertEqual(command.duration_ns, duration)
            self.assertIn(command.resource_id, ("chip0.dram", "chip0.sram"))
        score_bytes = 64 * 128 * 2048
        for op_id in ("attn.qk_fused.output_transfer",
                      "attn.sv_latent.input_transfer"):
            command = by_command[(op_id, "NoCCmd")]
            self.assertEqual(command.parameters["size_bytes"], score_bytes)
            self.assertEqual(command.parameters["noc_parallel_link_count"], 8)
            self.assertEqual(command.duration_ns, 8192)
        self.assertEqual(by_command[("attn.softmax", "VectorCmd")].resource_id,
                         "chip0.vector")
        self.assertLessEqual(
            by_command[("attn.qk_fused.output_transfer", "NoCCmd")].end_time_ns,
            by_command[("attn.softmax", "VectorCmd")].start_time_ns,
        )
        self.assertLessEqual(
            by_command[("attn.softmax", "VectorCmd")].end_time_ns,
            by_command[("attn.sv_latent.input_transfer", "NoCCmd")].start_time_ns,
        )
        for suffix in ("input_broadcast", "output_transfer"):
            command = by_command[("attn.dsa_qk_score." + suffix, "NoCCmd")]
            self.assertEqual(command.parameters["size_bytes"], 524288)
            self.assertEqual(command.parameters["noc_parallel_link_count"], 8)
            self.assertEqual(command.duration_ns, 256)

    def test_dsa_chip_traffic_scales_with_requests_and_microbatches(self):
        for name in ("deepseek-v3.2", "glm-5.1"):
            for batch in (32, 128, 256, 512, 1024, 2048):
                for mtp in (False, True):
                    with self.subTest(name=name, batch=batch, mtp=mtp):
                        current = mapped(name, batch, MoEParallelStrategy.TP, mtp)
                        result = Simulator().run(TreeParser().parse(
                            current.root_node, current.operators,
                            current.operator_mappings, layer_count=1,
                        ))
                        local_requests = batch // 16
                        expected_cache = (
                            local_requests * 32000
                            * current.model_config.indexer_head_dim
                        )
                        for kind in ("DramReadCmd", "SramReadCmd"):
                            command = next(c for c in result.command_results
                                           if c.op_id == "attn.dsa_qk_score"
                                           and c.command_type == kind)
                            self.assertEqual(
                                command.parameters["size_bytes"], expected_cache,
                            )

        current = mapped("deepseek-v3.2", 1024, MoEParallelStrategy.TP)
        attention, moe = current.root_node.root.children
        children = list(attention.children)
        score_index = next(index for index, child in enumerate(children)
                           if isinstance(child, GroupNode)
                           and any(getattr(leaf, "op_id", None) == "attn.dsa_qk_score"
                                   for leaf in child.children))
        children[score_index] = GroupNode(children[score_index].children, split=2)
        tree = PipeTree(current.root_node.batch_size,
                        current.root_node.operator_order,
                        GroupNode((GroupNode(tuple(children)), moe)))
        result = Simulator().run(TreeParser().parse(
            tree, current.operators, current.operator_mappings, layer_count=1,
        ))
        for op_id, kind, expected in (
            ("attn.dsa_qk_score", "DramReadCmd", 262144000),
            ("attn.dsa_qk_score", "SramReadCmd", 262144000),
            ("attn.dsa_qk_score.input_broadcast", "NoCCmd", 524288),
            ("attn.dsa_qk_score.output_transfer", "NoCCmd", 524288),
        ):
            commands = [c for c in result.command_results
                        if c.op_id == op_id and c.command_type == kind]
            self.assertEqual(len(commands), 2)
            self.assertEqual(sum(c.parameters["size_bytes"] for c in commands),
                             expected)

    def test_unfused_score_stream_and_cache_microbatch_conservation(self):
        current = mapped("deepseek-v3", 1024, MoEParallelStrategy.TP, True)
        baseline = Simulator().run(TreeParser().parse(
            current.root_node, current.operators,
            current.operator_mappings, layer_count=1,
        ))
        by_command = {(c.op_id, c.command_type): c
                      for c in baseline.command_results}
        for op_id in ("attn.qk_fused.output_transfer",
                      "attn.sv_latent.input_transfer"):
            command = by_command[(op_id, "NoCCmd")]
            self.assertEqual(command.parameters["size_bytes"], 524288000)
            self.assertEqual(command.parameters["noc_parallel_link_count"], 8)
            self.assertEqual(command.duration_ns, 256000)
        self.assertEqual(
            by_command[("attn.qk_fused", "DramReadCmd")].parameters["size_bytes"],
            1179648000,
        )
        self.assertEqual(
            by_command[("attn.sv_latent", "DramReadCmd")].parameters["size_bytes"],
            1048576000,
        )

        attention, moe = current.root_node.root.children
        children = list(attention.children)
        for index, child in enumerate(children):
            if isinstance(child, GroupNode) and any(
                getattr(leaf, "op_id", None) in ("attn.qk_fused", "attn.sv_latent")
                for leaf in child.children
            ):
                children[index] = GroupNode(child.children, split=2)
        tree = PipeTree(current.root_node.batch_size,
                        current.root_node.operator_order,
                        GroupNode((GroupNode(tuple(children)), moe)))
        split = Simulator().run(TreeParser().parse(
            tree, current.operators, current.operator_mappings,
            layer_count=1,
        ))
        for op_id, kind in (
            ("attn.qk_fused.output_transfer", "NoCCmd"),
            ("attn.sv_latent.input_transfer", "NoCCmd"),
            ("attn.qk_fused", "DramReadCmd"),
            ("attn.sv_latent", "DramReadCmd"),
            ("attn.qk_fused", "SramReadCmd"),
            ("attn.sv_latent", "SramReadCmd"),
        ):
            commands = [c for c in split.command_results
                        if c.op_id == op_id and c.command_type == kind]
            self.assertEqual(len(commands), 2)
            self.assertEqual(sum(c.parameters["size_bytes"] for c in commands),
                             by_command[(op_id, kind)].parameters["size_bytes"])

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
        graph = TreeParser().parse(
            tree, current.operators, current.operator_mappings, layer_count=1,
        )
        result = Simulator().run(graph)
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
