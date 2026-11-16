"""Focused invariants for the 16-chip TP/EP MoE 8×8 mesh layouts."""

# These regression fixtures intentionally cover the retained legacy mapper.
# DSA list-based mapping is covered by test_dsa_mapper.py.
import unittest

from srammachine.tests.hardware_fixtures import LEGACY_HARDWARE_CONFIG
from srammachine.inference import InferenceConfig, MoEParallelStrategy
from srammachine.mapping import (
    LegacyHardwareMapper as HardwareMapper, HardwareMappingRequest, load_model_config,
)
from srammachine.pipetree import TreeParser
from srammachine.simulator import Simulator


MODELS = ("deepseek-v3", "deepseek-v3.2", "kimi-k2.5", "glm-5.1")


class PureMoEMappingTests(unittest.TestCase):
    def test_ep_sparse_and_high_batch_parse_and_simulate(self):
        for name in MODELS:
            for batch in (32, 8192):
                for mtp in (False, True):
                    with self.subTest(name=name, batch=batch, mtp=mtp):
                        mapped = HardwareMapper(LEGACY_HARDWARE_CONFIG).map(HardwareMappingRequest(
                            name, InferenceConfig(
                                batch, 32000, 600, MoEParallelStrategy.EP,
                                mtp_enabled=mtp,
                            ),
                        ))
                        graph = TreeParser(LEGACY_HARDWARE_CONFIG).parse(
                            mapped.root_node, mapped.operators,
                            mapped.operator_mappings, layer_count=1,
                        )
                        result = Simulator(LEGACY_HARDWARE_CONFIG).run(graph)
                        self.assertTrue(result.command_results)
                        self.assertFalse(any(
                            command.op_id.startswith("moe.ep")
                            and "_die_" in command.op_id
                            for command in result.command_results
                        ))

    def test_all_models_modes_batches_and_mtp(self):
        mapper = HardwareMapper(LEGACY_HARDWARE_CONFIG)
        for name in MODELS:
            model = load_model_config(name)
            for strategy in (MoEParallelStrategy.TP, MoEParallelStrategy.EP):
                for batch in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
                    for mtp in (False, True):
                        with self.subTest(name=name, strategy=strategy.value,
                                          batch=batch, mtp=mtp):
                            mapped = mapper.map(HardwareMappingRequest(
                                name, InferenceConfig(
                                    batch, 128, 64, strategy, mtp_enabled=mtp,
                                ),
                            ))
                            ownership = mapped.expert_ownership
                            self.assertEqual(len(ownership), 16)
                            if strategy is MoEParallelStrategy.TP:
                                self.assertTrue(all(
                                    ids == tuple(range(model.num_experts))
                                    for ids in ownership
                                ))
                                chip_k, chip_n = 1, 16
                                die_k, die_n = 1, 1
                                pu_k, pu_n = 8, 8
                            else:
                                self.assertEqual(
                                    sorted(i for ids in ownership for i in ids),
                                    list(range(model.num_experts)),
                                )
                                self.assertTrue(all(
                                    len(ids) == model.num_experts // 16
                                    for ids in ownership
                                ))
                                chip_k, chip_n = 1, 1
                                die_k, die_n = 1, 1
                                pu_k, pu_n = 8, 8
                            order = mapped.root_node.operator_order
                            self.assertFalse(any("token_group" in op for op in order))
                            up_ids = [op for op in order if op.endswith(".up_gate")]
                            down_ids = [op for op in order if op.endswith(".down")]
                            self.assertEqual(len(up_ids), len(down_ids))
                            self.assertGreater(len(up_ids), 0)
                            for up_id, down_id in zip(up_ids, down_ids):
                                up = mapped.hardware_mappings[up_id]
                                down = mapped.hardware_mappings[down_id]
                                self.assertEqual(up.expert_ids, down.expert_ids)
                                self.assertEqual(up.pu_dimensions["B"],
                                                 len(up.expert_ids))
                                self.assertEqual(down.pu_dimensions["B"],
                                                 len(down.expert_ids))
                                self.assertEqual(up.pu_dimensions["K"],
                                                 model.hidden_size // (chip_k * die_k * pu_k))
                                self.assertEqual(up.pu_dimensions["N"],
                                                 2 * model.moe_intermediate_size
                                                 // (chip_n * die_n * pu_n))
                                self.assertEqual(down.pu_dimensions["K"],
                                                 up.pu_dimensions["N"] // 2)
                                self.assertEqual(down.pu_dimensions["N"],
                                                 up.pu_dimensions["K"])
                            self.assertEqual(
                                set(i for op in up_ids for i in
                                    mapped.hardware_mappings[op].expert_ids),
                                {i for i, load in enumerate(mapped.expert_token_loads)
                                 if load > 0 and i in ownership[0]},
                            )
                            for op_id in up_ids + down_ids:
                                hw = mapped.hardware_mappings[op_id]
                                self.assertEqual(
                                    mapped.operator_mappings[op_id].weight_read_fixed_bytes,
                                    hw.pu_dimensions["B"] * hw.pu_dimensions["K"]
                                    * hw.pu_dimensions["N"],
                                )
                            if strategy is MoEParallelStrategy.TP:
                                self.assertIn("moe.tp_input_alltoall", order)
                                self.assertEqual(
                                    "moe.tp_chip_reduce_scatter" in order,
                                    chip_n > 1,
                                )
                                self.assertNotIn("moe.tp_chip_hidden_exchange", order)
                                self.assertNotIn("moe.tp_output_die_allgather", order)
                                for up_id in up_ids:
                                    self.assertLess(
                                        order.index(up_id + ".output_transfer"),
                                        order.index(up_id.replace(".up_gate", ".silu")),
                                    )
                                self.assertFalse(any(
                                    "_die_" in op for op in order
                                    if op.startswith("moe.tp")
                                ))
                            else:
                                self.assertIn("moe.ep_dispatch", order)
                                self.assertIn("moe.ep_combine", order)
                                self.assertNotIn("moe.ep_output_die_allgather", order)
                                self.assertFalse(any(
                                    "_die_" in op for op in order
                                    if op.startswith("moe.ep")
                                ))
                            for up_id in up_ids:
                                self.assertLess(
                                    order.index(up_id + ".output_transfer"),
                                    order.index(up_id.replace(".up_gate", ".silu")),
                                )



if __name__ == "__main__":
    unittest.main()
