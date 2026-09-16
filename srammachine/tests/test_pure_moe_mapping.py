"""Focused invariants for the 16-chip TP/EP MoE 8×8 mesh layouts."""

import unittest

from srammachine.hardware import DEFAULT_HARDWARE_CONFIG
from srammachine.inference import InferenceConfig, MoEParallelStrategy
from srammachine.mapping import (
    HardwareMapper, HardwareMappingRequest, load_model_config,
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
                        mapped = HardwareMapper().map(HardwareMappingRequest(
                            name, InferenceConfig(
                                batch, 32000, 600, MoEParallelStrategy.EP,
                                mtp_enabled=mtp,
                            ),
                        ))
                        graph = TreeParser().parse(
                            mapped.root_node, mapped.operators,
                            mapped.operator_mappings, layer_count=1,
                        )
                        result = Simulator().run(graph)
                        self.assertTrue(result.command_results)
                        self.assertFalse(any(
                            command.op_id.startswith("moe.ep")
                            and "_die_" in command.op_id
                            for command in result.command_results
                        ))

    def test_chip_mesh_ep_reference_bytes_and_timing(self):
        mapped = HardwareMapper().map(HardwareMappingRequest(
            "deepseek-v3", InferenceConfig(
                1024, 32000, 600, MoEParallelStrategy.EP,
            ),
        ))
        graph = TreeParser().parse(
            mapped.root_node, mapped.operators, mapped.operator_mappings,
            layer_count=1,
        )
        result = Simulator().run(graph)
        by_op = {}
        for command in result.command_results:
            if command.op_id.startswith("moe.ep"):
                by_op.setdefault(command.op_id, []).append(command)

        up = "moe.ep.tokens32.up_gate"
        down = "moe.ep.tokens32.down"
        stages = (
            (up + ".input_transfer", "NoCCmd", 3_670_016, 448, 32),
            (up + ".output_transfer", "NoCCmd", 131_072, 512, 1),
            (down + ".input_transfer", "NoCCmd", 131_072, 512, 1),
            (down + ".output_reduce", "NoCCmd", 3_670_016, 448, 32),
            (up, "WeightPrefetchCmd", 469_762_048, 7341, None),
            (up, "WeightLoadCmd", 469_762_048, 1836, None),
            (down, "WeightPrefetchCmd", 234_881_024, 3671, None),
            (down, "WeightLoadCmd", 234_881_024, 918, None),
        )
        for op_id, kind, size, duration, links in stages:
            with self.subTest(op_id=op_id, kind=kind):
                command = next(c for c in by_op[op_id]
                               if c.command_type == kind)
                self.assertEqual(command.parameters["size_bytes"], size)
                self.assertEqual(command.duration_ns, duration)
                if links is not None:
                    self.assertEqual(
                        command.parameters["noc_parallel_link_count"], links,
                    )
        self.assertEqual(
            tuple(c.duration_ns for c in by_op["moe.ep_dispatch"]), (2151,),
        )
        self.assertEqual(
            tuple(c.duration_ns for c in by_op["moe.ep_combine"]), (2151,),
        )
        self.assertEqual(by_op["moe.ep.tokens32.silu"][0].resource_id,
                         "chip0.vector")
        self.assertFalse(any("_die_" in op for op in by_op))

    def test_chip_mesh_tp_reference_bytes_and_timing(self):
        mapped = HardwareMapper().map(HardwareMappingRequest(
            "deepseek-v3", InferenceConfig(
                1024, 32000, 600, MoEParallelStrategy.TP,
            ),
        ))
        graph = TreeParser().parse(
            mapped.root_node, mapped.operators, mapped.operator_mappings,
            layer_count=1,
        )
        result = Simulator().run(graph)
        by_op = {}
        for command in result.command_results:
            if command.op_id.startswith("moe.tp"):
                by_op.setdefault(command.op_id, []).append(command)

        up = "moe.tp.tokens32.up_gate"
        down = "moe.tp.tokens32.down"
        stages = (
            (up + ".input_transfer", "NoCCmd", 58_720_256, 7168, 32),
            (up + ".output_transfer", "NoCCmd", 131_072, 512, 1),
            (down + ".input_transfer", "NoCCmd", 131_072, 512, 1),
            (down + ".output_reduce", "NoCCmd", 58_720_256, 7168, 32),
            (up, "WeightPrefetchCmd", 469_762_048, 7341, None),
            (up, "WeightLoadCmd", 469_762_048, 1836, None),
            (down, "WeightPrefetchCmd", 234_881_024, 3671, None),
            (down, "WeightLoadCmd", 234_881_024, 918, None),
        )
        for op_id, kind, size, duration, links in stages:
            with self.subTest(op_id=op_id, kind=kind):
                command = next(c for c in by_op[op_id]
                               if c.command_type == kind)
                self.assertEqual(command.parameters["size_bytes"], size)
                self.assertEqual(command.duration_ns, duration)
                if links is not None:
                    self.assertEqual(
                        command.parameters["noc_parallel_link_count"], links,
                    )
        self.assertEqual(
            next(c for c in by_op[up] if c.command_type == "GemmCmd").duration_ns,
            3584,
        )
        self.assertEqual(
            next(c for c in by_op[down] if c.command_type == "GemmCmd").duration_ns,
            1792,
        )
        silu = by_op["moe.tp.tokens32.silu"][0]
        self.assertEqual(silu.resource_id, "chip0.vector")
        self.assertEqual(silu.duration_ns, 9)
        self.assertFalse(any("_die_" in op for op in by_op))

    def test_all_models_modes_batches_and_mtp(self):
        mapper = HardwareMapper()
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
                            weight_bytes = sum(
                                mapped.hardware_mappings[op].weight_bytes
                                for op in up_ids + down_ids
                            )
                            self.assertLessEqual(
                                weight_bytes,
                                DEFAULT_HARDWARE_CONFIG.chip.logic_die.memory.sram_capacity_bytes
                                * 4,
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
