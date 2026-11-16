"""EP fabric traffic, microbatch conservation, and dependency regression tests."""

from dataclasses import replace
import math
import unittest

from srammachine.commands import InterChipCmd
from srammachine.frontend.modules import FabricOp
from srammachine.inference import InferenceConfig, MoEParallelStrategy
from srammachine.mapping import DeepSeekV32Mapper, GLM51Mapper
from srammachine.pipetree import GroupNode, OpNode, PipeTree, TreeParser
from srammachine.pipetree.ep_routing import EPRouting
from srammachine.simulator.executor import GraphExecutor


class EPFabricMappingTests(unittest.TestCase):
    def test_routing_matches_individual_assignments_and_subranges(self):
        for chips, experts, top_k in ((1, 8, 3), (4, 16, 3), (8, 256, 8)):
            for batch in (1, 7, 33, 256):
                for mtp in (1, 2):
                    with self.subTest(chips=chips, batch=batch, mtp=mtp):
                        routing = EPRouting(chips, experts, top_k, mtp, 17)
                        expected = [[0] * chips for _ in range(chips)]
                        q, r = divmod(batch, chips)
                        request = 0
                        pieces = []
                        for source in range(chips):
                            for _ in range(q + (source < r)):
                                for token in range(request * mtp, (request + 1) * mtp):
                                    for k in range(top_k):
                                        expert = (token * top_k + k) % experts
                                        expected[source][expert // (experts // chips)] += 17
                                pieces.append(routing.transfer_bytes(request, request + 1, batch))
                                request += 1
                        full = routing.transfer_bytes(0, batch, batch)
                        self.assertEqual(full, tuple(map(tuple, expected)))
                        self.assertEqual(sum(map(sum, full)), batch * mtp * top_k * 17)
                        self.assertEqual(full, tuple(tuple(sum(p[i][j] for p in pieces)
                                                          for j in range(chips)) for i in range(chips)))
                        self.assertEqual(replace(routing, combine=True).transfer_bytes(0, batch, batch),
                                         tuple(zip(*full)))

    def test_models_traffic_widths_loads_and_lowered_timing(self):
        for mapper_type in (DeepSeekV32Mapper, GLM51Mapper):
            for fold in (False, True):
                for batch, mtp in ((1, False), (33, True), (256, False)):
                    with self.subTest(model=mapper_type, fold=fold, batch=batch, mtp=mtp):
                        mapper = mapper_type()
                        mapped = mapper.map(InferenceConfig(batch, 128, 1, mtp_enabled=mtp), xfold=fold)
                        dispatch = mapped.operators["moe.ep_dispatch"].transfer_bytes
                        fc1 = mapped.operators["moe.moe_fc1"]
                        payload = fc1.K * fc1.input_dtype_bytes
                        self.assertEqual(sum(map(sum, dispatch)), batch * (2 if mtp else 1)
                                         * mapped.model_config.top_k * payload)
                        for chip, experts in enumerate(mapped.expert_ownership):
                            self.assertEqual(sum(row[chip] for row in dispatch) // payload,
                                             sum(mapped.expert_token_loads[e] for e in experts))
                        self.assertEqual(mapped.operators["moe.ep_combine"].transfer_bytes,
                                         tuple(tuple(2 * value for value in row)
                                               for row in zip(*dispatch)))
                        if fold:
                            projection = mapped.operators["moe.fold_latent_q_indexer_k_w"]
                            expected = tuple(tuple(value // payload * projection.N * projection.output_dtype_bytes
                                                   for value in row) for row in zip(*dispatch))
                            self.assertEqual(mapped.operators["moe.ep_fold_combine"].transfer_bytes, expected)
                        # Isolate communication to test exact fabric lowering and timing.
                        fabric = {k: v for k, v in mapped.operators.items() if isinstance(v, FabricOp)}
                        tree = PipeTree(batch, tuple(fabric), GroupNode(tuple(OpNode(k) for k in fabric)))
                        graph = TreeParser(mapper.hardware_config).parse(
                            tree, fabric, {k: mapped.operator_mappings[k] for k in fabric}, layer_count=1)
                        result = GraphExecutor.simulate(graph, hardware_config=mapper.hardware_config)
                        bandwidth = mapper.hardware_config.inter_chip_fabric.per_chip_directional_bandwidth_bytes_per_second
                        for command in graph.commands:
                            matrix = command.transfer_bytes
                            sent = [sum(v for j, v in enumerate(row) if i != j) for i, row in enumerate(matrix)]
                            received = [sum(matrix[i][j] for i in range(len(matrix)) if i != j)
                                        for j in range(len(matrix))]
                            execution = result.executions[command.cmd_id]
                            self.assertEqual(execution.end_time_ns - execution.start_time_ns,
                                             math.ceil(max(sent + received) * 1_000_000_000 / bandwidth))

    def test_nested_split_conserves_every_matrix_entry(self):
        mapper = DeepSeekV32Mapper()
        mapped = mapper.map(InferenceConfig(34, 128, 1, mtp_enabled=True), xfold=True)
        fabric = {k: v for k, v in mapped.operators.items() if isinstance(v, FabricOp)}
        # Single-request intervals deliberately cross DP and expert boundaries.
        tree = PipeTree(34, tuple(fabric), GroupNode((GroupNode(
            tuple(OpNode(k) for k in fabric), split=17),), split=2))
        graph = TreeParser(mapper.hardware_config).parse(
            tree, fabric, {k: mapped.operator_mappings[k] for k in fabric}, layer_count=1)
        for op_id, op in fabric.items():
            matrices = [c.transfer_bytes for c in graph.commands if c.op_id == op_id]
            self.assertEqual(len(matrices), 34)
            self.assertEqual(op.transfer_bytes, tuple(tuple(sum(m[i][j] for m in matrices)
                             for j in range(mapped.chip_count)) for i in range(mapped.chip_count)))

    def test_full_graph_communication_dependencies_and_tp_unchanged(self):
        mapper = DeepSeekV32Mapper()
        for fold in (False, True):
            mapped = mapper.map(InferenceConfig(256, 128, 1), xfold=fold)
            graph = TreeParser(mapper.hardware_config).parse(
                mapped.root_node, mapped.operators, mapped.operator_mappings, layer_count=2)
            commands = {c.cmd_id: c for c in graph.commands}
            pairs = {(commands[a].op_id, commands[b].op_id) for a, b in graph.edges}
            self.assertIn(("moe.ep_dispatch", "moe.moe_fc1"), pairs)
            self.assertIn(("moe.moe_fc2", "moe.ep_combine"), pairs)
            self.assertIn(("moe.ep_combine", "attn.latent_down"), pairs)
            if fold:
                self.assertIn(("moe.fold_latent_q_indexer_k_w", "moe.ep_fold_combine"), pairs)
                self.assertIn(("moe.ep_fold_combine", "moe.indexer_q"), pairs)
            self.assertEqual(sum(isinstance(c, InterChipCmd) for c in graph.commands), 6 if fold else 4)
        mapped = mapper.map(InferenceConfig(256, 128, 1, moe_parallel_strategy=MoEParallelStrategy.TP))
        self.assertFalse(any(isinstance(op, FabricOp) for op in mapped.op_list))


if __name__ == "__main__":
    unittest.main()
