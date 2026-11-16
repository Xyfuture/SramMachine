"""Regression coverage for model-specific DSA lists and PU memory streams."""

from dataclasses import replace
from pathlib import Path
import unittest

from srammachine.commands import CommandGraph, PUMemoryReadCmd
from srammachine.frontend.modules import BMMOp, FlashAttentionOp, WeightMemory, WeightSource
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG
from srammachine.inference import InferenceConfig, MoEParallelStrategy
from srammachine.mapping import (
    DEFAULT_HBF_BANDWIDTH_CSV, HBFBandwidthTable,
    DeepSeekV32Mapper, GLM51Mapper, HardwareMappingRequest, MiniMaxM3Mapper,
    mapper_for_model,
)
from srammachine.mapping.components import ATTENTION_SPLIT, FC1_SPLIT, FC2_SPLIT, PROJECTION_SPLIT
from srammachine.pipetree import GroupNode, OpNode, OperatorMapping, PipeTree, TreeParser
from srammachine.simulator import Simulator
from srammachine.simulator.executor import GraphExecutor
from srammachine.tests.hardware_fixtures import LEGACY_HARDWARE_CONFIG


class DSAMapperTests(unittest.TestCase):
    def test_model_binding_json_and_ordinary_order(self):
        mapper = DeepSeekV32Mapper.from_json(
            Path(__file__).resolve().parents[1] / "model_cards/deepseek-v3.2.json",
        )
        mapped = mapper.map(InferenceConfig(256, 20000, 600))
        self.assertIsInstance(mapped.op_list, list)
        self.assertEqual(mapped.component_order, (
            "PreAttn", "FusedIndexer", "CoreAttn", "PostAttn", "FusedMoE",
        ))
        self.assertEqual([op.op_id for op in mapped.op_list], [
            "attn.latent_down", "attn.q_rope_projection", "attn.qk_nope", "attn.qk_absorb",
            "attn.indexer_q", "attn.indexer_k_w", "attn.indexer_score",
            "attn.flash_attention", "attn.vo_absorb", "attn.output_projection",
            "moe.ep_dispatch", "moe.moe_fc1", "moe.moe_fc2", "moe.ep_combine",
        ])
        self.assertEqual(tuple(op.op_id for op in mapped.op_list), mapped.root_node.operator_order)
        self.assertEqual(mapped.root_node.batch_size, 256)
        self.assertEqual(mapped.chip_count, 8)
        with self.assertRaises(ValueError):
            mapper.map(HardwareMappingRequest("glm-5.1", InferenceConfig(32, 128, 1)))
        self.assertIsInstance(mapper_for_model("glm-5.1"), GLM51Mapper)
        self.assertEqual(len({type(mapper_for_model(name)) for name in (
            "deepseek-v3", "deepseek-v3.2", "kimi-k2.5", "glm-5.1",
        )}), 4)
        with self.assertRaises(ValueError):
            DeepSeekV32Mapper(LEGACY_HARDWARE_CONFIG).map(InferenceConfig(256, 128, 1))

    def test_projection_shapes_and_partitions(self):
        ops = {op.op_id: op for op in DeepSeekV32Mapper().build_op_list(InferenceConfig(256, 20000, 600))}
        expected = {
            "attn.latent_down": (1, 256, 7168, 2112),
            "attn.q_rope_projection": (1, 256, 1536, 8192),
            "attn.qk_nope": (1, 256, 1536, 16384),
            "attn.qk_absorb": (128, 256, 128, 512),
            "attn.vo_absorb": (128, 256, 512, 128),
            "attn.output_projection": (1, 256, 16384, 7168),
            "attn.indexer_q": (1, 256, 1536, 8192),
            "attn.indexer_k_w": (1, 256, 7168, 192),
        }
        for op_id, shape in expected.items():
            with self.subTest(op=op_id):
                op = ops[op_id]
                self.assertEqual(tuple(getattr(op, axis) for axis in ("B", "M", "K", "N")), shape)
                self.assertEqual(op.partition_chain, PROJECTION_SPLIT)
                self.assertEqual(op.weight_memory.source, WeightSource.DRAM_FLASH)
        self.assertEqual(ops["moe.moe_fc1"].partition_chain, FC1_SPLIT)
        self.assertEqual(ops["moe.moe_fc2"].partition_chain, FC2_SPLIT)
        for key in ("attn.indexer_score", "attn.flash_attention"):
            self.assertEqual(ops[key].partition_chain, ATTENTION_SPLIT)
            self.assertEqual(ops[key].weight_memory.source, WeightSource.FLASH)
        # qk_nope must no longer pay for a 1536 -> 512 projection per head.
        q, absorb = ops["attn.qk_nope"], ops["attn.qk_absorb"]
        flops = sum(2 * op.B * op.M * op.K * op.N for op in (q, absorb))
        self.assertEqual(flops, 2 * 256 * (1536 * 16384 + 128 * 128 * 512))

    def test_deepseek_fold_keeps_only_indexer_key_cache(self):
        mapper = DeepSeekV32Mapper()
        for strategy in MoEParallelStrategy:
            for mtp in (False, True):
                for dtype in ("fp8", "fp16"):
                    with self.subTest(strategy=strategy, mtp=mtp, dtype=dtype):
                        inference = InferenceConfig(
                            256, 20000, 600, moe_parallel_strategy=strategy,
                            mtp_enabled=mtp, kv_cache_dtype=dtype,
                        )
                        global_ops = {op.op_id: op for op in mapper.build_op_list(inference, xfold=True)}
                        mapped = mapper.map(inference, xfold=True)
                        self.assertEqual(mapped.component_order, (
                            "PreAttn", "IndexerKeyCache", "CoreAttn", "PostAttn", "FusedFoldingMoE",
                        ))
                        order = tuple(op.op_id for op in mapped.op_list)
                        self.assertEqual(order, mapped.root_node.operator_order)
                        self.assertEqual(order[3:6], (
                            "attn.qk_absorb", "attn.indexer_k", "attn.flash_attention",
                        ))
                        for removed in ("attn.indexer_q", "attn.indexer_k_w", "attn.indexer_score"):
                            self.assertNotIn(removed, global_ops)
                            self.assertNotIn(removed, mapped.operators)
                        for ops, batch in ((global_ops, 256), (mapped.operators, 32)):
                            key = ops["attn.indexer_k"]
                            self.assertIsInstance(key, BMMOp)
                            self.assertEqual((key.B, key.M, key.K, key.N),
                                             (1, batch * inference.accepted_tokens_per_step, 7168, 128))
                            self.assertEqual(key.partition_chain, PROJECTION_SPLIT)
                            self.assertEqual(key.weight_memory.source, WeightSource.DRAM)
                        graph = TreeParser(mapper.hardware_config).parse(
                            mapped.root_node, mapped.operators, mapped.operator_mappings, layer_count=1,
                        )
                        score_reads = [c.op_id for c in graph.commands
                                       if isinstance(c, PUMemoryReadCmd) and c.data_kind == "indexer_cache"]
                        self.assertEqual(score_reads, ["moe.indexer_score"])

    def test_glm_fold_preserves_full_attention_indexer(self):
        mapped = GLM51Mapper().map(InferenceConfig(256, 20000, 600), xfold=True)
        self.assertEqual(mapped.component_order[1], "FusedIndexer")
        self.assertEqual(mapped.root_node.operator_order[4:7], (
            "attn.indexer_q", "attn.indexer_k_w", "attn.indexer_score",
        ))

    def test_xfold_pre_to_core_attention_uses_only_dram_with_measured_bandwidths(self):
        table = HBFBandwidthTable.from_csv(DEFAULT_HBF_BANDWIDTH_CSV)
        for mapper_type in (DeepSeekV32Mapper, GLM51Mapper, MiniMaxM3Mapper):
            for strategy in MoEParallelStrategy:
                for fold in (False, True):
                    with self.subTest(model=mapper_type.model_name, strategy=strategy, fold=fold):
                        mapper = mapper_type()
                        inference = InferenceConfig(64, 32768, 600, strategy, mtp_enabled=True)
                        global_ops = mapper.build_op_list(inference, xfold=fold)
                        mapped = mapper.map(inference, xfold=fold, hbf_bandwidth_table=table)
                        path = []
                        for op in global_ops:
                            path.append(op.op_id)
                            if isinstance(op, FlashAttentionOp):
                                break
                        self.assertIsInstance(mapped.operators[path[-1]], FlashAttentionOp)
                        if fold:
                            for op_id in path:
                                self.assertEqual(mapped.operators[op_id].weight_memory,
                                                 WeightMemory(WeightSource.DRAM))
                            for op in global_ops[:len(path)]:
                                self.assertEqual(op.weight_memory.source, WeightSource.DRAM)
                            self.assertEqual(mapped.operators["moe.moe_fc1"].weight_memory.source,
                                             WeightSource.DRAM_FLASH)
                            self.assertEqual(mapped.operators["moe.indexer_score"].weight_memory.source,
                                             WeightSource.FLASH)
                            self.assertEqual(mapped.operators["attn.output_projection"].weight_memory.source,
                                             WeightSource.DRAM_FLASH)
                        else:
                            self.assertEqual(mapped.operators[path[0]].weight_memory.source,
                                             WeightSource.DRAM_FLASH)
                            self.assertEqual(mapped.operators[path[-1]].weight_memory.source,
                                             WeightSource.FLASH)
                        graph = TreeParser().parse(mapped.root_node, mapped.operators,
                                                   mapped.operator_mappings, layer_count=2)
                        reads = [c for c in graph.commands if isinstance(c, PUMemoryReadCmd)
                                 and c.op_id in path]
                        self.assertTrue(reads)
                        self.assertEqual({c.port for c in reads},
                                         {"dram"} if fold else {"dram", "flash"})

    def test_fold_shapes_and_memory_sources(self):
        mapped = DeepSeekV32Mapper().map(InferenceConfig(256, 20000, 600), xfold=True)
        self.assertEqual(mapped.component_order[-1], "FusedFoldingMoE")
        self.assertEqual([op.op_id for op in mapped.op_list[-8:]], [
            "moe.ep_dispatch", "moe.moe_fc1", "moe.fold_latent_q_indexer_k_w",
            "moe.ep_fold_combine", "moe.indexer_q", "moe.indexer_score",
            "moe.moe_fc2", "moe.ep_combine",
        ])
        folded = mapped.operators["moe.fold_latent_q_indexer_k_w"]
        self.assertEqual((folded.B, folded.M, folded.K, folded.N), (32, 8, 2048, 1728))
        self.assertEqual(folded.partition_chain, FC2_SPLIT)
        self.assertEqual(mapped.operators["moe.indexer_q"].partition_chain, FC2_SPLIT)
        self.assertEqual(mapped.operators["moe.indexer_score"].partition_chain, ATTENTION_SPLIT)
        fc2 = mapped.operators["moe.moe_fc2"]
        self.assertEqual(fc2.weight_memory.source, WeightSource.DRAM)
        self.assertEqual(fc2.flash_prefetch_bytes_per_request, 2048 * 576)
        self.assertEqual(mapped.operators["attn.flash_attention"].weight_memory.source, WeightSource.DRAM)

    def test_fold_prefetch_forks_beside_fc2_and_joins_next_attention(self):
        mapper = DeepSeekV32Mapper()
        mapped = mapper.map(InferenceConfig(256, 128, 1, kv_cache_dtype="fp16"), xfold=True)
        tree = replace(mapped.root_node, root=replace(mapped.root_node.root, split=2))
        graph = TreeParser(mapper.hardware_config).parse(tree, mapped.operators, mapped.operator_mappings, layer_count=2)
        result = Simulator(mapper.hardware_config).run(graph, mapping_result=mapped)
        commands = {c.cmd_id: c for c in result.command_results}
        for layer in (0, 1):
            prefetch = [c for c in graph.commands if isinstance(c, PUMemoryReadCmd)
                        and c.data_kind == "next_layer_kv" and graph.traces[c.cmd_id].layer_index == layer]
            self.assertEqual(len(prefetch), 2)
            self.assertEqual(sum(c.size_bytes for c in prefetch) * 8 * 2048, 256 * 128 * 576 * 2)
            for read in prefetch:
                trace = graph.traces[read.cmd_id]
                core = f"layer{layer}.i{trace.instance_index}.core"
                output = f"layer{layer}.i{trace.instance_index}.output"
                self.assertEqual(set(graph.predecessors(read.cmd_id)),
                                 set(graph.predecessors(core)))
                self.assertNotIn((core, read.cmd_id), graph.start_edges)
                self.assertNotIn((read.cmd_id, output), graph.edges)
                if layer == 0:
                    targets = [c for c in graph.commands
                               if c.op_id == "attn.flash_attention" and c.cmd_id.endswith(".core")
                               and graph.traces[c.cmd_id].layer_index == 1
                               and max(trace.token_start, graph.traces[c.cmd_id].token_start)
                               < min(trace.token_stop, graph.traces[c.cmd_id].token_stop)]
                    successors = {s for c in targets for s in graph.successors(c.cmd_id)}
                    self.assertEqual(set(graph.successors(read.cmd_id)), successors)
                    for successor in successors:
                        self.assertGreaterEqual(commands[successor].start_time_ns,
                                                commands[read.cmd_id].end_time_ns)
                else:
                    self.assertEqual(graph.successors(read.cmd_id), ())
                fc2_reads = [c for c in graph.commands if isinstance(c, PUMemoryReadCmd)
                             and c.op_id == "moe.moe_fc2" and c.data_kind == "weight"]
                self.assertTrue(all(c.port == "dram" for c in fc2_reads))
        self.assertTrue(any(a.startswith("layer0.") and b.startswith("layer1.") for a, b in graph.edges))

    def test_models_context_dtype_mtp_and_small_batches(self):
        for mapper_type in (DeepSeekV32Mapper, GLM51Mapper):
            mapper = mapper_type()
            for fold in (False, True):
                for batch, context, dtype, mtp in ((1, 17, "fp8", False), (32, 4096, "fp16", True)):
                    with self.subTest(model=mapper.model_name, fold=fold, batch=batch):
                        mapped = mapper.map(InferenceConfig(batch, context, 1, kv_cache_dtype=dtype, mtp_enabled=mtp), xfold=fold)
                        graph = TreeParser(mapper.hardware_config).parse(mapped.root_node, mapped.operators, mapped.operator_mappings, layer_count=1)
                        result = Simulator(mapper.hardware_config).run(graph, mapping_result=mapped)
                        self.assertGreater(result.total_time_ns, 0)
                        self.assertEqual(mapped.operators["attn.flash_attention"].qk_N, min(context, 2048))
                        self.assertEqual(sum(mapped.expert_token_loads), batch * (2 if mtp else 1) * 8)

    def test_mtp_packs_query_heads_and_preserves_request_cache_tiles(self):
        for mapper_type in (DeepSeekV32Mapper, GLM51Mapper):
            mapper = mapper_type()
            for fold in (False, True):
                for batch in (1, 32, 256):
                    with self.subTest(model=mapper.model_name, fold=fold, batch=batch):
                        off, on = [mapper.map(InferenceConfig(batch, 20000, 1, mtp_enabled=mtp),
                                              xfold=fold) for mtp in (False, True)]
                        graphs = [TreeParser(mapper.hardware_config).parse(
                            r.root_node, r.operators, r.operator_mappings, layer_count=1,
                        ) for r in (off, on)]
                        ids = ["attn.flash_attention"]
                        if not fold or mapper.model_name == "glm-5.1":
                            ids.append("attn.indexer_score")
                        if fold:
                            ids.append("moe.indexer_score")
                        for op_id in ids:
                            a, b = off.operators[op_id], on.operators[op_id]
                            self.assertEqual(a.B, b.B)
                            self.assertEqual(b.M, a.M * 2)
                            self.assertEqual(on.operator_mappings[op_id].cache_query_group_size, 1)
                            index = on.root_node.operator_order.index(op_id)
                            cores = [g.command(f"i{index}.core") for g in graphs]
                            self.assertEqual(cores[1].B, cores[0].B)
                            self.assertEqual(cores[1].total_flops, 2 * cores[0].total_flops)
                            reads = [[(c.data_kind, c.port, c.size_bytes) for c in g.commands
                                      if isinstance(c, PUMemoryReadCmd) and c.op_id == op_id]
                                     for g in graphs]
                            self.assertTrue(reads[0])
                            self.assertEqual(reads[0], reads[1])
                            for suffix in ("input", "output"):
                                rows = [g.command(f"i{index}.{suffix}").io_bytes_per_row for g in graphs]
                                self.assertEqual(rows[1], tuple(2 * n for n in rows[0]))

    def test_optimizer_keeps_fold_when_rebinding_workload(self):
        from srammachine.search import SimulatedAnnealingConfig, SplitTreeOptimizer
        request = HardwareMappingRequest("deepseek-v3.2", InferenceConfig(32, 32, 1), xfold=True)
        optimizer = SplitTreeOptimizer()
        result = optimizer.run(request, config=SimulatedAnnealingConfig(write_result=False))
        order = result.initial_evaluation.split_tree.operator_order
        self.assertIn("moe.fold_latent_q_indexer_k_w", order)
        self.assertIn("attn.indexer_k", order)
        self.assertNotIn("attn.indexer_score", order)
        self.assertEqual(optimizer.hardware_config.chip_count, 8)


class PUMemoryTests(unittest.TestCase):
    def test_ports_are_instantiated_per_pu_and_only_same_port_contends(self):
        from Desim import SimSession

        hardware, _ = self._graph([WeightMemory()])
        commands = [
            PUMemoryReadCmd("dram0", "op", "chip0.die0.pu0.dram", 1000, "dram", 1e9),
            PUMemoryReadCmd("dram1", "op", "chip0.die0.pu0.dram", 1000, "dram", 1e9),
            PUMemoryReadCmd("flash0", "op", "chip0.die0.pu0.flash", 3000, "flash", 3e9),
            PUMemoryReadCmd("other_pu", "op", "chip0.die0.pu1.dram", 1000, "dram", 1e9),
        ]
        SimSession.reset()
        SimSession.init()
        try:
            executor = GraphExecutor(CommandGraph(commands), hardware)
            self.assertEqual(set(executor.stages), {
                "chip0.die0.pu0.dram", "chip0.die0.pu0.flash",
                "chip0.die0.pu1.dram", "chip0.die0.pu1.flash",
            })
            SimSession.scheduler.run()
            executions = executor.result().executions
            self.assertEqual(executions["dram1"].start_time_ns, 1000)
            for key in ("dram0", "flash0", "other_pu"):
                self.assertEqual(executions[key].start_time_ns, 0)
                self.assertEqual(executions[key].end_time_ns, 1000)
        finally:
            GraphExecutor._release_session(False)

    def _graph(self, policies):
        pu = replace(DEFAULT_HARDWARE_CONFIG.chip.logic_die.processing_unit,
                     dram_bandwidth_bytes_per_second=1e9, flash_bandwidth_bytes_per_second=3e9)
        die = replace(DEFAULT_HARDWARE_CONFIG.chip.logic_die, processing_unit=pu)
        hardware = replace(DEFAULT_HARDWARE_CONFIG, chip=replace(DEFAULT_HARDWARE_CONFIG.chip, logic_die=die))
        ops = {f"op{i}": BMMOp(f"op{i}", 1, 1, 1000, 1000,
                              input_source="none", output_destination="none", weight_memory=policy)
               for i, policy in enumerate(policies)}
        mappings = {key: OperatorMapping("M", "chip0.die0.pu0") for key in ops}
        tree = PipeTree(1, tuple(ops), GroupNode(tuple(OpNode(key) for key in ops)))
        return hardware, TreeParser(hardware).parse(tree, ops, mappings, layer_count=1)

    def test_single_dual_and_override_bandwidths(self):
        policies = [WeightMemory(WeightSource.DRAM), WeightMemory(WeightSource.FLASH),
                    WeightMemory(), WeightMemory(flash_bandwidth_bytes_per_second=9e9)]
        hardware, graph = self._graph(policies)
        result = Simulator(hardware).run(graph)
        execution = {c.cmd_id: c for c in result.command_results}
        for index, expected in enumerate((1_000_000, 333334, 250000, 100000)):
            core = graph.command(f"i{index}.core")
            self.assertEqual(core.memory_duration_ns, expected)
            reads = [c for c in graph.commands if isinstance(c, PUMemoryReadCmd) and c.op_id == f"op{index}"]
            self.assertAlmostEqual(sum(c.size_bytes for c in reads), 1_000_000)
            self.assertEqual({execution[c.cmd_id].duration_ns for c in reads}, {expected})
            self.assertEqual(len(reads), 1 if index < 2 else 2)
        rates = [c.bandwidth_bytes_per_second for c in graph.commands if isinstance(c, PUMemoryReadCmd) and c.port == "dram"]
        self.assertEqual(set(rates), {1e9})
        self.assertEqual({c.resource_id for c in graph.commands if isinstance(c, PUMemoryReadCmd)},
                         {"chip0.die0.pu0.dram", "chip0.die0.pu0.flash"})

    def test_memory_policies_validate_and_overrides_are_per_operator(self):
        for rate in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                WeightMemory(flash_bandwidth_bytes_per_second=rate)
        mapper = DeepSeekV32Mapper()
        overridden = mapper.map(InferenceConfig(32, 128, 1), memory_overrides={
            "attn.qk_nope": WeightMemory(flash_bandwidth_bytes_per_second=10e12),
        })
        self.assertEqual(overridden.operators["attn.qk_nope"].weight_memory.flash_bandwidth_bytes_per_second, 10e12)
        self.assertIsNone(overridden.operators["attn.qk_absorb"].weight_memory.flash_bandwidth_bytes_per_second)
        with self.assertRaises(ValueError):
            mapper.build_op_list(InferenceConfig(32, 128, 1), memory_overrides={"typo": WeightMemory()})


if __name__ == "__main__":
    unittest.main()
