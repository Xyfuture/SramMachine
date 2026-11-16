"""MiniMax M3 dimensions, grouped caches, block selection and simulator paths."""

from copy import deepcopy
from dataclasses import replace
import itertools
import json
from pathlib import Path
import tempfile
import unittest

from srammachine.commands import FlashMSACmd, MiniMaxFusedIndexerScoreCmd, PUMemoryReadCmd, VectorCmd
from srammachine.frontend.modules import (
    BMMOp, FlashAttentionOp, FlashMSAOp, FusedIndexerScoreOp,
    MiniMaxFusedIndexerScoreOp, PartitionChain, WeightMemory, WeightSource,
)
from srammachine.inference import InferenceConfig, MoEParallelStrategy
from srammachine.mapping import (
    HardwareMappingRequest, MiniMaxM3Config, MiniMaxM3Mapper,
    load_model_config, mapper_for_model,
)
from srammachine.mapping.components import ATTENTION_SPLIT, FC2_SPLIT
from srammachine.pipetree import GroupNode, OpNode, OperatorMapping, PipeTree, TreeParser
from srammachine.simulator import Simulator, export_perfetto_trace, steady_pu_metrics


def parse_op(op, split=1, layers=1):
    tree = PipeTree(op.B, (op.op_id,), GroupNode((OpNode(op.op_id),), split))
    return TreeParser().parse(tree, {op.op_id: op}, {
        op.op_id: OperatorMapping("B", "chip0.die0.pu0"),
    }, layer_count=layers)


class MiniMaxPartitionTests(unittest.TestCase):
    def test_bhs_partition_coverage_against_element_enumeration(self):
        shape = dict(B=3, H=3, S=5)
        for axes in itertools.product("BHS", repeat=4):
            with self.subTest(axes=axes):
                chain = PartitionChain(tuple(zip(axes, ("p", "q", "p", "q"), (2, 3, 2, 2))))
                covered, expected = set(), []
                for p in range(chain.grid_shape[0]):
                    sequences = {}
                    for q in range(chain.grid_shape[1]):
                        tile = chain.tile_ranges(shape, p, q)
                        elements = set(itertools.product(*(range(*tile[a]) for a in ("B", "H", "S"))))
                        self.assertFalse(covered & elements)
                        covered.update(elements)
                        for b, h, s in elements:
                            sequences.setdefault((b, h), set()).add(s)
                    complete = sum(len(s) == shape["S"] for s in sequences.values())
                    expected.append((complete, len(sequences) - complete))
                self.assertEqual(len(covered), 45)
                self.assertEqual(chain.attention_groups_per_row(shape), tuple(expected))

    def test_head_partition_reduces_cache_and_compute_without_cross_head_merge(self):
        op = FlashMSAOp("a", 2, 3, 8, 9, 9, 4, H=4,
                        partition_chain=PartitionChain((("H", "p", 4), ("S", "q", 3))),
                        weight_memory=WeightMemory(WeightSource.FLASH))
        graph = parse_op(op)
        core = graph.command("i0.core")
        self.assertIsInstance(core, FlashMSACmd)
        self.assertEqual((core.B, core.H, core.M, core.qk_N), (2, 1, 3, 3))
        self.assertEqual(core.total_flops, 2 * 2 * 1 * 3 * 3 * (8 + 4))
        reads = [c for c in graph.commands if isinstance(c, PUMemoryReadCmd)]
        self.assertEqual([c.size_bytes for c in reads], [2 * 1 * 3 * (8 + 4)])
        output = graph.command("i0.output")
        self.assertEqual(output.attention_complete_batches_per_row, (2,) * 4)
        self.assertEqual(output.attention_partial_batches_per_row, (0,) * 4)
        self.assertEqual(output.io_bytes_per_row, (2 * 3 * 4 * 2,) * 4)
        self.assertEqual(output.attention_count_unit, "request_kv_head")

    def test_sequence_rows_send_states_per_group_and_ignore_empty_heads(self):
        op = FlashMSAOp("a", 2, 3, 8, 9, 9, 4, H=3,
                        partition_chain=PartitionChain((("S", "p", 2), ("H", "q", 4))))
        volumes = op.communication_volumes()
        self.assertEqual(volumes["partial_batches_per_row"], (6, 6))
        self.assertEqual(volumes["output_bytes_per_row"], (6 * 3 * 6 * 4,) * 2)
        self.assertEqual(volumes["input_bytes_per_row"], (6 * 3 * 8 * 2,) * 2)

    def test_indexer_keeps_heads_separate_and_deduplicates_boundary_blocks(self):
        # Three token shards [0,3), [3,6), [6,9) overlap block IDs.
        op = MiniMaxFusedIndexerScoreOp("i", 2, 8, 16, 9, block_size=4, top_k=2,
                                       queries_per_request=2,
                                       partition_chain=PartitionChain((("S", "q", 3),)))
        graph = parse_op(op)
        core = graph.command("i0.core")
        self.assertIsInstance(core, MiniMaxFusedIndexerScoreCmd)
        self.assertEqual(core.output_candidates, 2 * 8)
        self.assertEqual(core.total_flops, 2 * 2 * 8 * 16 * 3)
        output = graph.command("i0.output")
        self.assertEqual(output.topk_candidates_per_row, (2 * 8 * 2,))
        self.assertEqual(output.topk_selection_unit, "block")
        reads = [c for c in graph.commands if isinstance(c, PUMemoryReadCmd)]
        self.assertEqual([c.size_bytes for c in reads], [2 * 16 * 3])
        many = replace(op, top_k=16).communication_volumes()
        self.assertEqual(many["topk_candidates_per_row"], (2 * 8 * 3,))

    def test_indexer_block_coverage_matches_enumeration(self):
        for axes in itertools.product("BS", repeat=3):
            chain = PartitionChain(tuple(zip(axes, ("p", "q", "q"), (3, 2, 2))))
            op = MiniMaxFusedIndexerScoreOp("i", 3, 4, 8, 11, top_k=3, block_size=4,
                                           partition_chain=chain)
            expected = []
            for p in range(chain.grid_shape[0]):
                by_request = {}
                for q in range(chain.grid_shape[1]):
                    tile = chain.tile_ranges(dict(B=3, S=11), p, q)
                    for b, s in itertools.product(range(*tile["B"]), range(*tile["S"])):
                        by_request.setdefault(b, set()).add(s // 4)
                expected.append(sum(4 * min(3, len(ids)) for ids in by_request.values()))
            self.assertEqual(op.communication_volumes()["topk_candidates_per_row"], tuple(expected))

    def test_head_split_does_not_leak_into_dsa_bmm_or_indexer(self):
        chain = PartitionChain((("H", "p", 2),))
        constructors = (
            lambda: BMMOp("x", 1, 2, 3, 4, partition_chain=chain),
            lambda: FusedIndexerScoreOp("x", 1, 2, 3, 4, partition_chain=chain),
            lambda: MiniMaxFusedIndexerScoreOp("x", 1, 2, 3, 4, partition_chain=chain),
            lambda: FlashAttentionOp("x", 1, 2, 3, 4, 4, 5, partition_chain=chain),
        )
        for construct in constructors:
            with self.assertRaises(ValueError):
                construct()
        for kwargs in ({"H": 0}, {"partition_chain": PartitionChain((("M", "q", 2),))}):
            with self.assertRaises(ValueError):
                FlashMSAOp("x", 1, 2, 3, 4, 4, 5, **kwargs)

    def test_microbatch_boundaries_and_memory_join(self):
        op = FlashMSAOp("a", 8, 2, 8, 9, 9, 4, H=4,
                        partition_chain=PartitionChain((("B", "p", 2), ("H", "q", 4))),
                        weight_memory=WeightMemory(flash_bandwidth_bytes_per_second=9e12))
        graph = parse_op(op, split=2, layers=2)
        self.assertEqual(graph.command("layer0.i0.core").B, 2)
        self.assertEqual(graph.command("layer0.i0.core").H, 1)
        self.assertIn(("layer0.i0.output", "layer1.i0.input"), graph.edges)
        reads = [c for c in graph.commands if isinstance(c, PUMemoryReadCmd)]
        first_reads = [c for c in reads if c.cmd_id.startswith("layer0.i0.")]
        self.assertEqual(len(first_reads), 2)
        self.assertEqual(sum(c.size_bytes for c in first_reads), 2 * 1 * 9 * 12)
        for read in first_reads:
            self.assertIn(("layer0.i0.core", read.cmd_id), graph.start_edges)
            self.assertIn((read.cmd_id, "layer0.i0.output"), graph.edges)
        result = Simulator().run(graph)
        times = {c.cmd_id: c for c in result.command_results}
        for read in first_reads:
            self.assertGreaterEqual(times["layer0.i0.output"].start_time_ns, times[read.cmd_id].end_time_ns)


class MiniMaxMapperTests(unittest.TestCase):
    def test_config_and_layer_masks(self):
        config = load_model_config("minimax-m3")
        self.assertIsInstance(config, MiniMaxM3Config)
        self.assertEqual((config.hidden_size, config.num_attention_heads, config.num_key_value_heads), (6144, 64, 4))
        self.assertEqual((config.num_experts, config.top_k), (128, 4))
        self.assertEqual(config.sparse_attention_layers, (False,) * 3 + (True,) * 57)
        self.assertEqual((config.sparse_topk_blocks, config.sparse_block_size), (16, 128))
        path = Path(__file__).resolve().parents[1] / "model_cards/minimax-m3.json"
        self.assertEqual(MiniMaxM3Mapper.from_json(path).model_config, config)
        self.assertIsInstance(mapper_for_model("minimax-m3"), MiniMaxM3Mapper)
        data = json.loads(path.read_text())
        for field, value in (("num_key_value_heads", 3), ("num_experts_per_tok", 129),
                             ("moe_layer_freq", [1]), ("head_dim", 0)):
            bad = deepcopy(data)
            bad["text_config"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                MiniMaxM3Config.from_dict(bad)

    def test_projection_shapes_and_dense_shared_routed_experts(self):
        mapper = MiniMaxM3Mapper()
        mapped = mapper.map(InferenceConfig(256, 20000, 1))
        ops = mapped.operators
        qkv = ops["attn.qkv_projection"]
        self.assertIs(type(qkv), BMMOp)
        self.assertEqual((qkv.B, qkv.M, qkv.K, qkv.N), (1, 32, 6144, 9216))
        self.assertEqual((ops["attn.indexer_q"].K, ops["attn.indexer_q"].N), (6144, 512))
        self.assertEqual(ops["attn.indexer_k"].N, 128)
        self.assertEqual(ops["attn.indexer_score"].partition_chain, ATTENTION_SPLIT)
        self.assertEqual((ops["attn.flash_msa"].H, ops["attn.flash_msa"].M), (4, 16))
        self.assertEqual(ops["attn.flash_msa"].qk_N, 2048)
        self.assertEqual(ops["moe.shared_fc1"].M, 32)
        self.assertEqual(ops["moe.moe_fc1"].M, 8)
        self.assertEqual(ops["moe.moe_fc1"].B, 16)
        self.assertEqual(mapped.represented_layer_count, 57)
        dense = mapped.additional_layer_mappings[0]
        self.assertEqual(dense.represented_layer_count, 3)
        self.assertNotIn("attn.indexer_score", dense.operators)
        self.assertNotIn("moe.ep_dispatch", dense.operators)
        self.assertEqual(dense.operators["attn.flash_msa"].qk_N, 20000)
        self.assertEqual(dense.operators["ffn.fc1"].N, 24576)
        self.assertEqual(dense.operators["ffn.fc2"].K, 12288)
        graph = TreeParser().parse(dense.root_node, dense.operators, dense.operator_mappings)
        simulator = Simulator()
        result = simulator.run(graph, mapping_result=dense)
        self.assertEqual(result.model_layer_count, 3)
        self.assertEqual(result.latency_ns, 3 * result.pipeline_layer_time_ns)
        self.assertEqual(simulator.run_latency(graph, mapping_result=dense), result.latency_ns)
        with self.assertRaises(ValueError):
            mapper.map(HardwareMappingRequest("deepseek-v3.2", InferenceConfig(32, 128, 1)))

    def test_tp_ep_short_long_dtype_and_mtp(self):
        for (strategy, fold), history, dtype, batch in itertools.product(
            ((MoEParallelStrategy.TP, False), (MoEParallelStrategy.EP, False),
             (MoEParallelStrategy.EP, True)),
            (17, 20000), ("fp8", "fp16"), (1, 32),
        ):
            with self.subTest(strategy=strategy, history=history, dtype=dtype, batch=batch, fold=fold):
                mapped = MiniMaxM3Mapper().map(InferenceConfig(batch, history, 1, strategy, dtype, True),
                                              xfold=fold)
                graph = TreeParser().parse(mapped.root_node, mapped.operators, mapped.operator_mappings, layer_count=1)
                result = Simulator().run(graph, mapping_result=mapped)
                self.assertGreater(result.latency_ns, 0)
                self.assertEqual(mapped.operators["attn.flash_msa"].qk_N, min(history, 2048))
                self.assertEqual(sum(mapped.expert_token_loads), batch * 2 * 4)
                self.assertFalse(any(isinstance(c, VectorCmd) for c in graph.commands))

    def test_mtp_doubles_compute_candidates_but_not_cache_traffic(self):
        graphs = []
        for mtp in (False, True):
            mapped = MiniMaxM3Mapper().map(InferenceConfig(32, 20000, 1, mtp_enabled=mtp))
            graphs.append(TreeParser().parse(mapped.root_node, mapped.operators, mapped.operator_mappings, layer_count=1))
        for op_id in ("attn.indexer_score", "attn.flash_msa"):
            cores = [next(c for c in g.commands if c.op_id == op_id and c.cmd_id.endswith(".core")) for g in graphs]
            self.assertEqual(cores[1].total_flops, 2 * cores[0].total_flops)
            reads = [[c.size_bytes for c in g.commands if c.op_id == op_id and isinstance(c, PUMemoryReadCmd)] for g in graphs]
            self.assertEqual(reads[0], reads[1])
            if op_id.endswith("indexer_score"):
                self.assertEqual(cores[1].output_candidates, 2 * cores[0].output_candidates)

    def test_weighted_latency_fast_path_and_perfetto(self):
        mapped = MiniMaxM3Mapper().map(InferenceConfig(32, 20000, 1))
        graph = TreeParser().parse(mapped.root_node, mapped.operators, mapped.operator_mappings)
        simulator = Simulator()
        result = simulator.run(graph, mapping_result=mapped)
        components = result.model_latency_components
        self.assertEqual([(name, count) for name, count, _ in components], [("sparse", 57), ("dense", 3)])
        self.assertEqual(result.latency_ns, sum(count * time for _, count, time in components))
        self.assertEqual(result.latency_ns, simulator.run_latency(graph, mapping_result=mapped))
        self.assertEqual(len(simulator._additional_layer_cache), 1)
        self.assertAlmostEqual(result.throughput_tokens_per_second, 32e9 / result.latency_ns)
        self.assertGreater(steady_pu_metrics(result.command_results)["steady_pu_busy_time_ns"], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = export_perfetto_trace(result, mapping_result=mapped, output_dir=directory, trace_label="m3")
            trace = json.loads(path.read_text())
        self.assertEqual([c["layer_count"] for c in trace["otherData"]["model_latency_components"]], [57, 3])
        events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
        self.assertTrue(any(e["args"].get("selection_unit") == "block" for e in events))
        self.assertTrue(any(e["args"].get("cache_layout") == "separate_k_v_per_group" for e in events))

    def test_search_accepts_m3(self):
        from srammachine.search import SimulatedAnnealingConfig, SplitTreeOptimizer
        request = HardwareMappingRequest("minimax-m3", InferenceConfig(32, 128, 1))
        result = SplitTreeOptimizer().run(request, config=SimulatedAnnealingConfig(write_result=False))
        evaluation = result.initial_evaluation
        self.assertIn("attn.flash_msa", evaluation.split_tree.operator_order)
        self.assertEqual(evaluation.latency_ns, evaluation.simulation_result.latency_ns)


class MiniMaxFoldTests(unittest.TestCase):
    def test_all_query_contributions_and_single_group_score(self):
        mapper = MiniMaxM3Mapper()
        inference = InferenceConfig(256, 20000, 1, moe_parallel_strategy=MoEParallelStrategy.EP)
        ops = {op.op_id: op for op in mapper.build_op_list(inference, xfold=True)}
        expected = {
            "moe.fold_base_indexer_q": (1, 256, 6144, 512),
            "moe.fold_shared_indexer_q": (1, 256, 3072, 512),
            "moe.fold_indexer_q": (128, 8, 3072, 512),
            "moe.indexer_score": (256, 4, 128, 20000),
        }
        for name, shape in expected.items():
            with self.subTest(op=name):
                self.assertEqual(tuple(getattr(ops[name], a) for a in ("B", "M", "K", "N")), shape)
        order = list(ops)
        for producer, consumer in (
            ("moe.shared_fc1", "moe.fold_shared_indexer_q"),
            ("moe.fold_shared_indexer_q", "moe.shared_fc2"),
            ("moe.fold_base_indexer_q", "moe.indexer_score"),
            ("moe.shared_fc2", "moe.indexer_score"),
            ("moe.moe_fc1", "moe.fold_indexer_q"),
            ("moe.fold_indexer_q", "moe.indexer_score"),
            ("moe.indexer_score", "moe.moe_fc2"),
        ):
            self.assertLess(order.index(producer), order.index(consumer))
        self.assertNotIn("attn.indexer_q", ops)
        self.assertNotIn("attn.indexer_score", ops)
        self.assertNotIn("moe.indexer_q", ops)
        self.assertEqual((ops["attn.indexer_k"].K, ops["attn.indexer_k"].N), (6144, 128))
        self.assertEqual(ops["moe.fold_indexer_q"].partition_chain, FC2_SPLIT)
        score = ops["moe.indexer_score"]
        self.assertIsInstance(score, MiniMaxFusedIndexerScoreOp)
        self.assertEqual((score.top_k, score.block_size), (16, 128))
        # The extra expert weights have 1/12 of the down projection's size.
        down, folded = ops["moe.moe_fc2"], ops["moe.fold_indexer_q"]
        self.assertEqual(folded.B * folded.K * folded.N * 12, down.B * down.K * down.N)

    def test_ep_fold_combine_before_score(self):
        mapped = MiniMaxM3Mapper().map(InferenceConfig(256, 4096, 1), xfold=True)
        op = mapped.operators["moe.fold_indexer_q"]
        self.assertEqual((op.B, op.M, op.K, op.N), (16, 8, 3072, 512))
        self.assertEqual(mapped.operator_mappings[op.op_id].batch_scaling_unit_count, 8)
        graph = TreeParser().parse(mapped.root_node, mapped.operators,
                                  mapped.operator_mappings, layer_count=1)
        combine = mapped.operators["moe.ep_fold_combine"]
        self.assertEqual(sum(map(sum, combine.transfer_bytes)), 256 * 4 * 512 * 2)
        dispatch = mapped.operators["moe.ep_dispatch"].transfer_bytes
        self.assertEqual(combine.transfer_bytes,
                         tuple(tuple(n // 6144 * 512 * 2 for n in row) for row in zip(*dispatch)))
        commands = {c.cmd_id: c for c in graph.commands}
        pairs = {(commands[a].op_id, commands[b].op_id) for a, b in graph.edges}
        self.assertIn(("moe.fold_indexer_q", "moe.ep_fold_combine"), pairs)
        self.assertIn(("moe.ep_fold_combine", "moe.indexer_score"), pairs)

    def test_prefetch_volume_overlap_and_layer_dependency(self):
        for dtype, mtp in itertools.product(("fp8", "fp16"), (False, True)):
            with self.subTest(dtype=dtype, mtp=mtp):
                inference = InferenceConfig(64, 4096, 1, kv_cache_dtype=dtype, mtp_enabled=mtp)
                mapped = MiniMaxM3Mapper().map(inference, xfold=True)
                expected_bytes = 2048 * 4 * (128 + 128) * inference.kv_cache_bytes_per_element
                fc2 = mapped.operators["moe.moe_fc2"]
                self.assertEqual(fc2.flash_prefetch_bytes_per_request, expected_bytes)
                self.assertEqual(fc2.weight_memory.source, WeightSource.DRAM)
                self.assertEqual(mapped.operators["attn.flash_msa"].weight_memory.source, WeightSource.DRAM)
                graph = TreeParser().parse(mapped.root_node, mapped.operators,
                                          mapped.operator_mappings, layer_count=2)
                result = Simulator().run(graph, mapping_result=mapped)
                times = {c.cmd_id: c for c in result.command_results}
                for layer in (0, 1):
                    reads = [c for c in graph.commands if isinstance(c, PUMemoryReadCmd)
                             and c.data_kind == "next_layer_kv" and graph.traces[c.cmd_id].layer_index == layer]
                    self.assertEqual(len(reads), 1)
                    self.assertEqual(sum(c.size_bytes for c in reads) * 8 * 2048, 64 * expected_bytes)
                    for read in reads:
                        instance = graph.traces[read.cmd_id].instance_index
                        core, output = f"layer{layer}.i{instance}.core", f"layer{layer}.i{instance}.output"
                        self.assertEqual(set(graph.predecessors(read.cmd_id)),
                                         set(graph.predecessors(core)))
                        self.assertNotIn((core, read.cmd_id), graph.start_edges)
                        self.assertNotIn((read.cmd_id, output), graph.edges)
                        if layer == 0:
                            attention = next(c for c in graph.commands
                                             if isinstance(c, FlashMSACmd)
                                             and graph.traces[c.cmd_id].layer_index == 1)
                            self.assertEqual(set(graph.successors(read.cmd_id)),
                                             set(graph.successors(attention.cmd_id)))
                            for successor in graph.successors(read.cmd_id):
                                self.assertGreaterEqual(times[successor].start_time_ns,
                                                        times[read.cmd_id].end_time_ns)
                        else:
                            self.assertEqual(graph.successors(read.cmd_id), ())
                self.assertTrue(any(a.startswith("layer0.") and b.startswith("layer1.") for a, b in graph.edges))
                self.assertFalse(any(isinstance(c, VectorCmd) for c in graph.commands))
                self.assertEqual([(name, count) for name, count, _ in result.model_latency_components],
                                 [("sparse", 57), ("dense", 3)])

    def test_dense_prefix_stays_normal_and_fold_overrides_are_preserved(self):
        mapper = MiniMaxM3Mapper()
        request = HardwareMappingRequest("minimax-m3", InferenceConfig(32, 128, 1), xfold=True)
        policy = WeightMemory(WeightSource.FLASH, flash_bandwidth_bytes_per_second=3e9)
        mapped = mapper.map(request, memory_overrides={"moe.fold_indexer_q": policy})
        self.assertEqual(mapped.operators["moe.fold_indexer_q"].weight_memory, policy)
        dense = mapped.additional_layer_mappings[0]
        self.assertFalse(dense.request.xfold)
        self.assertEqual(dense.operators["attn.flash_msa"].weight_memory.source, WeightSource.FLASH)
        self.assertFalse(any("fold" in key for key in dense.operators))
        with self.assertRaisesRegex(ValueError, "sparse MSA/MoE"):
            MiniMaxM3Mapper(layer_type="dense").map(request)

    def test_sequential_trace_preserves_fold(self):
        request = HardwareMappingRequest("minimax-m3", InferenceConfig(32, 128, 1), xfold=True)
        mapped = MiniMaxM3Mapper().map(request)
        graph = TreeParser().parse(mapped.root_node, mapped.operators, mapped.operator_mappings)
        result = Simulator().run(graph, mapping_result=mapped)
        with tempfile.TemporaryDirectory() as directory:
            path = export_perfetto_trace(result,
                                         mapping_result=mapped, output_dir=directory, trace_label="m3-fold")
            trace = json.loads(path.read_text())
        self.assertTrue(trace["otherData"]["xfold"])
        events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
        self.assertTrue(any(e["args"].get("selection_unit") == "block" for e in events))
        self.assertTrue(any(e["args"].get("data_kind") == "next_layer_kv" for e in events))


if __name__ == "__main__":
    unittest.main()
