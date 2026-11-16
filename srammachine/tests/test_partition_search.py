"""Partition enumeration, cached search mappings, and exact trace replay."""

from dataclasses import replace
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from srammachine.commands import PUMemoryReadCmd
from srammachine.frontend.modules import PartitionChain, WeightMemory, WeightSource
from srammachine.inference import InferenceConfig, MoEParallelStrategy
from srammachine.mapping import (
    DEFAULT_HBF_BANDWIDTH_CSV, HBFBandwidthTable, HardwareMappingRequest,
    MiniMaxM3Mapper, mapper_for_model,
)
from srammachine.pipetree import GroupNode, OpNode, RootNode, TreeParser
from srammachine.script.run_sa import (
    SACase, _make_cases, _run_case, build_parser, main, write_case_trace, write_csv, write_model_json,
)
from srammachine.search import (
    SimulatedAnnealingConfig, SplitTreeOptimizer, apply_partition_chains,
    load_pareto_partition_chains, load_pareto_split_tree, optimize_partition_chains,
    partition_chains_to_dict, split_tree_to_dict,
)
from srammachine.simulator import Simulator
from srammachine.search.partition import _attention_candidates


class PartitionSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapper = mapper_for_model("deepseek-v3.2")
        cls.table = HBFBandwidthTable.from_csv(DEFAULT_HBF_BANDWIDTH_CSV)
        cls.request = HardwareMappingRequest("deepseek-v3.2", InferenceConfig(
            256, 32768, 600, mtp_enabled=True,
        ))
        cls.original = cls.mapper.map(cls.request, hbf_bandwidth_table=cls.table)
        cls.selected = optimize_partition_chains(cls.original, cls.mapper.hardware_config)

    def simulate(self, mapping):
        graph = TreeParser(self.mapper.hardware_config).parse(
            mapping.root_node, mapping.operators, mapping.operator_mappings,
        )
        return Simulator(self.mapper.hardware_config).run(graph, mapping_result=mapping)

    def test_defaults_and_cli_opt_out_reach_worker_config(self):
        self.assertTrue(SimulatedAnnealingConfig().optimize_partition_chains)
        with self.assertRaises(TypeError):
            SimulatedAnnealingConfig(optimize_partition_chains=1)
        parser = build_parser()
        argv = ["--models", "deepseek-v3.2", "--mtp", "on", "--batch-sizes", "256",
                "--input-sequence-length", "32768",
                "--output-sequence-length", "600"]
        for extra, enabled in (([], True), (["--no-partition-optimization"], False)):
            with self.subTest(enabled=enabled):
                args = parser.parse_args(argv + extra)
                self.assertEqual(args.optimize_partition_chains, enabled)
                case = _make_cases(args)[0]
                self.assertEqual(case.optimize_partition_chains, enabled)
                self.assertEqual((case.rounds, case.warmup_rounds, case.restart_count), (0, 0, 0))

    def test_improves_known_decode_case_without_changing_policies_or_tree(self):
        before, after = self.simulate(self.original), self.simulate(self.selected)
        self.assertEqual(before.latency_ns, 21643288)
        self.assertEqual(after.latency_ns, 7926950)
        self.assertGreater(after.throughput_tokens_per_second, before.throughput_tokens_per_second)
        self.assertEqual(self.original.root_node, self.selected.root_node)
        self.assertEqual(self.original.expert_ownership, self.selected.expert_ownership)
        self.assertEqual(self.original.operator_mappings, self.selected.operator_mappings)
        for op_id in self.original.operators:
            self.assertEqual(self.original.operators[op_id].weight_memory,
                             self.selected.operators[op_id].weight_memory)
        self.assertEqual(self.original.operators["attn.flash_attention"].partition_chain,
                         PartitionChain((("B", "p", 2), ("B", "q", 4), ("S", "p", 16), ("S", "q", 16))))
        self.assertEqual(self.selected.operators["attn.flash_attention"].partition_chain,
                         PartitionChain((("B", "p", 32), ("S", "q", 64))))

    def test_metadata_and_saved_profiles_match_actual_core_dimensions(self):
        profile = json.loads(json.dumps(partition_chains_to_dict(self.selected)))
        restored = apply_partition_chains(self.original, profile, self.mapper.hardware_config)
        self.assertEqual(partition_chains_to_dict(restored), profile)
        graph = TreeParser(self.mapper.hardware_config).parse(
            restored.root_node, restored.operators, restored.operator_mappings, layer_count=1,
        )
        for index, op_id in enumerate(restored.root_node.operator_order):
            if op_id not in profile:
                continue
            command = graph.command(f"i{index}.core")
            for axis, extent in restored.hardware_mappings[op_id].pu_dimensions.items():
                self.assertEqual(getattr(command, axis), extent, (op_id, axis))
        self.assertEqual(self.simulate(restored).latency_ns, self.simulate(self.selected).latency_ns)

    def test_ragged_attention_tiles_cover_each_element_once(self):
        mapped = self.mapper.map(replace(self.request, inference_config=replace(
            self.request.inference_config, global_batch_size=192,
        )), hbf_bandwidth_table=self.table)
        selected = optimize_partition_chains(mapped, self.mapper.hardware_config)
        op = selected.operators["attn.flash_attention"]
        shape = {"B": op.B, "S": op.qk_N}
        volume = 0
        for p in range(32):
            for q in range(64):
                ranges = op.partition_chain.tile_ranges(shape, p, q)
                volume += (ranges["B"][1] - ranges["B"][0]) * (ranges["S"][1] - ranges["S"][0])
        self.assertEqual(volume, op.B * op.qk_N)
        self.assertEqual(op.B, 24)
        self.assertLess(self.simulate(selected).latency_ns, self.simulate(mapped).latency_ns)

    def test_fold_prefetch_and_tp_fp16_remain_valid(self):
        requests = [replace(self.request, xfold=True), HardwareMappingRequest(
            "glm-5.1", InferenceConfig(64, 128, 1, MoEParallelStrategy.TP, "fp16", True),
        )]
        for request in requests:
            with self.subTest(model=request.model_name, fold=request.xfold):
                mapper = mapper_for_model(request.model_name)
                original = mapper.map(request)
                selected = optimize_partition_chains(original, mapper.hardware_config)
                self.assertLessEqual(self.simulate(selected).latency_ns, self.simulate(original).latency_ns)
                for op_id, op in original.operators.items():
                    new_op = selected.operators[op_id]
                    self.assertEqual(new_op.weight_memory, op.weight_memory)
                    self.assertEqual(new_op.flash_prefetch_bytes_per_request, op.flash_prefetch_bytes_per_request)

    def test_other_models_are_unchanged(self):
        for name in ("kimi-k2.5",):
            with self.subTest(model=name):
                mapper = mapper_for_model(name)
                mapped = mapper.map(InferenceConfig(32, 128, 1))
                self.assertIs(optimize_partition_chains(mapped, mapper.hardware_config), mapped)
                self.assertEqual(partition_chains_to_dict(mapped), {})

    def test_direct_search_default_is_cached_once_and_can_be_disabled(self):
        with patch("srammachine.search.optimizer.optimize_partition_chains",
                   return_value=self.selected) as enumerate_layouts:
            optimizer = SplitTreeOptimizer(hbf_bandwidth_csv=DEFAULT_HBF_BANDWIDTH_CSV)
            enabled = optimizer.run(self.request, config=SimulatedAnnealingConfig(write_result=False))
            self.assertIs(optimizer._mapping(256, True), self.selected)
            self.assertEqual(enabled.initial_evaluation.latency_ns, 7926950)
            self.assertEqual(enumerate_layouts.call_count, 1)
        disabled = SplitTreeOptimizer(hbf_bandwidth_csv=DEFAULT_HBF_BANDWIDTH_CSV).run(
            self.request, config=SimulatedAnnealingConfig(
                write_result=False, optimize_partition_chains=False,
            ),
        )
        self.assertEqual(disabled.initial_evaluation.latency_ns, 21643288)

    def test_split_tree_search_is_rejected_before_work_starts(self):
        with self.assertRaisesRegex(NotImplementedError, "SplitTree search is disabled"):
            SimulatedAnnealingConfig(enabled=True)
        optimizer = SplitTreeOptimizer()
        config = SimulatedAnnealingConfig()
        # Also guard the entry point if a caller bypasses frozen-config validation.
        object.__setattr__(config, "enabled", True)
        with patch("srammachine.search.optimizer.mapper_for_model") as make_mapper:
            with self.assertRaises(NotImplementedError):
                optimizer.run(self.request, config=config)
            make_mapper.assert_not_called()
        with self.assertRaises(NotImplementedError):
            optimizer._search_workload(256, True, None, 0)
        argv = ["--models", "deepseek-v3.2", "--mtp", "on", "--batch-sizes", "256",
                "--input-sequence-length", "32768", "--output-sequence-length", "600"]
        for extra in (["--split-tree-search"], ["--rounds", "1"],
                      ["--warmup-rounds", "1"], ["--restarts", "1"]):
            with self.subTest(extra=extra), patch("srammachine.script.run_sa.create_csv_checkpoint") as write:
                with self.assertRaisesRegex(NotImplementedError, "SplitTree search is disabled"):
                    main(argv + extra)
                write.assert_not_called()

    def test_saved_layout_loader_supports_new_and_legacy_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            profile = partition_chains_to_dict(self.selected)
            payload = {"partition_optimization_enabled": True, "pareto_front": [{
                "split_tree_records": [{"split_tree": split_tree_to_dict(self.selected.root_node),
                                        "partition_chains": profile}],
            }]}
            path.write_text(json.dumps(payload))
            self.assertEqual(load_pareto_partition_chains(path), profile)
            self.assertEqual(split_tree_to_dict(load_pareto_split_tree(path)),
                             split_tree_to_dict(self.selected.root_node))
            payload["pareto_front"][0]["split_tree_records"][0].pop("partition_chains")
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "missing its saved"):
                load_pareto_partition_chains(path)
            payload.pop("partition_optimization_enabled")
            path.write_text(json.dumps(payload))
            self.assertEqual(load_pareto_partition_chains(path), {})

    def test_worker_checkpoint_and_trace_use_the_search_layout(self):
        case = SACase("deepseek-v3.2", 32, True, 128, 1, "ep", "fp8",
                      0, 0, 0, 1.0, 0.01, 4, 20260912, hbf_bandwidth_csv=None)
        with patch("srammachine.search.optimizer.legal_tree_mutations",
                   side_effect=AssertionError("SplitTree neighbors must not be generated")):
            row = _run_case(case)
        self.assertFalse(row["split_tree_search_enabled"])
        self.assertEqual(row["proposal_count"], 0)
        self.assertEqual(row["restart_count"], 0)
        self.assertEqual(row["restart_results"], [])
        self.assertGreaterEqual(row["throughput_improvement_fraction"], 0)
        with self.assertRaises(NotImplementedError):
            _run_case(replace(case, rounds=1))
        self.assertTrue(row["optimize_partition_chains"])
        self.assertTrue(row["partition_chains"])
        with tempfile.TemporaryDirectory() as directory:
            trace_path, _ = write_case_trace(row, Path(directory))
            trace = json.loads(trace_path.read_text())
            event = next(e["args"] for e in trace["traceEvents"]
                         if e.get("ph") == "X" and e["args"]["op_id"] == "attn.qk_absorb")
            expected = row["partition_chains"]["attn.qk_absorb"]
            self.assertEqual(event["partition_chain"], [f"{s['dimension']}-{s['pu_dimension']}-{s['factor']}" for s in expected])
            row["is_model_global_pareto"] = True
            path = Path(directory) / "checkpoint.json"
            write_model_json(path, "deepseek-v3.2", "ep", True, [row],
                             generated_at="test", detected_cpus=1, worker_count=1)
            self.assertEqual(load_pareto_partition_chains(path), row["partition_chains"])
            write_csv(Path(directory) / "result.csv", [row])
            # Replaying cannot silently drop layouts from an optimized checkpoint.
            missing = dict(row)
            missing.pop("partition_chains")
            with self.assertRaisesRegex(ValueError, "missing its saved"):
                write_case_trace(missing, Path(directory))

    def test_rejects_invalid_grid_or_operator_overrides(self):
        for profile in ({"unknown": []}, {"attn.latent_down": []}, {
            "attn.flash_attention": [dict(dimension="K", pu_dimension="p", factor=32),
                                      dict(dimension="S", pu_dimension="q", factor=64)],
        }):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                apply_partition_chains(self.original, profile, self.mapper.hardware_config)


class MiniMaxPartitionSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapper = MiniMaxM3Mapper()
        cls.request = HardwareMappingRequest("minimax-m3", InferenceConfig(
            256, 32768, 600, mtp_enabled=True,
        ))
        cls.original = cls.mapper.map(
            cls.request, hbf_bandwidth_table=HBFBandwidthTable.from_csv(DEFAULT_HBF_BANDWIDTH_CSV),
        )
        cls.selected = optimize_partition_chains(cls.original, cls.mapper.hardware_config)

    def graph(self, mapping, layers=4):
        return TreeParser(self.mapper.hardware_config).parse(
            mapping.root_node, mapping.operators, mapping.operator_mappings, layer_count=layers,
        )

    def simulate(self, mapping):
        return Simulator(self.mapper.hardware_config).run(self.graph(mapping), mapping_result=mapping)

    def test_improves_measured_workload_and_preserves_dense_baseline(self):
        before, after = self.simulate(self.original), self.simulate(self.selected)
        self.assertLess(after.latency_ns, before.latency_ns)
        self.assertGreater(after.throughput_tokens_per_second, before.throughput_tokens_per_second)
        self.assertEqual(self.original.root_node, self.selected.root_node)
        self.assertEqual(self.original.operator_mappings, self.selected.operator_mappings)
        self.assertEqual(self.original.expert_ownership, self.selected.expert_ownership)
        self.assertIs(self.original.additional_layer_mappings, self.selected.additional_layer_mappings)
        self.assertEqual(after.model_latency_components[1], before.model_latency_components[1])
        for op_id in ("attn.flash_msa", "attn.indexer_score", "moe.moe_fc1"):
            with self.subTest(op=op_id):
                self.assertNotEqual(self.original.operators[op_id].partition_chain,
                                    self.selected.operators[op_id].partition_chain)
        for op_id, op in self.original.operators.items():
            self.assertEqual(op.weight_memory, self.selected.operators[op_id].weight_memory)

    def test_saved_profiles_restore_group_dimensions_and_latency(self):
        profile = json.loads(json.dumps(partition_chains_to_dict(self.selected)))
        self.assertIn("attn.flash_msa", profile)
        self.assertIn("attn.indexer_score", profile)
        self.assertNotIn("moe.ep_dispatch", profile)
        restored = apply_partition_chains(self.original, profile, self.mapper.hardware_config)
        self.assertEqual(partition_chains_to_dict(restored), profile)
        graph = self.graph(restored, layers=1)
        for index, op_id in enumerate(restored.root_node.operator_order):
            if op_id not in profile:
                continue
            core = graph.command(f"i{index}.core")
            for axis, extent in restored.hardware_mappings[op_id].pu_dimensions.items():
                self.assertEqual(getattr(core, axis), extent, (op_id, axis))
        self.assertEqual(self.simulate(restored).latency_ns, self.simulate(self.selected).latency_ns)

    def test_msa_candidates_cover_bhs_and_select_simulated_minimum(self):
        # Independently distribute two binary splits per direction over B/H/S.
        # Ragged request, head and sequence dimensions exercise clipped tiles.
        hardware = self.mapper.hardware_config
        hardware = replace(hardware, chip=replace(
            hardware.chip, noc=replace(hardware.chip.noc, mesh_rows=4, mesh_columns=4),
            logic_die=replace(hardware.chip.logic_die, pu_mesh_rows=2, pu_mesh_columns=2),
        ))
        op_id = "attn.flash_msa"
        op = replace(self.original.operators[op_id], B=3, H=3, qk_N=5, sv_K=5,
                     partition_chain=PartitionChain((("S", "p", 4), ("H", "q", 4))))
        expected = set()
        for assignments in itertools.product("BHS", repeat=4):
            steps = tuple((axis, direction, 2 ** assignments[start:start + 2].count(axis))
                          for direction, start in (("p", 0), ("q", 2))
                          for axis in "BHS" if axis in assignments[start:start + 2])
            chain = PartitionChain(steps)
            factors = {axis: 1 for axis in "BHS"}
            for step in chain.steps:
                factors[step.dimension] *= step.factor
            if all(factor <= 4 for factor in factors.values()):
                expected.add(chain)
        candidates = set(_attention_candidates(op, hardware))
        self.assertEqual(candidates, expected)
        self.assertTrue(any(s.dimension == "H" and s.pu_dimension == "q"
                            for c in candidates for s in c.steps))
        tree = RootNode(256, (op_id,), GroupNode((OpNode(op_id),)))
        metadata = replace(self.original.hardware_mappings[op_id], chip_dimensions={
            axis: getattr(op, axis) for axis in ("B", "H", "M", "qk_K", "qk_N", "sv_K", "sv_N")
        })
        policy = self.original.operator_mappings[op_id]
        parser = TreeParser(hardware)

        def latency(candidate):
            return Simulator(hardware).run(parser.parse(
                tree, {op_id: candidate}, {op_id: policy}, layer_count=1,
            )).total_time_ns

        for source in WeightSource:
            with self.subTest(source=source):
                probe = replace(op, weight_memory=WeightMemory(
                    source, flash_bandwidth_bytes_per_second=3e9,
                ))
                mapping = replace(self.original, root_node=tree, operators={op_id: probe},
                                  operator_mappings={op_id: policy}, hardware_mappings={op_id: metadata},
                                  op_list=[probe], additional_layer_mappings=())
                selected = optimize_partition_chains(mapping, hardware)
                expected_cost = min(latency(replace(probe, partition_chain=c))
                                    for c in candidates | {probe.partition_chain})
                self.assertEqual(latency(selected.operators[op_id]), expected_cost)
                self.assertEqual(selected.operators[op_id].weight_memory, probe.weight_memory)

    def test_normal_fold_tp_ep_dtypes_mtp_and_ragged_batches(self):
        cases = (
            (False, "tp", "fp8", False, 32),
            (False, "ep", "fp16", True, 192),
            (True, "tp", "fp16", False, 192),
            (True, "ep", "fp8", True, 32),
        )
        for fold, strategy, dtype, mtp, batch in cases:
            with self.subTest(fold=fold, strategy=strategy, dtype=dtype, mtp=mtp, batch=batch):
                original = self.mapper.map(InferenceConfig(
                    batch, 4097, 1, MoEParallelStrategy(strategy), dtype, mtp,
                ), xfold=fold)
                selected = optimize_partition_chains(original, self.mapper.hardware_config)
                self.assertLessEqual(self.simulate(selected).latency_ns, self.simulate(original).latency_ns)
                score_id = "moe.indexer_score" if fold else "attn.indexer_score"
                profile = partition_chains_to_dict(selected)
                self.assertIn(score_id, profile)
                self.assertTrue(all(s["dimension"] in ("B", "S") for s in profile[score_id]))
                restored = apply_partition_chains(original, profile, self.mapper.hardware_config)
                self.assertEqual(self.simulate(restored).latency_ns, self.simulate(selected).latency_ns)
                for op_id, op in original.operators.items():
                    self.assertEqual(op.weight_memory, selected.operators[op_id].weight_memory)
                    self.assertEqual(op.flash_prefetch_bytes_per_request,
                                     selected.operators[op_id].flash_prefetch_bytes_per_request)
                if fold:
                    def prefetch(mapping):
                        graph = self.graph(mapping, layers=2)
                        return [(c.size_bytes, c.bandwidth_bytes_per_second,
                                 graph.predecessors(c.cmd_id), graph.successors(c.cmd_id))
                                for c in graph.commands if isinstance(c, PUMemoryReadCmd)
                                and c.data_kind == "next_layer_kv"]
                    self.assertEqual(prefetch(original), prefetch(selected))

    def test_standalone_dense_layer_can_be_optimized(self):
        mapper = MiniMaxM3Mapper(layer_type="dense")
        original = mapper.map(InferenceConfig(32, 4096, 1))
        selected = optimize_partition_chains(original, mapper.hardware_config)
        self.assertLessEqual(self.simulate(selected).latency_ns, self.simulate(original).latency_ns)
        self.assertIn("attn.flash_msa", partition_chains_to_dict(selected))
        self.assertNotIn("attn.indexer_score", partition_chains_to_dict(selected))

    def test_worker_checkpoint_trace_and_opt_out(self):
        case = SACase("minimax-m3", 32, True, 128, 1, "ep", "fp16",
                      0, 0, 0, 1.0, 0.01, 4, 20260912, xfold=True, hbf_bandwidth_csv=None)
        row = _run_case(case)
        self.assertTrue(row["partition_chains"])
        self.assertGreater(row["throughput_improvement_fraction"], 0)
        disabled = _run_case(replace(case, optimize_partition_chains=False))
        self.assertEqual(disabled["pareto_best_latency_ns"], disabled["baseline_latency_ns"])
        self.assertNotEqual(disabled["partition_chains"], row["partition_chains"])
        with tempfile.TemporaryDirectory() as directory:
            with patch("srammachine.search.optimizer.optimize_partition_chains",
                       side_effect=AssertionError("replay must use saved layouts")):
                trace_path, _ = write_case_trace(row, Path(directory))
            events = json.loads(trace_path.read_text())["traceEvents"]
            for op_id in ("attn.flash_msa", "moe.indexer_score"):
                event = next(e["args"] for e in events
                             if e.get("ph") == "X" and e["args"]["op_id"] == op_id)
                self.assertEqual(event["partition_chain"], [
                    f"{s['dimension']}-{s['pu_dimension']}-{s['factor']}"
                    for s in row["partition_chains"][op_id]
                ])
            row["is_model_global_pareto"] = True
            path = Path(directory) / "checkpoint.json"
            write_model_json(path, "minimax-m3", "ep", True, [row],
                             generated_at="test", detected_cpus=1, worker_count=1)
            self.assertEqual(load_pareto_partition_chains(path), row["partition_chains"])

    def test_rejects_illegal_minimax_partition_axes(self):
        for op_id, axis in (("attn.flash_msa", "M"), ("attn.indexer_score", "H")):
            with self.subTest(op=op_id), self.assertRaises(ValueError):
                apply_partition_chains(self.original, {op_id: [
                    dict(dimension=axis, pu_dimension="p", factor=32),
                    dict(dimension="S", pu_dimension="q", factor=64),
                ]}, self.mapper.hardware_config)


if __name__ == "__main__":
    unittest.main()
