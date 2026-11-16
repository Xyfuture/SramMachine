"""Dense V3 mapping, full-cache traffic, and PartitionChain search/replay."""

from dataclasses import replace
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

from srammachine.commands import PUMemoryReadCmd
from srammachine.frontend.modules import FabricOp, WeightMemory, WeightSource
from srammachine.inference import InferenceConfig, MoEParallelStrategy
from srammachine.mapping import (
    DEFAULT_HBF_BANDWIDTH_CSV, DeepSeekV3Mapper, DeepSeekV32Mapper,
    HBFBandwidthTable, HardwareMapper, HardwareMappingRequest, mapper_for_model,
)
from srammachine.pipetree import TreeParser
from srammachine.script.run_sa import (
    _make_cases, _run_case, _validate_args, build_parser, write_case_trace,
)
from srammachine.search import (
    SimulatedAnnealingConfig, SplitTreeOptimizer, apply_partition_chains,
    optimize_partition_chains, partition_chains_to_dict,
)
from srammachine.simulator import Simulator


class DeepSeekV3MapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapper = DeepSeekV3Mapper.from_json(
            Path(__file__).resolve().parents[1] / "model_cards/deepseek-v3.json",
        )
        cls.table = HBFBandwidthTable.from_csv(DEFAULT_HBF_BANDWIDTH_CSV)

    def graph(self, mapping):
        return TreeParser(self.mapper.hardware_config).parse(
            mapping.root_node, mapping.operators, mapping.operator_mappings, layer_count=1,
        )

    def simulate(self, mapping):
        return Simulator(self.mapper.hardware_config).run(
            self.graph(mapping), mapping_result=mapping,
        )

    def test_reuses_v32_ordinary_projections_and_moe_without_indexer(self):
        self.assertIsInstance(mapper_for_model("deepseek-v3"), DeepSeekV3Mapper)
        for strategy, mtp in itertools.product(MoEParallelStrategy, (False, True)):
            with self.subTest(strategy=strategy, mtp=mtp):
                inference = InferenceConfig(256, 20000, 600, strategy, mtp_enabled=mtp)
                request = HardwareMappingRequest("deepseek-v3", inference, xfold=True)
                v3 = HardwareMapper().map(request)
                v32 = DeepSeekV32Mapper().map(inference)
                self.assertEqual(v3.component_order, ("PreAttn", "CoreAttn", "PostAttn", "FusedMoE"))
                expected = [op for op in v32.op_list if "indexer" not in op.op_id]
                self.assertEqual(v3.root_node.operator_order, tuple(op.op_id for op in expected))
                self.assertEqual(v3.root_node.batch_size, 256)
                self.assertEqual(v3.local_batch_size, 32)
                self.assertEqual(v3.expert_ownership, v32.expert_ownership)
                self.assertEqual(v3.expert_token_loads, v32.expert_token_loads)
                for op in expected:
                    if op.op_id != "attn.flash_attention":
                        self.assertEqual(v3.operators[op.op_id], op)
                    self.assertEqual(v3.operator_mappings[op.op_id], v32.operator_mappings[op.op_id])
                self.assertEqual(v3.operators["attn.flash_attention"].qk_N, 20000)
                self.assertEqual(v32.operators["attn.flash_attention"].qk_N, 2048)
                self.assertEqual(any(isinstance(op, FabricOp) for op in v3.op_list),
                                 strategy is MoEParallelStrategy.EP)

    def test_xfold_is_ordinary_for_all_entry_points_and_memory_overrides(self):
        inference = InferenceConfig(256, 32768, 600)
        ordinary = self.mapper.map(inference, hbf_bandwidth_table=self.table)
        request = HardwareMappingRequest("deepseek-v3", inference, xfold=True)
        for mapped in (
            self.mapper.map(inference, xfold=True, hbf_bandwidth_table=self.table),
            self.mapper.map(request, hbf_bandwidth_table=self.table),
            HardwareMapper().map(request, hbf_bandwidth_table=self.table),
        ):
            self.assertFalse(mapped.request.xfold)
            self.assertEqual(mapped.op_list, ordinary.op_list)
            self.assertEqual(mapped.root_node, ordinary.root_node)
            self.assertFalse(any(op.flash_prefetch_bytes_per_request for op in mapped.op_list))
        self.assertTrue(request.xfold)  # The caller's immutable request is unchanged.
        self.assertEqual(self.mapper.build_op_list(inference, xfold=True),
                         self.mapper.build_op_list(inference))
        policy = WeightMemory(WeightSource.DRAM_FLASH, flash_bandwidth_bytes_per_second=9e9)
        overridden = self.mapper.map(inference, xfold=True, memory_overrides={
            "attn.flash_attention": policy,
        })
        self.assertEqual(overridden.operators["attn.flash_attention"].weight_memory, policy)
        with self.assertRaisesRegex(ValueError, "unknown operator"):
            self.mapper.build_op_list(inference, memory_overrides={"attn.indexer_q": policy})
        with self.assertRaises(TypeError):
            self.mapper.build_op_list(inference, xfold=1)
        with self.assertRaises(ValueError):
            self.mapper.map(replace(request, model_name="deepseek-v3.2"))

    def test_full_kv_flash_reads_use_csv_and_are_shared_by_mtp_queries(self):
        for sequence, dtype in itertools.product((20000, 32768), ("fp8", "fp16")):
            reads = []
            for mtp in (False, True):
                with self.subTest(sequence=sequence, dtype=dtype, mtp=mtp):
                    inference = InferenceConfig(256, sequence, 600, kv_cache_dtype=dtype,
                                                mtp_enabled=mtp)
                    mapped = self.mapper.map(inference, xfold=True, hbf_bandwidth_table=self.table)
                    op = mapped.operators["attn.flash_attention"]
                    self.assertEqual((op.B, op.M, op.qk_K, op.qk_N, op.sv_K, op.sv_N),
                                     (32, 128 * (2 if mtp else 1), 576, sequence, sequence, 512))
                    shape = {"B": op.B, "S": sequence}
                    tile = op.partition_chain.compute_shape(shape)
                    graph = self.graph(mapped)
                    cache_reads = [c for c in graph.commands if isinstance(c, PUMemoryReadCmd)
                                   and c.op_id == op.op_id]
                    self.assertEqual(len(cache_reads), 1)
                    read = cache_reads[0]
                    self.assertEqual(read.port, "flash")
                    self.assertEqual(read.size_bytes,
                                     tile["B"] * tile["S"] * 576 * inference.kv_cache_bytes_per_element)
                    rate = self.table.lookup("deepseek-v3", "dense_kv", mtp, 256, sequence)
                    self.assertEqual(read.bandwidth_bytes_per_second, rate)
                    index = mapped.root_node.operator_order.index(op.op_id)
                    self.assertEqual(graph.command(f"i{index}.core").memory_duration_ns,
                                     math.ceil(read.size_bytes * 1e9 / rate))
                    volume = 0
                    for p, q in itertools.product(range(32), range(64)):
                        ranges = op.partition_chain.tile_ranges(shape, p, q)
                        volume += math.prod(end - start for start, end in ranges.values())
                    self.assertEqual(volume, op.B * sequence)
                    reads.append(read.size_bytes)
            self.assertEqual(reads[0], reads[1])

    def test_small_ragged_workloads_preserve_full_context(self):
        for strategy, mtp in itertools.product(MoEParallelStrategy, (False, True)):
            with self.subTest(strategy=strategy, mtp=mtp):
                mapped = self.mapper.map(InferenceConfig(1, 17, 1, strategy, mtp_enabled=mtp),
                                         xfold=True)
                self.assertEqual(mapped.operators["attn.flash_attention"].qk_N, 17)
                self.assertGreater(self.simulate(mapped).total_time_ns, 0)

    def test_partition_search_improves_and_replays_dense_attention(self):
        for strategy, mtp, dtype in ((MoEParallelStrategy.EP, False, "fp8"),
                                     (MoEParallelStrategy.TP, True, "fp16")):
            with self.subTest(strategy=strategy, mtp=mtp, dtype=dtype):
                request = HardwareMappingRequest("deepseek-v3", InferenceConfig(
                    256, 32768, 600, strategy, dtype, mtp,
                ), xfold=True)
                original = self.mapper.map(request, hbf_bandwidth_table=self.table)
                selected = optimize_partition_chains(original, self.mapper.hardware_config)
                profile = json.loads(json.dumps(partition_chains_to_dict(selected)))
                self.assertEqual(set(profile), {op.op_id for op in original.op_list
                                               if not isinstance(op, FabricOp)})
                self.assertNotEqual(profile, partition_chains_to_dict(original))
                self.assertEqual(selected.operators["attn.flash_attention"].qk_N, 32768)
                self.assertEqual(selected.root_node, original.root_node)
                for op_id, op in original.operators.items():
                    self.assertEqual(selected.operators[op_id].weight_memory, op.weight_memory)
                restored = apply_partition_chains(original, profile, self.mapper.hardware_config)
                self.assertEqual(restored.op_list, selected.op_list)
                after = self.simulate(selected)
                self.assertLess(after.latency_ns, self.simulate(original).latency_ns)
                self.assertEqual(self.simulate(restored).latency_ns, after.latency_ns)
                optimizer = SplitTreeOptimizer(hbf_bandwidth_csv=DEFAULT_HBF_BANDWIDTH_CSV)
                result = optimizer.run(request, config=SimulatedAnnealingConfig(
                    layer_count=1, write_result=False,
                ))
                self.assertEqual(result.initial_evaluation.latency_ns, after.latency_ns)
                self.assertFalse(result.initial_evaluation.mapping_result.request.xfold)

    def test_cli_worker_saves_dense_chains_and_replays_xfold_as_ordinary(self):
        parser = build_parser()
        args = _validate_args(parser, parser.parse_args([
            "--models", "deepseek-v3", "deepseek-v3.2", "--xfold", "--mtp", "on",
            "--batch-sizes", "256", "--input-sequence-length", "32768",
            "--output-sequence-length", "600", "--layer-count", "4",
        ]))
        case = next(case for case in _make_cases(args) if case.model == "deepseek-v3")
        row = _run_case(case)
        self.assertFalse(row["xfold"])
        self.assertTrue(row["partition_chains"])
        self.assertGreater(row["throughput_improvement_fraction"], 0)
        with tempfile.TemporaryDirectory() as directory:
            trace_path, _ = write_case_trace(row, Path(directory))
            trace = json.loads(trace_path.read_text())
            self.assertFalse(trace["otherData"]["xfold"])
            events = [event["args"] for event in trace["traceEvents"] if event.get("ph") == "X"]
            self.assertFalse(any("indexer" in event["op_id"] for event in events))
            core = next(event for event in events if event["op_id"] == "attn.flash_attention"
                        and "partition_chain" in event)
            self.assertEqual(core["partition_chain"], [
                f"{s['dimension']}-{s['pu_dimension']}-{s['factor']}"
                for s in row["partition_chains"]["attn.flash_attention"]
            ])


if __name__ == "__main__":
    unittest.main()
