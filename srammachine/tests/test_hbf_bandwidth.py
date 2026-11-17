"""Measured HBF lookup, cache lowering, and experiment configuration regressions."""

import csv
import math
from pathlib import Path
import tempfile
import unittest
import warnings

from srammachine.commands import DramReadCmd, PUMemoryReadCmd
from srammachine.frontend.modules import WeightSource
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG
from srammachine.inference import InferenceConfig
from srammachine.mapping import (
    DEFAULT_HBF_BANDWIDTH_CSV, HBFBandwidthTable, HardwareMapper,
    HardwareMappingRequest, mapper_for_model,
)
from srammachine.mapping.hbf_bandwidth import normalize_hbf_csv
from srammachine.pipetree import GroupNode, OpNode, PipeTree, TreeParser
from srammachine.script.run_sa import build_parser
from srammachine.simulator import Simulator


def measurement(batch, sequence, gbps, *, kind="sparse_kv", mtp="0"):
    return dict(model_name="deepseek-v3.2", read_kind=kind, mtp_enabled=mtp,
                global_batch_size=batch, input_sequence_length=sequence,
                op_id="attn.indexer_score" if kind == "dense_indexer" else "attn.flash_attention",
                mean_GBps=gbps, bandwidth_scope="chip", pu_count=2048,
                flash_bandwidth_bytes_per_second=gbps * 1e9 / 2048)


class HBFTableTests(unittest.TestCase):
    def test_exact_linear_and_bilinear_interpolation(self):
        table = HBFBandwidthTable([
            measurement(16, 100, 2048), measurement(32, 100, 4096),
            measurement(16, 200, 6144), measurement(32, 200, 10240),
        ])
        for batch, sequence, expected in (
            (16, 100, 1e9), (32, 200, 5e9), (24, 100, 1.5e9),
            (16, 150, 2e9), (24, 150, 2.75e9),
        ):
            with self.subTest(batch=batch, sequence=sequence):
                self.assertEqual(table.lookup("deepseek-v3.2", "sparse_kv", False,
                                              batch, sequence), expected)
        for batch, sequence in ((15, 100), (33, 100), (16, 99), (16, 201)):
            with self.subTest(batch=batch, sequence=sequence), self.assertRaisesRegex(
                ValueError, "outside measured range",
            ):
                table.lookup("deepseek-v3.2", "sparse_kv", False, batch, sequence)

    def test_dense_mtp_fallback_and_sparse_mtp_isolation(self):
        table = HBFBandwidthTable([
            measurement(16, 100, 2048, kind="dense_indexer"),
            measurement(16, 100, 4096), measurement(16, 100, 8192, mtp="1"),
        ])
        self.assertEqual(table.lookup("deepseek-v3.2", "dense_indexer", True, 16, 100), 1e9)
        self.assertEqual(table.lookup("deepseek-v3.2", "sparse_kv", True, 16, 100), 4e9)
        sparse_only = HBFBandwidthTable([measurement(16, 100, 2048)])
        with self.assertRaisesRegex(ValueError, "no HBF measurements"):
            sparse_only.lookup("deepseek-v3.2", "sparse_kv", True, 16, 100)

    def test_invalid_csv_and_missing_interpolation_corner(self):
        row = measurement(16, 100, 2048)
        for bad in ([], [row, row], [dict(row, pu_count=16384)],
                    [dict(row, flash_bandwidth_bytes_per_second=float("nan"))],
                    [dict(row, flash_bandwidth_bytes_per_second=-1)],
                    [dict(row, flash_bandwidth_bytes_per_second=2048e9)]):
            with self.subTest(rows=bad), self.assertRaises(ValueError):
                HBFBandwidthTable(bad)
        table = HBFBandwidthTable([row, measurement(32, 100, 4096), measurement(16, 200, 6144)])
        with self.assertRaisesRegex(ValueError, "missing interpolation corner"):
            table.lookup("deepseek-v3.2", "sparse_kv", False, 24, 150)

    def test_normalization_preserves_provenance_and_converts_units(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "raw.csv", Path(directory) / "normalized.csv"
            fields = ("model", "part", "mtp", "global_bs", "local_bs", "mean_seqlen",
                      "token_bytes", "mean_GBps", "measurement_key")
            with source.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for model in ("Kimi-K2", "GLM-5.2"):
                    writer.writerow(dict(model=model, part="dense_kv", mtp=0,
                                         global_bs=16, local_bs=2, mean_seqlen=16384,
                                         token_bytes=576, mean_GBps=4096, measurement_key="original"))
            raw = source.read_bytes()
            self.assertEqual(normalize_hbf_csv(source, output), 2)
            self.assertEqual(source.read_bytes(), raw)
            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["model_name"] for r in rows], ["kimi-k2.5", "glm-5.2"])
            self.assertEqual(rows[0]["measurement_key"], "original")
            table = HBFBandwidthTable.from_csv(output)
            self.assertEqual(table.lookup("kimi-k2.5", "dense_kv", True, 16, 16384), 2e9)


class HBFMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = HBFBandwidthTable.from_csv(DEFAULT_HBF_BANDWIDTH_CSV)

    def map(self, model, *, mtp=False, xfold=False, batch=256):
        mapper = mapper_for_model(model)
        request = HardwareMappingRequest(model, InferenceConfig(batch, 20000, 600,
                                                               mtp_enabled=mtp), xfold=xfold)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mapped = mapper.map(request, hbf_bandwidth_table=self.table)
        return mapper, mapped

    def test_all_models_lower_measured_cache_rates_and_preserve_weight_policy(self):
        for model in ("deepseek-v3", "kimi-k2.5", "deepseek-v3.2", "glm-5.1", "minimax-m3"):
            for mtp in (False, True):
                with self.subTest(model=model, mtp=mtp):
                    mapper, mapped = self.map(model, mtp=mtp)
                    graph = TreeParser(mapper.hardware_config).parse(
                        mapped.root_node, mapped.operators, mapped.operator_mappings, layer_count=1)
                    op_id = "attn.flash_msa" if model == "minimax-m3" else "attn.flash_attention"
                    kind = "dense_kv" if model in ("deepseek-v3", "kimi-k2.5") else "sparse_kv"
                    expected = self.table.lookup(model, kind, mtp, 256, 20000)
                    reads = [c for c in graph.commands if isinstance(c, PUMemoryReadCmd) and c.op_id == op_id]
                    self.assertEqual(len(reads), 1)
                    self.assertEqual(reads[0].port, "flash")
                    self.assertEqual(reads[0].bandwidth_bytes_per_second, expected)
                    self.assertFalse(any(isinstance(c, DramReadCmd) and c.op_id == op_id
                                         for c in graph.commands))
                    for name, op in mapped.operators.items():
                        if "score" not in name and "flash" not in name and op.weight_memory:
                            self.assertIsNone(op.weight_memory.flash_bandwidth_bytes_per_second)

    def test_legacy_kv_bytes_are_per_request_not_per_head_or_mtp_query(self):
        for model in ("kimi-k2.5",):
            for mtp in (False, True):
                with self.subTest(model=model, mtp=mtp):
                    mapper, mapped = self.map(model, mtp=mtp)
                    graph = TreeParser(mapper.hardware_config).parse(
                        mapped.root_node, mapped.operators, mapped.operator_mappings, layer_count=1)
                    read = next(c for c in graph.commands if isinstance(c, PUMemoryReadCmd)
                                and c.op_id == "attn.flash_attention")
                    self.assertEqual(read.size_bytes, 32 * 20000 * 576 / 2048)
                    result = Simulator(mapper.hardware_config).run(graph)
                    execution = next(c for c in result.command_results if c.cmd_id == read.cmd_id)
                    self.assertEqual(execution.duration_ns,
                                     math.ceil(read.size_bytes * 1e9 / read.bandwidth_bytes_per_second))

    def test_fold_updates_remaining_indexer_and_prefetch_but_core_stays_dram(self):
        mapper, mapped = self.map("deepseek-v3.2", mtp=True, xfold=True)
        ops = mapped.operators
        self.assertEqual(ops["attn.flash_attention"].weight_memory.source, WeightSource.DRAM)
        self.assertIsNone(ops["attn.flash_attention"].weight_memory.flash_bandwidth_bytes_per_second)
        indexer = self.table.lookup("deepseek-v3.2", "dense_indexer", True, 256, 20000)
        sparse = self.table.lookup("deepseek-v3.2", "sparse_kv", True, 256, 20000)
        self.assertNotIn("attn.indexer_score", ops)
        self.assertEqual(ops["moe.indexer_score"].weight_memory.flash_bandwidth_bytes_per_second, indexer + 8e9)
        self.assertIsNone(ops["attn.indexer_k"].weight_memory.flash_bandwidth_bytes_per_second)
        self.assertEqual(ops["moe.moe_fc2"].weight_memory.source, WeightSource.DRAM)
        graph = TreeParser(mapper.hardware_config).parse(
            mapped.root_node, ops, mapped.operator_mappings, layer_count=1)
        reads = [c for c in graph.commands if isinstance(c, PUMemoryReadCmd) and c.op_id == "moe.moe_fc2"]
        self.assertEqual({c.port for c in reads}, {"dram", "flash"})
        self.assertEqual(next(c.bandwidth_bytes_per_second for c in reads if c.port == "flash"), sparse)

    def test_microbatches_keep_full_global_batch_rate(self):
        mapper, mapped = self.map("deepseek-v3.2", batch=256)
        op_id = "attn.flash_attention"
        # Each microbatch is smaller than the CSV's smallest batch (16).
        tree = PipeTree(256, (op_id,), GroupNode((OpNode(op_id),), split=32))
        graph = TreeParser(mapper.hardware_config).parse(
            tree, {op_id: mapped.operators[op_id]},
            {op_id: mapped.operator_mappings[op_id]}, layer_count=1)
        rates = [c.bandwidth_bytes_per_second for c in graph.commands if isinstance(c, PUMemoryReadCmd)]
        self.assertEqual(len(rates), 32)
        self.assertEqual(set(rates), {self.table.lookup("deepseek-v3.2", "sparse_kv", False, 256, 20000)})

    def test_minimax_fold_uses_indexer_and_sparse_prefetch_rates(self):
        for mtp in (False, True):
            with self.subTest(mtp=mtp):
                mapper, mapped = self.map("minimax-m3", mtp=mtp, xfold=True)
                self.assertEqual(mapped.operators["attn.flash_msa"].weight_memory.source, WeightSource.DRAM)
                graph = TreeParser(mapper.hardware_config).parse(
                    mapped.root_node, mapped.operators, mapped.operator_mappings, layer_count=1)
                reads = [c for c in graph.commands if isinstance(c, PUMemoryReadCmd)]
                index = next(c for c in reads if c.op_id == "moe.indexer_score")
                prefetch = next(c for c in reads if c.data_kind == "next_layer_kv")
                self.assertEqual(index.bandwidth_bytes_per_second,
                                 self.table.lookup("minimax-m3", "dense_indexer", mtp, 256, 20000) + 8e9)
                self.assertEqual(prefetch.bandwidth_bytes_per_second,
                                 self.table.lookup("minimax-m3", "sparse_kv", mtp, 256, 20000))
                self.assertEqual((index.port, prefetch.port), ("flash", "flash"))
                weights = [c for c in reads if c.op_id == "moe.moe_fc2" and c.data_kind == "weight"]
                self.assertEqual([c.port for c in weights], ["dram"])

    def test_minimax_dense_defaults_and_public_mapper(self):
        request = HardwareMappingRequest("minimax-m3", InferenceConfig(256, 20000, 600))
        with self.assertWarnsRegex(UserWarning, "dense KV.*default Flash"):
            mapped = HardwareMapper().map(request, hbf_bandwidth_table=self.table)
        dense = mapped.additional_layer_mappings[0]
        op = dense.operators["attn.flash_msa"]
        self.assertIsNone(op.weight_memory.flash_bandwidth_bytes_per_second)
        graph = TreeParser().parse(dense.root_node, dense.operators, dense.operator_mappings, layer_count=1)
        read = next(c for c in graph.commands if isinstance(c, PUMemoryReadCmd) and c.op_id == op.op_id)
        self.assertEqual(read.bandwidth_bytes_per_second,
                         DEFAULT_HARDWARE_CONFIG.chip.logic_die.processing_unit.flash_bandwidth_bytes_per_second)

    def test_cli_enables_bundled_table_and_can_override_or_disable_it(self):
        args = ["--models", "deepseek-v3", "--mtp", "off", "--batch-sizes", "16",
                "--rounds", "1", "--input-sequence-length", "16384", "--output-sequence-length", "600"]
        parser = build_parser()
        self.assertEqual(parser.parse_args(args).hbf_bandwidth_csv, DEFAULT_HBF_BANDWIDTH_CSV)
        self.assertIsNone(parser.parse_args(args + ["--no-hbf-bandwidth"]).hbf_bandwidth_csv)
        self.assertEqual(parser.parse_args(args + ["--hbf-bandwidth-csv", "custom.csv"]).hbf_bandwidth_csv,
                         Path("custom.csv"))


if __name__ == "__main__":
    unittest.main()
