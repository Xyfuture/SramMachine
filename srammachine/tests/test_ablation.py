"""Mixed-grid ablations must affect lowering, search and actual port execution."""

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

from exp.ablation.policy import ABLATIONS, apply_ablation, half_grid_hardware, latency_metrics, load_hardware
from exp.big.sweep import mark_pareto
from srammachine.commands import CommandGraph, PUMemoryReadCmd
from srammachine.frontend.modules import BMMOp, WeightMemory, WeightSource
from srammachine.hardware import HardwareConfig
from srammachine.inference import InferenceConfig
from srammachine.mapping import DEFAULT_HBF_BANDWIDTH_CSV, HBFBandwidthTable, HardwareMappingRequest, MiniMaxM3Mapper, mapper_for_model
from srammachine.pipetree import GroupNode, OpNode, OperatorMapping, RootNode, TreeParser
from srammachine.search import optimize_partition_chains
from srammachine.simulator import GraphExecutor, Simulator


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareConfig()
        cls.table = HBFBandwidthTable.from_csv(DEFAULT_HBF_BANDWIDTH_CSV)
        cls.mapper = MiniMaxM3Mapper(cls.hardware)
        cls.request = HardwareMappingRequest("minimax-m3", InferenceConfig(
            1024, 16384, 600, mtp_enabled=True), xfold=True)
        cls.original = cls.mapper.map(cls.request, hbf_bandwidth_table=cls.table)

    def test_all_variants_assign_grid_ports_and_preserve_base(self):
        for name in ABLATIONS:
            with self.subTest(ablation=name):
                source = self.mapper.map(replace(self.request, xfold=False), hbf_bandwidth_table=self.table) \
                    if name == "NoXfold" else self.original
                mapped = apply_ablation(source, self.hardware, name)
                for layer, before in zip((mapped, *mapped.additional_layer_mappings),
                                         (source, *source.additional_layer_mappings)):
                    for key, op in layer.operators.items():
                        special = key in ("attn.flash_msa", "attn.indexer_score", "moe.indexer_score")
                        dram = not special and name in ("NoDRAMShare", "NoDRAMFLashShare")
                        flash = special and name in ("NoFlashShare", "NoDRAMFLashShare")
                        hw = layer.operator_mappings[key].hardware_config
                        self.assertEqual(hw.chip.noc.mesh_rows, 16 if dram or flash else 32)
                        pu = hw.chip.logic_die.processing_unit
                        self.assertEqual(pu.dram_bandwidth_bytes_per_second, 32e9 if dram else 16e9)
                        self.assertEqual(pu.flash_bandwidth_bytes_per_second, 8e9 if flash else 4e9)
                        self.assertEqual(hw.chip.logic_die_count, 4)
                        self.assertEqual(hw.chip.logic_die.vector_unit, self.hardware.chip.logic_die.vector_unit)
                        if op.weight_memory is None:
                            continue
                        op.partition_chain.validate_grid(hw.chip.noc.mesh_rows, hw.chip.noc.mesh_columns)
                        rate = before.operators[key].weight_memory.flash_bandwidth_bytes_per_second or 4e9
                        factor = 2 if flash else 1
                        if name == "NoKVLayout" and key.endswith("indexer_score"):
                            factor *= 0.87
                        if (name == "NoMoELayout" and key.startswith("moe.")
                                and type(op) is BMMOp and not op.flash_prefetch_bytes_per_request):
                            factor *= 1 / 6
                        self.assertAlmostEqual(op.weight_memory.flash_bandwidth_bytes_per_second, rate * factor)
                self.assertEqual(mapped.request.xfold, name != "NoXfold")
        self.assertEqual(self.hardware.chip.noc.mesh_rows, 32)
        self.assertTrue(all(p.hardware_config is None for p in self.original.operator_mappings.values()))
        self.assertEqual(self.original.operators["moe.moe_fc2"].partition_chain.grid_shape, (32, 64))

    def test_measured_flash_and_kv_prefetch_are_scaled_not_replaced_by_default(self):
        score_rate = self.table.lookup("minimax-m3", "dense_indexer", True, 1024, 16384)
        kv_rate = self.table.lookup("minimax-m3", "sparse_kv", True, 1024, 16384)
        flash = apply_ablation(self.original, self.hardware, "NoFlashShare")
        self.assertAlmostEqual(flash.operators["moe.indexer_score"].weight_memory.flash_bandwidth_bytes_per_second,
                               2 * score_rate)
        self.assertAlmostEqual(flash.operators["moe.moe_fc2"].weight_memory.flash_bandwidth_bytes_per_second,
                               kv_rate)
        kv = apply_ablation(self.original, self.hardware, "NoKVLayout")
        graph = TreeParser(self.hardware).parse(kv.root_node, kv.operators, kv.operator_mappings)
        reads = [c for c in graph.commands if isinstance(c, PUMemoryReadCmd) and c.data_kind == "next_layer_kv"]
        self.assertEqual(len(reads), 4)
        self.assertTrue(all(c.bandwidth_bytes_per_second == kv_rate for c in reads))
        scores = [c for c in graph.commands if isinstance(c, PUMemoryReadCmd) and c.data_kind == "indexer_cache"]
        self.assertEqual(len(scores), 4)
        self.assertTrue(all(c.bandwidth_bytes_per_second == 0.87 * score_rate for c in scores))
        stage = apply_ablation(self.original, self.hardware, "NoMoELayout", moe_flash_scope="stage")
        self.assertTrue(math.isclose(stage.operators["moe.indexer_score"].weight_memory.flash_bandwidth_bytes_per_second,
                                     score_rate / 6))
        self.assertTrue(math.isclose(stage.operators["moe.moe_fc2"].weight_memory.flash_bandwidth_bytes_per_second,
                                     kv_rate / 6))
        weights_only = apply_ablation(self.original, self.hardware, "NoMoELayout", moe_flash_scope="weights")
        self.assertEqual(weights_only.operators["moe.indexer_score"].weight_memory.flash_bandwidth_bytes_per_second,
                         score_rate)
        self.assertEqual(weights_only.operators["moe.moe_fc2"].weight_memory.flash_bandwidth_bytes_per_second, kv_rate)

    def test_no_moe_layout_defaults_to_bmm_weights_and_preserves_all_cache_reads(self):
        for model in ("deepseek-v3", "deepseek-v3.2", "glm-5.1", "minimax-m3"):
            with self.subTest(model=model):
                mapped = mapper_for_model(model, self.hardware).map(
                    replace(self.request, model_name=model), hbf_bandwidth_table=self.table)
                baseline = apply_ablation(mapped, self.hardware, "XFold")
                ablated = apply_ablation(mapped, self.hardware, "NoMoELayout")
                dual_port_ops = 0
                for before, after in zip((baseline, *baseline.additional_layer_mappings),
                                         (ablated, *ablated.additional_layer_mappings)):
                    for key, op in before.operators.items():
                        changed = after.operators[key]
                        self.assertEqual(before.operator_mappings[key], after.operator_mappings[key])
                        if op.weight_memory is None:
                            continue
                        self.assertEqual(op.partition_chain, changed.partition_chain)
                        factor = 1 / 6 if (key.startswith("moe.") and type(op) is BMMOp
                                           and not op.flash_prefetch_bytes_per_request) else 1
                        self.assertTrue(math.isclose(
                            changed.weight_memory.flash_bandwidth_bytes_per_second,
                            op.weight_memory.flash_bandwidth_bytes_per_second * factor))
                    before_graph = TreeParser(self.hardware).parse(
                        before.root_node, before.operators, before.operator_mappings, layer_count=1)
                    after_graph = TreeParser(self.hardware).parse(
                        after.root_node, after.operators, after.operator_mappings, layer_count=1)
                    for command in before_graph.commands:
                        if not isinstance(command, PUMemoryReadCmd):
                            continue
                        changed = after_graph.command(command.cmd_id)
                        if command.data_kind != "weight" or not command.op_id.startswith("moe."):
                            self.assertEqual(command, changed)
                        elif command.port == "flash":
                            self.assertTrue(math.isclose(changed.bandwidth_bytes_per_second,
                                                        command.bandwidth_bytes_per_second / 6))
                        if (command.data_kind == "weight" and command.op_id.startswith("moe.")
                                and command.port == "dram"
                                and before.operators[command.op_id].weight_memory.source == WeightSource.DRAM_FLASH):
                            flash_before = next(c for c in before_graph.commands if isinstance(c, PUMemoryReadCmd)
                                                and c.op_id == command.op_id and c.port == "flash")
                            flash_after = after_graph.command(flash_before.cmd_id)
                            self.assertAlmostEqual(command.size_bytes + flash_before.size_bytes,
                                                   changed.size_bytes + flash_after.size_bytes)
                            self.assertGreater(changed.size_bytes, command.size_bytes)
                            self.assertLess(flash_after.size_bytes, flash_before.size_bytes)
                            self.assertTrue(math.isclose(changed.size_bytes / flash_after.size_bytes,
                                                        changed.bandwidth_bytes_per_second / flash_after.bandwidth_bytes_per_second))
                            dual_port_ops += 1
                self.assertGreater(dual_port_ops, 0)

    def test_no_kv_layout_changes_only_indexer_and_adds_delay_for_every_model_layer(self):
        for model in ("deepseek-v3", "deepseek-v3.2", "glm-5.1", "minimax-m3"):
            with self.subTest(model=model):
                original = mapper_for_model(model, self.hardware).map(
                    replace(self.request, model_name=model), hbf_bandwidth_table=self.table)
                mapped = apply_ablation(original, self.hardware, "NoKVLayout")
                for layer, before in zip((mapped, *mapped.additional_layer_mappings),
                                         (original, *original.additional_layer_mappings)):
                    for key, op in layer.operators.items():
                        self.assertEqual(layer.operator_mappings[key].hardware_config, self.hardware)
                        if op.weight_memory is None:
                            continue
                        self.assertEqual(op.partition_chain, before.operators[key].partition_chain)
                        base_rate = before.operators[key].weight_memory.flash_bandwidth_bytes_per_second or 4e9
                        measured_score = key.endswith("indexer_score") and op.weight_memory.source != WeightSource.DRAM
                        self.assertAlmostEqual(op.weight_memory.flash_bandwidth_bytes_per_second,
                                               base_rate * (0.87 if measured_score else 1))
                graph = TreeParser(self.hardware).parse(mapped.root_node, mapped.operators, mapped.operator_mappings)
                result = Simulator(self.hardware).run(graph, mapping_result=mapped)
                raw = result.latency_ns
                timing = latency_metrics(result, model, "NoKVLayout")
                per_layer = 25_000 if model == "minimax-m3" else 14_000
                self.assertEqual(timing["additional_model_latency_ns"], per_layer * original.model_config.num_hidden_layers)
                self.assertEqual(timing["model_step_latency_ns"], raw + per_layer * original.model_config.num_hidden_layers)
                self.assertEqual(latency_metrics(result, model, "NoKVLayout"), timing)  # No accumulation on repeated reporting.
                self.assertEqual(result.latency_ns, raw)
                self.assertEqual(latency_metrics(result, model, "XFold")["model_step_latency_ns"], raw)
                self.assertEqual(latency_metrics(result, model, "NoMoELayout")["additional_model_latency_ns"], 0)
                if model == "minimax-m3":
                    self.assertEqual(timing["additional_model_latency_ns"], 1_500_000)
                    self.assertEqual(sum(count * duration for _, count, duration in timing["adjusted_model_latency_components"]),
                                     timing["model_step_latency_ns"])

    def test_no_kv_layout_requires_measured_indexer_bandwidth(self):
        unmapped_rates = self.mapper.map(self.request)
        with self.assertRaisesRegex(ValueError, "CSV Indexer bandwidth"):
            apply_ablation(unmapped_rates, self.hardware, "NoKVLayout")

    def test_lowering_balances_dual_ports_using_operator_dram_bandwidth(self):
        op = BMMOp("op", 1, 1, 100, 100, input_source="none", output_destination="none",
                   weight_memory=WeightMemory(WeightSource.DRAM_FLASH, 8e9))
        hardware = half_grid_hardware(self.hardware, "dram")
        tree = RootNode(1, ("op",), GroupNode((OpNode("op"),)))
        graph = TreeParser(self.hardware).parse(tree, {"op": op}, {
            "op": OperatorMapping("M", "chip0.pu0", hardware_config=hardware)}, layer_count=1)
        reads = {cmd.port: cmd for cmd in graph.commands if isinstance(cmd, PUMemoryReadCmd)}
        self.assertEqual(reads["dram"].size_bytes, 8000)
        self.assertEqual(reads["flash"].size_bytes, 2000)
        result = GraphExecutor.simulate(graph, self.hardware, operator_hardware={"op": hardware})
        self.assertEqual(graph.command("i0.core").memory_duration_ns, 250)
        for read in reads.values():
            self.assertEqual(result.executions[read.cmd_id].duration_ns, 250)

    def test_different_hardware_views_still_contend_for_same_physical_port(self):
        commands = (
            PUMemoryReadCmd("a", "a", "chip0.pu0.dram", 32000, "dram", 32e9),
            PUMemoryReadCmd("b", "b", "chip0.pu0.dram", 32000, "dram", 16e9),
        )
        result = GraphExecutor.simulate(CommandGraph(commands), self.hardware,
                                       operator_hardware={"a": half_grid_hardware(self.hardware, "dram")})
        self.assertEqual(result.executions["a"].duration_ns, 1000)
        self.assertEqual(result.executions["b"].duration_ns, 2000)
        self.assertEqual(result.executions["b"].start_time_ns, result.executions["a"].end_time_ns)

    def test_mixed_grid_search_and_simulator_use_effective_hardware_including_dense(self):
        mapped = optimize_partition_chains(apply_ablation(self.original, self.hardware, "NoDRAMShare"), self.hardware)
        for key, op in mapped.operators.items():
            if op.weight_memory is not None:
                hw = mapped.operator_mappings[key].hardware_config
                op.partition_chain.validate_grid(hw.chip.noc.mesh_rows, hw.chip.noc.mesh_columns)
        graph = TreeParser(self.hardware).parse(mapped.root_node, mapped.operators, mapped.operator_mappings)
        simulator = Simulator(self.hardware)
        result = simulator.run(graph, mapping_result=mapped)
        self.assertEqual(simulator.run_latency(graph, mapping_result=mapped), result.latency_ns)
        for cmd in result.command_results:
            if cmd.parameters.get("port") == "dram":
                hw = mapped.operator_mappings[cmd.op_id].hardware_config
                expected = math.ceil(cmd.parameters["size_bytes"] * 1e9
                                     / hw.chip.logic_die.processing_unit.dram_bandwidth_bytes_per_second)
                self.assertEqual(cmd.end_time_ns - cmd.start_time_ns, expected)
        dense = mapped.additional_layer_mappings[0]
        dense_graph = TreeParser(self.hardware).parse(dense.root_node, dense.operators, dense.operator_mappings)
        dense_result = simulator.run(dense_graph, mapping_result=dense)
        self.assertEqual(dict((name, duration) for name, _, duration in result.model_latency_components)["dense"],
                         dense_result.pipeline_layer_time_ns)
        self.assertEqual(result.latency_ns, sum(count * duration for _, count, duration in result.model_latency_components))

    def test_baseline_hardware_views_do_not_change_timing(self):
        latencies = []
        for mapped in (self.original, apply_ablation(self.original, self.hardware, "XFold")):
            graph = TreeParser(self.hardware).parse(mapped.root_node, mapped.operators, mapped.operator_mappings)
            latencies.append(Simulator(self.hardware).run(graph, mapping_result=mapped).latency_ns)
        self.assertEqual(*latencies)

    def test_nested_base_config_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hardware.json"
            path.write_text(json.dumps({"chip": {"logic_die": {"processing_unit": {
                "dram_bandwidth_bytes_per_second": 20e9, "flash_bandwidth_bytes_per_second": 7e9}}}}))
            hardware = load_hardware(path)
            doubled = half_grid_hardware(hardware, "dram")
            self.assertEqual(doubled.chip.logic_die.processing_unit.dram_bandwidth_bytes_per_second, 40e9)
            self.assertEqual(doubled.chip.logic_die.processing_unit.flash_bandwidth_bytes_per_second, 7e9)
            path.write_text('{"typo": 1}')
            with self.assertRaises(ValueError):
                load_hardware(path)
        for factor in (0, -1, float("nan"), float("inf"), 1.1):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                apply_ablation(self.original, self.hardware, "NoKVLayout", kv_factor=factor)

    def test_pareto_keeps_equal_points_and_excludes_strictly_dominated(self):
        rows = [dict(tokens_per_second_per_user=x, tokens_per_second_per_chip=y)
                for x, y in ((1, 2), (2, 2), (2, 2), (3, 1))]
        self.assertEqual([row["is_pareto"] for row in mark_pareto(rows)], [False, True, True, True])


if __name__ == "__main__":
    unittest.main()
