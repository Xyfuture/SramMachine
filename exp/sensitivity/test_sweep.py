"""Regression checks confined to the sensitivity experiment directory."""

from dataclasses import replace
import unittest

from exp.sensitivity.sweep import FACTORS, PARAMETERS, bandwidth_table, mark_pareto, scaled_hardware
from srammachine.hardware import HardwareConfig
from srammachine.inference import InferenceConfig
from srammachine.mapping import DEFAULT_HBF_BANDWIDTH_CSV, HBFBandwidthTable, HardwareMappingRequest, mapper_for_model


class SensitivityTests(unittest.TestCase):
    """Check single-resource changes, measured rates and per-curve dominance."""

    def test_baselines_and_parameter_isolation(self) -> None:
        baseline = HardwareConfig()
        for parameter in PARAMETERS:
            for factor in FACTORS:
                with self.subTest(parameter=parameter, factor=factor):
                    changed = scaled_hardware(parameter, factor)
                    die = changed.chip.logic_die
                    pu = die.processing_unit
                    original_die = baseline.chip.logic_die
                    original_pu = original_die.processing_unit
                    self.assertEqual(changed.inter_chip_fabric, baseline.inter_chip_fabric)
                    if parameter == "hbf":
                        self.assertEqual(pu.flash_bandwidth_bytes_per_second,
                                         original_pu.flash_bandwidth_bytes_per_second * factor)
                        pu = replace(pu, flash_bandwidth_bytes_per_second=original_pu.flash_bandwidth_bytes_per_second)
                    elif parameter == "hbm":
                        self.assertEqual(pu.dram_bandwidth_bytes_per_second,
                                         original_pu.dram_bandwidth_bytes_per_second * factor)
                        self.assertEqual(die.memory.dram_bandwidth_bytes_per_second,
                                         original_die.memory.dram_bandwidth_bytes_per_second * factor)
                        pu = replace(pu, dram_bandwidth_bytes_per_second=original_pu.dram_bandwidth_bytes_per_second)
                        die = replace(die, memory=original_die.memory)
                    elif parameter == "noc":
                        for operation in ("PUHorizontal", "PUVertical", "PUMesh"):
                            self.assertEqual(changed.chip.noc.bandwidth_for(operation),
                                             baseline.chip.noc.bandwidth_for(operation) * factor)
                        changed = replace(changed, chip=replace(changed.chip, noc=baseline.chip.noc))
                    else:
                        self.assertEqual(pu.peak_flops, original_pu.peak_flops * factor)
                        self.assertEqual(die.vector_unit.peak_flops, original_die.vector_unit.peak_flops * factor)
                        pu = replace(pu, systolic_array=original_pu.systolic_array)
                        die = replace(die, vector_unit=original_die.vector_unit)
                    restored = replace(changed, chip=replace(changed.chip,
                        logic_die=replace(die, processing_unit=pu)))
                    self.assertEqual(restored, baseline)
                    if factor == 1:
                        self.assertEqual(scaled_hardware(parameter, factor), baseline)
        self.assertEqual(HardwareConfig(), baseline)

    def test_logic_scopes(self) -> None:
        baseline = HardwareConfig().chip.logic_die
        vector = scaled_hardware("logic", 0.5, "vector").chip.logic_die
        pu = scaled_hardware("logic", 0.5, "pu").chip.logic_die
        self.assertEqual(vector.processing_unit, baseline.processing_unit)
        self.assertEqual(vector.vector_unit.peak_flops, baseline.vector_unit.peak_flops / 2)
        self.assertEqual(pu.vector_unit, baseline.vector_unit)
        self.assertEqual(pu.processing_unit.peak_flops, baseline.processing_unit.peak_flops / 2)

    def test_csv_rates_reach_xfold_prefetch_and_indexer(self) -> None:
        baseline = HBFBandwidthTable.from_csv(DEFAULT_HBF_BANDWIDTH_CSV)
        hardware = HardwareConfig()
        request = HardwareMappingRequest("deepseek-v3.2", InferenceConfig(16, 16384, 600, mtp_enabled=True), xfold=True)
        original = mapper_for_model(request.model_name, hardware).map(request, hbf_bandwidth_table=baseline)
        for factor in FACTORS:
            table = bandwidth_table(DEFAULT_HBF_BANDWIDTH_CSV, factor)
            mapped = mapper_for_model(request.model_name, hardware).map(request, hbf_bandwidth_table=table)
            rates = {}
            for op_id, op in mapped.operators.items():
                if op.weight_memory and op.weight_memory.flash_bandwidth_bytes_per_second is not None:
                    base_rate = original.operators[op_id].weight_memory.flash_bandwidth_bytes_per_second
                    expected = ((base_rate - 8e9) * factor + 8e9 if op_id.endswith("indexer_score")
                                else base_rate * factor)
                    self.assertAlmostEqual(op.weight_memory.flash_bandwidth_bytes_per_second / expected, 1.0)
                    rates[op_id] = expected
            self.assertTrue(rates)
            self.assertTrue(any(mapped.operators[key].flash_prefetch_bytes_per_request for key in rates))
            self.assertIn("moe.indexer_score", rates)
            # Both sparse MTP1 and the dense MTP0 fallback are multiplied.
            for kind in ("sparse_kv", "dense_indexer"):
                self.assertEqual(table.lookup(request.model_name, kind, True, 16, 16384),
                                 baseline.lookup(request.model_name, kind, True, 16, 16384) * factor)

    def test_pareto_keeps_ties_and_drops_dominated_points(self) -> None:
        rows = [{"tokens_per_second_per_user": x, "tokens_per_second_per_chip": y}
                for x, y in ((1, 1), (2, 1), (1, 2), (2, 1))]
        self.assertEqual([r["is_pareto"] for r in mark_pareto(rows)], [False, True, True, True])


if __name__ == "__main__":
    unittest.main()


class RemoteSupplyTests(unittest.TestCase):
    def test_supply_caps_and_read_commands(self):
        from exp.sensitivity.sweep import cap_remote_supply
        from srammachine.pipetree import TreeParser
        from srammachine.commands import PUMemoryReadCmd
        table = HBFBandwidthTable.from_csv(DEFAULT_HBF_BANDWIDTH_CSV)
        request = HardwareMappingRequest("deepseek-v3.2", InferenceConfig(16, 32768, 600, mtp_enabled=False), xfold=True)
        baseline_hw = scaled_hardware("noc", 1)
        base = mapper_for_model(request.model_name, baseline_hw).map(request, hbf_bandwidth_table=table)
        self.assertIs(cap_remote_supply(base, baseline_hw, 1), base)
        for factor in (0.5, 0.75, 1.25, 1.5):
            hw = scaled_hardware("noc", factor)
            mapped = mapper_for_model(request.model_name, hw).map(request, hbf_bandwidth_table=table)
            capped = cap_remote_supply(mapped, hw, factor)
            remote, pooled = min(1, factor), (1 + min(1, factor)) / 2
            self.assertEqual(hw.chip.logic_die.processing_unit.dram_bandwidth_bytes_per_second, 16e9)
            for key, op in capped.operators.items():
                if op.weight_memory is None:
                    continue
                actual_hw = capped.operator_mappings[key].hardware_config
                self.assertEqual(actual_hw.chip.logic_die.processing_unit.dram_bandwidth_bytes_per_second, 16e9 * pooled)
                rate = base.operators[key].weight_memory.flash_bandwidth_bytes_per_second or 4e9
                expected = ((rate - 8e9) * pooled + 8e9 * remote if key.endswith("indexer_score")
                            else rate * (remote if op.flash_prefetch_bytes_per_request else pooled))
                self.assertAlmostEqual(op.weight_memory.flash_bandwidth_bytes_per_second, expected, delta=1e-5)
            graph = TreeParser(hw).parse(capped.root_node, capped.operators, capped.operator_mappings, layer_count=1)
            reads = [c for c in graph.commands if isinstance(c, PUMemoryReadCmd)]
            self.assertTrue(any(c.port == "dram" for c in reads))
            for cmd in reads:
                if cmd.port == "dram":
                    self.assertEqual(cmd.bandwidth_bytes_per_second, 16e9 * pooled)


class AttentionEfficiencyTests(unittest.TestCase):
    def test_only_attention_compute_peak_changes(self):
        from exp.sensitivity.sweep import apply_attention_efficiency, cap_remote_supply
        table = HBFBandwidthTable.from_csv(DEFAULT_HBF_BANDWIDTH_CSV)
        request = HardwareMappingRequest("deepseek-v3.2", InferenceConfig(16,32768,600,mtp_enabled=False), xfold=True)
        for factor in (.5,1,1.5):
            hw = scaled_hardware("noc", factor)
            mapped = mapper_for_model(request.model_name, hw).map(request, hbf_bandwidth_table=table)
            mapped = cap_remote_supply(mapped, hw, factor)
            result = apply_attention_efficiency(mapped, hw, .423)
            self.assertEqual(result.operators, mapped.operators)
            for key, policy in result.operator_mappings.items():
                before = mapped.operator_mappings[key]
                if key == "attn.flash_attention":
                    old = before.hardware_config or hw
                    new = policy.hardware_config
                    self.assertAlmostEqual(new.chip.logic_die.processing_unit.peak_flops / old.chip.logic_die.processing_unit.peak_flops, .423)
                    self.assertEqual(new.chip.logic_die.memory, old.chip.logic_die.memory)
                    self.assertEqual(new.chip.noc, old.chip.noc)
                    self.assertEqual(new.chip.logic_die.processing_unit.dram_bandwidth_bytes_per_second,old.chip.logic_die.processing_unit.dram_bandwidth_bytes_per_second)
                else:
                    self.assertEqual(policy, before)
