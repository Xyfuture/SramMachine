"""Indexer effective bandwidth must follow NoC and revert under NoFlashShare."""

import math
import unittest

from exp.ablation.policy import apply_ablation
from exp.sensitivity.sweep import scaled_hardware, bandwidth_table
from srammachine.commands import PUMemoryReadCmd
from srammachine.hardware import HardwareConfig
from srammachine.inference import InferenceConfig
from srammachine.mapping import DEFAULT_HBF_BANDWIDTH_CSV, HBFBandwidthTable, HardwareMappingRequest, mapper_for_model
from srammachine.pipetree import TreeParser
from srammachine.simulator import GraphExecutor


class IndexerNoCBandwidthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = HBFBandwidthTable.from_csv(DEFAULT_HBF_BANDWIDTH_CSV)

    def test_noc_hbf_scaling_and_half_grid_reversion(self):
        for model in ('minimax-m3', 'deepseek-v3.2', 'glm-5.1'):
            for parameter, factor in (('noc', 0.5), ('noc', 1), ('noc', 1.5), ('hbf', 0.5)):
                with self.subTest(model=model, parameter=parameter, factor=factor):
                    hw = scaled_hardware(parameter, factor)
                    table = bandwidth_table(DEFAULT_HBF_BANDWIDTH_CSV, factor) if parameter == 'hbf' else self.table
                    request = HardwareMappingRequest(model, InferenceConfig(256, 32768, 600, mtp_enabled=False), xfold=True)
                    mapped = mapper_for_model(model, hw).map(request, hbf_bandwidth_table=table)
                    raw = table.lookup(model, 'dense_indexer', False, 256, 32768)
                    extra = hw.chip.noc.link_bandwidth_bytes_per_second * 128
                    for variant in ('XFold', 'NoFlashShare', 'NoDRAMFLashShare'):
                        ablated = apply_ablation(mapped, hw, variant)
                        graph = TreeParser(hw).parse(ablated.root_node, ablated.operators, ablated.operator_mappings, layer_count=1)
                        read = next(c for c in graph.commands if isinstance(c, PUMemoryReadCmd)
                                    and c.op_id == 'moe.indexer_score')
                        pu_count = ablated.operator_mappings[read.op_id].hardware_config.chip.processing_unit_count
                        expected = raw * 2048 + (extra if variant == 'XFold' else 0)
                        self.assertAlmostEqual(read.bandwidth_bytes_per_second * pu_count / expected, 1)
                        if model == 'minimax-m3' and parameter == 'noc' and factor == 1:
                            execution = GraphExecutor.simulate(graph, hardware_config=hw).executions[read.cmd_id]
                            self.assertEqual(execution.end_time_ns - execution.start_time_ns,
                                             math.ceil(read.size_bytes * 1e9 / read.bandwidth_bytes_per_second))
                    # KV prefetch keeps the measured rate, not the Indexer increment.
                    prefetch = next(c for c in graph.commands if isinstance(c, PUMemoryReadCmd)
                                    and c.data_kind == 'next_layer_kv')
                    # Combined sharing doubles FC2 data per PU but not its Flash rate.
                    self.assertEqual(prefetch.bandwidth_bytes_per_second,
                                     table.lookup(model, 'sparse_kv', False, 256, 32768))

    def test_no_csv_fallback_and_ordinary_indexer(self):
        hw = HardwareConfig()
        for model in ('minimax-m3', 'deepseek-v3.2', 'glm-5.1'):
            mapped = mapper_for_model(model, hw).map(HardwareMappingRequest(
                model, InferenceConfig(256, 32768, 600, mtp_enabled=False)))
            self.assertEqual(mapped.operators['attn.indexer_score'].weight_memory.flash_bandwidth_bytes_per_second, 12e9)
            ablated = apply_ablation(mapped, hw, 'NoFlashShare')
            self.assertEqual(ablated.operators['attn.indexer_score'].weight_memory.flash_bandwidth_bytes_per_second, 8e9)
