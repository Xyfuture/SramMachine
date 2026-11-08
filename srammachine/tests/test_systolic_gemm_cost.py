"""Systolic-array tiling cost regressions."""

import unittest
from dataclasses import replace

from srammachine.commands import GemmCmd
from srammachine.hardware import (
    DEFAULT_HARDWARE_CONFIG,
    ProcessingUnitConfig,
    SystolicArrayConfig,
)
from srammachine.systolic_cost import systolic_gemm_duration_ns


class SystolicGemmCostTests(unittest.TestCase):
    def test_searches_array_count_factorizations(self):
        processing_unit = ProcessingUnitConfig(
            systolic_array_count=6,
            systolic_array=SystolicArrayConfig(
                rows=3,
                columns=2,
                frequency_hz=1_000_000_000,
            ),
        )
        logic_die = replace(
            DEFAULT_HARDWARE_CONFIG.chip.logic_die,
            processing_unit=processing_unit,
        )
        hardware_config = replace(
            DEFAULT_HARDWARE_CONFIG,
            chip=replace(DEFAULT_HARDWARE_CONFIG.chip, logic_die=logic_die),
        )
        command = GemmCmd("gemm", "gemm", "chip0.pu", B=2, M=1, K=25, N=13)

        # a=1, b=6 yields 2 * ceil(25 / 2) * ceil(13 / 18) = 26 cycles.
        self.assertEqual(
            systolic_gemm_duration_ns(command, hardware_config),
            26,
        )


if __name__ == "__main__":
    unittest.main()
