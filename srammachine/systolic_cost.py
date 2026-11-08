"""Systolic-array compute cost estimates."""

import math

from srammachine.commands import GemmCmd
from srammachine.hardware import HardwareConfig


def systolic_gemm_duration_ns(
    command: GemmCmd, hardware_config: HardwareConfig,
) -> int:
    """Return GEMM duration after searching array-count factorizations."""
    pu = hardware_config.chip.logic_die.processing_unit
    array = pu.systolic_array
    array_count = pu.systolic_array_count
    m = command.B * command.M
    best_duration_ns = None

    for a in range(1, array_count + 1):
        if array_count % a:
            continue
        b = array_count // a
        duration_numerator = (
            m
            * math.ceil(command.K / (a * array.columns))
            * math.ceil(command.N / (b * array.rows))
            * 1_000_000_000
        )
        duration_ns = (
            duration_numerator + array.frequency_hz - 1
        ) // array.frequency_hz
        if best_duration_ns is None or duration_ns < best_duration_ns:
            best_duration_ns = duration_ns

    return best_duration_ns
