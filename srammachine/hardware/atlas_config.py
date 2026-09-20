"""Atlas hardware preset built from the simulator's canonical config types."""

from dataclasses import replace

from .config import DEFAULT_HARDWARE_CONFIG as _BASE_HARDWARE_CONFIG


_atlas_processing_unit = replace(
    _BASE_HARDWARE_CONFIG.chip.logic_die.processing_unit,
    systolic_array_count=8,
)
_atlas_memory = replace(
    _BASE_HARDWARE_CONFIG.chip.logic_die.memory,
    dram_capacity_bytes=80 * 10**9,
    sram_bandwidth_bytes_per_second=16 * 10**12,
)
_atlas_logic_die = replace(
    _BASE_HARDWARE_CONFIG.chip.logic_die,
    processing_unit=_atlas_processing_unit,
    memory=_atlas_memory,
)

ATLAS_HARDWARE_CONFIG = replace(
    _BASE_HARDWARE_CONFIG,
    moe_time_multiplier=1.0,
    chip=replace(
        _BASE_HARDWARE_CONFIG.chip,
        logic_die=_atlas_logic_die,
    ),
)

# Preserve the module-level API used by standalone hardware config modules.
DEFAULT_HARDWARE_CONFIG = ATLAS_HARDWARE_CONFIG


__all__ = ["ATLAS_HARDWARE_CONFIG", "DEFAULT_HARDWARE_CONFIG"]
