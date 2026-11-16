"""Explicit hardware for regressions of historical mapping layouts."""

from dataclasses import replace

from srammachine.hardware import DEFAULT_HARDWARE_CONFIG


LEGACY_HARDWARE_CONFIG = replace(
    DEFAULT_HARDWARE_CONFIG,
    chip_count=16,
    chip=replace(
        DEFAULT_HARDWARE_CONFIG.chip,
        logic_die=replace(
            DEFAULT_HARDWARE_CONFIG.chip.logic_die,
            pu_mesh_rows=4,
            pu_mesh_columns=4,
        ),
        noc=replace(DEFAULT_HARDWARE_CONFIG.chip.noc, mesh_rows=8, mesh_columns=8),
    ),
)
