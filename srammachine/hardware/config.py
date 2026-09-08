"""Typed hardware configuration for the UltraStack simulator.

All byte and bandwidth values use decimal SI units: MB = 10**6 bytes,
GB = 10**9 bytes, and TB/s = 10**12 bytes per second.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Real


_MB = 10**6
_GB = 10**9
_TFLOPS = 10**12
_TB_PER_SECOND = 10**12


def _positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _positive_real(name: str, value: Real) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def _nonnegative_real(name: str, value: Real) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a nonnegative finite number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a nonnegative finite number")


def _nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


class NoCBandwidthPreset(Enum):
    """Supported directional bandwidths for one adjacent-PU NoC link."""

    GBPS_256 = 256 * _GB
    GBPS_512 = 512 * _GB

    @property
    def bytes_per_second(self) -> int:
        return self.value


@dataclass(frozen=True)
class SystolicArrayConfig:
    """Configuration shared by every systolic array in a PU."""

    rows: int = 16
    columns: int = 128
    frequency_hz: int = 1_000_000_000
    precision: str = "fp8"
    flops_per_mac: int = 2
    weight_bandwidth_bytes_per_second: int = 128 * _GB

    def __post_init__(self) -> None:
        for name in ("rows", "columns", "frequency_hz", "flops_per_mac"):
            _positive_integer(name, getattr(self, name))
        _nonempty("precision", self.precision)
        _positive_real(
            "weight_bandwidth_bytes_per_second",
            self.weight_bandwidth_bytes_per_second,
        )

    @property
    def peak_flops(self) -> int:
        return self.rows * self.columns * self.frequency_hz * self.flops_per_mac


@dataclass(frozen=True)
class ProcessingUnitConfig:
    """A PU containing identical systolic arrays."""

    systolic_array_count: int = 32
    systolic_array: SystolicArrayConfig = field(
        default_factory=SystolicArrayConfig
    )

    def __post_init__(self) -> None:
        _positive_integer("systolic_array_count", self.systolic_array_count)
        if not isinstance(self.systolic_array, SystolicArrayConfig):
            raise ValueError("systolic_array must be a SystolicArrayConfig")

    @property
    def peak_flops(self) -> int:
        return self.systolic_array_count * self.systolic_array.peak_flops

    @property
    def weight_bandwidth_bytes_per_second(self) -> int:
        return (
            self.systolic_array_count
            * self.systolic_array.weight_bandwidth_bytes_per_second
        )


@dataclass(frozen=True)
class VectorUnitConfig:
    """Aggregate vector-compute resource in one logic die."""

    peak_flops: int = 128 * _TFLOPS
    precision: str = "fp16"

    def __post_init__(self) -> None:
        _positive_real("peak_flops", self.peak_flops)
        _nonempty("precision", self.precision)


@dataclass(frozen=True)
class MemoryConfig:
    """DRAM and SRAM resources associated with one stack/logic die."""

    dram_capacity_bytes: int = 40 * _GB
    dram_bandwidth_bytes_per_second: int = 16 * _TB_PER_SECOND
    sram_capacity_bytes: int = 768 * _MB
    sram_bandwidth_bytes_per_second: int = 64 * _TB_PER_SECOND

    def __post_init__(self) -> None:
        for name in (
            "dram_capacity_bytes",
            "dram_bandwidth_bytes_per_second",
            "sram_capacity_bytes",
            "sram_bandwidth_bytes_per_second",
        ):
            _positive_real(name, getattr(self, name))


@dataclass(frozen=True)
class LogicDieConfig:
    """One logic die and its associated stack-local resources."""

    pu_mesh_rows: int = 4
    pu_mesh_columns: int = 4
    processing_unit: ProcessingUnitConfig = field(
        default_factory=ProcessingUnitConfig
    )
    vector_unit: VectorUnitConfig = field(default_factory=VectorUnitConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def __post_init__(self) -> None:
        _positive_integer("pu_mesh_rows", self.pu_mesh_rows)
        _positive_integer("pu_mesh_columns", self.pu_mesh_columns)
        if not isinstance(self.processing_unit, ProcessingUnitConfig):
            raise ValueError("processing_unit must be a ProcessingUnitConfig")
        if not isinstance(self.vector_unit, VectorUnitConfig):
            raise ValueError("vector_unit must be a VectorUnitConfig")
        if not isinstance(self.memory, MemoryConfig):
            raise ValueError("memory must be a MemoryConfig")

    @property
    def processing_unit_count(self) -> int:
        return self.pu_mesh_rows * self.pu_mesh_columns

    @property
    def systolic_array_count(self) -> int:
        return (
            self.processing_unit_count
            * self.processing_unit.systolic_array_count
        )

    @property
    def systolic_peak_flops(self) -> int:
        return self.processing_unit_count * self.processing_unit.peak_flops

    @property
    def systolic_weight_bandwidth_bytes_per_second(self) -> int:
        return (
            self.processing_unit_count
            * self.processing_unit.weight_bandwidth_bytes_per_second
        )


@dataclass(frozen=True)
class NoCConfig:
    """Unified chip-level PU mesh.

    A link is full duplex: ``link_bandwidth_bytes_per_second`` is the capacity
    of one direction and the reverse direction has an independent equal lane.
    """

    mesh_rows: int = 8
    mesh_columns: int = 8
    link_bandwidth_bytes_per_second: int = (
        NoCBandwidthPreset.GBPS_256.bytes_per_second
    )
    link_latency_ns: float = 0
    full_duplex: bool = True
    seamless_across_logic_dies: bool = True
    supports_in_network_reduction: bool = True

    def __post_init__(self) -> None:
        _positive_integer("mesh_rows", self.mesh_rows)
        _positive_integer("mesh_columns", self.mesh_columns)
        _positive_real(
            "link_bandwidth_bytes_per_second",
            self.link_bandwidth_bytes_per_second,
        )
        _nonnegative_real("link_latency_ns", self.link_latency_ns)
        for name in (
            "full_duplex",
            "seamless_across_logic_dies",
            "supports_in_network_reduction",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")

    @property
    def bidirectional_link_bandwidth_bytes_per_second(self) -> int:
        multiplier = 2 if self.full_duplex else 1
        return multiplier * self.link_bandwidth_bytes_per_second


@dataclass(frozen=True)
class ChipConfig:
    """A chip containing a tiled set of identical logic dies."""

    logic_die_mesh_rows: int = 2
    logic_die_mesh_columns: int = 2
    logic_die: LogicDieConfig = field(default_factory=LogicDieConfig)
    noc: NoCConfig = field(default_factory=NoCConfig)

    def __post_init__(self) -> None:
        _positive_integer("logic_die_mesh_rows", self.logic_die_mesh_rows)
        _positive_integer("logic_die_mesh_columns", self.logic_die_mesh_columns)
        if not isinstance(self.logic_die, LogicDieConfig):
            raise ValueError("logic_die must be a LogicDieConfig")
        if not isinstance(self.noc, NoCConfig):
            raise ValueError("noc must be a NoCConfig")

        expected_rows = self.logic_die_mesh_rows * self.logic_die.pu_mesh_rows
        expected_columns = (
            self.logic_die_mesh_columns * self.logic_die.pu_mesh_columns
        )
        if self.noc.mesh_rows != expected_rows:
            raise ValueError(
                "noc.mesh_rows must equal logic_die_mesh_rows * "
                "logic_die.pu_mesh_rows"
            )
        if self.noc.mesh_columns != expected_columns:
            raise ValueError(
                "noc.mesh_columns must equal logic_die_mesh_columns * "
                "logic_die.pu_mesh_columns"
            )

    @property
    def logic_die_count(self) -> int:
        return self.logic_die_mesh_rows * self.logic_die_mesh_columns

    @property
    def processing_unit_count(self) -> int:
        return self.logic_die_count * self.logic_die.processing_unit_count

    @property
    def systolic_array_count(self) -> int:
        return self.logic_die_count * self.logic_die.systolic_array_count

    @property
    def systolic_peak_flops(self) -> int:
        return self.logic_die_count * self.logic_die.systolic_peak_flops

    @property
    def vector_peak_flops(self) -> int:
        return self.logic_die_count * self.logic_die.vector_unit.peak_flops

    @property
    def horizontal_logic_die_boundary_link_count(self) -> int:
        """Links crossing one boundary between left/right logic dies."""

        return self.logic_die.pu_mesh_rows

    @property
    def vertical_logic_die_boundary_link_count(self) -> int:
        """Links crossing one boundary between upper/lower logic dies."""

        return self.logic_die.pu_mesh_columns

    @property
    def horizontal_logic_die_boundary_bandwidth_bytes_per_second(self) -> int:
        return (
            self.horizontal_logic_die_boundary_link_count
            * self.noc.link_bandwidth_bytes_per_second
        )

    @property
    def vertical_logic_die_boundary_bandwidth_bytes_per_second(self) -> int:
        return (
            self.vertical_logic_die_boundary_link_count
            * self.noc.link_bandwidth_bytes_per_second
        )

    @property
    def vertical_bisection_link_count(self) -> int:
        """Links crossing a vertical cut through the full chip mesh."""

        return self.noc.mesh_rows

    @property
    def horizontal_bisection_link_count(self) -> int:
        """Links crossing a horizontal cut through the full chip mesh."""

        return self.noc.mesh_columns

    @property
    def vertical_bisection_bandwidth_bytes_per_second(self) -> int:
        return (
            self.vertical_bisection_link_count
            * self.noc.link_bandwidth_bytes_per_second
        )

    @property
    def horizontal_bisection_bandwidth_bytes_per_second(self) -> int:
        return (
            self.horizontal_bisection_link_count
            * self.noc.link_bandwidth_bytes_per_second
        )


@dataclass(frozen=True)
class InterChipFabricConfig:
    """Per-chip interface to a full-mesh switched inter-chip fabric."""

    topology: str = "full_mesh_switch_fabric"
    per_chip_directional_bandwidth_bytes_per_second: int = 800 * _GB
    full_duplex: bool = True

    def __post_init__(self) -> None:
        _nonempty("topology", self.topology)
        _positive_real(
            "per_chip_directional_bandwidth_bytes_per_second",
            self.per_chip_directional_bandwidth_bytes_per_second,
        )
        if type(self.full_duplex) is not bool:
            raise ValueError("full_duplex must be a boolean")

    @property
    def per_chip_bidirectional_bandwidth_bytes_per_second(self) -> int:
        multiplier = 2 if self.full_duplex else 1
        return (
            multiplier
            * self.per_chip_directional_bandwidth_bytes_per_second
        )


@dataclass(frozen=True)
class HardwareConfig:
    """Top-level system hardware configuration."""

    chip_count: int = 16
    chip: ChipConfig = field(default_factory=ChipConfig)
    inter_chip_fabric: InterChipFabricConfig = field(
        default_factory=InterChipFabricConfig
    )

    def __post_init__(self) -> None:
        if type(self.chip_count) is not int or self.chip_count not in (16, 32):
            raise ValueError("chip_count must be either 16 or 32")
        if not isinstance(self.chip, ChipConfig):
            raise ValueError("chip must be a ChipConfig")
        if not isinstance(self.inter_chip_fabric, InterChipFabricConfig):
            raise ValueError(
                "inter_chip_fabric must be an InterChipFabricConfig"
            )

    @property
    def logic_die_count(self) -> int:
        return self.chip_count * self.chip.logic_die_count

    @property
    def processing_unit_count(self) -> int:
        return self.chip_count * self.chip.processing_unit_count

    @property
    def systolic_array_count(self) -> int:
        return self.chip_count * self.chip.systolic_array_count

    @property
    def systolic_peak_flops(self) -> int:
        return self.chip_count * self.chip.systolic_peak_flops

    @property
    def vector_peak_flops(self) -> int:
        return self.chip_count * self.chip.vector_peak_flops


DEFAULT_HARDWARE_CONFIG = HardwareConfig()
