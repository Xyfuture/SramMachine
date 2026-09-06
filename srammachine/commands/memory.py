"""DramResource read/write actions and SramResource weight loading."""
from dataclasses import dataclass
from .base import Command, integer


@dataclass(frozen=True)
class DramCmd(Command):
    size_bytes: int

    def __post_init__(self) -> None:
        super().__post_init__()
        integer("size_bytes", self.size_bytes)


@dataclass(frozen=True)
class DramReadCmd(DramCmd):
    """Read DRAM into SRAM, possibly ahead of the consuming operator."""


@dataclass(frozen=True)
class DramWriteCmd(DramCmd):
    """Write SRAM data, such as newly generated KV entries, to DRAM."""


@dataclass(frozen=True)
class WeightLoadCmd(Command):
    """Supply bytes from SRAM to the array independently of DRAM access."""
    size_bytes: int

    def __post_init__(self) -> None:
        super().__post_init__()
        integer("size_bytes", self.size_bytes)
