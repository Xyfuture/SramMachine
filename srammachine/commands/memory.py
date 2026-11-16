"""Direct DRAM/Flash transfers and DRAM writeback actions."""
from dataclasses import dataclass, field
import math
from typing import Optional

from .base import Command, integer, nonempty


@dataclass(frozen=True)
class PUMemoryReadCmd(Command):
    """One independent PU port stream; fractional bytes model ideal striping."""

    size_bytes: float
    port: str
    bandwidth_bytes_per_second: float
    data_kind: str = "weight"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.port not in ("dram", "flash"):
            raise ValueError("PU memory port must be dram or flash")
        if not self.resource_id.endswith("." + self.port):
            raise ValueError("memory resource must match the selected port")
        for name in ("size_bytes", "bandwidth_bytes_per_second"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.bandwidth_bytes_per_second == 0:
            raise ValueError("bandwidth must be positive")
        nonempty("data_kind", self.data_kind)


@dataclass(frozen=True)
class DramCmd(Command):
    size_bytes: int
    # ``size_bytes`` is the traffic actually charged to the representative
    # die.  These keyword-only fields preserve the logical, pre-sharing size
    # for traces when identical chip-level data is ideally striped over dies.
    logical_size_bytes: Optional[int] = field(default=None, kw_only=True)
    shared_die_factor: int = field(default=1, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        integer("size_bytes", self.size_bytes)
        _validate_shared_size(
            self.size_bytes, self.logical_size_bytes, self.shared_die_factor,
        )


@dataclass(frozen=True)
class DramReadCmd(DramCmd):
    """Read data directly from DRAM for the consuming operator."""


@dataclass(frozen=True)
class DramWriteCmd(DramCmd):
    """Write PU output, such as newly generated KV entries, to DRAM."""


def _validate_shared_size(
    effective_size_bytes: int,
    logical_size_bytes: Optional[int],
    shared_die_factor: int,
) -> None:
    integer("shared_die_factor", shared_die_factor, 1)
    if logical_size_bytes is None:
        if shared_die_factor != 1:
            raise ValueError(
                "shared_die_factor greater than one requires "
                "logical_size_bytes"
            )
        return
    integer("logical_size_bytes", logical_size_bytes)
    if logical_size_bytes < effective_size_bytes:
        raise ValueError(
            "logical_size_bytes must be at least the effective size_bytes"
        )
