"""Per-operator selection of the independent PU memory ports."""

from dataclasses import dataclass
from enum import Enum
import math


class WeightSource(str, Enum):
    """Ports supplying an operator's weights (or cached right operand)."""

    DRAM = "dram"
    FLASH = "flash"
    DRAM_FLASH = "dram+flash"


@dataclass(frozen=True)
class WeightMemory:
    """A memory policy; only Flash bandwidth may vary between operators."""

    source: WeightSource = WeightSource.DRAM_FLASH
    flash_bandwidth_bytes_per_second: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", WeightSource(self.source))
        value = self.flash_bandwidth_bytes_per_second
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0
        ):
            raise ValueError("Flash bandwidth must be positive and finite")

    def transfers(self, size_bytes: int, dram_bandwidth: float,
                  default_flash_bandwidth: float) -> tuple[tuple[str, float, float], ...]:
        """Return (port, bytes, bandwidth) with equal-time dual-port loads."""
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError("size_bytes must be a nonnegative integer")
        for rate in (dram_bandwidth, default_flash_bandwidth):
            if (isinstance(rate, bool) or not isinstance(rate, (int, float))
                    or not math.isfinite(rate) or rate <= 0):
                raise ValueError("memory bandwidth must be positive and finite")
        flash = self.flash_bandwidth_bytes_per_second or default_flash_bandwidth
        rates = ({"dram": dram_bandwidth} if self.source == WeightSource.DRAM else
                 {"flash": flash} if self.source == WeightSource.FLASH else
                 {"dram": dram_bandwidth, "flash": flash})
        total = sum(rates.values())
        return tuple((port, size_bytes * rate / total, rate)
                     for port, rate in rates.items())
