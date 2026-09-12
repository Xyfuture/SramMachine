"""Configuration for SplitTree simulated annealing."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


def _positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class SimulatedAnnealingConfig:
    """Immutable controls for direct execution or HVI-guided tree search."""

    enabled: bool = False
    batch_sizes: Tuple[int, ...] = ()
    warmup_rounds: int = 16
    rounds: int = 200
    random_seed: int = 0
    initial_temperature: float = 1.0
    final_temperature: float = 0.01
    layer_count: int = 4
    output_dir: Path = field(default_factory=lambda: Path("best split tree result"))

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        batches = tuple(self.batch_sizes)
        if any(type(batch) is not int or batch <= 0 for batch in batches):
            raise ValueError("batch_sizes must contain positive integers")
        if len(set(batches)) != len(batches):
            raise ValueError("batch_sizes must not contain duplicates")
        if batches != tuple(sorted(batches)):
            raise ValueError("batch_sizes must be strictly increasing")
        if self.enabled and not batches:
            raise ValueError("enabled annealing requires explicit batch_sizes")
        for name in ("warmup_rounds", "rounds", "layer_count"):
            _positive_integer(name, getattr(self, name))
        if type(self.random_seed) is not int:
            raise TypeError("random_seed must be an integer")
        for name in ("initial_temperature", "final_temperature"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a positive number")
            if value <= 0:
                raise ValueError(f"{name} must be a positive number")
        if self.final_temperature > self.initial_temperature:
            raise ValueError("final_temperature must not exceed initial_temperature")
        output_dir = Path(self.output_dir)
        object.__setattr__(self, "batch_sizes", batches)
        object.__setattr__(self, "output_dir", output_dir)


__all__ = ["SimulatedAnnealingConfig"]
