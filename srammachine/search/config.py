"""Configuration for SplitTree simulated annealing."""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Tuple


def _positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class SimulatedAnnealingConfig:
    """Immutable controls for direct execution or per-workload tree search."""

    enabled: bool = False
    batch_sizes: Tuple[int, ...] = ()
    mtp_values: Tuple[bool, ...] = (False, True)
    warmup_rounds: int = 16
    rounds: int = 50
    restart_count: int = 4
    random_seed: int = 20260912
    initial_temperature: float = 1.0
    final_temperature: float = 0.01
    layer_count: int = 4
    output_dir: Path = field(default_factory=lambda: Path("best split tree result"))
    write_result: bool = True

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
        mtp_values = tuple(self.mtp_values)
        if not mtp_values or any(type(value) is not bool for value in mtp_values):
            raise ValueError("mtp_values must contain booleans")
        if len(set(mtp_values)) != len(mtp_values):
            raise ValueError("mtp_values must not contain duplicates")
        for name in ("warmup_rounds", "rounds", "restart_count", "layer_count"):
            _positive_integer(name, getattr(self, name))
        if type(self.random_seed) is not int:
            raise TypeError("random_seed must be an integer")
        if type(self.write_result) is not bool:
            raise TypeError("write_result must be a boolean")
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
        object.__setattr__(self, "mtp_values", mtp_values)
        object.__setattr__(self, "output_dir", output_dir)


def derive_workload_seed(
    base_seed: int,
    model_name: str,
    moe_parallel_strategy: str,
    global_batch_size: int,
    mtp_enabled: bool,
) -> int:
    """Derive a stable per-workload seed independent of execution order.

    Python's built-in ``hash`` is deliberately randomized between processes,
    so a cryptographic digest is used to make parallel Windows ``spawn`` runs
    reproduce the same SplitTree walk as serial runs.
    """
    if type(base_seed) is not int:
        raise TypeError("base_seed must be an integer")
    for name, value in (
        ("model_name", model_name),
        ("moe_parallel_strategy", moe_parallel_strategy),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    _positive_integer("global_batch_size", global_batch_size)
    if type(mtp_enabled) is not bool:
        raise TypeError("mtp_enabled must be a boolean")
    payload = "\x00".join((
        str(base_seed), model_name, moe_parallel_strategy,
        str(global_batch_size), "1" if mtp_enabled else "0",
    )).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def derive_restart_seed(workload_seed: int, restart_index: int) -> int:
    """Derive a stable seed for one annealing restart."""
    if type(workload_seed) is not int:
        raise TypeError("workload_seed must be an integer")
    if type(restart_index) is not int or restart_index < 0:
        raise ValueError("restart_index must be a nonnegative integer")
    payload = f"{workload_seed}\x00restart\x00{restart_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


__all__ = [
    "SimulatedAnnealingConfig", "derive_workload_seed", "derive_restart_seed",
]
