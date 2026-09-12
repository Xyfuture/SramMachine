"""Typed configuration for one inference workload."""

from dataclasses import dataclass
from enum import Enum


class MoEParallelStrategy(Enum):
    """Supported chip-level MoE mappings."""

    TP = "tp"
    EP = "ep"


def _positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class InferenceConfig:
    """Model-independent shape and MoE policy for a decode workload.

    The current simulator models decoding only.  ``input_sequence_length`` is
    therefore the context length used by the representative decode step.
    Query length and derived history length are not stored.  KV-cache dtype is
    workload-selectable because it directly changes both DRAM and SRAM traffic;
    activation and weight dtypes remain properties of the hardware path.
    ``mtp_enabled`` force-applies the simulator's ideal MTP1 model: one extra
    prediction token is evaluated and always accepted.  This what-if switch is
    intentionally independent of native MTP metadata in a model card.
    """

    global_batch_size: int
    input_sequence_length: int
    output_sequence_length: int
    moe_parallel_strategy: MoEParallelStrategy = MoEParallelStrategy.TP
    kv_cache_dtype: str = "fp8"
    mtp_enabled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "global_batch_size",
            "input_sequence_length",
            "output_sequence_length",
        ):
            _positive_integer(name, getattr(self, name))
        if not isinstance(self.moe_parallel_strategy, MoEParallelStrategy):
            raise TypeError(
                "moe_parallel_strategy must be a MoEParallelStrategy"
            )
        if self.kv_cache_dtype not in ("fp8", "fp16"):
            raise ValueError("kv_cache_dtype must be fp8 or fp16")
        if type(self.mtp_enabled) is not bool:
            raise TypeError("mtp_enabled must be a boolean")

    @property
    def kv_cache_bytes_per_element(self) -> int:
        """Physical bytes transferred for one KV-cache element."""
        return 1 if self.kv_cache_dtype == "fp8" else 2

    @property
    def accepted_tokens_per_step(self) -> int:
        """Tokens credited to throughput; MTP1 assumes token two is accepted."""
        return 2 if self.mtp_enabled else 1


__all__ = ["InferenceConfig", "MoEParallelStrategy"]
