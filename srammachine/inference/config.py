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
    therefore the context length used by the representative decode step;
    query length, derived history length and tensor dtypes are intentionally
    not stored in this configuration.
    """

    global_batch_size: int
    input_sequence_length: int
    output_sequence_length: int
    moe_parallel_strategy: MoEParallelStrategy = MoEParallelStrategy.TP

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


__all__ = ["InferenceConfig", "MoEParallelStrategy"]
