"""Actions for SystolicArrayResource and VectorUnitResource."""
from dataclasses import dataclass, field
from typing import Any, Dict
from .base import Command, integer, nonempty


@dataclass(frozen=True)
class GemmCmd(Command):
    B: int
    M: int
    K: int
    N: int

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("B", "M", "K", "N"):
            integer(name, getattr(self, name), 1)


@dataclass(frozen=True)
class VectorCmd(Command):
    kind: str
    m: int
    n: int
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        nonempty("kind", self.kind)
        integer("m", self.m, 1)
        integer("n", self.n, 1)
        if not isinstance(self.params, dict):
            raise ValueError("params must be a dictionary")
        object.__setattr__(self, "params", self.params.copy())
