"""Communication commands with scope fixed by their resource-specific class."""
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple
from srammachine.frontend.modules.operators import CommOp, ParticipantId
from .base import Command


@dataclass(frozen=True)
class CommCmd(Command):
    kind: str
    group: Tuple[ParticipantId, ...]
    size_bytes: Optional[int] = None
    root: Optional[ParticipantId] = None
    reduce_kind: str = "sum"
    transfer_bytes: Optional[Tuple[Tuple[int, ...], ...]] = None
    parallel_link_count: int = 1
    scope: ClassVar[str] = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.scope:
            raise TypeError("instantiate NoCCmd or InterChipCmd")
        # Reuse the model's collective semantics and validation, not its identity.
        checked = CommOp(
            self.op_id, self.kind, self.scope, self.group, self.size_bytes,
            self.root, self.reduce_kind, self.transfer_bytes,
            self.parallel_link_count,
        )
        object.__setattr__(self, "group", checked.group)
        object.__setattr__(self, "transfer_bytes", checked.transfer_bytes)


@dataclass(frozen=True)
class NoCCmd(CommCmd):
    scope: ClassVar[str] = "intra_chip"


@dataclass(frozen=True)
class InterChipCmd(CommCmd):
    scope: ClassVar[str] = "inter_chip"
