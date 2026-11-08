"""Communication commands with scope fixed by their resource-specific class."""
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple
from srammachine.frontend.modules.operators import CommOp, ParticipantId
from .base import Command, nonempty


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
    additional_resource_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.additional_resource_ids, str):
            raise ValueError("additional_resource_ids must be a sequence of IDs")
        additional = tuple(self.additional_resource_ids)
        for resource_id in additional:
            nonempty("additional resource ID", resource_id)
        if len(set((self.resource_id,) + additional)) != 1 + len(additional):
            raise ValueError("resource IDs must be unique")
        object.__setattr__(self, "additional_resource_ids", additional)

    @property
    def resource_ids(self) -> Tuple[str, ...]:
        """NoC directions reserved atomically until communication completes."""
        return (self.resource_id,) + self.additional_resource_ids


@dataclass(frozen=True)
class InterChipCmd(CommCmd):
    scope: ClassVar[str] = "inter_chip"
