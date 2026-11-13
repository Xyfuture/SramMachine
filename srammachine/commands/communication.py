"""Communication commands with scope fixed by their resource-specific class."""
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple
from srammachine.frontend.modules.operators import FabricOp, NoCOp, ParticipantId
from .base import Command


@dataclass(frozen=True)
class CommCmd(Command):
    kind: str
    group: Tuple[ParticipantId, ...]
    size_bytes: Optional[int] = None
    root: Optional[ParticipantId] = None
    reduce_kind: str = "sum"
    transfer_bytes: Optional[Tuple[Tuple[int, ...], ...]] = None
    scope: ClassVar[str] = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.scope:
            raise TypeError("instantiate InterChipCmd")
        # Reuse the model's collective semantics and validation, not its identity.
        checked = FabricOp(
            self.op_id, self.kind, self.group, self.size_bytes,
            self.root, self.reduce_kind, self.transfer_bytes,
        )
        object.__setattr__(self, "group", checked.group)
        object.__setattr__(self, "transfer_bytes", checked.transfer_bytes)


@dataclass(frozen=True)
class NoCCmd(Command):
    """A NoC operation atomically reserves its required chip directions."""

    operation: str
    size_bytes: int
    kind: Optional[str] = None
    group: Tuple[ParticipantId, ...] = ()
    additional_resource_ids: Tuple[str, ...] = ()
    scope: ClassVar[str] = "intra_chip"

    def __post_init__(self) -> None:
        super().__post_init__()
        checked = NoCOp(self.op_id, self.operation, self.size_bytes,
                        self.kind, self.group)
        chip_id, separator, _ = self.resource_id.rpartition(".")
        if not separator:
            raise ValueError("NoC resource must include a chip ID")
        required = checked.resource_ids(chip_id)
        if isinstance(self.additional_resource_ids, str):
            raise ValueError("additional_resource_ids must be a sequence of IDs")
        additional = tuple(self.additional_resource_ids)
        if self.resource_id not in required:
            raise ValueError("NoC operation is bound to the wrong direction")
        if additional and set((self.resource_id,) + additional) != set(required):
            raise ValueError("NoC resources must match the operation")
        if len(set((self.resource_id,) + additional)) != 1 + len(additional):
            raise ValueError("resource IDs must be unique")
        object.__setattr__(self, "group", checked.group)
        object.__setattr__(self, "additional_resource_ids", tuple(
            item for item in required if item != self.resource_id))

    @property
    def resource_ids(self) -> Tuple[str, ...]:
        """Directions reserved together until communication completes."""
        return (self.resource_id,) + self.additional_resource_ids


@dataclass(frozen=True)
class InterChipCmd(CommCmd):
    scope: ClassVar[str] = "inter_chip"
