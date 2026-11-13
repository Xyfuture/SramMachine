"""Communication commands with scope fixed by their resource-specific class."""
from dataclasses import dataclass, field
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
    # Present for PartitionChain-derived IO: each row runs in parallel.
    io_bytes_per_row: Tuple[int, ...] = field(default=(), kw_only=True)
    # Attention rows merge S shards before sending complete O or (m, l, u).
    attention_complete_batches_per_row: Tuple[int, ...] = field(default=(), kw_only=True)
    attention_partial_batches_per_row: Tuple[int, ...] = field(default=(), kw_only=True)
    top_k: Optional[int] = field(default=None, kw_only=True)
    topk_candidates_per_row: Tuple[int, ...] = field(default=(), kw_only=True)
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
        rows = tuple(self.io_bytes_per_row)
        if rows:
            if self.operation not in ("FromIOdie", "ToIOdie"):
                raise ValueError("per-row IO volumes require an IO operation")
            if any(type(size) is not int or size < 0 for size in rows):
                raise ValueError("per-row IO volumes must be nonnegative integers")
            if max(rows) != self.size_bytes:
                raise ValueError("IO size_bytes must equal the maximum row volume")
        object.__setattr__(self, "io_bytes_per_row", rows)
        complete = tuple(self.attention_complete_batches_per_row)
        partial = tuple(self.attention_partial_batches_per_row)
        if complete or partial:
            if self.operation != "ToIOdie" or not rows or not (
                len(complete) == len(partial) == len(rows)
            ):
                raise ValueError("Attention output counts must match IO rows")
            if any(type(value) is not int or value < 0 for value in complete + partial):
                raise ValueError("Attention batch counts must be nonnegative integers")
        object.__setattr__(self, "attention_complete_batches_per_row", complete)
        object.__setattr__(self, "attention_partial_batches_per_row", partial)
        candidates = tuple(self.topk_candidates_per_row)
        if self.top_k is not None or candidates:
            if type(self.top_k) is not int or self.top_k < 1:
                raise ValueError("top_k must be a positive integer")
            if self.operation != "ToIOdie" or not rows or len(candidates) != len(rows):
                raise ValueError("top-k candidate counts must match output IO rows")
            if any(type(count) is not int or count < 0 for count in candidates):
                raise ValueError("top-k candidate counts must be nonnegative integers")
        object.__setattr__(self, "topk_candidates_per_row", candidates)
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
