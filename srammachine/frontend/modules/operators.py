"""Model-level operator descriptions, independent of hardware commands.

Validation runs at construction. Dimensions describe this operator instance;
no automatic conversion from PipeTree batch ranges is performed.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union


ParticipantId = Union[str, int]


def _integer(name: str, value: int, minimum: int) -> None:
    # bool is an int subclass but is not a meaningful dimension or byte count.
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _name(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass
class Operator:
    """An operator occurrence identified by the same ID used in OpNode."""

    op_id: str

    def __post_init__(self) -> None:
        _name("op_id", self.op_id)


@dataclass
class BMMOp(Operator):
    """B matrix products of [M, K] by [K, N], allowing shared operands."""

    B: int
    M: int
    K: int
    N: int

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("B", "M", "K", "N"):
            _integer(name, getattr(self, name), 1)


@dataclass
class VectorOp(Operator):
    """m vectors of dimension n; kind and params describe the operation.

    kind is extensible (e.g. softmax, rmsnorm, rope, topk). Kind-specific
    parameter interpretation is deferred to the future command generator.
    """

    kind: str
    m: int
    n: int
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        _name("kind", self.kind)
        _integer("m", self.m, 1)
        _integer("n", self.n, 1)
        if not isinstance(self.params, dict):
            raise ValueError("params must be a dictionary")
        self.params = self.params.copy()


@dataclass
class CommOp(Operator):
    """Communication demand, without bandwidth, topology or timing.

    group contains PU IDs for intra_chip and chip IDs for inter_chip.
    size_bytes is each participant's input size, not aggregate link traffic.
    For p2p, group is (sender, receiver) and size_bytes is the send size.
    For broadcast, size_bytes describes the root's payload. For reduce,
    size_bytes describes each participant's partial result and root receives
    the reduced result.
    For nonuniform alltoall, transfer_bytes[i][j] is the byte count sent from
    group[i] to group[j], replacing size_bytes. Diagonal entries are allowed
    and describe locally retained data rather than network traffic.
    reduce_kind applies to reduce, allreduce and reduce_scatter.
    parallel_link_count is the number of physical NoC links striping one
    logical participant-to-participant transfer.  It defaults to one for a
    PU row/column path; current die-level collectives use four boundary links.
    """

    kind: str
    scope: str
    group: Tuple[ParticipantId, ...]
    size_bytes: Optional[int] = None
    root: Optional[ParticipantId] = None
    reduce_kind: str = "sum"
    transfer_bytes: Optional[Tuple[Tuple[int, ...], ...]] = None
    parallel_link_count: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind not in (
            "p2p", "broadcast", "reduce", "allreduce", "allgather",
            "reduce_scatter", "alltoall",
        ):
            raise ValueError("unsupported communication kind")
        if self.scope not in ("intra_chip", "inter_chip"):
            raise ValueError("scope must be intra_chip or inter_chip")
        _integer("parallel_link_count", self.parallel_link_count, 1)
        if self.scope == "inter_chip" and self.parallel_link_count != 1:
            raise ValueError(
                "parallel_link_count is only configurable for intra_chip NoC"
            )
        if not isinstance(self.group, (tuple, list)) or not self.group:
            raise ValueError("group must be a nonempty sequence of participant IDs")
        self.group = tuple(self.group)
        for participant in self.group:
            if type(participant) is int:
                _integer("participant ID", participant, 0)
            else:
                _name("participant ID", participant)
        if len(set(self.group)) != len(self.group):
            raise ValueError("group must not contain duplicate participants")
        if self.kind == "p2p" and len(self.group) != 2:
            raise ValueError("p2p requires exactly (sender, receiver)")
        if self.kind in ("broadcast", "reduce"):
            if self.root is None or type(self.root) not in (str, int) or self.root not in self.group:
                raise ValueError(f"{self.kind} root must be a member of group")
        elif self.root is not None:
            raise ValueError("root is only used by broadcast or reduce")
        if self.kind == "reduce" and self.scope != "intra_chip":
            raise ValueError("reduce is only supported for intra_chip NoC")
        if self.reduce_kind not in ("sum", "max", "min"):
            raise ValueError("reduce_kind must be sum, max or min")
        if (self.size_bytes is None) == (self.transfer_bytes is None):
            raise ValueError("provide exactly one of size_bytes and transfer_bytes")
        if self.size_bytes is not None:
            _integer("size_bytes", self.size_bytes, 0)
        else:
            if self.kind != "alltoall":
                raise ValueError("transfer_bytes is only supported for alltoall")
            matrix = self.transfer_bytes
            count = len(self.group)
            if not isinstance(matrix, (tuple, list)) or len(matrix) != count:
                raise ValueError("transfer_bytes must have one row per participant")
            for row in matrix:
                if not isinstance(row, (tuple, list)) or len(row) != count:
                    raise ValueError("transfer_bytes must be a square matrix matching group")
                for size in row:
                    _integer("transfer_bytes entry", size, 0)
            self.transfer_bytes = tuple(tuple(row) for row in matrix)
