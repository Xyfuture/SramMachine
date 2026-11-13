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
class FusedIndexerScoreOp(BMMOp):
    """Full-K indexer QK with row-local ReLU and weighted head reduction."""

    main_gemm_flops_override: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.main_gemm_flops_override is not None:
            _integer(
                "main_gemm_flops_override",
                self.main_gemm_flops_override,
                1,
            )

    @property
    def qk_flops(self) -> int:
        if self.main_gemm_flops_override is not None:
            return self.main_gemm_flops_override
        return 2 * self.B * self.M * self.K * self.N

    @property
    def main_gemm_flops(self) -> int:
        return self.qk_flops

    @property
    def relu_flops(self) -> int:
        return self.B * self.M * self.N

    @property
    def head_reduce_flops(self) -> int:
        return self.B * self.N * (2 * self.M - 1)

    @property
    def total_flops(self) -> int:
        return self.qk_flops + self.relu_flops + self.head_reduce_flops


@dataclass
class FlashAttentionOp(Operator):
    """Fused QK/online-softmax/SV with two explicit GEMM shapes."""

    B: int
    M: int
    qk_K: int
    qk_N: int
    sv_K: int
    sv_N: int

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("B", "M", "qk_K", "qk_N", "sv_K", "sv_N"):
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
class FabricOp(Operator):
    """Inter-chip collective demand; size_bytes is each chip's input payload.

    Nonuniform alltoall uses transfer_bytes[i][j] for sender i to receiver j;
    diagonal entries represent locally retained data. Fabric collective
    algorithms and timing remain separate from explicit NoC send volumes.
    """

    kind: str
    scope: str = field(default="inter_chip", init=False)
    group: Tuple[ParticipantId, ...]
    size_bytes: Optional[int] = None
    root: Optional[ParticipantId] = None
    reduce_kind: str = "sum"
    transfer_bytes: Optional[Tuple[Tuple[int, ...], ...]] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind not in (
            "p2p", "broadcast", "reduce", "allreduce", "allgather",
            "reduce_scatter", "alltoall",
        ):
            raise ValueError("unsupported communication kind")
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
        if self.kind == "reduce":
            raise ValueError("reduce is not supported by Fabric")
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


NOC_OPERATION_RESOURCES = {
    "ToIOdie": ("noc_horizontal",),
    "FromIOdie": ("noc_horizontal",),
    "PUVertical": ("noc_vertical",),
    "PUHorizontal": ("noc_horizontal",),
    "PUMesh": ("noc_horizontal", "noc_vertical"),
}


@dataclass
class NoCOp(Operator):
    """Explicit NoC send demand; collective kind is descriptive only.

    size_bytes is the total traffic at IO-connected PUs for IO transfers,
    or the largest sending PU's traffic for PU communication. It is already
    the traffic to charge, not a collective input payload.
    """

    operation: str
    size_bytes: int
    kind: Optional[str] = None
    group: Tuple[ParticipantId, ...] = ()
    scope: str = field(default="intra_chip", init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.operation not in NOC_OPERATION_RESOURCES:
            raise ValueError("unsupported NoC operation")
        _integer("size_bytes", self.size_bytes, 0)
        if self.kind is not None and self.kind not in (
            "p2p", "broadcast", "reduce", "allreduce", "allgather",
            "reduce_scatter", "alltoall",
        ):
            raise ValueError("unsupported communication kind")
        if self.operation.startswith("PU") and self.kind is None:
            raise ValueError("PU communication requires a collective kind")
        self.group = tuple(self.group)
        for participant in self.group:
            if type(participant) is int:
                _integer("participant ID", participant, 0)
            else:
                _name("participant ID", participant)
        if len(set(self.group)) != len(self.group):
            raise ValueError("group must not contain duplicate participants")

    def resource_ids(self, chip_id: str = "chip0") -> Tuple[str, ...]:
        """Return all directional resources required by this operation."""
        return tuple(f"{chip_id}.{name}"
                     for name in NOC_OPERATION_RESOURCES[self.operation])
