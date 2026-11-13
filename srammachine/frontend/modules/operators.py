"""Model-level operator descriptions, independent of hardware commands.

Validation runs at construction. Dimensions describe this operator instance;
no automatic conversion from PipeTree batch ranges is performed.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional, Tuple, Union
from enum import Enum

from .partition import PartitionChain


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


class BMMInputSource(str, Enum):
    """Location supplying the left operand."""

    IO_DIE = "io_die"
    NONE = "none"


class BMMOutputDestination(str, Enum):
    """Destination of the computed (possibly partial) output."""

    IO_DIE = "io_die"
    NONE = "none"
    PU_VERTICAL = "pu_vertical"


@dataclass
class BMMOp(Operator):
    """Logical B products of [M, K] by [K, N] before PU partitioning.

    Empty partition_chain denotes a single PU. IO routes default to IO die;
    NONE means communication is unnecessary or supplied by explicit nodes.
    Element widths default to two-byte activations and one-byte weights.
    """

    PARTITION_AXES: ClassVar[Tuple[str, ...]] = ("B", "M", "K", "N")

    B: int
    M: int
    K: int
    N: int
    partition_chain: PartitionChain = field(default_factory=PartitionChain, kw_only=True)
    input_source: BMMInputSource = field(default=BMMInputSource.IO_DIE, kw_only=True)
    output_destination: BMMOutputDestination = field(
        default=BMMOutputDestination.IO_DIE, kw_only=True,
    )
    input_dtype_bytes: int = field(default=2, kw_only=True)
    output_dtype_bytes: int = field(default=2, kw_only=True)
    weight_dtype_bytes: int = field(default=1, kw_only=True)
    shared_weight: bool = field(default=False, kw_only=True)
    # Compatibility for callers whose mapping already supplies operand loads.
    auto_weight_load: bool = field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("B", "M", "K", "N"):
            _integer(name, getattr(self, name), 1)
        if not isinstance(self.partition_chain, PartitionChain):
            raise TypeError("partition_chain must be a PartitionChain")
        if any(step.dimension not in self.PARTITION_AXES
               for step in self.partition_chain.steps):
            raise ValueError(f"{type(self).__name__} partitions only support {self.PARTITION_AXES}")
        self.input_source = BMMInputSource(self.input_source)
        self.output_destination = BMMOutputDestination(self.output_destination)
        for name in ("input_dtype_bytes", "output_dtype_bytes", "weight_dtype_bytes"):
            _integer(name, getattr(self, name), 1)
        for name in ("shared_weight", "auto_weight_load"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")

    def communication_volumes(self, shape=None) -> dict:
        """Return exact per-row bytes and the critical vertical send volume."""
        shape = shape or {axis: getattr(self, axis) for axis in ("B", "M", "K", "N")}
        chain = self.partition_chain
        return {
            "input_bytes_per_row": tuple(
                size * self.input_dtype_bytes
                for size in chain.io_elements_per_row(shape, output=False)
            ) if self.input_source == BMMInputSource.IO_DIE else (),
            "output_bytes_per_row": tuple(
                size * self.output_dtype_bytes
                for size in chain.io_elements_per_row(shape, output=True)
            ) if self.output_destination == BMMOutputDestination.IO_DIE else (),
            "vertical_reduce_bytes": (
                chain.vertical_reduce_elements(shape) * self.output_dtype_bytes
                if self.output_destination == BMMOutputDestination.PU_VERTICAL else 0
            ),
        }


@dataclass
class FusedIndexerScoreOp(BMMOp):
    """Full-head QK/ReLU/head reduction and per-batch top-k selection.

    Only B and S (= N) can be partitioned. Local and row-merged candidates
    are capped at top_k independently for each batch, not across batches.
    IO candidates carry scores and global sequence indices for further merging.
    """

    PARTITION_AXES: ClassVar[Tuple[str, ...]] = ("B", "S")

    main_gemm_flops_override: Optional[int] = None
    top_k: int = field(default=2048, kw_only=True)
    index_dtype_bytes: int = field(default=4, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        _integer("top_k", self.top_k, 1)
        _integer("index_dtype_bytes", self.index_dtype_bytes, 1)
        if self.N > 1 << (8 * self.index_dtype_bytes):
            raise ValueError("index_dtype_bytes cannot represent global sequence indices")
        if self.output_destination == BMMOutputDestination.PU_VERTICAL:
            raise ValueError("FusedIndexerScore output_destination must be io_die or none")
        if self.main_gemm_flops_override is not None:
            if self.partition_chain.steps:
                raise ValueError("main_gemm_flops_override requires an unpartitioned indexer")
            _integer(
                "main_gemm_flops_override",
                self.main_gemm_flops_override,
                1,
            )

    def communication_volumes(self, shape: Optional[Dict[str, int]] = None) -> dict:
        """Return Q input bytes and row-merged top-k (score, index) bytes.

        shape uses the existing B/M/K/N fields after microbatch scaling.
        M heads are reduced before top-k, so output has no M multiplier.
        """
        shape = shape if shape is not None else {
            axis: getattr(self, axis) for axis in ("B", "M", "K", "N")
        }
        for axis in ("B", "M", "K", "N"):
            _integer(axis, shape[axis], 1)
        coverage = self.partition_chain.sequence_coverage_per_row(
            {"B": shape["B"], "S": shape["N"]},
        )
        candidates = tuple(sum((stop - start) * min(self.top_k, count)
                               for start, stop, count in row) for row in coverage)
        return {
            "input_bytes_per_row": tuple(
                sum(stop - start for start, stop, _ in row)
                * shape["M"] * shape["K"] * self.input_dtype_bytes
                for row in coverage
            ) if self.input_source == BMMInputSource.IO_DIE else (),
            "output_bytes_per_row": tuple(
                count * (self.output_dtype_bytes + self.index_dtype_bytes)
                for count in candidates
            ) if self.output_destination == BMMOutputDestination.IO_DIE else (),
            "topk_candidates_per_row": candidates,
            "vertical_reduce_bytes": 0,
        }

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
    """Full fused Attention, partitionable along batch B and KV sequence S.

    Q is the NoC input; KV storage traffic remains explicitly mapped.
    Row-local S shards merge before IO. Rows with incomplete S coverage send
    unnormalized (m, l, u) states for final merging on the IO die, not sums
    of normalized outputs. NONE leaves communication to the caller.
    """

    B: int
    M: int
    qk_K: int
    qk_N: int
    sv_K: int
    sv_N: int
    partition_chain: PartitionChain = field(default_factory=PartitionChain, kw_only=True)
    input_source: str = field(default="io_die", kw_only=True)
    output_destination: str = field(default="io_die", kw_only=True)
    input_dtype_bytes: int = field(default=2, kw_only=True)
    output_dtype_bytes: int = field(default=2, kw_only=True)
    state_dtype_bytes: int = field(default=4, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("B", "M", "qk_K", "qk_N", "sv_K", "sv_N"):
            _integer(name, getattr(self, name), 1)
        if self.qk_N != self.sv_K:
            raise ValueError("qk_N and sv_K must describe the same sequence length")
        if not isinstance(self.partition_chain, PartitionChain):
            raise TypeError("partition_chain must be a PartitionChain")
        if any(step.dimension not in ("B", "S") for step in self.partition_chain.steps):
            raise ValueError("FlashAttention partitions only support B/S")
        if self.input_source not in ("io_die", "none"):
            raise ValueError("FlashAttention input_source must be io_die or none")
        if self.output_destination not in ("io_die", "none"):
            raise ValueError("FlashAttention output_destination must be io_die or none")
        for name in ("input_dtype_bytes", "output_dtype_bytes", "state_dtype_bytes"):
            _integer(name, getattr(self, name), 1)

    def communication_volumes(self, shape=None) -> dict:
        """Return IO bytes after broadcast deduplication and row-local merging.

        shape optionally supplies the microbatch-scaled B/S/M dimensions.
        State vectors u and scalars m/l use state_dtype_bytes. Merge and
        normalization arithmetic are not charged by this communication model.
        """
        shape = shape if shape is not None else dict(B=self.B, S=self.qk_N, M=self.M)
        batches = self.partition_chain.attention_batches_per_row(
            {axis: shape[axis] for axis in ("B", "S")},
        )
        m = shape["M"]
        _integer("M", m, 1)
        return {
            "input_bytes_per_row": tuple(
                (complete + partial) * m * self.qk_K * self.input_dtype_bytes
                for complete, partial in batches
            ) if self.input_source == "io_die" else (),
            "output_bytes_per_row": tuple(
                m * (complete * self.sv_N * self.output_dtype_bytes
                     + partial * (self.sv_N + 2) * self.state_dtype_bytes)
                for complete, partial in batches
            ) if self.output_destination == "io_die" else (),
            "complete_batches_per_row": tuple(complete for complete, _ in batches),
            "partial_batches_per_row": tuple(partial for _, partial in batches),
            "vertical_reduce_bytes": 0,
        }


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
