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
    # DRAM traffic overlapped with this PU operation.  The PU occupies the
    # longer of compute time and this transfer time.
    memory_bytes: int = field(default=0, kw_only=True)
    # Ideal streaming lower bound. Separate PU memory commands account for
    # port contention, and the output waits for their actual completion.
    memory_duration_ns: int = field(default=0, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("B", "M", "K", "N"):
            integer(name, getattr(self, name), 1)
        integer("memory_bytes", self.memory_bytes)
        integer("memory_duration_ns", self.memory_duration_ns)


@dataclass(frozen=True)
class FusedIndexerScoreCmd(GemmCmd):
    """Full-K indexer QK, head aggregation and per-query local top-k.

    M packs queries_per_request groups, reduced and selected independently.
    Top-k selection arithmetic is not included in the current FLOP model.
    """

    main_gemm_flops_override: int | None = None
    top_k: int = field(default=2048, kw_only=True)
    queries_per_request: int = field(default=1, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        integer("top_k", self.top_k, 1)
        integer("queries_per_request", self.queries_per_request, 1)
        if self.M % self.queries_per_request:
            raise ValueError("M must be divisible by queries_per_request")
        if self.main_gemm_flops_override is not None:
            integer(
                "main_gemm_flops_override",
                self.main_gemm_flops_override,
                1,
            )

    @property
    def output_candidates(self) -> int:
        """Number of local candidates after head reduction and truncation."""
        return self.B * self.queries_per_request * min(self.top_k, self.N)

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
        return self.B * self.N * (2 * self.M - self.queries_per_request)

    @property
    def total_flops(self) -> int:
        return self.qk_flops + self.relu_flops + self.head_reduce_flops


@dataclass(frozen=True)
class FlashAttentionCmd(Command):
    """One fused QK/online-softmax/SV computation on a representative PU.

    Softmax/control arithmetic is intentionally excluded from the current
    cost model; only the two matrix products contribute FLOPs.
    """

    B: int
    M: int
    qk_K: int
    qk_N: int
    sv_K: int
    sv_N: int
    memory_bytes: int = field(default=0, kw_only=True)
    memory_duration_ns: int = field(default=0, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("B", "M", "qk_K", "qk_N", "sv_K", "sv_N"):
            integer(name, getattr(self, name), 1)
        integer("memory_bytes", self.memory_bytes)
        integer("memory_duration_ns", self.memory_duration_ns)

    @property
    def qk_flops(self) -> int:
        return 2 * self.B * self.M * self.qk_K * self.qk_N

    @property
    def sv_flops(self) -> int:
        return 2 * self.B * self.M * self.sv_K * self.sv_N

    @property
    def total_flops(self) -> int:
        return self.qk_flops + self.sv_flops


@dataclass(frozen=True)
class VectorCmd(Command):
    kind: str
    m: int
    n: int
    params: Dict[str, Any] = field(default_factory=dict)
    memory_bytes: int = field(default=0, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        nonempty("kind", self.kind)
        integer("m", self.m, 1)
        integer("n", self.n, 1)
        integer("memory_bytes", self.memory_bytes)
        if not isinstance(self.params, dict):
            raise ValueError("params must be a dictionary")
        object.__setattr__(self, "params", self.params.copy())
