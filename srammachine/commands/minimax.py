"""Compute commands for MiniMax M3; vector costs are deliberately excluded."""

from dataclasses import dataclass, field

from .base import integer
from .compute import FlashAttentionCmd, GemmCmd


@dataclass(frozen=True)
class FlashMSACmd(FlashAttentionCmd):
    """Two attention products for each of H independent GQA groups."""

    H: int = field(default=1, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        integer("H", self.H, 1)

    @property
    def qk_flops(self) -> int:
        return self.H * super().qk_flops

    @property
    def sv_flops(self) -> int:
        return self.H * super().sv_flops


@dataclass(frozen=True)
class MiniMaxFusedIndexerScoreCmd(GemmCmd):
    """Full token-level QK followed by untimed block-max and per-head top-k."""

    top_k: int = field(default=16, kw_only=True)
    block_size: int = field(default=128, kw_only=True)
    queries_per_request: int = field(default=1, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("top_k", "block_size", "queries_per_request"):
            integer(name, getattr(self, name), 1)
        if self.M % self.queries_per_request:
            raise ValueError("M must be divisible by queries_per_request")

    @property
    def output_candidates(self) -> int:
        """Candidates on the representative PU, whose token range starts at 0."""
        blocks = (self.N + self.block_size - 1) // self.block_size
        return self.B * self.M * min(self.top_k, blocks)

    @property
    def qk_flops(self) -> int:
        return 2 * self.B * self.M * self.K * self.N

    @property
    def total_flops(self) -> int:
        return self.qk_flops
