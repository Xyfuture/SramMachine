"""MiniMax M3 grouped sparse attention and block indexer operators.

Vector arithmetic (normalization, RoPE, max-pooling and top-k) is omitted
from timing. Block selection still determines communication and KV traffic.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Mapping

from .operators import BMMOp, BMMOutputDestination, FlashAttentionOp, _integer
from .partition import _union_volume


@dataclass
class FlashMSAOp(FlashAttentionOp):
    """GQA attention with H independent KV heads and M query rows per group.

    M = query tokens per request * query heads per KV head. H splits only
    whole GQA groups, preserving cache sharing inside each group. S is the
    selected-token budget (the full context for a dense GQA layer).
    K and V are separate caches, unlike absorbed MLA.
    """

    PARTITION_AXES: ClassVar[tuple[str, ...]] = ("B", "H", "S")
    H: int = field(default=1, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        _integer("H", self.H, 1)

    def communication_volumes(self, shape: Mapping[str, int] | None = None) -> dict:
        """Count Q/output once per request/group, merging only S shards."""
        shape = shape if shape is not None else dict(B=self.B, H=self.H, S=self.qk_N, M=self.M)
        groups = self.partition_chain.attention_groups_per_row(
            {axis: shape[axis] for axis in ("B", "H", "S")},
        )
        m = shape["M"]
        _integer("M", m, 1)
        return {
            "input_bytes_per_row": tuple(
                (complete + partial) * m * self.qk_K * self.input_dtype_bytes
                for complete, partial in groups
            ) if self.input_source == "io_die" else (),
            "output_bytes_per_row": tuple(
                m * (complete * self.sv_N * self.output_dtype_bytes
                     + partial * (self.sv_N + 2) * self.state_dtype_bytes)
                for complete, partial in groups
            ) if self.output_destination == "io_die" else (),
            # These counts refer to request/group pairs for MSA.
            "complete_batches_per_row": tuple(c for c, _ in groups),
            "partial_batches_per_row": tuple(p for _, p in groups),
            "vertical_reduce_bytes": 0,
        }


@dataclass
class MiniMaxFusedIndexerScoreOp(BMMOp):
    """Shared-key QK with independent block top-k for every query/group.

    B counts requests, M packs query tokens times index heads, K is the
    index width and N the full token history. Only B/S may be partitioned.
    There is no DSA ReLU, weighting projection, or cross-head reduction.
    """

    PARTITION_AXES: ClassVar[tuple[str, ...]] = ("B", "S")
    top_k: int = field(default=16, kw_only=True)
    block_size: int = field(default=128, kw_only=True)
    queries_per_request: int = field(default=1, kw_only=True)
    index_dtype_bytes: int = field(default=4, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("top_k", "block_size", "queries_per_request", "index_dtype_bytes"):
            _integer(name, getattr(self, name), 1)
        if self.M % self.queries_per_request:
            raise ValueError("M must be divisible by queries_per_request")
        if (self.N + self.block_size - 1) // self.block_size > 1 << (8 * self.index_dtype_bytes):
            raise ValueError("index_dtype_bytes cannot represent global block indices")
        if self.output_destination == BMMOutputDestination.PU_VERTICAL:
            raise ValueError("MiniMax indexer output_destination must be io_die or none")

    @property
    def qk_flops(self) -> int:
        """Only the matrix product contributes compute cost."""
        return 2 * self.B * self.M * self.K * self.N

    @property
    def total_flops(self) -> int:
        """Ignore vector work under the current simulator approximation."""
        return self.qk_flops

    def communication_volumes(self, shape: Mapping[str, int] | None = None) -> dict:
        """Merge block IDs, including blocks crossing PU shard boundaries.

        Each group keeps its own candidates. Scores accompany global block
        indices so row/IO merges can combine duplicate blocks by maximum.
        The mandatory local block occupies one of the top-k slots.
        """
        shape = shape if shape is not None else {
            axis: getattr(self, axis) for axis in ("B", "M", "K", "N")
        }
        for axis in ("B", "M", "K", "N"):
            _integer(axis, shape[axis], 1)
        chain = self.partition_chain
        inputs, candidates = [], []
        for p in range(chain.grid_shape[0]):
            tiles = [chain.tile_ranges({"B": shape["B"], "S": shape["N"]}, p, q)
                     for q in range(chain.grid_shape[1])]
            tiles = [t for t in tiles if all(b > a for a, b in t.values())]
            endpoints = sorted({v for t in tiles for v in t["B"]})
            batches = count = 0
            for start, stop in zip(endpoints, endpoints[1:]):
                blocks = tuple(((t["S"][0] // self.block_size,
                                 (t["S"][1] + self.block_size - 1) // self.block_size),)
                               for t in tiles if t["B"][0] <= start < t["B"][1])
                covered = _union_volume(blocks)
                if covered:
                    batches += stop - start
                    count += (stop - start) * shape["M"] * min(self.top_k, covered)
            inputs.append(batches * shape["M"] * shape["K"] * self.input_dtype_bytes)
            candidates.append(count)
        return {
            "input_bytes_per_row": tuple(inputs) if self.input_source == "io_die" else (),
            "output_bytes_per_row": tuple(
                n * (self.output_dtype_bytes + self.index_dtype_bytes) for n in candidates
            ) if self.output_destination == "io_die" else (),
            "topk_candidates_per_row": tuple(candidates),
            "vertical_reduce_bytes": 0,
        }
