"""Exact synthetic EP routing shared by full-batch mapping and microbatches."""

from dataclasses import dataclass

from srammachine.commands.base import integer


@dataclass(frozen=True)
class EPRouting:
    """Balanced cyclic experts with contiguous, balanced attention DP shards.

    Token t selects experts (t * top_k + k) % num_experts. Each assignment
    carries one activation vector (no source-side deduplication or destination-
    side expert aggregation). Combine returns each vector to its token owner;
    local weighted reduction is separate from this network-only demand.
    """

    chip_count: int
    num_experts: int
    top_k: int
    tokens_per_request: int
    bytes_per_assignment: int
    combine: bool = False

    def __post_init__(self) -> None:
        for name in ("chip_count", "num_experts", "top_k",
                     "tokens_per_request", "bytes_per_assignment"):
            integer(name, getattr(self, name), 1)
        if self.num_experts % self.chip_count:
            raise ValueError("expert count must be divisible by chip count")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed expert count")
        if type(self.combine) is not bool:
            raise TypeError("combine must be a boolean")

    def transfer_bytes(self, start: int, stop: int,
                       full_batch: int) -> tuple[tuple[int, ...], ...]:
        """Return exact bytes for a half-open global request interval."""
        integer("full_batch", full_batch, 1)
        integer("start", start)
        integer("stop", stop)
        if not 0 <= start <= stop <= full_batch:
            raise ValueError("routing interval must lie inside the full batch")
        experts_per_chip = self.num_experts // self.chip_count
        requests, remainder = divmod(full_batch, self.chip_count)
        assignments = self.tokens_per_request * self.top_k

        def prefix(count: int, destination: int) -> int:
            cycles, tail = divmod(count, self.num_experts)
            return cycles * experts_per_chip + min(
                experts_per_chip, max(0, tail - destination * experts_per_chip))

        matrix = []
        for source in range(self.chip_count):
            shard_start = source * requests + min(source, remainder)
            shard_stop = shard_start + requests + (source < remainder)
            lo, hi = max(start, shard_start), min(stop, shard_stop)
            matrix.append(tuple(
                (prefix(hi * assignments, destination)
                 - prefix(lo * assignments, destination)) * self.bytes_per_assignment
                if hi > lo else 0
                for destination in range(self.chip_count)
            ))
        if self.combine:
            return tuple(zip(*matrix))
        return tuple(matrix)
