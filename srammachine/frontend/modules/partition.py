"""Ordered nested BMM or Attention partitions and valid tensor projections."""

from dataclasses import dataclass
from math import prod
from typing import Mapping


AXES = ("B", "M", "K", "N")
ATTENTION_AXES = ("B", "S")


@dataclass(frozen=True)
class SplitStep:
    """Split one logical dimension onto one physical grid dimension."""

    dimension: str
    pu_dimension: str
    factor: int

    def __post_init__(self) -> None:
        if self.dimension not in (*AXES, "S") or self.pu_dimension not in ("p", "q"):
            raise ValueError("split dimensions must be B/M/K/N/S and p/q")
        if type(self.factor) is not int or self.factor < 1:
            raise ValueError("split factor must be a positive integer")


def _union_volume(boxes: tuple) -> int:
    """Measure a union of integer boxes without enumerating tensor elements."""
    boxes = tuple(set(boxes))
    if not boxes:
        return 0
    if not boxes[0]:
        return 1
    endpoints = sorted({point for box in boxes for point in box[0]})
    return sum(
        (stop - start) * _union_volume(tuple(
            box[1:] for box in boxes if box[0][0] <= start < box[0][1]
        ))
        for start, stop in zip(endpoints, endpoints[1:])
    )


@dataclass(frozen=True)
class PartitionChain:
    """Nested splits; the empty chain denotes a single PU.

    Grid coordinates use mixed-radix digits in step order. Each split uses
    the parent's nominal ceiling-sized tile, then clips its valid range.
    """

    steps: tuple[SplitStep, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(
            step if isinstance(step, SplitStep) else SplitStep(*step)
            for step in self.steps
        ))

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Return P/Q as the respective products of split factors."""
        return tuple(prod(s.factor for s in self.steps if s.pu_dimension == axis)
                     for axis in ("p", "q"))

    def validate_grid(self, P: int, Q: int) -> None:
        """Require the chain to cover exactly the supplied PU grid."""
        if self.grid_shape != (P, Q):
            raise ValueError(f"partition grid {self.grid_shape} does not match {(P, Q)}")

    def tile_ranges(self, shape: Mapping[str, int], p: int, q: int) -> dict:
        """Return each PU's clipped half-open logical index ranges."""
        if set(shape) not in (set(AXES), set(ATTENTION_AXES)) or any(
            type(size) is not int or size < 1 for size in shape.values()
        ):
            raise ValueError("shape must contain positive integer B/M/K/N or B/S")
        if any(step.dimension not in shape for step in self.steps):
            raise ValueError("partition dimension is not present in the logical shape")
        for coord, extent in zip((p, q), self.grid_shape):
            if type(coord) is not int or not 0 <= coord < extent:
                raise ValueError("PU coordinate outside partition grid")
        coordinates = {"p": p, "q": q}
        digits = []
        for step in reversed(self.steps):
            coordinates[step.pu_dimension], digit = divmod(
                coordinates[step.pu_dimension], step.factor,
            )
            digits.append(digit)
        starts = dict.fromkeys(shape, 0)
        sizes = dict(shape)
        ends = dict(shape)
        for step, digit in zip(self.steps, reversed(digits)):
            axis = step.dimension
            sizes[axis] = (sizes[axis] + step.factor - 1) // step.factor
            starts[axis] += digit * sizes[axis]
            ends[axis] = min(ends[axis], starts[axis] + sizes[axis])
        return {axis: (min(starts[axis], ends[axis]), ends[axis]) for axis in shape}

    def sequence_coverage_per_row(self, shape: Mapping[str, int]) -> tuple:
        """Return rows of (batch start, batch stop, covered sequence count).

        Sweep valid B interval boundaries and union S ranges, without
        enumerating tensor elements. Empty batch intervals are omitted.
        """
        if set(shape) != set(ATTENTION_AXES):
            raise ValueError("Attention shape must contain B/S")
        rows, columns = self.grid_shape
        result = []
        for p in range(rows):
            tiles = [self.tile_ranges(shape, p, q) for q in range(columns)]
            tiles = [tile for tile in tiles
                     if all(stop > start for start, stop in tile.values())]
            endpoints = sorted({point for tile in tiles for point in tile["B"]})
            intervals = []
            for start, stop in zip(endpoints, endpoints[1:]):
                sequences = tuple((tile["S"],) for tile in tiles
                                  if tile["B"][0] <= start < tile["B"][1])
                coverage = _union_volume(sequences)
                if coverage:
                    intervals.append((start, stop, coverage))
            result.append(tuple(intervals))
        return tuple(result)

    def attention_batches_per_row(self, shape: Mapping[str, int]) -> tuple:
        """Return complete/partial batch counts after row-local S merging."""
        return tuple((
            sum(stop - start for start, stop, count in row if count == shape["S"]),
            sum(stop - start for start, stop, count in row if count < shape["S"]),
        ) for row in self.sequence_coverage_per_row(shape))

    def compute_shape(self, shape: Mapping[str, int]) -> dict:
        """Largest active PU tile (the all-zero coordinate), without padding."""
        return {axis: stop - start for axis, (start, stop)
                in self.tile_ranges(shape, 0, 0).items()}

    def io_elements_per_row(self, shape: Mapping[str, int], *, output: bool) -> tuple[int, ...]:
        """Union input B/M/K or output B/M/N projections within each row."""
        axes = ("B", "M", "N" if output else "K")
        rows, columns = self.grid_shape
        volumes = []
        for p in range(rows):
            tiles = [self.tile_ranges(shape, p, q) for q in range(columns)]
            boxes = tuple(tuple(tile[axis] for axis in axes) for tile in tiles
                          if all(stop > start for start, stop in tile.values()))
            volumes.append(_union_volume(boxes))
        return tuple(volumes)

    def vertical_reduce_elements(self, shape: Mapping[str, int]) -> int:
        """Maximum adjacent vertical link load for reduce to the top member.

        Only equal B/M/N projections in a column reduce together. Partial
        sums are combined in-network; disjoint groups sharing a link add.
        Intermediate nonmembers forward data, so interleaved groups count.
        No horizontal gather or K reduction is implied.
        """
        rows, columns = self.grid_shape
        maximum = 0
        for q in range(columns):
            groups = {}
            for p in range(rows):
                tile = self.tile_ranges(shape, p, q)
                if all(stop > start for start, stop in tile.values()):
                    key = tuple(tile[axis] for axis in ("B", "M", "N"))
                    groups.setdefault(key, []).append(p)
            loads = [0] * max(0, rows - 1)
            for box, members in groups.items():
                volume = prod(stop - start for start, stop in box)
                for link in range(min(members), max(members)):
                    loads[link] += volume
            maximum = max(maximum, max(loads, default=0))
        return maximum
