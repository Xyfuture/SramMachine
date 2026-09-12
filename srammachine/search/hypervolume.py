"""Two-dimensional Pareto and hypervolume helpers."""

from dataclasses import dataclass
import math
from typing import Iterable, Tuple


Point2D = Tuple[float, float]


def _point(value: Point2D) -> Point2D:
    if len(value) != 2:
        raise ValueError("a point must contain exactly two coordinates")
    point = (float(value[0]), float(value[1]))
    if any(not math.isfinite(coordinate) for coordinate in point):
        raise ValueError("point coordinates must be finite")
    return point


def dominates(left: Point2D, right: Point2D) -> bool:
    """Return whether ``left`` strictly Pareto-dominates ``right``."""
    x1, y1 = _point(left)
    x2, y2 = _point(right)
    return x1 >= x2 and y1 >= y2 and (x1 > x2 or y1 > y2)


def nondominated_points(points: Iterable[Point2D]) -> Tuple[Point2D, ...]:
    """Return unique raw non-dominated points in deterministic x order."""
    unique = tuple(dict.fromkeys(_point(point) for point in points))
    front = tuple(
        point for point in unique
        if not any(dominates(other, point) for other in unique)
    )
    return tuple(sorted(front, key=lambda point: (point[0], -point[1])))


def hypervolume_2d(points: Iterable[Point2D]) -> float:
    """Area dominated by maximization points relative to reference ``(0, 0)``."""
    front = nondominated_points(
        (max(0.0, x), max(0.0, y)) for x, y in points
    )
    area = 0.0
    previous_x = 0.0
    for x, y in front:
        if x > previous_x:
            area += (x - previous_x) * y
            previous_x = x
    return area


def hypervolume_improvement(
    pareto_points: Iterable[Point2D], candidate: Point2D,
) -> float:
    """Additional two-dimensional hypervolume contributed by ``candidate``."""
    front = tuple(pareto_points)
    improvement = hypervolume_2d((*front, candidate)) - hypervolume_2d(front)
    return max(0.0, improvement)


@dataclass(frozen=True)
class HypervolumeBounds:
    """Frozen affine normalization bounds learned during warmup."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.x_max, self.y_min, self.y_max)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("normalization bounds must be finite")
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("normalization maxima must not be below minima")

    @classmethod
    def from_points(cls, points: Iterable[Point2D]) -> "HypervolumeBounds":
        copied = tuple(_point(point) for point in points)
        if not copied:
            raise ValueError("at least one warmup point is required")
        xs, ys = zip(*copied)
        return cls(min(xs), max(xs), min(ys), max(ys))

    @staticmethod
    def _normalize(value: float, lower: float, upper: float) -> float:
        if upper == lower:
            return 1.0
        return min(1.0, max(0.0, (value - lower) / (upper - lower)))

    def normalize(self, point: Point2D) -> Point2D:
        x, y = _point(point)
        return (
            self._normalize(x, self.x_min, self.x_max),
            self._normalize(y, self.y_min, self.y_max),
        )


__all__ = [
    "HypervolumeBounds", "dominates", "nondominated_points",
    "hypervolume_2d", "hypervolume_improvement",
]
