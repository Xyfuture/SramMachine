"""Hierarchical batch partitioning with child-major expansion."""

from dataclasses import dataclass
from typing import Iterator, Tuple, Union


@dataclass(frozen=True)
class OpNode:
    """Reference to one operator occurrence in the model's ordered operator list."""

    op_id: str


@dataclass(frozen=True)
class GroupNode:
    """Split the incoming batch; visit children in order, batches within each child."""

    children: Tuple["Node", ...]
    split: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", tuple(self.children))


Node = Union[OpNode, GroupNode]


@dataclass(frozen=True)
class OpInstance:
    """An expanded operator invocation; token range is half-open [start, stop)."""

    index: int
    op_id: str
    token_start: int
    token_stop: int
    node_path: Tuple[int, ...]

    @property
    def batch_size(self) -> int:
        return self.token_stop - self.token_start


@dataclass(frozen=True)
class PipeTree:
    batch_size: int
    operator_order: Tuple[str, ...]
    root: Node

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator_order", tuple(self.operator_order))
        self.validate()

    def validate(self) -> None:
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not self.operator_order or any(
            not isinstance(op, str) or not op for op in self.operator_order
        ):
            raise ValueError("operator_order must contain nonempty operator IDs")
        if len(set(self.operator_order)) != len(self.operator_order):
            raise ValueError("each operator occurrence must have a unique ID")

        leaves = []

        def visit(node: Node, batch: int) -> None:
            if isinstance(node, OpNode):
                leaves.append(node.op_id)
                return
            if not isinstance(node, GroupNode):
                raise TypeError("expected OpNode or GroupNode")
            if not node.children:
                raise ValueError("a group must have at least one child")
            if type(node.split) is not int or node.split <= 0:
                raise ValueError("split must be a positive integer")
            if batch % node.split:
                raise ValueError("split must divide the incoming batch size")
            for child in node.children:
                visit(child, batch // node.split)

        visit(self.root, self.batch_size)
        if tuple(leaves) != self.operator_order:
            raise ValueError("leaves must match operator_order exactly, in order")

    def expand(self) -> Iterator[OpInstance]:
        """Compatibility entry point; expansion is owned by TreeParser."""
        from .parser import TreeParser

        yield from TreeParser().iter_instances(self)
