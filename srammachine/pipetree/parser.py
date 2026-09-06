"""Lower PipeTree into operator instances, independently of hardware commands.

The resulting order is a scheduling input, not a chain of completion barriers.
Tensor dependencies, command generation and resource scheduling belong to later
lowering stages and cannot be inferred from a PipeTree alone.
"""

from dataclasses import dataclass
from typing import Iterator, Tuple

from .tree import GroupNode, Node, OpInstance, OpNode, PipeTree


@dataclass(frozen=True)
class OperatorPlan:
    """An immutable, materialized expansion of one PipeTree.

    Token ranges identify logical requests in the original batch. They are not
    physical tensor addresses or counts after MTP expansion / expert routing.
    """

    batch_size: int
    operator_order: Tuple[str, ...]
    instances: Tuple[OpInstance, ...]

    def instances_for(self, op_id: str) -> Tuple[OpInstance, ...]:
        """Return one operator's slices in expansion order; reject unknown IDs."""
        if op_id not in self.operator_order:
            raise KeyError(op_id)
        return tuple(item for item in self.instances if item.op_id == op_id)


class TreeParser:
    """Stateless parser; each invocation starts numbering instances at zero."""

    def parse(self, tree: PipeTree) -> OperatorPlan:
        return OperatorPlan(
            batch_size=tree.batch_size,
            operator_order=tree.operator_order,
            instances=tuple(self.iter_instances(tree)),
        )

    def iter_instances(self, tree: PipeTree) -> Iterator[OpInstance]:
        """Stream instances without materializing an entire large execution plan."""
        tree.validate()
        index = 0

        def walk(
            node: Node, start: int, stop: int, path: Tuple[int, ...]
        ) -> Iterator[OpInstance]:
            nonlocal index
            if isinstance(node, OpNode):
                yield OpInstance(index, node.op_id, start, stop, path)
                index += 1
                return
            if not isinstance(node, GroupNode):
                raise TypeError("expected OpNode or GroupNode")
            width = (stop - start) // node.split
            # Child-major order is intentional: visit every slice for this
            # child before moving to its next sibling.
            for child_index, child in enumerate(node.children):
                for part in range(node.split):
                    part_start = start + part * width
                    yield from walk(
                        child, part_start, part_start + width, path + (child_index,)
                    )

        yield from walk(tree.root, 0, tree.batch_size, ())
