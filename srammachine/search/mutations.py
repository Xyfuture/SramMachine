"""Order-preserving neighborhood operations for SplitTree annealing."""

from dataclasses import replace
from typing import Iterator, Tuple

from srammachine.pipetree import GroupNode, OpNode, PipeTree


TreeMutation = Tuple[str, PipeTree]


def is_atomic_compute_group(node) -> bool:
    """Recognize mapper-created input/compute/output blocks."""
    if not isinstance(node, GroupNode) or len(node.children) != 3:
        return False
    if not all(isinstance(child, OpNode) for child in node.children):
        return False
    first, middle, last = (child.op_id for child in node.children)
    for suffix in (".input_broadcast", ".input_transfer"):
        if first.endswith(suffix):
            base = first.removesuffix(suffix)
            return middle == base and last in (
                f"{base}.output_reduce", f"{base}.output_transfer",
            )
    return False


def _group_nodes(
    node, incoming_batch: int, path: Tuple[int, ...] = (),
) -> Iterator[tuple[Tuple[int, ...], GroupNode, int]]:
    if not isinstance(node, GroupNode):
        return
    yield path, node, incoming_batch
    child_batch = incoming_batch // node.split
    for index, child in enumerate(node.children):
        yield from _group_nodes(child, child_batch, path + (index,))


def _replace_node(node, path: Tuple[int, ...], replacement):
    if not path:
        return replacement
    if not isinstance(node, GroupNode):
        raise ValueError("node path does not identify a GroupNode")
    index = path[0]
    children = list(node.children)
    children[index] = _replace_node(children[index], path[1:], replacement)
    return replace(node, children=tuple(children))


def _replace_parent_children(
    root, parent_path: Tuple[int, ...], children: tuple,
):
    parent = root
    for index in parent_path:
        if not isinstance(parent, GroupNode):
            raise ValueError("parent path does not identify a GroupNode")
        parent = parent.children[index]
    return _replace_node(root, parent_path, replace(parent, children=children))


def _divisors(value: int) -> tuple[int, ...]:
    return tuple(divisor for divisor in range(1, value + 1) if value % divisor == 0)


def legal_tree_mutations(tree: PipeTree) -> Tuple[TreeMutation, ...]:
    """Enumerate all one-step legal mutations in stable order."""
    if not isinstance(tree, PipeTree):
        raise TypeError("tree must be a PipeTree")
    mutations = []

    def add_if_valid(name: str, root) -> None:
        """Discard mutations whose changed parent invalidates a descendant split."""
        try:
            candidate = PipeTree(tree.batch_size, tree.operator_order, root)
        except ValueError:
            return
        mutations.append((name, candidate))
    groups = tuple(_group_nodes(tree.root, tree.batch_size))

    for path, node, incoming_batch in groups:
        if is_atomic_compute_group(node):
            continue
        divisors = _divisors(incoming_batch)
        position = divisors.index(node.split)
        if position + 1 < len(divisors):
            updated = replace(node, split=divisors[position + 1])
            add_if_valid(
                "increase_split", _replace_node(tree.root, path, updated),
            )
        if position > 0:
            updated = replace(node, split=divisors[position - 1])
            add_if_valid(
                "decrease_split", _replace_node(tree.root, path, updated),
            )

        for index in range(len(node.children) - 1):
            grouped = GroupNode(node.children[index:index + 2], split=1)
            children = (
                node.children[:index] + (grouped,) + node.children[index + 2:]
            )
            add_if_valid(
                "group_siblings",
                _replace_parent_children(tree.root, path, children),
            )

        for index, child in enumerate(node.children):
            if (
                isinstance(child, GroupNode)
                and child.split == 1
                and not is_atomic_compute_group(child)
            ):
                children = (
                    node.children[:index]
                    + child.children
                    + node.children[index + 1:]
                )
                add_if_valid(
                    "ungroup",
                    _replace_parent_children(tree.root, path, children),
                )

    unique = {}
    for name, candidate in mutations:
        key = repr(candidate)
        unique.setdefault(key, (name, candidate))
    return tuple(unique.values())


def rebind_split_tree(
    tree: PipeTree, *, batch_size: int, operator_order: tuple[str, ...],
) -> PipeTree:
    """Preserve tree topology while rebinding leaves by stable operator ordinal."""
    if not isinstance(tree, PipeTree):
        raise TypeError("tree must be a PipeTree")
    order = tuple(operator_order)
    if len(order) != len(tree.operator_order):
        raise ValueError("cannot rebind a SplitTree with a different operator count")
    iterator = iter(order)

    def walk(node):
        if isinstance(node, OpNode):
            return OpNode(next(iterator))
        if isinstance(node, GroupNode):
            return GroupNode(tuple(walk(child) for child in node.children), node.split)
        raise TypeError("SplitTree nodes must be OpNode or GroupNode")

    rebound = PipeTree(batch_size, order, walk(tree.root))
    try:
        next(iterator)
    except StopIteration:
        return rebound
    raise ValueError("operator order contains unused entries")


__all__ = [
    "TreeMutation", "is_atomic_compute_group", "legal_tree_mutations",
    "rebind_split_tree",
]
