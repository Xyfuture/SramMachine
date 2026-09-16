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


def _deduplicate(mutations) -> Tuple[TreeMutation, ...]:
    unique = {}
    for name, candidate in mutations:
        key = repr(candidate)
        unique.setdefault(key, (name, candidate))
    return tuple(unique.values())


def _legacy_tree_mutations(tree: PipeTree) -> Tuple[TreeMutation, ...]:
    """Enumerate the original neighborhood for deterministic regression tests."""
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

        # The input/core/output children must remain an indivisible block, but
        # the block itself is a legal microbatch boundary.  Stopping here only
        # forbids regrouping or ungrouping its three commands; split mutations
        # above apply the same token partition to all of them.
        if is_atomic_compute_group(node):
            continue

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

    return _deduplicate(mutations)


def legal_tree_mutations(tree: PipeTree) -> Tuple[TreeMutation, ...]:
    """Enumerate the extended one-step neighborhood in stable order.

    The complete legacy neighborhood is kept first.  Wider wraps, unary
    microbatch layers, non-adjacent split choices, and boundary shifts are
    appended so the original candidates retain their relative order.
    """
    if not isinstance(tree, PipeTree):
        raise TypeError("tree must be a PipeTree")
    mutations = list(_legacy_tree_mutations(tree))

    def add_if_valid(name: str, root) -> None:
        try:
            candidate = PipeTree(tree.batch_size, tree.operator_order, root)
        except ValueError:
            return
        mutations.append((name, candidate))

    groups = tuple(_group_nodes(tree.root, tree.batch_size))
    for path, node, incoming_batch in groups:
        divisors = _divisors(incoming_batch)
        position = divisors.index(node.split)
        adjacent_targets = set()
        if position > 0:
            adjacent_targets.add(divisors[position - 1])
        if position + 1 < len(divisors):
            adjacent_targets.add(divisors[position + 1])
        for target in divisors:
            if target == node.split or target in adjacent_targets:
                continue
            updated = replace(node, split=target)
            add_if_valid(
                "increase_split" if target > node.split else "decrease_split",
                _replace_node(tree.root, path, updated),
            )

        # Structural mutations may wrap an atomic block as one child, but
        # never regroup the input/compute/output commands inside that block.
        if is_atomic_compute_group(node):
            continue

        child_count = len(node.children)
        for width in range(3, child_count):
            for start in range(child_count - width + 1):
                stop = start + width
                grouped = GroupNode(node.children[start:stop], split=1)
                children = (
                    node.children[:start] + (grouped,) + node.children[stop:]
                )
                add_if_valid(
                    "group_siblings",
                    _replace_parent_children(tree.root, path, children),
                )

        child_batch = incoming_batch // node.split
        for index, child in enumerate(node.children):
            for split in _divisors(child_batch):
                if split == 1:
                    continue
                wrapped = GroupNode((child,), split=split)
                children = (
                    node.children[:index] + (wrapped,)
                    + node.children[index + 1:]
                )
                add_if_valid(
                    "wrap_single",
                    _replace_parent_children(tree.root, path, children),
                )

        for index in range(child_count - 1):
            left = node.children[index]
            right = node.children[index + 1]
            if not isinstance(left, GroupNode) or not isinstance(right, GroupNode):
                continue
            if is_atomic_compute_group(left) or is_atomic_compute_group(right):
                continue

            if len(left.children) >= 2:
                shifted = (
                    node.children[:index]
                    + (
                        replace(left, children=left.children[:-1]),
                        replace(
                            right,
                            children=(left.children[-1],) + right.children,
                        ),
                    )
                    + node.children[index + 2:]
                )
                add_if_valid(
                    "boundary_shift",
                    _replace_parent_children(tree.root, path, shifted),
                )
            if len(right.children) >= 2:
                shifted = (
                    node.children[:index]
                    + (
                        replace(
                            left,
                            children=left.children + (right.children[0],),
                        ),
                        replace(right, children=right.children[1:]),
                    )
                    + node.children[index + 2:]
                )
                add_if_valid(
                    "boundary_shift",
                    _replace_parent_children(tree.root, path, shifted),
                )

    return _deduplicate(mutations)


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
