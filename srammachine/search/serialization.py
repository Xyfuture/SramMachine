"""Stable JSON representation for immutable SplitTrees."""

import json
from pathlib import Path
from typing import Any, Mapping

from srammachine.pipetree import GroupNode, OpNode, PipeTree


def _node_to_dict(node) -> dict[str, Any]:
    if isinstance(node, OpNode):
        return {"type": "op", "op_id": node.op_id}
    if isinstance(node, GroupNode):
        return {
            "type": "group",
            "split": node.split,
            "children": [_node_to_dict(child) for child in node.children],
        }
    raise TypeError("SplitTree nodes must be OpNode or GroupNode")


def split_tree_to_dict(tree: PipeTree) -> dict[str, Any]:
    if not isinstance(tree, PipeTree):
        raise TypeError("tree must be a PipeTree")
    return {
        "batch_size": tree.batch_size,
        "operator_order": list(tree.operator_order),
        "root": _node_to_dict(tree.root),
    }


def _node_from_dict(value: Mapping[str, Any]):
    if not isinstance(value, Mapping):
        raise TypeError("serialized node must be a mapping")
    kind = value.get("type")
    if kind == "op":
        if set(value) != {"type", "op_id"}:
            raise ValueError("serialized op node contains unknown fields")
        return OpNode(value["op_id"])
    if kind == "group":
        if set(value) != {"type", "split", "children"}:
            raise ValueError("serialized group node contains unknown fields")
        children = value["children"]
        if not isinstance(children, list):
            raise TypeError("serialized group children must be a list")
        return GroupNode(
            tuple(_node_from_dict(child) for child in children),
            split=value["split"],
        )
    raise ValueError(f"unknown serialized node type: {kind!r}")


def split_tree_from_dict(value: Mapping[str, Any]) -> PipeTree:
    if not isinstance(value, Mapping):
        raise TypeError("serialized SplitTree must be a mapping")
    if set(value) != {"batch_size", "operator_order", "root"}:
        raise ValueError("serialized SplitTree contains unknown fields")
    order = value["operator_order"]
    if not isinstance(order, list):
        raise TypeError("operator_order must be a list")
    return PipeTree(
        batch_size=value["batch_size"],
        operator_order=tuple(order),
        root=_node_from_dict(value["root"]),
    )


def load_pareto_split_tree(
    path: str | Path, *, point_index: int = 0, tree_index: int = 0,
) -> PipeTree:
    """Load one replayable tree from an annealing Pareto JSON file."""
    if type(point_index) is not int or point_index < 0:
        raise ValueError("point_index must be a nonnegative integer")
    if type(tree_index) is not int or tree_index < 0:
        raise ValueError("tree_index must be a nonnegative integer")
    with Path(path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    try:
        tree = payload["pareto_front"][point_index]["split_trees"][tree_index]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Pareto file does not contain the requested tree") from error
    return split_tree_from_dict(tree)


__all__ = [
    "split_tree_to_dict", "split_tree_from_dict", "load_pareto_split_tree",
]
