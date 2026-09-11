"""这是适用于模拟退火的初始 RootNode。

The ``RootNode`` is the model-derived initial solution shown before simulated
annealing in the simulator design.  It is also a valid ``PipeTree`` so it can
be parsed and simulated directly before a tree searcher is implemented.

The initial solution is deliberately conservative: every split is one, model
order is preserved, and each representative PU broadcast/GEMM/reduce sequence
is kept as one atomic group. Simulated annealing may copy this root node and
mutate its legal grouping and split choices without changing this definition.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from srammachine.frontend.modules import BMMOp, Operator

from .tree import GroupNode, Node, OpNode, PipeTree


@dataclass(frozen=True)
class RootNode(PipeTree):
    """ModelParser output used as the initial simulated-annealing solution."""


def _atomic_nodes(
    operator_order: Sequence[str], operators: Mapping[str, Operator],
) -> Tuple[Node, ...]:
    """Group every mapped BMM with its representative NoC communication."""
    nodes = []
    index = 0
    while index < len(operator_order):
        op_id = operator_order[index]
        if op_id.endswith(".input_broadcast"):
            if index + 2 >= len(operator_order):
                raise ValueError(f"incomplete BMM communication block: {op_id}")
            bmm_id = op_id.removesuffix(".input_broadcast")
            reduce_id = f"{bmm_id}.output_reduce"
            if (
                operator_order[index + 1] != bmm_id
                or operator_order[index + 2] != reduce_id
                or not isinstance(operators[bmm_id], BMMOp)
            ):
                raise ValueError(
                    f"expected broadcast/GEMM/reduce block for {bmm_id}"
                )
            nodes.append(GroupNode((
                OpNode(op_id),
                OpNode(bmm_id),
                OpNode(reduce_id),
            )))
            index += 3
            continue
        if isinstance(operators[op_id], BMMOp):
            raise ValueError(f"BMM is missing its input broadcast: {op_id}")
        if op_id.endswith(".output_reduce"):
            raise ValueError(f"orphan BMM output reduce: {op_id}")
        nodes.append(OpNode(op_id))
        index += 1
    return tuple(nodes)


def build_root_node(
    batch_size: int,
    operator_order: Sequence[str],
    operators: Mapping[str, Operator],
) -> RootNode:
    """Build the split=1 RootNode shared by supported MLA/MoE models.

    Model-specific operations, including DSA paths and TP/EP MoE
    collectives, come from ``operator_order``. This function only adds stable
    semantic grouping and never invents, removes, or reorders an operation.
    """
    order = tuple(operator_order)
    operator_map = dict(operators)
    if set(operator_map) != set(order):
        raise ValueError("operators must exactly match operator_order")
    if any(not isinstance(operator, Operator) for operator in operator_map.values()):
        raise TypeError("operators must contain Operator values")

    attention_order = tuple(op_id for op_id in order if op_id.startswith("attn."))
    moe_order = tuple(op_id for op_id in order if op_id.startswith("moe."))
    if len(attention_order) + len(moe_order) != len(order):
        raise ValueError("RootNode operators must belong to attn.* or moe.*")
    if not attention_order or not moe_order:
        raise ValueError("RootNode requires both attention and MoE operators")
    if order != attention_order + moe_order:
        raise ValueError("attention operators must precede MoE operators")

    attention_group = GroupNode(_atomic_nodes(attention_order, operator_map))
    moe_group = GroupNode(_atomic_nodes(moe_order, operator_map))
    return RootNode(
        batch_size=batch_size,
        operator_order=order,
        root=GroupNode((attention_group, moe_group)),
    )


__all__ = ["RootNode", "build_root_node"]
