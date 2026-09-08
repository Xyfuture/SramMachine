"""Default FP16-equivalent work estimates for vector commands."""

import math
from numbers import Real
from types import MappingProxyType

from srammachine.commands import VectorCmd


VECTOR_FLOPS_PER_ELEMENT = MappingProxyType({
    "softmax": 5,
    "rmsnorm": 5,
    "rope": 3,
    "silu": 4,
    "topk": 1,
})


def vector_flop_count(command: VectorCmd) -> Real:
    """Return the configured FP16-equivalent work for one vector command."""
    if not isinstance(command, VectorCmd):
        raise TypeError("command must be a VectorCmd")
    coefficient = command.params.get("flops_per_element")
    if coefficient is None:
        try:
            coefficient = VECTOR_FLOPS_PER_ELEMENT[command.kind]
        except KeyError as error:
            raise ValueError(
                f"unknown vector kind without flops_per_element: {command.kind}"
            ) from error
    if (isinstance(coefficient, bool) or not isinstance(coefficient, Real)
            or not math.isfinite(coefficient) or coefficient <= 0):
        raise ValueError("flops_per_element must be a positive finite number")
    return command.m * command.n * coefficient
