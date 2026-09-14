"""Steady-state hardware utilization metrics."""

from typing import Any, Iterable


PU_COMMAND_TYPES = frozenset((
    "GemmCmd",
    "FlashAttentionCmd",
    "FusedIndexerScoreCmd",
))


def steady_pu_metrics(command_results: Iterable[Any]) -> dict[str, Any]:
    """Measure representative-PU occupancy from the second to third layer.

    Command result layer indices start at zero, so the steady-state window is
    the interval between the first core command in layer 1 and layer 2.
    """
    commands = tuple(command_results)
    starts = []
    for layer in (1, 2):
        layer_starts = [
            item.start_time_ns for item in commands
            if item.layer_index == layer and item.cmd_id.endswith(".core")
        ]
        if not layer_starts:
            raise ValueError(f"layer {layer} has no core command")
        starts.append(min(layer_starts))
    window_start, window_end = starts
    if window_end <= window_start:
        raise ValueError("steady-state layer window must be positive")

    pu_commands = [
        item for item in commands
        if item.command_type in PU_COMMAND_TYPES
    ]
    resources = sorted({item.resource_id for item in pu_commands})
    if len(resources) != 1:
        raise ValueError(
            "expected exactly one representative PU resource, got "
            f"{resources}"
        )
    busy = 0
    overlapping_count = 0
    for item in pu_commands:
        overlap = max(
            0,
            min(item.end_time_ns, window_end)
            - max(item.start_time_ns, window_start),
        )
        if overlap:
            busy += overlap
            overlapping_count += 1
    window = window_end - window_start
    if busy > window:
        raise RuntimeError("representative PU commands overlap on a serial resource")
    return {
        "steady_start_time_ns": window_start,
        "steady_end_time_ns": window_end,
        "steady_window_time_ns": window,
        "steady_pu_busy_time_ns": busy,
        "steady_pu_command_count": overlapping_count,
        "pu_resource_id": resources[0],
        "pu_utilization_fraction": busy / window,
        "pu_utilization_percent": busy * 100.0 / window,
    }


__all__ = ["PU_COMMAND_TYPES", "steady_pu_metrics"]
