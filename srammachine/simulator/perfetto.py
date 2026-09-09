"""Export in-memory simulation results as Perfetto-compatible JSON traces."""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Union

from .results import SimulationResult


DEFAULT_TRACE_OUTPUT_DIR = Path("test result")
_TRACE_PROCESS_ID = 1


@dataclass(frozen=True)
class SimulationArtifacts:
    """A completed simulation and the relative or caller-selected trace path."""

    simulation_result: SimulationResult
    trace_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.simulation_result, SimulationResult):
            raise TypeError("simulation_result must be a SimulationResult")
        if not isinstance(self.trace_path, Path):
            raise TypeError("trace_path must be a pathlib.Path")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(f"trace argument is not JSON-compatible: {type(value).__name__}")


def _trace_label(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("trace_label must be a nonempty string")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not sanitized:
        raise ValueError("trace_label must contain a filename-safe character")
    return sanitized


def _trace_path(
    output_dir: Path, trace_label: str, timestamp: datetime,
) -> Path:
    prefix = timestamp.astimezone().strftime("%Y%m%d_%H%M")
    stem = f"{prefix}_{trace_label}_perfetto"
    candidate = output_dir / f"{stem}.json"
    suffix = 2
    while candidate.exists():
        candidate = output_dir / f"{stem}_{suffix:02d}.json"
        suffix += 1
    return candidate


def _trace_payload(result: SimulationResult) -> Mapping[str, Any]:
    resource_ids = tuple(dict.fromkeys(
        item.resource_id for item in result.command_results
    ))
    thread_ids = {
        resource_id: index
        for index, resource_id in enumerate(resource_ids, start=1)
    }
    events = [{
        "ph": "M",
        "pid": _TRACE_PROCESS_ID,
        "name": "process_name",
        "args": {"name": "SramMachine"},
    }]
    events.extend({
        "ph": "M",
        "pid": _TRACE_PROCESS_ID,
        "tid": thread_ids[resource_id],
        "name": "thread_name",
        "args": {"name": resource_id},
    } for resource_id in resource_ids)

    stable_order = {
        item.cmd_id: index for index, item in enumerate(result.command_results)
    }
    ordered_results = sorted(
        result.command_results,
        key=lambda item: (item.start_time_ns, stable_order[item.cmd_id]),
    )
    for item in ordered_results:
        args = {
            "cmd_id": item.cmd_id,
            "op_id": item.op_id,
            "command_type": item.command_type,
            "category": item.category.value,
            "resource_id": item.resource_id,
            "layer_index": item.layer_index,
            "instance_index": item.instance_index,
            "token_start": item.token_start,
            "token_stop": item.token_stop,
            "node_path": list(item.node_path),
            "start_time_ns": item.start_time_ns,
            "end_time_ns": item.end_time_ns,
            "duration_ns": item.duration_ns,
        }
        args.update(_json_value(item.parameters))
        event = {
            "name": _event_name(item),
            "cat": item.category.value,
            "pid": _TRACE_PROCESS_ID,
            "tid": thread_ids[item.resource_id],
            "ts": item.start_time_ns / 1_000,
            "args": args,
        }
        if item.duration_ns:
            event.update({"ph": "X", "dur": item.duration_ns / 1_000})
        else:
            event.update({"ph": "i", "s": "t"})
        events.append(event)

    return {
        "traceEvents": events,
        "displayTimeUnit": "ns",
    }


def _event_name(item) -> str:
    base = f"{item.op_id} [{item.command_type}]"
    parameters = item.parameters
    if "gemm_b" in parameters:
        return (
            f"{base} B={parameters['gemm_b']} M={parameters['gemm_m']} "
            f"K={parameters['gemm_k']} N={parameters['gemm_n']}"
        )
    if "vector_kind" in parameters:
        return (
            f"{base} {parameters['vector_kind']} "
            f"m={parameters['vector_m']} n={parameters['vector_n']}"
        )
    if "weight_size_bytes" in parameters:
        shape = parameters.get("weight_shape") or {}
        shape_text = _shape_text(shape, ("B", "K", "N"))
        suffix = f"weight={_format_bytes(parameters['weight_size_bytes'])}"
        if shape_text:
            suffix += f" {shape_text}"
        return f"{base} {suffix}"
    if "communication_kind" in parameters:
        size = parameters.get("communication_size_bytes")
        critical = parameters.get("communication_critical_path_bytes")
        suffix = (
            f"{parameters['communication_scope']} "
            f"{parameters['communication_kind']}"
        )
        if size is not None:
            suffix += f" size={_format_bytes(size)}"
        suffix += f" critical={_format_bytes(critical)}"
        return f"{base} {suffix}"
    if "size_bytes" in parameters:
        return f"{base} size={_format_bytes(parameters['size_bytes'])}"
    return base


def _shape_text(shape, order) -> str:
    parts = []
    for key in order:
        if key in shape:
            parts.append(f"{key}={shape[key]}")
    return " ".join(parts)


def _format_bytes(value) -> str:
    if value is None:
        return "0B"
    size = float(value)
    unit = "B"
    if abs(size) >= 1024 * 1024:
        size /= 1024 * 1024
        unit = "MB"
    elif abs(size) >= 1024:
        size /= 1024
        unit = "KB"
    if size.is_integer():
        return f"{int(size)}{unit}"
    return f"{size:.2f}{unit}"


def export_perfetto_trace(
    result: SimulationResult,
    *,
    trace_label: str = "srammachine",
    output_dir: Union[str, Path] = DEFAULT_TRACE_OUTPUT_DIR,
    timestamp: Optional[datetime] = None,
) -> Path:
    """Write one Chrome Trace Event JSON file without overwriting old traces."""
    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult")
    label = _trace_label(trace_label)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = _trace_path(directory, label, timestamp or datetime.now().astimezone())
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_trace_payload(result), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path


__all__ = [
    "DEFAULT_TRACE_OUTPUT_DIR",
    "SimulationArtifacts",
    "export_perfetto_trace",
]
