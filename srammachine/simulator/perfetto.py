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
            "name": f"{item.op_id} [{item.command_type}]",
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
