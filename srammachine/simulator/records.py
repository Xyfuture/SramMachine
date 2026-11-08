"""Immutable nanosecond records produced by command-graph simulation."""

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from srammachine.commands.base import integer, nonempty


class CommandState(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"


@dataclass(frozen=True)
class CommandExecution:
    cmd_id: str
    resource_id: str
    start_time_ns: int
    end_time_ns: int
    additional_resource_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        nonempty("cmd_id", self.cmd_id)
        nonempty("resource_id", self.resource_id)
        additional = tuple(self.additional_resource_ids)
        for resource_id in additional:
            nonempty("additional resource ID", resource_id)
        if len(set((self.resource_id,) + additional)) != 1 + len(additional):
            raise ValueError("resource IDs must be unique")
        object.__setattr__(self, "additional_resource_ids", additional)
        integer("start_time_ns", self.start_time_ns)
        integer("end_time_ns", self.end_time_ns)
        if self.end_time_ns < self.start_time_ns:
            raise ValueError("end_time_ns must not precede start_time_ns")

    @property
    def resource_ids(self) -> tuple[str, ...]:
        """All resources occupied during this execution interval."""
        return (self.resource_id,) + self.additional_resource_ids

    @property
    def duration_ns(self) -> int:
        return self.end_time_ns - self.start_time_ns


@dataclass(frozen=True)
class ExecutionResult:
    total_time_ns: int
    executions: Mapping[str, CommandExecution]
    states: Mapping[str, CommandState]

    def __post_init__(self) -> None:
        integer("total_time_ns", self.total_time_ns)
        executions = dict(self.executions)
        states = dict(self.states)
        if set(executions) != set(states):
            raise ValueError("executions and states must contain the same command IDs")
        if any(not isinstance(record, CommandExecution)
               for record in executions.values()):
            raise TypeError("executions must contain CommandExecution values")
        if any(key != record.cmd_id for key, record in executions.items()):
            raise ValueError("execution keys must match record command IDs")
        if any(not isinstance(state, CommandState) for state in states.values()):
            raise TypeError("states must contain CommandState values")
        expected_total_time_ns = max(
            (record.end_time_ns for record in executions.values()),
            default=0,
        )
        if self.total_time_ns != expected_total_time_ns:
            raise ValueError(
                "total_time_ns must equal the latest command end time"
            )
        object.__setattr__(self, "executions", MappingProxyType(executions))
        object.__setattr__(self, "states", MappingProxyType(states))
