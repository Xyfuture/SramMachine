"""Dependency DAG with stable order and optional scheduling priorities.

Scheduling priority affects only ready commands sharing one resource; it never
adds a dependency or makes an unready command block ready work.
"""
from dataclasses import dataclass
import heapq
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Tuple
from .base import Command, integer


@dataclass(frozen=True)
class CommandTrace:
    instance_index: Optional[int]
    token_start: int
    token_stop: int
    node_path: Tuple[int, ...]
    layer_index: int = 0

    def __post_init__(self) -> None:
        integer("layer_index", self.layer_index)
        if self.instance_index is not None:
            integer("instance_index", self.instance_index)
        integer("token_start", self.token_start)
        integer("token_stop", self.token_stop)
        if self.token_stop <= self.token_start:
            raise ValueError("token range must be nonempty")
        for index in self.node_path:
            integer("node_path index", index)
        object.__setattr__(self, "node_path", tuple(self.node_path))


class CommandGraph:
    def __init__(
        self, commands: Iterable[Command],
        edges: Iterable[Tuple[str, str]] = (),
        traces: Optional[Mapping[str, CommandTrace]] = None,
        scheduling_priorities: Optional[Mapping[str, int]] = None,
    ) -> None:
        self.commands = tuple(commands)
        lookup = {}
        for cmd in self.commands:
            if not isinstance(cmd, Command):
                raise TypeError("graph nodes must be Commands")
            if cmd.cmd_id in lookup:
                raise ValueError(f"duplicate command ID: {cmd.cmd_id}")
            lookup[cmd.cmd_id] = cmd
        self._lookup = lookup
        self._rank = {key: i for i, key in enumerate(lookup)}
        successors = {key: set() for key in lookup}
        predecessors = {key: set() for key in lookup}
        edge_set = set()
        for source, target in edges:
            if source not in lookup or target not in lookup:
                raise ValueError(f"edge references missing command: {(source, target)}")
            edge_set.add((source, target))
            successors[source].add(target)
            predecessors[target].add(source)
        self.edges = tuple(sorted(edge_set, key=lambda e: (self._rank[e[0]], self._rank[e[1]])))
        self._successors = {k: tuple(sorted(v, key=self._rank.get)) for k, v in successors.items()}
        self._predecessors = {k: tuple(sorted(v, key=self._rank.get)) for k, v in predecessors.items()}
        trace_map = dict(traces or {})
        if not set(trace_map).issubset(lookup):
            raise ValueError("trace references missing command")
        if any(not isinstance(t, CommandTrace) for t in trace_map.values()):
            raise TypeError("traces must contain CommandTrace values")
        self.traces = MappingProxyType(trace_map)
        priority_map = dict(scheduling_priorities or {})
        if not set(priority_map).issubset(lookup):
            raise ValueError("scheduling priority references missing command")
        for cmd_id, priority in priority_map.items():
            integer(f"scheduling priority for {cmd_id}", priority)
        self.scheduling_priorities = MappingProxyType({
            cmd_id: priority_map.get(cmd_id, rank)
            for cmd_id, rank in self._rank.items()
        })
        resources = {}
        for cmd in self.commands:
            resources.setdefault(cmd.resource_id, []).append(cmd.cmd_id)
        self.resource_order = MappingProxyType({k: tuple(v) for k, v in resources.items()})
        self.topological_order()

    def command(self, cmd_id: str) -> Command:
        return self._lookup[cmd_id]

    def predecessors(self, cmd_id: str) -> Tuple[str, ...]:
        return self._predecessors[cmd_id]

    def successors(self, cmd_id: str) -> Tuple[str, ...]:
        return self._successors[cmd_id]

    def topological_order(self) -> Tuple[str, ...]:
        """Return IDs in topological order, with insertion order breaking ties."""
        remaining = {k: len(v) for k, v in self._predecessors.items()}
        ready = [self._rank[k] for k, degree in remaining.items() if degree == 0]
        heapq.heapify(ready)
        result = []
        while ready:
            key = self.commands[heapq.heappop(ready)].cmd_id
            result.append(key)
            for target in self._successors[key]:
                remaining[target] -= 1
                if remaining[target] == 0:
                    heapq.heappush(ready, self._rank[target])
        if len(result) != len(self.commands):
            raise ValueError("command graph contains a cycle")
        return tuple(result)
