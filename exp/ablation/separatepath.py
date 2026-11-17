"""Paired, opt-in GEMM K-tile pipeline and shared-path experiment.

Layouts are selected by the existing XFold search and frozen for both variants.
K tiles share the original M/N placement. Input and weight bytes are partitioned
exactly; partial sums stay local until the original output command. Two buffer
slots permit at most one tile of lookahead. Weight loads finish before their
compute tile, while independent paths overlap input/weights/compute across tiles.
Attention/Indexer retain their existing compute model and cache commands.
"""
from dataclasses import replace

from srammachine.commands import CommandGraph, GemmCmd, NoCCmd, PUMemoryReadCmd
from srammachine.simulator import Simulator


PATH_VARIANTS = ("SeparatePath", "NoSeparatePath")


def tile_gemms(graph, k_tile, *, separate_path=True):
    """Lower ordinary GEMMs to double-buffered K slices, preserving traffic."""
    lookup = {c.cmd_id: c for c in graph.commands}
    groups, replacements, targets, sources, starts = {}, {}, {}, {}, {}
    added_edges = []
    traces, priorities = dict(graph.traces), dict(graph.scheduling_priorities)
    for core in graph.commands:
        if type(core) is not GemmCmd:
            continue
        prefix = core.cmd_id.removesuffix(".core")
        inp = lookup.get(prefix + ".input")
        if not isinstance(inp, NoCCmd) or inp.operation != "FromIOdie":
            continue
        weights = [c for c in graph.commands if isinstance(c, PUMemoryReadCmd)
                   and c.cmd_id.startswith(prefix + ".weight.")]
        if not weights or core.memory_bytes:
            continue
        members = [core, inp, *weights]
        for c in members:
            groups[c.cmd_id] = core.cmd_id
        tiles, inputs, computes, reads = [], [], [], []
        for i, lo in enumerate(range(0, core.K, k_tile)):
            hi = min(lo + k_tile, core.K)
            tile_prefix = f"{prefix}.tile{i}"
            # Cumulative integer division conserves each NoC row's byte count.
            part = lambda value: value * hi // core.K - value * lo // core.K
            rows = tuple(part(value) for value in inp.io_bytes_per_row)
            input_tile = replace(inp, cmd_id=tile_prefix + ".input",
                                 size_bytes=max(rows) if rows else part(inp.size_bytes),
                                 io_bytes_per_row=rows)
            compute = replace(core, cmd_id=tile_prefix + ".core", K=hi-lo,
                              memory_duration_ns=0)
            memory = [replace(w, cmd_id=f"{tile_prefix}.weight.{w.port}.{j}",
                              size_bytes=w.size_bytes * hi / core.K - w.size_bytes * lo / core.K)
                      for j, w in enumerate(weights)]
            inputs.append(input_tile.cmd_id)
            computes.append(compute.cmd_id)
            reads.append([w.cmd_id for w in memory])
            tiles.extend([input_tile, *memory, compute])
            added_edges.append((input_tile.cmd_id, compute.cmd_id))
            added_edges.extend((w.cmd_id, compute.cmd_id) for w in memory)
            if i:
                added_edges.append((computes[i-1], compute.cmd_id))
            if i >= 2:
                added_edges.extend((computes[i-2], target)
                                   for target in [input_tile.cmd_id, *reads[i]])
        # Gate both buffer slots at the original invocation's input dependencies.
        entry = [item for i in range(min(2, len(inputs))) for item in [inputs[i], *reads[i]]]
        targets[inp.cmd_id] = entry
        sources[inp.cmd_id], starts[inp.cmd_id] = inputs[-1], inputs[0]
        targets[core.cmd_id] = [computes[0]]
        sources[core.cmd_id], starts[core.cmd_id] = computes[-1], computes[0]
        for j, weight in enumerate(weights):
            targets[weight.cmd_id] = [reads[0][j]]
            sources[weight.cmd_id], starts[weight.cmd_id] = reads[-1][j], reads[0][j]
        replacements[core.cmd_id] = tiles
        for c in tiles:
            traces[c.cmd_id] = graph.traces[core.cmd_id]
            priorities[c.cmd_id] = graph.scheduling_priorities[core.cmd_id]
    commands = []
    for c in graph.commands:
        if c.cmd_id in replacements:
            commands.extend(replacements[c.cmd_id])
        elif c.cmd_id not in groups:
            commands.append(c)
    def rewrite(edges, start=False):
        out = []
        for a, b in edges:
            if a in groups and groups.get(b) == groups[a]:
                continue  # Tile readiness replaces the original fused streaming edges.
            source = (starts if start else sources).get(a, a)
            out.extend((source, target) for target in targets.get(b, [b]))
        return out
    ids = {c.cmd_id for c in commands}
    return CommandGraph(commands, [*rewrite(graph.edges), *added_edges],
                        start_edges=rewrite(graph.start_edges, start=True),
                        traces={k: v for k, v in traces.items() if k in ids},
                        scheduling_priorities={k: v for k, v in priorities.items() if k in ids},
                        noc_compute_memory_exclusive=not separate_path)


class SeparatePathSimulator(Simulator):
    """Apply identical tiling to primary and auxiliary layers before simulation."""
    def __init__(self, hardware, *, separate_path, k_tile):
        super().__init__(hardware)
        self.separate_path = separate_path
        self.k_tile = k_tile

    def run(self, graph, *, mapping_result=None):
        graph = tile_gemms(graph, self.k_tile, separate_path=self.separate_path)
        return super().run(graph, mapping_result=mapping_result)
