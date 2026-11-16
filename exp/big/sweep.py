"""Sweep long-context decode workloads; run from the repository root.

Example: .venv/bin/python -m exp.big.sweep --sequence-lengths 16384 32768 65536
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any
import zipfile

from srammachine.hardware import HardwareConfig, InterChipFabricConfig
from srammachine.inference import InferenceConfig
from srammachine.mapping import (
    DEFAULT_HBF_BANDWIDTH_CSV, HBFBandwidthTable, HardwareMappingRequest,
    mapper_for_model,
)
from srammachine.pipetree import TreeParser
from srammachine.script.run_sa import _without_desim_priority_queue_validation
from srammachine.search import optimize_partition_chains, partition_chains_to_dict, split_tree_to_dict
from srammachine.simulator import Simulator, export_perfetto_trace, steady_pu_metrics


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("deepseek-v3", "deepseek-v3.2", "glm-5.1", "minimax-m3")
BATCHES = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def plain(value: Any) -> Any:
    """Convert immutable simulator records to JSON without losing raw fields."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)
                if not f.name.startswith("_")}
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    """Commit a checkpoint atomically on the same filesystem."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(plain(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def hardware_config() -> HardwareConfig:
    """Pin the requested system while retaining all other hardware defaults."""
    return HardwareConfig(chip_count=8, inter_chip_fabric=InterChipFabricConfig(
        per_chip_directional_bandwidth_bytes_per_second=900_000_000_000,
        full_duplex=True,
    ))


def run_case(task: dict) -> dict:
    """Execute one point in a fresh process and persist its trace and raw data."""
    directory = Path(task["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        try:
            with _without_desim_priority_queue_validation():
                return simulate_case(task, started)
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
            write_json(directory / "error.json", {"task": task, "error": error})
            return {"status": "failed", "model": task["model"],
                    "global_batch_size": task["batch"],
                    "input_sequence_length": task["sequence"], "error": error}


def simulate_case(task: dict, started: float) -> dict:
    """Optimize PartitionChains and obtain metrics from the traced simulation."""
    directory = Path(task["directory"])
    hardware = hardware_config()
    inference = InferenceConfig(task["batch"], task["sequence"], 600, mtp_enabled=True)
    request = HardwareMappingRequest(task["model"], inference,
                                     xfold=task["model"] != "deepseek-v3")
    table = HBFBandwidthTable.from_csv(task["bandwidth_csv"])
    mapper = mapper_for_model(task["model"], hardware)
    mapped = mapper.map(request, hbf_bandwidth_table=table)
    optimization_start = time.monotonic()
    mapped = optimize_partition_chains(mapped, hardware)
    optimization_seconds = time.monotonic() - optimization_start
    graph = TreeParser(hardware).parse(mapped.root_node, mapped.operators,
                                     mapped.operator_mappings, layer_count=4)
    simulator = Simulator(hardware)
    result = simulator.run(graph, mapping_result=mapped)
    trace = export_perfetto_trace(result, trace_label="partition_optimized",
                                  output_dir=directory, mapping_result=mapped)
    # MiniMax's full-decoder metric includes its fixed dense-prefix baseline.
    # Export that baseline too, so every contribution can be inspected.
    auxiliary_traces = []
    for additional in mapped.additional_layer_mappings:
        auxiliary_graph = TreeParser(hardware).parse(
            additional.root_node, additional.operators, additional.operator_mappings,
        )
        auxiliary_result = simulator.run(auxiliary_graph, mapping_result=additional)
        auxiliary_trace = export_perfetto_trace(
            auxiliary_result, trace_label=f"{additional.layer_type}_baseline",
            output_dir=directory, mapping_result=additional,
        )
        auxiliary_traces.append(auxiliary_trace.name)
    latency = result.latency_ns
    total = result.throughput_tokens_per_second
    if latency is None or latency <= 0 or total is None or total <= 0:
        raise ValueError("Expected positive full-model latency and throughput")
    bandwidths = {
        op_id: op.weight_memory.flash_bandwidth_bytes_per_second
        for op_id, op in mapped.operators.items()
        if op.weight_memory is not None
        and op.weight_memory.flash_bandwidth_bytes_per_second is not None
    }
    row = {
        "status": "complete", "model": task["model"],
        "global_batch_size": task["batch"], "input_sequence_length": task["sequence"],
        "tokens_per_second_per_user": total / task["batch"],
        "tokens_per_second_per_chip": total / hardware.chip_count,
        "total_tokens_per_second": total, "model_step_latency_ns": latency,
        "model_step_latency_ms": latency / 1e6,
        "steady_layer_time_ns": result.pipeline_layer_time_ns,
        "trace_total_time_ns": result.total_time_ns,
        "model_layer_count": result.model_layer_count,
        "model_latency_components": plain(result.model_latency_components),
        "steady_pu_metrics": plain(steady_pu_metrics(result.command_results)),
        "chip_count": hardware.chip_count,
        "inter_chip_directional_GBps": 900, "inter_chip_bidirectional_GBps": 1800,
        "mtp_enabled": True, "accepted_tokens_per_step": inference.accepted_tokens_per_step,
        "xfold": mapped.request.xfold, "moe_strategy": inference.moe_parallel_strategy.value,
        "kv_cache_dtype": inference.kv_cache_dtype, "output_sequence_length": 600,
        "simulated_layer_count": result.layer_count,
        "optimize_partition_chains": True, "split_tree_search_enabled": False,
        "flash_bandwidth_source": "measured_csv",
        "flash_bandwidth_bytes_per_second_by_op": bandwidths,
        "minimax_dense_flash_source": "hardware_default" if auxiliary_traces else "",
        "trace_path": str(trace.relative_to(directory.parent.parent)),
        "auxiliary_trace_paths": [str((directory / name).relative_to(directory.parent.parent))
                                  for name in auxiliary_traces],
        "raw_result_path": str((directory / "result.json").relative_to(directory.parent.parent)),
        "optimization_wall_seconds": optimization_seconds,
        "wall_seconds": time.monotonic() - started,
    }
    write_json(directory / "result.json", {
        "metrics": row, "task": task, "hardware": hardware, "inference": inference,
        "partition_chains": partition_chains_to_dict(mapped),
        "split_tree": split_tree_to_dict(mapped.root_node),
        "layer_results": result.layer_results, "command_results": result.command_results,
        "auxiliary_partition_chains": {
            extra.layer_type: partition_chains_to_dict(extra)
            for extra in mapped.additional_layer_mappings
        },
    })
    return row


def mark_pareto(rows: list[dict]) -> list[dict]:
    """Mark nondominated points; both throughput objectives are maximized."""
    x, y = "tokens_per_second_per_user", "tokens_per_second_per_chip"
    return [dict(row, is_pareto=not any(
        other[x] >= row[x] and other[y] >= row[y]
        and (other[x] > row[x] or other[y] > row[y]) for other in rows
    )) for row in rows]


def write_reports(directory: Path, models: list[str]) -> None:
    """Rebuild per-model CSV, Excel and linear Pareto plots from checkpoints."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from openpyxl import Workbook

    for model in models:
        rows = [json.loads(path.read_text())["metrics"]
                for path in (directory / model).glob("bs*/result.json")]
        if not rows:
            continue
        rows = mark_pareto(sorted(rows, key=lambda row: row["global_batch_size"]))
        columns = list(rows[0])
        flat = [{key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list))
                 else value for key, value in row.items()} for row in rows]
        csv_path = directory / f"{model}.csv"
        with csv_path.with_suffix(".csv.tmp").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(flat)
        csv_path.with_suffix(".csv.tmp").replace(csv_path)
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "metrics"
        sheet.append(columns)
        for row in flat:
            sheet.append([row[key] for key in columns])
        sheet.freeze_panes = "D2"
        sheet.auto_filter.ref = sheet.dimensions
        for column in sheet.columns:
            sheet.column_dimensions[column[0].column_letter].width = min(48, max(16, len(str(column[0].value)) + 2))
        notes = workbook.create_sheet("methodology")
        for key, value in json.loads((directory / "manifest.json").read_text()).items():
            notes.append([key, json.dumps(value, ensure_ascii=False)])
        xlsx_path = directory / f"{model}.xlsx"
        workbook.save(xlsx_path.with_suffix(".tmp.xlsx"))
        xlsx_path.with_suffix(".tmp.xlsx").replace(xlsx_path)
        fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
        x, y = "tokens_per_second_per_user", "tokens_per_second_per_chip"
        ax.scatter([r[x] for r in rows], [r[y] for r in rows], color="#64748b", label="All batches")
        front = sorted((r for r in rows if r["is_pareto"]), key=lambda r: r[x])
        ax.plot([r[x] for r in front], [r[y] for r in front], "o-", color="#2563eb", label="Pareto frontier")
        for row in rows:
            ax.annotate(f"B{row['global_batch_size']}", (row[x], row[y]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
        sequence = rows[0]["input_sequence_length"]
        path_name = "XFold" if rows[0]["xfold"] else "ordinary MLA"
        ax.set(title=f"{model} | {sequence // 1024}k | {path_name}\n8 chips | EP | FP8 KV | ideal MTP1",
               xlabel="tokens/s/user", ylabel="tokens/s/chip", xlim=(0, None), ylim=(0, None))
        ax.grid(alpha=0.2)
        ax.legend()
        fig.savefig(directory / f"{model}-pareto.png", dpi=180)
        plt.close(fig)


def snapshot(directory: Path, bandwidth_csv: Path) -> str:
    """Archive actual dirty source, model cards, scripts and environment."""
    directory.mkdir()
    for path in Path(__file__).parent.iterdir():
        if path.is_file():
            shutil.copy2(path, directory / path.name)
    shutil.copy2(bandwidth_csv, directory / "hbf_read_bandwidth.csv")
    archive_path = directory / "source.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        paths = list((ROOT / "srammachine").rglob("*.py"))
        paths += list((ROOT / "srammachine/model_cards").glob("*.json"))
        paths += [ROOT / "AGENTS.md", ROOT / "pyproject.toml"]
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
    for filename, command in (
        ("git-status.txt", ["git", "status", "--short"]),
        ("git-diff.patch", ["git", "diff", "HEAD"]),
        ("git-head.txt", ["git", "rev-parse", "HEAD"]),
        ("environment.txt", [sys.executable, "-m", "pip", "freeze"]),
    ):
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (directory / filename).write_text(completed.stdout + completed.stderr)
    return hashlib.sha256(archive_path.read_bytes()).hexdigest()


def main() -> int:
    """Preflight, execute at most six isolated workers, and produce reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(BATCHES))
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[16384, 32768, 65536],
                        help="context lengths in tokens (default: 16384 32768 65536)")
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--bandwidth-csv", type=Path, default=DEFAULT_HBF_BANDWIDTH_CSV)
    parser.add_argument("--output-root", type=Path, default=Path("test result/big"))
    parser.add_argument("--run-id", default=datetime.now().astimezone().strftime("%Y-%m-%d-%H%M%S"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reports-only", action="store_true")
    args = parser.parse_args()
    if any(value <= 0 for value in args.batch_sizes + args.sequence_lengths):
        parser.error("batch sizes and sequence lengths must be positive")
    for values in (args.models, args.batch_sizes, args.sequence_lengths):
        if len(set(values)) != len(values):
            parser.error("duplicate sweep values are not allowed")
    if Path(args.run_id).name != args.run_id:
        parser.error("run-id must be a directory name")
    args.bandwidth_csv = args.bandwidth_csv.resolve()
    # Validate all lookups before creating results or launching workers.
    table = HBFBandwidthTable.from_csv(args.bandwidth_csv)
    for model in args.models:
        kinds = ("dense_kv",) if model == "deepseek-v3" else ("sparse_kv", "dense_indexer")
        for batch in args.batch_sizes:
            for sequence in args.sequence_lengths:
                for kind in kinds:
                    table.lookup(model, kind, True, batch, sequence)
    tasks, directories = [], []
    for sequence in args.sequence_lengths:
        label = f"{sequence // 1024}k" if sequence % 1024 == 0 else str(sequence)
        directory = args.output_root.resolve() / f"{args.run_id}-big-seqlen-{label}"
        config = {
            "models": args.models, "batch_sizes": args.batch_sizes,
            "input_sequence_length": sequence, "output_sequence_length": 600,
            "hardware": plain(hardware_config()), "mtp_enabled": True,
            "accepted_tokens_per_step": 2, "moe_strategy": "ep", "kv_cache_dtype": "fp8",
            "xfold_models": [model for model in args.models if model != "deepseek-v3"],
            "partition_chain_search": True, "split_tree_search": False,
            "simulated_layer_count": 4, "workers": args.workers,
            "bandwidth_csv_sha256": hashlib.sha256(args.bandwidth_csv.read_bytes()).hexdigest(),
            "throughput_definition": "user = 2e9 / model_step_latency_ns; chip = user * global_batch / 8",
            "latency_definition": "steady layer start interval extrapolated to full decoder; MiniMax weighted sparse+dense",
            "minimax_dense_flash": "hardware default, as existing mapper; CSV contains no dense MiniMax measurements",
        }
        if directory.exists():
            if not (args.resume or args.reports_only):
                parser.error(f"Already exists: {directory}; use --resume")
            saved = json.loads((directory / "manifest.json").read_text())
            if any(saved.get(key) != value for key, value in config.items()):
                parser.error(f"Resume configuration differs from manifest: {directory}")
        else:
            if args.reports_only:
                parser.error(f"No results: {directory}")
            directory.mkdir(parents=True)
            source_hash = snapshot(directory / "scripts", args.bandwidth_csv)
            write_json(directory / "manifest.json", dict(config,
                generated_at=datetime.now().astimezone().isoformat(), source_zip_sha256=source_hash,
                command=sys.argv, python=sys.version))
        directories.append(directory)
        for model in args.models:
            (directory / model).mkdir(exist_ok=True)
            for batch in args.batch_sizes:
                case_directory = directory / model / f"bs{batch}"
                result_path = case_directory / "result.json"
                if result_path.exists():
                    row = json.loads(result_path.read_text())["metrics"]
                    paths = [row["trace_path"], *row["auxiliary_trace_paths"]]
                    if all((directory / path).is_file() for path in paths):
                        continue
                tasks.append({"model": model, "batch": batch, "sequence": sequence,
                              "directory": str(case_directory),
                              "bandwidth_csv": str(directory / "scripts/hbf_read_bandwidth.csv")})
    if args.reports_only:
        for directory in directories:
            write_reports(directory, args.models)
        return 0
    tasks.sort(key=lambda task: (-task["batch"], -task["sequence"], task["model"]))
    print(f"Starting {len(tasks)} points, {args.workers} workers", flush=True)
    failures = []
    context = multiprocessing.get_context("spawn")
    with context.Pool(args.workers, maxtasksperchild=1) as pool:
        for count, row in enumerate(pool.imap_unordered(run_case, tasks, chunksize=1), 1):
            print(f"[{count}/{len(tasks)}] {row['model']} B{row['global_batch_size']} "
                  f"S{row.get('input_sequence_length', '?')} {row['status']} "
                  f"user={row.get('tokens_per_second_per_user', 0):.3f} "
                  f"chip={row.get('tokens_per_second_per_chip', 0):.3f}", flush=True)
            if row["status"] != "complete":
                failures.append(row)
            # Results survive interruption; refresh reports after every point.
            directory = directories[args.sequence_lengths.index(row["input_sequence_length"])]
            write_reports(directory, [row["model"]])
    for directory in directories:
        write_reports(directory, args.models)
        completed = len(list(directory.glob("*/bs*/result.json")))
        write_json(directory / "completion.json", {
            "completed_points": completed, "expected_points": len(args.models) * len(args.batch_sizes),
            "finished_at": datetime.now().astimezone().isoformat(),
        })
    print(f"Finished; {len(failures)} failed points", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
