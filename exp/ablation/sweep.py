"""Sweep operator-specific ablations and compare their throughput and latency."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from datetime import datetime
import hashlib
import itertools
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

from exp.ablation.policy import (
    ABLATIONS, KV_LAYOUT_ABLATIONS, MOE_LAYOUT_ABLATIONS, DEFAULT_MOE_FACTOR, DEFAULT_NO_KV_BANDWIDTH_CSV, load_no_kv_bandwidth,
    apply_ablation, latency_metrics, load_hardware,
)
from exp.ablation.separatepath import PATH_VARIANTS, SeparatePathSimulator
from exp.big.sweep import BATCHES, ROOT, mark_pareto, plain, write_json
from srammachine.inference import InferenceConfig
from srammachine.mapping import (
    DEFAULT_HBF_BANDWIDTH_CSV, HBFBandwidthTable, HardwareMappingRequest, mapper_for_model,
)
from srammachine.pipetree import TreeParser
from srammachine.script.run_sa import _without_desim_priority_queue_validation
from srammachine.search import optimize_partition_chains, partition_chains_to_dict, split_tree_to_dict
from srammachine.simulator import Simulator, export_perfetto_trace, steady_pu_metrics


MODELS = ("minimax-m3", "deepseek-v3", "deepseek-v3.2", "glm-5.1")


def mapping_details(mapped) -> dict:
    """Record the actual operator views, policies and layouts used for lowering."""
    return {
        "hardware_by_op": {key: plain(policy.hardware_config)
                           for key, policy in mapped.operator_mappings.items()},
        "weight_memory_by_op": {key: plain(op.weight_memory) for key, op in mapped.operators.items()},
        "partition_chains": partition_chains_to_dict(mapped),
        "hardware_mappings": plain(mapped.hardware_mappings),
        "split_tree": split_tree_to_dict(mapped.root_node),
    }


def run_case(task: dict) -> dict:
    """Run one point in an isolated worker and write a resumable checkpoint."""
    directory = Path(task["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        try:
            with _without_desim_priority_queue_validation():
                row = simulate_case(task, started)
            (directory / "error.json").unlink(missing_ok=True)
            return row
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
            write_json(directory / "error.json", {"task": task, "error": error})
            return {"status": "failed", "model": task["model"], "ablation": task["ablation"],
                    "global_batch_size": task["batch"], "input_sequence_length": task["sequence"],
                    "error": error}


def simulate_case(task: dict, started: float) -> dict:
    """Map, apply measured-rate ablations, search layouts and simulate four layers."""
    directory, root = Path(task["directory"]), Path(task["root"])
    hardware = load_hardware(Path(task["hardware_json"]))
    inference = InferenceConfig(task["batch"], task["sequence"], 600, mtp_enabled=task.get("mtp_enabled", True))
    request = HardwareMappingRequest(task["model"], inference, xfold=task["ablation"] != "NoXfold")
    table = HBFBandwidthTable.from_csv(task["bandwidth_csv"])
    mapped = mapper_for_model(task["model"], hardware).map(request, hbf_bandwidth_table=table)
    no_kv_table = (load_no_kv_bandwidth(task["no_kv_bandwidth_csv"])
                   if task["ablation"] in KV_LAYOUT_ABLATIONS else None)
    mapped = apply_ablation(mapped, hardware, task["ablation"],
                            moe_factor=task["moe_factor"], moe_flash_scope=task["moe_flash_scope"],
                            no_kv_table=no_kv_table)
    optimization_start = time.monotonic()
    mapped = optimize_partition_chains(mapped, hardware)
    optimization_seconds = time.monotonic() - optimization_start
    pipeline = task["ablation"] in PATH_VARIANTS
    k_tile = task.get("separatepath_k_tile") or (
        hardware.chip.logic_die.processing_unit.systolic_array.columns
        * hardware.chip.logic_die.processing_unit.systolic_array_count)
    simulator = (SeparatePathSimulator(hardware, k_tile=k_tile,
                                      separate_path=task["ablation"] == "SeparatePath")
                 if pipeline else Simulator(hardware))
    graph = TreeParser(hardware).parse(mapped.root_node, mapped.operators, mapped.operator_mappings)
    result = simulator.run(graph, mapping_result=mapped)

    def save_trace(layer_mapping, layer_result, label):
        path = export_perfetto_trace(layer_result, trace_label=label, output_dir=directory,
                                     mapping_result=layer_mapping)
        trace = json.loads(path.read_text())
        trace.setdefault("otherData", {}).update(
            ablation=task["ablation"], hardware_by_op=mapping_details(layer_mapping)["hardware_by_op"],
            latency_adjustment=latency_metrics(layer_result, task["model"], task["ablation"]),
            timing_scope="command timestamps exclude post-simulation additive layer delay",
            gemm_pipeline_k_tile=k_tile if pipeline else 0,
            noc_compute_memory_exclusive=task["ablation"] == "NoSeparatePath")
        write_json(path, trace)
        return str(path.relative_to(root))

    trace = save_trace(mapped, result, "partition_optimized")
    auxiliary_traces, auxiliary_results = [], {}
    for additional in mapped.additional_layer_mappings:
        auxiliary_graph = TreeParser(hardware).parse(
            additional.root_node, additional.operators, additional.operator_mappings)
        auxiliary_result = simulator.run(auxiliary_graph, mapping_result=additional)
        auxiliary_traces.append(save_trace(additional, auxiliary_result, f"{additional.layer_type}_baseline"))
        auxiliary_results[additional.layer_type] = dict(
            mapping_details(additional), command_results=plain(auxiliary_result.command_results),
            layer_results=plain(auxiliary_result.layer_results),
            steady_layer_time_ns=auxiliary_result.pipeline_layer_time_ns)
    timing = latency_metrics(result, task["model"], task["ablation"])
    total = inference.accepted_tokens_per_step * 1e9 * task["batch"] / timing["model_step_latency_ns"]
    row = {
        "status": "complete", "model": task["model"], "ablation": task["ablation"],
        "global_batch_size": task["batch"], "input_sequence_length": task["sequence"],
        "tokens_per_second_per_user": total / task["batch"],
        "tokens_per_second_per_chip": total / hardware.chip_count,
        "total_tokens_per_second": total, **timing,
        "steady_layer_time_ns": result.pipeline_layer_time_ns,
        "trace_total_time_ns": result.total_time_ns,
        "model_layer_count": result.model_layer_count,
        "model_latency_components": plain(result.model_latency_components),
        "steady_pu_metrics": plain(steady_pu_metrics(result.command_results)),
        "chip_count": hardware.chip_count, "mtp_enabled": inference.mtp_enabled,
        "accepted_tokens_per_step": inference.accepted_tokens_per_step,
        "xfold": mapped.request.xfold, "moe_strategy": inference.moe_parallel_strategy.value,
        "kv_cache_dtype": inference.kv_cache_dtype, "output_sequence_length": 600,
        "simulated_layer_count": result.layer_count,
        "optimize_partition_chains": True, "split_tree_search_enabled": False,
        "kv_layout_factor": 1.0,
        "no_kv_bandwidth_source": "scripts/no_kv_layout_bandwidth.csv" if no_kv_table else "",
        "moe_layout_factor": task["moe_factor"] if task["ablation"] in MOE_LAYOUT_ABLATIONS else 1.0,
        "moe_flash_scope": "weights" if task["ablation"] == "NoHBFLayout" else task["moe_flash_scope"],
        "gemm_pipeline_k_tile": k_tile if pipeline else 0,
        "noc_compute_memory_exclusive": task["ablation"] == "NoSeparatePath",
        "comparison_baseline": "XFold",
        "flash_bandwidth_source": "measured_csv_cache; hardware_default_weights; ablation_scaled",
        "flash_bandwidth_bytes_per_second_by_op": {
            key: op.weight_memory.flash_bandwidth_bytes_per_second
            for key, op in mapped.operators.items() if op.weight_memory is not None},
        "minimax_dense_flash_source": "hardware_default_ablation_scaled" if auxiliary_traces else "",
        "trace_path": trace, "auxiliary_trace_paths": auxiliary_traces,
        "raw_result_path": str((directory / "result.json").relative_to(root)),
        "optimization_wall_seconds": optimization_seconds, "wall_seconds": time.monotonic() - started,
    }
    write_json(directory / "result.json", dict(
        mapping_details(mapped), metrics=row, task=task, base_hardware=plain(hardware),
        inference=plain(inference), layer_results=plain(result.layer_results),
        command_results=plain(result.command_results), auxiliary_results=auxiliary_results))
    return row


def case_directory(root: Path, model: str, ablation: str, batch: int, sequence: int) -> Path:
    """Locate a case below the requested model/ablation hierarchy."""
    return root / model / ablation / f"bs{batch}-s{sequence}"


def checkpoint_complete(root: Path, directory: Path) -> bool:
    """Require both the result and every contributing trace before skipping work."""
    try:
        row = json.loads((directory / "result.json").read_text())["metrics"]
        paths = [row["trace_path"], *row["auxiliary_trace_paths"]]
        return row["status"] == "complete" and all((root / path).is_file() for path in paths)
    except (OSError, ValueError, KeyError):
        return False


def write_reports(root: Path, models: list[str]) -> None:
    """Write one CSV per model, one Pareto plot per context and a B1024 bar plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.loads((root / "manifest.json").read_text())
    variants, sequences = manifest["ablations"], manifest["sequence_lengths"]
    colors = dict(zip(ABLATIONS, plt.get_cmap("tab10").colors))
    x, y = "tokens_per_second_per_user", "tokens_per_second_per_chip"
    for model in models:
        rows = []
        for sequence, ablation in itertools.product(sequences, variants):
            group = []
            for batch in manifest["batch_sizes"]:
                directory = case_directory(root, model, ablation, batch, sequence)
                if checkpoint_complete(root, directory):
                    group.append(json.loads((directory / "result.json").read_text())["metrics"])
            rows.extend(mark_pareto(sorted(group, key=lambda row: row["global_batch_size"])))
        if not rows:
            continue
        path = root / f"{model}.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                              else value for key, value in row.items()} for row in rows)
        temporary.replace(path)
        for sequence in sequences:
            fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
            for ablation in variants:
                group = [row for row in rows if row["ablation"] == ablation
                         and row["input_sequence_length"] == sequence]
                front = sorted((row for row in group if row["is_pareto"]), key=lambda row: row[x])
                ax.scatter([row[x] for row in group], [row[y] for row in group],
                           color=colors[ablation], alpha=0.3, s=20)
                ax.plot([row[x] for row in front], [row[y] for row in front], "o-",
                        color=colors[ablation], label=ablation, markersize=4)
                for row in front:
                    ax.annotate(str(row["global_batch_size"]), (row[x], row[y]), xytext=(3, 3),
                                textcoords="offset points", fontsize=6, color=colors[ablation])
            ax.set(title=f"{model} | context {sequence:,} | {manifest['hardware']['chip_count']} chips\n"
                         f"EP | FP8 KV | MTP{int(manifest['mtp_enabled'])} | labels = global batch",
                   xlabel="tokens/s/user", ylabel="tokens/s/chip", xlim=(0, None), ylim=(0, None))
            ax.grid(alpha=0.2)
            ax.legend(fontsize=9)
            fig.savefig(root / f"{model}-pareto-s{sequence}.png", dpi=180)
            plt.close(fig)
        if 1024 in manifest["batch_sizes"]:
            fig, axes = plt.subplots(1, len(sequences), figsize=(7 * len(sequences), 5),
                                     layout="constrained", squeeze=False)
            for sequence, ax in zip(sequences, axes[0]):
                group = {row["ablation"]: row for row in rows if row["global_batch_size"] == 1024
                         and row["input_sequence_length"] == sequence}
                values = [group[name]["model_step_latency_ms"] if name in group else float("nan")
                          for name in variants]
                bars = ax.bar(variants, values, color=[colors[name] for name in variants])
                ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
                ax.set(title=f"{model} | context {sequence:,} | B1024", ylabel="Model step latency (ms)")
                ax.tick_params(axis="x", labelrotation=35, labelsize=8)
                ax.grid(axis="y", alpha=0.2)
                ax.set_ylim(bottom=0)
            fig.savefig(root / f"{model}-latency-bs1024.png", dpi=180)
            plt.close(fig)


def source_files() -> list[Path]:
    """Collect reproducible source, including untracked experiment code."""
    paths = [ROOT / "AGENTS.md", ROOT / "pyproject.toml", ROOT / "uv.lock"]
    for directory in (ROOT / "srammachine", ROOT / "exp"):
        paths.extend(path for path in directory.rglob("*") if path.is_file()
                     and path.suffix in (".py", ".json", ".md", ".txt", ".csv"))
    # Local editable packages are the actual simulator dependencies.
    for name in ("Desim", "PerfTracer"):
        paths.extend((ROOT / "local files" / name).rglob("*.py"))
    return sorted(set(path for path in paths if path.is_file()))


def source_digest() -> str:
    """Hash source names and bytes independently of ZIP timestamps."""
    digest = hashlib.sha256()
    for path in source_files():
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def snapshot(directory: Path, bandwidth_csv: Path) -> str:
    """Back up actual source, editable dependencies, CSV and environment."""
    directory.mkdir()
    for path in Path(__file__).parent.iterdir():
        if path.is_file():
            shutil.copy2(path, directory / path.name)
    shutil.copy2(bandwidth_csv, directory / "hbf_read_bandwidth.csv")
    archive_path = directory / "source.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source_files():
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
    """Validate inputs, execute isolated workers, and refresh resumable reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["minimax-m3"])
    parser.add_argument("--ablations", nargs="+", choices=ABLATIONS, default=list(ABLATIONS))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(BATCHES))
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[32768])
    parser.add_argument("--mtp", type=int, choices=(0, 1), default=1,
                        help="0: one token/step; 1: ideal two tokens/step (default)")
    parser.add_argument("--separatepath-k-tile", type=int, default=0,
                        help="GEMM K tile; 0 = array columns × array count (64 by default)")
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--hardware-config", type=Path, help="partial/full nested HardwareConfig JSON")
    parser.add_argument("--bandwidth-csv", type=Path, default=DEFAULT_HBF_BANDWIDTH_CSV)
    parser.add_argument("--no-kv-bandwidth-csv", type=Path, default=DEFAULT_NO_KV_BANDWIDTH_CSV,
                        help="raw measured NoKVLayout/NoHBFLayout sparse KV HBF bandwidth CSV")
    parser.add_argument("--moe-factor", type=float, default=DEFAULT_MOE_FACTOR)
    parser.add_argument("--moe-flash-scope", choices=("stage", "weights"), default="weights",
                        help="weights: ordinary MoE BMM weights only (default); stage: legacy cache/prefetch scaling")
    parser.add_argument("--output-root", type=Path, default=Path("test result/ablation"))
    parser.add_argument("--run-id", default=datetime.now().astimezone().strftime("%Y-%m-%d-%H%M%S"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reports-only", action="store_true")
    args = parser.parse_args()
    for values in (args.models, args.ablations, args.batch_sizes, args.sequence_lengths):
        if len(set(values)) != len(values):
            parser.error("duplicate sweep values are not allowed")
    if any(value <= 0 for value in args.batch_sizes + args.sequence_lengths):
        parser.error("batches and sequences must be positive")
    if not all(0 < factor <= 1 for factor in (args.moe_factor,)):
        parser.error("layout factors must be in (0, 1]")
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        parser.error("run-id must be a directory name")
    hardware = load_hardware(args.hardware_config)
    table = HBFBandwidthTable.from_csv(args.bandwidth_csv)
    for model, batch, sequence in itertools.product(args.models, args.batch_sizes, args.sequence_lengths):
        kinds = ("dense_kv",) if model == "deepseek-v3" else ("sparse_kv", "dense_indexer")
        for kind in kinds:
            table.lookup(model, kind, bool(args.mtp), batch, sequence)
    no_kv_enabled = any(name in KV_LAYOUT_ABLATIONS for name in args.ablations)
    if no_kv_enabled:
        no_kv_table = load_no_kv_bandwidth(args.no_kv_bandwidth_csv)
        for model, batch, sequence in itertools.product(args.models, args.batch_sizes, args.sequence_lengths):
            no_kv_table.lookup(model, "sparse_kv", bool(args.mtp), batch, sequence)
    config = {
        "models": args.models, "ablations": args.ablations, "batch_sizes": args.batch_sizes,
        "sequence_lengths": args.sequence_lengths, "hardware": plain(hardware),
        "kv_factor": 1.0, "moe_factor": args.moe_factor, "moe_flash_scope": args.moe_flash_scope,
        "ablation_revision": 7,
        "no_hbf_layout_definition": "measured NoKVLayout sparse KV rate + weights-only NoMoELayout Flash factor; no Indexer scaling or added delay",
        "no_kv_bandwidth_csv_sha256": hashlib.sha256(args.no_kv_bandwidth_csv.read_bytes()).hexdigest() if no_kv_enabled else None,
        "separatepath_k_tile": args.separatepath_k_tile,
        "separatepath_model": "paired K-tiled double-buffer GEMMs; frozen XFold layouts; NoSeparatePath excludes horizontal NoC from compute and all memory reads",
        "indexer_bandwidth_definition": "base Flash rate + NoC link bandwidth * 64 * 2 / base PU count; NoFlashShare and NoDRAMFLashShare remove the NoC contribution",
        "no_moe_bandwidth_scope": "ordinary MoE BMM weights only; Indexer, CoreAttention and KV prefetch unchanged"
                                  if args.moe_flash_scope == "weights" else "legacy whole MoE stage Flash reads",
        "no_kv_bandwidth_scope": "measured sparse KV HBF rate: direct KV reads or XFold KV prefetch; Indexer and NoC/HBM bonus unchanged",
        "no_kv_additional_latency_per_layer_ns": 0,
        "mtp_enabled": bool(args.mtp), "accepted_tokens_per_step": 1 + args.mtp, "output_sequence_length": 600,
        "moe_strategy": "ep", "kv_cache_dtype": "fp8", "simulated_layer_count": 4,
        "partition_chain_search": True, "split_tree_search": False,
        "bandwidth_csv_sha256": hashlib.sha256(args.bandwidth_csv.read_bytes()).hexdigest(),
        "throughput_definition": f"user = {1 + args.mtp}e9 / model_step_latency_ns; chip = user * global_batch / chip_count",
        "latency_definition": "steady core-start interval extrapolated to decoder; no additive latency; MiniMax = 57 sparse + 3 dense",
        "minimax_dense_flash": "no CSV measurement; hardware default with applicable ablation scaling",
        "prefetch_hardware": "FC2 execution view; NoKVLayout/NoHBFLayout replace KV prefetch with measured no-layout rate; default NoMoELayout preserves it",
        "deepseek_v3_mode": "ordinary path; NoKVLayout/NoHBFLayout unavailable without matching sparse KV measurements",
    }
    root = args.output_root.resolve() / f"{args.run_id}-ablation"
    if root.exists():
        if not (args.resume or args.reports_only):
            parser.error(f"Already exists: {root}; use --resume")
        saved = json.loads((root / "manifest.json").read_text())
        if any(saved.get(key) != value for key, value in config.items()):
            parser.error("configuration differs from saved manifest")
        if not args.reports_only and saved["source_digest"] != source_digest():
            parser.error("source differs from archived experiment; use a new run-id")
    else:
        if args.reports_only:
            parser.error(f"No results: {root}")
        root.mkdir(parents=True)
        archive_hash = snapshot(root / "scripts", args.bandwidth_csv)
        if no_kv_enabled:
            shutil.copy2(args.no_kv_bandwidth_csv, root / "scripts/no_kv_layout_bandwidth.csv")
        write_json(root / "scripts/base_hardware.json", hardware)
        write_json(root / "manifest.json", dict(
            config, generated_at=datetime.now().astimezone().isoformat(), command=sys.argv,
            source_zip_sha256=archive_hash, source_digest=source_digest(), python=sys.version))
    tasks = []
    for model, ablation, batch, sequence in itertools.product(
        args.models, args.ablations, args.batch_sizes, args.sequence_lengths
    ):
        directory = case_directory(root, model, ablation, batch, sequence)
        if checkpoint_complete(root, directory):
            continue
        tasks.append(dict(model=model, ablation=ablation, batch=batch, sequence=sequence,
                          directory=str(directory), root=str(root),
                          bandwidth_csv=str(root / "scripts/hbf_read_bandwidth.csv"),
                          hardware_json=str(root / "scripts/base_hardware.json"),
                          no_kv_bandwidth_csv=str(root / "scripts/no_kv_layout_bandwidth.csv"),
                          moe_factor=args.moe_factor,
                          moe_flash_scope=args.moe_flash_scope, mtp_enabled=bool(args.mtp),
                          separatepath_k_tile=args.separatepath_k_tile))
    if args.reports_only:
        write_reports(root, args.models)
        return 0
    tasks.sort(key=lambda task: (-task["batch"], -task["sequence"], task["ablation"]))
    print(f"Starting {len(tasks)} points, {args.workers} workers; {root}", flush=True)
    failures = []
    last_report = time.monotonic()
    with multiprocessing.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for count, row in enumerate(pool.imap_unordered(run_case, tasks, chunksize=1), 1):
            print(f"[{count}/{len(tasks)}] {row['model']} {row['ablation']} "
                  f"B{row['global_batch_size']} S{row['input_sequence_length']} {row['status']} "
                  f"latency_ms={row.get('model_step_latency_ms', 0):.3f}", flush=True)
            if row["status"] != "complete":
                failures.append(row)
                print(row["error"], file=sys.stderr, flush=True)
            # Checkpoints are durable per point; throttle expensive full plots.
            if time.monotonic() - last_report >= 30:
                write_reports(root, args.models)
                last_report = time.monotonic()
    write_reports(root, args.models)
    expected = len(args.models) * len(args.ablations) * len(args.batch_sizes) * len(args.sequence_lengths)
    complete = sum(checkpoint_complete(root, case_directory(root, *values)) for values in itertools.product(
        args.models, args.ablations, args.batch_sizes, args.sequence_lengths))
    write_json(root / "completion.json", {
        "status": "complete" if complete == expected else "incomplete",
        "completed_points": complete, "expected_points": expected, "failed_points": failures,
        "finished_at": datetime.now().astimezone().isoformat(),
    })
    print(f"Finished: {complete}/{expected}; {len(failures)} failed", flush=True)
    return 0 if complete == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
