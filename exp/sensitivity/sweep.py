"""One-factor-at-a-time decode sensitivity sweep (run from repository root)."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

from exp.big.sweep import BATCHES, ROOT, mark_pareto, plain, write_json
from srammachine.hardware import HardwareConfig
from srammachine.inference import InferenceConfig
from srammachine.mapping import (
    DEFAULT_HBF_BANDWIDTH_CSV, HBFBandwidthTable, HardwareMappingRequest,
    mapper_for_model,
)
from srammachine.pipetree import TreeParser
from srammachine.script.run_sa import _without_desim_priority_queue_validation
from srammachine.search import optimize_partition_chains, partition_chains_to_dict, split_tree_to_dict
from srammachine.simulator import Simulator, export_perfetto_trace


PARAMETERS = ("hbf", "hbm", "noc", "logic")
FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5)
LABELS = {"hbf": "HBF bandwidth", "hbm": "HBM bandwidth",
          "noc": "Intra-chip NoC bandwidth", "logic": "Logic compute"}
X = "tokens_per_second_per_user"
Y = "tokens_per_second_per_chip"


def scaled_hardware(parameter: str, factor: float, logic_scope: str = "both") -> HardwareConfig:
    """Return a fresh configuration, changing only the selected resource family."""
    if parameter not in PARAMETERS or factor not in FACTORS:
        raise ValueError("unsupported parameter or factor")
    if logic_scope not in ("both", "vector", "pu"):
        raise ValueError("logic_scope must be both, vector or pu")
    hardware = HardwareConfig()
    chip = hardware.chip
    die = chip.logic_die
    pu = die.processing_unit
    if parameter == "hbf":
        pu = replace(pu, flash_bandwidth_bytes_per_second=pu.flash_bandwidth_bytes_per_second * factor)
    elif parameter == "hbm":
        pu = replace(pu, dram_bandwidth_bytes_per_second=pu.dram_bandwidth_bytes_per_second * factor)
        die = replace(die, memory=replace(die.memory,
            dram_bandwidth_bytes_per_second=die.memory.dram_bandwidth_bytes_per_second * factor))
    elif parameter == "noc":
        noc = chip.noc
        chip = replace(chip, noc=replace(noc,
            link_bandwidth_bytes_per_second=noc.link_bandwidth_bytes_per_second * factor,
            horizontal_bandwidth_bytes_per_second=(None if noc.horizontal_bandwidth_bytes_per_second is None
                else noc.horizontal_bandwidth_bytes_per_second * factor),
            vertical_bandwidth_bytes_per_second=(None if noc.vertical_bandwidth_bytes_per_second is None
                else noc.vertical_bandwidth_bytes_per_second * factor)))
    elif parameter == "logic":
        if logic_scope in ("both", "pu"):
            # Scale frequency, retaining matrix dimensions and array count.
            array = pu.systolic_array
            frequency = array.frequency_hz * factor
            if not float(frequency).is_integer():
                raise ValueError("scaled systolic frequency must be integral")
            pu = replace(pu, systolic_array=replace(array, frequency_hz=int(frequency)))
        if logic_scope in ("both", "vector"):
            die = replace(die, vector_unit=replace(die.vector_unit,
                peak_flops=die.vector_unit.peak_flops * factor))
    return replace(hardware, chip=replace(chip, logic_die=replace(die, processing_unit=pu)))


def cap_remote_supply(mapping, hardware, factor):
    """Calibrated pooled-PU model; half local/half remote, no new link events.

    At 1x return the original mapping exactly. Prefetched KV is fully remote.
    Indexer's added HBM contribution is remote and saturates at its 1x rate.
    """
    if factor == 1.0:
        return mapping
    remote = min(1.0, factor)
    pooled = (1.0 + remote) / 2.0
    operators = dict(mapping.operators)
    policies = dict(mapping.operator_mappings)
    for key, op in operators.items():
        if op.weight_memory is None:
            continue
        policy = policies[key]
        hw = policy.hardware_config or hardware
        die = hw.chip.logic_die
        pu = die.processing_unit
        # These are equivalent per-PU rates over symmetric HBF/HBM sides;
        # the local half is unchanged, only the remote half is throttled.
        capped_pu = replace(pu,
            dram_bandwidth_bytes_per_second=pu.dram_bandwidth_bytes_per_second * pooled,
            flash_bandwidth_bytes_per_second=pu.flash_bandwidth_bytes_per_second * pooled)
        policies[key] = replace(policy, hardware_config=replace(hw, chip=replace(
            hw.chip, logic_die=replace(die, processing_unit=capped_pu))))
        weight = op.weight_memory
        rate = weight.flash_bandwidth_bytes_per_second or pu.flash_bandwidth_bytes_per_second
        if key in ("attn.indexer_score", "moe.indexer_score") and weight.source.value != "dram":
            # Mapping has already added the old linearly-scaled NoC term.
            old_extra = hardware.chip.noc.link_bandwidth_bytes_per_second * 128 / hardware.chip.processing_unit_count
            base_extra = old_extra / factor
            rate = (rate - old_extra) * pooled + base_extra * remote
        elif op.flash_prefetch_bytes_per_request:
            rate *= remote
        else:
            rate *= pooled
        operators[key] = replace(op, weight_memory=replace(weight,
            flash_bandwidth_bytes_per_second=rate))
    return replace(mapping, operators=operators, op_list=[operators[op.op_id] for op in mapping.op_list],
        operator_mappings=policies, additional_layer_mappings=tuple(
            cap_remote_supply(item, hardware, factor) for item in mapping.additional_layer_mappings))


def apply_attention_efficiency(mapping, hardware, efficiency):
    """Only core attention's roofline compute peak changes; memory is unchanged."""
    if efficiency == 1:
        return mapping
    policies = dict(mapping.operator_mappings)
    for key in ("attn.flash_attention", "attn.flash_msa"):
        if key not in policies:
            continue
        policy = policies[key]
        hw = policy.hardware_config or hardware
        die = hw.chip.logic_die
        pu = die.processing_unit
        array = replace(pu.systolic_array,
                        frequency_hz=round(pu.systolic_array.frequency_hz * efficiency))
        policies[key] = replace(policy, hardware_config=replace(hw, chip=replace(hw.chip,
            logic_die=replace(die, processing_unit=replace(pu, systolic_array=array)))))
    return replace(mapping, operator_mappings=policies, additional_layer_mappings=tuple(
        apply_attention_efficiency(item, hardware, efficiency) for item in mapping.additional_layer_mappings))


class ScaledHBFBandwidthTable(HBFBandwidthTable):
    """Multiply actual CSV lookups, including XFold prefetch, without editing CSV."""

    def __init__(self, rows: list[dict], factor: float = 1.0) -> None:
        super().__init__(rows)
        if factor not in FACTORS:
            raise ValueError("unsupported HBF factor")
        self.factor = factor

    def lookup(self, model_name: str, read_kind: str, mtp_enabled: bool,
               global_batch_size: int, input_sequence_length: int) -> float:
        """Preserve the original interpolation/fallback rules, then scale the rate."""
        return self.factor * super().lookup(model_name, read_kind, mtp_enabled,
                                           global_batch_size, input_sequence_length)


def bandwidth_table(path: Path, factor: float) -> ScaledHBFBandwidthTable:
    """Load original measurements with a run-local multiplier."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return ScaledHBFBandwidthTable(list(csv.DictReader(stream)), factor)


def case_stem(task: dict) -> str:
    """Unique filenames within a model/parameter directory."""
    return f"x{task['factor']:g}-bs{task['batch']}"


def run_case(task: dict) -> dict:
    """Optimize and trace one configuration in an isolated simulator process."""
    directory = Path(task["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    stem = case_stem(task)
    started = time.monotonic()
    identity = {"model": task["model"], "parameter": task["parameter"],
                "factor": task["factor"], "global_batch_size": task["batch"]}
    with (directory / f"{stem}.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        try:
            with _without_desim_priority_queue_validation():
                hardware = scaled_hardware(task["parameter"], task["factor"], task["logic_scope"])
                inference = InferenceConfig(task["batch"], task["sequence"], 600, mtp_enabled=task.get("mtp_enabled", True))
                hbf_factor = task["factor"] if task["parameter"] == "hbf" else 1.0
                table = bandwidth_table(Path(task["bandwidth_csv"]), hbf_factor)
                request = HardwareMappingRequest(task["model"], inference, xfold=True)
                mapped = mapper_for_model(task["model"], hardware).map(request, hbf_bandwidth_table=table)
                if task["parameter"] == "noc":
                    mapped = cap_remote_supply(mapped, hardware, task["factor"])
                mapped = optimize_partition_chains(mapped, hardware)
                mapped = apply_attention_efficiency(mapped, hardware, task.get("attention_efficiency", 1.0))
                graph = TreeParser(hardware).parse(mapped.root_node, mapped.operators,
                                                  mapped.operator_mappings, layer_count=4)
                simulator = Simulator(hardware)
                result = simulator.run(graph, mapping_result=mapped)
                latency = result.latency_ns
                total = result.throughput_tokens_per_second
                if latency is None or latency <= 0 or total is None or total <= 0:
                    raise ValueError("expected positive full-model latency and throughput")
                trace = export_perfetto_trace(result, trace_label=stem,
                    output_dir=directory, mapping_result=mapped)
                trace_path = directory / f"{stem}.perfetto.json"
                trace.replace(trace_path)
                result_path = directory / f"{stem}.result.json"
                root = directory.parent.parent
                auxiliary_results, auxiliary_paths = {}, []
                for additional in mapped.additional_layer_mappings:
                    aux_graph = TreeParser(hardware).parse(additional.root_node, additional.operators,
                                                          additional.operator_mappings, layer_count=4)
                    aux_result = simulator.run(aux_graph, mapping_result=additional)
                    aux_trace = export_perfetto_trace(aux_result, trace_label=f"{stem}-{additional.layer_type}",
                        output_dir=directory, mapping_result=additional)
                    aux_path = directory / f"{stem}.{additional.layer_type}.perfetto.json"
                    aux_trace.replace(aux_path)
                    auxiliary_paths.append(str(aux_path.relative_to(root)))
                    auxiliary_results[additional.layer_type] = dict(
                        partition_chains=partition_chains_to_dict(additional),
                        steady_layer_time_ns=aux_result.pipeline_layer_time_ns,
                        command_results=plain(aux_result.command_results),
                        layer_results=plain(aux_result.layer_results),
                        weight_memory_by_op={key: plain(op.weight_memory) for key, op in additional.operators.items()},
                        prefetch_op_ids=[key for key, op in additional.operators.items() if op.flash_prefetch_bytes_per_request])
                row = dict(identity, attention_efficiency=task.get("attention_efficiency", 1.0), status="complete", input_sequence_length=task["sequence"],
                    tokens_per_second_per_user=total / task["batch"],
                    tokens_per_second_per_chip=total / hardware.chip_count,
                    total_tokens_per_second=total, model_step_latency_ns=latency,
                    steady_layer_time_ns=result.pipeline_layer_time_ns,
                    trace_total_time_ns=result.total_time_ns, model_layer_count=result.model_layer_count,
                    chip_count=hardware.chip_count, mtp_enabled=inference.mtp_enabled,
                    accepted_tokens_per_step=inference.accepted_tokens_per_step,
                    xfold=True, moe_strategy=inference.moe_parallel_strategy.value,
                    kv_cache_dtype=inference.kv_cache_dtype, output_sequence_length=600,
                    simulated_layer_count=result.layer_count, logic_scope=task["logic_scope"],
                    optimize_partition_chains=True, split_tree_search_enabled=False,
                    hbf_csv_factor=hbf_factor,
                    flash_bandwidth_bytes_per_second_by_op={
                        key: op.weight_memory.flash_bandwidth_bytes_per_second
                        for key, op in mapped.operators.items()
                        if op.weight_memory is not None
                        and op.weight_memory.flash_bandwidth_bytes_per_second is not None},
                    hardware=plain(hardware), trace_path=str(trace_path.relative_to(root)),
                    auxiliary_trace_paths=auxiliary_paths,
                    model_latency_components=plain(result.model_latency_components),
                    raw_result_path=str(result_path.relative_to(root)),
                    wall_seconds=time.monotonic() - started)
                write_json(result_path, {"metrics": row, "task": task, "hardware": hardware,
                    "inference": inference, "partition_chains": partition_chains_to_dict(mapped),
                    "split_tree": split_tree_to_dict(mapped.root_node),
                    "command_results": result.command_results, "layer_results": result.layer_results,
                    "weight_memory_by_op": {key: plain(op.weight_memory) for key, op in mapped.operators.items()},
                    "prefetch_op_ids": [key for key, op in mapped.operators.items() if op.flash_prefetch_bytes_per_request],
                    "auxiliary_results": auxiliary_results})
                (directory / f"{stem}.error.json").unlink(missing_ok=True)
                return row
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
            write_json(directory / f"{stem}.error.json", {"task": task, "error": error})
            return dict(identity, status="failed", error=error)


def completed_row(task: dict) -> dict | None:
    """Accept a checkpoint only if its task and nonempty trace still match."""
    path = Path(task["directory"]) / f"{case_stem(task)}.result.json"
    try:
        payload = json.loads(path.read_text())
        row = payload["metrics"]
        trace = Path(task["directory"]).parent.parent / row["trace_path"]
        if (payload["task"] == task and row["status"] == "complete" and trace.stat().st_size
                and all((Path(task["directory"]).parent.parent / path).is_file()
                        for path in row.get("auxiliary_trace_paths", []))):
            return row
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def write_reports(directory: Path, config: dict, tasks: list[dict]) -> int:
    """Write four CSVs and one overlaid linear Pareto plot per model/parameter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rows = [row for task in tasks if (row := completed_row(task)) is not None]
    for parameter in config["parameters"]:
        rows = []
        for model in config["models"]:
            for factor in config["factors"]:
                group = sorted((r for r in all_rows if r["model"] == model
                    and r["parameter"] == parameter and r["factor"] == factor),
                    key=lambda r: r["global_batch_size"])
                rows.extend(mark_pareto(group))
        columns = list(rows[0]) if rows else ["model", "parameter", "factor", "global_batch_size", X, Y, "is_pareto"]
        target = directory / f"{parameter}.csv"
        temporary = target.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()} for row in rows)
        temporary.replace(target)
        for model in config["models"]:
            fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
            for index, factor in enumerate(config["factors"]):
                group = [r for r in rows if r["model"] == model and r["factor"] == factor]
                front = sorted((r for r in group if r["is_pareto"]), key=lambda r: r[X])
                color = f"C{index}"
                ax.scatter([r[X] for r in group], [r[Y] for r in group], color=color, alpha=0.4, s=22)
                ax.plot([r[X] for r in front], [r[Y] for r in front], "o-", color=color,
                        label=f"{factor:g}x ({len(group)}/{len(config['batch_sizes'])} batches)")
                for row in front:
                    ax.annotate(f"B{row['global_batch_size']}", (row[X], row[Y]),
                                xytext=(4, 4), textcoords="offset points", fontsize=7, color=color)
            scope = f" ({config['logic_scope']})" if parameter == "logic" else ""
            ax.set(title=f"{model} | {config['input_sequence_length']} tokens | {LABELS[parameter]}{scope}\n"
                   f"XFold | 8 chips | EP | FP8 KV | MTP{int(config['mtp_enabled'])} | PartitionChain optimized",
                   xlabel="tokens/s/user", ylabel="tokens/s/chip", xlim=(0, None), ylim=(0, None))
            ax.set_xscale("linear")
            ax.set_yscale("linear")
            ax.grid(alpha=0.2)
            ax.legend()
            target = directory / f"{model}-{parameter}-pareto.png"
            temporary = target.with_suffix(".tmp.png")
            fig.savefig(temporary, dpi=180)
            temporary.replace(target)
            plt.close(fig)
    return len(all_rows)


def source_files() -> list[Path]:
    """Collect actual source, including dirty/untracked experiment code and local dependencies."""
    paths = []
    for folder in ("srammachine", "exp/big", "exp/sensitivity", "local files/Desim", "local files/PerfTracer"):
        for path in (ROOT / folder).rglob("*"):
            if (path.is_file() and path.suffix in (".py", ".json", ".csv", ".md", ".toml", ".txt")
                    and not any(part in (".git", ".venv", "__pycache__") for part in path.parts)):
                paths.append(path)
    paths.extend(ROOT / name for name in ("AGENTS.md", "pyproject.toml", "uv.lock") if (ROOT / name).exists())
    return sorted(set(paths))


def source_digest() -> str:
    """Hash paths and bytes to reject a mixed-source resumed experiment."""
    digest = hashlib.sha256()
    for path in source_files():
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def snapshot(directory: Path, csv_path: Path) -> str:
    """Back up executable sources, original measurements and dependency metadata."""
    directory.mkdir()
    for path in Path(__file__).parent.iterdir():
        if path.is_file():
            shutil.copy2(path, directory / path.name)
    shutil.copy2(csv_path, directory / "hbf_read_bandwidth.csv")
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
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (directory / filename).write_text(result.stdout + result.stderr)
    return hashlib.sha256(archive_path.read_bytes()).hexdigest()


def main() -> int:
    """Validate, snapshot, run isolated cases and persist resumable reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", nargs="+", choices=PARAMETERS, default=list(PARAMETERS))
    parser.add_argument("--models", nargs="+", choices=["deepseek-v3.2", "minimax-m3"], default=["minimax-m3"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(BATCHES))
    parser.add_argument("--factors", nargs="+", type=float, choices=FACTORS, default=list(FACTORS))
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--attention-efficiency", type=float, default=1.0)
    parser.add_argument("--mtp", type=int, choices=(0, 1), default=0)
    parser.add_argument("--logic-scope", choices=["both", "vector", "pu"], default="both")
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--bandwidth-csv", type=Path, default=DEFAULT_HBF_BANDWIDTH_CSV)
    parser.add_argument("--output-root", type=Path, default=Path("test result/sensitivity"))
    parser.add_argument("--run-id", default=datetime.now().astimezone().strftime("%Y-%m-%d-%H%M%S"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reports-only", action="store_true")
    args = parser.parse_args()
    if args.sequence_length <= 0 or any(b <= 0 for b in args.batch_sizes):
        parser.error("sequence length and batch sizes must be positive")
    if any(len(set(values)) != len(values) for values in (args.models, args.batch_sizes, args.factors)):
        parser.error("duplicate sweep values are not allowed")
    if not args.run_id or args.run_id in (".", "..") or Path(args.run_id).name != args.run_id:
        parser.error("run-id must be a directory name")
    # Fail early if plots or measurement lookups cannot be produced.
    import matplotlib
    table = HBFBandwidthTable.from_csv(args.bandwidth_csv)
    for model in args.models:
        for batch in args.batch_sizes:
            for kind in ("sparse_kv", "dense_indexer"):
                table.lookup(model, kind, bool(args.mtp), batch, args.sequence_length)
    config = {
        "models": args.models, "parameters": args.parameters, "factors": args.factors,
        "batch_sizes": args.batch_sizes, "input_sequence_length": args.sequence_length,
        "logic_scope": args.logic_scope, "baseline_hardware": plain(HardwareConfig()),
        "attention_efficiency": args.attention_efficiency,
        "sensitivity_revision": 3, "xfold": True, "mtp_enabled": bool(args.mtp),
        "accepted_tokens_per_step": 1 + args.mtp, "output_sequence_length": 600,
        "moe_strategy": "ep", "kv_cache_dtype": "fp8", "simulated_layer_count": 4,
        "partition_chain_search": True, "split_tree_search": False,
        "bandwidth_csv_sha256": hashlib.sha256(args.bandwidth_csv.read_bytes()).hexdigest(),
        "source_digest": source_digest(),
        "throughput_definition": f"user = {1 + args.mtp}e9 / model_step_latency_ns; chip = user * global_batch / 8",
        "latency_definition": "weighted representative-layer core-start intervals; M3 = 57 sparse + 3 dense; decode only",
        "hbf_definition": "CSV per-op rates and hardware fallback Flash bandwidth multiplied together",
        "hbm_definition": "PU DRAM ports and logic-die DRAM bandwidth multiplied together",
        "noc_definition": "NoC links scale normally; pooled supply = local half + remote half * min(1,factor); KV prefetch fully remote; inter-chip fabric fixed",
        "indexer_noc_bandwidth": True,
        "remote_supply_cap": "pooled-half-local-v1",
        "indexer_bandwidth_definition": "HBF pooled rate + 1x Indexer HBM contribution * min(1,NoC factor)",
    }
    directory = args.output_root.resolve() / f"{args.run_id}-sensitivity"
    if directory.exists():
        if not (args.resume or args.reports_only):
            parser.error(f"Already exists: {directory}; use --resume")
        saved = json.loads((directory / "manifest.json").read_text())
        differences = [key for key in config if saved.get(key) != config[key]
                       and not (args.reports_only and key == "source_digest")]
        if differences:
            parser.error(f"Resume configuration/source differs: {differences}")
        for filename, key in (("source.zip", "source_zip_sha256"),
                              ("hbf_read_bandwidth.csv", "bandwidth_csv_sha256")):
            if hashlib.sha256((directory / "scripts" / filename).read_bytes()).hexdigest() != saved[key]:
                parser.error(f"Backup checksum differs: {filename}")
    else:
        if args.reports_only:
            parser.error(f"No results: {directory}")
        directory.mkdir(parents=True)
        source_hash = snapshot(directory / "scripts", args.bandwidth_csv)
        write_json(directory / "manifest.json", dict(config, source_zip_sha256=source_hash,
            generated_at=datetime.now().astimezone().isoformat(), command=sys.argv, python=sys.version))
    tasks = [{"model": model, "parameter": parameter, "factor": factor, "batch": batch,
              "attention_efficiency": args.attention_efficiency,
              "sequence": args.sequence_length, "logic_scope": args.logic_scope, "mtp_enabled": bool(args.mtp),
              "directory": str(directory / model / parameter),
              "bandwidth_csv": str(directory / "scripts/hbf_read_bandwidth.csv")}
             for model in args.models for parameter in args.parameters
             for factor in args.factors for batch in args.batch_sizes]
    if args.reports_only:
        completed = write_reports(directory, config, tasks)
        print(f"Reports refreshed: {completed}/{len(tasks)} points; {directory}")
        return 0
    pending = [task for task in tasks if completed_row(task) is None]
    print(f"Starting {len(pending)}/{len(tasks)} points, {args.workers} workers; {directory}", flush=True)
    failures = []
    last_report = time.monotonic()
    context = multiprocessing.get_context("spawn")
    with context.Pool(args.workers, maxtasksperchild=1) as pool:
        for count, row in enumerate(pool.imap_unordered(run_case, pending, chunksize=1), 1):
            print(f"[{count}/{len(pending)}] {row['parameter']} x{row['factor']:g} "
                  f"B{row['global_batch_size']} {row['status']} "
                  f"user={row.get(X, 0):.3f} chip={row.get(Y, 0):.3f}", flush=True)
            if row["status"] != "complete":
                failures.append(row)
            if time.monotonic() - last_report > 60:
                write_reports(directory, config, tasks)
                last_report = time.monotonic()
    completed = write_reports(directory, config, tasks)
    write_json(directory / "completion.json", {
        "status": "complete" if completed == len(tasks) else "incomplete",
        "completed_points": completed, "expected_points": len(tasks), "failures": failures,
        "finished_at": datetime.now().astimezone().isoformat(),
    })
    print(f"Finished: {completed}/{len(tasks)} points; {directory}", flush=True)
    return 0 if completed == len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
