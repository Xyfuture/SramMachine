"""Run PartitionChain-optimized sequential workloads in isolated processes.

Invoke from the SramMachine project directory, for example::

    python -m srammachine.script.run_sa \
      --models deepseek-v3 deepseek-v3.2 \
      --mtp off on --batch-sizes 256 512 \
      --input-sequence-length 20000 --output-sequence-length 600 \
      --moe-strategy tp ep

Desim owns process-global simulation state.  Each workload therefore runs in
its own process; using threads here would let concurrent cases reset each
other's ``SimSession``.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys
import tempfile
import traceback
from typing import Any, Callable, Iterable, Sequence

from Desim.Utils import UniquePriorityQueue

from srammachine.inference import InferenceConfig, MoEParallelStrategy
from srammachine.mapping import (
    DEFAULT_HBF_BANDWIDTH_CSV, HBFBandwidthTable, HardwareMappingRequest, mapper_for_model,
)
from srammachine.pipetree import TreeParser
from srammachine.search import (
    SimulatedAnnealingConfig,
    SplitTreeOptimizer,
    split_tree_from_dict,
    split_tree_to_dict,
    apply_partition_chains,
    partition_chains_to_dict,
    derive_workload_seed,
)
from srammachine.simulator import Simulator, steady_pu_metrics


SUPPORTED_MODELS = (
    "deepseek-v3", "deepseek-v3.2", "kimi-k2.5", "glm-5.1", "minimax-m3",
)
DEFAULT_SEED = SimulatedAnnealingConfig().random_seed
DEFAULT_OUTPUT_DIR = Path("best split tree result")
CSV_OMITTED_FIELDS = frozenset(("best_split_trees", "restart_results", "partition_chains"))
CSV_FIELDNAMES = (
    "model", "xfold", "mtp_enabled", "moe_strategy", "global_batch_size",
    "input_sequence_length", "output_sequence_length", "kv_cache_dtype",
    "pareto_best_total_throughput_tokens_per_second",
    "pareto_best_single_user_throughput_per_second",
    "steady_pu_utilization_percent",
    "baseline_latency_ns", "pareto_best_latency_ns",
    "baseline_single_user_throughput_per_second",
    "baseline_total_throughput_tokens_per_second",
    "latency_reduction_fraction", "throughput_improvement_fraction",
    "warmup_rounds", "rounds", "restart_count", "initial_temperature",
    "final_temperature", "layer_count", "base_seed", "chip_count",
    "case_seed",
    "proposal_count", "candidate_simulation_count",
    "materialization_simulation_count", "simulator_run_count",
    "cache_hit_count", "accepted_proposal_count",
    "is_model_global_pareto",
    "best_split_tree_path", "selected_split_tree_index", "trace_path",
    "hbf_bandwidth_csv",
    "optimize_partition_chains",
    "split_tree_search_enabled",
)


@dataclass(frozen=True)
class SACase:
    """Serializable input for one independent worker process."""

    model: str
    global_batch_size: int
    mtp_enabled: bool
    input_sequence_length: int
    output_sequence_length: int
    moe_strategy: str
    kv_cache_dtype: str
    warmup_rounds: int
    rounds: int
    restart_count: int
    initial_temperature: float
    final_temperature: float
    layer_count: int
    base_seed: int
    optimize_partition_chains: bool = True
    hbf_bandwidth_csv: str | None = str(DEFAULT_HBF_BANDWIDTH_CSV)
    xfold: bool = False
    artifact_dir: str | None = None
    case_checkpoint_dir: str | None = None
    run_stamp: str | None = None
    generated_at: str | None = None
    detected_cpus: int | None = None
    worker_count: int | None = None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate PartitionChains and simulate each sequential "
            "(model, MoE strategy, batch, MTP) case. SplitTree search is disabled."
        ),
    )
    parser.add_argument("--models", nargs="+", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--mtp", nargs="+", required=True, choices=("off", "on"))
    parser.add_argument(
        "--batch-sizes", nargs="+", required=True, type=_positive_int,
    )
    parser.add_argument("--split-tree-search", action="store_true",
                        help="unsupported: requesting SplitTree search raises NotImplementedError")
    parser.add_argument("--rounds", type=_nonnegative_int, default=0,
                        help="legacy SA option; nonzero values raise NotImplementedError")
    parser.add_argument(
        "--restarts", type=_nonnegative_int, default=0,
        help="legacy SA option; nonzero values raise NotImplementedError",
    )
    parser.add_argument(
        "--input-sequence-length", nargs="+", required=True, type=_positive_int,
    )
    parser.add_argument(
        "--output-sequence-length", required=True, type=_positive_int,
    )
    parser.add_argument(
        "--moe-strategy", nargs="+", default=["ep"], choices=("tp", "ep"),
        help="chip-level MoE parallel strategy (default: ep)",
    )
    parser.add_argument(
        "--kv-cache-dtype", choices=("fp8", "fp16"), default="fp8",
    )
    parser.add_argument("--xfold", action="store_true",
                        help="use the DSA or MiniMax MSA folding MoE path; DeepSeek V3 runs ordinary MLA")
    parser.add_argument(
        "--partition-optimization", dest="optimize_partition_chains",
        action=argparse.BooleanOptionalAction, default=True,
        help="enumerate MLA/DSA/MiniMax per-op PartitionChains before simulation (default: enabled)",
    )
    hbf = parser.add_mutually_exclusive_group()
    hbf.add_argument("--hbf-bandwidth-csv", type=Path, default=DEFAULT_HBF_BANDWIDTH_CSV,
                     help="normalized HBF measurements (default: bundled table)")
    hbf.add_argument("--no-hbf-bandwidth", dest="hbf_bandwidth_csv", action="store_const",
                     const=None, help="disable measured HBF bandwidths")
    parser.add_argument("--warmup-rounds", type=_nonnegative_int, default=0,
                        help="legacy SA option; nonzero values raise NotImplementedError")
    parser.add_argument("--initial-temperature", type=_positive_float, default=1.0)
    parser.add_argument("--final-temperature", type=_positive_float, default=0.01)
    parser.add_argument("--layer-count", type=_positive_int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--workers", type=_positive_int,
        help="worker processes; default uses every available logical CPU",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-csv", type=Path,
        help="exact CSV path; existing files are never overwritten",
    )
    return parser


def _validate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace,
) -> argparse.Namespace:
    if args.split_tree_search or args.rounds or args.restarts or args.warmup_rounds:
        raise NotImplementedError("SplitTree search is disabled; omit SA options and use PartitionChain optimization.")
    if args.xfold and any(model not in ("deepseek-v3", "deepseek-v3.2", "glm-5.1", "minimax-m3") for model in args.models):
        parser.error("--xfold supports deepseek-v3 (ordinary), deepseek-v3.2, glm-5.1 and minimax-m3")
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicates")
    if len(set(args.mtp)) != len(args.mtp):
        parser.error("--mtp must not contain duplicates")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("--batch-sizes must not contain duplicates")
    if len(set(args.input_sequence_length)) != len(args.input_sequence_length):
        parser.error("--input-sequence-length must not contain duplicates")
    if len(set(args.moe_strategy)) != len(args.moe_strategy):
        parser.error("--moe-strategy must not contain duplicates")
    if args.final_temperature > args.initial_temperature:
        parser.error("--final-temperature must not exceed --initial-temperature")
    if args.layer_count < 3:
        parser.error("--layer-count must be at least 3 for steady-state PU utilization")
    if args.output_csv is not None and args.output_csv.exists():
        parser.error(f"--output-csv already exists: {args.output_csv}")
    if args.hbf_bandwidth_csv is not None:
        args.hbf_bandwidth_csv = args.hbf_bandwidth_csv.resolve()
        try:
            HBFBandwidthTable.from_csv(args.hbf_bandwidth_csv)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    # Submit the heaviest workloads first.  There is no barrier between batch
    # sizes: each replacement worker immediately takes the next queued case.
    args.batch_sizes = sorted(args.batch_sizes, reverse=True)
    args.input_sequence_length = sorted(args.input_sequence_length, reverse=True)
    args.mtp = sorted(args.mtp, key=lambda value: value == "on")
    args.moe_strategy = sorted(
        args.moe_strategy, key=lambda value: ("tp", "ep").index(value),
    )
    return args


def _make_cases(args: argparse.Namespace) -> list[SACase]:
    # Keep batch as the outermost dimension so the process-pool queue is
    # globally ordered from the heaviest batch to the lightest one.  If model
    # were outermost, a BS32 case from the first model could start before a
    # BS4096 case from the next model even though batch_sizes is descending.
    return [
        SACase(
            model=model,
            xfold=args.xfold,
            optimize_partition_chains=args.optimize_partition_chains,
            global_batch_size=batch,
            mtp_enabled=mtp == "on",
            input_sequence_length=input_sequence_length,
            output_sequence_length=args.output_sequence_length,
            moe_strategy=strategy,
            kv_cache_dtype=args.kv_cache_dtype,
            warmup_rounds=args.warmup_rounds,
            rounds=args.rounds,
            restart_count=args.restarts,
            initial_temperature=args.initial_temperature,
            final_temperature=args.final_temperature,
            layer_count=args.layer_count,
            base_seed=args.seed,
            hbf_bandwidth_csv=(str(args.hbf_bandwidth_csv) if args.hbf_bandwidth_csv else None),
        )
        for batch in args.batch_sizes
        for input_sequence_length in args.input_sequence_length
        for model in args.models
        for strategy in args.moe_strategy
        for mtp in args.mtp
    ]


def choose_worker_count(
    case_count: int, requested: int | None = None,
    *, logical_cpu_count: int | None = None, platform: str | None = None,
) -> int:
    """Choose safe maximum process parallelism for the current platform."""
    if type(case_count) is not int or case_count <= 0:
        raise ValueError("case_count must be a positive integer")
    if requested is not None and (type(requested) is not int or requested <= 0):
        raise ValueError("requested workers must be a positive integer")
    detected = os.cpu_count() if logical_cpu_count is None else logical_cpu_count
    if detected is None or detected <= 0:
        detected = 1
    limit = requested if requested is not None else detected
    if (sys.platform if platform is None else platform).startswith("win"):
        limit = min(limit, 61)
    return min(case_count, limit)


@contextmanager
def _without_desim_priority_queue_validation():
    """Skip Desim's quadratic debug scan inside one isolated SA worker.

    ``UniquePriorityQueue`` already maintains a ``SortedList`` and a set on
    every mutation.  Its debug-only ``valid_check`` walks both containers and
    invokes ``SortedList._check`` before and after every queue operation.  That
    validates the library implementation rather than changing scheduling.
    Keep the patch scoped so ordinary Simulator users and the test suite retain
    all Desim diagnostics.
    """
    original = UniquePriorityQueue.valid_check

    def _already_valid(_queue) -> bool:
        return True

    UniquePriorityQueue.valid_check = _already_valid
    try:
        yield
    finally:
        UniquePriorityQueue.valid_check = original


def _run_case(case: SACase) -> dict[str, Any]:
    with _without_desim_priority_queue_validation():
        return _run_case_impl(case)


def _run_case_impl(case: SACase) -> dict[str, Any]:
    if case.rounds or case.restart_count or case.warmup_rounds:
        raise NotImplementedError("SplitTree search is disabled; all SA round/restart counts must be zero.")
    strategy = MoEParallelStrategy(case.moe_strategy)
    request = HardwareMappingRequest(case.model, InferenceConfig(
        global_batch_size=case.global_batch_size,
        input_sequence_length=case.input_sequence_length,
        output_sequence_length=case.output_sequence_length,
        moe_parallel_strategy=strategy,
        kv_cache_dtype=case.kv_cache_dtype,
        mtp_enabled=case.mtp_enabled,
    ), xfold=case.xfold)
    optimizer = SplitTreeOptimizer(hbf_bandwidth_csv=case.hbf_bandwidth_csv)
    direct_config = SimulatedAnnealingConfig(
        enabled=False, layer_count=case.layer_count, write_result=False,
        optimize_partition_chains=False,
    )
    baseline = optimizer.run(request, config=direct_config).initial_evaluation
    best = baseline
    if case.optimize_partition_chains:
        best = optimizer.run(request, config=replace(
            direct_config, optimize_partition_chains=True,
        )).initial_evaluation
    simulation_count = 2 if case.optimize_partition_chains else 1
    case_fields = asdict(case)
    for internal in (
        "artifact_dir", "case_checkpoint_dir", "run_stamp",
        "generated_at", "detected_cpus", "worker_count",
    ):
        case_fields.pop(internal)
    row = {
        **case_fields,
        "xfold": best.mapping_result.request.xfold,
        "chip_count": optimizer.hardware_config.chip_count,
        "partition_chains": partition_chains_to_dict(best.mapping_result),
        "split_tree_search_enabled": False,
        "case_seed": derive_workload_seed(case.base_seed, case.model, case.moe_strategy,
                                          case.global_batch_size, case.mtp_enabled),
        "restart_count": 0,
        "restart_results": [],
        "baseline_latency_ns": baseline.latency_ns,
        "pareto_best_latency_ns": best.latency_ns,
        "baseline_single_user_throughput_per_second": (
            baseline.single_user_throughput_per_second
        ),
        "pareto_best_single_user_throughput_per_second": (
            best.single_user_throughput_per_second
        ),
        "baseline_total_throughput_tokens_per_second": (
            baseline.total_throughput_tokens_per_second
        ),
        "pareto_best_total_throughput_tokens_per_second": (
            best.total_throughput_tokens_per_second
        ),
        "latency_reduction_fraction": 1.0 - best.latency_ns / baseline.latency_ns,
        "throughput_improvement_fraction": baseline.latency_ns / best.latency_ns - 1.0,
        "proposal_count": 0,
        "candidate_simulation_count": simulation_count,
        "materialization_simulation_count": 0,
        "simulator_run_count": simulation_count,
        "cache_hit_count": 0,
        "accepted_proposal_count": 0,
        "best_split_trees": [split_tree_to_dict(best.split_tree)],
    }
    if case.artifact_dir is not None:
        _persist_case_artifacts(row, case)
    return row


def _run_case_envelope(
    case: SACase,
) -> tuple[SACase, dict[str, Any] | None, str | None]:
    """Return a serializable result while retaining failing case context."""
    try:
        return case, _run_case(case), None
    except BaseException:
        return case, None, traceback.format_exc()


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_single = left["pareto_best_single_user_throughput_per_second"]
    left_total = left["pareto_best_total_throughput_tokens_per_second"]
    right_single = right["pareto_best_single_user_throughput_per_second"]
    right_total = right["pareto_best_total_throughput_tokens_per_second"]
    return (
        left_single >= right_single
        and left_total >= right_total
        and (left_single > right_single or left_total > right_total)
    )


def mark_model_pareto(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark a raw Pareto front within each fixed inference/hardware setup."""
    group_fields = (
        "model", "moe_strategy", "input_sequence_length",
        "output_sequence_length", "kv_cache_dtype", "layer_count",
        "chip_count",
    )
    marked = []
    for row in rows:
        group = tuple(row[field] for field in group_fields)
        peers = [
            item for item in rows
            if tuple(item[field] for field in group_fields) == group
        ]
        copied = dict(row)
        copied["is_model_global_pareto"] = not any(
            _dominates(other, row) for other in peers if other is not row
        )
        marked.append(copied)
    return marked


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    candidate = directory / f"{stem}{suffix}"
    index = 2
    while candidate.exists():
        candidate = directory / f"{stem}_{index:02d}{suffix}"
        index += 1
    return candidate


def _csv_rows(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        unexpected = set(row) - CSV_OMITTED_FIELDS - set(CSV_FIELDNAMES)
        if unexpected:
            raise ValueError(f"unexpected CSV fields: {sorted(unexpected)}")
        yield {field: row.get(field, "") for field in CSV_FIELDNAMES}


def _write_csv_rows(stream, rows: Iterable[dict[str, Any]]) -> None:
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    writer.writerows(_csv_rows(rows))


def create_csv_checkpoint(path: Path) -> None:
    """Create an empty, valid CSV before any worker process is launched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        _write_csv_rows(stream, ())
        stream.flush()
        os.fsync(stream.fileno())


def append_csv_checkpoint(path: Path, row: dict[str, Any]) -> None:
    """Durably append one completed case; its global Pareto status is unknown."""
    checkpoint_row = dict(row)
    checkpoint_row.pop("is_model_global_pareto", None)
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDNAMES)
        writer.writerow(next(_csv_rows((checkpoint_row,))))
        stream.flush()
        os.fsync(stream.fileno())


def finalize_csv_checkpoint(
    path: Path, rows: Sequence[dict[str, Any]],
) -> None:
    """Atomically replace a checkpoint with the sorted, finalized result."""
    if not rows:
        raise ValueError("cannot finalize an empty SA comparison")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="", delete=False,
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        ) as stream:
            temporary_path = Path(stream.name)
            _write_csv_rows(stream, rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty SA comparison")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        _write_csv_rows(stream, rows)


def write_model_json(
    path: Path, model: str, strategy: str, mtp_enabled: bool,
    rows: Sequence[dict[str, Any]],
    *, generated_at: str, detected_cpus: int, worker_count: int,
    input_sequence_length: int | None = None, overwrite: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    workloads = [
        dict(row) for row in rows
        if (
            row["model"] == model
            and row["moe_strategy"] == strategy
            and row["mtp_enabled"] == mtp_enabled
            and (
                input_sequence_length is None
                or row["input_sequence_length"] == input_sequence_length
            )
        )
    ]
    if not workloads:
        raise ValueError("model/strategy/MTP/ISL has no workload rows")
    if len({row["input_sequence_length"] for row in workloads}) != 1:
        raise ValueError("model JSON requires one input sequence length")
    def tree_records(row: dict[str, Any]) -> list[dict[str, Any]]:
        """Label every tree with the workload that produced it.

        A SplitTree only describes operator grouping/splitting, so its structural
        JSON cannot by itself say whether it was searched with MTP enabled.  Keep
        the legacy raw-tree list for replay compatibility, and provide these
        self-describing records for inspection and downstream analysis.
        """
        return [{
            "global_batch_size": row["global_batch_size"],
            "mtp_enabled": row["mtp_enabled"],
            "split_tree": tree,
            "partition_chains": row.get("partition_chains", {}),
        } for tree in row["best_split_trees"]]

    front = [{
        "global_batch_size": row["global_batch_size"],
        "mtp_enabled": row["mtp_enabled"],
        "single_user_throughput_per_second": (
            row["pareto_best_single_user_throughput_per_second"]
        ),
        "total_throughput_tokens_per_second": (
            row["pareto_best_total_throughput_tokens_per_second"]
        ),
        "latency_ns": row["pareto_best_latency_ns"],
        "case_seed": row["case_seed"],
        "split_trees": row["best_split_trees"],
        "split_tree_records": tree_records(row),
        "partition_chains": row.get("partition_chains", {}),
    } for row in workloads if row["is_model_global_pareto"]]
    best_by_workload = [{
        "global_batch_size": row["global_batch_size"],
        "mtp_enabled": row["mtp_enabled"],
        "latency_ns": row["pareto_best_latency_ns"],
        "single_user_throughput_per_second": (
            row["pareto_best_single_user_throughput_per_second"]
        ),
        "total_throughput_tokens_per_second": (
            row["pareto_best_total_throughput_tokens_per_second"]
        ),
        "split_trees": row["best_split_trees"],
        "split_tree_records": tree_records(row),
        "partition_chains": row.get("partition_chains", {}),
    } for row in workloads]
    first = workloads[0]
    payload = {
        "format_version": 1,
        "generated_at": generated_at,
        "model": model,
        "hardware": {"chip_count": first["chip_count"]},
        "xfold": first.get("xfold", False),
        "hbf_bandwidth_csv": first.get("hbf_bandwidth_csv"),
        "partition_optimization_enabled": first.get("optimize_partition_chains", False),
        "split_tree_search_enabled": False,
        "search": {
            "enabled": False,
            "moe_parallel_strategy": first["moe_strategy"],
            "mtp_enabled": mtp_enabled,
            "input_sequence_length": first["input_sequence_length"],
            "output_sequence_length": first["output_sequence_length"],
            "kv_cache_dtype": first["kv_cache_dtype"],
            "warmup_rounds_per_workload": first["warmup_rounds"],
            "rounds_per_workload": first["rounds"],
            "restart_count_per_workload": first["restart_count"],
            "initial_temperature": first["initial_temperature"],
            "final_temperature": first["final_temperature"],
            "layer_count": first["layer_count"],
            "base_seed": first["base_seed"],
            "detected_logical_cpu_count": detected_cpus,
            "worker_process_count": worker_count,
        },
        "workloads": workloads,
        # Unlike pareto_front, this contains one entry for every batch
        # workload for this fixed MTP state, including dominated points.
        "best_split_trees_by_workload": best_by_workload,
        "pareto_front": front,
    }
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False,
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_case_trace(
    row: dict[str, Any], output_dir: Path,
) -> tuple[Path, float]:
    """Replay one selected best tree and write its Perfetto trace."""
    best_trees = row.get("best_split_trees")
    if not best_trees:
        raise ValueError("SA row has no best SplitTree")
    tree_index = row.get("selected_split_tree_index", 0)
    if type(tree_index) is not int or not 0 <= tree_index < len(best_trees):
        raise ValueError("selected SplitTree index is out of range")

    request = HardwareMappingRequest(
        row["model"],
        InferenceConfig(
            global_batch_size=row["global_batch_size"],
            input_sequence_length=row["input_sequence_length"],
            output_sequence_length=row["output_sequence_length"],
            moe_parallel_strategy=MoEParallelStrategy(row["moe_strategy"]),
            kv_cache_dtype=row["kv_cache_dtype"],
            mtp_enabled=row["mtp_enabled"],
        ),
    )
    request = replace(request, xfold=row.get("xfold", False))
    mapper = mapper_for_model(request.model_name)
    table_path = row.get("hbf_bandwidth_csv")
    table = HBFBandwidthTable.from_csv(table_path) if table_path else None
    mapping = mapper.map(request, hbf_bandwidth_table=table)
    if "partition_chains" in row:
        mapping = apply_partition_chains(mapping, row["partition_chains"], mapper.hardware_config)
    elif row.get("optimize_partition_chains", False):
        raise ValueError("optimized case is missing its saved PartitionChains")
    tree = split_tree_from_dict(best_trees[tree_index])
    graph = TreeParser(mapper.hardware_config).parse(
        tree,
        mapping.operators,
        mapping.operator_mappings,
        layer_count=row["layer_count"],
    )
    trace_label = (
        f"{_safe_name(row['model'])}_{row['moe_strategy']}_"
        f"{'fold_' if row.get('xfold') else ''}mtp_{'on' if row['mtp_enabled'] else 'off'}_"
        f"bs{row['global_batch_size']}_isl{row['input_sequence_length']}"
    )
    artifacts = Simulator(mapper.hardware_config).run_and_trace(
        graph,
        mapping_result=mapping,
        trace_label=trace_label,
        output_dir=output_dir,
    )
    if artifacts.simulation_result.latency_ns != row["pareto_best_latency_ns"]:
        raise RuntimeError("trace replay latency does not match the searched result")
    metrics = steady_pu_metrics(artifacts.simulation_result.command_results)
    return artifacts.trace_path, metrics["pu_utilization_percent"]


def _persist_case_artifacts(row: dict[str, Any], case: SACase) -> None:
    """Atomically persist one completed case before its worker returns."""
    required = (
        case.artifact_dir, case.case_checkpoint_dir, case.run_stamp,
        case.generated_at, case.detected_cpus, case.worker_count,
    )
    if any(value is None for value in required):
        raise ValueError("case artifact context is incomplete")
    checkpoint_dir = Path(case.case_checkpoint_dir)
    case_stem = (
        f"{case.run_stamp}_{_safe_name(row['model'])}_{row['moe_strategy']}_"
        f"{'fold_' if row.get('xfold') else ''}mtp_{'on' if row['mtp_enabled'] else 'off'}_"
        f"bs{row['global_batch_size']}_isl{row['input_sequence_length']}_"
        "sa_case_checkpoint"
    )
    split_tree_path = _unique_path(checkpoint_dir, case_stem, ".json")
    row["best_split_tree_path"] = str(split_tree_path)
    row["selected_split_tree_index"] = 0
    write_model_json(
        split_tree_path, row["model"], row["moe_strategy"],
        row["mtp_enabled"], [dict(row, is_model_global_pareto=True)],
        generated_at=case.generated_at,
        detected_cpus=case.detected_cpus,
        worker_count=case.worker_count,
        input_sequence_length=row["input_sequence_length"],
    )
    trace_path, pu_utilization = write_case_trace(
        row, Path(case.artifact_dir),
    )
    row["steady_pu_utilization_percent"] = pu_utilization
    row["trace_path"] = str(trace_path)
    # Refresh the checkpoint atomically with the completed trace metadata.
    # The first version remains intact if trace replay is interrupted.
    write_model_json(
        split_tree_path, row["model"], row["moe_strategy"],
        row["mtp_enabled"], [dict(row, is_model_global_pareto=True)],
        generated_at=case.generated_at,
        detected_cpus=case.detected_cpus,
        worker_count=case.worker_count,
        input_sequence_length=row["input_sequence_length"],
        overwrite=True,
    )
    print(f"Case SplitTree checkpoint: {split_tree_path}", flush=True)


def _execute(
    cases: Sequence[SACase], worker_count: int,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    results = []
    context = multiprocessing.get_context("spawn")
    # One process handles exactly one case.  Besides isolating Desim's global
    # session, this makes the OS reclaim greenlet stacks and native allocator
    # arenas before a later case starts (especially important on Windows).
    pool = context.Pool(processes=worker_count, maxtasksperchild=1)
    try:
        completed = pool.imap_unordered(_run_case_envelope, cases, chunksize=1)
        for case, result, error_details in completed:
            if error_details is not None:
                raise RuntimeError(
                    "SA case failed: "
                    f"model={case.model}, strategy={case.moe_strategy}, "
                    f"batch={case.global_batch_size}, "
                    f"mtp={'on' if case.mtp_enabled else 'off'}\n"
                    f"{error_details}"
                )
            assert result is not None
            if on_result is not None:
                on_result(result)
            results.append(result)
            print(
                "completed "
                f"model={case.model} strategy={case.moe_strategy} "
                f"batch={case.global_batch_size} "
                f"isl={case.input_sequence_length} "
                f"mtp={'on' if case.mtp_enabled else 'off'} "
                "best="
                f"{result['pareto_best_total_throughput_tokens_per_second']:.3f} "
                "tokens/s",
                flush=True,
            )
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = _validate_args(parser, parser.parse_args(argv))
    cases = _make_cases(args)
    detected = os.cpu_count() or 1
    workers = choose_worker_count(
        len(cases), args.workers, logical_cpu_count=detected,
    )
    output_dir = args.output_dir
    timestamp = datetime.now().astimezone()
    minute = timestamp.strftime("%Y%m%d_%H%M")
    generated_at = timestamp.isoformat()
    csv_path = args.output_csv
    if csv_path is None:
        csv_path = _unique_path(
            output_dir, f"{minute}_sa_throughput_comparison", ".csv",
        )
    create_csv_checkpoint(csv_path)
    # Give every worker a private artifact context. It persists one reloadable
    # SplitTree and trace before returning its completed result to this process.
    artifact_dir = csv_path.parent
    artifact_dir.mkdir(parents=True, exist_ok=True)
    case_checkpoint_dir = (
        artifact_dir / "case checkpoints" / csv_path.stem
    )
    case_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        replace(
            case, artifact_dir=str(artifact_dir),
            case_checkpoint_dir=str(case_checkpoint_dir),
            run_stamp=minute, generated_at=generated_at,
            detected_cpus=detected, worker_count=workers,
        )
        for case in cases
    ]
    print(
        f"detected {detected} logical CPUs; running {len(cases)} cases "
        f"with {workers} worker processes",
        flush=True,
    )
    print(f"CSV checkpoint: {csv_path}", flush=True)

    saved_count = 0
    checkpoint_rows = []

    def save_completed_case(row: dict[str, Any]) -> None:
        nonlocal saved_count
        append_csv_checkpoint(csv_path, row)
        saved_count += 1
        checkpoint_rows.append(row)
        finalize_csv_checkpoint(csv_path, checkpoint_rows)

    try:
        results = _execute(cases, workers, on_result=save_completed_case)
    except BaseException:
        print(
            f"partial CSV retained: {csv_path} "
            f"({saved_count}/{len(cases)} completed cases)",
            file=sys.stderr, flush=True,
        )
        raise
    model_rank = {model: index for index, model in enumerate(args.models)}
    strategy_rank = {
        strategy: index for index, strategy in enumerate(args.moe_strategy)
    }
    results.sort(key=lambda row: (
        -row["global_batch_size"], -row["input_sequence_length"],
        model_rank[row["model"]], strategy_rank[row["moe_strategy"]],
        row["mtp_enabled"],
    ))
    results = mark_model_pareto(results)

    json_specs = []
    for model in args.models:
        for strategy in args.moe_strategy:
            for mtp_enabled in (False, True):
                for input_sequence_length in args.input_sequence_length:
                    if not any(
                        row["model"] == model
                        and row["moe_strategy"] == strategy
                        and row["mtp_enabled"] == mtp_enabled
                        and row["input_sequence_length"] == input_sequence_length
                        for row in results
                    ):
                        continue
                    path = _unique_path(
                        artifact_dir,
                        (
                            f"{minute}_{_safe_name(model)}_{strategy}_"
                            f"mtp_{'on' if mtp_enabled else 'off'}_"
                            f"isl{input_sequence_length}_sa_best_splittrees"
                        ),
                        ".json",
                    )
                    for row in results:
                        if (
                            row["model"] == model
                            and row["moe_strategy"] == strategy
                            and row["mtp_enabled"] == mtp_enabled
                            and row["input_sequence_length"] == input_sequence_length
                        ):
                            row["best_split_tree_path"] = str(path)
                            row["selected_split_tree_index"] = 0
                    json_specs.append((
                        path, model, strategy, mtp_enabled,
                        input_sequence_length,
                    ))

    json_paths = []
    for path, model, strategy, mtp_enabled, input_sequence_length in json_specs:
        write_model_json(
            path, model, strategy, mtp_enabled, results,
            generated_at=generated_at, detected_cpus=detected,
            worker_count=workers,
            input_sequence_length=input_sequence_length,
        )
        json_paths.append(path)

    finalize_csv_checkpoint(csv_path, results)

    print(f"CSV: {csv_path}")
    for path in json_paths:
        print(f"SplitTree JSON: {path}")
    print(f"Case checkpoints: {case_checkpoint_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
