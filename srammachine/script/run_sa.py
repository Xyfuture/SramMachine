"""Run independent SplitTree SA workloads in isolated worker processes.

Invoke from the SramMachine project directory, for example::

    python -m srammachine.script.run_sa \
      --models deepseek-v3 deepseek-v3.2 \
      --mtp off on --batch-sizes 256 512 --rounds 50 \
      --input-sequence-length 20000 --output-sequence-length 600 \
      --moe-strategy tp ep

Desim owns process-global simulation state.  Each workload therefore runs in
its own process; using threads here would let concurrent cases reset each
other's ``SimSession``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
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

from srammachine.hardware import DEFAULT_HARDWARE_CONFIG
from srammachine.inference import InferenceConfig, MoEParallelStrategy
from srammachine.mapping import HardwareMappingRequest
from srammachine.search import (
    SimulatedAnnealingConfig,
    SplitTreeOptimizer,
    split_tree_to_dict,
)


SUPPORTED_MODELS = (
    "deepseek-v3", "deepseek-v3.2", "kimi-k2.5", "glm-5.1",
)
DEFAULT_SEED = SimulatedAnnealingConfig().random_seed
DEFAULT_OUTPUT_DIR = Path("best split tree result")
CSV_OMITTED_FIELDS = frozenset(("best_split_trees", "restart_results"))
CSV_FIELDNAMES = (
    "model", "global_batch_size", "mtp_enabled", "input_sequence_length",
    "output_sequence_length", "moe_strategy", "kv_cache_dtype",
    "warmup_rounds", "rounds", "restart_count", "initial_temperature",
    "final_temperature", "layer_count", "base_seed", "chip_count",
    "case_seed", "baseline_latency_ns", "pareto_best_latency_ns",
    "baseline_single_user_throughput_per_second",
    "pareto_best_single_user_throughput_per_second",
    "baseline_total_throughput_tokens_per_second",
    "pareto_best_total_throughput_tokens_per_second",
    "latency_reduction_fraction", "throughput_improvement_fraction",
    "proposal_count", "candidate_simulation_count",
    "materialization_simulation_count", "simulator_run_count",
    "cache_hit_count", "accepted_proposal_count",
    "is_model_global_pareto",
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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one independent SplitTree simulated-annealing search per "
            "(model, MoE strategy, batch, MTP) case and export throughput "
            "comparisons."
        ),
    )
    parser.add_argument("--models", nargs="+", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--mtp", nargs="+", required=True, choices=("off", "on"))
    parser.add_argument(
        "--batch-sizes", nargs="+", required=True, type=_positive_int,
    )
    parser.add_argument("--rounds", required=True, type=_positive_int)
    parser.add_argument(
        "--restarts", type=_positive_int, default=4,
        help="full deterministic restarts per workload (default: 4)",
    )
    parser.add_argument(
        "--input-sequence-length", required=True, type=_positive_int,
    )
    parser.add_argument(
        "--output-sequence-length", required=True, type=_positive_int,
    )
    parser.add_argument(
        "--moe-strategy", nargs="+", required=True, choices=("tp", "ep"),
    )
    parser.add_argument(
        "--kv-cache-dtype", choices=("fp8", "fp16"), default="fp8",
    )
    parser.add_argument("--warmup-rounds", type=_positive_int, default=16)
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
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicates")
    if len(set(args.mtp)) != len(args.mtp):
        parser.error("--mtp must not contain duplicates")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("--batch-sizes must not contain duplicates")
    if len(set(args.moe_strategy)) != len(args.moe_strategy):
        parser.error("--moe-strategy must not contain duplicates")
    if args.final_temperature > args.initial_temperature:
        parser.error("--final-temperature must not exceed --initial-temperature")
    if args.output_csv is not None and args.output_csv.exists():
        parser.error(f"--output-csv already exists: {args.output_csv}")
    args.batch_sizes = sorted(args.batch_sizes)
    args.mtp = sorted(args.mtp, key=lambda value: value == "on")
    args.moe_strategy = sorted(
        args.moe_strategy, key=lambda value: ("tp", "ep").index(value),
    )
    return args


def _make_cases(args: argparse.Namespace) -> list[SACase]:
    return [
        SACase(
            model=model,
            global_batch_size=batch,
            mtp_enabled=mtp == "on",
            input_sequence_length=args.input_sequence_length,
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
        )
        for model in args.models
        for strategy in args.moe_strategy
        for batch in args.batch_sizes
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


def _run_case(case: SACase) -> dict[str, Any]:
    strategy = MoEParallelStrategy(case.moe_strategy)
    request = HardwareMappingRequest(case.model, InferenceConfig(
        global_batch_size=case.global_batch_size,
        input_sequence_length=case.input_sequence_length,
        output_sequence_length=case.output_sequence_length,
        moe_parallel_strategy=strategy,
        kv_cache_dtype=case.kv_cache_dtype,
        mtp_enabled=case.mtp_enabled,
    ))
    result = SplitTreeOptimizer().run(
        request,
        config=SimulatedAnnealingConfig(
            enabled=True,
            batch_sizes=(case.global_batch_size,),
            mtp_values=(case.mtp_enabled,),
            warmup_rounds=case.warmup_rounds,
            rounds=case.rounds,
            restart_count=case.restart_count,
            random_seed=case.base_seed,
            initial_temperature=case.initial_temperature,
            final_temperature=case.final_temperature,
            layer_count=case.layer_count,
            write_result=False,
        ),
    )
    if result.output_path is not None or len(result.workload_results) != 1:
        raise RuntimeError("independent SA worker produced an invalid result")
    workload = result.workload_results[0]
    baseline = workload.baseline_evaluation
    best = workload.best_evaluation
    return {
        **asdict(case),
        "chip_count": DEFAULT_HARDWARE_CONFIG.chip_count,
        "case_seed": workload.random_seed,
        "restart_count": workload.restart_count,
        "restart_results": [
            {
                "restart_index": item.restart_index,
                "random_seed": item.random_seed,
                "proposal_count": item.proposal_count,
                "candidate_simulation_count": item.candidate_simulation_count,
                "cache_hit_count": item.cache_hit_count,
                "accepted_proposal_count": item.accepted_proposal_count,
            }
            for item in workload.restart_results
        ],
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
        "latency_reduction_fraction": workload.latency_reduction_fraction,
        "throughput_improvement_fraction": workload.throughput_improvement_fraction,
        "proposal_count": workload.proposal_count,
        "candidate_simulation_count": workload.candidate_simulation_count,
        "materialization_simulation_count": (
            workload.materialization_simulation_count
        ),
        "simulator_run_count": workload.simulator_run_count,
        "cache_hit_count": workload.cache_hit_count,
        "accepted_proposal_count": workload.accepted_proposal_count,
        "best_split_trees": [
            split_tree_to_dict(item.split_tree)
            for item in workload.best_evaluations
        ],
    }


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
    path: Path, model: str, strategy: str, rows: Sequence[dict[str, Any]],
    *, generated_at: str, detected_cpus: int, worker_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workloads = [
        dict(row) for row in rows
        if row["model"] == model and row["moe_strategy"] == strategy
    ]
    if not workloads:
        raise ValueError("model/strategy has no workload rows")
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
    } for row in workloads if row["is_model_global_pareto"]]
    first = workloads[0]
    payload = {
        "format_version": 1,
        "generated_at": generated_at,
        "model": model,
        "hardware": {"chip_count": DEFAULT_HARDWARE_CONFIG.chip_count},
        "search": {
            "moe_parallel_strategy": first["moe_strategy"],
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
        "pareto_front": front,
    }
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


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
    print(
        f"detected {detected} logical CPUs; running {len(cases)} cases "
        f"with {workers} worker processes",
        flush=True,
    )
    print(f"CSV checkpoint: {csv_path}", flush=True)

    saved_count = 0

    def save_completed_case(row: dict[str, Any]) -> None:
        nonlocal saved_count
        append_csv_checkpoint(csv_path, row)
        saved_count += 1

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
        model_rank[row["model"]], strategy_rank[row["moe_strategy"]],
        row["global_batch_size"], row["mtp_enabled"],
    ))
    results = mark_model_pareto(results)

    finalize_csv_checkpoint(csv_path, results)

    json_paths = []
    for model in args.models:
        for strategy in args.moe_strategy:
            path = _unique_path(
                output_dir,
                f"{minute}_{_safe_name(model)}_{strategy}_sa_best_splittrees",
                ".json",
            )
            write_model_json(
                path, model, strategy, results, generated_at=generated_at,
                detected_cpus=detected, worker_count=workers,
            )
            json_paths.append(path)

    print(f"CSV: {csv_path}")
    for path in json_paths:
        print(f"SplitTree JSON: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
