"""Audit complete sweep artifacts without rerunning the simulator."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from openpyxl import load_workbook

from exp.big.sweep import write_json


def verify(directory: Path) -> dict:
    """Check coverage, configurations, raw timing, traces and both table formats."""
    manifest = json.loads((directory / "manifest.json").read_text())
    for filename, key in (("source.zip", "source_zip_sha256"),
                          ("hbf_read_bandwidth.csv", "bandwidth_csv_sha256")):
        assert hashlib.sha256((directory / "scripts" / filename).read_bytes()).hexdigest() == manifest[key]
    points, trace_count = 0, 0
    for model in manifest["models"]:
        with (directory / f"{model}.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        batches = manifest["batch_sizes"]
        assert sorted(int(row["global_batch_size"]) for row in rows) == sorted(batches)
        workbook = load_workbook(directory / f"{model}.xlsx", read_only=True)
        excel_values = list(workbook["metrics"].values)
        assert len(excel_values) == len(rows) + 1
        workbook.close()
        assert (directory / f"{model}-pareto.png").stat().st_size > 1000
        for row, excel in zip(rows, excel_values[1:]):
            payload = json.loads((directory / row["raw_result_path"]).read_text())
            metrics = payload["metrics"]
            assert metrics["status"] == "complete"
            assert metrics["mtp_enabled"] and metrics["accepted_tokens_per_step"] == 2
            assert metrics["xfold"] == (model != "deepseek-v3")
            assert metrics["moe_strategy"] == "ep" and metrics["kv_cache_dtype"] == "fp8"
            assert metrics["chip_count"] == 8 and metrics["simulated_layer_count"] == 4
            assert metrics["optimize_partition_chains"] and not metrics["split_tree_search_enabled"]
            assert metrics["input_sequence_length"] == manifest["input_sequence_length"]
            assert payload["hardware"] == manifest["hardware"]
            assert payload["partition_chains"]
            commands = payload["command_results"]
            core_start = [min(c["start_time_ns"] for c in commands
                              if c["layer_index"] == layer and c["cmd_id"].endswith(".core"))
                          for layer in (1, 2)]
            interval = core_start[1] - core_start[0]
            assert interval == metrics["steady_layer_time_ns"]
            components = metrics["model_latency_components"]
            latency = (sum(count * duration for _, count, duration in components)
                       if components else interval * metrics["model_layer_count"])
            assert latency == metrics["model_step_latency_ns"]
            assert math.isclose(metrics["tokens_per_second_per_user"], 2e9 / latency)
            assert math.isclose(metrics["tokens_per_second_per_chip"],
                                2e9 * metrics["global_batch_size"] / latency / 8)
            x, y = "tokens_per_second_per_user", "tokens_per_second_per_chip"
            expected_pareto = not any(
                float(other[x]) >= float(row[x]) and float(other[y]) >= float(row[y])
                and (float(other[x]) > float(row[x]) or float(other[y]) > float(row[y]))
                for other in rows
            )
            assert (row["is_pareto"] == "True") == expected_pareto
            for key in (x, y, "model_step_latency_ns"):
                assert math.isclose(float(row[key]), metrics[key], rel_tol=1e-12)
                assert math.isclose(excel[excel_values[0].index(key)], metrics[key], rel_tol=1e-12)
            trace = json.loads((directory / metrics["trace_path"]).read_text())
            context = trace["otherData"]
            assert context["global_batch_size"] == metrics["global_batch_size"]
            assert context["input_sequence_length"] == metrics["input_sequence_length"]
            assert context["xfold"] == metrics["xfold"]
            events = {e["args"]["cmd_id"]: e["args"] for e in trace["traceEvents"]
                      if e["ph"] in ("X", "i")}
            for command in commands:
                event = events[command["cmd_id"]]
                assert event["start_time_ns"] == command["start_time_ns"]
                assert event["end_time_ns"] == command["end_time_ns"]
            for path in metrics["auxiliary_trace_paths"]:
                auxiliary = json.loads((directory / path).read_text())
                assert auxiliary["traceEvents"]
            trace_count += 1 + len(metrics["auxiliary_trace_paths"])
            points += 1
    summary = {"status": "passed", "verified_points": points, "verified_traces": trace_count,
               "checks": ["coverage", "configuration", "source and CSV hashes", "raw timing",
                          "throughput formulas", "Pareto dominance", "CSV/Excel agreement",
                          "trace/raw command agreement"]}
    write_json(directory / "verification.json", summary)
    return summary


def main() -> None:
    """Verify each supplied sequence-length result directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    for directory in args.directories:
        print(directory, verify(directory))


if __name__ == "__main__":
    main()
