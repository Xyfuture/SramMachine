"""Audit sensitivity coverage, resource isolation, throughput and exported traces."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from exp.sensitivity.sweep import PARAMETERS, X, Y, plain, scaled_hardware, write_json
from srammachine.hardware import HardwareConfig
from srammachine.mapping import HBFBandwidthTable


def verify(directory: Path) -> dict:
    """Raise on inconsistent artifacts; persist a summary only after all checks pass."""
    config = json.loads((directory / "manifest.json").read_text())
    for filename, key in (("source.zip", "source_zip_sha256"),
                          ("hbf_read_bandwidth.csv", "bandwidth_csv_sha256")):
        assert hashlib.sha256((directory / "scripts" / filename).read_bytes()).hexdigest() == config[key]
    assert config["baseline_hardware"] == plain(HardwareConfig())
    table = HBFBandwidthTable.from_csv(directory / "scripts/hbf_read_bandwidth.csv")
    baseline_latencies = {}
    count = 0
    for parameter in PARAMETERS:
        with (directory / f"{parameter}.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        expected = {(model, factor, batch) for model in config["models"]
                    for factor in config["factors"] for batch in config["batch_sizes"]}
        actual = {(r["model"], float(r["factor"]), int(r["global_batch_size"])) for r in rows}
        assert actual == expected and len(rows) == len(expected), parameter
        for row in rows:
            payload = json.loads((directory / row["raw_result_path"]).read_text())
            metrics = payload["metrics"]
            factor, batch = metrics["factor"], metrics["global_batch_size"]
            hardware = scaled_hardware(parameter, factor, config["logic_scope"])
            assert payload["hardware"] == plain(hardware) == metrics["hardware"]
            assert metrics["parameter"] == parameter and metrics["status"] == "complete"
            assert metrics["input_sequence_length"] == config["input_sequence_length"]
            assert metrics["xfold"] and metrics["mtp_enabled"] and metrics["accepted_tokens_per_step"] == 2
            assert metrics["moe_strategy"] == "ep" and metrics["kv_cache_dtype"] == "fp8"
            assert metrics["chip_count"] == 8 and metrics["simulated_layer_count"] == 4
            assert payload["partition_chains"]
            commands = payload["command_results"]
            starts = [min(c["start_time_ns"] for c in commands
                          if c["layer_index"] == layer and c["cmd_id"].endswith(".core"))
                      for layer in (1, 2)]
            latency = (starts[1] - starts[0]) * metrics["model_layer_count"]
            assert latency == metrics["model_step_latency_ns"]
            assert math.isclose(metrics[X], 2e9 / latency)
            assert math.isclose(metrics[Y], metrics[X] * batch / 8)
            for key in (X, Y, "model_step_latency_ns"):
                assert math.isclose(float(row[key]), metrics[key], rel_tol=1e-12)
            group = [r for r in rows if r["model"] == row["model"] and r["factor"] == row["factor"]]
            nondominated = not any(float(r[X]) >= metrics[X] and float(r[Y]) >= metrics[Y]
                and (float(r[X]) > metrics[X] or float(r[Y]) > metrics[Y]) for r in group)
            assert (row["is_pareto"] == "True") == nondominated
            multiplier = factor if parameter == "hbf" else 1
            rates = metrics["flash_bandwidth_bytes_per_second_by_op"]
            assert "moe.indexer_score" in rates
            for op_id, rate in rates.items():
                kind = "dense_indexer" if "indexer_score" in op_id else "sparse_kv"
                expected_rate = table.lookup(row["model"], kind, True, batch,
                                             config["input_sequence_length"]) * multiplier
                assert math.isclose(rate, expected_rate), (op_id, rate, expected_rate)
            trace = json.loads((directory / metrics["trace_path"]).read_text())
            context = trace["otherData"]
            assert context["global_batch_size"] == batch and context["xfold"]
            assert context["input_sequence_length"] == config["input_sequence_length"]
            events = {e["args"]["cmd_id"]: e["args"] for e in trace["traceEvents"] if e["ph"] in ("X", "i")}
            for command in commands:
                event = events[command["cmd_id"]]
                assert event["start_time_ns"] == command["start_time_ns"]
                assert event["end_time_ns"] == command["end_time_ns"]
            if factor == 1:
                key = row["model"], batch
                assert baseline_latencies.setdefault(key, latency) == latency
            count += 1
        for model in config["models"]:
            assert (directory / f"{model}-{parameter}-pareto.png").stat().st_size > 1000
    completion = json.loads((directory / "completion.json").read_text())
    assert completion["status"] == "complete"
    assert completion["completed_points"] == completion["expected_points"] == count
    summary = {"status": "passed", "verified_points": count, "verified_traces": count,
               "checks": ["coverage", "hardware isolation", "HBF CSV scaling", "baseline agreement",
                          "raw timing", "throughput", "per-configuration Pareto", "CSV agreement",
                          "trace timings", "backup hashes"]}
    write_json(directory / "verification.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(verify(parser.parse_args().directory))
