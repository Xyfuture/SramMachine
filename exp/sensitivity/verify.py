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
    trace_count = 0
    accepted = 2 if config["mtp_enabled"] else 1
    for parameter in config["parameters"]:
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
            assert metrics["xfold"] and metrics["mtp_enabled"] == config["mtp_enabled"]
            assert metrics["accepted_tokens_per_step"] == accepted
            assert payload["inference"]["mtp_enabled"] == config["mtp_enabled"]
            assert metrics["moe_strategy"] == "ep" and metrics["kv_cache_dtype"] == "fp8"
            assert metrics["chip_count"] == 8 and metrics["simulated_layer_count"] == 4
            assert payload["partition_chains"]
            commands = payload["command_results"]
            starts = [min(c["start_time_ns"] for c in commands
                          if c["layer_index"] == layer and c["cmd_id"].endswith(".core"))
                      for layer in (1, 2)]
            interval = starts[1] - starts[0]
            assert interval == metrics["steady_layer_time_ns"]
            components = metrics.get("model_latency_components")
            latency = (sum(layers * duration for _, layers, duration in components)
                       if components else interval * metrics["model_layer_count"])
            if components:
                assert sum(layers for _, layers, _ in components) == metrics["model_layer_count"]
            assert latency == metrics["model_step_latency_ns"]
            assert math.isclose(metrics[X], accepted * 1e9 / latency)
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
            def check_layer(layer, trace_path, *, dense=False):
                nonlocal trace_count
                policies = layer.get("weight_memory_by_op", {})
                layer_rates = ({key: value["flash_bandwidth_bytes_per_second"]
                               for key, value in policies.items()
                               if value and value["flash_bandwidth_bytes_per_second"] is not None}
                              if policies else rates)
                pu = hardware.chip.logic_die.processing_unit
                for op_id, rate in layer_rates.items():
                    policy = policies.get(op_id)
                    expected_rate = pu.flash_bandwidth_bytes_per_second
                    if "indexer_score" in op_id and (not policy or policy["source"] != "dram"):
                        expected_rate = table.lookup(row["model"], "dense_indexer", config["mtp_enabled"], batch,
                                                    config["input_sequence_length"]) * multiplier
                        if config.get("indexer_noc_bandwidth", False):
                            expected_rate += hardware.chip.noc.link_bandwidth_bytes_per_second * 64 * 2 / hardware.chip.processing_unit_count
                    elif not dense and (op_id in layer.get("prefetch_op_ids", ["moe.moe_fc2"])
                                        or (op_id in ("attn.flash_attention", "attn.flash_msa")
                                            and (not policy or policy["source"] != "dram"))):
                        expected_rate = table.lookup(row["model"], "sparse_kv", config["mtp_enabled"], batch,
                                                    config["input_sequence_length"]) * multiplier
                    if parameter == "noc" and config.get("remote_supply_cap") and factor != 1:
                        remote = min(1, factor)
                        pooled = (1 + remote) / 2
                        if "indexer_score" in op_id and (not policy or policy["source"] != "dram"):
                            extra = hardware.chip.noc.link_bandwidth_bytes_per_second * 128 / hardware.chip.processing_unit_count
                            expected_rate = (expected_rate - extra) * pooled + extra / factor * remote
                        elif op_id in layer.get("prefetch_op_ids", ["moe.moe_fc2"]):
                            expected_rate *= remote
                        else:
                            expected_rate *= pooled
                    assert math.isclose(rate, expected_rate), (op_id, rate, expected_rate)
                trace = json.loads((directory / trace_path).read_text())
                context = trace["otherData"]
                assert context["global_batch_size"] == batch
                assert context["xfold"] == (False if dense else metrics["xfold"])
                assert context["input_sequence_length"] == config["input_sequence_length"]
                events = {e["args"]["cmd_id"]: e["args"] for e in trace["traceEvents"] if e["ph"] in ("X", "i")}
                assert set(events) == {c["cmd_id"] for c in layer["command_results"]}
                for command in layer["command_results"]:
                    event = events[command["cmd_id"]]
                    assert event["start_time_ns"] == command["start_time_ns"]
                    assert event["end_time_ns"] == command["end_time_ns"]
                    if command["command_type"] == "PUMemoryReadCmd":
                        params = command["parameters"]
                        rate = pu.dram_bandwidth_bytes_per_second if params["port"] == "dram" else layer_rates.get(command["op_id"], pu.flash_bandwidth_bytes_per_second)
                        if params["port"] == "dram" and parameter == "noc" and config.get("remote_supply_cap"):
                            rate *= (1 + min(1, factor)) / 2
                        assert math.isclose(params["bandwidth_bytes_per_second"], rate)
                        assert command["end_time_ns"] - command["start_time_ns"] == math.ceil(params["size_bytes"] * 1e9 / rate)
                trace_count += 1
            check_layer(payload, metrics["trace_path"])
            auxiliary = payload.get("auxiliary_results", {})
            paths = metrics.get("auxiliary_trace_paths", [])
            assert len(auxiliary) == len(paths)
            for path, (kind, layer) in zip(paths, auxiliary.items()):
                check_layer(layer, path, dense=kind == "dense")
                layer_starts = [min(c["start_time_ns"] for c in layer["command_results"]
                                    if c["layer_index"] == i and c["cmd_id"].endswith(".core")) for i in (1, 2)]
                assert layer_starts[1] - layer_starts[0] == layer["steady_layer_time_ns"]
                assert dict((name, duration) for name, _, duration in components)[kind] == layer["steady_layer_time_ns"]
            if factor == 1:
                key = row["model"], batch
                assert baseline_latencies.setdefault(key, latency) == latency
            count += 1
        for model in config["models"]:
            assert (directory / f"{model}-{parameter}-pareto.png").stat().st_size > 1000
    completion = json.loads((directory / "completion.json").read_text())
    assert completion["status"] == "complete"
    assert completion["completed_points"] == completion["expected_points"] == count
    summary = {"status": "passed", "verified_points": count, "verified_traces": trace_count,
               "checks": ["coverage", "hardware isolation", "HBF CSV scaling", "baseline agreement",
                          "raw timing", "throughput", "per-configuration Pareto", "CSV agreement",
                          "trace timings", "backup hashes"]}
    write_json(directory / "verification.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(verify(parser.parse_args().directory))
