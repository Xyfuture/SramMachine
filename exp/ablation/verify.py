"""Audit ablation coverage, effective hardware, CSV rates and traced timings."""

from collections import defaultdict
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

from exp.ablation.sweep import case_directory
from exp.ablation.policy import load_no_kv_bandwidth, KV_LAYOUT_ABLATIONS, MOE_LAYOUT_ABLATIONS
from exp.big.sweep import write_json
from srammachine.mapping import HBFBandwidthTable


def verify_layer(layer: dict, base: dict, variant: str, manifest: dict,
                 metrics: dict, table: HBFBandwidthTable, *, dense: bool = False, no_kv_table=None) -> None:
    """Check grids, per-op bandwidths and actual port durations independently."""
    base_pu = base["chip"]["logic_die"]["processing_unit"]
    intervals = defaultdict(list)
    for op_id, hardware in layer["hardware_by_op"].items():
        special = op_id in ("attn.flash_msa", "attn.flash_attention", "attn.indexer_score", "moe.indexer_score")
        dram = not special and variant in ("NoDRAMShare", "NoDRAMFLashShare")
        flash = special and variant in ("NoFlashShare", "NoDRAMFLashShare")
        assert hardware["chip"]["noc"]["mesh_rows"] == (16 if dram or flash else 32), op_id
        assert hardware["chip"]["noc"]["mesh_columns"] == 64
        pu = hardware["chip"]["logic_die"]["processing_unit"]
        for port, doubled in (("dram", dram), ("flash", flash)):
            key = f"{port}_bandwidth_bytes_per_second"
            assert pu[key] == base_pu[key] * (2 if doubled else 1), (op_id, key)
        if op_id in layer["partition_chains"]:
            steps = layer["partition_chains"][op_id]
            grid = [math.prod(step["factor"] for step in steps if step["pu_dimension"] == axis)
                    for axis in ("p", "q")]
            assert grid == [16 if dram or flash else 32, 64], op_id
        policy = layer["weight_memory_by_op"][op_id]
        if policy is None:
            continue
        rate = base_pu["flash_bandwidth_bytes_per_second"]
        if op_id.endswith("indexer_score") and policy["source"] != "dram":
            rate = table.lookup(metrics["model"], "dense_indexer", metrics["mtp_enabled"],
                                metrics["global_batch_size"], metrics["input_sequence_length"])
        elif not dense and ((special and not metrics["xfold"])
                            or (metrics["xfold"] and op_id == "moe.moe_fc2")):
            kind = "dense_kv" if metrics["model"] == "deepseek-v3" else "sparse_kv"
            rate = table.lookup(metrics["model"], kind, metrics["mtp_enabled"],
                                metrics["global_batch_size"], metrics["input_sequence_length"])
        if (manifest["ablation_revision"] >= 4 and op_id.endswith("indexer_score")
                and policy["source"] != "dram" and not flash):
            rate += base["chip"]["noc"]["link_bandwidth_bytes_per_second"] * 64 * 2 / 2048
        if flash:
            rate *= 2
        if manifest["ablation_revision"] < 6 and variant == "NoKVLayout" and op_id.endswith("indexer_score") and policy["source"] != "dram":
            rate *= manifest["kv_factor"]
        if (no_kv_table is not None and variant in KV_LAYOUT_ABLATIONS and not dense
                and ((op_id in ("attn.flash_attention", "attn.flash_msa") and policy["source"] != "dram")
                     or (metrics["xfold"] and op_id == "moe.moe_fc2"))):
            rate = no_kv_table.lookup(metrics["model"], "sparse_kv", metrics["mtp_enabled"],
                                      metrics["global_batch_size"], metrics["input_sequence_length"])
        if variant in MOE_LAYOUT_ABLATIONS and op_id.startswith("moe."):
            if (variant != "NoHBFLayout" and manifest["moe_flash_scope"] == "stage") or (
                layer["hardware_mappings"][op_id]["op_kind"] == "BMMOp"
                and (op_id != "moe.moe_fc2" or not metrics["xfold"])
            ):
                rate *= manifest["moe_factor"]
        assert math.isclose(policy["flash_bandwidth_bytes_per_second"], rate), (op_id, rate, policy)
    for command in layer["command_results"]:
        params = command["parameters"]
        if command["command_type"] == "PUMemoryReadCmd":
            hardware = layer["hardware_by_op"][command["op_id"]]
            pu = hardware["chip"]["logic_die"]["processing_unit"]
            rate = (pu["dram_bandwidth_bytes_per_second"] if params["port"] == "dram"
                    else layer["weight_memory_by_op"][command["op_id"]]["flash_bandwidth_bytes_per_second"])
            assert math.isclose(params["bandwidth_bytes_per_second"], rate)
            assert command["end_time_ns"] - command["start_time_ns"] == math.ceil(params["size_bytes"] * 1e9 / rate)
        if command["command_type"] != "InterChipCmd":
            for resource in params.get("resource_ids", [command["resource_id"]]):
                intervals[resource].append((command["start_time_ns"], command["end_time_ns"]))
    if variant == "NoSeparatePath":
        # End events precede starts at the same timestamp: touching is legal.
        events = []
        for command in layer["command_results"]:
            params = command["parameters"]
            horizontal = params.get("noc_direction") in ("horizontal", "both")
            work = (command["category"] == "compute"
                    or command["command_type"] in ("PUMemoryReadCmd", "DramReadCmd"))
            if command["end_time_ns"] > command["start_time_ns"] and (horizontal or work):
                chip = command["resource_id"].split(".")[0]
                kind = 0 if horizontal else 1
                events.extend(((command["start_time_ns"], 1, chip, kind),
                               (command["end_time_ns"], -1, chip, kind)))
        active = defaultdict(int)
        for _, delta, chip, kind in sorted(events):
            active[chip, kind] += delta
            assert not (active[chip, 0] and active[chip, 1]), "SeparatePath exclusion violated"
    for resource, spans in intervals.items():
        spans.sort()
        assert all(previous[1] <= current[0] for previous, current in zip(spans, spans[1:])), resource


def verify_trace(root: Path, path: str, commands: list[dict], variant: str,
                 expected_adjustment: dict | None = None) -> None:
    """Check every Perfetto command timestamp against the raw checkpoint."""
    trace = json.loads((root / path).read_text())
    assert trace["otherData"]["ablation"] == variant
    if expected_adjustment is not None:
        assert trace["otherData"]["latency_adjustment"] == expected_adjustment
    events = {event["args"]["cmd_id"]: event["args"] for event in trace["traceEvents"]
              if event["ph"] in ("X", "i")}
    assert set(events) == {command["cmd_id"] for command in commands}
    for command in commands:
        event = events[command["cmd_id"]]
        assert event["start_time_ns"] == command["start_time_ns"]
        assert event["end_time_ns"] == command["end_time_ns"]


def verify(root: Path) -> dict:
    """Require every requested case and audit derived metrics and figures."""
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["ablation_revision"] in (2, 3, 4, 5, 6, 7)
    for filename, key in (("source.zip", "source_zip_sha256"), ("hbf_read_bandwidth.csv", "bandwidth_csv_sha256")):
        assert hashlib.sha256((root / "scripts" / filename).read_bytes()).hexdigest() == manifest[key]
    assert json.loads((root / "scripts/base_hardware.json").read_text()) == manifest["hardware"]
    reuse_path = root / "reuse_manifest.json"
    if reuse_path.exists():
        reuse = json.loads(reuse_path.read_text())
        assert hashlib.sha256((root / reuse["source_archive"]).read_bytes()).hexdigest() == reuse["source_zip_sha256"]
        for filename, digest in reuse["file_sha256"].items():
            # Only unchanged variants may be imported into this revision.
            assert Path(filename).parts[1] != "NoMoELayout"
            assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == digest, filename
        assert sum(name.endswith("/result.json") for name in reuse["file_sha256"]) == reuse["reused_points"]
    table = HBFBandwidthTable.from_csv(root / "scripts/hbf_read_bandwidth.csv")
    no_kv_table = None
    if manifest.get("no_kv_bandwidth_csv_sha256"):
        path = root / "scripts/no_kv_layout_bandwidth.csv"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["no_kv_bandwidth_csv_sha256"]
        no_kv_table = load_no_kv_bandwidth(path)
    accepted = 2 if manifest["mtp_enabled"] else 1
    assert manifest["accepted_tokens_per_step"] == accepted
    points, traces = 0, 0
    for model in manifest["models"]:
        with (root / f"{model}.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        expected = set(itertools.product(manifest["ablations"], manifest["batch_sizes"], manifest["sequence_lengths"]))
        actual = [(row["ablation"], int(row["global_batch_size"]), int(row["input_sequence_length"])) for row in rows]
        assert len(actual) == len(expected) and set(actual) == expected
        for sequence in manifest["sequence_lengths"]:
            assert (root / f"{model}-pareto-s{sequence}.png").stat().st_size > 1000
        if 1024 in manifest["batch_sizes"]:
            assert (root / f"{model}-latency-bs1024.png").stat().st_size > 1000
        for row in rows:
            variant, batch, sequence = row["ablation"], int(row["global_batch_size"]), int(row["input_sequence_length"])
            directory = case_directory(root, model, variant, batch, sequence)
            payload = json.loads((directory / "result.json").read_text())
            metrics = payload["metrics"]
            assert (metrics["model"], metrics["ablation"], metrics["global_batch_size"], metrics["input_sequence_length"]) == (model, variant, batch, sequence)
            assert metrics["status"] == "complete" and not (directory / "error.json").exists()
            assert metrics["xfold"] == (variant != "NoXfold" and model != "deepseek-v3")
            assert metrics["mtp_enabled"] == payload["inference"]["mtp_enabled"] == manifest["mtp_enabled"]
            assert (row["mtp_enabled"] == "True") == manifest["mtp_enabled"]
            assert int(row["accepted_tokens_per_step"]) == accepted
            assert metrics["accepted_tokens_per_step"] == accepted and metrics["simulated_layer_count"] == 4
            assert payload["base_hardware"] == manifest["hardware"]
            if variant == "NoSeparatePath" and "SeparatePath" in manifest["ablations"]:
                paired = json.loads((case_directory(root, model, "SeparatePath", batch, sequence) / "result.json").read_text())
                for current, reference in [(payload, paired)] + [
                    (value, paired["auxiliary_results"][key])
                    for key, value in payload["auxiliary_results"].items()
                ]:
                    assert current["partition_chains"] == reference["partition_chains"]
                    # Identical commands/bytes/FLOPs; only scheduling may change.
                    def untimed(layer):
                        return [{k: v for k, v in c.items() if k not in ("start_time_ns", "end_time_ns")}
                                for c in layer["command_results"]]
                    assert untimed(current) == untimed(reference)
            verify_layer(payload, manifest["hardware"], variant, manifest, metrics, table, no_kv_table=no_kv_table)
            commands = payload["command_results"]
            starts = [min(command["start_time_ns"] for command in commands
                          if command["layer_index"] == index and command["cmd_id"].endswith(".core"))
                      for index in (1, 2)]
            interval = starts[1] - starts[0]
            assert interval == metrics["steady_layer_time_ns"]
            components = metrics["model_latency_components"]
            raw_latency = sum(count * duration for _, count, duration in components) if components else interval * metrics["model_layer_count"]
            assert raw_latency == metrics["simulated_model_step_latency_ns"]
            per_layer = (25_000 if model == "minimax-m3" else 14_000) if variant == "NoKVLayout" and manifest["ablation_revision"] < 6 else 0
            overhead = per_layer * metrics["model_layer_count"]
            assert metrics["additional_latency_per_layer_ns"] == per_layer
            assert metrics["additional_model_latency_ns"] == overhead
            assert metrics["adjusted_model_latency_components"] == [
                [name, count, duration + per_layer] for name, count, duration in components]
            latency = raw_latency + overhead
            assert latency == metrics["model_step_latency_ns"]
            assert math.isclose(metrics["tokens_per_second_per_user"], accepted * 1e9 / latency)
            assert math.isclose(metrics["tokens_per_second_per_chip"], accepted * 1e9 * batch / latency / metrics["chip_count"])
            for key in ("tokens_per_second_per_user", "tokens_per_second_per_chip", "model_step_latency_ns",
                        "simulated_model_step_latency_ns", "additional_model_latency_ns", "additional_latency_per_layer_ns"):
                assert math.isclose(float(row[key]), metrics[key], rel_tol=1e-12)
            group = [other for other in rows if other["ablation"] == variant and int(other["input_sequence_length"]) == sequence]
            x, y = "tokens_per_second_per_user", "tokens_per_second_per_chip"
            pareto = not any(float(other[x]) >= float(row[x]) and float(other[y]) >= float(row[y])
                             and (float(other[x]) > float(row[x]) or float(other[y]) > float(row[y])) for other in group)
            assert (row["is_pareto"] == "True") == pareto
            adjustment = {key: metrics[key] for key in (
                "simulated_model_step_latency_ns", "additional_latency_per_layer_ns",
                "additional_model_latency_ns", "model_step_latency_ns", "model_step_latency_ms",
                "adjusted_model_latency_components")}
            verify_trace(root, metrics["trace_path"], commands, variant, adjustment)
            assert len(metrics["auxiliary_trace_paths"]) == len(payload["auxiliary_results"])
            for path, (layer_type, additional) in zip(metrics["auxiliary_trace_paths"], payload["auxiliary_results"].items()):
                verify_layer(additional, manifest["hardware"], variant, manifest, metrics, table, dense=layer_type == "dense", no_kv_table=no_kv_table)
                verify_trace(root, path, additional["command_results"], variant)
                assert dict((name, duration) for name, _, duration in components)[layer_type] == additional["steady_layer_time_ns"]
            points += 1
            traces += 1 + len(metrics["auxiliary_trace_paths"])
    completion = json.loads((root / "completion.json").read_text())
    assert completion["status"] == "complete" and completion["completed_points"] == points == completion["expected_points"]
    summary = {"status": "passed", "verified_points": points, "verified_traces": traces,
               "reused_points_verified": reuse["reused_points"] if reuse_path.exists() else 0,
               "checks": ["coverage", "source and CSV hashes", "per-op grids and ports", "CSV bandwidth scaling",
                          "actual memory durations", "shared-resource contention", "weighted decoder latency",
                          "additive delay per full decoder layer", "adjusted throughput",
                          "per-variant Pareto dominance", "CSV/raw agreement", "trace/raw agreement"]}
    write_json(root / "verification.json", summary)
    return summary


def main() -> None:
    """Audit one or more completed result directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    for directory in parser.parse_args().directories:
        print(directory, verify(directory))


if __name__ == "__main__":
    main()
