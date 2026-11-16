"""Perfetto trace metadata and independent DSA memory track regressions."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from srammachine.inference import InferenceConfig
from srammachine.mapping import DeepSeekV32Mapper
from srammachine.pipetree import TreeParser
from srammachine.simulator import Simulator, export_perfetto_trace


class PerfettoExportTests(unittest.TestCase):
    def test_dsa_variants_export_shapes_ports_and_precise_timing(self):
        mapper = DeepSeekV32Mapper()
        for fold in (False, True):
            with self.subTest(fold=fold), tempfile.TemporaryDirectory() as directory:
                mapped = mapper.map(InferenceConfig(256, 20000, 600), xfold=fold)
                graph = TreeParser(mapper.hardware_config).parse(
                    mapped.root_node, mapped.operators, mapped.operator_mappings,
                )
                result = Simulator(mapper.hardware_config).run(graph, mapping_result=mapped)
                path = export_perfetto_trace(result, output_dir=directory, mapping_result=mapped)
                payload = json.loads(path.read_text())
                context = payload["otherData"]
                self.assertEqual(context["xfold"], fold)
                self.assertEqual(context["chip_count"], 8)
                self.assertEqual(context["simulated_layer_count"], 4)
                self.assertEqual(context["input_sequence_length"], 20000)
                events = [event for event in payload["traceEvents"] if event["ph"] in ("X", "i")]
                self.assertEqual(len(events), len(graph.commands))
                tracks = {event["args"]["name"]: event["tid"]
                          for event in payload["traceEvents"] if event["name"] == "thread_name"}
                dram = tracks["chip0.die0.pu0.dram"]
                flash = tracks["chip0.die0.pu0.flash"]
                self.assertNotEqual(dram, flash)
                self.assertEqual({event["args"]["layer_index"] for event in events}, set(range(4)))
                for event in events:
                    args = event["args"]
                    self.assertAlmostEqual(event["ts"] * 1000, args["start_time_ns"])
                    self.assertAlmostEqual(event.get("dur", 0) * 1000, args["duration_ns"])
                    self.assertEqual(args["xfold"], fold)
                    self.assertIn("component", args)
                    if args["op_id"] == "attn.qk_nope":
                        self.assertEqual(args["global_dimensions"], dict(B=1, M=256, K=1536, N=16384))
                        self.assertEqual(args["partition_chain"], ["K-p-32", "N-q-64"])
                        self.assertEqual(args["weight_source"], "dram" if fold else "dram+flash")
                    if args["command_type"] == "PUMemoryReadCmd":
                        self.assertEqual(event["tid"], dram if args["port"] == "dram" else flash)
                        self.assertIn(args["data_kind"], event["name"])
                        self.assertIn(args["port"].upper(), event["name"])
                        self.assertIn("TB/s", event["name"])
                prefetches = [event for event in events
                              if event["args"].get("data_kind") == "next_layer_kv"]
                self.assertEqual(len(prefetches), 4 if fold else 0)
                if fold:
                    for event in prefetches:
                        self.assertEqual(event["args"]["component"], "FusedFoldingMoE")
                        self.assertEqual(event["args"]["flash_prefetch_bytes_per_request"], 2048 * 576)
                        self.assertEqual(event["args"]["prefetch_target_layer_index"],
                                         event["args"]["layer_index"] + 1)
                        self.assertIn(".kv_prefetch L", event["name"])

    def test_generic_export_preserves_existing_files(self):
        mapper = DeepSeekV32Mapper()
        mapped = mapper.map(InferenceConfig(32, 32, 1))
        graph = TreeParser(mapper.hardware_config).parse(
            mapped.root_node, mapped.operators, mapped.operator_mappings, layer_count=1,
        )
        result = Simulator(mapper.hardware_config).run(graph)
        with tempfile.TemporaryDirectory() as directory:
            timestamp = datetime(2026, 11, 14, tzinfo=timezone.utc)
            first = export_perfetto_trace(result, output_dir=directory, timestamp=timestamp)
            original = first.read_bytes()
            second = export_perfetto_trace(result, output_dir=directory, timestamp=timestamp)
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), original)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 2)
            self.assertNotIn("model_name", json.loads(original)["otherData"])


if __name__ == "__main__":
    unittest.main()
