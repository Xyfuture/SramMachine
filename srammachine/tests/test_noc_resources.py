"""Directional NoC mapping, atomic reservations and trace regressions."""

from dataclasses import replace
import unittest

from srammachine.commands import CommandGraph, InterChipCmd, NoCCmd
from srammachine.frontend.modules import NoCOp, FabricOp
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG, NoCConfig
from srammachine.inference import InferenceConfig, MoEParallelStrategy
from srammachine.mapping import HardwareMapper, HardwareMappingRequest
from srammachine.mapping.mapper import _LayerBuilder
from srammachine.pipetree import OperatorMapping, OpNode, PipeTree, TreeParser
from srammachine.simulator import GraphExecutor, Simulator
from srammachine.simulator.perfetto import _trace_payload


HORIZONTAL = "chip0.noc_horizontal"
VERTICAL = "chip0.noc_vertical"


def transfer(cmd_id, resource_id, duration=10, additional=()):
    """Create a transfer at the default 256 bytes/ns link bandwidth."""
    return NoCCmd(
        cmd_id, cmd_id, resource_id,
        "PUMesh" if additional else "PUHorizontal" if resource_id == HORIZONTAL else "PUVertical",
        duration * 256, "p2p", (0, 1),
        additional_resource_ids=additional,
    )


class NoCResourceTests(unittest.TestCase):
    def test_all_operations_charge_explicit_bytes_only(self):
        for operation in ("ToIOdie", "FromIOdie", "PUVertical", "PUHorizontal", "PUMesh"):
            for kind in ("p2p", "broadcast", "reduce", "allreduce", "allgather",
                         "reduce_scatter", "alltoall"):
                with self.subTest(operation=operation, kind=kind):
                    op = NoCOp("op", operation, 2561, kind, (0, 1, 2, 3))
                    cmd = NoCCmd("cmd", "op", op.resource_ids()[0], operation,
                                 op.size_bytes, kind, op.group)
                    graph = TreeParser().parse(
                        PipeTree(1, (op.op_id,), OpNode(op.op_id)), {op.op_id: op},
                        {op.op_id: OperatorMapping("size_bytes", cmd.resource_id)},
                    )
                    result = Simulator().run(graph)
                    self.assertEqual(result.command_results[0].duration_ns, 11)
                    item = result.command_results[0]
                    self.assertEqual(item.parameters["size_bytes"], 2561)
                    self.assertEqual(item.parameters["operation"], operation)
                    self.assertEqual(cmd.resource_ids, op.resource_ids())

    def test_io_transfers_contend_while_vertical_overlaps(self):
        commands = (
            NoCCmd("input", "input", HORIZONTAL, "FromIOdie", 2560),
            NoCCmd("output", "output", HORIZONTAL, "ToIOdie", 2560),
            transfer("vertical", VERTICAL, 20),
        )
        result = GraphExecutor.simulate(CommandGraph(commands))
        self.assertEqual(result.total_time_ns, 20)
        self.assertEqual(result.executions["output"].start_time_ns, 10)
        self.assertEqual(result.executions["vertical"].start_time_ns, 0)

    def test_directional_bandwidth_and_no_fixed_latency(self):
        noc = NoCConfig(horizontal_bandwidth_bytes_per_second=256e9,
                        vertical_bandwidth_bytes_per_second=128e9,
                        link_latency_ns=100)
        hardware = replace(DEFAULT_HARDWARE_CONFIG, chip=replace(
            DEFAULT_HARDWARE_CONFIG.chip, noc=noc))
        for operation, resource, duration in (
            ("FromIOdie", HORIZONTAL, 10), ("ToIOdie", HORIZONTAL, 10),
            ("PUHorizontal", HORIZONTAL, 10), ("PUVertical", VERTICAL, 20),
            ("PUMesh", HORIZONTAL, 20),
        ):
            with self.subTest(operation=operation):
                cmd = NoCCmd("cmd", "op", resource, operation, 2560, "allreduce")
                result = GraphExecutor.simulate(CommandGraph((cmd,)), hardware)
                self.assertEqual(result.total_time_ns, duration)
        for bandwidth in (0, -1, float("inf"), float("nan"), True):
            with self.assertRaises(ValueError):
                NoCConfig(horizontal_bandwidth_bytes_per_second=bandwidth)

    def test_noc_validation_and_operation_resource_binding(self):
        for operation, resource in (("ToIOdie", VERTICAL), ("FromIOdie", VERTICAL),
                                    ("PUVertical", HORIZONTAL), ("PUHorizontal", VERTICAL)):
            with self.assertRaises(ValueError):
                NoCCmd("cmd", "op", resource, operation, 2560, "allgather")
        for size in (-1, 0.5, True):
            with self.assertRaises(ValueError):
                NoCOp("op", "FromIOdie", size)
        with self.assertRaises(ValueError):
            NoCOp("op", "unknown", 0)
        with self.assertRaises(ValueError):
            NoCOp("op", "PUMesh", 0)

    def test_nonuniform_mapping_uses_max_sender_not_receiver(self):
        request = HardwareMappingRequest(
            "deepseek-v3", InferenceConfig(256, 20000, 600, MoEParallelStrategy.TP))
        mapped = HardwareMapper().map(request)
        builder = _LayerBuilder(request, DEFAULT_HARDWARE_CONFIG, mapped.model_config)
        builder.add_comm("exchange", "exchange", scope="intra_chip", kind="alltoall",
                         group=(0, 1, 2), parallel_strategy="tp",
                         transfer_bytes=((900, 0, 0), (2560, 900, 0), (2560, 0, 900)))
        op = builder.operators["exchange"]
        self.assertIsInstance(op, NoCOp)
        self.assertEqual(op.size_bytes, 2560)
        self.assertEqual(op.operation, "PUMesh")

    def test_independent_directions_overlap(self):
        result = GraphExecutor.simulate(CommandGraph((
            transfer("h", HORIZONTAL), transfer("v", VERTICAL),
        )))
        self.assertEqual(result.total_time_ns, 10)
        self.assertTrue(all(item.start_time_ns == 0
                            for item in result.executions.values()))

    def test_dual_transfer_waits_for_both_and_releases_both(self):
        for primary, secondary in ((HORIZONTAL, VERTICAL), (VERTICAL, HORIZONTAL)):
            with self.subTest(primary=primary):
                graph = CommandGraph((
                    transfer("first", secondary, 20),
                    transfer("both", primary, additional=(secondary,)),
                    transfer("other", primary),
                    transfer("after_h", HORIZONTAL),
                    transfer("after_v", VERTICAL),
                ), edges=(("both", "after_h"), ("both", "after_v")))
                result = GraphExecutor.simulate(graph)
                # A waiting dual transfer must not reserve its free direction.
                self.assertEqual(result.executions["other"].start_time_ns, 0)
                self.assertEqual(result.executions["both"].start_time_ns, 20)
                self.assertEqual(result.executions["both"].resource_ids,
                                 (primary, secondary))
                self.assertEqual(result.executions["after_h"].start_time_ns, 30)
                self.assertEqual(result.executions["after_v"].start_time_ns, 30)
                self.assertEqual(result.total_time_ns, 40)
                self.assertIn("both", graph.resource_order[secondary])

    def test_opposite_resource_orders_and_priorities(self):
        graph = CommandGraph((
            transfer("a", HORIZONTAL, additional=(VERTICAL,)),
            transfer("b", VERTICAL, additional=(HORIZONTAL,)),
        ), scheduling_priorities={"a": 1, "b": 0})
        result = GraphExecutor.simulate(graph)
        self.assertEqual(result.executions["b"].start_time_ns, 0)
        self.assertEqual(result.executions["a"].start_time_ns, 10)
        self.assertEqual(result.total_time_ns, 20)

    def test_zero_duration_and_start_dependencies(self):
        graph = CommandGraph((
            transfer("both", HORIZONTAL, 0, (VERTICAL,)),
            transfer("next", VERTICAL),
        ), start_edges=(("both", "next"),))
        result = GraphExecutor.simulate(graph)
        self.assertEqual(result.executions["next"].start_time_ns, 0)
        self.assertEqual(result.total_time_ns, 10)

    def test_lowering_layers_and_perfetto_occupy_both_tracks(self):
        op = NoCOp("exchange", "PUMesh", 2560, "alltoall", (0, 1))
        tree = PipeTree(1, (op.op_id,), OpNode(op.op_id))
        graph = TreeParser().parse(tree, {op.op_id: op}, {
            op.op_id: OperatorMapping(
                "size_bytes", HORIZONTAL, additional_resource_ids=(VERTICAL,),
            ),
        }, layer_count=2)
        self.assertTrue(all(cmd.resource_ids == (HORIZONTAL, VERTICAL)
                            for cmd in graph.commands))
        result = Simulator().run(graph)
        events = _trace_payload(result)["traceEvents"]
        tracks = {event["tid"]: event["args"]["name"]
                  for event in events if event["name"] == "thread_name"}
        for item in result.command_results:
            self.assertEqual(item.resource_ids, (HORIZONTAL, VERTICAL))
            self.assertEqual(item.parameters["noc_direction"], "both")
            slices = [event for event in events
                      if event.get("args", {}).get("cmd_id") == item.cmd_id]
            self.assertEqual({tracks[event["tid"]] for event in slices},
                             {HORIZONTAL, VERTICAL})
            self.assertEqual(len(slices), 2)
            self.assertEqual(slices[0]["ts"], slices[1]["ts"])
            self.assertEqual(slices[0]["dur"], slices[1]["dur"])

    def test_mapper_migrates_resources_and_supports_special_collectives(self):
        request = HardwareMappingRequest(
            "deepseek-v3", InferenceConfig(256, 20000, 600, MoEParallelStrategy.TP),
        )
        mapped = HardwareMapper().map(request)
        graph = TreeParser().parse(
            mapped.root_node, mapped.operators, mapped.operator_mappings,
            layer_count=1,
        )
        noc_commands = [cmd for cmd in graph.commands if isinstance(cmd, NoCCmd)]
        self.assertTrue(noc_commands)
        for command in noc_commands:
            self.assertTrue(set(command.resource_ids) <= {HORIZONTAL, VERTICAL})
            if command.operation in ("FromIOdie", "ToIOdie", "PUHorizontal"):
                self.assertEqual(command.resource_ids, (HORIZONTAL,))
            elif command.operation == "PUVertical":
                self.assertEqual(command.resource_ids, (VERTICAL,))
            else:
                self.assertEqual(set(command.resource_ids), {HORIZONTAL, VERTICAL})
        self.assertTrue(any(isinstance(op, FabricOp) for op in mapped.operators.values()))
        builder = _LayerBuilder(request, DEFAULT_HARDWARE_CONFIG, mapped.model_config)
        for kind in ("allreduce", "alltoall", "p2p"):
            with self.subTest(kind=kind):
                builder.add_comm(
                    kind, kind, scope="intra_chip", kind=kind, group=(0, 1),
                    size_bytes=2560, parallel_strategy="tp",
                    **({"noc_direction": "both"} if kind == "p2p" else {}),
                )
                mapping = builder.mappings[kind]
                self.assertEqual({mapping.resource_id, *mapping.additional_resource_ids},
                                 {HORIZONTAL, VERTICAL})

    def test_invalid_resource_sets(self):
        for additional in ((HORIZONTAL,), (VERTICAL, VERTICAL), ("",), VERTICAL):
            with self.subTest(additional=additional):
                with self.assertRaises(ValueError):
                    transfer("bad", HORIZONTAL, additional=additional)
                with self.assertRaises(ValueError):
                    OperatorMapping("size_bytes", HORIZONTAL,
                                    additional_resource_ids=additional)

    def test_fabric_transfers_still_overlap(self):
        commands = tuple(InterChipCmd(
            name, name, "system.fabric", "p2p", (0, 1), 2560,
        ) for name in ("a", "b"))
        result = GraphExecutor.simulate(CommandGraph(commands))
        self.assertTrue(all(item.start_time_ns == 0
                            for item in result.executions.values()))


if __name__ == "__main__":
    unittest.main()
