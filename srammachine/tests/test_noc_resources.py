"""Directional NoC mapping, atomic reservations and trace regressions."""

import unittest

from srammachine.commands import CommandGraph, InterChipCmd, NoCCmd
from srammachine.frontend.modules import CommOp
from srammachine.hardware import DEFAULT_HARDWARE_CONFIG
from srammachine.inference import InferenceConfig, MoEParallelStrategy
from srammachine.mapping import HardwareMapper, HardwareMappingRequest
from srammachine.mapping.mapper import _LayerBuilder
from srammachine.pipetree import OperatorMapping, OpNode, PipeTree, TreeParser
from srammachine.simulator import GraphExecutor, Simulator
from srammachine.simulator.perfetto import _trace_payload


HORIZONTAL = "chip0.noc_horizotal"
VERTICAL = "chip0.noc_vertical"


def transfer(cmd_id, resource_id, duration=10, additional=()):
    """Create a transfer at the default 256 bytes/ns link bandwidth."""
    return NoCCmd(
        cmd_id, cmd_id, resource_id, "p2p", (0, 1), duration * 256,
        additional_resource_ids=additional,
    )


class NoCResourceTests(unittest.TestCase):
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
        op = CommOp("exchange", "alltoall", "intra_chip", (0, 1), 2560)
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
            if command.kind in ("broadcast", "allgather"):
                self.assertEqual(command.resource_id, HORIZONTAL)
            elif command.kind in ("reduce", "reduce_scatter"):
                self.assertEqual(command.resource_id, VERTICAL)
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
