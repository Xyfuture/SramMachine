"""Traffic conservation, tile readiness and shared-path scheduling invariants."""
import unittest

from exp.ablation.separatepath import tile_gemms
from srammachine.commands import CommandGraph, CommandTrace, GemmCmd, NoCCmd, PUMemoryReadCmd
from srammachine.hardware import HardwareConfig
from srammachine.simulator import Simulator


class SeparatePathTests(unittest.TestCase):
    def graph(self, k=130):
        commands = [
            NoCCmd("i0.input", "gemm", "chip0.noc_horizontal", "FromIOdie", k*128,
                   io_bytes_per_row=(k*128, k*64)),
            GemmCmd("i0.core", "gemm", "chip0.die0.pu0", 1, 1, k, 64, memory_duration_ns=k*4),
            NoCCmd("i0.output", "gemm", "chip0.noc_horizontal", "ToIOdie", 128,
                   io_bytes_per_row=(128,64)),
            PUMemoryReadCmd("i0.weight.dram.0", "gemm", "chip0.die0.pu0.dram", k*48, "dram", 16e9),
            PUMemoryReadCmd("i0.weight.flash.1", "gemm", "chip0.die0.pu0.flash", k*16, "flash", 4e9),
        ]
        return CommandGraph(commands, [("i0.input","i0.core"),("i0.core","i0.output"),
                                      ("i0.weight.dram.0","i0.output"),("i0.weight.flash.1","i0.output")],
                            start_edges=[("i0.core","i0.weight.dram.0"),("i0.core","i0.weight.flash.1")],
                            traces={c.cmd_id:CommandTrace(0,0,1,(0,)) for c in commands})

    def test_ragged_tiles_conserve_bytes_flops_and_output(self):
        original=self.graph()
        tiled=tile_gemms(original,64)
        cores=[c for c in tiled.commands if type(c) is GemmCmd]
        self.assertEqual([c.K for c in cores],[64,64,2])
        self.assertEqual(sum(2*c.B*c.M*c.K*c.N for c in cores),2*130*64)
        inputs=[c for c in tiled.commands if isinstance(c,NoCCmd) and c.operation=="FromIOdie"]
        self.assertEqual(tuple(sum(c.io_bytes_per_row[i] for c in inputs) for i in range(2)),(130*128,130*64))
        for port,width in (("dram",48),("flash",16)):
            self.assertEqual(sum(c.size_bytes for c in tiled.commands if isinstance(c,PUMemoryReadCmd) and c.port==port),130*width)
        self.assertEqual(tiled.command("i0.output"),original.command("i0.output"))
        self.assertIn(("i0.tile0.core","i0.tile2.input"),tiled.edges)
        self.assertEqual(len(original.commands),5)

    def test_small_batch_has_real_overlap_and_exclusion(self):
        original=self.graph(256)
        results=[]
        for separate in (True,False):
            graph=tile_gemms(original,64,separate_path=separate)
            result=Simulator(HardwareConfig()).run(graph)
            results.append(result)
            indexed={c.cmd_id:c for c in result.command_results}
            for a,b in graph.edges:
                self.assertLessEqual(indexed[a].end_time_ns,indexed[b].start_time_ns)
            noc=[c for c in result.command_results if c.command_type=="NoCCmd"]
            work=[c for c in result.command_results if c.command_type in ("GemmCmd","PUMemoryReadCmd")]
            overlaps=[(a,b) for a in noc for b in work if max(a.start_time_ns,b.start_time_ns)<min(a.end_time_ns,b.end_time_ns)]
            self.assertEqual(bool(overlaps),separate)
            memory=[c for c in result.command_results if c.command_type=="PUMemoryReadCmd"]
            self.assertTrue(any(a.resource_id!=b.resource_id and max(a.start_time_ns,b.start_time_ns)<min(a.end_time_ns,b.end_time_ns) for a in memory for b in memory))
        self.assertGreater(results[1].total_time_ns,results[0].total_time_ns)

    def test_single_tile_and_non_gemm_graph(self):
        graph=tile_gemms(self.graph(16),64)
        self.assertEqual(sum(type(c) is GemmCmd for c in graph.commands),1)
        empty=CommandGraph([])
        self.assertEqual(tile_gemms(empty,64).commands,())


if __name__ == "__main__":
    unittest.main()
