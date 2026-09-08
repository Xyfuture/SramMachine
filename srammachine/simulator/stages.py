"""Desim-backed workers for the hardware resources used by commands."""

from abc import ABC, abstractmethod
import math
from typing import Tuple, Type

from Desim import FIFO, SimModule, SimSession, SimTime

from srammachine.commands import (
    Command, DramCmd, GemmCmd, InterChipCmd, NoCCmd, VectorCmd,
    WeightLoadCmd,
)
from srammachine.hardware import HardwareConfig
from .records import CommandExecution
from .vector_costs import vector_flop_count


def _rate_duration_ns(work, rate) -> int:
    if work == 0:
        return 0
    return math.ceil(work * 1_000_000_000 / rate)


def _communication_volume_ratio(command) -> Tuple[int, int]:
    """Return critical-path bytes as an exact numerator/denominator pair."""
    participant_count = len(command.group)
    if command.transfer_bytes is not None:
        matrix = command.transfer_bytes
        sent = [
            sum(size for column, size in enumerate(row) if column != row_index)
            for row_index, row in enumerate(matrix)
        ]
        received = [
            sum(matrix[row][column] for row in range(participant_count)
                if row != column)
            for column in range(participant_count)
        ]
        return max(sent + received, default=0), 1

    size_bytes = command.size_bytes
    if command.kind in ("p2p", "broadcast"):
        return size_bytes, 1
    if command.kind == "allreduce":
        return 2 * (participant_count - 1) * size_bytes, participant_count
    if command.kind == "allgather":
        return (participant_count - 1) * size_bytes, 1
    if command.kind in ("reduce_scatter", "alltoall"):
        return (participant_count - 1) * size_bytes, participant_count
    raise ValueError(f"unsupported communication kind: {command.kind}")


class HardwareResourceStage(SimModule, ABC):
    """One persistent, serial Desim worker for a logical resource ID."""

    accepted_command_types: Tuple[Type[Command], ...] = ()

    def __init__(
        self, resource_id: str, hardware_config: HardwareConfig,
        completion_fifo: FIFO,
    ) -> None:
        if SimSession.scheduler is None:
            raise RuntimeError("initialize SimSession before creating resource stages")
        if not isinstance(resource_id, str) or not resource_id.strip():
            raise ValueError("resource_id must be a nonempty string")
        if not isinstance(hardware_config, HardwareConfig):
            raise TypeError("hardware_config must be a HardwareConfig")
        if not isinstance(completion_fifo, FIFO):
            raise TypeError("completion_fifo must be a Desim FIFO")
        super().__init__()
        self.resource_id = resource_id
        self.hardware_config = hardware_config
        self.completion_fifo = completion_fifo
        self.command_fifo = FIFO(1)
        self._outstanding = False
        self.register_coroutine(self._process)

    @property
    def has_outstanding_command(self) -> bool:
        return self._outstanding

    def submit(self, command: Command) -> None:
        self._validate_command(command)
        if self._outstanding:
            raise RuntimeError(f"resource stage is busy: {self.resource_id}")
        self._outstanding = True
        self.command_fifo.write(command)

    def _validate_command(self, command: Command) -> None:
        if not isinstance(command, self.accepted_command_types):
            expected = ", ".join(cls.__name__ for cls in self.accepted_command_types)
            raise TypeError(f"{type(self).__name__} requires {expected}")
        if command.resource_id != self.resource_id:
            raise ValueError("command resource_id does not match its stage")

    def _process(self) -> None:
        while True:
            command = self.command_fifo.read()
            start_time_ns = SimSession.sim_time.cycle
            duration_ns = self.latency_ns(command)
            if type(duration_ns) is not int or duration_ns < 0:
                raise ValueError("latency_ns must return a nonnegative integer")
            SimModule.wait_time(SimTime(duration_ns))
            execution = CommandExecution(
                command.cmd_id,
                command.resource_id,
                start_time_ns,
                SimSession.sim_time.cycle,
            )
            self._outstanding = False
            self.completion_fifo.write(execution)

    @abstractmethod
    def latency_ns(self, command: Command) -> int:
        raise NotImplementedError


class DramResourceStage(HardwareResourceStage):
    accepted_command_types = (DramCmd,)

    def latency_ns(self, command: DramCmd) -> int:
        self._validate_command(command)
        bandwidth = (
            self.hardware_config.chip.logic_die.memory
            .dram_bandwidth_bytes_per_second
        )
        return _rate_duration_ns(command.size_bytes, bandwidth)


class SramResourceStage(HardwareResourceStage):
    accepted_command_types = (WeightLoadCmd,)

    def latency_ns(self, command: WeightLoadCmd) -> int:
        self._validate_command(command)
        logic_die = self.hardware_config.chip.logic_die
        bandwidth = min(
            logic_die.memory.sram_bandwidth_bytes_per_second,
            logic_die.processing_unit.weight_bandwidth_bytes_per_second,
        )
        return _rate_duration_ns(command.size_bytes, bandwidth)


class ProcessingUnitStage(HardwareResourceStage):
    accepted_command_types = (GemmCmd,)

    def latency_ns(self, command: GemmCmd) -> int:
        self._validate_command(command)
        flops = 2 * command.B * command.M * command.K * command.N
        peak_flops = (
            self.hardware_config.chip.logic_die.processing_unit.peak_flops
        )
        return _rate_duration_ns(flops, peak_flops)


class VectorUnitStage(HardwareResourceStage):
    accepted_command_types = (VectorCmd,)

    def latency_ns(self, command: VectorCmd) -> int:
        self._validate_command(command)
        flops = vector_flop_count(command)
        peak_flops = self.hardware_config.chip.logic_die.vector_unit.peak_flops
        return _rate_duration_ns(flops, peak_flops)


class NoCResourceStage(HardwareResourceStage):
    accepted_command_types = (NoCCmd,)

    def latency_ns(self, command: NoCCmd) -> int:
        self._validate_command(command)
        numerator, denominator = _communication_volume_ratio(command)
        if numerator == 0:
            return 0
        noc = self.hardware_config.chip.noc
        transfer_time_ns = _rate_duration_ns(
            numerator, denominator * noc.link_bandwidth_bytes_per_second,
        )
        fixed_time_ns = math.ceil(noc.link_latency_ns)
        return transfer_time_ns + fixed_time_ns


class InterChipFabricStage(HardwareResourceStage):
    accepted_command_types = (InterChipCmd,)

    def latency_ns(self, command: InterChipCmd) -> int:
        self._validate_command(command)
        chip_count = self.hardware_config.chip_count
        if len(command.group) > chip_count:
            raise ValueError(
                "inter-chip command has more participants than system chips"
            )
        if any(
            type(participant) is int and participant >= chip_count
            for participant in command.group
        ):
            raise ValueError("inter-chip participant ID exceeds system chip count")
        numerator, denominator = _communication_volume_ratio(command)
        bandwidth = (
            self.hardware_config.inter_chip_fabric
            .per_chip_directional_bandwidth_bytes_per_second
        )
        return _rate_duration_ns(numerator, denominator * bandwidth)


def stage_class_for_command(command: Command):
    if isinstance(command, DramCmd):
        return DramResourceStage
    if isinstance(command, WeightLoadCmd):
        return SramResourceStage
    if isinstance(command, GemmCmd):
        return ProcessingUnitStage
    if isinstance(command, VectorCmd):
        return VectorUnitStage
    if isinstance(command, NoCCmd):
        return NoCResourceStage
    if isinstance(command, InterChipCmd):
        return InterChipFabricStage
    raise TypeError(f"unsupported command type: {type(command).__name__}")
