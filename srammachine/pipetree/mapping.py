"""Explicit lowering policy; byte counts are per mapped resource instance.

Operator dimensions describe the full PipeTree batch at the intended hardware
mapping. This policy does not automatically partition chips or PUs.
Per-token counts refer to entries assigned to the mapped resource.  When
``batch_partition_degree`` is greater than one, TreeParser first converts a
logical tree batch to the representative partition's ceiling-sized batch.
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional
from srammachine.commands.base import integer, nonempty


@dataclass(frozen=True)
class OperatorMapping:
    batch_axis: str
    resource_id: str
    dram_resource_id: Optional[str] = None
    sram_resource_id: Optional[str] = None
    dram_read_once_bytes: int = 0
    dram_read_bytes_per_token: int = 0
    dram_write_bytes_per_token: int = 0
    weight_load_fixed_bytes: int = 0
    weight_load_bytes_per_token: int = 0
    # Exact resident weight shape.  HardwareMapper supplies the die-local
    # matrix shape here so trace metadata does not mistake the 4x4 PU tile
    # count for the batched-GEMM B dimension.
    weight_shape: Optional[Mapping[str, int]] = None
    batch_partition_degree: int = 1
    sram_read_bytes_per_mapped_token: int = 0
    sram_read_data_kind: Optional[str] = None

    def __post_init__(self) -> None:
        if self.batch_axis not in ("B", "M", "m", "size_bytes"):
            raise ValueError("batch_axis must be B, M, m or size_bytes")
        nonempty("resource_id", self.resource_id)
        for name in ("dram_resource_id", "sram_resource_id"):
            value = getattr(self, name)
            if value is not None:
                nonempty(name, value)
        for name in (
            "dram_read_once_bytes", "dram_read_bytes_per_token",
            "dram_write_bytes_per_token", "weight_load_fixed_bytes",
            "weight_load_bytes_per_token",
            "sram_read_bytes_per_mapped_token",
        ):
            integer(name, getattr(self, name))
        integer("batch_partition_degree", self.batch_partition_degree, 1)
        if (self.dram_read_once_bytes or self.dram_read_bytes_per_token
                or self.dram_write_bytes_per_token) and self.dram_resource_id is None:
            raise ValueError("DRAM traffic requires dram_resource_id")
        if self.dram_read_once_bytes and self.sram_resource_id is None:
            raise ValueError("weight prefetch requires sram_resource_id")
        if (self.weight_load_fixed_bytes or self.weight_load_bytes_per_token
                ) and self.sram_resource_id is None:
            raise ValueError("weight loading requires sram_resource_id")
        if self.sram_read_bytes_per_mapped_token:
            if self.sram_resource_id is None:
                raise ValueError("SRAM demand read requires sram_resource_id")
            nonempty("sram_read_data_kind", self.sram_read_data_kind)
        elif self.sram_read_data_kind is not None:
            raise ValueError(
                "sram_read_data_kind requires SRAM demand-read traffic"
            )
        if self.weight_shape is not None:
            if not isinstance(self.weight_shape, Mapping):
                raise TypeError("weight_shape must be a mapping")
            copied = {}
            for name, value in self.weight_shape.items():
                nonempty("weight shape dimension", name)
                integer(name, value, 0)
                copied[name] = value
            object.__setattr__(self, "weight_shape", MappingProxyType(copied))
