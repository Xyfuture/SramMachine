"""Explicit lowering policy; byte counts are per mapped resource instance.

Operator dimensions describe the full PipeTree batch at the intended hardware
mapping. This policy does not automatically partition chips or PUs.
Per-token counts refer to logical entries in the tree batch, not context length.
"""
from dataclasses import dataclass
from typing import Optional
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
        ):
            integer(name, getattr(self, name))
        if (self.dram_read_once_bytes or self.dram_read_bytes_per_token
                or self.dram_write_bytes_per_token) and self.dram_resource_id is None:
            raise ValueError("DRAM traffic requires dram_resource_id")
        if self.dram_read_once_bytes and self.sram_resource_id is None:
            raise ValueError("weight prefetch requires sram_resource_id")
        if (self.weight_load_fixed_bytes or self.weight_load_bytes_per_token
                ) and self.sram_resource_id is None:
            raise ValueError("weight loading requires sram_resource_id")
