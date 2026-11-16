"""Explicit lowering policy; byte counts are per mapped resource instance.

Operator dimensions describe the full PipeTree batch at the intended hardware
mapping. This policy does not automatically partition chips or PUs.
Per-token counts refer to entries assigned to the mapped resource.  When
``batch_partition_degree`` is greater than one, TreeParser first converts a
logical tree batch to the representative partition's ceiling-sized batch.
"""
from dataclasses import dataclass
import math
from typing import Optional
from srammachine.commands.base import integer, nonempty
from srammachine.hardware import HardwareConfig
from .ep_routing import EPRouting


@dataclass(frozen=True)
class OperatorMapping:
    batch_axis: str
    resource_id: str
    dram_resource_id: Optional[str] = None
    dram_read_bytes_per_token: int = 0
    dram_write_bytes_per_token: int = 0
    # Explicit per-PU weight bytes for prepartitioned legacy operators.
    weight_read_fixed_bytes: int = 0
    batch_partition_degree: int = 1
    dram_read_logical_bytes_per_token: Optional[int] = None
    dram_write_logical_bytes_per_token: Optional[int] = None
    shared_die_factor: int = 1
    # Optional number of indivisible logical work units along batch_axis.
    # TreeParser distributes these units across microbatches with quotient /
    # remainder partitioning, so adjacent instances differ by at most one.
    batch_scaling_unit_count: Optional[int] = None
    # Demand cache traffic can cover every chip-local request while the
    # representative PU core covers only one request partition.
    demand_batch_partition_degree: Optional[int] = None
    # Extra NoC directions occupied by the same communication command.
    additional_resource_ids: tuple[str, ...] = ()
    cache_query_group_size: int = 1
    ep_routing: Optional[EPRouting] = None
    # Uniform per-PU KV share for legacy head-expanded attention tiles.
    cache_read_bytes_per_request: Optional[float] = None
    # Per-op execution view; resource IDs still identify shared physical ports.
    hardware_config: Optional[HardwareConfig] = None

    def __post_init__(self) -> None:
        if self.hardware_config is not None and not isinstance(self.hardware_config, HardwareConfig):
            raise TypeError("hardware_config must be a HardwareConfig")
        if self.cache_read_bytes_per_request is not None:
            size = self.cache_read_bytes_per_request
            if (isinstance(size, bool) or not isinstance(size, (int, float))
                    or not math.isfinite(size) or size <= 0):
                raise ValueError("cache_read_bytes_per_request must be positive and finite")
        if self.ep_routing is not None:
            if not isinstance(self.ep_routing, EPRouting):
                raise TypeError("ep_routing must be EPRouting")
            if (self.batch_axis != "size_bytes" or self.batch_partition_degree != 1
                    or self.batch_scaling_unit_count is not None):
                raise ValueError("EP routing requires unpartitioned byte mapping")
        integer("cache_query_group_size", self.cache_query_group_size, 1)
        if self.batch_axis not in ("B", "M", "m", "size_bytes"):
            raise ValueError("batch_axis must be B, M, m or size_bytes")
        nonempty("resource_id", self.resource_id)
        if isinstance(self.additional_resource_ids, str):
            raise ValueError("additional_resource_ids must be a sequence of IDs")
        additional = tuple(self.additional_resource_ids)
        for resource_id in additional:
            nonempty("additional resource ID", resource_id)
        if len(set((self.resource_id,) + additional)) != 1 + len(additional):
            raise ValueError("resource IDs must be unique")
        object.__setattr__(self, "additional_resource_ids", additional)
        if self.dram_resource_id is not None:
            nonempty("dram_resource_id", self.dram_resource_id)
        for name in (
            "dram_read_bytes_per_token", "dram_write_bytes_per_token",
            "weight_read_fixed_bytes",
        ):
            integer(name, getattr(self, name))
        for effective_name, logical_name in (
            ("dram_read_bytes_per_token", "dram_read_logical_bytes_per_token"),
            ("dram_write_bytes_per_token", "dram_write_logical_bytes_per_token"),
        ):
            logical = getattr(self, logical_name)
            if logical is not None:
                integer(logical_name, logical)
                if logical < getattr(self, effective_name):
                    raise ValueError(
                        f"{logical_name} must be at least {effective_name}"
                    )
        integer("shared_die_factor", self.shared_die_factor, 1)
        has_shared_size = any(
            getattr(self, name) is not None
            for name in (
                "dram_read_logical_bytes_per_token",
                "dram_write_logical_bytes_per_token",
            )
        )
        if self.shared_die_factor > 1 and not has_shared_size:
            raise ValueError(
                "shared_die_factor greater than one requires a logical size"
            )
        integer("batch_partition_degree", self.batch_partition_degree, 1)
        if self.demand_batch_partition_degree is not None:
            integer("demand_batch_partition_degree",
                    self.demand_batch_partition_degree, 1)
        if self.batch_scaling_unit_count is not None:
            integer("batch_scaling_unit_count", self.batch_scaling_unit_count, 1)
            if self.batch_partition_degree != 1:
                raise ValueError(
                    "nonuniform batch scaling requires batch_partition_degree=1"
                )
        if (self.dram_read_bytes_per_token
                or self.dram_write_bytes_per_token) and self.dram_resource_id is None:
            raise ValueError("DRAM traffic requires dram_resource_id")
