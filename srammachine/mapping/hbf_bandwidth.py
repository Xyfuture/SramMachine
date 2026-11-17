"""Measured chip HBF bandwidths, normalized to individual PU Flash ports."""

from bisect import bisect_left
import csv
from dataclasses import replace
import math
from pathlib import Path
from typing import TYPE_CHECKING
import warnings

from srammachine.frontend.modules import WeightMemory, WeightSource

if TYPE_CHECKING:
    from srammachine.hardware import HardwareConfig
    from .mapper import HardwareMappingResult


DEFAULT_HBF_BANDWIDTH_CSV = (
    Path(__file__).resolve().parents[1] / "model_cards" / "hbf_read_bandwidth.csv"
)
MEASURED_PU_COUNT = 2048
MODEL_NAMES = {
    "DeepSeek-V3": "deepseek-v3",
    "DeepSeek-V3.2": "deepseek-v3.2",
    "Kimi-K2": "kimi-k2.5",
    "GLM-5.1": "glm-5.1",
    "GLM-5.2": "glm-5.2",
    "MiniMax-M3": "minimax-m3",
}
READ_KINDS = {"dense_kv", "sparse_kv", "dense_indexer"}


def _op_id(model: str, kind: str) -> str:
    if kind == "dense_indexer":
        return "attn.indexer_score"
    return "attn.flash_msa" if model == "minimax-m3" else "attn.flash_attention"


def normalize_hbf_csv(source: str | Path, destination: str | Path) -> int:
    """Export canonical names and per-PU bytes/s, retaining measurement metadata.

    The supplied measurements cover one chip with 2048 PUs. GLM 5.2 is
    retained as data only; normalization does not add simulator model support.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("normalized output must differ from the source CSV")
    renames = {
        "model": "model_name", "part": "read_kind", "mtp": "mtp_enabled",
        "global_bs": "global_batch_size", "local_bs": "local_batch_size",
        "mean_seqlen": "input_sequence_length",
    }
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = set(renames) | {"mean_GBps", "token_bytes"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"source CSV requires columns {sorted(required)}")
        fields = [renames.get(name, name) for name in reader.fieldnames]
        fields += ["op_id", "bandwidth_scope", "pu_count",
                   "flash_bandwidth_bytes_per_second", "source_model", "source_part"]
        rows = []
        for original in reader:
            row = {renames.get(key, key): value for key, value in original.items()}
            row["model_name"] = MODEL_NAMES[original["model"]]
            part = original["part"]
            if part in ("sparse_kv_mtp0", "sparse_kv_mtp1"):
                if original["mtp"] != part[-1]:
                    raise ValueError(f"inconsistent part and mtp: {part}")
                row["read_kind"] = "sparse_kv"
            if row["read_kind"] not in READ_KINDS:
                raise ValueError(f"unknown HBF part: {part}")
            row.update(
                op_id=_op_id(row["model_name"], row["read_kind"]),
                bandwidth_scope="chip", pu_count=MEASURED_PU_COUNT,
                flash_bandwidth_bytes_per_second=(
                    float(original["mean_GBps"]) * 1e9 / MEASURED_PU_COUNT
                ),
                source_model=original["model"], source_part=part,
            )
            rows.append(row)
    # Validate before creating the output; never overwrite the raw measurements.
    HBFBandwidthTable(rows)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


class HBFBandwidthTable:
    """Exact lookup or bilinear interpolation over global batch and context.

    Interpolation is linear in the numeric batch/sequence coordinates, with no
    extrapolation. Dense reads may reuse MTP0 only when no MTP1 series exists.
    """

    def __init__(self, rows: list[dict]) -> None:
        self._series: dict[tuple[str, str, bool], dict[tuple[int, int], float]] = {}
        for number, row in enumerate(rows, start=2):
            try:
                model, kind = row["model_name"], row["read_kind"]
                mtp = str(row["mtp_enabled"])
                batch = int(row["global_batch_size"])
                sequence = int(row["input_sequence_length"])
                rate = float(row["flash_bandwidth_bytes_per_second"])
                mean = float(row["mean_GBps"])
                if (model not in MODEL_NAMES.values() or kind not in READ_KINDS
                        or mtp not in ("0", "1") or batch <= 0 or sequence <= 0
                        or row["op_id"] != _op_id(model, kind)
                        or row["bandwidth_scope"] != "chip"
                        or int(row["pu_count"]) != MEASURED_PU_COUNT
                        or not math.isfinite(rate) or rate <= 0
                        or not math.isfinite(mean) or mean <= 0
                        or not math.isclose(rate, mean * 1e9 / MEASURED_PU_COUNT,
                                            rel_tol=1e-9)):
                    raise ValueError("invalid names, dimensions, scope or bandwidth")
                points = self._series.setdefault((model, kind, mtp == "1"), {})
                if (batch, sequence) in points:
                    raise ValueError("duplicate measurement key")
                points[batch, sequence] = rate
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid HBF CSV row {number}: {error}") from error
        if not self._series:
            raise ValueError("HBF CSV contains no measurements")

    @classmethod
    def from_csv(cls, path: str | Path) -> "HBFBandwidthTable":
        """Read and validate a normalized CSV once for a run."""
        with Path(path).open(newline="", encoding="utf-8-sig") as stream:
            return cls(list(csv.DictReader(stream)))

    def lookup(self, model_name: str, read_kind: str, mtp_enabled: bool,
               global_batch_size: int, input_sequence_length: int) -> float:
        """Return bytes/s per PU; reject unavailable series or out-of-range inputs."""
        key = (model_name, read_kind, mtp_enabled)
        if (key not in self._series and mtp_enabled
                and read_kind in ("dense_kv", "dense_indexer")):
            key = (model_name, read_kind, False)
        if key not in self._series:
            raise ValueError(f"no HBF measurements for {key}")
        points = self._series[key]

        def bracket(value: int, values: list[int], name: str) -> tuple[int, int]:
            if value < values[0] or value > values[-1]:
                raise ValueError(
                    f"HBF {key}: {name}={value} outside measured range "
                    f"[{values[0]}, {values[-1]}]"
                )
            index = bisect_left(values, value)
            return (value, value) if values[index] == value else (values[index - 1], values[index])

        b0, b1 = bracket(global_batch_size, sorted({b for b, _ in points}), "global_batch_size")
        s0, s1 = bracket(input_sequence_length, sorted({s for _, s in points}), "input_sequence_length")
        x = 0 if b0 == b1 else (global_batch_size - b0) / (b1 - b0)
        y = 0 if s0 == s1 else (input_sequence_length - s0) / (s1 - s0)
        try:
            low = points[b0, s0] * (1 - x) + points[b1, s0] * x
            high = points[b0, s1] * (1 - x) + points[b1, s1] * x
        except KeyError as error:
            raise ValueError(f"HBF {key}: missing interpolation corner {error}") from error
        return low * (1 - y) + high * y


def indexer_noc_bandwidth_per_pu(hardware: "HardwareConfig") -> float:
    """Equivalent Indexer read bandwidth: 64 bidirectional NoC links per chip."""
    # Indexer cache 可同时利用 HBF 和 HBM 的带宽；HBM 中的 cache 通过 NoC 共享。
    # 因此以 NoC 链路带宽 × 64 × 2 估算每卡共享 HBM 提供的额外带宽，
    # 再均摊到每 PU，与原 HBF 带宽相加；这里不单独模拟 HBM/NoC 传输。
    return (hardware.chip.noc.link_bandwidth_bytes_per_second * 64 * 2
            / hardware.chip.processing_unit_count)


def apply_indexer_noc_bandwidth(mapping: "HardwareMappingResult",
                                hardware: "HardwareConfig") -> "HardwareMappingResult":
    """Add the NoC contribution to Flash-backed Indexer cache reads once.

    This is an effective-bandwidth approximation, not a separate NoC transfer.
    DRAM-only Indexers, KV reads/prefetch and projection weights are unchanged.
    """
    operators = dict(mapping.operators)
    extra = indexer_noc_bandwidth_per_pu(hardware)
    for op_id in ("attn.indexer_score", "moe.indexer_score"):
        if op_id not in operators:
            continue
        op = operators[op_id]
        policy = op.weight_memory
        if policy is None or policy.source == WeightSource.DRAM:
            continue
        rate = (policy.flash_bandwidth_bytes_per_second
                or hardware.chip.logic_die.processing_unit.flash_bandwidth_bytes_per_second)
        operators[op_id] = replace(op, weight_memory=replace(
            policy, flash_bandwidth_bytes_per_second=rate + extra))
    return replace(mapping, operators=operators,
                   op_list=[operators[op.op_id] for op in mapping.op_list])


def apply_hbf_bandwidths(mapping: "HardwareMappingResult", hardware: "HardwareConfig",
                         table: HBFBandwidthTable) -> "HardwareMappingResult":
    """Apply measured cache rates, including Fold prefetch and legacy MLA reads."""
    if hardware.chip.processing_unit_count != MEASURED_PU_COUNT:
        raise ValueError("HBF measurements require 2048 PUs per chip")
    request = mapping.request
    inference = request.inference_config
    model = request.model_name
    operators = dict(mapping.operators)
    policies = dict(mapping.operator_mappings)
    for op_id, op in operators.items():
        if (op.weight_memory is not None
                and op.weight_memory.source == WeightSource.DRAM
                and not op.flash_prefetch_bytes_per_request):
            continue
        if op_id in ("attn.indexer_score", "moe.indexer_score"):
            kind = "dense_indexer"
        elif op_id in ("attn.flash_attention", "attn.flash_msa") or op.flash_prefetch_bytes_per_request:
            if model == "minimax-m3" and mapping.layer_type == "dense":
                warnings.warn(
                    "minimax-m3 dense KV has no HBF measurements; using hardware default Flash bandwidth",
                    UserWarning, stacklevel=2,
                )
                continue
            kind = "dense_kv" if model in ("deepseek-v3", "kimi-k2.5") else "sparse_kv"
        else:
            continue
        rate = table.lookup(model, kind, inference.mtp_enabled,
                            inference.global_batch_size, inference.input_sequence_length)
        policy = op.weight_memory or WeightMemory(WeightSource.FLASH)
        operators[op_id] = replace(op, weight_memory=replace(
            policy, flash_bandwidth_bytes_per_second=rate,
        ))
        if policies[op_id].dram_read_bytes_per_token:
            # Legacy MLA compute tiles include query heads. Read each cached
            # token once per request, uniformly striped across the chip's PUs.
            # This avoids charging every head (or MTP query) for a duplicate KV.
            bytes_per_request = (
                inference.input_sequence_length
                * (mapping.model_config.kv_lora_rank + mapping.model_config.qk_rope_head_dim)
                * inference.kv_cache_bytes_per_element / MEASURED_PU_COUNT
            )
            policies[op_id] = replace(
                policies[op_id], dram_read_bytes_per_token=0,
                dram_read_logical_bytes_per_token=None,
                # Legacy MLA PipeTree batches already describe one chip.
                demand_batch_partition_degree=1,
                cache_read_bytes_per_request=bytes_per_request,
            )
    additional = tuple(apply_hbf_bandwidths(item, hardware, table)
                       for item in mapping.additional_layer_mappings)
    result = replace(mapping, operators=operators, operator_mappings=policies,
                     op_list=[operators[op.op_id] for op in mapping.op_list],
                     additional_layer_mappings=additional)
    return apply_indexer_noc_bandwidth(result, hardware)
