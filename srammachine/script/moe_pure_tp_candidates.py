"""Offline communication sweep for the fixed, pure-PU-TP MoE mapping.

Run ``python -m srammachine.script.moe_pure_tp_candidates`` to regenerate the
adjacent CSV.  This is intentionally not called by the runtime mapper.
"""

import csv
import json
from fractions import Fraction
from pathlib import Path

from srammachine.hardware import DEFAULT_HARDWARE_CONFIG
from srammachine.mapping import HardwareMapper, load_model_config


MODELS = ("deepseek-v3", "deepseek-v3.2", "kimi-k2.5", "glm-5.1")
BATCHES = (4096, 8192)
MULTIPLIERS = (1, 2)
CHIP_K = (1, 2, 4, 8, 16)
DIE_K = (1, 2, 4)
PU_K = (1, 2, 4, 8, 16)
CSV_PATH = Path(__file__).with_name("moe_pure_tp_candidates.csv")


def _ceil_fraction(value):
    return (value.numerator + value.denominator - 1) // value.denominator


def _stage(name, direction, size_bytes, critical_bytes, *, links=1,
           participants=1):
    hardware = DEFAULT_HARDWARE_CONFIG
    bandwidth = (
        hardware.inter_chip_fabric.per_chip_directional_bandwidth_bytes_per_second
        if direction == "fabric" else
        hardware.chip.noc.link_bandwidth_bytes_per_second * links
    )
    duration_ns = _ceil_fraction(
        Fraction(critical_bytes * 1_000_000_000, bandwidth)
    )
    if direction != "fabric":
        duration_ns += _ceil_fraction(Fraction(hardware.chip.noc.link_latency_ns))
    return {
        "stage": name, "direction": direction, "size_bytes": size_bytes,
        "critical_bytes": float(critical_bytes), "participants": participants,
        "parallel_links": links, "duration_ns": duration_ns,
    }


def _critical_alltoall(matrix):
    count = len(matrix)
    sent = [sum(row[j] for j in range(count) if j != i)
            for i, row in enumerate(matrix)]
    received = [sum(matrix[i][j] for i in range(count) if i != j)
                for j in range(count)]
    return max(sent + received)


def _candidate(model_name, strategy, batch, multiplier, chip_k, die_k, pu_k):
    model = load_model_config(model_name)
    hardware = DEFAULT_HARDWARE_CONFIG
    chips = hardware.chip_count
    dies = hardware.chip.logic_die_count
    pu_count = hardware.chip.logic_die.processing_unit_count
    chip_n = chips // chip_k if strategy == "tp" else 1
    die_n = dies // die_k
    pu_n = pu_count // pu_k
    global_batch = batch * multiplier
    local_batch = batch // chips * multiplier
    loads = HardwareMapper._balanced_expert_loads(
        global_batch, model.top_k, model.num_experts,
    )
    if strategy == "ep":
        expert_per_chip = model.num_experts // chips
        ownership = tuple(
            tuple(range(i * expert_per_chip, (i + 1) * expert_per_chip))
            for i in range(chips)
        )
        loads = HardwareMapper._balanced_ep_expert_loads(
            global_batch, model.top_k, model.num_experts, ownership,
        )
        groups = HardwareMapper._groups_for_experts(ownership[0], loads)
    else:
        groups = HardwareMapper._groups_for_experts(range(model.num_experts), loads)
    stages = []
    add = lambda *args, **kwargs: stages.append(_stage(*args, **kwargs))
    h = model.hidden_size
    intermediate = model.moe_intermediate_size
    byte = 1  # FP8 activations; weights are also one byte per element.
    if strategy == "tp":
        entry_shard = local_batch * h // chip_k * byte
        add("entry_16chip_alltoall", "fabric", entry_shard,
            (chips - 1) * entry_shard, participants=chips)
    else:
        effective = sum(loads)
        rows = (effective // chips,) * chips
        cols = tuple(sum(loads[i] for i in ids) for ids in ownership)
        matrix = HardwareMapper._transport_matrix(rows, cols, h * byte)
        add("ep_dispatch", "fabric", sum(map(sum, matrix)),
            _critical_alltoall(matrix), participants=chips)
    for expert_ids, token_count in groups:
        em = len(expert_ids) * token_count
        label = f"tokens{token_count}"
        hidden_die_bytes = em * h // (chip_k * die_k) * byte
        up_bytes = em * 2 * intermediate // (chip_n * die_n) * byte
        down_input_bytes = em * intermediate // (chip_n * die_n) * byte
        add(f"{label}.up_input", "noc_input", hidden_die_bytes,
            hidden_die_bytes, participants=2 if pu_k > 1 else pu_count)
        add(f"{label}.up_output", "noc_output", up_bytes,
            up_bytes, participants=2 if pu_k == 1 else pu_k)
        if die_k > 1:
            add(f"{label}.up_die_reduce_scatter", "noc_output", up_bytes,
                Fraction(die_k - 1, die_k) * up_bytes,
                links=4, participants=die_k)
            add(f"{label}.up_die_allgather", "noc_input", up_bytes // die_k,
                (die_k - 1) * (up_bytes // die_k),
                links=4, participants=die_k)
        if chip_k > 1:
            add(f"{label}.up_chip_allreduce", "fabric", up_bytes,
                Fraction(2 * (chip_k - 1), chip_k) * up_bytes,
                participants=chip_k)
        add(f"{label}.down_input", "noc_input", down_input_bytes,
            down_input_bytes, participants=2 if pu_k == 1 else pu_k)
        add(f"{label}.down_output", "noc_output", hidden_die_bytes,
            hidden_die_bytes, participants=2 if pu_n == 1 else pu_n)
        if die_n > 1:
            add(f"{label}.down_die_reduce_scatter", "noc_output",
                hidden_die_bytes,
                Fraction(die_n - 1, die_n) * hidden_die_bytes,
                links=4, participants=die_n)
    if strategy == "tp":
        hidden = global_batch * h // chip_k * byte
        if chip_n > 1:
            add("exit_chip_reduce_scatter", "fabric", hidden,
                Fraction(chip_n - 1, chip_n) * hidden,
                participants=chip_n)
        if chip_k > 1:
            shard = local_batch * h // chip_k * byte
            exchanged = (chip_k - 1) * shard
            add("exit_chip_hidden_exchange", "fabric", shard,
                exchanged, participants=chip_k)
            add("exit_die_distribute", "noc_input", exchanged,
                exchanged, participants=2)
    else:
        add("ep_combine", "fabric", sum(map(sum, matrix)),
            _critical_alltoall(matrix), participants=chips)
    final_shard = local_batch * h // dies * byte
    add("final_die_allgather", "noc_input", final_shard,
        (dies - 1) * final_shard, links=4, participants=dies)
    totals = {key: sum(stage["duration_ns"] for stage in stages
                       if stage["direction"] == key)
              for key in ("fabric", "noc_input", "noc_output")}
    return {
        "model": model_name, "strategy": strategy, "batch_size": batch,
        "mtp": "on" if multiplier == 2 else "off",
        "chip_k": chip_k, "chip_n": chip_n, "die_k": die_k, "die_n": die_n,
        "pu_k": pu_k, "pu_n": pu_n,
        "up_pu_B": len(groups[0][0]), "up_pu_M": groups[0][1],
        "up_pu_K": h // (chip_k * die_k * pu_k),
        "up_pu_N": 2 * intermediate // (chip_n * die_n * pu_n),
        "down_pu_K": intermediate // (chip_n * die_n * pu_n),
        "down_pu_N": h // (chip_k * die_k * pu_k),
        "die_weight_bytes": (
            model.num_experts * h // (chip_k * die_k)
            * (2 * intermediate // (chip_n * die_n))
            + model.num_experts * intermediate // (chip_n * die_n)
            * (h // (chip_k * die_k))
        ) if strategy == "tp" else (
            (model.num_experts // chips) * h // die_k
            * (2 * intermediate // die_n)
            + (model.num_experts // chips) * intermediate // die_n
            * (h // die_k)
        ),
        "fabric_ns": totals["fabric"],
        "noc_input_ns": totals["noc_input"],
        "noc_output_ns": totals["noc_output"],
        "total_ns": sum(totals.values()),
        "collective_count": len(stages),
        "stages_json": json.dumps(stages, separators=(",", ":")),
    }


def main():
    rows = []
    for model in MODELS:
        for strategy in ("tp", "ep"):
            chip_factors = CHIP_K if strategy == "tp" else (1,)
            for batch in BATCHES:
                for multiplier in MULTIPLIERS:
                    for chip_k in chip_factors:
                        for die_k in DIE_K:
                            for pu_k in PU_K:
                                rows.append(_candidate(
                                    model, strategy, batch, multiplier,
                                    chip_k, die_k, pu_k,
                                ))
    # This experiment deliberately fixes the user-requested factors rather
    # than selecting the communication minimum from the offline sweep.
    choices = {
        (model, strategy): (1, 4, 2) if strategy == "tp" else (1, 2, 1)
        for model in MODELS for strategy in ("tp", "ep")
    }
    for row in rows:
        row["selected"] = (
            (row["chip_k"], row["die_k"], row["pu_k"])
            == choices[row["model"], row["strategy"]]
        )
        row["selection_reason"] = (
            "user-requested chip N16, intra-chip K8xN8 comparison"
            if row["selected"] and row["strategy"] == "tp"
            else "existing EP K2xN32" if row["selected"]
            else "alternative candidate"
        )
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for key, value in choices.items():
        print(key, value)


if __name__ == "__main__":
    main()
