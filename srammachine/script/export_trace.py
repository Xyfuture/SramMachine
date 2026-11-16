"""Export an ordered model-layer baseline directly to Perfetto JSON."""

import argparse
from pathlib import Path

from srammachine.inference import InferenceConfig
from srammachine.mapping import (
    DEFAULT_HBF_BANDWIDTH_CSV, HBFBandwidthTable,
    HardwareMappingRequest, MiniMaxM3Mapper, mapper_for_model,
)
from srammachine.pipetree import TreeParser
from srammachine.simulator import Simulator
from srammachine.search import optimize_partition_chains
from srammachine.search.partition import SUPPORTED_PARTITION_MODELS


def main() -> None:
    """Simulate the mapper's sequential baseline and export its resource tracks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek-v3.2", choices=(
        "deepseek-v3", "deepseek-v3.2", "kimi-k2.5", "glm-5.1", "minimax-m3",
    ))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--input-sequence-length", type=int, default=20000)
    parser.add_argument("--output-sequence-length", type=int, default=600)
    parser.add_argument("--layer-count", type=int, default=4)
    parser.add_argument("--kv-cache-dtype", choices=("fp8", "fp16"), default="fp8")
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--xfold", action="store_true",
                        help="enable folding; DeepSeek V3 runs ordinary MLA")
    parser.add_argument(
        "--partition-optimization", action=argparse.BooleanOptionalAction, default=True,
        help="enumerate MLA/DSA/MiniMax per-op PartitionChains (default: enabled)",
    )
    hbf = parser.add_mutually_exclusive_group()
    hbf.add_argument("--hbf-bandwidth-csv", type=Path, default=DEFAULT_HBF_BANDWIDTH_CSV,
                     help="normalized HBF measurements (default: bundled table)")
    hbf.add_argument("--no-hbf-bandwidth", dest="hbf_bandwidth_csv", action="store_const",
                     const=None, help="disable measured HBF bandwidths")
    parser.add_argument("--minimax-layer-type", choices=("sparse", "dense"), default="sparse",
                        help="M3 trace layer type; dense reports only the dense prefix")
    parser.add_argument("--output-dir", type=Path, default=Path("test result/perfetto"))
    args = parser.parse_args()
    if args.xfold and args.model not in ("deepseek-v3", "deepseek-v3.2", "glm-5.1", "minimax-m3"):
        parser.error("--xfold supports deepseek-v3 (ordinary), deepseek-v3.2, glm-5.1 and minimax-m3")
    if args.xfold and args.model == "minimax-m3" and args.minimax_layer_type == "dense":
        parser.error("--xfold requires --minimax-layer-type sparse")
    if args.minimax_layer_type == "dense" and args.model != "minimax-m3":
        parser.error("--minimax-layer-type dense requires minimax-m3")
    for name in ("batch_size", "input_sequence_length", "output_sequence_length", "layer_count"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    mapper = mapper_for_model(args.model)
    if args.model == "minimax-m3":
        mapper = MiniMaxM3Mapper(layer_type=args.minimax_layer_type)
    request = HardwareMappingRequest(args.model, InferenceConfig(
        args.batch_size, args.input_sequence_length, args.output_sequence_length,
        kv_cache_dtype=args.kv_cache_dtype, mtp_enabled=args.mtp,
    ), xfold=args.xfold)
    table = (HBFBandwidthTable.from_csv(args.hbf_bandwidth_csv)
             if args.hbf_bandwidth_csv is not None else None)
    mapped = mapper.map(request, hbf_bandwidth_table=table)
    if args.partition_optimization:
        mapped = optimize_partition_chains(mapped, mapper.hardware_config)
    graph = TreeParser(mapper.hardware_config).parse(
        mapped.root_node, mapped.operators, mapped.operator_mappings,
        layer_count=args.layer_count,
    )
    label = (
        f"{args.model}_{'fold' if mapped.request.xfold else 'normal'}_baseline_"
        f"bs{args.batch_size}_isl{args.input_sequence_length}_"
        f"{args.kv_cache_dtype}_mtp_{'on' if args.mtp else 'off'}"
    )
    if args.model == "minimax-m3":
        label += f"_{args.minimax_layer_type}"
    if args.partition_optimization and args.model in SUPPORTED_PARTITION_MODELS:
        label += "_partition_optimized"
    Simulator(mapper.hardware_config).run_and_trace(
        graph, mapping_result=mapped, trace_label=label, output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
