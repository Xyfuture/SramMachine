"""Normalize measured HBF CSV names and convert chip GB/s to PU bytes/s."""

import argparse
from pathlib import Path

from srammachine.mapping.hbf_bandwidth import normalize_hbf_csv


def main() -> None:
    """Export a normalized CSV beside the source unless an output is specified."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.source.with_name(args.source.stem + "-normalized.csv")
    count = normalize_hbf_csv(args.source, output)
    print(f"Exported {count} measurements to {output}")


if __name__ == "__main__":
    main()
