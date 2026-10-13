#!/usr/bin/env python3
"""
Collect per_pos_mse arrays from all per_pos_mse.json files under a base directory
and write a consolidated JSON file with a mapping from relative file path (or a
short name) to the per_pos_mse list and a few metadata fields.

Usage:
    python3 scripts/collect_per_pos_mse.py \
        --base /media/disk7t/PADP1018/accurency_test/data/eval/test_dp_h40/pmse \
        --out /media/disk7t/PADP1018/accurency_test/data/eval/test_dp_h40/pmse/per_pos_mse_aggregated.json

If no arguments are provided, the script uses the test_dp_h40/pmse directory in the repo.
"""

import argparse
import json
from pathlib import Path
import sys


def find_and_collect(base_dir: Path):
    results = {}
    for p in sorted(base_dir.rglob('per_pos_mse.json')):
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            print(f"WARNING: failed to read/parse {p}: {e}", file=sys.stderr)
            continue
        if 'per_pos_mse' not in data:
            print(f"WARNING: no 'per_pos_mse' in {p}", file=sys.stderr)
            continue
        # short name: last two path parts if possible, else file stem
        parts = p.parts
        short_name = p.parent.name
        # also include relative path from base_dir for disambiguation
        try:
            rel = str(p.relative_to(base_dir))
        except Exception:
            rel = str(p)
        results[rel] = {
            'short_name': short_name,
            'per_pos_mse': data.get('per_pos_mse'),
            'horizon': data.get('horizon'),
            'count': data.get('count'),
            'time_steps': data.get('time_steps')
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path,
                        default=Path('/media/disk7t/PADP1018/accurency_test/data/eval/test_dp_h40/pmse'),
                        help='Base directory to search (defaults to test_dp_h40/pmse)')
    parser.add_argument('--out', type=Path,
                        default=Path('/media/disk7t/PADP1018/accurency_test/data/eval/test_dp_h40/pmse/per_pos_mse_aggregated.json'),
                        help='Output JSON file path')
    args = parser.parse_args()

    base = args.base
    if not base.exists():
        print(f"ERROR: base dir {base} does not exist", file=sys.stderr)
        sys.exit(2)

    collected = find_and_collect(base)
    if not collected:
        print("No per_pos_mse.json files found or none with 'per_pos_mse' field.")

    # Ensure parent exists
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'base': str(base), 'files': collected}, indent=2))
    print(f"Wrote aggregated data for {len(collected)} files to {args.out}")


if __name__ == '__main__':
    main()
