"""Command-line wrapper for the U(1)' repair census.

Quick demo:
    .venv/bin/python scripts/repair_map.py

Paper-scale run (parallel over all cores):
    .venv/bin/python scripts/repair_map.py --radius 2 --max-added 6 \
        --timeout-ms 8000 --output markdown
"""
import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsm.repair_map import (
    TIER_ORDER,
    build_atlas,
    canonical_grid,
    classify_grid,
    csv_rows,
    integer_grid,
    latex_summary_table,
    summary_markdown,
)


def main(radius=1, max_added=4, timeout_ms=4000, output="summary", workers=None):
    if workers is None:
        workers = os.cpu_count() or 1
    atlas = build_atlas(radius=radius, max_added=max_added, timeout_ms=timeout_ms,
                        workers=workers)
    if output == "csv":
        print(csv_rows(atlas), end="")
        return
    if output == "latex":
        print(latex_summary_table(atlas), end="")
        return
    if output == "markdown":
        print(summary_markdown(atlas), end="")
        return

    counts = atlas.counts
    print(f"U(1)' repair census: {atlas.raw_count} raw cells -> "
          f"{atlas.distinct_count} primitive sign-fixed reps "
          f"({atlas.dedup_ratio:.2f}x dedup)")
    print(f"Parameters: {atlas.parameters()}\n")
    for tier in TIER_ORDER:
        if counts.get(tier):
            example = next(cell for cell in atlas.cells if cell.tier == tier)
            added = list(example.added)
            ex = f"   e.g. X={example.charges} -> {added}" if added else ""
            print(f"  {tier:30} {counts[tier]:5}{ex}")
    print(f"\nUncertified cells: {atlas.uncertified_count}/{atlas.distinct_count}")
    print("Note: budget-limited means no repair was found within max_added; it "
          "is not an unrepairability claim.")


def _parse_args():
    parser = argparse.ArgumentParser(description="Build a U(1)' repair-census summary.")
    parser.add_argument("positional", nargs="*",
                        help="legacy form: radius max_added timeout_ms output workers")
    parser.add_argument("--radius", type=int, default=None,
                        help="integer charge radius N (default: 1)")
    parser.add_argument("--max-added", type=int, default=None,
                        help="maximum added Weyl multiplets (default: 4)")
    parser.add_argument("--timeout-ms", type=int, default=None,
                        help="per-SMT-call timeout in milliseconds (default: 4000)")
    parser.add_argument("--output", choices=("summary", "markdown", "latex", "csv"),
                        default=None, help="output format (default: summary)")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel worker count (default: os.cpu_count())")
    args = parser.parse_args()

    legacy = list(args.positional)
    if len(legacy) > 5:
        parser.error("expected at most five positional arguments")

    radius = args.radius if args.radius is not None else int(legacy[0]) if len(legacy) > 0 else 1
    max_added = args.max_added if args.max_added is not None else int(legacy[1]) if len(legacy) > 1 else 4
    timeout_ms = args.timeout_ms if args.timeout_ms is not None else int(legacy[2]) if len(legacy) > 2 else 4000
    output = args.output if args.output is not None else legacy[3] if len(legacy) > 3 else "summary"
    workers = args.workers if args.workers is not None else int(legacy[4]) if len(legacy) > 4 else None
    if output not in {"summary", "markdown", "latex", "csv"}:
        parser.error(f"invalid output format: {output}")
    return radius, max_added, timeout_ms, output, workers


if __name__ == "__main__":
    main(*_parse_args())
