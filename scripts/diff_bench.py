#!/usr/bin/env python3
"""Diff two benchmark reports produced by ``scripts/benchmark_stm32h743.py``.

Prints a per-metric delta table on stdout. Exit code is 0 on success or 1 if
any "phase B passes" gate fails — currently we only enforce that
``semantic_at_5`` does not regress.

Usage:
    python scripts/diff_bench.py validation/benchmarks/baseline.json \
                                 validation/benchmarks/phase-A.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


METRICS = [
    ("p_at_1", "P@1", "higher"),
    ("p_at_5", "P@5", "higher"),
    ("p_at_10", "P@10", "higher"),
    ("mrr", "MRR", "higher"),
    ("semantic_at_5", "Sem@5", "higher"),
    ("avg_distinct_pages_top10", "DistinctPages@10", "higher"),
    ("avg_query_wall_s", "QueryWall (s)", "lower"),
    ("index_wall_s", "IndexWall (s)", "lower"),
]


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _print_matrix(summaries: list[tuple[str, dict]], markdown: bool) -> int:
    """Print a metric-x-phase matrix when more than two reports are given."""
    labels = [label for _, label, _ in METRICS]
    header_names = [name for name, _ in summaries]
    if markdown:
        print("| Metric | " + " | ".join(header_names) + " |")
        print("|---|" + "|".join([":---:"] * len(header_names)) + "|")
        for key, label, direction in METRICS:
            row = f"| {label} | "
            cells: list[str] = []
            for _name, summary in summaries:
                v = float(summary.get(key) or 0)
                cells.append(f"{v:.3f}")
            print(row + " | ".join(cells) + " |")
    else:
        header = f"{'METRIC':22}" + "".join(f"{n:>14}" for n in header_names)
        print(header)
        print("-" * len(header))
        for key, label, direction in METRICS:
            cells: list[str] = []
            for _name, summary in summaries:
                v = float(summary.get(key) or 0)
                cells.append(f"{v:>14.3f}")
            print(f"{label:22}" + "".join(cells))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Diff two ragrag benchmark reports.")
    parser.add_argument("a", type=Path, nargs="?")
    parser.add_argument("b", type=Path, nargs="?")
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print a Markdown table suitable for PR bodies.",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        nargs="+",
        help="Print a metric-x-phase matrix for 2+ reports in order.",
    )
    args = parser.parse_args()

    if args.matrix:
        summaries = [
            (p.stem, _load(p).get("summary", {})) for p in args.matrix
        ]
        return _print_matrix(summaries, args.markdown)

    if not args.a or not args.b:
        parser.error("two positional reports required unless --matrix is given")

    summary_a = _load(args.a).get("summary", {})
    summary_b = _load(args.b).get("summary", {})

    rows = []
    regression = False
    for key, label, direction in METRICS:
        va = float(summary_a.get(key) or 0)
        vb = float(summary_b.get(key) or 0)
        delta = vb - va
        better = (delta >= 0) if direction == "higher" else (delta <= 0)
        marker = "✓" if better else "✗"
        rows.append((label, va, vb, delta, marker, direction))
        if direction == "higher" and key == "semantic_at_5" and delta < -1e-6:
            regression = True

    if args.markdown:
        print(f"| Metric | {args.a.stem} | {args.b.stem} | Δ |")
        print("|---|---:|---:|---:|")
        for label, va, vb, delta, marker, direction in rows:
            sign = "+" if delta >= 0 else ""
            print(f"| {label} | {va:.3f} | {vb:.3f} | {sign}{delta:.3f} {marker} |")
    else:
        print(f"{'METRIC':22} {args.a.stem:>14} {args.b.stem:>14} {'DELTA':>10}")
        print("-" * 64)
        for label, va, vb, delta, marker, direction in rows:
            sign = "+" if delta >= 0 else ""
            print(f"{label:22} {va:>14.3f} {vb:>14.3f} {sign}{delta:>9.3f} {marker}")
        if regression:
            print()
            print("WARN: semantic_at_5 regressed", file=sys.stderr)

    return 1 if regression else 0


if __name__ == "__main__":
    sys.exit(main())
