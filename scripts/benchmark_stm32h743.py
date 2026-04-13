#!/usr/bin/env python3
"""STM32H743VI datasheet quality benchmark for ragrag.

Indexes the STM32H743VI datasheet in a temp directory (or reuses an existing
index), runs a fixed question set against the installed `ragrag` CLI, and
emits a JSON report with per-query grading and aggregate metrics.

Usage:
    python scripts/benchmark_stm32h743.py [--index-dir DIR] [--report report.json]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


QUESTIONS = [
    {
        "id": "Q01_max_junction_temp",
        "query": "maximum junction temperature TJ",
        "expected_pages": (105, 105),
        "must_contain_any": ["125", "TJ"],
        "difficulty": "E",
    },
    {
        "id": "Q02_min_vdd",
        "query": "minimum VDD operating voltage range",
        "expected_pages": (105, 106),
        "must_contain_any": ["1.62", "3.6", "VDD"],
        "difficulty": "E",
    },
    {
        "id": "Q03_max_cpu_clock_vs_voltage",
        "query": "maximum CPU clock frequency at VOS0 voltage scaling",
        "expected_pages": (105, 130),
        "must_contain_any": ["480", "MHz", "VOS"],
        "difficulty": "M",
    },
    {
        "id": "Q04_flash_wait_states",
        "query": "flash memory access wait states at 200 MHz",
        "expected_pages": (131, 135),
        "must_contain_any": ["wait state", "WS", "latency"],
        "difficulty": "M",
    },
    {
        "id": "Q05_adc_resolution_sample_rate",
        "query": "ADC resolution and maximum sampling rate",
        "expected_pages": (166, 170),
        "must_contain_any": ["16-bit", "MSPS", "3.6", "4.5"],
        "difficulty": "E",
    },
    {
        "id": "Q06_adc_inl",
        "query": "ADC integral non-linearity INL specification",
        "expected_pages": (167, 170),
        "must_contain_any": ["INL", "LSB"],
        "difficulty": "M",
    },
    {
        "id": "Q07_idd_run_400mhz",
        "query": "supply current IDD in run mode at 400 MHz",
        "expected_pages": (109, 121),
        "must_contain_any": ["IDD", "Run", "mA"],
        "difficulty": "H",
    },
    {
        "id": "Q08_idd_stop_mode",
        "query": "current consumption in stop mode",
        "expected_pages": (109, 122),
        "must_contain_any": ["Stop", "µA", "uA", "IDD"],
        "difficulty": "M",
    },
    {
        "id": "Q09_gpio_drive_current",
        "query": "GPIO output drive current and slew rate",
        "expected_pages": (134, 140),
        "must_contain_any": ["I/O", "drive", "mA"],
        "difficulty": "M",
    },
    {
        "id": "Q10_vrefbuf_accuracy",
        "query": "VREFBUF internal reference voltage accuracy",
        "expected_pages": (170, 176),
        "must_contain_any": ["VREFBUF", "ppm", "%"],
        "difficulty": "M",
    },
    {
        "id": "Q11_temp_sensor_calibration",
        "query": "internal temperature sensor calibration values",
        "expected_pages": (174, 178),
        "must_contain_any": ["TS_CAL", "calibration"],
        "difficulty": "M",
    },
    {
        "id": "Q12_dac_settling_time",
        "query": "DAC settling time and output impedance",
        "expected_pages": (171, 174),
        "must_contain_any": ["settling", "DAC", "µs", "us"],
        "difficulty": "M",
    },
]


def run_ragrag(
    cli: str,
    index_dir: Path,
    args: list[str],
    timeout: int = 7200,
) -> tuple[dict, float, str]:
    """Run the ragrag CLI, return (parsed_json, wall_seconds, stderr_tail)."""
    t0 = time.monotonic()
    proc = subprocess.run(
        [cli] + args,
        cwd=str(index_dir),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    wall = time.monotonic() - t0
    if proc.returncode != 0:
        sys.stderr.write(f"ragrag exited {proc.returncode}\nstdout: {proc.stdout[:500]}\nstderr: {proc.stderr[-2000:]}\n")
        proc.check_returncode()
    try:
        resp = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"failed to parse stdout as JSON: {e}\nstdout head: {proc.stdout[:500]}\n")
        raise
    return resp, wall, proc.stderr[-2000:]


def grade(question: dict, response: dict, wall_s: float) -> dict:
    results = response.get("results", [])
    expected_lo, expected_hi = question["expected_pages"]
    in_range = lambda p: p is not None and expected_lo <= p <= expected_hi
    pages_top10 = [r.get("page") for r in results[:10]]

    top1 = bool(results) and in_range(results[0].get("page"))
    top5 = any(in_range(r.get("page")) for r in results[:5])
    top10 = any(in_range(r.get("page")) for r in results[:10])

    mrr = 0.0
    for i, r in enumerate(results, start=1):
        if in_range(r.get("page")):
            mrr = 1.0 / i
            break

    must = [s.lower() for s in question["must_contain_any"]]
    excerpts5 = " ".join((r.get("excerpt") or "").lower() for r in results[:5])
    semantic5 = any(s in excerpts5 for s in must)

    timing = response.get("timing_ms", {}) or {}
    return {
        "id": question["id"],
        "query": question["query"],
        "difficulty": question["difficulty"],
        "expected_pages": list(question["expected_pages"]),
        "top1": top1,
        "top5": top5,
        "top10": top10,
        "mrr": mrr,
        "semantic5": semantic5,
        "distinct_pages_top10": len({p for p in pages_top10 if p is not None}),
        "top1_modality": results[0].get("modality") if results else None,
        "top1_page": results[0].get("page") if results else None,
        "top1_excerpt_head": (results[0].get("excerpt") or "")[:160] if results else "",
        "wall_s": round(wall_s, 3),
        "indexing_ms": int(timing.get("indexing_ms", 0)),
        "query_embedding_ms": int(timing.get("query_embedding_ms", 0)),
        "retrieval_ms": int(timing.get("retrieval_ms", 0)),
    }


def summarize(rows: list[dict], index_wall_s: float) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    by_diff: dict[str, list[dict]] = {"E": [], "M": [], "H": []}
    for r in rows:
        by_diff[r["difficulty"]].append(r)

    def avg(rs: list[dict], k: str) -> float:
        if not rs:
            return 0.0
        vals = [float(r[k]) for r in rs]
        return round(sum(vals) / len(vals), 4)

    return {
        "n": n,
        "p_at_1": avg(rows, "top1"),
        "p_at_5": avg(rows, "top5"),
        "p_at_10": avg(rows, "top10"),
        "mrr": avg(rows, "mrr"),
        "semantic_at_5": avg(rows, "semantic5"),
        "avg_distinct_pages_top10": avg(rows, "distinct_pages_top10"),
        "avg_query_wall_s": avg(rows, "wall_s"),
        "by_difficulty": {
            d: {
                "n": len(rs),
                "p_at_1": avg(rs, "top1"),
                "p_at_5": avg(rs, "top5"),
                "mrr": avg(rs, "mrr"),
                "semantic_at_5": avg(rs, "semantic5"),
            }
            for d, rs in by_diff.items() if rs
        },
        "index_wall_s": round(index_wall_s, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run STM32H743VI quality benchmark.")
    ap.add_argument("--index-dir", default="/tmp/stm32h743_bench", help="Directory holding the PDF and .ragrag index.")
    ap.add_argument("--pdf", default="/home/pavel/Downloads/stm32h743vi.pdf", help="Source PDF path.")
    ap.add_argument("--report", default="bench_report.json", help="Report JSON output path.")
    ap.add_argument("--cli", default="ragrag", help="Path to ragrag CLI (default: from PATH).")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--reset", action="store_true", help="Delete .ragrag in index dir before running.")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    pdf_dest = index_dir / Path(args.pdf).name
    if not pdf_dest.exists():
        print(f"Copying {args.pdf} -> {pdf_dest}", flush=True)
        shutil.copyfile(args.pdf, pdf_dest)

    if args.reset:
        ragrag_dir = index_dir / ".ragrag"
        if ragrag_dir.exists():
            print(f"Resetting {ragrag_dir}", flush=True)
            shutil.rmtree(ragrag_dir)

    needs_new = not (index_dir / ".ragrag").exists()
    print(f"Cold call (indexing if needed) — needs_new={needs_new}", flush=True)
    cold_args = ["warmup query", "--top-k", str(args.top_k), "--log-level", "INFO"]
    if needs_new:
        cold_args.append("--new")
    try:
        cold_resp, index_wall, _ = run_ragrag(args.cli, index_dir, cold_args, timeout=7200)
    except subprocess.CalledProcessError:
        return 2
    cold_indexing_ms = int((cold_resp.get("timing_ms") or {}).get("indexing_ms", 0))
    print(f"  cold wall {index_wall:.1f}s; indexing_ms={cold_indexing_ms}", flush=True)

    rows = []
    print(f"Running {len(QUESTIONS)} queries...", flush=True)
    for q in QUESTIONS:
        print(f"  {q['id']}: {q['query']}", flush=True)
        try:
            resp, wall, _ = run_ragrag(
                args.cli, index_dir,
                [q["query"], "--top-k", str(args.top_k), "--log-level", "WARNING"],
                timeout=300,
            )
        except subprocess.CalledProcessError as e:
            print(f"    FAILED: {e}", flush=True)
            continue
        row = grade(q, resp, wall)
        rows.append(row)
        verdict = []
        if row["top1"]: verdict.append("top1")
        if row["top5"]: verdict.append("top5")
        if row["semantic5"]: verdict.append("sem")
        verdict_s = ",".join(verdict) or "MISS"
        print(f"    [{verdict_s}] page={row['top1_page']} mod={row['top1_modality']} mrr={row['mrr']:.2f} wall={row['wall_s']:.1f}s", flush=True)
        print(f"    excerpt: {row['top1_excerpt_head']!r}", flush=True)

    summary = summarize(rows, index_wall)
    report = {"summary": summary, "rows": rows, "cold_indexing_ms": cold_indexing_ms}
    Path(args.report).write_text(json.dumps(report, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nReport written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
