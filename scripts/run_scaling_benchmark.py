#!/usr/bin/env python3
"""Automated strong scaling benchmark harness for PlastBurstAlign on Beocat.

Evaluates speedup and parallel efficiency across p in {1, 2, 4, 8, 16}.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def run_benchmark_trial(
    in_dir: str,
    base_out_dir: str,
    p: int,
    select_mode: str = "cds",
    alignment_tool: str = "mafft",
    extra_flags: list[str] | None = None,
) -> dict[str, float | int | str]:
    trial_dir = os.path.join(base_out_dir, f"run_p{p}")
    if os.path.exists(trial_dir):
        shutil.rmtree(trial_dir)
    os.makedirs(trial_dir, exist_ok=True)

    cmd = [
        "python3",
        "-m",
        "plastburstalign",
        "-i",
        in_dir,
        "-o",
        trial_dir,
        "-s",
        select_mode,
        "--alignmenttool",
        alignment_tool,
        "--numthreads",
        str(p),
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    log_path = os.path.join(trial_dir, "benchmark_run.log")
    print(f"\n[p={p:>2}] Launching: {' '.join(cmd)}")

    start_time = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed_sec = time.perf_counter() - start_time

    if proc.returncode != 0:
        print(f"[p={p:>2}] FAILED (Exit Code {proc.returncode}). See {log_path}")
        return {
            "p": p,
            "status": "FAILED",
            "walltime_sec": elapsed_sec,
            "speedup": 0.0,
            "efficiency": 0.0,
            "output_dir": trial_dir,
        }

    print(f"[p={p:>2}] COMPLETED in {elapsed_sec:.2f}s")
    return {
        "p": p,
        "status": "SUCCESS",
        "walltime_sec": elapsed_sec,
        "speedup": 0.0,
        "efficiency": 0.0,
        "output_dir": trial_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Run PBA strong scaling benchmark.")
    parser.add_argument(
        "-i", "--in-dir", required=True, help="Input directory containing GenBank files"
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        default="benchmark_results",
        help="Base directory for benchmark outputs",
    )
    parser.add_argument(
        "-s",
        "--select-mode",
        default="cds",
        choices=["cds", "igs", "rna"],
        help="Feature selection mode",
    )
    parser.add_argument(
        "-t", "--alignmenttool", default="mafft", help="Alignment tool binary name"
    )
    parser.add_argument(
        "--cores",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16],
        help="List of core/thread counts to test",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results: list[dict[str, float | int | str]] = []

    print("================================================================")
    print("        PlastBurstAlign Strong Scaling Benchmark (Beocat)       ")
    print("================================================================")
    print(f"Input Dataset : {Path(args.in_dir).resolve()}")
    print(f"Output Root   : {Path(args.out_dir).resolve()}")
    print(f"Selection Mode: {args.select_mode}")
    print(f"Cores (p)     : {args.cores}")
    print("----------------------------------------------------------------")

    t1_time: float | None = None

    for p in args.cores:
        res = run_benchmark_trial(
            in_dir=args.in_dir,
            base_out_dir=args.out_dir,
            p=p,
            select_mode=args.select_mode,
            alignment_tool=args.alignmenttool,
        )
        if p == 1 and res["status"] == "SUCCESS":
            t1_time = float(res["walltime_sec"])

        if t1_time and res["status"] == "SUCCESS":
            wall = float(res["walltime_sec"])
            speedup = t1_time / wall
            eff = speedup / p
            res["speedup"] = round(speedup, 3)
            res["efficiency"] = round(eff, 3)

        results.append(res)

    # Output formatted Markdown summary
    print("\n================================================================")
    print("                     BENCHMARK SCALING RESULTS                  ")
    print("================================================================")
    print(
        f"| {'p (cores)':<9} | {'Status':<8} | {'Walltime (s)':<14} | {'Speedup (Sp)':<12} | {'Efficiency (Ep)':<15} |"
    )
    print(f"|{'-' * 11}|{'-' * 10}|{'-' * 16}|{'-' * 14}|{'-' * 17}|")
    for r in results:
        wall_str = (
            f"{r['walltime_sec']:.2f}"
            if isinstance(r["walltime_sec"], float)
            else str(r["walltime_sec"])
        )
        print(
            f"| {r['p']:<9} | {r['status']:<8} | {wall_str:<14} | {r['speedup']:<12} | {r['efficiency']:<15} |"
        )
    print("================================================================\n")

    # Save CSV and JSON artifacts
    csv_file = os.path.join(args.out_dir, "scaling_summary.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "p",
                "status",
                "walltime_sec",
                "speedup",
                "efficiency",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    json_file = os.path.join(args.out_dir, "scaling_summary.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Artifacts saved to:\n  - {csv_file}\n  - {json_file}")


if __name__ == "__main__":
    main()
