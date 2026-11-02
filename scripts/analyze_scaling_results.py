#!/usr/bin/env python3
"""Aggregates empirical scaling benchmarks (p in {1, 2, 4, 8, 16}) into CIS 625 proposal tables."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_time_output(file_path: Path) -> dict[str, float]:
    """Parses GNU /usr/bin/time -v output log."""
    metrics: dict[str, float] = {}
    if not file_path.exists():
        return metrics

    with open(file_path, "r") as f:
        content = f.read()

    # Wall-clock elapsed time (m:ss.ss or h:mm:ss)
    m_elapsed = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s+([\d:.]+)", content
    )
    if m_elapsed:
        raw_t = m_elapsed.group(1)
        parts = raw_t.split(":")
        if len(parts) == 3:
            metrics["elapsed_sec"] = (
                float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            )
        elif len(parts) == 2:
            metrics["elapsed_sec"] = float(parts[0]) * 60 + float(parts[1])

    # User CPU time
    m_user = re.search(r"User time \(seconds\):\s+([\d.]+)", content)
    if m_user:
        metrics["user_sec"] = float(m_user.group(1))

    # System CPU time
    m_sys = re.search(r"System time \(seconds\):\s+([\d.]+)", content)
    if m_sys:
        metrics["sys_sec"] = float(m_sys.group(1))

    # CPU utilization percentage
    m_cpu = re.search(r"Percent of CPU this job got:\s+([\d.]+)%", content)
    if m_cpu:
        metrics["cpu_percent"] = float(m_cpu.group(1))

    # Peak resident memory (kbytes -> GB)
    m_rss = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", content)
    if m_rss:
        metrics["peak_rss_gb"] = float(m_rss.group(1)) / (1024 * 1024)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="PBAv2 Scaling Sweep Aggregator")
    parser.add_argument(
        "-b",
        "--base-dir",
        required=True,
        help="Base directory containing run_p* outputs",
    )
    parser.add_argument(
        "-o", "--out-csv", default="scaling_metrics.csv", help="Target CSV output path"
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    p_values = [1, 2, 4, 8, 16]
    results: list[dict] = []

    t1_sec: float | None = None

    for p in p_values:
        trial_dir = base_dir / f"run_p{p}"
        time_file = trial_dir / "profile_time_v.txt"
        data = parse_time_output(time_file)

        if not data or "elapsed_sec" not in data:
            results.append({"p": p, "status": "IN_PROGRESS / PENDING"})
            continue

        if p == 1:
            t1_sec = data["elapsed_sec"]

        if t1_sec:
            speedup = t1_sec / data["elapsed_sec"]
            efficiency = (speedup / p) * 100.0
            speedup_str = f"{speedup:.2f}x"
            eff_str = f"{efficiency:.1f}%"
        else:
            speedup_str = "--"
            eff_str = "--"

        # Loci count verification across fasta/ or root
        fasta_dir = trial_dir / "fasta"
        loci_count = (
            len(list(fasta_dir.glob("*.aligned.fasta")))
            if fasta_dir.exists()
            else len(list(trial_dir.glob("*.aligned.fasta")))
        )

        row = {
            "p": p,
            "status": "COMPLETED",
            "elapsed_sec": data.get("elapsed_sec", 0.0),
            "wall_clock": f"{data.get('elapsed_sec', 0.0) / 60.0:.2f} min",
            "speedup": speedup_str,
            "efficiency": eff_str,
            "cpu_util": f"{data.get('cpu_percent', 0.0):.0f}%",
            "peak_rss_gb": f"{data.get('peak_rss_gb', 0.0):.2f} GB",
            "aligned_loci": loci_count,
        }
        results.append(row)

    # Print Formatted Markdown Table
    print("\n### PlastBurstAlign v2 (PR #43) Empirical Scaling Metrics")
    print(f"**Target Suite:** `{base_dir.name}`\n")
    print(
        "| Cores ($p$) | Status | Wall-Clock | Speedup ($S_p$) | Efficiency ($E_p$) | CPU Util (%) | Peak RAM | Aligned Loci |"
    )
    print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for r in results:
        if r["status"] != "COMPLETED":
            print(f"| {r['p']} | {r['status']} | -- | -- | -- | -- | -- | -- |")
        else:
            print(
                f"| {r['p']} | {r['status']} | {r['wall_clock']} | {r['speedup']} | {r['efficiency']} | {r['cpu_util']} | {r['peak_rss_gb']} | {r['aligned_loci']} |"
            )

    # Save to CSV
    csv_path = Path(args.out_csv)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "p",
                "status",
                "elapsed_sec",
                "wall_clock",
                "speedup",
                "efficiency",
                "cpu_util",
                "peak_rss_gb",
                "aligned_loci",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved raw scaling metrics to: `{csv_path}`")


if __name__ == "__main__":
    main()
