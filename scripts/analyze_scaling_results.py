#!/usr/bin/env python3
"""analyze_scaling_results.py — ingest run manifests and emit scaling metrics.

Supports:
  - Primary: glob-based harvesting of ``run_manifest.json`` artifacts.
  - Legacy fallback: GNU ``/usr/bin/time -v`` logs under ``--base-dir``
    (``run_p{N}/profile_time_v.txt``) when manifests are missing or incomplete.
  - Metrics: Speedup (S_p), Parallel Efficiency (E_p), Karp–Flatt serial fraction.
  - Outputs: Markdown/CSV summary tables and vector plots (PDF/SVG).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional analysis stack (pandas / matplotlib / numpy). Core package stays lean.
# ---------------------------------------------------------------------------

try:
    import matplotlib

    # Headless / HPC-safe default (Beocat login nodes, CI, no DISPLAY).
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    HAS_ANALYSIS_DEPS = True
except ImportError:  # pragma: no cover - exercised when bench extras absent
    HAS_ANALYSIS_DEPS = False
    plt = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]


def _require_analysis_deps() -> None:
    if HAS_ANALYSIS_DEPS:
        return
    print(
        "Analysis dependencies (pandas, matplotlib, numpy) are missing.\n"
        "Install them with:  pip install -e '.[bench]'",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _karp_flatt(speedup: float | None, procs: int) -> float | None:
    """Karp–Flatt serial fraction; prefer package helper when importable."""
    try:
        from plastburstalign.scaling import karp_flatt as _kf

        return _kf(speedup, procs)
    except Exception:
        if procs <= 1 or speedup is None or not math.isfinite(speedup) or speedup <= 0:
            return None
        e = (1.0 / speedup - 1.0 / procs) / (1.0 - 1.0 / procs)
        return min(1.0, max(0.0, e))


# ---------------------------------------------------------------------------
# Legacy GNU /usr/bin/time -v parser (kept for Beocat sweeps without RSS)
# ---------------------------------------------------------------------------


def parse_time_output(file_path: Path) -> dict[str, float]:
    """Parse GNU ``/usr/bin/time -v`` output log into numeric metrics."""
    metrics: dict[str, float] = {}
    if not file_path.exists():
        return metrics

    content = file_path.read_text(encoding="utf-8", errors="replace")

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

    m_user = re.search(r"User time \(seconds\):\s+([\d.]+)", content)
    if m_user:
        metrics["user_sec"] = float(m_user.group(1))

    m_sys = re.search(r"System time \(seconds\):\s+([\d.]+)", content)
    if m_sys:
        metrics["sys_sec"] = float(m_sys.group(1))

    m_cpu = re.search(r"Percent of CPU this job got:\s+([\d.]+)%", content)
    if m_cpu:
        metrics["cpu_percent"] = float(m_cpu.group(1))

    m_rss = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", content)
    if m_rss:
        # Keep both GB (legacy table) and MB (manifest-aligned).
        kbytes = float(m_rss.group(1))
        metrics["peak_rss_gb"] = kbytes / (1024 * 1024)
        metrics["peak_rss_mb"] = kbytes / 1024.0

    return metrics


def harvest_legacy_base_dir(base_dir: Path) -> list[dict[str, Any]]:
    """Harvest ``run_p{N}/profile_time_v.txt`` trees (legacy GNU-time path)."""
    records: list[dict[str, Any]] = []
    if not base_dir.is_dir():
        return records

    dataset = base_dir.name
    for trial_dir in sorted(base_dir.glob("run_p*")):
        if not trial_dir.is_dir():
            continue
        m = re.fullmatch(r"run_p(\d+)", trial_dir.name)
        if not m:
            continue
        cores = int(m.group(1))
        time_file = trial_dir / "profile_time_v.txt"
        data = parse_time_output(time_file)
        if "elapsed_sec" not in data:
            continue

        fasta_dir = trial_dir / "fasta"
        if fasta_dir.is_dir():
            loci_count = len(list(fasta_dir.glob("*.aligned.fasta")))
        else:
            loci_count = len(list(trial_dir.glob("*.aligned.fasta")))

        # Prefer colocated run_manifest.json when present (fill gaps).
        manifest_path = trial_dir / "run_manifest.json"
        ntax = nchar = None
        mode = "legacy"
        status = "COMPLETE"
        cpu_seconds = data.get("user_sec", 0.0) + data.get("sys_sec", 0.0)
        peak_rss_mb = data.get("peak_rss_mb")
        if manifest_path.is_file():
            try:
                with open(manifest_path, encoding="utf-8") as handle:
                    doc = json.load(handle) or {}
                sm = doc.get("supermatrix") or {}
                ntax = sm.get("ntax")
                nchar = sm.get("nchar")
                loci_count = sm.get("n_loci") or sm.get("nloci") or loci_count
                params = doc.get("params") or {}
                mode = (
                    doc.get("mode")
                    or params.get("select_mode")
                    or (doc.get("run") or {}).get("mode")
                    or mode
                )
                status = (doc.get("run") or {}).get("status") or status
                sched = doc.get("scheduler") or {}
                if sched.get("total_cpu_seconds") is not None:
                    cpu_seconds = float(sched["total_cpu_seconds"])
                peak_bytes = sched.get("peak_memory_bytes")
                if peak_bytes and not peak_rss_mb:
                    peak_rss_mb = float(peak_bytes) / (1024 * 1024)
            except Exception as exc:
                sys.stderr.write(
                    f"Warning: Failed to enrich legacy run from {manifest_path}: {exc}\n"
                )

        records.append(
            {
                "manifest_path": str(manifest_path) if manifest_path.is_file() else "",
                "source": "legacy_time",
                "dataset": dataset,
                "mode": str(mode),
                "cores": cores,
                "makespan": float(data["elapsed_sec"]),
                "total_cpu_seconds": float(cpu_seconds) if cpu_seconds else None,
                "peak_rss_mb": float(peak_rss_mb) if peak_rss_mb is not None else None,
                "ntax": ntax,
                "nchar": nchar,
                "nloci": loci_count,
                "status": status,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Manifest harvest
# ---------------------------------------------------------------------------


def _bytes_to_mb(peak_bytes: Any) -> float | None:
    if peak_bytes is None:
        return None
    try:
        value = float(peak_bytes)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value / (1024.0 * 1024.0)


def _record_from_manifest(path: Path, doc: dict[str, Any]) -> dict[str, Any] | None:
    sched = doc.get("scheduler") or {}
    tuning = doc.get("tuning") or {}
    sm = doc.get("supermatrix") or {}
    run_meta = doc.get("run") or {}
    params = doc.get("params") or {}

    cores = sched.get("total_cores")
    if cores is None:
        cores = tuning.get("cpu_budget")
    makespan = sched.get("makespan")
    if makespan is None:
        makespan = run_meta.get("wall_seconds")
    cpu_secs = sched.get("total_cpu_seconds")

    peak_rss_mb: float | None = _bytes_to_mb(sched.get("peak_memory_bytes"))
    if peak_rss_mb is None:
        resource = doc.get("resource_usage") or {}
        if resource.get("max_rss_bytes") is not None:
            peak_rss_mb = _bytes_to_mb(resource.get("max_rss_bytes"))
        elif resource.get("peak_rss_mb") is not None:
            try:
                peak_rss_mb = float(resource["peak_rss_mb"])
            except (TypeError, ValueError):
                peak_rss_mb = None

    # Infer dataset / mode from path when not embedded in the manifest.
    if len(path.parts) >= 3:
        ds_name = doc.get("dataset_name") or path.parts[-3].replace(
            "benchmark_results_", ""
        )
    else:
        ds_name = doc.get("dataset_name") or path.parent.name
    mode = (
        doc.get("mode")
        or params.get("select_mode")
        or run_meta.get("mode")
        or "unknown"
    )

    ntax = sm.get("ntax") if isinstance(sm, dict) else None
    status = run_meta.get("status")
    if not status:
        status = "COMPLETE" if ntax else "PARTIAL"

    if cores is None and makespan is None:
        return None

    return {
        "manifest_path": str(path),
        "source": "manifest",
        "dataset": str(ds_name),
        "mode": str(mode),
        "cores": int(cores) if cores is not None else None,
        "makespan": float(makespan) if makespan is not None else None,
        "total_cpu_seconds": float(cpu_secs) if cpu_secs is not None else None,
        "peak_rss_mb": peak_rss_mb,
        "ntax": ntax,
        "nchar": sm.get("nchar") if isinstance(sm, dict) else None,
        "nloci": (sm.get("n_loci") or sm.get("nloci"))
        if isinstance(sm, dict)
        else None,
        "status": str(status).upper(),
    }


def _status_rank(status: str | None) -> int:
    """Higher is better when deduplicating dataset/mode/cores."""
    token = (status or "").upper()
    if token == "COMPLETE":
        return 3
    if token == "FAILED":
        return 2
    if token == "PARTIAL":
        return 1
    return 0


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per (dataset, mode, cores); prefer COMPLETE over PARTIAL."""
    best: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for rec in records:
        key = (rec.get("dataset"), rec.get("mode"), rec.get("cores"))
        prev = best.get(key)
        if prev is None:
            best[key] = rec
            continue
        rank_new = _status_rank(rec.get("status"))
        rank_old = _status_rank(prev.get("status"))
        if rank_new > rank_old:
            best[key] = rec
            continue
        if rank_new == rank_old:
            # Prefer rows with a makespan, then longer makespan stability via path.
            new_ms = rec.get("makespan")
            old_ms = prev.get("makespan")
            if old_ms is None and new_ms is not None:
                best[key] = rec
            elif new_ms is not None and old_ms is not None:
                # Prefer COMPLETE already tied; break ties by non-null RSS / path.
                if (
                    prev.get("peak_rss_mb") is None
                    and rec.get("peak_rss_mb") is not None
                ):
                    best[key] = rec
    return list(best.values())


def harvest_manifests(patterns: list[str]) -> list[dict[str, Any]]:
    """Load run manifests matching one or more glob patterns."""
    records: list[dict[str, Any]] = []
    matched_files: list[str] = []
    for pat in patterns:
        matched_files.extend(glob.glob(pat, recursive=True))

    for path_str in sorted(set(matched_files)):
        path = Path(path_str)
        try:
            with open(path, encoding="utf-8") as handle:
                doc = json.load(handle) or {}
            if not isinstance(doc, dict):
                continue
            rec = _record_from_manifest(path, doc)
            if rec is not None:
                records.append(rec)
        except Exception as exc:
            sys.stderr.write(f"Warning: Failed to parse {path}: {exc}\n")
    return records


# ---------------------------------------------------------------------------
# Metrics + reporting
# ---------------------------------------------------------------------------


def compute_metrics(df: Any) -> Any:
    """Attach S_p, E_p, and Karp–Flatt columns grouped by dataset/mode."""
    _require_analysis_deps()
    assert pd is not None and np is not None

    if df.empty:
        return df.copy()

    processed: list[Any] = []
    for (_ds, _mode), group in df.groupby(["dataset", "mode"], dropna=False):
        group = group.sort_values("cores").copy()
        t1_rows = group[group["cores"] == 1]
        t1_val = None
        if not t1_rows.empty:
            raw = t1_rows["makespan"].iloc[0]
            if pd.notna(raw) and float(raw) > 0:
                t1_val = float(raw)

        speedups: list[float] = []
        efficiencies: list[float] = []
        karps: list[float] = []
        for _, row in group.iterrows():
            p = row["cores"]
            tp = row["makespan"]
            if (
                t1_val is not None
                and p is not None
                and pd.notna(p)
                and pd.notna(tp)
                and float(tp) > 0
                and int(p) > 0
            ):
                sp = t1_val / float(tp)
                ep = sp / float(p)
                kf = _karp_flatt(sp, int(p))
                e = (
                    float(kf)
                    if kf is not None
                    else (0.0 if int(p) <= 1 else float("nan"))
                )
            else:
                sp = ep = e = float("nan")
            speedups.append(sp)
            efficiencies.append(ep)
            karps.append(e)
        group["speedup"] = speedups
        group["efficiency"] = efficiencies
        group["karp_flatt"] = karps
        processed.append(group)

    if not processed:
        return pd.DataFrame()
    return pd.concat(processed, ignore_index=True)


def dataframe_to_markdown(df: Any) -> str:
    """Render a compact Markdown table without requiring tabulate."""
    _require_analysis_deps()
    assert pd is not None

    cols = [
        c
        for c in [
            "dataset",
            "mode",
            "cores",
            "status",
            "makespan",
            "speedup",
            "efficiency",
            "karp_flatt",
            "peak_rss_mb",
            "ntax",
            "nloci",
        ]
        if c in df.columns
    ]
    if df.empty or not cols:
        return "_No scaling rows available._\n"

    view = df[cols].copy()
    for col in ("makespan", "speedup", "efficiency", "karp_flatt", "peak_rss_mb"):
        if col in view.columns:
            view[col] = view[col].map(
                lambda x: (
                    ""
                    if x is None or (isinstance(x, float) and not math.isfinite(x))
                    else (f"{float(x):.4g}")
                )
            )

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in view.iterrows():
        cells = [str(row[c]) if pd.notna(row[c]) else "" for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_scaling_figures(df: Any, out_dir: Path) -> list[Path]:
    """Write fig1 (strong scaling), fig2 (memory), fig3 (Karp–Flatt) as PDF+SVG."""
    _require_analysis_deps()
    assert plt is not None and pd is not None

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    style = (
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    plt.style.use(style)

    # Fig 1: Speedup vs cores
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for (ds, mode), grp in df.groupby(["dataset", "mode"], dropna=False):
        valid = grp.dropna(subset=["cores", "speedup"]).sort_values("cores")
        if valid.empty:
            continue
        ax.plot(
            valid["cores"],
            valid["speedup"],
            marker="o",
            linewidth=2,
            label=f"{ds} ({mode})",
        )
    all_cores = sorted({int(c) for c in df["cores"].dropna().unique()})
    if all_cores:
        max_c = max(all_cores)
        ax.plot([1, max_c], [1, max_c], "k--", alpha=0.6, label="Ideal linear")
        ax.set_xticks(all_cores)
    ax.set_xlabel("Physical cores ($p$)")
    ax.set_ylabel(r"Speedup ($S_p = T_1 / T_p$)")
    ax.set_title("PlastBurstAlign strong scaling")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        path = out_dir / f"fig1_strong_scaling.{ext}"
        fig.savefig(path)
        written.append(path)
    plt.close(fig)

    # Fig 2: Memory vs taxon count
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    plotted = False
    for (ds, mode), grp in df.groupby(["dataset", "mode"], dropna=False):
        valid = grp.dropna(subset=["ntax", "peak_rss_mb"])
        if valid.empty:
            continue
        plotted = True
        ax.scatter(
            valid["ntax"],
            valid["peak_rss_mb"] / 1024.0,
            s=60,
            alpha=0.8,
            label=f"{ds} ({mode})",
        )
    ax.set_xlabel("Taxon count ($N$)")
    ax.set_ylabel("Peak memory (GiB)")
    ax.set_title("Memory-bounded streaming invariance")
    if plotted:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No peak RSS / ntax pairs", ha="center", va="center")
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        path = out_dir / f"fig2_memory_scaling.{ext}"
        fig.savefig(path)
        written.append(path)
    plt.close(fig)

    # Fig 3: Karp–Flatt vs cores
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for (ds, mode), grp in df.groupby(["dataset", "mode"], dropna=False):
        valid = grp.dropna(subset=["cores", "karp_flatt"]).sort_values("cores")
        valid = valid[valid["cores"] > 1]
        if valid.empty:
            continue
        ax.plot(
            valid["cores"],
            valid["karp_flatt"],
            marker="s",
            linewidth=2,
            label=f"{ds} ({mode})",
        )
    if all_cores:
        ax.set_xticks([c for c in all_cores if c > 1] or all_cores)
    ax.set_xlabel("Physical cores ($p$)")
    ax.set_ylabel(r"Karp–Flatt serial fraction ($e$)")
    ax.set_title("Karp–Flatt contention profile")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        path = out_dir / f"fig3_karp_flatt.{ext}"
        fig.savefig(path)
        written.append(path)
    plt.close(fig)

    return written


def _legacy_only_table(base_dir: Path, out_csv: Path) -> None:
    """Original CIS-625 markdown/CSV path when only ``--base-dir`` is used."""
    p_values = [1, 2, 4, 8, 16]
    results: list[dict[str, Any]] = []
    t1_sec: float | None = None

    for p in p_values:
        trial_dir = base_dir / f"run_p{p}"
        data = parse_time_output(trial_dir / "profile_time_v.txt")
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
        fasta_dir = trial_dir / "fasta"
        loci_count = (
            len(list(fasta_dir.glob("*.aligned.fasta")))
            if fasta_dir.exists()
            else len(list(trial_dir.glob("*.aligned.fasta")))
        )
        results.append(
            {
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
        )

    print("\n### PlastBurstAlign Empirical Scaling Metrics (legacy GNU-time)")
    print(f"**Target Suite:** `{base_dir.name}`\n")
    print(
        "| Cores ($p$) | Status | Wall-Clock | Speedup ($S_p$) | "
        "Efficiency ($E_p$) | CPU Util (%) | Peak RAM | Aligned Loci |"
    )
    print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in results:
        if r["status"] != "COMPLETED":
            print(f"| {r['p']} | {r['status']} | -- | -- | -- | -- | -- | -- |")
        else:
            print(
                f"| {r['p']} | {r['status']} | {r['wall_clock']} | {r['speedup']} | "
                f"{r['efficiency']} | {r['cpu_util']} | {r['peak_rss_gb']} | "
                f"{r['aligned_loci']} |"
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
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
    print(f"\nSaved raw scaling metrics to: `{out_csv}`")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate PlastBurstAlign run manifests (primary) or GNU /usr/bin/time "
            "logs (legacy --base-dir) into scaling metrics and publication figures."
        )
    )
    parser.add_argument(
        "--manifest-glob",
        nargs="+",
        default=None,
        help=(
            "Glob pattern(s) for run_manifest.json files "
            "(default: benchmark_results_*/run_*/run_manifest.json)."
        ),
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        type=Path,
        default=None,
        help=(
            "Legacy mode: directory containing run_p{N}/profile_time_v.txt. "
            "Also merges GNU-time RSS/makespan when manifests lack those fields."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scaling_analysis_output"),
        help="Directory for CSV, Markdown, and figures (default: scaling_analysis_output).",
    )
    parser.add_argument(
        "-o",
        "--out-csv",
        type=Path,
        default=None,
        help="Optional explicit CSV path (defaults to <output-dir>/scaling_metrics_summary.csv).",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Optional figure directory (defaults to --output-dir).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation.",
    )
    parser.add_argument(
        "--legacy-only",
        action="store_true",
        help="Force the original GNU-time table path (requires --base-dir).",
    )
    args = parser.parse_args(argv)

    if args.legacy_only:
        if args.base_dir is None:
            parser.error("--legacy-only requires --base-dir")
        out_csv = args.out_csv or (args.output_dir / "scaling_metrics.csv")
        _legacy_only_table(Path(args.base_dir), out_csv)
        return 0

    patterns = args.manifest_glob or ["benchmark_results_*/run_*/run_manifest.json"]
    records = harvest_manifests(patterns)

    # Merge / fill from legacy GNU-time tree when provided.
    if args.base_dir is not None:
        legacy = harvest_legacy_base_dir(Path(args.base_dir))
        if not records:
            records = legacy
        else:
            # Index legacy by cores for RSS/makespan gap-fill on same dataset.
            by_cores: dict[int, dict[str, Any]] = {}
            for rec in legacy:
                if rec.get("cores") is not None:
                    by_cores[int(rec["cores"])] = rec
            for rec in records:
                cores = rec.get("cores")
                if cores is None:
                    continue
                fill = by_cores.get(int(cores))
                if fill is None:
                    continue
                if rec.get("makespan") is None and fill.get("makespan") is not None:
                    rec["makespan"] = fill["makespan"]
                    rec["source"] = "manifest+legacy"
                if (
                    rec.get("peak_rss_mb") is None
                    and fill.get("peak_rss_mb") is not None
                ):
                    rec["peak_rss_mb"] = fill["peak_rss_mb"]
                    rec["source"] = "manifest+legacy"
            # Add legacy-only core counts not present in manifests.
            seen_cores = {
                int(r["cores"]) for r in records if r.get("cores") is not None
            }
            for cores, fill in by_cores.items():
                if cores not in seen_cores:
                    records.append(fill)

    if not records:
        print("No run manifests found matching specified pattern(s).")
        if args.base_dir is not None:
            print(f"(Also no usable GNU-time logs under {args.base_dir})")
        return 0

    records = dedupe_records(records)
    _require_analysis_deps()
    assert pd is not None

    df_raw = pd.DataFrame(records)
    # Drop rows that cannot participate in scaling (need cores + makespan).
    df_raw = df_raw.dropna(subset=["cores", "makespan"]).copy()
    if not df_raw.empty:
        df_raw["cores"] = df_raw["cores"].astype(int)

    df_metrics = compute_metrics(df_raw)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = (
        Path(args.out_csv) if args.out_csv else out_dir / "scaling_metrics_summary.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(csv_path, index=False)
    print(f"Metrics table exported to {csv_path}")

    md_path = out_dir / "scaling_metrics_summary.md"
    md_body = (
        "# PlastBurstAlign scaling metrics\n\n"
        f"Rows: {len(df_metrics)} after dedupe "
        f"(prefer COMPLETE over PARTIAL).\n\n" + dataframe_to_markdown(df_metrics)
    )
    md_path.write_text(md_body, encoding="utf-8")
    print(f"Markdown summary exported to {md_path}")
    print(md_body)

    if not args.no_plots and not df_metrics.empty:
        fig_dir = Path(args.fig_dir) if args.fig_dir else out_dir
        paths = plot_scaling_figures(df_metrics, fig_dir)
        print(f"Vector figures saved ({len(paths)} files) under {fig_dir}/")
    elif args.no_plots:
        print("Skipping plots (--no-plots).")
    else:
        print("No plottable rows; skipping figures.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
