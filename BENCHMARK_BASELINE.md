# Performance Benchmark Baseline

- **Date:** 2026-10-25
- **Git Commit:** $(git rev-parse --short HEAD)
- **Branch:** chore/env-baseline-verification
- **Hardware/OS:** Pop!_OS Linux (x86_64), 2 CPU workers assigned
- **Dataset:** 2 Complete Plastid Genomes (NC_000932.1, NC_001879.2)

## Baseline Metrics (CDS Mode, MAFFT)
- **Status:** Exit 0 (Success)
- **Wall Clock Time:** 10.31 seconds
- **User CPU Time:** 8.10 seconds
- **System CPU Time:** 18.42 seconds
- **CPU Utilization:** 257%
- **Peak RAM (Maximum Resident Set Size):** 78,568 KB (~78.5 MB)
- **Minor Page Faults:** 1,850,550
- **Context Switches:** 23,683 voluntary / 5,856 involuntary

## Test Suite Baseline
- **Execution Command:** `pytest tests/ -v`
- **Results:** 12 passed, 9 subtests failed (Upstream CLI argument mismatch: missing `--trna` / `--rrna` flags in `user_parameters.py`)
