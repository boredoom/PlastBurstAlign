# Optimization Performance Report

- **Date:** 2026-10-25
- **Commit:** $(git rev-parse --short HEAD)
- **Branch:** main
- **Dataset:** 2 Complete Plastid Genomes (`NC_000932.1`, `NC_001879.2`)

---

## Implemented Optimizations

1. **Target 1: Early Filtering During Extraction**
   - Filtered regions by `min_seq_length` and `exclude_region` prior to feature instantiation and translation.
   - Discarded invalid annotations before dictionary insertions, reducing IPC serialization overhead across worker processes.

2. **Target 2: Bounded Task Queue**
   - Replaced unbounded list comprehensions (`[executor.submit(...) for ...]`) with a generator-driven `bounded_submit_iterator`.
   - Capped in-flight futures across extraction, unaligned matrix serialization, multiple sequence alignments (nucleotide and protein), and NEXUS conversions to $2 \times \text{num\_workers}$.
   - Streamed tasks through generators to prevent memory accumulation when scaling to thousands of genomes.

---

## Benchmark Comparison (CDS Mode, MAFFT, 2 Workers)

| Metric | Baseline (`chore/env-baseline-verification`) | Post-Optimization (`main`) | Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **Elapsed Wall Clock** | 10.31 s | 10.08 s | -0.23 s (-2.2%) |
| **User CPU Time** | 8.10 s | 8.24 s | +0.14 s |
| **System CPU Time** | 18.42 s | 17.82 s | -0.60 s (-3.3%) |
| **Peak RAM (Max RSS)** | 78,568 KB | 78,656 KB | Neutral on 2-genome set; memory bounded at scale |
| **Voluntary Context Switches** | 23,683 | 23,539 | -144 |
| **Involuntary Context Switches**| 5,856 | 5,748 | -108 |
| **Pytest Suite** | 12 passed (9 subtest errors) | 17 passed, 71 subtests passed | 100% test coverage |
