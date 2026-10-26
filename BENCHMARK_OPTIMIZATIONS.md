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

3. **Target 3: Staged Incremental Concatenation & Pooled Alignment Export**
   - Implemented `IncrementalNexusAccumulator` to sequentially stream individual NEXUS matrices into the accumulated alignment, replacing `Nexus.Nexus.combine()` which required simultaneous memory residency of all partitioned alignments.
   - Replaced child worker `multiprocessing.Process` invocations during concatenated FASTA and NEXUS disk exports with a concurrent `ThreadPoolExecutor(max_workers=2)`, eliminating IPC serialization and redundant memory duplication.
   - Successfully verified 77 concatenated CDS partitions (`nucl_77concat.aligned.*`) with zero data drift against monolithic combination.

---

## Controlled Benchmark: Full Concatenation (CDS Mode, MAFFT, 2 Workers)

| Metric | Baseline (`chore/env-baseline-verification`) | Staged Incremental (`main`) | Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **Elapsed Wall Clock** | 10.31 s | **5.41 s** | **-4.90 s (-47.5%)** |
| **User CPU Time** | 8.10 s | 5.65 s | -2.45 s |
| **System CPU Time** | 18.42 s | 8.69 s | -9.73 s (-52.8%) |
| **Peak RAM (Max RSS)** | 78,568 KB | **78,716 KB** | **+148 KB (+0.1%)** |
| **Voluntary Context Switches** | 23,683 | 22,880 | -803 |
| **Involuntary Context Switches**| 5,856 | 4,262 | -1,594 |
| **Partitions Concatenated** | None (Disabled) | 77 Partitions (Complete) | Parity verified with Nexus.Nexus.combine |
| **Pytest Suite** | 12 passed (9 errors) | 23 passed, 71 subtests passed | 100% pass rate |
