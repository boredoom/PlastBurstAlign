#!/usr/bin/env bash
# ==============================================================================
# Master Benchmark Launcher: PlastBurstAlign Scaling Sweep
# ==============================================================================
set -euo pipefail

BASE_BENCH="/homes/arronweber/repos/PlastBurstAlign/benchmarking"
REPO_DIR="/homes/arronweber/repos/v2/PlastBurstAlign"
cd "${REPO_DIR}"

mkdir -p slurm_logs

echo "=================================================================="
echo "Submitting Full Benchmark Matrix to Beocat"
echo "Timestamp: $(date)"
echo "=================================================================="

# 1. Monocots (N = 733)
MONOCOTS="${BASE_BENCH}/benchmark_YangEtAl2022_monocots"
for mode in cds int igs; do
    echo ">>> Queuing Monocots | Mode: ${mode}"
    ./run_scaling_sweep_slurm.sh "${MONOCOTS}" "${mode}" 32G 01:00:00
done

# 2. Asteraceae (N ~ 2,500)
ASTER="${BASE_BENCH}/benchmark_YangEtAl2022_Asteraceae"
for mode in cds int igs; do
    echo ">>> Queuing Asteraceae | Mode: ${mode}"
    ./run_scaling_sweep_slurm.sh "${ASTER}" "${mode}" 48G 04:00:00
done

# 3. Angiosperms (N ~ 4,500)
ANGIO="${BASE_BENCH}/benchmark_YangEtAl2022_angiosperms"
for mode in cds int igs; do
    echo ">>> Queuing Angiosperms | Mode: ${mode}"
    ./run_scaling_sweep_slurm.sh "${ANGIO}" "${mode}" 64G 08:00:00
done

# 4. Zhang et al. (N = 13,724)
ZHANG="${BASE_BENCH}/benchmark_ZhangNNEtAl2025_13724angiosperms"
for mode in cds int; do
    echo ">>> Queuing Zhang 13k | Mode: ${mode}"
    ./run_scaling_sweep_slurm.sh "${ZHANG}" "${mode}" 96G 18:00:00
done

echo "=================================================================="
echo "All 55 benchmark jobs dispatched to SLURM."
echo "=================================================================="
