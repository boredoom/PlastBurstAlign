#!/usr/bin/env bash
set -euo pipefail

BASE_BENCH="/homes/arronweber/repos/PlastBurstAlign/benchmarking"
REPO_DIR="/homes/arronweber/repos/v2/PlastBurstAlign"
cd "${REPO_DIR}"

mkdir -p slurm_logs

echo "=================================================================="
echo "Submitting Non-Coding Benchmark Sweeps (INT & IGS)"
echo "Timestamp: $(date)"
echo "=================================================================="

# 1. Monocots (N = 733) - IGS
MONOCOTS="${BASE_BENCH}/benchmark_YangEtAl2022_monocots"
echo ">>> Queuing Monocots | Mode: igs"
./run_scaling_sweep_slurm.sh "${MONOCOTS}" igs 32G 01:00:00

# 2. Asteraceae (N ~ 2,500) - IGS
ASTER="${BASE_BENCH}/benchmark_YangEtAl2022_Asteraceae"
echo ">>> Queuing Asteraceae | Mode: igs"
./run_scaling_sweep_slurm.sh "${ASTER}" igs 48G 02:00:00

# 3. Angiosperms (N ~ 4,500) - INT & IGS
ANGIO="${BASE_BENCH}/benchmark_YangEtAl2022_angiosperms"
for mode in int igs; do
    echo ">>> Queuing Angiosperms | Mode: ${mode}"
    ./run_scaling_sweep_slurm.sh "${ANGIO}" "${mode}" 64G 06:00:00
done

# 4. Zhang et al. (N = 13,724) - INT (Fast NW-NS-PartTree-1 with --nofft)
ZHANG="${BASE_BENCH}/benchmark_ZhangNNEtAl2025_13724angiosperms"
echo ">>> Queuing Zhang 13k | Mode: int"
./run_scaling_sweep_slurm.sh "${ZHANG}" int 96G 18:00:00

echo "=================================================================="
echo "All non-coding benchmark jobs submitted."
echo "=================================================================="
