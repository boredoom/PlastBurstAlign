#!/usr/bin/env bash
# ==============================================================================
# PlastBurstAlign v2 (PR #43) - Universal Slurm Scaling Sweep
# Usage:
#    ./run_scaling_sweep_slurm.sh <DATASET_PATH> <MODE> [MEM_GB] [TIME_LIMIT]
# Examples:
#    ./run_scaling_sweep_slurm.sh .../benchmark_YangEtAl2022_monocots int 32G 06:00:00
#    ./run_scaling_sweep_slurm.sh .../benchmark_YangEtAl2022_angiosperm igs 32G 08:00:00
#    ./run_scaling_sweep_slurm.sh .../benchmark_ZhangNNEtAl2025_13724angiosperms int 64G 24:00:00
# ==============================================================================
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <DATASET_PATH> <MODE> [MEM_ALLOC] [TIME_LIMIT]"
    echo "  MODE: cds, int, igs, or rna"
    exit 1
fi

DATASET_PATH="$1"
MODE="$2"
MEM_ALLOC="${3:-32G}"
TIME_LIMIT="${4:-08:00:00}"

if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "ERROR: Dataset directory not found: ${DATASET_PATH}"
    exit 1
fi

DATASET_NAME=$(basename "${DATASET_PATH}" | sed 's/^benchmark_//')
OUT_BASE="benchmark_results_${DATASET_NAME}_${MODE}_v2"

mkdir -p slurm_logs "${OUT_BASE}"

echo "=================================================================="
echo "Initializing PBAv2 Scaling Sweep: ${DATASET_NAME} | Mode: ${MODE}"
echo "Input Directory : ${DATASET_PATH}"
echo "Output Directory: ${OUT_BASE}"
echo "Memory Ceiling  : ${MEM_ALLOC} | Time Limit: ${TIME_LIMIT}"
echo "=================================================================="

P_VALUES=(1 2 4 8 16)
SUBMITTED_JOBS=()

for P in "${P_VALUES[@]}"; do
    TRIAL_DIR="${OUT_BASE}/run_p${P}"
    JOB_SCRIPT="${OUT_BASE}/submit_p${P}.slurm"

    cat << SLURM_EOF > "${JOB_SCRIPT}"
#!/usr/bin/env bash
#SBATCH --job-name=pba_${DATASET_NAME:0:8}_${MODE}_p${P}
#SBATCH --partition=batch.q,killable.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${P}
#SBATCH --mem=${MEM_ALLOC}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=slurm_logs/${DATASET_NAME}_${MODE}_p${P}_%j.out
#SBATCH --error=slurm_logs/${DATASET_NAME}_${MODE}_p${P}_%j.err

set -euo pipefail

# Activate conda environment and pin internal BLAS/MKL threads to 1
source /homes/arronweber/miniforge3/etc/profile.d/conda.sh
conda activate plastburst-beocat
export PYTHONPATH="/homes/arronweber/repos/v2/PlastBurstAlign:${PYTHONPATH:-}"

export PATH="/homes/arronweber/miniforge3/envs/plastburst-beocat/bin:\$PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

FINAL_DIR="${TRIAL_DIR}"
TIME_LOG="\${FINAL_DIR}/profile_time_v.txt"

# 1. Allocate isolated node-local scratch on compute node NVMe/SSD
NODE_SCRATCH="\${SLURM_TMPDIR:-\${TMPDIR:-/tmp}}/pba_\${SLURM_JOB_ID}_p${P}"
rm -rf "\${NODE_SCRATCH}"
mkdir -p "\${NODE_SCRATCH}" "\${FINAL_DIR}"

# 2. Ensure local scratch is purged when job finishes or terminates
cleanup() {
    rm -rf "\${NODE_SCRATCH}"
}
trap cleanup EXIT

echo "=================================================================="
echo "PBAv2 Trial Initialized: p=${P} cores on \$(hostname)"
echo "Local Scratch: \${NODE_SCRATCH}"
echo "Network Dest : \${FINAL_DIR}"
echo "Dataset      : ${DATASET_PATH} | Mode: ${MODE}"
echo "Start Time   : \$(date)"
echo "=================================================================="

# 3. Execute pipeline on node-local NVMe
/usr/bin/time -v -o "\${TIME_LOG}" python3 -m plastburstalign \
    -i "${DATASET_PATH}" \
    -o "\${NODE_SCRATCH}" \
    -s "${MODE}" \
    -c \
    --adaptiveheuristics \
    --order alpha \
    --numthreads "${P}"

EXIT_CODE=\$?

# 4. Bulk stream all deliverables and matrices to network storage
echo "Staging deliverables to network storage..."
cp -a "\${NODE_SCRATCH}/." "\${FINAL_DIR}/"

echo "=================================================================="
echo "Trial p=${P} finished at \$(date) with exit code: \${EXIT_CODE}"
echo "=================================================================="
exit \${EXIT_CODE}
SLURM_EOF

    JOB_ID=$(sbatch "${JOB_SCRIPT}" | awk '{print $NF}')
    SUBMITTED_JOBS+=("${JOB_ID}")
    echo "  -> Submitted p=${P} job: Slurm ID ${JOB_ID} (${P} cores, ${MEM_ALLOC} RAM)"
done

echo ""
echo "All 5 scaling trials queued on Slurm."
echo "Active Jobs: ${SUBMITTED_JOBS[*]}"
echo "Monitor with: squeue -u arronweber"
