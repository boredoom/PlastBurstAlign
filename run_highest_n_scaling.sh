#!/usr/bin/env bash
set -euo pipefail

DATASET="/homes/arronweber/repos/PlastBurstAlign/benchmarking/benchmark_ZhangNNEtAl2025_13724angiosperms"
MODE="cds"  # Or "int"
OUT_BASE="benchmark_results_zhang13724"
mkdir -p slurm_logs "${OUT_BASE}"

for P in 1 2 4 8 16; do
  MEM=$(( P * 4 ))
  if [ "${MEM}" -lt 16 ]; then MEM=16; fi

  JOB_SCRIPT="${OUT_BASE}/submit_p${P}.slurm"
  cat << SLURM_EOF > "${JOB_SCRIPT}"
#!/usr/bin/env bash
#SBATCH --job-name=pba_z13k_p${P}
#SBATCH --partition=batch.q,killable.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${P}
#SBATCH --mem=${MEM}G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/z13k_p${P}_%j.out
#SBATCH --error=slurm_logs/z13k_p${P}_%j.err

source /homes/arronweber/miniforge3/etc/profile.d/conda.sh
conda activate plastburst-beocat

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

TRIAL_DIR="${OUT_BASE}/run_p${P}"
rm -rf "\${TRIAL_DIR}"
mkdir -p "\${TRIAL_DIR}"

echo "=========================================================="
echo "PlastBurstAlign v2 Benchmark: n=13,724 | p=${P} cores"
echo "Host: \$(hostname) | Start: \$(date)"
echo "=========================================================="

/usr/bin/time -v python3 -m plastburstalign \
  -i "${DATASET}" \
  -o "\${TRIAL_DIR}" \
  -s "${MODE}" \
  --numthreads "${P}"

echo "=========================================================="
echo "Completed p=${P} run at: \$(date)"
echo "=========================================================="
SLURM_EOF

  JOB_ID=$(sbatch "${JOB_SCRIPT}" | awk '{print $NF}')
  echo "Submitted p=$P trial (Cores: $P, RAM: ${MEM}G) -> Slurm Job ID: ${JOB_ID}"
done

echo ""
echo "All 5 benchmark trials queued concurrently. Monitor with: squeue --me"
