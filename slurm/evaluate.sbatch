#!/bin/bash
# Evaluate a trained serial plan policy with scripts/plan_space/evaluate_serial.py on a Slurm cluster.
#
#   sbatch --export=ALL,DOMAIN_DIR=data/gripper,CHECKPOINT=runs/gripper/best.pt,EXTRA_ARGS="--max-rounds 60" \
#       --partition=<partition> slurm/evaluate.sbatch
#
# Variables:
#   VENV        path of the virtual environment to activate (default .venv)
#   DOMAIN_DIR  folder with domain.pddl and the problem folder (required)
#   CHECKPOINT  policy checkpoint (default ${OUTPUT}/best.pt; one of the two is required)
#   OUTPUT      run folder; results.json is written to ${RESULTS:-${OUTPUT}/results.json}
#   PROBLEMS    problem folder (default ${DOMAIN_DIR}/test)
#   DEVICE      torch device (default: cuda when a GPU is allocated, else cpu)
#   EXTRA_ARGS  additional evaluate_serial.py arguments (default empty)
#
# Resource defaults (override with sbatch --partition/--time/--mem/--cpus-per-task/--gres):
#SBATCH --job-name=plan-space-eval
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

VENV="${VENV:-.venv}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
: "${DOMAIN_DIR:?set DOMAIN_DIR}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT:?set CHECKPOINT or OUTPUT}/best.pt}"
PROBLEMS="${PROBLEMS:-${DOMAIN_DIR}/test}"
RESULTS="${RESULTS:-$(dirname "${CHECKPOINT}")/results.json}"

if [[ -z "${DEVICE:-}" ]]; then
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" || -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        DEVICE=cuda
    else
        DEVICE=cpu
    fi
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "host $(hostname) job ${SLURM_JOB_ID:-local} checkpoint ${CHECKPOINT} problems ${PROBLEMS} device ${DEVICE}"
# shellcheck disable=SC2086
python scripts/plan_space/evaluate_serial.py \
    --checkpoint "${CHECKPOINT}" \
    --domain-dir "${DOMAIN_DIR}" \
    --problems "${PROBLEMS}" \
    --output "${RESULTS}" \
    --device "${DEVICE}" \
    ${EXTRA_ARGS}
