#!/bin/bash
# Train a plan-space policy with scripts/plan_space/train.py (imitation of planner
# references) or scripts/plan_space/train_hei.py (hindsight expert iteration, no
# planner) on a Slurm cluster.
#
# Configuration comes from environment variables (defaults below) passed with
# `sbatch --export`; the #SBATCH directives below are the resource defaults and
# can be overridden on the command line:
#
#   sbatch --export=ALL,DOMAIN_DIR=data/gripper,OUTPUT=runs/gripper,EXTRA_ARGS="--epochs 50 --dagger" \
#       --partition=<partition> --time=04:00:00 slurm/train.sbatch
#   sbatch --export=ALL,SCRIPT=scripts/plan_space/train_hei.py,DOMAIN_DIR=data/gripper,OUTPUT=runs/gripper_hei,EXTRA_ARGS="--iterations 12" \
#       --partition=<partition> --time=04:00:00 slurm/train.sbatch
#
# Variables:
#   SCRIPT      training script (default scripts/plan_space/train.py; use scripts/plan_space/train_hei.py for HEI)
#   VENV        path of the virtual environment to activate (default .venv)
#   DOMAIN_DIR  folder with domain.pddl, train/ and validation/ (required)
#   OUTPUT      folder for best.pt, latest.pt and metrics.jsonl (required)
#   SEED        random seed (default 0)
#   DEVICE      torch device (default: cuda when a GPU is allocated, else cpu)
#   EXTRA_ARGS  additional arguments of the training script (default empty)
#   ACCOUNT     optional Slurm account; use `sbatch --account=...` or set it here
#
# Resource defaults (override with sbatch --partition/--time/--mem/--cpus-per-task/--gres):
#SBATCH --job-name=plan-space-train
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT="${SCRIPT:-scripts/plan_space/train.py}"
VENV="${VENV:-.venv}"
SEED="${SEED:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
: "${DOMAIN_DIR:?set DOMAIN_DIR}"
: "${OUTPUT:?set OUTPUT}"

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

mkdir -p "${OUTPUT}"
echo "host $(hostname) job ${SLURM_JOB_ID:-local} script ${SCRIPT} domain ${DOMAIN_DIR} output ${OUTPUT} seed ${SEED} device ${DEVICE}"
# shellcheck disable=SC2086
python "${SCRIPT}" \
    --domain-dir "${DOMAIN_DIR}" \
    --output "${OUTPUT}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    ${EXTRA_ARGS}
