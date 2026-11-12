#!/bin/bash
# Submit one training job per (domain dir, seed) and, chained after each, its evaluation job.
#
#   DOMAIN_DIRS="data/blocks data/gripper" SEEDS="0 1 2" RUNS=runs PARTITION=<partition> TIME=04:00:00 \
#       EXTRA_ARGS="--iterations 15 --layers 8" slurm/submit_sweep.sh
#
# Variables:
#   DOMAIN_DIRS  space-separated domain folders (required)
#   SEEDS        space-separated seeds (default "0")
#   RUNS         root of the run folders; a run is ${RUNS}/<domain name>/seed<seed> (default runs)
#   PARTITION    Slurm partition (optional)
#   ACCOUNT      Slurm account (optional)
#   TIME         wall-clock limit of the training job (default 02:00:00)
#   EVAL_TIME    wall-clock limit of the evaluation job (default 01:00:00)
#   MEM          memory of each job (default 16G)
#   CPUS         CPUs of each job (default 4)
#   GPUS         GPUs of the training job, passed as --gres=gpu:${GPUS} (default 0: none)
#   VENV         virtual environment (default .venv)
#   EXTRA_ARGS   additional train_serial.py arguments (default empty)
#   EVAL_ARGS    additional evaluate_serial.py arguments (default empty)
#   DRY_RUN      set to 1 to print the sbatch commands without submitting

set -euo pipefail

: "${DOMAIN_DIRS:?set DOMAIN_DIRS}"
SEEDS="${SEEDS:-0}"
RUNS="${RUNS:-runs}"
TIME="${TIME:-02:00:00}"
EVAL_TIME="${EVAL_TIME:-01:00:00}"
MEM="${MEM:-16G}"
CPUS="${CPUS:-4}"
GPUS="${GPUS:-0}"
VENV="${VENV:-.venv}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EVAL_ARGS="${EVAL_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
common=(--mem="${MEM}" --cpus-per-task="${CPUS}")
[[ -n "${PARTITION:-}" ]] && common+=(--partition="${PARTITION}")
[[ -n "${ACCOUNT:-}" ]] && common+=(--account="${ACCOUNT}")
train_resources=("${common[@]}" --time="${TIME}")
[[ "${GPUS}" != "0" ]] && train_resources+=(--gres="gpu:${GPUS}")
eval_resources=("${common[@]}" --time="${EVAL_TIME}")

submit() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "sbatch $*" >&2
        echo "0"
    else
        sbatch --parsable "$@"
    fi
}

for domain_dir in ${DOMAIN_DIRS}; do
    domain_name="$(basename "${domain_dir}")"
    for seed in ${SEEDS}; do
        output="${RUNS}/${domain_name}/seed${seed}"
        mkdir -p "${output}"
        train_id="$(submit "${train_resources[@]}" \
            --job-name="train-${domain_name}-s${seed}" \
            --export=ALL,VENV="${VENV}",DOMAIN_DIR="${domain_dir}",OUTPUT="${output}",SEED="${seed}",EXTRA_ARGS="${EXTRA_ARGS}" \
            "${here}/train.sbatch")"
        eval_id="$(submit "${eval_resources[@]}" \
            --job-name="eval-${domain_name}-s${seed}" \
            --dependency="afterok:${train_id}" \
            --export=ALL,VENV="${VENV}",DOMAIN_DIR="${domain_dir}",OUTPUT="${output}",EXTRA_ARGS="${EVAL_ARGS}" \
            "${here}/evaluate.sbatch")"
        echo "${domain_name} seed ${seed}: train job ${train_id}, evaluation job ${eval_id} -> ${output}"
    done
done
