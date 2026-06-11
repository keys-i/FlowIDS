#!/bin/bash
#SBATCH --job-name=flowids
#SBATCH --array=0-7
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=cuda80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00

# Run one configured model per Slurm array task
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
models=(M0 M0 M0 M1 M1 M2 M2 M2)
variants=(base small matched reconstruct teacher hybrid future-hybrid future-jepa)
index="${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array job}"
if [[ ! "$index" =~ ^[0-9]+$ ]] || (( index >= ${#variants[@]} )); then
    printf 'Unknown Slurm array task %s\n' "$index" >&2
    exit 1
fi
variant="${variants[index]}"
model="${models[index]}"
config="tools/config/$(printf '%s.%s' "$model" "$variant" | tr '[:upper:]' '[:lower:]').toml"
if [[ ! -f pixi.lock || ! -f "$config" ]]; then
    printf 'Submit from the project root containing pixi.lock and %s\n' "$config" >&2
    exit 1
fi
output="results/${model}-${variant}"
mkdir -p "$output"
exec >"$output/slurm-${SLURM_JOB_ID:?Slurm must provide a job ID}.log" 2>&1
export PYTHONUNBUFFERED=1

pixi install --frozen
srun pixi run --frozen python -c \
    'import sys, torch; sys.exit(0 if torch.cuda.is_available() else "CUDA is unavailable")'
srun pixi run --frozen model "$model" "$variant"
