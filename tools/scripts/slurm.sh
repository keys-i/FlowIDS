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
#SBATCH --time=03:00:00

# Run one configured model per Slurm array task
set -euo pipefail

hours=3
mode=normal
action=model
if [[ "${1:-}" == max ]]; then
    hours=72
    mode=max
    shift
fi
if [[ "${1:-}" == timing ]]; then
    action=timing
    shift
fi
if [[ $# -gt 0 && "$1" != -* ]]; then
    printf 'Usage: bash tools/scripts/slurm.sh [max] [timing] [sbatch options]\n' >&2
    exit 1
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    script_args=()
    if [[ "$mode" == max ]]; then script_args+=(max); fi
    if [[ "$action" == timing ]]; then script_args+=(timing); fi
    exec sbatch --time="${hours}:00:00" "$@" "$0" "${script_args[@]}"
fi
if (( $# )); then
    printf 'Usage: bash tools/scripts/slurm.sh [max] [timing] [sbatch options]\n' >&2
    exit 1
fi

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
arguments=("$model" "$variant")
if (( hours == 72 )); then
    output="$output/max"
    arguments=(max "${arguments[@]}")
fi
if [[ "$action" == timing ]]; then
    output="timings/${model}-${variant}"
    arguments=(--array "$index" --mode "$mode")
fi
mkdir -p "$output"
exec >"$output/slurm-${SLURM_JOB_ID:?Slurm must provide a job ID}.log" 2>&1
export PYTHONUNBUFFERED=1

srun pixi run --frozen --no-install python -c \
    'import sys, torch; sys.exit(0 if torch.cuda.is_available() else "CUDA is unavailable")'
srun pixi run --frozen --no-install "$action" "${arguments[@]}"
