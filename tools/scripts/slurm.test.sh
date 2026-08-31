#!/bin/bash
# M2 hybrid, future-hybrid and future-jepa occupy array entries 5-7
exec sbatch "$@" --array=5-7 --time=03:00:00 "$(dirname "$0")/slurm.sh"
