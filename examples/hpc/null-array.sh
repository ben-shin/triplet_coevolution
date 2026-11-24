#!/usr/bin/env bash
#SBATCH --job-name=triplet-null
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Supply --array, --cpus-per-task, --time, --mem and --output at submission.
set -euo pipefail
: "${TRIPLET_BIN:?absolute path to installed triplet}"
: "${TRIPLET_PLAN:?absolute path to the prepared shard plan}"
: "${SLURM_ARRAY_TASK_ID:?zero-based shard index}"
: "${SLURM_CPUS_PER_TASK:?allocated CPUs for this task}"
[[ "$TRIPLET_BIN" == /* && -x "$TRIPLET_BIN" && "$TRIPLET_PLAN" == /* ]]
[[ "$SLURM_ARRAY_TASK_ID" =~ ^(0|[1-9][0-9]*)$ && "$SLURM_CPUS_PER_TASK" =~ ^[1-9][0-9]*$ ]]
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 GOTO_NUM_THREADS=1
printf 'Slurm job %s, array task %s, CPUs %s\n' "${SLURM_ARRAY_JOB_ID:-local}" "$SLURM_ARRAY_TASK_ID" "$SLURM_CPUS_PER_TASK" >&2
"$TRIPLET_BIN" --version >&2
exec "$TRIPLET_BIN" null-shard --plan "$TRIPLET_PLAN" --shard-id "$SLURM_ARRAY_TASK_ID" \
    --workers "$SLURM_CPUS_PER_TASK" --resume --progress
