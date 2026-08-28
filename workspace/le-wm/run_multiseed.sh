#!/bin/bash
# Multi-seed comparison: baseline / uvlowr_r4 / rand_diff at 3 seeds × 500 steps.
set -e
export STABLEWM_HOME=/workspace/stablewm_home
export WANDB_MODE=offline

cd "$(dirname "$0")"
mkdir -p multiseed_runs

run() {
  local name=$1
  local seed=$2
  shift 2
  local outdir="multiseed_runs/${name}_seed${seed}"
  if [ -f "${outdir}/csv/version_0/metrics.csv" ] && [ "$(wc -l < ${outdir}/csv/version_0/metrics.csv)" -gt 30 ]; then
    echo "skip ${name} seed=${seed}"; return
  fi
  echo "=== ${name} seed=${seed} ==="
  /workspace/le-wm/.venv/bin/python train.py \
    data=pusht \
    seed=${seed} \
    loader.batch_size=16 loader.num_workers=0 loader.persistent_workers=False \
    loader.pin_memory=False loader.prefetch_factor=null \
    trainer.max_epochs=1 \
    +trainer.limit_train_batches=500 \
    +trainer.limit_val_batches=20 \
    wandb.enabled=False \
    output_model_name="${name}_seed${seed}" \
    hydra.job.id="${name}_seed${seed}" \
    hydra.run.dir="multiseed_runs/${name}_seed${seed}" \
    "$@" 2>&1 | tail -3
}

# Three seeds × three variants = 9 runs
for seed in 0 1 2; do
  run baseline   $seed
  run uvlowr_r4  $seed +predictor.ffn_rank=4
  run randdiff   $seed +loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2
done

echo "DONE_MULTISEED"
