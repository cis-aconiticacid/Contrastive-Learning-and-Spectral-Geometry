#!/bin/bash
# Real-LeWM baseline vs uv_lowr comparison on Push-T (single 4060 8GB, batch=16, 200 steps).
set -e
export STABLEWM_HOME=/workspace/stablewm_home
export WANDB_MODE=offline

cd "$(dirname "$0")"

mkdir -p compare_runs

run() {
  local name=$1
  shift
  local extra="$@"
  local outdir="compare_runs/${name}"
  if [ -f "${outdir}/csv/version_0/metrics.csv" ]; then
    echo "skip ${name}"; return
  fi
  echo "=== ${name} (extra: ${extra}) ==="
  /workspace/le-wm/.venv/bin/python train.py \
    data=pusht \
    loader.batch_size=16 \
    loader.num_workers=0 \
    loader.persistent_workers=False \
    loader.pin_memory=False \
    loader.prefetch_factor=null \
    trainer.max_epochs=1 \
    +trainer.limit_train_batches=200 \
    +trainer.limit_val_batches=20 \
    wandb.enabled=False \
    output_model_name="${name}" \
    hydra.job.id="${name}" \
    hydra.run.dir="compare_runs/${name}" \
    ${extra} 2>&1 | tail -8
}

run baseline
run uvlowr_r4   "+predictor.ffn_rank=4"
run uvlowr_r8   "+predictor.ffn_rank=8"
run uvlowr_r16  "+predictor.ffn_rank=16"

echo "DONE_COMPARISON"
