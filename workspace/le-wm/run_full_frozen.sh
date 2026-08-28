#!/bin/bash
# Frozen-encoder ablation pipeline (Stage-2 of v3 experiment):
#   - Stage-1 source: baseline_seed0 weights (encoder + projector + pred_proj frozen)
#   - 3 variants × 2 seeds × ${STEPS} steps (default 10000)
#   - SIGReg loss skipped (no gradient consumer with encoder frozen)
#
# Usage: ./run_full_frozen.sh [STEPS] [PROBE_EVERY] [BATCH] [NUM_WORKERS]
set -e
STEPS=${1:-10000}
PROBE_EVERY=${2:-500}
BATCH=${3:-32}
NUM_WORKERS=${4:-4}

cd /workspace/le-wm
export STABLEWM_HOME=${STABLEWM_HOME:-/workspace/stablewm_home}
export WANDB_MODE=offline
export SDL_VIDEODRIVER=dummy
unset http_proxy https_proxy

PY=/root/miniconda3/envs/lewm310/bin/python
STAGE1_CKPT="${STABLEWM_HOME}/baseline_seed0/baseline_seed0_weights.ckpt"
mkdir -p aggregated_frozen

if [ ! -f "${STAGE1_CKPT}" ]; then
  echo "ERROR: Stage-1 ckpt not found at ${STAGE1_CKPT}"
  echo "       Need v2 baseline_seed0 weights. Sync from local before running."
  exit 2
fi

train_one_frozen() {
  local name=$1
  local seed=$2
  shift 2
  local outdir="${STABLEWM_HOME}/${name}_seed${seed}"
  if [ -f "${outdir}/final_analyses.json" ]; then
    echo "SKIP ${name} seed=${seed} (final_analyses.json exists)"
    return
  fi
  echo "===> train [FROZEN] ${name} seed=${seed} (steps=${STEPS})"
  $PY train.py \
    data=pusht seed=${seed} \
    loader.batch_size=${BATCH} loader.num_workers=${NUM_WORKERS} \
    loader.persistent_workers=True loader.pin_memory=True loader.prefetch_factor=2 \
    trainer.max_epochs=1 +trainer.limit_train_batches=${STEPS} +trainer.limit_val_batches=20 \
    wandb.enabled=False \
    +probe_every=${PROBE_EVERY} \
    +jacobian_probe_n_samples=6 \
    +probing_n_train=256 +probing_n_test=64 \
    +latent_cov_n_samples=256 \
    +freeze.enabled=true \
    +freeze.stage1_ckpt=${STAGE1_CKPT} \
    +freeze.skip_sigreg=true \
    output_model_name=${name}_seed${seed} hydra.job.id=${name}_seed${seed} \
    hydra.run.dir=hydra_runs/${name}_seed${seed} "$@" 2>&1 | tail -12

  echo "===> rollout eval ${name} seed=${seed}"
  ckpt=${outdir}/${name}_seed${seed}_epoch_1_object.ckpt
  $PY eval_rollout.py "$ckpt" \
    --horizons 1 3 5 10 15 20 \
    --n-rollout 256 --n-probe-train 1024 --n-probe-test 256 --batch-size 16 2>&1 | tail -6

  echo "===> final analyses ${name} seed=${seed}"
  $PY final_analyses.py "$ckpt" \
    --encoder-jac-samples 4 --mlp-n-train 2048 --mlp-n-test 512 2>&1 | tail -10
}

aggregate_one() {
  local name=$1
  $PY aggregate.py --variant "${name}" \
    --runs ${STABLEWM_HOME}/${name}_seed0 ${STABLEWM_HOME}/${name}_seed1 \
    --out-dir aggregated_frozen/${name}
}

# Sanity smoke (M0): 1 short run, baseline-only — gated, will only run if SANITY_STEPS env set
if [ -n "${SANITY_STEPS:-}" ]; then
  echo "===> M0 SANITY: frozen-baseline ${SANITY_STEPS} steps"
  STEPS_BAK=${STEPS}; STEPS=${SANITY_STEPS}
  train_one_frozen frozen_baseline_sanity 0
  STEPS=${STEPS_BAK}
  echo "===> M0 SANITY done; proceeding to M1"
fi

for seed in 0 1; do
  train_one_frozen frozen_baseline   $seed
  train_one_frozen frozen_uvlowr_r4  $seed +predictor.ffn_rank=4
  train_one_frozen frozen_randdiff   $seed +loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2
done

for v in frozen_baseline frozen_uvlowr_r4 frozen_randdiff; do
  aggregate_one "$v"
done

echo "DONE_FROZEN_PIPELINE  steps=${STEPS}"
