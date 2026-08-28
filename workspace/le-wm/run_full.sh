#!/bin/bash
# Full instrumented pipeline: 9 training jobs (3 variants × 3 seeds), each followed by
# eval + final-checkpoint analyses. All metrics + raw values stored as CSV/JSON.
set -e
STEPS=${1:-40000}
PROBE_EVERY=${2:-500}
BATCH=${3:-32}
NUM_WORKERS=${4:-4}

cd /workspace/le-wm
export STABLEWM_HOME=${STABLEWM_HOME:-/workspace/stablewm_home}
export WANDB_MODE=offline
export SDL_VIDEODRIVER=dummy
unset http_proxy https_proxy

PY=/workspace/le-wm/.venv/bin/python
mkdir -p aggregated

train_one() {
  local name=$1
  local seed=$2
  shift 2
  local outdir="${STABLEWM_HOME}/${name}_seed${seed}"
  if [ -f "${outdir}/final_analyses.json" ]; then
    echo "SKIP ${name} seed=${seed} (final_analyses.json exists)"
    return
  fi
  echo "===> train ${name} seed=${seed} (steps=${STEPS})"
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
    output_model_name=${name}_seed${seed} hydra.job.id=${name}_seed${seed} \
    hydra.run.dir=hydra_runs/${name}_seed${seed} "$@" 2>&1 | tail -8

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
    --runs ${STABLEWM_HOME}/${name}_seed0 ${STABLEWM_HOME}/${name}_seed1 ${STABLEWM_HOME}/${name}_seed2 \
    --out-dir aggregated/${name}
}

for seed in 0 1 2; do
  train_one baseline   $seed
  train_one uvlowr_r4  $seed +predictor.ffn_rank=4
  train_one randdiff   $seed +loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2
done

for v in baseline uvlowr_r4 randdiff; do
  aggregate_one "$v"
done

echo "DONE_FULL_PIPELINE  steps=${STEPS}"
