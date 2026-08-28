#!/bin/bash
# B4 randdiff λ-sweep on Push-T frozen 40K (paper claim: nuclear regularization
# beats baseline under appropriate conditions). Also re-runs uvlowr seed1 for
# noise confirmation.
#
# Runs sequentially on one 4090-48G. Each run ~42 min. Total ~3h.
#
# Hooks into run_pipeline_v5.sh's train_frozen() helper. Stage-1 ckpt must be at
# /workspace/lewm_autodl_results_v3/stage1_baseline_seed0_weights.ckpt.

set -e
cd /workspace/le-wm
export STABLEWM_HOME=${STABLEWM_HOME:-/workspace/stablewm_home}
export WANDB_MODE=offline
export HF_ENDPOINT=https://hf-mirror.com
export HYDRA_FULL_ERROR=1

PY=.venv/bin/python
NW=${NUM_WORKERS:-4}
PF=${PREFETCH_FACTOR:-2}
STAGE1="/workspace/lewm_autodl_results_v3/stage1_baseline_seed0_weights.ckpt"
STEPS=40000

if [ ! -f "$STAGE1" ]; then
  echo "FATAL: stage1 ckpt missing at $STAGE1"
  exit 2
fi

mkdir -p ${STABLEWM_HOME} hydra_runs

train_one() {
  local name=$1; shift
  local seed=$1; shift
  local outdir="${STABLEWM_HOME}/${name}"
  if [ -f "${outdir}/${name}_epoch_1_object.ckpt" ]; then
    echo "===> SKIP ${name} (ckpt exists)"
    return 0
  fi
  echo "===> [$(date +%H:%M:%S)] train ${name}  seed=${seed}  steps=${STEPS}  overrides: $@"
  local t0=$(date +%s)
  $PY train.py \
    data=pusht seed=${seed} \
    loader.batch_size=32 loader.num_workers=${NW} \
    loader.persistent_workers=True loader.pin_memory=True loader.prefetch_factor=${PF} \
    trainer.max_epochs=1 +trainer.limit_train_batches=${STEPS} +trainer.limit_val_batches=20 \
    wandb.enabled=False \
    +probe_every=500 +jacobian_probe_n_samples=6 \
    +probing_n_train=256 +probing_n_test=64 +latent_cov_n_samples=256 \
    +freeze.enabled=true \
    +freeze.stage1_ckpt=${STAGE1} \
    +freeze.skip_sigreg=true \
    output_model_name=${name} hydra.job.id=${name} \
    hydra.run.dir=hydra_runs/${name} "$@" 2>&1 | tail -12
  local ckpt="${outdir}/${name}_epoch_1_object.ckpt"
  echo "===> eval_rollout ${name}"
  $PY eval_rollout.py "$ckpt" --horizons 1 3 5 10 15 20 \
    --n-rollout 256 --n-probe-train 1024 --n-probe-test 256 --batch-size 16 2>&1 | tail -8
  echo "===> final_analyses ${name}"
  $PY final_analyses.py "$ckpt" \
    --encoder-jac-samples 4 --mlp-n-train 2048 --mlp-n-test 512 2>&1 | tail -8
  echo "===> spectral_postprocess ${name} (|λ_max| + TwoNN(enc/pred))"
  $PY spectral_postprocess.py "$ckpt" \
    --n-samples 6 --twonn-n 2048 --batch-size 16 2>&1 | tail -10
  local dt=$(( $(date +%s) - t0 ))
  echo "===> [$(date +%H:%M:%S)] ${name} done in ${dt}s ($(( dt / 60 ))m)"
}

echo "================================================================"
echo "B4 λ-sweep + uvlowr seed1 (~3h total)"
echo "Start: $(date +%H:%M:%S)"
echo "================================================================"

# 1. randdiff lam=0.001 frozen 40K seed0 (most critical — match uvlowr setting)
train_one v5_randdiff_lam001_40K 0 \
  +loss.rand_diff.weight=0.001 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2

# 2. randdiff lam=0.003 frozen 40K seed0
train_one v5_randdiff_lam003_40K 0 \
  +loss.rand_diff.weight=0.003 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2

# 3. uvlowr seed1 (noise confirmation for +1.46pp @ h=3, +3.68pp @ h=5)
train_one v5_uvlowr_40K_seed1 1 +predictor.ffn_rank=4

# 4. randdiff lam=0.005 frozen 40K seed0 (fine sweep)
train_one v5_randdiff_lam005_40K 0 \
  +loss.rand_diff.weight=0.005 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2

echo "================================================================"
echo "ALL 4 RUNS DONE.  End: $(date +%H:%M:%S)"
echo "================================================================"
ls -la ${STABLEWM_HOME}/v5_*_40K* 2>/dev/null | head -20
