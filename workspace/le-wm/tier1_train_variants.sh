#!/bin/bash
# Tier-1 local reproduction: train 3 frozen-encoder variants for trajectory-metric study.
#
# v3 frozen ckpts are remote; this trains 3 variants × 1 seed × 3000 steps locally
# (~17 min on RTX 4060). Training count is 37% of v3's 8K — variant ORDERING should
# match v3 but absolute metric magnitudes will differ.
set -e

STEPS=${1:-3000}
SEED=${2:-0}
BATCH=8

cd /workspace/le-wm
export STABLEWM_HOME=/workspace/stablewm_home
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1

STAGE1_CKPT="/workspace/lewm_autodl_results_v3/stage1_baseline_seed0_weights.ckpt"

train_one() {
    local name=$1; shift
    local outdir="${STABLEWM_HOME}/${name}"
    if [ -f "${outdir}/${name}_epoch_1_object.ckpt" ]; then
        echo "===> SKIP ${name} (ckpt exists)"
        return
    fi
    echo "===> [tier1] train ${name} (${STEPS} steps, batch=${BATCH}, seed=${SEED})"
    local t0=$(date +%s)
    .venv/bin/python train.py \
        data=pusht seed=${SEED} \
        loader.batch_size=${BATCH} loader.num_workers=0 loader.persistent_workers=False loader.pin_memory=False '~loader.prefetch_factor' \
        trainer.max_epochs=1 +trainer.limit_train_batches=${STEPS} +trainer.limit_val_batches=4 \
        wandb.enabled=False \
        +probe_every=999999 +jacobian_probe_n_samples=1 \
        +probing_n_train=128 +probing_n_test=32 +latent_cov_n_samples=128 \
        +freeze.enabled=true \
        +freeze.stage1_ckpt=${STAGE1_CKPT} \
        +freeze.skip_sigreg=true \
        output_model_name=${name} hydra.job.id=${name} \
        hydra.run.dir=hydra_runs/${name} "$@" 2>&1 | tail -8
    local dt=$(( $(date +%s) - t0 ))
    echo "===> ${name} done in ${dt}s"
}

train_one tier1_baseline
train_one tier1_uvlowr  +predictor.ffn_rank=4
train_one tier1_randdiff +loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2

echo "===> Tier-1 training complete"
ls -lh ${STABLEWM_HOME}/tier1_*/{*epoch_1_object.ckpt,engineering_metrics.json} 2>/dev/null
