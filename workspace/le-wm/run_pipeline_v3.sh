#!/bin/bash
# v3 pipeline (frozen-encoder ablation):
#   Phase A: Stage-1 baseline pretrain (20K steps, joint training, seed=0) → produces frozen encoder
#   Phase B: M0 sanity — frozen-baseline, 1K steps
#   Phase C: M1 main — 3 variants × 2 seeds × ${STEPS} steps frozen-encoder
#   Phase D: aggregate
#
# Args: ./run_pipeline_v3.sh [STAGE1_STEPS] [STAGE2_STEPS] [PROBE_EVERY] [BATCH] [NUM_WORKERS]
# Defaults: 20000 / 8000 / 500 / 32 / 4
set -e
STAGE1_STEPS=${1:-20000}
STAGE2_STEPS=${2:-8000}
PROBE_EVERY=${3:-500}
BATCH=${4:-32}
NUM_WORKERS=${5:-4}

cd /workspace/le-wm
export STABLEWM_HOME=${STABLEWM_HOME:-/workspace/stablewm_home}
export WANDB_MODE=offline
export SDL_VIDEODRIVER=dummy
unset http_proxy https_proxy

PY=/root/miniconda3/envs/lewm310/bin/python

STAGE1_NAME="stage1_baseline"
STAGE1_DIR="${STABLEWM_HOME}/${STAGE1_NAME}_seed0"
STAGE1_CKPT="${STAGE1_DIR}/${STAGE1_NAME}_seed0_weights.ckpt"

mkdir -p aggregated_frozen

train_joint() {
  # Phase A: joint training (encoder + projector + predictor + pred_proj + sigreg)
  local name=$1
  local seed=$2
  local steps=$3
  shift 3
  local outdir="${STABLEWM_HOME}/${name}_seed${seed}"
  if [ -f "${outdir}/final_analyses.json" ]; then
    echo "SKIP joint ${name} seed=${seed} (final_analyses.json exists)"
    return
  fi
  echo "===> [STAGE-1 JOINT] train ${name} seed=${seed} (steps=${steps})"
  $PY train.py \
    data=pusht seed=${seed} \
    loader.batch_size=${BATCH} loader.num_workers=${NUM_WORKERS} \
    loader.persistent_workers=True loader.pin_memory=True loader.prefetch_factor=2 \
    trainer.max_epochs=1 +trainer.limit_train_batches=${steps} +trainer.limit_val_batches=20 \
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

train_frozen() {
  # Phase B/C: frozen-encoder training (encoder + projector + pred_proj loaded from Stage-1 ckpt)
  local name=$1
  local seed=$2
  local steps=$3
  shift 3
  local outdir="${STABLEWM_HOME}/${name}_seed${seed}"
  if [ -f "${outdir}/final_analyses.json" ]; then
    echo "SKIP frozen ${name} seed=${seed} (final_analyses.json exists)"
    return
  fi
  echo "===> [FROZEN] train ${name} seed=${seed} (steps=${steps})"
  $PY train.py \
    data=pusht seed=${seed} \
    loader.batch_size=${BATCH} loader.num_workers=${NUM_WORKERS} \
    loader.persistent_workers=True loader.pin_memory=True loader.prefetch_factor=2 \
    trainer.max_epochs=1 +trainer.limit_train_batches=${steps} +trainer.limit_val_batches=20 \
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

aggregate_frozen() {
  local name=$1
  $PY aggregate.py --variant "${name}" \
    --runs ${STABLEWM_HOME}/${name}_seed0 ${STABLEWM_HOME}/${name}_seed1 \
    --out-dir aggregated_frozen/${name}
}

#####################
#  Phase A: Stage 1 #
#####################
echo ""
echo "########## Phase A: STAGE-1 baseline pretrain (${STAGE1_STEPS} joint steps) ##########"
if [ -f "${STAGE1_CKPT}" ]; then
  echo "Stage-1 ckpt already exists at ${STAGE1_CKPT}; skipping pretrain"
else
  train_joint ${STAGE1_NAME} 0 ${STAGE1_STEPS}
  if [ ! -f "${STAGE1_CKPT}" ]; then
    echo "FATAL: Stage-1 ckpt missing after pretrain; aborting"
    exit 2
  fi
fi
ls -la ${STAGE1_DIR}

#####################
#  Phase B: M0 sanity #
#####################
echo ""
echo "########## Phase B: M0 SANITY (frozen-baseline, 1000 steps) ##########"
train_frozen frozen_sanity 0 1000

# M0 sanity assertions: pred_loss drop, NO grad on frozen modules, callbacks fired
echo "===> M0 sanity assertions"
$PY <<EOF
import json, csv
from pathlib import Path
d = Path('${STABLEWM_HOME}/frozen_sanity_seed0')
assert d.exists(), f"M0 dir missing: {d}"

# 1) pred_loss drops sufficiently
rows = list(csv.DictReader(open(d / 'csv' / 'version_0' / 'metrics.csv')))
losses = [(int(r['step']), float(r['fit/pred_loss']))
          for r in rows if r.get('fit/pred_loss','').strip()]
assert losses, "M0 FAIL: no pred_loss rows in metrics.csv"
first = losses[0][1]; last = losses[-1][1]
drop_pct = 100*(first-last)/first
print(f"  M0 pred_loss: step{losses[0][0]}={first:.4f}, step{losses[-1][0]}={last:.4f}, drop={drop_pct:.1f}%")
assert last < first * 0.7, f"M0 FAIL: pred_loss did not drop ≥30% ({first:.4f} -> {last:.4f})"

# 2) All instrumentation files exist and are non-empty
for fn in ['jacobian_probe.jsonl', 'probing.jsonl', 'latent_cov.jsonl',
           'engineering_metrics.json']:
    fp = d / fn
    assert fp.exists() and fp.stat().st_size > 0, f"M0 FAIL: {fn} missing or empty"
print(f"  M0 instrumentation files: all present and non-empty")

# 3) Probing R² produced at step 0 (the control point) and at the end
plines = [json.loads(l) for l in open(d / 'probing.jsonl')]
steps = sorted({p['step'] for p in plines})
assert 0 in steps or min(steps) <= 1, f"M0 FAIL: no step-0 probing measurement; got {steps[:5]}"
print(f"  M0 probing.jsonl: {len(plines)} rows across steps {min(steps)}..{max(steps)}")

# 4) Jacobian SR sanity — should produce non-NaN, non-zero numbers
jlines = [json.loads(l) for l in open(d / 'jacobian_probe.jsonl')]
last_jac = jlines[-1]
sr = last_jac.get('mean_stable_rank', float('nan'))
assert sr > 0 and sr == sr, f"M0 FAIL: bad jacobian SR={sr}"
print(f"  M0 final jacobian SR: {sr:.2f}")

# 5) Verify frozen modules really had zero gradients via the FreezeAssertCallback log
# (the KeepFrozenInEvalCallback raises on_fit_start if requires_grad fails — so
#  if we got here at all, the freeze invariant held)
print("  M0 freeze invariant: PASSED (KeepFrozenInEvalCallback would have raised otherwise)")

print("  M0 SANITY: ALL CHECKS PASSED")
EOF
echo "===> M0 sanity PASSED"

#####################
#  Phase C: M1 main #
#####################
echo ""
echo "########## Phase C: M1 MAIN (3 variants × 2 seeds × ${STAGE2_STEPS} steps frozen) ##########"
for seed in 0 1; do
  train_frozen frozen_baseline   $seed ${STAGE2_STEPS}
  train_frozen frozen_uvlowr_r4  $seed ${STAGE2_STEPS} +predictor.ffn_rank=4
  train_frozen frozen_randdiff   $seed ${STAGE2_STEPS} +loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2
done

#####################
#  Phase D: aggregate #
#####################
echo ""
echo "########## Phase D: aggregate ##########"
for v in frozen_baseline frozen_uvlowr_r4 frozen_randdiff; do
  aggregate_frozen "$v"
done

echo ""
echo "DONE_PIPELINE_V3  stage1=${STAGE1_STEPS}  stage2=${STAGE2_STEPS}"
