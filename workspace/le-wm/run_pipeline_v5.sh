#!/bin/bash
# v5 orchestrator — Encoder-mediated compensation paper, post-novelty-check pivot.
#
# Runs B1 (ε sweep) + B2 (uvlowr-Stage-1 + freeze) + B3 (rank sweep) +
# B4 (λ dose-response) + B5 (DMD post-hoc) sequentially on a single AutoDL 4090.
#
# Reuses v3 ckpts where possible (frozen_baseline_seed0, frozen_uvlowr_r4_seed0,
# frozen_randdiff_seed0). Total ~20 GPU-hr.
#
# Usage:
#   ./run_pipeline_v5.sh           # full pipeline
#   ./run_pipeline_v5.sh --phase B3  # run only one phase

set -e

PHASES="${1:-all}"

cd /workspace/le-wm
export STABLEWM_HOME=${STABLEWM_HOME:-/workspace/stablewm_home}
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
unset http_proxy https_proxy

PY=${PY:-.venv/bin/python}
# WSL2 /dev/shm=64MB → forces num_workers=0; set NUM_WORKERS=4 to override on real Linux.
NW=${NUM_WORKERS:-0}
PW=${PERSISTENT_WORKERS:-False}
PF=${PREFETCH_FACTOR:-2}
STAGE1_BASELINE_CKPT="/workspace/lewm_autodl_results_v3/stage1_baseline_seed0_weights.ckpt"

if [ ! -f "${STAGE1_BASELINE_CKPT}" ]; then
    echo "FATAL: v3 Stage-1 ckpt not found at ${STAGE1_BASELINE_CKPT}"
    echo "       Upload via: scp stage1_baseline_seed0_weights.ckpt instance:/workspace/lewm_autodl_results_v3/"
    exit 2
fi

mkdir -p ${STABLEWM_HOME} hydra_runs aggregated_v5

# ---------------------------------------------------------------------------
# Helper: train one frozen run (uniform interface for B1, B3, B4)
# ---------------------------------------------------------------------------
train_frozen() {
    local name=$1; shift
    local stage1_ckpt=$1; shift
    local steps=${1:-8000}; shift
    local outdir="${STABLEWM_HOME}/${name}"
    if [ -f "${outdir}/${name}_epoch_1_object.ckpt" ]; then
        echo "===> SKIP ${name} (ckpt exists)"
        return 0
    fi
    echo "===> [$(date +%H:%M:%S)] train_frozen ${name}  steps=${steps}  overrides: $@"
    local t0=$(date +%s)
    local pf_arg
    if [ "$NW" -gt 0 ]; then pf_arg="loader.prefetch_factor=${PF}"; else pf_arg="loader.prefetch_factor=null"; fi
    $PY train.py \
        data=pusht seed=0 \
        loader.batch_size=32 loader.num_workers=${NW} \
        loader.persistent_workers=${PW} loader.pin_memory=True ${pf_arg} \
        trainer.max_epochs=1 +trainer.limit_train_batches=${steps} +trainer.limit_val_batches=20 \
        wandb.enabled=False \
        +probe_every=500 +jacobian_probe_n_samples=6 \
        +probing_n_train=256 +probing_n_test=64 +latent_cov_n_samples=256 \
        +freeze.enabled=true \
        +freeze.stage1_ckpt=${stage1_ckpt} \
        +freeze.skip_sigreg=true \
        output_model_name=${name} hydra.job.id=${name} \
        hydra.run.dir=hydra_runs/${name} "$@" 2>&1 | tail -10
    local ckpt="${outdir}/${name}_epoch_1_object.ckpt"
    echo "===> eval_rollout ${name}"
    $PY eval_rollout.py "$ckpt" --horizons 1 3 5 10 15 20 \
        --n-rollout 256 --n-probe-train 1024 --n-probe-test 256 --batch-size 16 2>&1 | tail -6
    echo "===> final_analyses ${name}"
    $PY final_analyses.py "$ckpt" \
        --encoder-jac-samples 4 --mlp-n-train 2048 --mlp-n-test 512 2>&1 | tail -8
    local dt=$(( $(date +%s) - t0 ))
    echo "===> [$(date +%H:%M:%S)] ${name} done in ${dt}s"
}

# ---------------------------------------------------------------------------
# Helper: train Stage-1 (uvlowr-shaped joint training)
# ---------------------------------------------------------------------------
train_stage1_uvlowr() {
    local name=$1; shift
    local steps=${1:-20000}; shift
    local seed=${1:-0}; shift
    local outdir="${STABLEWM_HOME}/${name}"
    if [ -f "${outdir}/${name}_epoch_1_object.ckpt" ]; then
        echo "===> SKIP ${name} (ckpt exists)"
        return 0
    fi
    echo "===> [$(date +%H:%M:%S)] Stage-1 ${name}  steps=${steps}  seed=${seed}"
    local t0=$(date +%s)
    $PY train.py \
        data=pusht seed=${seed} \
        loader.batch_size=32 loader.num_workers=4 \
        loader.persistent_workers=True loader.pin_memory=True loader.prefetch_factor=2 \
        trainer.max_epochs=1 +trainer.limit_train_batches=${steps} +trainer.limit_val_batches=20 \
        wandb.enabled=False \
        +probe_every=500 +jacobian_probe_n_samples=6 \
        +probing_n_train=256 +probing_n_test=64 +latent_cov_n_samples=256 \
        +predictor.ffn_rank=4 \
        output_model_name=${name} hydra.job.id=${name} \
        hydra.run.dir=hydra_runs/${name} 2>&1 | tail -10
    # also save weights-only ckpt for downstream freeze
    local obj="${outdir}/${name}_epoch_1_object.ckpt"
    local w="${outdir}/${name}_weights.ckpt"
    if [ -f "$obj" ] && [ ! -f "$w" ]; then
        $PY -c "import torch; sd = torch.load('$obj', map_location='cpu', weights_only=False); \
                sd2 = sd.state_dict() if hasattr(sd,'state_dict') else sd; \
                torch.save({'state_dict': sd2}, '$w'); print('saved weights ckpt')"
    fi
    local dt=$(( $(date +%s) - t0 ))
    echo "===> [$(date +%H:%M:%S)] ${name} done in ${dt}s"
}

# ---------------------------------------------------------------------------
# M0 — sanity check (1K freeze)
# ---------------------------------------------------------------------------
run_M0() {
    echo "##### M0 — freeze sanity #####"
    train_frozen v5_sanity ${STAGE1_BASELINE_CKPT} 1000
}

# ---------------------------------------------------------------------------
# B1 — ε sensitivity sweep ×2 seeds (v5.2)
# ---------------------------------------------------------------------------
run_B1() {
    echo "##### B1 — ε sweep ×2 seeds on frozen randdiff #####"
    for s in 0 1; do
        for eps in 0.0125 0.05 0.2 0.8; do
            tag=$(echo $eps | tr '.' '_')
            if [ "$eps" = "0.05" ] && [ "$s" = "0" ]; then
                echo "===> SKIP v5_eps_${tag}_seed${s} (reuse v3 R006)"
                continue
            fi
            train_frozen v5_eps_${tag}_seed${s} ${STAGE1_BASELINE_CKPT} 8000 seed=${s} \
                +loss.rand_diff.weight=0.01 +loss.rand_diff.eps=${eps} +loss.rand_diff.num_dirs=2
        done
    done
}

# ---------------------------------------------------------------------------
# B2 — uvlowr-co-trained Stage-1 + freeze 3 variants × 2 seeds
# ---------------------------------------------------------------------------
run_B2() {
    echo "##### B2 — uvlowr-Stage-1 + freeze multi-predictor #####"
    train_stage1_uvlowr v5_stage1_uvlowr 20000 0
    local NEW_S1="${STABLEWM_HOME}/v5_stage1_uvlowr/v5_stage1_uvlowr_weights.ckpt"
    [ -f "$NEW_S1" ] || NEW_S1="${STABLEWM_HOME}/v5_stage1_uvlowr/v5_stage1_uvlowr_epoch_1_object.ckpt"

    for seed in 0 1; do
        train_frozen v5_frozen_baseline_uvS1_seed${seed} $NEW_S1 8000 seed=${seed}
        train_frozen v5_frozen_uvlowr_uvS1_seed${seed}   $NEW_S1 8000 seed=${seed} \
            +predictor.ffn_rank=4
        # ε for B2 randdiff: use the v3 ε=0.05 default (B1 confirms or contradicts)
        train_frozen v5_frozen_randdiff_uvS1_seed${seed} $NEW_S1 8000 seed=${seed} \
            +loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2
    done
}

# ---------------------------------------------------------------------------
# B3 — uvlowr rank sweep ×2 seeds (v5.2; r=4 s=0 reused from v3)
# ---------------------------------------------------------------------------
run_B3() {
    echo "##### B3 — uvlowr rank sweep ×2 seeds #####"
    for s in 0 1; do
        for r in 1 2 4 8 16; do
            if [ "$r" = "4" ] && [ "$s" = "0" ]; then
                echo "===> SKIP v5_uvlowr_r4_seed0 (reuse v3 R004)"
                continue
            fi
            train_frozen v5_uvlowr_r${r}_seed${s} ${STAGE1_BASELINE_CKPT} 8000 seed=${s} \
                +predictor.ffn_rank=${r}
        done
    done
}

# ---------------------------------------------------------------------------
# B4 — randdiff λ dose-response ×2 seeds (v5.2)
#  λ=0 s={0,1} reused from R002/R003; λ=0.01 s=0 reused from R006
# ---------------------------------------------------------------------------
run_B4() {
    echo "##### B4 — randdiff λ dose-response ×2 seeds #####"
    for s in 0 1; do
        for lam in 0.001 0.003 0.03 0.1; do
            tag=$(echo $lam | tr '.' '_')
            train_frozen v5_lam_${tag}_seed${s} ${STAGE1_BASELINE_CKPT} 8000 seed=${s} \
                +loss.rand_diff.weight=${lam} +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2
        done
    done
    # λ=0.01 s=1 (s=0 reused from v3 R006)
    train_frozen v5_lam_0_01_seed1 ${STAGE1_BASELINE_CKPT} 8000 seed=1 \
        +loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2
}

# ---------------------------------------------------------------------------
# B5 — post-hoc DMD on all v5 + v3 frozen ckpts (no training)
# ---------------------------------------------------------------------------
run_B5() {
    echo "##### B5 — DMD post-hoc on v5 + v3 frozen ckpts #####"
    local outdir=${STABLEWM_HOME}/dmd_full_budget
    mkdir -p $outdir
    # We'll need to extend tier1_dump_trajectories.py to accept arbitrary ckpt path.
    # Quick approach: loop over ckpts and invoke a unified dump+metrics script.
    for run in v5_eps_0125 v5_eps_200 v5_eps_800 \
               v5_uvlowr_r1 v5_uvlowr_r2 v5_uvlowr_r8 v5_uvlowr_r16 \
               v5_lam_001 v5_lam_003 v5_lam_030 v5_lam_100 \
               v5_frozen_baseline_uvS1_seed0 v5_frozen_uvlowr_uvS1_seed0 v5_frozen_randdiff_uvS1_seed0
    do
        local ckpt=${STABLEWM_HOME}/${run}/${run}_epoch_1_object.ckpt
        [ -f "$ckpt" ] || { echo "SKIP $run (no ckpt)"; continue; }
        echo "===> DMD $run"
        $PY tier1_dump_and_metrics.py --ckpt "$ckpt" --name "$run" \
            --out-dir $outdir 2>&1 | tail -4
    done
}

# ---------------------------------------------------------------------------
# B6 — aggregate + summary
# ---------------------------------------------------------------------------
run_B6() {
    echo "##### B6 — aggregate v5 results #####"
    $PY aggregate.py --variant v5_eps_sweep \
        --runs ${STABLEWM_HOME}/v5_eps_0125 \
               /workspace/lewm_autodl_results_v3/frozen_randdiff_seed0 \
               ${STABLEWM_HOME}/v5_eps_200 \
               ${STABLEWM_HOME}/v5_eps_800 \
        --out-dir aggregated_v5/eps_sweep 2>&1 | tail -4
    $PY aggregate.py --variant v5_rank_sweep \
        --runs ${STABLEWM_HOME}/v5_uvlowr_r1 ${STABLEWM_HOME}/v5_uvlowr_r2 \
               /workspace/lewm_autodl_results_v3/frozen_uvlowr_r4_seed0 \
               ${STABLEWM_HOME}/v5_uvlowr_r8 ${STABLEWM_HOME}/v5_uvlowr_r16 \
        --out-dir aggregated_v5/rank_sweep 2>&1 | tail -4
    $PY aggregate.py --variant v5_lam_sweep \
        --runs /workspace/lewm_autodl_results_v3/frozen_baseline_seed0 \
               ${STABLEWM_HOME}/v5_lam_001 ${STABLEWM_HOME}/v5_lam_003 \
               /workspace/lewm_autodl_results_v3/frozen_randdiff_seed0 \
               ${STABLEWM_HOME}/v5_lam_030 ${STABLEWM_HOME}/v5_lam_100 \
        --out-dir aggregated_v5/lam_sweep 2>&1 | tail -4
    echo "DONE: aggregated_v5/*; tier-1 DMD outputs in ${STABLEWM_HOME}/dmd_full_budget/"
}

# ---------------------------------------------------------------------------
# B0' — Cross-seed noise floor (v5.2: n=7, 5 new seeds × frozen_baseline)
# ---------------------------------------------------------------------------
run_B0prime() {
    echo "##### B0' — cross-seed noise floor (frozen_baseline seeds 2..6) #####"
    # B0PRIME_SEEDS env override allows running a subset, e.g. "2" for a single seed.
    local seeds_to_run=${B0PRIME_SEEDS:-"2 3 4 5 6"}
    for s in $seeds_to_run; do
        train_frozen v5_baseline_seed${s} ${STAGE1_BASELINE_CKPT} 8000 seed=${s}
    done
    echo "===> B0' aggregate σ_noise"
    $PY -c "
import json, glob
import numpy as np
runs = ['/workspace/lewm_autodl_results_v3/frozen_baseline_seed0',
        '/workspace/lewm_autodl_results_v3/frozen_baseline_seed1',
        '${STABLEWM_HOME}/v5_baseline_seed2',
        '${STABLEWM_HOME}/v5_baseline_seed3',
        '${STABLEWM_HOME}/v5_baseline_seed4',
        '${STABLEWM_HOME}/v5_baseline_seed5',
        '${STABLEWM_HOME}/v5_baseline_seed6']
metrics = {'sigma_1': [], 'F_sq': [], 'SR': []}
for r in runs:
    try:
        d = json.loads(open(r + '/jacobian_probe.jsonl').readlines()[-1])
        metrics['sigma_1'].append(d['mean_spectral_norm'])
        metrics['F_sq'].append(np.mean([s['frobenius_norm_sq'] for s in d['samples']]))
        metrics['SR'].append(d['mean_stable_rank'])
    except: pass
out = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'two_sigma': float(2*np.std(v)), 'n': len(v)}
       for k, v in metrics.items() if v}
json.dump(out, open('${STABLEWM_HOME}/v5_noise_floor.json', 'w'), indent=2)
print(json.dumps(out, indent=2))
"
}

# ---------------------------------------------------------------------------
# Gate M1 → M2: check if any ε in B1 reproduces v2-like inflation
# ---------------------------------------------------------------------------
gate_M1_to_M2() {
    echo "##### GATE M1→M2: checking if any ε produces σ₁ > baseline + 0.20 #####"
    $PY -c "
import json
import numpy as np
nf = json.load(open('${STABLEWM_HOME}/v5_noise_floor.json'))
baseline_mean = nf['sigma_1']['mean']
runs = {
    'eps_0125': '${STABLEWM_HOME}/v5_eps_0125',
    'eps_050':  '/workspace/lewm_autodl_results_v3/frozen_randdiff_seed0',
    'eps_200':  '${STABLEWM_HOME}/v5_eps_200',
    'eps_800':  '${STABLEWM_HOME}/v5_eps_800',
}
sigmas = {}
for name, r in runs.items():
    try:
        d = json.loads(open(r + '/jacobian_probe.jsonl').readlines()[-1])
        sigmas[name] = d['mean_spectral_norm']
    except Exception as e:
        sigmas[name] = None
        print(f'WARN: {name} missing ({e})')
print(f'baseline_mean σ₁ = {baseline_mean:.3f}')
print(f'σ₁ per ε:')
for name, s in sigmas.items():
    if s is None: continue
    delta = s - baseline_mean
    flag = 'INFLATION' if delta > 0.20 else 'shrinkage' if delta < -nf['sigma_1']['two_sigma'] else 'neutral'
    print(f'  {name}: σ₁={s:.3f}  Δ={delta:+.3f}  → {flag}')
ABORT = any((s - baseline_mean) > 0.20 for s in sigmas.values() if s is not None)
if ABORT:
    print('===> ABORT GATE: some ε produces v2-like inflation. DO NOT RUN M2.')
    open('${STABLEWM_HOME}/v5_abort_gate.flag', 'w').write('ABORTED')
else:
    print('===> GATE PASSED: no ε reproduces v2-like inflation. Proceed to M2.')
"
    if [ -f "${STABLEWM_HOME}/v5_abort_gate.flag" ]; then
        echo "ABORTING: v5_abort_gate.flag exists."
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# B4' — 40K extension for ALL 3 variants (v5.2 full)
# ---------------------------------------------------------------------------
run_B4prime() {
    echo "##### B4' — 40K extension for all 3 variants #####"
    train_frozen v5_baseline_40K ${STAGE1_BASELINE_CKPT} 40000
    train_frozen v5_uvlowr_40K   ${STAGE1_BASELINE_CKPT} 40000 +predictor.ffn_rank=4
    train_frozen v5_randdiff_40K ${STAGE1_BASELINE_CKPT} 40000 \
        +loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2
}

# ---------------------------------------------------------------------------
# B4' partial — 40K randdiff only (v5.2 partial budget; saves ~6 GPU-h)
# ---------------------------------------------------------------------------
run_B4prime_partial() {
    echo "##### B4' partial — 40K frozen_randdiff only #####"
    train_frozen v5_randdiff_40K ${STAGE1_BASELINE_CKPT} 40000 \
        +loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2
}

# ---------------------------------------------------------------------------
# B4' AutoDL helpers — single variants (Plan E split: user does randdiff
# locally; AutoDL does baseline + uvlowr)
# ---------------------------------------------------------------------------
run_B4prime_baseline_only() {
    echo "##### B4' (AutoDL) — frozen_baseline_40K #####"
    train_frozen v5_baseline_40K ${STAGE1_BASELINE_CKPT} 40000
}

run_B4prime_uvlowr_only() {
    echo "##### B4' (AutoDL) — frozen_uvlowr_40K (ffn_rank=4) #####"
    train_frozen v5_uvlowr_40K ${STAGE1_BASELINE_CKPT} 40000 +predictor.ffn_rank=4
}

run_B4prime_randdiff_only() {
    echo "##### B4' (local) — frozen_randdiff_40K (P-Anti-I closure) #####"
    train_frozen v5_randdiff_40K ${STAGE1_BASELINE_CKPT} 40000 \
        +loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2
}

# ---------------------------------------------------------------------------
# M0 quick sanity for AutoDL freshness (smaller version of M0)
# ---------------------------------------------------------------------------
run_M0_quick_sanity() {
    echo "##### M0 quick sanity (200-step frozen_baseline check) #####"
    train_frozen autodl_sanity ${STAGE1_BASELINE_CKPT} 200
}

# ---------------------------------------------------------------------------
# B5' — (ε, λ) 4-corner cross-product  ⭐ NEW v5.1
# ---------------------------------------------------------------------------
run_B5prime() {
    echo "##### B5' — (ε, λ) 4-corner cross-product #####"
    train_frozen v5_corner_LL ${STAGE1_BASELINE_CKPT} 8000 \
        +loss.rand_diff.weight=0.001 +loss.rand_diff.eps=0.0125 +loss.rand_diff.num_dirs=2
    train_frozen v5_corner_LH ${STAGE1_BASELINE_CKPT} 8000 \
        +loss.rand_diff.weight=0.1   +loss.rand_diff.eps=0.0125 +loss.rand_diff.num_dirs=2
    train_frozen v5_corner_HL ${STAGE1_BASELINE_CKPT} 8000 \
        +loss.rand_diff.weight=0.001 +loss.rand_diff.eps=0.8 +loss.rand_diff.num_dirs=2
    train_frozen v5_corner_HH ${STAGE1_BASELINE_CKPT} 8000 \
        +loss.rand_diff.weight=0.1   +loss.rand_diff.eps=0.8 +loss.rand_diff.num_dirs=2
}

# ---------------------------------------------------------------------------
# B7 — Optimizer/LR robustness (v5.2 MUST)
# ---------------------------------------------------------------------------
run_B7() {
    echo "##### B7 — optimizer/LR robustness #####"
    train_frozen v5_LRhi_baseline ${STAGE1_BASELINE_CKPT} 8000 optimizer.lr=5e-4
    train_frozen v5_LRhi_randdiff ${STAGE1_BASELINE_CKPT} 8000 optimizer.lr=5e-4 \
        +loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2
}

# ---------------------------------------------------------------------------
# B8 — 2nd Stage-1 baseline seed (v5.2 MUST)
# ---------------------------------------------------------------------------
run_B8() {
    echo "##### B8 — Stage-1 seed=1 + 3 frozen variants #####"
    train_stage1_baseline_seed1 v5_stage1_baseline_seed1 20000 1
    local S1S1="${STABLEWM_HOME}/v5_stage1_baseline_seed1/v5_stage1_baseline_seed1_weights.ckpt"
    [ -f "$S1S1" ] || S1S1="${STABLEWM_HOME}/v5_stage1_baseline_seed1/v5_stage1_baseline_seed1_epoch_1_object.ckpt"
    train_frozen v5_S1seed1_baseline $S1S1 8000
    train_frozen v5_S1seed1_uvlowr   $S1S1 8000 +predictor.ffn_rank=4
    train_frozen v5_S1seed1_randdiff $S1S1 8000 \
        +loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2
}

train_stage1_baseline_seed1() {
    local name=$1; shift
    local steps=${1:-20000}; shift
    local seed=${1:-1}; shift
    local outdir="${STABLEWM_HOME}/${name}"
    if [ -f "${outdir}/${name}_epoch_1_object.ckpt" ]; then
        echo "===> SKIP ${name} (ckpt exists)"
        return 0
    fi
    echo "===> [$(date +%H:%M:%S)] Stage-1 baseline ${name}  steps=${steps}  seed=${seed}"
    $PY train.py \
        data=pusht seed=${seed} \
        loader.batch_size=32 loader.num_workers=4 \
        loader.persistent_workers=True loader.pin_memory=True loader.prefetch_factor=2 \
        trainer.max_epochs=1 +trainer.limit_train_batches=${steps} +trainer.limit_val_batches=20 \
        wandb.enabled=False \
        +probe_every=500 +jacobian_probe_n_samples=6 \
        +probing_n_train=256 +probing_n_test=64 +latent_cov_n_samples=256 \
        output_model_name=${name} hydra.job.id=${name} \
        hydra.run.dir=hydra_runs/${name} 2>&1 | tail -10
}

# ---------------------------------------------------------------------------
# B9 — PointMaze cross-dataset (v5.2 MUST ⭐⭐⭐)
# ---------------------------------------------------------------------------
run_B9() {
    echo "##### B9 — PointMaze cross-dataset (Stage-1 + 3v × 2s frozen) #####"
    # Stage-1 on PointMaze (data=pointmaze hydra override)
    local S1_PM_NAME=v5_pm_stage1
    if [ ! -f "${STABLEWM_HOME}/${S1_PM_NAME}/${S1_PM_NAME}_epoch_1_object.ckpt" ]; then
        echo "===> Stage-1 PointMaze"
        $PY train.py \
            data=pointmaze seed=0 \
            loader.batch_size=32 loader.num_workers=4 \
            loader.persistent_workers=True loader.pin_memory=True loader.prefetch_factor=2 \
            trainer.max_epochs=1 +trainer.limit_train_batches=20000 +trainer.limit_val_batches=20 \
            wandb.enabled=False \
            +probe_every=500 +jacobian_probe_n_samples=6 \
            +probing_n_train=256 +probing_n_test=64 +latent_cov_n_samples=256 \
            output_model_name=${S1_PM_NAME} hydra.job.id=${S1_PM_NAME} \
            hydra.run.dir=hydra_runs/${S1_PM_NAME} 2>&1 | tail -10
    else
        echo "===> SKIP ${S1_PM_NAME} (ckpt exists)"
    fi
    local S1PM="${STABLEWM_HOME}/${S1_PM_NAME}/${S1_PM_NAME}_weights.ckpt"
    [ -f "$S1PM" ] || S1PM="${STABLEWM_HOME}/${S1_PM_NAME}/${S1_PM_NAME}_epoch_1_object.ckpt"

    # 3 variants × 2 seeds on PointMaze
    for s in 0 1; do
        train_frozen v5_pm_baseline_seed${s} $S1PM 8000 seed=${s} data=pointmaze
        train_frozen v5_pm_uvlowr_seed${s}   $S1PM 8000 seed=${s} data=pointmaze +predictor.ffn_rank=4
        train_frozen v5_pm_randdiff_seed${s} $S1PM 8000 seed=${s} data=pointmaze \
            +loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2
    done
}

# ---------------------------------------------------------------------------
# B10 — DINOv2 cross-pretraining (v5.2 MUST, ×2 seeds)
# ---------------------------------------------------------------------------
run_B10() {
    echo "##### B10 — DINOv2 cross-pretraining ×2 seeds #####"
    # Build adapter first (no training)
    if [ ! -f "${STABLEWM_HOME}/v5_dino_adapter.pt" ]; then
        $PY tier1_dinov2_adapter.py --build-adapter \
            --out ${STABLEWM_HOME}/v5_dino_adapter.pt \
            --init random --seed 42 2>&1 | tail -5
    fi
    # Run 3 variants × 2 seeds using DINOv2-adapter encoder
    for seed in 0 1; do
        for variant in baseline uvlowr randdiff; do
            case $variant in
                baseline) overrides="" ;;
                uvlowr)   overrides="+predictor.ffn_rank=4" ;;
                randdiff) overrides="+loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2" ;;
            esac
            local name="v5_dino_${variant}_seed${seed}"
            local outdir="${STABLEWM_HOME}/${name}"
            if [ -f "${outdir}/${name}_epoch_1_object.ckpt" ]; then
                echo "===> SKIP $name"
                continue
            fi
            echo "===> [DINOv2] train ${name}"
            $PY tier1_dinov2_adapter.py --train \
                --adapter ${STABLEWM_HOME}/v5_dino_adapter.pt \
                --name ${name} \
                --steps 8000 --batch 16 --seed ${seed} \
                ${overrides} 2>&1 | tail -10
        done
    done
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "$PHASES" in
    all)
        run_M0
        run_B0prime
        run_B1
        gate_M1_to_M2 || { echo "ABORTED at M1→M2 gate"; exit 0; }
        run_B2
        run_B3
        run_B4
        run_B4prime
        run_B5prime
        run_B7
        run_B8
        run_B9
        run_B10
        run_B5
        run_B6
        ;;
    autodl-subset)
        # AutoDL portion of Plan E — fresh Claude on server runs this.
        # Local user runs everything else.
        # Total: ~14 GPU-h ≈ ¥42
        run_M0_quick_sanity || { echo "Sanity failed"; exit 1; }
        run_B4prime_baseline_only
        run_B4prime_uvlowr_only
        run_B9
        echo "===> AutoDL subset complete. Pull results + release instance."
        ;;
    all-partial)
        # v5.2 partial — fits within ¥175 budget (~¥158)
        # CUT: B7 LR robustness, B9 PointMaze (data setup blocker), B4' restricted to randdiff-only
        # KEEP: M0, B0' n=7, B1×2, gate, B2, B3×2, B4×2, B4' randdiff-only, B5'×2, B8, B10×2, B5, B6
        run_M0
        run_B0prime
        run_B1
        gate_M1_to_M2 || { echo "ABORTED at M1→M2 gate"; exit 0; }
        run_B2
        run_B3
        run_B4
        run_B4prime_partial
        run_B5prime
        run_B8
        run_B10
        run_B5
        run_B6
        ;;
    M0)         run_M0 ;;
    B0prime|"B0'")    run_B0prime ;;
    B1)         run_B1 ;;
    gate)       gate_M1_to_M2 ;;
    B2)         run_B2 ;;
    B3)         run_B3 ;;
    B4)         run_B4 ;;
    B4prime|"B4'")    run_B4prime ;;
    B5prime|"B5'")    run_B5prime ;;
    B5)         run_B5 ;;
    B6)         run_B6 ;;
    B7)         run_B7 ;;
    B8)         run_B8 ;;
    B9)         run_B9 ;;
    B10)        run_B10 ;;
    B4prime_baseline)   run_B4prime_baseline_only ;;
    B4prime_uvlowr)     run_B4prime_uvlowr_only ;;
    B4prime_randdiff)   run_B4prime_randdiff_only ;;
    sanity-quick)       run_M0_quick_sanity ;;
    --phase)    shift; eval "run_$1" ;;
    *)
        echo "Usage: $0 [all|all-partial|autodl-subset|M0|B0'|B1|gate|B2|B3|B4|B4'|B5'|B5|B6|B7|B8|B9|B10|B4prime_baseline|B4prime_uvlowr|sanity-quick]"
        exit 1
        ;;
esac

echo "===> [$(date +%H:%M:%S)] v5 pipeline phase=${PHASES} COMPLETE"
