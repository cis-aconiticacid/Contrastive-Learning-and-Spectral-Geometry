# Experiment Tracker v4 — Discharging v3 confounds + uvlowr rank sweep

**Created**: 2026-05-12
**Plan doc**: `EXPERIMENT_PLAN_v4.md`
**v3 reference data**: `/workspace/lewm_autodl_results_v3/`
**v3 Stage-1 ckpt** (reused by B1, B3): `/workspace/lewm_autodl_results_v3/stage1_baseline_seed0_weights.ckpt` (216 MB)
**v4 results dir** (to create): `/workspace/lewm_autodl_results_v4/`

## Status legend
- **TODO** — not yet started
- **RUNNING** — in flight
- **DONE** — finished, results pulled
- **NOT RUN** — explicitly skipped (with reason)

## Runs

| Run ID | Milestone | Block | Purpose | Variant / Override | Stage-1 source | Steps | Seeds | Priority | Status | Notes / key results |
|---|---|---|---|---|---|---|---|---|---|---|
| R100 | M0 | — | Latent-norm probe + ε bracket selection | `v4_eps_bracket_probe.py` | v3 baseline | — | — | MUST | **DONE locally 2026-05-12** | post-proj ‖z‖₂ mean = **33.28** (CLS = 13.95, both nearly fixed-radius); bracket ε ∈ {0.0125, 0.05, 0.2, 0.8} (frac ε/‖z‖ = 0.0004..0.024). Saved `refine-logs/v4_eps_bracket.json` |
| R101 | M0 | — | Frozen-encoder sanity rerun (verify freeze on fresh instance) | `frozen_sanity` | v3 baseline | 1000 (target) / 20 (local smoke) | {0} | MUST | **smoke PASS locally; 1K rerun on AutoDL TODO** | Local 20-step smoke at batch 8 confirmed: probing R² byte-identical across step 0/10/20 (=0.2899), predictor jac σ₁ drift only in trainable predictor (1.956→1.955 over 20 steps), pred_loss converging |
| R110 | M1 | B1 | ε sweep — ε_1 (smallest) | `frozen_randdiff +loss.rand_diff.eps=ε_1` | v3 baseline | 8000 | {0} | MUST | TODO | metric: σ₁, SR, F² vs frozen_baseline |
| R111 | M1 | B1 | ε sweep — ε_2 | `frozen_randdiff +loss.rand_diff.eps=ε_2` | v3 baseline | 8000 | {0} | MUST | TODO | |
| R112 | M1 | B1 | ε sweep — ε_3 (≈ v3 ε=0.05) | `frozen_randdiff +loss.rand_diff.eps=0.05` | v3 baseline | 8000 | {0} | MUST | TODO | reproduces v3 randdiff_seed0 as a check; can skip if exact match expected |
| R113 | M1 | B1 | ε sweep — ε_4 (largest) | `frozen_randdiff +loss.rand_diff.eps=ε_4` | v3 baseline | 8000 | {0} | MUST | TODO | |
| R120 | M2 | B2 | Stage-1 with uvlowr_r4 predictor (joint training) | `stage1_uvlowr +predictor.ffn_rank=4` | — (this IS Stage-1) | 20000 | {0} | MUST | TODO | full LeWM stack; SIGReg active; expect 3 GPU-h |
| R130 | M2 | B2 | Frozen baseline on uvlowr Stage-1 | `frozen_baseline` | uvlowr Stage-1 (R120) | 8000 | {0} | MUST | TODO | depends on R120 |
| R131 | M2 | B2 | Frozen baseline on uvlowr Stage-1 — seed 1 | `frozen_baseline` | uvlowr Stage-1 (R120) | 8000 | {1} | MUST | TODO | depends on R120 |
| R132 | M2 | B2 | Frozen uvlowr_r4 on uvlowr Stage-1 (co-shaped) | `frozen_uvlowr_r4` | uvlowr Stage-1 (R120) | 8000 | {0} | MUST | TODO | depends on R120 |
| R133 | M2 | B2 | Frozen uvlowr_r4 on uvlowr Stage-1 — seed 1 | `frozen_uvlowr_r4` | uvlowr Stage-1 (R120) | 8000 | {1} | MUST | TODO | depends on R120 |
| R134 | M2 | B2 | Frozen randdiff on uvlowr Stage-1 (cross-shaped — the falsifier) | `frozen_randdiff` (ε from M1 winning value) | uvlowr Stage-1 (R120) | 8000 | {0} | MUST | TODO | depends on R120 + M1 verdict |
| R135 | M2 | B2 | Frozen randdiff on uvlowr Stage-1 — seed 1 | `frozen_randdiff` | uvlowr Stage-1 (R120) | 8000 | {1} | MUST | TODO | depends on R120 |
| R140 | M3 | B3 | Rank sweep r=1 | `frozen_uvlowr_r1 +predictor.ffn_rank=1` | v3 baseline | 8000 | {0} | MUST | TODO | watch for NaN |
| R141 | M3 | B3 | Rank sweep r=2 | `frozen_uvlowr_r2 +predictor.ffn_rank=2` | v3 baseline | 8000 | {0} | MUST | TODO | |
| R142 | M3 | B3 | Rank sweep r=4 (reproduces v3 frozen_uvlowr_r4 seed 0 — sanity) | `frozen_uvlowr_r4 +predictor.ffn_rank=4` | v3 baseline | 8000 | {0} | OPTIONAL | NOT RUN | can reuse v3 R004 (h=10=0.318) — same Stage-1, same seed |
| R143 | M3 | B3 | Rank sweep r=8 | `frozen_uvlowr_r8 +predictor.ffn_rank=8` | v3 baseline | 8000 | {0} | MUST | TODO | |
| R144 | M3 | B3 | Rank sweep r=16 | `frozen_uvlowr_r16 +predictor.ffn_rank=16` | v3 baseline | 8000 | {0} | MUST | TODO | |
| R150 | M4 | B4 | Aggregate B1 ε-sweep + B2 cross-Stage-1 + B3 rank-sweep into v4 ANALYSIS.md | aggregate.py + manual writeup | — | — | — | MUST | TODO | output: `lewm_autodl_results_v4/ANALYSIS.md` |

## Decision gates (to fill in post-execution)

### After M0 (R100, R101): GATE
- R100 ε bracket: chose ε ∈ {___, ___, ___, ___}
- R101 sanity: pred_loss drop ___, encoder grad norm ___ → GATE: ___

### After M1 (R110-R113): VERDICTS on Anti-claim A
- For each ε, did frozen_randdiff predictor σ₁ exceed `frozen_baseline σ₁` by ≥30%?
  - ε_1 = ___: σ₁ = ___ (___ vs baseline)
  - ε_2 = ___: σ₁ = ___
  - ε_3 = 0.05: σ₁ = ___
  - ε_4 = ___: σ₁ = ___
- **Anti-claim A verdict**: ___ (falsified / supported / mixed)
- **C2 update from M1**: ___

### After M2 (R120-R135): VERDICTS on Anti-claim B
- Cross-Stage-1 randdiff direction:
  - on baseline-Stage-1 (v3 data): σ₁ −21% vs baseline, SR +10%
  - on uvlowr-Stage-1 (R134, R135): σ₁ ___% vs baseline, SR ___%
- **Anti-claim B verdict**: ___
- **C2 update from M2**: ___

### After M3 (R140-R144): VERDICTS on Anti-claim C + C13
- Rollout MSE at h=10 vs r:
  - r=1: ___
  - r=2: ___
  - r=4: 0.318 (from v3 R004)
  - r=8: ___
  - r=16: ___
- Knee/plateau location: ___
- **Anti-claim C verdict**: ___
- **C13 status post-v4**: ___

### Final claim status (to update CLAIM.md after M4)

| Claim | Pre-v4 status | Post-v4 status | Notes |
|---|---|---|---|
| C1 | non-monotonic in 3 SR points | TBD | rank-sweep adds more SR points |
| C2 | suggestive (v3) | TBD via M1 + M2 | main push |
| C3 | weakened (only h=10 in v3) | TBD via M2 + M3 | |
| C4 | confirmed (v2+v3) | TBD | likely unchanged |
| C13 | confirmed at 3 points | TBD via M3 | upgrade to "knee at r=4" if rank-sweep shows it |

## Reuse / external references

- v2 reference data: `/workspace/lewm_autodl_results_v2/` (3 variants × 3 seeds × 40K joint)
- v3 results: `/workspace/lewm_autodl_results_v3/`
- v3 Stage-1 ckpt: `/workspace/lewm_autodl_results_v3/stage1_baseline_seed0_weights.ckpt` (used by B1, B3)
- v3 frozen_uvlowr_r4_seed0 result: rollout h=10 = 0.318 (reused as the r=4 point in B3)
- v3 ANALYSIS.md: `/workspace/lewm_autodl_results_v3/ANALYSIS.md`

## Budget tracking

- Planned: 15.5 GPU-h (~¥47)
- Spent: ¥0 (not started)
- Remaining balance pre-v4: ¥175.32
- Expected balance post-v4: ~¥128

## Local smoke-test results (2026-05-12, RTX 4060 8GB, batch 8)

Validated launch commands BEFORE renting AutoDL:

| Test | Cmd shape | Verified |
|---|---|---|
| R100 ε-bracket probe | `python v4_eps_bracket_probe.py` | Loads ckpt → encodes 100 episodes → ‖z_proj‖ = 33.28, narrow band; bracket saved. **End-to-end works, ε bracket usable** |
| R101 freeze sanity (20 steps) | `train.py +freeze.enabled=true +freeze.stage1_ckpt=<path> +freeze.skip_sigreg=true` | Probing R² byte-identical across steps (freeze callback works); pred jac drift only in trainable layers; pred_loss=0.35 at step 20 vs 0.23 val |
| R110 ε override (15 steps) | `... +loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.2 +loss.rand_diff.num_dirs=2` | `fit/rand_diff_loss=0.564` logged correctly; `fit/loss = pred_loss + 0.01·rand_diff_loss` arithmetic checks out; pred jac σ₁ shrinks 1.95→1.83 at ε=0.2 (matches v3 direction) |
| R140 rank override (15 steps) | `... +predictor.ffn_rank=2` | `predictor_params: 6.13M` vs baseline `10.79M` (−43%) — UV factorization actually applied |

**Local note**: `num_workers=0` is required on WSL2 (default `/dev/shm` is 64 MB; multiprocessing dataloader OOMs SHM). On AutoDL set `num_workers=4` as in v3.

**Speed**: 4060 at batch 8 runs at ~7 it/s, so 8000 steps = ~19 min. 4090 at batch 32 will be similar wall-clock. v4 total compute estimate (15.5 GPU-hr) holds.

## Operational checklist (v3 lessons; do BEFORE first launch)

- [ ] Create AutoDL instance with `expand_system_disk_by_gb=80` from start (108 GB total)
- [ ] First `pip` speedtest >2 MB/s; if not, release + recreate
- [ ] `~/.pip/pip.conf` set to tsinghua mirror BEFORE running pip install torch
- [ ] `flock -n /workspace/_setup_v4.lock` wraps setup script
- [ ] All SSH commands prefix with `cd /workspace/le-wm &&`
- [ ] v3 Stage-1 ckpt uploaded to instance (`scp stage1_baseline_seed0_weights.ckpt`) before B1 launches
- [ ] After last `scp` pull, **release instance immediately** (¥3/hr while running)
