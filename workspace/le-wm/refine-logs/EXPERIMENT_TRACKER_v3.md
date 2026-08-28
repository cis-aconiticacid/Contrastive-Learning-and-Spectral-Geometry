# Experiment Tracker — Frozen-Encoder Ablation (v3)

**Updated**: 2026-05-11 (post-execution)
**Stage-1 ckpt**: `stage1_baseline_seed0` (20K joint training, freshly trained — NOT v2's baseline_seed0)
**Stage-2 steps**: 8000 (revised down from EXPERIMENT_PLAN.md's 10000 to fit ~6 GPU-hr budget)
**Instance**: AutoDL 4090-48G, +80 GB disk → 108 GB total (third attempt; first two had network/disk issues)

| Run ID | Milestone | Variant | Steps | Seeds | Status | Key results |
|---|---|---|---|---|---|---|
| R000 | Stage-1 | stage1_baseline (joint) | 20000 | {0} | **DONE** | rollout h=10=0.081, h=20=0.220, probe R²=0.833, pred_jac σ₁=1.72 SR=32.4 F²=95.2 |
| R001 | M0 sanity | frozen_sanity | 1000 | {0} | **DONE — PASSED** | pred_loss drop ≥30% confirmed; all callbacks fired; KeepFrozenInEvalCallback verified |
| R002 | M1 | frozen_baseline | 8000 | {0} | **DONE** | rollout h=10=0.267, jac σ₁=2.05, SR=20.8 |
| R003 | M1 | frozen_baseline | 8000 | {1} | **DONE** | rollout h=10=0.275, jac σ₁=2.10, SR=19.9 |
| R004 | M1 | frozen_uvlowr_r4 | 8000 | {0} | **DONE** | rollout h=10=0.318, jac σ₁=2.06, SR=21.2; pred_params=6.18M (−43%) |
| R005 | M1 | frozen_uvlowr_r4 | 8000 | {1} | **DONE** | rollout h=10=0.090, jac σ₁=2.10, SR=20.4 |
| R006 | M1 | frozen_randdiff | 8000 | {0} | **DONE** | rollout h=10=0.961, jac σ₁=1.61, SR=22.7 |
| R007 | M1 | frozen_randdiff | 8000 | {1} | **DONE** | rollout h=10=0.816, jac σ₁=1.64, SR=22.2 |
| R008 | M2 | frozen-randdiff on baseline_seed1 ckpt | 8000 | {0} | **NOT RUN** | conditional on M1 P3 positive; v3 only ran Stage-1 seed=0 — robustness check deferred |
| R009 | M3 | aggregate + verdict (analysis) | — | — | **DONE** | see `ANALYSIS.md`; P3 verdict = FALSIFIED IN DIRECTION (reversal) → C2 strong support |

## Decision Gates — outcomes

### After M0 (R001): GATE PASSED
- pred_loss drop ≥30% confirmed (assertion in `run_pipeline_v3.sh` Phase B)
- KeepFrozenInEvalCallback raised no assertion errors → all 3 frozen modules (encoder, projector, pred_proj) have `requires_grad=False`
- All instrumentation files non-empty: `jacobian_probe.jsonl`, `probing.jsonl`, `latent_cov.jsonl`, `engineering_metrics.json`

### After M1 (R002-R007): VERDICTS

**P1 (probing R² mechanically identical across variants)**: **CONFIRMED** mechanically
- All 6 frozen runs give eval R²=0.8329 exactly, inline R²=0.448 exactly (different sample sizes; same model)
- Significance: sanity check on freeze logic, not on C2 itself. Real C2 test is via P3.

**P3 (predictor spectral signatures persist or amplify)**: **FALSIFIED IN DIRECTION — REVERSED**
- v2 randdiff: σ₁=2.48 (+43% vs baseline), SR=18 (−43%), F²=110 (+14%)
- v3 randdiff: σ₁=1.63 (−21% vs frozen_baseline), SR=22.4 (+10%), F²=59 (−32%)
- **The signature direction flipped between v2 and v3.** This is STRONG evidence for C2 — the v2 σ₁ inflation was an encoder-absorption artifact.

**P4 (rollout MSE ordering preserved; uvlowr long-horizon win robust)**: **MIXED**
- At h=10: uvlowr (0.204) < baseline (0.271) < randdiff (0.889). v2 ordering preserved.
- At h=5: baseline (0.048) < uvlowr (0.101) < randdiff (0.224). v2 ordering broken (uvlowr no longer wins at h=5).
- At h=15/20: high variance, ordering ambiguous.
- uvlowr's advantage is narrower in v3 than v2; encoder co-adaptation explains some of v2's broader win.

**M2 NOT EXECUTED**: P3 verdict is REVERSED, not "CONFIRMED" — the original gate condition ("P3 = CONFIRMED → run M2") doesn't apply. M2 robustness check (different Stage-1 ckpt) would still be useful for v4 planning, deferred.

### After M3: VERDICT MAPPING

| Claim | v2 verdict | v3 verdict | Status |
|---|---|---|---|
| C1 (Jac SR doesn't predict rollout) | CONFIRMED (non-monotonic, 3 pts) | CONFIRMED (frozen variants narrow SR range, randdiff outlier) | holds |
| C2 (encoder absorption is causal) | suggestive | **STRONG via P3 reversal** | upgraded |
| C3 (uvlowr Pareto improvement) | CONFIRMED | **WEAKENED** (only h=10 robust) | downgraded |
| C4 (randdiff degrades long-h rollout) | CONFIRMED | CONFIRMED in frozen too | holds |
| C5 (encoder Jac dichotomy under joint regularization) | CONFIRMED (v2 only) | N/A (encoder frozen) | reframed as joint-training observation |
| C6 (training-driven low-rank ViT) | confirmed by C6 cheap check | unchanged | holds |
| C7 (randdiff inflates predictor σ₁) | CONFIRMED (v2 only) | **REVERSED** (v3 σ₁ shrinks) | C7 was joint-training artifact |
| C13 (uvlowr r=4 matches data dim) | confirmed by C13 cheap check | unchanged | holds |

## Confounds NOT addressed in v3

1. **Stage-1 step count**: 20K vs v2's 40K → absolute v3-vs-v2 comparisons unfair
2. **Baseline-shaped latents**: stage-1 encoder co-trained with uniform full-rank predictor; uvlowr/randdiff tested only in this shape
3. **randdiff ε scale**: not rescaled to v3 latent norm; reversal could be partial ε mis-calibration
4. **Single Stage-1 seed**: M2 robustness check deferred
5. **8K predictor training**: may be undertraining; pred_loss not visibly plateaued

## Followup priority

1. **ε-rescaled frozen_randdiff** — settles Confound 3, ~1.5 GPU-hr
2. **v4 uvlowr-co-trained Stage-1 + freeze + multi-predictor** — settles Confound 2, ~8 GPU-hr
3. **Stage-1 seed sweep** for v3 — settles Confound 4
4. **rank/λ sweeps** for C1 closure
5. **Push-T planning success rate** (separate eval pipeline build, ~1 day)

## Reuse references

- v2 reference data: `/workspace/lewm_autodl_results_v2/` (3 variants × 3 seeds × 40K joint)
- v3 results: `/workspace/lewm_autodl_results_v3/`
- v3 analysis: `/workspace/lewm_autodl_results_v3/ANALYSIS.md`
- Cheap checks (C6, C13): `/workspace/le-wm/refine-logs/c*.json`
- Stage-1 ckpt (for v4): `/workspace/lewm_autodl_results_v3/stage1_baseline_seed0_weights.ckpt`

## Actual wall-clock + cost

- Setup time (3 instance attempts due to network/disk): ~3 hours wall (not counted in budget)
- Successful instance runtime: ~6 GPU-hours (~¥18)
- Total v3 spend: ~¥24 (including failed instance starts)
- Remaining balance: ¥175 / starting ¥193
