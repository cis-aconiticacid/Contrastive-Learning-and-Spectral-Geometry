# Experiment Tracker (canonical, v5.2 — "no cost saving")

**Created**: 2026-05-12
**Plan**: `EXPERIMENT_PLAN.md` (v5.2)
**Framing**: encoder-mediated compensation (C2) + nuclear-norm over-contraction (C_dyn) + NDE/DEQ contrast. Multi-seed everywhere; cross-dataset (PointMaze) + cross-encoder (DINOv2).
**Stage-1 ckpts**:
- v3 baseline (Push-T, 20K joint): `/workspace/lewm_autodl_results_v3/stage1_baseline_seed0_weights.ckpt`
- v5 uvlowr (Push-T): R220 to be trained
- v5 baseline seed=1: R310 to be trained
- v5 PointMaze: R400 to be trained
- DINOv2 ViT-S/14: torch.hub at runtime

**v5 results dir**: `/workspace/lewm_autodl_results_v5/`

## Pre-registered predictions (v5.2 — multi-seed mean ± empirical σ_noise)

| ID | Block | Prediction |
|---|---|---|
| P-C2 | B1 (×2 seeds) | ≥3/4 ε: mean σ₁(randdiff,ε) ≤ baseline_mean − 2σ_noise[σ₁] |
| P-Cdyn | B4 + B5 | mean \|λ_max\|(λ=0.1) ≤ \|λ_max\|(λ=0) − 2σ_noise AND ρ_mean(λ=0.01) < 0.95 |
| P-C13 | B3 (×2 seeds) | argmin rollout-MSE@h=10 in {2,4} AND r=16 ≥ r=4 − 2σ_noise |
| P-Anti-I | B4' (3 variants × 40K) | sign(σ₁(randdiff) − σ₁(baseline)) preserved 8K → 40K |
| P-Anti-H | B5' (×2 seeds) | All 4 corners: σ₁ ≤ baseline + 2σ_noise |
| P-Anti-J | B10 (×2 seeds) | DINOv2 randdiff σ₁ < DINOv2 baseline σ₁ |
| P-Anti-K | B9 (×2 seeds) | PointMaze randdiff σ₁ − PointMaze baseline σ₁ < 0 |

**Abort gate M1→M2**: any ε B1 produces mean σ₁ > baseline + 0.20 → STOP (saves 60+ GPU-h).
**Overall**: ≥2/3 main P-* AND P-Anti-I AND P-Anti-H AND P-Anti-J AND P-Anti-K all hold → paper draft.

## Runs (multi-seed expanded)

| Run ID | M | Block | Variant / Override | Stage-1 | Steps | Seed | Status |
|---|---|---|---|---|---|---|---|
| R200 | M0 | — | ε bracket probe (analysis) | v3 base | — | — | **DONE locally** |
| R201 | M0 | — | freeze sanity | v3 base | 1000 | 0 | TODO |
| **B0' noise floor (n=7)** ||||||||
| R002 | M0.5 | B0' | `frozen_baseline` (v3 R002) | v3 base | 8000 | 0 | **REUSE** |
| R003 | M0.5 | B0' | `frozen_baseline` (v3 R003) | v3 base | 8000 | 1 | **REUSE** |
| R202 | M0.5 | B0' | `frozen_baseline seed=2` | v3 base | 8000 | 2 | TODO |
| R203 | M0.5 | B0' | `frozen_baseline seed=3` | v3 base | 8000 | 3 | TODO |
| R204 | M0.5 | B0' | `frozen_baseline seed=4` | v3 base | 8000 | 4 | TODO |
| R205 | M0.5 | B0' | `frozen_baseline seed=5` | v3 base | 8000 | 5 | TODO |
| R206 | M0.5 | B0' | `frozen_baseline seed=6` | v3 base | 8000 | 6 | TODO |
| **B1 ε sweep ×2 seeds** ||||||||
| R210 | M1 | B1 | `frozen_randdiff +eps=0.0125` | v3 base | 8000 | 0 | TODO |
| R211 | M1 | B1 | `frozen_randdiff +eps=0.0125` | v3 base | 8000 | 1 | TODO |
| R006 | M1 | B1 | `frozen_randdiff +eps=0.05` (v3 R006) | v3 base | 8000 | 0 | **REUSE** |
| R212 | M1 | B1 | `frozen_randdiff +eps=0.05` | v3 base | 8000 | 1 | TODO |
| R213 | M1 | B1 | `frozen_randdiff +eps=0.2` | v3 base | 8000 | 0 | TODO |
| R214 | M1 | B1 | `frozen_randdiff +eps=0.2` | v3 base | 8000 | 1 | TODO |
| R215 | M1 | B1 | `frozen_randdiff +eps=0.8` | v3 base | 8000 | 0 | TODO |
| R216 | M1 | B1 | `frozen_randdiff +eps=0.8` | v3 base | 8000 | 1 | TODO |
| **⭐ GATE M1→M2** ||||||||
| **B2 uvlowr-Stage-1 + freeze 3×2** ||||||||
| R220 | M2 | B2 | Stage-1 `+ffn_rank=4`, SIGReg | — | 20000 (+10K opt) | 0 | TODO |
| R230 | M2 | B2 | `frozen_baseline` | R220 | 8000 | 0 | TODO |
| R231 | M2 | B2 | `frozen_baseline` | R220 | 8000 | 1 | TODO |
| R232 | M2 | B2 | `frozen_uvlowr_r4 +ffn_rank=4` | R220 | 8000 | 0 | TODO |
| R233 | M2 | B2 | `frozen_uvlowr_r4 +ffn_rank=4` | R220 | 8000 | 1 | TODO |
| R234 | M2 | B2 | `frozen_randdiff` (ε from M1) | R220 | 8000 | 0 | TODO |
| R235 | M2 | B2 | `frozen_randdiff` | R220 | 8000 | 1 | TODO |
| **B3 rank sweep ×2 seeds** ||||||||
| R240 | M3 | B3 | `+ffn_rank=1` | v3 base | 8000 | 0 | TODO |
| R241 | M3 | B3 | `+ffn_rank=1` | v3 base | 8000 | 1 | TODO |
| R242 | M3 | B3 | `+ffn_rank=2` | v3 base | 8000 | 0 | TODO |
| R243 | M3 | B3 | `+ffn_rank=2` | v3 base | 8000 | 1 | TODO |
| R004 | M3 | B3 | `+ffn_rank=4` (v3 R004) | v3 base | 8000 | 0 | **REUSE** |
| R244 | M3 | B3 | `+ffn_rank=4` | v3 base | 8000 | 1 | TODO |
| R245 | M3 | B3 | `+ffn_rank=8` | v3 base | 8000 | 0 | TODO |
| R246 | M3 | B3 | `+ffn_rank=8` | v3 base | 8000 | 1 | TODO |
| R247 | M3 | B3 | `+ffn_rank=16` | v3 base | 8000 | 0 | TODO |
| R248 | M3 | B3 | `+ffn_rank=16` | v3 base | 8000 | 1 | TODO |
| **B4 λ sweep ×2 seeds** ||||||||
| R250 | M4 | B4 | λ=0 reuse R002 | v3 base | — | 0 | **REUSE** |
| R251 | M4 | B4 | λ=0 reuse R003 | v3 base | — | 1 | **REUSE** |
| R252 | M4 | B4 | `+weight=0.001` | v3 base | 8000 | 0 | TODO |
| R253 | M4 | B4 | `+weight=0.001` | v3 base | 8000 | 1 | TODO |
| R254 | M4 | B4 | `+weight=0.003` | v3 base | 8000 | 0 | TODO |
| R255 | M4 | B4 | `+weight=0.003` | v3 base | 8000 | 1 | TODO |
| R256 | M4 | B4 | λ=0.01 reuse R006 | v3 base | — | 0 | **REUSE** |
| R257 | M4 | B4 | `+weight=0.01` | v3 base | 8000 | 1 | TODO |
| R258 | M4 | B4 | `+weight=0.03` | v3 base | 8000 | 0 | TODO |
| R259 | M4 | B4 | `+weight=0.03` | v3 base | 8000 | 1 | TODO |
| R260 | M4 | B4 | `+weight=0.1` | v3 base | 8000 | 0 | TODO |
| R261 | M4 | B4 | `+weight=0.1` | v3 base | 8000 | 1 | TODO |
| **⭐ B4' 40K extension all 3 variants** ||||||||
| R270 | M4.5 | B4' | `frozen_baseline_40K` | v3 base | **40000** | 0 | TODO |
| R271 | M4.5 | B4' | `frozen_uvlowr_40K +ffn_rank=4` | v3 base | **40000** | 0 | TODO |
| R272 | M4.5 | B4' | `frozen_randdiff_40K` | v3 base | **40000** | 0 | TODO |
| **⭐ B5' (ε,λ) 4 corners ×2 seeds** ||||||||
| R280 | M4.6 | B5' | LL: (0.0125, 0.001) | v3 base | 8000 | 0 | TODO |
| R281 | M4.6 | B5' | LL: (0.0125, 0.001) | v3 base | 8000 | 1 | TODO |
| R282 | M4.6 | B5' | LH: (0.0125, 0.1) | v3 base | 8000 | 0 | TODO |
| R283 | M4.6 | B5' | LH: (0.0125, 0.1) | v3 base | 8000 | 1 | TODO |
| R284 | M4.6 | B5' | HL: (0.8, 0.001) | v3 base | 8000 | 0 | TODO |
| R285 | M4.6 | B5' | HL: (0.8, 0.001) | v3 base | 8000 | 1 | TODO |
| R286 | M4.6 | B5' | HH: (0.8, 0.1) | v3 base | 8000 | 0 | TODO |
| R287 | M4.6 | B5' | HH: (0.8, 0.1) | v3 base | 8000 | 1 | TODO |
| **⭐ B5 DMD + ρ post-hoc (no training)** ||||||||
| R290 | M5 | B5 | DMD+ρ on ALL v5 + reuse ckpts | analysis | — | — | TODO |
| **B6 aggregation** ||||||||
| R295 | M6 | B6 | Aggregate + Fig 2-6 + ANALYSIS.md (NDE contrast caption) | analysis | — | — | TODO |
| **⭐ B7 LR robustness (PROMOTED MUST)** ||||||||
| R002b | M7 | B7 | reuse R002 as LR=5e-5 control | reuse | — | 0 | **REUSE** |
| R006b | M7 | B7 | reuse R006 as LR=5e-5 control | reuse | — | 0 | **REUSE** |
| R300 | M7 | B7 | `frozen_baseline +lr=5e-4` | v3 base | 8000 | 0 | TODO |
| R301 | M7 | B7 | `frozen_randdiff +lr=5e-4` | v3 base | 8000 | 0 | TODO |
| **⭐ B8 2nd Stage-1 baseline seed (PROMOTED MUST)** ||||||||
| R310 | M8 | B8 | Stage-1 baseline seed=1, 20K joint | — | 20000 | 1 | TODO |
| R311 | M8 | B8 | `frozen_baseline` | R310 | 8000 | 0 | TODO |
| R312 | M8 | B8 | `frozen_uvlowr_r4 +ffn_rank=4` | R310 | 8000 | 0 | TODO |
| R313 | M8 | B8 | `frozen_randdiff` | R310 | 8000 | 0 | TODO |
| **⭐⭐⭐ B9 PointMaze cross-dataset (PROMOTED FROM CUT to MUST)** ||||||||
| R400 | M9 | B9 | Stage-1 PointMaze, 20K joint | — | 20000 | 0 | TODO |
| R410 | M9 | B9 | `frozen_baseline` PM | R400 | 8000 | 0 | TODO |
| R411 | M9 | B9 | `frozen_baseline` PM | R400 | 8000 | 1 | TODO |
| R412 | M9 | B9 | `frozen_uvlowr_r4` PM | R400 | 8000 | 0 | TODO |
| R413 | M9 | B9 | `frozen_uvlowr_r4` PM | R400 | 8000 | 1 | TODO |
| R414 | M9 | B9 | `frozen_randdiff` PM | R400 | 8000 | 0 | TODO |
| R415 | M9 | B9 | `frozen_randdiff` PM | R400 | 8000 | 1 | TODO |
| **⭐ B10 DINOv2 cross-pretraining ×2 seeds (PROMOTED MUST)** ||||||||
| R500 | M10 | B10 | DINOv2 adapter setup (analysis) | — | — | — | TODO |
| R510 | M10 | B10 | `dino_baseline` | DINOv2 | 8000 | 0 | TODO |
| R511 | M10 | B10 | `dino_baseline` | DINOv2 | 8000 | 1 | TODO |
| R512 | M10 | B10 | `dino_uvlowr +ffn_rank=4` | DINOv2 | 8000 | 0 | TODO |
| R513 | M10 | B10 | `dino_uvlowr +ffn_rank=4` | DINOv2 | 8000 | 1 | TODO |
| R514 | M10 | B10 | `dino_randdiff` | DINOv2 | 8000 | 0 | TODO |
| R515 | M10 | B10 | `dino_randdiff` | DINOv2 | 8000 | 1 | TODO |

**Totals**:
- New training runs: **70**
- Reused from v3: **8 contexts** (R002 ×3, R003 ×2, R004 ×1, R006 ×3)
- Analysis-only: **2** (B5 DMD+ρ; B6 aggregate)
- Wall-clock: ~36h AutoDL session

## Budget

| Bucket | Runs | hr/run | Subtotal |
|---|---|---|---|
| 8K frozen | 56 | 0.8 | 44.8 |
| 20K Stage-1 (B2 R220, B8 R310, B9 R400) | 3 | 3.0 | 9.0 |
| 40K extension (B4') | 3 | 3.0 | 9.0 |
| DINOv2 8K | 6 | 0.8 | 4.8 |
| Adapter + analysis | — | — | 0.7 |
| **Total** | | | **~67.0 GPU-h ≈ ¥201** |

**Balance** ¥175 → **over by ~¥26**. Top up ¥30 OR apply cuts:
1. B0' to n=5 (save ¥6)
2. B4' to randdiff-only (save ¥18)
3. B7 LR robustness (save ¥5)

## Decision gates (post-execution)

### After M0 (R201)
pred_loss drop ≥30% AND enc grad=0 → PASS → M0.5

### After M0.5 (R002, R003, R202-R206): σ_noise table

| Metric | mean | std (σ_noise) | 2σ |
|---|---|---|---|
| σ₁ | ___ | ___ | ___ |
| F² | ___ | ___ | ___ |
| SR | ___ | ___ | ___ |
| rollout MSE h=10 | ___ | ___ | ___ |
| \|λ_max\| | ___ | ___ | ___ |
| ρ_mean | ___ | ___ | ___ |

Sanity: max |seed value − mean| < 3σ for each → PASS

### After M1 (B1 8 runs): Anti-A + abort gate
For each ε, mean over 2 seeds σ₁(randdiff,ε):
- ε=0.0125: ___
- ε=0.05: ___ (R006 reused for s=0)
- ε=0.2: ___
- ε=0.8: ___

**P-C2**: ___  **ABORT GATE**: ___

### After M2 (B2 7 runs): Anti-B
Stage-1 convergence: ___
randdiff sign-flip on uvlowr-S1 (mean over 2 seeds): σ₁ Δ = ___, F² Δ = ___, SR Δ = ___

### After M3 (B3 9 runs): Anti-D + C13
Rollout MSE@h=10 mean over 2 seeds: r=1 ___ r=2 ___ r=4 ___ r=8 ___ r=16 ___
P-C13: ___

### After M4 (B4 9 runs): Anti-C + C_dyn
\|λ_max\| mean over 2 seeds across λ ∈ {0, 0.001, 0.003, 0.01, 0.03, 0.1}: ___, ___, ___, ___, ___, ___
Monotone? ___  P-Cdyn: ___

### After M4.5 (B4' 3 runs): Anti-I
σ₁ at step 40K — baseline ___, uvlowr ___, randdiff ___
sign(σ₁(randdiff) − σ₁(baseline)) preserved? ___
P-Anti-I: ___

### After M4.6 (B5' 8 runs): Anti-H
4-corner σ₁ mean over 2 seeds: LL ___, LH ___, HL ___, HH ___
Any corner > baseline + 2σ_noise: ___
P-Anti-H: ___

### After M7 (B7 2 new): Anti-E
F² randdiff vs baseline at LR=5e-4: ___
Match at LR=5e-5: ___ (reuse v3)

### After M8 (B8 4 runs): Anti-L
Sign-flip on Stage-1 seed=1: σ₁ Δ = ___
Match Stage-1 seed=0 direction: ___

### After M9 (B9 7 runs): Anti-K
PointMaze randdiff σ₁ direction mean over 2 seeds: σ₁(pm_randdiff) − σ₁(pm_baseline) = ___
Match Push-T v3 direction: ___
P-Anti-K: ___

### After M10 (B10 6 runs): Anti-J
DINOv2 randdiff σ₁ direction mean over 2 seeds: σ₁(dino_randdiff) − σ₁(dino_baseline) = ___
Match Push-T v3 direction: ___
P-Anti-J: ___

### After M5 (R290): DMD + ρ
\|λ_max\|(8K) baseline ___, randdiff ___ (tier-1 3K: 1.095 / 0.996)
ρ_mean(randdiff λ=0.01): ___
r(\|λ_max\|, ρ_mean) across λ sweep: ___

### Final claim status (post-M6)

| Claim | Pre-v5 | Post-v5.2 | Driving evidence |
|---|---|---|---|
| C2 | suggestive | TBD | B0' + B1 + B2 + B4' + B5' + B7 + B8 + B9 — 8 lines |
| C_dyn | tier-1 | TBD | B4 + B5 (ρ) + B10 + Anti-NDE/DEQ contrast |
| C13 | confirmed 3-point | TBD | B3 × 2 seeds knee |
| C3 (uvlowr Pareto) | weakened | TBD | B3 + B2 + B9 |
| C4 | confirmed | TBD | B4 + B4' + B9 |

## Reuse / external references

- v2 (40K joint, Push-T): `/workspace/lewm_autodl_results_v2/`
- v3 (8K frozen, Push-T): `/workspace/lewm_autodl_results_v3/`
- v3 Stage-1 ckpt: `/workspace/lewm_autodl_results_v3/stage1_baseline_seed0_weights.ckpt`
- R002 (frozen_baseline_seed0): σ₁=2.05, F²=87.5, SR=20.8, h=10=0.267 — REUSED in B0', B4 λ=0 s=0, B7 LR=5e-5
- R003 (frozen_baseline_seed1): σ₁=2.10, h=10=0.275 — REUSED in B0', B4 λ=0 s=1
- R004 (frozen_uvlowr_r4_seed0): h=10=0.318 — REUSED in B3 r=4 s=0
- R006 (frozen_randdiff_seed0): σ₁=1.61, F²=59.2, SR=22.7, h=10=0.961 — REUSED in B1 ε=0.05, B4 λ=0.01 s=0, B7

## Operational checklist

- [ ] AutoDL balance ≥¥200 (top up ¥30 if needed)
- [ ] `expand_system_disk_by_gb=120` (extra for PointMaze + DINOv2)
- [ ] First pip speedtest ≥2 MB/s
- [ ] tsinghua mirror
- [ ] flock guards
- [ ] SSH cmd prefix `cd /workspace/le-wm &&`
- [ ] Upload v3 Stage-1 ckpt
- [ ] **PointMaze data prep**: LeWM repo download script
- [ ] **DINOv2 cache prep**: pre-warm `torch.hub.load(...)`
- [ ] B0' first → `v5_noise_floor.json` → P-* gate scripts read it
- [ ] After B1, run abort-gate check before B2
- [ ] After B2 Stage-1 (R220), run convergence-gate check
- [ ] B5 (DMD + ρ) batched at end over all v5 ckpts
- [ ] Release immediately after final scp pull
