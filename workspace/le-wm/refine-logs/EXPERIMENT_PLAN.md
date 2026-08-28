# Experiment Plan (canonical, v5.2 — 2026-05-12, "go big")

## Motivation

**Problem**: In jointly-trained JEPA (LeWM on Push-T), predictor-side regularization is not predictor-local. v2 (joint, 40K) vs v3 (frozen, 8K) experiments show the Scarvelis-Solomon nuclear-norm (randdiff) penalty produces opposite predictor-Jacobian signatures across 6 independent metrics. Tier-1 DMD on frozen runs additionally shows randdiff drives `|λ_max| < 1` — predictor becomes a contraction map, converging to a fixed point, explaining its rollout blowup.

**Method Thesis**: *Encoder-mediated compensation can mask and even invert predictor-side regularization effects in end-to-end JEPAs.* When the encoder is allowed to co-adapt, randdiff's apparent "σ₁ inflation" is an encoder-shaping artifact, not an intrinsic predictor property; with the encoder frozen, randdiff over-contracts the predictor into fixed-point dynamics.

**Contrast with prior literature**: Counter-intuitively, this finding *contradicts* the prevailing wisdom in neural-ODE / DEQ literature where Jacobian regularization **stabilizes** long-horizon predictions (Finlay et al. NODE, Bai et al. DEQ, Scarvelis-Solomon NeurIPS 2024). We show that for JEPA-style **discrete recurrent predictors with a shared embedding space**, the same regularization causes **over-contraction and rollout collapse** — the opposite of the NDE result. Propagated to **Fig 4** caption.

**Date**: 2026-05-12 (v5.2 "no cost saving" pass)
**Budget**: MUST-RUN **~67 GPU-h (~¥202)** of ¥175 balance — over by ¥27, top up or accept.

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| **C2 (anchor)** | Reframes joint-JEPA predictor claims | (i) sign-flip survives ε sweep ×2 seeds (B1); (ii) (ε,λ) 4-corner ×2 seeds (B5'); (iii) Stage-1 shape (B2); (iv) 40K Stage-2 ALL 3 variants (B4'); (v) LR perturbation (B7); (vi) Stage-1 seed (B8); (vii) **DIFFERENT DATASET (B9 PointMaze)**; (viii) σ_noise from n=7 baseline (B0') | B0', B1, B2, B4', B5', B7, B8, B9 |
| **C_dyn (anchor 2)** — contradicts NDE/DEQ stabilization narrative | Mechanistic explanation; paper's most surprising finding | (ix) λ dose-response ×2 seeds (B4); (x) **nonlinear ρ < 1** (B5); (xi) DMD replicates at 8K (B5); (xii) **DIFFERENT ENCODER (B10 DINOv2)** | B4, B5, B10 |
| **C13 (supporting)** | "Lean LeWM" framing | rank sweep ×2 seeds (B3) knee at r=4 | B3 |
| **Anti-A** (ε mis-calibration) | Killed by B1 | B1 |
| **Anti-B** (Stage-1 shape) | Killed by B2 | B2 |
| **Anti-C** (λ-specific) | Killed by B4 monotone | B4 |
| **Anti-D** (r=4 lucky) | Killed by B3 knee | B3 |
| **Anti-E** (optimizer/LR) | Killed by B7 | B7 |
| **Anti-F** (n=1 ≤ noise) | Killed by B0' n=7 σ_noise | B0' |
| **Anti-G** (linear spectrum only) | Killed by B5 ρ < 1 | B5 |
| **Anti-H** ((ε,λ) interaction) | Killed by B5' 4 corners ×2 | B5' |
| **Anti-I** (8K under-training) | Killed by B4' ALL 3 at 40K | B4' |
| **Anti-J** (LeWM-encoder specific) | Killed by B10 DINOv2 | B10 |
| **Anti-K** ⭐ NEW (Push-T only) | Killed by B9 PointMaze | B9 |
| **Anti-L** ⭐ NEW (Stage-1 seed specific) | Killed by B8 | B8 |

**Demoted to supporting** (per novelty check): TLPS, cross-metric correlation, C6.

---

## Paper Storyline

**Main paper proves:**
1. **C2** — joint vs frozen sign reversal, robust across **8 dimensions** (ε, λ, (ε,λ) factorial, Stage-1 shape, Stage-1 seed, Stage-2 duration, optimizer/LR, **second dataset**)
2. **C_dyn** — λ dose-response + nonlinear ρ + cross-encoder (DINOv2) replication; explicit NDE/DEQ contrast in Fig 4 caption
3. **C13** — rank knee at r=4 ×2 seeds

**Appendix:** v2/v3 full tables, Tier-0 figures, Tier-1 trajectory metrics with Wang et al. citation, C6 cheap-check, v3→v5.2 verdict evolution, B0' σ_noise table, B7 LR ablation, B8 Stage-1-seed-1, raw DMD scatter plots.

**Cut for v5.2 → follow-up:** Reacher (3rd dataset), full ε×λ factorial heatmap, planning-success eval, depth/width ablation.

---

## Experiment Blocks

### B0 (M0): pre-flight (ε bracket done locally + AutoDL sanity)
- R200 analysis-only (DONE), R201 1K freeze sanity. **Cost**: 0.3 GPU-h. **MUST**

### B0' (M0.5) — cross-seed noise floor (n=7, ⭐ expanded from 5) — kills Anti-F
- **Variants**: `frozen_baseline` × seeds {0 (reuse), 1 (reuse), **2, 3, 4, 5, 6** (NEW)} = 5 new runs
- **Metrics**: σ₁, F², SR, σ₁/σ₁₀, β, |λ_max|, ρ_mean, rollout MSE per seed
- **Output**: `v5_noise_floor.json` with mean ± std → 2σ thresholds for P-* gates
- **Cost**: 5 × 0.8h = **4.0 GPU-h**. **MUST**

### B1 (M1): ε sweep — kills Anti-A. **Multi-seed (n=2)** ⭐
- **Variants**: `frozen_randdiff` at ε ∈ {0.0125, 0.05 (reuse R006 s=0), 0.2, 0.8} × **2 seeds**
- Cells: 4×2 = 8; new: 7
- **Decision (P-C2)**: ≥3 of 4 ε give mean σ₁ ≤ baseline_mean − 2·σ_noise[σ₁]
- **Abort gate M1→M2**: any ε σ₁ > baseline+0.20 → STOP
- **Cost**: 7 × 0.8h = **5.6 GPU-h**. **MUST**

### B2 (M2): uvlowr-Stage-1 + freeze multi-predictor — kills Anti-B
- Stage-1 (20K joint with convergence gate; auto-extend to 30K if needed) + 6 frozen (3v × 2 seeds × 8K)
- **Stage-1 gate**: pred_loss plateau in [18K, 20K] AND probing R² @ 20K ≥ 0.362
- **Decision**: randdiff σ₁ on uvlowr-S1 still ≤ baseline − 2σ_noise across both seeds
- **Cost**: 3 + 6 × 0.8 = **7.8 GPU-h** (+1.5h opt). **MUST**

### B3 (M3): rank sweep — kills Anti-D. **Multi-seed (n=2)** ⭐
- r ∈ {1, 2, 4 (reuse R004 s=0), 8, 16} × **2 seeds**
- Cells: 5×2 = 10; new: 9
- **Decision (P-C13)**: mean rollout MSE@h=10 has argmin in r ∈ {2,4}
- **Cost**: 9 × 0.8h = **7.2 GPU-h**. **MUST**

### B4 (M4): λ dose-response — kills Anti-C. **Multi-seed (n=2)** ⭐
- λ ∈ {0 (reuse R002/R003), 0.001, 0.003, 0.01 (reuse R006 s=0), 0.03, 0.1} × **2 seeds**
- Cells: 6×2 = 12; new: 9 (3 reused)
- **Metrics**: σ₁/F²/SR + |λ_max| (DMD) + ρ (nonlinear) + rollout MSE
- **Decision (P-Cdyn)**: |λ_max|(λ=0.1) ≤ |λ_max|(λ=0) − 2σ_noise AND monotone AND ρ_mean(λ=0.01) < 0.95
- **Figure**: Fig 4 with NDE/DEQ contrast caption
- **Cost**: 9 × 0.8h = **7.2 GPU-h**. **MUST**

### B4' (M4.5) — 40K extension for **ALL 3 variants** (⭐ expanded) — kills Anti-I
- 3 runs: `frozen_baseline_40K`, `frozen_uvlowr_40K`, `frozen_randdiff_40K`, seed=0, 40K steps
- Tracks σ₁/F²/SR/|λ_max|/rollout at {8K, 16K, 24K, 32K, 40K}
- **Decision**: sign(σ₁(randdiff) − σ₁(baseline)) preserved 8K → 40K
- **Bonus**: provides encoded TLPS at 40K (addresses self-review #7)
- **Cost**: 3 × 3h = **9.0 GPU-h**. **MUST**

### B5' (M4.6) — (ε,λ) 4-corner cross-product — kills Anti-H. **Multi-seed (n=2)** ⭐
- 4 corners × 2 seeds = 8 new runs
- Corners: (ε=0.0125, λ=0.001), (0.0125, 0.1), (0.8, 0.001), (0.8, 0.1)
- **Decision (P-Anti-H)**: all 4 corners σ₁ ≤ baseline + 2σ_noise
- **Cost**: 8 × 0.8h = **6.4 GPU-h**. **MUST**

### B5 (M5) — DMD + nonlinear ρ post-hoc
- All v5 + reused ckpts (~28 ckpts) via `tier1_dump_and_metrics.py`
- Output per ckpt: |λ_max|, n_unstable, top-5 |λ|, ρ_mean, ρ_p95
- **Cost**: 0.3 GPU-h. **MUST**

### B6 (M6) — aggregate + paper figures + ANALYSIS.md
- Fig 2 (ε sweep), Fig 3 (rank), Fig 4 (λ dose-response + NDE contrast caption), Fig 5 (B9 cross-dataset), Fig 6 (B10 cross-encoder), Fig A* appendix
- **Cost**: 0.2 GPU-h. **MUST**

### B7 (M7) ⭐ PROMOTED MUST — optimizer/LR robustness
- 2 × {frozen_baseline, frozen_randdiff} at LR ∈ {5e-5 (reuse R002/R006), 5e-4}, seed=0, 8K
- **Decision**: F²(randdiff) < baseline F² at LR=5e-4 too
- **Cost**: 2 × 0.8h = **1.6 GPU-h**. **MUST**

### B8 (M8) ⭐ PROMOTED MUST — 2nd Stage-1 baseline seed — kills Anti-L
- 1 Stage-1 baseline seed=1 (20K joint) + 3 frozen × 1 seed × 8K
- **Decision**: cross-Stage-1-seed reproduction of sign-flip
- **Cost**: 3 + 2.4 = **5.4 GPU-h**. **MUST**

### B9 (M9) ⭐⭐⭐ PROMOTED FROM CUT TO MUST — PointMaze cross-dataset — kills Anti-K
- **Why critical**: novelty-check flagged "Push-T only" as the biggest reviewer attack. Cross-dataset replication is the single most-defensible add.
- **Dataset**: PointMaze (LeWM upstream dataset; in LeWM repo's data scripts)
- **Stage-1** (20K joint) + 3 variants × **2 seeds** × 8K = 7 runs
- **Decision (P-Anti-K)**: randdiff predictor σ₁ < baseline σ₁ on PointMaze (v3-direction match)
- **Failure**: if randdiff opposite direction on PointMaze → C2 is Push-T-specific; major reframe
- **Setup overhead**: PointMaze data prep (~10 min via existing scripts)
- **Cost**: 3 + 6 × 0.8 = **7.8 GPU-h**. **MUST**

### B10 (M10) ⭐ PROMOTED MUST — DINOv2 cross-pretraining ×2 seeds — kills Anti-J
- Adapter setup (0.5h) + 3 variants × **2 seeds** × 8K
- **Decision**: randdiff direction matches v3 LeWM-encoder direction
- **Cost**: 0.5 + 6 × 0.8 = **5.3 GPU-h**. **MUST**

### CUT for v5.2 (deferred to follow-up paper)
- B11 Stage-1 step-count sweep (partially addressed by B4')
- B12 architecture depth/width ablation
- B15 full ε×λ factorial heatmap (24×2 = 48 cells, ~38 GPU-h)
- Reacher dataset (3rd dataset)
- Push-T planning-success-rate eval

---

## Run Order and Milestones

| Milestone | Goal | Runs (new + reused) | Cost (GPU-h) | Decision Gate |
|---|---|---|---|---|
| **M0** | sanity rerun | 1 + 0 | 0.3 | pred_loss drop ≥30% |
| **M0.5 (B0', n=7)** ⭐ | noise floor | 5 + 2 | 4.0 | All 7 seeds within 3σ |
| **M1 (B1)** ⭐ multi-seed | ε sweep | 7 + 1 | 5.6 | **ABORT M2 if any ε σ₁ > baseline+0.20** |
| **M2 (B2)** | uvlowr-Stage-1 + freeze 3×2 | 7 + 0 | 7.8 | Stage-1 gate + sign-flip |
| **M3 (B3)** ⭐ multi-seed | rank sweep | 9 + 1 | 7.2 | Knee at r∈{2,4}? |
| **M4 (B4)** ⭐ multi-seed | λ dose-response | 9 + 3 | 7.2 | Monotone \|λ_max\|? ρ<0.95? |
| **M4.5 (B4')** ⭐ ALL 3 variants | 40K extension | 3 + 0 | 9.0 | Sign preserved at 40K? |
| **M4.6 (B5')** ⭐ multi-seed | 4 corners | 8 + 0 | 6.4 | No corner reproduces v2 inflation? |
| **M5 (B5)** | DMD + ρ post-hoc | analysis | 0.3 | Tier-1 \|λ_max\| replicates? |
| **M6 (B6)** | Aggregate + Fig 2-6 + ANALYSIS.md | analysis | 0.2 | — |
| **M7 (B7)** ⭐ MUST | LR robustness | 2 + 2 | 1.6 | F² randdiff < baseline at 10× LR? |
| **M8 (B8)** ⭐ MUST | Stage-1 seed=1 + 3 frozen | 4 + 0 | 5.4 | Cross-seed reproduction? |
| **M9 (B9)** ⭐⭐⭐ MUST | **PointMaze cross-dataset** | 7 + 0 | 7.8 | Randdiff v3-direction on PointMaze? |
| **M10 (B10)** ⭐ MUST multi-seed | DINOv2 cross-pretraining | 6 + 0 | 5.3 | Randdiff v3-direction on DINOv2? |
| **MUST TOTAL** | | **70 new + 8 reused = 78 runs** | **~67.0 GPU-h ≈ ¥201** | over ¥175 by ~¥26 |

**Pre-registered predictions** (with empirical σ_noise from B0' n=7):

```
P-C2:    For ≥3 of 4 ε, mean over 2 seeds:  σ₁(randdiff, ε) ≤ baseline_mean − 2·σ_noise[σ₁]
P-Cdyn:  mean: |λ_max|(λ=0.1) ≤ |λ_max|(λ=0) − 2·σ_noise[|λ_max|]
         AND  monotone non-increasing across λ (±σ_noise tolerance)
         AND  ρ_mean(randdiff at λ=0.01) < 0.95
P-C13:   mean argmin_r rollout-MSE(h=10) in {2, 4}
         AND  rollout-MSE(r=16) ≥ rollout-MSE(r=4) − 2·σ_noise[rollout]
P-Anti-I: sign(σ₁(randdiff) − σ₁(baseline)) preserved 8K → 40K (B4')
P-Anti-H: all 4 (ε,λ) corners: σ₁ ≤ baseline + 2·σ_noise (B5')
P-Anti-J: B10 DINOv2 σ₁(randdiff) < σ₁(baseline)
P-Anti-K: B9 PointMaze σ₁(randdiff) − σ₁(baseline) < 0
Abort gate M1→M2: any ε σ₁ > baseline + 0.20 → STOP (save 60+ GPU-h)
```

**Overall**: ≥2/3 main P-* AND P-Anti-I AND P-Anti-H AND P-Anti-J AND P-Anti-K all hold → paper draft.

---

## Compute and Data Budget

- **MUST-RUN**: 67.0 GPU-h (~¥201)
- **Balance**: ¥175 → **over by ~¥26**, top up or apply prioritized cuts
- **Wall-clock**: ~36h single AutoDL session (one weekend)
- **Data prep**: Push-T already on AutoDL; **PointMaze needs download** (LeWM repo has scripts; ~10 min); DINOv2 cached via torch.hub
- **Biggest bottleneck**: B2 + B8 + B9 each have 20K Stage-1 (3h serial each) = 9h Stage-1 chain

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Budget overrun (¥201 vs ¥175) | high | medium | top up ¥30 OR drop B0' to n=5 (save ¥6) + drop B4' baseline_40K + uvlowr_40K (save ¥18) = ¥177 fit |
| B0' cross-seed std huge | low | medium | use 3σ thresholds or expand to n=10 |
| B2 Stage-1 doesn't converge | medium | medium | gate auto-extends to 30K |
| B4 \|λ_max\| non-monotone | medium | medium | regime-dependent reporting |
| B4' σ₁ flips back at 40K | medium | **HIGH** | most important confound; honest reporting |
| B5' corner reproduces v2 inflation | medium | high | identify regime; reframe paper |
| B9 PointMaze randdiff opposite direction | medium | **HIGH** | major reframe to "Push-T specific" |
| B10 DINOv2 download timeout | low | low | pre-stage via huggingface mirror |
| AutoDL net/disk issues | medium | medium | flock + expand_disk=120 |

---

## Final Checklist (v5.2)

- [x] Main paper tables covered (cross-Stage-1, ε, rank, λ dose-response, **cross-dataset**, **cross-encoder**, 40K, (ε,λ) factorial)
- [x] Novelty isolated (mechanism, not technique)
- [x] Simplicity defended (frozen narrative simpler than v2)
- [x] Frontier component — N/A
- [x] MUST (all blocks) vs CUT (B11/B12/B15, Reacher, planning eval) separated
- [x] Failure interpretation for every block
- [x] Anti-claims A through L assigned
- [x] Pre-registered P-* with empirical σ_noise
- [x] **Multi-seed (n=2) for all sweeps** (B1, B3, B4, B5', B9, B10)
- [x] **B0' n=7 seeds** (5 new)
- [x] **B4' all 3 variants at 40K**
- [x] M1→M2 abort gate explicit
- [x] Stage-1 convergence gate explicit
- [x] Nonlinear contraction ρ in B5
- [x] **B7 / B8 / B9 / B10 PROMOTED to MUST**
- [x] **NDE/DEQ contrast in motivation + Fig 4 caption**
- [ ] Budget: ~¥201 vs ¥175 — over by ~¥26, need top-up or modest cuts

---

# Execution Playbook (v5.2)

## Pre-flight checklist

```
[ ] Top up AutoDL balance to ≥¥200
[ ] Create 4090-48G instance, expand_system_disk_by_gb=120
[ ] First pip speedtest ≥2 MB/s
[ ] tsinghua mirror in ~/.pip/pip.conf
[ ] flock -n /workspace/_setup_v5.lock bash setup_v5.sh
[ ] scp v3 Stage-1 ckpt
[ ] **Pre-stage PointMaze data** (LeWM repo download script)
[ ] **Pre-cache DINOv2 weights** (torch.hub.load once)
[ ] Verify CUDA
[ ] Run M0 sanity (R201) — 5 min smoke
```

## Wall-clock schedule (~36h session)

| Hour | Action | Cost (cum) |
|---|---|---|
| 0:00–0:30 | Pre-flight + M0 sanity + start B0' | 0.3 |
| 0:30–4:30 | B0' 5 new seeds × 0.8h | 4.3 |
| 4:30–10:00 | B1 ε sweep × 2 seeds (7 runs × 0.8h) | 9.9 |
| 10:00 GATE | **Abort check** | — |
| 10:00–17:30 | B2 Stage-1 + 6 frozen | 17.7 |
| 17:30–24:30 | B3 rank sweep × 2 seeds (9 runs × 0.8h) | 24.9 |
| 24:30–32:00 | B4 λ sweep × 2 seeds (9 runs × 0.8h) | 32.1 |
| 32:00–41:00 | B4' 40K all 3 variants (3 × 3h) | 41.1 |
| 41:00–47:30 | B5' (ε,λ) corners × 2 seeds (8 × 0.8h) | 47.5 |
| 47:30–49:30 | B7 LR (2 × 0.8h) + B8 Stage-1-seed-1 (4 runs: 3h S1 + 3 × 0.8h frozen) | 54.9 |
| 49:30–57:30 | B9 PointMaze (S1 3h + 6 × 0.8h frozen) | 62.7 |
| 57:30–63:00 | B10 DINOv2 × 2 seeds (6 × 0.8h + adapter setup) | 68.0 |
| 63:00–63:30 | B5 DMD + ρ post-hoc on ALL ckpts | 68.3 |
| 63:30–67:00 | scp pull + RELEASE | — |

(Note: actual cost-tracked time is GPU-burning time; wall-clock includes data transfer.)

## Decision tree (operational)

```
M0 sanity OK? → M0.5
M0.5 σ_noise reasonable (max seed dev < 3σ)? → M1
M1: any ε σ₁ > baseline+0.20? → ABORT (skip M2-M10; save 60+ GPU-h)
   else → M2
M2: Stage-1 converges at 20K? → frozen 6 runs; else extend to 30K
M3 → M4 → M4.5 → M4.6 → M7 → M8 → M9 → M10 → M5 → M6 (all unconditional)
```

## Per-block VARIANT_OVERRIDES (multi-seed expansion)

```bash
# B0' noise floor — 5 new seeds (2,3,4,5,6)
for s in 2 3 4 5 6; do
    train_frozen v5_baseline_seed${s} ${STAGE1_BASELINE_CKPT} 8000 seed=${s}
done

# B1 ε sweep × 2 seeds (ε=0.05 s=0 reused from R006)
for s in 0 1; do for eps in 0.0125 0.05 0.2 0.8; do
    [ "$eps" = "0.05" ] && [ "$s" = "0" ] && continue
    tag=$(echo $eps | tr '.' '_')
    train_frozen v5_eps_${tag}_seed${s} ${STAGE1_BASELINE_CKPT} 8000 seed=${s} \
        +loss.rand_diff.weight=0.01 +loss.rand_diff.eps=${eps} +loss.rand_diff.num_dirs=2
done; done

# B3 rank sweep × 2 seeds (r=4 s=0 reused from R004)
for s in 0 1; do for r in 1 2 4 8 16; do
    [ "$r" = "4" ] && [ "$s" = "0" ] && continue
    train_frozen v5_uvlowr_r${r}_seed${s} ${STAGE1_BASELINE_CKPT} 8000 seed=${s} \
        +predictor.ffn_rank=${r}
done; done

# B4 λ sweep × 2 seeds (λ=0 reused R002/R003; λ=0.01 s=0 reused R006)
for s in 0 1; do for lam in 0.001 0.003 0.03 0.1; do
    tag=$(echo $lam | tr '.' '_')
    train_frozen v5_lam_${tag}_seed${s} ${STAGE1_BASELINE_CKPT} 8000 seed=${s} \
        +loss.rand_diff.weight=${lam} +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2
done; done
train_frozen v5_lam_0_01_seed1 ${STAGE1_BASELINE_CKPT} 8000 seed=1 \
    +loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2

# B4' 40K extension all 3 variants
train_frozen v5_baseline_40K  ${STAGE1_BASELINE_CKPT} 40000
train_frozen v5_uvlowr_40K    ${STAGE1_BASELINE_CKPT} 40000 +predictor.ffn_rank=4
train_frozen v5_randdiff_40K  ${STAGE1_BASELINE_CKPT} 40000 \
    +loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2

# B5' (ε,λ) 4 corners × 2 seeds
for s in 0 1; do for ep_lam in \
    "0.0125 0.001 LL" "0.0125 0.1 LH" "0.8 0.001 HL" "0.8 0.1 HH"; do
    set -- $ep_lam; eps=$1; lam=$2; tag=$3
    train_frozen v5_corner_${tag}_seed${s} ${STAGE1_BASELINE_CKPT} 8000 seed=${s} \
        +loss.rand_diff.weight=${lam} +loss.rand_diff.eps=${eps} +loss.rand_diff.num_dirs=2
done; done

# B7 LR ablation
train_frozen v5_LRhi_baseline ${STAGE1_BASELINE_CKPT} 8000 optimizer.lr=5e-4
train_frozen v5_LRhi_randdiff ${STAGE1_BASELINE_CKPT} 8000 optimizer.lr=5e-4 \
    +loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2

# B8 Stage-1 seed=1
train_stage1 v5_stage1_baseline_seed1 20000 seed=1
S1S1="${STABLEWM_HOME}/v5_stage1_baseline_seed1/v5_stage1_baseline_seed1_weights.ckpt"
for var in baseline uvlowr randdiff; do
    overrides=""
    [ "$var" = "uvlowr" ] && overrides="+predictor.ffn_rank=4"
    [ "$var" = "randdiff" ] && overrides="+loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2"
    train_frozen v5_S1seed1_${var} $S1S1 8000 ${overrides}
done

# B9 PointMaze cross-dataset
.venv/bin/python download_pointmaze.py  # or use LeWM data prep
train_stage1 v5_pm_stage1 20000 data=pointmaze
S1PM="${STABLEWM_HOME}/v5_pm_stage1/v5_pm_stage1_weights.ckpt"
for s in 0 1; do for var in baseline uvlowr randdiff; do
    overrides=""
    [ "$var" = "uvlowr" ] && overrides="+predictor.ffn_rank=4"
    [ "$var" = "randdiff" ] && overrides="+loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2"
    train_frozen v5_pm_${var}_seed${s} $S1PM 8000 data=pointmaze seed=${s} ${overrides}
done; done

# B10 DINOv2 × 2 seeds
.venv/bin/python tier1_dinov2_adapter.py --build-adapter \
    --out ${STABLEWM_HOME}/v5_dino_adapter.pt --init random --seed 42
for s in 0 1; do for var in baseline uvlowr randdiff; do
    overrides=""
    [ "$var" = "uvlowr" ] && overrides="+predictor.ffn_rank=4"
    [ "$var" = "randdiff" ] && overrides="+loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2"
    .venv/bin/python tier1_dinov2_adapter.py --train \
        --adapter ${STABLEWM_HOME}/v5_dino_adapter.pt \
        --name v5_dino_${var}_seed${s} --steps 8000 --batch 16 --seed ${s} ${overrides}
done; done

# B5 DMD + ρ post-hoc
.venv/bin/python tier1_dump_and_metrics.py --batch-all \
    --dir-glob "${STABLEWM_HOME}/v5_*" \
    --output ${STABLEWM_HOME}/dmd_v5.json
```

## File outputs in `lewm_autodl_results_v5/`

```
lewm_autodl_results_v5/
├── ANALYSIS.md                                  ← v5 writeup with all P-* verdicts + NDE contrast
├── noise_floor/                                 ← B0' R002/R003/R202-R206 (n=7)
├── eps_sweep/v5_eps_*_seed{0,1}/                ← B1 7 dirs (eps_050_seed0 reused from v3)
├── stage1_uvlowr_seed0/                         ← B2 Stage-1
├── frozen_*_on_uvlowr_seed{0,1}/                ← B2 Stage-2 × 6
├── rank_sweep/v5_uvlowr_r*_seed{0,1}/           ← B3 9 new dirs (r=4 s=0 reused)
├── lam_sweep/v5_lam_*_seed{0,1}/                ← B4 9 new dirs
├── long_40K/v5_{baseline,uvlowr,randdiff}_40K/  ← B4' 3 dirs
├── corner/v5_corner_{LL,LH,HL,HH}_seed{0,1}/    ← B5' 8 dirs
├── lr_robustness/v5_LRhi_{baseline,randdiff}/   ← B7
├── stage1_seed1/v5_S1seed1_{baseline,uvlowr,randdiff}/  ← B8
├── pointmaze/v5_pm_{stage1,baseline_*,uvlowr_*,randdiff_*}/  ← B9 7 dirs
├── dinov2/v5_dino_{baseline,uvlowr,randdiff}_seed{0,1}/  ← B10 6 dirs
├── dmd_full_budget/dmd_v5.json                  ← B5 DMD + ρ per ckpt
└── figures/Fig{2,3,4,5,6}.{pdf,png}             ← main paper figures
```

---

## Pivot summary (v5.1 → v5.2)

| Aspect | v5.1 | v5.2 |
|---|---|---|
| Noise floor seeds | 5 (3 new) | **7 (5 new)** |
| ε sweep seeds | 1 | **2** |
| Rank sweep seeds | 1 | **2** |
| λ sweep seeds | 1 | **2** |
| (ε,λ) corner seeds | 1 | **2** |
| DINOv2 seeds | 1 | **2** |
| 40K extension | randdiff only | **all 3 variants** |
| B7 LR | NICE | **MUST** |
| B8 Stage-1 seed=1 | NICE | **MUST** |
| **B9 PointMaze** | CUT | **MUST** ⭐⭐⭐ |
| B10 DINOv2 | NICE | **MUST** |
| Anti-claims covered | A-J | **A-L** (+ K cross-dataset, L cross-Stage-1-seed) |
| Compute cost | 35.4 GPU-h (¥106) | **~67 GPU-h (¥201)** |

---

## Lessons inherited (v2 → v3 → v4 → v5 → v5.1 → v5.2)

- AutoDL disk: `expand_system_disk_by_gb=120` (extra for PointMaze + DINOv2)
- Network speedtest <2 MB/s → release + recreate
- flock for setup; never `pkill -f "pip install"`
- SSH cmds prefix `cd /workspace/le-wm &&`
- KeepFrozenInEvalCallback for picklable freeze
- Release immediately after final scp pull
- WSL2: num_workers=0
- **v5.1 NEW**: noise floor before single-seed sweeps; linear+nonlinear ρ co-reporting; abort gates
- **v5.2 NEW**: 2 seeds = minimum credible n for any frozen comparison; cross-dataset (B9) is the most-defensible single upgrade
