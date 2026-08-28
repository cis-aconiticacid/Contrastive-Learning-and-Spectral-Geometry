# Experiment Plan v5 — Post-Novelty-Check Pivot

**Problem**: We've shown that in jointly-trained JEPA (v2 vs v3), predictor-side regularization is **not predictor-local**: the randdiff (Scarvelis-Solomon nuclear-norm) penalty produces opposite predictor-Jacobian signatures under joint training vs. frozen-encoder ablation. Six metrics (σ₁, F², SR, σ₁/σ₁₀, β, PR) all reverse direction. Tier-1 DMD on frozen runs additionally shows randdiff drives `|λ_max| < 1` (predictor becomes a contraction map → fixed-point convergence) — a new mechanistic explanation for randdiff's bad rollout. Codex novelty check (gpt-5.4, xhigh) says: broad framing 4/10, **tight mechanistic framing 6/10**, do not abandon — abandon only the weak framing.

**Method Thesis (new headline, post-novelty)**: *Encoder-mediated compensation can mask and even invert predictor-side regularization effects in end-to-end JEPAs.* Specifically: (a) joint vs frozen produces sign-reversed spectral signatures; (b) nuclear-norm regularization on the predictor over-contracts latent dynamics into a fixed-point regime when the encoder is allowed to absorb the gradient.

**Date**: 2026-05-12

**Budget**: 19.5 GPU-h must-run (~¥59) + up to 20 GPU-h nice-to-have (~¥60). Balance ¥175 → comfortable.

**Inputs**:
- v3 Stage-1 ckpt: `/workspace/lewm_autodl_results_v3/stage1_baseline_seed0_weights.ckpt` (216 MB)
- v2 reference data: `/workspace/lewm_autodl_results_v2/` (3 variants × 3 seeds × 40K joint)
- v3 results: `/workspace/lewm_autodl_results_v3/`
- Tier-0 results: `refine-logs/tier0_results.json`, `tier0_correlation_matrix.json`
- Tier-1 results: `refine-logs/tier1_traj/predicted_metrics.json` (local 3K-step repro)
- Code: `/workspace/le-wm` (already patched with freeze + KeepFrozenInEvalCallback)

---

## Claim Map (post-pivot)

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| **C2 (anchor)**: encoder-mediated compensation in joint JEPA reverses predictor-Jacobian signatures | This is the headline. Reframes any claim about predictor properties in joint training. | (i) Sign-flip in σ₁/SR/F² between v2 joint and v3 frozen survives ε sweep (B1); (ii) survives different Stage-1 shape (B2); (iii) survives optimizer/LR perturbation (B7) | B1, B2, B7 |
| **C_dyn (anchor #2)**: nuclear-norm regularization over-contracts the predictor into fixed-point dynamics | New from Tier-1 DMD. Explains the apparent paradox that the most "stable" predictor (\|λ_max\|<1) gives the worst rollout MSE. Pairs with C2 as the mechanistic punchline. | (iv) λ-sweep (B4) shows a dose-response: as λ increases, \|λ_max\| decreases monotonically toward 0 AND rollout MSE first improves then explodes; (v) DMD spectrum is rerun on full-budget v4 ckpts (B5), not just 3K-step local reproductions | B4, B5 |
| **C13 (supporting)**: uvlowr r=4 sits near the knee of a rollout-vs-rank curve at a value matching Push-T's intrinsic dim ≈ 4 | Saves the "lean LeWM" story from the post-hoc-numerology attack. | A rollout-MSE-vs-r curve over r∈{1,2,4,8,16} with a clear knee at r=4 | B3 |
| **Anti-claim A** (reversal is ε-calibration): one ε value reproduces v2-like signature | Falsifies C2 | Killed by B1 negative result (no ε reproduces v2 inflation) | B1 |
| **Anti-claim B** (reversal is Stage-1-shape-specific): uvlowr-shaped Stage-1 makes randdiff revert to v2 direction | Falsifies C2 generality | Killed by B2 negative result | B2 |
| **Anti-claim C** (over-stabilization is λ-specific): only λ=0.01 produces \|λ_max\|<1, others don't | Falsifies C_dyn | Killed by B4 showing monotone λ → \|λ_max\| dose-response curve | B4 |
| **Anti-claim D** (rank=4 is lucky): rollout MSE smooth in r, no knee at 4 | Falsifies C13 alignment story | Killed by B3 showing knee at r=4 in rollout curve | B3 |
| **Anti-claim E** (reversal is optimizer-dependent): different LR or scheduler removes the sign flip | Falsifies C2 robustness | Killed by B7 negative result | B7 |

**Demoted to supporting roles** (per novelty-check feedback):
- TLPS as diagnostic (Wang et al. 2603.12231 owns this) — cite, present as one panel
- Cross-metric correlation structure — present as supporting evidence, do NOT label "causal"
- C6 spectral regime dichotomy — appendix
- C8/C9/C10 — appendix

---

## Paper Storyline

**Main paper proves:**
1. **C2 (encoder-mediated compensation)** — joint vs frozen sign reversal, survives ε sweep, different Stage-1 shape, and optimizer perturbation
2. **C_dyn (over-contraction failure)** — λ dose-response curve linking spectral migration (|λ_max| → 0) to rollout failure mode
3. **C13 (sharpened)** — rank sweep with knee at r=4 matching data intrinsic dim

**Appendix:**
- v2 joint-training tables (for the joint condition baseline)
- v3 frozen-encoder tables (baseline-shaped Stage-1 condition)
- C6 ImageNet/random/JEPA Jacobian dichotomy (cheap-check from Tier-0)
- Cross-metric correlation matrix (tier-0)
- Tier-1 trajectory metrics (TLPS, curvature, TwoNN ID) — explicit credit to Wang et al.
- v3 → v5 evolution of the claim verdicts

**Cut (or future work):**
- Second dataset beyond Push-T (deferred to follow-up paper; novelty check flagged as biggest cost item)
- Push-T planning success rate (separate eval pipeline build)
- λ sweep + ε sweep cross-product (factorial design — too expensive)
- TLPS as a loss (Wang et al. already did this)

---

## Experiment Blocks

### Block 0 (M0): ε-norm probe + freeze sanity (DONE in v4 prep; rerun on fresh instance)

- Already executed locally; will rerun on AutoDL to confirm env reproducibility
- Output: `v4_eps_bracket.json` with ε ∈ {0.0125, 0.05, 0.2, 0.8}, ‖z_proj‖₂ = 33.28
- Cost: 0.3 GPU-h
- Priority: MUST-RUN

### Block 1 (M1, B1): ε sensitivity sweep on frozen randdiff — anti-claim A

- **Claim tested**: C2's reversal is NOT ε mis-calibration
- **Why this block exists**: The single most-cited "easy attack" on the v3 reversal
- **Dataset**: Push-T expert (identical splits to v3)
- **Compared systems**: 4 frozen_randdiff runs at ε ∈ {0.0125, 0.05, 0.2, 0.8} × seed 0 × 8K, on v3 Stage-1 ckpt
- **Metrics** (decisive first):
  - **Predictor Jacobian σ₁, SR, F²** at final step — sign-flip indicator
  - **|λ_max|, n_unstable** (DMD from rollout latents) — over-contraction indicator (C_dyn pre-test)
  - **Rollout MSE at h∈{1,3,5,10}** — operational
  - rand_diff_loss trajectory (sanity that penalty fires at each ε)
- **Setup**: same as v3 frozen runs; Stage-1 source = v3 baseline ckpt; 8K predictor-only steps, batch 32; ε override via Hydra
- **Success criterion**: at NO ε does randdiff predictor reproduce v2-like σ₁ inflation (≥+30% over baseline) — sign stays negative or near zero across the 64× geometric sweep
- **Failure interpretation**:
  - Any ε reproduces v2 inflation → C2 reversal is ε-driven, downgrade to "suggestive in a regime"
  - 2/4 reproduce → middle ground; characterize the regime
- **Table/figure target**: main paper Fig 2 inset (σ₁(ε), SR(ε), |λ_max|(ε) curves)
- **Priority**: MUST-RUN
- **Cost**: 4 runs × 0.8 GPU-h = **3.2 GPU-h**

### Block 2 (M2, B2): uvlowr-co-trained Stage-1 + freeze multi-predictor — anti-claim B

- **Claim tested**: C2 generalizes across encoder shapes
- **Why this block exists**: anti-claim B says "you only tested in baseline-shaped Stage-1". Train a fresh Stage-1 with uvlowr_r4 as the joint predictor; freeze; rerun the 3-variant ablation.
- **Dataset**: Push-T expert
- **Compared systems**:
  - **Stage-1**: `stage1_uvlowr_r4` (joint 20K, uvlowr_r4 predictor) — new
  - **Stage-2 (frozen, 3 variants × 2 seeds)**: frozen_baseline, frozen_uvlowr_r4 (matched), frozen_randdiff (cross-shaped)
- **Metrics**: predictor Jacobian σ₁/SR/F² (decisive: sign-flip on randdiff?); rollout MSE at h∈{1,3,5,10,15,20}; |λ_max|, n_unstable
- **Setup**: Stage-1 20K joint with `+predictor.ffn_rank=4` and SIGReg active, seed 0; Stage-2 8K frozen with the same 3 variants × 2 seeds as v3
- **Success criterion**: cross-Stage-1 comparison plot shows randdiff direction (σ₁ shrinkage, SR mild increase) holds on the uvlowr-shaped encoder too. C2 generalizes.
- **Failure interpretation**:
  - Direction reverses again in uvlowr-shaped Stage-1 → reversal is Stage-1-shape-specific; reframe paper
  - Direction holds → C2 strong on Confound 2
- **Table/figure target**: main paper Table 3 (cross-Stage-1 cross-variant); Fig 2 expanded
- **Priority**: MUST-RUN
- **Cost**: Stage-1 (3 GPU-h) + 6 frozen × 0.8h = **7.8 GPU-h**

### Block 3 (M3, B3): uvlowr rank sweep — defends C13 against numerology attack

- **Claim tested**: C13 — uvlowr's rollout advantage has a knee at r=4 matching Push-T's data intrinsic dim
- **Why this block exists**: Codex flagged C13 as the weakest claim; reviewer attack "rank=4 is lucky". Needs falsification test = a curve.
- **Dataset**: Push-T
- **Compared systems**: `frozen_uvlowr_r{r}` for r ∈ {1, 2, 4, 8, 16}, single seed, on v3 baseline Stage-1 (r=4 may reuse v3 R004 to save a run)
- **Metrics** (decisive first): rollout MSE at h=10 vs r (the knee); predictor params; predictor Jacobian σ₁/SR; |λ_max|
- **Setup**: Stage-1 = v3 baseline; Stage-2 = 8K predictor-only at each r; identical to v3 M1
- **Success criterion**: rollout-MSE-vs-r curve has a minimum or plateau onset at r=4; worse below (under-parameterized); flat or worse above
- **Failure interpretation**:
  - Monotone in r → no special r=4; C13 reframed as parameter-budget story
  - Knee but at r≠4 → revisit data intrinsic dim measurement
- **Table/figure target**: main paper Fig 3 (rank-vs-rollout curve)
- **Priority**: MUST-RUN
- **Cost**: 4 new runs (skip r=4 if reusing v3) × 1.0 GPU-h = **4.0 GPU-h**

### Block 4 (M4, B4) ⭐ NEW: randdiff λ dose-response — defends C_dyn

- **Claim tested**: C_dyn — over-contraction is a dose-dependent function of λ; not specific to λ=0.01
- **Why this block exists**: The strongest single defense against reviewer attack "your over-stabilization claim is one λ". Predicts: monotone migration of |λ_max| toward 0 as λ grows; non-monotone rollout MSE (improves at low λ, explodes at high λ).
- **Dataset**: Push-T
- **Compared systems**: `frozen_randdiff` at λ ∈ {0, 0.001, 0.003, 0.01, 0.03, 0.1} × seed 0 × 8K on v3 Stage-1 (λ=0 ≡ frozen_baseline; reuse v3 R002)
- **Metrics** (decisive first):
  - **|λ_max|, n_unstable** (DMD on rollout latents) — primary dose-response
  - **Predictor Jacobian σ₁, F², SR** — secondary
  - **Rollout MSE at h∈{1,5,10}** — operational
  - rand_diff_loss trajectory + final value
- **Setup**: Stage-1 = v3 baseline, Stage-2 8K, single seed, ε=0.05, num_dirs=2; only λ varies
- **Success criterion**:
  - |λ_max| decreases monotonically with λ → over-contraction is **dose-dependent**, defends C_dyn
  - Rollout MSE has a U-shape vs λ (or monotone if even tiny λ is too much) → links spectral migration to operational failure
  - At λ=0.1, predictor latent rollout converges to a fixed point visibly (TLPS approaches 1.0, predicted ‖v_t‖ → 0)
- **Failure interpretation**:
  - |λ_max| is U-shaped or non-monotone in λ → C_dyn reframed
  - |λ_max| < 1 at all λ > 0 including tiny → C_dyn even stronger
- **Table/figure target**: main paper Fig 4 (λ dose-response curve, dual y-axis: |λ_max| left, rollout MSE right)
- **Priority**: MUST-RUN
- **Cost**: 5 new runs (skip λ=0 / λ=0.01 if reusing v3) × 0.8 GPU-h = **4.0 GPU-h**

### Block 5 (M5, B5) ⭐ NEW: post-hoc DMD extraction for full-budget ckpts

- **Claim tested**: Tier-1 DMD findings (from 3K-step local reproduction) replicate at v3/v4's 8K budget
- **Why this block exists**: Tier-1 numbers (|λ_max|, n_unstable) were from 3K-step local repro because v3 frozen ckpts weren't preserved. v4 produces fresh 8K ckpts → can replicate the |λ_max|<1 finding at full budget. Pure analysis script, no retraining.
- **Dataset**: Push-T val trajectories (same 80 used in tier-1)
- **Compared systems**: re-process all v4 frozen ckpts produced by B1/B2/B3/B4 — extract predicted/encoded trajectories + DMD
- **Metrics**: |λ_max|, n_unstable, top-5 |λ|, spectral abscissa per ckpt
- **Setup**: reuse `tier1_dump_trajectories.py` + `tier1_metrics.py` (extended to take any ckpt path); pure inference, ≤5 min/ckpt
- **Success criterion**: |λ_max|(randdiff) < |λ_max|(baseline) < |λ_max|(uvlowr) preserves at 8K → tier-1 numbers (|λ_max|≈0.996) replicate within ±0.05 → fixed-point claim is real, not 3K-step artifact
- **Failure interpretation**:
  - At 8K all variants have |λ_max|=1 → randdiff still trains a stable predictor but no extreme contraction → C_dyn weakened to "more stable than baseline"
- **Table/figure target**: appendix Table A2 (DMD numbers at 3K vs 8K, side-by-side)
- **Priority**: MUST-RUN
- **Cost**: 0.3 GPU-h (eval only, no training)

### Block 6 (M6, B6): aggregation + paper figure generation

- **Tasks**: combine B1-B5 outputs into `lewm_autodl_results_v5/ANALYSIS.md`, generate Fig 2/3/4 with paper-style typography, write claim-by-claim verdict update for CLAIM.md
- **Cost**: 0.2 GPU-h
- **Priority**: MUST-RUN

### Block 7 (M7, B7) NICE: optimizer/LR robustness — anti-claim E

- **Claim tested**: C2 reversal survives optimizer/LR perturbation
- **Why this block exists**: Codex listed "maybe reversal is optimizer/schedule-dependent" as a real attack. A 4-run check directly addresses it.
- **Dataset**: Push-T
- **Compared systems**: 2 × {frozen_baseline, frozen_randdiff} at LR ∈ {5e-5 default, 5e-4 10× higher} × seed 0 × 8K on v3 Stage-1
- **Metrics**: predictor Jacobian σ₁, F², SR — does sign-flip on randdiff vs baseline survive 10× LR change?
- **Setup**: minimal change from v3 M1; only `optimizer.lr` differs
- **Success criterion**: at LR=5e-4, randdiff still has F² < baseline F² by ≥20%
- **Failure interpretation**: LR-dependent reversal → C2 weakened
- **Table/figure target**: appendix Table A3
- **Priority**: NICE-TO-HAVE
- **Cost**: 4 runs × 0.8 = **3.2 GPU-h**

### Block 8 (M8, B8) NICE: 2nd Stage-1 seed robustness — extra anti-Confound-4 layer

- Single Stage-1 seed was an open v3 confound. B2's uvlowr-Stage-1 gives one robustness data point, but a 2nd baseline-Stage-1 seed would be a cleaner ablation.
- 1 Stage-1 baseline seed=1 (20K) + 3 frozen variants × 1 seed = 4 runs
- **Cost**: 3 + 3 × 0.8 = **5.4 GPU-h**
- **Priority**: NICE-TO-HAVE

### Block 9 (M9, B9) DEFER: second dataset (PointMaze or Reacher)

- Biggest attack surface from novelty check ("Push-T only"). But cost is high.
- Stage-1 (20K) on new dataset + 3 frozen variants × 1 seed = 4 runs × 3 GPU-h = 12 GPU-h. Plus dataset prep + config + eval pipeline.
- Defer to follow-up paper unless v5 lands and we want a 2-dataset version.
- **Priority**: CUT for v5; rerun consideration after v5 main results.

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost (GPU-h) | Risk |
|---|---|---|---|---|---|
| **M0** | ε-norm probe + freeze sanity on fresh instance | 1 (already done locally; rerun for AutoDL env) | gate: pred_loss ≥30% drop, enc grad=0 → proceed | 0.3 | low |
| **M1 (B1)** | ε sweep frozen_randdiff (anti-claim A) | 4 | If ANY ε reproduces v2 inflation → flag; else C2 strong on ε | 3.2 | medium — possibility of finding v2-like ε |
| **M2 (B2)** | uvlowr-Stage-1 + freeze 3-variant (anti-claim B) | 1 + 6 = 7 | randdiff sign-flip in uvlowr-Stage-1: same direction? | 7.8 | medium — Stage-1 training noise |
| **M3 (B3)** | rank sweep (defends C13) | 4 (r=4 reuses v3 R004) | Knee at r=4 in rollout-vs-r curve? | 4.0 | low |
| **M4 (B4) ⭐** | λ dose-response (defends C_dyn) | 5 (λ=0 / λ=0.01 reuse v3) | Monotone migration of \|λ_max\| toward 0? | 4.0 | medium — non-monotone possible |
| **M5 (B5) ⭐** | DMD extraction on all v4 ckpts | post-hoc, no training | \|λ_max\| < 1 holds at 8K too? | 0.3 | low |
| **M6 (B6)** | Aggregate + paper figs | analysis only | none | 0.2 | low |
| **MUST-RUN total** | | 22 runs (12 new + 1 Stage-1 + 9 reused) | | **19.5 GPU-h ≈ ¥59** | |
| **M7 (B7) NICE** | optimizer/LR robustness | 4 | LR=5e-4 preserves sign? | 3.2 | low |
| **M8 (B8) NICE** | 2nd Stage-1 baseline seed | 4 | 3-variant verdict on new Stage-1 | 5.4 | low |
| **NICE total** | | +8 runs | | **+8.6 GPU-h ≈ ¥26** | |
| **All-up total** | | 30 runs | | **28.1 GPU-h ≈ ¥85** of ¥175 balance | |

**Parallelism**: M0 must precede M1. M1 and M3 are independent (different Stage-1 source). M2's Stage-1 must precede M2's Stage-2. M4 depends on M0. M5 depends on M1-M4 outputs. Single GPU → serial.

**Pre-registered prediction** (Codex defense for "post-hoc metrics"):
- C2 prediction: for at least 3 of 4 ε values in B1, frozen_randdiff σ₁ < frozen_baseline σ₁
- C_dyn prediction: in B4, |λ_max| at λ=0.1 is at least 0.05 lower than |λ_max| at λ=0
- C13 prediction: in B3, rollout MSE at h=10 has a minimum at r ∈ {2, 4}, not at r ∈ {8, 16}

If 2+ of 3 pre-registered predictions hold, paper proceeds to write-up. If 2+ fail, full reframe.

---

## Compute and Data Budget

- **MUST-RUN GPU-hours**: 19.5 (~¥59)
- **MUST-RUN + NICE**: 28.1 (~¥85) of ¥175 balance → ¥90 reserve
- **Data prep**: none (Push-T already on AutoDL volume; v3 Stage-1 ckpt local)
- **Human eval**: none
- **Biggest bottleneck**: B2 Stage-1 (~3 GPU-h serial; cannot parallelize on single instance)
- **Wall clock budget**: ~24 hours of GPU time + ~6 hours of setup/transfer = single AutoDL session of ~30 hours

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| B1 ε sweep finds v2-like inflation at some ε | medium | high — softens C2 | report honestly; the ε-dependent regime characterization is itself publishable |
| B2 uvlowr-Stage-1 doesn't converge cleanly in 20K | medium | medium — B2 verdict ambiguous | budget +5K extension if pred_loss not plateaued; cap at 25K |
| B4 \|λ_max\| not monotone in λ | medium | medium — C_dyn reframed but not killed | the regime-dependent λ→\|λ_max\| mapping is still mechanistic; report shape honestly |
| B3 NaNs at r=1 (extreme low rank) | low | low — drop that point | report sweep over r∈{2,4,8,16} |
| AutoDL network/disk issues (v3 lesson) | medium | medium | use the v3 setup script with flock + expand_system_disk_by_gb=80 from the start |
| Stage-1 ckpt for B2 trained with bad seed | low | medium | v3 Stage-1 already passed sanity; if B2's Stage-1 has anomalous probing R² at end-of-Stage-1, retrain |
| 8K Stage-2 is undertraining | medium | low — patterns should hold by 8K | preregister verdict at h=10 (where v3 already showed clean ordering at 8K) |
| Reviewer pushes back on "frozen-encoder ablation is common" | high | low — defended by mechanism finding, not by the technique | lead with the *finding* (sign reversal), not the *technique* (freezing); cite DINO-WM |

---

## Implementation Notes

What changes vs. v4 codebase:
1. **train.py**: no change needed beyond v3 (freeze flag works; rank flag works; ε flag works)
2. **`run_pipeline_v5.sh`** (new): orchestrator for B1+B2+B3+B4 with phase boundaries
3. **DMD/Koopman extraction module**: extend `tier1_dump_trajectories.py` to accept any LeWM ckpt + run `tier1_metrics.py` over the results. Generalize from `VARIANTS = ["tier1_baseline", "tier1_uvlowr", "tier1_randdiff"]` to a CLI-passed list.
4. **Aggregation**: extend `aggregate.py` to emit `v5_cross_condition.csv` with columns `(stage1_source, variant, ε, λ, rank, seed, h, mse, σ₁, SR, F², |λ_max|, n_unstable)`
5. **Stage-1 source for B2**: train fresh with `+predictor.ffn_rank=4 +loss.rand_diff.weight=0.0` (uvlowr arch only, NO randdiff in Stage-1)

---

## Final Checklist

- [x] Main paper tables covered (cross-Stage-1, ε sweep, rank sweep, λ dose-response)
- [x] Novelty isolated (mechanism finding — sign reversal — is the headline, not a regularization technique)
- [x] Simplicity defended (the simpler frozen-encoder explanation is the one we promote; we DEMOTE the over-built v2 narrative)
- [x] Frontier component justified — N/A, no LLM/VLM/Diffusion in this work
- [x] MUST-RUN (B1-B6) vs NICE-TO-HAVE (B7, B8) vs CUT (B9) separated
- [x] Failure interpretations specified for each block
- [x] Anti-claims A/B/C/D/E listed and assigned to blocks
- [x] Pre-registered predictions written down BEFORE running (Codex defense for "post-hoc metrics")
- [x] Within budget (28.1 / 58 remaining GPU-hours)

---

## Lessons inherited (v2 → v3 → v4 plan)

- AutoDL disk: create with `expand_system_disk_by_gb=80` (108 GB total)
- AutoDL network: if pip speedtest <2 MB/s, release + recreate
- flock for setup; pkill safety (never -f on "pip install" pattern that matches own shell)
- SSH commands always prefix `cd /workspace/le-wm &&`
- Reuse `KeepFrozenInEvalCallback` from v3 (picklable freeze)
- Release immediately after final `scp` pull

---

## Pivot summary (per novelty check)

| Aspect | v3/v4 framing | v5 framing |
|---|---|---|
| Headline | "Lean LeWM via low-rank predictor regularization" | "Encoder-mediated compensation can mask and even invert predictor-side regularization effects" |
| Pillar 1 | C13 (rank=4 matches intrinsic dim) | **C2 (sign reversal under freeze)** |
| Pillar 2 | C3 (uvlowr Pareto improvement) | **C_dyn (nuclear-norm → over-contraction)** |
| Demoted | — | C13 (supporting), TLPS, cross-correlation, C6 |
| Differentiator | "We try low-rank in JEPA" (overlap with TD-JEPA arXiv:2510.00739) | "Sign-reversed predictor-Jacobian under freeze/unfreeze intervention + λ dose-response of |λ_max|" (not found elsewhere) |

---

## Composition with downstream skills

After v5:
- `/auto-review-loop` — iterate on CLAIM.md verdicts with v5 evidence
- `/research-refine` — finalize the paper draft with v5 framing
- If v5 lands cleanly, optional `/experiment-plan` for v6 with second dataset (B9)
